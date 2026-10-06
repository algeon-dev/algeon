//! Casts into decimal types with the semantics of a DataFusion `CAST`.
//!
//! A libcudf decimal type stores a scale and no precision, so libcudf wraps
//! values that overflow the declared precision and truncates where Arrow
//! rounds. The caller passes the declared precision. This module rejects
//! values that do not fit it and reproduces Arrow's rescaling:
//!
//! - Integer to decimal multiplies by `10^scale` and fails when the product
//!   leaves the precision. A negative target scale truncates toward zero.
//! - Float to decimal evaluates `round(10^scale * value)` in `f64`, rounding
//!   half away from zero, and fails on NaN, infinity, and out-of-range
//!   results. libcudf converts the exact binary value, truncating instead of
//!   rounding, and turns NaN and infinity into nulls.
//! - Decimal to decimal rounds half away from zero when digits are dropped,
//!   where libcudf truncates, and fails when the rescaled value leaves the
//!   precision.
//!
//! Sources other than integers, floats, and decimals keep libcudf's behavior.

use std::sync::Arc;

use arrow::datatypes::{Decimal32Type, Decimal64Type, Decimal128Type, DecimalType};

use super::{contains_nan, float_scalar_range, integral_range, libcudf_cast, round_half_up_impl, scalar_raw_range};
use crate::{
    core::{
        column::{Column, ColumnView, copy_view_in},
        error::{Error, ErrorCode, Result},
        scalar::{Scalar, ScalarValue},
        types::{DataType, TypeId},
    },
    ops::{
        binary::{self, BinaryOperator},
        reduction,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// `f64` conversion of 2^127, the exclusive bound of values `i128` holds.
const I128_EXCLUSIVE_BOUND_AS_F64: f64 = i128::MAX as f64;

/// The declared type of a cast into a decimal.
struct DecimalTarget {
    id: TypeId,
    /// Digits after the decimal point. cuDF stores the negated value as scale.
    scale: i32,
    precision: u8,
    /// Largest precision the cuDF storage type holds.
    storage_precision: u8,
    /// Largest raw magnitude the precision allows, `10^precision - 1`.
    max_raw: u128,
}

/// How the cast reads its input.
#[derive(Clone, Copy)]
enum Source {
    Integer,
    Float,
    /// A decimal with this many digits after the decimal point.
    Decimal {
        scale: i32,
    },
}

fn storage_precision(id: TypeId) -> Option<u8> {
    match id {
        TypeId::Decimal32 => Some(Decimal32Type::MAX_PRECISION),
        TypeId::Decimal64 => Some(Decimal64Type::MAX_PRECISION),
        TypeId::Decimal128 => Some(Decimal128Type::MAX_PRECISION),
        _ => None,
    }
}

fn pow10(exponent: u32) -> Option<i128> {
    10i128.checked_pow(exponent)
}

/// `raw / divisor` rounded half away from zero, as Arrow rescales decimals.
fn divide_rounding_half_away(raw: i128, divisor: i128) -> i128 {
    let quotient = raw / divisor;
    let remainder = raw % divisor;
    let half = divisor / 2;
    if raw >= 0 && remainder >= half {
        quotient + 1
    } else if raw < 0 && remainder <= -half {
        quotient - 1
    } else {
        quotient
    }
}

impl DecimalTarget {
    fn new(output_type: &DataType, precision: u8, operation: &'static str) -> Result<Self> {
        let id = output_type.id();
        let Some(storage_precision) = storage_precision(id) else {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "a decimal precision applies only to a decimal cast target",
            )
            .with_detail("reason", "decimal_precision_requires_decimal_target")
            .with_detail("target_type_id", id.name()));
        };
        if precision == 0 || precision > storage_precision {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "decimal precision is outside the range of the target storage type",
            )
            .with_detail("reason", "invalid_decimal_precision")
            .with_detail("target_type_id", id.name())
            .with_detail("target_precision", precision.to_string())
            .with_detail("storage_precision", storage_precision.to_string()));
        }
        Ok(Self {
            id,
            scale: output_type.scale().saturating_neg(),
            precision,
            storage_precision,
            max_raw: 10u128.pow(u32::from(precision)) - 1,
        })
    }

    fn contains(&self, raw: i128) -> bool {
        raw.unsigned_abs() <= self.max_raw
    }

    /// Arrow fails a cast whose power-of-ten factor overflows the target
    /// storage type, whatever the values are.
    fn scale_factor_fits(&self, source: Source) -> bool {
        let storage_precision = i64::from(self.storage_precision);
        match source {
            Source::Integer => i64::from(self.scale).abs() <= storage_precision,
            Source::Decimal { scale } => i64::from(self.scale) - i64::from(scale) <= storage_precision,
            Source::Float => true,
        }
    }

    /// The raw value the target stores for an integer or decimal source value,
    /// or `None` when scaling it up overflows.
    fn rescaled(&self, source: Source, raw: i128) -> Option<i128> {
        let source_scale = match source {
            Source::Decimal { scale } => scale,
            Source::Integer | Source::Float => 0,
        };
        let delta = i64::from(self.scale) - i64::from(source_scale);
        let digits = u32::try_from(delta.unsigned_abs()).ok()?;
        if delta >= 0 {
            return raw.checked_mul(pow10(digits)?);
        }
        // A divisor beyond i128 exceeds every raw value, so the result is zero.
        let Some(divisor) = pow10(digits) else {
            return Some(0);
        };
        Some(match source {
            Source::Integer => raw / divisor,
            Source::Decimal { .. } | Source::Float => divide_rounding_half_away(raw, divisor),
        })
    }

    fn accepts(&self, source: Source, raw: i128) -> bool {
        self.rescaled(source, raw).is_some_and(|rescaled| self.contains(rescaled))
    }

    fn accepts_float(&self, value: f64) -> bool {
        let scaled = (10_f64.powi(self.scale) * value).round();
        scaled.abs() < I128_EXCLUSIVE_BOUND_AS_F64 && self.contains(scaled as i128)
    }
}

impl Source {
    fn of(input_type: &DataType) -> Option<Self> {
        let id = input_type.id();
        if id.is_float() {
            Some(Self::Float)
        } else if id.is_decimal() {
            Some(Self::Decimal { scale: input_type.scale().saturating_neg() })
        } else {
            integral_range(id).map(|_| Self::Integer)
        }
    }

    /// The raw values every column of the type can hold, or `None` for floats.
    fn storage_raw_range(self, id: TypeId) -> Option<(i128, i128)> {
        match self {
            Self::Integer => integral_range(id),
            Self::Decimal { .. } => {
                let bound = pow10(u32::from(storage_precision(id)?))? - 1;
                Some((-bound, bound))
            },
            Self::Float => None,
        }
    }
}

fn out_of_range_error(
    operation: &'static str,
    source_type: &DataType,
    target: &DecimalTarget,
    reason: &'static str,
) -> Error {
    Error::new_with(
        ErrorCode::ValueOutOfRange,
        operation,
        "cast input cannot be represented in the target decimal precision",
    )
    .with_detail("reason", reason)
    .with_detail("source_type_id", source_type.id().name())
    .with_detail("target_type_id", target.id.name())
    .with_detail("target_precision", target.precision.to_string())
    .with_detail("target_scale", target.scale.to_string())
}

/// Rejects input values that a DataFusion `CAST` (non-safe mode) to the
/// target decimal would reject.
///
/// Scaling and rounding never decrease a value, so the minimum and maximum of
/// the valid rows decide whether every row fits. A cast whose whole source
/// range fits needs no reduction.
fn validate_input(
    input: &ColumnView<'_>,
    input_type: &DataType,
    source: Source,
    target: &DecimalTarget,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<()> {
    let out_of_range = |reason: &'static str| out_of_range_error(operation, input_type, target, reason);
    if !target.scale_factor_fits(source) {
        return Err(out_of_range("decimal_scale_overflow"));
    }
    if let Some((low, high)) = source.storage_raw_range(input_type.id())
        && target.accepts(source, low)
        && target.accepts(source, high)
    {
        return Ok(());
    }
    if matches!(source, Source::Float) && contains_nan(input, output_domain, exec_options, operation)? {
        return Err(out_of_range("nan"));
    }
    let (min_scalar, max_scalar) = reduction::minmax_impl(input, output_domain, exec_options, operation)?;
    if !min_scalar.is_valid()? {
        return Ok(());
    }
    let fits = if matches!(source, Source::Float) {
        let (min, max) = float_scalar_range(input_type.id(), &min_scalar, &max_scalar)?;
        target.accepts_float(min) && target.accepts_float(max)
    } else {
        let (min, max) = scalar_raw_range(input_type.id(), &min_scalar, &max_scalar)?;
        target.accepts(source, min) && target.accepts(source, max)
    };
    if fits { Ok(()) } else { Err(out_of_range("decimal_precision_overflow")) }
}

/// Casts `input` to `output_type`, a decimal type declared with `precision`.
///
/// Fails with `ValueOutOfRange` when a valid row does not fit the precision
/// after rescaling.
pub(crate) fn cast_to_decimal_impl(
    input: &ColumnView<'_>,
    output_type: &DataType,
    precision: u8,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let target = DecimalTarget::new(output_type, precision, operation)?;
    let input_type = input.data_type()?;
    let Some(source) = Source::of(&input_type) else {
        return libcudf_cast(input, &input_type, output_type, output_domain, exec_options, operation);
    };
    validate_input(input, &input_type, source, &target, output_domain, exec_options, operation)?;
    match source {
        Source::Float => {
            float_to_decimal(input, &input_type, output_type, &target, output_domain, exec_options, operation)
        },
        Source::Decimal { scale } if scale > target.scale => {
            let rounded = round_half_up_impl(input, target.scale, output_domain, exec_options, operation)?;
            if input_type.id() == output_type.id() {
                return Ok(rounded);
            }
            let rounded_type = DataType::fixed_point(input_type.id(), output_type.scale())?;
            libcudf_cast(&rounded.view()?, &rounded_type, output_type, output_domain, exec_options, operation)
        },
        _ if input_type == *output_type => {
            copy_view_in(input, output_domain, exec_options).map_err(|error| error.reanchor(operation))
        },
        _ => libcudf_cast(input, &input_type, output_type, output_domain, exec_options, operation),
    }
}

/// Evaluates Arrow's `round(10^scale * value)` on the device, then moves the
/// rounded integer into the decimal type.
fn float_to_decimal(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    target: &DecimalTarget,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let float64 = DataType::new(TypeId::Float64);
    let widened;
    let widened_view;
    let values = if input_type.id() == TypeId::Float64 {
        input
    } else {
        widened = libcudf_cast(input, input_type, &float64, output_domain, exec_options, operation)?;
        widened_view = widened.view()?;
        &widened_view
    };
    let factor = Scalar::from_value_in_domain(10_f64.powi(target.scale), output_domain)?;
    let scaled = binary::binary_operation_vs_impl(
        values,
        &factor,
        BinaryOperator::Mul,
        &float64,
        output_domain,
        exec_options,
        operation,
    )?;
    let rounded = round_half_up_impl(&scaled.view()?, 0, output_domain, exec_options, operation)?;
    // The rounded values are integers, which libcudf converts exactly at scale 0.
    let whole_type = DataType::fixed_point(TypeId::Decimal128, 0)?;
    let whole = libcudf_cast(&rounded.view()?, &float64, &whole_type, output_domain, exec_options, operation)?;
    // libcudf fixed-point division divides the raw values and labels the
    // quotient with the requested scale. Dividing by a raw 1 at scale
    // `target.scale` therefore moves the label from scale 0 to the target
    // scale and keeps the digits.
    let unit = ScalarValue::decimal(1, DataType::fixed_point(TypeId::Decimal128, target.scale)?)
        .and_then(|unit| Scalar::from_scalar_value_in_domain(&unit, output_domain))?;
    let wide_type = DataType::fixed_point(TypeId::Decimal128, output_type.scale())?;
    let wide = binary::binary_operation_vs_impl(
        &whole.view()?,
        &unit,
        BinaryOperator::Div,
        &wide_type,
        output_domain,
        exec_options,
        operation,
    )?;
    if output_type.id() == TypeId::Decimal128 {
        return Ok(wide);
    }
    libcudf_cast(&wide.view()?, &wide_type, output_type, output_domain, exec_options, operation)
}
