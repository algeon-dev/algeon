//! GPU-accelerated unary operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView, copy_view_in},
        error::{self, Error, ErrorCode, Result},
        scalar::Scalar,
        types::{DataType, TypeId},
    },
    frame::series::Series,
    ops::{reduction, strings as string_ops},
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

mod decimal_cast;

pub(crate) use decimal_cast::cast_to_decimal_impl;

/// Unary operators matching `cudf::unary_operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum UnaryOperator {
    /// Trigonometric sine.
    Sin = 0,
    /// Trigonometric cosine.
    Cos,
    /// Trigonometric tangent.
    Tan,
    /// Inverse sine (arcsin).
    ArcSin,
    /// Inverse cosine (arccos).
    ArcCos,
    /// Inverse tangent (arctan).
    ArcTan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    ArcSinh,
    /// Inverse hyperbolic cosine.
    ArcCosh,
    /// Inverse hyperbolic tangent.
    ArcTanh,
    /// Natural exponential (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Square root.
    Sqrt,
    /// Cube root.
    Cbrt,
    /// Ceiling; smallest integer >= x.
    Ceil,
    /// Floor; largest integer <= x.
    Floor,
    /// Absolute value.
    Abs,
    /// Round to nearest integer (half-to-even / banker's rounding).
    Rint,
    /// Population count; number of `1`-bits in the binary representation.
    BitCount,
    /// Bitwise inversion (flips every bit). For integer types.
    BitInvert,
    /// Logical NOT. For `Bool8` columns: `true` becomes `false` and vice versa.
    Not,
    /// Arithmetic negation (`-x`). For numeric types.
    Negate,
}

impl UnaryOperator {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_unary_operator_t {
        match self {
            UnaryOperator::Sin => cudf_sys::CUDF_UNARY_SIN,
            UnaryOperator::Cos => cudf_sys::CUDF_UNARY_COS,
            UnaryOperator::Tan => cudf_sys::CUDF_UNARY_TAN,
            UnaryOperator::ArcSin => cudf_sys::CUDF_UNARY_ARCSIN,
            UnaryOperator::ArcCos => cudf_sys::CUDF_UNARY_ARCCOS,
            UnaryOperator::ArcTan => cudf_sys::CUDF_UNARY_ARCTAN,
            UnaryOperator::Sinh => cudf_sys::CUDF_UNARY_SINH,
            UnaryOperator::Cosh => cudf_sys::CUDF_UNARY_COSH,
            UnaryOperator::Tanh => cudf_sys::CUDF_UNARY_TANH,
            UnaryOperator::ArcSinh => cudf_sys::CUDF_UNARY_ARCSINH,
            UnaryOperator::ArcCosh => cudf_sys::CUDF_UNARY_ARCCOSH,
            UnaryOperator::ArcTanh => cudf_sys::CUDF_UNARY_ARCTANH,
            UnaryOperator::Exp => cudf_sys::CUDF_UNARY_EXP,
            UnaryOperator::Log => cudf_sys::CUDF_UNARY_LOG,
            UnaryOperator::Sqrt => cudf_sys::CUDF_UNARY_SQRT,
            UnaryOperator::Cbrt => cudf_sys::CUDF_UNARY_CBRT,
            UnaryOperator::Ceil => cudf_sys::CUDF_UNARY_CEIL,
            UnaryOperator::Floor => cudf_sys::CUDF_UNARY_FLOOR,
            UnaryOperator::Abs => cudf_sys::CUDF_UNARY_ABS,
            UnaryOperator::Rint => cudf_sys::CUDF_UNARY_RINT,
            UnaryOperator::BitCount => cudf_sys::CUDF_UNARY_BIT_COUNT,
            UnaryOperator::BitInvert => cudf_sys::CUDF_UNARY_BIT_INVERT,
            UnaryOperator::Not => cudf_sys::CUDF_UNARY_NOT,
            UnaryOperator::Negate => cudf_sys::CUDF_UNARY_NEGATE,
        }
    }
}

pub(crate) fn unary_operation_impl(
    input: &ColumnView<'_>,
    op: UnaryOperator,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let op_ffi = op.to_raw();
    let status = unsafe { cudf_sys::cudf_unary_operation_ex(input.raw_handle(), op_ffi, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("operator", format!("{:?}", op).to_lowercase()), ("execution_authority", "retained_domain".to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn round_half_up_impl(
    input: &ColumnView<'_>,
    decimal_places: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_round_half_up_ex(input.raw_handle(), decimal_places, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("decimal_places", decimal_places.to_string()),
            ("rounding_method", "half_up".to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn is_null_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_null_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Return a boolean column: `true` where elements are valid (non-null).
pub(crate) fn is_valid(input: &ColumnView<'_>) -> Result<Column> {
    let ctx = Exec::RETAINED;
    let execution = ResolvedExecution::unary(input.execution_domain(), ctx, "unary::is_valid")?;
    is_valid_impl(input, execution.output_domain(), execution.exec_options(), "unary::is_valid")
}

pub(crate) fn is_valid_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_valid_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Cast a column to the specified data type.
pub(crate) fn cast(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let ctx = Exec::RETAINED;
    let execution = ResolvedExecution::unary(input.execution_domain(), ctx, "unary::cast")?;
    cast_impl(input, output_type, execution.output_domain(), execution.exec_options(), "unary::cast")
}

pub(crate) fn cast_impl(
    input: &ColumnView<'_>,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let input_type = input.data_type()?;
    if input_type == *output_type {
        return copy_view_in(input, output_domain, exec_options).map_err(|error| error.reanchor(operation));
    }
    if input_type.id().is_integral() && output_type.id() == TypeId::String {
        return string_ops::from_integers_impl(input, output_domain, exec_options, operation);
    }
    validate_integral_target_cast_input(input, &input_type, output_type, output_domain, exec_options, operation)?;
    libcudf_cast(input, &input_type, output_type, output_domain, exec_options, operation)
}

fn libcudf_cast(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for("unary::cast")?;
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). dtype_handle was returned by
    // a successful to_raw_handle_for call.
    let status = unsafe { cudf_sys::cudf_cast_ex(input.raw_handle(), dtype_handle, &mut out, exec_options) };
    let status_result = error::check_status_in_with_details(
        status,
        operation,
        [
            ("input_type_id", format!("{:?}", input_type.id()).to_lowercase()),
            ("input_type_scale", input_type.scale().to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
            ("output_type_scale", output_type.scale().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    );
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    status_result?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

// libcudf's fixed-point scalar uses RMM's default device-allocation alignment.
const DECIMAL_RESCALE_SCALAR_RESERVATION_BYTES: usize = 256;

/// Returns the exact scalar workspace reservation used by libcudf's same-width decimal rescale.
///
/// Returns `None` outside that dispatch and `Some(0)` when the decimal scale is unchanged.
pub fn decimal_rescale_temporary_workspace_reservation_bytes(
    input_type: &DataType,
    output_type: &DataType,
) -> Option<usize> {
    if input_type.id() != output_type.id() || !input_type.id().is_decimal() {
        return None;
    }
    Some(if input_type.scale() == output_type.scale() { 0 } else { DECIMAL_RESCALE_SCALAR_RESERVATION_BYTES })
}

/// Inclusive value range of an integer type, or `None` for any other type.
fn integral_range(id: TypeId) -> Option<(i128, i128)> {
    Some(match id {
        TypeId::Int8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        TypeId::Int16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        TypeId::Int32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        TypeId::Int64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        TypeId::UInt8 => (0, i128::from(u8::MAX)),
        TypeId::UInt16 => (0, i128::from(u16::MAX)),
        TypeId::UInt32 => (0, i128::from(u32::MAX)),
        TypeId::UInt64 => (0, i128::from(u64::MAX)),
        _ => return None,
    })
}

/// Rejects input values that a DataFusion `CAST` (non-safe mode) to an integer
/// type would reject, before libcudf wraps or saturates them silently.
///
/// A cast whose source range fits the target needs no check. Every other
/// integer, floating-point, and fixed-point source takes one min/max reduction
/// over its valid rows, and floating-point sources add a NaN check.
/// Floating-point and fixed-point values truncate toward zero before the range
/// comparison.
fn validate_integral_target_cast_input(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<()> {
    let source_id = input_type.id();
    let Some((target_min, target_max)) = integral_range(output_type.id()) else {
        return Ok(());
    };
    match integral_range(source_id) {
        Some((source_min, source_max)) if source_min >= target_min && source_max <= target_max => return Ok(()),
        None if !source_id.is_float() && !source_id.is_decimal() => return Ok(()),
        _ => {},
    }
    let out_of_range = |reason: &'static str| cast_input_out_of_range_error(operation, input_type, output_type, reason);

    if source_id.is_float() && contains_nan(input, output_domain, exec_options, operation)? {
        return Err(out_of_range("nan"));
    }
    let (min_scalar, max_scalar) = reduction::minmax_impl(input, output_domain, exec_options, operation)?;
    if !min_scalar.is_valid()? {
        return Ok(());
    }
    let (below, above) = if source_id.is_float() {
        let (min_value, max_value) = float_scalar_range(source_id, &min_scalar, &max_scalar)?;
        // The target minimum is zero or a negated power of two and the
        // exclusive maximum is a power of two, so both convert exactly.
        (min_value.trunc() < target_min as f64, max_value.trunc() >= target_max as f64 + 1.0)
    } else {
        let (min_value, max_value) = integral_scalar_range(source_id, input_type.scale(), &min_scalar, &max_scalar)?;
        (min_value < target_min, max_value > target_max)
    };
    if below {
        return Err(out_of_range("below_target_range"));
    }
    if above {
        return Err(out_of_range("above_target_range"));
    }
    Ok(())
}

/// Whether any valid row of a floating-point column is NaN.
fn contains_nan(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<bool> {
    let nan_mask = is_nan_impl(input, output_domain, exec_options, operation)?;
    Series::from_column(nan_mask)?.on(Exec::RETAINED).any_value()
}

/// The min and max scalars of a Float32 or Float64 column as `f64`.
fn float_scalar_range(id: TypeId, min: &Scalar, max: &Scalar) -> Result<(f64, f64)> {
    if id == TypeId::Float32 {
        Ok((f64::from(min.value::<f32>()?), f64::from(max.value::<f32>()?)))
    } else {
        Ok((min.value::<f64>()?, max.value::<f64>()?))
    }
}

/// Integer value of the min and max scalars of an integer or fixed-point
/// column. Fixed-point values scale to their integer part.
fn integral_scalar_range(id: TypeId, scale: i32, min: &Scalar, max: &Scalar) -> Result<(i128, i128)> {
    let (min_raw, max_raw) = scalar_raw_range(id, min, max)?;
    if id.is_decimal() {
        Ok((fixed_point_integral_value(min_raw, scale), fixed_point_integral_value(max_raw, scale)))
    } else {
        Ok((min_raw, max_raw))
    }
}

/// The min and max scalars of an integer or fixed-point column as stored: a
/// fixed-point value is its raw integer, not scaled.
fn scalar_raw_range(id: TypeId, min: &Scalar, max: &Scalar) -> Result<(i128, i128)> {
    macro_rules! widen {
        ($ty:ty) => {
            (i128::from(min.value::<$ty>()?), i128::from(max.value::<$ty>()?))
        };
    }
    Ok(match id {
        TypeId::Int8 => widen!(i8),
        TypeId::Int16 => widen!(i16),
        TypeId::Int32 => widen!(i32),
        TypeId::Int64 => widen!(i64),
        TypeId::UInt8 => widen!(u8),
        TypeId::UInt16 => widen!(u16),
        TypeId::UInt32 => widen!(u32),
        TypeId::UInt64 => widen!(u64),
        TypeId::Decimal32 => (i128::from(min.decimal32_raw_value()?), i128::from(max.decimal32_raw_value()?)),
        TypeId::Decimal64 => (i128::from(min.decimal64_raw_value()?), i128::from(max.decimal64_raw_value()?)),
        _ => (min.decimal128_raw_value()?, max.decimal128_raw_value()?),
    })
}

/// Integer part of `raw * 10^scale` truncated toward zero, saturating at the
/// `i128` limits when the product overflows.
fn fixed_point_integral_value(raw: i128, scale: i32) -> i128 {
    let factor = 10i128.checked_pow(scale.unsigned_abs());
    let value = if scale < 0 {
        Some(factor.map_or(0, |factor| raw / factor))
    } else {
        match factor {
            Some(factor) => raw.checked_mul(factor),
            None => (raw == 0).then_some(0),
        }
    };
    value.unwrap_or(if raw < 0 { i128::MIN } else { i128::MAX })
}

fn cast_input_out_of_range_error(
    operation: &'static str,
    source_type: &DataType,
    target_type: &DataType,
    reason: &'static str,
) -> Error {
    Error::new_with(
        ErrorCode::ValueOutOfRange,
        operation,
        "cast input cannot be represented in the target integer type",
    )
    .with_detail("reason", reason)
    .with_detail("source_type_id", format!("{:?}", source_type.id()).to_lowercase())
    .with_detail("target_type_id", format!("{:?}", target_type.id()).to_lowercase())
}

/// Check whether a cast between two data types is supported.
pub fn is_supported_cast(from: &DataType, to: &DataType) -> Result<bool> {
    let from_handle = from.to_raw_handle_for("unary::is_supported_cast")?;
    let to_handle = to.to_raw_handle_for("unary::is_supported_cast")?;
    let mut out: i32 = 0;
    // SAFETY: from_handle and to_handle were returned by successful
    // to_raw_handle_for calls.
    let status = unsafe { cudf_sys::cudf_is_supported_cast(from_handle, to_handle, &mut out) };
    let status_result = error::check_status_in_with_details(
        status,
        "unary::is_supported_cast",
        [
            ("from_type_id", format!("{:?}", from.id()).to_lowercase()),
            ("to_type_id", format!("{:?}", to.id()).to_lowercase()),
        ],
    );
    // SAFETY: Both handles are valid and must be destroyed.
    unsafe {
        cudf_sys::cudf_data_type_destroy(from_handle);
        cudf_sys::cudf_data_type_destroy(to_handle);
    }
    status_result?;
    Ok(out != 0 || (from.id().is_integral() && to.id() == TypeId::String))
}

pub(crate) fn is_nan_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_nan_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn is_not_nan_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_is_not_nan_ex(input.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::memory::DeviceMemoryResource,
        runtime::{CudaStream, DeviceId, Runtime},
    };

    /// A same-dtype cast with an explicit output domain copies on that
    /// domain. The input budget fits the input once but never twice, so a
    /// copy charged to the input MR is rejected while the output-domain copy
    /// succeeds.
    #[test]
    fn same_dtype_cast_copies_on_the_output_domain() {
        let device = DeviceId::current().expect("test device");
        let input_budget = DeviceMemoryResource::cuda_on(device)
            .expect("input upstream MR")
            .budget_tracking_limit(12 * 1024 * 1024)
            .expect("input budget");
        let input_runtime = Runtime::builder()
            .with_stream(CudaStream::new().expect("input stream"))
            .with_memory_resource(input_budget)
            .build()
            .expect("input runtime");
        let output_runtime = Runtime::builder()
            .with_stream(CudaStream::new().expect("output stream"))
            .with_pool_memory_resource(64 * 1024 * 1024, None)
            .build()
            .expect("output runtime");

        let values: Vec<i32> = (0..2_000_000).collect();
        let series = input_runtime.series(&values).expect("input series");
        let view = series.as_column().view().expect("input view");
        let dtype = view.data_type().expect("input dtype");

        let output_domain = Arc::clone(output_runtime.execution_domain());
        let exec_options = output_domain.enter_execution().expect("output execution");
        let result = cast_impl(&view, &dtype, &output_domain, exec_options.options(), "test::same_dtype_cast")
            .expect("same-dtype cast must copy on the output domain, not the capped input budget");
        assert!(
            ExecutionDomain::same_provenance(result.execution_domain(), &output_domain),
            "cast output must retain the output domain"
        );
        let result_series = Series::from_column(result).expect("result series");
        assert_eq!(result_series.to_vec_strict::<i32>().expect("result values"), values);
    }
}
