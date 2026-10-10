//! Arrow type and cast semantics for plan validation.
//!
//! These helpers answer what the query language admits: which Arrow types a
//! plan may carry, which casts it may insert or preserve, and how a type reads
//! in a diagnostic. Whether an execution backend can represent a type is a
//! separate question, asked through
//! [`crate::backend::BackendCapabilities::supports_type`], which is why the
//! cast policy takes the backend for the one branch that needs it.

use std::collections::HashMap;

use arrow::datatypes::{
    DECIMAL128_MAX_SCALE, DataType as ArrowDataType, Decimal32Type, Decimal64Type, Decimal128Type, Field, FieldRef,
    Fields, TimeUnit, validate_decimal_precision_and_scale,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::{backend::BackendCapabilities, expr::BinaryOp, validation::temporal_duration_binary_output_type};

/// Returns whether this Arrow type is a timestamp carrying timezone metadata.
pub fn is_timestamp_with_timezone(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Timestamp(_, Some(_)))
}

/// Returns whether this Arrow type is a list admitted only for carry/projection.
pub fn is_supported_list_carry_type(data_type: &ArrowDataType) -> bool {
    match data_type {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) => is_supported_basic_list_child(field),
        ArrowDataType::FixedSizeList(field, dimension) => *dimension > 0 && is_supported_fixed_size_list_child(field),
        _ => false,
    }
}

/// Returns whether this Arrow type is a narrow decimal admitted only for carry/projection.
pub fn is_narrow_decimal_carry_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Decimal32(precision, scale) if is_supported_decimal32(*precision, *scale)
    ) || matches!(
        data_type,
        ArrowDataType::Decimal64(precision, scale) if is_supported_decimal64(*precision, *scale)
    )
}

/// Returns the physical byte width for Arrow scalar fixed-width values used by
/// native sizing heuristics.
///
/// This is not a general Arrow memory-layout classifier: it intentionally
/// preserves the native execution paths that do not size `Float16` or
/// fixed-size binary values as fixed-width inputs.
pub fn arrow_fixed_width_byte_size(data_type: &ArrowDataType) -> Option<u64> {
    match data_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some(2),
        ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_)
        | ArrowDataType::Decimal32(_, _) => Some(4),
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_)
        | ArrowDataType::Decimal64(_, _) => Some(8),
        ArrowDataType::Decimal128(_, _) => Some(16),
        ArrowDataType::Decimal256(_, _) => Some(32),
        ArrowDataType::Null
        | ArrowDataType::Float16
        | ArrowDataType::Interval(_)
        | ArrowDataType::Binary
        | ArrowDataType::FixedSizeBinary(_)
        | ArrowDataType::LargeBinary
        | ArrowDataType::BinaryView
        | ArrowDataType::Utf8
        | ArrowDataType::LargeUtf8
        | ArrowDataType::Utf8View
        | ArrowDataType::List(_)
        | ArrowDataType::ListView(_)
        | ArrowDataType::FixedSizeList(_, _)
        | ArrowDataType::LargeList(_)
        | ArrowDataType::LargeListView(_)
        | ArrowDataType::Struct(_)
        | ArrowDataType::Union(_, _)
        | ArrowDataType::Dictionary(_, _)
        | ArrowDataType::Map(_, _)
        | ArrowDataType::RunEndEncoded(_, _) => None,
    }
}

/// Returns whether an Arrow scalar type is fixed-width under the native sizing policy.
///
/// See [`arrow_fixed_width_byte_size`] for the policy boundary. In particular,
/// some physically fixed-width Arrow types remain excluded for behavior
/// compatibility with existing native execution paths.
pub fn is_arrow_fixed_width(data_type: &ArrowDataType) -> bool {
    arrow_fixed_width_byte_size(data_type).is_some()
}

/// Human-readable Arrow type name for client-facing diagnostics.
pub fn diagnostic_arrow_type(data_type: &ArrowDataType) -> String {
    match data_type {
        ArrowDataType::List(field) => format!("List({})", diagnostic_arrow_type(field.data_type())),
        ArrowDataType::LargeList(field) => format!("LargeList({})", diagnostic_arrow_type(field.data_type())),
        ArrowDataType::FixedSizeList(field, size) => {
            format!("FixedSizeList({}, size={size})", diagnostic_arrow_field(field))
        },
        ArrowDataType::Struct(fields) => diagnostic_arrow_struct(fields),
        ArrowDataType::Map(field, sorted) => {
            format!("Map({}, {})", diagnostic_arrow_field(field), if *sorted { "sorted" } else { "unsorted" })
        },
        ArrowDataType::Dictionary(key, value) => {
            format!("Dictionary({}, {})", diagnostic_arrow_type(key), diagnostic_arrow_type(value))
        },
        ArrowDataType::Timestamp(unit, Some(timezone)) => {
            format!("Timestamp({unit:?}, timezone={timezone})")
        },
        ArrowDataType::Timestamp(unit, None) => format!("Timestamp({unit:?})"),
        ArrowDataType::Duration(unit) => format!("Duration({unit:?})"),
        ArrowDataType::Time32(unit) => format!("Time32({unit:?})"),
        ArrowDataType::Time64(unit) => format!("Time64({unit:?})"),
        ArrowDataType::Decimal32(precision, scale) => format!("Decimal32({precision}, {scale})"),
        ArrowDataType::Decimal64(precision, scale) => format!("Decimal64({precision}, {scale})"),
        ArrowDataType::Decimal128(precision, scale) => format!("Decimal128({precision}, {scale})"),
        ArrowDataType::Decimal256(precision, scale) => format!("Decimal256({precision}, {scale})"),
        _ => format!("{data_type:?}"),
    }
}

fn diagnostic_arrow_field(field: &Field) -> String {
    let nullability = if field.is_nullable() { "" } else { "non-null " };
    format!("\"{}\": {nullability}{}", field.name(), diagnostic_arrow_type(field.data_type()))
}

fn diagnostic_arrow_struct(fields: &Fields) -> String {
    let fields = fields.iter().map(|field| diagnostic_arrow_field(field)).collect::<Vec<_>>().join(", ");
    format!("Struct({fields})")
}

pub(crate) fn arrow_fields_match_except_name(left: &Field, right: &Field) -> bool {
    left.is_nullable() == right.is_nullable()
        && arrow_field_metadata_matches_except_parquet_field_id(left.metadata(), right.metadata())
        && arrow_data_types_match_except_list_child_names(left.data_type(), right.data_type())
}

pub(crate) fn arrow_data_types_match_except_list_child_names(left: &ArrowDataType, right: &ArrowDataType) -> bool {
    match (left, right) {
        (ArrowDataType::List(left_field), ArrowDataType::List(right_field))
        | (ArrowDataType::LargeList(left_field), ArrowDataType::LargeList(right_field)) => {
            arrow_fields_match_except_name(left_field.as_ref(), right_field.as_ref())
        },
        (
            ArrowDataType::FixedSizeList(left_field, left_dimension),
            ArrowDataType::FixedSizeList(right_field, right_dimension),
        ) => {
            left_dimension == right_dimension
                && arrow_fields_match_except_name(left_field.as_ref(), right_field.as_ref())
        },
        _ => left == right,
    }
}

fn is_supported_basic_list_child(field: &FieldRef) -> bool {
    matches!(
        field.data_type(),
        ArrowDataType::Boolean
            | ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64
            | ArrowDataType::Utf8
            | ArrowDataType::LargeUtf8
            | ArrowDataType::Utf8View
    )
}

fn is_supported_fixed_size_list_child(field: &FieldRef) -> bool {
    matches!(
        field.data_type(),
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64
    )
}

pub(crate) fn arrow_field_metadata_matches_except_parquet_field_id(
    left: &HashMap<String, String>,
    right: &HashMap<String, String>,
) -> bool {
    let visible_len = |metadata: &HashMap<String, String>| {
        metadata.keys().filter(|key| key.as_str() != PARQUET_FIELD_ID_META_KEY).count()
    };
    visible_len(left) == visible_len(right)
        && left
            .iter()
            .filter(|(key, _value)| key.as_str() != PARQUET_FIELD_ID_META_KEY)
            .all(|(key, value)| right.get(key) == Some(value))
}

/// Converts Arrow's decimal scale to cuDF fixed-point scale convention.
pub fn arrow_decimal_scale_to_cudf_scale(scale: i8) -> i32 {
    debug_assert!(
        scale <= DECIMAL128_MAX_SCALE,
        "Decimal128 support policy must reject positive scales larger than Arrow's Decimal128 max scale"
    );
    -i32::from(scale)
}

/// Returns whether the native engine supports an explicit cast.
pub fn is_supported_arrow_cast(
    source_type: &ArrowDataType,
    target_type: &ArrowDataType,
    backend: &dyn BackendCapabilities,
) -> bool {
    !is_narrow_decimal_carry_type(source_type)
        && !is_narrow_decimal_carry_type(target_type)
        && (is_supported_arrow_implicit_cast(source_type, target_type, backend)
            || is_supported_decimal_to_float64_cast(source_type, target_type)
            || is_supported_numeric_to_decimal_cast(source_type, target_type)
            || is_supported_decimal_to_integral_cast(source_type, target_type)
            || is_supported_range_checked_integral_cast(source_type, target_type)
            || is_supported_integral_to_string_cast(source_type, target_type)
            || is_supported_temporal_count_cast(source_type, target_type)
            || is_supported_float64_to_float32_cast(source_type, target_type))
}

/// Returns whether the cuDF AST evaluates an explicit cast.
///
/// The AST casts integer, floating-point, and decimal values to `Int64`,
/// `UInt64`, and `Float64` without the range, `NaN`, and precision checks
/// DataFusion applies, and it has no cast over temporal values. A cast is
/// therefore AST-safe only when it is supported, takes a numeric or decimal
/// source, and cannot fail at runtime.
pub fn is_supported_ast_cast(
    source_type: &ArrowDataType,
    target_type: &ArrowDataType,
    backend: &dyn BackendCapabilities,
) -> bool {
    matches!(target_type, ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64)
        && (source_type.is_integer()
            || matches!(source_type, ArrowDataType::Float32 | ArrowDataType::Float64 | ArrowDataType::Decimal128(_, _)))
        && is_supported_arrow_cast(source_type, target_type, backend)
        && !cast_can_fail_at_runtime(source_type, target_type)
}

/// Returns whether native evaluation of a supported explicit cast can fail on
/// the values it reads.
///
/// An integer, floating-point, or `Decimal128` source cast to an integer fails
/// the query when a valid value, truncated toward zero, is outside the target
/// range, as DataFusion `CAST` does. An integer source whose whole range fits
/// the target cannot fail.
///
/// An integer, floating-point, or `Decimal128` source cast to `Decimal128`
/// fails when a valid value does not fit the target precision after rescaling.
/// Floating-point sources always can, because NaN and infinity never fit. An
/// integer or decimal source cannot fail when every value its type allows
/// fits, the same condition under which Arrow skips the check.
///
/// A timestamp or duration cast to `Int64` cannot fail, because it reads the
/// stored count unchanged.
///
/// A timestamp, duration, or `Date32` cast to a finer unit multiplies the
/// stored count and fails when a valid product leaves `Int64`, which every
/// source count can do except the `Int32` day count of a `Date32` at a unit
/// no finer than milliseconds. A cast to a coarser unit divides and cannot
/// fail.
pub fn cast_can_fail_at_runtime(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    if source_type == target_type {
        return false;
    }
    if let Some(overflows) = temporal_unit_cast_overflows(source_type, target_type) {
        return overflows;
    }
    if let ArrowDataType::Decimal128(target_precision, target_scale) = target_type {
        return (is_supported_numeric_to_decimal_cast(source_type, target_type)
            || is_supported_decimal_to_decimal_cast(source_type, target_type))
            && !decimal_cast_always_fits(source_type, *target_precision, *target_scale);
    }
    target_type.is_integer()
        && (is_supported_decimal_to_integral_cast(source_type, target_type)
            || is_supported_range_checked_integral_cast(source_type, target_type))
        && !is_supported_numeric_widening_cast(source_type, target_type)
}

/// For a cast between two timestamp units, two duration units, or from a
/// `Date32` to a timestamp, whether the scaled stored count can leave `Int64`.
/// `None` for any other cast.
fn temporal_unit_cast_overflows(source_type: &ArrowDataType, target_type: &ArrowDataType) -> Option<bool> {
    const NANOSECONDS_PER_DAY: i128 = 86_400_000_000_000;
    let unit_nanoseconds = |unit: &TimeUnit| match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    let int64_counts = (i128::from(i64::MIN), i128::from(i64::MAX));
    let (source_nanoseconds, (lowest_count, highest_count), target_nanoseconds) = match (source_type, target_type) {
        (ArrowDataType::Timestamp(source, None), ArrowDataType::Timestamp(target, None))
        | (ArrowDataType::Duration(source), ArrowDataType::Duration(target)) => {
            (unit_nanoseconds(source), int64_counts, unit_nanoseconds(target))
        },
        (ArrowDataType::Date32, ArrowDataType::Timestamp(target, None)) => {
            (NANOSECONDS_PER_DAY, (i128::from(i32::MIN), i128::from(i32::MAX)), unit_nanoseconds(target))
        },
        _ => return None,
    };
    let factor = source_nanoseconds / target_nanoseconds;
    Some(factor > 1 && (lowest_count * factor < int64_counts.0 || highest_count * factor > int64_counts.1))
}

/// Returns whether native evaluation of a supported temporal `+` or `-` can
/// fail on the values it reads.
///
/// Timestamp plus or minus duration, and timestamp minus timestamp, fail the
/// query in a row where both operands are valid and the result leaves `Int64`.
/// Arrow computes them with `add_checked` and `sub_checked` even for the
/// wrapping SQL operators, and DataFusion inherits that.
pub fn temporal_arithmetic_can_fail_at_runtime(op: BinaryOp, left: &ArrowDataType, right: &ArrowDataType) -> bool {
    let is_timestamp = |data_type: &ArrowDataType| matches!(data_type, ArrowDataType::Timestamp(_, None));
    (is_timestamp(left) || is_timestamp(right))
        && matches!(temporal_duration_binary_output_type(op, left, right), Ok(Some(_)))
}

/// Whether every value of an integer or `Decimal128` source type fits the
/// target precision after rescaling to the target scale.
fn decimal_cast_always_fits(source_type: &ArrowDataType, target_precision: u8, target_scale: i8) -> bool {
    let source_digits_and_scale = match source_type {
        ArrowDataType::Int8 | ArrowDataType::UInt8 => Some((3, 0)),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some((5, 0)),
        ArrowDataType::Int32 | ArrowDataType::UInt32 => Some((10, 0)),
        ArrowDataType::Int64 => Some((19, 0)),
        ArrowDataType::UInt64 => Some((20, 0)),
        ArrowDataType::Decimal128(source_precision, source_scale) => {
            Some((i32::from(*source_precision), *source_scale))
        },
        _ => None,
    };
    let Some((source_digits, source_scale)) = source_digits_and_scale else {
        return false;
    };
    let scale_change = i32::from(target_scale) - i32::from(source_scale);
    if scale_change >= 0 {
        source_digits + scale_change <= i32::from(target_precision)
    } else {
        // Dropping digits can round up into one more integer digit.
        source_digits + scale_change < i32::from(target_precision)
    }
}

/// Returns whether the native engine may insert or preserve an implicit cast.
pub fn is_supported_arrow_implicit_cast(
    source_type: &ArrowDataType,
    target_type: &ArrowDataType,
    backend: &dyn BackendCapabilities,
) -> bool {
    (source_type == target_type && backend.supports_type(source_type))
        || is_supported_numeric_widening_cast(source_type, target_type)
        || is_supported_decimal_to_decimal_cast(source_type, target_type)
        || is_supported_string_cast(source_type, target_type)
        || is_supported_timestamp_cast(source_type, target_type)
        || is_supported_duration_cast(source_type, target_type)
}

/// Returns whether a numeric cast widens without narrowing range or precision class.
pub(crate) fn is_supported_numeric_widening_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(
        (source_type, target_type),
        (ArrowDataType::Int8, ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::Int16, ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::Int32, ArrowDataType::Int64)
            | (ArrowDataType::UInt8, ArrowDataType::UInt16 | ArrowDataType::UInt32 | ArrowDataType::UInt64)
            | (ArrowDataType::UInt16, ArrowDataType::UInt32 | ArrowDataType::UInt64)
            | (ArrowDataType::UInt32, ArrowDataType::UInt64)
            | (ArrowDataType::UInt8, ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::UInt16, ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::UInt32, ArrowDataType::Int64)
            | (
                ArrowDataType::Int8
                    | ArrowDataType::Int16
                    | ArrowDataType::Int32
                    | ArrowDataType::Int64
                    | ArrowDataType::UInt8
                    | ArrowDataType::UInt16
                    | ArrowDataType::UInt32
                    | ArrowDataType::UInt64,
                ArrowDataType::Float32 | ArrowDataType::Float64,
            )
            | (ArrowDataType::Float32, ArrowDataType::Float64)
    )
}

fn is_supported_decimal_to_float64_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    match (source_type, target_type) {
        (ArrowDataType::Decimal128(source_precision, source_scale), ArrowDataType::Float64) => {
            is_supported_decimal128(*source_precision, *source_scale)
        },
        _ => false,
    }
}

fn is_supported_numeric_to_decimal_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    match (source_type, target_type) {
        (
            ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64,
            ArrowDataType::Decimal128(target_precision, target_scale),
        ) => is_supported_decimal128(*target_precision, *target_scale),
        _ => false,
    }
}

fn is_supported_decimal_to_decimal_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    match (source_type, target_type) {
        (
            ArrowDataType::Decimal128(source_precision, source_scale),
            ArrowDataType::Decimal128(target_precision, target_scale),
        ) => {
            is_supported_decimal128(*source_precision, *source_scale)
                && is_supported_decimal128(*target_precision, *target_scale)
        },
        _ => false,
    }
}

fn is_supported_decimal_to_integral_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    match (source_type, target_type) {
        (
            ArrowDataType::Decimal128(source_precision, 0),
            ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64,
        ) => is_supported_decimal128(*source_precision, 0),
        _ => false,
    }
}

/// Integer or floating-point source to an integer target. Native evaluation
/// checks the valid values against the target range and fails the query on
/// NaN, infinity, or an out-of-range value, as DataFusion `CAST` does.
fn is_supported_range_checked_integral_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    target_type.is_integer()
        && (source_type.is_integer() || matches!(source_type, ArrowDataType::Float32 | ArrowDataType::Float64))
}

/// Formats integers as plain decimal text, like Arrow.
fn is_supported_integral_to_string_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    source_type.is_integer()
        && matches!(target_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

/// Timestamp without a time zone, or duration, to `Int64`. Arrow reinterprets
/// the stored count in the type's own unit, so the value is not rescaled to a
/// common unit and a null stays null. A time zone does not change the stored
/// count, but timezone-aware timestamps are carry types and are not expression
/// values, so they stay outside the matrix.
fn is_supported_temporal_count_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(
        (source_type, target_type),
        (ArrowDataType::Timestamp(_, None) | ArrowDataType::Duration(_), ArrowDataType::Int64)
    )
}

/// IEEE round-to-nearest narrowing; values beyond the Float32 range become
/// infinities in both Arrow and libcudf.
fn is_supported_float64_to_float32_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!((source_type, target_type), (ArrowDataType::Float64, ArrowDataType::Float32))
}

fn is_supported_decimal128(precision: u8, scale: i8) -> bool {
    validate_decimal_precision_and_scale::<Decimal128Type>(precision, scale).is_ok()
}

fn is_supported_decimal32(precision: u8, scale: i8) -> bool {
    validate_decimal_precision_and_scale::<Decimal32Type>(precision, scale).is_ok()
}

fn is_supported_decimal64(precision: u8, scale: i8) -> bool {
    validate_decimal_precision_and_scale::<Decimal64Type>(precision, scale).is_ok()
}

fn is_supported_string_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(
        (source_type, target_type),
        (
            ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View,
            ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View,
        )
    )
}

fn is_supported_timestamp_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(
        (source_type, target_type),
        (ArrowDataType::Date32, ArrowDataType::Timestamp(_, None))
            | (ArrowDataType::Timestamp(_, None), ArrowDataType::Date32)
            | (ArrowDataType::Timestamp(_, None), ArrowDataType::Timestamp(_, None))
            | (
                ArrowDataType::Int8
                    | ArrowDataType::Int16
                    | ArrowDataType::Int32
                    | ArrowDataType::UInt8
                    | ArrowDataType::UInt16,
                ArrowDataType::Date32,
            )
            | (
                ArrowDataType::Int8
                    | ArrowDataType::Int16
                    | ArrowDataType::Int32
                    | ArrowDataType::Int64
                    | ArrowDataType::UInt8
                    | ArrowDataType::UInt16
                    | ArrowDataType::UInt32,
                ArrowDataType::Timestamp(_, None),
            )
    )
}

fn is_supported_duration_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!((source_type, target_type), (ArrowDataType::Duration(_), ArrowDataType::Duration(_)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Fields, TimeUnit};

    use super::{arrow_fixed_width_byte_size, diagnostic_arrow_type, is_arrow_fixed_width};

    #[test]
    fn arrow_fixed_width_byte_size_covers_scalar_fixed_width_types() {
        let cases = [
            (ArrowDataType::Boolean, 1),
            (ArrowDataType::Int8, 1),
            (ArrowDataType::UInt8, 1),
            (ArrowDataType::Int16, 2),
            (ArrowDataType::UInt16, 2),
            (ArrowDataType::Int32, 4),
            (ArrowDataType::UInt32, 4),
            (ArrowDataType::Float32, 4),
            (ArrowDataType::Date32, 4),
            (ArrowDataType::Time32(TimeUnit::Millisecond), 4),
            (ArrowDataType::Int64, 8),
            (ArrowDataType::UInt64, 8),
            (ArrowDataType::Float64, 8),
            (ArrowDataType::Date64, 8),
            (ArrowDataType::Time64(TimeUnit::Nanosecond), 8),
            (ArrowDataType::Timestamp(TimeUnit::Microsecond, None), 8),
            (ArrowDataType::Duration(TimeUnit::Nanosecond), 8),
            (ArrowDataType::Decimal32(9, 2), 4),
            (ArrowDataType::Decimal64(18, 2), 8),
            (ArrowDataType::Decimal128(10, 2), 16),
            (ArrowDataType::Decimal256(30, 2), 32),
        ];

        for (data_type, expected_size) in cases {
            assert_eq!(arrow_fixed_width_byte_size(&data_type), Some(expected_size), "{data_type:?}");
            assert!(is_arrow_fixed_width(&data_type), "{data_type:?}");
        }
    }

    #[test]
    fn arrow_fixed_width_byte_size_rejects_types_outside_native_sizing_policy() {
        let data_type = ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::Int32, true)));

        assert_eq!(arrow_fixed_width_byte_size(&ArrowDataType::Float16), None);
        assert_eq!(arrow_fixed_width_byte_size(&ArrowDataType::FixedSizeBinary(16)), None);
        assert_eq!(arrow_fixed_width_byte_size(&ArrowDataType::Utf8), None);
        assert_eq!(arrow_fixed_width_byte_size(&ArrowDataType::LargeUtf8), None);
        assert_eq!(arrow_fixed_width_byte_size(&ArrowDataType::Utf8View), None);
        assert_eq!(arrow_fixed_width_byte_size(&data_type), None);
        assert!(!is_arrow_fixed_width(&ArrowDataType::Float16));
        assert!(!is_arrow_fixed_width(&ArrowDataType::Utf8));
    }

    #[test]
    fn diagnostic_arrow_type_formats_map_without_arrow_debug_noise() {
        let data_type = ArrowDataType::Map(
            Arc::new(Field::new(
                "key_value",
                ArrowDataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                    Arc::new(Field::new("value", ArrowDataType::Int32, true)),
                ])),
                false,
            )),
            false,
        );

        assert_eq!(
            diagnostic_arrow_type(&data_type),
            "Map(\"key_value\": non-null Struct(\"key\": non-null Utf8, \"value\": Int32), unsorted)"
        );
    }

    #[test]
    fn diagnostic_arrow_type_formats_sorted_map_and_lists() {
        let data_type = ArrowDataType::Map(
            Arc::new(Field::new(
                "entries",
                ArrowDataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                    Arc::new(Field::new(
                        "value",
                        ArrowDataType::List(Arc::new(Field::new("element", ArrowDataType::LargeUtf8, true))),
                        false,
                    )),
                ])),
                false,
            )),
            true,
        );

        assert_eq!(
            diagnostic_arrow_type(&data_type),
            "Map(\"entries\": non-null Struct(\"key\": non-null Utf8, \"value\": non-null List(LargeUtf8)), sorted)"
        );
    }
}
