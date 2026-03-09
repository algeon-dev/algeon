//! Arrow-to-cuDF datatype support policy.
//!
//! These helpers define the native engine's admitted Arrow surface before a
//! plan reaches cudf-rs. Returning `false` or `None` means the query should be
//! rejected by capability analysis rather than attempted at execution time.

use std::collections::HashMap;

use arrow::datatypes::{
    DECIMAL128_MAX_SCALE, DataType as ArrowDataType, Decimal128Type, Field, Fields, TimeUnit,
    validate_decimal_precision_and_scale,
};
use cudf::{DataType as CudfDataType, TimestampResolution, TypeId};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

/// Returns whether this Arrow type has a native cuDF physical representation.
pub fn is_supported_arrow_type(data_type: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(data_type).is_some()
}

/// Converts a supported Arrow type to its cuDF dtype.
///
/// Unsupported nested types and decimal shapes outside the native policy return
/// `None`. Timezone-aware timestamps and string-list columns are admitted only
/// as carry types; operator validation decides where their semantics are
/// unsupported.
pub fn cudf_dtype_from_arrow_type(data_type: &ArrowDataType) -> Option<CudfDataType> {
    match data_type {
        ArrowDataType::Boolean => Some(CudfDataType::new(TypeId::Bool8)),
        ArrowDataType::Int8 => Some(CudfDataType::new(TypeId::Int8)),
        ArrowDataType::Int16 => Some(CudfDataType::new(TypeId::Int16)),
        ArrowDataType::Int32 => Some(CudfDataType::new(TypeId::Int32)),
        ArrowDataType::Int64 => Some(CudfDataType::new(TypeId::Int64)),
        ArrowDataType::UInt8 => Some(CudfDataType::new(TypeId::UInt8)),
        ArrowDataType::UInt16 => Some(CudfDataType::new(TypeId::UInt16)),
        ArrowDataType::UInt32 => Some(CudfDataType::new(TypeId::UInt32)),
        ArrowDataType::UInt64 => Some(CudfDataType::new(TypeId::UInt64)),
        ArrowDataType::Float32 => Some(CudfDataType::new(TypeId::Float32)),
        ArrowDataType::Float64 => Some(CudfDataType::new(TypeId::Float64)),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            Some(CudfDataType::new(TypeId::String))
        },
        ArrowDataType::Decimal128(precision, scale) => is_supported_decimal128(*precision, *scale)
            .then(|| CudfDataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale))),
        ArrowDataType::Date32 => Some(CudfDataType::new(TypeId::TimestampDays)),
        ArrowDataType::Timestamp(unit, _) => Some(CudfDataType::new(timestamp_unit_to_cudf_resolution(unit).type_id())),
        ArrowDataType::Duration(unit) => {
            Some(CudfDataType::new(timestamp_unit_to_cudf_resolution(unit).duration_type_id()))
        },
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if is_supported_string_list_child(field) => {
            Some(CudfDataType::new(TypeId::List))
        },
        _ => None,
    }
}

/// Returns whether this Arrow type is a timestamp carrying timezone metadata.
pub fn is_timestamp_with_timezone(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Timestamp(_, Some(_)))
}

/// Returns whether this Arrow type is a list admitted only for carry/projection.
pub fn is_supported_list_carry_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::List(field) | ArrowDataType::LargeList(field) if is_supported_string_list_child(field))
}

/// Human-readable Arrow type name for client-facing diagnostics.
pub fn diagnostic_arrow_type(data_type: &ArrowDataType) -> String {
    match data_type {
        ArrowDataType::List(field) => format!("List({})", diagnostic_arrow_field(field)),
        ArrowDataType::LargeList(field) => format!("LargeList({})", diagnostic_arrow_field(field)),
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
        _ => left == right,
    }
}

fn is_supported_string_list_child(field: &arrow::datatypes::FieldRef) -> bool {
    matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn arrow_field_metadata_matches_except_parquet_field_id(
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

/// Converts an Arrow temporal unit to the matching cuDF timestamp resolution.
pub fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}

/// Returns whether the native engine supports an explicit cast.
pub fn is_supported_arrow_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    is_supported_arrow_implicit_cast(source_type, target_type)
        || is_supported_decimal_to_float64_cast(source_type, target_type)
        || is_supported_numeric_to_decimal_cast(source_type, target_type)
        || is_supported_decimal_to_integral_cast(source_type, target_type)
        || is_supported_float64_to_int64_cast(source_type, target_type)
}

/// Returns whether the native engine may insert or preserve an implicit cast.
pub fn is_supported_arrow_implicit_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    (source_type == target_type && is_supported_arrow_type(source_type))
        || is_supported_numeric_widening_cast(source_type, target_type)
        || is_supported_decimal_to_decimal_cast(source_type, target_type)
        || is_supported_string_cast(source_type, target_type)
        || is_supported_timestamp_cast(source_type, target_type)
        || is_supported_duration_cast(source_type, target_type)
}

/// Returns whether a numeric cast widens without narrowing range or precision class.
pub fn is_supported_numeric_widening_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
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

fn is_supported_float64_to_int64_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!((source_type, target_type), (ArrowDataType::Float64, ArrowDataType::Int64))
}

fn is_supported_decimal128(precision: u8, scale: i8) -> bool {
    validate_decimal_precision_and_scale::<Decimal128Type>(precision, scale).is_ok()
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

    use arrow::datatypes::{DataType as ArrowDataType, Field, Fields};

    use super::diagnostic_arrow_type;

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
            "Map(\"entries\": non-null Struct(\"key\": non-null Utf8, \"value\": non-null List(\"element\": LargeUtf8)), sorted)"
        );
    }
}
