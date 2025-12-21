use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use cudf::{DataType as CudfDataType, TimestampResolution, TypeId};

pub fn is_supported_arrow_type(data_type: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(data_type).is_some()
}

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
        ArrowDataType::Decimal128(_precision, scale) => {
            Some(CudfDataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale)))
        },
        ArrowDataType::Date32 => Some(CudfDataType::new(TypeId::TimestampDays)),
        ArrowDataType::Timestamp(unit, None) => {
            Some(CudfDataType::new(timestamp_unit_to_cudf_resolution(unit).type_id()))
        },
        ArrowDataType::Timestamp(_, Some(_)) => None,
        ArrowDataType::Duration(unit) => {
            Some(CudfDataType::new(timestamp_unit_to_cudf_resolution(unit).duration_type_id()))
        },
        _ => None,
    }
}

pub fn arrow_decimal_scale_to_cudf_scale(scale: i8) -> i32 {
    -i32::from(scale)
}

pub fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}

pub fn is_supported_arrow_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    source_type == target_type
        || is_supported_numeric_widening_cast(source_type, target_type)
        || is_supported_decimal_cast(source_type, target_type)
        || is_supported_string_cast(source_type, target_type)
        || is_supported_timestamp_cast(source_type, target_type)
        || is_supported_duration_cast(source_type, target_type)
}

pub fn is_supported_numeric_widening_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(
        (source_type, target_type),
        (ArrowDataType::Int8, ArrowDataType::Int16 | ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::Int16, ArrowDataType::Int32 | ArrowDataType::Int64)
            | (ArrowDataType::Int32, ArrowDataType::Int64)
            | (ArrowDataType::UInt8, ArrowDataType::UInt16 | ArrowDataType::UInt32 | ArrowDataType::UInt64)
            | (ArrowDataType::UInt16, ArrowDataType::UInt32 | ArrowDataType::UInt64)
            | (ArrowDataType::UInt32, ArrowDataType::UInt64)
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

fn is_supported_decimal_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!((source_type, target_type), (ArrowDataType::Decimal128(_, _), ArrowDataType::Decimal128(_, _)))
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
    )
}

fn is_supported_duration_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!((source_type, target_type), (ArrowDataType::Duration(_), ArrowDataType::Duration(_)))
}
