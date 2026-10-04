//! Native cuDF dtype conversion facade.
//!
//! Arrow-to-cuDF dtype conversion and native type support are re-exported
//! from the component's own Arrow interop contract
//! (`cudf::interop::arrow_host`). A rejected conversion returns the cuDF error
//! with its code and `reason` detail. Timezone-aware timestamps and supported
//! list columns convert only as carry types; operator validation rejects them
//! where their semantics are unsupported.
//!
//! The Arrow type support and cast policy helpers exposed here live in
//! [`query_planning::datatypes`] and are re-exported unchanged.

use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
pub use cudf::interop::arrow_host::{cudf_dtype_from_arrow_type, is_supported_arrow_type};
use cudf::{ScalarValue, TimestampResolution};
pub(crate) use query_planning::datatypes::{
    arrow_decimal_scale_to_cudf_scale, arrow_fixed_width_byte_size, is_arrow_fixed_width,
};
pub use query_planning::datatypes::{
    diagnostic_arrow_type, is_narrow_decimal_carry_type, is_supported_arrow_cast, is_supported_arrow_implicit_cast,
    is_supported_list_carry_type, is_timestamp_with_timezone,
};

use crate::expr::LiteralValue;

/// Converts an Arrow temporal unit to the matching cuDF timestamp resolution.
pub(crate) fn timestamp_unit_to_cudf_resolution(unit: &TimeUnit) -> TimestampResolution {
    match unit {
        TimeUnit::Second => TimestampResolution::Seconds,
        TimeUnit::Millisecond => TimestampResolution::Milliseconds,
        TimeUnit::Microsecond => TimestampResolution::Microseconds,
        TimeUnit::Nanosecond => TimestampResolution::Nanoseconds,
    }
}

/// A literal with no cuDF host value of its own.
pub(crate) enum UnrepresentableLiteral<'a> {
    /// Month-day-nano intervals lower only inside date and timestamp
    /// arithmetic.
    Interval,
    /// A typed null, whose Arrow type selects the cuDF null type.
    Null(&'a ArrowDataType),
}

/// Converts a non-null, non-interval literal to its cuDF host value.
pub(crate) fn cudf_scalar_value(value: &LiteralValue) -> Result<ScalarValue, UnrepresentableLiteral<'_>> {
    Ok(match value {
        LiteralValue::Boolean(value) => ScalarValue::Bool(*value),
        LiteralValue::Int8(value) => ScalarValue::Int8(*value),
        LiteralValue::Int16(value) => ScalarValue::Int16(*value),
        LiteralValue::Int32(value) => ScalarValue::Int32(*value),
        LiteralValue::Int64(value) => ScalarValue::Int64(*value),
        LiteralValue::UInt8(value) => ScalarValue::UInt8(*value),
        LiteralValue::UInt16(value) => ScalarValue::UInt16(*value),
        LiteralValue::UInt32(value) => ScalarValue::UInt32(*value),
        LiteralValue::UInt64(value) => ScalarValue::UInt64(*value),
        LiteralValue::Float32(value) => ScalarValue::Float32(*value),
        LiteralValue::Float64(value) => ScalarValue::Float64(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => ScalarValue::String(value.clone()),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            ScalarValue::Decimal128 { raw: *raw, scale: arrow_decimal_scale_to_cudf_scale(*scale) }
        },
        LiteralValue::Date32(value) => {
            ScalarValue::Timestamp { raw: i64::from(*value), resolution: TimestampResolution::Days }
        },
        LiteralValue::Date64(value) => {
            ScalarValue::Timestamp { raw: *value, resolution: TimestampResolution::Milliseconds }
        },
        LiteralValue::Time32 { raw, unit } => {
            ScalarValue::Duration { raw: i64::from(*raw), resolution: timestamp_unit_to_cudf_resolution(unit) }
        },
        LiteralValue::Time64 { raw, unit } | LiteralValue::Duration { raw, unit } => {
            ScalarValue::Duration { raw: *raw, resolution: timestamp_unit_to_cudf_resolution(unit) }
        },
        LiteralValue::Timestamp { raw, unit } => {
            ScalarValue::Timestamp { raw: *raw, resolution: timestamp_unit_to_cudf_resolution(unit) }
        },
        LiteralValue::IntervalMonthDayNano { .. } => return Err(UnrepresentableLiteral::Interval),
        LiteralValue::Null(data_type) => return Err(UnrepresentableLiteral::Null(data_type)),
    })
}
