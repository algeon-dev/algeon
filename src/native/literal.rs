use arrow::datatypes::{DataType, TimeUnit};
use cudf_query_engine::{LiteralValue as NativeLiteralValue, is_supported_arrow_type};
use datafusion::common::ScalarValue;

use crate::policy::FallbackReason;

#[derive(Debug, Clone, Copy)]
pub(super) enum NativeLiteralContext {
    Expression,
    SourceValue,
}

impl NativeLiteralContext {
    fn fallback_reason(self) -> FallbackReason {
        match self {
            Self::Expression => FallbackReason::UnsupportedExpression,
            Self::SourceValue => FallbackReason::UnsupportedType,
        }
    }
}

pub(super) fn native_literal_value(
    value: &ScalarValue,
    context: NativeLiteralContext,
) -> Result<NativeLiteralValue, FallbackReason> {
    let literal = match value {
        ScalarValue::Boolean(Some(value)) => NativeLiteralValue::Boolean(*value),
        ScalarValue::Int8(Some(value)) => NativeLiteralValue::Int8(*value),
        ScalarValue::Int16(Some(value)) => NativeLiteralValue::Int16(*value),
        ScalarValue::Int32(Some(value)) => NativeLiteralValue::Int32(*value),
        ScalarValue::Int64(Some(value)) => NativeLiteralValue::Int64(*value),
        ScalarValue::UInt8(Some(value)) => NativeLiteralValue::UInt8(*value),
        ScalarValue::UInt16(Some(value)) => NativeLiteralValue::UInt16(*value),
        ScalarValue::UInt32(Some(value)) => NativeLiteralValue::UInt32(*value),
        ScalarValue::UInt64(Some(value)) => NativeLiteralValue::UInt64(*value),
        ScalarValue::Float32(Some(value)) => NativeLiteralValue::Float32(*value),
        ScalarValue::Float64(Some(value)) => NativeLiteralValue::Float64(*value),
        // Utf8View has no distinct native literal; non-null values are canonicalized to Utf8.
        ScalarValue::Utf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => NativeLiteralValue::Utf8(value.clone()),
        ScalarValue::LargeUtf8(Some(value)) => NativeLiteralValue::LargeUtf8(value.clone()),
        ScalarValue::Decimal128(Some(raw), precision, scale) => {
            NativeLiteralValue::Decimal128 { raw: *raw, precision: *precision, scale: *scale }
        },
        ScalarValue::Date32(Some(value)) => NativeLiteralValue::Date32(*value),
        ScalarValue::TimestampSecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: TimeUnit::Second }
        },
        ScalarValue::TimestampMillisecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: TimeUnit::Millisecond }
        },
        ScalarValue::TimestampMicrosecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: TimeUnit::Microsecond }
        },
        ScalarValue::TimestampNanosecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: TimeUnit::Nanosecond }
        },
        ScalarValue::DurationSecond(Some(raw)) => NativeLiteralValue::Duration { raw: *raw, unit: TimeUnit::Second },
        ScalarValue::DurationMillisecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: TimeUnit::Millisecond }
        },
        ScalarValue::DurationMicrosecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: TimeUnit::Microsecond }
        },
        ScalarValue::DurationNanosecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: TimeUnit::Nanosecond }
        },
        ScalarValue::Boolean(None) => NativeLiteralValue::Null(DataType::Boolean),
        ScalarValue::Int8(None) => NativeLiteralValue::Null(DataType::Int8),
        ScalarValue::Int16(None) => NativeLiteralValue::Null(DataType::Int16),
        ScalarValue::Int32(None) => NativeLiteralValue::Null(DataType::Int32),
        ScalarValue::Int64(None) => NativeLiteralValue::Null(DataType::Int64),
        ScalarValue::UInt8(None) => NativeLiteralValue::Null(DataType::UInt8),
        ScalarValue::UInt16(None) => NativeLiteralValue::Null(DataType::UInt16),
        ScalarValue::UInt32(None) => NativeLiteralValue::Null(DataType::UInt32),
        ScalarValue::UInt64(None) => NativeLiteralValue::Null(DataType::UInt64),
        ScalarValue::Float32(None) => NativeLiteralValue::Null(DataType::Float32),
        ScalarValue::Float64(None) => NativeLiteralValue::Null(DataType::Float64),
        ScalarValue::Utf8(None) => NativeLiteralValue::Null(DataType::Utf8),
        // Typed Utf8View null is preserved; native dtype maps it to a cuDF string.
        ScalarValue::Utf8View(None) => NativeLiteralValue::Null(DataType::Utf8View),
        ScalarValue::LargeUtf8(None) => NativeLiteralValue::Null(DataType::LargeUtf8),
        ScalarValue::Decimal128(None, precision, scale) => {
            NativeLiteralValue::Null(DataType::Decimal128(*precision, *scale))
        },
        ScalarValue::Date32(None) => NativeLiteralValue::Null(DataType::Date32),
        ScalarValue::TimestampSecond(None, None) => {
            NativeLiteralValue::Null(DataType::Timestamp(TimeUnit::Second, None))
        },
        ScalarValue::TimestampMillisecond(None, None) => {
            NativeLiteralValue::Null(DataType::Timestamp(TimeUnit::Millisecond, None))
        },
        ScalarValue::TimestampMicrosecond(None, None) => {
            NativeLiteralValue::Null(DataType::Timestamp(TimeUnit::Microsecond, None))
        },
        ScalarValue::TimestampNanosecond(None, None) => {
            NativeLiteralValue::Null(DataType::Timestamp(TimeUnit::Nanosecond, None))
        },
        ScalarValue::DurationSecond(None) => NativeLiteralValue::Null(DataType::Duration(TimeUnit::Second)),
        ScalarValue::DurationMillisecond(None) => NativeLiteralValue::Null(DataType::Duration(TimeUnit::Millisecond)),
        ScalarValue::DurationMicrosecond(None) => NativeLiteralValue::Null(DataType::Duration(TimeUnit::Microsecond)),
        ScalarValue::DurationNanosecond(None) => NativeLiteralValue::Null(DataType::Duration(TimeUnit::Nanosecond)),
        _ => return Err(context.fallback_reason()),
    };

    if native_literal_type_is_supported(&literal.data_type()) { Ok(literal) } else { Err(context.fallback_reason()) }
}

pub(super) fn validate_native_literal(
    value: &ScalarValue,
    context: NativeLiteralContext,
) -> Result<(), FallbackReason> {
    native_literal_value(value, context).map(|_literal| ())
}

fn native_literal_type_is_supported(data_type: &DataType) -> bool {
    is_supported_arrow_type(data_type)
}
