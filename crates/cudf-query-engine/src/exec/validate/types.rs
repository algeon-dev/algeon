use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};

use super::*;
use crate::datatypes::is_supported_arrow_implicit_cast;

pub(super) fn require_boolean(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Boolean) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected boolean")
    }
}

pub(super) fn require_string(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected string")
    }
}

pub(super) fn require_date_or_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected date or timezone-free timestamp")
    }
}

pub(super) fn require_timestamp_second(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Timestamp(TimeUnit::Second, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected timezone-free second timestamp")
    }
}

pub(super) fn require_integral_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_integral_type(data_type) { Ok(()) } else { unsupported_type(context, data_type, "expected integral") }
}

pub(super) fn require_arithmetic_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_numeric_type(data_type) || is_decimal_type(data_type) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected numeric or decimal")
    }
}

pub(super) fn require_compatible_types(
    left: &ArrowDataType,
    right: &ArrowDataType,
    context: &'static str,
) -> Result<()> {
    if left == right || is_supported_arrow_implicit_cast(right, left) || is_supported_arrow_implicit_cast(left, right) {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

pub(super) fn require_comparable_types(left: &ArrowDataType, right: &ArrowDataType) -> Result<()> {
    if left == right
        || ((is_numeric_type(left) || is_decimal_type(left)) && (is_numeric_type(right) || is_decimal_type(right)))
        || (is_string_type(left) && is_string_type(right))
        || (is_temporal_type(left) && is_temporal_type(right))
    {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native comparison operand types are not compatible",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

pub(super) fn temporal_duration_binary_output_type(
    op: BinaryOp,
    left: &ArrowDataType,
    right: &ArrowDataType,
) -> Result<Option<ArrowDataType>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Ok(None);
    }
    let left_temporal = is_date_or_timestamp_type(left);
    let right_temporal = is_date_or_timestamp_type(right);
    let left_duration = is_duration_type(left);
    let right_duration = is_duration_type(right);

    match (op, left_temporal, right_temporal, left_duration, right_duration) {
        (BinaryOp::Add, true, false, false, true) | (BinaryOp::Subtract, true, false, false, true) => {
            Ok(Some(left.clone()))
        },
        (BinaryOp::Add, false, true, true, false) => Ok(Some(right.clone())),
        (_, true | false, true | false, true, true) | (_, true, true, false, false) => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native temporal arithmetic supports only datetime plus/minus duration",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}"))),
        _ => Ok(None),
    }
}

fn unsupported_type(context: &'static str, data_type: &ArrowDataType, expected: &'static str) -> Result<()> {
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "cudf_query_engine.plan.validate_expr",
        "native expression type does not satisfy its operator contract",
    )
    .with_fact("context", context)
    .with_fact("expected", expected)
    .with_fact("data_type", format!("{data_type:?}")))
}

fn is_integral_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
    )
}

pub(super) fn is_numeric_type(data_type: &ArrowDataType) -> bool {
    is_integral_type(data_type) || matches!(data_type, ArrowDataType::Float32 | ArrowDataType::Float64)
}

pub(super) fn is_decimal_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Decimal128(_, _))
}

fn is_string_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn is_temporal_type(data_type: &ArrowDataType) -> bool {
    is_date_or_timestamp_type(data_type) || is_duration_type(data_type)
}

fn is_date_or_timestamp_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Date32
            | ArrowDataType::Timestamp(
                TimeUnit::Second | TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond,
                None
            )
    )
}

fn is_duration_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Duration(
            TimeUnit::Second | TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond
        )
    )
}
