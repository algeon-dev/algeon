use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};

use super::*;
use crate::datatypes::{
    is_narrow_decimal_carry_type, is_supported_arrow_implicit_cast, is_supported_list_carry_type,
    is_timestamp_with_timezone,
};

#[doc(hidden)]
pub fn require_boolean(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Boolean) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected boolean")
    }
}

#[doc(hidden)]
pub fn require_string(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected string")
    }
}

#[doc(hidden)]
pub fn require_date_or_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected date or timezone-free timestamp")
    }
}

#[doc(hidden)]
pub fn require_timestamp_second(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Timestamp(TimeUnit::Second, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected timezone-free second timestamp")
    }
}

#[doc(hidden)]
pub fn require_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected timezone-free timestamp")
    }
}

#[doc(hidden)]
pub fn require_timestamp_preserving_output(
    input_type: &ArrowDataType,
    output_type: &ArrowDataType,
    context: &'static str,
) -> Result<()> {
    require_timestamp(input_type, context)?;
    require_timestamp(output_type, context)?;
    if input_type == output_type {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.plan.validate_expr",
            "native timestamp expression output type must preserve input type",
        )
        .with_fact("context", context)
        .with_fact("input_type", format!("{input_type:?}"))
        .with_fact("output_type", format!("{output_type:?}")))
    }
}

#[doc(hidden)]
pub fn require_timezone_free_operator_key(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_timestamp_with_timezone(data_type) {
        unsupported_type(context, data_type, "expected timezone-free operator key")
    } else if is_supported_list_carry_type(data_type) || is_narrow_decimal_carry_type(data_type) {
        unsupported_type(context, data_type, "expected scalar operator key")
    } else {
        Ok(())
    }
}

#[doc(hidden)]
pub fn require_scalar_operator_value(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_timestamp_with_timezone(data_type) {
        unsupported_type(context, data_type, "expected timezone-free scalar operator value")
    } else if is_supported_list_carry_type(data_type) || is_narrow_decimal_carry_type(data_type) {
        unsupported_type(context, data_type, "expected scalar operator value")
    } else {
        Ok(())
    }
}

#[doc(hidden)]
pub fn require_integral_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_integral_type(data_type) { Ok(()) } else { unsupported_type(context, data_type, "expected integral") }
}

#[doc(hidden)]
pub fn require_arithmetic_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_numeric_type(data_type) || is_decimal_type(data_type) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected numeric or decimal")
    }
}

#[doc(hidden)]
pub fn require_compatible_types(left: &ArrowDataType, right: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_timestamp_with_timezone(left)
        || is_timestamp_with_timezone(right)
        || carry_only_operand(left)
        || carry_only_operand(right)
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}"))
        .with_fact("reason", carry_operand_rejection_reason(left, right, "operator")));
    }
    if left == right || is_supported_arrow_implicit_cast(right, left) || is_supported_arrow_implicit_cast(left, right) {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

#[doc(hidden)]
pub fn require_comparable_types(left: &ArrowDataType, right: &ArrowDataType) -> Result<()> {
    if is_timestamp_with_timezone(left)
        || is_timestamp_with_timezone(right)
        || carry_only_operand(left)
        || carry_only_operand(right)
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.plan.validate_expr",
            "native comparison operand types are not compatible",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}"))
        .with_fact("reason", carry_operand_rejection_reason(left, right, "comparison")));
    }
    if left == right
        || ((is_numeric_type(left) || is_decimal_type(left)) && (is_numeric_type(right) || is_decimal_type(right)))
        || (is_string_type(left) && is_string_type(right))
        || (is_temporal_type(left) && is_temporal_type(right))
    {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.plan.validate_expr",
            "native comparison operand types are not compatible",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

fn carry_only_operand(data_type: &ArrowDataType) -> bool {
    is_supported_list_carry_type(data_type) || is_narrow_decimal_carry_type(data_type)
}

fn carry_operand_rejection_reason(
    left: &ArrowDataType,
    right: &ArrowDataType,
    operation: &'static str,
) -> &'static str {
    if is_timestamp_with_timezone(left) || is_timestamp_with_timezone(right) {
        match operation {
            "comparison" => "timestamp_timezone_comparison_unsupported",
            _ => "timestamp_timezone_operator_unsupported",
        }
    } else if is_narrow_decimal_carry_type(left) || is_narrow_decimal_carry_type(right) {
        match operation {
            "comparison" => "narrow_decimal_comparison_unsupported",
            _ => "narrow_decimal_operator_unsupported",
        }
    } else {
        "list_operator_unsupported"
    }
}

#[doc(hidden)]
pub fn temporal_duration_binary_output_type(
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
            "nexus_query_engine.plan.validate_expr",
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
        "nexus_query_engine.plan.validate_expr",
        "native expression type does not satisfy its operator contract",
    )
    .with_fact("context", context)
    .with_fact("expected", expected)
    .with_fact("data_type", format!("{data_type:?}")))
}

#[doc(hidden)]
pub fn is_integral_type(data_type: &ArrowDataType) -> bool {
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

#[doc(hidden)]
pub fn is_numeric_type(data_type: &ArrowDataType) -> bool {
    is_integral_type(data_type) || matches!(data_type, ArrowDataType::Float32 | ArrowDataType::Float64)
}

#[doc(hidden)]
pub fn is_decimal_type(data_type: &ArrowDataType) -> bool {
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
