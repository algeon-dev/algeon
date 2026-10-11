//! Arrow type predicates and operand rules shared by expression, join and node
//! validation.
//!
//! Every `require_*` rule records the operator context in its error, so one
//! `UnsupportedType` code can say which operand failed. Two families the
//! backend represents are refused as operator inputs here. Lists and narrow
//! decimals are carry types: a node may read and project them unchanged but no
//! kernel computes on them. Timestamps with a timezone have no cuDF
//! representation beyond the raw instant. Keys, comparison operands and
//! arithmetic operands must therefore be scalar, timezone-free types.

use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};

use super::{BinaryOp, QueryError, QueryErrorCode, Result};
use crate::{
    backend::AnalysisContext,
    datatypes::{
        is_narrow_decimal_carry_type, is_supported_arrow_implicit_cast, is_supported_list_carry_type,
        is_timestamp_with_timezone,
    },
};

pub(crate) fn require_boolean(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Boolean) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected boolean")
    }
}

pub(crate) fn require_string(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected string")
    }
}

pub(crate) fn require_date_or_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected date or timezone-free timestamp")
    }
}

pub(crate) fn require_timestamp_second(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Timestamp(TimeUnit::Second, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected timezone-free second timestamp")
    }
}

pub(crate) fn require_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected timezone-free timestamp")
    }
}

pub(crate) fn require_timestamp_preserving_output(
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
            "query_engine.plan.validate_expr",
            "native timestamp expression output type must preserve input type",
        )
        .with_fact("context", context)
        .with_fact("input_type", format!("{input_type:?}"))
        .with_fact("output_type", format!("{output_type:?}")))
    }
}

/// Rule for group, sort, join, partition and order keys: scalar and
/// timezone-free. Carry types cannot be hashed or compared by a kernel.
pub(crate) fn require_timezone_free_operator_key(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_timestamp_with_timezone(data_type) {
        unsupported_type(context, data_type, "expected timezone-free operator key")
    } else if is_supported_list_carry_type(data_type) || is_narrow_decimal_carry_type(data_type) {
        unsupported_type(context, data_type, "expected scalar operator key")
    } else {
        Ok(())
    }
}

/// The key rule applied to a value operand (`IS NULL` input, window aggregate
/// input); it differs only in the wording of the expectation.
pub(crate) fn require_scalar_operator_value(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_timestamp_with_timezone(data_type) {
        unsupported_type(context, data_type, "expected timezone-free scalar operator value")
    } else if is_supported_list_carry_type(data_type) || is_narrow_decimal_carry_type(data_type) {
        unsupported_type(context, data_type, "expected scalar operator value")
    } else {
        Ok(())
    }
}

pub(crate) fn require_integral_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_integral_type(data_type) { Ok(()) } else { unsupported_type(context, data_type, "expected integral") }
}

pub(crate) fn require_remainder_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_numeric_type(data_type) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected integral or floating-point")
    }
}

pub(crate) fn require_arithmetic_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_numeric_type(data_type) || is_decimal_type(data_type) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected numeric or decimal")
    }
}

/// Operands of arithmetic, `IN` lists and `CASE` branches may differ only by
/// an implicit cast the backend supports in either direction. The carry and
/// timezone rejections come first, so their `reason` fact names the family.
/// Other mismatches carry no `reason` fact.
pub(crate) fn require_compatible_types(
    left: &ArrowDataType,
    right: &ArrowDataType,
    context: &'static str,
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    if is_timestamp_with_timezone(left)
        || is_timestamp_with_timezone(right)
        || carry_only_operand(left)
        || carry_only_operand(right)
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}"))
        .with_fact("reason", carry_operand_rejection_reason(left, right, "operator")));
    }
    if left == right
        || is_supported_arrow_implicit_cast(right, left, ctx.backend())
        || is_supported_arrow_implicit_cast(left, right, ctx.backend())
    {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

/// Comparison admits any two numeric or decimal types, any two string
/// representations (Utf8, LargeUtf8 and Utf8View compare by value), and any two
/// temporal types, without consulting the backend; the engine casts the
/// operands to a common type when the kernel needs one.
pub(crate) fn require_comparable_types(left: &ArrowDataType, right: &ArrowDataType) -> Result<()> {
    if is_timestamp_with_timezone(left)
        || is_timestamp_with_timezone(right)
        || carry_only_operand(left)
        || carry_only_operand(right)
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.plan.validate_expr",
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
            "query_engine.plan.validate_expr",
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

/// Output type of a temporal `+` or `-` that native evaluation supports, `None`
/// when the operands are not temporal, and an error for a temporal combination
/// native evaluation does not run.
///
/// Every accepted combination has a libcudf operation for its operand types:
/// a timezone-free timestamp plus or minus a duration of the same unit,
/// a duration plus such a timestamp, and the difference of two timestamps of
/// one unit. libcudf has no `Date32` plus a second-or-finer duration, and
/// Arrow rejects `Date32` plus or minus a duration outright. DataFusion casts a
/// duration to the timestamp's unit before it adds, so mixed units reach
/// execution only as an explicit cast. Accepting either here would fail the
/// query after planning chose the native path.
pub(crate) fn temporal_duration_binary_output_type(
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

    if let (BinaryOp::Subtract, ArrowDataType::Timestamp(left_unit, None), ArrowDataType::Timestamp(right_unit, None)) =
        (op, left, right)
        && left_unit == right_unit
    {
        return Ok(Some(ArrowDataType::Duration(*left_unit)));
    }
    match (op, left_temporal, right_temporal, left_duration, right_duration) {
        (_, true, false, false, true) if timestamp_has_duration_unit(left, right) => Ok(Some(left.clone())),
        (BinaryOp::Add, false, true, true, false) if timestamp_has_duration_unit(right, left) => {
            Ok(Some(right.clone()))
        },
        (_, true, false, false, true) | (BinaryOp::Add, false, true, true, false) => {
            let reason = if left == &ArrowDataType::Date32 || right == &ArrowDataType::Date32 {
                "date_duration_arithmetic_unsupported"
            } else {
                "timestamp_duration_unit_mismatch"
            };
            Err(unsupported_temporal_arithmetic(left, right, reason))
        },
        (_, true | false, true | false, true, true) | (_, true, true, false, false) => {
            Err(unsupported_temporal_arithmetic(left, right, "temporal_operand_combination_unsupported"))
        },
        _ => Ok(None),
    }
}

fn timestamp_has_duration_unit(timestamp: &ArrowDataType, duration: &ArrowDataType) -> bool {
    matches!(
        (timestamp, duration),
        (ArrowDataType::Timestamp(timestamp_unit, None), ArrowDataType::Duration(duration_unit))
            if timestamp_unit == duration_unit
    )
}

fn unsupported_temporal_arithmetic(left: &ArrowDataType, right: &ArrowDataType, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "query_engine.plan.validate_expr",
        "native temporal arithmetic supports only a timezone-free timestamp plus or minus a duration of the same \
         unit, and the difference of two timestamps of one unit",
    )
    .with_fact("left_type", format!("{left:?}"))
    .with_fact("right_type", format!("{right:?}"))
    .with_fact("reason", reason)
}

fn unsupported_type(context: &'static str, data_type: &ArrowDataType, expected: &'static str) -> Result<()> {
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "query_engine.plan.validate_expr",
        "native expression type does not satisfy its operator contract",
    )
    .with_fact("context", context)
    .with_fact("expected", expected)
    .with_fact("data_type", format!("{data_type:?}")))
}

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

pub(crate) fn is_numeric_type(data_type: &ArrowDataType) -> bool {
    is_integral_type(data_type) || matches!(data_type, ArrowDataType::Float32 | ArrowDataType::Float64)
}

pub(crate) fn is_decimal_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Decimal128(_, _))
}

pub fn is_string_type(data_type: &ArrowDataType) -> bool {
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
