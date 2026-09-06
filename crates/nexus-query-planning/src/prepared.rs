use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use cudf::{
    AggSpec as CudfAggSpec, GroupByAgg,
    ops::{
        binary::BinaryOperator,
        col,
        expr::{concat, lit_decimal128_raw, lit_duration_raw, lit_timestamp_raw},
        lit, lit_null,
        strings::StringSliceOptions,
        when,
    },
};

use crate::{
    datatypes::{cudf_dtype_from_arrow_type, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, DatePart, Expr, LiteralValue, SortSpec,
        interval_month_day_nano,
    },
    expression_lowering_policy::ExpressionLoweringPolicy,
    plan::ProjectionOutput,
};

mod aggregate;

#[doc(hidden)]
pub use aggregate::{
    AvgStateAliases, DataFusionExactHostAccumulator, LoweredAggregate, aggregate_expr_is_exact_nunique,
    aggregate_spec_has_avg, aggregate_spec_has_distinct, aggregate_spec_has_filters,
    aggregate_spec_has_floating_statistical_aggregate, aggregate_spec_has_median, aggregate_spec_requires_whole_input,
    datafusion_exact_host_accumulator, datafusion_exact_host_group_key_supported, lower_aggregate_function,
    lower_aggregate_spec, lower_aggregate_spec_keys, lower_final_aggregate_function, lower_final_aggregate_spec,
    lower_partial_aggregate_spec, lower_single_stage_grouped_aggregate_spec,
};

pub fn lower_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Column(column) => Ok(col(column.name())),
        Expr::Literal(value) => lower_literal(value),
        Expr::Not(expr) => Ok(!lower_expr(expr)?),
        Expr::IsNull(expr) => Ok(lower_expr(expr)?.is_null()),
        Expr::IsNotNull(expr) => Ok(lower_expr(expr)?.is_not_null()),
        Expr::Cast { expr, target_type } => {
            if expr_has_declared_type(expr, target_type) {
                return lower_expr_with_expected_type(expr, Some(target_type));
            }
            if let Some(cast) = lower_integer_temporal_cast(expr, target_type)? {
                return Ok(cast);
            }
            let target_dtype = cudf_dtype_from_arrow_type(target_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.expr.lower_cast",
                    "native execution does not support casts to this target type",
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })?;
            Ok(lower_expr_with_expected_type(expr, Some(target_type))?.cast(target_dtype))
        },
        Expr::TryCast { expr, target_type } => lower_try_cast_expr(expr, target_type),
        Expr::Binary { left, op, right, output_type } => lower_binary_expr(left, *op, right, output_type.as_ref()),
        Expr::Like { expr, pattern, escape, negated } => {
            let expr = match escape {
                Some(escape) => {
                    validate_like_escape(escape)?;
                    lower_expr(expr)?.str().like_escape(pattern.clone(), escape.clone())
                },
                None => lower_expr(expr)?.str().like(pattern.clone()),
            };
            Ok(if *negated { !expr } else { expr })
        },
        Expr::Substring { expr, start, length } => lower_substring(expr, *start, *length),
        Expr::Lower(expr) => Ok(lower_expr(expr)?.str().lower()),
        Expr::Upper(expr) => Ok(lower_expr(expr)?.str().upper()),
        Expr::Trim { expr, trim_chars } if trim_chars.is_empty() => lower_expr(expr),
        Expr::Trim { expr, trim_chars } => Ok(lower_expr(expr)?.str().strip_chars(trim_chars.clone())),
        Expr::LTrim { expr, trim_chars } if trim_chars.is_empty() => lower_expr(expr),
        Expr::LTrim { expr, trim_chars } => Ok(lower_expr(expr)?.str().lstrip_chars(trim_chars.clone())),
        Expr::RTrim { expr, trim_chars } if trim_chars.is_empty() => lower_expr(expr),
        Expr::RTrim { expr, trim_chars } => Ok(lower_expr(expr)?.str().rstrip_chars(trim_chars.clone())),
        Expr::StringContains { expr, pattern } => Ok(lower_expr(expr)?.str().contains(pattern.clone())),
        Expr::StartsWith { expr, prefix } => Ok(lower_expr(expr)?.str().starts_with(prefix.clone())),
        Expr::EndsWith { expr, suffix } => Ok(lower_expr(expr)?.str().ends_with(suffix.clone())),
        Expr::StringLength { expr, output_type } => lower_string_length(expr, output_type),
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => {
            lower_regexp_replace(expr, pattern, replacement, output_type)
        },
        Expr::Concat(exprs) => lower_concat(exprs),
        Expr::Abs(expr) => Ok(lower_expr(expr)?.abs()),
        Expr::Sqrt { expr, output_type } => lower_sqrt(expr, output_type),
        Expr::Round { expr, decimal_places, output_type } => lower_round(expr, *decimal_places, output_type),
        Expr::InList { expr, list, negated } => lower_in_list(expr, list, *negated),
        Expr::IsDistinctFrom { left, right } => {
            Ok(cudf_binary_expr(lower_expr(left)?, BinaryOperator::NullNotEquals, lower_expr(right)?))
        },
        Expr::IsNotDistinctFrom { left, right } => {
            Ok(cudf_binary_expr(lower_expr(left)?, BinaryOperator::NullEquals, lower_expr(right)?))
        },
        Expr::Case { branches, else_expr } => lower_case(branches, else_expr),
        Expr::Extract { field, expr } => lower_extract(*field, expr),
        Expr::DateTrunc { field, expr, output_type } => lower_date_trunc(*field, expr, output_type),
        Expr::Alias { expr, alias } => Ok(lower_expr(expr)?.alias(alias)),
    }
}

pub fn lower_filter_predicate_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Not(expr) => Ok(!lower_filter_predicate_expr(expr)?),
        Expr::IsNull(expr) => Ok(lower_filter_predicate_expr(expr)?.is_null()),
        Expr::IsNotNull(expr) => Ok(lower_filter_predicate_expr(expr)?.is_not_null()),
        Expr::Binary { left, op, right, output_type } => {
            lower_filter_binary_expr(left, *op, right, output_type.as_ref())
        },
        Expr::InList { expr, list, negated } => lower_filter_in_list(expr, list, *negated),
        Expr::Alias { expr, alias } => Ok(lower_filter_predicate_expr(expr)?.alias(alias)),
        _ => lower_expr(expr),
    }
}

#[doc(hidden)]
pub fn lower_projection_output_expr(output: &ProjectionOutput) -> Result<cudf::Expr> {
    lower_expr_with_expected_type(&output.expr, Some(output.field.data_type()))
}

fn lower_expr_with_expected_type(expr: &Expr, expected_type: Option<&ArrowDataType>) -> Result<cudf::Expr> {
    match expr {
        Expr::Cast { expr, target_type } => {
            if expr_has_declared_type(expr, target_type) {
                return lower_expr_with_expected_type(expr, Some(target_type));
            }
            if let Some(cast) = lower_integer_temporal_cast(expr, target_type)? {
                return Ok(cast);
            }
            let target_dtype = cudf_dtype_from_arrow_type(target_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.expr.lower_cast",
                    "native execution does not support casts to this target type",
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })?;
            Ok(lower_expr_with_expected_type(expr, Some(target_type))?.cast(target_dtype))
        },
        Expr::TryCast { expr, target_type } => lower_try_cast_expr(expr, target_type),
        Expr::Binary { left, op, right, output_type } => {
            lower_binary_expr_with_expected_type(left, *op, right, output_type.as_ref(), expected_type)
        },
        Expr::Alias { expr, alias } => Ok(lower_expr_with_expected_type(expr, expected_type)?.alias(alias)),
        _ => lower_expr(expr),
    }
}

/// Lower a `TRY_CAST(input AS target_type)` expression to a cudf expression.
///
/// Native execution currently admits only string -> integer try-casts. The IR
/// only carries the target type, so this rejects unsupported targets here;
/// the source type is enforced at runtime by `cudf::Expr::TryCast` eval (and
/// by the adapter-side validator before the IR is built).
fn lower_try_cast_expr(input: &Expr, target_type: &ArrowDataType) -> Result<cudf::Expr> {
    if !is_integer_type(target_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_try_cast",
            "native execution supports TRY_CAST only when the target type is an integer",
        )
        .with_fact("target_type", format!("{target_type:?}")));
    }
    let target_dtype = cudf_dtype_from_arrow_type(target_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_try_cast",
            "native execution does not support try-casts to this target type",
        )
        .with_fact("target_type", format!("{target_type:?}"))
    })?;
    Ok(lower_expr(input)?.try_cast(target_dtype))
}

fn is_integer_type(data_type: &ArrowDataType) -> bool {
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
pub fn lower_literal(value: &LiteralValue) -> Result<cudf::Expr> {
    match value {
        LiteralValue::Boolean(value) => Ok(lit(*value)),
        LiteralValue::Int8(value) => Ok(lit(*value)),
        LiteralValue::Int16(value) => Ok(lit(*value)),
        LiteralValue::Int32(value) => Ok(lit(*value)),
        LiteralValue::Int64(value) => Ok(lit(*value)),
        LiteralValue::UInt8(value) => Ok(lit(*value)),
        LiteralValue::UInt16(value) => Ok(lit(*value)),
        LiteralValue::UInt32(value) => Ok(lit(*value)),
        LiteralValue::UInt64(value) => Ok(lit(*value)),
        LiteralValue::Float32(value) => Ok(lit(*value)),
        LiteralValue::Float64(value) => Ok(lit(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => Ok(lit(value.clone())),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            Ok(lit_decimal128_raw(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale)))
        },
        LiteralValue::Date32(value) => Ok(lit_timestamp_raw(i64::from(*value), cudf::TimestampResolution::Days)),
        LiteralValue::Date64(value) => Ok(lit_timestamp_raw(*value, cudf::TimestampResolution::Milliseconds)),
        LiteralValue::Time32 { raw, unit } => {
            Ok(lit_duration_raw(i64::from(*raw), timestamp_unit_to_cudf_resolution(unit)))
        },
        LiteralValue::Time64 { raw, unit } => Ok(lit_duration_raw(*raw, timestamp_unit_to_cudf_resolution(unit))),
        LiteralValue::Timestamp { raw, unit } => Ok(lit_timestamp_raw(*raw, timestamp_unit_to_cudf_resolution(unit))),
        LiteralValue::Duration { raw, unit } => Ok(lit_duration_raw(*raw, timestamp_unit_to_cudf_resolution(unit))),
        LiteralValue::IntervalMonthDayNano { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_literal",
            "interval literals must be lowered as part of timestamp/date interval arithmetic",
        )
        .with_fact("literal", value)),
        LiteralValue::Null(data_type) => {
            let dtype = cudf_dtype_from_arrow_type(data_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.expr.lower_literal",
                    "native execution does not support typed null literals for this type",
                )
                .with_fact("literal", value)
                .with_fact("data_type", format!("{data_type:?}"))
            })?;
            Ok(lit_null(dtype))
        },
    }
}

fn lower_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
) -> Result<cudf::Expr> {
    lower_binary_expr_with_expected_type(left, op, right, output_type, None)
}

fn lower_filter_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
) -> Result<cudf::Expr> {
    // Filter predicates lower AND/OR through the null-aware AST operators so the
    // post-read FilterNode path matches SQL/Spark 3-valued logic exactly (e.g.
    // `true OR null` is `true`, `false AND null` is `false`). This is the same
    // semantics libcudf's reader_predicate uses (NULL_LOGICAL_*), so a predicate
    // replayed as a FilterNode by `local_parquet_source_reuse` keeps the same row
    // set it would have had as a reader-side filter.
    match op {
        BinaryOp::And | BinaryOp::Or => lower_filter_logical_expr(left, op, right),
        _ => lower_binary_expr(left, op, right, output_type),
    }
}

fn lower_filter_logical_expr(left: &Expr, op: BinaryOp, right: &Expr) -> Result<cudf::Expr> {
    let combine_op = match op {
        BinaryOp::And => BinaryOperator::NullLogicalAnd,
        BinaryOp::Or => BinaryOperator::NullLogicalOr,
        _ => unreachable!("filter logical lowering is only used for AND/OR"),
    };
    let mut operands = Vec::new();
    let mut stack = vec![right, left];
    while let Some(expr) = stack.pop() {
        match expr {
            Expr::Binary { left, op: nested_op, right, output_type: None } if *nested_op == op => {
                stack.push(right);
                stack.push(left);
            },
            _ => operands.push(expr),
        }
    }

    let mut lowered = operands.into_iter().map(lower_filter_predicate_expr);
    let first = lowered.next().expect("filter logical expression always has at least two operands")?;
    lowered.try_fold(first, |left, right| Ok(cudf_binary_expr(left, combine_op, right?)))
}

fn lower_binary_expr_with_expected_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    expected_type: Option<&ArrowDataType>,
) -> Result<cudf::Expr> {
    if let Some((left, right)) = date32_integral_cast_comparison_operands(left, right, op) {
        return lower_binary_with_expected_type(
            lower_expr(left)?,
            op,
            lower_expr(&right)?,
            output_type.or(expected_type),
        );
    }
    let declared_type = output_type.or(expected_type);
    if let Some((months, days, nanoseconds)) = interval_month_day_nano(right) {
        return match op {
            BinaryOp::Add => lower_datetime_interval_offset(left, months, days, nanoseconds),
            BinaryOp::Subtract => {
                let (months, days, nanoseconds) = negate_interval(months, days, nanoseconds)?;
                lower_datetime_interval_offset(left, months, days, nanoseconds)
            },
            _ => lower_binary_with_expected_type(lower_expr(left)?, op, lower_expr(right)?, declared_type),
        };
    }
    if let Some((months, days, nanoseconds)) = interval_month_day_nano(left)
        && matches!(op, BinaryOp::Add)
    {
        return lower_datetime_interval_offset(right, months, days, nanoseconds);
    }
    lower_binary_with_expected_type(lower_expr(left)?, op, lower_expr(right)?, declared_type)
}

fn expr_has_declared_type(expr: &Expr, target_type: &ArrowDataType) -> bool {
    expr_declared_type(expr).is_some_and(|data_type| {
        data_type == *target_type || (is_string_type(&data_type) && is_string_type(target_type))
    })
}

fn expr_declared_type(expr: &Expr) -> Option<ArrowDataType> {
    match expr {
        Expr::Literal(value) => Some(value.data_type()),
        Expr::Cast { target_type, .. } => Some(target_type.clone()),
        Expr::Binary { output_type, .. } => output_type.clone(),
        Expr::Substring { .. } => Some(ArrowDataType::Utf8),
        Expr::StringLength { output_type, .. }
        | Expr::RegexpReplace { output_type, .. }
        | Expr::Sqrt { output_type, .. }
        | Expr::Round { output_type, .. }
        | Expr::DateTrunc { output_type, .. } => Some(output_type.clone()),
        Expr::Extract { .. } => Some(ArrowDataType::Int32),
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => Some(ArrowDataType::Boolean),
        Expr::Alias { expr, .. } => expr_declared_type(expr),
        _ => None,
    }
}

fn is_string_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn lower_integer_temporal_cast(expr: &Expr, target_type: &ArrowDataType) -> Result<Option<cudf::Expr>> {
    if expr_declared_type(expr)
        .is_some_and(|source_type| matches!(source_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None)))
    {
        return Ok(None);
    }
    let Some((timestamp_resolution, duration_resolution)) = temporal_cast_resolutions(target_type) else {
        return Ok(None);
    };
    let duration = lower_expr(expr)?.cast(cudf::DataType::new(duration_resolution.duration_type_id()));
    Ok(Some(duration.cast(cudf::DataType::new(timestamp_resolution.type_id()))))
}

fn temporal_cast_resolutions(
    target_type: &ArrowDataType,
) -> Option<(cudf::TimestampResolution, cudf::TimestampResolution)> {
    match target_type {
        ArrowDataType::Date32 => Some((cudf::TimestampResolution::Days, cudf::TimestampResolution::Days)),
        ArrowDataType::Timestamp(unit, None) => {
            let resolution = timestamp_unit_to_cudf_resolution(unit);
            Some((resolution, resolution))
        },
        _ => None,
    }
}

fn date32_integral_cast_comparison_operands<'a>(
    left: &'a Expr,
    right: &'a Expr,
    op: BinaryOp,
) -> Option<(&'a Expr, Expr)> {
    if !matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq) {
        return None;
    }
    let Expr::Cast { expr, target_type } = left else {
        return None;
    };
    if !matches!(target_type, ArrowDataType::Date32) {
        return None;
    }
    let Expr::Literal(LiteralValue::Date32(value)) = right else {
        return None;
    };
    Some((expr, Expr::Literal(LiteralValue::Int32(*value))))
}

fn lower_datetime_interval_offset(expr: &Expr, months: i32, days: i32, nanoseconds: i64) -> Result<cudf::Expr> {
    Ok(lower_expr(expr)?.dt().offset(months, days, nanoseconds))
}

fn negate_interval(months: i32, days: i32, nanoseconds: i64) -> Result<(i32, i32, i64)> {
    let months = months.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_interval",
            "interval month component cannot be negated without overflow",
        )
        .with_fact("months", months)
    })?;
    let days = days.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_interval",
            "interval day component cannot be negated without overflow",
        )
        .with_fact("days", days)
    })?;
    let nanoseconds = nanoseconds.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_interval",
            "interval nanosecond component cannot be negated without overflow",
        )
        .with_fact("nanoseconds", nanoseconds)
    })?;
    Ok((months, days, nanoseconds))
}

fn lower_binary_with_expected_type(
    left: cudf::Expr,
    op: BinaryOp,
    right: cudf::Expr,
    expected_type: Option<&ArrowDataType>,
) -> Result<cudf::Expr> {
    if op == BinaryOp::Divide
        && let Some(expected_type) = expected_type
        && let Some(output_type) = expected_decimal_dtype(expected_type)?
    {
        return Ok(left.binary_op_with_type(right, BinaryOperator::Div, output_type));
    }

    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Subtract => Ok(left - right),
        BinaryOp::Multiply => Ok(left * right),
        BinaryOp::Divide => Ok(left / right),
        BinaryOp::Modulo => Ok(left % right),
        BinaryOp::BitwiseAnd => Ok(cudf_binary_expr(left, BinaryOperator::BitwiseAnd, right)),
        BinaryOp::BitwiseOr => Ok(cudf_binary_expr(left, BinaryOperator::BitwiseOr, right)),
        BinaryOp::BitwiseXor => Ok(cudf_binary_expr(left, BinaryOperator::BitwiseXor, right)),
        BinaryOp::ShiftLeft => Ok(cudf_binary_expr(left, BinaryOperator::ShiftLeft, right)),
        BinaryOp::ShiftRight => Ok(cudf_binary_expr(left, BinaryOperator::ShiftRight, right)),
        BinaryOp::Eq => Ok(left.eq(right)),
        BinaryOp::NotEq => Ok(left.ne(right)),
        BinaryOp::Lt => Ok(left.lt(right)),
        BinaryOp::LtEq => Ok(left.le(right)),
        BinaryOp::Gt => Ok(left.gt(right)),
        BinaryOp::GtEq => Ok(left.ge(right)),
        BinaryOp::And => Ok(sql_boolean_and(left, right)),
        BinaryOp::Or => Ok(sql_boolean_or(left, right)),
    }
}

fn cudf_binary_expr(left: cudf::Expr, op: BinaryOperator, right: cudf::Expr) -> cudf::Expr {
    cudf::Expr::BinaryOp { lhs: Box::new(left), op, rhs: Box::new(right) }
}

fn sql_boolean_and(left: cudf::Expr, right: cudf::Expr) -> cudf::Expr {
    let any_null = left.clone().is_null() | right.clone().is_null();
    when(left.clone().eq(lit(false))).then(lit(false)).otherwise(
        when(right.clone().eq(lit(false)))
            .then(lit(false))
            .otherwise(when(any_null).then(lower_boolean_null_literal()).otherwise(lit(true))),
    )
}

fn sql_boolean_or(left: cudf::Expr, right: cudf::Expr) -> cudf::Expr {
    let any_null = left.clone().is_null() | right.clone().is_null();
    when(left.clone().eq(lit(true))).then(lit(true)).otherwise(
        when(right.clone().eq(lit(true)))
            .then(lit(true))
            .otherwise(when(any_null).then(lower_boolean_null_literal()).otherwise(lit(false))),
    )
}

fn lower_boolean_null_literal() -> cudf::Expr {
    lit_null(cudf_dtype_from_arrow_type(&ArrowDataType::Boolean).expect("Boolean must be a supported cudf dtype"))
}

fn expected_decimal_dtype(expected_type: &ArrowDataType) -> Result<Option<cudf::DataType>> {
    if !matches!(expected_type, ArrowDataType::Decimal128(_, _)) {
        return Ok(None);
    }
    let dtype = cudf_dtype_from_arrow_type(expected_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_binary",
            "native decimal division output type is not supported",
        )
        .with_fact("expected_type", format!("{expected_type:?}"))
    })?;
    Ok(Some(dtype))
}

fn lower_substring(expr: &Expr, start: i32, length: Option<i32>) -> Result<cudf::Expr> {
    if start < 1 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_substring",
            "native substring requires a one-based positive start position",
        )
        .with_fact("start", start));
    }
    if let Some(length) = length
        && length < 0
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_substring",
            "native substring requires a non-negative length",
        )
        .with_fact("length", length));
    }

    let zero_based_start = start - 1;
    let stop = length
        .map(|length| {
            zero_based_start.checked_add(length).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "nexus_query_engine.expr.lower_substring",
                    "native substring stop position exceeds i32 range",
                )
                .with_fact("start", start)
                .with_fact("length", length)
            })
        })
        .transpose()?;
    let options = StringSliceOptions::new(Some(zero_based_start), stop);
    Ok(lower_expr(expr)?.str().slice(options))
}

fn lower_string_length(expr: &Expr, output_type: &ArrowDataType) -> Result<cudf::Expr> {
    let length = lower_expr(expr)?.str().len();
    let target = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_string_length",
            "native string length output type is not supported",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    Ok(length.cast(target))
}

fn lower_regexp_replace(
    expr: &Expr,
    pattern: &str,
    replacement: &str,
    output_type: &ArrowDataType,
) -> Result<cudf::Expr> {
    let replaced = lower_expr(expr)?.str().replace_with_backrefs(pattern, replacement);
    if is_string_type(output_type) {
        return Ok(replaced);
    }
    let target = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_regexp_replace",
            "native REGEXP_REPLACE output type is not supported",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    Ok(replaced.cast(target))
}

fn lower_concat(exprs: &[Expr]) -> Result<cudf::Expr> {
    if exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_concat",
            "native CONCAT requires at least one argument",
        ));
    }
    exprs.iter().map(lower_expr).collect::<Result<Vec<_>>>().map(concat)
}

fn lower_round(expr: &Expr, decimal_places: i32, output_type: &ArrowDataType) -> Result<cudf::Expr> {
    let dtype = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_round",
            "native ROUND output type is not supported",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    if matches!(output_type, ArrowDataType::Float32 | ArrowDataType::Float64) {
        return lower_float_round(expr, decimal_places).map(|expr| expr.cast(dtype));
    }
    Ok(lower_expr(expr)?.round(decimal_places).cast(dtype))
}

fn lower_sqrt(expr: &Expr, output_type: &ArrowDataType) -> Result<cudf::Expr> {
    let dtype = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_sqrt",
            "native SQRT output type is not supported",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    Ok(lower_expr(expr)?.cast(dtype).sqrt().cast(dtype))
}

fn lower_float_round(expr: &Expr, decimal_places: i32) -> Result<cudf::Expr> {
    let factor = 10_f64.powi(decimal_places);
    if !factor.is_finite() || factor == 0.0 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_round",
            "native float ROUND decimal places produce a non-finite scale factor",
        )
        .with_fact("decimal_places", decimal_places));
    }
    let scaled = lower_expr(expr)? * lit(factor);
    let rounded = when(scaled.clone().ge(lit(0.0)))
        .then((scaled.clone() + lit(0.5)).floor())
        .otherwise((scaled - lit(0.5)).ceil());
    Ok(rounded / lit(factor))
}

fn lower_in_list(expr: &Expr, list: &[Expr], negated: bool) -> Result<cudf::Expr> {
    validate_in_list_literal_values(list)?;
    let values = list.iter().map(lower_expr).collect::<Result<Vec<_>>>()?;
    Ok(if negated { lower_expr(expr)?.is_not_in_list(values) } else { lower_expr(expr)?.is_in_list(values) })
}

fn lower_filter_in_list(expr: &Expr, list: &[Expr], negated: bool) -> Result<cudf::Expr> {
    validate_in_list_literal_values(list)?;
    if !ExpressionLoweringPolicy::default().filter_in_list_should_use_row_predicate_tree(list.len()) {
        // Large dynamic-filter keysets are post-decode filters. Keeping them as
        // cudf InList lets cudf use its boolean-mask path instead of recursively
        // lowering a deeply nested row-predicate OR/AND tree.
        return lower_in_list(expr, list, negated);
    }
    let Some((first, rest)) = list.split_first() else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_filter_in_list",
            "native filter IN list requires at least one value",
        ));
    };

    let op = if negated { BinaryOp::NotEq } else { BinaryOp::Eq };
    let combine_op = if negated { BinaryOperator::NullLogicalAnd } else { BinaryOperator::NullLogicalOr };
    let mut lowered = lower_filter_binary_expr(expr, op, first, Some(&ArrowDataType::Boolean))?;
    for item in rest {
        lowered = cudf_binary_expr(
            lowered,
            combine_op,
            lower_filter_binary_expr(expr, op, item, Some(&ArrowDataType::Boolean))?,
        );
    }
    Ok(lowered)
}

fn validate_in_list_literal_values(list: &[Expr]) -> Result<()> {
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "nexus_query_engine.expr.lower_in_list",
                "native IN list values must be literals",
            ));
        };
        if matches!(value, LiteralValue::Null(_)) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "nexus_query_engine.expr.lower_in_list",
                "native IN list does not yet model SQL null-containing list semantics",
            ));
        }
    }
    Ok(())
}

fn validate_like_escape(escape: &str) -> Result<()> {
    if escape.len() <= 1 && escape.is_ascii() {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_like",
            "native LIKE escape must be empty or one ASCII byte",
        )
        .with_fact("escape", escape))
    }
}

fn lower_case(branches: &[CaseWhen], else_expr: &Expr) -> Result<cudf::Expr> {
    let mut branches = branches.iter().rev();
    let Some(last) = branches.next() else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_case",
            "native CASE requires at least one WHEN branch",
        ));
    };

    let mut lowered = when(lower_expr(&last.when)?).then(lower_expr(&last.then)?).otherwise(lower_expr(else_expr)?);
    for branch in branches {
        lowered = when(lower_expr(&branch.when)?).then(lower_expr(&branch.then)?).otherwise(lowered);
    }
    Ok(lowered)
}

fn lower_extract(field: DatePart, expr: &Expr) -> Result<cudf::Expr> {
    if matches!(field, DatePart::Minute) {
        let raw_seconds = lower_timestamp_second_integer_cast_raw(expr)?;
        return Ok(lower_timestamp_second_minute(raw_seconds).cast(cudf::DataType::new(cudf::TypeId::Int32)));
    }
    let lowered = lower_expr(expr)?;
    let component = match field {
        DatePart::Year => lowered.dt().year(),
        DatePart::Month => lowered.dt().month(),
        DatePart::Day => lowered.dt().day(),
        DatePart::Minute => unreachable!("minute extraction is handled before datetime component lowering"),
        DatePart::Hour | DatePart::Second | DatePart::Millisecond | DatePart::Microsecond | DatePart::Nanosecond => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "nexus_query_engine.expr.lower_extract",
                "native EXTRACT does not support this field",
            )
            .with_fact("field", field.to_string()));
        },
    };
    Ok(component.cast(cudf::DataType::new(cudf::TypeId::Int32)))
}

fn lower_timestamp_second_minute(expr: cudf::Expr) -> cudf::Expr {
    cudf_binary_expr(expr.floor_div(lit(60_i64)), BinaryOperator::Pymod, lit(60_i64))
}

fn lower_date_trunc(field: DatePart, expr: &Expr, output_type: &ArrowDataType) -> Result<cudf::Expr> {
    let frequency = match field {
        DatePart::Day => cudf::ops::datetime::DatetimeRoundingFrequency::Day,
        DatePart::Hour => cudf::ops::datetime::DatetimeRoundingFrequency::Hour,
        DatePart::Minute => cudf::ops::datetime::DatetimeRoundingFrequency::Minute,
        DatePart::Second => cudf::ops::datetime::DatetimeRoundingFrequency::Second,
        DatePart::Millisecond => cudf::ops::datetime::DatetimeRoundingFrequency::Millisecond,
        DatePart::Microsecond => cudf::ops::datetime::DatetimeRoundingFrequency::Microsecond,
        DatePart::Nanosecond => cudf::ops::datetime::DatetimeRoundingFrequency::Nanosecond,
        DatePart::Year | DatePart::Month => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "nexus_query_engine.expr.lower_date_trunc",
                "native DATE_TRUNC currently supports only fixed-duration fields",
            )
            .with_fact("field", field.to_string()));
        },
    };
    if !matches!(output_type, ArrowDataType::Timestamp(_, None)) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_date_trunc",
            "native DATE_TRUNC output type is not supported",
        )
        .with_fact("field", field.to_string())
        .with_fact("output_type", format!("{output_type:?}")));
    }
    let target = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.expr.lower_date_trunc",
            "native DATE_TRUNC output type is not supported",
        )
        .with_fact("output_type", format!("{output_type:?}"))
    })?;
    Ok(lower_expr(expr)?.dt().floor(frequency).cast(target))
}

fn lower_timestamp_second_integer_cast_raw(expr: &Expr) -> Result<cudf::Expr> {
    let Expr::Cast { expr, target_type } = expr else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_timestamp_second_raw",
            "native second timestamp minute operations currently require an integer-to-timestamp cast",
        )
        .with_fact("expression", expr));
    };
    if !matches!(target_type, ArrowDataType::Timestamp(TimeUnit::Second, None)) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.expr.lower_timestamp_second_raw",
            "native second timestamp minute operations require Timestamp(Second, None)",
        )
        .with_fact("target_type", format!("{target_type:?}")));
    }
    Ok(lower_expr(expr)?.cast(cudf::DataType::new(cudf::TypeId::Int64)))
}

// cuDF-ready sort keys split into the parallel vectors expected by execution.
#[doc(hidden)]
pub struct LoweredSort {
    #[doc(hidden)]
    pub columns: Vec<String>,
    #[doc(hidden)]
    pub descending: Vec<bool>,
    #[doc(hidden)]
    pub nulls_first: Vec<bool>,
}

#[doc(hidden)]
pub fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "nexus_query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

#[doc(hidden)]
pub fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
    match expr {
        Expr::Column(column) => Ok((column.name().to_string(), column.display_name().to_string())),
        Expr::Alias { expr, alias } => Ok((lower_column_expr(expr, operation)?, alias.clone())),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

#[doc(hidden)]
pub fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
    match expr {
        Expr::Column(column) => Ok(column.name().to_string()),
        Expr::Alias { expr, .. } => lower_column_expr(expr, operation),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int64_literals(count: usize) -> Vec<Expr> {
        (0..count).map(|value| Expr::literal(LiteralValue::Int64(value as i64))).collect()
    }

    #[test]
    fn large_filter_in_list_stays_as_cudf_in_list() {
        let predicate = Expr::in_list(
            Expr::column("k"),
            int64_literals(ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT + 1),
        );

        let lowered = lower_filter_predicate_expr(&predicate).unwrap();

        match lowered {
            cudf::Expr::InList { values, negated, .. } => {
                assert_eq!(values.len(), ExpressionLoweringPolicy::DEFAULT_ROW_PREDICATE_IN_LIST_EXPANSION_LIMIT + 1);
                assert!(!negated);
            },
            other => panic!("large filter IN list should not expand to row-predicate tree: {}", expr_kind(&other)),
        }
    }

    #[test]
    fn small_filter_in_list_can_use_row_predicate_tree() {
        let predicate = Expr::in_list(Expr::column("k"), int64_literals(2));

        let lowered = lower_filter_predicate_expr(&predicate).unwrap();

        assert!(
            matches!(lowered, cudf::Expr::BinaryOp { .. }),
            "small filter IN list should still lower to a row-predicate-friendly tree"
        );
    }

    fn expr_kind(expr: &cudf::Expr) -> &'static str {
        match expr {
            cudf::Expr::Column(_) => "column",
            cudf::Expr::Literal(_) => "literal",
            cudf::Expr::BinaryOp { .. } => "binary_op",
            cudf::Expr::BinaryOpWithType { .. } => "binary_op_with_type",
            cudf::Expr::UnaryOp { .. } => "unary_op",
            cudf::Expr::Round { .. } => "round",
            cudf::Expr::Cast { .. } => "cast",
            cudf::Expr::TryCast { .. } => "try_cast",
            cudf::Expr::IsNull(_) => "is_null",
            cudf::Expr::IsNotNull(_) => "is_not_null",
            cudf::Expr::Alias { .. } => "alias",
            cudf::Expr::FillNaScalar { .. } => "fill_na_scalar",
            cudf::Expr::FillNaExpr { .. } => "fill_na_expr",
            cudf::Expr::FillNullPolicy { .. } => "fill_null_policy",
            cudf::Expr::Scan { .. } => "scan",
            cudf::Expr::Reduce { .. } => "reduce",
            cudf::Expr::StringOp { .. } => "string_op",
            cudf::Expr::StringConcat(_) => "string_concat",
            cudf::Expr::DateTimeComponent { .. } => "datetime_component",
            cudf::Expr::DateTimeFloor { .. } => "datetime_floor",
            cudf::Expr::DateTimeOffset { .. } => "datetime_offset",
            cudf::Expr::IfElse { .. } => "if_else",
            cudf::Expr::InList { .. } => "in_list",
        }
    }
}
