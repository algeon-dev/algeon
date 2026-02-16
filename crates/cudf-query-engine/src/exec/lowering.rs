use cudf::ops::{
    binary::BinaryOperator,
    expr::{lit_decimal128_raw, lit_duration_raw, lit_timestamp_raw},
    lit_null,
    strings::StringSliceOptions,
    when,
};

use super::*;
use crate::{
    expr::{CaseWhen, DatePart},
    plan::ProjectionOutput,
};

#[path = "lowering/aggregate.rs"]
mod aggregate;

pub(super) use aggregate::*;

pub(super) fn lower_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Column(column) => Ok(col(column.name())),
        Expr::Literal(value) => lower_literal(value),
        Expr::Not(expr) => Ok(lower_expr(expr)?.not()),
        Expr::IsNull(expr) => Ok(lower_expr(expr)?.is_null()),
        Expr::IsNotNull(expr) => Ok(lower_expr(expr)?.is_not_null()),
        Expr::Cast { expr, target_type } => {
            let target_dtype = cudf_dtype_from_arrow_type(target_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.expr.lower_cast",
                    "native execution does not support casts to this target type",
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })?;
            Ok(lower_expr_with_expected_type(expr, Some(target_type))?.cast(target_dtype))
        },
        Expr::Binary { left, op, right, output_type } => lower_binary_expr(left, *op, right, output_type.as_ref()),
        Expr::Like { expr, pattern, escape, negated } => {
            let expr = match escape {
                Some(escape) => {
                    validate_like_escape(escape)?;
                    lower_expr(expr)?.str().like_escape(pattern.clone(), escape.clone())
                },
                None => lower_expr(expr)?.str().like(pattern.clone()),
            };
            Ok(if *negated { expr.not() } else { expr })
        },
        Expr::Substring { expr, start, length } => lower_substring(expr, *start, *length),
        Expr::Lower(expr) => Ok(lower_expr(expr)?.str().lower()),
        Expr::Upper(expr) => Ok(lower_expr(expr)?.str().upper()),
        Expr::InList { expr, list, negated } => lower_in_list(expr, list, *negated),
        Expr::Case { branches, else_expr } => lower_case(branches, else_expr),
        Expr::Extract { field, expr } => lower_extract(*field, expr),
        Expr::Alias { expr, alias } => Ok(lower_expr(expr)?.alias(alias)),
    }
}

pub(super) fn lower_projection_output_expr(output: &ProjectionOutput) -> Result<cudf::Expr> {
    lower_expr_with_expected_type(&output.expr, Some(output.field.data_type()))
}

fn lower_expr_with_expected_type(expr: &Expr, expected_type: Option<&ArrowDataType>) -> Result<cudf::Expr> {
    match expr {
        Expr::Cast { expr, target_type } => {
            let target_dtype = cudf_dtype_from_arrow_type(target_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.expr.lower_cast",
                    "native execution does not support casts to this target type",
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })?;
            Ok(lower_expr_with_expected_type(expr, Some(target_type))?.cast(target_dtype))
        },
        Expr::Binary { left, op, right, output_type } => {
            lower_binary_expr_with_expected_type(left, *op, right, output_type.as_ref(), expected_type)
        },
        Expr::Alias { expr, alias } => Ok(lower_expr_with_expected_type(expr, expected_type)?.alias(alias)),
        _ => lower_expr(expr),
    }
}

pub(super) fn lower_literal(value: &LiteralValue) -> Result<cudf::Expr> {
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
        LiteralValue::Timestamp { raw, unit } => Ok(lit_timestamp_raw(*raw, timestamp_unit_to_cudf_resolution(unit))),
        LiteralValue::Duration { raw, unit } => Ok(lit_duration_raw(*raw, timestamp_unit_to_cudf_resolution(unit))),
        LiteralValue::IntervalMonthDayNano { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_literal",
            "interval literals must be lowered as part of timestamp/date interval arithmetic",
        )
        .with_fact("literal", value)),
        LiteralValue::Null(data_type) => {
            let dtype = cudf_dtype_from_arrow_type(data_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.expr.lower_literal",
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

fn lower_binary_expr_with_expected_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    expected_type: Option<&ArrowDataType>,
) -> Result<cudf::Expr> {
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

fn lower_datetime_interval_offset(expr: &Expr, months: i32, days: i32, nanoseconds: i64) -> Result<cudf::Expr> {
    Ok(lower_expr(expr)?.dt().offset(months, days, nanoseconds))
}

fn interval_month_day_nano(expr: &Expr) -> Option<(i32, i32, i64)> {
    let Expr::Literal(LiteralValue::IntervalMonthDayNano { months, days, nanoseconds }) = expr else {
        return None;
    };
    Some((*months, *days, *nanoseconds))
}

fn negate_interval(months: i32, days: i32, nanoseconds: i64) -> Result<(i32, i32, i64)> {
    let months = months.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_interval",
            "interval month component cannot be negated without overflow",
        )
        .with_fact("months", months)
    })?;
    let days = days.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_interval",
            "interval day component cannot be negated without overflow",
        )
        .with_fact("days", days)
    })?;
    let nanoseconds = nanoseconds.checked_neg().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_interval",
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
            "cudf_query_engine.expr.lower_binary",
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
            "cudf_query_engine.expr.lower_substring",
            "native substring requires a one-based positive start position",
        )
        .with_fact("start", start));
    }
    if let Some(length) = length
        && length < 0
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_substring",
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
                    "cudf_query_engine.expr.lower_substring",
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

fn lower_in_list(expr: &Expr, list: &[Expr], negated: bool) -> Result<cudf::Expr> {
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.expr.lower_in_list",
                "native IN list values must be literals",
            ));
        };
        if matches!(value, LiteralValue::Null(_)) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.expr.lower_in_list",
                "native IN list does not yet model SQL null-containing list semantics",
            ));
        }
    }
    let values = list.iter().map(lower_expr).collect::<Result<Vec<_>>>()?;
    Ok(if negated { lower_expr(expr)?.is_not_in_list(values) } else { lower_expr(expr)?.is_in_list(values) })
}

fn validate_like_escape(escape: &str) -> Result<()> {
    if escape.len() <= 1 && escape.is_ascii() {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_like",
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
            "cudf_query_engine.expr.lower_case",
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
    let expr = lower_expr(expr)?.dt();
    let component = match field {
        DatePart::Year => expr.year(),
        DatePart::Month => expr.month(),
        DatePart::Day => expr.day(),
    };
    Ok(component.cast(cudf::DataType::new(cudf::TypeId::Int32)))
}

pub(super) struct LoweredSort {
    pub(super) columns: Vec<String>,
    pub(super) descending: Vec<bool>,
    pub(super) nulls_first: Vec<bool>,
}

pub(super) fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "cudf_query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

pub(super) fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
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

pub(super) fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
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
