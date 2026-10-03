use std::collections::{BTreeMap, BTreeSet};

use cudf::{
    ScalarValue, TimestampResolution,
    ops::ast::{AstExpression, AstOperator},
};
use query_planning::expr::{ColumnQualifier, interval_month_day_nano};

use super::{
    ArrowDataType, BinaryOp, ExecutionContext, Expr, JoinNode, JoinSide, LiteralValue, QueryError, QueryErrorCode,
    Result, cudf_query_error,
};
use crate::datatypes::{UnrepresentableLiteral, cudf_scalar_value};

#[derive(Debug, Clone)]
struct ResidualColumnLayout {
    names: Vec<String>,
    positions: BTreeMap<String, usize>,
}

impl ResidualColumnLayout {
    fn new(names: Vec<String>) -> Self {
        let positions = names.iter().enumerate().map(|(index, name)| (name.clone(), index)).collect();
        Self { names, positions }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }

    fn names(&self) -> &[String] {
        &self.names
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.positions.get(name).copied()
    }
}

#[derive(Debug, Clone)]
pub(in crate::exec) struct JoinResidualColumns {
    left: ResidualColumnLayout,
    right: ResidualColumnLayout,
}

impl Default for JoinResidualColumns {
    fn default() -> Self {
        Self { left: ResidualColumnLayout::empty(), right: ResidualColumnLayout::empty() }
    }
}

impl JoinResidualColumns {
    fn new(left: Vec<String>, right: Vec<String>) -> Self {
        Self { left: ResidualColumnLayout::new(left), right: ResidualColumnLayout::new(right) }
    }

    pub(in crate::exec) fn names(&self, side: JoinSide) -> &[String] {
        match side {
            JoinSide::Left => self.left.names(),
            JoinSide::Right => self.right.names(),
        }
    }

    fn index_of(&self, side: JoinSide, name: &str) -> Option<usize> {
        match side {
            JoinSide::Left => self.left.index_of(name),
            JoinSide::Right => self.right.index_of(name),
        }
    }
}

pub(in crate::exec) fn residual_input_columns(node: &JoinNode) -> Result<JoinResidualColumns> {
    let Some(predicate) = &node.spec.residual_predicate else {
        return Ok(JoinResidualColumns::default());
    };
    let mut left = BTreeSet::new();
    let mut right = BTreeSet::new();
    collect_join_expr_columns(predicate, &mut left, &mut right)?;
    for key in &node.spec.keys {
        left.insert(key.left.name().to_string());
        right.insert(key.right.name().to_string());
    }
    if left.is_empty() || right.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.residual_columns",
            "native join residual predicates must reference at least one column from each input",
        )
        .with_fact("predicate", predicate));
    }
    Ok(JoinResidualColumns::new(left.into_iter().collect(), right.into_iter().collect()))
}

fn collect_join_expr_columns(expr: &Expr, left: &mut BTreeSet<String>, right: &mut BTreeSet<String>) -> Result<()> {
    let mut collect = |input, name: &str| match input {
        0 => {
            left.insert(name.to_string());
        },
        1 => {
            right.insert(name.to_string());
        },
        _ => unreachable!("join residual walker only yields binary input columns"),
    };
    walk_join_residual_columns(
        expr,
        "query_engine.join.residual_columns",
        "native join residual predicates require input-qualified column references",
        &mut collect,
    )
}

pub(in crate::exec) fn walk_join_residual_columns(
    expr: &Expr,
    operation: &'static str,
    bad_qualifier_message: &'static str,
    on_input: &mut impl FnMut(usize, &str),
) -> Result<()> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(ColumnQualifier::Input { index }) if matches!(*index, 0 | 1) => {
                on_input(*index, column.name());
                Ok(())
            },
            qualifier => Err(QueryError::new(QueryErrorCode::UnsupportedExpression, operation, bad_qualifier_message)
                .with_fact("column", column.name())
                .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(_) => Ok(()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::DateTrunc { expr: inner, .. }
        | Expr::Abs(inner)
        | Expr::UnaryMath { expr: inner, .. }
        | Expr::Round { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => {
            walk_join_residual_columns(inner, operation, bad_qualifier_message, on_input)
        },
        Expr::Binary { left: lhs, right: rhs, .. } => {
            walk_join_residual_columns(lhs, operation, bad_qualifier_message, on_input)?;
            walk_join_residual_columns(rhs, operation, bad_qualifier_message, on_input)
        },
        Expr::IsDistinctFrom { left: lhs, right: rhs } | Expr::IsNotDistinctFrom { left: lhs, right: rhs } => {
            walk_join_residual_columns(lhs, operation, bad_qualifier_message, on_input)?;
            walk_join_residual_columns(rhs, operation, bad_qualifier_message, on_input)
        },
        Expr::InList { expr, list, .. } => {
            walk_join_residual_columns(expr, operation, bad_qualifier_message, on_input)?;
            for item in list {
                walk_join_residual_columns(item, operation, bad_qualifier_message, on_input)?;
            }
            Ok(())
        },
        Expr::Concat(exprs) => {
            for expr in exprs {
                walk_join_residual_columns(expr, operation, bad_qualifier_message, on_input)?;
            }
            Ok(())
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                walk_join_residual_columns(&branch.when, operation, bad_qualifier_message, on_input)?;
                walk_join_residual_columns(&branch.then, operation, bad_qualifier_message, on_input)?;
            }
            walk_join_residual_columns(else_expr, operation, bad_qualifier_message, on_input)
        },
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

pub(in crate::exec) fn lower_join_residual_predicate(
    context: &ExecutionContext,
    node: &JoinNode,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    let Some(predicate) = &node.spec.residual_predicate else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.join.lower_residual",
            "join residual lowering requires a residual predicate",
        ));
    };
    lower_join_ast_expr(context, predicate, residual_columns)
}

fn lower_join_ast_expr(
    context: &ExecutionContext,
    expr: &Expr,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let (side, index) = match column.qualifier() {
                Some(ColumnQualifier::Input { index: 0 }) => {
                    (JoinSide::Left, residual_columns.index_of(JoinSide::Left, column.name()))
                },
                Some(ColumnQualifier::Input { index: 1 }) => {
                    (JoinSide::Right, residual_columns.index_of(JoinSide::Right, column.name()))
                },
                qualifier => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "query_engine.join.lower_residual",
                        "join residual column must be input-qualified",
                    )
                    .with_fact("column", column.name())
                    .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string())));
                },
            };
            let Some(index) = index else {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.join.lower_residual",
                    "join residual column was not included in predicate columns",
                )
                .with_fact("side", side)
                .with_fact("column", column.name()));
            };
            let index = i32::try_from(index).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.join.lower_residual",
                    "join residual predicate column index exceeds i32 range",
                )
                .with_fact("side", side)
                .with_fact("column", column.name())
                .with_fact("index", index)
            })?;
            let expression = match side {
                JoinSide::Left => AstExpression::left_column(index),
                JoinSide::Right => AstExpression::right_column(index),
            };
            expression.map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST column reference", error)
            })
        },
        Expr::Literal(value) => lower_join_ast_literal(context, value),
        Expr::Not(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::Not, input).map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST NOT", error)
            })
        },
        Expr::IsNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST IS NULL", error)
            })
        },
        Expr::IsNotNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST IS NULL", error)
            })?;
            AstExpression::unary(AstOperator::Not, is_null).map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST IS NOT NULL", error)
            })
        },
        Expr::Cast { expr, target_type } => {
            let input = lower_join_ast_expr(context, expr, residual_columns)?;
            let expression = match target_type {
                ArrowDataType::Int64 => AstExpression::cast_to_int64(input),
                ArrowDataType::UInt64 => AstExpression::cast_to_uint64(input),
                ArrowDataType::Float64 => AstExpression::cast_to_float64(input),
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "query_engine.join.lower_residual",
                        "native join residual cast is not supported by the cudf AST contract",
                    )
                    .with_fact("target_type", format!("{target_type:?}")));
                },
            };
            expression.map_err(|error| {
                cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST cast expression", error)
                    .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        Expr::Binary { left, op, right, .. }
            if join_ast_operator_is_comparison(*op)
                && (expr_contains_interval_literal(left) || expr_contains_interval_literal(right)) =>
        {
            lower_join_ast_datetime_interval_comparison(context, left, *op, right, residual_columns)
        },
        Expr::Binary { left, op, right, .. }
            if expr_contains_interval_literal(left) || expr_contains_interval_literal(right) =>
        {
            Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.join.lower_residual",
                "join residual interval arithmetic is supported only inside a comparison",
            ))
        },
        Expr::Binary { left, op, right, .. } => {
            let left = lower_join_ast_expr(context, left, residual_columns)?;
            let right = lower_join_ast_expr(context, right, residual_columns)?;
            let op = lower_join_ast_operator(*op)?;
            AstExpression::binary(op, left, right).map_err(|error| {
                cudf_query_error(
                    "query_engine.join.lower_residual",
                    "failed to build cudf AST binary expression",
                    error,
                )
            })
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.join.lower_residual",
                    "native join residual IN list must not be empty",
                ));
            }
            let input = lower_join_ast_expr(context, expr, residual_columns)?;
            let values = list
                .iter()
                .map(|value| {
                    if !matches!(value, Expr::Literal(_)) {
                        return Err(QueryError::new(
                            QueryErrorCode::UnsupportedExpression,
                            "query_engine.join.lower_residual",
                            "native join residual IN list values must be literals",
                        )
                        .with_fact("value", value));
                    }
                    lower_join_ast_expr(context, value, residual_columns)
                })
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.join.lower_residual",
                    "failed to build cudf AST IN list expression",
                    error,
                )
                .with_fact("negated", negated.to_string())
                .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => lower_join_ast_expr(context, expr, residual_columns),
        Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::UnaryMath { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_join_ast_datetime_interval_comparison(
    context: &ExecutionContext,
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    let (diff_left, raw_days) = if let Some((base, raw_days)) = interval_date_offset_expr(right)? {
        let left = lower_join_ast_expr(context, left, residual_columns)?;
        let base = lower_join_ast_expr(context, base, residual_columns)?;
        (
            AstExpression::binary(AstOperator::Sub, left, base).map_err(|error| {
                cudf_query_error(
                    "query_engine.join.lower_residual",
                    "failed to build cudf AST timestamp difference for interval comparison",
                    error,
                )
            })?,
            raw_days,
        )
    } else if let Some((base, raw_days)) = interval_date_offset_expr(left)? {
        let base = lower_join_ast_expr(context, base, residual_columns)?;
        let right = lower_join_ast_expr(context, right, residual_columns)?;
        (
            AstExpression::binary(AstOperator::Sub, base, right).map_err(|error| {
                cudf_query_error(
                    "query_engine.join.lower_residual",
                    "failed to build cudf AST timestamp difference for interval comparison",
                    error,
                )
            })?,
            raw_days.checked_neg().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.join.lower_residual",
                    "join residual interval day count cannot be negated without overflow",
                )
                .with_fact("raw_days", raw_days)
            })?,
        )
    } else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual interval comparison requires one datetime plus/minus fixed-day interval operand",
        ));
    };
    let duration = lower_duration_days_ast_literal(context, raw_days)?;
    let op = lower_join_ast_operator(op)?;
    AstExpression::binary(op, diff_left, duration).map_err(|error| {
        cudf_query_error(
            "query_engine.join.lower_residual",
            "failed to build cudf AST datetime interval comparison expression",
            error,
        )
    })
}

fn interval_date_offset_expr(expr: &Expr) -> Result<Option<(&Expr, i32)>> {
    let Expr::Binary { left, op, right, .. } = expr else {
        return Ok(None);
    };
    let left_interval = interval_month_day_nano(left);
    let right_interval = interval_month_day_nano(right);
    if left_interval.is_none() && right_interval.is_none() {
        return Ok(None);
    }
    if left_interval.is_some() == right_interval.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual interval arithmetic requires exactly one interval literal operand",
        ));
    }
    if let Some(interval) = left_interval {
        if *op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.join.lower_residual",
                "join residual interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        return Ok(Some((right, fixed_day_interval_raw(interval.0, interval.1, interval.2)?)));
    }
    let interval = right_interval.expect("right interval checked above");
    if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual interval arithmetic supports only datetime plus/minus interval",
        ));
    }
    let raw_days = fixed_day_interval_raw(interval.0, interval.1, interval.2)?;
    let raw_days = if *op == BinaryOp::Subtract {
        raw_days.checked_neg().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.join.lower_residual",
                "join residual interval day count cannot be negated without overflow",
            )
            .with_fact("raw_days", raw_days)
        })?
    } else {
        raw_days
    };
    Ok(Some((left, raw_days)))
}

fn lower_duration_days_ast_literal(context: &ExecutionContext, value: i32) -> Result<AstExpression> {
    let scalar = context
        .runtime()?
        .scalar(ScalarValue::Duration { raw: i64::from(value), resolution: TimestampResolution::Days })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.join.lower_residual",
                "failed to build duration-days scalar for join residual interval literal",
                error,
            )
            .with_fact("value", value)
        })?;
    AstExpression::literal(scalar).map_err(|error| {
        cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST duration-days literal", error)
            .with_fact("value", value)
    })
}

fn fixed_day_interval_raw(months: i32, days: i32, nanoseconds: i64) -> Result<i32> {
    const NANOS_PER_DAY: i64 = 86_400_000_000_000;
    if months != 0 || nanoseconds % NANOS_PER_DAY != 0 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual AST interval arithmetic currently supports only fixed day-resolution intervals",
        )
        .with_fact("months", months)
        .with_fact("nanoseconds", nanoseconds));
    }
    let nanos_days = i32::try_from(nanoseconds / NANOS_PER_DAY).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual interval day count exceeds i32 range",
        )
        .with_fact("nanoseconds", nanoseconds)
    })?;
    days.checked_add(nanos_days).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.lower_residual",
            "join residual interval day count overflows i32",
        )
        .with_fact("days", days)
        .with_fact("nanoseconds", nanoseconds)
    })
}

fn expr_contains_interval_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }) => true,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Abs(expr)
        | Expr::UnaryMath { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Alias { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Like { expr, .. } => expr_contains_interval_literal(expr),
        Expr::Binary { left, right, .. } => {
            expr_contains_interval_literal(left) || expr_contains_interval_literal(right)
        },
        Expr::Concat(exprs) => exprs.iter().any(expr_contains_interval_literal),
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            expr_contains_interval_literal(left) || expr_contains_interval_literal(right)
        },
        Expr::InList { expr, list, .. } => {
            expr_contains_interval_literal(expr) || list.iter().any(expr_contains_interval_literal)
        },
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                expr_contains_interval_literal(&branch.when) || expr_contains_interval_literal(&branch.then)
            }) || expr_contains_interval_literal(else_expr)
        },
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

fn join_ast_operator_is_comparison(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq)
}

fn lower_join_ast_operator(op: BinaryOp) -> Result<AstOperator> {
    match op {
        BinaryOp::Add => Ok(AstOperator::Add),
        BinaryOp::Subtract => Ok(AstOperator::Sub),
        BinaryOp::Multiply => Ok(AstOperator::Mul),
        BinaryOp::Divide => Ok(AstOperator::Div),
        BinaryOp::Modulo => Ok(AstOperator::Mod),
        BinaryOp::BitwiseAnd => Ok(AstOperator::BitwiseAnd),
        BinaryOp::BitwiseOr => Ok(AstOperator::BitwiseOr),
        BinaryOp::BitwiseXor => Ok(AstOperator::BitwiseXor),
        BinaryOp::Eq => Ok(AstOperator::Equal),
        BinaryOp::NotEq => Ok(AstOperator::NotEqual),
        BinaryOp::Lt => Ok(AstOperator::Less),
        BinaryOp::LtEq => Ok(AstOperator::LessEqual),
        BinaryOp::Gt => Ok(AstOperator::Greater),
        BinaryOp::GtEq => Ok(AstOperator::GreaterEqual),
        BinaryOp::And => Ok(AstOperator::NullLogicalAnd),
        BinaryOp::Or => Ok(AstOperator::NullLogicalOr),
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.join.residual",
            "native join residual AST does not support bit-shift operators",
        )
        .with_fact("operator", op.to_string())),
    }
}

fn lower_join_ast_literal(context: &ExecutionContext, value: &LiteralValue) -> Result<AstExpression> {
    let scalar_value = match cudf_scalar_value(value) {
        Ok(scalar_value) => scalar_value,
        Err(UnrepresentableLiteral::Interval) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.join.lower_residual",
                "join residual AST literals do not support interval values",
            )
            .with_fact("literal", value));
        },
        Err(UnrepresentableLiteral::Null(data_type)) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.join.lower_residual",
                "join residual AST literals do not yet support typed null values",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    };
    let scalar = context.runtime()?.scalar(scalar_value).map_err(|error| {
        cudf_query_error(
            "query_engine.join.lower_residual",
            "failed to build cudf scalar for join residual AST literal",
            error,
        )
        .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar).map_err(|error| {
        cudf_query_error("query_engine.join.lower_residual", "failed to build cudf AST literal", error)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_residual_column_walker_collects_binary_input_columns() {
        let expr = Expr::binary(
            Expr::is_not_null(Expr::input_column(0, "l_orderkey", "l_orderkey")),
            BinaryOp::And,
            Expr::binary(
                Expr::input_column(0, "l_quantity", "l_quantity"),
                BinaryOp::Gt,
                Expr::input_column(1, "p_size", "p_size"),
            ),
        );
        let mut columns = Vec::new();

        walk_join_residual_columns(&expr, "test.join_residual_columns", "bad qualifier", &mut |input, name| {
            columns.push((input, name.to_string()));
        })
        .expect("walker should accept binary input-qualified columns");

        assert_eq!(
            columns,
            vec![(0, "l_orderkey".to_string()), (0, "l_quantity".to_string()), (1, "p_size".to_string()),]
        );
    }

    #[test]
    fn join_residual_column_walker_rejects_non_binary_input_qualifier() {
        let expr = Expr::input_column(2, "bad_key", "bad_key");
        let error =
            walk_join_residual_columns(&expr, "test.join_residual_columns", "bad qualifier", &mut |_input, _name| {})
                .expect_err("walker should reject input qualifiers outside the binary join inputs");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
        assert_eq!(error.operation(), "test.join_residual_columns");
        assert_eq!(error.message(), "bad qualifier");
        assert_eq!(error.fact_value("column"), Some("bad_key"));
        assert_eq!(error.fact_value("qualifier"), Some("input2"));
    }
}
