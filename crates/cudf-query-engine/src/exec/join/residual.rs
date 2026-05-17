use std::collections::{BTreeMap, BTreeSet};

use cudf::ops::ast::{AstExpression, AstOperator};

use super::*;

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
            "cudf_query_engine.join.residual_columns",
            "native join residual predicates must reference at least one column from each input",
        )
        .with_fact("predicate", predicate));
    }
    Ok(JoinResidualColumns::new(left.into_iter().collect(), right.into_iter().collect()))
}

fn collect_join_expr_columns(expr: &Expr, left: &mut BTreeSet<String>, right: &mut BTreeSet<String>) -> Result<()> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(crate::expr::ColumnQualifier::Input { index: 0 }) => {
                left.insert(column.name().to_string());
                Ok(())
            },
            Some(crate::expr::ColumnQualifier::Input { index: 1 }) => {
                right.insert(column.name().to_string());
                Ok(())
            },
            qualifier => Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.residual_columns",
                "native join residual predicates require input-qualified column references",
            )
            .with_fact("column", column.name())
            .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(_) => Ok(()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => collect_join_expr_columns(inner, left, right),
        Expr::Binary { left: lhs, right: rhs, .. } => {
            collect_join_expr_columns(lhs, left, right)?;
            collect_join_expr_columns(rhs, left, right)
        },
        Expr::InList { expr, list, .. } => {
            collect_join_expr_columns(expr, left, right)?;
            for item in list {
                collect_join_expr_columns(item, left, right)?;
            }
            Ok(())
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_join_expr_columns(&branch.when, left, right)?;
                collect_join_expr_columns(&branch.then, left, right)?;
            }
            collect_join_expr_columns(else_expr, left, right)
        },
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Lower(_) | Expr::Upper(_) | Expr::Extract { .. } => {
            Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.residual_columns",
                "native join residual predicates do not yet support this expression family",
            )
            .with_fact("expression", expr))
        },
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
            "cudf_query_engine.join.lower_residual",
            "join residual lowering requires a residual predicate",
        ));
    };
    lower_join_ast_expr(context, predicate, residual_columns)
}

pub(in crate::exec) fn lower_join_match_predicate(
    context: &ExecutionContext,
    node: &JoinNode,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    let mut predicate = node.spec.residual_predicate.clone().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.lower_residual",
            "join match predicate lowering requires a residual predicate",
        )
    })?;

    for key in &node.spec.keys {
        let equality = join_key_match_expr(key, node.spec.null_equality);
        predicate = Expr::binary(equality, BinaryOp::And, predicate);
    }

    lower_join_ast_expr(context, &predicate, residual_columns)
}

fn join_key_match_expr(key: &crate::plan::JoinKey, null_equality: JoinNullEquality) -> Expr {
    let left = Expr::Column(key.left.clone());
    let right = Expr::Column(key.right.clone());
    let equality = Expr::binary(left.clone(), BinaryOp::Eq, right.clone());
    match null_equality {
        JoinNullEquality::NullsDoNotMatch => equality,
        JoinNullEquality::NullsMatch => {
            Expr::binary(equality, BinaryOp::Or, Expr::binary(Expr::is_null(left), BinaryOp::And, Expr::is_null(right)))
        },
    }
}

fn lower_join_ast_expr(
    context: &ExecutionContext,
    expr: &Expr,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let (side, index) = match column.qualifier() {
                Some(crate::expr::ColumnQualifier::Input { index: 0 }) => {
                    (JoinSide::Left, residual_columns.index_of(JoinSide::Left, column.name()))
                },
                Some(crate::expr::ColumnQualifier::Input { index: 1 }) => {
                    (JoinSide::Right, residual_columns.index_of(JoinSide::Right, column.name()))
                },
                qualifier => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.lower_residual",
                        "join residual column must be input-qualified",
                    )
                    .with_fact("column", column.name())
                    .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string())));
                },
            };
            let Some(index) = index else {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.join.lower_residual",
                    "join residual column was not included in predicate columns",
                )
                .with_fact("side", side)
                .with_fact("column", column.name()));
            };
            let index = i32::try_from(index).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.lower_residual",
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
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST column reference",
                    error,
                )
            })
        },
        Expr::Literal(value) => lower_join_ast_literal(context, value),
        Expr::Not(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::Not, input).map_err(|error| {
                QueryError::from_cudf("cudf_query_engine.join.lower_residual", "failed to build cudf AST NOT", error)
            })
        },
        Expr::IsNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })
        },
        Expr::IsNotNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })?;
            AstExpression::unary(AstOperator::Not, is_null).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NOT NULL",
                    error,
                )
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
                        "cudf_query_engine.join.lower_residual",
                        "native join residual cast is not supported by the cudf AST contract",
                    )
                    .with_fact("target_type", format!("{target_type:?}")));
                },
            };
            expression.map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST cast expression",
                    error,
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        Expr::Binary { left, op, right, .. } => {
            let left = lower_join_ast_expr(context, left, residual_columns)?;
            let right = lower_join_ast_expr(context, right, residual_columns)?;
            let op = lower_join_ast_operator(*op)?;
            AstExpression::binary(op, left, right).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST binary expression",
                    error,
                )
            })
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.lower_residual",
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
                            "cudf_query_engine.join.lower_residual",
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
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IN list expression",
                    error,
                )
                .with_fact("negated", negated.to_string())
                .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => lower_join_ast_expr(context, expr, residual_columns),
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Case { .. }
        | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.lower_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_join_ast_operator(op: BinaryOp) -> Result<AstOperator> {
    match op {
        BinaryOp::Add => Ok(AstOperator::Add),
        BinaryOp::Subtract => Ok(AstOperator::Sub),
        BinaryOp::Multiply => Ok(AstOperator::Mul),
        BinaryOp::Divide => Ok(AstOperator::Div),
        BinaryOp::Modulo => Ok(AstOperator::Mod),
        BinaryOp::Eq => Ok(AstOperator::Equal),
        BinaryOp::NotEq => Ok(AstOperator::NotEqual),
        BinaryOp::Lt => Ok(AstOperator::Less),
        BinaryOp::LtEq => Ok(AstOperator::LessEqual),
        BinaryOp::Gt => Ok(AstOperator::Greater),
        BinaryOp::GtEq => Ok(AstOperator::GreaterEqual),
        BinaryOp::And => Ok(AstOperator::LogicalAnd),
        BinaryOp::Or => Ok(AstOperator::LogicalOr),
    }
}

fn lower_join_ast_literal(context: &ExecutionContext, value: &LiteralValue) -> Result<AstExpression> {
    let runtime = context.runtime();
    let scalar = match value {
        LiteralValue::Boolean(value) => runtime.scalar(*value),
        LiteralValue::Int8(value) => runtime.scalar(*value),
        LiteralValue::Int16(value) => runtime.scalar(*value),
        LiteralValue::Int32(value) => runtime.scalar(*value),
        LiteralValue::Int64(value) => runtime.scalar(*value),
        LiteralValue::UInt8(value) => runtime.scalar(*value),
        LiteralValue::UInt16(value) => runtime.scalar(*value),
        LiteralValue::UInt32(value) => runtime.scalar(*value),
        LiteralValue::UInt64(value) => runtime.scalar(*value),
        LiteralValue::Float32(value) => runtime.scalar(*value),
        LiteralValue::Float64(value) => runtime.scalar(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => runtime.string_scalar(value),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            runtime.decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        LiteralValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), cudf::TimestampResolution::Days),
        LiteralValue::Timestamp { raw, unit } => {
            runtime.timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            runtime.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.lower_residual",
                "join residual AST literals do not support interval values",
            )
            .with_fact("literal", value));
        },
        LiteralValue::Null(data_type) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.lower_residual",
                "join residual AST literals do not yet support typed null values",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.lower_residual",
            "failed to build cudf scalar for join residual AST literal",
            error,
        )
        .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.join.lower_residual", "failed to build cudf AST literal", error)
    })
}
