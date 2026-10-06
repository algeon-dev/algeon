//! Expression-level capability analysis.
//!
//! Classifies whether each native [`Expr`] form is supported and, when not,
//! produces a stable `(reason, expression_family)` rejection for reports.

use super::{
    AggregateExpr, AggregateFunction, BinaryOp, Expr, LiteralValue, NodeId, QueryError, model::CapabilityDecision,
    rejection::Reason,
};
use crate::{
    error::{QueryErrorCode, Result},
    expr::{ColumnQualifier, DatePart, UnaryMathOp},
};

// Stable report facts for an expression capability miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpressionCapabilityRejection {
    reason: &'static str,
    expression_family: &'static str,
}

pub(super) fn expression_rejection_decision(
    node_id: NodeId,
    reason: Reason,
    expression: &Expr,
    rejection: ExpressionCapabilityRejection,
) -> CapabilityDecision {
    CapabilityDecision::rejected(
        reason,
        [
            ("node_id", node_id.to_string()),
            ("reason", rejection.reason.to_string()),
            ("expression_family", rejection.expression_family.to_string()),
            ("expression", expression.to_string()),
        ],
    )
}

/// The error a node rule reports when the backend cannot evaluate `expr`'s
/// form.
pub(crate) fn require_expression_capability(
    result: std::result::Result<(), ExpressionCapabilityRejection>,
    expr: &Expr,
) -> Result<()> {
    result.map_err(|rejection| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.plan.check_node",
            "native backend does not evaluate this expression form",
        )
        .with_fact("reason", rejection.reason)
        .with_fact("expression_family", rejection.expression_family)
        .with_fact("expression", expr.to_string())
    })
}

pub(crate) fn validate_aggregate_expr_capability(
    aggregate_expr: &AggregateExpr,
    is_scalar_aggregate: bool,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if aggregate_expr.distinct && aggregate_expr.function != AggregateFunction::Count {
        return Err(ExpressionCapabilityRejection {
            reason: "distinct_aggregate_requires_count",
            expression_family: "aggregate",
        });
    }
    if let Some(filter) = &aggregate_expr.filter {
        validate_general_expr_capability(filter).map_err(|rejection| ExpressionCapabilityRejection {
            reason: rejection.reason,
            expression_family: "aggregate_filter",
        })?;
    }
    if (is_scalar_aggregate || aggregate_expr.filter.is_some())
        && aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && is_count_star_expr(&aggregate_expr.expr)
    {
        return Ok(());
    }
    if aggregate_expr.function == AggregateFunction::NUnique && is_count_star_expr(&aggregate_expr.expr) {
        return Err(ExpressionCapabilityRejection {
            reason: "nunique_requires_value_expression",
            expression_family: "aggregate",
        });
    }
    validate_column_expr_capability(&aggregate_expr.expr)
}

fn is_count_star_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))
        || matches!(expr, Expr::Alias { expr, .. } if is_count_star_expr(expr))
}

pub(super) fn validate_named_column_expr_capability(
    expr: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_named_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

pub(crate) fn validate_column_expr_capability(expr: &Expr) -> std::result::Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

pub(crate) fn validate_general_expr_capability(expr: &Expr) -> std::result::Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_general_expr_capability(expr)
        },
        Expr::Cast { expr, .. }
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
        | Expr::RegexpReplace { expr, .. } => validate_general_expr_capability(expr),
        Expr::Extract { field, expr } => {
            validate_general_expr_capability(expr)?;
            validate_extract_field_capability(*field)
        },
        Expr::DateTrunc { field, expr, .. } => {
            validate_general_expr_capability(expr)?;
            validate_date_trunc_field_capability(*field)
        },
        Expr::Abs(expr) => validate_general_expr_capability(expr),
        Expr::UnaryMath { expr, .. } => validate_general_expr_capability(expr),
        Expr::Concat(exprs) => {
            if exprs.is_empty() {
                return Err(ExpressionCapabilityRejection {
                    reason: "concat_requires_argument",
                    expression_family: "concat",
                });
            }
            for expr in exprs {
                validate_general_expr_capability(expr)?;
            }
            Ok(())
        },
        Expr::Round { expr, .. } => validate_general_expr_capability(expr),
        Expr::Binary { left, op, right, .. } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_interval_binary_expr_capability(left, *op, right)
        },
        Expr::Binary { left, right, .. } => {
            validate_general_expr_capability(left)?;
            validate_general_expr_capability(right)
        },
        Expr::Like { expr, escape, .. } => {
            validate_general_expr_capability(expr)?;
            if matches!(escape, Some(escape) if escape.len() > 1 || !escape.is_ascii()) {
                return Err(ExpressionCapabilityRejection {
                    reason: "like_escape_must_be_empty_or_one_ascii_byte",
                    expression_family: "like",
                });
            }
            Ok(())
        },
        Expr::Substring { expr, start, length } => {
            validate_general_expr_capability(expr)?;
            if *start < 1 {
                return Err(ExpressionCapabilityRejection {
                    reason: "substring_start_must_be_positive",
                    expression_family: "substring",
                });
            }
            if let Some(length) = length
                && (*length < 0 || start.checked_sub(1).and_then(|start| start.checked_add(*length)).is_none())
            {
                return Err(ExpressionCapabilityRejection {
                    reason: "substring_length_must_be_non_negative_and_in_range",
                    expression_family: "substring",
                });
            }
            Ok(())
        },
        Expr::InList { expr, list, .. } => {
            validate_general_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            validate_general_expr_capability(left)?;
            validate_general_expr_capability(right)
        },
        Expr::Case { branches, else_expr } => {
            if branches.is_empty() {
                return Err(ExpressionCapabilityRejection {
                    reason: "case_requires_when_branch",
                    expression_family: "case",
                });
            }
            for branch in branches {
                validate_general_expr_capability(&branch.when)?;
                validate_general_expr_capability(&branch.then)?;
            }
            validate_general_expr_capability(else_expr)
        },
    }
}

pub(super) fn validate_join_residual_expr_capability(
    expr: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(column) if matches!(column.qualifier(), Some(ColumnQualifier::Input { index: 0 | 1 })) => Ok(()),
        Expr::Column(_) => Err(ExpressionCapabilityRejection {
            reason: "join_residual_column_must_be_input_qualified",
            expression_family: "column",
        }),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_join_residual_expr_capability(expr)
        },
        Expr::Cast { expr, .. } => validate_join_residual_expr_capability(expr),
        // TRY_CAST is rejected from join residuals: null-on-failure changes
        // the filter semantics versus strict CAST.
        Expr::TryCast { .. } => Err(ExpressionCapabilityRejection {
            reason: "join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
        Expr::Abs(_) | Expr::UnaryMath { .. } | Expr::Round { .. } => Err(ExpressionCapabilityRejection {
            reason: "join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
        Expr::Binary { left, op, right, .. } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_join_interval_binary_expr_capability(left, *op, right)
        },
        Expr::Binary { left, right, .. } => {
            validate_join_residual_expr_capability(left)?;
            validate_join_residual_expr_capability(right)
        },
        Expr::InList { expr, list, .. } => {
            validate_join_residual_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => Err(ExpressionCapabilityRejection {
            reason: "join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
        Expr::Like { .. }
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
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => Err(ExpressionCapabilityRejection {
            reason: "join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
    }
}

pub(super) fn validate_post_join_residual_expr_capability(
    expr: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(column) if matches!(column.qualifier(), Some(ColumnQualifier::Input { index: 0 | 1 })) => Ok(()),
        Expr::Column(_) => Err(ExpressionCapabilityRejection {
            reason: "join_residual_column_must_be_input_qualified",
            expression_family: "column",
        }),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_post_join_residual_expr_capability(expr)
        },
        Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Abs(expr)
        | Expr::UnaryMath { expr, .. } => validate_post_join_residual_expr_capability(expr),
        Expr::Extract { field, expr } => {
            validate_post_join_residual_expr_capability(expr)?;
            validate_post_join_extract_field_capability(*field)
        },
        Expr::Round { expr, .. } => validate_post_join_residual_expr_capability(expr),
        Expr::Concat(exprs) => {
            if exprs.is_empty() {
                return Err(ExpressionCapabilityRejection {
                    reason: "concat_requires_argument",
                    expression_family: "concat",
                });
            }
            for expr in exprs {
                validate_post_join_residual_expr_capability(expr)?;
            }
            Ok(())
        },
        Expr::Binary { left, op, right, .. } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_post_join_interval_binary_expr_capability(left, *op, right)
        },
        Expr::Binary { left, right, .. } => {
            validate_post_join_residual_expr_capability(left)?;
            validate_post_join_residual_expr_capability(right)
        },
        Expr::InList { expr, list, .. } => {
            validate_post_join_residual_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            validate_post_join_residual_expr_capability(left)?;
            validate_post_join_residual_expr_capability(right)
        },
        Expr::Case { branches, else_expr } => {
            if branches.is_empty() {
                return Err(ExpressionCapabilityRejection {
                    reason: "case_requires_when_branch",
                    expression_family: "case",
                });
            }
            for branch in branches {
                validate_post_join_residual_expr_capability(&branch.when)?;
                validate_post_join_residual_expr_capability(&branch.then)?;
            }
            validate_post_join_residual_expr_capability(else_expr)
        },
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::DateTrunc { .. } => Err(ExpressionCapabilityRejection {
            reason: "post_join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
    }
}

fn validate_extract_field_capability(field: DatePart) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if DatePart::EXTRACT_FIELDS.contains(&field) {
        Ok(())
    } else {
        Err(ExpressionCapabilityRejection { reason: "extract_field_not_supported", expression_family: "extract" })
    }
}

fn validate_post_join_extract_field_capability(
    field: DatePart,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if DatePart::POST_JOIN_EXTRACT_FIELDS.contains(&field) {
        Ok(())
    } else {
        Err(ExpressionCapabilityRejection {
            reason: "post_join_extract_field_not_supported",
            expression_family: "extract",
        })
    }
}

fn validate_date_trunc_field_capability(field: DatePart) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if DatePart::DATE_TRUNC_FIELDS.contains(&field) {
        Ok(())
    } else {
        Err(ExpressionCapabilityRejection { reason: "date_trunc_field_not_supported", expression_family: "date_trunc" })
    }
}

fn validate_in_list_capability(
    list: &[Expr],
    expression_family: &'static str,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if list.is_empty() {
        return Err(ExpressionCapabilityRejection { reason: "in_list_must_not_be_empty", expression_family });
    }
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(ExpressionCapabilityRejection { reason: "in_list_values_must_be_literals", expression_family });
        };
        validate_literal_capability(value, expression_family)?;
        if matches!(value, LiteralValue::Null(_)) {
            return Err(ExpressionCapabilityRejection {
                reason: "in_list_null_semantics_not_modeled",
                expression_family,
            });
        }
    }
    Ok(())
}

fn validate_literal_capability(
    value: &LiteralValue,
    expression_family: &'static str,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    if matches!(value, LiteralValue::IntervalMonthDayNano { .. }) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_literal_requires_datetime_interval_arithmetic",
            expression_family,
        });
    }
    Ok(())
}

fn validate_interval_binary_expr_capability(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_arithmetic_requires_exactly_one_interval_literal",
            expression_family: "binary",
        });
    }
    if left_interval && op != BinaryOp::Add {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_left_operand_requires_add",
            expression_family: "binary",
        });
    }
    if right_interval && !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_right_operand_requires_add_or_subtract",
            expression_family: "binary",
        });
    }
    validate_general_expr_capability(if left_interval { right } else { left })
}

fn validate_join_interval_binary_expr_capability(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_arithmetic_requires_exactly_one_interval_literal",
            expression_family: "binary",
        });
    }
    if left_interval && op != BinaryOp::Add {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_left_operand_requires_add",
            expression_family: "binary",
        });
    }
    if right_interval && !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_right_operand_requires_add_or_subtract",
            expression_family: "binary",
        });
    }
    let interval = if left_interval { left } else { right };
    validate_fixed_day_interval_literal(interval)?;
    validate_join_residual_expr_capability(if left_interval { right } else { left })
}

fn validate_fixed_day_interval_literal(expr: &Expr) -> std::result::Result<(), ExpressionCapabilityRejection> {
    let Expr::Literal(LiteralValue::IntervalMonthDayNano { months, nanoseconds, .. }) = expr else {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_arithmetic_requires_interval_literal",
            expression_family: "binary",
        });
    };
    const NANOS_PER_DAY: i64 = 86_400_000_000_000;
    if *months != 0 || *nanoseconds % NANOS_PER_DAY != 0 {
        return Err(ExpressionCapabilityRejection {
            reason: "join_residual_interval_requires_fixed_day_resolution",
            expression_family: "binary",
        });
    }
    Ok(())
}

fn validate_post_join_interval_binary_expr_capability(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> std::result::Result<(), ExpressionCapabilityRejection> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_arithmetic_requires_exactly_one_interval_literal",
            expression_family: "binary",
        });
    }
    if left_interval && op != BinaryOp::Add {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_left_operand_requires_add",
            expression_family: "binary",
        });
    }
    if right_interval && !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_right_operand_requires_add_or_subtract",
            expression_family: "binary",
        });
    }
    validate_post_join_residual_expr_capability(if left_interval { right } else { left })
}

fn expr_is_interval_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }))
}

pub(super) fn expr_family(expr: &Expr) -> &'static str {
    match expr {
        Expr::Column(_) => "column",
        Expr::Literal(_) => "literal",
        Expr::Not(_) => "not",
        Expr::IsNull(_) => "is_null",
        Expr::IsNotNull(_) => "is_not_null",
        Expr::Cast { .. } => "cast",
        Expr::TryCast { .. } => "try_cast",
        Expr::Binary { .. } => "binary",
        Expr::Like { .. } => "like",
        Expr::Substring { .. } => "substring",
        Expr::Lower(_) => "lower",
        Expr::Upper(_) => "upper",
        Expr::Trim { .. } => "trim",
        Expr::LTrim { .. } => "ltrim",
        Expr::RTrim { .. } => "rtrim",
        Expr::StringContains { .. } => "string_contains",
        Expr::StartsWith { .. } => "starts_with",
        Expr::EndsWith { .. } => "ends_with",
        Expr::StringLength { .. } => "string_length",
        Expr::RegexpReplace { .. } => "regexp_replace",
        Expr::Concat(_) => "concat",
        Expr::Abs(_) => "abs",
        Expr::UnaryMath { op: UnaryMathOp::Negate, .. } => "negate",
        Expr::UnaryMath { op: UnaryMathOp::Sqrt, .. } => "sqrt",
        Expr::UnaryMath { op: UnaryMathOp::Ln, .. } => "ln",
        Expr::UnaryMath { op: UnaryMathOp::Exp, .. } => "exp",
        Expr::Round { .. } => "round",
        Expr::InList { .. } => "in_list",
        Expr::IsDistinctFrom { .. } => "is_distinct_from",
        Expr::IsNotDistinctFrom { .. } => "is_not_distinct_from",
        Expr::Case { .. } => "case",
        Expr::Extract { .. } => "extract",
        Expr::DateTrunc { .. } => "date_trunc",
        Expr::Alias { .. } => "alias",
    }
}
