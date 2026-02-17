use super::{model::*, *};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExpressionCapabilityRejection {
    reason: &'static str,
    expression_family: &'static str,
}

pub(super) fn expression_rejection_decision(
    node_id: NodeId,
    reason: &'static str,
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

pub(super) fn validation_result_decision(
    result: crate::Result<()>,
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityDecision {
    match result {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(error) => validation_error_decision(reason, error, facts),
    }
}

pub(super) fn validation_error_decision(
    reason: &'static str,
    error: QueryError,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityDecision {
    let mut facts = facts.into_iter().collect::<BTreeMap<_, _>>();
    facts.insert("error_code", error.code().as_str().to_string());
    facts.insert("error_operation", error.operation().to_string());
    facts.insert("error_message", error.message().to_string());
    if let Some(error_reason) = error.fact_value("reason") {
        facts.insert("error_reason", error_reason.to_string());
    }
    CapabilityDecision::Rejected { reason, facts }
}

pub(super) fn validate_aggregate_expr_capability(
    aggregate_expr: &AggregateExpr,
    is_scalar_aggregate: bool,
) -> Result<(), ExpressionCapabilityRejection> {
    if is_scalar_aggregate
        && aggregate_expr.function == AggregateFunction::Count
        && is_count_star_expr(&aggregate_expr.expr)
    {
        return Ok(());
    }
    validate_column_expr_capability(&aggregate_expr.expr)
}

fn is_count_star_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))
        || matches!(expr, Expr::Alias { expr, .. } if is_count_star_expr(expr))
}

pub(super) fn validate_named_column_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_named_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

pub(super) fn validate_column_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

pub(super) fn validate_general_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_general_expr_capability(expr)
        },
        Expr::Cast { expr, .. } | Expr::Extract { expr, .. } | Expr::Lower(expr) | Expr::Upper(expr) => {
            validate_general_expr_capability(expr)
        },
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

pub(super) fn validate_join_residual_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(column)
            if matches!(column.qualifier(), Some(crate::expr::ColumnQualifier::Input { index: 0 | 1 })) =>
        {
            Ok(())
        },
        Expr::Column(_) => Err(ExpressionCapabilityRejection {
            reason: "join_residual_column_must_be_input_qualified",
            expression_family: "column",
        }),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_join_residual_expr_capability(expr)
        },
        Expr::Cast { expr, .. } => validate_join_residual_expr_capability(expr),
        Expr::Binary { left, right, .. } => {
            validate_join_residual_expr_capability(left)?;
            validate_join_residual_expr_capability(right)
        },
        Expr::InList { expr, list, .. } => {
            validate_join_residual_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Case { .. }
        | Expr::Extract { .. } => Err(ExpressionCapabilityRejection {
            reason: "join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
    }
}

pub(super) fn validate_post_join_residual_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(column)
            if matches!(column.qualifier(), Some(crate::expr::ColumnQualifier::Input { index: 0 | 1 })) =>
        {
            Ok(())
        },
        Expr::Column(_) => Err(ExpressionCapabilityRejection {
            reason: "join_residual_column_must_be_input_qualified",
            expression_family: "column",
        }),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_post_join_residual_expr_capability(expr)
        },
        Expr::Cast { expr, .. } | Expr::Lower(expr) | Expr::Upper(expr) | Expr::Extract { expr, .. } => {
            validate_post_join_residual_expr_capability(expr)
        },
        Expr::Binary { left, right, .. } => {
            validate_post_join_residual_expr_capability(left)?;
            validate_post_join_residual_expr_capability(right)
        },
        Expr::InList { expr, list, .. } => {
            validate_post_join_residual_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
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
        Expr::Like { .. } | Expr::Substring { .. } => Err(ExpressionCapabilityRejection {
            reason: "post_join_residual_expression_family_not_supported",
            expression_family: expr_family(expr),
        }),
    }
}

fn validate_in_list_capability(
    list: &[Expr],
    expression_family: &'static str,
) -> Result<(), ExpressionCapabilityRejection> {
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
) -> Result<(), ExpressionCapabilityRejection> {
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
) -> Result<(), ExpressionCapabilityRejection> {
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
        Expr::Binary { .. } => "binary",
        Expr::Like { .. } => "like",
        Expr::Substring { .. } => "substring",
        Expr::Lower(_) => "lower",
        Expr::Upper(_) => "upper",
        Expr::InList { .. } => "in_list",
        Expr::Case { .. } => "case",
        Expr::Extract { .. } => "extract",
        Expr::Alias { .. } => "alias",
    }
}
