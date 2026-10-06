use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    common::JoinSide,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
        },
    },
    physical_plan::joins::utils::JoinFilter,
};
use query_planning::{datatypes::cast_can_fail_at_runtime, validation::is_integral_type};

use super::{
    ExpressionCapability, ExpressionRejection, ExpressionRejectionSubject, ScalarFunctionKind, ValidatedPhysicalExpr,
    binary_operator_subject, is_decimal_division, physical_expr_data_type, validate_physical_expr_contract,
};

/// Representation selected by the join-residual-specific cast contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum JoinResidualCastMode {
    Ast,
    Materialize,
}

/// Proof that a join filter satisfies both the general physical-expression
/// contract and the stricter native join residual vocabulary.
#[derive(Debug)]
pub(in crate::native) struct ValidatedJoinResidual<'a> {
    filter: &'a JoinFilter,
    expression: ValidatedPhysicalExpr,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
}

impl<'a> ValidatedJoinResidual<'a> {
    pub(in crate::native) fn filter(&self) -> &'a JoinFilter {
        self.filter
    }

    pub(in crate::native) fn expression(&self) -> &ValidatedPhysicalExpr {
        &self.expression
    }

    pub(in crate::native) fn left_schema(&self) -> &SchemaRef {
        &self.left_schema
    }

    pub(in crate::native) fn right_schema(&self) -> &SchemaRef {
        &self.right_schema
    }

    pub(in crate::native) fn single_side(&self, expr: &Arc<dyn PhysicalExpr>) -> Result<JoinSide, ExpressionRejection> {
        join_filter_expr_single_side(expr, self.filter)
    }
}

pub(in crate::native) fn validate_join_residual<'a>(
    filter: &'a JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
) -> Result<ValidatedJoinResidual<'a>, ExpressionRejection> {
    let expression = validate_physical_expr_contract(filter.expression(), filter.schema())?;
    validate_join_residual_expr(filter.expression(), filter)?;
    validate_join_residual_side_indices(filter.expression(), filter, left_schema, right_schema)?;

    let mut side = None;
    let mut saw_left = false;
    let mut saw_right = false;
    collect_join_filter_expr_sides(filter.expression(), filter, &mut side, &mut saw_left, &mut saw_right, false)?;
    if !saw_left || !saw_right {
        return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSides));
    }

    Ok(ValidatedJoinResidual {
        filter,
        expression,
        left_schema: Arc::clone(left_schema),
        right_schema: Arc::clone(right_schema),
    })
}

fn validate_join_residual_side_indices(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
) -> Result<(), ExpressionRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let filter_field =
            filter.schema().fields().get(column.index()).ok_or_else(|| {
                ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSideSchema)
            })?;
        let column_index = filter
            .column_indices()
            .get(column.index())
            .ok_or_else(|| ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSideSchema))?;
        let side_schema = match column_index.side {
            JoinSide::Left => left_schema,
            JoinSide::Right => right_schema,
            JoinSide::None => {
                return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSideSchema));
            },
        };
        let Some(side_field) = side_schema.fields().get(column_index.index) else {
            return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSideSchema));
        };
        if side_field.data_type() != filter_field.data_type() {
            return Err(ExpressionRejection::unsupported_type(ExpressionCapability::JoinResidualSideSchema));
        }
        return Ok(());
    }
    for child in expr.children() {
        validate_join_residual_side_indices(child, filter, left_schema, right_schema)?;
    }
    Ok(())
}

pub(in crate::native) fn join_residual_cast_mode(
    target_type: &DataType,
) -> Result<JoinResidualCastMode, ExpressionRejection> {
    if matches!(target_type, DataType::Int64 | DataType::UInt64 | DataType::Float64) {
        return Ok(JoinResidualCastMode::Ast);
    }
    if matches!(target_type, DataType::Decimal128(_, _) | DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
        return Ok(JoinResidualCastMode::Materialize);
    }
    Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualCastTarget))
}

pub(in crate::native) fn validate_join_residual_scalar_function(
    kind: ScalarFunctionKind,
) -> Result<(), ExpressionRejection> {
    match kind {
        ScalarFunctionKind::Substring | ScalarFunctionKind::Lower | ScalarFunctionKind::Upper => Ok(()),
        _ => Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualScalarFunction)),
    }
}

fn validate_join_residual_expr(expr: &Arc<dyn PhysicalExpr>, filter: &JoinFilter) -> Result<(), ExpressionRejection> {
    if expr.downcast_ref::<Column>().is_some() || expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        // The cudf AST has no zero-divisor or overflow checks and no verified
        // floating-point remainder. A residual is also lowered without
        // restricting a CASE branch to the rows DataFusion evaluates, so a
        // division that native evaluation checks would fail on a guarded row.
        // Only floating-point division is admitted.
        let ast_arithmetic_is_unchecked = match binary.op() {
            Operator::Modulo => true,
            Operator::Divide => {
                is_decimal_division(binary, filter.schema())
                    || physical_expr_data_type(binary.left(), filter.schema())
                        .map_or(true, |data_type| is_integral_type(&data_type))
            },
            _ => false,
        };
        if ast_arithmetic_is_unchecked {
            return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape)
                .named_by_operator(binary_operator_subject(binary, filter.schema())));
        }
        validate_join_residual_expr(binary.left(), filter)?;
        return validate_join_residual_expr(binary.right(), filter);
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return validate_join_residual_expr(not.arg(), filter);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return validate_join_residual_expr(is_null.arg(), filter);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return validate_join_residual_expr(is_not_null.arg(), filter);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        if matches!(join_residual_cast_mode(cast.cast_type())?, JoinResidualCastMode::Materialize) {
            join_filter_expr_single_side(expr, filter)?;
        } else {
            // The AST cast wraps or saturates where DataFusion fails the query.
            let source_type = physical_expr_data_type(cast.expr(), filter.schema())?;
            if cast_can_fail_at_runtime(&source_type, cast.cast_type()) {
                return Err(ExpressionRejection::unsupported_type(ExpressionCapability::CastPair).with_subject(
                    ExpressionRejectionSubject::Cast { source: source_type, target: cast.cast_type().clone() },
                ));
            }
        }
        return validate_join_residual_expr(cast.expr(), filter);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        validate_join_residual_expr(in_list.expr(), filter)?;
        for item in in_list.list() {
            validate_join_residual_expr(item, filter)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(base) = case.expr() {
            validate_join_residual_expr(base, filter)?;
        }
        for (when, then) in case.when_then_expr() {
            validate_join_residual_expr(when, filter)?;
            validate_join_residual_expr(then, filter)?;
        }
        if let Some(else_expr) = case.else_expr() {
            validate_join_residual_expr(else_expr, filter)?;
        }
        return Ok(());
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        validate_join_residual_scalar_function(ScalarFunctionKind::from_scalar_function(function)?).map_err(
            |rejection| rejection.with_subject(ExpressionRejectionSubject::ScalarFunction(function.name().to_string())),
        )?;
        join_filter_expr_single_side(expr, filter)?;
        for arg in function.args() {
            validate_join_residual_expr(arg, filter)?;
        }
        return Ok(());
    }
    Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape))
}

fn join_filter_expr_single_side(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
) -> Result<JoinSide, ExpressionRejection> {
    let mut side = None;
    let mut saw_left = false;
    let mut saw_right = false;
    collect_join_filter_expr_sides(expr, filter, &mut side, &mut saw_left, &mut saw_right, true)?;
    side.ok_or_else(|| ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSides))
}

fn collect_join_filter_expr_sides(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
    side: &mut Option<JoinSide>,
    saw_left: &mut bool,
    saw_right: &mut bool,
    enforce_single_side: bool,
) -> Result<(), ExpressionRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let column_index = filter
            .column_indices()
            .get(column.index())
            .ok_or_else(|| ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape))?;
        match column_index.side {
            JoinSide::Left | JoinSide::Right => {
                *saw_left |= column_index.side == JoinSide::Left;
                *saw_right |= column_index.side == JoinSide::Right;
                if enforce_single_side && side.is_some_and(|existing| existing != column_index.side) {
                    return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSides));
                }
                *side = Some(column_index.side);
                return Ok(());
            },
            JoinSide::None => {
                return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape));
            },
        }
    }
    if expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        collect_join_filter_expr_sides(binary.left(), filter, side, saw_left, saw_right, enforce_single_side)?;
        return collect_join_filter_expr_sides(binary.right(), filter, side, saw_left, saw_right, enforce_single_side);
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return collect_join_filter_expr_sides(not.arg(), filter, side, saw_left, saw_right, enforce_single_side);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return collect_join_filter_expr_sides(is_null.arg(), filter, side, saw_left, saw_right, enforce_single_side);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return collect_join_filter_expr_sides(
            is_not_null.arg(),
            filter,
            side,
            saw_left,
            saw_right,
            enforce_single_side,
        );
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return collect_join_filter_expr_sides(cast.expr(), filter, side, saw_left, saw_right, enforce_single_side);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        collect_join_filter_expr_sides(in_list.expr(), filter, side, saw_left, saw_right, enforce_single_side)?;
        for item in in_list.list() {
            collect_join_filter_expr_sides(item, filter, side, saw_left, saw_right, enforce_single_side)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(base) = case.expr() {
            collect_join_filter_expr_sides(base, filter, side, saw_left, saw_right, enforce_single_side)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_join_filter_expr_sides(when, filter, side, saw_left, saw_right, enforce_single_side)?;
            collect_join_filter_expr_sides(then, filter, side, saw_left, saw_right, enforce_single_side)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_join_filter_expr_sides(else_expr, filter, side, saw_left, saw_right, enforce_single_side)?;
        }
        return Ok(());
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        for arg in function.args() {
            collect_join_filter_expr_sides(arg, filter, side, saw_left, saw_right, enforce_single_side)?;
        }
        return Ok(());
    }
    Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape))
}
