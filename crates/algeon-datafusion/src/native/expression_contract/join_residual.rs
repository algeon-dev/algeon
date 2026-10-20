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
use query_engine::rapids_backend::rapids_backend;
use query_planning::{datatypes::is_supported_ast_cast, validation::is_integral_type};

use super::{
    ExpressionCapability, ExpressionRejection, ExpressionRejectionSubject, ValidatedPhysicalExpr,
    binary_operator_subject, contains_fallible_operation, fallible_operand, is_checked_temporal_arithmetic,
    is_decimal_division, physical_expr_data_type, validate_physical_expr_contract,
};

/// Proof that a join filter satisfies both the general physical-expression
/// contract and the native join residual contract.
///
/// The residual keeps the operators the cuDF join AST evaluates. Each other
/// subexpression that reads one input and cannot fail is a
/// [`JoinResidualMaterializationSite`]: lowering evaluates it on every row of
/// that input before the join and the residual reads the result as a column.
#[derive(Debug)]
pub(in crate::native) struct ValidatedJoinResidual<'a> {
    filter: &'a JoinFilter,
    expression: ValidatedPhysicalExpr,
    left_schema: SchemaRef,
    right_schema: SchemaRef,
    materializations: Vec<JoinResidualMaterializationSite>,
}

/// A residual subexpression that lowering evaluates on one join input before
/// the join. Sites are distinct `Arc`s in residual pre-order, and lowering
/// substitutes each by pointer identity.
#[derive(Debug)]
pub(in crate::native) struct JoinResidualMaterializationSite {
    pub(in crate::native) expr: Arc<dyn PhysicalExpr>,
    pub(in crate::native) side: JoinSide,
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

    pub(in crate::native) fn contains_decimal_division(&self) -> bool {
        contains_decimal_division(self.filter.expression(), self.filter.schema())
    }

    pub(in crate::native) fn materializations(&self) -> &[JoinResidualMaterializationSite] {
        &self.materializations
    }
}

pub(in crate::native) fn validate_join_residual<'a>(
    filter: &'a JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
) -> Result<ValidatedJoinResidual<'a>, ExpressionRejection> {
    let expression = validate_physical_expr_contract(filter.expression(), filter.schema())?;
    let residual = filter.expression();
    let mut shape = ResidualShape {
        filter,
        post_join: contains_decimal_division(residual, filter.schema()),
        materializations: Vec::new(),
    };
    shape.visit(residual)?;
    validate_join_residual_side_indices(residual, filter, left_schema, right_schema)
        .map_err(|rejection| rejection.at_expression(residual))?;

    let (reads_left, reads_right) =
        join_filter_expr_sides(residual, filter).map_err(|rejection| rejection.at_expression(residual))?;
    if !reads_left || !reads_right {
        return Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualSides)
            .at_expression(residual));
    }

    Ok(ValidatedJoinResidual {
        filter,
        expression,
        left_schema: Arc::clone(left_schema),
        right_schema: Arc::clone(right_schema),
        materializations: shape.materializations,
    })
}

fn contains_decimal_division(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    expr.downcast_ref::<BinaryExpr>().is_some_and(|binary| is_decimal_division(binary, schema))
        || expr.children().into_iter().any(|child| contains_decimal_division(child, schema))
}

/// Walk of a residual that admits its shape and records its materialization
/// sites.
struct ResidualShape<'f> {
    filter: &'f JoinFilter,
    /// Whether lowering runs the residual as the checked filter over
    /// key-matched rows, which it does for a decimal `/`. That filter
    /// evaluates a fallible operation only on matched pairs, as DataFusion
    /// does, and the operation stays in the residual.
    post_join: bool,
    materializations: Vec<JoinResidualMaterializationSite>,
}

impl ResidualShape<'_> {
    fn visit(&mut self, expr: &Arc<dyn PhysicalExpr>) -> Result<(), ExpressionRejection> {
        self.visit_shape(expr).map_err(|rejection| rejection.at_expression(expr))
    }

    fn visit_children(&mut self, expr: &Arc<dyn PhysicalExpr>) -> Result<(), ExpressionRejection> {
        expr.children().into_iter().try_for_each(|child| self.visit(child))
    }

    fn visit_shape(&mut self, expr: &Arc<dyn PhysicalExpr>) -> Result<(), ExpressionRejection> {
        if expr.downcast_ref::<Column>().is_some() || expr.downcast_ref::<Literal>().is_some() {
            return Ok(());
        }
        let schema = self.filter.schema();
        if is_inline_operator(expr, schema) {
            return self.visit_children(expr);
        }
        let Some(side) = join_filter_expr_single_side(expr, self.filter) else {
            return self.visit_cross_input(expr);
        };
        // A pre-join value is computed for every row of its input, including
        // rows that match no key and that DataFusion never filters. Only an
        // expression that cannot fail moves before the join.
        if !contains_fallible_operation(expr, schema) {
            if !self.materializations.iter().any(|site| Arc::ptr_eq(&site.expr, expr)) {
                self.materializations.push(JoinResidualMaterializationSite { expr: Arc::clone(expr), side });
            }
            return Ok(());
        }
        if self.post_join && !is_unchecked_ast_arithmetic(expr, schema) {
            return self.visit_children(expr);
        }
        Err(fallible_operation_rejection(expr, schema))
    }

    /// Admits an expression over both inputs that the AST does not evaluate.
    /// The planner evaluates an inner join residual with `CASE` or null-aware
    /// comparison as a filter over key-matched pairs.
    fn visit_cross_input(&mut self, expr: &Arc<dyn PhysicalExpr>) -> Result<(), ExpressionRejection> {
        let null_aware = expr
            .downcast_ref::<BinaryExpr>()
            .is_some_and(|binary| matches!(binary.op(), Operator::IsDistinctFrom | Operator::IsNotDistinctFrom));
        if null_aware || expr.downcast_ref::<CaseExpr>().is_some() {
            return self.visit_children(expr);
        }
        Err(cross_input_rejection(expr, self.filter.schema()))
    }
}

/// Whether the residual keeps `expr` as an operator over its operands: the
/// cuDF join AST evaluates it, or, for a decimal `/`, the checked post-join
/// filter does. The AST has no concatenation, null-aware comparison or bit
/// shift.
fn is_inline_operator(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return !matches!(
            binary.op(),
            Operator::StringConcat
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom
                | Operator::BitwiseShiftLeft
                | Operator::BitwiseShiftRight
        ) && !is_unchecked_ast_arithmetic(expr, schema);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return physical_expr_data_type(cast.expr(), schema)
            .is_ok_and(|source| is_supported_ast_cast(&source, cast.cast_type(), rapids_backend()));
    }
    expr.downcast_ref::<NotExpr>().is_some()
        || expr.downcast_ref::<IsNullExpr>().is_some()
        || expr.downcast_ref::<IsNotNullExpr>().is_some()
        || expr.downcast_ref::<InListExpr>().is_some()
}

/// Whether `expr` is arithmetic the cuDF AST would evaluate without the checks
/// DataFusion applies. The AST has no zero-divisor or overflow checks and no
/// verified floating-point remainder, and it wraps timestamp arithmetic that
/// Arrow checks. Decimal division uses the checked post-join filter.
fn is_unchecked_ast_arithmetic(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    match binary.op() {
        Operator::Modulo => true,
        Operator::Plus | Operator::Minus => is_checked_temporal_arithmetic(binary, schema),
        Operator::Divide => {
            physical_expr_data_type(binary.left(), schema).map_or(true, |data_type| is_integral_type(&data_type))
        },
        _ => false,
    }
}

/// The rejection for `expr`, an expression over both inputs that neither the
/// AST nor the post-join filter evaluates.
fn cross_input_rejection(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> ExpressionRejection {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape)
            .named_by_operator(binary_operator_subject(binary, schema));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return cast_rejection(cast, schema);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualScalarFunction)
            .with_subject(ExpressionRejectionSubject::ScalarFunction(function.name().to_string()));
    }
    ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape)
}

/// The rejection for `expr`, a single-input expression whose evaluation can
/// fail, naming its first fallible operation.
fn fallible_operation_rejection(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> ExpressionRejection {
    let operation = first_fallible_operation(expr, schema).unwrap_or(expr);
    let rejection = if let Some(binary) = operation.downcast_ref::<BinaryExpr>() {
        ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape)
            .named_by_operator(binary_operator_subject(binary, schema))
    } else if let Some(cast) = operation.downcast_ref::<CastExpr>() {
        cast_rejection(cast, schema)
    } else {
        ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape)
    };
    rejection.at_expression(operation)
}

fn first_fallible_operation<'e>(
    expr: &'e Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
) -> Option<&'e Arc<dyn PhysicalExpr>> {
    if fallible_operand(expr, schema).is_some() {
        return Some(expr);
    }
    expr.children().into_iter().find_map(|child| first_fallible_operation(child, schema))
}

/// A cast the residual cannot evaluate. A cast to an AST target type reports
/// its type pair; the AST has no other cast target.
fn cast_rejection(cast: &CastExpr, schema: &SchemaRef) -> ExpressionRejection {
    let rejection = if matches!(cast.cast_type(), DataType::Int64 | DataType::UInt64 | DataType::Float64) {
        ExpressionRejection::unsupported_type(ExpressionCapability::CastPair)
    } else {
        ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualCastTarget)
    };
    match physical_expr_data_type(cast.expr(), schema) {
        Ok(source) => {
            rejection.with_subject(ExpressionRejectionSubject::Cast { source, target: cast.cast_type().clone() })
        },
        Err(type_rejection) => type_rejection,
    }
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

/// Which join inputs `expr` reads, as `(left, right)`.
fn join_filter_expr_sides(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
) -> Result<(bool, bool), ExpressionRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return match filter.column_indices().get(column.index()).map(|column_index| column_index.side) {
            Some(JoinSide::Left) => Ok((true, false)),
            Some(JoinSide::Right) => Ok((false, true)),
            Some(JoinSide::None) | None => {
                Err(ExpressionRejection::unsupported_expression(ExpressionCapability::JoinResidualShape))
            },
        };
    }
    expr.children().into_iter().try_fold((false, false), |(left, right), child| {
        let (child_left, child_right) = join_filter_expr_sides(child, filter)?;
        Ok((left || child_left, right || child_right))
    })
}

/// The one input `expr` reads, or `None` when it reads both or neither.
fn join_filter_expr_single_side(expr: &Arc<dyn PhysicalExpr>, filter: &JoinFilter) -> Option<JoinSide> {
    match join_filter_expr_sides(expr, filter) {
        Ok((true, false)) => Some(JoinSide::Left),
        Ok((false, true)) => Some(JoinSide::Right),
        _ => None,
    }
}
