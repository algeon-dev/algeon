use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, SchemaRef};
use datafusion::{
    common::JoinSide as DataFusionJoinSide,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
        },
    },
    physical_plan::joins::utils::JoinFilter,
};
use query_engine::{
    expr::{
        BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen, Expr as NativeExpr, LiteralValue as NativeLiteralValue,
    },
    validation::{is_string_type, substring_stop_position},
};

use super::{
    LoweredNativeInput,
    expr::{alias_if_needed, lower_binary_operator, lower_literal, physical_expr_field_from_validated},
    join::CanonicalJoin,
    naming::unique_name_with_suffix,
};
use crate::{
    capability::CapabilityReason,
    native::{
        LoweringRejection,
        expression_contract::{
            JoinResidualCastMode, ScalarFunctionKind, ValidatedJoinResidual, case_expr_data_type,
            declared_decimal_divide_output_type, join_residual_cast_mode, physical_expr_data_type,
        },
        literal::literal_i32,
    },
};

/// Residual predicate lowered for a native join, plus the input columns and
/// pre-join synthetic expressions needed to evaluate it.
pub(super) struct LoweredJoinResidual {
    pub(super) predicate: NativeExpr,
    pub(super) left_required: BTreeSet<usize>,
    pub(super) right_required: BTreeSet<usize>,
    pub(super) materializations: JoinResidualMaterializations,
}

/// Side-partitioned materializations. They are keyed by original DataFusion
/// side because projections are inserted before canonical join input ordering.
#[derive(Default)]
pub(super) struct JoinResidualMaterializations {
    left: Vec<JoinResidualMaterialization>,
    right: Vec<JoinResidualMaterialization>,
}

impl JoinResidualMaterializations {
    fn side_mut(
        &mut self,
        side: DataFusionJoinSide,
    ) -> Result<&mut Vec<JoinResidualMaterialization>, LoweringRejection> {
        match side {
            DataFusionJoinSide::Left => Ok(&mut self.left),
            DataFusionJoinSide::Right => Ok(&mut self.right),
            DataFusionJoinSide::None => Err(CapabilityReason::UnsupportedExpression.into()),
        }
    }

    pub(super) fn side(&self, side: DataFusionJoinSide) -> Result<&[JoinResidualMaterialization], LoweringRejection> {
        match side {
            DataFusionJoinSide::Left => Ok(&self.left),
            DataFusionJoinSide::Right => Ok(&self.right),
            DataFusionJoinSide::None => Err(CapabilityReason::UnsupportedExpression.into()),
        }
    }
}

/// Synthetic column appended to one join input so the residual predicate can
/// reference an expression that cannot be represented directly in the native
/// join predicate AST.
pub(super) struct JoinResidualMaterialization {
    pub(super) alias: String,
    pub(super) expr: NativeExpr,
    pub(super) field: Field,
}

#[derive(Debug, Clone, Copy)]
struct JoinFilterLoweringContext<'a> {
    contract: &'a ValidatedJoinResidual<'a>,
    filter: &'a JoinFilter,
    left_schema: &'a SchemaRef,
    right_schema: &'a SchemaRef,
    canonical: CanonicalJoin,
    bindings: Option<JoinResidualInputBindings<'a>>,
}

/// Executable join inputs available when residual expressions reference columns.
#[derive(Debug, Clone, Copy)]
pub(super) struct JoinResidualInputBindings<'a> {
    pub(super) left: &'a LoweredNativeInput,
    pub(super) right: &'a LoweredNativeInput,
}

pub(super) fn lower_join_filter(
    contract: &ValidatedJoinResidual<'_>,
    canonical: CanonicalJoin,
    bindings: Option<JoinResidualInputBindings<'_>>,
) -> Result<LoweredJoinResidual, LoweringRejection> {
    // Supported residual vocabulary is deliberately small. Expressions the
    // cuDF join AST cannot represent directly may be materialized on one input
    // side, but a residual predicate must still reference both sides.
    let filter = contract.filter();
    debug_assert!(Arc::ptr_eq(contract.expression().expr(), filter.expression()));
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let mut materializations = JoinResidualMaterializations::default();
    let context = JoinFilterLoweringContext {
        contract,
        filter,
        left_schema: contract.left_schema(),
        right_schema: contract.right_schema(),
        canonical,
        bindings,
    };
    let predicate = lower_join_filter_expr(
        filter.expression(),
        context,
        &mut left_required,
        &mut right_required,
        &mut materializations,
    )?;
    debug_assert!(
        !left_required.is_empty() && !right_required.is_empty(),
        "validated join residual must reference both inputs"
    );
    Ok(LoweredJoinResidual { predicate, left_required, right_required, materializations })
}

fn lower_join_filter_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, LoweringRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return lower_join_filter_column(column, context, left_required, right_required);
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(lower_literal(literal.value()).expect("validated join-residual literal must lower"));
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_expr(binary.left(), context, left_required, right_required, materializations)?;
        let right = lower_join_filter_expr(binary.right(), context, left_required, right_required, materializations)?;
        let op = lower_binary_operator(binary.op());
        if matches!(binary.op(), Operator::Divide)
            && let Some(output_type) = declared_decimal_divide_output_type(binary, context.filter.schema())
                .expect("validated join-residual decimal division must have a supported output type")
        {
            return Ok(NativeExpr::binary_with_output_type(left, op, right, output_type));
        }
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Ok(!lower_join_filter_expr(not.arg(), context, left_required, right_required, materializations)?);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_join_filter_expr(
            is_null.arg(),
            context,
            left_required,
            right_required,
            materializations,
        )?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_join_filter_expr(
            is_not_null.arg(),
            context,
            left_required,
            right_required,
            materializations,
        )?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        if matches!(
            join_residual_cast_mode(cast.cast_type()).expect("validated join-residual cast target"),
            JoinResidualCastMode::Materialize
        ) {
            // Some casts are valid native expressions but not accepted in the
            // join residual AST. Materialize those on one input side and let
            // the residual predicate reference the synthetic column.
            return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
        }
        return Ok(NativeExpr::cast(
            lower_join_filter_expr(cast.expr(), context, left_required, right_required, materializations)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        debug_assert!(!in_list.is_empty(), "empty IN list cannot carry a join-residual validation proof");
        let expr = lower_join_filter_expr(in_list.expr(), context, left_required, right_required, materializations)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let literal = item.downcast_ref::<Literal>().expect("validated join-residual IN entries are literals");
                debug_assert!(
                    !literal.value().is_null(),
                    "null IN entry cannot carry a join-residual validation proof"
                );
                lower_join_filter_expr(item, context, left_required, right_required, materializations)
            })
            .collect::<Result<Vec<_>, LoweringRejection>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(expr, list)
        } else {
            NativeExpr::in_list(expr, list)
        });
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if context.contract.single_side(expr).is_ok() {
            return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
        }
        return lower_join_filter_case_expr(case, context, left_required, right_required, materializations);
    }
    if expr.downcast_ref::<ScalarFunctionExpr>().is_some() {
        return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
    }
    unreachable!("validated join residual must have a lowering branch")
}

fn lower_join_filter_case_expr(
    case: &CaseExpr,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, LoweringRejection> {
    let base_expr = case
        .expr()
        .map(|expr| lower_join_filter_expr(expr, context, left_required, right_required, materializations))
        .transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(
                    base_expr.clone(),
                    NativeBinaryOp::Eq,
                    lower_join_filter_expr(when, context, left_required, right_required, materializations)?,
                ),
                None => lower_join_filter_expr(when, context, left_required, right_required, materializations)?,
            };
            Ok(NativeCaseWhen::new(
                when,
                lower_join_filter_expr(then, context, left_required, right_required, materializations)?,
            ))
        })
        .collect::<Result<Vec<_>, LoweringRejection>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_join_filter_expr(else_expr, context, left_required, right_required, materializations)?,
        None => {
            let data_type = case_expr_data_type(case, context.filter.schema())
                .expect("validated join-residual CASE must have an output type");
            NativeExpr::literal(NativeLiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn materialize_join_residual_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, LoweringRejection> {
    // Materializations are valid only when the whole expression belongs to one
    // DataFusion side. Cross-side expressions must remain in the residual AST.
    let side = context.contract.single_side(expr).expect("validated materialized residual must belong to one side");
    let (schema, required) = match side {
        DataFusionJoinSide::Left => (context.left_schema, left_required),
        DataFusionJoinSide::Right => (context.right_schema, right_required),
        DataFusionJoinSide::None => unreachable!("validated materialized residual must belong to left or right"),
    };
    let native_expr = lower_join_filter_side_expr(expr, context, schema, side, required)?;
    let side_materializations =
        materializations.side_mut(side).expect("validated materialization side must be left or right");
    let alias = unique_synthetic_join_residual_alias(side, side_materializations.len(), schema, side_materializations);
    let field =
        physical_expr_field_from_validated(&context.contract.expression().trusted_descendant(expr), alias.as_str())
            .expect("validated materialized residual must have a native field");
    side_materializations.push(JoinResidualMaterialization {
        alias: alias.clone(),
        expr: alias_if_needed(native_expr, alias.as_str()),
        field,
    });
    let native_input_index = context
        .canonical
        .native_input_index_for_datafusion_side(side)
        .expect("validated materialization side must map to a native input");
    Ok(NativeExpr::input_column(native_input_index, alias.as_str(), alias.as_str()))
}

fn lower_join_filter_side_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringRejection> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let column_index = context
            .filter
            .column_indices()
            .get(column.index())
            .expect("validated residual column must have a side mapping");
        if column_index.side != side {
            unreachable!("validated single-side residual cannot cross join inputs");
        }
        let field = side_schema
            .fields()
            .get(column_index.index)
            .expect("validated residual column must exist in its side schema");
        required.insert(column_index.index);
        return Ok(NativeExpr::column(side_native_name(context, side, column_index.index, field.name())?));
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(lower_literal(literal.value()).expect("validated materialized literal must lower"));
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_side_expr(binary.left(), context, side_schema, side, required)?;
        let right = lower_join_filter_side_expr(binary.right(), context, side_schema, side, required)?;
        let op = lower_binary_operator(binary.op());
        if matches!(binary.op(), Operator::Divide)
            && let Some(output_type) = declared_decimal_divide_output_type(binary, context.filter.schema())
                .expect("validated materialized decimal division must have a supported output type")
        {
            return Ok(NativeExpr::binary_with_output_type(left, op, right, output_type));
        }
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Ok(!lower_join_filter_side_expr(not.arg(), context, side_schema, side, required)?);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_join_filter_side_expr(
            is_null.arg(),
            context,
            side_schema,
            side,
            required,
        )?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_join_filter_side_expr(
            is_not_null.arg(),
            context,
            side_schema,
            side,
            required,
        )?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let source_type = physical_expr_data_type(cast.expr(), context.filter.schema())
            .expect("validated materialized cast source must have a type");
        if is_string_type(&source_type) && is_string_type(cast.cast_type()) {
            return lower_join_filter_side_expr(cast.expr(), context, side_schema, side, required);
        }
        return Ok(NativeExpr::cast(
            lower_join_filter_side_expr(cast.expr(), context, side_schema, side, required)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        debug_assert!(!in_list.is_empty(), "empty materialized IN list cannot carry a validation proof");
        let expr = lower_join_filter_side_expr(in_list.expr(), context, side_schema, side, required)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let literal = item.downcast_ref::<Literal>().expect("validated materialized IN entries are literals");
                debug_assert!(!literal.value().is_null(), "null materialized IN entry cannot carry a validation proof");
                lower_join_filter_side_expr(item, context, side_schema, side, required)
            })
            .collect::<Result<Vec<_>, LoweringRejection>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(expr, list)
        } else {
            NativeExpr::in_list(expr, list)
        });
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return lower_join_filter_side_case_expr(case, context, side_schema, side, required);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return lower_join_filter_side_scalar_function_expr(function, context, side_schema, side, required);
    }
    unreachable!("validated materialized residual must have a lowering branch")
}

fn lower_join_filter_side_case_expr(
    case: &CaseExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringRejection> {
    let base_expr =
        case.expr().map(|expr| lower_join_filter_side_expr(expr, context, side_schema, side, required)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(
                    base_expr.clone(),
                    NativeBinaryOp::Eq,
                    lower_join_filter_side_expr(when, context, side_schema, side, required)?,
                ),
                None => lower_join_filter_side_expr(when, context, side_schema, side, required)?,
            };
            Ok(NativeCaseWhen::new(when, lower_join_filter_side_expr(then, context, side_schema, side, required)?))
        })
        .collect::<Result<Vec<_>, LoweringRejection>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_join_filter_side_expr(else_expr, context, side_schema, side, required)?,
        None => {
            let data_type = case_expr_data_type(case, context.filter.schema())
                .expect("validated materialized CASE must have an output type");
            NativeExpr::literal(NativeLiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn lower_join_filter_side_scalar_function_expr(
    function: &ScalarFunctionExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringRejection> {
    match ScalarFunctionKind::from_name(function.name()).expect("validated join residual scalar function") {
        ScalarFunctionKind::Substring => {
            lower_join_filter_side_substring_function(function, context, side_schema, side, required)
        },
        ScalarFunctionKind::Lower => lower_join_filter_side_string_case_function(
            function,
            context,
            side_schema,
            side,
            required,
            NativeExpr::lower,
        ),
        ScalarFunctionKind::Upper => lower_join_filter_side_string_case_function(
            function,
            context,
            side_schema,
            side,
            required,
            NativeExpr::upper,
        ),
        _ => unreachable!("join residual proof excludes other scalar functions"),
    }
}

fn lower_join_filter_side_substring_function(
    function: &ScalarFunctionExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringRejection> {
    let args = function.args();
    debug_assert!((2..=3).contains(&args.len()), "invalid SUBSTRING arity cannot carry a residual proof");
    let start = literal_i32(&args[1]).expect("validated residual SUBSTRING start must be an i32 literal");
    debug_assert!(start >= 1, "invalid residual SUBSTRING start cannot carry a validation proof");
    let length = if args.len() == 3 {
        Some(literal_i32(&args[2]).expect("validated residual SUBSTRING length must be an i32 literal"))
    } else {
        None
    };
    debug_assert!(
        length.is_none_or(|length| length >= 0 && substring_stop_position(start, length).is_some()),
        "invalid residual SUBSTRING length cannot carry a validation proof"
    );
    Ok(NativeExpr::substring(
        lower_join_filter_side_expr(&args[0], context, side_schema, side, required)?,
        start,
        length,
    ))
}

fn lower_join_filter_side_string_case_function(
    function: &ScalarFunctionExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
    constructor: fn(NativeExpr) -> NativeExpr,
) -> Result<NativeExpr, LoweringRejection> {
    let [expr] = function.args() else { unreachable!("validated residual string case must have one argument") };
    Ok(constructor(lower_join_filter_side_expr(expr, context, side_schema, side, required)?))
}

fn unique_synthetic_join_residual_alias(
    side: DataFusionJoinSide,
    index: usize,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> String {
    // The `__algeon_native_join_residual_*` prefix is visible in native plans and tests.
    // Treat it as stable unless a migration updates diagnostics consumers.
    let side_name = match side {
        DataFusionJoinSide::Left => "left",
        DataFusionJoinSide::Right => "right",
        DataFusionJoinSide::None => "none",
    };
    let base = format!("__algeon_native_join_residual_{side_name}_{index}");
    unique_name_with_suffix(base.as_str(), |alias| {
        synthetic_join_residual_alias_is_available(alias, input_schema, materializations)
    })
}

fn synthetic_join_residual_alias_is_available(
    alias: &str,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> bool {
    input_schema.index_of(alias).is_err()
        && materializations.iter().all(|materialization| materialization.alias != alias)
}

fn lower_join_filter_column(
    column: &Column,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringRejection> {
    let column_index = context
        .filter
        .column_indices()
        .get(column.index())
        .expect("validated residual column must have a side mapping");
    let (datafusion_side, schema, required) = match column_index.side {
        DataFusionJoinSide::Left => (DataFusionJoinSide::Left, context.left_schema, left_required),
        DataFusionJoinSide::Right => (DataFusionJoinSide::Right, context.right_schema, right_required),
        DataFusionJoinSide::None => unreachable!("validated residual column must belong to a join input"),
    };
    let field =
        schema.fields().get(column_index.index).expect("validated residual column must exist in its side schema");
    required.insert(column_index.index);
    let native_input_index = context
        .canonical
        .native_input_index_for_datafusion_side(datafusion_side)
        .expect("validated residual side must map to a native input");
    let native_name = side_native_name(context, datafusion_side, column_index.index, field.name())?;
    Ok(NativeExpr::input_column(native_input_index, native_name, field.name()))
}

fn side_native_name<'a>(
    context: JoinFilterLoweringContext<'a>,
    side: DataFusionJoinSide,
    frontend_index: usize,
    fallback_name: &'a str,
) -> Result<&'a str, LoweringRejection> {
    let Some(bindings) = context.bindings else {
        return Ok(fallback_name);
    };
    let input = match side {
        DataFusionJoinSide::Left => bindings.left,
        DataFusionJoinSide::Right => bindings.right,
        DataFusionJoinSide::None => unreachable!("validated residual binding side must be left or right"),
    };
    Ok(input.binding_for_frontend_index(frontend_index)?.native_name.as_str())
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        common::{JoinSide, ScalarValue},
        config::ConfigOptions,
        functions::{string, unicode},
        logical_expr::{Operator, ScalarUDF},
        physical_expr::{
            PhysicalExpr, ScalarFunctionExpr,
            expressions::{CaseExpr, CastExpr, IsNotNullExpr, IsNullExpr, NotExpr, binary, col, in_list, lit},
        },
        physical_plan::joins::utils::{ColumnIndex, JoinFilter},
    };
    use query_engine::plan::JoinType as NativeJoinType;

    use super::*;
    use crate::native::expression_contract::validate_join_residual;

    fn scalar_function(
        name: &str,
        function: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        output_type: DataType,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(ScalarFunctionExpr::new(
            name,
            function,
            args,
            Arc::new(Field::new(name, output_type, true)),
            Arc::new(ConfigOptions::new()),
        ))
    }

    fn residual_schemas(data_type: DataType) -> (SchemaRef, SchemaRef, SchemaRef) {
        (
            Arc::new(Schema::new(vec![Field::new("left", data_type.clone(), true)])),
            Arc::new(Schema::new(vec![Field::new("right", data_type.clone(), true)])),
            Arc::new(Schema::new(vec![
                Field::new("left", data_type.clone(), true),
                Field::new("right", data_type, true),
            ])),
        )
    }

    fn assert_join_residual_parity(
        label: &str,
        expr: Arc<dyn PhysicalExpr>,
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
        filter_schema: &SchemaRef,
    ) {
        let filter = JoinFilter::new(
            expr,
            vec![ColumnIndex { index: 0, side: JoinSide::Left }, ColumnIndex { index: 0, side: JoinSide::Right }],
            Arc::clone(filter_schema),
        );
        let validated = validate_join_residual(&filter, left_schema, right_schema)
            .unwrap_or_else(|error| panic!("{label} must validate: {error:?}"));
        let lowered = lower_join_filter(&validated, CanonicalJoin::new(NativeJoinType::Inner, false), None)
            .unwrap_or_else(|error| panic!("{label} must lower: {error:?}"));
        assert!(!lowered.left_required.is_empty(), "{label} must retain a left input dependency");
        assert!(!lowered.right_required.is_empty(), "{label} must retain a right input dependency");
    }

    #[test]
    fn every_supported_join_residual_shape_validates_then_lowers() {
        let (left_schema, right_schema, schema) = residual_schemas(DataType::Int32);
        let left = || col("left", &schema).unwrap();
        let right = || col("right", &schema).unwrap();
        let cross_side = || binary(left(), Operator::Eq, right(), &schema).unwrap();

        assert_join_residual_parity("columns and binary", cross_side(), &left_schema, &right_schema, &schema);

        let not_and_null = binary(
            Arc::new(NotExpr::new(cross_side())),
            Operator::And,
            binary(Arc::new(IsNullExpr::new(left())), Operator::Or, Arc::new(IsNotNullExpr::new(right())), &schema)
                .unwrap(),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("not and null predicates", not_and_null, &left_schema, &right_schema, &schema);

        let ast_cast = binary(
            Arc::new(CastExpr::new(left(), DataType::Int64, None)),
            Operator::Eq,
            Arc::new(CastExpr::new(right(), DataType::Int64, None)),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("AST casts", ast_cast, &left_schema, &right_schema, &schema);

        let materialized_cast = binary(
            Arc::new(CastExpr::new(left(), DataType::Decimal128(10, 0), None)),
            Operator::Eq,
            Arc::new(CastExpr::new(right(), DataType::Decimal128(10, 0), None)),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("materialized casts", materialized_cast, &left_schema, &right_schema, &schema);

        let in_list_expr = binary(
            in_list(
                left(),
                vec![lit(ScalarValue::Int32(Some(1))), lit(ScalarValue::Int32(Some(2)))],
                &false,
                schema.as_ref(),
            )
            .unwrap(),
            Operator::And,
            cross_side(),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("IN list", in_list_expr, &left_schema, &right_schema, &schema);

        let single_side_case: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(binary(left(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &schema).unwrap(), left())],
                Some(lit(ScalarValue::Int32(Some(0)))),
            )
            .unwrap(),
        );
        let materialized_case = binary(single_side_case, Operator::Eq, right(), &schema).unwrap();
        assert_join_residual_parity(
            "single-side materialized CASE",
            materialized_case,
            &left_schema,
            &right_schema,
            &schema,
        );

        let cross_side_case: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(cross_side(), lit(ScalarValue::Boolean(Some(true))))],
                Some(lit(ScalarValue::Boolean(Some(false)))),
            )
            .unwrap(),
        );
        assert_join_residual_parity("cross-side CASE", cross_side_case, &left_schema, &right_schema, &schema);

        let (left_schema, right_schema, schema) = residual_schemas(DataType::Utf8);
        let left = || col("left", &schema).unwrap();
        let right = || col("right", &schema).unwrap();
        let supported_functions = [
            scalar_function(
                "substring",
                unicode::substr(),
                vec![left(), lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(Some(2)))],
                DataType::Utf8,
            ),
            scalar_function("lower", string::lower(), vec![left()], DataType::Utf8),
            scalar_function("upper", string::upper(), vec![left()], DataType::Utf8),
        ];
        for function in supported_functions {
            let name = function.downcast_ref::<ScalarFunctionExpr>().unwrap().name().to_string();
            let predicate = binary(function, Operator::Eq, right(), &schema).unwrap();
            assert_join_residual_parity(&name, predicate, &left_schema, &right_schema, &schema);
        }
    }
}
