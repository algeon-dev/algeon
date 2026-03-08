use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::{
    common::{JoinSide as DataFusionJoinSide, ScalarValue},
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
        },
    },
    physical_plan::joins::utils::JoinFilter,
};
use nexus_query_engine::{CaseWhen as NativeCaseWhen, Expr as NativeExpr, LiteralValue as NativeLiteralValue};

use super::{
    LoweredNativeInput,
    expr::{alias_if_needed, lower_binary_operator, lower_literal, physical_expr_field},
    join::CanonicalJoin,
};
use crate::{
    native::validate::{is_supported_native_cast, is_supported_native_type, validate_physical_expr},
    policy::FallbackReason,
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
    fn side_mut(&mut self, side: DataFusionJoinSide) -> Result<&mut Vec<JoinResidualMaterialization>, FallbackReason> {
        match side {
            DataFusionJoinSide::Left => Ok(&mut self.left),
            DataFusionJoinSide::Right => Ok(&mut self.right),
            DataFusionJoinSide::None => Err(FallbackReason::UnsupportedExpression),
        }
    }

    pub(super) fn side(&self, side: DataFusionJoinSide) -> Result<&[JoinResidualMaterialization], FallbackReason> {
        match side {
            DataFusionJoinSide::Left => Ok(&self.left),
            DataFusionJoinSide::Right => Ok(&self.right),
            DataFusionJoinSide::None => Err(FallbackReason::UnsupportedExpression),
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
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
    bindings: Option<JoinResidualInputBindings<'_>>,
) -> Result<LoweredJoinResidual, FallbackReason> {
    // Supported residual vocabulary is deliberately small. Expressions the
    // cuDF join AST cannot represent directly may be materialized on one input
    // side, but a residual predicate must still reference both sides.
    validate_physical_expr(filter.expression(), filter.schema())?;
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let mut materializations = JoinResidualMaterializations::default();
    let context = JoinFilterLoweringContext { filter, left_schema, right_schema, canonical, bindings };
    let predicate = lower_join_filter_expr(
        filter.expression(),
        context,
        &mut left_required,
        &mut right_required,
        &mut materializations,
    )?;
    if left_required.is_empty() || right_required.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(LoweredJoinResidual { predicate, left_required, right_required, materializations })
}

fn lower_join_filter_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return lower_join_filter_column(column, context, left_required, right_required);
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_expr(binary.left(), context, left_required, right_required, materializations)?;
        let right = lower_join_filter_expr(binary.right(), context, left_required, right_required, materializations)?;
        let op = lower_binary_operator(binary.op())?;
        if matches!(binary.op(), datafusion::logical_expr::Operator::Divide)
            && let Some(output_type) = declared_decimal_divide_output_type(binary, context.filter.schema())?
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
        let source_type = cast
            .expr()
            .data_type(context.filter.schema().as_ref())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        if !is_supported_join_residual_ast_cast_target(cast.cast_type()) {
            // Some casts are valid native expressions but not accepted in the
            // join residual AST. Materialize those on one input side and let
            // the residual predicate reference the synthetic column.
            if !is_supported_join_residual_materialized_cast_target(cast.cast_type()) {
                return Err(FallbackReason::UnsupportedExpression);
            }
            return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
        }
        return Ok(NativeExpr::cast(
            lower_join_filter_expr(cast.expr(), context, left_required, right_required, materializations)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let expr = lower_join_filter_expr(in_list.expr(), context, left_required, right_required, materializations)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let Some(literal) = item.downcast_ref::<Literal>() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                if literal.value().is_null() {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                lower_join_filter_expr(item, context, left_required, right_required, materializations)
            })
            .collect::<Result<Vec<_>, FallbackReason>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(expr, list)
        } else {
            NativeExpr::in_list(expr, list)
        });
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if join_filter_expr_single_side(expr, context.filter).is_ok() {
            return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
        }
        return lower_join_filter_case_expr(case, context, left_required, right_required, materializations);
    }
    if expr.downcast_ref::<ScalarFunctionExpr>().is_some() {
        return materialize_join_residual_expr(expr, context, left_required, right_required, materializations);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_join_filter_case_expr(
    case: &CaseExpr,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, FallbackReason> {
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
                    nexus_query_engine::BinaryOp::Eq,
                    lower_join_filter_expr(when, context, left_required, right_required, materializations)?,
                ),
                None => lower_join_filter_expr(when, context, left_required, right_required, materializations)?,
            };
            Ok(NativeCaseWhen::new(
                when,
                lower_join_filter_expr(then, context, left_required, right_required, materializations)?,
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_join_filter_expr(else_expr, context, left_required, right_required, materializations)?,
        None => {
            let data_type = case
                .data_type(context.filter.schema().as_ref())
                .map_err(|_error| FallbackReason::UnsupportedExpression)?;
            NativeExpr::literal(NativeLiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn is_supported_join_residual_ast_cast_target(target_type: &DataType) -> bool {
    matches!(target_type, DataType::Int64 | DataType::UInt64 | DataType::Float64)
}

fn is_supported_join_residual_materialized_cast_target(target_type: &DataType) -> bool {
    matches!(target_type, DataType::Decimal128(_, _) | DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn materialize_join_residual_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, FallbackReason> {
    // Materializations are valid only when the whole expression belongs to one
    // DataFusion side. Cross-side expressions must remain in the residual AST.
    let side = join_filter_expr_single_side(expr, context.filter)?;
    let (schema, required) = match side {
        DataFusionJoinSide::Left => (context.left_schema, left_required),
        DataFusionJoinSide::Right => (context.right_schema, right_required),
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    let native_expr = lower_join_filter_side_expr(expr, context, schema, side, required)?;
    let side_materializations = materializations.side_mut(side)?;
    let alias = unique_synthetic_join_residual_alias(side, side_materializations.len(), schema, side_materializations);
    let field = physical_expr_field(expr, context.filter.schema(), alias.as_str())?;
    side_materializations.push(JoinResidualMaterialization {
        alias: alias.clone(),
        expr: alias_if_needed(native_expr, alias.as_str()),
        field,
    });
    let native_input_index = context.canonical.native_input_index_for_datafusion_side(side)?;
    Ok(NativeExpr::input_column(native_input_index, alias.as_str(), alias.as_str()))
}

fn join_filter_expr_single_side(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
) -> Result<DataFusionJoinSide, FallbackReason> {
    let mut side = None;
    collect_join_filter_expr_sides(expr, filter, &mut side)?;
    side.ok_or(FallbackReason::UnsupportedExpression)
}

fn collect_join_filter_expr_sides(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
    side: &mut Option<DataFusionJoinSide>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let column_index = filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
        match column_index.side {
            DataFusionJoinSide::Left | DataFusionJoinSide::Right => {
                if side.is_some_and(|existing| existing != column_index.side) {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                *side = Some(column_index.side);
                return Ok(());
            },
            DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
        }
    }
    if expr.downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        collect_join_filter_expr_sides(binary.left(), filter, side)?;
        return collect_join_filter_expr_sides(binary.right(), filter, side);
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return collect_join_filter_expr_sides(not.arg(), filter, side);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return collect_join_filter_expr_sides(is_null.arg(), filter, side);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return collect_join_filter_expr_sides(is_not_null.arg(), filter, side);
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return collect_join_filter_expr_sides(cast.expr(), filter, side);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        collect_join_filter_expr_sides(in_list.expr(), filter, side)?;
        for item in in_list.list() {
            collect_join_filter_expr_sides(item, filter, side)?;
        }
        return Ok(());
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr() {
            collect_join_filter_expr_sides(expr, filter, side)?;
        }
        for (when, then) in case.when_then_expr() {
            collect_join_filter_expr_sides(when, filter, side)?;
            collect_join_filter_expr_sides(then, filter, side)?;
        }
        if let Some(else_expr) = case.else_expr() {
            collect_join_filter_expr_sides(else_expr, filter, side)?;
        }
        return Ok(());
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        for arg in function.args() {
            collect_join_filter_expr_sides(arg, filter, side)?;
        }
        return Ok(());
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_join_filter_side_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let column_index =
            context.filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
        if column_index.side != side {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let field = side_schema.fields().get(column_index.index).ok_or(FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        required.insert(column_index.index);
        return Ok(NativeExpr::column(side_native_name(context, side, column_index.index, field.name())?));
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_side_expr(binary.left(), context, side_schema, side, required)?;
        let right = lower_join_filter_side_expr(binary.right(), context, side_schema, side, required)?;
        let op = lower_binary_operator(binary.op())?;
        if matches!(binary.op(), datafusion::logical_expr::Operator::Divide)
            && let Some(output_type) = declared_decimal_divide_output_type(binary, context.filter.schema())?
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
        let source_type = cast
            .expr()
            .data_type(context.filter.schema().as_ref())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        if is_string_type(&source_type) && is_string_type(cast.cast_type()) {
            return lower_join_filter_side_expr(cast.expr(), context, side_schema, side, required);
        }
        return Ok(NativeExpr::cast(
            lower_join_filter_side_expr(cast.expr(), context, side_schema, side, required)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let expr = lower_join_filter_side_expr(in_list.expr(), context, side_schema, side, required)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let Some(literal) = item.downcast_ref::<Literal>() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                if literal.value().is_null() {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                lower_join_filter_side_expr(item, context, side_schema, side, required)
            })
            .collect::<Result<Vec<_>, FallbackReason>>()?;
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
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_join_filter_side_case_expr(
    case: &CaseExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, FallbackReason> {
    let base_expr =
        case.expr().map(|expr| lower_join_filter_side_expr(expr, context, side_schema, side, required)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(
                    base_expr.clone(),
                    nexus_query_engine::BinaryOp::Eq,
                    lower_join_filter_side_expr(when, context, side_schema, side, required)?,
                ),
                None => lower_join_filter_side_expr(when, context, side_schema, side, required)?,
            };
            Ok(NativeCaseWhen::new(when, lower_join_filter_side_expr(then, context, side_schema, side, required)?))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_join_filter_side_expr(else_expr, context, side_schema, side, required)?,
        None => {
            let data_type = case
                .data_type(context.filter.schema().as_ref())
                .map_err(|_error| FallbackReason::UnsupportedExpression)?;
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
) -> Result<NativeExpr, FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => {
            lower_join_filter_side_substring_function(function, context, side_schema, side, required)
        },
        "lower" => lower_join_filter_side_string_case_function(
            function,
            context,
            side_schema,
            side,
            required,
            NativeExpr::lower,
        ),
        "upper" => lower_join_filter_side_string_case_function(
            function,
            context,
            side_schema,
            side,
            required,
            NativeExpr::upper,
        ),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn lower_join_filter_side_substring_function(
    function: &ScalarFunctionExpr,
    context: JoinFilterLoweringContext<'_>,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    validate_join_filter_string_expr(&args[0], context.filter.schema())?;
    let start = join_filter_literal_i32(&args[1])?;
    if start < 1 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let length = if args.len() == 3 { Some(join_filter_literal_i32(&args[2])?) } else { None };
    if let Some(length) = length
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(FallbackReason::UnsupportedExpression);
    }
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
) -> Result<NativeExpr, FallbackReason> {
    let [expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_join_filter_string_expr(expr, context.filter.schema())?;
    Ok(constructor(lower_join_filter_side_expr(expr, context, side_schema, side, required)?))
}

fn validate_join_filter_string_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn is_string_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn join_filter_literal_i32(expr: &Arc<dyn PhysicalExpr>) -> Result<i32, FallbackReason> {
    let Some(literal) = expr.downcast_ref::<Literal>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal.value() {
        ScalarValue::Int8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int32(Some(value)) => Ok(*value),
        ScalarValue::Int64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn declared_decimal_divide_output_type(
    binary: &BinaryExpr,
    input_schema: &SchemaRef,
) -> Result<Option<DataType>, FallbackReason> {
    let left_type =
        binary.left().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let right_type =
        binary.right().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(left_type, DataType::Decimal128(_, _)) && !matches!(right_type, DataType::Decimal128(_, _)) {
        return Ok(None);
    }
    let declared_type =
        binary.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !matches!(declared_type, DataType::Decimal128(_, _)) {
        return Ok(None);
    }
    if !is_supported_native_type(&declared_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    Ok(Some(declared_type))
}

fn unique_synthetic_join_residual_alias(
    side: DataFusionJoinSide,
    index: usize,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> String {
    // The `__cudf_join_residual_*` prefix is visible in native plans and tests.
    // Treat it as stable unless a migration updates diagnostics consumers.
    let side_name = match side {
        DataFusionJoinSide::Left => "left",
        DataFusionJoinSide::Right => "right",
        DataFusionJoinSide::None => "none",
    };
    let base = format!("__cudf_join_residual_{side_name}_{index}");
    if synthetic_join_residual_alias_is_available(base.as_str(), input_schema, materializations) {
        return base;
    }

    for suffix in 1usize.. {
        let alias = format!("{base}_{suffix}");
        if synthetic_join_residual_alias_is_available(alias.as_str(), input_schema, materializations) {
            return alias;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
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
) -> Result<NativeExpr, FallbackReason> {
    let column_index =
        context.filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
    let (datafusion_side, schema, required) = match column_index.side {
        DataFusionJoinSide::Left => (DataFusionJoinSide::Left, context.left_schema, left_required),
        DataFusionJoinSide::Right => (DataFusionJoinSide::Right, context.right_schema, right_required),
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    let field = schema.fields().get(column_index.index).ok_or(FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(field.data_type()) {
        return Err(FallbackReason::UnsupportedType);
    }
    required.insert(column_index.index);
    let native_input_index = context.canonical.native_input_index_for_datafusion_side(datafusion_side)?;
    let native_name = side_native_name(context, datafusion_side, column_index.index, field.name())?;
    Ok(NativeExpr::input_column(native_input_index, native_name, field.name()))
}

fn side_native_name<'a>(
    context: JoinFilterLoweringContext<'a>,
    side: DataFusionJoinSide,
    frontend_index: usize,
    fallback_name: &'a str,
) -> Result<&'a str, FallbackReason> {
    let Some(bindings) = context.bindings else {
        return Ok(fallback_name);
    };
    let input = match side {
        DataFusionJoinSide::Left => bindings.left,
        DataFusionJoinSide::Right => bindings.right,
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(input.binding_for_frontend_index(frontend_index)?.native_name.as_str())
}
