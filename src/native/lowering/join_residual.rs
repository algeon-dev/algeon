use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, Field, SchemaRef};
use cudf_query_engine::Expr as NativeExpr;
use datafusion::{
    common::JoinSide as DataFusionJoinSide,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr},
    },
    physical_plan::joins::utils::JoinFilter,
};

use super::{
    expr::{alias_if_needed, lower_binary_operator, lower_literal, physical_expr_field},
    join::CanonicalJoin,
};
use crate::{
    native::validate::{is_supported_native_cast, is_supported_native_type},
    policy::FallbackReason,
};

pub(super) struct LoweredJoinResidual {
    pub(super) predicate: NativeExpr,
    pub(super) left_required: BTreeSet<String>,
    pub(super) right_required: BTreeSet<String>,
    pub(super) materializations: JoinResidualMaterializations,
}

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
}

pub(super) fn lower_join_filter(
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
) -> Result<LoweredJoinResidual, FallbackReason> {
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let mut materializations = JoinResidualMaterializations::default();
    let context = JoinFilterLoweringContext { filter, left_schema, right_schema, canonical };
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
    left_required: &mut BTreeSet<String>,
    right_required: &mut BTreeSet<String>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return lower_join_filter_column(
            column,
            context.filter,
            context.left_schema,
            context.right_schema,
            context.canonical,
            left_required,
            right_required,
        );
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_expr(binary.left(), context, left_required, right_required, materializations)?;
        let right = lower_join_filter_expr(binary.right(), context, left_required, right_required, materializations)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(!lower_join_filter_expr(not.arg(), context, left_required, right_required, materializations)?);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_join_filter_expr(
            is_null.arg(),
            context,
            left_required,
            right_required,
            materializations,
        )?));
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_join_filter_expr(
            is_not_null.arg(),
            context,
            left_required,
            right_required,
            materializations,
        )?));
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type = cast
            .expr()
            .data_type(context.filter.schema().as_ref())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        if !is_supported_join_residual_ast_cast_target(cast.cast_type()) {
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
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let expr = lower_join_filter_expr(in_list.expr(), context, left_required, right_required, materializations)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let Some(literal) = item.as_any().downcast_ref::<Literal>() else {
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
    Err(FallbackReason::UnsupportedExpression)
}

fn is_supported_join_residual_ast_cast_target(target_type: &DataType) -> bool {
    matches!(target_type, DataType::Int64 | DataType::UInt64 | DataType::Float64)
}

fn is_supported_join_residual_materialized_cast_target(target_type: &DataType) -> bool {
    matches!(target_type, DataType::Decimal128(_, _))
}

fn materialize_join_residual_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<String>,
    right_required: &mut BTreeSet<String>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, FallbackReason> {
    let side = join_filter_expr_single_side(expr, context.filter)?;
    let (schema, required) = match side {
        DataFusionJoinSide::Left => (context.left_schema, left_required),
        DataFusionJoinSide::Right => (context.right_schema, right_required),
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    let native_expr = lower_join_filter_side_expr(expr, context.filter, schema, side, required)?;
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
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
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
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_join_filter_expr_sides(binary.left(), filter, side)?;
        return collect_join_filter_expr_sides(binary.right(), filter, side);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return collect_join_filter_expr_sides(not.arg(), filter, side);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return collect_join_filter_expr_sides(is_null.arg(), filter, side);
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return collect_join_filter_expr_sides(is_not_null.arg(), filter, side);
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        return collect_join_filter_expr_sides(cast.expr(), filter, side);
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        collect_join_filter_expr_sides(in_list.expr(), filter, side)?;
        for item in in_list.list() {
            collect_join_filter_expr_sides(item, filter, side)?;
        }
        return Ok(());
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_join_filter_side_expr(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
    side_schema: &SchemaRef,
    side: DataFusionJoinSide,
    required: &mut BTreeSet<String>,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        let column_index = filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
        if column_index.side != side {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let field = side_schema.fields().get(column_index.index).ok_or(FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        required.insert(field.name().clone());
        return Ok(NativeExpr::column(field.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_side_expr(binary.left(), filter, side_schema, side, required)?;
        let right = lower_join_filter_side_expr(binary.right(), filter, side_schema, side, required)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(!lower_join_filter_side_expr(not.arg(), filter, side_schema, side, required)?);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_join_filter_side_expr(
            is_null.arg(),
            filter,
            side_schema,
            side,
            required,
        )?));
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_join_filter_side_expr(
            is_not_null.arg(),
            filter,
            side_schema,
            side,
            required,
        )?));
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(filter.schema().as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return Ok(NativeExpr::cast(
            lower_join_filter_side_expr(cast.expr(), filter, side_schema, side, required)?,
            cast.cast_type().clone(),
        ));
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let expr = lower_join_filter_side_expr(in_list.expr(), filter, side_schema, side, required)?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let Some(literal) = item.as_any().downcast_ref::<Literal>() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                if literal.value().is_null() {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                lower_join_filter_side_expr(item, filter, side_schema, side, required)
            })
            .collect::<Result<Vec<_>, FallbackReason>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(expr, list)
        } else {
            NativeExpr::in_list(expr, list)
        });
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn unique_synthetic_join_residual_alias(
    side: DataFusionJoinSide,
    index: usize,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> String {
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
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
    left_required: &mut BTreeSet<String>,
    right_required: &mut BTreeSet<String>,
) -> Result<NativeExpr, FallbackReason> {
    let column_index = filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
    let (datafusion_side, schema, required) = match column_index.side {
        DataFusionJoinSide::Left => (DataFusionJoinSide::Left, left_schema, left_required),
        DataFusionJoinSide::Right => (DataFusionJoinSide::Right, right_schema, right_required),
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    let field = schema.fields().get(column_index.index).ok_or(FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(field.data_type()) {
        return Err(FallbackReason::UnsupportedType);
    }
    required.insert(field.name().clone());
    let native_input_index = canonical.native_input_index_for_datafusion_side(datafusion_side)?;
    Ok(NativeExpr::input_column(native_input_index, field.name(), field.name()))
}
