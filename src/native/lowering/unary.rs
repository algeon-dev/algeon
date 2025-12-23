use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use cudf_query_engine::{Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder};
use datafusion::{
    physical_expr::PhysicalExpr,
    physical_plan::{
        ExecutionPlan,
        projection::{ProjectionExec, ProjectionExpr},
    },
};

use super::{
    LoweredNativeInput,
    expr::{alias_if_needed, lower_physical_expr},
    lower_native_input,
};
use crate::{
    native::{
        normalize::{NormalizedNativeFilter, NormalizedNativeProjection},
        validate::{collect_required_columns, validate_physical_expr},
    },
    policy::FallbackReason,
};

pub(super) fn lower_filter_input(
    filter: &NormalizedNativeFilter<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    validate_physical_expr(filter.filter.predicate(), &filter.filter.input().schema())?;
    let selection = required_filter_output_selection(filter.filter, required_output_columns)?;
    let mut child_required = BTreeSet::new();
    collect_required_columns(filter.filter.predicate(), &filter.filter.input().schema(), &mut child_required)?;
    child_required.extend(selection.child_required_columns.iter().cloned());
    let child_required = child_required.into_iter().collect::<Vec<_>>();
    let input = lower_native_input(&filter.input, builder, &child_required)?;
    let filtered_node = builder.add_filter(
        input.node,
        lower_physical_expr(filter.filter.predicate(), &input.schema)?,
        Arc::clone(&input.schema),
    );
    if !selection.requires_projection_from(&input.schema) {
        return Ok(LoweredNativeInput { node: filtered_node, schema: input.schema });
    }
    let node = builder.add_projection(filtered_node, selection.expressions, Arc::clone(&selection.schema));
    Ok(LoweredNativeInput { node, schema: selection.schema })
}

struct FilterOutputSelection {
    expressions: Vec<NativeExpr>,
    schema: SchemaRef,
    child_required_columns: Vec<String>,
}

impl FilterOutputSelection {
    fn requires_projection_from(&self, input_schema: &SchemaRef) -> bool {
        self.schema.fields().len() != input_schema.fields().len()
            || self
                .schema
                .fields()
                .iter()
                .zip(input_schema.fields())
                .any(|(selected, input)| selected.name() != input.name())
    }
}

fn required_filter_output_selection(
    filter: &datafusion::physical_plan::filter::FilterExec,
    required_output_columns: &[String],
) -> Result<FilterOutputSelection, FallbackReason> {
    let required = required_output_columns.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let input_schema = filter.input().schema();
    let output_schema = filter.schema();
    let mut expressions = Vec::new();
    let mut fields = Vec::new();
    let mut child_required_columns = Vec::new();
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if !required.is_empty() && !required.contains(field.name().as_str()) {
            continue;
        }
        let input_field = if let Some(projection) = filter.projection() {
            let input_index = *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?;
            input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?
        } else {
            input_schema.field_with_name(field.name()).map_err(|_error| FallbackReason::UnsupportedExpression)?
        };
        expressions.push(NativeExpr::column(input_field.name()));
        fields.push(field.as_ref().clone());
        child_required_columns.push(input_field.name().clone());
    }
    if !required.is_empty() && fields.len() != required.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(FilterOutputSelection { expressions, schema: Arc::new(Schema::new(fields)), child_required_columns })
}

pub(super) fn lower_projection_input(
    projection: &NormalizedNativeProjection<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let required_projection = required_projection_exprs(projection.projection, required_output_columns)?;
    let mut child_required = BTreeSet::new();
    for projection_expr in &required_projection {
        validate_physical_expr(projection_expr.expr, &projection.projection.input().schema())?;
        collect_required_columns(projection_expr.expr, &projection.projection.input().schema(), &mut child_required)?;
    }
    let child_required = child_required.into_iter().collect::<Vec<_>>();
    let input = lower_native_input(&projection.input, builder, &child_required)?;
    let expressions = required_projection
        .iter()
        .map(|projection_expr| {
            let expr = lower_physical_expr(projection_expr.expr, &input.schema)?;
            Ok(alias_if_needed(expr, projection_expr.alias))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let schema = required_projection_schema(&required_projection, projection.projection.schema())?;
    let node = builder.add_projection(input.node, expressions, Arc::clone(&schema));
    Ok(LoweredNativeInput { node, schema })
}

pub(super) fn lower_projection_exprs(projection: &ProjectionExec) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .expr()
        .iter()
        .map(|projection_expr| lower_projection_expr(projection_expr, &projection.input().schema()))
        .collect()
}

struct RequiredProjectionExpr<'a> {
    expr: &'a Arc<dyn PhysicalExpr>,
    alias: &'a str,
}

fn required_projection_exprs<'a>(
    projection: &'a ProjectionExec,
    required_output_columns: &[String],
) -> Result<Vec<RequiredProjectionExpr<'a>>, FallbackReason> {
    let required = required_output_columns.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let exprs = projection
        .expr()
        .iter()
        .filter(|projection_expr| required.is_empty() || required.contains(projection_expr.alias.as_str()))
        .map(|projection_expr| RequiredProjectionExpr {
            expr: &projection_expr.expr,
            alias: projection_expr.alias.as_str(),
        })
        .collect::<Vec<_>>();
    if !required.is_empty() && exprs.len() != required.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(exprs)
}

fn required_projection_schema(
    projection: &[RequiredProjectionExpr<'_>],
    output_schema: SchemaRef,
) -> Result<SchemaRef, FallbackReason> {
    let fields = projection
        .iter()
        .map(|projection_expr| {
            output_schema
                .field_with_name(projection_expr.alias)
                .cloned()
                .map_err(|_error| FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn lower_filter_projection_exprs(
    projection: &[usize],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .iter()
        .map(|index| {
            let field = input_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
            Ok(NativeExpr::column(field.name()))
        })
        .collect()
}

fn lower_projection_expr(
    projection_expr: &ProjectionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let expr = lower_physical_expr(&projection_expr.expr, input_schema)?;
    Ok(alias_if_needed(expr, projection_expr.alias.as_str()))
}
