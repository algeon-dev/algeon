use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    physical_expr::PhysicalExpr,
    physical_plan::{ExecutionPlan, projection::ProjectionExec},
};
use nexus_query_engine::{Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input, projection_outputs,
};
use crate::{
    native::{
        normalize::{NormalizedNativeFilter, NormalizedNativeProjection},
        validate::validate_physical_expr,
    },
    policy::FallbackReason,
};

pub(super) fn lower_filter_input(
    filter: &NormalizedNativeFilter<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    validate_physical_expr(filter.filter.predicate(), &filter.filter.input().schema())?;
    let selection = required_filter_output_selection(filter.filter, required_output_fields)?;
    let mut child_required = BTreeSet::new();
    collect_required_field_indices(filter.filter.predicate(), &filter.filter.input().schema(), &mut child_required)?;
    child_required.extend(selection.child_required_indices.iter().copied());
    let input = lower_native_input(&filter.input, builder, &RequiredFields::only(child_required))?;
    let filtered_node = builder.add_filter(
        input.node,
        lower_physical_expr_with_bindings(
            filter.filter.predicate(),
            &filter.filter.input().schema(),
            Some(&input.bindings),
        )?,
        Arc::clone(&input.schema),
    );
    if !selection.requires_projection_from(&input) {
        return Ok(LoweredNativeInput { node: filtered_node, schema: input.schema, bindings: input.bindings });
    }
    let expressions = selection
        .child_required_indices
        .iter()
        .zip(selection.bindings.iter())
        .map(|(input_index, output_binding)| {
            let input_binding = input.binding_for_frontend_index(*input_index)?;
            Ok(alias_if_needed(
                NativeExpr::column(input_binding.native_name.as_str()),
                output_binding.native_name.as_str(),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(filtered_node, projection_outputs(expressions, &selection.schema)?);
    Ok(LoweredNativeInput::with_bindings(node, selection.schema, selection.bindings))
}

struct FilterOutputSelection {
    schema: SchemaRef,
    bindings: Vec<super::FieldBinding>,
    child_required_indices: Vec<usize>,
}

impl FilterOutputSelection {
    fn requires_projection_from(&self, input: &LoweredNativeInput) -> bool {
        self.bindings.len() != input.bindings.len()
            || self.bindings.iter().zip(&input.bindings).any(|(selected, input)| {
                selected.frontend_index != input.frontend_index || selected.native_name != input.native_name
            })
    }
}

fn required_filter_output_selection(
    filter: &datafusion::physical_plan::filter::FilterExec,
    required_output_fields: &RequiredFields,
) -> Result<FilterOutputSelection, FallbackReason> {
    let input_schema = filter.input().schema();
    let output_schema = filter.schema();
    let (schema, bindings) = executable_schema_and_bindings(&output_schema, required_output_fields)?;
    let mut child_required_indices = Vec::new();
    for binding in &bindings {
        let output_index = binding.frontend_index;
        if let Some(projection) = filter.projection() {
            let input_index = *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?;
            input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?;
        } else {
            input_schema.fields().get(output_index).ok_or(FallbackReason::UnsupportedExpression)?;
        };
        let input_index = if let Some(projection) = filter.projection() {
            *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?
        } else {
            output_index
        };
        child_required_indices.push(input_index);
    }
    Ok(FilterOutputSelection { schema, bindings, child_required_indices })
}

pub(super) fn lower_projection_input(
    projection: &NormalizedNativeProjection<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let required_projection = required_projection_exprs(projection.projection, required_output_fields)?;
    let mut child_required = BTreeSet::new();
    for projection_expr in &required_projection {
        validate_physical_expr(projection_expr.expr, &projection.projection.input().schema())?;
        collect_required_field_indices(
            projection_expr.expr,
            &projection.projection.input().schema(),
            &mut child_required,
        )?;
    }
    let input = lower_native_input(&projection.input, builder, &RequiredFields::only(child_required))?;
    let (schema, bindings) = executable_schema_and_bindings(&projection.projection.schema(), required_output_fields)?;
    let expressions = required_projection
        .iter()
        .zip(bindings.iter())
        .map(|projection_expr| {
            let (projection_expr, binding) = projection_expr;
            let expr = lower_physical_expr_with_bindings(
                projection_expr.expr,
                &projection.projection.input().schema(),
                Some(&input.bindings),
            )?;
            Ok(alias_if_needed(expr, binding.native_name.as_str()))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) fn add_projection_node_with_input(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    projection: &ProjectionExec,
) -> Result<LoweredNativeInput, FallbackReason> {
    let required = RequiredFields::all();
    let (schema, bindings) = executable_schema_and_bindings(&projection.schema(), &required)?;
    let expressions = projection
        .expr()
        .iter()
        .zip(bindings.iter())
        .map(|(projection_expr, binding)| {
            let expr = lower_physical_expr_with_bindings(
                &projection_expr.expr,
                &projection.input().schema(),
                Some(&input.bindings),
            )?;
            Ok(alias_if_needed(expr, binding.native_name.as_str()))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

struct RequiredProjectionExpr<'a> {
    expr: &'a Arc<dyn PhysicalExpr>,
}

fn required_projection_exprs<'a>(
    projection: &'a ProjectionExec,
    required_output_fields: &RequiredFields,
) -> Result<Vec<RequiredProjectionExpr<'a>>, FallbackReason> {
    let indices = required_output_fields.requested_indices(projection.expr().len())?;
    let exprs = indices
        .into_iter()
        .map(|index| {
            let projection_expr = projection.expr().get(index).ok_or(FallbackReason::UnsupportedExpression)?;
            Ok(RequiredProjectionExpr { expr: &projection_expr.expr })
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    Ok(exprs)
}
