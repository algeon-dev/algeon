use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    physical_expr::PhysicalExpr,
    physical_plan::{ExecutionPlan, filter::FilterExec, projection::ProjectionExec},
};
use query_engine::{expr::Expr as NativeExpr, plan::QueryPlanBuilder as NativeQueryPlanBuilder};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{
        alias_if_needed, collect_required_field_indices, lower_physical_expr_with_bindings,
        lower_validated_physical_expr_with_bindings,
    },
    lower_native_input, projection_outputs,
};
use crate::{
    capability::CapabilityReason,
    native::{
        LoweringFailure,
        expression_contract::validate_physical_expr_contract,
        normalize::{NormalizedNativeFilter, NormalizedNativeProjection},
    },
};

pub(super) fn lower_filter_input(
    filter: &NormalizedNativeFilter<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let validated_predicate =
        validate_physical_expr_contract(filter.filter.predicate(), &filter.filter.input().schema())?;
    let selection = required_filter_output_selection(filter.filter, required_output_fields)?;
    let mut predicate_required = BTreeSet::new();
    collect_required_field_indices(
        filter.filter.predicate(),
        &filter.filter.input().schema(),
        &mut predicate_required,
    )?;
    for output_index in required_output_fields.predicate_indices(filter.filter.schema().fields().len())? {
        let input_index = if let Some(projection) = filter.filter.projection() {
            *projection.get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?
        } else {
            output_index
        };
        predicate_required.insert(input_index);
    }
    let mut child_required = predicate_required.clone();
    child_required.extend(selection.child_required_indices.iter().copied());
    let input = lower_native_input(
        &filter.input,
        builder,
        &RequiredFields::only_with_predicates(child_required, predicate_required),
    )?;
    let filtered_node = builder.add_filter(
        input.node,
        lower_validated_physical_expr_with_bindings(&validated_predicate, Some(&input.bindings))?,
    )?;
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
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let node = builder.add_projection(filtered_node, projection_outputs(expressions, &selection.schema)?)?;
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
    filter: &FilterExec,
    required_output_fields: &RequiredFields,
) -> Result<FilterOutputSelection, LoweringFailure> {
    let input_schema = filter.input().schema();
    let output_schema = filter.schema();
    let (schema, bindings) = executable_schema_and_bindings(&output_schema, required_output_fields)?;
    let mut child_required_indices = Vec::new();
    for binding in &bindings {
        let output_index = binding.frontend_index;
        if let Some(projection) = filter.projection() {
            let input_index = *projection.get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?;
            input_schema.fields().get(input_index).ok_or(CapabilityReason::UnsupportedExpression)?;
        } else {
            input_schema.fields().get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?;
        };
        let input_index = if let Some(projection) = filter.projection() {
            *projection.get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?
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
) -> Result<LoweredNativeInput, LoweringFailure> {
    let required_projection = required_projection_exprs(projection.projection, required_output_fields)?;
    let mut child_required = BTreeSet::new();
    let mut validated_projection = Vec::with_capacity(required_projection.len());
    for projection_expr in required_projection.iter().copied() {
        validated_projection
            .push(validate_physical_expr_contract(projection_expr, &projection.projection.input().schema())?);
        collect_required_field_indices(projection_expr, &projection.projection.input().schema(), &mut child_required)?;
    }
    let mut child_predicate_required = BTreeSet::new();
    for index in required_output_fields.predicate_indices(projection.projection.schema().fields().len())? {
        let projection_expr = projection.projection.expr().get(index).ok_or(CapabilityReason::UnsupportedExpression)?;
        collect_required_field_indices(
            &projection_expr.expr,
            &projection.projection.input().schema(),
            &mut child_predicate_required,
        )?;
    }
    let input = lower_native_input(
        &projection.input,
        builder,
        &RequiredFields::only_with_predicates(child_required, child_predicate_required),
    )?;
    let (schema, bindings) = executable_schema_and_bindings(&projection.projection.schema(), required_output_fields)?;
    let expressions = required_projection
        .iter()
        .copied()
        .zip(validated_projection.iter())
        .zip(bindings.iter())
        .map(|((projection_expr, validated), binding)| {
            debug_assert!(Arc::ptr_eq(projection_expr, validated.expr()));
            let expr = lower_validated_physical_expr_with_bindings(validated, Some(&input.bindings))?;
            Ok(alias_if_needed(expr, binding.native_name.as_str()))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) fn add_projection_node_with_input(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    projection: &ProjectionExec,
) -> Result<LoweredNativeInput, LoweringFailure> {
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
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?)?;
    super::advisory::attach(builder, node, &schema, projection)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

fn required_projection_exprs<'a>(
    projection: &'a ProjectionExec,
    required_output_fields: &RequiredFields,
) -> Result<Vec<&'a Arc<dyn PhysicalExpr>>, LoweringFailure> {
    let indices = required_output_fields.requested_indices(projection.expr().len())?;
    let exprs = indices
        .into_iter()
        .map(|index| {
            let projection_expr = projection.expr().get(index).ok_or(CapabilityReason::UnsupportedExpression)?;
            Ok(&projection_expr.expr)
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    Ok(exprs)
}
