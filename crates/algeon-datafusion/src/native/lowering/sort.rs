use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::physical_expr::{LexOrdering, expressions::Column};
use query_engine::{
    expr::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec},
    plan::QueryPlanBuilder as NativeQueryPlanBuilder,
};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr_with_bindings},
    naming::unique_name_with_suffix,
    project_bindings_to_schema, projection_outputs,
};
use crate::{capability::CapabilityReason, native::LoweringFailure};

pub(super) fn lower_sort_ordering_with_input(
    builder: &mut NativeQueryPlanBuilder,
    ordering: &LexOrdering,
    sort_input_schema: &SchemaRef,
    sort_output_schema: &SchemaRef,
    input: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let sorted = lower_sort_ordering_unprojected_with_input(builder, ordering, sort_input_schema, input)?;
    project_sort_output(builder, sort_output_schema, sorted.input, required_output_fields)
}

/// A lowered sort before the projection back to the sort's output schema.
pub(super) struct SortedNativeInput {
    /// The sort node's output, bound to the sort's frontend fields. Its schema
    /// also carries the unbound columns computed for expression sort keys.
    pub(super) input: LoweredNativeInput,
    /// `key_columns[i]` is the native column the sort ordered by for
    /// `ordering[i]`.
    pub(super) key_columns: Vec<String>,
}

pub(super) fn lower_sort_ordering_unprojected_with_input(
    builder: &mut NativeQueryPlanBuilder,
    ordering: &LexOrdering,
    sort_input_schema: &SchemaRef,
    input: LoweredNativeInput,
) -> Result<SortedNativeInput, LoweringFailure> {
    let prepared = materialize_sort_expressions(builder, ordering, sort_input_schema, input)?;
    let node = builder.add_sort(
        prepared.input.node,
        lower_sort_spec_with_prepared_input(ordering, &prepared.input, &prepared.sort_columns)?,
    )?;
    Ok(SortedNativeInput {
        input: LoweredNativeInput::with_bindings(node, prepared.input.schema, prepared.original_bindings),
        key_columns: prepared.sort_columns,
    })
}

struct PreparedSortInput {
    input: LoweredNativeInput,
    original_bindings: Vec<FieldBinding>,
    sort_columns: Vec<String>,
}

fn materialize_sort_expressions(
    builder: &mut NativeQueryPlanBuilder,
    ordering: &LexOrdering,
    sort_input_schema: &SchemaRef,
    input: LoweredNativeInput,
) -> Result<PreparedSortInput, LoweringFailure> {
    let mut sort_columns = Vec::with_capacity(ordering.len());
    let mut temp_exprs = Vec::new();
    let mut used_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    for (index, sort_expr) in ordering.iter().enumerate() {
        if let Some(column) = sort_expr.expr.downcast_ref::<Column>() {
            sort_columns.push(input.binding_for_column(column)?.native_name.clone());
            continue;
        }
        let name = unique_sort_key_name(index, &used_names);
        used_names.insert(name.clone());
        sort_columns.push(name.clone());
        temp_exprs.push((name, sort_expr));
    }

    if temp_exprs.is_empty() {
        let original_bindings = input.bindings.clone();
        return Ok(PreparedSortInput { input, original_bindings, sort_columns });
    }

    let original_bindings = input.bindings.clone();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut expressions = input
        .schema
        .fields()
        .iter()
        .map(|field| alias_if_needed(NativeExpr::column(field.name().as_str()), field.name().as_str()))
        .collect::<Vec<_>>();
    for (name, sort_expr) in temp_exprs {
        let data_type = sort_expr
            .expr
            .data_type(sort_input_schema.as_ref())
            .map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        let nullable = sort_expr
            .expr
            .nullable(sort_input_schema.as_ref())
            .map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        fields.push(Field::new(name.as_str(), data_type, nullable));
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(&sort_expr.expr, sort_input_schema, Some(&input.bindings))?,
            name.as_str(),
        ));
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?)?;
    Ok(PreparedSortInput {
        input: LoweredNativeInput::with_bindings(node, schema, input.bindings),
        original_bindings,
        sort_columns,
    })
}

pub(super) fn project_sort_output(
    builder: &mut NativeQueryPlanBuilder,
    sort_output_schema: &SchemaRef,
    sorted: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let (schema, bindings) = executable_schema_and_bindings(sort_output_schema, required_output_fields)?;
    if bindings.len() == sorted.bindings.len()
        && bindings
            .iter()
            .zip(&sorted.bindings)
            .all(|(left, right)| left.frontend_index == right.frontend_index && left.native_name == right.native_name)
        && sorted.schema.fields().len() == sorted.bindings.len()
    {
        return Ok(sorted);
    }

    project_bindings_to_schema(builder, sorted, schema, bindings)
}

fn unique_sort_key_name(index: usize, used_names: &BTreeSet<String>) -> String {
    let base = format!("__algeon_native_sort_key_{index}");
    unique_name_with_suffix(base.as_str(), |candidate| !used_names.contains(candidate))
}

fn lower_sort_spec_with_prepared_input(
    ordering: &LexOrdering,
    input: &LoweredNativeInput,
    sort_columns: &[String],
) -> Result<NativeSortSpec, LoweringFailure> {
    if ordering.len() != sort_columns.len() {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    let expressions = ordering
        .iter()
        .zip(sort_columns.iter())
        .map(|(sort_expr, column)| {
            if input.schema.field_with_name(column.as_str()).is_err() {
                return Err(LoweringFailure::from(CapabilityReason::UnsupportedExpression));
            }
            Ok(NativeSortExpr::new(
                NativeExpr::column(column.as_str()),
                sort_expr.options.descending,
                sort_expr.options.nulls_first,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}
