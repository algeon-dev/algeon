use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema};
use datafusion::{
    physical_expr::expressions::Column,
    physical_plan::{ExecutionPlan, sorts::sort::SortExec},
};
use nexus_query_engine::{
    Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr,
    SortSpec as NativeSortSpec,
};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr_with_bindings},
    projection_outputs,
};
use crate::policy::FallbackReason;

pub(super) fn lower_sort_with_input(
    builder: &mut NativeQueryPlanBuilder,
    sort: &SortExec,
    input: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let prepared = materialize_sort_expressions(builder, sort, input)?;
    let node = builder.add_sort(
        prepared.input.node,
        lower_sort_spec_with_prepared_input(sort, &prepared.input, &prepared.sort_columns)?,
        Arc::clone(&prepared.input.schema),
    );
    let sorted = LoweredNativeInput::with_bindings(node, prepared.input.schema, prepared.original_bindings);
    project_sort_output(builder, sort, sorted, required_output_fields)
}

struct PreparedSortInput {
    input: LoweredNativeInput,
    original_bindings: Vec<FieldBinding>,
    sort_columns: Vec<String>,
}

fn materialize_sort_expressions(
    builder: &mut NativeQueryPlanBuilder,
    sort: &SortExec,
    input: LoweredNativeInput,
) -> Result<PreparedSortInput, FallbackReason> {
    let mut sort_columns = Vec::with_capacity(sort.expr().len());
    let mut temp_exprs = Vec::new();
    let mut used_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    for (index, sort_expr) in sort.expr().iter().enumerate() {
        if let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() {
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
            .data_type(sort.input().schema().as_ref())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        let nullable = sort_expr
            .expr
            .nullable(sort.input().schema().as_ref())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        fields.push(Field::new(name.as_str(), data_type, nullable));
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(&sort_expr.expr, &sort.input().schema(), Some(&input.bindings))?,
            name.as_str(),
        ));
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?);
    Ok(PreparedSortInput {
        input: LoweredNativeInput::with_bindings(node, schema, input.bindings),
        original_bindings,
        sort_columns,
    })
}

fn project_sort_output(
    builder: &mut NativeQueryPlanBuilder,
    sort: &SortExec,
    sorted: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let (schema, bindings) = executable_schema_and_bindings(&sort.schema(), required_output_fields)?;
    if bindings.len() == sorted.bindings.len()
        && bindings
            .iter()
            .zip(&sorted.bindings)
            .all(|(left, right)| left.frontend_index == right.frontend_index && left.native_name == right.native_name)
        && sorted.schema.fields().len() == sorted.bindings.len()
    {
        return Ok(sorted);
    }

    let expressions = bindings
        .iter()
        .map(|binding| {
            let input_binding = sorted.binding_for_frontend_index(binding.frontend_index)?;
            Ok(alias_if_needed(NativeExpr::column(input_binding.native_name.as_str()), binding.native_name.as_str()))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(sorted.node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

fn unique_sort_key_name(index: usize, used_names: &BTreeSet<String>) -> String {
    let base = format!("__cudf_sort_key_{index}");
    if !used_names.contains(base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !used_names.contains(candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn lower_sort_spec_with_prepared_input(
    sort: &SortExec,
    input: &LoweredNativeInput,
    sort_columns: &[String],
) -> Result<NativeSortSpec, FallbackReason> {
    if sort.expr().len() != sort_columns.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let expressions = sort
        .expr()
        .iter()
        .zip(sort_columns.iter())
        .map(|(sort_expr, column)| {
            if input.schema.field_with_name(column.as_str()).is_err() {
                return Err(FallbackReason::UnsupportedExpression);
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
