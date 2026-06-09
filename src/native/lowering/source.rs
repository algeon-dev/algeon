use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::{Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder};

mod local_parquet;
mod reader_pushdown;
mod table_format;

use local_parquet::lower_local_parquet_source_input;
use reader_pushdown::{SourceReaderPushdownPredicate, supported_reader_pushdown_predicate};
pub(super) use reader_pushdown::{
    reader_predicate_proves_local_parquet_at_most_one_row, source_reader_pushdown_predicate,
};
use table_format::lower_table_format_source_input;
pub(super) use table_format::{lower_table_format_zero_column_count_source, table_format_scan_rejection};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr},
    projection_outputs,
};
use crate::{
    native::normalize::{NormalizedNativeScan, NormalizedNativeSource},
    policy::FallbackReason,
};

pub(super) fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    match source.scan {
        NormalizedNativeScan::LocalParquet(source_exec) => lower_local_parquet_source_input(
            source_exec,
            source.filter,
            builder,
            required_output_fields,
            pushed_scan_limit,
        ),
        NormalizedNativeScan::TableFormat(scan_exec) => lower_table_format_source_input(
            scan_exec,
            source.filter,
            builder,
            required_output_fields,
            pushed_scan_limit,
        ),
    }
}

pub(super) fn combine_scan_limits(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(limit), None) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    }
}

pub(super) fn project_scan_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: nexus_query_engine::NodeId,
    input_schema: SchemaRef,
    input_bindings: Vec<super::FieldBinding>,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let selected_bindings = input_bindings
        .iter()
        .filter(|binding| required_output_fields.contains(binding.frontend_index))
        .cloned()
        .collect::<Vec<_>>();
    if selected_bindings.len() == input_schema.fields().len()
        && selected_bindings.iter().zip(input_bindings.iter()).all(|(selected, input)| {
            selected.frontend_index == input.frontend_index && selected.native_name == input.native_name
        })
    {
        return Ok(LoweredNativeInput::with_bindings(input_node, input_schema, input_bindings));
    }

    let mut expressions = Vec::with_capacity(selected_bindings.len());
    let mut fields = Vec::with_capacity(selected_bindings.len());
    for binding in &selected_bindings {
        expressions.push(NativeExpr::column(binding.native_name.as_str()));
        let field = input_schema
            .field_with_name(binding.native_name.as_str())
            .map_err(|_error| FallbackReason::UnsupportedExpression)?;
        fields.push(field.clone());
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input_node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, selected_bindings))
}

pub(super) fn source_bindings_for_schema(schema: &SchemaRef, frontend_schema: SchemaRef) -> Vec<super::FieldBinding> {
    let mut used_frontend_indices = BTreeSet::new();
    schema
        .fields()
        .iter()
        .filter_map(|field| {
            let frontend_index = frontend_schema
                .fields()
                .iter()
                .enumerate()
                .find(|(index, frontend_field)| {
                    !used_frontend_indices.contains(index) && frontend_field.name() == field.name()
                })
                .map(|(index, _)| index)?;
            used_frontend_indices.insert(frontend_index);
            Some(super::FieldBinding {
                frontend_index,
                frontend_name: field.name().clone(),
                native_name: field.name().clone(),
            })
        })
        .collect()
}

pub(super) fn required_source_scan_output_indices(
    source_schema: SchemaRef,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<Vec<usize>, FallbackReason> {
    if let Some(filter) = filter
        && let Some(projection) = filter.projection()
    {
        return required_output_fields
            .requested_indices(filter.schema().fields().len())?
            .into_iter()
            .map(|output_index| projection.get(output_index).copied().ok_or(FallbackReason::UnsupportedExpression))
            .collect();
    }
    required_output_fields.requested_indices(source_schema.fields().len())
}

pub(super) fn add_source_filter_projection(
    builder: &mut NativeQueryPlanBuilder,
    filtered_node: nexus_query_engine::NodeId,
    scan_output: &LoweredNativeInput,
    filter_schema: SchemaRef,
    projection: &[usize],
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let requested_output_indices = required_output_fields.requested_indices(filter_schema.fields().len())?;
    let requested = RequiredFields::only(requested_output_indices.iter().copied().collect());
    let (schema, bindings) = executable_schema_and_bindings(&filter_schema, &requested)?;
    let expressions = requested_output_indices
        .iter()
        .zip(bindings.iter())
        .map(|(output_index, output_binding)| {
            let input_index = projection.get(*output_index).copied().ok_or(FallbackReason::UnsupportedExpression)?;
            let input_binding = scan_output.binding_for_frontend_index(input_index)?;
            Ok(alias_if_needed(
                nexus_query_engine::Expr::column(input_binding.native_name.as_str()),
                output_binding.native_name.as_str(),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(filtered_node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) struct RequiredScanProjectionExpr<'a> {
    pub(super) output_index: usize,
    pub(super) expr: &'a Arc<dyn PhysicalExpr>,
    pub(super) alias: &'a str,
}

pub(super) fn required_scan_projection<'a>(
    projection: &'a ProjectionExprs,
    scan_output_schema: SchemaRef,
    required_scan_output_columns: &[String],
) -> Vec<RequiredScanProjectionExpr<'a>> {
    projection
        .iter()
        .enumerate()
        .filter(|projection_expr| {
            let (_output_index, projection_expr) = projection_expr;
            required_scan_output_columns.iter().any(|column| column == &projection_expr.alias)
                || scan_output_schema.index_of(projection_expr.alias.as_str()).is_ok_and(|index| {
                    scan_output_schema
                        .fields()
                        .get(index)
                        .is_some_and(|field| required_scan_output_columns.iter().any(|column| column == field.name()))
                })
        })
        .map(|(output_index, projection_expr)| RequiredScanProjectionExpr {
            output_index,
            expr: &projection_expr.expr,
            alias: projection_expr.alias.as_str(),
        })
        .collect()
}

pub(super) fn lower_required_file_source_projection_exprs(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .iter()
        .map(|projection_expr| {
            let expr = lower_physical_expr(projection_expr.expr, input_schema)?;
            Ok(alias_if_needed(expr, projection_expr.alias))
        })
        .collect()
}

pub(super) fn required_scan_projection_requires_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> bool {
    projection.iter().any(|projection_expr| {
        let Some(column) = projection_expr.expr.downcast_ref::<Column>() else {
            return true;
        };
        input_schema.fields().get(column.index()).is_none_or(|field| field.name() != projection_expr.alias)
    })
}

pub(super) fn required_scan_projection_needs_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
    current_source_schema: &SchemaRef,
    scan_output_schema: SchemaRef,
) -> Result<bool, FallbackReason> {
    if required_scan_projection_requires_native_projection(projection, input_schema) {
        return Ok(true);
    }
    let projected_schema = required_scan_projection_schema(projection, scan_output_schema)?;
    Ok(projected_schema.as_ref() != current_source_schema.as_ref())
}

pub(super) fn required_scan_projection_schema(
    projection: &[RequiredScanProjectionExpr<'_>],
    scan_output_schema: SchemaRef,
) -> Result<SchemaRef, FallbackReason> {
    let fields = projection
        .iter()
        .map(|projection_expr| {
            scan_output_schema
                .fields()
                .get(projection_expr.output_index)
                .cloned()
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}
