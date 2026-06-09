mod aggregate;
mod bindings;
mod expr;
mod join;
mod join_output;
mod join_residual;
mod root;
mod sort;
mod source;
mod unary;
mod union;
mod window;

use std::sync::Arc;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use bindings::{FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings, rename_field};
use datafusion::{datasource::memory::MemorySourceConfig, physical_plan::ExecutionPlan};
use nexus_query_engine::{
    Expr as NativeExpr, GpuResidentSource as NativeGpuResidentSource, LimitSpec as NativeLimitSpec,
    LiteralBatchSource as NativeLiteralBatchSource, ProjectionOutput as NativeProjectionOutput,
    QueryPlan as NativeQueryPlan, QueryPlanBuilder as NativeQueryPlanBuilder, SourceNode as NativeSourceNode,
};

use super::normalize::{NormalizedNativeInput, NormalizedNativePlan, NormalizedNativeRoot};
use crate::policy::FallbackReason;

/// Lowers a normalized native tree into the DataFusion-free query IR.
///
/// Aggregate roots over `single_source_input` use the slice1 path so source
/// pruning, filter pushdown, zero-column count, aggregate, sort, and limit can
/// be planned together. All other aggregate roots lower their input tree first,
/// which covers joins and recursive unary/aggregate inputs.
pub(super) fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    match &normalized.root {
        NormalizedNativeRoot::Aggregate(root)
            if normalized.single_source_input().is_some_and(|source| source.local_parquet_source_exec().is_some()) =>
        {
            root::lower_slice1_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root)
            if normalized.single_source_input().is_some_and(|source| {
                source.table_format_scan_exec().is_some()
                    && crate::native::validate::native_zero_column_count_source_is_safe(
                        source.filter,
                        root.pre_aggregate_projection,
                        root.aggregate,
                    )
            }) =>
        {
            root::lower_table_format_zero_column_count_plan(normalized, root)
        },
        NormalizedNativeRoot::Aggregate(root) => root::lower_joined_aggregate_plan(normalized, root),
        NormalizedNativeRoot::Input(root) => root::lower_input_root_plan(normalized, root),
    }
}

pub(super) fn projection_outputs(
    expressions: Vec<NativeExpr>,
    schema: &SchemaRef,
) -> Result<Vec<NativeProjectionOutput>, FallbackReason> {
    if expressions.len() != schema.fields().len() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| NativeProjectionOutput::new(expr, field.as_ref().clone()))
        .collect())
}

/// Recursively lowers a normalized input subtree.
///
/// `required_output_fields` is the set of output field identities requested by
/// the parent lowering step. `All` means no explicit parent pruning. `Only({})`
/// means the parent explicitly needs zero output fields.
fn lower_native_input(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    lower_native_input_with_scan_limit(input, builder, required_output_fields, None)
}

fn lower_native_input_with_scan_limit(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    source_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, FallbackReason> {
    match input {
        NormalizedNativeInput::Source(source) => {
            source::lower_source_input(source, builder, required_output_fields, source_scan_limit)
        },
        NormalizedNativeInput::GpuResidentSource(source) => {
            lower_gpu_resident_source_input(source, builder, required_output_fields)
        },
        NormalizedNativeInput::LiteralSource(source) => {
            lower_literal_source_input(source, builder, required_output_fields)
        },
        NormalizedNativeInput::Filter(filter) => unary::lower_filter_input(filter, builder, required_output_fields),
        NormalizedNativeInput::Projection(projection) => {
            unary::lower_projection_input(projection, builder, required_output_fields)
        },
        NormalizedNativeInput::Limit(limit) => {
            let pushed_scan_limit = source_scan_limit_through_limit(limit.spec, source_scan_limit);
            let input =
                lower_native_input_with_scan_limit(&limit.input, builder, required_output_fields, pushed_scan_limit)?;
            let node = builder.add_limit(input.node, limit.spec, Arc::clone(&input.schema));
            Ok(LoweredNativeInput::with_bindings(node, input.schema, input.bindings))
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            aggregate::lower_aggregate_input(aggregate, builder, required_output_fields)
        },
        NormalizedNativeInput::Sort(sort) => window::lower_sort_input(sort, builder, required_output_fields),
        NormalizedNativeInput::Window(window) => window::lower_window_input(window, builder, required_output_fields),
        NormalizedNativeInput::HashJoin(join) => join::lower_hash_join_input(join, builder, required_output_fields),
        NormalizedNativeInput::CrossJoin(join) => join::lower_cross_join_input(join, builder, required_output_fields),
        NormalizedNativeInput::NestedLoopJoin(join) => {
            join::lower_nested_loop_join_input(join, builder, required_output_fields)
        },
        NormalizedNativeInput::Union(union) => union::lower_union_input(union, builder, required_output_fields),
    }
}

pub(super) fn source_scan_limit_from_limit_spec(limit: Option<NativeLimitSpec>) -> Option<usize> {
    let limit = limit?;
    let fetch = limit.fetch?;
    if fetch == 0 {
        return Some(0);
    }
    limit.skip.checked_add(fetch)
}

fn source_scan_limit_through_limit(limit: NativeLimitSpec, downstream_scan_limit: Option<usize>) -> Option<usize> {
    let effective_fetch = match (limit.fetch, downstream_scan_limit) {
        (Some(fetch), Some(downstream)) => fetch.min(downstream),
        (Some(fetch), None) => fetch,
        (None, Some(downstream)) => downstream,
        (None, None) => return None,
    };
    if effective_fetch == 0 {
        return Some(0);
    }
    limit.skip.checked_add(effective_fetch)
}

fn lower_literal_source_input(
    source: &super::normalize::NormalizedNativeLiteralSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let (source_schema, batches, fetch) = match source {
        super::normalize::NormalizedNativeLiteralSource::PlaceholderRow(placeholder) => {
            if !placeholder.schema().fields().is_empty() {
                return Err(FallbackReason::UnsupportedOperator);
            }
            (placeholder.schema(), NativeLiteralBatchSource::placeholder_row().batches, None)
        },
        super::normalize::NormalizedNativeLiteralSource::Empty(empty) => {
            let schema = empty.schema();
            let literal = NativeLiteralBatchSource::empty(Arc::clone(&schema));
            (schema, literal.batches, None)
        },
        super::normalize::NormalizedNativeLiteralSource::Memory(source_exec) => {
            let memory = source_exec
                .data_source()
                .downcast_ref::<MemorySourceConfig>()
                .ok_or(FallbackReason::UnsupportedOperator)?;
            let batches = memory
                .partitions()
                .iter()
                .flat_map(|partition| partition.iter())
                .map(|batch| project_memory_batch(batch, memory.projection()))
                .collect::<Result<Vec<_>, _>>()?;
            (source_exec.schema(), batches, source_exec.fetch())
        },
    };

    let (native_schema, source_bindings) = executable_schema_and_bindings(&source_schema, &RequiredFields::all())?;
    let source_node = builder
        .add_source(NativeSourceNode::LiteralBatch(NativeLiteralBatchSource::new(Arc::clone(&native_schema), batches)));
    let source_node = if let Some(fetch) = fetch {
        builder.add_limit(source_node, nexus_query_engine::LimitSpec::fetch(fetch), Arc::clone(&native_schema))
    } else {
        source_node
    };
    let source_input = LoweredNativeInput::with_bindings(source_node, Arc::clone(&native_schema), source_bindings);
    project_literal_output_columns(builder, source_input, required_output_fields)
}

fn project_memory_batch(batch: &RecordBatch, projection: &Option<Vec<usize>>) -> Result<RecordBatch, FallbackReason> {
    projection.as_ref().map_or_else(
        || Ok(batch.clone()),
        |projection| batch.project(projection).map_err(|_| FallbackReason::UnsupportedOperator),
    )
}

fn project_literal_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    if required_output_fields.is_all() {
        return Ok(input);
    }
    let (projected_schema, projected_bindings) =
        executable_schema_and_bindings(&input.frontend_schema()?, required_output_fields)?;
    let expressions = projected_bindings
        .iter()
        .map(|projected| {
            let input_binding = input.binding_for_frontend_index(projected.frontend_index)?;
            let field = input
                .schema
                .field_with_name(input_binding.native_name.as_str())
                .map_err(|_| FallbackReason::UnsupportedExpression)?;
            Ok(NativeProjectionOutput::new(
                NativeExpr::column(input_binding.native_name.as_str()),
                rename_field(field, projected.native_name.as_str()),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let projected_node = builder.add_projection(input.node, expressions);
    Ok(LoweredNativeInput::with_bindings(projected_node, projected_schema, projected_bindings))
}

fn lower_gpu_resident_source_input(
    source: &super::normalize::NormalizedNativeGpuResidentSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    let full_schema = source.source_exec.schema();
    let full_required = RequiredFields::all();
    let (source_schema, source_bindings) = executable_schema_and_bindings(&full_schema, &full_required)?;
    let source_node = builder.add_source(NativeSourceNode::GpuResident(NativeGpuResidentSource::new(
        source.input_name.as_str(),
        Arc::clone(&source_schema),
    )));
    if required_output_fields.is_all() {
        return Ok(LoweredNativeInput::with_bindings(source_node, source_schema, source_bindings));
    }

    let (projected_schema, projected_bindings) = executable_schema_and_bindings(&full_schema, required_output_fields)?;
    let expressions = projected_bindings
        .iter()
        .map(|projected| {
            let input = source_bindings
                .iter()
                .find(|binding| binding.frontend_index == projected.frontend_index)
                .ok_or(FallbackReason::UnsupportedExpression)?;
            Ok(NativeProjectionOutput::new(
                NativeExpr::column(input.native_name.as_str()),
                rename_field(
                    source_schema
                        .fields()
                        .get(input.frontend_index)
                        .ok_or(FallbackReason::UnsupportedExpression)?
                        .as_ref(),
                    projected.native_name.as_str(),
                ),
            ))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let projected_node = builder.add_projection(source_node, expressions);
    Ok(LoweredNativeInput::with_bindings(projected_node, projected_schema, projected_bindings))
}
