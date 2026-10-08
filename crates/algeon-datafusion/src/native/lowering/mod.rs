mod advisory;
mod aggregate;
mod bindings;
mod expr;
mod join;
mod join_output;
mod join_residual;
mod naming;
mod sort;
mod source;
mod unary;
mod union;
mod window;

use std::{collections::BTreeSet, sync::Arc};

pub(in crate::native) use aggregate::aggregate_input_demand;
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
pub(in crate::native) use bindings::Demand;
use bindings::{FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings, rename_field};
use datafusion::{datasource::memory::MemorySourceConfig, physical_plan::ExecutionPlan};
#[cfg(test)]
pub(in crate::native) use expr::lower_physical_expr as lower_physical_expr_for_test;
use query_planning::{
    expr::{Expr as NativeExpr, LimitSpec as NativeLimitSpec},
    plan::{
        ProjectionOutput as NativeProjectionOutput, QueryPlan as NativeQueryPlan,
        QueryPlanBuilder as NativeQueryPlanBuilder,
    },
    source::{
        GpuResidentSource as NativeGpuResidentSource, LiteralBatchSource as NativeLiteralBatchSource,
        SourceNode as NativeSourceNode,
    },
};
pub(in crate::native) use source::{parquet_column_requirements, parquet_reads_row_count_only};

use super::normalize::{NormalizedNativeInput, NormalizedNativePlan};
use crate::{
    capability::CapabilityReason,
    native::{LoweringFailure, ParquetReadStrategyConfig},
};

/// Lowers a normalized native tree into the DataFusion-free query IR. The
/// root output is the host sink in the frontend schema; `ordered` says whether
/// the plan declares an output ordering.
pub(super) fn lower_normalized_plan(
    normalized: &NormalizedNativePlan<'_>,
    ordered: bool,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeQueryPlan, LoweringFailure> {
    let mut builder = NativeQueryPlanBuilder::new();
    let demand = Demand { ordered, ..Demand::all() };
    let output = lower_native_input(&normalized.input, &mut builder, &demand, source_reads)?;
    Ok(builder.finish_with_host_sink_schema(output.node, output.frontend_schema()?)?)
}

pub(super) fn projection_outputs(
    expressions: Vec<NativeExpr>,
    schema: &SchemaRef,
) -> Result<Vec<NativeProjectionOutput>, LoweringFailure> {
    if expressions.len() != schema.fields().len() {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    Ok(expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| NativeProjectionOutput::new(expr, field.as_ref().clone()))
        .collect())
}

fn project_bindings_to_schema(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    schema: SchemaRef,
    bindings: Vec<FieldBinding>,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let expressions = bindings
        .iter()
        .map(|binding| {
            let input_binding = input.binding_for_frontend_index(binding.frontend_index)?;
            Ok(expr::alias_if_needed(
                NativeExpr::column(input_binding.native_name.as_str()),
                binding.native_name.as_str(),
            ))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

/// Recursively lowers a normalized input subtree for what its parent
/// demands. `demand.fields` of `All` means no explicit parent pruning, and
/// `Only({})` means the parent needs zero output fields.
fn lower_native_input(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    demand: &Demand,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let fields = &demand.fields;
    let output = match input {
        NormalizedNativeInput::Source(source) => source::lower_source_input(source, builder, demand, source_reads),
        NormalizedNativeInput::GpuResidentSource(source) => lower_gpu_resident_source_input(source, builder, fields),
        NormalizedNativeInput::LiteralSource(source) => lower_literal_source_input(source, builder, fields),
        NormalizedNativeInput::Filter(filter) => unary::lower_filter_input(filter, builder, demand, source_reads),
        NormalizedNativeInput::Projection(projection) => {
            unary::lower_projection_input(projection, builder, demand, source_reads)
        },
        NormalizedNativeInput::Limit(limit) => {
            let input_demand = Demand {
                fields: fields.clone(),
                row_count_only: false,
                row_limit: source_scan_limit_through_limit(limit.spec, demand.row_limit),
                extra_fields_allowed: demand.extra_fields_allowed,
                ordered: demand.ordered || limit.input_ordered,
            };
            let input = lower_native_input(&limit.input, builder, &input_demand, source_reads)?;
            let node = builder.add_limit(input.node, limit.spec)?;
            Ok(LoweredNativeInput::with_bindings(node, input.schema, input.bindings))
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            aggregate::lower_aggregate_input(aggregate, builder, fields, source_reads)
        },
        NormalizedNativeInput::Sort(sort) => window::lower_sort_input(sort, builder, fields, source_reads),
        NormalizedNativeInput::Window(window) => window::lower_window_input(window, builder, fields, source_reads),
        NormalizedNativeInput::HashJoin(join) => join::lower_hash_join_input(join, builder, fields, source_reads),
        NormalizedNativeInput::CrossJoin(join) => join::lower_cross_join_input(join, builder, fields, source_reads),
        NormalizedNativeInput::NestedLoopJoin(join) => {
            join::lower_nested_loop_join_input(join, builder, fields, source_reads)
        },
        NormalizedNativeInput::Union(union) => union::lower_union_input(union, builder, fields, source_reads),
    }
    .map_err(|failure| failure.within_operator(|| input.shape().operator().to_string()))?;
    advisory::attach_input(builder, input, &output)?;
    Ok(output)
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
) -> Result<LoweredNativeInput, LoweringFailure> {
    let (source_schema, batches, fetch) = match source {
        super::normalize::NormalizedNativeLiteralSource::PlaceholderRow(placeholder) => {
            if !placeholder.schema().fields().is_empty() {
                return Err(CapabilityReason::UnsupportedOperator.into());
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
                .ok_or(CapabilityReason::UnsupportedOperator)?;
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
        builder.add_limit(source_node, NativeLimitSpec::fetch(fetch))?
    } else {
        source_node
    };
    let source_input = LoweredNativeInput::with_bindings(source_node, Arc::clone(&native_schema), source_bindings);
    project_literal_output_columns(builder, source_input, required_output_fields)
}

fn project_memory_batch(batch: &RecordBatch, projection: &Option<Vec<usize>>) -> Result<RecordBatch, LoweringFailure> {
    projection.as_ref().map_or_else(
        || Ok(batch.clone()),
        |projection| batch.project(projection).map_err(|_| CapabilityReason::UnsupportedOperator.into()),
    )
}

fn project_literal_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
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
                .map_err(|_| CapabilityReason::UnsupportedExpression)?;
            Ok(NativeProjectionOutput::new(
                NativeExpr::column(input_binding.native_name.as_str()),
                rename_field(field, projected.native_name.as_str()),
            ))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let projected_node = builder.add_projection(input.node, expressions)?;
    Ok(LoweredNativeInput::with_bindings(projected_node, projected_schema, projected_bindings))
}

fn lower_gpu_resident_source_input(
    source: &super::normalize::NormalizedNativeGpuResidentSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let full_schema = source.source_exec.schema();
    let full_required = RequiredFields::all();
    let (source_schema, source_bindings) = executable_schema_and_bindings(&full_schema, &full_required)?;
    let unique_key_columns = source
        .unique_output_key
        .as_ref()
        .map(|columns| {
            columns
                .iter()
                .map(|column| {
                    source_bindings
                        .iter()
                        .find(|binding| binding.frontend_name == *column)
                        .map(|binding| binding.native_name.clone())
                        .ok_or(CapabilityReason::UnsupportedExpression)
                })
                .collect::<Result<BTreeSet<_>, _>>()
        })
        .transpose()?;
    let mut resident_source = NativeGpuResidentSource::new(source.input_name.as_str(), Arc::clone(&source_schema));
    if let Some(row_count_upper_bound) = source.row_count_upper_bound {
        resident_source = resident_source.with_row_count_upper_bound(row_count_upper_bound);
    }
    if let Some(unique_key_columns) = unique_key_columns {
        resident_source = resident_source.with_unique_key_columns(unique_key_columns);
    }
    let source_node = builder.add_source(NativeSourceNode::GpuResident(resident_source));
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
                .ok_or(CapabilityReason::UnsupportedExpression)?;
            Ok(NativeProjectionOutput::new(
                NativeExpr::column(input.native_name.as_str()),
                rename_field(
                    source_schema
                        .fields()
                        .get(input.frontend_index)
                        .ok_or(CapabilityReason::UnsupportedExpression)?
                        .as_ref(),
                    projected.native_name.as_str(),
                ),
            ))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let projected_node = builder.add_projection(source_node, expressions)?;
    Ok(LoweredNativeInput::with_bindings(projected_node, projected_schema, projected_bindings))
}
