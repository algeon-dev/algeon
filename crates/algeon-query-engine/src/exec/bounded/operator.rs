//! Bounded execution of one plan operator from its resident inputs, shared by
//! compiled pipeline tasks and the bounded aggregate path.

#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;
use std::time::Instant;

use arrow::datatypes::SchemaRef;
use query_planning::{plan::EdgeNormalizeNode, source::LiteralBatchSource};
#[cfg(feature = "cugraph")]
use query_planning::{plan::GraphAlgorithmNode, plan::GraphAlgorithmSpec};
use query_runtime::observability::SourceReadFact;

use super::{
    BoundedExecutionAttempt, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, GpuDataFrame,
    GpuResidentInputMap, GpuResidentSource, LocalParquetSource, NodeId, ProjectionNode, QueryError, QueryErrorCode,
    ResidentGpuFrame, Result, SourceChunkPlanningPolicy, concatenate_source_read_outputs, cudf_query_error,
    execute_edge_normalize, execute_projection_with_budget, execute_projection_with_budget_and_prepared,
    execution_budget_token_for_boundary, observe_device_memory_bytes,
    plan_local_parquet_source_chunks_with_execution_context, projection_regexp_replace_prepare,
    read_gpu_resident_source, read_literal_batch_source, read_local_parquet_chunks_required_only,
    validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
use super::{DataFrame, execute_graph_algorithm_sanitized};
#[cfg(feature = "cugraph")]
use crate::exec::graph_algorithm::{bounded_betweenness::*, graph_empty_input_prevalidated, statistics::*};
use crate::{
    exec::source::{read_local_parquet_source, record_bounded_parquet_source_metrics},
    rapids_backend::rapids_analysis_context,
};

/// A union branch named like the union's declared output.
///
/// Union inputs match the declared schema by position; their own column names
/// may differ. The rename is schema-only, so the branch keeps its columns and
/// their resident owners.
pub(in crate::exec::bounded) fn union_branch_with_output_names(
    node_id: NodeId,
    branch: ResidentGpuFrame,
    output_schema: &SchemaRef,
) -> Result<ResidentGpuFrame> {
    let names = branch.frame.frame().column_names();
    let output_names = output_schema.fields().iter().map(|field| field.name().as_str());
    if names.iter().copied().eq(output_names.clone()) {
        return Ok(branch);
    }
    if names.len() != output_schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.union",
            "union branch column count differs from the union output schema",
        )
        .with_fact("node_id", node_id)
        .with_fact("branch_columns", names.len())
        .with_fact("output_fields", output_schema.fields().len()));
    }
    let renamed = branch.frame.frame().rename_columns(names.iter().copied().zip(output_names)).map_err(|error| {
        cudf_query_error("query_engine.bounded.union.rename", "failed to name a union branch like its output", error)
            .with_fact("node_id", node_id)
    })?;
    let frame = GpuDataFrame::try_new_with_source_residency(renamed, &branch.frame)?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, branch.device_bytes()))
}

pub(in crate::exec::bounded) fn execute_bounded_edge_normalize_from_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    edge_normalize: &EdgeNormalizeNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    execute_bounded_unary(execution, node_id, input.frame, |context, input, metrics| {
        execute_edge_normalize(context, node_id, edge_normalize, input, metrics)
    })
}

pub(in crate::exec::bounded) fn execute_bounded_union(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    inputs: Vec<ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.union",
            "bounded union requires at least one input",
        )
        .with_fact("node_id", node_id));
    }
    if inputs.len() == 1 {
        return Ok(inputs.into_iter().next().expect("single union input checked"));
    }

    let input_count = inputs.len();
    let output = (|| {
        let mut frames = Vec::with_capacity(input_count);
        for input in inputs {
            frames.push(input.frame.into_frame());
        }
        let first = frames.remove(0);
        let rest = frames.iter().collect::<Vec<_>>();
        let exec_runtime = execution.context().runtime()?;
        let overrides = exec_runtime.exec();
        let timer = Instant::now();
        let output = first.on(overrides).concat_many(&rest).map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.union.concat",
                "failed to concatenate bounded native union inputs",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("input_count", input_count)
        })?;
        execution.metrics().add_operator_time(timer.elapsed());

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics())?;
        Ok::<_, QueryError>((output_frame, output_bytes))
    })();
    match output {
        Ok((frame, device_bytes)) => Ok(ResidentGpuFrame { frame, device_bytes }),
        Err(error) => Err(error),
    }
}

#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) fn execute_bounded_graph_algorithm_from_inputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    graph: &GraphAlgorithmNode,
    input: ResidentGpuFrame,
    side_input_frames: BTreeMap<usize, ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    let ResidentGpuFrame { frame: mut input_frame, .. } = input;
    let _retained_normalization_source = input_frame.take_retained_normalization_source();
    let input_rows = input_frame.num_rows() as u64;
    let input_columns = input_frame.num_columns() as u64;
    let input_graph_residency = input_frame.graph_residency().cloned();
    let prior_null_edge_rows_dropped = execution.metrics().null_edge_rows_dropped(graph.edge_input());
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope {
        weight: graph.spec.algorithm.consumes_weight(),
        edge_id: graph.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input_frame = match super::edge_null_filter::filter_null_edge_rows(
        execution.context(),
        node_id,
        "graph_edge_null_filter",
        input_frame.into_frame(),
        &graph.spec.edge_columns,
        null_filter_scope,
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let sanitized_input_rows = filtered_input_frame.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    let mut filtered_input = GpuDataFrame::new(filtered_input_frame);
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        filtered_input = filtered_input.with_graph_residency(graph_residency);
    }
    let mut side_inputs = BTreeMap::new();
    for (input_id, frame) in side_input_frames {
        side_inputs.insert(input_id, frame.frame);
    }
    if let Err(error) = validate_betweenness_static_seed_policy(&graph.spec) {
        return Err(error.with_fact_if_absent("node_id", node_id));
    }
    let empty_input_prevalidated = match graph_empty_input_prevalidated(
        execution.context(),
        graph,
        filtered_input.frame(),
        &side_inputs,
        input_rows,
        sanitized_input_rows,
    ) {
        Ok(empty_input_prevalidated) => empty_input_prevalidated,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let mut runtime_spec = RuntimeGraphSpecResolution::CheckedNoRewrite;
    if !empty_input_prevalidated {
        execution.metrics().record_graph_claim_calibration(node_id, graph.spec.algorithm);
        let graph_statistics_probe =
            match graph_runtime_vertex_count_probe(execution.context(), &graph.spec, filtered_input.frame()) {
                Ok(probe) => probe,
                Err(error) => {
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            };
        if let Some(vertex_count) = graph_statistics_probe
            && let Err(error) = validate_betweenness_seed_policy_for_vertex_count(&graph.spec, vertex_count)
        {
            return Err(error.with_fact_if_absent("node_id", node_id));
        }
        runtime_spec =
            runtime_graph_spec_resolution_after_bounded_seed_validation(&graph.spec, graph_statistics_probe.is_some());
    }
    let output = match execute_graph_algorithm_sanitized(
        execution.context(),
        node_id,
        graph,
        runtime_spec,
        filtered_input,
        side_inputs,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        empty_input_prevalidated,
        execution.metrics(),
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&output, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame { frame: output, device_bytes: output_bytes })
}

#[cfg(feature = "cugraph")]
/// Runs the runtime graph vertex-count probe on the attempt's own runtime.
///
/// This runtime sizing only informs betweenness seed selection. It carries no
/// reservation or rejection authority, so an allocation failure degrades the
/// estimate instead of failing the query.
#[cfg(feature = "cugraph")]
fn graph_runtime_vertex_count_probe(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    input: &DataFrame,
) -> Result<Option<u64>> {
    match runtime_graph_statistics_probe(context, input, spec) {
        Ok(probe) => Ok(probe),
        Err(error)
            if matches!(
                error.code(),
                QueryErrorCode::QueryMemoryLimitExceeded
                    | QueryErrorCode::DeviceOutOfMemory
                    | QueryErrorCode::DeviceAllocationFailed
            ) =>
        {
            Ok(None)
        },
        Err(error) => Err(error),
    }
}

pub(in crate::exec::bounded) fn execute_bounded_local_parquet_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source, rapids_analysis_context(execution.context().planning_policy()))?;
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        node_id,
        source,
        SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
            source,
            execution.context().memory_control_policy(),
            execution.context().source_chunk_planning_mode(),
            execution.context().source_chunk_planning_options(),
        ),
        execution.context(),
    )?;
    for chunk in &chunk_plan.chunks {
        execution.record_source_read(SourceReadFact::ChunkPlanned { row_groups: chunk.row_group_count() as u64 });
    }
    if chunk_plan.chunks.is_empty() {
        let frame = read_local_parquet_source(execution.context(), node_id, source, execution.metrics())?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        return Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes });
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, execution.metrics());

    let chunks =
        read_local_parquet_chunks_required_only(execution, source, &chunk_plan.chunks, "bounded_generic_source_read")?;

    concatenate_source_read_outputs(
        execution,
        node_id,
        chunks,
        "query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

pub(in crate::exec::bounded) fn execute_bounded_gpu_resident_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &GpuResidentSource,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(execution.context(), source, gpu_inputs, execution.metrics())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_literal_batch_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LiteralBatchSource,
) -> Result<ResidentGpuFrame> {
    let frame = read_literal_batch_source(execution.context(), source, execution.metrics())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes })
}

/// Runs a unary operator that takes no budget token and observes its output.
pub(in crate::exec::bounded) fn execute_bounded_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    input: GpuDataFrame,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionMetrics) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let frame = execute(execution.context(), input, execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    Ok(ResidentGpuFrame { frame, device_bytes: output_bytes })
}

/// Runs a unary operator under a `stage` budget token. The token shares the
/// context's allocator member: it labels the stage in diagnostics and never
/// reserves or resizes memory.
pub(in crate::exec::bounded) fn run_budgeted(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: GpuDataFrame,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &ExecutionMetrics) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let token = execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token",
    )
    .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    execute_bounded_unary(execution, node_id, input, |context, input, metrics| execute(context, input, &token, metrics))
}

pub(in crate::exec::bounded) fn execute_bounded_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return run_budgeted(execution, node_id, stage, input.frame, |context, input, token, metrics| {
            execute_projection_with_budget(context, projection, input, token, metrics)
        });
    }
    execute_bounded_regexp_projection(execution, node_id, stage, projection, input.frame)
}

fn execute_bounded_regexp_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input_frame: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    // Prepared state stays owned by the native operation and runs under the
    // attempt's allocator-enforced budget token.
    let prepared =
        match projection_regexp_replace_prepare(execution.context(), projection, &input_frame, execution.metrics()) {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                // Defense in depth: no regexp after the gate above.
                let token = match execution_budget_token_for_boundary(
                    execution.context(),
                    execution.device_budget_bytes(),
                    stage,
                    "query_engine.bounded_generic.budget_token",
                    "failed to create bounded native generic operator budget token",
                ) {
                    Ok(token) => token,
                    Err(error) => {
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                };
                match execute_projection_with_budget(
                    execution.context(),
                    projection,
                    input_frame,
                    &token,
                    execution.metrics(),
                ) {
                    Ok(frame) => {
                        let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                return Err(error.with_fact_if_absent("node_id", node_id));
                            },
                        };
                        return Ok(ResidentGpuFrame { frame, device_bytes: output_bytes });
                    },
                    Err(error) => {
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                }
            },
            Err(error) => {
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };

    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let result = execute_projection_with_budget_and_prepared(
        execution.context(),
        projection,
        input_frame,
        &token,
        Some(prepared),
        execution.metrics(),
    );
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame { frame, device_bytes: output_bytes })
}
