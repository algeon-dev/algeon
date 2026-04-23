use std::{collections::BTreeMap, sync::atomic::Ordering};

use super::super::*;

#[path = "adaptive_betweenness.rs"]
pub(in crate::exec) mod adaptive_betweenness;
#[path = "build.rs"]
pub(in crate::exec) mod build;
#[path = "centrality.rs"]
mod centrality;
#[path = "community.rs"]
mod community;
#[path = "diagnostics.rs"]
pub(in crate::exec) mod diagnostics;
#[path = "layout.rs"]
mod layout;
#[path = "memory.rs"]
pub(in crate::exec) mod memory;
#[path = "params.rs"]
mod params;
#[path = "resident_cache.rs"]
pub(in crate::exec) mod resident_cache;
#[path = "statistics.rs"]
pub(in crate::exec) mod statistics;
#[path = "structure.rs"]
mod structure;
#[path = "traversal.rs"]
mod traversal;

use adaptive_betweenness::{
    FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED, RuntimeGraphSpecResolution,
    execute_chunked_betweenness_retry, graph_adaptive_memory_observation_from_error, graph_adaptive_retry_seed_batches,
    record_graph_adaptive_memory_history_from_error, resolve_runtime_graph_spec_for_execution,
    runtime_graph_spec_resolution_after_bounded_seed_validation, validate_betweenness_static_seed_policy,
};
pub use adaptive_betweenness::{
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test,
};
use build::*;
use centrality::*;
use community::*;
use diagnostics::*;
use layout::*;
use memory::*;
use params::*;
use resident_cache::*;
pub use statistics::force_next_graph_statistics_optional_resource_exhausted_for_test;
use structure::*;
use traversal::{execute_bfs, execute_sssp};

pub(in crate::exec) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_graph_algorithm_with_cache_requirement(context, node_id, node, input, side_inputs, None, metrics)
}

pub(in crate::exec) fn execute_graph_algorithm_cache_hit(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    preparation_pin_id: u64,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_graph_algorithm_with_cache_requirement(
        context,
        node_id,
        node,
        input,
        side_inputs,
        Some(preparation_pin_id),
        metrics,
    )
}

fn execute_graph_algorithm_with_cache_requirement(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    preparation_pin_id: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let input_graph_residency = input.graph_residency().cloned();
    let prior_null_edge_rows_dropped = graph_input_null_edge_rows_dropped(metrics, node.input);
    let null_filter_scope = super::super::edge_null_filter::EdgeNullFilterScope {
        weight: node.spec.algorithm.consumes_weight(),
        edge_id: node.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input = super::super::edge_null_filter::filter_null_edge_rows(
        context,
        node_id,
        "graph_edge_null_filter",
        input.frame().clone(),
        &node.spec.edge_columns,
        null_filter_scope,
        metrics,
    )?;
    let sanitized_input_rows = filtered_input.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    validate_betweenness_static_seed_policy(&node.spec)?;
    let graph_memory_limit_bytes =
        admit_graph_memory(context, node_id, &node.spec, &filtered_input, sanitized_input_rows, metrics)?;
    let runtime_spec = runtime_graph_spec_resolution_after_bounded_seed_validation(&node.spec, false);
    let mut sanitized_input = GpuDataFrame::try_new_with_source_residency(filtered_input, &input)?;
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        sanitized_input = sanitized_input.with_graph_residency(graph_residency);
    }
    execute_graph_algorithm_sanitized(
        context,
        node_id,
        node,
        runtime_spec,
        sanitized_input,
        side_inputs,
        graph_memory_limit_bytes,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        preparation_pin_id,
        metrics,
    )
}

pub(in crate::exec) fn execute_graph_algorithm_sanitized(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    runtime_spec: RuntimeGraphSpecResolution,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    graph_memory_limit_bytes: Option<u64>,
    input_rows: u64,
    input_columns: u64,
    null_edge_rows_dropped: u64,
    preparation_pin_id: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let boundary_before = metrics.boundary.clone();
    let graph_budget_event_start = metrics.budgeted_execution_events.len();
    let pool_usage_before = graph_pool_usage_snapshot(context);
    let timer = Instant::now();
    let runtime_spec = match runtime_spec {
        RuntimeGraphSpecResolution::Unresolved => resolve_runtime_graph_spec_for_execution(
            context,
            input.frame(),
            &node.spec,
            graph_memory_limit_bytes,
            metrics,
        )?,
        runtime_spec => runtime_spec,
    };
    let spec = runtime_spec.spec_or(&node.spec);
    let executor = graph_algorithm_executor(spec.algorithm)?;
    let adaptive_retry_inputs = graph_adaptive_retry_seed_batches(spec).map(|batches| {
        (
            GraphAlgorithmExecutionInputs {
                edge_list: input.clone(),
                side_inputs: side_inputs.clone(),
                graph_memory_limit_bytes,
                preparation_pin_id,
            },
            batches,
        )
    });
    let inputs =
        GraphAlgorithmExecutionInputs { edge_list: input, side_inputs, graph_memory_limit_bytes, preparation_pin_id };
    let result = match execute_graph_algorithm_attempt(context, spec, inputs, executor) {
        Ok(result) => result,
        Err(error) if should_retry_graph_resource_exhausted(context, &error, adaptive_retry_inputs.as_ref()) => {
            let Some((retry_inputs, seed_batches)) = adaptive_retry_inputs else {
                return Err(error);
            };
            record_graph_adaptive_memory_history_from_error(context, &node.spec, &error);
            record_graph_budget_observation_from_error(metrics, &error);
            metrics.record_adaptive_runtime_action_counts(1, 0, 0);
            let retry_result = execute_chunked_betweenness_retry(context, spec, retry_inputs, seed_batches, executor);
            if let Err(retry_error) = &retry_result {
                record_graph_adaptive_memory_history_from_error(context, &node.spec, retry_error);
                record_graph_budget_observation_from_error(metrics, retry_error);
            }
            retry_result?
        },
        Err(error) => return Err(error),
    };
    let pool_usage_after = graph_pool_usage_snapshot(context);
    let elapsed = timer.elapsed();
    metrics.operator_time += elapsed;
    for observation in &result.budget_observations {
        observation.record(metrics);
    }

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
    let budget_graph_peak_bytes = metrics
        .budgeted_execution_events
        .iter()
        .skip(graph_budget_event_start)
        .filter(|event| event.boundary == crate::graph_memory::GRAPH_MEMORY_BOUNDARY)
        .filter_map(|event| event.allocator_high_water_bytes)
        .max();
    let pool_graph_peak_bytes = graph_pool_high_water_delta(pool_usage_before, pool_usage_after);
    let measured_graph_peak_bytes = budget_graph_peak_bytes.or(pool_graph_peak_bytes);
    let graph_peak_bytes = measured_graph_peak_bytes.or(Some(0));
    metrics.record_graph_model_error_ratio(
        &node_id.to_string(),
        node.spec.algorithm.as_str(),
        measured_graph_peak_bytes,
    );
    let boundary_delta = boundary_delta(&boundary_before, &metrics.boundary);
    metrics.record_graph_algorithm_runtime(crate::metrics::GraphAlgorithmRuntimeMetrics {
        query_id: None,
        node_id: node_id.to_string(),
        algorithm: node.spec.algorithm.as_str().to_string(),
        runtime_outcome: "succeeded".to_string(),
        input_boundary: "gpu_resident".to_string(),
        output_boundary: "gpu_resident".to_string(),
        construction_policy: node.spec.graph_options.policy.as_str().to_string(),
        source_column: node.spec.edge_columns.source.clone(),
        destination_column: node.spec.edge_columns.destination.clone(),
        weight_column: node.spec.edge_columns.weight.clone(),
        directed: node.spec.graph_options.directed,
        renumber: node.spec.graph_options.renumber,
        algorithm_parameters: graph_algorithm_parameters(&node.spec.params),
        algorithm_metadata: graph_algorithm_metadata_with_null_drops(result.algorithm_metadata, null_edge_rows_dropped),
        input_rows,
        input_columns,
        output_rows: gpu.num_rows() as u64,
        output_columns: gpu.num_columns() as u64,
        output_device_bytes,
        graph_peak_bytes,
        elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        gpu_resident_input_handoffs: boundary_delta.gpu_resident_input_handoffs,
        host_to_device_uploads: boundary_delta.host_to_device_uploads,
        device_to_host_materializations: boundary_delta.device_to_host_materializations,
    });
    Ok(gpu)
}

fn execute_graph_algorithm_attempt(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
    executor: GraphAlgorithmExecutor,
) -> Result<GraphAlgorithmExecutionResult> {
    if context.native_pipeline_execution_mode() == crate::mode::NativePipelineExecutionMode::Adaptive
        && graph_adaptive_retry_seed_batches(spec).is_some()
        && FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED.swap(false, Ordering::SeqCst)
    {
        let mut error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.graph_algorithm.adaptive_retry_test",
            "forced graph adaptive retry resource exhaustion",
        )
        .with_fact("reason", "forced_graph_adaptive_oom_retry_test")
        .with_fact("algorithm", spec.algorithm.as_str());
        if let Some(limit) = inputs.graph_memory_limit_bytes {
            error = error
                .with_fact("graph_budget_limit_bytes", limit)
                .with_fact("graph_budget_peak_bytes", limit)
                .with_fact("graph_budget_status", "reservation_resource_exhausted");
        }
        return Err(error);
    }
    executor(context, spec, inputs)
}

fn should_retry_graph_resource_exhausted(
    context: &ExecutionContext,
    error: &QueryError,
    retry_inputs: Option<&(GraphAlgorithmExecutionInputs, Vec<crate::plan::GraphBetweennessSeeds>)>,
) -> bool {
    let Some((inputs, _)) = retry_inputs else {
        return false;
    };
    context.native_pipeline_execution_mode() == crate::mode::NativePipelineExecutionMode::Adaptive
        && error.code() == QueryErrorCode::CudfResourceExhausted
        && inputs.graph_memory_limit_bytes.is_some()
        && graph_adaptive_memory_observation_from_error(error).is_some()
}

// Common return envelope for algorithm-specific cuGraph executors.
pub(super) struct GraphAlgorithmExecutionResult {
    pub(super) frame: DataFrame,
    pub(super) algorithm_metadata: String,
    pub(super) budget_observations: Vec<GraphBudgetObservation>,
}

// Shared graph inputs after edge-list preparation and side-input materialization.
#[derive(Clone)]
pub(super) struct GraphAlgorithmExecutionInputs {
    pub(super) edge_list: GpuDataFrame,
    pub(super) side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    pub(super) graph_memory_limit_bytes: Option<u64>,
    pub(super) preparation_pin_id: Option<u64>,
}

type GraphAlgorithmExecutor = fn(
    &ExecutionContext,
    &crate::plan::GraphAlgorithmSpec,
    GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult>;

const GRAPH_ALGORITHM_EXECUTORS: &[(crate::plan::GraphAlgorithm, GraphAlgorithmExecutor)] = &[
    (crate::plan::GraphAlgorithm::PageRank, execute_pagerank),
    (crate::plan::GraphAlgorithm::PersonalizedPageRank, execute_personalized_pagerank),
    (crate::plan::GraphAlgorithm::Bfs, execute_bfs),
    (crate::plan::GraphAlgorithm::Louvain, execute_louvain),
    (crate::plan::GraphAlgorithm::EigenvectorCentrality, execute_eigenvector_centrality),
    (crate::plan::GraphAlgorithm::KatzCentrality, execute_katz_centrality),
    (crate::plan::GraphAlgorithm::Hits, execute_hits),
    (crate::plan::GraphAlgorithm::BetweennessCentrality, execute_betweenness_centrality),
    (crate::plan::GraphAlgorithm::EdgeBetweennessCentrality, execute_edge_betweenness_centrality),
    (crate::plan::GraphAlgorithm::Sssp, execute_sssp),
    (crate::plan::GraphAlgorithm::DegreesAll, execute_degrees_all),
    (crate::plan::GraphAlgorithm::InDegreesAll, execute_in_degrees_all),
    (crate::plan::GraphAlgorithm::OutDegreesAll, execute_out_degrees_all),
    (crate::plan::GraphAlgorithm::WeaklyConnectedComponents, execute_weakly_connected_components),
    (crate::plan::GraphAlgorithm::StronglyConnectedComponents, execute_strongly_connected_components),
    (crate::plan::GraphAlgorithm::TriangleCountAll, execute_triangle_count_all),
    (crate::plan::GraphAlgorithm::CoreNumber, execute_core_number),
    (crate::plan::GraphAlgorithm::KCore, execute_k_core),
    (crate::plan::GraphAlgorithm::MinimumSpanningTree, execute_minimum_spanning_tree),
    (crate::plan::GraphAlgorithm::Leiden, execute_leiden),
    (crate::plan::GraphAlgorithm::Ecg, execute_ecg),
    (crate::plan::GraphAlgorithm::SpectralModularityMaximization, execute_spectral_modularity_maximization),
    (crate::plan::GraphAlgorithm::ForceAtlas2, execute_force_atlas2),
];

fn graph_algorithm_executor(algorithm: crate::plan::GraphAlgorithm) -> Result<GraphAlgorithmExecutor> {
    GRAPH_ALGORITHM_EXECUTORS
        .iter()
        .find(|(registered, _)| *registered == algorithm)
        .map(|(_, executor)| *executor)
        .ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.graph_algorithm.executor_lookup",
                "graph algorithm executor table is missing a registered variant",
            )
            .with_fact("algorithm", algorithm.as_str())
        })
}
