use std::collections::BTreeMap;
#[cfg(all(test, feature = "cugraph"))]
use std::sync::atomic::Ordering;

use super::super::*;

pub(in crate::exec) mod bounded_betweenness;
#[path = "build.rs"]
pub(in crate::exec) mod build;
#[path = "centrality.rs"]
mod centrality;
#[path = "community.rs"]
mod community;
#[path = "diagnostics.rs"]
pub(in crate::exec) mod diagnostics;
#[path = "empty_input.rs"]
mod empty_input;
#[path = "graph_builder.rs"]
pub(in crate::exec) mod graph_builder;
#[path = "layout.rs"]
mod layout;
#[path = "memory.rs"]
pub(in crate::exec) mod memory;
#[path = "params.rs"]
mod params;
#[path = "similarity.rs"]
mod similarity;
#[path = "statistics.rs"]
pub(in crate::exec) mod statistics;
#[path = "structure.rs"]
mod structure;
#[path = "traversal.rs"]
mod traversal;

use bounded_betweenness::{
    RuntimeGraphSpecResolution, execute_bounded_betweenness_seed_chunk_retry,
    graph_retry_memory_observation_from_error, graph_seed_chunk_retry_batches,
    resolve_runtime_graph_spec_for_execution, runtime_graph_spec_resolution_after_bounded_seed_validation,
    validate_betweenness_static_seed_policy,
};
#[cfg(any(test, feature = "test-hooks"))]
pub use bounded_betweenness::{
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
};
use build::*;
use centrality::*;
use community::*;
use diagnostics::*;
use empty_input::*;
use graph_builder::*;
use layout::*;
use memory::*;
use params::*;
use similarity::*;
#[cfg(any(test, feature = "test-hooks"))]
pub use statistics::force_next_graph_statistics_optional_resource_exhausted_for_test;
use structure::*;
use traversal::{execute_bfs, execute_sssp};

use super::super::nvtx::{NvtxRange, graph_build_range};

pub(in crate::exec) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_prepared_graph_algorithm(context, node_id, node, input, side_inputs, metrics)
}

pub(in crate::exec) fn graph_empty_input_prevalidated(
    context: &ExecutionContext,
    node: &crate::plan::GraphAlgorithmNode,
    sanitized_edges: &DataFrame,
    side_inputs: &BTreeMap<NodeId, GpuDataFrame>,
    input_rows: u64,
    sanitized_input_rows: u64,
) -> Result<bool> {
    resolve_empty_graph_input(context, node, sanitized_edges, side_inputs, input_rows, sanitized_input_rows)
        .map(|result| result.is_some())
}

fn execute_prepared_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
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
    let empty_input_prevalidated =
        graph_empty_input_prevalidated(context, node, &filtered_input, &side_inputs, input_rows, sanitized_input_rows)?;
    // The descriptor is created only from a complete Local Parquet edge
    // rowset. Filtering null edges can only remove rows, so its footer-backed
    // vertex bound remains a conservative admission upper bound.
    let vertex_estimate = source_graph_vertex_estimate(input_graph_residency.as_ref());
    let raw_input_edges = source_graph_raw_edge_row_count(input_graph_residency.as_ref());
    let graph_memory_limit_bytes = if empty_input_prevalidated {
        None
    } else {
        admit_graph_memory(
            context,
            node_id,
            &node.spec,
            &filtered_input,
            &side_inputs,
            sanitized_input_rows,
            vertex_estimate,
            raw_input_edges,
            metrics,
        )?
    };
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
        empty_input_prevalidated,
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
    empty_input_prevalidated: bool,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let boundary_before = metrics.boundary.clone();
    let pool_usage_before = graph_pool_usage_snapshot(context);
    let timer = Instant::now();
    let result = if empty_input_prevalidated {
        resolve_empty_graph_input(context, node, input.frame(), &side_inputs, input_rows, input.num_rows() as u64)?
            .expect("prevalidated empty graph input must resolve to an empty result")
    } else {
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
        let seed_chunk_retry_inputs = graph_seed_chunk_retry_batches(spec).map(|batches| {
            (
                GraphAlgorithmExecutionInputs {
                    edge_list: input.clone(),
                    side_inputs: side_inputs.clone(),
                    graph_memory_limit_bytes,
                },
                batches,
            )
        });
        let inputs = GraphAlgorithmExecutionInputs { edge_list: input, side_inputs, graph_memory_limit_bytes };
        // This is the first call that can enter the native graph algorithm.
        // Earlier staging, graph-memory pricing, and runtime-spec resolution
        // must not classify an attempt as physical graph execution.
        metrics.record_graph_claim_calibration_algorithm_started(node_id, spec.algorithm);
        match execute_graph_algorithm_attempt(context, spec, inputs, executor) {
            Ok(result) => result,
            Err(error) if should_run_bounded_seed_chunk_retry(context, &error, seed_chunk_retry_inputs.as_ref()) => {
                let first_cache_scope_peak_bytes = error.graph_cache_scope_peak_bytes();
                let Some((retry_inputs, seed_batches)) = seed_chunk_retry_inputs else {
                    record_graph_claim_peak_on_error(
                        metrics,
                        node_id,
                        node.spec.algorithm,
                        &error,
                        first_cache_scope_peak_bytes,
                        pool_usage_before,
                        graph_pool_usage_snapshot(context),
                    );
                    return Err(error);
                };
                record_graph_budget_observation_from_error(metrics, &error);
                let retry_result =
                    execute_bounded_betweenness_seed_chunk_retry(context, spec, retry_inputs, seed_batches, executor);
                match retry_result {
                    Ok(mut result) => {
                        result.cache_scope_peak_bytes =
                            max_graph_peak(first_cache_scope_peak_bytes, result.cache_scope_peak_bytes);
                        result
                    },
                    Err(error) => {
                        record_graph_claim_peak_on_error(
                            metrics,
                            node_id,
                            node.spec.algorithm,
                            &error,
                            max_graph_peak(first_cache_scope_peak_bytes, error.graph_cache_scope_peak_bytes()),
                            pool_usage_before,
                            graph_pool_usage_snapshot(context),
                        );
                        return Err(error);
                    },
                }
            },
            Err(error) => {
                record_graph_claim_peak_on_error(
                    metrics,
                    node_id,
                    node.spec.algorithm,
                    &error,
                    error.graph_cache_scope_peak_bytes(),
                    pool_usage_before,
                    graph_pool_usage_snapshot(context),
                );
                return Err(error);
            },
        }
    };
    let pool_usage_after = graph_pool_usage_snapshot(context);
    let elapsed = timer.elapsed();
    metrics.operator_time += elapsed;
    for observation in &result.budget_observations {
        observation.record(metrics);
    }

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
    let reservation_graph_peak_bytes =
        result.budget_observations.iter().filter_map(GraphBudgetObservation::allocator_high_water_bytes).max();
    let pool_graph_peak_bytes = graph_pool_high_water_delta(pool_usage_before, pool_usage_after);
    let (measured_graph_peak_bytes, measured_peak_provenance) = graph_claim_peak_measurement(
        reservation_graph_peak_bytes,
        result.cache_scope_peak_bytes,
        pool_graph_peak_bytes,
    );
    let graph_peak_bytes = measured_graph_peak_bytes.or(Some(0));
    metrics.record_graph_claim_calibration_measurement(
        node_id,
        node.spec.algorithm,
        measured_graph_peak_bytes,
        measured_peak_provenance,
    );
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
        elapsed_ns: elapsed.as_nanos() as u64,
        gpu_resident_input_handoffs: boundary_delta.gpu_resident_input_handoffs,
        host_to_device_uploads: boundary_delta.host_to_device_uploads,
        device_to_host_materializations: boundary_delta.device_to_host_materializations,
    });
    Ok(gpu)
}

fn max_graph_peak(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(peak), None) | (None, Some(peak)) => Some(peak),
        (None, None) => None,
    }
}

fn graph_claim_peak_measurement(
    reservation_high_water_bytes: Option<u64>,
    cache_scope_high_water_bytes: Option<u64>,
    pool_delta_bytes: Option<u64>,
) -> (Option<u64>, crate::graph_memory::GraphMeasuredPeakProvenance) {
    match (reservation_high_water_bytes, cache_scope_high_water_bytes, pool_delta_bytes) {
        (Some(peak), _, _) => (Some(peak), crate::graph_memory::GraphMeasuredPeakProvenance::ReservationHighWater),
        (None, Some(peak), _) => (Some(peak), crate::graph_memory::GraphMeasuredPeakProvenance::CacheScopeHighWater),
        (None, None, Some(peak)) => (Some(peak), crate::graph_memory::GraphMeasuredPeakProvenance::PoolDelta),
        (None, None, None) => (None, crate::graph_memory::GraphMeasuredPeakProvenance::Unavailable),
    }
}

fn record_graph_claim_peak_on_error(
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    algorithm: crate::plan::GraphAlgorithm,
    error: &QueryError,
    cache_scope_peak_bytes: Option<u64>,
    pool_usage_before: Option<cudf::DeviceMemoryPoolUsage>,
    pool_usage_after: Option<cudf::DeviceMemoryPoolUsage>,
) {
    record_graph_budget_observation_from_error(metrics, error);
    let (measured_peak_bytes, measured_peak_provenance) = graph_claim_peak_measurement(
        graph_budget_peak_from_error(error),
        cache_scope_peak_bytes,
        graph_pool_high_water_delta(pool_usage_before, pool_usage_after),
    );
    metrics.record_graph_claim_calibration_measurement(
        node_id,
        algorithm,
        measured_peak_bytes,
        measured_peak_provenance,
    );
}

fn execute_graph_algorithm_attempt(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
    executor: GraphAlgorithmExecutor,
) -> Result<GraphAlgorithmExecutionResult> {
    #[cfg(any(test, feature = "test-hooks"))]
    if graph_seed_chunk_retry_batches(spec).is_some()
        && bounded_betweenness::FORCE_NEXT_GRAPH_SEED_CHUNK_RETRY_FULL_ATTEMPT_RESOURCE_EXHAUSTED
            .swap(false, Ordering::SeqCst)
    {
        let mut error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.graph_algorithm.bounded_seed_chunk_retry_test",
            "forced graph bounded seed-chunk retry resource exhaustion",
        )
        .with_fact("reason", "forced_graph_bounded_seed_chunk_retry_resource_exhausted_test")
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

fn should_run_bounded_seed_chunk_retry(
    _context: &ExecutionContext,
    error: &QueryError,
    retry_inputs: Option<&(GraphAlgorithmExecutionInputs, Vec<crate::plan::GraphBetweennessSeeds>)>,
) -> bool {
    let Some((inputs, _)) = retry_inputs else {
        return false;
    };
    error.code() == QueryErrorCode::CudfResourceExhausted
        && inputs.graph_memory_limit_bytes.is_some()
        && graph_retry_memory_observation_from_error(error).is_some()
}

// Common return envelope for algorithm-specific cuGraph executors.
pub(super) struct GraphAlgorithmExecutionResult {
    pub(super) frame: DataFrame,
    pub(super) algorithm_metadata: String,
    pub(super) budget_observations: Vec<GraphBudgetObservation>,
    /// Exact high-water bytes from the exclusive resident-cache allocation
    /// scope, when this physical execution used a resident graph runtime.
    /// It is intentionally absent for ordinary attempt-runtime executions.
    pub(super) cache_scope_peak_bytes: Option<u64>,
}

// Shared graph inputs after edge-list preparation and side-input materialization.
#[derive(Clone)]
pub(super) struct GraphAlgorithmExecutionInputs {
    pub(super) edge_list: GpuDataFrame,
    pub(super) side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    pub(super) graph_memory_limit_bytes: Option<u64>,
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
    (crate::plan::GraphAlgorithm::Jaccard, execute_similarity),
    (crate::plan::GraphAlgorithm::Sorensen, execute_similarity),
    (crate::plan::GraphAlgorithm::Overlap, execute_similarity),
    (crate::plan::GraphAlgorithm::Cosine, execute_similarity),
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

#[cfg(test)]
mod graph_claim_peak_measurement_tests {
    use super::{graph_claim_peak_measurement, max_graph_peak, record_graph_claim_peak_on_error};
    use crate::{
        error::{QueryError, QueryErrorCode},
        graph_memory::{GraphClaimCalibrationRow, GraphClaimExecutionState, GraphMeasuredPeakProvenance},
        metrics::QueryMetrics,
        plan::{GraphAlgorithm, NodeId},
    };

    #[test]
    fn cache_scope_peak_is_the_direct_fallback_before_attempt_pool_delta() {
        assert_eq!(
            graph_claim_peak_measurement(None, Some(2_048), Some(4_096)),
            (Some(2_048), GraphMeasuredPeakProvenance::CacheScopeHighWater)
        );
    }

    #[test]
    fn reservation_peak_remains_preferred_over_cache_scope_peak() {
        assert_eq!(
            graph_claim_peak_measurement(Some(1_024), Some(2_048), Some(4_096)),
            (Some(1_024), GraphMeasuredPeakProvenance::ReservationHighWater)
        );
    }

    #[test]
    fn cache_scope_error_and_retry_peaks_keep_the_larger_direct_observation() {
        assert_eq!(max_graph_peak(Some(2_048), Some(4_096)), Some(4_096));
        assert_eq!(max_graph_peak(None, Some(4_096)), Some(4_096));
    }

    #[test]
    fn graph_claim_error_records_the_typed_reservation_high_water() {
        let node_id = NodeId::from_index_for_internal_use(7);
        let mut metrics = QueryMetrics::new("graph-claim-error");
        metrics.graph_claim_calibration_rows.push(GraphClaimCalibrationRow {
            query_id: None,
            node_id: node_id.to_string(),
            node_index: node_id.index(),
            algorithm: GraphAlgorithm::PageRank,
            model_requested_bytes: 4_096,
            requested_bytes: 2_048,
            budget_bytes: Some(2_048),
            execution_state: GraphClaimExecutionState::AlgorithmStarted,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
            graph_model_error_ratio: None,
        });
        let error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.graph_algorithm.test",
            "reservation-backed graph execution exhausted memory",
        )
        .with_fact("graph_budget_operation", "nexus_query_engine.graph_algorithm.pagerank")
        .with_fact("graph_budget_enforcement", "stream_bound_reservation")
        .with_fact("graph_budget_limit_bytes", 2_048)
        .with_fact("graph_budget_status", "reservation_resource_exhausted")
        .with_fact("graph_budget_peak_bytes", 1_536);

        record_graph_claim_peak_on_error(&mut metrics, node_id, GraphAlgorithm::PageRank, &error, None, None, None);

        let row = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(row.measured_peak_bytes, Some(1_536));
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::ReservationHighWater);
        assert_eq!(metrics.max_task_observed_peak_bytes, Some(1_536));
    }

    #[test]
    fn graph_claim_error_rejects_an_incomplete_reservation_projection() {
        let node_id = NodeId::from_index_for_internal_use(8);
        let mut metrics = QueryMetrics::new("graph-claim-incomplete-error");
        metrics.graph_claim_calibration_rows.push(GraphClaimCalibrationRow {
            query_id: None,
            node_id: node_id.to_string(),
            node_index: node_id.index(),
            algorithm: GraphAlgorithm::PageRank,
            model_requested_bytes: 4_096,
            requested_bytes: 2_048,
            budget_bytes: Some(2_048),
            execution_state: GraphClaimExecutionState::AlgorithmStarted,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
            graph_model_error_ratio: None,
        });
        let error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.graph_algorithm.test",
            "graph execution returned an incomplete reservation projection",
        )
        .with_fact("graph_budget_operation", "nexus_query_engine.graph_algorithm.pagerank")
        .with_fact("graph_budget_limit_bytes", 2_048)
        .with_fact("graph_budget_peak_bytes", 1_536);

        record_graph_claim_peak_on_error(&mut metrics, node_id, GraphAlgorithm::PageRank, &error, None, None, None);

        let row = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(row.measured_peak_bytes, None);
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
    }
}
