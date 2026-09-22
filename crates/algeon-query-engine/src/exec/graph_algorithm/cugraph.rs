use std::collections::BTreeMap;

use crate::{
    exec::{
        edge_null_filter::{EdgeNullFilterScope, filter_null_edge_rows},
        nvtx::{NvtxRange, graph_build_range},
        *,
    },
    graph_observation::GraphMeasuredPeakProvenance,
    metrics::GraphAlgorithmRuntimeMetrics,
    plan::{GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmSpec},
};

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
#[path = "validation.rs"]
mod validation;

use bounded_betweenness::{
    RuntimeGraphSpecResolution, resolve_runtime_graph_spec_for_execution,
    runtime_graph_spec_resolution_after_bounded_seed_validation, validate_betweenness_static_seed_policy,
};
use build::*;
use centrality::*;
use community::*;
use diagnostics::*;
use empty_input::*;
use graph_builder::*;
use layout::*;
use params::*;
use similarity::*;
use structure::*;
use traversal::{execute_bfs, execute_sssp};
use validation::*;

pub(in crate::exec) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    execute_prepared_graph_algorithm(context, node_id, node, input, side_inputs, metrics)
}

pub(in crate::exec) fn graph_empty_input_prevalidated(
    context: &ExecutionContext,
    node: &GraphAlgorithmNode,
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
    node: &GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let input_graph_residency = input.graph_residency().cloned();
    let prior_null_edge_rows_dropped = metrics.null_edge_rows_dropped(node.input);
    let null_filter_scope = EdgeNullFilterScope {
        weight: node.spec.algorithm.consumes_weight(),
        edge_id: node.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input = filter_null_edge_rows(
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
    if !empty_input_prevalidated {
        metrics.record_graph_claim_calibration(node_id, node.spec.algorithm);
    }
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
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        empty_input_prevalidated,
        metrics,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuDF/cuGraph/cuVS call parameters stay explicit at the FFI boundary"
)]
pub(in crate::exec) fn execute_graph_algorithm_sanitized(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &GraphAlgorithmNode,
    runtime_spec: RuntimeGraphSpecResolution,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    input_rows: u64,
    input_columns: u64,
    null_edge_rows_dropped: u64,
    empty_input_prevalidated: bool,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let boundary_before = metrics.boundary();
    let timer = Instant::now();
    let result = if empty_input_prevalidated {
        resolve_empty_graph_input(context, node, input.frame(), &side_inputs, input_rows, input.num_rows() as u64)?
            .expect("prevalidated empty graph input must resolve to an empty result")
    } else {
        let runtime_spec = match runtime_spec {
            RuntimeGraphSpecResolution::Unresolved => {
                resolve_runtime_graph_spec_for_execution(input.frame(), &node.spec)?
            },
            runtime_spec => runtime_spec,
        };
        let spec = runtime_spec.spec_or(&node.spec);
        let executor = graph_algorithm_executor(spec.algorithm)?;
        let inputs = GraphAlgorithmExecutionInputs { edge_list: input, side_inputs };
        // This is the first call that can enter the native graph algorithm.
        // Earlier staging and runtime-spec resolution must not classify an
        // attempt as physical graph execution.
        metrics.record_graph_claim_calibration_algorithm_started(node_id, spec.algorithm);
        match executor(context, spec, inputs) {
            Ok(result) => result,
            Err(error) => {
                record_graph_claim_peak_on_error(
                    metrics,
                    node_id,
                    node.spec.algorithm,
                    error.graph_cache_scope_peak_bytes(),
                );
                return Err(error);
            },
        }
    };
    let elapsed = timer.elapsed();
    metrics.add_operator_time(elapsed);

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
    let (measured_graph_peak_bytes, measured_peak_provenance) = graph_peak_measurement(result.cache_scope_peak_bytes);
    let graph_peak_bytes = measured_graph_peak_bytes;
    metrics.record_graph_claim_calibration_measurement(
        node_id,
        node.spec.algorithm,
        measured_graph_peak_bytes,
        measured_peak_provenance,
    );
    let boundary_delta = boundary_delta(&boundary_before, &metrics.boundary());
    metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
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

#[cfg(test)]
fn max_graph_peak(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(peak), None) | (None, Some(peak)) => Some(peak),
        (None, None) => None,
    }
}

fn graph_peak_measurement(cache_scope_high_water_bytes: Option<u64>) -> (Option<u64>, GraphMeasuredPeakProvenance) {
    match cache_scope_high_water_bytes {
        Some(peak) => (Some(peak), GraphMeasuredPeakProvenance::CacheScopeHighWater),
        None => (None, GraphMeasuredPeakProvenance::Unavailable),
    }
}

fn record_graph_claim_peak_on_error(
    metrics: &ExecutionMetrics,
    node_id: NodeId,
    algorithm: GraphAlgorithm,
    cache_scope_peak_bytes: Option<u64>,
) {
    let (measured_peak_bytes, measured_peak_provenance) = graph_peak_measurement(cache_scope_peak_bytes);
    metrics.record_graph_claim_calibration_measurement(
        node_id,
        algorithm,
        measured_peak_bytes,
        measured_peak_provenance,
    );
}

// Common return envelope for algorithm-specific cuGraph executors.
pub(super) struct GraphAlgorithmExecutionResult {
    pub(super) frame: DataFrame,
    pub(super) algorithm_metadata: String,
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
}

type GraphAlgorithmExecutor =
    fn(&ExecutionContext, &GraphAlgorithmSpec, GraphAlgorithmExecutionInputs) -> Result<GraphAlgorithmExecutionResult>;

const GRAPH_ALGORITHM_EXECUTORS: &[(GraphAlgorithm, GraphAlgorithmExecutor)] = &[
    (GraphAlgorithm::PageRank, execute_pagerank),
    (GraphAlgorithm::PersonalizedPageRank, execute_personalized_pagerank),
    (GraphAlgorithm::Bfs, execute_bfs),
    (GraphAlgorithm::Louvain, execute_louvain),
    (GraphAlgorithm::EigenvectorCentrality, execute_eigenvector_centrality),
    (GraphAlgorithm::KatzCentrality, execute_katz_centrality),
    (GraphAlgorithm::Hits, execute_hits),
    (GraphAlgorithm::BetweennessCentrality, execute_betweenness_centrality),
    (GraphAlgorithm::EdgeBetweennessCentrality, execute_edge_betweenness_centrality),
    (GraphAlgorithm::Sssp, execute_sssp),
    (GraphAlgorithm::DegreesAll, execute_degrees_all),
    (GraphAlgorithm::InDegreesAll, execute_in_degrees_all),
    (GraphAlgorithm::OutDegreesAll, execute_out_degrees_all),
    (GraphAlgorithm::WeaklyConnectedComponents, execute_weakly_connected_components),
    (GraphAlgorithm::StronglyConnectedComponents, execute_strongly_connected_components),
    (GraphAlgorithm::TriangleCountAll, execute_triangle_count_all),
    (GraphAlgorithm::CoreNumber, execute_core_number),
    (GraphAlgorithm::KCore, execute_k_core),
    (GraphAlgorithm::MinimumSpanningTree, execute_minimum_spanning_tree),
    (GraphAlgorithm::Leiden, execute_leiden),
    (GraphAlgorithm::Ecg, execute_ecg),
    (GraphAlgorithm::SpectralModularityMaximization, execute_spectral_modularity_maximization),
    (GraphAlgorithm::ForceAtlas2, execute_force_atlas2),
    (GraphAlgorithm::Jaccard, execute_similarity),
    (GraphAlgorithm::Sorensen, execute_similarity),
    (GraphAlgorithm::Overlap, execute_similarity),
    (GraphAlgorithm::Cosine, execute_similarity),
];

fn graph_algorithm_executor(algorithm: GraphAlgorithm) -> Result<GraphAlgorithmExecutor> {
    GRAPH_ALGORITHM_EXECUTORS
        .iter()
        .find(|(registered, _)| *registered == algorithm)
        .map(|(_, executor)| *executor)
        .ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.graph_algorithm.executor_lookup",
                "graph algorithm executor table is missing a registered variant",
            )
            .with_fact("algorithm", algorithm.as_str())
        })
}

#[cfg(test)]
mod graph_peak_measurement_tests {
    use super::{graph_peak_measurement, max_graph_peak, record_graph_claim_peak_on_error};
    use crate::{
        exec::ExecutionMetrics,
        graph_observation::GraphMeasuredPeakProvenance,
        plan::{GraphAlgorithm, NodeId},
    };

    #[test]
    fn cache_scope_peak_is_the_direct_attributed_measurement() {
        assert_eq!(
            graph_peak_measurement(Some(2_048)),
            (Some(2_048), GraphMeasuredPeakProvenance::CacheScopeHighWater)
        );
    }

    #[test]
    fn graph_peak_is_unavailable_without_an_attributed_scope() {
        assert_eq!(graph_peak_measurement(None), (None, GraphMeasuredPeakProvenance::Unavailable));
    }

    #[test]
    fn cache_scope_error_and_retry_peaks_keep_the_larger_direct_observation() {
        assert_eq!(max_graph_peak(Some(2_048), Some(4_096)), Some(4_096));
        assert_eq!(max_graph_peak(None, Some(4_096)), Some(4_096));
    }

    #[test]
    fn graph_claim_error_records_the_measured_cache_scope_peak() {
        let node_id = NodeId::from_index_for_internal_use(7);
        let metrics = ExecutionMetrics::detached("graph-claim-error");
        metrics.record_graph_claim_calibration(node_id, GraphAlgorithm::PageRank);

        record_graph_claim_peak_on_error(&metrics, node_id, GraphAlgorithm::PageRank, Some(1_536));

        let metrics = metrics.snapshot();
        let row = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(row.measured_peak_bytes, Some(1_536));
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::CacheScopeHighWater);
        assert_eq!(metrics.max_task_observed_peak_bytes, Some(1_536));
    }

    #[test]
    fn graph_claim_error_without_any_measurement_stays_unavailable() {
        let node_id = NodeId::from_index_for_internal_use(8);
        let metrics = ExecutionMetrics::detached("graph-claim-incomplete-error");
        metrics.record_graph_claim_calibration(node_id, GraphAlgorithm::PageRank);

        record_graph_claim_peak_on_error(&metrics, node_id, GraphAlgorithm::PageRank, None);

        let metrics = metrics.snapshot();
        let row = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(row.measured_peak_bytes, None);
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
    }
}
