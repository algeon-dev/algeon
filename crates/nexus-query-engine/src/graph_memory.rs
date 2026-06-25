//! Approximate cuGraph memory pricing outside the strict bounded gate.
//!
//! These estimates are intentionally not strict-bounded contracts. They price a
//! graph fragment before entering cuGraph so bounded service admission can fail
//! locally with structured facts instead of relying on a raw RMM OOM.

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    metrics::{PlanNodeId, QueryId},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy, GraphEdgeColumnBindings,
    },
    runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
};

/// Stable graph memory-tier boundary used in reports and reservation facts.
pub const GRAPH_MEMORY_BOUNDARY: &str = "graph_memory";

/// Temporary bounded-admission probe that measures distinct graph endpoints.
pub const GRAPH_VERTEX_ESTIMATE_BOUNDARY: &str = "graph_memory_vertex_estimate";

/// Stable local rejection reason for graph estimate denial.
pub const GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON: &str = "graph_memory_estimate_exceeds_budget";

/// Minimum live reservation used when a graph fragment is hard-limited by RMM.
///
/// Tiny model estimates are useful diagnostics, but cuGraph still needs a
/// practical allocator envelope for graph construction and result conversion.
/// Bounded admission reserves this floor so the RMM cap never exceeds the
/// live graph reservation.
pub const GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES: u64 = 1024 * 1024;

/// cuGraph's single-GPU vertex-betweenness implementation tags concurrent
/// sources with a `uint16_t` origin id. The requested seed count can be much
/// larger, but those sources are not all resident in one source-by-vertex
/// workspace at the same time.
const BETWEENNESS_MAX_CONCURRENT_SOURCES: u64 = u16::MAX as u64;

/// Conservative class assigned to a cuGraph algorithm for approximate pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMemoryClass {
    /// Output-sized algorithms with `O(V + E)` storage.
    OutputSized,
    /// Multi-array algorithms with extra per-level or ensemble state.
    MultiArray,
    /// Workspace-heavy algorithms with data-dependent allocation shape.
    WorkspaceHeavyDataDependent,
    /// Workspace-heavy algorithms whose memory is primarily seed-count driven.
    WorkspaceHeavySeedDependent,
}

impl GraphMemoryClass {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutputSized => "output_sized",
            Self::MultiArray => "multi_array",
            Self::WorkspaceHeavyDataDependent => "workspace_heavy_data_dependent",
            Self::WorkspaceHeavySeedDependent => "workspace_heavy_seed_dependent",
        }
    }

    const fn is_data_dependent(self) -> bool {
        matches!(self, Self::WorkspaceHeavyDataDependent | Self::WorkspaceHeavySeedDependent)
    }
}

impl std::fmt::Display for GraphMemoryClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of graph memory-tier admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphReservationOutcome {
    /// No bounded budget was active, so the estimate was diagnostic-only.
    NotBounded,
    /// Estimate fit the current device budget.
    Admitted,
    /// Estimate exceeded the current device budget.
    Denied,
}

impl GraphReservationOutcome {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotBounded => "not_bounded",
            Self::Admitted => "admitted",
            Self::Denied => "denied",
        }
    }
}

impl std::fmt::Display for GraphReservationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provenance for the vertex-count signal used by graph memory pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphVertexEstimateSource {
    /// The model fell back to `2 * input_edges`.
    EdgeEndpointUpperBound,
    /// Exact distinct endpoint count measured from the runtime edge frame.
    RuntimeExactDistinct,
    /// Source-level statistics supplied a trusted vertex count.
    SourceStatistics,
    /// Graph dataset metadata supplied a trusted vertex count.
    GraphMetadata,
}

impl GraphVertexEstimateSource {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EdgeEndpointUpperBound => "edge_endpoint_upper_bound",
            Self::RuntimeExactDistinct => "runtime_exact_distinct",
            Self::SourceStatistics => "source_statistics",
            Self::GraphMetadata => "graph_metadata",
        }
    }
}

impl std::fmt::Display for GraphVertexEstimateSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Refined vertex-count evidence for graph memory pricing.
///
/// Non-runtime callers must pass an exact or safety-padded count; admission uses
/// this value directly after clamping it to the endpoint upper bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphVertexEstimate {
    /// Vertex-count signal before clamping to the endpoint upper bound.
    pub count: u64,
    /// Provenance for the count.
    pub source: GraphVertexEstimateSource,
}

impl GraphVertexEstimate {
    /// Exact endpoint count measured from the runtime edge-list frame.
    pub const fn runtime_exact_distinct(count: u64) -> Self {
        Self { count, source: GraphVertexEstimateSource::RuntimeExactDistinct }
    }

    /// Trusted vertex count supplied by source-level statistics.
    pub const fn source_statistics(count: u64) -> Self {
        Self { count, source: GraphVertexEstimateSource::SourceStatistics }
    }

    /// Trusted vertex count supplied by graph dataset metadata.
    pub const fn graph_metadata(count: u64) -> Self {
        Self { count, source: GraphVertexEstimateSource::GraphMetadata }
    }
}

/// Plan/runtime inputs used by the approximate graph memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryModelInput {
    /// Edge-row count after graph edge filtering/normalization.
    pub input_edges: u64,
    /// Width in bytes of the vertex id type.
    pub vertex_width_bytes: u64,
    /// Width in bytes of the optional weight column consumed by the algorithm.
    pub weight_width_bytes: Option<u64>,
    /// Whether the graph has a stable edge-id column.
    pub has_edge_id: bool,
    /// Total bytes reported by the CUDA device. This lets seed-dependent
    /// algorithms mirror cuGraph's physical-memory source batching heuristic.
    pub device_total_memory_bytes: Option<u64>,
    /// Better vertex-count signal when available.
    pub vertex_estimate: Option<GraphVertexEstimate>,
}

impl GraphMemoryModelInput {
    /// Conservative upper bound on vertex count: every edge contributes two
    /// distinct endpoints.
    pub fn vertex_upper_bound(self) -> u64 {
        self.input_edges.saturating_mul(2)
    }

    /// Attach refined vertex-count evidence.
    pub fn with_vertex_estimate(mut self, estimate: GraphVertexEstimate) -> Self {
        self.vertex_estimate = Some(estimate);
        self
    }

    /// Attach physical device-memory evidence for algorithms whose cuGraph
    /// implementation chooses resident batch width from total GPU memory.
    pub fn with_device_total_memory_bytes(mut self, total_bytes: u64) -> Self {
        self.device_total_memory_bytes = Some(total_bytes);
        self
    }
}

/// Approximate graph memory estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryEstimate {
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub seed_count: Option<u64>,
    pub data_dependent: bool,
    pub requested_bytes: u64,
}

/// Public graph memory-tier report row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReservationReport {
    pub query_id: Option<QueryId>,
    pub node_id: String,
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub outcome: GraphReservationOutcome,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub seed_count: Option<u64>,
    pub data_dependent: bool,
    pub requested_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub reason: Option<&'static str>,
}

impl GraphReservationReport {
    /// Builds a report from an estimate and optional bounded budget.
    pub fn from_estimate(node_id: impl ToString, estimate: GraphMemoryEstimate, budget_bytes: Option<u64>) -> Self {
        let requested_bytes = if budget_bytes.is_some() {
            estimate.requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
        } else {
            estimate.requested_bytes
        };
        let denied = budget_bytes.is_some_and(|budget| requested_bytes > budget);
        let outcome = match (budget_bytes, denied) {
            (None, _) => GraphReservationOutcome::NotBounded,
            (Some(_), true) => GraphReservationOutcome::Denied,
            (Some(_), false) => GraphReservationOutcome::Admitted,
        };
        Self {
            query_id: None,
            node_id: node_id.to_string(),
            algorithm: estimate.algorithm,
            class: estimate.class,
            outcome,
            input_edges: estimate.input_edges,
            vertex_upper_bound: estimate.vertex_upper_bound,
            vertex_estimate: estimate.vertex_estimate,
            vertex_estimate_source: estimate.vertex_estimate_source,
            vertex_width_bytes: estimate.vertex_width_bytes,
            weight_width_bytes: estimate.weight_width_bytes,
            has_edge_id: estimate.has_edge_id,
            directed: estimate.directed,
            renumber: estimate.renumber,
            construction_policy: estimate.construction_policy,
            seed_count: estimate.seed_count,
            data_dependent: estimate.data_dependent,
            requested_bytes,
            budget_bytes,
            reason: denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON),
        }
    }

    /// Builds the structured local admission error for a denied graph fragment.
    pub fn denied_error(&self, request: &MemoryReservationRequest, granted_bytes: u64) -> QueryError {
        let mut error = request
            .denied_error(granted_bytes)
            .with_fact("reason", GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON)
            .with_fact("algorithm", self.algorithm.as_str())
            .with_fact("graph_memory_class", self.class.as_str())
            .with_fact("input_edges", self.input_edges)
            .with_fact("vertex_upper_bound", self.vertex_upper_bound)
            .with_fact("vertex_estimate", self.vertex_estimate)
            .with_fact("vertex_estimate_source", self.vertex_estimate_source.as_str())
            .with_fact("vertex_width_bytes", self.vertex_width_bytes)
            .with_fact("directed", self.directed)
            .with_fact("renumber", self.renumber)
            .with_fact("construction_policy", self.construction_policy.as_str())
            .with_fact("data_dependent", self.data_dependent);
        if let Some(seed_count) = self.seed_count {
            error = error.with_fact("seed_count", seed_count);
        }
        error
    }
}

/// Approximate model for the graph memory tier.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphMemoryModel;

impl GraphMemoryModel {
    /// Estimate the graph fragment memory for an algorithm and graph input.
    pub fn estimate(&self, spec: &GraphAlgorithmSpec, input: GraphMemoryModelInput) -> Result<GraphMemoryEstimate> {
        if input.vertex_width_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.estimate",
                "graph memory model requires a positive vertex dtype width",
            )
            .with_fact("vertex_width_bytes", input.vertex_width_bytes));
        }

        let class = graph_memory_class(spec.algorithm);
        let vertex_upper_bound = input.vertex_upper_bound();
        let (vertex_estimate, vertex_estimate_source) = input
            .vertex_estimate
            .map(|estimate| (estimate.count.min(vertex_upper_bound), estimate.source))
            .unwrap_or((vertex_upper_bound, GraphVertexEstimateSource::EdgeEndpointUpperBound));
        let edge_row_width = input
            .vertex_width_bytes
            .saturating_mul(2)
            .saturating_add(input.weight_width_bytes.unwrap_or(0))
            .saturating_add(if input.has_edge_id { input.vertex_width_bytes } else { 0 });
        let construction_multiplier = construction_multiplier(spec.graph_options.policy, spec.graph_options.directed);
        let construction_bytes = saturating_product(input.input_edges, edge_row_width)
            .saturating_mul(construction_multiplier)
            .saturating_add(if spec.graph_options.renumber {
                saturating_product(vertex_estimate, input.vertex_width_bytes.saturating_mul(3))
            } else {
                0
            });
        let output_bytes = output_bytes_for_algorithm(spec.algorithm, input, vertex_estimate);
        let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
        let seed_count = seed_input.map(|seed_input| seed_input.requested_count);
        let workspace_bytes = workspace_bytes_for_algorithm(spec.algorithm, class, input, vertex_estimate, seed_input);
        let requested_bytes = construction_bytes.saturating_add(output_bytes).saturating_add(workspace_bytes).max(1);

        Ok(GraphMemoryEstimate {
            algorithm: spec.algorithm,
            class,
            input_edges: input.input_edges,
            vertex_upper_bound,
            vertex_estimate,
            vertex_estimate_source,
            vertex_width_bytes: input.vertex_width_bytes,
            weight_width_bytes: input.weight_width_bytes,
            has_edge_id: input.has_edge_id,
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            construction_policy: spec.graph_options.policy,
            seed_count,
            data_dependent: class.is_data_dependent(),
            requested_bytes,
        })
    }

    /// Build the reservation request used by graph-tier admission.
    pub fn reservation_request(
        &self,
        node_id: impl ToString,
        report: &GraphReservationReport,
        memory_space: MemorySpaceId,
    ) -> Option<MemoryReservationRequest> {
        let budget_bytes = report.budget_bytes?;
        Some(
            MemoryReservationRequest::new(ReservationKind::GraphAlgorithm, report.requested_bytes, budget_bytes)
                .with_memory_space(memory_space)
                .with_memory_boundary(GRAPH_MEMORY_BOUNDARY)
                .with_node_id(PlanNodeId::new(node_id.to_string()))
                .with_stage("graph_memory_admission"),
        )
    }
}

/// Return the memory class for a graph algorithm.
pub const fn graph_memory_class(algorithm: GraphAlgorithm) -> GraphMemoryClass {
    match algorithm {
        GraphAlgorithm::PageRank
        | GraphAlgorithm::PersonalizedPageRank
        | GraphAlgorithm::EigenvectorCentrality
        | GraphAlgorithm::KatzCentrality
        | GraphAlgorithm::Hits
        | GraphAlgorithm::Bfs
        | GraphAlgorithm::Sssp
        | GraphAlgorithm::DegreesAll
        | GraphAlgorithm::InDegreesAll
        | GraphAlgorithm::OutDegreesAll
        | GraphAlgorithm::WeaklyConnectedComponents
        | GraphAlgorithm::CoreNumber
        | GraphAlgorithm::MinimumSpanningTree => GraphMemoryClass::OutputSized,
        GraphAlgorithm::Louvain
        | GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::KCore
        | GraphAlgorithm::StronglyConnectedComponents => GraphMemoryClass::MultiArray,
        GraphAlgorithm::TriangleCountAll
        | GraphAlgorithm::SpectralModularityMaximization
        | GraphAlgorithm::BalancedCutClustering
        | GraphAlgorithm::ForceAtlas2 => GraphMemoryClass::WorkspaceHeavyDataDependent,
        GraphAlgorithm::BetweennessCentrality | GraphAlgorithm::EdgeBetweennessCentrality => {
            GraphMemoryClass::WorkspaceHeavySeedDependent
        },
    }
}

/// Whether bounded graph execution must reserve an algorithm workspace tier.
///
/// Algorithms stay on the graph-workspace path until their cuGraph execution is
/// audited as table/view-only over already-accounted resident inputs.
pub const fn graph_memory_reservation_required(algorithm: GraphAlgorithm) -> bool {
    match algorithm {
        GraphAlgorithm::Bfs => false,
        GraphAlgorithm::PageRank
        | GraphAlgorithm::PersonalizedPageRank
        | GraphAlgorithm::Louvain
        | GraphAlgorithm::EigenvectorCentrality
        | GraphAlgorithm::KatzCentrality
        | GraphAlgorithm::Hits
        | GraphAlgorithm::BetweennessCentrality
        | GraphAlgorithm::EdgeBetweennessCentrality
        | GraphAlgorithm::Sssp
        | GraphAlgorithm::DegreesAll
        | GraphAlgorithm::InDegreesAll
        | GraphAlgorithm::OutDegreesAll
        | GraphAlgorithm::WeaklyConnectedComponents
        | GraphAlgorithm::StronglyConnectedComponents
        | GraphAlgorithm::TriangleCountAll
        | GraphAlgorithm::CoreNumber
        | GraphAlgorithm::KCore
        | GraphAlgorithm::MinimumSpanningTree
        | GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::SpectralModularityMaximization
        | GraphAlgorithm::BalancedCutClustering
        | GraphAlgorithm::ForceAtlas2 => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphSeedMemoryInput {
    requested_count: u64,
    explicit_seed_list: bool,
    requires_runtime_seed_selection: bool,
}

fn graph_seed_memory_input(params: &GraphAlgorithmParams, vertex_estimate: u64) -> Option<GraphSeedMemoryInput> {
    match params {
        GraphAlgorithmParams::BetweennessCentrality(params) => {
            Some(betweenness_seed_memory_input(params.seeds.as_ref(), params.k, vertex_estimate))
        },
        GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            Some(betweenness_seed_memory_input(params.seeds.as_ref(), params.k, vertex_estimate))
        },
        _ => None,
    }
}

fn betweenness_seed_memory_input(
    seeds: Option<&crate::plan::GraphBetweennessSeeds>,
    k: Option<u64>,
    vertex_estimate: u64,
) -> GraphSeedMemoryInput {
    match seeds {
        Some(seeds) => GraphSeedMemoryInput {
            requested_count: seeds.len() as u64,
            explicit_seed_list: true,
            requires_runtime_seed_selection: false,
        },
        None => GraphSeedMemoryInput {
            requested_count: k.unwrap_or(vertex_estimate),
            explicit_seed_list: k.is_some(),
            requires_runtime_seed_selection: k.is_some(),
        },
    }
}

fn construction_multiplier(policy: GraphConstructionPolicy, directed: bool) -> u64 {
    match (policy, directed) {
        (GraphConstructionPolicy::PythonCugraph, false) => 3,
        (GraphConstructionPolicy::PythonCugraph, true) => 2,
        (GraphConstructionPolicy::RawLibcugraph, _) => 1,
    }
}

fn output_bytes_for_algorithm(algorithm: GraphAlgorithm, input: GraphMemoryModelInput, vertex_estimate: u64) -> u64 {
    match algorithm {
        GraphAlgorithm::EdgeBetweennessCentrality => {
            input.input_edges.saturating_mul(edge_betweenness_output_row_width(input))
        },
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => input.input_edges.saturating_mul(16),
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll => vertex_estimate.saturating_mul(24),
        GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => vertex_estimate.saturating_mul(24),
        GraphAlgorithm::ForceAtlas2 => vertex_estimate.saturating_mul(16),
        _ => vertex_estimate.saturating_mul(16),
    }
}

fn edge_betweenness_output_row_width(input: GraphMemoryModelInput) -> u64 {
    input.vertex_width_bytes.max(8).saturating_mul(2).saturating_add(8).saturating_add(if input.has_edge_id {
        input.vertex_width_bytes.max(8)
    } else {
        0
    })
}

fn workspace_bytes_for_algorithm(
    algorithm: GraphAlgorithm,
    class: GraphMemoryClass,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> u64 {
    let vertex_array = vertex_estimate.saturating_mul(input.vertex_width_bytes.max(8));
    let edge_index = input.input_edges.saturating_mul(input.vertex_width_bytes.saturating_mul(2));
    let output_sized = vertex_array.saturating_mul(3).saturating_add(edge_index);
    match class {
        GraphMemoryClass::OutputSized => output_sized,
        GraphMemoryClass::MultiArray => output_sized.saturating_mul(4),
        GraphMemoryClass::WorkspaceHeavyDataDependent => {
            let superlinear_probe = vertex_estimate.saturating_mul(vertex_estimate.min(1024)).saturating_mul(4);
            output_sized.saturating_mul(8).saturating_add(superlinear_probe)
        },
        GraphMemoryClass::WorkspaceHeavySeedDependent => match algorithm {
            GraphAlgorithm::BetweennessCentrality => {
                output_sized.saturating_add(vertex_betweenness_workspace_bytes(input, vertex_estimate, seed_input))
            },
            GraphAlgorithm::EdgeBetweennessCentrality => {
                output_sized.saturating_add(edge_betweenness_workspace_bytes(input, vertex_estimate, seed_input))
            },
            _ => output_sized,
        },
    }
}

fn vertex_betweenness_workspace_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> u64 {
    let requested_sources = seed_input.map_or(vertex_estimate, |seed_input| seed_input.requested_count);
    let concurrent_sources = vertex_betweenness_concurrent_source_count(input, vertex_estimate, requested_sources);
    let source_state_bytes = concurrent_sources
        .saturating_mul(vertex_estimate)
        .saturating_mul(vertex_betweenness_per_source_vertex_state_width(input));
    source_state_bytes
        .saturating_add(explicit_seed_list_bytes(input, seed_input))
        .saturating_add(runtime_seed_selection_bytes(input, vertex_estimate, seed_input))
}

fn edge_betweenness_workspace_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> u64 {
    vertex_estimate
        .saturating_mul(edge_betweenness_per_source_vertex_state_width(input))
        .saturating_add(explicit_seed_list_bytes(input, seed_input))
        .saturating_add(runtime_seed_selection_bytes(input, vertex_estimate, seed_input))
}

fn explicit_seed_list_bytes(input: GraphMemoryModelInput, seed_input: Option<GraphSeedMemoryInput>) -> u64 {
    let Some(seed_input) = seed_input else {
        return 0;
    };
    if seed_input.explicit_seed_list { seed_input.requested_count.saturating_mul(input.vertex_width_bytes) } else { 0 }
}

fn runtime_seed_selection_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> u64 {
    let Some(seed_input) = seed_input else {
        return 0;
    };
    if !seed_input.requires_runtime_seed_selection {
        return 0;
    }
    let vertex_width = input.vertex_width_bytes.max(8);
    let endpoint_copy_and_concat = input.input_edges.saturating_mul(vertex_width).saturating_mul(4);
    let distinct_sort_buffers = vertex_estimate.saturating_mul(vertex_width).saturating_mul(3);
    endpoint_copy_and_concat.saturating_add(distinct_sort_buffers)
}

fn vertex_betweenness_concurrent_source_count(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    requested_sources: u64,
) -> u64 {
    let mut sources = requested_sources.clamp(1, BETWEENNESS_MAX_CONCURRENT_SOURCES);
    let Some(total_memory_bytes) = input.device_total_memory_bytes else {
        return sources;
    };
    if sources <= 1 || vertex_estimate == 0 {
        return sources;
    }

    let bfs_result_bytes_per_source =
        vertex_estimate.saturating_mul(input.vertex_width_bytes.saturating_add(input.vertex_width_bytes));
    if bfs_result_bytes_per_source == 0 {
        return sources;
    }

    let physical_batch_cap = total_memory_bytes / 4 / bfs_result_bytes_per_source;
    sources = sources.min(physical_batch_cap).max(1);
    sources
}

fn vertex_betweenness_per_source_vertex_state_width(input: GraphMemoryModelInput) -> u64 {
    let vertex_width = input.vertex_width_bytes.max(8);
    let edge_width = input.vertex_width_bytes.max(8);
    let weight_width = input.weight_width_bytes.unwrap_or(8).max(8);
    let origin_width = 2;

    // Mirrors the live arrays in cuGraph's single-GPU multisource Brandes path:
    // distances, sigmas, deltas, and distance-level vertex/source buckets plus
    // their sorted copies. This is intentionally a resident-state shape, not a
    // total `requested_sources * vertices` traversal-work counter.
    vertex_width
        .saturating_add(edge_width)
        .saturating_add(weight_width)
        .saturating_add(vertex_width.saturating_mul(2))
        .saturating_add(origin_width * 2)
}

fn edge_betweenness_per_source_vertex_state_width(input: GraphMemoryModelInput) -> u64 {
    let vertex_width = input.vertex_width_bytes.max(8);
    let edge_width = input.vertex_width_bytes.max(8);
    let weight_width = input.weight_width_bytes.unwrap_or(8).max(8);

    // Edge betweenness walks requested sources serially in cuGraph, so resident
    // vertex state is one Brandes source at a time.
    vertex_width.saturating_add(edge_width).saturating_add(weight_width)
}

fn saturating_product(left: u64, right: u64) -> u64 {
    left.saturating_mul(right)
}

/// Derive model inputs from runtime-observed graph edge metadata.
pub fn graph_memory_model_input(
    input_edges: u64,
    vertex_width_bytes: u64,
    edge_columns: &GraphEdgeColumnBindings,
    consumes_weight: bool,
    weight_width_bytes: Option<u64>,
) -> GraphMemoryModelInput {
    GraphMemoryModelInput {
        input_edges,
        vertex_width_bytes,
        weight_width_bytes: consumes_weight.then_some(weight_width_bytes).flatten(),
        has_edge_id: edge_columns.edge_id.is_some(),
        device_total_memory_bytes: None,
        vertex_estimate: None,
    }
}
