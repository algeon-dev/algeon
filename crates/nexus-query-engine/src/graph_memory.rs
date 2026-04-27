//! Approximate cuGraph memory pricing outside the strict bounded gate.
//!
//! These estimates are intentionally not strict-bounded contracts. They price a
//! graph fragment before entering cuGraph so bounded service admission can fail
//! locally with structured facts instead of relying on a raw RMM OOM.

use arrow::datatypes::{DataType, SchemaRef};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    metrics::{PlanNodeId, QueryId},
    plan::{
        EdgeNormalizeSpec, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy,
        GraphEdgeColumnBindings, NodeId, PlanNode, QueryPlan, estimated_output_rows,
    },
    runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
};

/// Stable graph memory-tier boundary used in reports and reservation facts.
pub const GRAPH_MEMORY_BOUNDARY: &str = "graph_memory";

/// Temporary bounded-admission probe that measures distinct graph endpoints.
pub const GRAPH_VERTEX_ESTIMATE_BOUNDARY: &str = "graph_memory_vertex_estimate";

/// Minimum exact-vertex probe reservation.
///
/// The edge-scaled model dominates for real graph inputs. This floor handles
/// tiny graphs where RMM allocation alignment and cudf distinct intermediates
/// exceed the byte-exact column copy size.
pub const GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES: u64 = 64 * 1024;

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

// The constants below mirror allocation shapes in the pinned cuGraph checkout;
// `graph_memory_pricing_constants_match_pinned_cugraph_sources` fails loudly
// when the mirrored upstream code drifts.

/// `sizeof(origin_t)` in cuGraph's resident push tuple
/// `(vertex_t, origin_t, edge_t)` (`betweenness_centrality_impl.cuh`); the two
/// vertex-width components are added per-call, this is only the middle field.
const BETWEENNESS_SOURCE_ORIGIN_WIDTH_BYTES: u64 = 2;
/// cuGraph's `max_edge_tuple_data_ratio = 0.25`: multi-source BFS batches its
/// frontier so resident push tuples never exceed a quarter of total device
/// memory. Pricing above this ceiling would charge for an allocation
/// libcugraph cannot make.
const BETWEENNESS_PUSH_TUPLE_DEVICE_MEMORY_DENOMINATOR: u64 = 4;
/// After 2-core pruning, cuGraph extracts the low-to-high-degree oriented edge
/// list and rebuilds a graph from it (`extract_transform_if_e` →
/// `create_graph_from_edgelist`): one src plus one dst column per DFG edge.
const TRIANGLE_COUNT_DFG_REBUILD_VERTEX_COLUMNS: u64 = 2;
/// Endpoint (src, dst) pair list that
/// `transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v` iterates;
/// priced over all DFG edges at once as the unchunked upper bound.
const TRIANGLE_COUNT_PAIR_LIST_VERTEX_COLUMNS: u64 = 2;
/// `nbr_intersection` allocates its intersection index buffer from per-pair
/// min degrees ("upper bound for intersection sizes") and keeps a same-order
/// companion live across compaction — two buffers that scale with the chunk's
/// min-degree sum rather than with E.
const TRIANGLE_COUNT_PEAK_INTERSECTION_VERTEX_COLUMNS: u64 = 2;
/// Envelope for the V-sized arrays concurrently live across the 2-core /
/// renumber / count phases of `triangle_count_impl.cuh` (`core_numbers`,
/// `renumber_map`, flag and count arrays).
const TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS: u64 = 4;
/// Graph construction sorts/renumbers the edge list holding a staged copy of
/// both endpoint columns. This transient is what the construction term alone
/// under-covered on the DBLP calibration point.
const TRIANGLE_COUNT_SORT_STAGING_VERTEX_COLUMNS: u64 = 2;
/// Statistics-absent fallback envelope, deliberately LINEAR in edge count so
/// the plan-time path (which never sees runtime statistics) cannot collapse
/// back into the superlinear `V*min(V,1024)` whole-device ask, while still
/// dominating the measured DBLP pool peak with wide margin. Pinned on both
/// sides by `graph_memory_model_keeps_triangle_no_stats_fallback_topology_independent`
/// (floor: ask >= measured peak) and
/// `graph_memory_model_keeps_triangle_plan_time_fallback_below_whole_device_ticket`
/// (ceiling: ticket stays well under device capacity).
const TRIANGLE_COUNT_NO_STATS_EDGE_VERTEX_COLUMNS: u64 = 24;
/// Numerator of the 5/4 peak-chunk safety margin: cuGraph chunks intersection
/// work by dense renumbered dst id, which cannot be reproduced from sparse
/// external ids, so pricing assumes peak chunk = average chunk (min-degree
/// sum / chunk count) times this margin. On skewed degree distributions the
/// true peak can exceed the margin; that residual under-ask is deliberately
/// left to the graph stream lease, which contains it as a per-query retryable
/// allocation failure instead of shared-pool pressure.
const TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR: u64 = 5;
/// Denominator of the 5/4 peak-chunk safety margin described on
/// [`TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR`].
const TRIANGLE_COUNT_CHUNK_SAFETY_DENOMINATOR: u64 = 4;

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
    /// Explicit-pair similarity with one output row per pair relation row.
    PairOutputSized,
}

impl GraphMemoryClass {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutputSized => "output_sized",
            Self::MultiArray => "multi_array",
            Self::WorkspaceHeavyDataDependent => "workspace_heavy_data_dependent",
            Self::WorkspaceHeavySeedDependent => "workspace_heavy_seed_dependent",
            Self::PairOutputSized => "pair_output_sized",
        }
    }

    const fn is_data_dependent(self) -> bool {
        matches!(self, Self::WorkspaceHeavyDataDependent | Self::WorkspaceHeavySeedDependent)
    }
}

/// Provenance for cardinality evidence used by explicit-pair similarity memory
/// admission.
///
/// A similarity call cannot reuse graph vertex or edge cardinality as a proxy:
/// one pair relation row produces one output row and the relation is a
/// multiset.  The source is therefore retained on the reservation report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPairCardinalitySource {
    /// A deterministic literal native-plan input supplied the exact row count.
    PlanExact,
    /// An explicit plan-level upper bound constrained the pair relation.
    PlanHardUpperBound,
    /// The side input was materialized before graph admission.
    RuntimeMaterialized,
}

impl GraphPairCardinalitySource {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanExact => "plan_exact",
            Self::PlanHardUpperBound => "plan_hard_upper_bound",
            Self::RuntimeMaterialized => "runtime_materialized",
        }
    }

    /// Returns whether the evidence is admissible for strict bounded pricing.
    pub const fn is_strict_bounded_evidence(self) -> bool {
        matches!(self, Self::PlanExact | Self::PlanHardUpperBound)
    }
}

impl std::fmt::Display for GraphPairCardinalitySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Admission contract that consumed explicit-pair cardinality evidence.
///
/// A runtime-materialized count is valid for adaptive bounded execution, but
/// never proves a strict pre-admission bound.  Preserve that distinction on
/// the estimate and its report projection rather than inferring it from a
/// later execution mode label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPairCardinalityAdmissionMode {
    /// No bounded admission was active; the row records observed execution
    /// evidence only.
    Unbounded,
    /// Strict bounded admission used a plan-exact count or hard upper bound.
    StrictBounded,
    /// Adaptive bounded admission used materialized/preflight evidence.
    AdaptiveBounded,
}

impl GraphPairCardinalityAdmissionMode {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::StrictBounded => "strict_bounded",
            Self::AdaptiveBounded => "adaptive_bounded",
        }
    }
}

impl std::fmt::Display for GraphPairCardinalityAdmissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Display for GraphMemoryClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Optional externally supplied graph calibration metadata.
///
/// Defaults intentionally carry no calibration. Callers should only provide
/// these values from fixture metadata or a documented calibration run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GraphMemoryCalibrationOverride {
    /// Raw edge rows before explicit edge normalization, when known.
    pub raw_input_edges: Option<u64>,
}

impl GraphMemoryCalibrationOverride {
    /// Returns an override with no calibration metadata.
    pub const fn none() -> Self {
        Self { raw_input_edges: None }
    }

    /// Attaches raw pre-normalization edge-row evidence.
    pub const fn with_raw_input_edges(mut self, raw_input_edges: u64) -> Self {
        self.raw_input_edges = Some(raw_input_edges);
        self
    }

    /// Applies this override to model input.
    pub fn apply_to_input(self, mut input: GraphMemoryModelInput) -> GraphMemoryModelInput {
        if input.raw_input_edges.is_none()
            && let Some(raw_input_edges) = self.raw_input_edges
        {
            input = input.with_raw_input_edges(raw_input_edges);
        }
        input
    }
}

/// Source-selection mode for seed-dependent graph algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphSeedMode {
    /// Caller supplied a concrete seed list.
    ExplicitSeedList,
    /// Runtime must select `k` sources from graph vertices.
    RuntimeKSelection,
    /// Algorithm executes over every vertex as a source.
    AllVertices,
}

impl GraphSeedMode {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitSeedList => "explicit_seed_list",
            Self::RuntimeKSelection => "runtime_k_selection",
            Self::AllVertices => "all_vertices",
        }
    }
}

impl std::fmt::Display for GraphSeedMode {
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

/// Provenance for non-default graph calibration inputs visible on reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GraphCalibrationSource {
    /// No explicit graph calibration key was selected.
    #[default]
    None,
    /// Raw pre-normalization edge-count evidence was available.
    RawInputEdges,
}

impl GraphCalibrationSource {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::RawInputEdges => "raw_input_edges",
        }
    }
}

impl std::fmt::Display for GraphCalibrationSource {
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

/// Runtime-derived topology evidence collected from the resident graph edge frame.
///
/// Some fields are direct measurements and some are conservative pricing terms
/// derived from measured degree distributions. Fields that are not relevant to
/// the probed algorithm are left unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphStatistics {
    /// Exact distinct endpoint count measured from the runtime edge-list frame.
    pub exact_vertices: u64,
    /// Maximum source out-degree when degree statistics were requested.
    pub degree_max: Option<u64>,
    /// 99th percentile source out-degree when degree statistics were requested.
    pub degree_p99: Option<u64>,
    /// Sum of oriented min-degree terms used by triangle-count workspace pricing.
    pub sum_min_degree_oriented: Option<u64>,
    /// Edge count in cuGraph's low-degree-to-high-degree oriented triangle DFG.
    pub oriented_edge_count: Option<u64>,
    /// Runtime-derived peak chunk min-degree pricing term used by triangle-count workspace pricing.
    pub peak_chunk_min_degree_sum: Option<u64>,
    /// Sum of resident source out-degrees used by betweenness push batching.
    pub sum_out_degree: Option<u64>,
}

impl GraphStatistics {
    /// Start a statistics row from exact runtime vertex evidence.
    pub const fn runtime_exact_vertices(exact_vertices: u64) -> Self {
        Self {
            exact_vertices,
            degree_max: None,
            degree_p99: None,
            sum_min_degree_oriented: None,
            oriented_edge_count: None,
            peak_chunk_min_degree_sum: None,
            sum_out_degree: None,
        }
    }

    /// Attach source out-degree distribution evidence.
    pub const fn with_degree_distribution(mut self, degree_max: u64, degree_p99: u64) -> Self {
        self.degree_max = Some(degree_max);
        self.degree_p99 = Some(degree_p99);
        self
    }

    /// Attach betweenness push-batching evidence.
    pub const fn with_sum_out_degree(mut self, sum_out_degree: u64) -> Self {
        self.sum_out_degree = Some(sum_out_degree);
        self
    }

    /// Attach triangle-count oriented min-degree evidence and derived chunk pricing term.
    pub const fn with_triangle_min_degree(
        mut self,
        sum_min_degree_oriented: u64,
        oriented_edge_count: u64,
        peak_chunk_min_degree_sum: u64,
    ) -> Self {
        self.sum_min_degree_oriented = Some(sum_min_degree_oriented);
        self.oriented_edge_count = Some(oriented_edge_count);
        self.peak_chunk_min_degree_sum = Some(peak_chunk_min_degree_sum);
        self
    }
}

/// Plan/runtime inputs used by the approximate graph memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryModelInput {
    /// Raw edge rows before any explicit edge-normalize operator, when known.
    pub raw_input_edges: Option<u64>,
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
    /// Whether an edge-normalize operator was inserted before graph execution.
    pub edge_normalize_inserted: bool,
    /// Runtime topology evidence used by data-dependent pricing, when available.
    pub statistics: Option<GraphStatistics>,
    /// Exact or hard-bounded candidate-pair rows for a similarity operation.
    pub pair_rows: Option<u64>,
    /// Width in bytes of each vertex column in the candidate-pair relation.
    pub pair_vertex_width_bytes: Option<u64>,
    /// Provenance for [`Self::pair_rows`].
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
}

impl GraphMemoryModelInput {
    /// Conservative upper bound on vertex count: every edge contributes two
    /// distinct endpoints.
    pub fn vertex_upper_bound(self) -> u64 {
        self.input_edges.saturating_mul(2)
    }

    /// Raw edge count when known, otherwise the effective graph-stage count.
    pub fn raw_edge_count(self) -> u64 {
        self.raw_input_edges.unwrap_or(self.input_edges)
    }

    /// Attach raw pre-normalization edge-row evidence.
    pub fn with_raw_input_edges(mut self, raw_input_edges: u64) -> Self {
        self.raw_input_edges = Some(raw_input_edges);
        self
    }

    /// Attach refined vertex-count evidence.
    pub fn with_vertex_estimate(mut self, estimate: GraphVertexEstimate) -> Self {
        self.vertex_estimate = Some(estimate);
        self
    }

    /// Attach runtime graph statistics used by data-dependent pricing.
    pub fn with_statistics(mut self, statistics: GraphStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Attach physical device-memory evidence for algorithms whose cuGraph
    /// implementation chooses resident batch width from total GPU memory.
    pub fn with_device_total_memory_bytes(mut self, total_bytes: u64) -> Self {
        self.device_total_memory_bytes = Some(total_bytes);
        self
    }

    /// Mark the input as having passed through explicit edge normalization.
    pub fn with_edge_normalize_inserted(mut self, inserted: bool) -> Self {
        self.edge_normalize_inserted = inserted;
        self
    }

    /// Attach explicit-pair cardinality and dtype evidence for a similarity
    /// operation.
    pub fn with_similarity_pair_input(
        mut self,
        pair_rows: u64,
        pair_vertex_width_bytes: u64,
        pair_cardinality_source: GraphPairCardinalitySource,
    ) -> Self {
        self.pair_rows = Some(pair_rows);
        self.pair_vertex_width_bytes = Some(pair_vertex_width_bytes);
        self.pair_cardinality_source = Some(pair_cardinality_source);
        self
    }
}

/// Approximate graph memory estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryEstimate {
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub raw_input_edges: Option<u64>,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub pair_rows: Option<u64>,
    pub pair_vertex_width_bytes: Option<u64>,
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
    /// Admission contract that consumed [`Self::pair_cardinality_source`].
    pub pair_cardinality_admission_mode: Option<GraphPairCardinalityAdmissionMode>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub edge_normalize_inserted: bool,
    pub calibration_source: GraphCalibrationSource,
    pub seed_count: Option<u64>,
    pub seed_mode: Option<GraphSeedMode>,
    pub data_dependent: bool,
    /// Bytes needed to construct and retain the native graph representation.
    ///
    /// Warm cache admission may omit this term only when query-service
    /// admission has reserved an exact-runtime resident graph entry.
    pub construction_bytes: u64,
    /// Algorithm output and workspace bytes that remain necessary on both
    /// cold and warm executions.
    pub execution_bytes: u64,
    /// Cold execution price: construction plus output and workspace.
    pub requested_bytes: u64,
}

/// Planning-time graph memory estimate for a node in a native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPlanMemoryEstimate {
    pub node_id: NodeId,
    pub estimate: GraphMemoryEstimate,
}

/// Fixed-point `ask / measured_peak` ratio for graph model calibration drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphModelErrorRatio {
    milli_ratio: u64,
}

impl GraphModelErrorRatio {
    /// Build a ratio from model-requested bytes and measured graph peak bytes.
    pub fn from_requested_and_measured(requested_bytes: u64, measured_peak_bytes: u64) -> Option<Self> {
        if measured_peak_bytes == 0 {
            return None;
        }
        let numerator = u128::from(requested_bytes).saturating_mul(1000);
        let rounded = numerator.saturating_add(u128::from(measured_peak_bytes / 2)) / u128::from(measured_peak_bytes);
        Some(Self { milli_ratio: rounded.min(u128::from(u64::MAX)) as u64 })
    }

    /// Stable decimal string with three fractional digits.
    pub fn as_decimal_string(self) -> String {
        format!("{}.{:03}", self.milli_ratio / 1000, self.milli_ratio % 1000)
    }
}

impl std::fmt::Display for GraphModelErrorRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_decimal_string())
    }
}

/// Public graph memory-tier report row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReservationReport {
    pub query_id: Option<QueryId>,
    pub node_id: String,
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub outcome: GraphReservationOutcome,
    pub raw_input_edges: Option<u64>,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub pair_rows: Option<u64>,
    pub pair_vertex_width_bytes: Option<u64>,
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
    /// Admission contract that consumed explicit-pair cardinality evidence.
    pub pair_cardinality_admission_mode: Option<GraphPairCardinalityAdmissionMode>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub edge_normalize_inserted: bool,
    pub calibration_source: GraphCalibrationSource,
    pub seed_count: Option<u64>,
    pub seed_mode: Option<GraphSeedMode>,
    pub data_dependent: bool,
    pub model_requested_bytes: u64,
    pub requested_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub graph_model_error_ratio: Option<GraphModelErrorRatio>,
    /// Measured graph-stage allocator high-water bytes, when runtime evidence
    /// was available.  This is the denominator of [`Self::graph_model_error_ratio`].
    pub measured_peak_bytes: Option<u64>,
    pub statistics: Option<GraphStatistics>,
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
            raw_input_edges: estimate.raw_input_edges,
            input_edges: estimate.input_edges,
            vertex_upper_bound: estimate.vertex_upper_bound,
            vertex_estimate: estimate.vertex_estimate,
            vertex_estimate_source: estimate.vertex_estimate_source,
            vertex_width_bytes: estimate.vertex_width_bytes,
            weight_width_bytes: estimate.weight_width_bytes,
            pair_rows: estimate.pair_rows,
            pair_vertex_width_bytes: estimate.pair_vertex_width_bytes,
            pair_cardinality_source: estimate.pair_cardinality_source,
            pair_cardinality_admission_mode: estimate.pair_cardinality_admission_mode,
            has_edge_id: estimate.has_edge_id,
            directed: estimate.directed,
            renumber: estimate.renumber,
            construction_policy: estimate.construction_policy,
            edge_normalize_inserted: estimate.edge_normalize_inserted,
            calibration_source: estimate.calibration_source,
            seed_count: estimate.seed_count,
            seed_mode: estimate.seed_mode,
            data_dependent: estimate.data_dependent,
            model_requested_bytes: estimate.requested_bytes,
            requested_bytes,
            budget_bytes,
            graph_model_error_ratio: None,
            measured_peak_bytes: None,
            statistics: None,
            reason: denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON),
        }
    }

    /// Converts an isolated similarity model price into the incremental
    /// reservation required while its explicit pair relation is already live.
    ///
    /// The model intentionally includes the pair columns because it is also
    /// used for unbounded and planning-time estimates. Bounded execution must
    /// reserve only the additional graph work when those exact bytes already
    /// consume the same query grant. Credit no more than the modeled pair
    /// overlap so a wider source frame cannot reduce unrelated graph work.
    pub fn credit_live_similarity_pair_input(mut self, already_accounted_bytes: u64) -> Self {
        let Some(budget_bytes) = self.budget_bytes else {
            return self;
        };
        if !self.algorithm.is_similarity() {
            return self;
        }
        let pair_input_overlap_bytes = self
            .pair_rows
            .zip(self.pair_vertex_width_bytes)
            .map(|(rows, width)| rows.saturating_mul(width.saturating_mul(2)))
            .unwrap_or(0);
        let credit_bytes = pair_input_overlap_bytes.min(already_accounted_bytes);
        if credit_bytes == 0 {
            return self;
        }

        self.requested_bytes =
            self.requested_bytes.saturating_sub(credit_bytes).max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
        let denied = self.requested_bytes > budget_bytes;
        self.outcome = if denied { GraphReservationOutcome::Denied } else { GraphReservationOutcome::Admitted };
        self.reason = denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON);
        self
    }

    /// Attach runtime graph statistics evidence to this row.
    pub fn with_statistics(mut self, statistics: GraphStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Attach the observed model error ratio for a completed graph execution.
    pub fn record_model_error_ratio(&mut self, measured_peak_bytes: u64) {
        self.measured_peak_bytes = Some(measured_peak_bytes);
        self.graph_model_error_ratio =
            GraphModelErrorRatio::from_requested_and_measured(self.model_requested_bytes, measured_peak_bytes);
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
            .with_fact("pair_rows", self.pair_rows.unwrap_or(0))
            .with_fact(
                "pair_cardinality_source",
                self.pair_cardinality_source.map_or("none", GraphPairCardinalitySource::as_str),
            )
            .with_fact("directed", self.directed)
            .with_fact("renumber", self.renumber)
            .with_fact("construction_policy", self.construction_policy.as_str())
            .with_fact("edge_normalize_inserted", self.edge_normalize_inserted)
            .with_fact("calibration_source", self.calibration_source.as_str())
            .with_fact("data_dependent", self.data_dependent);
        if let Some(raw_input_edges) = self.raw_input_edges {
            error = error.with_fact("raw_input_edges", raw_input_edges);
        }
        if let Some(seed_count) = self.seed_count {
            error = error.with_fact("seed_count", seed_count);
        }
        if let Some(seed_mode) = self.seed_mode {
            error = error.with_fact("seed_mode", seed_mode.as_str());
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
        self.estimate_with_pair_cardinality_admission_mode(spec, input, GraphPairCardinalityAdmissionMode::Unbounded)
    }

    /// Estimate a graph fragment for strict bounded admission.
    ///
    /// Similarity requires plan-exact or hard-upper-bound pair cardinality
    /// evidence here.  A count obtained by executing the pair relation is
    /// deliberately rejected, because it is adaptive evidence and cannot be
    /// retroactively promoted to a strict admission proof.
    pub fn estimate_for_strict_bounded(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
    ) -> Result<GraphMemoryEstimate> {
        require_strict_similarity_pair_cardinality_evidence(spec, input)?;
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::StrictBounded,
        )
    }

    /// Estimate a graph fragment for adaptive bounded admission.
    ///
    /// The adaptive path may consume a pair relation after its own
    /// materialization boundary.  Its report is labelled separately so that
    /// runtime evidence is never presented as a strict bound.
    pub fn estimate_for_adaptive_bounded(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
    ) -> Result<GraphMemoryEstimate> {
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::AdaptiveBounded,
        )
    }

    fn estimate_with_pair_cardinality_admission_mode(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
        pair_cardinality_admission_mode: GraphPairCardinalityAdmissionMode,
    ) -> Result<GraphMemoryEstimate> {
        if input.vertex_width_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.estimate",
                "graph memory model requires a positive vertex dtype width",
            )
            .with_fact("vertex_width_bytes", input.vertex_width_bytes));
        }

        let class = graph_memory_class(spec.algorithm);
        let (pair_rows, pair_vertex_width_bytes, pair_cardinality_source) =
            similarity_pair_memory_evidence(spec, input)?;
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
        let normalized_edge_bytes = saturating_product(input.input_edges, edge_row_width);
        let normalization_transient_bytes = if input.edge_normalize_inserted { normalized_edge_bytes } else { 0 };
        let construction_bytes = normalized_edge_bytes
            .saturating_mul(construction_multiplier)
            .saturating_add(normalization_transient_bytes)
            .saturating_add(if spec.graph_options.renumber {
                saturating_product(vertex_estimate, input.vertex_width_bytes.saturating_mul(3))
            } else {
                0
            });
        let output_bytes = output_bytes_for_algorithm(spec.algorithm, input, vertex_estimate, pair_rows);
        let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
        let seed_count = seed_input.map(|seed_input| seed_input.requested_count);
        let seed_mode = seed_input.map(|seed_input| seed_input.mode);
        let workspace_bytes = workspace_bytes_for_algorithm(
            spec.algorithm,
            class,
            input,
            vertex_estimate,
            seed_input,
            pair_rows,
            pair_vertex_width_bytes,
        );
        let execution_bytes = output_bytes.saturating_add(workspace_bytes).max(1);
        let requested_bytes = construction_bytes.saturating_add(output_bytes).saturating_add(workspace_bytes).max(1);
        let calibration_source = graph_calibration_source(input);

        Ok(GraphMemoryEstimate {
            algorithm: spec.algorithm,
            class,
            raw_input_edges: input.raw_input_edges,
            input_edges: input.input_edges,
            vertex_upper_bound,
            vertex_estimate,
            vertex_estimate_source,
            vertex_width_bytes: input.vertex_width_bytes,
            weight_width_bytes: input.weight_width_bytes,
            pair_rows,
            pair_vertex_width_bytes,
            pair_cardinality_source,
            pair_cardinality_admission_mode: spec.algorithm.is_similarity().then_some(pair_cardinality_admission_mode),
            has_edge_id: input.has_edge_id,
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            construction_policy: spec.graph_options.policy,
            edge_normalize_inserted: input.edge_normalize_inserted,
            calibration_source,
            seed_count,
            seed_mode,
            data_dependent: class.is_data_dependent(),
            construction_bytes,
            execution_bytes,
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
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            GraphMemoryClass::PairOutputSized
        },
        GraphAlgorithm::Louvain
        | GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::KCore
        | GraphAlgorithm::StronglyConnectedComponents => GraphMemoryClass::MultiArray,
        GraphAlgorithm::TriangleCountAll
        | GraphAlgorithm::SpectralModularityMaximization
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
        | GraphAlgorithm::ForceAtlas2
        | GraphAlgorithm::Jaccard
        | GraphAlgorithm::Sorensen
        | GraphAlgorithm::Overlap
        | GraphAlgorithm::Cosine => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphSeedMemoryInput {
    requested_count: u64,
    mode: GraphSeedMode,
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
            mode: GraphSeedMode::ExplicitSeedList,
            explicit_seed_list: true,
            requires_runtime_seed_selection: false,
        },
        None => GraphSeedMemoryInput {
            requested_count: k.unwrap_or(vertex_estimate),
            mode: if k.is_some() { GraphSeedMode::RuntimeKSelection } else { GraphSeedMode::AllVertices },
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

fn output_bytes_for_algorithm(
    algorithm: GraphAlgorithm,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
) -> u64 {
    match algorithm {
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            pair_rows.unwrap_or(0).saturating_mul(24)
        },
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
    pair_rows: Option<u64>,
    pair_vertex_width_bytes: Option<u64>,
) -> u64 {
    let vertex_array = vertex_estimate.saturating_mul(input.vertex_width_bytes.max(8));
    // Graph construction keeps copied endpoints live while sorting/renumbering
    // through an additional per-edge vertex-width staging vector.
    let edge_index_workspace = input.input_edges.saturating_mul(input.vertex_width_bytes.saturating_mul(3));
    let output_sized = vertex_array.saturating_mul(3).saturating_add(edge_index_workspace);
    match class {
        GraphMemoryClass::OutputSized => output_sized,
        GraphMemoryClass::MultiArray => output_sized.saturating_mul(4),
        GraphMemoryClass::WorkspaceHeavyDataDependent => match algorithm {
            GraphAlgorithm::TriangleCountAll => triangle_count_workspace_bytes(input, vertex_estimate),
            _ => legacy_workspace_heavy_data_dependent_bytes(vertex_estimate, output_sized),
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
        GraphMemoryClass::PairOutputSized => similarity_workspace_bytes(
            pair_rows.expect("similarity memory evidence validated before workspace pricing"),
            pair_vertex_width_bytes.expect("similarity memory evidence validated before workspace pricing"),
            input.weight_width_bytes.is_some(),
        ),
    }
}

/// Fixed cuGraph allocator floor retained during pairwise similarity
/// coefficient dispatch. The linear term is calibrated separately for the
/// selected vertex width and weighted path: weighted coefficients retain
/// additional per-pair temporary state in libcugraph. Input overlap and
/// `VertexPairs` are accounted for in [`similarity_workspace_bytes`] as
/// separate live buffers.
///
/// The 2026-07-25 `cugraph-similarity-calibrate` 100k-pair observations pin
/// these terms for both Int32/Int64 and weighted/unweighted Jaccard across
/// sparse-uniform and hub-skewed graph topologies. The sparse topology is the
/// observed high-water case; the hub case remains admitted without projecting
/// its lower peak into the model. The same libcugraph pairwise workspace is
/// used by all four exposed metrics. See the calibration-band contract test
/// for the captured peaks and commands.
const SIMILARITY_WORKSPACE_FIXED_BYTES: u64 = 120 * 1024;
const SIMILARITY_NATIVE_SCORE_WIDTH_BYTES: u64 = 8;

fn similarity_workspace_bytes(pair_rows: u64, pair_vertex_width_bytes: u64, weighted: bool) -> u64 {
    let pair_columns = pair_rows.saturating_mul(pair_vertex_width_bytes.saturating_mul(2));
    let pair_input_overlap = pair_columns;
    let native_vertex_pairs = pair_columns;
    let native_scores = pair_rows.saturating_mul(SIMILARITY_NATIVE_SCORE_WIDTH_BYTES);
    let kernel_workspace = SIMILARITY_WORKSPACE_FIXED_BYTES.saturating_add(
        pair_rows.saturating_mul(similarity_workspace_per_pair_bytes(pair_vertex_width_bytes, weighted)),
    );
    pair_input_overlap
        .saturating_add(native_vertex_pairs)
        .saturating_add(native_scores)
        .saturating_add(kernel_workspace)
}

fn similarity_workspace_per_pair_bytes(pair_vertex_width_bytes: u64, weighted: bool) -> u64 {
    if weighted {
        // The supported vertex domains are exact-width Int32 and Int64. The
        // isolated calibration matrix observes 66 B and 85 B respectively;
        // a single linear term would either under-admit Int64 or add an
        // unexplained broad multiplier to Int32.
        match pair_vertex_width_bytes {
            4 => 66,
            8 => 85,
            _ => unreachable!("similarity validation admits only Int32 or Int64 vertex domains"),
        }
    } else {
        // As above, price the two supported widths from the isolated matrix
        // rather than using a broad common slope: Int32 => 8 B, Int64 => 28 B.
        match pair_vertex_width_bytes {
            4 => 8,
            8 => 28,
            _ => unreachable!("similarity validation admits only Int32 or Int64 vertex domains"),
        }
    }
}

fn similarity_pair_memory_evidence(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<(Option<u64>, Option<u64>, Option<GraphPairCardinalitySource>)> {
    if !spec.algorithm.is_similarity() {
        return Ok((None, None, None));
    }
    let (Some(pair_rows), Some(pair_vertex_width_bytes), Some(pair_cardinality_source)) =
        (input.pair_rows, input.pair_vertex_width_bytes, input.pair_cardinality_source)
    else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair similarity requires pair-cardinality evidence before memory admission",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "missing_pair_cardinality_evidence")
        .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
        .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
        .with_fact(
            "pair_cardinality_source",
            input.pair_cardinality_source.map_or("none", GraphPairCardinalitySource::as_str),
        ));
    };
    if pair_vertex_width_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair similarity requires a positive pair vertex dtype width",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "invalid_pair_vertex_width")
        .with_fact("pair_vertex_width_bytes", pair_vertex_width_bytes));
    }
    if pair_vertex_width_bytes != input.vertex_width_bytes {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair vertex width must match the graph vertex width",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "pair_graph_vertex_width_mismatch")
        .with_fact("pair_vertex_width_bytes", pair_vertex_width_bytes)
        .with_fact("vertex_width_bytes", input.vertex_width_bytes));
    }
    Ok((Some(pair_rows), Some(pair_vertex_width_bytes), Some(pair_cardinality_source)))
}

fn require_strict_similarity_pair_cardinality_evidence(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<()> {
    if !spec.algorithm.is_similarity() {
        return Ok(());
    }

    let Some(source) = input.pair_cardinality_source else {
        return Err(strict_similarity_pair_cardinality_error(
            spec.algorithm,
            "strict_missing_pair_cardinality_evidence",
        )
        .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
        .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
        .with_fact("pair_cardinality_source", "none"));
    };
    if !source.is_strict_bounded_evidence() {
        return Err(strict_similarity_pair_cardinality_error(
            spec.algorithm,
            "strict_pair_cardinality_evidence_not_plan_bound",
        )
        .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
        .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
        .with_fact("pair_cardinality_source", source.as_str()));
    }
    Ok(())
}

fn strict_similarity_pair_cardinality_error(algorithm: GraphAlgorithm, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.strict_similarity_admission",
        "strict bounded similarity admission requires plan-exact pair cardinality or a hard pair-row upper bound",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("admission_mode", GraphPairCardinalityAdmissionMode::StrictBounded.as_str())
    .with_fact("reason", reason)
}

fn legacy_workspace_heavy_data_dependent_bytes(vertex_estimate: u64, output_sized: u64) -> u64 {
    let superlinear_probe = vertex_estimate.saturating_mul(vertex_estimate.min(1024)).saturating_mul(4);
    output_sized.saturating_mul(8).saturating_add(superlinear_probe)
}

fn triangle_count_workspace_bytes(input: GraphMemoryModelInput, vertex_estimate: u64) -> u64 {
    if let Some(statistics) = input.statistics
        && statistics.oriented_edge_count.is_some()
        && statistics.peak_chunk_min_degree_sum.is_some()
    {
        return triangle_count_mechanistic_workspace_bytes(input, statistics);
    }
    triangle_count_conservative_workspace_bytes(input, vertex_estimate)
}

fn triangle_count_conservative_workspace_bytes(input: GraphMemoryModelInput, vertex_estimate: u64) -> u64 {
    let vertex_width = input.vertex_width_bytes.max(1);
    let vertex_scratch =
        vertex_estimate.saturating_mul(vertex_width.max(8)).saturating_mul(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS);
    let edge_workspace =
        input.input_edges.saturating_mul(vertex_width).saturating_mul(TRIANGLE_COUNT_NO_STATS_EDGE_VERTEX_COLUMNS);
    edge_workspace.saturating_add(vertex_scratch)
}

fn triangle_count_mechanistic_workspace_bytes(input: GraphMemoryModelInput, statistics: GraphStatistics) -> u64 {
    let vertex_width = input.vertex_width_bytes.max(1);
    let exact_vertices = statistics.exact_vertices;
    let oriented_edges = statistics.oriented_edge_count.unwrap_or(0);
    let peak_min_degree = statistics.peak_chunk_min_degree_sum.unwrap_or(0);
    let dfg_rebuild =
        oriented_edges.saturating_mul(vertex_width).saturating_mul(TRIANGLE_COUNT_DFG_REBUILD_VERTEX_COLUMNS);
    let pair_list = oriented_edges.saturating_mul(vertex_width).saturating_mul(TRIANGLE_COUNT_PAIR_LIST_VERTEX_COLUMNS);
    let peak_intersection =
        peak_min_degree.saturating_mul(vertex_width).saturating_mul(TRIANGLE_COUNT_PEAK_INTERSECTION_VERTEX_COLUMNS);
    let vertex_scratch =
        exact_vertices.saturating_mul(vertex_width.max(8)).saturating_mul(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS);
    let sort_staging =
        input.input_edges.saturating_mul(vertex_width).saturating_mul(TRIANGLE_COUNT_SORT_STAGING_VERTEX_COLUMNS);
    dfg_rebuild
        .saturating_add(pair_list)
        .saturating_add(peak_intersection)
        .saturating_add(vertex_scratch)
        .saturating_add(sort_staging)
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
        .saturating_add(betweenness_push_batch_workspace_bytes(input, concurrent_sources))
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
        .saturating_add(betweenness_push_batch_workspace_bytes(input, 1))
        .saturating_add(explicit_seed_list_bytes(input, seed_input))
        .saturating_add(runtime_seed_selection_bytes(input, vertex_estimate, seed_input))
}

fn betweenness_push_batch_workspace_bytes(input: GraphMemoryModelInput, resident_source_copies: u64) -> u64 {
    let tuple_width = input
        .vertex_width_bytes
        .max(8)
        .saturating_add(BETWEENNESS_SOURCE_ORIGIN_WIDTH_BYTES)
        .saturating_add(input.vertex_width_bytes.max(8));
    let pushes = input
        .statistics
        .and_then(|statistics| statistics.sum_out_degree)
        .unwrap_or_else(|| input.input_edges.saturating_mul(resident_source_copies.max(1)));
    if tuple_width == 0 || pushes == 0 {
        return 0;
    }
    let resident_pushes = input
        .device_total_memory_bytes
        .map(|total_memory_bytes| {
            let cap = total_memory_bytes / BETWEENNESS_PUSH_TUPLE_DEVICE_MEMORY_DENOMINATOR / tuple_width;
            pushes.min(cap.max(1))
        })
        .unwrap_or(pushes);
    resident_pushes.saturating_mul(tuple_width)
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
        raw_input_edges: None,
        input_edges,
        vertex_width_bytes,
        weight_width_bytes: consumes_weight.then_some(weight_width_bytes).flatten(),
        has_edge_id: edge_columns.edge_id.is_some(),
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    }
}

/// Estimate graph memory asks for strict bounded admission when plan/config
/// facts are known.
///
/// Missing ordinary edge-row evidence leaves a graph node unpriced. In
/// contrast, a similarity node with known edge rows but no plan-exact or hard
/// pair-row bound is a strict admission failure: materializing the pair source
/// first would create adaptive evidence, not a strict pre-admission proof.
pub fn graph_memory_estimates_for_plan(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<Vec<GraphPlanMemoryEstimate>> {
    let mut estimates = Vec::new();
    for node in plan.nodes() {
        let PlanNode::GraphAlgorithm(graph) = &node.kind else {
            continue;
        };
        let Some(mut input) = graph_memory_model_input_for_plan_node(plan, graph, calibration_override)? else {
            continue;
        };
        input = calibration_override.apply_to_input(input);
        if let Some(device_total_memory_bytes) = device_total_memory_bytes {
            input = input.with_device_total_memory_bytes(device_total_memory_bytes);
        }
        let estimate = GraphMemoryModel.estimate_for_strict_bounded(&graph.spec, input)?;
        estimates.push(GraphPlanMemoryEstimate { node_id: node.id, estimate });
    }
    Ok(estimates)
}

/// Estimate graph memory for a graph edge list when planning has direct edge-count evidence.
pub fn graph_memory_estimate_for_edge_list(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryEstimate> {
    let mut input = graph_memory_model_input_for_edge_list(spec, input_schema, input_edges, raw_input_edges)?;
    if let Some(device_total_memory_bytes) = device_total_memory_bytes {
        input = input.with_device_total_memory_bytes(device_total_memory_bytes);
    }
    GraphMemoryModel.estimate(spec, calibration_override.apply_to_input(input))
}

/// Estimate the row count produced by a native edge-list plan, when the plan
/// carries deterministic planning evidence.
pub fn graph_edge_list_output_rows_for_plan(plan: &QueryPlan) -> Option<u64> {
    estimated_output_rows(plan, plan.root_id())
}

/// Returns a mathematically hard upper bound on the root row count when the
/// native plan can prove one without executing any source.
///
/// This is intentionally narrower than [`graph_edge_list_output_rows_for_plan`]:
/// strict admission may use it for resident side inputs, while descriptive
/// planning estimates remain unsuitable as an admission proof.
pub fn query_plan_hard_row_upper_bound(plan: &QueryPlan) -> Option<u64> {
    hard_row_upper_bound_for_plan_node(plan, plan.root_id()).map(|(rows, _)| rows)
}

fn graph_memory_model_input_for_plan_node(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    calibration_override: GraphMemoryCalibrationOverride,
) -> Result<Option<GraphMemoryModelInput>> {
    let Some(edge_counts) = graph_plan_edge_counts(plan, graph.input, calibration_override.raw_input_edges) else {
        return Ok(None);
    };
    let input_schema =
        plan.node(graph.input).ok_or_else(|| missing_graph_input_error(graph.input))?.output_schema().clone();
    let mut input = graph_memory_model_input_for_edge_list(
        &graph.spec,
        &input_schema,
        edge_counts.input_edges,
        edge_counts.raw_input_edges,
    )?;
    if let Some((pair_rows, pair_cardinality_source)) = strict_similarity_pair_cardinality_for_plan(plan, graph)? {
        let crate::plan::GraphAlgorithmParams::Similarity(params) = &graph.spec.params else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.plan_estimate",
                "similarity algorithm node carried non-similarity parameters",
            )
            .with_fact("algorithm", graph.spec.algorithm.as_str()));
        };
        let pair_schema = plan
            .node(params.vertex_pairs.input)
            .ok_or_else(|| missing_graph_input_error(params.vertex_pairs.input))?
            .output_schema();
        let first_width = graph_vertex_field_width(
            pair_schema,
            &params.vertex_pairs.first_column,
            "pair_first",
            graph.spec.algorithm,
        )?;
        let second_width = graph_vertex_field_width(
            pair_schema,
            &params.vertex_pairs.second_column,
            "pair_second",
            graph.spec.algorithm,
        )?;
        if first_width != second_width || first_width != input.vertex_width_bytes {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_memory.plan_estimate",
                "similarity pair columns must match each other and the graph vertex width",
            )
            .with_fact("algorithm", graph.spec.algorithm.as_str())
            .with_fact("first_vertex_width_bytes", first_width)
            .with_fact("second_vertex_width_bytes", second_width)
            .with_fact("graph_vertex_width_bytes", input.vertex_width_bytes));
        }
        input = input.with_similarity_pair_input(pair_rows, first_width, pair_cardinality_source);
    }
    Ok(Some(input))
}

/// Returns only the pair-cardinality evidence that can prove strict bounded
/// similarity admission before the candidate relation is materialized.
///
/// This intentionally excludes runtime counts.  Callers that execute a
/// strict bounded graph node must invoke it before evaluating graph side
/// inputs, then price the graph reservation with the returned exact count or
/// upper bound.
pub fn strict_similarity_pair_cardinality_for_plan(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
) -> Result<Option<(u64, GraphPairCardinalitySource)>> {
    if !graph.spec.algorithm.is_similarity() {
        return Ok(None);
    }
    let crate::plan::GraphAlgorithmParams::Similarity(params) = &graph.spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.strict_similarity_admission",
            "similarity algorithm node carried non-similarity parameters",
        )
        .with_fact("algorithm", graph.spec.algorithm.as_str()));
    };
    hard_row_upper_bound_for_plan_node(plan, params.vertex_pairs.input).map(Some).ok_or_else(|| {
        strict_similarity_pair_cardinality_error(graph.spec.algorithm, "strict_missing_pair_cardinality_evidence")
    })
}

/// Derive only mathematically hard cardinality evidence.  This deliberately
/// does not reuse [`estimated_output_rows`], whose contract is descriptive and
/// unsuitable for strict bounded admission.
fn hard_row_upper_bound_for_plan_node(plan: &QueryPlan, node_id: NodeId) -> Option<(u64, GraphPairCardinalitySource)> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LiteralBatch(source) => {
                Some((source.row_count() as u64, GraphPairCardinalitySource::PlanExact))
            },
            crate::source::SourceNode::LocalParquet(source) => {
                let rows = source
                    .files
                    .iter()
                    .map(|file| {
                        file.row_group_rows.as_ref()?.iter().try_fold(0_u64, |total, rows| total.checked_add(*rows))
                    })
                    .try_fold(0_u64, |total, rows| total.checked_add(rows?))?;
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some((
                    source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)),
                    GraphPairCardinalitySource::PlanHardUpperBound,
                ))
            },
            crate::source::SourceNode::GpuResident(source) => {
                source.row_count_upper_bound.map(|rows| (rows, GraphPairCardinalitySource::PlanHardUpperBound))
            },
        },
        PlanNode::Projection(projection) => hard_row_upper_bound_for_plan_node(plan, projection.input),
        PlanNode::Filter(filter) => hard_row_upper_bound_for_plan_node(plan, filter.input)
            .map(|(rows, _)| (rows, GraphPairCardinalitySource::PlanHardUpperBound)),
        PlanNode::Window(window) => hard_row_upper_bound_for_plan_node(plan, window.input),
        PlanNode::Sort(sort) => hard_row_upper_bound_for_plan_node(plan, sort.input),
        PlanNode::Limit(limit) => match limit.spec.fetch {
            Some(fetch) => Some((fetch as u64, GraphPairCardinalitySource::PlanHardUpperBound)),
            None => hard_row_upper_bound_for_plan_node(plan, limit.input).map(|(rows, _)| {
                (rows.saturating_sub(limit.spec.skip as u64), GraphPairCardinalitySource::PlanHardUpperBound)
            }),
        },
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| hard_row_upper_bound_for_plan_node(plan, *input)?.0.checked_add(total))
            .map(|rows| (rows, GraphPairCardinalitySource::PlanHardUpperBound)),
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            Some((1, GraphPairCardinalitySource::PlanExact))
        },
        PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => None,
    }
}

fn graph_memory_model_input_for_edge_list(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
) -> Result<GraphMemoryModelInput> {
    let source_width = graph_vertex_field_width(input_schema, &spec.edge_columns.source, "source", spec.algorithm)?;
    let destination_width =
        graph_vertex_field_width(input_schema, &spec.edge_columns.destination, "destination", spec.algorithm)?;
    if source_width != destination_width {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_memory.plan_estimate",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_width_bytes", source_width)
        .with_fact("dst_width_bytes", destination_width));
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_width = graph_vertex_field_width(input_schema, edge_id, "edge_id", spec.algorithm)?;
        if edge_id_width != source_width {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_memory.plan_estimate",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_width_bytes", edge_id_width)
            .with_fact("vertex_width_bytes", source_width));
        }
    }

    let weight_width_bytes = if spec.algorithm.consumes_weight() {
        spec.edge_columns
            .weight
            .as_ref()
            .map(|weight| graph_weight_field_width(input_schema, weight, spec.algorithm))
            .transpose()?
    } else {
        None
    };
    let mut input = graph_memory_model_input(
        input_edges,
        source_width,
        &spec.edge_columns,
        spec.algorithm.consumes_weight(),
        weight_width_bytes,
    )
    .with_edge_normalize_inserted(spec.edge_normalize_inserted);
    if let Some(raw_input_edges) = raw_input_edges {
        input = input.with_raw_input_edges(raw_input_edges);
    }
    Ok(input)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphPlanEdgeCounts {
    raw_input_edges: Option<u64>,
    input_edges: u64,
}

fn graph_plan_edge_counts(
    plan: &QueryPlan,
    graph_input: NodeId,
    configured_raw_input_edges: Option<u64>,
) -> Option<GraphPlanEdgeCounts> {
    let node = plan.node(graph_input)?;
    if let PlanNode::EdgeNormalize(edge_normalize) = &node.kind {
        let raw_input_edges = estimated_output_rows(plan, edge_normalize.input).or(configured_raw_input_edges)?;
        return Some(GraphPlanEdgeCounts {
            raw_input_edges: Some(raw_input_edges),
            input_edges: edge_normalize_output_row_upper_bound(raw_input_edges, &edge_normalize.spec),
        });
    }
    let input_edges = estimated_output_rows(plan, graph_input).or(configured_raw_input_edges)?;
    Some(GraphPlanEdgeCounts { raw_input_edges: configured_raw_input_edges, input_edges })
}

/// Size the exact-vertex probe reservation.
///
/// The probe copies both endpoint columns, concatenates them, distincts the
/// result, and sorts vertices for deterministic seed selection. Under the
/// graph stream lease this includes output buffers and cudf temporary
/// workspaces, so the multiplier is intentionally higher than the old
/// output-only budget-tracking estimate.
pub fn graph_vertex_estimate_probe_bytes(input: GraphMemoryModelInput, remaining_bytes: u64) -> u64 {
    input
        .input_edges
        .saturating_mul(input.vertex_width_bytes.saturating_mul(16).saturating_add(16))
        .max(GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES)
        .min(remaining_bytes)
        .max(1)
}

/// Size the bounded runtime statistics probe.
///
/// Algorithms that only need exact endpoint cardinality use the narrower
/// exact-vertex probe. Algorithms whose admission depends on additional
/// topology evidence get the full transient graph-statistics window; the probe
/// reservation is released before the graph-stage reservation, so this does not
/// inflate the final graph ask.
pub fn graph_runtime_statistics_probe_bytes(
    algorithm: GraphAlgorithm,
    input: GraphMemoryModelInput,
    remaining_bytes: u64,
) -> u64 {
    if graph_runtime_statistics_requires_algorithm_evidence(algorithm) {
        remaining_bytes.max(1)
    } else {
        graph_vertex_estimate_probe_bytes(input, remaining_bytes)
    }
}

const fn graph_runtime_statistics_requires_algorithm_evidence(algorithm: GraphAlgorithm) -> bool {
    matches!(
        algorithm,
        GraphAlgorithm::TriangleCountAll
            | GraphAlgorithm::BetweennessCentrality
            | GraphAlgorithm::EdgeBetweennessCentrality
    )
}

/// Derive the triangle-count peak chunk min-degree pricing term.
///
/// cuGraph chunks DFG endpoint pairs by dense destination id, which is not
/// directly observable from sparse external vertex ids. Use the average chunk
/// load implied by the measured full min-degree sum plus a fixed safety margin;
/// the graph stream lease turns residual under-margin into a contained,
/// retryable allocation failure instead of shared-pool pressure.
pub fn triangle_count_peak_chunk_min_degree_pricing_term(
    sum_min_degree_oriented: u64,
    oriented_edge_count: u64,
    device_multiprocessor_count: Option<u32>,
) -> u64 {
    if sum_min_degree_oriented == 0 || oriented_edge_count == 0 {
        return 0;
    }
    let sm_count = u64::from(device_multiprocessor_count.unwrap_or(1)).max(1);
    let max_chunk_size = sm_count.saturating_mul(1 << 15).max(1);
    let chunk_count = ceil_div_u64(oriented_edge_count, max_chunk_size).max(1);
    let average_chunk = ceil_div_u64(sum_min_degree_oriented, chunk_count);
    ceil_saturating_ratio(average_chunk, TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR, TRIANGLE_COUNT_CHUNK_SAFETY_DENOMINATOR)
        .min(sum_min_degree_oriented)
}

fn graph_calibration_source(input: GraphMemoryModelInput) -> GraphCalibrationSource {
    if input.raw_input_edges.is_some() { GraphCalibrationSource::RawInputEdges } else { GraphCalibrationSource::None }
}

fn ceil_div_u64(value: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return u64::MAX;
    }
    let numerator = u128::from(value);
    let denominator = u128::from(denominator);
    numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64
}

fn ceil_saturating_ratio(value: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return u64::MAX;
    }
    let product = u128::from(value).saturating_mul(u128::from(numerator));
    let rounded = product.saturating_add(u128::from(denominator - 1)) / u128::from(denominator);
    rounded.min(u128::from(u64::MAX)) as u64
}

fn edge_normalize_output_row_upper_bound(raw_input_edges: u64, spec: &EdgeNormalizeSpec) -> u64 {
    raw_input_edges.saturating_mul(if spec.add_reverse { 2 } else { 1 })
}

fn graph_vertex_field_width(
    schema: &SchemaRef,
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<u64> {
    let field = schema.field_with_name(column).map_err(|_error| graph_field_error(column, role, algorithm))?;
    match field.data_type() {
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(8),
        other => Err(graph_field_type_error(column, role, algorithm, other)),
    }
}

fn graph_weight_field_width(schema: &SchemaRef, column: &str, algorithm: GraphAlgorithm) -> Result<u64> {
    let field = schema.field_with_name(column).map_err(|_error| graph_field_error(column, "weight", algorithm))?;
    match field.data_type() {
        DataType::Float32 => Ok(4),
        DataType::Float64 => Ok(8),
        other => Err(graph_field_type_error(column, "weight", algorithm, other)),
    }
}

fn graph_field_error(column: &str, role: &'static str, algorithm: GraphAlgorithm) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate could not resolve an edge-list column",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("column", column)
    .with_fact("role", role)
}

fn graph_field_type_error(
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
    data_type: &DataType,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate only supports fixed-width cuGraph edge types",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("column", column)
    .with_fact("role", role)
    .with_fact("arrow_type", format!("{data_type:?}"))
}

fn missing_graph_input_error(input: NodeId) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate referenced a missing graph input node",
    )
    .with_fact("input", input.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::{
        plan::{GraphConstructionOptions, GraphTriangleCountAllParams, QueryPlanBuilder},
        source::{LocalParquetFile, LocalParquetSource, SourceNode},
    };

    fn edge_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    fn triangle_plan(source: LocalParquetSource) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::TriangleCountAll,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::PythonCugraph,
            },
            GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
        );
        let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
        builder.finish_with_host_sink(graph)
    }

    #[test]
    fn graph_memory_estimates_for_plan_use_row_group_rows_and_edge_normalize_upper_bound() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![10, 5])],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let plan = triangle_plan(source);

        let estimates = graph_memory_estimates_for_plan(
            &plan,
            GraphMemoryCalibrationOverride::none().with_raw_input_edges(100),
            None,
        )
        .unwrap();

        assert_eq!(estimates.len(), 1);
        let estimate = &estimates[0].estimate;
        assert_eq!(estimate.raw_input_edges, Some(15));
        assert_eq!(estimate.input_edges, 30);
        assert_eq!(estimate.vertex_width_bytes, 8);
        assert_eq!(estimate.weight_width_bytes, None);
        assert!(estimate.edge_normalize_inserted);
        assert_eq!(estimate.calibration_source, GraphCalibrationSource::RawInputEdges);
        assert_eq!(estimate.requested_bytes, estimate.construction_bytes.saturating_add(estimate.execution_bytes));
        assert!(estimate.construction_bytes > 0);
        assert!(estimate.execution_bytes < estimate.requested_bytes);
    }

    #[test]
    fn graph_memory_estimates_for_plan_can_use_configured_raw_edges_when_plan_rows_are_unknown() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet")],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let plan = triangle_plan(source);
        let calibration = GraphMemoryCalibrationOverride::none().with_raw_input_edges(100);

        let estimates = graph_memory_estimates_for_plan(&plan, calibration, None).unwrap();

        assert_eq!(estimates.len(), 1);
        let estimate = &estimates[0].estimate;
        assert_eq!(estimate.raw_input_edges, Some(100));
        assert_eq!(estimate.input_edges, 200);
        assert_eq!(estimate.calibration_source, GraphCalibrationSource::RawInputEdges);
    }

    #[test]
    fn graph_memory_estimates_for_plan_skip_graph_nodes_without_edge_count_evidence() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet")],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let plan = triangle_plan(source);

        let estimates = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();

        assert!(estimates.is_empty());
    }
}
