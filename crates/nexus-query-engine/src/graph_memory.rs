//! Approximate cuGraph memory pricing outside the strict bounded gate.
//!
//! These estimates are intentionally not strict-bounded contracts. They price a
//! graph fragment before entering cuGraph so bounded service admission can fail
//! locally with structured facts instead of relying on a raw RMM OOM.

use arrow::datatypes::{DataType, SchemaRef};

use crate::{
    admission::canonical_execution_scope_fingerprint,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
    metrics::{PlanNodeId, QueryId},
    plan::{
        EdgeNormalizeSpec, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy,
        GraphEdgeColumnBindings, JoinType, NodeId, PlanNode, QueryPlan, estimated_output_rows,
    },
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

    /// Returns whether the evidence is available before bounded admission.
    pub(crate) const fn is_pre_admission_bound(self) -> bool {
        matches!(self, Self::PlanExact | Self::PlanHardUpperBound)
    }
}

impl std::fmt::Display for GraphPairCardinalitySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Capability admission status projected into graph report rows.
///
/// The bounded controller emits [`Self::Bounded`]. Compatibility string
/// labels are emitted only from the bounded capability contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPairCardinalityAdmissionMode {
    /// No bounded admission was active; the row records observed execution
    /// evidence only.
    Unbounded,
    /// Bounded admission used a plan-exact count or hard upper bound.
    Bounded,
}

impl GraphPairCardinalityAdmissionMode {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded => "strict_bounded",
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

/// Owner-provided phase bounds for raw single-GPU graph construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct GraphConstructionWorkspaceBounds {
    renumber_workspace_bytes: u64,
    compression_workspace_bytes: u64,
}

impl GraphConstructionWorkspaceBounds {
    /// Create construction workspace bounds from the owner-provided phases.
    pub const fn new(renumber_workspace_bytes: u64, compression_workspace_bytes: u64) -> Self {
        Self { renumber_workspace_bytes, compression_workspace_bytes }
    }

    /// Peak workspace used while renumbering endpoints.
    pub const fn renumber_workspace_bytes(self) -> u64 {
        self.renumber_workspace_bytes
    }

    /// Peak workspace used while sorting and compressing endpoints.
    pub const fn compression_workspace_bytes(self) -> u64 {
        self.compression_workspace_bytes
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
    /// Upper bounds for raw libcugraph graph-construction phases, supplied by
    /// the owning cuGraph preflight contract. `None` makes bounded admission
    /// reject rather than invent an allowance.
    pub construction_workspace_bounds: Option<GraphConstructionWorkspaceBounds>,
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
    pub fn vertex_upper_bound(self) -> Result<u64> {
        self.input_edges
            .checked_mul(2)
            .ok_or_else(|| graph_memory_arithmetic_error("vertex_upper_bound", "multiplication"))
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

    /// Attach the owning libcugraph graph-construction phase bounds.
    pub fn with_construction_workspace_bounds(mut self, bounds: GraphConstructionWorkspaceBounds) -> Self {
        self.construction_workspace_bounds = Some(bounds);
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

/// Typed retained-output contribution from the graph-memory owner.
///
/// The retained output is kept as a typed contribution instead of being
/// recovered from a difference between rounded phase peaks.  That distinction
/// matters when composing graph producers: a producer's output handle remains
/// live after its isolated phase and is therefore an independent owner in the
/// schedule fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryRetainedOutputContribution {
    bytes: u64,
    row_count_upper_bound: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GraphMemoryOwnerTermRole {
    GraphPersistent,
    ConstructionTransient,
    ExecutionWorkspace,
    GraphLiveReservationFloorTopUp,
    SideInputWorkspace,
    RetainedOutput,
}

impl GraphMemoryOwnerTermRole {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::GraphPersistent => "graph_persistent",
            Self::ConstructionTransient => "construction_transient",
            Self::ExecutionWorkspace => "execution_workspace",
            Self::GraphLiveReservationFloorTopUp => "graph_live_reservation_floor_top_up",
            Self::SideInputWorkspace => "side_input_workspace",
            Self::RetainedOutput => "retained_output",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphMemoryOwnerTermFact {
    execution_scope_fingerprint: String,
    graph_node_id: NodeId,
    role: GraphMemoryOwnerTermRole,
    bytes: u64,
    row_count_upper_bound: Option<u64>,
}

impl GraphMemoryOwnerTermFact {
    fn new(
        execution_scope_fingerprint: String,
        graph_node_id: NodeId,
        role: GraphMemoryOwnerTermRole,
        bytes: u64,
        row_count_upper_bound: Option<u64>,
    ) -> Result<Self> {
        if bytes == u64::MAX {
            return Err(graph_memory_arithmetic_error(role.as_str(), "saturated_owner_term"));
        }
        Ok(Self { execution_scope_fingerprint, graph_node_id, role, bytes, row_count_upper_bound })
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(crate) const fn graph_node_id(&self) -> NodeId {
        self.graph_node_id
    }

    pub(crate) const fn role(&self) -> GraphMemoryOwnerTermRole {
        self.role
    }

    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) const fn row_count_upper_bound(&self) -> Option<u64> {
        self.row_count_upper_bound
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphMemoryOwnerComponents {
    graph_persistent_bytes: u64,
    construction_transient_bytes: u64,
    execution_workspace_bytes: u64,
}

impl GraphMemoryRetainedOutputContribution {
    /// Builds a model-owned retained-output contribution.
    pub(crate) const fn new(bytes: u64, row_count_upper_bound: u64) -> Self {
        Self { bytes, row_count_upper_bound }
    }

    /// Device bytes retained by the producer after its isolated phase.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Upper bound on rows represented by the retained producer output.
    ///
    /// The row bound is carried with the bytes so an engine-owned consumer
    /// preflight can price a downstream join without asking an adapter to
    /// reverse-engineer graph output cardinality.
    pub const fn row_count_upper_bound(self) -> u64 {
        self.row_count_upper_bound
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
    /// Peak bytes needed while constructing the native graph representation.
    ///
    /// The execution-only estimate may omit this diagnostic term, but
    /// admission never uses that estimate to select a grant. A resident cache
    /// needs its own accounted reservation before retaining a graph across
    /// attempts.
    pub construction_bytes: u64,
    /// Peak bytes needed by the algorithm after graph construction.
    pub execution_bytes: u64,
    /// Typed bytes and row-count bound retained by the producer result handle.
    ///
    /// This is the owner-calculated `output_bytes` contribution. It is not
    /// inferred from `requested_bytes` or `execution_bytes`, both of which may
    /// include rounded workspace and phase terms.
    pub retained_output_bytes: GraphMemoryRetainedOutputContribution,
    /// Cold native-graph peak estimate. For algorithms with distinct native
    /// lifetime phases this is the phase-aware peak, not a naïve sum of
    /// construction and execution estimates. Admission composes any retained
    /// source-stage owner separately when the plan proves one.
    pub requested_bytes: u64,
    owner_components: GraphMemoryOwnerComponents,
}

/// Opaque graph-memory model output.
///
/// The model is the only constructor for this token. It contains descriptive
/// bytes only: it cannot form a generic admission requirement or alter the
/// ledger's floor, capability, or fit proof. An engine-issued capability proof
/// may retain its cold estimate as a soft immutable-grant target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphSourceStageLifetime {
    /// The cold graph peak already owns the prepared edge frame emitted by the
    /// native source. The source diagnostic is not a second GPU allocation.
    SharedPreparedEdgeFrame,
    /// Explicit edge normalization retains the source frame while it creates
    /// the normalized graph input, so both owners are live at the cold peak.
    SeparateNormalizedInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryAdmissionEvidence {
    execution_scope_fingerprint: String,
    cold_model_requested_bytes: u64,
    warm_model_requested_bytes: u64,
    retained_output_bytes: GraphMemoryRetainedOutputContribution,
    owner_terms: Vec<GraphMemoryOwnerTermFact>,
    source_stage_lifetime: GraphSourceStageLifetime,
}

impl GraphMemoryAdmissionEvidence {
    fn from_bounded_plan_estimate(plan: &QueryPlan, node_id: NodeId, estimate: &GraphMemoryEstimate) -> Result<Self> {
        // `requested_bytes` is already phase-aware: persistent graph/input
        // owners survive into execution, while construction-only scratch does not.
        let cold_model_requested_bytes = estimate.construction_bytes.max(estimate.requested_bytes).max(1);
        let execution_scope_fingerprint = canonical_execution_scope_fingerprint(plan);
        let live_reservation_floor_top_up_bytes =
            graph_live_reservation_floor_top_up_bytes(cold_model_requested_bytes)?;
        let mut owner_terms = estimate.owner_components.owner_terms(
            execution_scope_fingerprint.clone(),
            node_id,
            estimate.retained_output_bytes,
            live_reservation_floor_top_up_bytes,
        )?;
        if let Some(side_input_peak_bytes) = graph_side_input_workspace_peak_bytes(plan, node_id, estimate)? {
            owner_terms.push(GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                node_id,
                GraphMemoryOwnerTermRole::SideInputWorkspace,
                side_input_peak_bytes,
                None,
            )?);
        }
        Ok(Self {
            execution_scope_fingerprint,
            cold_model_requested_bytes,
            warm_model_requested_bytes: estimate.execution_bytes,
            retained_output_bytes: estimate.retained_output_bytes,
            owner_terms,
            source_stage_lifetime: if estimate.edge_normalize_inserted {
                GraphSourceStageLifetime::SeparateNormalizedInput
            } else {
                GraphSourceStageLifetime::SharedPreparedEdgeFrame
            },
        })
    }

    pub(crate) fn matches_execution_scope_fingerprint(&self, fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == fingerprint
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(crate) fn owner_term_facts(&self) -> &[GraphMemoryOwnerTermFact] {
        &self.owner_terms
    }

    /// Returns the construction-inclusive model's descriptive byte estimate.
    ///
    /// This is diagnostic/model output. It may refine an engine-issued graph
    /// capability proof's soft grant target, but is not accepted as a raw
    /// public admission-request input.
    pub fn cold_model_requested_bytes(&self) -> u64 {
        self.cold_model_requested_bytes
    }

    /// Returns the execution-only model's descriptive byte estimate.
    ///
    /// This is a hypothetical cache-price diagnostic only. It is not accepted
    /// by public admission request construction or device placement.
    pub fn warm_model_requested_bytes(&self) -> u64 {
        self.warm_model_requested_bytes
    }

    /// Returns the typed producer output contribution retained through the
    /// composed consumer phase.
    pub fn retained_output(&self) -> GraphMemoryRetainedOutputContribution {
        self.retained_output_bytes
    }

    /// Returns the retained output bytes without exposing a caller-authored
    /// byte input path.
    pub fn retained_output_bytes(&self) -> u64 {
        self.retained_output_bytes.bytes()
    }

    /// Returns the model-owned retained output row bound.
    pub fn retained_output_rows_upper_bound(&self) -> u64 {
        self.retained_output_bytes.row_count_upper_bound()
    }

    pub(crate) const fn source_stage_lifetime(&self) -> GraphSourceStageLifetime {
        self.source_stage_lifetime
    }

    /// Composes the model's cold graph peak with the exact source-stage hint
    /// retained by the matching native edge fragment.
    ///
    /// The lifecycle classification is model-owned: raw input shares the
    /// prepared edge frame already priced by the graph peak, while an explicit
    /// normalize operator keeps the source and its derived graph input live
    /// together. Callers cannot choose this relation or inject a raw graph
    /// model byte value.
    pub(crate) fn cold_model_with_source_stage_bytes(&self, source_stage_bytes: u64) -> Result<u64> {
        self.graph_peak_with_source_stage_bytes(self.cold_model_requested_bytes, source_stage_bytes)
    }

    /// Returns the graph-native live reservation after applying the canonical
    /// bounded execution minimum. The model value remains available through
    /// [`Self::cold_model_requested_bytes`] as diagnostics; this lower bound
    /// is a reservation contract, not an observed allocation term.
    pub(crate) fn bounded_live_graph_peak_bytes(&self) -> Result<u64> {
        let Some(top_up) = self.graph_live_reservation_floor_top_up_bytes()? else {
            return Ok(self.cold_model_requested_bytes);
        };
        self.cold_model_requested_bytes
            .checked_add(top_up)
            .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation", "addition"))
    }

    /// Composes the bounded graph-native live minimum with a prepared source
    /// frame using the same owner lifetime relation as the raw model.
    pub(crate) fn bounded_live_graph_peak_with_source_stage_bytes(&self, source_stage_bytes: u64) -> Result<u64> {
        let Some(top_up) = self.graph_live_reservation_floor_top_up_bytes()? else {
            return self.cold_model_with_source_stage_bytes(source_stage_bytes);
        };
        let graph_peak = self
            .cold_model_requested_bytes
            .checked_add(top_up)
            .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation", "addition"))?;
        self.graph_peak_with_source_stage_bytes(graph_peak, source_stage_bytes)
    }

    fn graph_peak_with_source_stage_bytes(&self, graph_peak_bytes: u64, source_stage_bytes: u64) -> Result<u64> {
        if source_stage_bytes == u64::MAX || graph_peak_bytes == u64::MAX {
            return Err(graph_memory_arithmetic_error("source_stage_graph_peak", "saturated"));
        }
        match self.source_stage_lifetime {
            GraphSourceStageLifetime::SharedPreparedEdgeFrame => Ok(source_stage_bytes.max(graph_peak_bytes)),
            GraphSourceStageLifetime::SeparateNormalizedInput => source_stage_bytes
                .checked_add(graph_peak_bytes)
                .ok_or_else(|| graph_memory_arithmetic_error("source_stage_graph_peak", "addition")),
        }
    }

    pub(crate) fn graph_live_reservation_floor_top_up_bytes(&self) -> Result<Option<u64>> {
        let expected = graph_live_reservation_floor_top_up_bytes(self.cold_model_requested_bytes)?;
        let mut facts = self
            .owner_terms
            .iter()
            .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp);
        let actual = facts.next().map(GraphMemoryOwnerTermFact::bytes);
        if facts.next().is_some() || actual != expected {
            return Err(graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "owner_term_mismatch")
                .with_fact("reason", "graph_memory_owner_term_floor_mismatch"));
        }
        Ok(expected)
    }
}

fn graph_side_input_workspace_peak_bytes(
    plan: &QueryPlan,
    graph_node_id: NodeId,
    _estimate: &GraphMemoryEstimate,
) -> Result<Option<u64>> {
    let graph = match &plan.node(graph_node_id).ok_or_else(|| missing_graph_input_error(graph_node_id))?.kind {
        PlanNode::GraphAlgorithm(graph) => graph,
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.side_input_workspace_preflight",
                "graph memory evidence referenced a non-graph node",
            )
            .with_fact("reason", "side_input_workspace_preflight_unavailable")
            .with_fact("side_input_reason", "graph_node_missing"));
        },
    };

    #[cfg(feature = "cugraph")]
    {
        graph_side_input_workspace_peak_bytes_with_owner(plan, graph, _estimate)
    }
    #[cfg(not(feature = "cugraph"))]
    {
        let has_bounded_side_input = match &graph.spec.params {
            GraphAlgorithmParams::Bfs(params) => params.has_predicates() || params.source_vertices.is_some(),
            GraphAlgorithmParams::PersonalizedPageRank(_) => true,
            _ => false,
        };
        if has_bounded_side_input {
            return Err(side_input_workspace_preflight_unavailable("owner_contract_unavailable", None, None));
        }
        Ok(None)
    }
}

fn side_input_workspace_preflight_unavailable(
    side_input_reason: &'static str,
    role: Option<&'static str>,
    algorithm: Option<GraphAlgorithm>,
) -> QueryError {
    side_input_workspace_error(QueryErrorCode::UnsupportedPlanShape, side_input_reason, role, algorithm)
}

fn side_input_workspace_error(
    code: QueryErrorCode,
    side_input_reason: &'static str,
    role: Option<&'static str>,
    algorithm: Option<GraphAlgorithm>,
) -> QueryError {
    let error = QueryError::new(
        code,
        "nexus_query_engine.graph_memory.side_input_workspace_preflight",
        "cuGraph side-input workspace preflight is unavailable",
    )
    .with_fact("reason", "side_input_workspace_preflight_unavailable")
    .with_fact("side_input_reason", side_input_reason);
    let error = match role {
        Some(role) => error.with_fact("side_input_role", role),
        None => error,
    };
    match algorithm {
        Some(algorithm) => error.with_fact("algorithm", algorithm.as_str()),
        None => error,
    }
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
enum GraphSideInputColumns<'a> {
    Vertex(&'a str),
    Personalization { vertex: &'a str, value: &'a str },
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
struct GraphSideInputBinding<'a> {
    role: &'static str,
    input: NodeId,
    columns: GraphSideInputColumns<'a>,
}

#[cfg(feature = "cugraph")]
fn graph_side_input_workspace_peak_bytes_with_owner(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    estimate: &GraphMemoryEstimate,
) -> Result<Option<u64>> {
    match &graph.spec.params {
        GraphAlgorithmParams::Bfs(_params) => {
            let bindings = graph_side_input_bindings(graph)?;
            let graph_vertex_type = graph_side_input_graph_vertex_type(plan, graph)?;
            let source_rows = bindings
                .iter()
                .find(|binding| binding.role == "source_vertices")
                .map(|binding| graph_side_input_rows(plan, binding))
                .transpose()?
                .unwrap_or(1);
            let mut include_vertices_rows = None;
            let mut exclude_vertices_rows = None;
            let mut target_vertices_rows = None;
            let mut include_edge_ids_rows = None;
            for binding in &bindings {
                let rows = graph_side_input_rows(plan, binding)?;
                let GraphSideInputColumns::Vertex(column) = binding.columns else {
                    return Err(side_input_workspace_preflight_unavailable(
                        "invalid_bfs_binding",
                        Some(binding.role),
                        Some(graph.spec.algorithm),
                    ));
                };
                graph_side_input_vertex_type(
                    plan,
                    binding.input,
                    column,
                    graph_vertex_type,
                    binding.role,
                    graph.spec.algorithm,
                )?;
                match binding.role {
                    "source_vertices" => {},
                    "include_vertices" => include_vertices_rows = Some(rows),
                    "exclude_vertices" => exclude_vertices_rows = Some(rows),
                    "target_vertices" => target_vertices_rows = Some(rows),
                    "include_edge_ids" => include_edge_ids_rows = Some(rows),
                    _ => {
                        return Err(side_input_workspace_preflight_unavailable(
                            "unknown_bfs_binding",
                            Some(binding.role),
                            Some(graph.spec.algorithm),
                        ));
                    },
                }
            }
            let num_vertices = graph_side_input_count(estimate.vertex_estimate, "num_vertices", graph.spec.algorithm)?;
            let num_edges = graph_side_input_count(estimate.input_edges, "num_edges", graph.spec.algorithm)?;
            if include_vertices_rows.is_some() && exclude_vertices_rows.is_some() {
                return Err(side_input_workspace_preflight_unavailable(
                    "owner_input_rejected",
                    None,
                    Some(graph.spec.algorithm),
                ));
            }
            if source_rows == 0 {
                return Ok(Some(0));
            }
            let input = cugraph::cudf_interop::BfsSideInputPreflightInput::new(
                graph_vertex_type,
                graph_vertex_type,
                num_vertices,
                num_edges,
                source_rows,
                include_vertices_rows,
                exclude_vertices_rows,
                target_vertices_rows,
                include_edge_ids_rows,
            )
            .map_err(|error| {
                side_input_workspace_preflight_unavailable("owner_input_rejected", None, Some(graph.spec.algorithm))
                    .with_source(error)
            })?;
            let preflight = cugraph::cudf_interop::bfs_side_input_workspace_preflight(&input).map_err(|error| {
                side_input_workspace_preflight_unavailable("owner_query_failed", None, Some(graph.spec.algorithm))
                    .with_source(error)
            })?;
            Ok(Some(preflight.peak_bytes()))
        },
        GraphAlgorithmParams::PersonalizedPageRank(_params) => {
            let bindings = graph_side_input_bindings(graph)?;
            let [binding] = bindings.as_slice() else {
                return Err(side_input_workspace_preflight_unavailable(
                    "personalization_binding_missing",
                    Some("personalization"),
                    Some(graph.spec.algorithm),
                ));
            };
            let GraphSideInputColumns::Personalization { vertex, value } = binding.columns else {
                return Err(side_input_workspace_preflight_unavailable(
                    "invalid_personalization_binding",
                    Some(binding.role),
                    Some(graph.spec.algorithm),
                ));
            };
            let graph_vertex_type = graph_side_input_graph_vertex_type(plan, graph)?;
            graph_side_input_vertex_type(
                plan,
                binding.input,
                vertex,
                graph_vertex_type,
                binding.role,
                graph.spec.algorithm,
            )?;
            let value_type =
                graph_side_input_value_type(plan, binding.input, value, binding.role, graph.spec.algorithm)?;
            let rows = graph_side_input_rows(plan, binding)?;
            let input = cugraph::cudf_interop::PersonalizedPageRankSideInputPreflightInput::new(
                graph_vertex_type,
                value_type,
                rows,
            )
            .map_err(|error| {
                side_input_workspace_preflight_unavailable(
                    "owner_input_rejected",
                    Some(binding.role),
                    Some(graph.spec.algorithm),
                )
                .with_source(error)
            })?;
            let preflight = cugraph::cudf_interop::personalized_pagerank_side_input_workspace_preflight(&input)
                .map_err(|error| {
                    side_input_workspace_preflight_unavailable(
                        "owner_query_failed",
                        Some(binding.role),
                        Some(graph.spec.algorithm),
                    )
                    .with_source(error)
                })?;
            Ok(Some(preflight.peak_bytes()))
        },
        _ => Ok(None),
    }
}

#[cfg(feature = "cugraph")]
fn graph_side_input_bindings<'a>(graph: &'a crate::plan::GraphAlgorithmNode) -> Result<Vec<GraphSideInputBinding<'a>>> {
    let mut bindings = Vec::new();
    match &graph.spec.params {
        GraphAlgorithmParams::Bfs(params) => {
            if params.predicates.include_edges.is_some() {
                return Err(side_input_workspace_preflight_unavailable(
                    "include_edges_owner_contract_missing",
                    Some("include_edges"),
                    Some(graph.spec.algorithm),
                ));
            }
            if let Some(input) = params.source_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "source_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.include_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "include_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.exclude_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "exclude_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.target_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "target_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.include_edge_ids.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "include_edge_ids",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
        },
        GraphAlgorithmParams::PersonalizedPageRank(params) => {
            let input = &params.personalization;
            bindings.push(GraphSideInputBinding {
                role: "personalization",
                input: input.input,
                columns: GraphSideInputColumns::Personalization {
                    vertex: input.vertex.as_str(),
                    value: input.value.as_str(),
                },
            });
        },
        _ => return Ok(bindings),
    }

    let mut expected_inputs = Vec::with_capacity(bindings.len() + 1);
    expected_inputs.push(graph.input);
    expected_inputs.extend(bindings.iter().map(|binding| binding.input));
    if graph.inputs != expected_inputs {
        return Err(side_input_workspace_preflight_unavailable(
            "graph_input_binding_mismatch",
            None,
            Some(graph.spec.algorithm),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(expected_inputs.len());
    for input in expected_inputs {
        if !seen.insert(input) {
            return Err(side_input_workspace_preflight_unavailable(
                "side_input_node_reused",
                None,
                Some(graph.spec.algorithm),
            )
            .with_fact("node_id", input));
        }
    }
    Ok(bindings)
}

#[cfg(feature = "cugraph")]
fn graph_side_input_rows(plan: &QueryPlan, binding: &GraphSideInputBinding<'_>) -> Result<usize> {
    let Some((rows, _)) = hard_row_upper_bound_for_plan_node(plan, binding.input) else {
        return Err(side_input_workspace_preflight_unavailable("missing_row_upper_bound", Some(binding.role), None)
            .with_fact("node_id", binding.input));
    };
    usize::try_from(rows).map_err(|_| {
        side_input_workspace_preflight_unavailable("row_count_conversion_overflow", Some(binding.role), None)
            .with_fact("row_count", rows)
            .with_fact("node_id", binding.input)
    })
}

#[cfg(feature = "cugraph")]
fn graph_side_input_count(value: u64, term: &'static str, algorithm: GraphAlgorithm) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        side_input_workspace_preflight_unavailable("graph_count_conversion_overflow", None, Some(algorithm))
            .with_fact("term", term)
            .with_fact("value", value)
    })
}

#[cfg(feature = "cugraph")]
fn graph_side_input_graph_vertex_type(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
) -> Result<cugraph::types::DataType> {
    let schema = plan.node(graph.input).ok_or_else(|| missing_graph_input_error(graph.input))?.output_schema();
    let field = schema.field_with_name(&graph.spec.edge_columns.source).map_err(|_| {
        side_input_workspace_preflight_unavailable(
            "graph_vertex_column_missing",
            Some("graph_vertex"),
            Some(graph.spec.algorithm),
        )
    })?;
    cugraph_index_type_for_arrow(field.data_type(), "graph_vertex", graph.spec.algorithm)
}

#[cfg(feature = "cugraph")]
fn graph_side_input_vertex_type(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    expected: cugraph::types::DataType,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<()> {
    let schema = plan
        .node(node_id)
        .ok_or_else(|| {
            side_input_workspace_preflight_unavailable("side_input_node_missing", Some(role), Some(algorithm))
                .with_fact("node_id", node_id)
        })?
        .output_schema();
    let field = schema.field_with_name(column).map_err(|_| {
        side_input_workspace_preflight_unavailable("side_input_column_missing", Some(role), Some(algorithm))
            .with_fact("column", column)
            .with_fact("node_id", node_id)
    })?;
    let actual = cugraph_index_type_for_arrow(field.data_type(), role, algorithm)?;
    if actual != expected {
        return Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_vertex_dtype_mismatch",
            Some(role),
            Some(algorithm),
        )
        .with_fact("expected_dtype", expected)
        .with_fact("actual_dtype", actual)
        .with_fact("column", column));
    }
    Ok(())
}

#[cfg(feature = "cugraph")]
fn graph_side_input_value_type(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<cugraph::types::DataType> {
    let schema = plan
        .node(node_id)
        .ok_or_else(|| {
            side_input_workspace_preflight_unavailable("side_input_node_missing", Some(role), Some(algorithm))
                .with_fact("node_id", node_id)
        })?
        .output_schema();
    let field = schema.field_with_name(column).map_err(|_| {
        side_input_workspace_preflight_unavailable("side_input_column_missing", Some(role), Some(algorithm))
            .with_fact("column", column)
            .with_fact("node_id", node_id)
    })?;
    match field.data_type() {
        DataType::Float32 => Ok(cugraph::types::DataType::Float32),
        DataType::Float64 => Ok(cugraph::types::DataType::Float64),
        _ => Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_value_dtype_unsupported",
            Some(role),
            Some(algorithm),
        )
        .with_fact("column", column)),
    }
}

#[cfg(feature = "cugraph")]
fn cugraph_index_type_for_arrow(
    data_type: &DataType,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<cugraph::types::DataType> {
    match data_type {
        DataType::Int32 => Ok(cugraph::types::DataType::Int32),
        DataType::Int64 => Ok(cugraph::types::DataType::Int64),
        _ => Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_vertex_dtype_unsupported",
            Some(role),
            Some(algorithm),
        )
        .with_fact("data_type", data_type)),
    }
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

/// Direct source for one graph attempt's measured device-memory peak.
///
/// This is deliberately separate from [`GraphCalibrationSource`]: that enum
/// describes model inputs, while this one describes the runtime observation
/// used to calibrate the completed graph attempt.  Consumers must not infer a
/// source from a missing value or from an error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMeasuredPeakProvenance {
    /// The graph execution's reservation resource reported its own high-water
    /// mark. This is the preferred graph-stage measurement.
    ReservationHighWater,
    /// The graph execution did not expose a reservation high-water mark, so
    /// the CUDA async-pool high-water delta bounded around that attempt was
    /// used instead.
    PoolDelta,
    /// An exclusive resident-graph cache scope reported its own allocator
    /// high-water mark after the warm graph execution had retired. This is
    /// the only physical peak source for work that runs on the cache runtime
    /// rather than the admitted attempt runtime.
    CacheScopeHighWater,
    /// Neither runtime measurement completed. `measured_peak_bytes` is
    /// unavailable rather than a synthetic zero.
    Unavailable,
}

impl GraphMeasuredPeakProvenance {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReservationHighWater => "reservation_high_water",
            Self::PoolDelta => "pool_delta",
            Self::CacheScopeHighWater => "cache_scope_high_water",
            Self::Unavailable => "unavailable",
        }
    }
}

impl std::fmt::Display for GraphMeasuredPeakProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Engine-owned lifecycle state for one graph model claim.
///
/// A claim exists before graph-memory admission so a missing terminal state is
/// deliberately fail-closed. The two terminal states distinguish a model
/// admission denial from an actual entry into the graph executor; neither is
/// inferred from an error string, an allocator event, or elapsed time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphClaimExecutionState {
    /// The claim was recorded but no exact terminal graph boundary was
    /// observed. This state is not execution evidence.
    Pending,
    /// The native graph algorithm executor was entered.
    AlgorithmStarted,
    /// Graph-memory pricing rejected the request because its tracked request
    /// exceeded the current graph-admission budget, before the executor ran.
    GraphMemoryAdmissionDenied,
}

impl GraphClaimExecutionState {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::AlgorithmStarted => "algorithm_started",
            Self::GraphMemoryAdmissionDenied => "graph_memory_admission_denied",
        }
    }
}

impl std::fmt::Display for GraphClaimExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-shape model-claim versus measured-peak calibration row for one graph
/// attempt.
///
/// The engine creates one of these before it enters the graph executor and
/// fills the runtime measurement when that attempt leaves the executor. It is
/// diagnostic only: the row cannot mint admission evidence or modify a
/// device grant. `node_index` is retained solely as typed executor identity;
/// consumers should use `node_id` for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphClaimCalibrationRow {
    /// Query-wide diagnostic correlation id, when the execution context has
    /// one. This is assigned by [`crate::metrics::QueryMetrics`].
    pub query_id: Option<QueryId>,
    /// Stable native graph node id for reports.
    pub node_id: String,
    /// Typed graph node identity used to correlate the post-execution sample
    /// without parsing a report string.
    pub node_index: usize,
    /// The graph algorithm that owned this model claim and measured peak.
    pub algorithm: GraphAlgorithm,
    /// Construction-inclusive bytes requested by the graph memory model.
    pub model_requested_bytes: u64,
    /// Bytes requested from the graph pricing tier after its documented floor
    /// and any live graph-input or similarity-input credit.
    pub requested_bytes: u64,
    /// The graph-pricing budget considered for this attempt, when one was
    /// present. A missing budget is not rendered as a zero-byte budget.
    pub budget_bytes: Option<u64>,
    /// Exact engine state at the graph executor boundary. A pending claim is
    /// intentionally not promoted to physical execution by an adapter.
    pub execution_state: GraphClaimExecutionState,
    /// Measured graph-stage peak bytes when a runtime observation completed.
    pub measured_peak_bytes: Option<u64>,
    /// Source of [`Self::measured_peak_bytes`].
    pub measured_peak_provenance: GraphMeasuredPeakProvenance,
    /// Fixed-point model-ask / measured-peak ratio, when a nonzero measured
    /// peak was available.
    pub graph_model_error_ratio: Option<GraphModelErrorRatio>,
}

impl GraphClaimCalibrationRow {
    /// Builds the pending calibration row from the typed graph-pricing report.
    ///
    /// The report has no private execution identity, so the caller supplies
    /// the plan node that produced it. Keeping this identity outside the
    /// report avoids changing existing reservation-row semantics.
    pub fn from_reservation_report(report: &GraphReservationReport, node_id: NodeId) -> Self {
        Self {
            query_id: report.query_id.clone(),
            node_id: report.node_id.clone(),
            node_index: node_id.index(),
            algorithm: report.algorithm,
            model_requested_bytes: report.model_requested_bytes,
            requested_bytes: report.requested_bytes,
            budget_bytes: report.budget_bytes,
            execution_state: GraphClaimExecutionState::Pending,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
            graph_model_error_ratio: None,
        }
    }

    /// Records one terminal graph-boundary state without allowing a later
    /// observation to overwrite the first exact boundary.
    pub(crate) fn record_execution_state(&mut self, execution_state: GraphClaimExecutionState) {
        if self.execution_state == GraphClaimExecutionState::Pending {
            self.execution_state = execution_state;
        }
    }

    /// Records the one direct peak observation for this graph attempt.
    ///
    /// An unavailable source always clears the measured value and ratio. A
    /// zero high-water mark remains an observed zero, but cannot form a
    /// meaningful model-error ratio.
    pub fn record_measured_peak(
        &mut self,
        measured_peak_bytes: Option<u64>,
        measured_peak_provenance: GraphMeasuredPeakProvenance,
    ) {
        match measured_peak_provenance {
            GraphMeasuredPeakProvenance::Unavailable => {
                self.measured_peak_bytes = None;
                self.graph_model_error_ratio = None;
            },
            GraphMeasuredPeakProvenance::ReservationHighWater
            | GraphMeasuredPeakProvenance::PoolDelta
            | GraphMeasuredPeakProvenance::CacheScopeHighWater => {
                self.measured_peak_bytes = measured_peak_bytes;
                self.graph_model_error_ratio = measured_peak_bytes.and_then(|peak| {
                    GraphModelErrorRatio::from_requested_and_measured(self.model_requested_bytes, peak)
                });
            },
        }
        self.measured_peak_provenance = measured_peak_provenance;
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

    /// Converts the cold whole-peak model into the incremental reservation
    /// needed when its graph-input edge frame is already live in the same query
    /// grant.
    ///
    /// The model includes the normalized semantic edge frame and, after an
    /// explicit edge-normalize stage, its retained source frame. Each live
    /// owner is credited separately and capped at its modeled overlap, so a
    /// wider input cannot erase graph construction, output, or workspace
    /// bytes. The exact normalized-frame owner may use its physical tracker
    /// charge as a tighter cap than semantic column widths.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn credit_live_graph_input(
        mut self,
        already_accounted_normalized_frame_bytes: u64,
        exact_normalized_frame_charge_bytes: Option<u64>,
        retained_normalization_source_bytes: u64,
    ) -> Self {
        let Some(budget_bytes) = self.budget_bytes else {
            return self;
        };
        let credit_cap_bytes =
            exact_normalized_frame_charge_bytes.unwrap_or_else(|| self.modeled_live_graph_input_frame_bytes());
        let normalized_frame_credit_bytes =
            credit_cap_bytes.min(already_accounted_normalized_frame_bytes).min(self.requested_bytes);
        self.requested_bytes -= normalized_frame_credit_bytes;
        let source_credit_cap_bytes =
            if self.edge_normalize_inserted { self.modeled_live_graph_input_frame_bytes() } else { 0 };
        let retained_source_credit_bytes =
            source_credit_cap_bytes.min(retained_normalization_source_bytes).min(self.requested_bytes);
        self.requested_bytes -= retained_source_credit_bytes;
        let denied = self.requested_bytes > budget_bytes;
        self.outcome = if denied { GraphReservationOutcome::Denied } else { GraphReservationOutcome::Admitted };
        self.reason = denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON);
        self
    }

    #[cfg(any(feature = "cugraph", test))]
    fn modeled_live_graph_input_frame_bytes(&self) -> u64 {
        let semantic_edge_row_width =
            semantic_edge_row_width_bytes(self.vertex_width_bytes, self.weight_width_bytes, self.has_edge_id);
        self.input_edges.saturating_mul(semantic_edge_row_width)
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

        self.requested_bytes = self.requested_bytes.saturating_sub(credit_bytes);
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
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::Unbounded,
            true,
        )
    }

    /// Estimate a graph fragment for strict bounded admission.
    ///
    /// Similarity requires plan-exact or hard-upper-bound pair cardinality
    /// evidence here.  A count obtained by executing the pair relation is
    /// deliberately rejected, because it is post-admission runtime evidence
    /// and cannot be retroactively promoted to an admission proof.
    pub(crate) fn estimate_for_bounded(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
    ) -> Result<GraphMemoryEstimate> {
        require_strict_similarity_pair_cardinality_evidence(spec, input)?;
        require_graph_construction_workspace_preflight(spec, input)?;
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::Bounded,
            false,
        )
    }

    fn estimate_with_pair_cardinality_admission_mode(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
        pair_cardinality_admission_mode: GraphPairCardinalityAdmissionMode,
        diagnostic_projection: bool,
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
        let endpoint_bound = model_product(input.input_edges, 2, "vertex_upper_bound")?;
        let _ = project_model_bytes(endpoint_bound, "vertex_upper_bound", diagnostic_projection)?;
        let (vertex_estimate_model, vertex_estimate_source) = input
            .vertex_estimate
            .map(|estimate| (ModelBytes::from_u64(estimate.count).min_model(endpoint_bound), estimate.source))
            .unwrap_or((endpoint_bound, GraphVertexEstimateSource::EdgeEndpointUpperBound));
        let vertex_estimate = project_model_bytes(vertex_estimate_model, "vertex_estimate", diagnostic_projection)?;
        let derived = derive_graph_terms(spec, input, vertex_estimate, pair_rows, pair_vertex_width_bytes)?;
        let execution_components = derived.execution_workspace.add(derived.output, "execution_output_workspace")?;
        if execution_components != derived.execution {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "graph execution owner components do not match the inclusive execution estimate",
            )
            .with_fact("reason", "graph_memory_owner_component_mismatch"));
        }
        let construction_components =
            derived.graph_persistent.add(derived.construction_transient, "construction_persistent_transient")?;
        if construction_components != derived.construction {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "graph construction owner components do not match the inclusive construction estimate",
            )
            .with_fact("reason", "graph_memory_owner_component_mismatch"));
        }
        let owner_components = GraphMemoryOwnerComponents::from_derived(derived, diagnostic_projection)?;
        let output_row_count_upper_bound = derived.output_rows;
        let retained_output_bytes = GraphMemoryRetainedOutputContribution::new(
            project_model_bytes(derived.output, "output_bytes", diagnostic_projection)?,
            output_row_count_upper_bound,
        );
        let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
        let seed_count = seed_input.map(|seed_input| seed_input.requested_count);
        let seed_mode = seed_input.map(|seed_input| seed_input.mode);
        let calibration_source = graph_calibration_source(input);

        Ok(GraphMemoryEstimate {
            algorithm: spec.algorithm,
            class,
            raw_input_edges: input.raw_input_edges,
            input_edges: input.input_edges,
            vertex_upper_bound: project_model_bytes(endpoint_bound, "vertex_upper_bound", diagnostic_projection)?,
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
            construction_bytes: project_model_bytes(derived.construction, "construction_bytes", diagnostic_projection)?,
            execution_bytes: project_model_bytes(derived.execution, "execution_bytes", diagnostic_projection)?,
            retained_output_bytes,
            requested_bytes: project_model_bytes(derived.requested, "requested_bytes", diagnostic_projection)?,
            owner_components,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelBytes(u128);

impl ModelBytes {
    fn from_u64(value: u64) -> Self {
        Self(u128::from(value))
    }

    fn add(self, other: Self, term: &'static str) -> Result<Self> {
        self.0.checked_add(other.0).map(Self).ok_or_else(|| graph_memory_arithmetic_error(term, "addition"))
    }

    fn mul(self, other: Self, term: &'static str) -> Result<Self> {
        self.0.checked_mul(other.0).map(Self).ok_or_else(|| graph_memory_arithmetic_error(term, "multiplication"))
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min_model(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn diagnostic_u64(self) -> u64 {
        self.0.min(u128::from(u64::MAX)) as u64
    }

    fn bounded_u64(self, term: &'static str) -> Result<u64> {
        if self.0 >= u128::from(u64::MAX) {
            return Err(graph_memory_arithmetic_error(term, "saturated"));
        }
        u64::try_from(self.0).map_err(|_| graph_memory_arithmetic_error(term, "u128_to_u64"))
    }
}

fn graph_memory_arithmetic_error(term: &'static str, operation: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.arithmetic",
        "graph memory bounded arithmetic is not finite",
    )
    .with_fact("reason", "graph_memory_arithmetic_overflow")
    .with_fact("term", term)
    .with_fact("operation", operation)
}

fn project_model_bytes(value: ModelBytes, term: &'static str, diagnostic: bool) -> Result<u64> {
    if diagnostic { Ok(value.diagnostic_u64()) } else { value.bounded_u64(term) }
}

fn graph_live_reservation_floor_top_up_bytes(cold_model_requested_bytes: u64) -> Result<Option<u64>> {
    if cold_model_requested_bytes == u64::MAX {
        return Err(graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "saturated"));
    }
    if cold_model_requested_bytes >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES {
        return Ok(None);
    }
    GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES
        .checked_sub(cold_model_requested_bytes)
        .map(Some)
        .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "subtraction"))
}

#[derive(Debug, Clone, Copy)]
struct GraphDerivedTerms {
    construction: ModelBytes,
    graph_persistent: ModelBytes,
    construction_transient: ModelBytes,
    output: ModelBytes,
    execution_workspace: ModelBytes,
    execution: ModelBytes,
    requested: ModelBytes,
    output_rows: u64,
}

impl GraphMemoryOwnerComponents {
    fn from_derived(derived: GraphDerivedTerms, diagnostic_projection: bool) -> Result<Self> {
        Ok(Self {
            graph_persistent_bytes: project_model_bytes(
                derived.graph_persistent,
                "graph_persistent_bytes",
                diagnostic_projection,
            )?,
            construction_transient_bytes: project_model_bytes(
                derived.construction_transient,
                "construction_transient_bytes",
                diagnostic_projection,
            )?,
            execution_workspace_bytes: project_model_bytes(
                derived.execution_workspace,
                "execution_workspace_bytes",
                diagnostic_projection,
            )?,
        })
    }

    fn owner_terms(
        self,
        execution_scope_fingerprint: String,
        graph_node_id: NodeId,
        retained_output: GraphMemoryRetainedOutputContribution,
        live_reservation_floor_top_up_bytes: Option<u64>,
    ) -> Result<Vec<GraphMemoryOwnerTermFact>> {
        let mut terms = vec![
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::GraphPersistent,
                self.graph_persistent_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::ConstructionTransient,
                self.construction_transient_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::ExecutionWorkspace,
                self.execution_workspace_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::RetainedOutput,
                retained_output.bytes(),
                Some(retained_output.row_count_upper_bound()),
            )?,
        ];
        if let Some(bytes) = live_reservation_floor_top_up_bytes {
            terms.push(GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint,
                graph_node_id,
                GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp,
                bytes,
                None,
            )?);
        }
        Ok(terms)
    }
}

fn checked_add_all(values: impl IntoIterator<Item = (ModelBytes, &'static str)>) -> Result<ModelBytes> {
    values.into_iter().try_fold(ModelBytes(0), |total, (value, term)| total.add(value, term))
}

fn model_product(left: u64, right: u64, term: &'static str) -> Result<ModelBytes> {
    ModelBytes::from_u64(left).mul(ModelBytes::from_u64(right), term)
}

fn construction_multiplier(policy: GraphConstructionPolicy, directed: bool) -> u64 {
    match (policy, directed) {
        (GraphConstructionPolicy::PythonCugraph, false) => 3,
        (GraphConstructionPolicy::PythonCugraph, true) => 2,
        (GraphConstructionPolicy::RawLibcugraph, _) => 1,
    }
}

fn derive_graph_terms(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
    pair_vertex_width_bytes: Option<u64>,
) -> Result<GraphDerivedTerms> {
    let edge_row_width = semantic_edge_row_width_model(input)?;
    let normalized_edge = ModelBytes::from_u64(input.input_edges).mul(edge_row_width, "normalized_edge_bytes")?;
    let construction_base = normalized_edge.mul(
        ModelBytes::from_u64(construction_multiplier(spec.graph_options.policy, spec.graph_options.directed)),
        "construction_base",
    )?;
    let normalization = if input.edge_normalize_inserted { normalized_edge } else { ModelBytes(0) };
    let retained_normalization_source = normalization;
    let live_graph_inputs = normalization.add(retained_normalization_source, "live_graph_inputs")?;
    let construction_transient = if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        let unnormalized_input_copy = if input.edge_normalize_inserted { ModelBytes(0) } else { normalized_edge };
        let renumber_workspace = ModelBytes::from_u64(graph_construction_renumber_workspace_bound_bytes(spec, input));
        let compression_workspace =
            ModelBytes::from_u64(graph_construction_compression_workspace_bound_bytes(spec, input));
        let property_output_staging = raw_libcugraph_property_output_staging(input)?;
        let renumber_phase = if spec.graph_options.renumber {
            checked_add_all([
                (live_graph_inputs, "renumber_live_graph_inputs"),
                (unnormalized_input_copy, "renumber_unnormalized_input_copy"),
                (renumber_workspace, "renumber_workspace"),
            ])?
        } else {
            live_graph_inputs.add(unnormalized_input_copy, "construction_live_graph_inputs")?
        };
        let retained_number_map = if spec.graph_options.renumber {
            model_product(vertex_estimate, input.vertex_width_bytes, "retained_number_map")?
        } else {
            ModelBytes(0)
        };
        let compression_phase = checked_add_all([
            (live_graph_inputs, "compression_live_graph_inputs"),
            (unnormalized_input_copy, "compression_unnormalized_input_copy"),
            (retained_number_map, "retained_number_map"),
            (property_output_staging, "property_output_staging"),
            (compression_workspace, "compression_workspace"),
        ])?;
        renumber_phase.max(compression_phase)
    } else {
        let renumber_sort_staging = model_product(
            input.input_edges,
            input
                .vertex_width_bytes
                .checked_mul(2)
                .ok_or_else(|| graph_memory_arithmetic_error("construction_sort_staging_width", "multiplication"))?,
            "renumber_sort_staging",
        )?;
        let renumber_workspace = if spec.graph_options.renumber {
            model_product(
                vertex_estimate,
                input
                    .vertex_width_bytes
                    .checked_mul(3)
                    .ok_or_else(|| graph_memory_arithmetic_error("renumber_staging_width", "multiplication"))?,
                "renumber_staging",
            )?
        } else {
            ModelBytes(0)
        };
        checked_add_all([
            (live_graph_inputs, "live_graph_inputs"),
            (renumber_sort_staging, "renumber_sort_staging"),
            (renumber_workspace, "renumber_staging"),
        ])?
    };
    let construction = construction_base.add(construction_transient, "construction")?;
    let output = output_bytes_model(spec.algorithm, input, vertex_estimate, pair_rows)?;
    let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
    let workspace = workspace_bytes_model(
        spec.algorithm,
        &spec.params,
        graph_memory_class(spec.algorithm),
        input,
        vertex_estimate,
        seed_input,
        pair_rows,
        pair_vertex_width_bytes,
    )?;
    let execution = output.add(workspace, "execution_output_workspace")?;
    let requested = match spec.algorithm {
        GraphAlgorithm::ForceAtlas2 => {
            force_atlas2_phase_peak_model(input, vertex_estimate, &spec.params, construction, output)?
        },
        GraphAlgorithm::PageRank => construction
            .max(
                live_graph_inputs
                    .add(construction_base, "persistent_graph_inputs")?
                    .add(execution, "persistent_graph_execution")?,
            )
            .max(ModelBytes(1)),
        GraphAlgorithm::Bfs if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph => construction
            .max(
                live_graph_inputs
                    .add(construction_base, "persistent_graph_inputs")?
                    .add(execution, "persistent_graph_execution")?,
            )
            .max(ModelBytes(1)),
        _ => construction.add(execution, "requested_construction_execution")?.max(ModelBytes(1)),
    };
    Ok(GraphDerivedTerms {
        construction,
        graph_persistent: construction_base,
        construction_transient,
        output,
        execution_workspace: workspace,
        execution,
        requested,
        output_rows: output_row_count_upper_bound_for_algorithm(spec.algorithm, input, vertex_estimate, pair_rows),
    })
}

fn semantic_edge_row_width_model(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    checked_add_all([
        (
            ModelBytes::from_u64(input.vertex_width_bytes).mul(ModelBytes::from_u64(2), "edge_endpoint_width")?,
            "edge_endpoint_width",
        ),
        (ModelBytes::from_u64(input.weight_width_bytes.unwrap_or(0)), "edge_weight_width"),
        (
            if input.has_edge_id { ModelBytes::from_u64(input.vertex_width_bytes) } else { ModelBytes(0) },
            "edge_id_width",
        ),
    ])
}

fn raw_libcugraph_property_output_staging(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    let property_width = match (input.weight_width_bytes, input.has_edge_id) {
        (None, false) => 0,
        (Some(width), false) => width,
        (None, true) => input.vertex_width_bytes,
        (Some(_), true) => input.vertex_width_bytes,
    };
    model_product(input.input_edges, property_width, "property_output_staging")
}

fn output_bytes_model(
    algorithm: GraphAlgorithm,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
) -> Result<ModelBytes> {
    let (rows, width) = match algorithm {
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            (pair_rows.unwrap_or(0), 24)
        },
        GraphAlgorithm::EdgeBetweennessCentrality => {
            (input.input_edges, edge_betweenness_output_row_width_model(input)?)
        },
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => (input.input_edges, 16),
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll | GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => {
            (vertex_estimate, 24)
        },
        GraphAlgorithm::ForceAtlas2 => (vertex_estimate, 16),
        _ => (vertex_estimate, 16),
    };
    model_product(rows, width, "output_bytes")
}

fn edge_betweenness_output_row_width_model(input: GraphMemoryModelInput) -> Result<u64> {
    let vertex = input.vertex_width_bytes.max(8);
    vertex
        .checked_mul(2)
        .and_then(|value| value.checked_add(8))
        .and_then(|value| if input.has_edge_id { value.checked_add(vertex) } else { Some(value) })
        .ok_or_else(|| graph_memory_arithmetic_error("edge_betweenness_output_row_width", "addition"))
}

fn workspace_bytes_model(
    algorithm: GraphAlgorithm,
    params: &GraphAlgorithmParams,
    class: GraphMemoryClass,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
    pair_rows: Option<u64>,
    pair_vertex_width_bytes: Option<u64>,
) -> Result<ModelBytes> {
    let vertex_array = model_product(vertex_estimate, input.vertex_width_bytes.max(8), "workspace_vertex_array")?;
    let edge_index = model_product(
        input.input_edges,
        input
            .vertex_width_bytes
            .checked_mul(3)
            .ok_or_else(|| graph_memory_arithmetic_error("workspace_edge_index_width", "multiplication"))?,
        "workspace_edge_index",
    )?;
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "workspace_output_arrays")?
        .add(edge_index, "workspace_output_sized")?;
    match class {
        GraphMemoryClass::OutputSized => Ok(output_sized),
        GraphMemoryClass::MultiArray => output_sized.mul(ModelBytes::from_u64(4), "multi_array_workspace"),
        GraphMemoryClass::WorkspaceHeavyDataDependent => match algorithm {
            GraphAlgorithm::TriangleCountAll => triangle_count_workspace_model(input, vertex_estimate),
            GraphAlgorithm::ForceAtlas2 => force_atlas2_workspace_model(input, vertex_estimate, params, output_sized),
            GraphAlgorithm::SpectralModularityMaximization => {
                legacy_workspace_heavy_model(vertex_estimate, output_sized)
            },
            _ => Ok(output_sized),
        },
        GraphMemoryClass::WorkspaceHeavySeedDependent => match algorithm {
            GraphAlgorithm::BetweennessCentrality => output_sized.add(
                vertex_betweenness_workspace_model(input, vertex_estimate, seed_input)?,
                "vertex_betweenness_workspace",
            ),
            GraphAlgorithm::EdgeBetweennessCentrality => output_sized.add(
                edge_betweenness_workspace_model(input, vertex_estimate, seed_input)?,
                "edge_betweenness_workspace",
            ),
            _ => Ok(output_sized),
        },
        GraphMemoryClass::PairOutputSized => similarity_workspace_model(
            pair_rows.expect("similarity memory evidence validated before workspace pricing"),
            pair_vertex_width_bytes.expect("similarity memory evidence validated before workspace pricing"),
            input.weight_width_bytes.is_some(),
        ),
    }
}

fn legacy_workspace_heavy_model(vertex_estimate: u64, output_sized: ModelBytes) -> Result<ModelBytes> {
    let probe = model_product(vertex_estimate, vertex_estimate.min(1024), "spectral_probe")?
        .mul(ModelBytes::from_u64(4), "spectral_probe_scale")?;
    output_sized.mul(ModelBytes::from_u64(8), "spectral_output_workspace")?.add(probe, "spectral_workspace")
}

fn similarity_workspace_model(rows: u64, width: u64, weighted: bool) -> Result<ModelBytes> {
    let pair_width =
        width.checked_mul(2).ok_or_else(|| graph_memory_arithmetic_error("similarity_pair_width", "multiplication"))?;
    let pair_columns = model_product(rows, pair_width, "similarity_pair_columns")?;
    let scores = model_product(rows, SIMILARITY_NATIVE_SCORE_WIDTH_BYTES, "similarity_scores")?;
    let kernel = ModelBytes::from_u64(SIMILARITY_WORKSPACE_FIXED_BYTES).add(
        model_product(rows, similarity_workspace_per_pair_bytes(width, weighted), "similarity_kernel")?,
        "similarity_kernel_total",
    )?;
    checked_add_all([
        (pair_columns, "similarity_pair_input"),
        (pair_columns, "similarity_native_pairs"),
        (scores, "similarity_scores"),
        (kernel, "similarity_kernel"),
    ])
}

fn force_atlas2_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: ModelBytes,
) -> Result<ModelBytes> {
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)?;
    checked_add_all([
        (phase.persistent_csr_and_decompressed_coo_bytes, "force_atlas2_persistent_graph"),
        (phase.barnes_hut_workspace_bytes, "force_atlas2_barnes_hut"),
        (phase.legacy_coo_sort_workspace_bytes, "force_atlas2_coo_sort"),
    ])
}

#[derive(Debug, Clone, Copy)]
struct ForceAtlas2WorkspacePhaseModel {
    persistent_csr_and_decompressed_coo_bytes: ModelBytes,
    persistent_number_map_and_positions_bytes: ModelBytes,
    legacy_coo_sort_workspace_bytes: ModelBytes,
    barnes_hut_workspace_bytes: ModelBytes,
}

fn force_atlas2_workspace_phase_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: ModelBytes,
) -> Result<ForceAtlas2WorkspacePhaseModel> {
    let barnes_hut = match params {
        GraphAlgorithmParams::ForceAtlas2(params) if params.barnes_hut_optimize => checked_add_all([
            (model_product(vertex_estimate, 120, "force_atlas2_barnes_hut")?, "force_atlas2_barnes_hut"),
            (
                model_product(
                    vertex_estimate,
                    input
                        .vertex_width_bytes
                        .checked_mul(2)
                        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_degree_width", "multiplication"))?,
                    "force_atlas2_degree",
                )?,
                "force_atlas2_degree",
            ),
        ])?,
        GraphAlgorithmParams::ForceAtlas2(_) => model_product(vertex_estimate, 44, "force_atlas2_exact")?,
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "ForceAtlas2 memory model received incompatible parameters",
            ));
        },
    };
    let position_width = input
        .vertex_width_bytes
        .checked_add(8)
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_position_width", "addition"))?;
    let coo_record_width = input
        .vertex_width_bytes
        .checked_mul(2)
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_coo_width", "multiplication"))?
        .checked_add(input.weight_width_bytes.unwrap_or(4))
        .and_then(|width| width.checked_add(1))
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_coo_record", "addition"))?;
    Ok(ForceAtlas2WorkspacePhaseModel {
        persistent_csr_and_decompressed_coo_bytes: output_sized,
        persistent_number_map_and_positions_bytes: model_product(
            vertex_estimate,
            position_width,
            "force_atlas2_positions",
        )?,
        legacy_coo_sort_workspace_bytes: model_product(input.input_edges, coo_record_width, "force_atlas2_coo_sort")?,
        barnes_hut_workspace_bytes: barnes_hut,
    })
}

fn force_atlas2_phase_peak_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    construction: ModelBytes,
    result_conversion: ModelBytes,
) -> Result<ModelBytes> {
    let vertex_array = model_product(vertex_estimate, input.vertex_width_bytes.max(8), "force_atlas2_vertex_array")?;
    let edge_index = model_product(
        input.input_edges,
        input
            .vertex_width_bytes
            .checked_mul(3)
            .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_edge_index_width", "multiplication"))?,
        "force_atlas2_edge_index",
    )?;
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "force_atlas2_output_arrays")?
        .add(edge_index, "force_atlas2_output_sized")?;
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)?;
    let property_payload =
        model_product(input.input_edges, input.weight_width_bytes.unwrap_or(4), "force_atlas2_property_payload")?;
    let base = checked_add_all([
        (construction, "force_atlas2_construction"),
        (property_payload, "force_atlas2_property_payload"),
        (phase.persistent_csr_and_decompressed_coo_bytes, "force_atlas2_persistent_graph"),
        (phase.persistent_number_map_and_positions_bytes, "force_atlas2_positions"),
    ])?;
    let sort_peak = base.add(phase.legacy_coo_sort_workspace_bytes, "force_atlas2_sort_peak")?;
    let barnes_peak = base.add(phase.barnes_hut_workspace_bytes, "force_atlas2_barnes_peak")?;
    let result_peak = base.add(result_conversion, "force_atlas2_result_peak")?;
    Ok(sort_peak.max(barnes_peak).max(result_peak).max(ModelBytes(1)))
}

#[cfg(test)]
fn force_atlas2_workspace_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: u64,
) -> u64 {
    force_atlas2_workspace_model(input, vertex_estimate, params, ModelBytes::from_u64(output_sized))
        .expect("test ForceAtlas2 input is finite")
        .diagnostic_u64()
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ForceAtlas2WorkspacePhaseBytes {
    persistent_csr_and_decompressed_coo_bytes: u64,
    persistent_number_map_and_positions_bytes: u64,
    legacy_coo_sort_workspace_bytes: u64,
    barnes_hut_workspace_bytes: u64,
}

#[cfg(test)]
fn force_atlas2_workspace_phase_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: u64,
) -> ForceAtlas2WorkspacePhaseBytes {
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, ModelBytes::from_u64(output_sized))
        .expect("test ForceAtlas2 input is finite");
    ForceAtlas2WorkspacePhaseBytes {
        persistent_csr_and_decompressed_coo_bytes: phase.persistent_csr_and_decompressed_coo_bytes.diagnostic_u64(),
        persistent_number_map_and_positions_bytes: phase.persistent_number_map_and_positions_bytes.diagnostic_u64(),
        legacy_coo_sort_workspace_bytes: phase.legacy_coo_sort_workspace_bytes.diagnostic_u64(),
        barnes_hut_workspace_bytes: phase.barnes_hut_workspace_bytes.diagnostic_u64(),
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ForceAtlas2PhasePeaks {
    persistent_base_bytes: u64,
    legacy_coo_sort_before_degree_bytes: u64,
    barnes_hut_with_degree_bytes: u64,
    post_barnes_result_conversion_bytes: u64,
}

#[cfg(test)]
impl ForceAtlas2PhasePeaks {
    fn peak_bytes(self) -> u64 {
        self.legacy_coo_sort_before_degree_bytes
            .max(self.barnes_hut_with_degree_bytes)
            .max(self.post_barnes_result_conversion_bytes)
            .max(1)
    }
}

#[cfg(test)]
fn force_atlas2_phase_peaks(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    construction_bytes: u64,
    result_conversion_bytes: u64,
) -> ForceAtlas2PhasePeaks {
    let vertex_array =
        model_product(vertex_estimate, input.vertex_width_bytes.max(8), "test_force_atlas2_vertex_array")
            .expect("test ForceAtlas2 input is finite");
    let edge_index = model_product(
        input.input_edges,
        input.vertex_width_bytes.checked_mul(3).expect("test ForceAtlas2 width is finite"),
        "test_force_atlas2_edge_index",
    )
    .expect("test ForceAtlas2 input is finite");
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "test_force_atlas2_output_arrays")
        .and_then(|value| value.add(edge_index, "test_force_atlas2_output_sized"))
        .expect("test ForceAtlas2 input is finite");
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)
        .expect("test ForceAtlas2 input is finite");
    let base = checked_add_all([
        (ModelBytes::from_u64(construction_bytes), "test_force_atlas2_construction"),
        (
            model_product(input.input_edges, input.weight_width_bytes.unwrap_or(4), "test_force_atlas2_property")
                .expect("test ForceAtlas2 input is finite"),
            "test_force_atlas2_property",
        ),
        (phase.persistent_csr_and_decompressed_coo_bytes, "test_force_atlas2_graph"),
        (phase.persistent_number_map_and_positions_bytes, "test_force_atlas2_positions"),
    ])
    .expect("test ForceAtlas2 input is finite");
    ForceAtlas2PhasePeaks {
        persistent_base_bytes: base.diagnostic_u64(),
        legacy_coo_sort_before_degree_bytes: base
            .add(phase.legacy_coo_sort_workspace_bytes, "test_force_atlas2_sort")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
        barnes_hut_with_degree_bytes: base
            .add(phase.barnes_hut_workspace_bytes, "test_force_atlas2_barnes")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
        post_barnes_result_conversion_bytes: base
            .add(ModelBytes::from_u64(result_conversion_bytes), "test_force_atlas2_result")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
    }
}

fn triangle_count_workspace_model(input: GraphMemoryModelInput, vertex_estimate: u64) -> Result<ModelBytes> {
    let width = input.vertex_width_bytes.max(1);
    if let Some(statistics) = input.statistics
        && let (Some(oriented_edges), Some(peak_min_degree)) =
            (statistics.oriented_edge_count, statistics.peak_chunk_min_degree_sum)
    {
        return checked_add_all([
            (
                model_product(oriented_edges, width, "triangle_dfg_edges")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_DFG_REBUILD_VERTEX_COLUMNS), "triangle_dfg_columns")?,
                "triangle_dfg",
            ),
            (
                model_product(oriented_edges, width, "triangle_pair_edges")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_PAIR_LIST_VERTEX_COLUMNS), "triangle_pair_columns")?,
                "triangle_pairs",
            ),
            (
                model_product(peak_min_degree, width, "triangle_intersection")?.mul(
                    ModelBytes::from_u64(TRIANGLE_COUNT_PEAK_INTERSECTION_VERTEX_COLUMNS),
                    "triangle_intersection_buffers",
                )?,
                "triangle_intersection",
            ),
            (
                model_product(statistics.exact_vertices, width.max(8), "triangle_vertex_scratch")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS), "triangle_vertex_columns")?,
                "triangle_vertex_scratch",
            ),
            (
                model_product(input.input_edges, width, "triangle_sort")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_SORT_STAGING_VERTEX_COLUMNS), "triangle_sort_columns")?,
                "triangle_sort",
            ),
        ]);
    }
    checked_add_all([
        (
            model_product(input.input_edges, width, "triangle_no_stats_edges")?.mul(
                ModelBytes::from_u64(TRIANGLE_COUNT_NO_STATS_EDGE_VERTEX_COLUMNS),
                "triangle_no_stats_edge_columns",
            )?,
            "triangle_no_stats_edges",
        ),
        (
            model_product(vertex_estimate, width.max(8), "triangle_no_stats_vertices")?
                .mul(ModelBytes::from_u64(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS), "triangle_no_stats_vertex_columns")?,
            "triangle_no_stats_vertices",
        ),
    ])
}

fn vertex_betweenness_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    let requested_sources = seed_input.map_or(vertex_estimate, |seed| seed.requested_count);
    let concurrent_sources =
        vertex_betweenness_concurrent_source_count_model(input, vertex_estimate, requested_sources)?;
    checked_add_all([
        (
            model_product(concurrent_sources, vertex_estimate, "betweenness_source_count")?
                .mul(vertex_betweenness_state_width_model(input)?, "betweenness_source_state")?,
            "betweenness_source_state",
        ),
        (betweenness_push_batch_model(input, concurrent_sources)?, "betweenness_push_batch"),
        (explicit_seed_model(input, seed_input)?, "explicit_seed_list"),
        (runtime_seed_model(input, vertex_estimate, seed_input)?, "runtime_seed_selection"),
    ])
}

fn edge_betweenness_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    checked_add_all([
        (
            model_product(vertex_estimate, edge_betweenness_state_width_model(input)?, "edge_betweenness_state")?,
            "edge_betweenness_state",
        ),
        (betweenness_push_batch_model(input, 1)?, "betweenness_push_batch"),
        (explicit_seed_model(input, seed_input)?, "explicit_seed_list"),
        (runtime_seed_model(input, vertex_estimate, seed_input)?, "runtime_seed_selection"),
    ])
}

fn betweenness_push_batch_model(input: GraphMemoryModelInput, resident_source_copies: u64) -> Result<ModelBytes> {
    let vertex_width = input.vertex_width_bytes.max(8);
    let tuple_width = vertex_width
        .checked_add(BETWEENNESS_SOURCE_ORIGIN_WIDTH_BYTES)
        .and_then(|value| value.checked_add(vertex_width))
        .ok_or_else(|| graph_memory_arithmetic_error("betweenness_tuple_width", "addition"))?;
    let pushes = input.statistics.and_then(|statistics| statistics.sum_out_degree).map_or_else(
        || {
            input
                .input_edges
                .checked_mul(resident_source_copies.max(1))
                .ok_or_else(|| graph_memory_arithmetic_error("betweenness_pushes", "multiplication"))
        },
        Ok,
    )?;
    let resident_pushes = input.device_total_memory_bytes.map_or(pushes, |total| {
        let cap = total / BETWEENNESS_PUSH_TUPLE_DEVICE_MEMORY_DENOMINATOR / tuple_width;
        pushes.min(cap.max(1))
    });
    model_product(resident_pushes, tuple_width, "betweenness_push_batch")
}

fn explicit_seed_model(input: GraphMemoryModelInput, seed_input: Option<GraphSeedMemoryInput>) -> Result<ModelBytes> {
    match seed_input {
        Some(seed) if seed.explicit_seed_list => {
            model_product(seed.requested_count, input.vertex_width_bytes, "explicit_seed_list")
        },
        _ => Ok(ModelBytes(0)),
    }
}

fn runtime_seed_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    let Some(seed) = seed_input else {
        return Ok(ModelBytes(0));
    };
    if !seed.requires_runtime_seed_selection {
        return Ok(ModelBytes(0));
    }
    let vertex_width = input.vertex_width_bytes.max(8);
    let endpoint_copy = model_product(input.input_edges, vertex_width, "runtime_seed_endpoint")?
        .mul(ModelBytes::from_u64(4), "runtime_seed_endpoint_copies")?;
    let distinct_sort = model_product(vertex_estimate, vertex_width, "runtime_seed_distinct")?
        .mul(ModelBytes::from_u64(3), "runtime_seed_distinct_buffers")?;
    endpoint_copy.add(distinct_sort, "runtime_seed_selection")
}

fn vertex_betweenness_concurrent_source_count_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    requested_sources: u64,
) -> Result<u64> {
    let mut sources = requested_sources.clamp(1, BETWEENNESS_MAX_CONCURRENT_SOURCES);
    let Some(total_memory_bytes) = input.device_total_memory_bytes else {
        return Ok(sources);
    };
    if sources <= 1 || vertex_estimate == 0 {
        return Ok(sources);
    }
    let per_source = model_product(
        vertex_estimate,
        input
            .vertex_width_bytes
            .checked_add(input.vertex_width_bytes)
            .ok_or_else(|| graph_memory_arithmetic_error("betweenness_bfs_result_width", "addition"))?,
        "betweenness_bfs_result_bytes",
    )?;
    if per_source.0 == 0 {
        return Ok(sources);
    }
    let physical_batch_cap = u128::from(total_memory_bytes) / 4 / per_source.0;
    sources = sources.min(u64::try_from(physical_batch_cap).unwrap_or(u64::MAX)).max(1);
    Ok(sources)
}

fn vertex_betweenness_state_width_model(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    checked_add_all([
        (ModelBytes::from_u64(input.vertex_width_bytes.max(8)), "betweenness_vertex_width"),
        (ModelBytes::from_u64(input.vertex_width_bytes.max(8)), "betweenness_edge_width"),
        (ModelBytes::from_u64(input.weight_width_bytes.unwrap_or(8).max(8)), "betweenness_weight_width"),
        (model_product(input.vertex_width_bytes.max(8), 2, "betweenness_vertex_state")?, "betweenness_vertex_state"),
        (ModelBytes::from_u64(4), "betweenness_origin_state"),
    ])
}

fn edge_betweenness_state_width_model(input: GraphMemoryModelInput) -> Result<u64> {
    input
        .vertex_width_bytes
        .max(8)
        .checked_add(input.vertex_width_bytes.max(8))
        .and_then(|value| value.checked_add(input.weight_width_bytes.unwrap_or(8).max(8)))
        .ok_or_else(|| graph_memory_arithmetic_error("edge_betweenness_state_width", "addition"))
}

/// Returns the model-owned row bound represented by a graph result.
///
/// Every supported graph algorithm emits one result row per vertex/edge/pair
/// (depending on its output contract).  Keeping this cardinality beside the
/// retained byte contribution lets an engine-owned consumer proof derive a
/// downstream output bound without asking an adapter to guess bytes or rows.
fn output_row_count_upper_bound_for_algorithm(
    algorithm: GraphAlgorithm,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
) -> u64 {
    match algorithm {
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            pair_rows.unwrap_or(0)
        },
        GraphAlgorithm::EdgeBetweennessCentrality | GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => {
            input.input_edges
        },
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll | GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => {
            vertex_estimate
        },
        _ => vertex_estimate,
    }
}

#[cfg(any(feature = "cugraph", test))]
fn semantic_edge_row_width_bytes(vertex_width_bytes: u64, weight_width_bytes: Option<u64>, has_edge_id: bool) -> u64 {
    semantic_edge_row_width_model(GraphMemoryModelInput {
        vertex_width_bytes,
        weight_width_bytes,
        has_edge_id,
        ..graph_memory_model_input(
            0,
            vertex_width_bytes,
            &GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            false,
            None,
        )
    })
    .map(ModelBytes::diagnostic_u64)
    .unwrap_or(u64::MAX)
}

fn graph_construction_renumber_workspace_bound_bytes(spec: &GraphAlgorithmSpec, input: GraphMemoryModelInput) -> u64 {
    if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        input.construction_workspace_bounds.map_or(0, GraphConstructionWorkspaceBounds::renumber_workspace_bytes)
    } else {
        0
    }
}

fn graph_construction_compression_workspace_bound_bytes(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> u64 {
    if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        input.construction_workspace_bounds.map_or(0, GraphConstructionWorkspaceBounds::compression_workspace_bytes)
    } else {
        0
    }
}

fn require_graph_construction_workspace_preflight(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<()> {
    if spec.graph_options.policy != GraphConstructionPolicy::RawLibcugraph
        || input.construction_workspace_bounds.is_some()
    {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.estimate_for_bounded",
        "raw cuGraph construction lacks pre-execution workspace bounds",
    )
    .with_fact("reason", "graph_construction_workspace_preflight_unavailable")
    .with_fact("algorithm", spec.algorithm.as_str())
    .with_fact("construction_policy", spec.graph_options.policy.as_str())
    .with_fact("edge_normalize_inserted", input.edge_normalize_inserted))
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
    if !source.is_pre_admission_bound() {
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
    .with_fact("admission_mode", GraphPairCardinalityAdmissionMode::Bounded.as_str())
    .with_fact("reason", reason)
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
        construction_workspace_bounds: None,
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
/// pair-row bound is an admission failure: materializing the pair source first
/// would create post-admission evidence, not a pre-admission proof.
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
        #[cfg(feature = "cugraph")]
        {
            input = with_graph_construction_workspace_bounds_from_owner(&graph.spec, input);
        }
        let estimate = GraphMemoryModel.estimate_for_bounded(&graph.spec, input)?;
        estimates.push(GraphPlanMemoryEstimate { node_id: node.id, estimate });
    }
    Ok(estimates)
}

/// Derive one opaque graph admission evidence token for an exact lowered graph
/// execution plan. The plan must contain exactly one graph algorithm node;
/// ambiguous multi-graph plans cannot be attached to one capability proof.
pub fn graph_memory_bounded_admission_evidence_for_plan(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryAdmissionEvidence> {
    let mut estimates = graph_memory_estimates_for_plan(plan, calibration_override, device_total_memory_bytes)?;
    let Some(estimate) = (estimates.len() == 1).then(|| estimates.pop()).flatten() else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.admission_evidence",
            "bounded graph admission evidence requires exactly one graph node in the execution plan",
        )
        .with_fact("reason", if estimates.is_empty() { "missing_graph_scope" } else { "ambiguous_graph_scope" })
        .with_fact("graph_node_count", estimates.len()));
    };
    GraphMemoryAdmissionEvidence::from_bounded_plan_estimate(plan, estimate.node_id, &estimate.estimate)
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
    graph_memory_estimate_for_edge_list_with_vertex_estimate(
        spec,
        input_schema,
        input_edges,
        raw_input_edges,
        None,
        calibration_override,
        device_total_memory_bytes,
    )
}

/// Computes graph memory for one edge-list input, optionally carrying a
/// source-rowset-bound graph vertex estimate.
///
/// The estimate is an input to the engine-owned admission model, not an
/// externally supplied calibration. Frontends may provide it only from source
/// metadata bound to the exact edge rowset they lower.
pub fn graph_memory_estimate_for_edge_list_with_vertex_estimate(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
    vertex_estimate: Option<GraphVertexEstimate>,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryEstimate> {
    let mut input = graph_memory_model_input_for_edge_list(spec, input_schema, input_edges, raw_input_edges)?;
    if let Some(vertex_estimate) = vertex_estimate {
        input = input.with_vertex_estimate(vertex_estimate);
    }
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
    query_plan_node_hard_row_upper_bound(plan, plan.root_id())
}

pub(crate) fn query_plan_node_hard_row_upper_bound(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    hard_row_upper_bound_for_plan_node(plan, node_id).map(|(rows, _)| rows)
}

/// Returns a source-owned graph vertex upper bound when the entire native plan
/// preserves the edge-file rowset that carried the footer metadata.
///
/// Unlike [`query_plan_hard_row_upper_bound`], this intentionally refuses
/// filters, limits, and other row-changing nodes. A vertex bound is tied to an
/// exact source rowset; forwarding it through a transformed relation would
/// turn a source fact into detached query metadata.
pub fn query_plan_graph_vertex_upper_bound(plan: &QueryPlan) -> Option<u64> {
    graph_vertex_upper_bound_for_plan_node(plan, plan.root_id())
}

fn graph_memory_model_input_for_plan_node(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    calibration_override: GraphMemoryCalibrationOverride,
) -> Result<Option<GraphMemoryModelInput>> {
    let Some(edge_counts) = graph_plan_edge_counts(plan, graph.input, calibration_override.raw_input_edges)? else {
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
    if let Some(vertex_upper_bound) = edge_counts.graph_vertex_upper_bound {
        input = input.with_vertex_estimate(GraphVertexEstimate::graph_metadata(vertex_upper_bound));
    }
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

/// Queries the owning cuGraph host-only preflight for raw graph construction.
///
/// This deliberately maps every unavailable input or owner-query failure to
/// `None`: bounded admission then emits its existing structured preflight
/// rejection rather than converting an owner failure into a guessed byte
/// allowance. The graph definition supplies the actual transposed-layout
/// requirement used by the native constructor.
#[cfg(feature = "cugraph")]
pub(crate) fn with_graph_construction_workspace_bounds_from_owner(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> GraphMemoryModelInput {
    graph_construction_workspace_bounds_from_owner(spec, input)
        .map(|bounds| input.with_construction_workspace_bounds(bounds))
        .unwrap_or(input)
}

#[cfg(feature = "cugraph")]
fn graph_construction_workspace_bounds_from_owner(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Option<GraphConstructionWorkspaceBounds> {
    graph_construction_workspace_bounds_from_owner_query(spec, input, |preflight| {
        let result = cugraph::cudf_interop::graph_construction_workspace_preflight(preflight).ok()?;
        Some(GraphConstructionWorkspaceBounds::new(
            result.renumber_workspace_bytes(),
            result.compression_workspace_bytes(),
        ))
    })
}

/// Builds the exact owner preflight input, then projects a failed owner query
/// to absence so the bounded model retains the stable fail-closed rejection.
///
/// The query callback is deliberately narrow: it lets the no-bound outcome be
/// unit-tested without introducing an engine-side substitute for the owner
/// contract.
#[cfg(feature = "cugraph")]
fn graph_construction_workspace_bounds_from_owner_query(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    query: impl FnOnce(&cugraph::cudf_interop::GraphConstructionPreflightInput) -> Option<GraphConstructionWorkspaceBounds>,
) -> Option<GraphConstructionWorkspaceBounds> {
    if spec.graph_options.policy != GraphConstructionPolicy::RawLibcugraph {
        return None;
    }

    let vertex_type = match input.vertex_width_bytes {
        4 => cugraph::types::DataType::Int32,
        8 => cugraph::types::DataType::Int64,
        _ => return None,
    };
    let weight_type = match input.weight_width_bytes {
        None => None,
        Some(4) => Some(cugraph::types::DataType::Float32),
        Some(8) => Some(cugraph::types::DataType::Float64),
        Some(_) => return None,
    };
    let store_transposed =
        matches!(spec.algorithm.definition().layout_requirement, crate::plan::GraphLayoutRequirement::Transposed);
    let vertex_upper_bound = input.vertex_upper_bound().ok()?;
    let num_vertices =
        input.vertex_estimate.map_or(vertex_upper_bound, |estimate| estimate.count.min(vertex_upper_bound));
    let preflight = cugraph::cudf_interop::GraphConstructionPreflightInput::new(
        vertex_type,
        vertex_type,
        weight_type,
        input.has_edge_id,
        store_transposed,
        spec.graph_options.renumber,
        usize::try_from(input.input_edges).ok()?,
        usize::try_from(num_vertices).ok()?,
    )
    .ok()?;
    query(&preflight)
}

/// Returns only the pair-cardinality evidence that can prove strict bounded
/// similarity admission before the candidate relation is materialized.
///
/// This intentionally excludes runtime counts.  Callers that execute a
/// strict bounded graph node must invoke it before evaluating graph side
/// inputs, then price the graph reservation with the returned exact count or
/// upper bound.
pub(crate) fn strict_similarity_pair_cardinality_for_plan(
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
                let rows = source.exact_selected_row_count()?;
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
        PlanNode::Aggregate(aggregate) => hard_row_upper_bound_for_plan_node(plan, aggregate.input)
            .map(|(rows, _)| (rows, GraphPairCardinalitySource::PlanHardUpperBound)),
        PlanNode::Join(join) => {
            let left_rows = hard_row_upper_bound_for_plan_node(plan, join.inputs[0])?.0;
            let rows = match join.spec.join_type {
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => Some(left_rows),
                JoinType::Inner | JoinType::Cross => {
                    left_rows.checked_mul(hard_row_upper_bound_for_plan_node(plan, join.inputs[1])?.0)
                },
                JoinType::Left => {
                    left_rows.checked_mul(hard_row_upper_bound_for_plan_node(plan, join.inputs[1])?.0.max(1))
                },
                JoinType::Full => {
                    let right_rows = hard_row_upper_bound_for_plan_node(plan, join.inputs[1])?.0;
                    left_rows.checked_mul(right_rows)?.checked_add(left_rows)?.checked_add(right_rows)
                },
            }?;
            Some((rows, GraphPairCardinalitySource::PlanHardUpperBound))
        },
        PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) | PlanNode::VectorAlgorithm(_) => None,
    }
}

fn graph_vertex_upper_bound_for_plan_node(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => source.graph_vertex_upper_bound(),
            crate::source::SourceNode::GpuResident(source) => source.graph_vertex_upper_bound,
            crate::source::SourceNode::LiteralBatch(_) => None,
        },
        PlanNode::Projection(projection) => graph_vertex_upper_bound_for_plan_node(plan, projection.input),
        PlanNode::Window(window) => graph_vertex_upper_bound_for_plan_node(plan, window.input),
        PlanNode::Sort(sort) => graph_vertex_upper_bound_for_plan_node(plan, sort.input),
        PlanNode::EdgeNormalize(edge_normalize) => graph_vertex_upper_bound_for_plan_node(plan, edge_normalize.input),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| total.checked_add(graph_vertex_upper_bound_for_plan_node(plan, *input)?)),
        PlanNode::Filter(_)
        | PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Join(_)
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
    graph_vertex_upper_bound: Option<u64>,
}

fn graph_plan_edge_counts(
    plan: &QueryPlan,
    graph_input: NodeId,
    configured_raw_input_edges: Option<u64>,
) -> Result<Option<GraphPlanEdgeCounts>> {
    let Some(node) = plan.node(graph_input) else {
        return Ok(None);
    };
    if let PlanNode::EdgeNormalize(edge_normalize) = &node.kind {
        let Some(raw_input_edges) = hard_row_upper_bound_for_plan_node(plan, edge_normalize.input)
            .map(|(rows, _)| rows)
            .or(configured_raw_input_edges)
        else {
            return Ok(None);
        };
        return Ok(Some(GraphPlanEdgeCounts {
            raw_input_edges: Some(raw_input_edges),
            input_edges: edge_normalize_output_row_upper_bound(raw_input_edges, &edge_normalize.spec)?,
            graph_vertex_upper_bound: graph_vertex_upper_bound_for_plan_node(plan, edge_normalize.input),
        }));
    }
    let Some(input_edges) =
        hard_row_upper_bound_for_plan_node(plan, graph_input).map(|(rows, _)| rows).or(configured_raw_input_edges)
    else {
        return Ok(None);
    };
    Ok(Some(GraphPlanEdgeCounts {
        raw_input_edges: configured_raw_input_edges,
        input_edges,
        graph_vertex_upper_bound: graph_vertex_upper_bound_for_plan_node(plan, graph_input),
    }))
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
) -> Result<u64> {
    if sum_min_degree_oriented == 0 || oriented_edge_count == 0 {
        return Ok(0);
    }
    let sm_count = u64::from(device_multiprocessor_count.unwrap_or(1)).max(1);
    let max_chunk_size = sm_count
        .checked_mul(1 << 15)
        .ok_or_else(|| graph_memory_arithmetic_error("triangle_chunk_size", "multiplication"))?;
    let chunk_count = ceil_div_u64(oriented_edge_count, max_chunk_size)?.max(1);
    let average_chunk = ceil_div_u64(sum_min_degree_oriented, chunk_count)?;
    Ok(ceil_checked_ratio(
        average_chunk,
        TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR,
        TRIANGLE_COUNT_CHUNK_SAFETY_DENOMINATOR,
        "triangle_peak_chunk_min_degree",
    )?
    .min(sum_min_degree_oriented))
}

fn graph_calibration_source(input: GraphMemoryModelInput) -> GraphCalibrationSource {
    if input.raw_input_edges.is_some() { GraphCalibrationSource::RawInputEdges } else { GraphCalibrationSource::None }
}

fn ceil_div_u64(value: u64, denominator: u64) -> Result<u64> {
    if denominator == 0 {
        return Err(graph_memory_arithmetic_error("triangle_chunk_count", "division"));
    }
    let numerator = u128::from(value);
    let denominator = u128::from(denominator);
    u64::try_from(numerator.div_ceil(denominator))
        .map_err(|_| graph_memory_arithmetic_error("triangle_ceil_division", "u128_to_u64"))
}

fn ceil_checked_ratio(value: u64, numerator: u64, denominator: u64, term: &'static str) -> Result<u64> {
    if denominator == 0 {
        return Err(graph_memory_arithmetic_error(term, "division"));
    }
    let product = u128::from(value)
        .checked_mul(u128::from(numerator))
        .ok_or_else(|| graph_memory_arithmetic_error(term, "multiplication"))?;
    let rounded = product
        .checked_add(u128::from(denominator - 1))
        .ok_or_else(|| graph_memory_arithmetic_error(term, "addition"))?
        / u128::from(denominator);
    u64::try_from(rounded).map_err(|_| graph_memory_arithmetic_error(term, "u128_to_u64"))
}

fn edge_normalize_output_row_upper_bound(raw_input_edges: u64, spec: &EdgeNormalizeSpec) -> Result<u64> {
    raw_input_edges
        .checked_mul(if spec.add_reverse { 2 } else { 1 })
        .ok_or_else(|| graph_memory_arithmetic_error("edge_normalize_output_rows", "multiplication"))
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
    #[cfg(feature = "cugraph")]
    use crate::plan::{
        GraphEdgePredicateInput, GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
        GraphPredicateInput,
    };
    use crate::{
        expr::{AggregateSpec, ColumnRef, Expr, LiteralValue},
        plan::{
            GraphBfsParams, GraphConstructionOptions, GraphForceAtlas2Params, GraphPageRankParams,
            GraphSimilarityParams, GraphTriangleCountAllParams, JoinKey, JoinSpec, QueryPlanBuilder,
        },
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    fn edge_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    fn bounded_evidence_for_spec(spec: &GraphAlgorithmSpec, rows: u64) -> GraphMemoryAdmissionEvidence {
        bounded_evidence_for_spec_with_vertex_bound(spec, rows, None)
    }

    fn bounded_evidence_for_spec_with_vertex_bound(
        spec: &GraphAlgorithmSpec,
        rows: u64,
        graph_vertex_upper_bound: Option<u64>,
    ) -> GraphMemoryAdmissionEvidence {
        let mut builder = QueryPlanBuilder::new();
        let mut source =
            crate::source::GpuResidentSource::new("graph-test-input", edge_schema()).with_row_count_upper_bound(rows);
        if let Some(graph_vertex_upper_bound) = graph_vertex_upper_bound {
            source = source.with_graph_vertex_upper_bound(graph_vertex_upper_bound);
        }
        let source = builder.add_source(SourceNode::GpuResident(source));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_gpu_resident(graph);
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded graph test plan must produce evidence")
    }

    #[cfg(feature = "cugraph")]
    fn bfs_side_input_plan<F>(side_schema: SchemaRef, side_rows: Option<u64>, configure: F) -> QueryPlan
    where F: FnOnce(NodeId) -> GraphBfsParams {
        let mut builder = QueryPlanBuilder::new();
        let edge = builder.add_source(SourceNode::GpuResident(
            crate::source::GpuResidentSource::new("bfs-side-edges", edge_schema()).with_row_count_upper_bound(20),
        ));
        let mut side = crate::source::GpuResidentSource::new("bfs-side-input", side_schema);
        if let Some(side_rows) = side_rows {
            side = side.with_row_count_upper_bound(side_rows);
        }
        let side = builder.add_source(SourceNode::GpuResident(side));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(configure(side))),
        );
        let graph = builder.add_graph_algorithm(edge, spec);
        builder.finish_gpu_resident(graph)
    }

    #[cfg(feature = "cugraph")]
    fn personalized_pagerank_side_input_plan(side_rows: Option<u64>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let edge = builder.add_source(SourceNode::GpuResident(
            crate::source::GpuResidentSource::new("ppr-side-edges", edge_schema()).with_row_count_upper_bound(20),
        ));
        let side_schema = Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let mut side = crate::source::GpuResidentSource::new("ppr-side-input", side_schema);
        if let Some(side_rows) = side_rows {
            side = side.with_row_count_upper_bound(side_rows);
        }
        let side = builder.add_source(SourceNode::GpuResident(side));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PersonalizedPageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
                pagerank: GraphPageRankParams::default(),
                personalization: GraphPageRankPersonalizationInput::new(side, "vertex", "value"),
            }),
        );
        let graph = builder.add_graph_algorithm(edge, spec);
        builder.finish_gpu_resident(graph)
    }

    fn bounded_join_plan(
        left_rows: Option<u64>,
        right_rows: Option<u64>,
        join_type: JoinType,
        keyed: bool,
        residual: bool,
    ) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let mut left_source = GpuResidentSource::new("bounded-join-left", Arc::clone(&schema));
        if let Some(rows) = left_rows {
            left_source = left_source.with_row_count_upper_bound(rows);
        }
        let mut right_source = GpuResidentSource::new("bounded-join-right", Arc::clone(&schema));
        if let Some(rows) = right_rows {
            right_source = right_source.with_row_count_upper_bound(rows);
        }
        let left = builder.add_source(SourceNode::GpuResident(left_source));
        let right = builder.add_source(SourceNode::GpuResident(right_source));
        let keys = keyed
            .then(|| JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key")))
            .into_iter()
            .collect();
        let mut spec = JoinSpec::new(join_type, keys);
        if residual {
            spec = spec.with_residual_predicate(Expr::literal(LiteralValue::Boolean(true)));
        }
        let join = builder.add_join(left, right, spec, schema);
        builder.finish_gpu_resident(join)
    }

    #[test]
    fn join_hard_row_bounds_are_checked_and_fail_closed() {
        let bounded = bounded_join_plan(Some(3), Some(4), JoinType::Inner, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&bounded), Some(12));

        let empty = bounded_join_plan(Some(0), Some(4), JoinType::Inner, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&empty), Some(0));

        let unknown = bounded_join_plan(None, Some(4), JoinType::Inner, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&unknown), None);

        let overflow = bounded_join_plan(Some(u64::MAX), Some(2), JoinType::Inner, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&overflow), None);

        let residual = bounded_join_plan(Some(3), Some(4), JoinType::Inner, true, true);
        assert_eq!(query_plan_hard_row_upper_bound(&residual), Some(12));

        let keyless = bounded_join_plan(Some(3), Some(4), JoinType::Inner, false, false);
        assert_eq!(query_plan_hard_row_upper_bound(&keyless), Some(12));

        for (join_type, keyed, expected) in [
            (JoinType::Left, true, 12),
            (JoinType::Full, true, 19),
            (JoinType::LeftSemi, true, 3),
            (JoinType::LeftAnti, true, 3),
            (JoinType::Cross, false, 12),
            (JoinType::LeftMark, true, 3),
        ] {
            let plan = bounded_join_plan(Some(3), Some(4), join_type, keyed, false);
            assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(expected), "join type {join_type}");
        }

        let empty_right_left = bounded_join_plan(Some(3), Some(0), JoinType::Left, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&empty_right_left), Some(3));

        let empty_left_full = bounded_join_plan(Some(0), Some(4), JoinType::Full, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&empty_left_full), Some(4));

        let unknown_right_semi = bounded_join_plan(Some(3), None, JoinType::LeftSemi, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&unknown_right_semi), Some(3));

        let full_overflow = bounded_join_plan(Some(u64::MAX), Some(1), JoinType::Full, true, false);
        assert_eq!(query_plan_hard_row_upper_bound(&full_overflow), None);
    }

    #[test]
    fn grouped_aggregate_hard_row_bound_cannot_exceed_its_input() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("bounded-grouped-aggregate", Arc::clone(&schema)).with_row_count_upper_bound(3),
        ));
        let aggregate =
            builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column("key")], Vec::new()), schema);
        let plan = builder.finish_gpu_resident(aggregate);

        assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(3));
    }

    #[test]
    fn bounded_graph_evidence_exposes_exclusive_owner_terms() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            crate::source::GpuResidentSource::new("graph-owner-input", edge_schema()).with_row_count_upper_bound(20),
        ));
        let graph = builder.add_graph_algorithm(source, spec);
        let plan = builder.finish_gpu_resident(graph);
        let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded graph estimate")
            .pop()
            .expect("one graph estimate");
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("bounded graph evidence");

        let facts = evidence.owner_term_facts();
        let expected_owner_term_count = 4
            + usize::from(evidence.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
            + usize::from(cfg!(feature = "cugraph"));
        assert_eq!(facts.len(), expected_owner_term_count);
        assert_eq!(facts.iter().filter(|fact| fact.row_count_upper_bound().is_some()).count(), 1);
        assert_eq!(
            facts
                .iter()
                .find(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphPersistent)
                .expect("persistent owner")
                .bytes(),
            estimate.estimate.owner_components.graph_persistent_bytes
        );
        assert_eq!(
            facts
                .iter()
                .find(|fact| fact.role() == GraphMemoryOwnerTermRole::ConstructionTransient)
                .expect("construction transient owner")
                .bytes(),
            estimate.estimate.owner_components.construction_transient_bytes
        );
        assert_eq!(
            facts
                .iter()
                .find(|fact| fact.role() == GraphMemoryOwnerTermRole::ExecutionWorkspace)
                .expect("execution workspace owner")
                .bytes(),
            estimate.estimate.owner_components.execution_workspace_bytes
        );
        assert_eq!(
            facts
                .iter()
                .find(|fact| fact.role() == GraphMemoryOwnerTermRole::RetainedOutput)
                .expect("retained output owner")
                .bytes(),
            estimate.estimate.retained_output_bytes.bytes()
        );
        let floor_top_up = facts
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
            .map(GraphMemoryOwnerTermFact::bytes);
        assert_eq!(
            floor_top_up,
            (evidence.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
                .then(|| GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - evidence.cold_model_requested_bytes())
        );
        if cfg!(feature = "cugraph") {
            assert!(
                facts
                    .iter()
                    .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
                    .is_some_and(|fact| fact.bytes() > 0)
            );
        }
        assert_eq!(
            estimate
                .estimate
                .owner_components
                .execution_workspace_bytes
                .checked_add(estimate.estimate.retained_output_bytes.bytes()),
            Some(estimate.estimate.execution_bytes)
        );
        assert!(facts.iter().all(|fact| fact.role() != GraphMemoryOwnerTermRole::GraphPersistent
            || fact.bytes() != estimate.estimate.construction_bytes));
    }

    #[test]
    fn bounded_graph_live_floor_is_a_checked_owner_top_up_over_raw_model() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let tiny = bounded_evidence_for_spec(&spec, 1);
        assert!(tiny.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
        assert_eq!(
            tiny.bounded_live_graph_peak_bytes().expect("tiny graph floor fits"),
            GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES
        );
        assert_eq!(
            tiny.owner_term_facts()
                .iter()
                .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
                .map(GraphMemoryOwnerTermFact::bytes)
                .collect::<Vec<_>>(),
            vec![GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - tiny.cold_model_requested_bytes()]
        );

        let large = bounded_evidence_for_spec(&spec, 100_000);
        assert!(large.cold_model_requested_bytes() >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
        assert_eq!(
            large.bounded_live_graph_peak_bytes().expect("large graph model fits"),
            large.cold_model_requested_bytes(),
            "the canonical floor must not alter sufficiently large graph models"
        );
        assert!(
            large
                .owner_term_facts()
                .iter()
                .all(|fact| fact.role() != GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
        );
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn bfs_side_input_owner_uses_the_owner_peak_not_component_sum() {
        let side_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
        let plan = bfs_side_input_plan(side_schema, Some(3), |side| {
            let mut params = GraphBfsParams::new(0, None);
            params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
            params
        });
        let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded BFS estimate")
            .pop()
            .expect("one BFS estimate");
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("bounded BFS side-input evidence");
        let owner_peak = evidence
            .owner_term_facts()
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
            .expect("side-input owner term")
            .bytes();
        let preflight = cugraph::cudf_interop::bfs_side_input_workspace_preflight(
            &cugraph::cudf_interop::BfsSideInputPreflightInput::new(
                cugraph::types::DataType::Int64,
                cugraph::types::DataType::Int64,
                usize::try_from(estimate.estimate.vertex_estimate).expect("vertex estimate fits usize"),
                usize::try_from(estimate.estimate.input_edges).expect("edge estimate fits usize"),
                1,
                Some(3),
                None,
                None,
                None,
            )
            .expect("BFS side-input preflight input"),
        )
        .expect("BFS side-input preflight");
        assert_eq!(owner_peak, preflight.peak_bytes());
        assert_ne!(
            owner_peak,
            preflight.source_copy_bytes() + preflight.predicate_bitmap_bytes() + preflight.edge_mask_bytes(),
            "the owner term must preserve the co-resident peak rather than sum sequential components"
        );
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn bfs_empty_source_side_input_owner_is_zero() {
        let side_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
        let plan = bfs_side_input_plan(side_schema, Some(0), |side| {
            let mut params = GraphBfsParams::new(0, None);
            params.source_vertices = Some(GraphPredicateInput::new(side, "vertex"));
            params.endpoint_policy = crate::plan::GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
            params
        });
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("zero-row BFS source must retain finite graph-memory evidence");
        let side_input_terms = evidence
            .owner_term_facts()
            .iter()
            .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
            .collect::<Vec<_>>();
        assert_eq!(side_input_terms.len(), 1);
        assert_eq!(side_input_terms[0].bytes(), 0);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn personalized_pagerank_side_input_owner_requires_a_hard_row_bound() {
        let plan = personalized_pagerank_side_input_plan(Some(3));
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("bounded PPR side-input evidence");
        let owner_peak = evidence
            .owner_term_facts()
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
            .expect("PPR side-input owner term")
            .bytes();
        assert!(owner_peak > 0);

        let missing_rows = personalized_pagerank_side_input_plan(None);
        let error = graph_memory_bounded_admission_evidence_for_plan(
            &missing_rows,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect_err("PPR side input without a hard row bound must fail closed");
        assert_eq!(error.fact_value("reason"), Some("side_input_workspace_preflight_unavailable"));
        assert_eq!(error.fact_value("side_input_reason"), Some("missing_row_upper_bound"));
        assert_eq!(error.fact_value("side_input_role"), Some("personalization"));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn bfs_side_input_preflight_rejects_unsupported_dtype_include_edges_and_aliases() {
        let unsupported = bfs_side_input_plan(
            Arc::new(Schema::new(vec![Field::new("vertex", DataType::UInt64, false)])),
            Some(3),
            |side| {
                let mut params = GraphBfsParams::new(0, None);
                params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
                params
            },
        );
        let error = graph_memory_bounded_admission_evidence_for_plan(
            &unsupported,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect_err("unsupported BFS side-input dtype must fail closed");
        assert_eq!(error.kind(), crate::error::QueryErrorKind::Unsupported);
        assert_eq!(error.fact_value("side_input_reason"), Some("side_input_vertex_dtype_unsupported"));

        let include_edges = bfs_side_input_plan(edge_schema(), Some(3), |side| {
            let mut params = GraphBfsParams::new(0, None);
            params.predicates.include_edges = Some(GraphEdgePredicateInput::new(side, "src", "dst", None::<&str>));
            params
        });
        let error = graph_memory_bounded_admission_evidence_for_plan(
            &include_edges,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect_err("include_edges without an owner contract must fail closed");
        assert_eq!(error.fact_value("side_input_reason"), Some("include_edges_owner_contract_missing"));

        let duplicate = bfs_side_input_plan(
            Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)])),
            Some(3),
            |side| {
                let mut params = GraphBfsParams::new(0, None);
                params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
                params.predicates.exclude_vertices = Some(GraphPredicateInput::new(side, "vertex"));
                params
            },
        );
        let error =
            graph_memory_bounded_admission_evidence_for_plan(&duplicate, GraphMemoryCalibrationOverride::none(), None)
                .expect_err("reused BFS side-input nodes must fail closed");
        assert_eq!(error.fact_value("side_input_reason"), Some("side_input_node_reused"));
    }

    #[test]
    fn malformed_graph_owner_evidence_rejects_saturation_and_missing_terms() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            crate::source::GpuResidentSource::new("graph-owner-input", edge_schema()).with_row_count_upper_bound(20),
        ));
        let graph = builder.add_graph_algorithm(source, spec);
        let plan = builder.finish_gpu_resident(graph);
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("bounded graph evidence");
        let mut saturated = evidence.clone();
        saturated.owner_terms[0].bytes = u64::MAX;
        let proof = crate::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
        let error = proof.with_graph_memory_admission_evidence(&saturated).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_saturated"));

        let mut incomplete = evidence;
        incomplete.owner_terms.remove(0);
        let proof = crate::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
        let error = proof.with_graph_memory_admission_evidence(&incomplete).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_owner_term_role_missing"));
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

    fn raw_bfs_plan(source: LocalParquetSource) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
        builder.finish_with_host_sink(graph)
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn graph_memory_estimates_for_plan_raw_bfs_consumes_owner_construction_workspace_bounds() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![100])],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string()],
        );
        let plan = raw_bfs_plan(source);

        let planned = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("the owner preflight supplies raw BFS bounded-admission evidence");
        let graph = plan
            .nodes()
            .iter()
            .find_map(|node| match &node.kind {
                PlanNode::GraphAlgorithm(graph) => Some(graph),
                _ => None,
            })
            .expect("raw BFS plan contains a graph node");
        let input = graph_memory_model_input_for_plan_node(&plan, graph, GraphMemoryCalibrationOverride::none())
            .unwrap()
            .expect("literal row count is hard graph-input evidence");
        let workspace_bounds = graph_construction_workspace_bounds_from_owner(&graph.spec, input)
            .expect("same typed owner query must produce the construction bounds");
        let expected = GraphMemoryModel
            .estimate_for_bounded(&graph.spec, input.with_construction_workspace_bounds(workspace_bounds))
            .unwrap();

        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].estimate, expected);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn raw_bfs_owner_preflight_query_failure_leaves_the_bound_unavailable() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None);

        assert_eq!(
            graph_construction_workspace_bounds_from_owner_query(&spec, input, |_preflight| None),
            None,
            "an owner query error must remain an absent bound"
        );
        let error = GraphMemoryModel
            .estimate_for_bounded(&spec, input)
            .expect_err("an absent owner bound must retain the raw BFS rejection");
        assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn raw_bfs_owner_preflight_passes_renumber_and_refined_vertex_bound() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None)
            .with_vertex_estimate(GraphVertexEstimate::runtime_exact_distinct(37));
        let bounds = graph_construction_workspace_bounds_from_owner_query(&spec, input, |preflight| {
            assert!(preflight.renumber());
            assert_eq!(preflight.num_edges(), 100);
            assert_eq!(preflight.num_vertices(), 37);
            Some(GraphConstructionWorkspaceBounds::new(11, 17))
        })
        .expect("callback returns owner bounds");
        assert_eq!(bounds.renumber_workspace_bytes(), 11);
        assert_eq!(bounds.compression_workspace_bytes(), 17);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn raw_bfs_owner_preflight_falls_back_to_endpoint_vertex_bound() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None);
        let bounds = graph_construction_workspace_bounds_from_owner_query(&spec, input, |preflight| {
            assert!(preflight.renumber());
            assert_eq!(preflight.num_edges(), 100);
            assert_eq!(preflight.num_vertices(), 200);
            Some(GraphConstructionWorkspaceBounds::new(13, 19))
        })
        .expect("callback returns owner bounds");
        assert_eq!(bounds, GraphConstructionWorkspaceBounds::new(13, 19));
    }

    #[cfg(not(feature = "cugraph"))]
    #[test]
    fn graph_memory_estimates_for_plan_raw_bfs_refuses_without_cugraph_preflight() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![100])],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string()],
        );

        let error =
            graph_memory_estimates_for_plan(&raw_bfs_plan(source), GraphMemoryCalibrationOverride::none(), None)
                .expect_err("feature-off planning has no owner preflight contract");

        assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
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
    fn graph_memory_evidence_keeps_typed_retained_output_distinct_from_execution_workspace() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::ForceAtlas2,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::PythonCugraph,
            },
            GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        );
        let evidence = bounded_evidence_for_spec(&spec, 10);

        assert!(evidence.retained_output_bytes() > 0);
        assert!(evidence.retained_output_rows_upper_bound() > 0);
        assert!(evidence.warm_model_requested_bytes() > evidence.retained_output_bytes());
        assert_ne!(
            evidence.retained_output_bytes(),
            evidence.warm_model_requested_bytes(),
            "retained output must not be recovered from the rounded execution peak"
        );
    }

    #[test]
    fn graph_memory_source_lifetime_composes_from_owner_evidence() {
        let base_spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        let source_bytes = 2_048;
        let shared = bounded_evidence_for_spec(&base_spec, 20);
        let normalized = bounded_evidence_for_spec(&base_spec.clone().with_edge_normalize_inserted(true), 20);

        assert_eq!(
            shared.cold_model_with_source_stage_bytes(source_bytes).unwrap(),
            shared.cold_model_requested_bytes().max(source_bytes),
            "a prepared edge frame is already owned by the shared graph peak"
        );
        assert_eq!(
            normalized.cold_model_with_source_stage_bytes(source_bytes).unwrap(),
            source_bytes.checked_add(normalized.cold_model_requested_bytes()).unwrap(),
            "explicit normalization retains source and normalized graph input together"
        );
    }

    #[test]
    fn bounded_model_rejects_exact_u64_max_projection() {
        let error = ModelBytes(u128::from(u64::MAX)).bounded_u64("bounded_test_term").unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
        assert_eq!(error.fact_value("term"), Some("bounded_test_term"));
        assert_eq!(error.fact_value("operation"), Some("saturated"));
    }

    #[test]
    fn shared_graph_source_composition_rejects_exact_u64_max() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        let evidence = bounded_evidence_for_spec(&spec, 1);
        let error = evidence.cold_model_with_source_stage_bytes(u64::MAX).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
        assert_eq!(error.fact_value("term"), Some("source_stage_graph_peak"));
        assert_eq!(error.fact_value("operation"), Some("saturated"));
    }

    #[test]
    fn bounded_graph_reservation_credits_only_its_live_normalized_input_owner() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![1_000_000])],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let estimate =
            graph_memory_estimates_for_plan(&triangle_plan(source), GraphMemoryCalibrationOverride::none(), None)
                .unwrap()
                .into_iter()
                .next()
                .expect("graph plan yields one model estimate");
        let full_model_bytes = estimate.estimate.requested_bytes;
        let semantic_edge_row_width = semantic_edge_row_width_bytes(
            estimate.estimate.vertex_width_bytes,
            estimate.estimate.weight_width_bytes,
            estimate.estimate.has_edge_id,
        );
        let modeled_input_bytes = estimate.estimate.input_edges.saturating_mul(semantic_edge_row_width);
        let raw_source_provenance_bytes = estimate
            .estimate
            .raw_input_edges
            .expect("complete normalized source retains an exact raw row count")
            .saturating_mul(semantic_edge_row_width);
        let remaining_budget_bytes = full_model_bytes.saturating_sub(modeled_input_bytes);
        assert!(remaining_budget_bytes >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);

        let admitted = GraphReservationReport::from_estimate(
            estimate.node_id,
            estimate.estimate.clone(),
            Some(remaining_budget_bytes),
        )
        .credit_live_graph_input(modeled_input_bytes, None, 0);
        assert_eq!(admitted.model_requested_bytes, full_model_bytes);
        assert_eq!(admitted.requested_bytes, remaining_budget_bytes);
        assert_eq!(admitted.outcome, GraphReservationOutcome::Admitted);

        let capped = GraphReservationReport::from_estimate(
            estimate.node_id,
            estimate.estimate.clone(),
            Some(remaining_budget_bytes),
        )
        .credit_live_graph_input(u64::MAX, None, 0);
        assert_eq!(capped.model_requested_bytes, full_model_bytes);
        assert_eq!(capped.requested_bytes, remaining_budget_bytes);
        assert_eq!(capped.outcome, GraphReservationOutcome::Admitted);

        let provenance_only = GraphReservationReport::from_estimate(
            estimate.node_id,
            estimate.estimate.clone(),
            Some(remaining_budget_bytes),
        )
        .credit_live_graph_input(modeled_input_bytes.saturating_add(raw_source_provenance_bytes), None, 0);
        assert_eq!(
            provenance_only.requested_bytes, remaining_budget_bytes,
            "raw source rows are provenance only; they cannot increase graph-input credit"
        );

        let denied = GraphReservationReport::from_estimate(
            estimate.node_id,
            estimate.estimate.clone(),
            Some(remaining_budget_bytes),
        )
        .credit_live_graph_input(modeled_input_bytes.saturating_sub(1), None, 0);
        assert_eq!(denied.model_requested_bytes, full_model_bytes);
        assert_eq!(denied.requested_bytes, remaining_budget_bytes.saturating_add(1));
        assert_eq!(denied.outcome, GraphReservationOutcome::Denied);

        // EdgeNormalize is the exact owner of the logical graph edge frame.
        // Its native output can retain aliases beyond the semantic column
        // widths, so a caller that proves this ownership may credit the
        // tracker charge rather than pretending those bytes are a second raw
        // source frame.
        let exact_normalized_frame_charge_bytes = modeled_input_bytes.saturating_add(4_096);
        let exact_owner_budget_bytes = full_model_bytes.saturating_sub(exact_normalized_frame_charge_bytes);
        let normalized_and_source_budget_bytes = full_model_bytes
            .checked_sub(modeled_input_bytes.checked_mul(2).expect("test input bytes fit u64"))
            .expect("graph model covers normalized and retained source frames");
        let normalized_and_source = GraphReservationReport::from_estimate(
            estimate.node_id,
            estimate.estimate.clone(),
            Some(normalized_and_source_budget_bytes),
        )
        .credit_live_graph_input(modeled_input_bytes, Some(modeled_input_bytes), modeled_input_bytes);
        assert_eq!(normalized_and_source.requested_bytes, normalized_and_source_budget_bytes);
        assert_eq!(normalized_and_source.outcome, GraphReservationOutcome::Admitted);

        let exact_owner =
            GraphReservationReport::from_estimate(estimate.node_id, estimate.estimate, Some(exact_owner_budget_bytes))
                .credit_live_graph_input(u64::MAX, Some(exact_normalized_frame_charge_bytes), 0);
        assert_eq!(exact_owner.requested_bytes, exact_owner_budget_bytes);
        assert_eq!(exact_owner.outcome, GraphReservationOutcome::Admitted);
    }

    #[test]
    fn normalized_graph_raw_source_count_is_provenance_not_physical_residency() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::ForceAtlas2,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::PythonCugraph,
            },
            GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        )
        .with_edge_normalize_inserted(true);
        let normalized_input = GraphMemoryModelInput {
            raw_input_edges: Some(200),
            input_edges: 400,
            vertex_width_bytes: 8,
            weight_width_bytes: Some(8),
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: Some(GraphVertexEstimate::graph_metadata(50)),
            edge_normalize_inserted: true,
            construction_workspace_bounds: None,
            statistics: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };
        let without_raw = GraphMemoryModel.estimate_for_bounded(&spec, normalized_input).unwrap();
        let with_raw =
            GraphMemoryModel.estimate_for_bounded(&spec, normalized_input.with_raw_input_edges(100)).unwrap();
        assert_eq!(with_raw.raw_input_edges, Some(100));
        assert_eq!(with_raw.construction_bytes, without_raw.construction_bytes);
        assert_eq!(with_raw.execution_bytes, without_raw.execution_bytes);
        assert_eq!(with_raw.requested_bytes, without_raw.requested_bytes);

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            crate::source::GpuResidentSource::new("normalized-graph-input", edge_schema())
                .with_row_count_upper_bound(200)
                .with_graph_vertex_upper_bound(50),
        ));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_gpu_resident(graph);
        let canonical_estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("canonical lowered graph estimate")
            .pop()
            .expect("one canonical graph estimate");
        assert_eq!(canonical_estimate.estimate.raw_input_edges, Some(200));
        assert_eq!(canonical_estimate.estimate.input_edges, 400);
        assert_eq!(canonical_estimate.estimate.vertex_estimate, 50);
        let cold_evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("canonical bounded graph evidence");
        assert_eq!(cold_evidence.cold_model_requested_bytes(), canonical_estimate.estimate.requested_bytes);
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

    #[test]
    fn graph_plan_memory_uses_footer_vertex_bound_only_for_complete_rowsets() {
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet").with_row_group_rows(vec![10]).with_graph_vertex_upper_bound(30),
                LocalParquetFile::new("second.parquet").with_row_group_rows(vec![20]).with_graph_vertex_upper_bound(50),
            ],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let plan = triangle_plan(source);

        let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .unwrap()
            .into_iter()
            .next()
            .expect("footer-backed graph plan has a model estimate")
            .estimate;
        assert_eq!(estimate.vertex_estimate, 80);
        assert_eq!(estimate.vertex_estimate_source, GraphVertexEstimateSource::GraphMetadata);

        let partial_source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet")
                    .with_row_group_rows(vec![10, 20])
                    .with_row_groups(vec![0])
                    .with_graph_vertex_upper_bound(300),
            ],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let partial_estimate = graph_memory_estimates_for_plan(
            &triangle_plan(partial_source),
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap()
        .into_iter()
        .next()
        .expect("selected row groups still have a hard row bound")
        .estimate;
        assert_eq!(partial_estimate.vertex_estimate_source, GraphVertexEstimateSource::EdgeEndpointUpperBound);
    }

    #[test]
    fn force_atlas2_workspace_is_linear_and_uses_the_dispatch_path() {
        let input = GraphMemoryModelInput {
            raw_input_edges: None,
            input_edges: 0,
            vertex_width_bytes: 8,
            weight_width_bytes: None,
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: None,
            statistics: None,
            edge_normalize_inserted: false,
            construction_workspace_bounds: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };
        let output_sized = 4_096;
        let exact_params = GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default());
        let barnes_hut = GraphForceAtlas2Params { barnes_hut_optimize: true, ..Default::default() };
        let barnes_hut_params = GraphAlgorithmParams::ForceAtlas2(barnes_hut);

        assert_eq!(force_atlas2_workspace_bytes(input, 10, &exact_params, output_sized), output_sized + 10 * 44);
        assert_eq!(
            force_atlas2_workspace_bytes(input, 10, &barnes_hut_params, output_sized),
            output_sized + 10 * (120 + 16),
            "Barnes-Hut includes its Int64 degree vector"
        );
        assert_eq!(
            force_atlas2_workspace_bytes(input, 20, &exact_params, output_sized)
                - force_atlas2_workspace_bytes(input, 10, &exact_params, output_sized),
            10 * 44,
            "the exact path must stay linear in vertices"
        );

        let weighted_edges = GraphMemoryModelInput { input_edges: 10, weight_width_bytes: Some(8), ..input };
        assert_eq!(
            force_atlas2_workspace_bytes(weighted_edges, 0, &exact_params, 0),
            10 * 25,
            "the legacy COO stable-sort workspace is one weighted record plus its CUB spine"
        );

        let unweighted_edges = GraphMemoryModelInput { input_edges: 10, weight_width_bytes: None, ..input };
        assert_eq!(
            force_atlas2_workspace_bytes(unweighted_edges, 0, &exact_params, 0),
            10 * 21,
            "the C-API materializes Float32 unit weights for unweighted ForceAtlas2"
        );
    }

    #[test]
    fn force_atlas2_phase_peaks_keep_sequential_allocations_separate() {
        let input = GraphMemoryModelInput {
            raw_input_edges: Some(50),
            input_edges: 100,
            vertex_width_bytes: 8,
            weight_width_bytes: Some(8),
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: Some(GraphVertexEstimate::graph_metadata(10)),
            statistics: None,
            edge_normalize_inserted: true,
            construction_workspace_bounds: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };
        let barnes_hut = GraphForceAtlas2Params { barnes_hut_optimize: true, ..Default::default() };
        let params = GraphAlgorithmParams::ForceAtlas2(barnes_hut);
        let construction_bytes = 4_000;
        let result_conversion_bytes = 160;
        let phases = force_atlas2_phase_peaks(input, 10, &params, construction_bytes, result_conversion_bytes);
        let workspace = force_atlas2_workspace_phase_bytes(input, 10, &params, 2_640);

        assert_eq!(
            phases.legacy_coo_sort_before_degree_bytes,
            phases.persistent_base_bytes.saturating_add(workspace.legacy_coo_sort_workspace_bytes),
            "COO sorting completes before the Barnes-Hut degree vector is allocated"
        );
        assert_eq!(
            phases.barnes_hut_with_degree_bytes,
            phases.persistent_base_bytes.saturating_add(workspace.barnes_hut_workspace_bytes),
            "the degree vector belongs only to the Barnes-Hut phase"
        );
        assert_eq!(
            phases.post_barnes_result_conversion_bytes,
            phases.persistent_base_bytes.saturating_add(result_conversion_bytes),
            "result vectors are created only after Barnes-Hut returns"
        );
        assert_eq!(
            phases.peak_bytes(),
            phases
                .legacy_coo_sort_before_degree_bytes
                .max(phases.barnes_hut_with_degree_bytes)
                .max(phases.post_barnes_result_conversion_bytes)
        );
    }

    #[test]
    fn graph_construction_prices_sequential_renumber_and_property_compression_phases() {
        let mut input = GraphMemoryModelInput {
            raw_input_edges: None,
            input_edges: 10,
            vertex_width_bytes: 8,
            weight_width_bytes: Some(8),
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: None,
            statistics: None,
            edge_normalize_inserted: false,
            construction_workspace_bounds: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };

        assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 10 * 8);

        input.weight_width_bytes = None;
        assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 0);

        input.has_edge_id = true;
        assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 10 * 8);

        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::ForceAtlas2,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions {
                directed: true,
                renumber: false,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        );
        input.has_edge_id = false;
        input.weight_width_bytes = Some(8);
        let construction_with_edges = GraphMemoryModel.estimate(&spec, input).unwrap().construction_bytes;
        input.input_edges = 0;
        let construction_without_edges = GraphMemoryModel.estimate(&spec, input).unwrap().construction_bytes;
        assert_eq!(construction_with_edges - construction_without_edges, 10 * (24 + 24 + 8));

        let weighted_pagerank = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        input.input_edges = 10;
        input.vertex_estimate = Some(GraphVertexEstimate::graph_metadata(5));
        let error = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
        assert_eq!(error.fact_value("algorithm"), Some("pagerank"));

        input.construction_workspace_bounds = Some(GraphConstructionWorkspaceBounds::new(200, 0));
        let renumber_dominant = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap();
        assert_eq!(renumber_dominant.construction_bytes, 240 + 240 + 200);

        input.construction_workspace_bounds = Some(GraphConstructionWorkspaceBounds::new(0, 200));
        let compression_dominant = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap();
        assert_eq!(compression_dominant.construction_bytes, 240 + 240 + 40 + 80 + 200);
    }

    #[test]
    fn raw_bfs_phase_price_retains_normalization_source_and_prepared_input() {
        const EDGES: u64 = 45_564_149;
        const VERTICES: u64 = 4_894_081;
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(1_091, None))),
        )
        .with_edge_normalize_inserted(true);
        let input = GraphMemoryModelInput {
            raw_input_edges: Some(EDGES),
            input_edges: EDGES,
            vertex_width_bytes: 8,
            weight_width_bytes: None,
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: Some(GraphVertexEstimate::graph_metadata(VERTICES)),
            statistics: None,
            edge_normalize_inserted: true,
            construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };

        let estimate = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap();
        let normalized_input_bytes = EDGES * 16;

        assert_eq!(estimate.construction_bytes, 2_226_231_800);
        assert_eq!(estimate.execution_bytes, 1_328_455_464);
        assert_eq!(estimate.requested_bytes, 3_515_534_616);
        assert_eq!(
            estimate.requested_bytes,
            2 * normalized_input_bytes + normalized_input_bytes + estimate.execution_bytes,
            "the source and normalized edge owners remain live for BFS, while construction-only work does not"
        );
    }

    #[test]
    fn bounded_graph_arithmetic_rejects_early_cardinality_overflow() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: true,
                renumber: false,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        let input = graph_memory_model_input(u64::MAX, u64::MAX, &spec.edge_columns, false, None)
            .with_construction_workspace_bounds(GraphConstructionWorkspaceBounds::new(0, 0));
        let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
        assert_eq!(error.fact_value("term"), Some("vertex_upper_bound"));
    }

    #[test]
    fn bounded_graph_arithmetic_rejects_late_phase_overflow() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: true,
                renumber: false,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        let input = graph_memory_model_input(u64::MAX / 24, 1, &spec.edge_columns, false, None)
            .with_construction_workspace_bounds(GraphConstructionWorkspaceBounds::new(0, 0));
        let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
        assert_eq!(error.fact_value("term"), Some("execution_workspace_bytes"));
    }

    #[test]
    fn bounded_normalized_source_and_graph_peak_reject_addition_overflow() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: true,
                renumber: false,
                policy: GraphConstructionPolicy::PythonCugraph,
            },
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        );
        let evidence = bounded_evidence_for_spec(&spec, 1);
        let error = evidence.cold_model_with_source_stage_bytes(u64::MAX).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
        assert_eq!(error.fact_value("term"), Some("source_stage_graph_peak"));
    }

    #[test]
    fn bounded_raw_bfs_requires_a_preflight_workspace_contract() {
        let mut spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let input = GraphMemoryModelInput {
            raw_input_edges: Some(100),
            input_edges: 100,
            vertex_width_bytes: 8,
            weight_width_bytes: None,
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: Some(GraphVertexEstimate::graph_metadata(50)),
            edge_normalize_inserted: false,
            construction_workspace_bounds: None,
            statistics: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };

        let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
        assert_eq!(error.fact_value("algorithm"), Some("bfs"));

        let renumber_workspace_bytes = 10_000;
        let compression_workspace_bytes = 0;
        let workspace_bounds =
            GraphConstructionWorkspaceBounds::new(renumber_workspace_bytes, compression_workspace_bytes);
        let admitted = GraphMemoryModel
            .estimate_for_bounded(&spec, input.with_construction_workspace_bounds(workspace_bounds))
            .expect("the owner-provided bound completes raw BFS admission evidence");
        let persistent_graph_and_input_bytes = input.input_edges * input.vertex_width_bytes * 2;

        assert_eq!(admitted.construction_bytes, 13_200);
        assert_eq!(admitted.execution_bytes, 4_800);
        assert_eq!(
            admitted.requested_bytes,
            admitted.construction_bytes.max(persistent_graph_and_input_bytes.saturating_add(admitted.execution_bytes)),
            "construction-only workspace must not remain live during execution"
        );
        assert_eq!(admitted.requested_bytes, admitted.construction_bytes);

        spec.edge_normalize_inserted = true;
        let normalized = input.with_edge_normalize_inserted(true).with_construction_workspace_bounds(workspace_bounds);
        assert!(GraphMemoryModel.estimate_for_bounded(&spec, normalized).is_ok());
    }

    #[test]
    fn raw_bfs_owner_bounds_retain_renumber_map_across_compression_and_execution() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let input = graph_memory_model_input(1_000, 8, &spec.edge_columns, false, None)
            .with_vertex_estimate(GraphVertexEstimate::graph_metadata(50));
        let bounds = GraphConstructionWorkspaceBounds::new(6_000, 10_000);
        let estimate = GraphMemoryModel
            .estimate_for_bounded(&spec, input.with_construction_workspace_bounds(bounds))
            .expect("owner bounds complete bounded raw BFS pricing");

        assert_eq!(estimate.owner_components.graph_persistent_bytes, 16_000);
        assert_eq!(estimate.owner_components.construction_transient_bytes, 26_400);
        assert_eq!(estimate.construction_bytes, 42_400);
        assert_eq!(estimate.execution_bytes, 26_400);
        assert_eq!(estimate.requested_bytes, 42_400);
    }

    #[test]
    fn graph_claim_calibration_row_keeps_claim_and_peak_provenance_distinct() {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![10, 5])],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let plan = triangle_plan(source);
        let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .unwrap()
            .into_iter()
            .next()
            .expect("graph plan yields one model estimate");
        let report = GraphReservationReport::from_estimate(estimate.node_id, estimate.estimate, Some(64 * 1024 * 1024));
        let mut row = GraphClaimCalibrationRow::from_reservation_report(&report, estimate.node_id);

        assert_eq!(row.node_id, estimate.node_id.to_string());
        assert_eq!(row.node_index, estimate.node_id.index());
        assert_eq!(row.algorithm, GraphAlgorithm::TriangleCountAll);
        assert_eq!(row.model_requested_bytes, report.model_requested_bytes);
        assert_eq!(row.requested_bytes, report.requested_bytes);
        assert_eq!(row.budget_bytes, report.budget_bytes);
        assert_eq!(row.execution_state, GraphClaimExecutionState::Pending);
        assert_eq!(row.measured_peak_bytes, None);
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
        assert_eq!(row.graph_model_error_ratio, None);

        row.record_measured_peak(Some(4096), GraphMeasuredPeakProvenance::ReservationHighWater);
        assert_eq!(row.measured_peak_bytes, Some(4096));
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::ReservationHighWater);
        assert_eq!(
            row.graph_model_error_ratio,
            GraphModelErrorRatio::from_requested_and_measured(report.model_requested_bytes, 4096)
        );

        row.record_measured_peak(Some(8192), GraphMeasuredPeakProvenance::Unavailable);
        assert_eq!(row.measured_peak_bytes, None, "unavailable provenance cannot retain a guessed numeric peak");
        assert_eq!(row.graph_model_error_ratio, None, "unavailable provenance cannot retain a ratio");
        assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
    }

    #[test]
    fn bounded_similarity_model_accepts_only_pre_admission_pair_evidence() {
        assert!(GraphPairCardinalitySource::PlanExact.is_pre_admission_bound());
        assert!(GraphPairCardinalitySource::PlanHardUpperBound.is_pre_admission_bound());
        assert!(!GraphPairCardinalitySource::RuntimeMaterialized.is_pre_admission_bound());

        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Jaccard,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
        );
        let input = GraphMemoryModelInput {
            raw_input_edges: None,
            input_edges: 20,
            vertex_width_bytes: 8,
            weight_width_bytes: None,
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: None,
            statistics: None,
            edge_normalize_inserted: false,
            construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        };

        let runtime_only = GraphMemoryModel
            .estimate_for_bounded(
                &spec,
                input.with_similarity_pair_input(100, 8, GraphPairCardinalitySource::RuntimeMaterialized),
            )
            .unwrap_err();
        assert_eq!(runtime_only.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(runtime_only.fact_value("reason"), Some("strict_pair_cardinality_evidence_not_plan_bound"));

        for source in [GraphPairCardinalitySource::PlanExact, GraphPairCardinalitySource::PlanHardUpperBound] {
            let estimate =
                GraphMemoryModel.estimate_for_bounded(&spec, input.with_similarity_pair_input(100, 8, source)).unwrap();
            assert_eq!(estimate.pair_cardinality_source, Some(source));
            assert_eq!(estimate.pair_cardinality_admission_mode, Some(GraphPairCardinalityAdmissionMode::Bounded));
        }
    }

    #[test]
    fn bounded_graph_floor_is_applied_before_live_input_credits() {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Jaccard,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
        );
        let input = GraphMemoryModelInput {
            raw_input_edges: None,
            input_edges: 6,
            vertex_width_bytes: 8,
            weight_width_bytes: None,
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: None,
            statistics: None,
            edge_normalize_inserted: true,
            construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        }
        .with_similarity_pair_input(1, 8, GraphPairCardinalitySource::PlanHardUpperBound);
        let estimate = GraphMemoryModel.estimate_for_bounded(&spec, input).expect("small similarity model");
        assert!(estimate.requested_bytes < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);

        let report =
            GraphReservationReport::from_estimate("n3", estimate, Some(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES))
                .credit_live_graph_input(6 * 8, None, 0)
                .credit_live_similarity_pair_input(16);
        assert_eq!(
            report.requested_bytes,
            GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - 6 * 8 - 16,
            "the floor applies to the complete live graph peak, so accounted input bytes reduce the incremental ask"
        );
        assert_eq!(report.outcome, GraphReservationOutcome::Admitted);
    }
}
