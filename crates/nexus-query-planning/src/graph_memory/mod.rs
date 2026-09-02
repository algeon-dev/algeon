//! Approximate cuGraph memory pricing outside the bounded gate.
//!
//! These estimates are intentionally not bounded contracts. They price a
//! graph fragment before entering cuGraph so bounded service admission can fail
//! locally with structured facts instead of relying on a raw RMM OOM.

mod evidence;
mod model;
mod side_input;

#[cfg(test)]
use arrow::datatypes::{DataType, SchemaRef};
pub use evidence::{
    GraphClaimCalibrationRow, GraphClaimExecutionState, GraphMeasuredPeakProvenance, GraphMemoryAdmissionEvidence,
    GraphMemoryEstimate, GraphMemoryRetainedOutputContribution, GraphModelErrorRatio, GraphPlanMemoryEstimate,
    GraphReservationReport,
};
#[doc(hidden)]
pub use evidence::{
    GraphMemoryOwnerComponents, GraphMemoryOwnerTermFact, GraphMemoryOwnerTermRole, GraphSourceStageLifetime,
    graph_output_is_final,
};
#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub use model::bounded_similarity_pair_cardinality_for_plan;
#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub use model::with_graph_construction_workspace_bounds_from_owner;
pub use model::{
    GraphMemoryModel, GraphMemoryModelInput, graph_edge_list_output_rows_for_plan,
    graph_memory_bounded_admission_evidence_for_plan, graph_memory_class, graph_memory_estimate_for_edge_list,
    graph_memory_estimate_for_edge_list_with_vertex_estimate, graph_memory_estimates_for_plan,
    graph_memory_model_input, graph_memory_reservation_required, graph_runtime_statistics_probe_bytes,
    graph_vertex_estimate_probe_bytes, query_plan_graph_vertex_upper_bound, query_plan_hard_row_upper_bound,
    triangle_count_peak_chunk_min_degree_pricing_term,
};
#[cfg(test)]
#[doc(hidden)]
pub use model::{
    ModelBytes, force_atlas2_phase_peaks, force_atlas2_workspace_bytes, force_atlas2_workspace_phase_bytes,
    graph_memory_model_input_for_plan_node, raw_libcugraph_property_output_staging, semantic_edge_row_width_bytes,
};
#[cfg(all(test, feature = "cugraph"))]
#[doc(hidden)]
pub use model::{graph_construction_workspace_bounds_from_owner, graph_construction_workspace_bounds_from_owner_query};
#[doc(hidden)]
pub use model::{
    graph_memory_bounded_admission_evidence_for_plan_with_resident_rows, query_plan_node_hard_row_upper_bound,
    query_plan_node_hard_row_upper_bound_with_resident_rows,
};

#[cfg(test)]
use crate::{
    error::QueryErrorCode,
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy, GraphEdgeColumnBindings,
        JoinSide, JoinType, NodeId, PlanNode, QueryPlan,
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
    #[doc(hidden)]
    pub const fn is_pre_admission_bound(self) -> bool {
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
            Self::Bounded => "bounded",
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

#[cfg(test)]
mod tests;
