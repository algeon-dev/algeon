//! Graph traversal algorithm parameter contracts.

use super::*;

/// Breadth-first search parameters.
///
/// `source_vertex` is the traversal root. `depth_limit = None` means the
/// traversal is unbounded by depth; when set, the limit must be non-negative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBfsParams {
    /// Root vertex for traversal.
    pub source_vertex: i64,
    /// Optional one-row GPU-resident source-vertex input. When present, this
    /// takes precedence over `source_vertex` so callers can derive the source
    /// id on device and avoid host round-trips during logical-id mapping.
    pub source_vertices: Option<GraphPredicateInput>,
    /// Optional non-negative traversal depth limit.
    pub depth_limit: Option<i64>,
    /// Optional relation-backed predicate inputs consumed by predicate BFS.
    pub predicates: GraphBfsPredicateInputs,
    /// SQL-facing policy for absent source/target endpoints after edge filters.
    pub endpoint_policy: GraphBfsEndpointPolicy,
    /// Whether the BFS output should include per-row target metadata columns.
    ///
    /// When true, `target_vertices` must be present. The executor appends
    /// `target_found BOOLEAN` and nullable `target_distance INT64`; it does
    /// not expose a target vertex because cuGraph does not define a stable
    /// tie-break for equally near targets.
    pub return_target_info: bool,
}

impl GraphBfsParams {
    /// Creates plain BFS parameters without predicate side inputs.
    pub fn new(source_vertex: i64, depth_limit: Option<i64>) -> Self {
        Self {
            source_vertex,
            source_vertices: None,
            depth_limit,
            predicates: GraphBfsPredicateInputs::default(),
            endpoint_policy: GraphBfsEndpointPolicy::Strict,
            return_target_info: false,
        }
    }

    /// Returns true when this BFS call has any predicate side input.
    pub fn has_predicates(&self) -> bool {
        self.predicates.has_inputs()
    }

    /// Visits every side-input node id in stable role order.
    pub fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.source_vertices.iter().map(|input| input.input).chain(self.predicates.node_ids())
    }
}

/// BFS behavior when requested source/target endpoints are absent from the
/// materialized graph domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphBfsEndpointPolicy {
    /// Preserve the native/cuGraph contract: absent endpoints are invalid input.
    Strict,
    /// Return an empty result when all path endpoints are absent. Partial
    /// multi-source misses remain errors so callers do not lose diagnostics.
    EmptyOnMissingPathEndpoint,
}

impl GraphBfsEndpointPolicy {
    /// Stable label used in diagnostics and graph runtime metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::EmptyOnMissingPathEndpoint => "empty_on_missing_path_endpoint",
        }
    }
}

/// Optional predicate side inputs for predicate-aware BFS.
///
/// Each input identifies another native plan node plus the column(s) the BFS
/// executor should read from that node's GPU-resident output. SQL table names
/// and option JSON are deliberately absent from this IR.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphBfsPredicateInputs {
    /// Allowed vertex id set.
    pub include_vertices: Option<GraphPredicateInput>,
    /// Excluded vertex id set.
    pub exclude_vertices: Option<GraphPredicateInput>,
    /// Early-stop target vertex id set.
    pub target_vertices: Option<GraphPredicateInput>,
    /// Stable edge-id allow set. Requires graph construction with edge ids.
    pub include_edge_ids: Option<GraphPredicateInput>,
    /// Relation-backed edge allow set aligned by declared edge columns.
    pub include_edges: Option<GraphEdgePredicateInput>,
}

impl GraphBfsPredicateInputs {
    /// Returns true when any predicate input is present.
    pub fn has_inputs(&self) -> bool {
        self.include_vertices.is_some()
            || self.exclude_vertices.is_some()
            || self.target_vertices.is_some()
            || self.include_edge_ids.is_some()
            || self.include_edges.is_some()
    }

    /// Visits every side-input node id in stable role order.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.include_vertices
            .iter()
            .map(|input| input.input)
            .chain(self.exclude_vertices.iter().map(|input| input.input))
            .chain(self.target_vertices.iter().map(|input| input.input))
            .chain(self.include_edge_ids.iter().map(|input| input.input))
            .chain(self.include_edges.iter().map(|input| input.input))
    }
}

/// Single-column graph predicate input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPredicateInput {
    /// Native plan node producing the predicate relation.
    pub input: NodeId,
    /// Non-null id column in the predicate relation.
    pub column: String,
}

impl GraphPredicateInput {
    /// Creates a single-column predicate input binding.
    pub fn new(input: NodeId, column: impl Into<String>) -> Self {
        Self { input, column: column.into() }
    }
}

/// Edge relation predicate input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgePredicateInput {
    /// Native plan node producing the allowed-edge relation.
    pub input: NodeId,
    /// Source vertex column used to align with the primary edge relation.
    pub source: String,
    /// Destination vertex column used to align with the primary edge relation.
    pub destination: String,
    /// Optional stable edge-id column when the source relation already has one.
    pub edge_id: Option<String>,
}

impl GraphEdgePredicateInput {
    /// Creates an edge predicate input binding.
    pub fn new(
        input: NodeId,
        source: impl Into<String>,
        destination: impl Into<String>,
        edge_id: Option<impl Into<String>>,
    ) -> Self {
        Self { input, source: source.into(), destination: destination.into(), edge_id: edge_id.map(Into::into) }
    }
}

/// Parameters for single-source shortest paths (SSSP).
///
/// Mirrors `cugraph::traversal::SsspParams` without cugraph-rs newtype
/// wrappers so the native IR can be constructed without the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSsspParams {
    /// Source vertex for the SSSP traversal.
    pub source_vertex: i64,
    /// Maximum edge-weight sum to consider. `None` means no cutoff.
    /// Must be >= 0 when set.
    pub cutoff: Option<f64>,
}
