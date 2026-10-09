//! Graph traversal algorithm parameter contracts.

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
    /// Bounded source-to-target path output request. The source and target
    /// bindings remain the canonical `source_vertices`/`source_vertex` and
    /// `predicates.target_vertices` fields above.
    pub path_output: Option<GraphBfsPathOutput>,
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
            path_output: None,
            return_target_info: false,
        }
    }

    /// Visits every side-input ordinal in stable role order.
    pub(crate) fn side_input_ordinals(&self) -> impl Iterator<Item = usize> + '_ {
        self.source_vertices.iter().map(|input| input.input).chain(self.predicates.ordinals())
    }
}

/// Hard bounds required to extract one BFS path on device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphBfsPathOutput {
    /// Maximum rows in the source-to-target path frame.
    pub max_output_rows: u64,
}

impl GraphBfsPathOutput {
    /// Create a bounded path-output request.
    pub const fn new(max_output_rows: u64) -> Self {
        Self { max_output_rows }
    }
}

/// BFS behavior when requested source/target endpoints are absent from the
/// materialized graph domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphBfsEndpointPolicy {
    /// Matches the native/cuGraph contract. Absent endpoints are invalid input.
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
    /// Visits every side-input ordinal in stable role order.
    pub(crate) fn ordinals(&self) -> impl Iterator<Item = usize> + '_ {
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
    /// Position of the relation among the algorithm node inputs.
    pub input: usize,
    /// Non-null id column in the predicate relation.
    pub column: String,
}

impl GraphPredicateInput {
    /// Creates a single-column predicate input binding.
    pub fn new(input: usize, column: impl Into<String>) -> Self {
        Self { input, column: column.into() }
    }
}

/// Edge relation predicate input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgePredicateInput {
    /// Position of the relation among the algorithm node inputs.
    pub input: usize,
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
        input: usize,
        source: impl Into<String>,
        destination: impl Into<String>,
        edge_id: Option<impl Into<String>>,
    ) -> Self {
        Self { input, source: source.into(), destination: destination.into(), edge_id: edge_id.map(Into::into) }
    }
}

/// Parameters for single-source shortest paths (SSSP).
///
/// Mirrors `cugraph::traversal::SsspParams` without the `cugraph` crate's
/// new type wrappers so the native IR can be constructed without the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSsspParams {
    /// Source vertex for the SSSP traversal.
    pub source_vertex: GraphSsspSource,
    /// Maximum edge-weight sum to consider. `None` means no cutoff.
    /// Must be >= 0 when set.
    pub cutoff: Option<f64>,
}

/// SSSP source binding. Numeric calls retain the direct scalar path; logical
/// string calls bind one already-encoded vertex side input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSsspSource {
    /// Direct numeric source vertex.
    I64(i64),
    /// One-row encoded source-vertex input.
    Input(GraphPredicateInput),
}

impl std::fmt::Display for GraphSsspSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I64(value) => write!(formatter, "{value}"),
            Self::Input(input) => write!(formatter, "input:{}:{}", input.input, input.column),
        }
    }
}

impl GraphSsspParams {
    /// Visits the source side-input node when this call uses one.
    pub(crate) fn side_input_ordinals(&self) -> impl Iterator<Item = usize> + '_ {
        match &self.source_vertex {
            GraphSsspSource::I64(_) => None,
            GraphSsspSource::Input(input) => Some(input.input),
        }
        .into_iter()
    }
}
