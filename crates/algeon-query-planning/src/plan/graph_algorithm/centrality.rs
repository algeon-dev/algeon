//! Centrality algorithm parameter contracts.

use super::{GraphPredicateInput, NodeId};

/// PageRank parameters.
///
/// `alpha` is the damping factor, `epsilon` is the positive convergence
/// tolerance, and `max_iterations` is the positive iteration cap. Defaults
/// match the pinned local `cugraph` crate defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPageRankParams {
    /// Damping factor.
    pub alpha: f64,
    /// Positive convergence tolerance.
    pub epsilon: f64,
    /// Positive iteration cap.
    pub max_iterations: u32,
}

impl Default for GraphPageRankParams {
    fn default() -> Self {
        // Matches cugraph::centrality::PageRankParams::default().
        Self { alpha: 0.85, epsilon: 1e-5, max_iterations: 100 }
    }
}

/// Personalized PageRank parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPersonalizedPageRankParams {
    /// Shared PageRank solver options.
    pub pagerank: GraphPageRankParams,
    /// GPU-resident personalization vector side input.
    pub personalization: GraphPageRankPersonalizationInput,
}

impl GraphPersonalizedPageRankParams {
    /// Visits every side-input node id in stable role order.
    pub(crate) fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::once(self.personalization.input)
    }
}

impl Default for GraphPersonalizedPageRankParams {
    fn default() -> Self {
        Self {
            pagerank: GraphPageRankParams::default(),
            personalization: GraphPageRankPersonalizationInput::new(
                NodeId::from_index_for_internal_use(0),
                "vertex",
                "value",
            ),
        }
    }
}

/// Personalized PageRank side-input column binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPageRankPersonalizationInput {
    /// Native plan node producing the personalization relation.
    pub input: NodeId,
    /// Vertex id column in the personalization relation.
    pub vertex: String,
    /// Personalization value column in the personalization relation.
    pub value: String,
}

impl GraphPageRankPersonalizationInput {
    /// Creates a personalization input binding.
    pub fn new(input: NodeId, vertex: impl Into<String>, value: impl Into<String>) -> Self {
        Self { input, vertex: vertex.into(), value: value.into() }
    }
}

/// Parameters for cuGraph eigenvector centrality.
///
/// Mirrors `cugraph::centrality::EigenvectorCentralityParams` but stays free
/// of the `cugraph` crate's new type wrappers so the native IR can be constructed (and
/// validated) without depending on the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEigenvectorCentralityParams {
    /// Convergence tolerance for the power iteration. Must be > 0.
    pub epsilon: f64,
    /// Maximum power-iteration count. Must be >= 1.
    pub max_iterations: u32,
}

impl Default for GraphEigenvectorCentralityParams {
    fn default() -> Self {
        // Matches cugraph::centrality::EigenvectorCentralityParams::default().
        Self { epsilon: 1.0e-6, max_iterations: 200 }
    }
}

/// Parameters for cuGraph Katz centrality.
///
/// Mirrors `cugraph::centrality::KatzCentralityParams` but stays free of
/// the `cugraph` crate's new type wrappers so the native IR can be constructed and
/// validated without depending on the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphKatzCentralityParams {
    /// Katz attenuation factor. cuGraph requires this to be smaller than
    /// the inverse of the graph's maximum eigenvalue. Must be > 0; the
    /// adapter rejects 0 at planning so the kernel does not produce
    /// degenerate output.
    pub alpha: f64,
    /// Constant added to each vertex score in every iteration. Must be
    /// >= 0.
    pub beta: f64,
    /// Convergence tolerance. Must be > 0; libcugraph's convergence
    /// test never accepts on a non-empty graph at epsilon=0.
    pub epsilon: f64,
    /// Maximum number of power-iteration steps. Must be >= 1.
    pub max_iterations: u32,
}

impl Default for GraphKatzCentralityParams {
    fn default() -> Self {
        // Matches cugraph::centrality::KatzCentralityParams::default().
        Self { alpha: 0.01, beta: 1.0, epsilon: 1.0e-6, max_iterations: 1000 }
    }
}

/// Parameters for cuGraph HITS (Hyperlink-Induced Topic Search).
///
/// Mirrors `cugraph::centrality::HitsParams` minus the `cugraph` crate's
/// new type wrappers, so the native IR can be constructed without
/// depending on the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHitsParams {
    /// Convergence tolerance. Must be > 0; libcugraph's convergence
    /// test never accepts at epsilon=0 on a non-empty graph.
    pub epsilon: f64,
    /// Maximum number of power-iteration steps. Must be >= 1.
    pub max_iterations: u32,
    /// Whether to normalize the resulting hub/authority scores so their
    /// L1 norm sums to 1. Off by default to match the `cugraph` crate's
    /// `HitsParams::default()`.
    pub normalize: bool,
}

impl Default for GraphHitsParams {
    fn default() -> Self {
        // Matches cugraph::centrality::HitsParams::default().
        Self { epsilon: 1.0e-5, max_iterations: 100, normalize: false }
    }
}

/// Default upper bound on exact betweenness source count.
///
/// Exact betweenness runs one traversal per vertex. Calls above this
/// conservative upper bound must provide explicit seed vertices so the
/// approximation is visible in the request.
pub const DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD: u64 = 100_000;

/// Explicit seed set for approximate betweenness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphBetweennessSeeds {
    /// Host literal seed vertices from SQL options.
    I64(Vec<i64>),
    /// Already-encoded seed vertices supplied by a native side-input node.
    Input {
        /// Native node/column binding for the encoded seed values.
        input: GraphPredicateInput,
        /// Proven upper bound used by graph admission before execution.
        row_count_upper_bound: u64,
    },
}

impl GraphBetweennessSeeds {
    /// Literal seed count, or the proven row-count upper bound for encoded
    /// relation-backed seeds.
    pub fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::Input { row_count_upper_bound, .. } => usize::try_from(*row_count_upper_bound).unwrap_or(usize::MAX),
        }
    }

    /// Whether the seed list is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::I64(values) => values.is_empty(),
            Self::Input { row_count_upper_bound, .. } => *row_count_upper_bound == 0,
        }
    }
}

impl GraphBetweennessSeeds {
    /// Visits the encoded seed side-input node, when present.
    pub(crate) fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        match self {
            Self::I64(_) => None,
            Self::Input { input, .. } => Some(input.input),
        }
        .into_iter()
    }
}

/// Parameters for vertex betweenness centrality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBetweennessCentralityParams {
    /// Explicit seed vertices for approximate betweenness. `None` requests
    /// exact all-vertex betweenness unless `k` requests a deterministic
    /// approximate seed set.
    pub seeds: Option<GraphBetweennessSeeds>,
    /// Deterministic approximate seed count. Mutually exclusive with `seeds`;
    /// execution resolves it to the first `k` distinct graph vertices in stable
    /// order before calling cuGraph.
    pub k: Option<u64>,
    /// Whether to normalize the result.
    pub normalized: bool,
    /// Whether endpoints contribute to vertex betweenness.
    pub include_endpoints: bool,
    /// Conservative all-vertex threshold for exact execution.
    pub exact_vertex_threshold: u64,
}

impl GraphBetweennessCentralityParams {
    /// Visits explicit seed side-input nodes in stable role order.
    pub(crate) fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.seeds.iter().flat_map(GraphBetweennessSeeds::side_input_node_ids)
    }
}

impl Default for GraphBetweennessCentralityParams {
    fn default() -> Self {
        Self {
            seeds: None,
            k: None,
            normalized: true,
            include_endpoints: false,
            exact_vertex_threshold: DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD,
        }
    }
}

/// Parameters for edge betweenness centrality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeBetweennessCentralityParams {
    /// Explicit seed vertices for approximate edge betweenness. `None`
    /// requests exact all-vertex edge betweenness unless `k` requests a
    /// deterministic approximate seed set.
    pub seeds: Option<GraphBetweennessSeeds>,
    /// Deterministic approximate seed count. Mutually exclusive with `seeds`;
    /// execution resolves it to the first `k` distinct graph vertices in stable
    /// order before calling cuGraph.
    pub k: Option<u64>,
    /// Whether to normalize the result.
    pub normalized: bool,
    /// Conservative all-vertex threshold for exact execution.
    pub exact_vertex_threshold: u64,
}

impl GraphEdgeBetweennessCentralityParams {
    /// Visits explicit seed side-input nodes in stable role order.
    pub(crate) fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.seeds.iter().flat_map(GraphBetweennessSeeds::side_input_node_ids)
    }
}

impl Default for GraphEdgeBetweennessCentralityParams {
    fn default() -> Self {
        Self {
            seeds: None,
            k: None,
            normalized: true,
            exact_vertex_threshold: DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD,
        }
    }
}
