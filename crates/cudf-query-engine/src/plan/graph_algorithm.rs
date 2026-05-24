//! DataFusion-free graph algorithm IR.
//!
//! This module defines stable graph algorithm names, graph construction
//! options, edge-list normalization, parameter defaults, and result schemas.
//! The IR can be built without the optional cuGraph feature; execution support
//! is selected later by the adapter/runtime.

use super::*;

/// Strict bounded memory-boundary label for graph algorithm rejection facts.
pub const GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY: &str = "graph_algorithm";
/// Strict bounded required-contract label for graph algorithm rejection facts.
pub const GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT: &str = "unsupported_graph_algorithm_reservation_contract";

/// Lower-level cuGraph/cudf-rs contracts required before graph algorithms can
/// be admitted under strict bounded execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphAlgorithmStrictBoundedLowerLevelContract {
    /// Stable short contract name.
    pub name: &'static str,
    /// Required lower-level cudf-rs/libcugraph surface before strict admission.
    pub required_surface: &'static str,
}

/// Lower-level graph contracts that currently block strict bounded admission.
pub const GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS: &[GraphAlgorithmStrictBoundedLowerLevelContract] = &[
    GraphAlgorithmStrictBoundedLowerLevelContract {
        name: "graph_construction_memory",
        required_surface: "preflight or hard upper-bound bytes for cuGraph graph construction from cudf table input",
    },
    GraphAlgorithmStrictBoundedLowerLevelContract {
        name: "renumbering_memory",
        required_surface: "preflight or hard upper-bound bytes for optional cuGraph vertex renumbering",
    },
    GraphAlgorithmStrictBoundedLowerLevelContract {
        name: "algorithm_workspace_memory",
        required_surface: "preflight or hard upper-bound bytes for algorithm-specific cuGraph workspace",
    },
    GraphAlgorithmStrictBoundedLowerLevelContract {
        name: "result_table_memory",
        required_surface: "preflight or hard upper-bound rows and bytes for the cuGraph result table",
    },
    GraphAlgorithmStrictBoundedLowerLevelContract {
        name: "retained_resource_lifetime_boundaries",
        required_surface: "deterministic ownership and release boundaries for graph and algorithm retained resources",
    },
];

/// Native graph algorithm supported by the cuGraph operator boundary.
///
/// Every variant must have a matching entry in [`GRAPH_OPERATION_REGISTRY`].
/// Ref: https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphAlgorithm {
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/centrality/#pagerank
    PageRank,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/traversal/#breadth-first-search-bfs
    Bfs,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#louvain
    Louvain,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/centrality/#eigenvector-centrality
    EigenvectorCentrality,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/centrality/#katz-centrality
    KatzCentrality,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/centrality/#hits-centrality
    Hits,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/traversal/#single-source-shortest-path-sssp
    Sssp,
    /// In-degree and out-degree for every vertex.
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.structure.graph_implementation.simplegraphimpl.degrees/
    DegreesAll,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.structure.graph_implementation.simplegraphimpl.in_degree/
    InDegreesAll,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.structure.graph_implementation.simplegraphimpl.out_degree/
    OutDegreesAll,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/labeling/#weakly-connected-components
    WeaklyConnectedComponents,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/labeling/#strongly-connected-components
    StronglyConnectedComponents,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#triangle-counting
    TriangleCountAll,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/core/#core-number
    CoreNumber,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/core/#k-core
    KCore,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.tree.minimum_spanning_tree.minimum_spanning_tree/
    MinimumSpanningTree,
    /// PageRank variant that uses strict convergence/error behavior.
    /// TODO: link?
    PageRankStrict,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#leiden
    Leiden,
    /// Ensemble clustering for graphs.
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#ecg
    Ecg,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#spectral-clustering-modularity-maximization
    SpectralModularityMaximization,
    /// https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph_c/community/#balanced-cut
    BalancedCutClustering,
}

impl GraphAlgorithm {
    /// Stable lowercase algorithm code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PageRank => "pagerank",
            Self::Bfs => "bfs",
            Self::Louvain => "louvain",
            Self::EigenvectorCentrality => "eigenvector_centrality",
            Self::KatzCentrality => "katz_centrality",
            Self::Hits => "hits",
            Self::Sssp => "sssp",
            Self::DegreesAll => "degrees_all",
            Self::InDegreesAll => "in_degrees_all",
            Self::OutDegreesAll => "out_degrees_all",
            Self::WeaklyConnectedComponents => "weakly_connected_components",
            Self::StronglyConnectedComponents => "strongly_connected_components",
            Self::TriangleCountAll => "triangle_count_all",
            Self::CoreNumber => "core_number",
            Self::KCore => "k_core",
            Self::MinimumSpanningTree => "minimum_spanning_tree",
            Self::PageRankStrict => "pagerank_strict",
            Self::Leiden => "leiden",
            Self::Ecg => "ecg",
            Self::SpectralModularityMaximization => "spectral_modularity_maximization",
            Self::BalancedCutClustering => "balanced_cut_clustering",
        }
    }

    /// Look up the registry definition for this algorithm.
    ///
    /// Panics if the algorithm variant lacks a registry entry. The registry is
    /// the single source of truth for algorithm metadata; new
    /// [`GraphAlgorithm`] variants must add an entry to
    /// [`GRAPH_OPERATION_REGISTRY`].
    pub fn definition(self) -> &'static GraphOperationDefinition {
        GRAPH_OPERATION_REGISTRY
            .iter()
            .find(|def| def.algorithm == self)
            .expect("graph operation registry is missing an algorithm variant")
    }

    /// All algorithms exposed by the registry, in registry order.
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        GRAPH_OPERATION_REGISTRY.iter().map(|def| def.algorithm)
    }
}

impl fmt::Display for GraphAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Required cuGraph graph-layout state for an algorithm.
///
/// Mirrors the `Graph<Forward>` / `Graph<Transposed>` type-state in cugraph-rs.
/// [`Any`](Self::Any) indicates that the algorithm runs under either layout;
/// the executor still picks one concrete layout at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphLayoutRequirement {
    /// Algorithm requires a forward graph layout.
    Forward,
    /// Algorithm requires a transposed graph layout.
    Transposed,
    /// Algorithm can run on either layout.
    Any,
}

impl GraphLayoutRequirement {
    /// Stable lowercase layout code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Transposed => "transposed",
            Self::Any => "any",
        }
    }
}

impl fmt::Display for GraphLayoutRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Registry-level metadata for a graph operation.
///
/// One entry per [`GraphAlgorithm`] variant. The registry is the place new
/// algorithms must declare their layout requirement, stable SQL name, and any
/// future metadata fields (parameter schema, weighted-edge support, etc.) so
/// the planner, executor, and SQL surface stay in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphOperationDefinition {
    /// Algorithm variant described by this registry row.
    pub algorithm: GraphAlgorithm,
    /// Stable identifier used in plan display, runtime metrics, and SQL
    /// function naming (`cugraph_<name>`).
    pub name: &'static str,
    /// cuGraph layout requirement for this operation.
    pub layout_requirement: GraphLayoutRequirement,
}

impl GraphOperationDefinition {
    /// Stable SQL-facing result schema for this algorithm.
    pub fn output_schema(&self) -> SchemaRef {
        stable_graph_result_schema(self.algorithm)
    }
}

/// Source-of-truth list of supported cuGraph operations.
///
/// Order is preserved by [`GraphAlgorithm::all`] and is the order in which
/// the SQL surface iterates algorithms when registering UDTFs.
pub const GRAPH_OPERATION_REGISTRY: &[GraphOperationDefinition] = &[
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::PageRank,
        name: "pagerank",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Bfs,
        name: "bfs",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Louvain,
        name: "louvain",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::EigenvectorCentrality,
        name: "eigenvector_centrality",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::KatzCentrality,
        name: "katz_centrality",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Hits,
        name: "hits",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Sssp,
        name: "sssp",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::DegreesAll,
        name: "degrees_all",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::InDegreesAll,
        name: "in_degrees_all",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::OutDegreesAll,
        name: "out_degrees_all",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::WeaklyConnectedComponents,
        name: "weakly_connected_components",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::StronglyConnectedComponents,
        name: "strongly_connected_components",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::TriangleCountAll,
        name: "triangle_count_all",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::CoreNumber,
        name: "core_number",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::KCore,
        name: "k_core",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::MinimumSpanningTree,
        name: "minimum_spanning_tree",
        layout_requirement: GraphLayoutRequirement::Any,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::PageRankStrict,
        name: "pagerank_strict",
        layout_requirement: GraphLayoutRequirement::Transposed,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Leiden,
        name: "leiden",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Ecg,
        name: "ecg",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::SpectralModularityMaximization,
        name: "spectral_modularity_maximization",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::BalancedCutClustering,
        name: "balanced_cut_clustering",
        layout_requirement: GraphLayoutRequirement::Forward,
    },
];

/// Edge-list column bindings consumed by a graph algorithm.
///
/// `weight` is optional. When present, duplicate-edge normalization follows
/// the selected [`GraphConstructionPolicy`] and [`EdgeWeightCombiner`] so
/// conflicting weights are deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeColumnBindings {
    /// Source-vertex column name in the edge-list input.
    pub source: String,
    /// Destination-vertex column name in the edge-list input.
    pub destination: String,
    /// Optional edge-weight column name.
    pub weight: Option<String>,
}

impl GraphEdgeColumnBindings {
    /// Create edge-list column bindings.
    pub fn new(source: impl Into<String>, destination: impl Into<String>, weight: Option<impl Into<String>>) -> Self {
        Self { source: source.into(), destination: destination.into(), weight: weight.map(Into::into) }
    }
}

/// Graph construction options shared by cuGraph algorithms.
///
/// Defaults match Python cuGraph: directed graph construction, vertex
/// renumbering enabled, and Python-compatible edge-list normalization. Changing
/// `directed` affects whether reverse edges are inserted by the
/// Python-compatible policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphConstructionOptions {
    /// Whether graph construction treats edges as directed.
    pub directed: bool,
    /// Whether cuGraph may renumber vertices internally.
    pub renumber: bool,
    /// Edge-list normalization policy used before graph construction.
    pub policy: GraphConstructionPolicy,
}

impl Default for GraphConstructionOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, policy: GraphConstructionPolicy::PythonCugraph }
    }
}

/// Graph construction policy understood by the native IR.
///
/// Mirrors `cugraph::cudf_interop::GraphConstructionPolicy`. See
/// `datafusion_cudf::CugraphGraphConstructionPolicy` for the SQL-level
/// semantics and trade-offs between the two variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphConstructionPolicy {
    /// Normalize edge lists to match Python cuGraph semantics.
    PythonCugraph,
    /// Pass edge lists directly to libcugraph-compatible construction.
    RawLibcugraph,
}

impl GraphConstructionPolicy {
    /// Stable lowercase construction-policy code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PythonCugraph => "python_cugraph",
            Self::RawLibcugraph => "raw_libcugraph",
        }
    }
}

impl fmt::Display for GraphConstructionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Edge-list normalization inserted before raw cuGraph construction.
///
/// The Python-compatible policy drops duplicate directed edges and inserts
/// reverse edges for undirected graphs before cuGraph construction. Weighted
/// duplicate conflicts use [`EdgeWeightCombiner`] to define the retained value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeNormalizeSpec {
    /// Edge-list columns used by normalization.
    pub edge_columns: GraphEdgeColumnBindings,
    /// Whether reverse edges are inserted before deduplication.
    pub add_reverse: bool,
    /// Duplicate-edge retention policy.
    pub dedup: EdgeDedupStrategy,
    /// Duplicate-weight conflict policy.
    pub weight_combiner: EdgeWeightCombiner,
}

impl EdgeNormalizeSpec {
    /// Build the Python-compatible edge normalization spec.
    pub fn python_cugraph(edge_columns: GraphEdgeColumnBindings, graph_options: &GraphConstructionOptions) -> Self {
        let has_weight = edge_columns.weight.is_some();
        let (dedup, weight_combiner) = match (graph_options.directed, has_weight) {
            (_, false) => (EdgeDedupStrategy::KeepLast, EdgeWeightCombiner::None),
            (true, true) => (EdgeDedupStrategy::KeepLast, EdgeWeightCombiner::KeepLastInputOrder),
            (false, true) => (EdgeDedupStrategy::KeepFirst, EdgeWeightCombiner::MinWeight),
        };
        Self { edge_columns: edge_columns.clone(), add_reverse: !graph_options.directed, dedup, weight_combiner }
    }
}

/// Duplicate-edge handling applied by edge-list normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDedupStrategy {
    /// Keep duplicate edges.
    None,
    /// Keep the first duplicate edge in input order.
    KeepFirst,
    /// Keep the last duplicate edge in input order.
    KeepLast,
}

impl EdgeDedupStrategy {
    /// Stable lowercase duplicate-edge policy code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::KeepFirst => "keep_first",
            Self::KeepLast => "keep_last",
        }
    }
}

impl fmt::Display for EdgeDedupStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Weighted duplicate-edge conflict policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeWeightCombiner {
    /// No weight combination is required.
    None,
    /// Keep the minimum weight among duplicate edges.
    MinWeight,
    /// Keep the weight from the last duplicate in input order.
    KeepLastInputOrder,
}

impl EdgeWeightCombiner {
    /// Stable lowercase weight-combiner code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MinWeight => "min_weight",
            Self::KeepLastInputOrder => "keep_last_input_order",
        }
    }
}

impl fmt::Display for EdgeWeightCombiner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Degree direction for core-number and k-core computations.
///
/// Mirrors `cugraph::structure::KCoreDegreeType` without depending on the
/// cugraph feature so the native IR can be constructed without the cuGraph
/// feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphKCoreDegreeType {
    /// Use incoming edges.
    In,
    /// Use outgoing edges.
    Out,
    #[default]
    /// Use both incoming and outgoing edges.
    InOut,
}

impl GraphKCoreDegreeType {
    /// Stable lowercase degree-type code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "in_out",
        }
    }
}

impl fmt::Display for GraphKCoreDegreeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Algorithm-specific parameters.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum GraphAlgorithmParams {
    PageRank(GraphPageRankParams),
    Bfs(GraphBfsParams),
    Louvain(GraphLouvainParams),
    EigenvectorCentrality(GraphEigenvectorCentralityParams),
    KatzCentrality(GraphKatzCentralityParams),
    Hits(GraphHitsParams),
    Sssp(GraphSsspParams),
    DegreesAll(GraphDegreesAllParams),
    InDegreesAll(GraphInDegreesAllParams),
    OutDegreesAll(GraphOutDegreesAllParams),
    WeaklyConnectedComponents(GraphWccParams),
    StronglyConnectedComponents(GraphSccParams),
    TriangleCountAll(GraphTriangleCountAllParams),
    CoreNumber(GraphCoreNumberParams),
    KCore(GraphKCoreParams),
    MinimumSpanningTree(GraphMstParams),
    PageRankStrict(GraphPageRankParams),
    Leiden(GraphLeidenParams),
    Ecg(GraphEcgParams),
    SpectralModularityMaximization(GraphSpectralClusteringParams),
    BalancedCutClustering(GraphSpectralClusteringParams),
}

impl GraphAlgorithmParams {
    /// Algorithm variant associated with this parameter payload.
    pub const fn algorithm(&self) -> GraphAlgorithm {
        match self {
            Self::PageRank(_) => GraphAlgorithm::PageRank,
            Self::Bfs(_) => GraphAlgorithm::Bfs,
            Self::Louvain(_) => GraphAlgorithm::Louvain,
            Self::EigenvectorCentrality(_) => GraphAlgorithm::EigenvectorCentrality,
            Self::KatzCentrality(_) => GraphAlgorithm::KatzCentrality,
            Self::Hits(_) => GraphAlgorithm::Hits,
            Self::Sssp(_) => GraphAlgorithm::Sssp,
            Self::DegreesAll(_) => GraphAlgorithm::DegreesAll,
            Self::InDegreesAll(_) => GraphAlgorithm::InDegreesAll,
            Self::OutDegreesAll(_) => GraphAlgorithm::OutDegreesAll,
            Self::WeaklyConnectedComponents(_) => GraphAlgorithm::WeaklyConnectedComponents,
            Self::StronglyConnectedComponents(_) => GraphAlgorithm::StronglyConnectedComponents,
            Self::TriangleCountAll(_) => GraphAlgorithm::TriangleCountAll,
            Self::CoreNumber(_) => GraphAlgorithm::CoreNumber,
            Self::KCore(_) => GraphAlgorithm::KCore,
            Self::MinimumSpanningTree(_) => GraphAlgorithm::MinimumSpanningTree,
            Self::PageRankStrict(_) => GraphAlgorithm::PageRankStrict,
            Self::Leiden(_) => GraphAlgorithm::Leiden,
            Self::Ecg(_) => GraphAlgorithm::Ecg,
            Self::SpectralModularityMaximization(_) => GraphAlgorithm::SpectralModularityMaximization,
            Self::BalancedCutClustering(_) => GraphAlgorithm::BalancedCutClustering,
        }
    }
}

/// PageRank parameters.
///
/// `alpha` is the damping factor, `epsilon` is the positive convergence
/// tolerance, and `max_iterations` is the positive iteration cap. Defaults
/// match cuGraph's public defaults.
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
        Self { alpha: 0.85, epsilon: 1e-6, max_iterations: 100 }
    }
}

/// Breadth-first search parameters.
///
/// `source_vertex` is the traversal root. `depth_limit = None` means the
/// traversal is unbounded by depth; when set, the limit must be non-negative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBfsParams {
    /// Root vertex for traversal.
    pub source_vertex: i64,
    /// Optional non-negative traversal depth limit.
    pub depth_limit: Option<i64>,
}

/// Louvain community detection parameters.
///
/// `resolution` controls community granularity and defaults to `1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLouvainParams {
    /// Community granularity parameter.
    pub resolution: f64,
}

impl Default for GraphLouvainParams {
    fn default() -> Self {
        Self { resolution: 1.0 }
    }
}

/// Parameters for cuGraph eigenvector centrality.
///
/// Mirrors `cugraph::centrality::EigenvectorCentralityParams` but stays free
/// of cugraph-rs newtype wrappers so the native IR can be constructed (and
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
        Self { epsilon: 1.0e-6, max_iterations: 500 }
    }
}

/// Parameters for cuGraph Katz centrality.
///
/// Mirrors `cugraph::centrality::KatzCentralityParams` but stays free of
/// cugraph-rs newtype wrappers so the native IR can be constructed and
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
/// Mirrors `cugraph::centrality::HitsParams` minus the cugraph-rs
/// newtype wrappers, so the native IR can be constructed without
/// depending on the cuGraph feature.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHitsParams {
    /// Convergence tolerance. Must be > 0 — libcugraph's convergence
    /// test never accepts at epsilon=0 on a non-empty graph.
    pub epsilon: f64,
    /// Maximum number of power-iteration steps. Must be >= 1.
    pub max_iterations: u32,
    /// Whether to normalize the resulting hub/authority scores so their
    /// L1 norm sums to 1. Off by default to match the cugraph-rs
    /// `HitsParams::default()`.
    pub normalize: bool,
}

impl Default for GraphHitsParams {
    fn default() -> Self {
        // Matches cugraph::centrality::HitsParams::default().
        Self { epsilon: 1.0e-5, max_iterations: 100, normalize: false }
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

/// Zero-argument all-vertex degree computation.
///
/// No algorithm-specific options beyond the shared graph construction options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphDegreesAllParams;

/// Zero-argument all-vertex in-degree computation.
///
/// Direction is fixed by the algorithm; shared graph construction options
/// still control directedness, renumbering, and edge normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphInDegreesAllParams;

/// Zero-argument all-vertex out-degree computation.
///
/// Direction is fixed by the algorithm; shared graph construction options
/// still control directedness, renumbering, and edge normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphOutDegreesAllParams;

/// Zero-argument weakly connected components.
///
/// Component labels follow cuGraph's stable result schema for the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphWccParams;

/// Zero-argument strongly connected components.
///
/// Component labels follow cuGraph's stable result schema for the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphSccParams;

/// Zero-argument all-vertex triangle count.
///
/// Output schema is fixed by [`stable_graph_result_schema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphTriangleCountAllParams;

/// Parameters for core-number computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCoreNumberParams {
    /// Which edge direction to use when computing core numbers.
    pub degree_type: GraphKCoreDegreeType,
}

impl Default for GraphCoreNumberParams {
    fn default() -> Self {
        Self { degree_type: GraphKCoreDegreeType::InOut }
    }
}

/// Parameters for k-core subgraph extraction.
///
/// Mirrors `cugraph::structure::KCoreParams` without cugraph-rs newtype
/// wrappers so the native IR can be constructed without the cuGraph feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphKCoreParams {
    /// K value for the extracted core. Must be >= 1.
    pub k: u32,
    /// Degree direction to use when computing core numbers internally.
    pub degree_type: GraphKCoreDegreeType,
}

impl Default for GraphKCoreParams {
    fn default() -> Self {
        Self { k: 2, degree_type: GraphKCoreDegreeType::InOut }
    }
}

/// Parameters for minimum spanning tree extraction.
///
/// No algorithm-specific options beyond the shared graph construction options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphMstParams;

/// Parameters for Leiden community detection.
///
/// Mirrors `cugraph::community::LeidenParams` without the cugraph-rs newtype
/// wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLeidenParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// Modularity-gain scaling for the refinement phase. Must be > 0.
    pub theta: f64,
    /// RNG seed for the Leiden algorithm. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphLeidenParams {
    fn default() -> Self {
        // Matches cugraph::community::LeidenParams::default().
        Self { max_level: 100, resolution: 1.0, theta: 1.0, seed: 0 }
    }
}

/// Parameters for ECG community detection.
///
/// Mirrors `cugraph::community::EcgParams` without cugraph-rs newtype
/// wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEcgParams {
    /// Minimum edge weight in the final Louvain graph. Must be >= 0.
    pub min_weight: f64,
    /// Number of ensemble Louvain iterations. Must be >= 1.
    pub ensemble_size: u32,
    /// Maximum hierarchy depth for final Louvain. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for final Louvain. Must be >= 0.
    pub threshold: f64,
    /// Resolution parameter in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// RNG seed. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphEcgParams {
    fn default() -> Self {
        // Matches cugraph::community::EcgParams::default().
        Self { min_weight: 0.001, ensemble_size: 10, max_level: 100, threshold: 1.0e-7, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for spectral modularity maximization and balanced-cut clustering.
///
/// Mirrors `cugraph::community::SpectralClusteringParams` without cugraph-rs
/// newtype wrappers and without `do_expensive_check` (always false in the
/// adapter). Both `SpectralModularityMaximization` and the deprecated
/// `BalancedCutClustering` share this struct.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSpectralClusteringParams {
    /// Desired number of clusters. Must be >= 1.
    pub n_clusters: u32,
    /// Number of eigenvectors to use. Must be >= 1.
    pub n_eigenvectors: u32,
    /// Eigenvalue-solver tolerance. Must be >= 0.
    pub evs_tolerance: f64,
    /// Eigenvalue-solver maximum iteration count. Must be >= 1.
    pub evs_max_iterations: u32,
    /// k-means tolerance. Must be >= 0.
    pub k_means_tolerance: f64,
    /// k-means maximum iteration count. Must be >= 1.
    pub k_means_max_iterations: u32,
    /// RNG seed. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphSpectralClusteringParams {
    fn default() -> Self {
        // Matches cugraph::community::SpectralClusteringParams::default().
        Self {
            n_clusters: 2,
            n_eigenvectors: 2,
            evs_tolerance: 0.01,
            evs_max_iterations: 500,
            k_means_tolerance: 0.001,
            k_means_max_iterations: 100,
            seed: 0,
        }
    }
}

/// DataFusion-free graph algorithm specification.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphAlgorithmSpec {
    /// Algorithm to execute.
    pub algorithm: GraphAlgorithm,
    /// Edge-list column bindings.
    pub edge_columns: GraphEdgeColumnBindings,
    /// Shared graph construction options.
    pub graph_options: GraphConstructionOptions,
    /// Algorithm-specific parameter payload.
    pub params: GraphAlgorithmParams,
    /// Stable output schema expected from execution.
    pub output_schema: SchemaRef,
}

impl GraphAlgorithmSpec {
    /// Create a graph algorithm spec with an explicit output schema.
    pub fn new(
        algorithm: GraphAlgorithm,
        edge_columns: GraphEdgeColumnBindings,
        graph_options: GraphConstructionOptions,
        params: GraphAlgorithmParams,
        output_schema: SchemaRef,
    ) -> Self {
        assert_eq!(algorithm, params.algorithm(), "graph algorithm and parameter variant must match");
        Self { algorithm, edge_columns, graph_options, params, output_schema }
    }

    /// Create a graph algorithm spec using this crate's stable result schema.
    pub fn with_stable_schema(
        algorithm: GraphAlgorithm,
        edge_columns: GraphEdgeColumnBindings,
        graph_options: GraphConstructionOptions,
        params: GraphAlgorithmParams,
    ) -> Self {
        Self::new(algorithm, edge_columns, graph_options, params, stable_graph_result_schema(algorithm))
    }
}

/// Native graph algorithm operation over one edge-list input.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphAlgorithmNode {
    /// Edge-list input node.
    pub input: NodeId,
    /// Graph algorithm specification.
    pub spec: GraphAlgorithmSpec,
}

impl GraphAlgorithmNode {
    /// Output schema declared by the graph algorithm spec.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}

/// Stable SQL-facing result schema for graph algorithms.
pub fn stable_graph_result_schema(algorithm: GraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        GraphAlgorithm::PageRank | GraphAlgorithm::EigenvectorCentrality | GraphAlgorithm::KatzCentrality => {
            // Centrality-style algorithms expose one score per vertex.
            vec![
                Field::new("vertex", arrow::datatypes::DataType::Int64, false),
                Field::new("value", arrow::datatypes::DataType::Float64, false),
            ]
        },
        GraphAlgorithm::Bfs => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("distance", arrow::datatypes::DataType::Int64, false),
            Field::new("predecessor", arrow::datatypes::DataType::Int64, true),
        ],
        GraphAlgorithm::Louvain => {
            vec![
                Field::new("vertex", arrow::datatypes::DataType::Int64, false),
                Field::new("partition", arrow::datatypes::DataType::Int64, false),
            ]
        },
        GraphAlgorithm::Hits => {
            // HITS exposes both hub and authority scores per vertex;
            // upstream cuGraph names them `hub` / `authority`. The
            // adapter renames to `hub_score` / `authority_score` in
            // the stable SQL schema so the meaning is obvious to query
            // authors.
            vec![
                Field::new("vertex", arrow::datatypes::DataType::Int64, false),
                Field::new("hub_score", arrow::datatypes::DataType::Float64, false),
                Field::new("authority_score", arrow::datatypes::DataType::Float64, false),
            ]
        },
        GraphAlgorithm::Sssp => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            // SSSP distances are weighted (Float64); BFS uses Int64 hop counts.
            Field::new("distance", arrow::datatypes::DataType::Float64, false),
            // -1 sentinel means no predecessor (source vertex or unreachable);
            // nullable=true matches the BFS predecessor convention.
            Field::new("predecessor", arrow::datatypes::DataType::Int64, true),
        ],
        GraphAlgorithm::DegreesAll => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("in_degree", arrow::datatypes::DataType::Int64, false),
            Field::new("out_degree", arrow::datatypes::DataType::Int64, false),
        ],
        GraphAlgorithm::InDegreesAll => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("in_degree", arrow::datatypes::DataType::Int64, false),
        ],
        GraphAlgorithm::OutDegreesAll => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("out_degree", arrow::datatypes::DataType::Int64, false),
        ],
        // WCC and SCC both return a vertex-label assignment. Labels are
        // component identifiers and are not semantically ordered; parity
        // tests use partition-equivalence comparison.
        GraphAlgorithm::WeaklyConnectedComponents | GraphAlgorithm::StronglyConnectedComponents => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("label", arrow::datatypes::DataType::Int64, false),
        ],
        GraphAlgorithm::TriangleCountAll => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("triangle_count", arrow::datatypes::DataType::Int64, false),
        ],
        GraphAlgorithm::CoreNumber => vec![
            Field::new("vertex", arrow::datatypes::DataType::Int64, false),
            Field::new("core_number", arrow::datatypes::DataType::Int64, false),
        ],
        // KCore and MST return an edge list; column names follow the
        // graph-SQL `src`/`dst` convention. Weight is deferred to Phase 2.
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => vec![
            Field::new("src", arrow::datatypes::DataType::Int64, false),
            Field::new("dst", arrow::datatypes::DataType::Int64, false),
        ],
        // PageRank strict shares the same centrality schema as PageRank.
        GraphAlgorithm::PageRankStrict => {
            vec![
                Field::new("vertex", arrow::datatypes::DataType::Int64, false),
                Field::new("value", arrow::datatypes::DataType::Float64, false),
            ]
        },
        // Leiden, ECG, spectral modularity maximization, and balanced-cut
        // clustering all return a vertex-partition assignment. Labels are
        // partition identifiers; parity tests use partition-equivalence.
        GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::SpectralModularityMaximization
        | GraphAlgorithm::BalancedCutClustering => {
            vec![
                Field::new("vertex", arrow::datatypes::DataType::Int64, false),
                Field::new("partition", arrow::datatypes::DataType::Int64, false),
            ]
        },
    };
    Arc::new(Schema::new(fields))
}
