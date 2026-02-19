use super::*;

pub const GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY: &str = "graph_algorithm";
pub const GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT: &str = "unsupported_graph_algorithm_reservation_contract";

/// Lower-level cuGraph/cudf-rs contracts required before graph algorithms can
/// be admitted under strict bounded execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphAlgorithmStrictBoundedLowerLevelContract {
    pub name: &'static str,
    pub required_surface: &'static str,
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphAlgorithm {
    PageRank,
    Bfs,
    Louvain,
}

impl GraphAlgorithm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PageRank => "pagerank",
            Self::Bfs => "bfs",
            Self::Louvain => "louvain",
        }
    }
}

impl fmt::Display for GraphAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Edge-list column bindings consumed by a graph algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeColumnBindings {
    pub source: String,
    pub destination: String,
    pub weight: Option<String>,
}

impl GraphEdgeColumnBindings {
    pub fn new(source: impl Into<String>, destination: impl Into<String>, weight: Option<impl Into<String>>) -> Self {
        Self { source: source.into(), destination: destination.into(), weight: weight.map(Into::into) }
    }
}

/// Graph construction options shared by the initial cuGraph algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphConstructionOptions {
    pub directed: bool,
    pub renumber: bool,
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
    PythonCugraph,
    RawLibcugraph,
}

impl GraphConstructionPolicy {
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeNormalizeSpec {
    pub edge_columns: GraphEdgeColumnBindings,
    pub add_reverse: bool,
    pub dedup: EdgeDedupStrategy,
    pub weight_combiner: EdgeWeightCombiner,
}

impl EdgeNormalizeSpec {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDedupStrategy {
    None,
    KeepFirst,
    KeepLast,
}

impl EdgeDedupStrategy {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeWeightCombiner {
    None,
    MinWeight,
    KeepLastInputOrder,
}

impl EdgeWeightCombiner {
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

/// Algorithm-specific parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphAlgorithmParams {
    PageRank(GraphPageRankParams),
    Bfs(GraphBfsParams),
    Louvain(GraphLouvainParams),
}

impl GraphAlgorithmParams {
    pub const fn algorithm(&self) -> GraphAlgorithm {
        match self {
            Self::PageRank(_) => GraphAlgorithm::PageRank,
            Self::Bfs(_) => GraphAlgorithm::Bfs,
            Self::Louvain(_) => GraphAlgorithm::Louvain,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphPageRankParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for GraphPageRankParams {
    fn default() -> Self {
        Self { alpha: 0.85, epsilon: 1e-6, max_iterations: 100 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBfsParams {
    pub source_vertex: i64,
    pub depth_limit: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphLouvainParams {
    pub resolution: f64,
}

impl Default for GraphLouvainParams {
    fn default() -> Self {
        Self { resolution: 1.0 }
    }
}

/// DataFusion-free graph algorithm specification.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphAlgorithmSpec {
    pub algorithm: GraphAlgorithm,
    pub edge_columns: GraphEdgeColumnBindings,
    pub graph_options: GraphConstructionOptions,
    pub params: GraphAlgorithmParams,
    pub output_schema: SchemaRef,
}

impl GraphAlgorithmSpec {
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
    pub input: NodeId,
    pub spec: GraphAlgorithmSpec,
}

impl GraphAlgorithmNode {
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}

/// Stable SQL-facing result schema for graph algorithms.
pub fn stable_graph_result_schema(algorithm: GraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        GraphAlgorithm::PageRank => {
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
    };
    Arc::new(Schema::new(fields))
}
