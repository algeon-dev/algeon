use std::{collections::BTreeSet, fmt, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use cudf_query_engine::GraphAlgorithm as NativeGraphAlgorithm;
use datafusion::common::{DataFusionError, Result};

use super::plan_error;
use crate::NativeCudfExecutionConfig;

pub const CUGRAPH_PAGERANK_FUNCTION: &str = "cugraph_pagerank";
pub const CUGRAPH_BFS_FUNCTION: &str = "cugraph_bfs";
pub const CUGRAPH_BFS_PATH_FUNCTION: &str = "cugraph_bfs_path";
pub const CUGRAPH_LOUVAIN_FUNCTION: &str = "cugraph_louvain";
pub const CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION: &str = "cugraph_eigenvector_centrality";
pub const CUGRAPH_KATZ_CENTRALITY_FUNCTION: &str = "cugraph_katz_centrality";
pub const CUGRAPH_HITS_FUNCTION: &str = "cugraph_hits";
pub const CUGRAPH_SSSP_FUNCTION: &str = "cugraph_sssp";
pub const CUGRAPH_DEGREES_ALL_FUNCTION: &str = "cugraph_degrees_all";
pub const CUGRAPH_IN_DEGREES_ALL_FUNCTION: &str = "cugraph_in_degrees_all";
pub const CUGRAPH_OUT_DEGREES_ALL_FUNCTION: &str = "cugraph_out_degrees_all";
pub const CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION: &str = "cugraph_weakly_connected_components";
pub const CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION: &str = "cugraph_strongly_connected_components";
pub const CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION: &str = "cugraph_triangle_count_all";
pub const CUGRAPH_CORE_NUMBER_FUNCTION: &str = "cugraph_core_number";
pub const CUGRAPH_K_CORE_FUNCTION: &str = "cugraph_k_core";
pub const CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION: &str = "cugraph_minimum_spanning_tree";
pub const CUGRAPH_PAGERANK_STRICT_FUNCTION: &str = "cugraph_pagerank_strict";
pub const CUGRAPH_LEIDEN_FUNCTION: &str = "cugraph_leiden";
pub const CUGRAPH_ECG_FUNCTION: &str = "cugraph_ecg";
pub const CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION: &str = "cugraph_spectral_modularity_maximization";
pub const CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION: &str = "cugraph_balanced_cut_clustering";
pub const CUGRAPH_LIST_ALGORITHMS_FUNCTION: &str = "cugraph_list_algorithms";
pub const CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION: &str = "cugraph_describe_algorithm";
pub const CUGRAPH_VALIDATE_CALL_FUNCTION: &str = "cugraph_validate_call";

#[derive(Debug, Clone)]
pub struct CugraphSqlConfig {
    /// SQL-visible cuGraph algorithms registered by `with_cugraph_sql`.
    pub allowed_algorithms: BTreeSet<CugraphAlgorithm>,
    /// Graph construction defaults applied before per-call `options_json`
    /// overrides.
    pub default_graph_options: CugraphGraphOptions,
    /// Whether experimental SQL functions should be exposed.
    pub experimental_functions: bool,
    /// Native execution policy used when a cuGraph edge source has not already
    /// been rewritten into a native cudf execution node.
    pub native_execution_config: NativeCudfExecutionConfig,
}

impl Default for CugraphSqlConfig {
    fn default() -> Self {
        Self {
            allowed_algorithms: CugraphAlgorithm::all().into_iter().collect(),
            default_graph_options: CugraphGraphOptions::default(),
            experimental_functions: false,
            native_execution_config: NativeCudfExecutionConfig::bounded_auto_sized(),
        }
    }
}

impl CugraphSqlConfig {
    pub fn with_allowed_algorithms(mut self, algorithms: impl IntoIterator<Item = CugraphAlgorithm>) -> Self {
        self.allowed_algorithms = algorithms.into_iter().collect();
        self
    }

    /// Set the native execution policy used by `CugraphAlgoExec`.
    ///
    /// The server uses this to keep cuGraph SQL execution aligned with the
    /// configured bounded/unbounded memory policy. If an edge-source child is
    /// already a native cudf execution node, cuGraph execution still inherits
    /// that child node's config so composed GPU-resident pipelines use one
    /// native context.
    pub fn with_native_execution_config(mut self, native_execution_config: NativeCudfExecutionConfig) -> Self {
        self.native_execution_config = native_execution_config;
        self
    }

    pub(crate) fn allows(&self, algorithm: CugraphAlgorithm) -> bool {
        self.allowed_algorithms.contains(&algorithm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CugraphAlgorithm {
    PageRank,
    Bfs,
    Louvain,
    EigenvectorCentrality,
    KatzCentrality,
    Hits,
    Sssp,
    DegreesAll,
    InDegreesAll,
    OutDegreesAll,
    WeaklyConnectedComponents,
    StronglyConnectedComponents,
    TriangleCountAll,
    CoreNumber,
    KCore,
    MinimumSpanningTree,
    PageRankStrict,
    Leiden,
    Ecg,
    SpectralModularityMaximization,
    BalancedCutClustering,
}

impl CugraphAlgorithm {
    /// All algorithms exposed by the SQL surface, in registry order.
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.algorithm)
    }

    pub fn function_name(self) -> &'static str {
        self.definition().sql_function_name
    }

    /// Look up the SQL-side registry definition for this algorithm.
    pub fn definition(self) -> &'static CugraphOperationDefinition {
        CUGRAPH_OPERATION_REGISTRY
            .iter()
            .find(|entry| entry.algorithm == self)
            .expect("cuGraph SQL operation registry is missing an algorithm variant")
    }

    pub fn native_algorithm(self) -> NativeGraphAlgorithm {
        self.definition().native_algorithm
    }
}

/// Registry-level metadata for a SQL-visible cuGraph operation.
///
/// One entry per [`CugraphAlgorithm`] variant. Mirrors
/// [`cudf_query_engine::GRAPH_OPERATION_REGISTRY`]: each SQL UDTF resolves to
/// the same native algorithm advertised by the engine, and the SQL function
/// name follows the `cugraph_<native_name>` convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CugraphOperationDefinition {
    pub algorithm: CugraphAlgorithm,
    pub native_algorithm: NativeGraphAlgorithm,
    pub sql_function_name: &'static str,
}

/// Source-of-truth list of SQL-visible cuGraph algorithms.
///
/// Order is preserved by [`CugraphAlgorithm::all`] and is the order in which
/// [`cugraph_table_functions`](crate::cugraph_table_functions) registers UDTFs.
pub const CUGRAPH_OPERATION_REGISTRY: &[CugraphOperationDefinition] = &[
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::PageRank,
        native_algorithm: NativeGraphAlgorithm::PageRank,
        sql_function_name: CUGRAPH_PAGERANK_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Bfs,
        native_algorithm: NativeGraphAlgorithm::Bfs,
        sql_function_name: CUGRAPH_BFS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Louvain,
        native_algorithm: NativeGraphAlgorithm::Louvain,
        sql_function_name: CUGRAPH_LOUVAIN_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::EigenvectorCentrality,
        native_algorithm: NativeGraphAlgorithm::EigenvectorCentrality,
        sql_function_name: CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::KatzCentrality,
        native_algorithm: NativeGraphAlgorithm::KatzCentrality,
        sql_function_name: CUGRAPH_KATZ_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Hits,
        native_algorithm: NativeGraphAlgorithm::Hits,
        sql_function_name: CUGRAPH_HITS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Sssp,
        native_algorithm: NativeGraphAlgorithm::Sssp,
        sql_function_name: CUGRAPH_SSSP_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::DegreesAll,
        native_algorithm: NativeGraphAlgorithm::DegreesAll,
        sql_function_name: CUGRAPH_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::InDegreesAll,
        native_algorithm: NativeGraphAlgorithm::InDegreesAll,
        sql_function_name: CUGRAPH_IN_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::OutDegreesAll,
        native_algorithm: NativeGraphAlgorithm::OutDegreesAll,
        sql_function_name: CUGRAPH_OUT_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::WeaklyConnectedComponents,
        native_algorithm: NativeGraphAlgorithm::WeaklyConnectedComponents,
        sql_function_name: CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::StronglyConnectedComponents,
        native_algorithm: NativeGraphAlgorithm::StronglyConnectedComponents,
        sql_function_name: CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::TriangleCountAll,
        native_algorithm: NativeGraphAlgorithm::TriangleCountAll,
        sql_function_name: CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::CoreNumber,
        native_algorithm: NativeGraphAlgorithm::CoreNumber,
        sql_function_name: CUGRAPH_CORE_NUMBER_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::KCore,
        native_algorithm: NativeGraphAlgorithm::KCore,
        sql_function_name: CUGRAPH_K_CORE_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::MinimumSpanningTree,
        native_algorithm: NativeGraphAlgorithm::MinimumSpanningTree,
        sql_function_name: CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::PageRankStrict,
        native_algorithm: NativeGraphAlgorithm::PageRankStrict,
        sql_function_name: CUGRAPH_PAGERANK_STRICT_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Leiden,
        native_algorithm: NativeGraphAlgorithm::Leiden,
        sql_function_name: CUGRAPH_LEIDEN_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Ecg,
        native_algorithm: NativeGraphAlgorithm::Ecg,
        sql_function_name: CUGRAPH_ECG_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::SpectralModularityMaximization,
        native_algorithm: NativeGraphAlgorithm::SpectralModularityMaximization,
        sql_function_name: CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::BalancedCutClustering,
        native_algorithm: NativeGraphAlgorithm::BalancedCutClustering,
        sql_function_name: CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION,
    },
];

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphAlgorithmSpec {
    pub algorithm: CugraphAlgorithm,
    pub table_name: CugraphTableReference,
    pub src_col: String,
    pub dst_col: String,
    pub weight_col: Option<String>,
    pub edge_id_col: Option<String>,
    pub graph_options: CugraphGraphOptions,
    pub params: CugraphAlgorithmParams,
    pub output_schema: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CugraphTableReference {
    parts: Vec<String>,
}

impl CugraphTableReference {
    pub fn parse(raw: &str) -> Result<Self> {
        let parts = raw.split('.').collect::<Vec<_>>();
        if !(1..=3).contains(&parts.len()) || parts.iter().any(|part| !is_simple_identifier(part)) {
            return Err(DataFusionError::Plan(format!(
                "cuGraph table source '{raw}' must be a simple table, schema.table, or catalog.schema.table identifier"
            )));
        }
        Ok(Self { parts: parts.into_iter().map(ToOwned::to_owned).collect() })
    }

    pub fn parts(&self) -> &[String] {
        &self.parts
    }
}

impl fmt::Display for CugraphTableReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphGraphOptions {
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: CugraphGraphConstructionPolicy,
}

impl Default for CugraphGraphOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, construction_policy: CugraphGraphConstructionPolicy::PythonCugraph }
    }
}

/// SQL-facing selector for the cuGraph edge-list ingress contract.
///
/// Mirrors `cugraph::cudf_interop::GraphConstructionPolicy` and is the public
/// `construction_policy` option exposed in the cuGraph table-function JSON.
///
/// - [`PythonCugraph`](Self::PythonCugraph): default. Match Python
///   `cugraph.Graph.from_cudf_edgelist(...)` semantics. The ingress layer
///   normalizes the edge DataFrame before raw graph construction (drop
///   duplicate directed edges according to Python cuGraph's direction-sensitive
///   weighted semantics, and add reverse edges for undirected simple graphs).
///   Safe for callers whose SQL has not pre-normalized the edge list. Carries
///   an extra sort + dedup cost and may import borrowed/prepared edge columns
///   into owned cudf objects, which breaks the zero-copy ingress path documented in
///   `cugraph::cudf_interop`.
/// - [`RawLibcugraph`](Self::RawLibcugraph): pass the SQL edge list straight
///   through to libcugraph without ingress-side normalization. Use when the
///   query plan (or the caller) already guarantees a well-formed edge list:
///   no duplicate directed edges in simple-graph mode, and reverse edges
///   already present for undirected simple graphs. Preserves zero-copy
///   ingress and avoids a second pass of dedup work that DataFusion can
///   already express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphGraphConstructionPolicy {
    PythonCugraph,
    RawLibcugraph,
}

impl CugraphGraphConstructionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PythonCugraph => "python_cugraph",
            Self::RawLibcugraph => "raw_libcugraph",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CugraphAlgorithmParams {
    PageRank(CugraphPageRankParams),
    Bfs(CugraphBfsParams),
    Louvain(CugraphLouvainParams),
    EigenvectorCentrality(CugraphEigenvectorCentralityParams),
    KatzCentrality(CugraphKatzCentralityParams),
    Hits(CugraphHitsParams),
    Sssp(CugraphSsspParams),
    DegreesAll(CugraphDegreesAllParams),
    InDegreesAll(CugraphInDegreesAllParams),
    OutDegreesAll(CugraphOutDegreesAllParams),
    WeaklyConnectedComponents(CugraphWccParams),
    StronglyConnectedComponents(CugraphSccParams),
    TriangleCountAll(CugraphTriangleCountAllParams),
    CoreNumber(CugraphCoreNumberParams),
    KCore(CugraphKCoreParams),
    MinimumSpanningTree(CugraphMstParams),
    PageRankStrict(CugraphPageRankParams),
    Leiden(CugraphLeidenParams),
    Ecg(CugraphEcgParams),
    SpectralModularityMaximization(CugraphSpectralClusteringParams),
    BalancedCutClustering(CugraphSpectralClusteringParams),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphPageRankParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for CugraphPageRankParams {
    fn default() -> Self {
        Self { alpha: 0.85, epsilon: 1e-6, max_iterations: 100 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CugraphBfsParams {
    /// Root vertex for the SQL BFS traversal.
    pub source_vertex: CugraphBfsSourceVertex,
    /// Optional non-negative BFS depth limit. `None` means unbounded depth.
    pub depth_limit: Option<i64>,
    /// Optional relation-backed predicate inputs.
    pub predicates: CugraphBfsPredicateInputs,
    /// SQL result-shape options. Defaults preserve the legacy raw BFS schema.
    pub output: CugraphBfsOutputOptions,
}

/// SQL-visible BFS source vertex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CugraphBfsSourceVertex {
    /// Numeric source vertex. This preserves the legacy SQL contract.
    Int64(i64),
    /// Logical string source vertex. The SQL execution layer maps it to a
    /// deterministic numeric cuGraph vertex id for one graph call.
    Utf8(String),
}

impl Default for CugraphBfsSourceVertex {
    fn default() -> Self {
        Self::Int64(0)
    }
}

impl CugraphBfsSourceVertex {
    pub const fn is_utf8(&self) -> bool {
        matches!(self, Self::Utf8(_))
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(value) => Some(*value),
            Self::Utf8(_) => None,
        }
    }
}

/// SQL graph vertex-id domain after applying the algorithm-specific logical-id
/// policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CugraphVertexIdDomain {
    Int32,
    Int64,
    StringLogical,
}

impl CugraphVertexIdDomain {
    pub(crate) fn accepts(&self, data_type: &DataType) -> bool {
        match self {
            Self::Int32 => matches!(data_type, DataType::Int32),
            Self::Int64 => matches!(data_type, DataType::Int64),
            Self::StringLogical => is_string_vertex_dtype(data_type),
        }
    }

    pub(crate) const fn is_string_logical(&self) -> bool {
        matches!(self, Self::StringLogical)
    }

    pub(crate) const fn label(&self) -> &'static str {
        match self {
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::StringLogical => "Arrow string",
        }
    }
}

pub(crate) fn vertex_id_domain_for_call(
    algorithm: CugraphAlgorithm,
    params: &CugraphAlgorithmParams,
    src_type: &DataType,
    dst_type: &DataType,
) -> Result<CugraphVertexIdDomain> {
    let is_bfs_string_source = algorithm == CugraphAlgorithm::Bfs
        && matches!(params, CugraphAlgorithmParams::Bfs(params) if params.source_vertex.is_utf8());
    if is_bfs_string_source {
        if !is_string_vertex_dtype(src_type) {
            return Err(plan_error(format!(
                "cuGraph BFS string source_vertex requires a string source column, found {src_type:?}"
            )));
        }
        if !is_string_vertex_dtype(dst_type) {
            return Err(plan_error(format!(
                "cuGraph BFS string source_vertex requires a string destination column, found {dst_type:?}"
            )));
        }
        return Ok(CugraphVertexIdDomain::StringLogical);
    }

    if algorithm == CugraphAlgorithm::Bfs && (is_string_vertex_dtype(src_type) || is_string_vertex_dtype(dst_type)) {
        return Err(plan_error(format!(
            "cuGraph BFS numeric source_vertex requires Int32 or Int64 edge columns, found source={src_type:?}, destination={dst_type:?}"
        )));
    }

    if !is_numeric_vertex_dtype(src_type) {
        return Err(plan_error(format!(
            "cuGraph {} source column has unsupported dtype {src_type:?}",
            algorithm.function_name()
        )));
    }
    if !is_numeric_vertex_dtype(dst_type) {
        return Err(plan_error(format!(
            "cuGraph {} destination column has unsupported dtype {dst_type:?}",
            algorithm.function_name()
        )));
    }
    if src_type != dst_type {
        return Err(plan_error("cuGraph source and destination columns must have the same numeric vertex dtype"));
    }
    match src_type {
        DataType::Int32 => Ok(CugraphVertexIdDomain::Int32),
        DataType::Int64 => Ok(CugraphVertexIdDomain::Int64),
        _ => unreachable!("numeric vertex dtype checked above"),
    }
}

pub(crate) fn is_numeric_vertex_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Int32 | DataType::Int64)
}

pub(crate) fn is_string_vertex_dtype(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// SQL-visible BFS output options.
///
/// `mode = Raw` preserves the stable legacy schema and cuGraph distance
/// sentinel values. `mode = Normalized` adds `reachable` and converts
/// unreachable distance / absent predecessor sentinels to nulls. `mode = Path`
/// reconstructs one source-to-target path from the BFS predecessor output.
/// `return_target_info = true` is an explicit schema-widening request that
/// requires a target-vertex predicate and appends `target_found` plus nullable
/// `target_distance` to every output row for raw / normalized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphBfsOutputOptions {
    /// Result schema and sentinel-normalization mode.
    pub mode: CugraphBfsOutputMode,
    /// Whether target metadata is requested.
    pub return_target_info: bool,
}

/// BFS SQL result-shape selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CugraphBfsOutputMode {
    /// Preserve the raw `vertex`, `distance`, `predecessor` schema.
    #[default]
    Raw,
    /// Return `vertex`, nullable `distance`, nullable `predecessor`, and
    /// `reachable`.
    Normalized,
    /// Return one row per vertex on the reconstructed source-to-target path.
    Path,
}

impl CugraphBfsOutputMode {
    /// Stable JSON label used by `output_mode` and metadata UDTFs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Normalized => "normalized",
            Self::Path => "path",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CugraphBfsPredicateInputs {
    pub include_vertices: Option<CugraphGraphPredicateTable>,
    pub exclude_vertices: Option<CugraphGraphPredicateTable>,
    pub target_vertices: Option<CugraphGraphPredicateTable>,
    pub include_edge_ids: Option<CugraphGraphPredicateTable>,
    pub include_edges: Option<CugraphGraphEdgePredicateTable>,
}

impl CugraphBfsPredicateInputs {
    pub fn has_inputs(&self) -> bool {
        self.include_vertices.is_some()
            || self.exclude_vertices.is_some()
            || self.target_vertices.is_some()
            || self.include_edge_ids.is_some()
            || self.include_edges.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CugraphGraphPredicateTable {
    pub table_name: CugraphTableReference,
    pub column: String,
}

impl CugraphGraphPredicateTable {
    pub fn new(table_name: CugraphTableReference, column: impl Into<String>) -> Self {
        Self { table_name, column: column.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CugraphGraphEdgePredicateTable {
    pub table_name: CugraphTableReference,
    pub source_col: String,
    pub destination_col: String,
    pub edge_id_col: Option<String>,
}

impl CugraphGraphEdgePredicateTable {
    pub fn new(
        table_name: CugraphTableReference,
        source_col: impl Into<String>,
        destination_col: impl Into<String>,
        edge_id_col: Option<impl Into<String>>,
    ) -> Self {
        Self {
            table_name,
            source_col: source_col.into(),
            destination_col: destination_col.into(),
            edge_id_col: edge_id_col.map(Into::into),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphLouvainParams {
    pub resolution: f64,
}

impl Default for CugraphLouvainParams {
    fn default() -> Self {
        Self { resolution: 1.0 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphEigenvectorCentralityParams {
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for CugraphEigenvectorCentralityParams {
    fn default() -> Self {
        // Mirrors `cudf_query_engine::GraphEigenvectorCentralityParams::default`.
        Self { epsilon: 1.0e-6, max_iterations: 500 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphKatzCentralityParams {
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub max_iterations: u32,
}

impl Default for CugraphKatzCentralityParams {
    fn default() -> Self {
        // Mirrors `cudf_query_engine::GraphKatzCentralityParams::default`.
        Self { alpha: 0.01, beta: 1.0, epsilon: 1.0e-6, max_iterations: 1000 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphHitsParams {
    pub epsilon: f64,
    pub max_iterations: u32,
    pub normalize: bool,
}

impl Default for CugraphHitsParams {
    fn default() -> Self {
        // Mirrors `cudf_query_engine::GraphHitsParams::default`.
        Self { epsilon: 1.0e-5, max_iterations: 100, normalize: false }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CugraphSsspParams {
    pub source_vertex: i64,
    /// Maximum edge-weight sum to consider. `None` means no cutoff; must be >= 0 when set.
    pub cutoff: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphInDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphOutDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphWccParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphSccParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphTriangleCountAllParams;

/// Degree direction for core-number and k-core SQL parameters.
///
/// Mirrors `cudf_query_engine::GraphKCoreDegreeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CugraphKCoreDegreeType {
    In,
    Out,
    #[default]
    InOut,
}

impl CugraphKCoreDegreeType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "in_out",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CugraphCoreNumberParams {
    pub degree_type: CugraphKCoreDegreeType,
}

impl Default for CugraphCoreNumberParams {
    fn default() -> Self {
        Self { degree_type: CugraphKCoreDegreeType::InOut }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CugraphKCoreParams {
    pub k: u32,
    pub degree_type: CugraphKCoreDegreeType,
}

impl Default for CugraphKCoreParams {
    fn default() -> Self {
        // Mirrors `cudf_query_engine::GraphKCoreParams::default`.
        Self { k: 2, degree_type: CugraphKCoreDegreeType::InOut }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphMstParams;

/// Parameters for Leiden community detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphLeidenParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// Modularity-gain scaling for the refinement phase. Must be > 0.
    pub theta: f64,
    /// RNG seed. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for CugraphLeidenParams {
    fn default() -> Self {
        // Matches cudf_query_engine::GraphLeidenParams::default().
        Self { max_level: 100, resolution: 1.0, theta: 1.0, seed: 0 }
    }
}

/// Parameters for ECG community detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphEcgParams {
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

impl Default for CugraphEcgParams {
    fn default() -> Self {
        // Matches cudf_query_engine::GraphEcgParams::default().
        Self { min_weight: 0.001, ensemble_size: 10, max_level: 100, threshold: 1.0e-7, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for spectral modularity maximization and balanced-cut clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphSpectralClusteringParams {
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

impl Default for CugraphSpectralClusteringParams {
    fn default() -> Self {
        // Matches cudf_query_engine::GraphSpectralClusteringParams::default().
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

pub fn stable_result_schema(algorithm: CugraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::EigenvectorCentrality | CugraphAlgorithm::KatzCentrality => {
            // Centrality-style algorithms expose one score per vertex.
            vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)]
        },
        CugraphAlgorithm::Bfs => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
        CugraphAlgorithm::Louvain => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)]
        },
        CugraphAlgorithm::Hits => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("hub_score", DataType::Float64, false),
            Field::new("authority_score", DataType::Float64, false),
        ],
        CugraphAlgorithm::Sssp => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Float64, false),
            Field::new("predecessor", DataType::Int64, true),
        ],
        CugraphAlgorithm::DegreesAll => vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("in_degree", DataType::Int64, false),
            Field::new("out_degree", DataType::Int64, false),
        ],
        CugraphAlgorithm::InDegreesAll => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("in_degree", DataType::Int64, false)]
        },
        CugraphAlgorithm::OutDegreesAll => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("out_degree", DataType::Int64, false)]
        },
        CugraphAlgorithm::WeaklyConnectedComponents | CugraphAlgorithm::StronglyConnectedComponents => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("label", DataType::Int64, false)]
        },
        CugraphAlgorithm::TriangleCountAll => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("triangle_count", DataType::Int64, false)]
        },
        CugraphAlgorithm::CoreNumber => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("core_number", DataType::Int64, false)]
        },
        CugraphAlgorithm::KCore | CugraphAlgorithm::MinimumSpanningTree => {
            vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)]
        },
        CugraphAlgorithm::PageRankStrict => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("value", DataType::Float64, false)]
        },
        CugraphAlgorithm::Leiden
        | CugraphAlgorithm::Ecg
        | CugraphAlgorithm::SpectralModularityMaximization
        | CugraphAlgorithm::BalancedCutClustering => {
            vec![Field::new("vertex", DataType::Int64, false), Field::new("partition", DataType::Int64, false)]
        },
    };
    Arc::new(Schema::new(fields))
}

/// SQL output schema for one BFS call after applying output options.
pub fn bfs_result_schema(params: &CugraphBfsParams) -> SchemaRef {
    let vertex_type = if params.source_vertex.is_utf8() { DataType::Utf8 } else { DataType::Int64 };
    let predecessor_type = vertex_type.clone();
    let mut fields = match params.output.mode {
        CugraphBfsOutputMode::Raw if params.source_vertex.is_utf8() => vec![
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", predecessor_type, true),
        ],
        CugraphBfsOutputMode::Raw => {
            stable_result_schema(CugraphAlgorithm::Bfs).fields().iter().map(|field| field.as_ref().clone()).collect()
        },
        CugraphBfsOutputMode::Normalized => vec![
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", predecessor_type, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
        CugraphBfsOutputMode::Path => vec![
            Field::new("path_index", DataType::Int64, false),
            Field::new("source", vertex_type.clone(), false),
            Field::new("target", vertex_type.clone(), false),
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, false),
        ],
    };
    if params.output.mode != CugraphBfsOutputMode::Path && params.output.return_target_info {
        fields.push(Field::new("target_found", DataType::Boolean, false));
        fields.push(Field::new("target_distance", DataType::Int64, true));
    }
    Arc::new(Schema::new(fields))
}

/// SQL output schema for one parsed cuGraph algorithm call.
pub fn result_schema_for_params(algorithm: CugraphAlgorithm, params: &CugraphAlgorithmParams) -> SchemaRef {
    match (algorithm, params) {
        (CugraphAlgorithm::Bfs, CugraphAlgorithmParams::Bfs(params)) => bfs_result_schema(params),
        _ => stable_result_schema(algorithm),
    }
}

fn is_simple_identifier(part: &str) -> bool {
    let mut chars = part.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
