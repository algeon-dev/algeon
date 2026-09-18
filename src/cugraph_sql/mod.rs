//! SQL-facing cuGraph table-function contracts.
//!
//! This family recognizes table-function calls in `parser`, resolves their
//! named relations through `table_function`, and executes native graph plans
//! through `exec`. Registry identity, option validation, call types, and GPU
//! metadata remain separate owners even though planning begins at the table
//! provider rather than a relation-aware logical node.

#[cfg(all(test, feature = "cugraph"))]
#[path = "composed_planning_tests.rs"]
mod composed_planning_tests;
mod error;
#[cfg(feature = "cugraph")]
mod exec;
mod gpu_family;
mod options;
/// Registry of cuGraph parity specs shared by parity tests and oracle-refresh tooling.
/// Hidden from the normal report contract; enable the `bench-support` feature to use it.
#[cfg(feature = "bench-support")]
#[doc(hidden)]
pub mod parity;
mod parser;
mod registry;
mod table_function;
mod types;
#[cfg(not(feature = "cugraph"))]
use error::feature_disabled as cugraph_feature_disabled;
use error::plan as plan_error;
#[cfg(feature = "cugraph")]
pub use exec::CugraphAlgoExec;
#[cfg(feature = "cugraph")]
pub(crate) use exec::native_graph_input_execution_plan;
pub(crate) use gpu_family::CugraphGpuFunctionFamily;
pub use parser::parse_cugraph_algorithm_spec;
#[cfg(feature = "cugraph")]
pub(crate) use registry::CugraphLogicalOutputMultiplicity;
pub use registry::{
    CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_COSINE_FUNCTION, CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_ECG_FUNCTION,
    CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
    CUGRAPH_FORCE_ATLAS2_FUNCTION, CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_JACCARD_FUNCTION,
    CUGRAPH_K_CORE_FUNCTION, CUGRAPH_KATZ_CENTRALITY_FUNCTION, CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LOUVAIN_FUNCTION,
    CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION, CUGRAPH_OPERATION_REGISTRY, CUGRAPH_OUT_DEGREES_ALL_FUNCTION,
    CUGRAPH_OVERLAP_FUNCTION, CUGRAPH_PAGERANK_FUNCTION, CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION,
    CUGRAPH_SORENSEN_FUNCTION, CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, CugraphAlgorithm, CugraphOperationDefinition,
};
#[cfg(feature = "cugraph")]
pub use table_function::CugraphAlgoTableProvider;
pub(crate) use table_function::{cugraph_table_functions, is_feature_disabled_execution_plan};
pub use types::{
    CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessCentralityParams, CugraphBetweennessSeeds,
    CugraphBfsOutputMode, CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsPredicateInputs, CugraphBfsSource,
    CugraphBfsSourceVertex, CugraphCoreNumberParams, CugraphDegreesAllParams, CugraphEcgParams,
    CugraphEdgeBetweennessCentralityParams, CugraphEigenvectorCentralityParams,
    CugraphForceAtlas2InitialPositionsTable, CugraphForceAtlas2Params, CugraphForceAtlas2VertexAttributesTable,
    CugraphGraphConstructionPolicy, CugraphGraphEdgePredicateTable, CugraphGraphOptions, CugraphGraphPredicateTable,
    CugraphHitsParams, CugraphInDegreesAllParams, CugraphKCoreDegreeType, CugraphKCoreParams,
    CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams, CugraphMstParams,
    CugraphOutDegreesAllParams, CugraphPageRankParams, CugraphPageRankPersonalizationTable,
    CugraphPersonalizedPageRankParams, CugraphSccParams, CugraphSimilarityParams, CugraphSpectralClusteringParams,
    CugraphSqlConfig, CugraphSsspParams, CugraphTableReference, CugraphTriangleCountAllParams, CugraphVertexPairTable,
    CugraphVertexSelector, CugraphWccParams, GpuIslandHostBoundaryConfig, bfs_result_schema, result_schema_for_params,
    stable_result_schema,
};
pub(crate) use types::{
    CugraphVertexIdDomain, PredicateSourceColumnRole, PredicateSourceSpec, result_schema_for_vertex_domain,
    validate_similarity_pair_table_schema, validate_vertex_id_domain_support_for_call, vertex_id_domain_for_call,
};
