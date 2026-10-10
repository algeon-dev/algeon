//! Relation-aware cuGraph SQL planning and execution contracts.
//!
//! The adapter binds named relation and value arguments, validates canonical
//! child schemas, and lowers one logical graph node through the shared Algeon
//! query planner. Native graph IR and device execution remain owned below this
//! DataFusion boundary.

#[cfg(all(feature = "bench-support", feature = "cugraph"))]
#[doc(hidden)]
pub mod bench_support;
#[cfg(all(test, feature = "cugraph"))]
mod composed_planning_tests;
mod definitions;
mod error;
#[cfg(feature = "cugraph")]
mod exec;
mod gpu_family;
mod logical;
mod options;
/// Registry of cuGraph parity specs shared by parity tests and oracle-refresh tooling.
/// Hidden from the normal report contract; enable the `bench-support` feature to use it.
#[cfg(feature = "bench-support")]
#[doc(hidden)]
pub mod parity;
mod planning;
mod query_planner;
mod registry;
mod relation_planner;
mod types;
#[cfg(not(feature = "cugraph"))]
use error::feature_disabled;
use error::plan;
#[cfg(feature = "cugraph")]
pub use exec::CugraphAlgoExec;
#[cfg(feature = "cugraph")]
pub(crate) use exec::native_graph_input_execution_plan;
#[cfg(feature = "cugraph")]
pub(crate) use exec::singleflight::GraphBfsSingleflight;
#[cfg(feature = "cugraph")]
pub use exec::singleflight::{GraphBfsSingleflightEventSnapshot, GraphBfsSingleflightSnapshot};
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use exec::tests::CountingGpuResidentProducer;
pub(crate) use gpu_family::CugraphGpuFunctionFamily;
pub(crate) use logical::{CugraphLogicalNode, CugraphRelationRole};
pub(crate) use planning::is_feature_disabled_execution_plan;
pub(crate) use query_planner::CugraphExtensionPlanner;
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
pub(crate) use relation_planner::cugraph_relation_planner;
pub(crate) use types::{
    CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN, CUGRAPH_EDGE_WEIGHT_COLUMN,
    CugraphVertexIdDomain, PredicateSourceColumnRole, result_schema_for_vertex_domain,
    validate_similarity_pair_table_schema, vertex_id_domain_for_call,
};
#[cfg(feature = "cugraph")]
pub(crate) use types::{
    CUGRAPH_MASS_COLUMN, CUGRAPH_MOBILITY_COLUMN, CUGRAPH_RADIUS_COLUMN, CUGRAPH_VALUE_COLUMN, CUGRAPH_VERTEX_COLUMN,
    CUGRAPH_X_COLUMN, CUGRAPH_Y_COLUMN,
};
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
    CugraphSqlConfig, CugraphSsspParams, CugraphTriangleCountAllParams, CugraphVertexPairTable, CugraphVertexSelector,
    CugraphWccParams, GpuIslandHostBoundaryConfig, bfs_result_schema, result_schema_for_params, stable_result_schema,
};
