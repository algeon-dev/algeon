//! Native query plan IR.
//!
//! [`QueryPlan`] is the native IR that `execute_query_plan` runs.
//! Builder, node, facts, and graph-algorithm items are re-exported from here
//! as the engine's public plan surface.

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::{
    expr::{AggregateFunction, AggregateSpec, ColumnRef, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec},
    source::SourceNode,
};

mod advisory;
mod builder;
mod core;
mod display;
mod estimates;
mod facts;
mod gpu_function_options;
mod graph;
mod graph_algorithm;
mod identity;
mod join;
mod ml_algorithm;
mod ml_forest;
mod nodes;
mod preference;
mod refinement;
mod remap;
pub mod row_bounds;
mod sink;
mod uniqueness;
mod vector_algorithm;

pub(crate) use core::{Annotations, RewrittenNodes};
pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use advisory::{AdvisoryOutputBinding, AdvisoryOutputStatistics, AdvisoryStatistic, AdvisoryUnknownReason};
pub use builder::QueryPlanBuilder;
#[doc(hidden)]
pub use estimates::estimated_output_bytes;
pub(crate) use estimates::{advisory_output_bytes, advisory_output_rows, estimated_output_rows};
pub use facts::PlanFacts;
pub use gpu_function_options::{
    DenseVectorBinding, GpuOptionDefinition, GpuOptionKind, GpuOptionValidationError, GpuOutputFieldDefinition,
    RELATION_ID_TYPES, RelationIdType, VectorElementType,
};
pub use graph_algorithm::{
    DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner,
    EmptyGraphEdgeInputAction, EmptyGraphInputPolicy, EmptyGraphSideInputPolicy, GRAPH_OPERATION_REGISTRY,
    GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
    GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPathOutput, GraphBfsPredicateInputs,
    GraphConstructionOptions, GraphConstructionPolicy, GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams,
    GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings, GraphEdgePredicateInput,
    GraphEigenvectorCentralityParams, GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params,
    GraphForceAtlas2VertexAttributesInput, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType,
    GraphKCoreParams, GraphKatzCentralityParams, GraphLayoutRequirement, GraphLeidenParams, GraphLouvainParams,
    GraphMstParams, GraphOperationDefinition, GraphOutDegreesAllParams, GraphOutputSchemaKind, GraphPageRankParams,
    GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams, GraphPredicateInput, GraphResidencyDescriptor,
    GraphResidencyOrientation, GraphResidencyPlanAnalysis, GraphSccParams, GraphSimilarityParams,
    GraphSourceFingerprint, GraphSourceFingerprintBypassReason, GraphSourceVertexRequirement,
    GraphSpectralClusteringParams, GraphSsspParams, GraphSsspSource, GraphTriangleCountAllParams,
    GraphVertexIdTypeSupport, GraphVertexPairInput, GraphWccParams, GraphWeightPolicy, ResidentGraphKey,
    edge_plan_has_simple_directed_edge_topology, graph_residency_descriptor_for_edge_plan,
    graph_residency_descriptor_for_local_parquet_source, graph_result_schema_for_params,
    graph_result_schema_for_vertex_type, planned_resident_graph_key, stable_graph_result_schema,
};
#[doc(hidden)]
pub use identity::DeferredGraphExecutionScopeIdentity;
pub(crate) use identity::stable_hash_fields;
pub use identity::{PlanFingerprint, SchemaFingerprint};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use ml_algorithm::{
    CUML_FOREST_PREDICT_FUNCTION, CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION, CUML_OPERATION_REGISTRY,
    CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, CUML_TSVD_FUNCTION,
    CUML_UMAP_FUNCTION, CumlOperation, CumlOperationDescriptor, CumlOutputContract, MlAlgorithm, MlAlgorithmNode,
    MlAlgorithmParams, MlAlgorithmSpec, MlClusterSelectionMethod, MlDecompositionSolver, MlForestPredictParams,
    MlHdbscanParams, MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding, MlRelationRole,
    MlSplitCriterion, MlTsvdParams, MlUmapInit, MlUmapParams, canonicalize_cuml_options,
    ml_classification_output_schema, ml_embedding_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema,
};
pub use ml_forest::{
    MlForestModelDigest, MlForestModelFacts, MlForestModelFormat, MlForestModelSnapshot, MlForestOutputContract,
    MlForestPrecision, ml_forest_output_schema,
};
pub use nodes::{
    AggregateNode, EdgeNormalizeNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, ShiftDirection,
    SortNode, UnionNode, WindowFrame, WindowFunction, WindowNode, WindowSpec,
};
pub use preference::{NativePlanCostReason, native_plan_cost_decision};
#[cfg(test)]
pub(crate) use remap::remap_plan_node_inputs;
pub(crate) use remap::{NodeRemap, clone_plan_node_with_inputs, compact_reachable_nodes};
pub use sink::ResultSink;
pub use vector_algorithm::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_CAGRA_FUNCTION, CUVS_IVF_FLAT_FUNCTION, CUVS_IVF_PQ_FUNCTION,
    CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PAIRWISE_DISTANCE_FUNCTION, CUVS_PCA_FUNCTION,
    CUVS_TOP_K_FUNCTION, CuvsOperation, CuvsOperationDescriptor, CuvsOutputContract, CuvsRelationRole,
    CuvsVectorElementSupport, VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams, VectorAlgorithmSpec,
    VectorBruteForceKnnParams, VectorCagraBuildAlgo, VectorCagraParams, VectorCagraSearchAlgo, VectorDistanceMetric,
    VectorIvfFlatParams, VectorIvfPqCodebookKind, VectorIvfPqParams, VectorKMeansInit, VectorKMeansMetric,
    VectorKMeansParams, VectorPairwiseDistanceParams, VectorPcaParams, VectorPcaSolver, VectorRelationBinding,
    VectorRowIdBinding, VectorTopKParams, canonicalize_cuvs_options, vector_kmeans_output_schema,
    vector_knn_output_schema, vector_pairwise_distance_output_schema, vector_pca_output_schema,
    vector_top_k_output_schema,
};
