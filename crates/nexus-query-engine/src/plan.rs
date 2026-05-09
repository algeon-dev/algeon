//! Native query plan IR.
//!
//! [`QueryPlan`](crate::plan::QueryPlan) is the DataFusion-free IR that
//! `execute_query_plan` runs.
//! Builder, node, expression-spec, facts, and graph-algorithm modules are
//! re-exported from here as the engine's public plan surface.

use std::{fmt, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::{
    expr::{AggregateFunction, AggregateSpec, ColumnRef, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec},
    source::SourceNode,
};

mod builder;
mod core;
mod display;
mod estimates;
mod facts;
mod graph_algorithm;
mod join;
mod nodes;
mod preference;
mod sink;
mod vector_algorithm;

pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use builder::QueryPlanBuilder;
pub(crate) use estimates::{
    estimated_local_parquet_source_bytes, estimated_output_bytes, estimated_output_rows,
    estimated_reachable_local_parquet_source_bytes,
};
pub use facts::{DataPlacement, PlanFacts, SchemaBehavior};
pub use graph_algorithm::{
    DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner,
    EmptyGraphEdgeInputAction, EmptyGraphInputPolicy, EmptyGraphSideInputPolicy,
    GRAPH_ALGORITHM_CAPABILITY_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
    GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT, GRAPH_OPERATION_REGISTRY, GraphAlgorithm,
    GraphAlgorithmLowerLevelContract, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
    GraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy, GraphCoreNumberParams,
    GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings,
    GraphEdgePredicateInput, GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams,
    GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLayoutRequirement,
    GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOperationDefinition, GraphOutDegreesAllParams,
    GraphOutputSchemaKind, GraphPageRankParams, GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
    GraphPredicateInput, GraphResidencyDescriptor, GraphResidencyOrientation, GraphResidencyPlanAnalysis,
    GraphSccParams, GraphSimilarityParams, GraphSourceFingerprint, GraphSourceFingerprintBypassReason,
    GraphSourceVertexRequirement, GraphSpectralClusteringParams, GraphSsspParams, GraphTriangleCountAllParams,
    GraphVertexIdTypeSupport, GraphVertexPairInput, GraphWccParams, GraphWeightPolicy, ResidentGraphKey,
    edge_plan_has_simple_directed_edge_topology, graph_residency_descriptor_for_edge_plan,
    graph_residency_descriptor_for_local_parquet_source, graph_result_schema_for_params, planned_resident_graph_key,
    stable_graph_result_schema,
};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use nodes::{
    AggregateNode, EdgeNormalizeNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, ShiftDirection,
    SortNode, UnionNode, WindowFrame, WindowFunction, WindowNode, WindowSpec,
};
pub use preference::{
    NativePlanCostDecision, NativePlanCostFacts, NativePlanCostOptions, NativePlanCostReason, native_plan_cost_decision,
};
pub use sink::ResultSink;
pub use vector_algorithm::{
    CUVS_BRUTE_FORCE_KNN_FUNCTION, CUVS_ID_TYPES, CUVS_KMEANS_FUNCTION, CUVS_OPERATION_REGISTRY, CUVS_PCA_FUNCTION,
    CuvsDenseVectorBinding, CuvsIdType, CuvsOperation, CuvsOperationDescriptor, CuvsOptionDefinition, CuvsOptionKind,
    CuvsOptionValidationError, CuvsOutputContract, CuvsOutputFieldDefinition, CuvsRelationArgumentSyntax,
    CuvsRelationRole, VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY, VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT,
    VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams, VectorAlgorithmSpec, VectorBruteForceKnnParams,
    VectorDistanceMetric, VectorKMeansInit, VectorKMeansMetric, VectorKMeansParams, VectorPcaParams, VectorPcaSolver,
    VectorRelationBinding, VectorRowIdBinding, canonicalize_cuvs_options, vector_brute_force_knn_output_schema,
    vector_kmeans_output_schema, vector_pca_output_schema,
};
