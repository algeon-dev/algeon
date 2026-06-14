//! Native query plan IR.
//!
//! [`QueryPlan`] is the DataFusion-free IR that `execute_query_plan` runs.
//! Builder, node, expression-spec, facts, and graph-algorithm modules are
//! re-exported from here as the engine's public plan surface.

use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};

use crate::{
    bounded::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    expr::{AggregateFunction, AggregateSpec, ColumnRef, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec},
    runtime::ExecutionBoundaryPolicy,
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

pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use builder::QueryPlanBuilder;
pub(crate) use estimates::{estimated_output_bytes, estimated_output_rows};
pub use facts::{DataPlacement, PlanFacts, SchemaBehavior};
pub use graph_algorithm::{
    EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner, GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS,
    GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT,
    GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphAlgorithmStrictBoundedLowerLevelContract, GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions,
    GraphConstructionPolicy, GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeColumnBindings,
    GraphEdgePredicateInput, GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams,
    GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLayoutRequirement,
    GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOperationDefinition, GraphOutDegreesAllParams,
    GraphOutputSchemaKind, GraphPageRankParams, GraphPredicateInput, GraphSccParams, GraphSourceVertexRequirement,
    GraphSpectralClusteringParams, GraphSsspParams, GraphTriangleCountAllParams, GraphWccParams, GraphWeightPolicy,
    graph_result_schema_for_params, stable_graph_result_schema,
};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use nodes::{
    AggregateNode, EdgeNormalizeNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, ShiftDirection,
    SortNode, UnionNode, WindowFrame, WindowFunction, WindowNode, WindowSpec,
};
pub use preference::{
    NativePlanCpuPreferenceReason, NativePlanPreferenceDecision, NativePlanPreferenceFacts,
    NativePlanPreferenceOptions, native_plan_preference_decision,
};
pub use sink::ResultSink;
