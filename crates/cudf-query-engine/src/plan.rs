use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};

use crate::{
    bounded::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    expr::{AggregateFunction, AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
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
mod sink;

pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use builder::QueryPlanBuilder;
pub(crate) use estimates::{estimated_output_bytes, estimated_output_rows};
pub use facts::{DataPlacement, PlanFacts, SchemaBehavior};
pub use graph_algorithm::{
    EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner, GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS,
    GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, GraphAlgorithm,
    GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec, GraphAlgorithmStrictBoundedLowerLevelContract,
    GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy, GraphEdgeColumnBindings, GraphLouvainParams,
    GraphPageRankParams, stable_graph_result_schema,
};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use nodes::{
    AggregateNode, EdgeNormalizeNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, SortNode, UnionNode,
    WindowFrame, WindowFunction, WindowNode, WindowSpec,
};
pub use sink::ResultSink;
