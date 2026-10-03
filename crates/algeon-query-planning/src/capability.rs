//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` walks a `QueryPlan` and returns a
//! `CapabilityAnalysis` that classifies every node, expression, edge, and
//! subgraph invariant. Rejections are local and structured (missing operator,
//! expression, Arrow type, edge contract, or invariant); they use canonical
//! capability report keys rather than
//! execution-family labels.

use crate::{
    error::QueryError,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnQualifier, Expr, GroupingSetsSpec,
        LiteralValue,
    },
    plan::{
        GraphAlgorithm, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan, ResultSink,
    },
    source::{SourceNode, validate_local_parquet_source},
};

mod aggregate;
mod edge;
mod entry;
mod expr;
mod graph;
mod invariant;
mod join;
mod join_strategy;
mod lineage;
mod model;
mod node;
mod rejection;

pub use aggregate::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStateContract, AggregateStrategyCandidateRejection,
    AggregateStrategyCapabilityDecision, AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision,
};
pub use join::KeyedHashJoinStrategyOptions;
#[doc(hidden)]
pub use join::{
    JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
    KeylessConditionalJoinKind, join_node_execution_capability, keyed_filtered_semi_anti_join_capable,
    keyed_hash_join_build_is_large, keyed_hash_join_capable, keyed_hash_join_strategy_input_facts_for_plan,
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options,
    keyed_residual_left_match_join_capable, keyed_residual_mixed_join_capable, partitioned_equi_join_supported,
    partitioned_keyed_residual_join_supported,
};
pub(crate) use join::{
    join_residual_requires_post_join_filter, keyed_hash_join_strategy_decision_for_plan,
    keyed_hash_join_strategy_decision_with_options, one_shot_equi_join_supported,
};
#[doc(hidden)]
pub use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
pub use model::{
    AggregateGrouping, CapabilityAnalysis, CapabilityDecision, CapabilityRejectionCategory, CapabilityScope,
    EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, InvariantId, NodeCapability, NodeOperator,
    OrderingRequirement, ResidentResource, ResidualRequirement, ResultSinkKind, SourceFormatCapability,
    SubgraphInvariant, SubgraphInvariantKind, VectorInputRole,
};
#[doc(hidden)]
pub use node::is_cudf_median_input_type;
pub use rejection::{Reason, Rejection, RejectionOwner, UnsupportedPart};
