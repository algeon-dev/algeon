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
        AggregateNode, GraphAlgorithm, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan,
        ResultSink, WindowFunction,
    },
    source::{SourceNode, validate_local_parquet_source},
};

mod admission;
mod aggregate;
mod edge;
mod entry;
mod expr;
mod graph;
mod invariant;
mod join;
mod join_strategy;
mod model;
mod node;
mod shape;

#[doc(hidden)]
pub use aggregate::analyze_aggregate_strategies;
pub use aggregate::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStateContract, AggregateStrategyCandidateRejection,
    AggregateStrategyCapability, AggregateStrategyCapabilityDecision, AggregateStrategyExecutionOptions,
    aggregate_state_contract, aggregate_strategy_capability_decision,
};
pub use entry::analyze_query_capabilities;
pub use join::KeyedHashJoinStrategyOptions;
#[doc(hidden)]
pub use join::{
    JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
    KeylessConditionalJoinKind, join_node_execution_capability, join_residual_requires_post_join_filter,
    keyed_filtered_semi_anti_join_capable, keyed_hash_join_build_is_large, keyed_hash_join_capable,
    keyed_hash_join_strategy_decision_for_plan, keyed_hash_join_strategy_decision_for_plan_with_options,
    keyed_hash_join_strategy_decision_with_options, keyed_hash_join_strategy_input_facts_for_plan,
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options,
    keyed_residual_left_match_join_capable, keyed_residual_mixed_join_capable, keyless_outer_cross_join_capable,
    multi_join_keyed_residual_mixed_join_capable, multi_join_subtree_join_execution_capability,
    one_shot_equi_join_supported, partitioned_equi_join_supported, partitioned_keyed_residual_join_supported,
    projected_cross_join_capable, single_join_node_execution_capability,
};
#[doc(hidden)]
pub use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
pub use model::{
    AggregateGrouping, CapabilityAnalysis, CapabilityDecision, CapabilityEnvelope, CapabilityEnvelopeAdmission,
    CapabilityRejectionCategory, CapabilityRejectionOwner, CapabilityRejectionProjection, CapabilityScope,
    ConstraintRef, ConstraintScope, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, InvariantId,
    NodeCapability, NodeOperator, OrderingRequirement, ProducedProperty, ResidentResource, ResidualRequirement,
    ResultSinkKind, STATISTICAL_JOIN_FINAL_SORT_REQUIRES_EXACT_CONTRACTS, ScheduleConstraint, SourceFormatCapability,
    SubgraphInvariant, SubgraphInvariantKind, VectorInputRole,
};
#[doc(hidden)]
pub use node::is_cudf_median_input_type;
#[doc(hidden)]
pub use shape::capability_shape_contract_projection_ids;
