//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` walks a `QueryPlan` and returns a
//! `CapabilityAnalysis` that classifies every node, expression, edge, and
//! subgraph invariant. Rejections are local and structured (missing operator,
//! expression, Arrow type, edge contract, invariant, or reservation
//! contract); they use canonical capability report keys rather than
//! execution-family labels.

use crate::{
    chunk::local_parquet_source_has_exact_read_preflight,
    error::QueryError,
    execution_contracts::ReservationKind,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnQualifier, Expr, GroupingSetsSpec,
        LiteralValue,
    },
    plan::{
        AggregateNode, GraphAlgorithm, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan,
        WindowFunction,
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
mod model;
mod node;
mod requirements;
mod shape;

#[doc(hidden)]
pub use aggregate::analyze_aggregate_strategies;
pub use aggregate::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
    AggregateStrategyCapability, AggregateStrategyCapabilityDecision, AggregateStrategyExecutionOptions,
    aggregate_strategy_capability_decision,
};
pub use entry::analyze_query_capabilities;
pub use join::KeyedHashJoinStrategyOptions;
#[doc(hidden)]
pub use join::{
    JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
    KeylessConditionalJoinKind, hash_join_prebuild_reservation_unsupported_build_key_for_plan,
    join_node_execution_capability, join_residual_requires_post_join_filter, keyed_filtered_semi_anti_join_capable,
    keyed_hash_join_capable, keyed_hash_join_strategy_decision_for_plan,
    keyed_hash_join_strategy_decision_for_plan_with_options, keyed_hash_join_strategy_decision_with_options,
    keyed_hash_join_strategy_input_facts_for_plan,
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options,
    keyed_residual_mixed_join_capable, keyless_outer_cross_join_capable, multi_join_keyed_residual_mixed_join_capable,
    multi_join_subtree_join_execution_capability, one_shot_equi_join_supported, partitioned_equi_join_supported,
    partitioned_keyed_residual_join_supported, projected_cross_join_capable, single_join_node_execution_capability,
};
pub use model::{
    AggregateGrouping, BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS, CapabilityAnalysis, CapabilityDecision,
    CapabilityEnvelope, CapabilityEnvelopeAdmission, CapabilityRejectionCategory, CapabilityRejectionOwner,
    CapabilityRejectionProjection, ConstraintRef, ConstraintScope, ContractQuality, ContractStatus, EdgeConsumer,
    EdgeContract, EdgeContractKind, InputRole, InvariantId, NodeCapability, NodeOperator, OrderingRequirement,
    ProducedProperty, RequiredProperty, ReservationRequirement, ReservationRequirementScope, ResidentResource,
    ResidualRequirement, ResultSinkKind, STATISTICAL_JOIN_FINAL_SORT_REQUIRES_EXACT_CONTRACTS, ScheduleConstraint,
    SharedProducerIdentity, SourceFormatCapability, SubgraphInvariant, SubgraphInvariantKind,
    UNRESOLVED_PREFLIGHT_GAPS, UnresolvedPreflightGap, VectorInputRole, lower_level_unresolved_preflight_gap,
    lower_level_unresolved_preflight_gap_for_contract, unresolved_preflight_gap,
};
pub use requirements::reservation_requirement_scope;
#[doc(hidden)]
pub use shape::capability_shape_contract_projection_ids;
