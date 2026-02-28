//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` walks a `QueryPlan` and returns a
//! `CapabilityAnalysis` that classifies every node, expression, edge, and
//! subgraph invariant. Rejections are local and structured (missing operator,
//! expression, Arrow type, edge contract, invariant, or reservation
//! contract); they never use legacy named "execution family" report keys.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    chunk::local_parquet_source_has_exact_read_preflight,
    error::QueryError,
    exec::validate_local_parquet_source,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnQualifier, Expr, GroupingSetsSpec,
        LiteralValue,
    },
    plan::{GraphAlgorithm, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan},
    runtime::ReservationKind,
    source::SourceNode,
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

pub(crate) use aggregate::analyze_aggregate_strategies;
pub use aggregate::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
    AggregateStrategyCapability, AggregateStrategyCapabilityDecision, aggregate_strategy_capability_decision,
};
pub use entry::analyze_query_capabilities;
pub(crate) use join::{
    JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
    KeylessConditionalJoinKind, LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD, LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD,
    LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD, SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD,
    SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD, join_node_execution_capability, join_residual_requires_post_join_filter,
    keyed_filtered_semi_anti_join_capable, keyed_hash_join_capable, keyed_hash_join_strategy_decision,
    keyed_hash_join_strategy_decision_for_plan, keyed_hash_join_strategy_input_facts_for_plan,
    multi_join_subtree_join_execution_capability, one_shot_equi_join_supported,
};
pub use model::{
    AggregateGrouping, BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS, CapabilityAnalysis, CapabilityDecision,
    CapabilityEnvelopeAdmission, CapabilityRejectionCategory, CapabilityRejectionOwner, CapabilityRejectionProjection,
    ConstraintRef, ConstraintScope, ContractQuality, ContractStatus, EdgeConsumer, EdgeContract, EdgeContractKind,
    InputRole, InvariantId, NodeCapability, NodeOperator, OrderingRequirement, ProducedProperty, RequiredProperty,
    ReservationRequirement, ReservationRequirementScope, ResidentResource, ResidualRequirement, ResultSinkKind,
    STRICT_UNRESOLVED_PREFLIGHT_GAPS, ScheduleConstraint, SharedProducerIdentity, SourceFormatCapability,
    StrictBoundedEnvelope, StrictUnresolvedPreflightGap, SubgraphInvariant, SubgraphInvariantKind,
    lower_level_unresolved_preflight_gap, lower_level_unresolved_preflight_gap_for_contract,
    strict_unresolved_preflight_gap,
};
pub use requirements::reservation_requirement_scope;
pub(crate) use shape::strict_bounded_shape_contract_projection_ids;
