//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` walks a `QueryPlan` and returns a
//! `CapabilityAnalysis` that classifies every node, expression, edge, and
//! subgraph invariant. Rejections are local and structured (missing operator,
//! expression, Arrow type, edge contract, invariant, or reservation
//! contract); they use canonical capability report keys rather than
//! execution-family labels.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    chunk::local_parquet_source_has_exact_read_preflight,
    error::QueryError,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnQualifier, Expr, GroupingSetsSpec,
        LiteralValue,
    },
    plan::{
        AggregateNode, GraphAlgorithm, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan,
        WindowFunction,
    },
    runtime::ReservationKind,
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

pub(crate) use aggregate::analyze_aggregate_strategies;
pub use aggregate::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
    AggregateStrategyCapability, AggregateStrategyCapabilityDecision, AggregateStrategyExecutionOptions,
    aggregate_strategy_capability_decision,
};
pub use entry::analyze_query_capabilities;
pub use join::KeyedHashJoinStrategyOptions;
pub(crate) use join::*;
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
pub(crate) use shape::capability_shape_contract_projection_ids;
