use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::QueryError,
    exec::validate_local_parquet_source,
    expr::{AggregateExpr, AggregateFunction, BinaryOp, ColumnQualifier, Expr, LiteralValue},
    plan::{JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan},
    runtime::ReservationKind,
    source::SourceNode,
};

mod admission;
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

pub use entry::analyze_query_capabilities;
pub(crate) use join::{
    JoinExecutionCapability, KeylessConditionalJoinKind, join_node_execution_capability,
    keyed_filtered_semi_anti_join_capable, keyed_hash_join_capable, multi_join_subtree_join_execution_capability,
};
pub use model::{
    AggregateGrouping, CapabilityAnalysis, CapabilityDecision, CapabilityRejectionOwner, CapabilityRejectionProjection,
    ConstraintRef, ConstraintScope, ContractQuality, ContractStatus, EdgeConsumer, EdgeContract, EdgeContractKind,
    InputRole, InvariantId, NodeCapability, NodeOperator, OrderingRequirement, ProducedProperty, RequiredProperty,
    ReservationRequirement, ReservationRequirementScope, ResidentResource, ResidualRequirement, ResultSinkKind,
    ScheduleConstraint, SharedProducerIdentity, SourceFormatCapability, SubgraphInvariant, SubgraphInvariantKind,
    lower_level_unresolved_preflight_gap, lower_level_unresolved_preflight_gap_for_contract,
};
pub use requirements::reservation_requirement_scope;
