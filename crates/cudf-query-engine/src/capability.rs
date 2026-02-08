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
pub(crate) use entry::{
    admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
    admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
    admits_multi_join_aggregate_over_join_subplan_topk_pipeline,
    admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
};
pub(crate) use join::{
    JoinExecutionCapability, KeylessConditionalJoinKind, join_node_execution_capability,
    keyed_filtered_semi_anti_join_capable, keyed_hash_join_capable, keyed_residual_mixed_join_capable,
    multi_join_keyed_residual_mixed_join_capable, multi_join_subtree_join_execution_capability,
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
