use std::collections::{BTreeMap, BTreeSet};

use crate::{
    bounded::{
        BoundedJoinNodeCapability, BoundedJoinPlanSupport, BoundedKeylessConditionalJoinType,
        analyze_bounded_join_plan_support, bounded_join_node_capability, bounded_multi_join_node_capability,
        bounded_root_filtered_join_compatible, bounded_root_hash_join_compatible,
    },
    error::QueryError,
    exec::validate_local_parquet_source,
    expr::{AggregateExpr, AggregateFunction, BinaryOp, ColumnQualifier, Expr, LiteralValue},
    plan::{JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan},
    runtime::ReservationKind,
    source::SourceNode,
};

mod admission;
mod compatibility;
mod edge;
mod entry;
mod expr;
mod graph;
mod invariant;
mod model;
mod node;
mod requirements;
mod shape;

pub(crate) use compatibility::CompatibilityReportProjection;
pub use entry::analyze_query_capabilities;
pub(crate) use entry::{
    admits_closed_multi_join_subgraph,
    admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
    admits_keyed_hash_join_resident_subplan_pipeline,
    admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
    admits_keyless_conditional_join_input_subplan_pipeline,
    admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
    admits_multi_join_aggregate_over_join_subplan_topk_pipeline,
    admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
    closed_multi_join_subgraph_admission_decision,
    hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission_decision,
    keyed_hash_join_resident_subplan_admission_decision,
    keyless_conditional_join_aggregate_over_join_subplan_inputs_admission_decision,
    keyless_conditional_join_input_subplan_admission_decision,
    mixed_keyed_residual_join_resident_grouped_aggregate_input_admission_decision,
    multi_join_aggregate_over_join_subplan_topk_admission_decision,
    multi_join_resident_grouped_aggregate_subquery_filter_admission_decision,
};
pub use model::{
    AggregateGrouping, CapabilityAnalysis, CapabilityDecision, CompatibilityFamilyProjection,
    CompatibilityRejectionProjection, ConstraintRef, ConstraintScope, ContractQuality, ContractStatus, EdgeConsumer,
    EdgeContract, EdgeContractKind, InputRole, InvariantId, NodeCapability, NodeOperator, OrderingRequirement,
    ProducedProperty, RequiredProperty, ReservationRequirement, ReservationRequirementScope, ResidentResource,
    ResidualRequirement, ResultSinkKind, ScheduleConstraint, SharedProducerIdentity, SourceFormatCapability,
    SubgraphInvariant, SubgraphInvariantKind,
};
pub use requirements::reservation_requirement_scope;
