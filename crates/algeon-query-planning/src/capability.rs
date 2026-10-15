//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` traverses a `QueryPlan` and produces a
//! `CapabilityAnalysis` that classifies each node, expression, edge, and
//! subgraph invariant. Rejections are localized and structured, identifying the
//! specific missing operator, expression, Arrow type, edge contract, or invariant.
//! They use canonical capability report keys rather than execution-family labels.
//!
//! Capability analysis verifies structural execution support, confirming that a
//! native executor implements the required SQL semantics. Memory allocation and
//! sizing boundaries are excluded from capability admission; missing size estimates
//! do not trigger candidate rejections, as query limits are enforced by the
//! query domain allocator during execution
//! (`website/docs/design/admission.mdx#allocation-enforcement`).
//! The analysis pass is exhaustive. Every node receives a classification, and
//! rejections preserve diagnostic facts identifying the first blocking contract
//! for inclusion in planning reports.

use crate::{
    error::QueryError,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnQualifier, Expr, GroupingSetsSpec,
        LiteralValue,
    },
    plan::{GraphAlgorithm, JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryNode, QueryPlan, ResultSink},
    source::SourceNode,
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
pub(crate) use edge::input_role;
pub(crate) use expr::{
    require_expression_capability, validate_aggregate_expr_capability, validate_column_expr_capability,
    validate_general_expr_capability,
};
pub use join::KeyedHashJoinStrategyOptions;
#[doc(hidden)]
pub use join::{
    JoinForm, KeyedHashJoinStrategyInputFacts, KeylessConditionalJoinKind, keyed_hash_join_build_is_large,
    keyed_hash_join_strategy_input_facts_for_plan,
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness,
};
pub(crate) use join::{choose_join_strategy, classify_join, join_probe_streams_chunks};
#[doc(hidden)]
pub use join_strategy::{JoinStrategy, JoinStrategyChoice, JoinStrategyEvidence};
pub use model::{
    AggregateGrouping, CapabilityAnalysis, CapabilityDecision, CapabilityRejectionCategory, CapabilityScope,
    EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, InvariantId, NodeCapability, NodeOperator,
    OrderingRequirement, ResidentResource, ResidualRequirement, ResultSinkKind, SourceFormatCapability,
    SubgraphInvariant, SubgraphInvariantKind, VectorInputRole,
};
#[doc(hidden)]
pub use node::is_cudf_median_input_type;
pub use rejection::{Reason, Rejection, RejectionOwner, UnsupportedPart};
