//! Capability-first admission for native query plans.
//!
//! `analyze_query_capabilities` walks a `QueryPlan` and returns a
//! `CapabilityAnalysis` that classifies every node, expression, edge, and
//! subgraph invariant. Rejections are local and structured (missing operator,
//! expression, Arrow type, edge contract, or invariant); they use canonical
//! capability report keys rather than
//! execution-family labels.
//!
//! "Capable" is a structural claim: a node, expression, Arrow type, edge
//! contract, or subgraph invariant has a native executor whose semantics match
//! the SQL contract. It says nothing about memory. A missing size estimate
//! cannot reject a candidate; the query domain's allocator enforces limits
//! during execution (`website/docs/design/admission.mdx`). The analysis itself
//! cannot fail: every node receives a classification, and a rejection carries
//! the first unmet contract with the facts that identify the blocking element;
//! planning reports name that element.

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
