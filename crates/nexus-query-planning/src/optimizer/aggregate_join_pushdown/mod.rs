use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use super::{
    aggregate_join_pushdown_policy::AggregateJoinPushdownPolicy, rewrite, simple_projection_column,
    unique_join_output_for_name,
};
use crate::{
    datatypes::arrow_fixed_width_byte_size,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, LiteralValue},
    plan::{
        AggregateNode, JoinKey, JoinNode, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanFacts, PlanNode,
        ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
    },
};

mod finalize;
mod lineage;
mod model;
mod preaggregate;
mod rewriter;

#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::optimizer::aggregate_join_pushdown) use finalize::*;
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::optimizer::aggregate_join_pushdown) use lineage::*;
#[doc(hidden)]
pub use model::AggregateJoinPushdownResult;
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::optimizer::aggregate_join_pushdown) use model::*;
// Preserve public API surface for optimizer.rs re-exports.
pub use model::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::optimizer::aggregate_join_pushdown) use preaggregate::*;
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::optimizer::aggregate_join_pushdown) use rewriter::*;

#[doc(hidden)]
pub fn optimize_aggregate_join_pushdown(
    plan: QueryPlan,
    options: &AggregateJoinPushdownOptions,
) -> AggregateJoinPushdownResult {
    if !options.enabled {
        return AggregateJoinPushdownResult {
            plan,
            change: crate::optimizer::NativeOptimizerChange::Unchanged,
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
        };
    }

    let mut rewriter = AggregateJoinPushdownRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let AggregateJoinPushdownRewriter { diagnostics, .. } = rewriter;
        return AggregateJoinPushdownResult {
            plan,
            change: crate::optimizer::NativeOptimizerChange::Unchanged,
            diagnostics,
        };
    }

    AggregateJoinPushdownResult {
        plan: rewrite::finish_compacted_rewritten_plan(rewriter.nodes, new_root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan),
        change: crate::optimizer::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}
