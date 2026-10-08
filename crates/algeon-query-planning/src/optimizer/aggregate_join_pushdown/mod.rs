use arrow::datatypes::{Field, Schema, SchemaRef};

use super::{
    aggregate_join_pushdown_policy::AggregateJoinPushdownPolicy, rewrite, simple_projection_column,
    unique_join_output_for_name,
};
use crate::{
    datatypes::arrow_fixed_width_byte_size,
    error::Result,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, CaseWhen, ColumnRef, Expr},
    optimizer::NativeOptimizerChange,
    plan::{
        AggregateNode, JoinKey, JoinNode, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanFacts, PlanNode,
        ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, RewrittenNodes,
    },
    prepared::{
        AVG_COUNT_STATE_TYPE, avg_declared_sum_state_type, avg_final_expr, avg_merge_aggregates,
        avg_partial_aggregates, merged_count_field, zero_filled_count_expr,
    },
};

mod finalize;
mod lineage;
mod model;
mod preaggregate;
mod rewriter;

pub(in crate::optimizer::aggregate_join_pushdown) use finalize::final_aggregate_rewrite_after_preagg;
pub(in crate::optimizer::aggregate_join_pushdown) use lineage::{
    AggregateJoinLineage, join_key_for_side, join_side_index, join_side_introduces_nulls, opposite_side,
    rewritten_join_key_after_preagg,
};
pub(crate) use model::AggregateJoinPushdownResult;
// Preserve public API surface for optimizer.rs re-exports.
pub use model::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
};
pub(in crate::optimizer::aggregate_join_pushdown) use preaggregate::{
    PreAggregateColumn, PreAggregateMeasureState, append_group_projection_outputs, coalesce_preagg_projection_outputs,
    decomposable_union_partial_aggregate_expr, final_aggregate_spec_after_union_preagg, join_with_replaced_input,
    measure_input_projection_outputs, new_join_output_schema, preaggregate_measure_states, preaggregate_schema,
    preaggregate_spec, preaggregate_state_aliases, push_unique_preagg_group_column, unique_measure_input_alias,
};
pub(in crate::optimizer::aggregate_join_pushdown) use rewriter::AggregateJoinPushdownRewriter;

pub(crate) fn optimize_aggregate_join_pushdown(
    plan: QueryPlan,
    options: &AggregateJoinPushdownOptions,
) -> Result<AggregateJoinPushdownResult> {
    if !options.enabled {
        return Ok(AggregateJoinPushdownResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
        });
    }

    let mut rewriter = AggregateJoinPushdownRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let AggregateJoinPushdownRewriter { diagnostics, .. } = rewriter;
        return Ok(AggregateJoinPushdownResult { plan, change: NativeOptimizerChange::Unchanged, diagnostics });
    }

    Ok(AggregateJoinPushdownResult {
        plan: plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?,
        change: NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    })
}
