use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use super::{
    aggregate_join_pushdown_policy::AggregateJoinPushdownPolicy, rewrite, simple_projection_column,
    unique_join_output_for_name,
};
use crate::{
    datatypes::arrow_fixed_width_byte_size,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, LiteralValue},
    optimizer::NativeOptimizerChange,
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

pub(in crate::optimizer::aggregate_join_pushdown) use finalize::{
    avg_pushdown_sum_state_data_type, final_aggregate_rewrite_after_preagg,
};
pub(in crate::optimizer::aggregate_join_pushdown) use lineage::{
    AggregateJoinLineage, join_key_for_side, join_side_index, join_side_introduces_nulls, opposite_side,
    rewritten_join_key_after_preagg,
};
#[doc(hidden)]
pub use model::AggregateJoinPushdownResult;
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

#[doc(hidden)]
pub fn optimize_aggregate_join_pushdown(
    plan: QueryPlan,
    options: &AggregateJoinPushdownOptions,
) -> AggregateJoinPushdownResult {
    if !options.enabled {
        return AggregateJoinPushdownResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
        };
    }

    let mut rewriter = AggregateJoinPushdownRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let AggregateJoinPushdownRewriter { diagnostics, .. } = rewriter;
        return AggregateJoinPushdownResult { plan, change: NativeOptimizerChange::Unchanged, diagnostics };
    }

    AggregateJoinPushdownResult {
        plan: rewrite::finish_compacted_rewritten_plan(rewriter.nodes, new_root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan)
            .with_advisory_outputs_from(&plan),
        change: NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}
