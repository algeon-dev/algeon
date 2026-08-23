//! Plan-shape catalogues used by capability-envelope admission.
//!
//! Each submodule recognises one named shape family (root join, mixed
//! residual, keyless conditional, etc.) and exposes its recognisers to
//! [`super::admission`].

use std::collections::BTreeSet;

use super::{
    AggregateFunction, AggregateNode, AggregateSpec, Expr, JoinExecutionCapability, JoinNode, JoinSide, JoinType,
    KeylessConditionalJoinKind, NodeId, PlanNode, QueryPlan, SourceNode, WindowFunction,
    edge::node_operator_fact,
    expr::expr_family,
    graph::{
        collect_reachable_node_ids, collect_reachable_node_ids_from, consumer_nodes_for_reachable,
        reachable_join_count_from_sink,
    },
    invariant::expr_references_input,
    join_node_execution_capability,
    model::{AggregateGrouping, CapabilityDecision, ConstraintRef},
    multi_join_subtree_join_execution_capability,
};

mod access;
mod branch;
mod keyed_hash_resident;
mod keyless_conditional;
mod mixed_residual;
mod post_join_topk;
mod projection_global_limit;
mod projection_scalar_setop;
mod rollup_rank_topk;
mod root_full_sort_join_aggregate;
mod root_join;
mod root_topk_join_aggregate;
mod single_join_post_root;
mod subquery_filter;
mod types;

pub(in crate::capability) use access::{
    closed_multi_join_subgraph_structure_decision, column_expr_lineage, expression_lineage_for_input,
    generic_join_node, grouped_aggregate_input, join_key_lineage_for_side, join_node_with_missing_reason,
    join_side_index, limit_input, limit_input_with_missing_reason, node_kind_for_fact, opposite_join_side,
    output_lineage, output_projection_single_side, peel_projection_chain_ids_for_shape,
    peel_resident_unary_chain_for_capability, post_root_join_id, projected_output_side, projection_input,
    projection_input_with_missing_reason, projection_or_filter_wrapped_grouped_local_parquet_aggregate_root,
    projection_wrapped_grouped_aggregate_over_join_subplan_root,
    projection_wrapped_grouped_local_parquet_aggregate_root,
    projection_wrapped_grouped_sum_local_parquet_aggregate_root, projection_wrapped_join_subtree_root,
    require_join_node, require_keyed_inner_hash_join_without_residual,
    require_keyed_inner_hash_join_without_residual_with_nonempty_keys,
    require_left_semi_keyed_hash_join_with_right_build, scalar_aggregate_input, sort_input,
    sort_input_with_missing_reason, validate_bounded_join_subtree_for_capability,
    validate_bounded_join_subtree_with_preaggregate_leaves_rooted_at_join_for_capability,
    validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan,
};
pub(in crate::capability) use branch::{
    keyed_hash_joined_grouped_branch_shape_decision, keyed_hash_scalar_max_branch_shape_decision,
};
pub(in crate::capability) use keyed_hash_resident::keyed_hash_join_resident_subplan_shape_decision;
pub(in crate::capability) use keyless_conditional::{
    keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision,
    keyless_conditional_join_input_subplan_shape_decision,
};
pub(in crate::capability) use mixed_residual::mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision;
pub(in crate::capability) use post_join_topk::{
    hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision,
    multi_join_aggregate_over_join_subplan_topk_shape_decision,
};
pub(in crate::capability) use projection_global_limit::projection_global_limit_shape_decision;
pub(in crate::capability) use projection_scalar_setop::projection_scalar_setop_shape_decision;
pub(in crate::capability) use rollup_rank_topk::rollup_rank_topk_after_aggregate_join_pushdown_shape_decision;
pub(in crate::capability) use root_full_sort_join_aggregate::{
    root_full_sort_join_aggregate_shape_decision, root_full_sort_window_allows_sink_path_grouped_aggregate_search,
};
pub(in crate::capability) use root_join::{
    root_join_shape_decision, root_join_with_post_root_stages, validate_general_post_root_stages,
    validate_optional_projection_then_sort_post_root_stages, validate_resident_leaf_pipeline_for_capability,
};
pub(in crate::capability) use root_topk_join_aggregate::root_topk_join_aggregate_shape_decision;
pub(in crate::capability) use single_join_post_root::single_join_post_root_pipeline_shape_decision;
pub(in crate::capability) use subquery_filter::multi_join_resident_grouped_aggregate_subquery_filter_shape_decision;
pub(in crate::capability) use types::{
    FinalRollupWidthClass, HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape,
    KeyedHashJoinResidentSubplanShape, KeylessConditionalJoinAggregateOverJoinSubplanInputsShape,
    KeylessConditionalJoinInputSubplanShape, MixedResidualResidentGroupedAggregateInputShape,
    MultiJoinAggregateOverJoinSubplanTopKShape, MultiJoinResidentGroupedAggregateSubqueryFilterShape, OutputLineage,
    PostRootStage, ProjectionGlobalLimitShape, ProjectionScalarSetopKind, ProjectionScalarSetopShape,
    ResidentGroupedAggregateSubqueryFilterStats, RollupRankTopKAfterAggregateJoinPushdownShape,
    RootFullSortJoinAggregateShape, RootFullSortJoinAggregateSubcontract, RootTopKJoinAggregateShape,
};

#[doc(hidden)]
pub fn capability_shape_contract_projection_ids(plan: &QueryPlan) -> BTreeSet<NodeId> {
    let mut ids = BTreeSet::new();

    if let Ok(shape) = mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) {
        ids.insert(shape.join_subtree_projection_id);
        ids.insert(shape.grouped_aggregate_projection_id);
    }
    if let Ok(shape) = hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
    }
    if let Ok(shape) = multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) {
        ids.insert(shape.aggregate_projection_id);
    }
    if let Ok(shape) = root_topk_join_aggregate_shape_decision(plan) {
        ids.insert(shape.aggregate_projection_id);
        ids.extend(shape.post_root_projection_ids);
    }
    if let Ok(shape) = root_full_sort_join_aggregate_shape_decision(plan) {
        ids.insert(shape.aggregate_projection_id);
        ids.extend(shape.post_root_projection_ids);
    }
    if let Ok(shape) = projection_global_limit_shape_decision(plan)
        && let Some(post_limit_projection_id) = shape.post_limit_projection_id
    {
        ids.insert(post_limit_projection_id);
    }
    if let Ok(shape) = keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
        ids.insert(shape.scalar_aggregate_projection_id);
    }
    if let Ok(shape) = multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
    }

    ids
}
