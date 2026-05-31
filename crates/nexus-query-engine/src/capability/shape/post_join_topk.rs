use super::*;

pub(in crate::capability) fn hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(
    plan: &QueryPlan,
) -> Result<HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape, CapabilityDecision> {
    let final_limit_id = plan.sink().input();
    let final_sort_id =
        limit_input(plan, final_limit_id, "hash_resident_grouped_aggregate_topk_requires_final_limit", "final_limit")?;
    let final_projection_id =
        sort_input(plan, final_sort_id, "hash_resident_grouped_aggregate_topk_requires_final_sort", "final_sort")?;
    let post_root_aggregate_id = projection_input_with_missing_reason(
        plan,
        final_projection_id,
        "hash_resident_grouped_aggregate_topk_missing_node",
        "hash_resident_grouped_aggregate_topk_requires_projection_before_sort",
        "final_projection",
    )?;
    let post_root_aggregate_input =
        grouped_aggregate_input(plan, post_root_aggregate_id, "post_join_topk_requires_grouped_aggregate")?;
    let post_root_projection_id = post_root_aggregate_input;
    let root_join_id = projection_input_with_missing_reason(
        plan,
        post_root_projection_id,
        "hash_resident_grouped_aggregate_topk_missing_node",
        "post_join_topk_requires_projection_before_grouped_aggregate",
        "pre_aggregate_projection",
    )?;

    let root_join = generic_join_node(plan, root_join_id, "post_join_topk_requires_root_join_below_projection")?;
    require_left_semi_keyed_hash_join_with_right_build(root_join_id, root_join)?;

    let join_subtree_root_id = root_join.inputs[join_side_index(JoinSide::Left)];
    let mut join_count = 0;
    validate_bounded_join_subtree_for_capability(plan, join_subtree_root_id, &mut join_count)?;
    if join_count < 2 {
        return Err(CapabilityDecision::rejected(
            "hash_resident_grouped_aggregate_topk_requires_multi_join_subtree",
            [("node_id", join_subtree_root_id.to_string()), ("join_count", join_count.to_string())],
        ));
    }

    let grouped_aggregate_input = root_join.inputs[join_side_index(JoinSide::Right)];
    let (grouped_aggregate_id, grouped_aggregate_projection_id) =
        projection_or_filter_wrapped_grouped_local_parquet_aggregate_root(plan, grouped_aggregate_input)?;

    Ok(HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape {
        root_join_id,
        grouped_aggregate_id,
        grouped_aggregate_projection_id,
        post_root_projection_id,
        post_root_aggregate_id,
        final_projection_id,
        final_sort_id,
        final_limit_id,
    })
}

pub(in crate::capability) fn multi_join_aggregate_over_join_subplan_topk_shape_decision(
    plan: &QueryPlan,
) -> Result<MultiJoinAggregateOverJoinSubplanTopKShape, CapabilityDecision> {
    let final_limit_id = plan.sink().input();
    let final_sort_id = limit_input_with_missing_reason(
        plan,
        final_limit_id,
        "multi_join_aggregate_over_join_topk_missing_node",
        "multi_join_aggregate_over_join_topk_requires_final_limit",
        "final_limit",
    )?;
    let post_root_projection_id = sort_input_with_missing_reason(
        plan,
        final_sort_id,
        "multi_join_aggregate_over_join_topk_missing_node",
        "multi_join_aggregate_over_join_topk_requires_final_sort",
        "final_sort",
    )?;
    let root_join_id = projection_input_with_missing_reason(
        plan,
        post_root_projection_id,
        "multi_join_aggregate_over_join_topk_missing_node",
        "multi_join_aggregate_over_join_topk_requires_projection_before_sort",
        "post_root_projection",
    )?;

    let root_join = join_node_with_missing_reason(
        plan,
        root_join_id,
        "multi_join_aggregate_over_join_topk_missing_node",
        "multi_join_aggregate_over_join_topk_requires_root_join_below_projection",
    )?;
    require_keyed_inner_hash_join_without_residual_with_nonempty_keys(
        root_join_id,
        root_join,
        "multi_join_aggregate_over_join_topk_requires_keyed_inner_hash_join",
    )?;
    let multi_join_side =
        output_projection_single_side(root_join, root_join_id, "multi_join_aggregate_over_join_topk_root_output_side")?;
    let aggregate_over_join_side = opposite_join_side(multi_join_side);

    let multi_join_input = root_join.inputs[join_side_index(multi_join_side)];
    let mut multi_join_count = 0;
    validate_bounded_join_subtree_for_capability(plan, multi_join_input, &mut multi_join_count)?;
    if multi_join_count < 3 {
        return Err(CapabilityDecision::rejected(
            "multi_join_aggregate_over_join_topk_requires_deep_multi_join_side",
            [("node_id", multi_join_input.to_string()), ("join_count", multi_join_count.to_string())],
        ));
    }

    let aggregate_input = root_join.inputs[join_side_index(aggregate_over_join_side)];
    let (aggregate_id, aggregate_projection_id) =
        projection_wrapped_grouped_aggregate_over_join_subplan_root(plan, aggregate_input)?;

    Ok(MultiJoinAggregateOverJoinSubplanTopKShape {
        root_join_id,
        aggregate_id,
        aggregate_projection_id,
        post_root_projection_id,
        final_sort_id,
        final_limit_id,
    })
}
