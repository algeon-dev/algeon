use super::{
    CapabilityDecision, HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape, JoinSide,
    MultiJoinAggregateOverJoinSubplanTopKShape, NodeId, PlanNode, QueryPlan, generic_join_node,
    grouped_aggregate_input, join_node_with_missing_reason, join_side_index, limit_input,
    limit_input_with_missing_reason, node_kind_for_fact, node_operator_fact, opposite_join_side,
    output_projection_single_side, peel_projection_chain_ids_for_shape, projection_input_with_missing_reason,
    projection_or_filter_wrapped_grouped_local_parquet_aggregate_root,
    projection_wrapped_grouped_aggregate_over_join_subplan_root,
    require_keyed_inner_hash_join_without_residual_with_nonempty_keys,
    require_left_semi_keyed_hash_join_with_right_build, sort_input, sort_input_with_missing_reason,
    validate_bounded_join_subtree_for_capability,
};

pub(in crate::capability) fn hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(
    plan: &QueryPlan,
) -> Result<HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape, CapabilityDecision> {
    let final_limit_id = plan.sink().input();
    let limit_child_id =
        limit_input(plan, final_limit_id, "hash_resident_grouped_aggregate_topk_requires_final_limit", "final_limit")?;
    let (final_sort_id, final_projection_id, post_root_aggregate_id) = match plan
        .node(limit_child_id)
        .map(|node| &node.kind)
    {
        Some(PlanNode::Projection(proj)) => {
            let sort_id = proj.input;
            let sort_input_id =
                sort_input(plan, sort_id, "hash_resident_grouped_aggregate_topk_requires_final_sort", "final_sort")?;
            (sort_id, limit_child_id, sort_input_id)
        },
        Some(PlanNode::Sort(sort)) => {
            let proj_id = sort.input;
            let proj_input_id = projection_input_with_missing_reason(
                plan,
                proj_id,
                "hash_resident_grouped_aggregate_topk_missing_node",
                "hash_resident_grouped_aggregate_topk_requires_projection_before_sort",
                "final_projection",
            )?;
            (limit_child_id, proj_id, proj_input_id)
        },
        Some(node) => {
            return Err(CapabilityDecision::rejected(
                "hash_resident_grouped_aggregate_topk_requires_final_sort",
                [("node_id", limit_child_id.to_string()), ("operator", node_operator_fact(node).to_string())],
            ));
        },
        None => {
            return Err(CapabilityDecision::rejected(
                "hash_resident_grouped_aggregate_topk_missing_node",
                [("node_id", limit_child_id.to_string()), ("role", "final_sort".to_string())],
            ));
        },
    };
    let post_root_aggregate_input =
        grouped_aggregate_input(plan, post_root_aggregate_id, "post_join_topk_requires_grouped_aggregate")?;
    let (root_join_id, post_root_projection_id) = match plan.node(post_root_aggregate_input).map(|node| &node.kind) {
        Some(PlanNode::Projection(projection)) => (projection.input, post_root_aggregate_input),
        Some(PlanNode::Join(_)) => (post_root_aggregate_input, post_root_aggregate_input),
        Some(node) => {
            return Err(CapabilityDecision::rejected(
                "post_join_topk_requires_projection_or_join_before_grouped_aggregate",
                [
                    ("node_id", post_root_aggregate_input.to_string()),
                    ("operator", node_operator_fact(node).to_string()),
                ],
            ));
        },
        None => {
            return Err(CapabilityDecision::rejected(
                "hash_resident_grouped_aggregate_topk_missing_node",
                [("node_id", post_root_aggregate_input.to_string()), ("role", "pre_aggregate_input".to_string())],
            ));
        },
    };

    let root_join = generic_join_node(plan, root_join_id, "post_join_topk_requires_root_join_below_projection")?;
    require_left_semi_keyed_hash_join_with_right_build(root_join_id, root_join)?;

    let join_subtree_root_id = root_join.inputs[join_side_index(JoinSide::Left)];
    let join_count = validate_hash_resident_grouped_aggregate_topk_preserved_side(plan, join_subtree_root_id)?;
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

fn validate_hash_resident_grouped_aggregate_topk_preserved_side(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<usize, CapabilityDecision> {
    let mut join_count = 0;
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Aggregate(aggregate)) => {
            if aggregate.spec.group_exprs.is_empty() || aggregate.spec.grouping_sets.is_some() {
                return Err(CapabilityDecision::rejected(
                    "hash_resident_grouped_aggregate_topk_requires_grouped_preserved_aggregate",
                    [
                        ("node_id", node_id.to_string()),
                        ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                        ("has_grouping_sets", aggregate.spec.grouping_sets.is_some().to_string()),
                    ],
                ));
            }
            validate_bounded_join_subtree_for_capability(plan, aggregate.input, &mut join_count)?;
        },
        Some(_) => {
            validate_bounded_join_subtree_for_capability(plan, node_id, &mut join_count)?;
        },
        None => {
            return Err(CapabilityDecision::rejected(
                "hash_resident_grouped_aggregate_topk_missing_preserved_side",
                [("node_id", node_id.to_string())],
            ));
        },
    }
    Ok(join_count)
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
    let (post_projection_input_id, post_root_projections) =
        peel_projection_chain_ids_for_shape(plan, post_root_projection_id)?;
    let direct_sort_fetch_join_input =
        matches!(plan.node(post_projection_input_id).map(|node| &node.kind), Some(PlanNode::Join(_)));
    if post_root_projections.is_empty() && !direct_sort_fetch_join_input {
        return Err(CapabilityDecision::rejected(
            "multi_join_aggregate_over_join_topk_requires_projection_before_sort",
            [
                ("node_id", post_root_projection_id.to_string()),
                ("operator", node_kind_for_fact(plan, post_root_projection_id)),
            ],
        ));
    }
    let (root_join_id, post_root_join_ids) =
        peel_single_output_post_root_joins_for_multi_join_aggregate_topk(plan, post_projection_input_id)?;

    let (aggregate_id, aggregate_projection_id) =
        multi_join_aggregate_over_join_subplan_topk_root_shape(plan, root_join_id)?;

    Ok(MultiJoinAggregateOverJoinSubplanTopKShape {
        root_join_id,
        aggregate_id,
        aggregate_projection_id,
        post_root_join_ids,
        post_root_projection_id,
        final_sort_id,
        final_limit_id,
    })
}

fn multi_join_aggregate_over_join_subplan_topk_root_shape(
    plan: &QueryPlan,
    root_join_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
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
    projection_wrapped_grouped_aggregate_over_join_subplan_root(plan, aggregate_input)
}

fn peel_single_output_post_root_joins_for_multi_join_aggregate_topk(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, Vec<NodeId>), CapabilityDecision> {
    let mut current = node_id;
    let mut post_root_join_ids = Vec::new();
    let mut visited = std::collections::BTreeSet::new();

    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "multi_join_aggregate_over_join_topk_post_root_join_cycle",
                [("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "multi_join_aggregate_over_join_topk_missing_node",
                [("node_id", current.to_string())],
            ));
        };
        let PlanNode::Join(join) = &node.kind else {
            return Ok((current, post_root_join_ids));
        };
        require_keyed_inner_hash_join_without_residual_with_nonempty_keys(
            current,
            join,
            "multi_join_aggregate_over_join_topk_requires_keyed_inner_hash_join",
        )?;
        let output_side = output_projection_single_side(
            join,
            current,
            "multi_join_aggregate_over_join_topk_post_root_join_output_side",
        )?;
        let output_input = join.inputs[join_side_index(output_side)];
        if multi_join_aggregate_over_join_subplan_topk_root_shape(plan, output_input).is_err() {
            return Ok((current, post_root_join_ids));
        }
        post_root_join_ids.push(current);
        current = output_input;
    }
}
