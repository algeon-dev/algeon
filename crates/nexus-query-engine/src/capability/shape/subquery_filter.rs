use super::*;

pub(in crate::capability) fn multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(
    plan: &QueryPlan,
) -> Result<MultiJoinResidentGroupedAggregateSubqueryFilterShape, CapabilityDecision> {
    let final_sort_id = plan.sink().input();
    let root_join_id = sort_input_with_missing_reason(
        plan,
        final_sort_id,
        "resident_grouped_subquery_filter_missing_node",
        "resident_grouped_subquery_filter_requires_final_sort",
        "final_sort",
    )?;
    let root_join = join_node_with_missing_reason(
        plan,
        root_join_id,
        "resident_grouped_subquery_filter_missing_node",
        "resident_grouped_subquery_filter_requires_root_join_below_final_sort",
    )?;
    require_left_semi_keyed_hash_join_with_right_build_for_reason(
        root_join_id,
        root_join,
        "resident_grouped_subquery_filter_requires_left_semi_keyed_hash_join",
    )?;

    let output_side = output_projection_single_side(
        root_join,
        root_join_id,
        "resident_grouped_subquery_filter_requires_single_output_side",
    )?;
    let subquery_filter_side = opposite_join_side(output_side);
    let output_input = root_join.inputs[join_side_index(output_side)];
    let mut output_join_count = 0;
    validate_bounded_join_subtree_for_capability(plan, output_input, &mut output_join_count)?;
    if output_join_count < 1 {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_requires_join_output_side",
            [("node_id", output_input.to_string()), ("join_count", output_join_count.to_string())],
        ));
    }

    let subquery_input = root_join.inputs[join_side_index(subquery_filter_side)];
    let stats = resident_grouped_aggregate_subquery_filter_stats(plan, subquery_input)?;
    if stats.join_count < 2 {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_requires_deep_filter_side",
            [("node_id", subquery_input.to_string()), ("join_count", stats.join_count.to_string())],
        ));
    }
    if stats.grouped_aggregate_count != 1 {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_requires_one_grouped_aggregate",
            [
                ("node_id", subquery_input.to_string()),
                ("grouped_aggregate_count", stats.grouped_aggregate_count.to_string()),
            ],
        ));
    }
    let Some((residual_join_id, grouped_aggregate_id, grouped_aggregate_projection_id)) = stats.residual_comparison
    else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_requires_residual_grouped_aggregate_comparison",
            [("node_id", subquery_input.to_string())],
        ));
    };
    if stats.residual_comparison_count != 1 {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_requires_single_residual_grouped_aggregate_comparison",
            [
                ("node_id", subquery_input.to_string()),
                ("residual_comparison_count", stats.residual_comparison_count.to_string()),
            ],
        ));
    }

    Ok(MultiJoinResidentGroupedAggregateSubqueryFilterShape {
        final_sort_id,
        root_join_id,
        grouped_aggregate_id,
        grouped_aggregate_projection_id,
        residual_join_id,
    })
}

pub(in crate::capability) fn require_left_semi_keyed_hash_join_with_right_build_for_reason(
    join_id: NodeId,
    join: &JoinNode,
    reason: &'static str,
) -> Result<(), CapabilityDecision> {
    if join.spec.join_type == JoinType::LeftSemi
        && join.spec.build_side == JoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
    {
        return Ok(());
    }
    Err(CapabilityDecision::rejected(
        reason,
        [
            ("node_id", join_id.to_string()),
            ("join_type", join.spec.join_type.to_string()),
            ("key_count", join.spec.keys.len().to_string()),
            ("has_residual", join.spec.residual_predicate.is_some().to_string()),
            ("build_side", join.spec.build_side.to_string()),
            (
                "projects_only_left",
                join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left).to_string(),
            ),
        ],
    ))
}

pub(in crate::capability) fn resident_grouped_aggregate_subquery_filter_stats(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<ResidentGroupedAggregateSubqueryFilterStats, CapabilityDecision> {
    if let Ok((aggregate_id, projection_id)) =
        projection_or_filter_wrapped_grouped_local_parquet_aggregate_root(plan, node_id)
    {
        return Ok(ResidentGroupedAggregateSubqueryFilterStats {
            grouped_aggregate_count: 1,
            aggregate_leaf: Some((aggregate_id, projection_id)),
            ..ResidentGroupedAggregateSubqueryFilterStats::default()
        });
    }

    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(join) => {
            if multi_join_subtree_join_execution_capability(join).is_none() {
                return Err(CapabilityDecision::rejected(
                    "resident_grouped_subquery_filter_unsupported_join",
                    [
                        ("node_id", node_id.to_string()),
                        ("join_type", join.spec.join_type.to_string()),
                        ("key_count", join.spec.keys.len().to_string()),
                        ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                        ("build_side", join.spec.build_side.to_string()),
                    ],
                ));
            }
            let left = resident_grouped_aggregate_subquery_filter_stats(plan, join.inputs[0])?;
            let right = resident_grouped_aggregate_subquery_filter_stats(plan, join.inputs[1])?;
            ResidentGroupedAggregateSubqueryFilterStats::combine_with_join(node_id, join, left, right)
        },
        PlanNode::Source(_) => Ok(ResidentGroupedAggregateSubqueryFilterStats::default()),
        PlanNode::Filter(filter) => resident_grouped_aggregate_subquery_filter_stats(plan, filter.input),
        PlanNode::Projection(projection) => resident_grouped_aggregate_subquery_filter_stats(plan, projection.input),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => Err(CapabilityDecision::rejected(
            "resident_grouped_subquery_filter_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}
