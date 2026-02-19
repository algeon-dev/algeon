use super::*;

pub(in crate::capability) fn keyless_conditional_join_input_subplan_shape_decision(
    plan: &QueryPlan,
) -> Result<KeylessConditionalJoinInputSubplanShape, CapabilityDecision> {
    let (root_join_id, post_root_stages) = root_join_with_post_root_stages(plan, "keyless_input_subplan")?;
    let root_join = generic_join_node(
        plan,
        root_join_id,
        "keyless_conditional_input_subplan_requires_keyless_conditional_root_join",
    )?;
    require_keyless_conditional_inner_join(
        root_join_id,
        root_join,
        "keyless_conditional_input_subplan_requires_keyless_conditional_inner_join",
    )?;
    validate_general_post_root_stages(plan, &post_root_stages, "keyless_conditional_input_subplan")?;

    let left_contains_subplan = validate_keyless_conditional_input_subplan_for_capability(
        plan,
        root_join.inputs[join_side_index(JoinSide::Left)],
        false,
    )?;
    let right_contains_subplan = validate_keyless_conditional_input_subplan_for_capability(
        plan,
        root_join.inputs[join_side_index(JoinSide::Right)],
        false,
    )?;
    if !left_contains_subplan && !right_contains_subplan {
        return Err(CapabilityDecision::rejected(
            "keyless_conditional_input_subplan_requires_resident_subplan_input",
            [("root_join_id", root_join_id.to_string())],
        ));
    }

    Ok(KeylessConditionalJoinInputSubplanShape {
        root_join_id,
        reachable_join_count: reachable_join_count_from_sink(plan)?,
    })
}

pub(in crate::capability) fn keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(
    plan: &QueryPlan,
) -> Result<KeylessConditionalJoinAggregateOverJoinSubplanInputsShape, CapabilityDecision> {
    let (root_join_id, post_root_stages) = root_join_with_post_root_stages(plan, "keyless_aggregate_subplan_inputs")?;
    let root_join = generic_join_node(
        plan,
        root_join_id,
        "keyless_aggregate_subplan_inputs_requires_keyless_conditional_root_join",
    )?;
    require_keyless_conditional_inner_join(
        root_join_id,
        root_join,
        "keyless_aggregate_subplan_inputs_requires_keyless_conditional_inner_join",
    )?;
    if root_join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
        return Err(CapabilityDecision::rejected(
            "keyless_aggregate_subplan_inputs_requires_left_only_root_projection",
            [("root_join_id", root_join_id.to_string())],
        ));
    }

    let (post_root_projection_id, final_sort_id) =
        validate_optional_projection_then_sort_post_root_stages(plan, &post_root_stages)?;
    let (grouped_aggregate_id, grouped_aggregate_projection_id) = projection_wrapped_aggregate_over_join_subplan_root(
        plan,
        root_join.inputs[join_side_index(JoinSide::Left)],
        AggregateGrouping::Grouped,
        "keyless_grouped_aggregate_subplan_input",
    )?;
    let (scalar_aggregate_id, scalar_aggregate_projection_id) = projection_wrapped_aggregate_over_join_subplan_root(
        plan,
        root_join.inputs[join_side_index(JoinSide::Right)],
        AggregateGrouping::Scalar,
        "keyless_scalar_aggregate_subplan_input",
    )?;

    Ok(KeylessConditionalJoinAggregateOverJoinSubplanInputsShape {
        root_join_id,
        grouped_aggregate_id,
        grouped_aggregate_projection_id,
        scalar_aggregate_id,
        scalar_aggregate_projection_id,
        post_root_projection_id,
        final_sort_id,
    })
}

pub(in crate::capability) fn require_keyless_conditional_inner_join(
    join_id: NodeId,
    join: &JoinNode,
    reason: &'static str,
) -> Result<(), CapabilityDecision> {
    if join.spec.join_type == JoinType::Inner
        && join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
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
            ("output_projection_count", join.spec.output_projection.len().to_string()),
        ],
    ))
}

pub(in crate::capability) fn validate_keyless_conditional_input_subplan_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
    join_reached_through_projection: bool,
) -> Result<bool, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "keyless_conditional_input_subplan_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Projection(projection) => {
            validate_keyless_conditional_input_subplan_for_capability(plan, projection.input, true)
        },
        PlanNode::Aggregate(aggregate) => {
            if !aggregate.spec.group_exprs.is_empty() {
                return Err(CapabilityDecision::rejected(
                    "keyless_conditional_input_subplan_requires_scalar_aggregate_or_leaf",
                    [
                        ("node_id", node_id.to_string()),
                        ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                    ],
                ));
            }
            validate_source_filter_projection_leaf_pipeline_for_capability(plan, aggregate.input)?;
            Ok(true)
        },
        PlanNode::Join(join) => {
            if !join_reached_through_projection {
                return Err(CapabilityDecision::rejected(
                    "keyless_conditional_input_subplan_requires_projection_wrapped_join",
                    [("node_id", node_id.to_string())],
                ));
            }
            require_left_anti_filtered_source_leaf_join(node_id, join)?;
            validate_source_filter_projection_leaf_pipeline_for_capability(plan, join.inputs[0])?;
            validate_source_filter_projection_leaf_pipeline_for_capability(plan, join.inputs[1])?;
            Ok(true)
        },
        PlanNode::Source(_) | PlanNode::Filter(_) => {
            validate_source_filter_projection_leaf_pipeline_for_capability(plan, node_id)?;
            Ok(false)
        },
        PlanNode::Window(_) | PlanNode::Sort(_) | PlanNode::Limit(_) | PlanNode::Union(_) => {
            Err(CapabilityDecision::rejected(
                "keyless_conditional_input_subplan_unsupported_input_stage",
                [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            ))
        },
    }
}

pub(in crate::capability) fn require_left_anti_filtered_source_leaf_join(
    join_id: NodeId,
    join: &JoinNode,
) -> Result<(), CapabilityDecision> {
    if join.spec.join_type == JoinType::LeftAnti
        && join.spec.build_side == JoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
    {
        return Ok(());
    }
    Err(CapabilityDecision::rejected(
        "keyless_conditional_input_subplan_requires_projection_wrapped_left_anti_join",
        [
            ("node_id", join_id.to_string()),
            ("join_type", join.spec.join_type.to_string()),
            ("key_count", join.spec.keys.len().to_string()),
            ("has_residual", join.spec.residual_predicate.is_some().to_string()),
            ("build_side", join.spec.build_side.to_string()),
        ],
    ))
}

pub(in crate::capability) fn projection_wrapped_aggregate_over_join_subplan_root(
    plan: &QueryPlan,
    node_id: NodeId,
    grouping: AggregateGrouping,
    reason_prefix: &'static str,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_subplan_input_missing_projection",
            [("reason_prefix", reason_prefix.to_string()), ("node_id", node_id.to_string())],
        ));
    };
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_subplan_input_requires_projection_wrapped_aggregate",
            [
                ("reason_prefix", reason_prefix.to_string()),
                ("node_id", node_id.to_string()),
                ("operator", node_operator_fact(&node.kind).to_string()),
            ],
        ));
    };
    let aggregate_id = projection.input;
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_subplan_input_missing_aggregate",
            [("reason_prefix", reason_prefix.to_string()), ("node_id", aggregate_id.to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_subplan_input_requires_aggregate",
            [
                ("reason_prefix", reason_prefix.to_string()),
                ("node_id", aggregate_id.to_string()),
                ("operator", node_operator_fact(&aggregate_node.kind).to_string()),
            ],
        ));
    };
    match grouping {
        AggregateGrouping::Grouped if aggregate.spec.group_exprs.is_empty() => {
            return Err(CapabilityDecision::rejected(
                "aggregate_over_join_subplan_input_requires_grouped_aggregate",
                [("reason_prefix", reason_prefix.to_string()), ("node_id", aggregate_id.to_string())],
            ));
        },
        AggregateGrouping::Scalar if !aggregate.spec.group_exprs.is_empty() => {
            return Err(CapabilityDecision::rejected(
                "aggregate_over_join_subplan_input_requires_scalar_aggregate",
                [
                    ("reason_prefix", reason_prefix.to_string()),
                    ("node_id", aggregate_id.to_string()),
                    ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                ],
            ));
        },
        AggregateGrouping::Grouped | AggregateGrouping::Scalar | AggregateGrouping::GroupingSets => {},
    }
    validate_bounded_join_subtree_rooted_at_join_for_capability(plan, aggregate.input)?;
    Ok((aggregate_id, node_id))
}

pub(in crate::capability) fn validate_source_filter_projection_leaf_pipeline_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "source_leaf_pipeline_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Ok(()),
        PlanNode::Source(_) => Err(CapabilityDecision::rejected(
            "source_leaf_pipeline_requires_local_parquet_source",
            [("node_id", node_id.to_string()), ("source_format", "gpu_resident".to_string())],
        )),
        PlanNode::Filter(filter) => validate_source_filter_projection_leaf_pipeline_for_capability(plan, filter.input),
        PlanNode::Projection(projection) => {
            validate_source_filter_projection_leaf_pipeline_for_capability(plan, projection.input)
        },
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_) => Err(CapabilityDecision::rejected(
            "source_leaf_pipeline_requires_source_filter_projection",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}
