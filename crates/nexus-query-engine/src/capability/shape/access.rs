use super::*;

pub(in crate::capability) fn projection_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "mixed_residual_resident_aggregate_missing_node",
            [("node_id", node_id.to_string()), ("role", role.to_string())],
        ));
    };
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("role", role.to_string()),
                ("operator", node_operator_fact(&node.kind).to_string()),
            ],
        ));
    };
    Ok(projection.input)
}

pub(in crate::capability) fn projection_input_with_missing_reason(
    plan: &QueryPlan,
    node_id: NodeId,
    missing_reason: &'static str,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            missing_reason,
            [("node_id", node_id.to_string()), ("role", role.to_string())],
        ));
    };
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("role", role.to_string()),
                ("operator", node_operator_fact(&node.kind).to_string()),
            ],
        ));
    };
    Ok(projection.input)
}

pub(in crate::capability) fn limit_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    limit_input_with_missing_reason(plan, node_id, "hash_resident_grouped_aggregate_topk_missing_node", reason, role)
}

pub(in crate::capability) fn limit_input_with_missing_reason(
    plan: &QueryPlan,
    node_id: NodeId,
    missing_reason: &'static str,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            missing_reason,
            [("node_id", node_id.to_string()), ("role", role.to_string())],
        ));
    };
    let PlanNode::Limit(limit) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("role", role.to_string()),
                ("operator", node_operator_fact(&node.kind).to_string()),
            ],
        ));
    };
    Ok(limit.input)
}

pub(in crate::capability) fn sort_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    sort_input_with_missing_reason(plan, node_id, "hash_resident_grouped_aggregate_topk_missing_node", reason, role)
}

pub(in crate::capability) fn sort_input_with_missing_reason(
    plan: &QueryPlan,
    node_id: NodeId,
    missing_reason: &'static str,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            missing_reason,
            [("node_id", node_id.to_string()), ("role", role.to_string())],
        ));
    };
    let PlanNode::Sort(sort) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("role", role.to_string()),
                ("operator", node_operator_fact(&node.kind).to_string()),
            ],
        ));
    };
    Ok(sort.input)
}

pub(in crate::capability) fn scalar_aggregate_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "mixed_residual_resident_aggregate_missing_node",
            [("node_id", node_id.to_string()), ("role", "post_root_aggregate".to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    if !aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("operator", "aggregate".to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
            ],
        ));
    }
    Ok(aggregate.input)
}

pub(in crate::capability) fn grouped_aggregate_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "hash_resident_grouped_aggregate_topk_missing_node",
            [("node_id", node_id.to_string()), ("role", "post_root_aggregate".to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    if aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("operator", "aggregate".to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
            ],
        ));
    }
    Ok(aggregate.input)
}

pub(in crate::capability) fn require_join_node<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a JoinNode, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "mixed_residual_resident_aggregate_missing_node",
            [("node_id", node_id.to_string()), ("role", "root_join".to_string())],
        ));
    };
    let PlanNode::Join(join) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    Ok(join)
}

pub(in crate::capability) fn projected_output_side(
    join: &JoinNode,
    join_id: NodeId,
) -> Result<JoinSide, CapabilityDecision> {
    let Some(first) = join.spec.output_projection.first().map(|column| column.side) else {
        return Err(CapabilityDecision::rejected(
            "root_join_output_projection_requires_projected_columns",
            [("node_id", join_id.to_string())],
        ));
    };
    if join.spec.output_projection.iter().any(|column| column.side != first) {
        return Err(CapabilityDecision::rejected(
            "root_join_output_projection_requires_join_subtree_side_only",
            [("node_id", join_id.to_string()), ("output_projection_side", "mixed".to_string())],
        ));
    }
    Ok(first)
}

pub(in crate::capability) fn output_projection_single_side(
    join: &JoinNode,
    join_id: NodeId,
    reason: &'static str,
) -> Result<JoinSide, CapabilityDecision> {
    let Some(first) = join.spec.output_projection.first().map(|column| column.side) else {
        return Err(CapabilityDecision::rejected(reason, [("node_id", join_id.to_string())]));
    };
    if join.spec.output_projection.iter().any(|column| column.side != first) {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", join_id.to_string()), ("output_projection_side", "mixed".to_string())],
        ));
    }
    Ok(first)
}

pub(in crate::capability) fn projection_wrapped_join_subtree_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let (subtree_root, projections) = peel_projection_chain_ids_for_shape(plan, node_id)?;
    let Some(projection_id) = projections.first().copied() else {
        return Err(CapabilityDecision::rejected(
            "join_subtree_input_requires_projection_wrapped_join_subtree",
            [("node_id", node_id.to_string()), ("operator", node_kind_for_fact(plan, node_id))],
        ));
    };

    let mut join_count = 0;
    validate_bounded_join_subtree_for_capability(plan, subtree_root, &mut join_count)?;
    if join_count == 0 {
        return Err(CapabilityDecision::rejected(
            "join_subtree_input_requires_join_subtree",
            [("node_id", subtree_root.to_string()), ("operator", node_kind_for_fact(plan, subtree_root))],
        ));
    }
    Ok((subtree_root, projection_id))
}

pub(in crate::capability) fn projection_wrapped_grouped_local_parquet_aggregate_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_missing_projection",
            [("node_id", node_id.to_string())],
        ));
    };
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_requires_projection_wrapped_grouped_aggregate",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    let aggregate_id = projection.input;
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_missing_aggregate",
            [("node_id", aggregate_id.to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_requires_grouped_aggregate",
            [("node_id", aggregate_id.to_string()), ("operator", node_operator_fact(&aggregate_node.kind).to_string())],
        ));
    };
    if aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_requires_grouped_aggregate",
            [("node_id", aggregate_id.to_string()), ("group_expr_count", "0".to_string())],
        ));
    }
    validate_local_parquet_leaf_pipeline_for_capability(plan, aggregate.input)?;
    Ok((aggregate_id, node_id))
}

pub(in crate::capability) fn projection_wrapped_grouped_aggregate_over_join_subplan_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_missing_projection",
            [("node_id", node_id.to_string())],
        ));
    };
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_requires_projection_wrapped_grouped_aggregate",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    let aggregate_id = projection.input;
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_missing_aggregate",
            [("node_id", aggregate_id.to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_requires_grouped_aggregate",
            [("node_id", aggregate_id.to_string()), ("operator", node_operator_fact(&aggregate_node.kind).to_string())],
        ));
    };
    if aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_requires_grouped_aggregate",
            [("node_id", aggregate_id.to_string()), ("group_expr_count", "0".to_string())],
        ));
    }
    validate_bounded_join_subtree_rooted_at_join_for_capability(plan, aggregate.input)?;
    Ok((aggregate_id, node_id))
}

pub(in crate::capability) fn projection_wrapped_grouped_sum_local_parquet_aggregate_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<NodeId, CapabilityDecision> {
    let (aggregate_id, projection_ids) = peel_projection_chain_ids_for_shape(plan, node_id)?;
    if projection_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_wrapped_grouped_sum_aggregate_requires_projection",
            [("node_id", node_id.to_string()), ("operator", node_kind_for_fact(plan, node_id))],
        ));
    }
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return Err(CapabilityDecision::rejected(
            "projection_wrapped_grouped_sum_aggregate_missing_aggregate",
            [("node_id", aggregate_id.to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(CapabilityDecision::rejected(
            "projection_wrapped_grouped_sum_aggregate_requires_grouped_aggregate",
            [("node_id", aggregate_id.to_string()), ("operator", node_operator_fact(&aggregate_node.kind).to_string())],
        ));
    };
    if aggregate.spec.group_exprs.len() != 1 || aggregate.spec.aggr_exprs.len() != 1 {
        return Err(CapabilityDecision::rejected(
            "projection_wrapped_grouped_sum_aggregate_requires_single_group_and_measure",
            [
                ("node_id", aggregate_id.to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                ("aggregate_expr_count", aggregate.spec.aggr_exprs.len().to_string()),
            ],
        ));
    }
    if aggregate.spec.aggr_exprs[0].function != AggregateFunction::Sum {
        return Err(CapabilityDecision::rejected(
            "projection_wrapped_grouped_sum_aggregate_requires_sum",
            [
                ("node_id", aggregate_id.to_string()),
                ("function", format!("{:?}", aggregate.spec.aggr_exprs[0].function)),
            ],
        ));
    }
    validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(plan, aggregate.input)?;
    Ok(aggregate_id)
}

pub(in crate::capability) fn projection_or_filter_wrapped_grouped_local_parquet_aggregate_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let mut current = node_id;
    let mut saw_wrapper = false;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "resident_grouped_aggregate_topk_input_cycle",
                [("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "resident_grouped_aggregate_topk_input_missing_node",
                [("node_id", current.to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Projection(projection) => {
                saw_wrapper = true;
                current = projection.input;
            },
            PlanNode::Filter(filter) => {
                saw_wrapper = true;
                current = filter.input;
            },
            PlanNode::Aggregate(aggregate) => {
                if !saw_wrapper {
                    return Err(CapabilityDecision::rejected(
                        "resident_grouped_aggregate_topk_input_requires_projection_or_filter_wrapped_grouped_aggregate",
                        [("node_id", node_id.to_string())],
                    ));
                }
                if aggregate.spec.group_exprs.is_empty() {
                    return Err(CapabilityDecision::rejected(
                        "resident_grouped_aggregate_topk_input_requires_grouped_aggregate",
                        [("node_id", current.to_string()), ("group_expr_count", "0".to_string())],
                    ));
                }
                validate_local_parquet_leaf_pipeline_for_capability(plan, aggregate.input)?;
                return Ok((current, node_id));
            },
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {
                return Err(CapabilityDecision::rejected(
                    "resident_grouped_aggregate_topk_input_requires_grouped_aggregate",
                    [("node_id", current.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
                ));
            },
        }
    }
}

pub(in crate::capability) fn peel_resident_unary_chain_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Option<(NodeId, Vec<NodeId>)> {
    let mut current = node_id;
    let mut chain = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        match &node.kind {
            PlanNode::Projection(projection) => {
                chain.push(current);
                current = projection.input;
            },
            PlanNode::Filter(filter) => {
                chain.push(current);
                current = filter.input;
            },
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {
                return Some((current, chain));
            },
        }
    }
}

pub(in crate::capability) fn peel_projection_chain_ids_for_shape(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, Vec<NodeId>), CapabilityDecision> {
    let mut current = node_id;
    let mut projections = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected("projection_chain_cycle", [("node_id", current.to_string())]));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "projection_chain_missing_node",
                [("node_id", current.to_string())],
            ));
        };
        let PlanNode::Projection(projection) = &node.kind else {
            return Ok((current, projections));
        };
        projections.push(current);
        current = projection.input;
    }
}

pub(in crate::capability) fn validate_bounded_join_subtree_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
    join_count: &mut usize,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "join_subtree_input_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(join) => {
            if multi_join_subtree_join_execution_capability(join).is_none() {
                return Err(CapabilityDecision::rejected(
                    "join_subtree_input_unsupported_join",
                    [
                        ("node_id", node_id.to_string()),
                        ("join_type", join.spec.join_type.to_string()),
                        ("key_count", join.spec.keys.len().to_string()),
                        ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                        ("build_side", join.spec.build_side.to_string()),
                    ],
                ));
            }
            *join_count += 1;
            validate_bounded_join_subtree_for_capability(plan, join.inputs[0], join_count)?;
            validate_bounded_join_subtree_for_capability(plan, join.inputs[1], join_count)
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_bounded_join_subtree_for_capability(plan, filter.input, join_count),
        PlanNode::Projection(projection) => {
            validate_bounded_join_subtree_for_capability(plan, projection.input, join_count)
        },
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "join_subtree_input_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn validate_bounded_join_subtree_with_preaggregate_leaves_rooted_at_join_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_join_subplan_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(_) => {
            let mut join_count = 0;
            validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(plan, node_id, &mut join_count)
        },
        PlanNode::Projection(projection) => {
            validate_bounded_join_subtree_with_preaggregate_leaves_rooted_at_join_for_capability(plan, projection.input)
        },
        PlanNode::Source(_)
        | PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_requires_join_subplan",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

fn validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
    join_count: &mut usize,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "join_subtree_input_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(join) => {
            if multi_join_subtree_join_execution_capability(join).is_none() {
                return Err(CapabilityDecision::rejected(
                    "join_subtree_input_unsupported_join",
                    [
                        ("node_id", node_id.to_string()),
                        ("join_type", join.spec.join_type.to_string()),
                        ("key_count", join.spec.keys.len().to_string()),
                        ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                        ("build_side", join.spec.build_side.to_string()),
                    ],
                ));
            }
            *join_count += 1;
            validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(plan, join.inputs[0], join_count)?;
            validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(plan, join.inputs[1], join_count)
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => {
            validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(plan, filter.input, join_count)
        },
        PlanNode::Projection(projection) => {
            validate_bounded_join_subtree_with_preaggregate_leaves_for_capability(plan, projection.input, join_count)
        },
        PlanNode::Aggregate(aggregate) => {
            validate_grouped_sum_preaggregate_leaf(plan, node_id, aggregate)?;
            Ok(())
        },
        PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "join_subtree_input_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

fn validate_grouped_sum_preaggregate_leaf(
    plan: &QueryPlan,
    node_id: NodeId,
    aggregate: &AggregateNode,
) -> Result<(), CapabilityDecision> {
    if aggregate.spec.group_exprs.is_empty()
        || aggregate.spec.aggr_exprs.is_empty()
        || aggregate.spec.grouping_sets.is_some()
    {
        return Err(CapabilityDecision::rejected(
            "join_subtree_input_unsupported_preaggregate",
            [
                ("node_id", node_id.to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                ("aggregate_expr_count", aggregate.spec.aggr_exprs.len().to_string()),
                ("has_grouping_sets", aggregate.spec.grouping_sets.is_some().to_string()),
            ],
        ));
    }
    for expr in &aggregate.spec.aggr_exprs {
        if expr.function != AggregateFunction::Sum || expr.distinct || expr.filter.is_some() {
            return Err(CapabilityDecision::rejected(
                "join_subtree_input_unsupported_preaggregate",
                [
                    ("node_id", node_id.to_string()),
                    ("function", expr.function.as_str().to_string()),
                    ("distinct", expr.distinct.to_string()),
                    ("has_filter", expr.filter.is_some().to_string()),
                ],
            ));
        }
    }
    validate_local_parquet_leaf_pipeline_for_capability(plan, aggregate.input)
}

pub(in crate::capability) fn validate_bounded_join_subtree_rooted_at_join_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_join_subplan_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(_) => {
            let mut join_count = 0;
            validate_bounded_join_subtree_for_capability(plan, node_id, &mut join_count)
        },
        PlanNode::Projection(projection) => {
            validate_bounded_join_subtree_rooted_at_join_for_capability(plan, projection.input)
        },
        PlanNode::Source(_)
        | PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "aggregate_over_join_topk_input_requires_join_subplan",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn validate_local_parquet_leaf_pipeline_for_capability(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_missing_leaf_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Ok(()),
        PlanNode::Source(_) => Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_requires_local_parquet_source",
            [("node_id", node_id.to_string()), ("source_format", "gpu_resident".to_string())],
        )),
        PlanNode::Filter(filter) => validate_local_parquet_leaf_pipeline_for_capability(plan, filter.input),
        PlanNode::Projection(projection) => validate_local_parquet_leaf_pipeline_for_capability(plan, projection.input),
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "resident_grouped_aggregate_input_requires_source_filter_projection_leaf",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_leaf_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Ok(()),
        PlanNode::Source(_) => Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_leaf_requires_local_parquet_source",
            [("node_id", node_id.to_string()), ("source_format", "gpu_resident".to_string())],
        )),
        PlanNode::Filter(filter) => {
            validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(plan, filter.input)
        },
        PlanNode::Projection(projection) => {
            validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(plan, projection.input)
        },
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_leaf_requires_source_filter_projection",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn generic_join_node<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a JoinNode, CapabilityDecision> {
    join_node_with_missing_reason(plan, node_id, "keyed_hash_resident_subplan_missing_node", reason)
}

pub(in crate::capability) fn join_node_with_missing_reason<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    missing_reason: &'static str,
    reason: &'static str,
) -> Result<&'a JoinNode, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(missing_reason, [("node_id", node_id.to_string())]));
    };
    let PlanNode::Join(join) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    Ok(join)
}

pub(in crate::capability) fn require_keyed_inner_hash_join_without_residual(
    join_id: NodeId,
    join: &JoinNode,
    reason: &'static str,
) -> Result<(), CapabilityDecision> {
    if join.spec.join_type == JoinType::Inner
        && join.spec.keys.len() == 1
        && join.spec.residual_predicate.is_none()
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

pub(in crate::capability) fn require_keyed_inner_hash_join_without_residual_with_nonempty_keys(
    join_id: NodeId,
    join: &JoinNode,
    reason: &'static str,
) -> Result<(), CapabilityDecision> {
    if join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
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

pub(in crate::capability) fn require_left_semi_keyed_hash_join_with_right_build(
    join_id: NodeId,
    join: &JoinNode,
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
        "hash_resident_grouped_aggregate_topk_requires_left_semi_keyed_hash_join",
        [
            ("node_id", join_id.to_string()),
            ("join_type", join.spec.join_type.to_string()),
            ("key_count", join.spec.keys.len().to_string()),
            ("has_residual", join.spec.residual_predicate.is_some().to_string()),
            ("build_side", join.spec.build_side.to_string()),
            ("output_projection_count", join.spec.output_projection.len().to_string()),
            (
                "projects_only_left",
                join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left).to_string(),
            ),
        ],
    ))
}

pub(in crate::capability) fn join_key_lineage_for_side(
    plan: &QueryPlan,
    join: &JoinNode,
    side: JoinSide,
) -> Result<OutputLineage, CapabilityDecision> {
    let [key] = join.spec.keys.as_slice() else {
        return Err(CapabilityDecision::rejected(
            "join_key_lineage_requires_single_key",
            [("key_count", join.spec.keys.len().to_string())],
        ));
    };
    let input_id = join.inputs[join_side_index(side)];
    let column = match side {
        JoinSide::Left => &key.left,
        JoinSide::Right => &key.right,
    };
    let field_index = resolve_column_field_index(plan, input_id, column, "join_key_lineage_column_resolution")?;
    output_lineage(plan, input_id, field_index)
}

pub(in crate::capability) fn column_expr_lineage(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
    reason: &'static str,
) -> Result<OutputLineage, CapabilityDecision> {
    match expr {
        Expr::Column(column) => {
            let field_index = resolve_column_field_index(plan, input_id, column, reason)?;
            output_lineage(plan, input_id, field_index)
        },
        Expr::Alias { expr, .. } => column_expr_lineage(plan, input_id, expr, reason),
        _ => Err(CapabilityDecision::rejected(reason, [("expression_family", expr_family(expr).to_string())])),
    }
}

pub(in crate::capability) fn expression_lineage_for_input(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
) -> Result<OutputLineage, CapabilityDecision> {
    column_expr_lineage(plan, input_id, expr, "expression_lineage_requires_resolved_input_column")
}

pub(in crate::capability) fn output_lineage(
    plan: &QueryPlan,
    node_id: NodeId,
    field_index: usize,
) -> Result<OutputLineage, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "output_lineage_missing_node",
            [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
        ));
    };
    if field_index >= node.output_schema().fields().len() {
        return Err(CapabilityDecision::rejected(
            "output_lineage_field_index_out_of_range",
            [
                ("node_id", node_id.to_string()),
                ("field_index", field_index.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
            ],
        ));
    }
    match &node.kind {
        PlanNode::Source(_) => Ok(OutputLineage::SourceField { source_id: node_id, field_index }),
        PlanNode::Filter(filter) => output_lineage(plan, filter.input, field_index),
        PlanNode::Sort(sort) => output_lineage(plan, sort.input, field_index),
        PlanNode::Limit(limit) => output_lineage(plan, limit.input, field_index),
        PlanNode::Projection(projection) => {
            let Some(output) = projection.outputs.get(field_index) else {
                return Err(CapabilityDecision::rejected(
                    "output_lineage_projection_field_index_out_of_range",
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            match &output.expr {
                Expr::Column(column) => {
                    let input_field = resolve_column_field_index(
                        plan,
                        projection.input,
                        column,
                        "output_lineage_projection_column_resolution",
                    )?;
                    output_lineage(plan, projection.input, input_field)
                },
                Expr::Alias { expr, .. } => expression_lineage_for_input(plan, projection.input, expr),
                _ => Ok(OutputLineage::Derived { node_id, field_index }),
            }
        },
        PlanNode::Aggregate(aggregate) => {
            let group_count = aggregate.spec.group_exprs.len();
            if field_index < group_count {
                return Ok(OutputLineage::AggregateGroup { aggregate_id: node_id, group_index: field_index });
            }
            let aggr_index = field_index - group_count;
            let Some(aggregate_expr) = aggregate.spec.aggr_exprs.get(aggr_index) else {
                return Err(CapabilityDecision::rejected(
                    "output_lineage_aggregate_field_index_out_of_range",
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            if group_count == 0 {
                let input = Box::new(expression_lineage_for_input(plan, aggregate.input, &aggregate_expr.expr)?);
                Ok(OutputLineage::ScalarAggregateMeasure {
                    aggregate_id: node_id,
                    aggr_index,
                    function: aggregate_expr.function,
                    input,
                })
            } else {
                Ok(OutputLineage::AggregateMeasure {
                    aggregate_id: node_id,
                    aggr_index,
                    function: aggregate_expr.function,
                })
            }
        },
        PlanNode::Join(join) => {
            let Some(output) = join.spec.output_projection.get(field_index) else {
                return Err(CapabilityDecision::rejected(
                    "output_lineage_join_projection_index_out_of_range",
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            let input_id = join.inputs[join_side_index(output.side)];
            let input_field = resolve_column_field_index(
                plan,
                input_id,
                &output.column,
                "output_lineage_join_output_column_resolution",
            )?;
            output_lineage(plan, input_id, input_field)
        },
        PlanNode::Window(window) => {
            let input_field_count =
                plan.node(window.input).map(|input| input.output_schema().fields().len()).ok_or_else(|| {
                    CapabilityDecision::rejected(
                        "output_lineage_window_missing_input",
                        [("node_id", node_id.to_string()), ("input_id", window.input.to_string())],
                    )
                })?;
            if field_index < input_field_count {
                output_lineage(plan, window.input, field_index)
            } else {
                Ok(OutputLineage::Derived { node_id, field_index })
            }
        },
        PlanNode::Union(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => {
            Ok(OutputLineage::Derived { node_id, field_index })
        },
    }
}

pub(in crate::capability) fn resolve_column_field_index(
    plan: &QueryPlan,
    input_id: NodeId,
    column: &crate::expr::ColumnRef,
    reason: &'static str,
) -> Result<usize, CapabilityDecision> {
    let Some(node) = plan.node(input_id) else {
        return Err(CapabilityDecision::rejected(reason, [("input_id", input_id.to_string())]));
    };
    let mut matches = node
        .output_schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| (field.name() == column.name()).then_some(index));
    let Some(index) = matches.next() else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("input_id", input_id.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
                ("resolution", "missing".to_string()),
            ],
        ));
    };
    if matches.next().is_some() {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("input_id", input_id.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
                ("resolution", "ambiguous".to_string()),
            ],
        ));
    }
    Ok(index)
}

pub(in crate::capability) fn node_kind_for_fact(plan: &QueryPlan, node_id: NodeId) -> String {
    plan.node(node_id).map(|node| node_operator_fact(&node.kind).to_string()).unwrap_or_else(|| "missing".to_string())
}

pub(in crate::capability) const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(in crate::capability) const fn opposite_join_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

pub(in crate::capability) fn closed_multi_join_subgraph_structure_decision(plan: &QueryPlan) -> CapabilityDecision {
    let root_join_id = match post_root_join_id(plan) {
        Ok(root_join_id) => root_join_id,
        Err(decision) => return decision,
    };
    let mut join_count = 0;
    if let Err(decision) = validate_closed_multi_join_subtree(plan, root_join_id, &mut join_count) {
        return decision;
    }
    if join_count <= 1 {
        return CapabilityDecision::rejected(
            "closed_multi_join_subgraph_requires_multiple_joins",
            [("root_join_id", root_join_id.to_string()), ("join_count", join_count.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn validate_closed_multi_join_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
    join_count: &mut usize,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "closed_multi_join_subgraph_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(join) => {
            if multi_join_subtree_join_execution_capability(join).is_none() {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_unsupported_join",
                    [
                        ("node_id", node_id.to_string()),
                        ("join_type", join.spec.join_type.to_string()),
                        ("key_count", join.spec.keys.len().to_string()),
                        ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                        ("build_side", join.spec.build_side.to_string()),
                    ],
                ));
            }
            *join_count += 1;
            validate_closed_multi_join_subtree(plan, join.inputs[0], join_count)?;
            validate_closed_multi_join_subtree(plan, join.inputs[1], join_count)?;
            Ok(())
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_closed_multi_join_subtree(plan, filter.input, join_count),
        PlanNode::Projection(projection) => validate_closed_multi_join_subtree(plan, projection.input, join_count),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "closed_multi_join_subgraph_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn post_root_join_id(plan: &QueryPlan) -> Result<NodeId, CapabilityDecision> {
    let mut current = plan.sink().input();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "closed_multi_join_subgraph_cycle",
                [("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "closed_multi_join_subgraph_missing_node",
                [("node_id", current.to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Join(_) => return Ok(current),
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Aggregate(aggregate) => current = aggregate.input,
            PlanNode::Window(window) => current = window.input,
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Limit(limit) => current = limit.input,
            PlanNode::Union(_) => {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_unsupported_node",
                    [("node_id", current.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
                ));
            },
            PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_unsupported_node",
                    [("node_id", current.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
                ));
            },
            PlanNode::Source(_) => {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_missing_root_join",
                    [("sink_input", plan.sink().input().to_string())],
                ));
            },
        }
    }
}
