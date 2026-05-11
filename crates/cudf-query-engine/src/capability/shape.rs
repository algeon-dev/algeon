use super::{edge::*, expr::*, model::*, *};

#[derive(Debug, Clone, Copy)]
pub(super) struct MixedResidualResidentGroupedAggregateInputShape {
    pub(super) root_join_id: NodeId,
    pub(super) join_subtree_root_id: NodeId,
    pub(super) join_subtree_projection_id: NodeId,
    pub(super) grouped_aggregate_id: NodeId,
    pub(super) grouped_aggregate_projection_id: NodeId,
    pub(super) post_root_projection_id: NodeId,
    pub(super) post_root_aggregate_id: NodeId,
    pub(super) final_projection_id: NodeId,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKShape {
    pub(super) root_join_id: NodeId,
    pub(super) grouped_aggregate_id: NodeId,
    pub(super) grouped_aggregate_projection_id: NodeId,
    pub(super) post_root_projection_id: NodeId,
    pub(super) post_root_aggregate_id: NodeId,
    pub(super) final_projection_id: NodeId,
    pub(super) final_sort_id: NodeId,
    pub(super) final_limit_id: NodeId,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MultiJoinAggregateOverJoinSubplanTopKShape {
    pub(super) root_join_id: NodeId,
    pub(super) aggregate_id: NodeId,
    pub(super) aggregate_projection_id: NodeId,
    pub(super) post_root_projection_id: NodeId,
    pub(super) final_sort_id: NodeId,
    pub(super) final_limit_id: NodeId,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KeyedHashJoinResidentSubplanShape {
    pub(super) final_sort_id: NodeId,
    pub(super) root_join_id: NodeId,
    pub(super) joined_grouped_branch_join_id: NodeId,
    pub(super) shared_grouped_aggregate_id: NodeId,
    pub(super) scalar_aggregate_id: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum OutputLineage {
    SourceField {
        source_id: NodeId,
        field_index: usize,
    },
    AggregateGroup {
        aggregate_id: NodeId,
        group_index: usize,
    },
    AggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
    },
    ScalarAggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
        input: Box<OutputLineage>,
    },
    Derived {
        node_id: NodeId,
        field_index: usize,
    },
}

pub(super) fn mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(
    plan: &QueryPlan,
) -> Result<MixedResidualResidentGroupedAggregateInputShape, CapabilityDecision> {
    let final_projection_id = plan.sink().input();
    let final_projection_input = projection_input(
        plan,
        final_projection_id,
        "post_root_stage_order_requires_final_projection",
        "final_projection",
    )?;

    let post_root_aggregate_id = final_projection_input;
    let post_root_aggregate_input =
        scalar_aggregate_input(plan, post_root_aggregate_id, "post_root_stage_order_requires_scalar_aggregate")?;

    let post_root_projection_id = post_root_aggregate_input;
    let root_join_id = projection_input(
        plan,
        post_root_projection_id,
        "post_root_stage_order_requires_projection_before_scalar_aggregate",
        "pre_aggregate_projection",
    )?;

    let root_join = require_join_node(plan, root_join_id, "post_root_stage_order_requires_root_join_below_projection")?;
    if root_join.spec.join_type != JoinType::Inner
        || root_join.spec.keys.is_empty()
        || root_join.spec.residual_predicate.is_none()
        || root_join.spec.output_projection.is_empty()
    {
        return Err(CapabilityDecision::rejected(
            "root_join_requires_inner_keyed_residual",
            [
                ("node_id", root_join_id.to_string()),
                ("join_type", root_join.spec.join_type.to_string()),
                ("key_count", root_join.spec.keys.len().to_string()),
                ("has_residual", root_join.spec.residual_predicate.is_some().to_string()),
                ("output_projection_count", root_join.spec.output_projection.len().to_string()),
            ],
        ));
    }

    let join_subtree_side = projected_output_side(root_join, root_join_id)?;
    let grouped_aggregate_side = opposite_join_side(join_subtree_side);

    let join_subtree_input = root_join.inputs[join_side_index(join_subtree_side)];
    let (join_subtree_root_id, join_subtree_projection_id) =
        projection_wrapped_join_subtree_root(plan, join_subtree_input)?;

    let grouped_aggregate_input = root_join.inputs[join_side_index(grouped_aggregate_side)];
    let (grouped_aggregate_id, grouped_aggregate_projection_id) =
        projection_wrapped_grouped_local_parquet_aggregate_root(plan, grouped_aggregate_input)?;

    Ok(MixedResidualResidentGroupedAggregateInputShape {
        root_join_id,
        join_subtree_root_id,
        join_subtree_projection_id,
        grouped_aggregate_id,
        grouped_aggregate_projection_id,
        post_root_projection_id,
        post_root_aggregate_id,
        final_projection_id,
    })
}

pub(super) fn keyed_hash_join_resident_subplan_shape_decision(
    plan: &QueryPlan,
) -> Result<KeyedHashJoinResidentSubplanShape, CapabilityDecision> {
    let final_sort_id = plan.sink().input();
    let final_sort_node = plan.node(final_sort_id).ok_or_else(|| {
        CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_missing_node",
            [("node_id", final_sort_id.to_string())],
        )
    })?;
    let PlanNode::Sort(final_sort) = &final_sort_node.kind else {
        return Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_requires_final_sort",
            [
                ("node_id", final_sort_id.to_string()),
                ("operator", node_operator_fact(&final_sort_node.kind).to_string()),
            ],
        ));
    };
    let [sort_expr] = final_sort.spec.expressions.as_slice() else {
        return Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_requires_single_final_sort_key",
            [("node_id", final_sort_id.to_string()), ("sort_key_count", final_sort.spec.expressions.len().to_string())],
        ));
    };
    if sort_expr.descending {
        return Err(CapabilityDecision::rejected(
            "keyed_hash_resident_subplan_requires_ascending_final_sort",
            [("node_id", final_sort_id.to_string())],
        ));
    }

    let root_join_id = final_sort.input;
    let root_join =
        generic_join_node(plan, root_join_id, "keyed_hash_resident_subplan_requires_root_join_below_final_sort")?;
    require_keyed_inner_hash_join_without_residual(root_join_id, root_join, "keyed_hash_resident_subplan_root_join")?;
    let _ = column_expr_lineage(
        plan,
        root_join_id,
        &sort_expr.expr,
        "keyed_hash_resident_subplan_final_sort_key_must_resolve_to_root_output",
    )?;

    let joined_grouped_branch_side =
        output_projection_single_side(root_join, root_join_id, "keyed_hash_resident_subplan_root_output_side")?;
    let scalar_max_side = opposite_join_side(joined_grouped_branch_side);
    let joined_grouped_branch_input = root_join.inputs[join_side_index(joined_grouped_branch_side)];
    let scalar_max_input = root_join.inputs[join_side_index(scalar_max_side)];

    let joined_grouped_branch = keyed_hash_joined_grouped_branch_shape_decision(plan, joined_grouped_branch_input)?;
    let scalar_max = keyed_hash_scalar_max_branch_shape_decision(plan, scalar_max_input)?;
    if joined_grouped_branch.grouped_aggregate_id != scalar_max.shared_grouped_aggregate_id {
        return Err(CapabilityDecision::rejected(
            "resident_subplan_requires_same_node_id_after_cse",
            [
                ("joined_branch_grouped_aggregate", joined_grouped_branch.grouped_aggregate_id.to_string()),
                ("scalar_grouped_aggregate", scalar_max.shared_grouped_aggregate_id.to_string()),
                ("identity", "same_node_id_after_cse".to_string()),
            ],
        ));
    }

    let shared_measure = OutputLineage::AggregateMeasure {
        aggregate_id: joined_grouped_branch.grouped_aggregate_id,
        aggr_index: 0,
        function: AggregateFunction::Sum,
    };
    let joined_branch_key_lineage = join_key_lineage_for_side(plan, root_join, joined_grouped_branch_side)?;
    if joined_branch_key_lineage != shared_measure {
        return Err(CapabilityDecision::rejected(
            "root_join_key_requires_shared_grouped_aggregate_measure",
            [("root_join_id", root_join_id.to_string())],
        ));
    }
    let scalar_key_lineage = join_key_lineage_for_side(plan, root_join, scalar_max_side)?;
    let expected_scalar_lineage = OutputLineage::ScalarAggregateMeasure {
        aggregate_id: scalar_max.aggregate_id,
        aggr_index: 0,
        function: AggregateFunction::Max,
        input: Box::new(shared_measure),
    };
    if scalar_key_lineage != expected_scalar_lineage {
        return Err(CapabilityDecision::rejected(
            "root_join_key_requires_scalar_max_of_shared_grouped_aggregate_measure",
            [("root_join_id", root_join_id.to_string()), ("scalar_aggregate_id", scalar_max.aggregate_id.to_string())],
        ));
    }

    Ok(KeyedHashJoinResidentSubplanShape {
        final_sort_id,
        root_join_id,
        joined_grouped_branch_join_id: joined_grouped_branch.join_id,
        shared_grouped_aggregate_id: joined_grouped_branch.grouped_aggregate_id,
        scalar_aggregate_id: scalar_max.aggregate_id,
    })
}

pub(super) fn hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(
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

pub(super) fn multi_join_aggregate_over_join_subplan_topk_shape_decision(
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

#[derive(Debug, Clone, Copy)]
pub(super) struct KeyedHashJoinedGroupedBranchShape {
    join_id: NodeId,
    grouped_aggregate_id: NodeId,
}

pub(super) fn keyed_hash_joined_grouped_branch_shape_decision(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<KeyedHashJoinedGroupedBranchShape, CapabilityDecision> {
    let (join_id, _) = peel_projection_chain_ids_for_shape(plan, node_id)?;
    let join = generic_join_node(plan, join_id, "keyed_hash_resident_subplan_requires_joined_grouped_branch_join")?;
    require_keyed_inner_hash_join_without_residual(
        join_id,
        join,
        "keyed_hash_resident_subplan_joined_grouped_branch_join",
    )?;

    for grouped_side in [JoinSide::Left, JoinSide::Right] {
        let grouped_input = join.inputs[join_side_index(grouped_side)];
        let Ok(grouped_aggregate_id) = projection_wrapped_grouped_sum_local_parquet_aggregate_root(plan, grouped_input)
        else {
            continue;
        };
        let source_side = opposite_join_side(grouped_side);
        validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(
            plan,
            join.inputs[join_side_index(source_side)],
        )?;
        let grouped_key_lineage = join_key_lineage_for_side(plan, join, grouped_side)?;
        let expected_group_lineage =
            OutputLineage::AggregateGroup { aggregate_id: grouped_aggregate_id, group_index: 0 };
        if grouped_key_lineage != expected_group_lineage {
            return Err(CapabilityDecision::rejected(
                "joined_grouped_branch_join_key_requires_grouped_aggregate_group_key",
                [("join_id", join_id.to_string()), ("grouped_aggregate_id", grouped_aggregate_id.to_string())],
            ));
        }
        return Ok(KeyedHashJoinedGroupedBranchShape { join_id, grouped_aggregate_id });
    }
    Err(CapabilityDecision::rejected(
        "joined_grouped_branch_requires_projection_wrapped_grouped_aggregate_input",
        [("join_id", join_id.to_string())],
    ))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KeyedHashScalarMaxBranchShape {
    aggregate_id: NodeId,
    shared_grouped_aggregate_id: NodeId,
}

pub(super) fn keyed_hash_scalar_max_branch_shape_decision(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<KeyedHashScalarMaxBranchShape, CapabilityDecision> {
    let node = plan.node(node_id).ok_or_else(|| {
        CapabilityDecision::rejected("keyed_hash_resident_subplan_missing_node", [("node_id", node_id.to_string())])
    })?;
    let PlanNode::Aggregate(aggregate) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_scalar_aggregate",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    if !aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_scalar_aggregate",
            [("node_id", node_id.to_string()), ("group_expr_count", aggregate.spec.group_exprs.len().to_string())],
        ));
    }
    let [aggregate_expr] = aggregate.spec.aggr_exprs.as_slice() else {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_single_max_aggregate",
            [("node_id", node_id.to_string()), ("aggregate_expr_count", aggregate.spec.aggr_exprs.len().to_string())],
        ));
    };
    if aggregate_expr.function != AggregateFunction::Max {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_max_aggregate",
            [("node_id", node_id.to_string()), ("function", format!("{:?}", aggregate_expr.function))],
        ));
    }
    let (shared_grouped_aggregate_id, projection_ids) = peel_projection_chain_ids_for_shape(plan, aggregate.input)?;
    if projection_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_projection_wrapped_grouped_aggregate_input",
            [("node_id", aggregate.input.to_string())],
        ));
    }
    projection_wrapped_grouped_sum_local_parquet_aggregate_root(plan, aggregate.input)?;
    let aggregate_input_lineage = expression_lineage_for_input(plan, aggregate.input, &aggregate_expr.expr)?;
    let expected_lineage = OutputLineage::AggregateMeasure {
        aggregate_id: shared_grouped_aggregate_id,
        aggr_index: 0,
        function: AggregateFunction::Sum,
    };
    if aggregate_input_lineage != expected_lineage {
        return Err(CapabilityDecision::rejected(
            "scalar_max_branch_requires_max_over_shared_grouped_aggregate_measure",
            [
                ("aggregate_id", node_id.to_string()),
                ("shared_grouped_aggregate_id", shared_grouped_aggregate_id.to_string()),
            ],
        ));
    }
    Ok(KeyedHashScalarMaxBranchShape { aggregate_id: node_id, shared_grouped_aggregate_id })
}

pub(super) fn projection_input(
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

pub(super) fn projection_input_with_missing_reason(
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

pub(super) fn limit_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    limit_input_with_missing_reason(plan, node_id, "hash_resident_grouped_aggregate_topk_missing_node", reason, role)
}

pub(super) fn limit_input_with_missing_reason(
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

pub(super) fn sort_input(
    plan: &QueryPlan,
    node_id: NodeId,
    reason: &'static str,
    role: &'static str,
) -> Result<NodeId, CapabilityDecision> {
    sort_input_with_missing_reason(plan, node_id, "hash_resident_grouped_aggregate_topk_missing_node", reason, role)
}

pub(super) fn sort_input_with_missing_reason(
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

pub(super) fn scalar_aggregate_input(
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

pub(super) fn grouped_aggregate_input(
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

pub(super) fn require_join_node<'a>(
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

pub(super) fn projected_output_side(join: &JoinNode, join_id: NodeId) -> Result<JoinSide, CapabilityDecision> {
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

pub(super) fn output_projection_single_side(
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

pub(super) fn projection_wrapped_join_subtree_root(
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

pub(super) fn projection_wrapped_grouped_local_parquet_aggregate_root(
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

pub(super) fn projection_wrapped_grouped_aggregate_over_join_subplan_root(
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

pub(super) fn projection_wrapped_grouped_sum_local_parquet_aggregate_root(
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

pub(super) fn projection_or_filter_wrapped_grouped_local_parquet_aggregate_root(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let mut current = node_id;
    let mut first_projection_id = None;
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
                first_projection_id.get_or_insert(current);
                current = projection.input;
            },
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Aggregate(aggregate) => {
                let Some(projection_id) = first_projection_id else {
                    return Err(CapabilityDecision::rejected(
                        "resident_grouped_aggregate_topk_input_requires_projection_wrapped_grouped_aggregate",
                        [("node_id", node_id.to_string())],
                    ));
                };
                if aggregate.spec.group_exprs.is_empty() {
                    return Err(CapabilityDecision::rejected(
                        "resident_grouped_aggregate_topk_input_requires_grouped_aggregate",
                        [("node_id", current.to_string()), ("group_expr_count", "0".to_string())],
                    ));
                }
                validate_local_parquet_leaf_pipeline_for_capability(plan, aggregate.input)?;
                return Ok((current, projection_id));
            },
            PlanNode::Source(_) | PlanNode::Join(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
                return Err(CapabilityDecision::rejected(
                    "resident_grouped_aggregate_topk_input_requires_grouped_aggregate",
                    [("node_id", current.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
                ));
            },
        }
    }
}

pub(super) fn peel_resident_unary_chain_for_capability(
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
            | PlanNode::Sort(_)
            | PlanNode::Limit(_) => {
                return Some((current, chain));
            },
        }
    }
}

pub(super) fn peel_projection_chain_ids_for_shape(
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

pub(super) fn validate_bounded_join_subtree_for_capability(
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
            if bounded_multi_join_node_capability(join).is_none() {
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
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(CapabilityDecision::rejected(
            "join_subtree_input_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(super) fn validate_bounded_join_subtree_rooted_at_join_for_capability(
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
        PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
            Err(CapabilityDecision::rejected(
                "aggregate_over_join_topk_input_requires_join_subplan",
                [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            ))
        },
    }
}

pub(super) fn validate_local_parquet_leaf_pipeline_for_capability(
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
        PlanNode::Join(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
            Err(CapabilityDecision::rejected(
                "resident_grouped_aggregate_input_requires_source_filter_projection_leaf",
                [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            ))
        },
    }
}

pub(super) fn validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan(
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
        PlanNode::Join(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
            Err(CapabilityDecision::rejected(
                "keyed_hash_resident_subplan_leaf_requires_source_filter_projection",
                [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            ))
        },
    }
}

pub(super) fn generic_join_node<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a JoinNode, CapabilityDecision> {
    join_node_with_missing_reason(plan, node_id, "keyed_hash_resident_subplan_missing_node", reason)
}

pub(super) fn join_node_with_missing_reason<'a>(
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

pub(super) fn require_keyed_inner_hash_join_without_residual(
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

pub(super) fn require_keyed_inner_hash_join_without_residual_with_nonempty_keys(
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

pub(super) fn require_left_semi_keyed_hash_join_with_right_build(
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

pub(super) fn join_key_lineage_for_side(
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

pub(super) fn column_expr_lineage(
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

pub(super) fn expression_lineage_for_input(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
) -> Result<OutputLineage, CapabilityDecision> {
    column_expr_lineage(plan, input_id, expr, "expression_lineage_requires_resolved_input_column")
}

pub(super) fn output_lineage(
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
    }
}

pub(super) fn resolve_column_field_index(
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

pub(super) fn node_kind_for_fact(plan: &QueryPlan, node_id: NodeId) -> String {
    plan.node(node_id).map(|node| node_operator_fact(&node.kind).to_string()).unwrap_or_else(|| "missing".to_string())
}

pub(super) const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(super) const fn opposite_join_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

pub(super) fn closed_multi_join_subgraph_structure_decision(plan: &QueryPlan) -> CapabilityDecision {
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

pub(super) fn validate_closed_multi_join_subtree(
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
            if bounded_multi_join_node_capability(join).is_none() {
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
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(CapabilityDecision::rejected(
            "closed_multi_join_subgraph_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(super) fn post_root_join_id(plan: &QueryPlan) -> Result<NodeId, CapabilityDecision> {
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
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Limit(limit) => current = limit.input,
            PlanNode::Source(_) => {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_missing_root_join",
                    [("sink_input", plan.sink().input().to_string())],
                ));
            },
        }
    }
}
