use super::{
    AggregateFunction, CapabilityDecision, JoinSide, NodeId, OutputLineage, PlanNode, QueryPlan,
    expression_lineage_for_input, generic_join_node, join_key_lineage_for_side, join_side_index, node_operator_fact,
    opposite_join_side, peel_projection_chain_ids_for_shape,
    projection_wrapped_grouped_sum_local_parquet_aggregate_root, require_keyed_inner_hash_join_without_residual,
    validate_local_parquet_leaf_pipeline_for_keyed_hash_resident_subplan,
};

#[derive(Debug, Clone, Copy)]
pub(in crate::capability) struct KeyedHashJoinedGroupedBranchShape {
    pub(in crate::capability) join_id: NodeId,
    pub(in crate::capability) grouped_aggregate_id: NodeId,
}

pub(in crate::capability) fn keyed_hash_joined_grouped_branch_shape_decision(
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
pub(in crate::capability) struct KeyedHashScalarMaxBranchShape {
    pub(in crate::capability) aggregate_id: NodeId,
    pub(in crate::capability) shared_grouped_aggregate_id: NodeId,
}

pub(in crate::capability) fn keyed_hash_scalar_max_branch_shape_decision(
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
