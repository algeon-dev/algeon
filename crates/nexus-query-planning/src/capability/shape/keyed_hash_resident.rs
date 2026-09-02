use super::{
    AggregateFunction, CapabilityDecision, KeyedHashJoinResidentSubplanShape, OutputLineage, PlanNode, QueryPlan,
    column_expr_lineage, generic_join_node, join_key_lineage_for_side, join_side_index,
    keyed_hash_joined_grouped_branch_shape_decision, keyed_hash_scalar_max_branch_shape_decision, node_operator_fact,
    opposite_join_side, output_projection_single_side, require_keyed_inner_hash_join_without_residual,
};

pub(in crate::capability) fn keyed_hash_join_resident_subplan_shape_decision(
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
