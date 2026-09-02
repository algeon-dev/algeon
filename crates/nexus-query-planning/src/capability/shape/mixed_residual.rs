use super::*;

pub(in crate::capability) fn mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(
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
    if !matches!(root_join.spec.join_type, JoinType::Inner | JoinType::LeftSemi)
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
