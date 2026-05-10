use super::*;

mod inputs;

pub(super) use inputs::{
    AggregateOverJoinSubplanInputKind, bounded_join_inputs_include_resident_subplan,
    bounded_join_source_leaf_inputs_compatible, validate_bounded_join_source_leaf_inputs,
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_post_root_pipeline,
    validate_bounded_keyless_conditional_join_input_subplan,
    validate_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
    validate_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_sides,
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_post_root_pipeline,
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_sides,
    validate_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
    validate_multi_join_resident_grouped_aggregate_subquery_filter_sides,
    validate_projection_wrapped_aggregate_over_join_subplan_input, validate_source_filter_projection_leaf_pipeline,
};

pub(super) fn analyze_bounded_join_pipeline_shape(
    plan: &QueryPlan,
    require_single_reachable_join: bool,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let mut current = plan.sink().input();
    let mut stages = Vec::new();

    loop {
        let node = plan.node(current).ok_or(BoundedJoinPipelineRejection::MissingNode {
            node_id: current,
            message: "bounded join pipeline references a missing plan node",
        })?;
        match &node.kind {
            PlanNode::Join(join) => {
                if !bounded_root_join_compatible(join) {
                    return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
                }
                if require_single_reachable_join && reachable_join_count(plan) != 1 {
                    return Err(BoundedJoinPipelineRejection::MultipleReachableJoins);
                }
                if require_single_reachable_join {
                    validate_bounded_join_source_leaf_inputs(plan, join)?;
                }
                stages.reverse();
                return Ok(BoundedJoinPipelineShape { join_id: current, join, stages });
            },
            PlanNode::Filter(filter) => {
                stages.push(BoundedJoinPipelineStage::Filter { node_id: current, node: filter });
                current = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(BoundedJoinPipelineStage::Projection {
                    node_id: current,
                    node: projection,
                    output_schema: &node.facts.output_schema,
                });
                current = projection.input;
            },
            PlanNode::Aggregate(aggregate) => {
                stages.push(BoundedJoinPipelineStage::Aggregate {
                    node_id: current,
                    node: aggregate,
                    output_schema: &node.facts.output_schema,
                });
                current = aggregate.input;
            },
            PlanNode::Sort(sort) => {
                stages.push(BoundedJoinPipelineStage::Sort { node_id: current, node: sort });
                current = sort.input;
            },
            PlanNode::Limit(limit) => {
                let child = plan.node(limit.input).ok_or(BoundedJoinPipelineRejection::MissingNode {
                    node_id: limit.input,
                    message: "bounded join pipeline limit references a missing input",
                })?;
                if !matches!(child.kind, PlanNode::Sort(_)) {
                    return Err(BoundedJoinPipelineRejection::LimitNotDirectlyAboveSort);
                }
                stages.push(BoundedJoinPipelineStage::Limit { node_id: current, node: limit });
                current = limit.input;
            },
            PlanNode::Source(_) => return Err(BoundedJoinPipelineRejection::MissingRootJoin),
        }
    }
}

pub(super) fn validate_bounded_multi_join_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded multi-join pipeline references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Join(join) => {
            if !bounded_multi_join_compatible(join) {
                return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
            }
            validate_bounded_multi_join_subtree(plan, join.inputs[0])?;
            validate_bounded_multi_join_subtree(plan, join.inputs[1])?;
            Ok(())
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_bounded_multi_join_subtree(plan, filter.input),
        PlanNode::Projection(projection) => validate_bounded_multi_join_subtree(plan, projection.input),
        PlanNode::Aggregate(_) => {
            Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "limit" }),
    }
}

pub(super) fn bounded_join_subtree_join_count(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<usize, BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded multi-join pipeline references a missing plan node",
    })?;
    let mut count = usize::from(matches!(node.kind, PlanNode::Join(_)));
    for input in node.input_ids() {
        count += bounded_join_subtree_join_count(plan, *input)?;
    }
    Ok(count)
}

pub(super) fn peel_projection_chain(
    plan: &QueryPlan,
    node_id: NodeId,
    max_projection_count: usize,
) -> Option<(NodeId, usize)> {
    let mut current = node_id;
    let mut projection_count = 0;
    while let Some(projection) = projection_node(plan, current) {
        if projection_count == max_projection_count {
            return None;
        }
        projection_count += 1;
        current = projection.input;
    }
    Some((current, projection_count))
}

pub(super) fn projection_node(plan: &QueryPlan, node_id: NodeId) -> Option<&ProjectionNode> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Projection(projection) => Some(projection),
        _ => None,
    }
}

pub(super) fn reachable_join_count(plan: &QueryPlan) -> usize {
    let mut reachable = vec![false; plan.nodes().len()];
    let mut stack = vec![plan.sink().input()];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            continue;
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        if matches!(node.kind, PlanNode::Join(_)) {
            count += 1;
        }
        stack.extend(node.input_ids().iter().copied());
    }
    count
}
