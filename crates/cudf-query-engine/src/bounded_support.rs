use arrow::datatypes::SchemaRef;

use crate::plan::{
    AggregateNode, FilterNode, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode, QueryPlan,
    SortNode,
};

pub(crate) enum BoundedJoinPlanSupport {
    Root,
    UnaryPipeline,
    Unsupported,
}

pub(crate) struct BoundedJoinPipelineShape<'a> {
    pub(crate) join_id: NodeId,
    pub(crate) join: &'a JoinNode,
    pub(crate) stages: Vec<BoundedJoinPipelineStage<'a>>,
}

pub(crate) enum BoundedJoinPipelineStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode, output_schema: &'a SchemaRef },
    Aggregate { node_id: NodeId, node: &'a AggregateNode, output_schema: &'a SchemaRef },
    Sort { node_id: NodeId, node: &'a SortNode },
    Limit { node_id: NodeId, node: &'a LimitNode },
}

pub(crate) enum BoundedJoinPipelineRejection<'a> {
    MissingNode { node_id: NodeId, message: &'static str },
    UnsupportedJoin { join: &'a JoinNode },
    LimitWithoutSort,
    MissingRootJoin,
    MultipleReachableJoins,
}

pub(crate) fn analyze_bounded_join_plan_support(plan: &QueryPlan) -> BoundedJoinPlanSupport {
    if let Some(node) = plan.node(plan.sink().input())
        && let PlanNode::Join(join) = &node.kind
        && bounded_root_hash_join_compatible(join)
    {
        return BoundedJoinPlanSupport::Root;
    }

    match analyze_bounded_join_pipeline(plan) {
        Ok(_) => BoundedJoinPlanSupport::UnaryPipeline,
        Err(_) => BoundedJoinPlanSupport::Unsupported,
    }
}

pub(crate) fn analyze_bounded_join_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let mut current = plan.sink().input();
    let mut stages = Vec::new();
    let mut saw_aggregate = false;

    loop {
        let node = plan.node(current).ok_or(BoundedJoinPipelineRejection::MissingNode {
            node_id: current,
            message: "bounded join pipeline references a missing plan node",
        })?;
        match &node.kind {
            PlanNode::Join(join) => {
                if !bounded_root_hash_join_compatible(join) {
                    return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
                }
                if reachable_join_count(plan) != 1 {
                    return Err(BoundedJoinPipelineRejection::MultipleReachableJoins);
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
            PlanNode::Aggregate(aggregate) if !saw_aggregate => {
                saw_aggregate = true;
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
                    return Err(BoundedJoinPipelineRejection::LimitWithoutSort);
                }
                stages.push(BoundedJoinPipelineStage::Limit { node_id: current, node: limit });
                current = limit.input;
            },
            PlanNode::Source(_) | PlanNode::Aggregate(_) => return Err(BoundedJoinPipelineRejection::MissingRootJoin),
        }
    }
}

pub(crate) fn bounded_root_hash_join_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

fn reachable_join_count(plan: &QueryPlan) -> usize {
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
