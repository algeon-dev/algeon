use std::collections::BTreeMap;

use arrow::datatypes::SchemaRef;

use crate::{
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode,
        QueryPlan, SortNode,
    },
    runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy},
};

pub(crate) enum BoundedJoinPlanSupport {
    Root,
    UnaryPipeline,
    MultiJoinPipeline,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedConditionalJoinType {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
}

impl BoundedConditionalJoinType {
    pub(crate) fn try_from_join(join: &JoinNode) -> Option<Self> {
        if !join.spec.keys.is_empty()
            || join.spec.residual_predicate.is_none()
            || join.spec.output_projection.is_empty()
        {
            return None;
        }

        match join.spec.join_type {
            JoinType::Inner => Some(Self::Inner),
            JoinType::Left if join.spec.build_side == JoinSide::Right => Some(Self::Left),
            JoinType::Full => Some(Self::Full),
            JoinType::LeftSemi if join.spec.build_side == JoinSide::Left => Some(Self::LeftSemi),
            JoinType::LeftAnti if join.spec.build_side == JoinSide::Left => Some(Self::LeftAnti),
            JoinType::Left | JoinType::Cross => None,
            JoinType::LeftSemi | JoinType::LeftAnti => None,
        }
    }
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
    UnsupportedLeaf { node_id: NodeId, node_kind: &'static str },
    LimitWithoutSort,
    MissingRootJoin,
    MultipleReachableJoins,
    SingleReachableJoin,
}

pub(crate) fn analyze_bounded_join_plan_support(plan: &QueryPlan) -> BoundedJoinPlanSupport {
    if let Some(node) = plan.node(plan.sink().input())
        && let PlanNode::Join(join) = &node.kind
        && bounded_root_join_compatible(join)
        && reachable_join_count(plan) == 1
        && bounded_join_source_leaf_inputs_compatible(plan, join)
    {
        return BoundedJoinPlanSupport::Root;
    }

    match analyze_bounded_join_pipeline(plan) {
        Ok(_) => BoundedJoinPlanSupport::UnaryPipeline,
        Err(_) => match analyze_bounded_multi_join_pipeline(plan) {
            Ok(_) => BoundedJoinPlanSupport::MultiJoinPipeline,
            Err(_) => BoundedJoinPlanSupport::Unsupported,
        },
    }
}

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    if let Some(join) = first_keyed_residual_semi_anti_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_only_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left_semi_left_anti".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_only_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_only_join_shape_or_orientation".to_string(),
        );
    } else if let Some(join) = first_keyed_residual_full_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_full_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_full".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_full_shape_contract(plan, join));
        facts.insert("bounded_join_shape_blocker", "bounded_mixed_keyed_residual_full_join_shape".to_string());
    } else if let Some(join) = first_keyed_residual_left_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_join_shape_or_orientation".to_string(),
        );
    }
    facts
}

pub(crate) fn analyze_bounded_join_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    analyze_bounded_join_pipeline_shape(plan, true)
}

pub(crate) fn analyze_bounded_multi_join_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let shape = analyze_bounded_join_pipeline_shape(plan, false)?;
    let join_count = bounded_join_subtree_join_count(plan, shape.join_id)?;
    if join_count <= 1 {
        return Err(BoundedJoinPipelineRejection::SingleReachableJoin);
    }
    validate_bounded_multi_join_subtree(plan, shape.join_id)?;
    Ok(shape)
}

pub(crate) fn bounded_root_hash_join_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

pub(crate) fn bounded_root_filtered_join_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Right
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(crate) fn bounded_root_conditional_join_compatible(join: &JoinNode) -> bool {
    BoundedConditionalJoinType::try_from_join(join).is_some()
}

pub(crate) fn bounded_root_projected_cross_join_compatible(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Cross
}

pub(crate) fn bounded_root_mixed_join_compatible(join: &JoinNode) -> bool {
    if join.spec.keys.is_empty() || join.spec.residual_predicate.is_none() || join.spec.output_projection.is_empty() {
        return false;
    }

    match join.spec.join_type {
        JoinType::Inner | JoinType::Full => true,
        JoinType::Left => join.spec.build_side == JoinSide::Right,
        JoinType::LeftSemi | JoinType::LeftAnti => {
            join.spec.build_side == JoinSide::Right
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        },
        JoinType::Cross => false,
    }
}

pub(crate) fn bounded_root_join_compatible(join: &JoinNode) -> bool {
    bounded_root_hash_join_compatible(join)
        || bounded_root_filtered_join_compatible(join)
        || bounded_root_conditional_join_compatible(join)
        || bounded_root_projected_cross_join_compatible(join)
        || bounded_root_mixed_join_compatible(join)
}

pub(crate) fn bounded_multi_join_compatible(join: &JoinNode) -> bool {
    bounded_root_hash_join_compatible(join)
        || bounded_root_filtered_join_compatible(join)
        || bounded_multi_join_mixed_inner_compatible(join)
}

pub(crate) fn bounded_multi_join_mixed_inner_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Inner
}

fn analyze_bounded_join_pipeline_shape(
    plan: &QueryPlan,
    require_single_reachable_join: bool,
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

fn bounded_join_source_leaf_inputs_compatible(plan: &QueryPlan, join: &JoinNode) -> bool {
    validate_bounded_join_source_leaf_inputs(plan, join).is_ok()
}

fn validate_bounded_join_source_leaf_inputs<'a>(
    plan: &'a QueryPlan,
    join: &'a JoinNode,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    validate_bounded_join_source_leaf_input(plan, join.inputs[0])?;
    validate_bounded_join_source_leaf_input(plan, join.inputs[1])
}

fn validate_bounded_join_source_leaf_input(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded join input references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_bounded_join_source_leaf_input(plan, filter.input),
        PlanNode::Projection(projection) => validate_bounded_join_source_leaf_input(plan, projection.input),
        PlanNode::Join(_) => Err(BoundedJoinPipelineRejection::MultipleReachableJoins),
        PlanNode::Aggregate(_) => {
            Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "limit" }),
    }
}

fn validate_bounded_multi_join_subtree(
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
            Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind: "limit" }),
    }
}

fn bounded_join_subtree_join_count(
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

fn first_keyed_residual_semi_anti_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty()
            && join.spec.residual_predicate.is_some()
            && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti))
        .then_some(join)
    })
}

fn first_keyed_residual_full_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Full)
            .then_some(join)
    })
}

fn first_keyed_residual_left_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Left)
            .then_some(join)
    })
}

fn mixed_keyed_residual_left_only_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_semi_anti_topology_unsupported"
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_semi_anti_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_semi_anti_requires_left_projection"
    } else if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
        "root_mixed_keyed_residual_semi_anti_requires_left_only_projection"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_semi_anti_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_only_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_joins".to_string()
    } else if join.spec.build_side != JoinSide::Right
        || join.spec.output_projection.is_empty()
        || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        "root_mixed_keyed_residual_left_only_join_requires_right_build_left_projection".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_full_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_full_topology_unsupported"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_full_shape_unsupported"
    }
}

fn mixed_keyed_residual_full_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_joins".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_left_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_left_topology_unsupported"
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_left_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_joins".to_string()
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_join_requires_right_build".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn root_join_leaf_blocker(plan: &QueryPlan, join: &JoinNode) -> Option<&'static str> {
    join.inputs.iter().find_map(|input| root_join_input_leaf_blocker(plan, *input))
}

fn root_join_input_leaf_blocker(plan: &QueryPlan, node_id: NodeId) -> Option<&'static str> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(_) => None,
        PlanNode::Filter(filter) => root_join_input_leaf_blocker(plan, filter.input),
        PlanNode::Projection(projection) => root_join_input_leaf_blocker(plan, projection.input),
        PlanNode::Join(_) => Some("bounded_root_join_nested_join_input_topology_unsupported"),
        PlanNode::Aggregate(_) => Some("bounded_root_join_aggregate_leaf_unsupported"),
        PlanNode::Sort(_) => Some("bounded_root_join_sort_leaf_unsupported"),
        PlanNode::Limit(_) => Some("bounded_root_join_limit_leaf_unsupported"),
    }
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
