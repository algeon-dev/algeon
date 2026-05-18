use super::*;

pub(in crate::capability) fn root_join_with_post_root_stages(
    plan: &QueryPlan,
    reason_prefix: &'static str,
) -> Result<(NodeId, Vec<PostRootStage>), CapabilityDecision> {
    let mut current = plan.sink().input();
    let mut stages = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "post_root_stage_cycle",
                [("reason_prefix", reason_prefix.to_string()), ("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "post_root_stage_missing_node",
                [("reason_prefix", reason_prefix.to_string()), ("node_id", current.to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Join(_) => {
                stages.reverse();
                return Ok((current, stages));
            },
            PlanNode::Filter(filter) => {
                stages.push(PostRootStage::Filter(current));
                current = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(PostRootStage::Projection(current));
                current = projection.input;
            },
            PlanNode::Aggregate(aggregate) => {
                stages.push(PostRootStage::Aggregate(current));
                current = aggregate.input;
            },
            PlanNode::Window(window) => {
                stages.push(PostRootStage::Window(current));
                current = window.input;
            },
            PlanNode::Sort(sort) => {
                stages.push(PostRootStage::Sort(current));
                current = sort.input;
            },
            PlanNode::Limit(limit) => {
                stages.push(PostRootStage::Limit(current));
                current = limit.input;
            },
            PlanNode::Source(_) => {
                return Err(CapabilityDecision::rejected(
                    "post_root_stage_requires_root_join",
                    [("reason_prefix", reason_prefix.to_string()), ("node_id", current.to_string())],
                ));
            },
            PlanNode::Union(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => {
                return Err(CapabilityDecision::rejected(
                    "post_root_stage_unsupported_node",
                    [
                        ("reason_prefix", reason_prefix.to_string()),
                        ("node_id", current.to_string()),
                        ("operator", node_operator_fact(&node.kind).to_string()),
                    ],
                ));
            },
        }
    }
}

pub(in crate::capability) fn root_join_shape_decision(plan: &QueryPlan) -> Result<(), CapabilityDecision> {
    let root_join_id = plan.sink().input();
    let root_join = generic_join_node(plan, root_join_id, "root_join_admission_requires_sink_root_join")?;
    let reachable_join_count = reachable_join_count_from_sink(plan)?;
    if reachable_join_count != 1 {
        return Err(CapabilityDecision::rejected(
            "root_join_admission_requires_single_reachable_join",
            [("reachable_join_count", reachable_join_count.to_string())],
        ));
    }
    validate_resident_source_filter_projection_leaf_pipeline_for_capability(
        plan,
        root_join.inputs[join_side_index(JoinSide::Left)],
    )?;
    validate_resident_source_filter_projection_leaf_pipeline_for_capability(
        plan,
        root_join.inputs[join_side_index(JoinSide::Right)],
    )?;
    Ok(())
}

pub(in crate::capability) fn validate_general_post_root_stages(
    plan: &QueryPlan,
    stages: &[PostRootStage],
    reason_prefix: &'static str,
) -> Result<(), CapabilityDecision> {
    for stage in stages {
        if let PostRootStage::Limit(limit_id) = *stage {
            let Some(PlanNode::Limit(limit)) = plan.node(limit_id).map(|node| &node.kind) else {
                return Err(CapabilityDecision::rejected(
                    "post_root_limit_missing_node",
                    [("reason_prefix", reason_prefix.to_string()), ("node_id", limit_id.to_string())],
                ));
            };
            if !matches!(plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
                return Err(CapabilityDecision::rejected(
                    "post_root_limit_requires_sort_input",
                    [("reason_prefix", reason_prefix.to_string()), ("node_id", limit_id.to_string())],
                ));
            }
        }
    }
    Ok(())
}

pub(in crate::capability) fn validate_resident_source_filter_projection_leaf_pipeline_for_capability(
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
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => {
            validate_resident_source_filter_projection_leaf_pipeline_for_capability(plan, filter.input)
        },
        PlanNode::Projection(projection) => {
            validate_resident_source_filter_projection_leaf_pipeline_for_capability(plan, projection.input)
        },
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Err(CapabilityDecision::rejected(
            "source_leaf_pipeline_requires_source_filter_projection",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

pub(in crate::capability) fn validate_optional_projection_then_sort_post_root_stages(
    plan: &QueryPlan,
    stages: &[PostRootStage],
) -> Result<(Option<NodeId>, Option<NodeId>), CapabilityDecision> {
    let mut seen_projection = None;
    let mut seen_sort = None;
    for stage in stages {
        match *stage {
            PostRootStage::Projection(node_id) if seen_projection.is_none() && seen_sort.is_none() => {
                seen_projection = Some(node_id);
            },
            PostRootStage::Sort(node_id) if seen_sort.is_none() => {
                seen_sort = Some(node_id);
            },
            PostRootStage::Filter(node_id)
            | PostRootStage::Projection(node_id)
            | PostRootStage::Aggregate(node_id)
            | PostRootStage::Window(node_id)
            | PostRootStage::Sort(node_id)
            | PostRootStage::Limit(node_id) => {
                return Err(CapabilityDecision::rejected(
                    "keyless_aggregate_subplan_inputs_unsupported_post_root_stage",
                    [("node_id", node_id.to_string()), ("operator", node_kind_for_fact(plan, node_id))],
                ));
            },
        }
    }
    Ok((seen_projection, seen_sort))
}
