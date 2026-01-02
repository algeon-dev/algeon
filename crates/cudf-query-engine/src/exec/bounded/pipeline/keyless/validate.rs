use super::*;

pub(super) fn validate_bounded_keyless_conditional_join_input_subplan_capabilities(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(plan, node_id, false)
}

fn validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    join_reached_through_projection: bool,
) -> Result<()> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join input-subplan pipeline references a missing node",
        )
    })?;
    match &node.kind {
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(_) => validate_bounded_join_leaf_input_capabilities(plan, node_id),
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(plan, projection.input, true)
        },
        PlanNode::Aggregate(aggregate) => {
            if !aggregate.spec.group_exprs.is_empty() {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution supports only scalar/global aggregate inputs",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("node_id", node_id)
                .with_fact("node_kind", "grouped_aggregate"));
            }
            validate_aggregate_capability(aggregate, &node.facts.output_schema)?;
            validate_bounded_join_leaf_input_capabilities(plan, aggregate.input)
        },
        PlanNode::Join(join) => {
            if !join_reached_through_projection {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution requires left anti hash-join subplans to be projection-wrapped",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("node_id", node_id)
                .with_fact("node_kind", "bare_join")
                .with_fact("join_type", join.spec.join_type));
            }
            validate_executable_join_subset(join)?;
            if !(join.spec.join_type == JoinType::LeftAnti && join.is_bounded_root_filtered_join_compatible()) {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution supports only left anti hash-join subplans",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("join_type", join.spec.join_type)
                .with_fact("build_side", join.spec.build_side)
                .with_fact("key_count", join.spec.keys.len())
                .with_fact("residual_predicate", join.spec.residual_predicate.is_some())
                .with_fact("output_projection_len", join.spec.output_projection.len()));
            }
            validate_bounded_join_leaf_input_capabilities(plan, join.inputs[0])?;
            validate_bounded_join_leaf_input_capabilities(plan, join.inputs[1])
        },
        PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded keyless conditional join input-subplan pipeline execution does not support sort or limit below root join inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&node.kind))),
    }
}

pub(super) fn validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    kind: AggregateOverJoinSubplanInputKind,
) -> Result<()> {
    let projection_node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline references a missing node",
        )
    })?;
    let PlanNode::Projection(projection) = &projection_node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires projection-wrapped aggregate inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&projection_node.kind)));
    };
    validate_projection_capability(projection, &projection_node.facts.output_schema)?;

    let aggregate_node = plan.node(projection.input).ok_or_else(|| {
        invalid_node_reference(
            projection.input,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline projection references a missing aggregate",
        )
    })?;
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires projection-wrapped aggregate inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", projection.input)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&aggregate_node.kind)));
    };
    match kind {
        AggregateOverJoinSubplanInputKind::Grouped if aggregate.spec.group_exprs.is_empty() => {
            return Err(unsupported_bounded_shape(
                "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires grouped aggregate on the left input",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", projection.input)
            .with_fact("node_kind", "scalar_aggregate"));
        },
        AggregateOverJoinSubplanInputKind::Scalar if !aggregate.spec.group_exprs.is_empty() => {
            return Err(unsupported_bounded_shape(
                "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires scalar/global aggregate on the right input",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", projection.input)
            .with_fact("node_kind", "grouped_aggregate"));
        },
        AggregateOverJoinSubplanInputKind::Grouped | AggregateOverJoinSubplanInputKind::Scalar => {},
    }
    validate_aggregate_capability(aggregate, &aggregate_node.facts.output_schema)?;
    validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(context, plan, aggregate.input)
}

fn validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline aggregate references a missing join subtree",
        )
    })?;
    match &node.kind {
        PlanNode::Join(_) => validate_bounded_multi_join_subtree_capabilities(context, plan, node_id),
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(
                context,
                plan,
                projection.input,
            )
        },
        _ => Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires aggregate inputs rooted at joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&node.kind))),
    }
}

pub(super) fn bounded_keyless_conditional_join_input_subplan_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}
