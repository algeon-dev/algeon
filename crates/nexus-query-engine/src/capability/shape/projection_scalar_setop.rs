use super::*;

pub(in crate::capability) fn projection_scalar_setop_shape_decision(
    plan: &QueryPlan,
) -> Result<ProjectionScalarSetopShape, CapabilityDecision> {
    let final_projection_id = plan.sink().input();
    let Some(final_projection_node) = plan.node(final_projection_id) else {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_missing_node",
            [("node_id", final_projection_id.to_string()), ("role", "final_projection".to_string())],
        ));
    };
    let final_projection = match &final_projection_node.kind {
        PlanNode::Projection(final_projection) => Some(final_projection),
        PlanNode::Aggregate(aggregate) => {
            let reachable = collect_reachable_node_ids(plan)?;
            let components = projection_scalar_setop_components(plan, &reachable, final_projection_id)?;
            let mut aggregate_ids = components.aggregate_ids.clone();
            if !aggregate_ids.contains(&final_projection_id) {
                aggregate_ids.push(final_projection_id);
            }
            let kind = if aggregate.spec.group_exprs.is_empty() {
                projection_scalar_setop_input_kind(final_projection_id, &components)?
            } else if aggregate.spec.aggr_exprs.is_empty() {
                ProjectionScalarSetopKind::SetopDedupInput
            } else {
                return Err(CapabilityDecision::rejected(
                    "projection_scalar_setop_requires_scalar_or_setop_contract",
                    [("final_projection_id", final_projection_id.to_string())],
                ));
            };
            return Ok(ProjectionScalarSetopShape {
                final_projection_id,
                join_ids: components.join_ids,
                aggregate_ids,
                kind,
            });
        },
        PlanNode::Join(_) => {
            let reachable = collect_reachable_node_ids(plan)?;
            let components = projection_scalar_setop_components(plan, &reachable, final_projection_id)?;
            let kind = projection_scalar_setop_input_kind(final_projection_id, &components)?;
            return Ok(ProjectionScalarSetopShape {
                final_projection_id,
                join_ids: components.join_ids,
                aggregate_ids: components.aggregate_ids,
                kind,
            });
        },
        _ => {
            return Err(CapabilityDecision::rejected(
                "projection_scalar_setop_requires_sink_projection",
                [
                    ("node_id", final_projection_id.to_string()),
                    ("operator", node_operator_fact(&final_projection_node.kind).to_string()),
                ],
            ));
        },
    };

    let Some(final_projection) = final_projection else {
        unreachable!("aggregate sink returned above");
    };

    if matches!(plan.node(final_projection.input).map(|node| &node.kind), Some(PlanNode::Limit(_))) {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_defers_limit_projection",
            [("projection_input", final_projection.input.to_string())],
        ));
    }

    let reachable = collect_reachable_node_ids(plan)?;
    let components = projection_scalar_setop_components(plan, &reachable, final_projection_id)?;
    let ProjectionScalarSetopComponents {
        join_ids,
        aggregate_ids,
        grouped_dedup_aggregate_ids,
        keyless_or_cross_join_ids,
        left_anti_join_ids,
        distinct_aggregate_ids,
    } = components;

    if join_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_requires_join_fragment",
            [("final_projection_id", final_projection_id.to_string())],
        ));
    }
    if aggregate_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_requires_aggregate_fragment",
            [("final_projection_id", final_projection_id.to_string())],
        ));
    }
    if !distinct_aggregate_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_rejects_distinct_aggregate",
            [
                ("final_projection_id", final_projection_id.to_string()),
                ("distinct_aggregate_ids", node_ids_fact(&distinct_aggregate_ids)),
                ("required_remediation", "distinct_aggregate_contract".to_string()),
            ],
        ));
    }

    let scalar_aggregate_ids = aggregate_ids
        .iter()
        .copied()
        .filter(|aggregate_id| {
            matches!(
                plan.node(*aggregate_id).map(|node| &node.kind),
                Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
            )
        })
        .collect::<Vec<_>>();
    let kind = if grouped_dedup_aggregate_ids.is_empty()
        && scalar_aggregate_ids.len() == aggregate_ids.len()
        && !keyless_or_cross_join_ids.is_empty()
    {
        ProjectionScalarSetopKind::ScalarFanout
    } else if !left_anti_join_ids.is_empty()
        && !grouped_dedup_aggregate_ids.is_empty()
        && projection_scalar_setop_has_final_scalar_aggregate(plan, final_projection.input)
    {
        ProjectionScalarSetopKind::SetopDedupScalar
    } else if projection_scalar_setop_has_final_scalar_aggregate(plan, final_projection.input)
        && projection_scalar_setop_has_grouped_aggregate_input(plan, &aggregate_ids, &scalar_aggregate_ids)
        && !join_ids.is_empty()
    {
        ProjectionScalarSetopKind::ScalarJoinGroupedAggregateInput
    } else if scalar_aggregate_ids.len() == aggregate_ids.len() && join_ids.len() > 1 {
        ProjectionScalarSetopKind::ScalarJoinAggregateInput
    } else if !left_anti_join_ids.is_empty() && !grouped_dedup_aggregate_ids.is_empty() {
        ProjectionScalarSetopKind::SetopDedupInput
    } else if !keyless_or_cross_join_ids.is_empty() {
        ProjectionScalarSetopKind::KeylessOrCrossJoinInput
    } else {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_requires_scalar_or_setop_contract",
            [
                ("final_projection_id", final_projection_id.to_string()),
                ("scalar_aggregate_ids", node_ids_fact(&scalar_aggregate_ids)),
                ("grouped_dedup_aggregate_ids", node_ids_fact(&grouped_dedup_aggregate_ids)),
                ("keyless_or_cross_join_ids", node_ids_fact(&keyless_or_cross_join_ids)),
                ("left_anti_join_ids", node_ids_fact(&left_anti_join_ids)),
            ],
        ));
    };

    Ok(ProjectionScalarSetopShape { final_projection_id, join_ids, aggregate_ids, kind })
}

fn projection_scalar_setop_input_kind(
    final_projection_id: NodeId,
    components: &ProjectionScalarSetopComponents,
) -> Result<ProjectionScalarSetopKind, CapabilityDecision> {
    if components.join_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_requires_join_fragment",
            [("final_projection_id", final_projection_id.to_string())],
        ));
    }
    if !components.left_anti_join_ids.is_empty() && !components.grouped_dedup_aggregate_ids.is_empty() {
        return Ok(ProjectionScalarSetopKind::SetopDedupInput);
    }
    if !components.keyless_or_cross_join_ids.is_empty() {
        return Ok(ProjectionScalarSetopKind::KeylessOrCrossJoinInput);
    }
    Err(CapabilityDecision::rejected(
        "projection_scalar_setop_requires_scalar_or_setop_contract",
        [
            ("join_node_ids", node_ids_fact(&components.join_ids)),
            ("aggregate_node_ids", node_ids_fact(&components.aggregate_ids)),
            ("grouped_dedup_aggregate_ids", node_ids_fact(&components.grouped_dedup_aggregate_ids)),
            ("keyless_or_cross_join_ids", node_ids_fact(&components.keyless_or_cross_join_ids)),
            ("left_anti_join_ids", node_ids_fact(&components.left_anti_join_ids)),
        ],
    ))
}

struct ProjectionScalarSetopComponents {
    join_ids: Vec<NodeId>,
    aggregate_ids: Vec<NodeId>,
    grouped_dedup_aggregate_ids: Vec<NodeId>,
    keyless_or_cross_join_ids: Vec<NodeId>,
    left_anti_join_ids: Vec<NodeId>,
    distinct_aggregate_ids: Vec<NodeId>,
}

fn projection_scalar_setop_components(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
    final_projection_id: NodeId,
) -> Result<ProjectionScalarSetopComponents, CapabilityDecision> {
    let mut join_ids = Vec::new();
    let mut aggregate_ids = Vec::new();
    let mut grouped_dedup_aggregate_ids = Vec::new();
    let mut keyless_or_cross_join_ids = Vec::new();
    let mut left_anti_join_ids = Vec::new();
    let mut distinct_aggregate_ids = Vec::new();

    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            return Err(CapabilityDecision::rejected(
                "projection_scalar_setop_missing_node",
                [("node_id", node_id.to_string()), ("role", "reachable".to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Join(join) => {
                join_ids.push(*node_id);
                if join.spec.keys.is_empty() || join.spec.join_type == JoinType::Cross {
                    keyless_or_cross_join_ids.push(*node_id);
                }
                if join.spec.join_type == JoinType::LeftAnti {
                    left_anti_join_ids.push(*node_id);
                }
            },
            PlanNode::Aggregate(aggregate) => {
                aggregate_ids.push(*node_id);
                if aggregate.spec.grouping_sets.is_some()
                    || aggregate
                        .spec
                        .aggr_exprs
                        .iter()
                        .any(|expr| expr.distinct || expr.function == AggregateFunction::NUnique)
                {
                    distinct_aggregate_ids.push(*node_id);
                }
                if aggregate.spec.aggr_exprs.is_empty() && !aggregate.spec.group_exprs.is_empty() {
                    grouped_dedup_aggregate_ids.push(*node_id);
                }
            },
            PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Window(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_) => {},
        }
    }

    if aggregate_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "projection_scalar_setop_requires_aggregate_fragment",
            [("final_projection_id", final_projection_id.to_string())],
        ));
    }

    Ok(ProjectionScalarSetopComponents {
        join_ids,
        aggregate_ids,
        grouped_dedup_aggregate_ids,
        keyless_or_cross_join_ids,
        left_anti_join_ids,
        distinct_aggregate_ids,
    })
}

fn projection_scalar_setop_has_final_scalar_aggregate(plan: &QueryPlan, mut current: NodeId) -> bool {
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(node) = plan.node(current) else {
            return false;
        };
        match &node.kind {
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Aggregate(aggregate) => return aggregate.spec.group_exprs.is_empty(),
            _ => return false,
        }
    }
}

fn projection_scalar_setop_has_grouped_aggregate_input(
    plan: &QueryPlan,
    aggregate_ids: &[NodeId],
    scalar_aggregate_ids: &[NodeId],
) -> bool {
    aggregate_ids.iter().any(|aggregate_id| {
        !scalar_aggregate_ids.contains(aggregate_id)
            && matches!(
                plan.node(*aggregate_id).map(|node| &node.kind),
                Some(PlanNode::Aggregate(aggregate))
                    if !aggregate.spec.group_exprs.is_empty()
                        && aggregate.spec.grouping_sets.is_none()
                        && !aggregate.spec.aggr_exprs.is_empty()
            )
    })
}

fn node_ids_fact(node_ids: &[NodeId]) -> String {
    node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}
