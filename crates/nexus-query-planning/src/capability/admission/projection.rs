use std::collections::{BTreeMap, BTreeSet};

use super::{
    CapabilityDecision, EdgeContract, InvariantSelector, JoinExecutionCapability, JoinNode, JoinSide, JoinType,
    NodeCapability, NodeId, PlanNode, ProjectionScalarSetopKind, ProjectionScalarSetopShape, QueryPlan, SourceNode,
    SubgraphInvariant, collect_reachable_node_ids, consumer_nodes_for_reachable, first_reachable_edge_rejection,
    first_reachable_node_rejection, join_node_execution_capability, join_node_with_missing_reason, join_side_index,
    node_ids_fact, projection_global_limit_shape_decision, projection_scalar_setop_shape_decision,
    solved_invariant_decision,
};

pub(in crate::capability) fn analyze_projection_global_limit_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match projection_global_limit_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for join_id in &shape.join_ids {
        let join = match join_node_with_missing_reason(
            plan,
            *join_id,
            "projection_global_limit_missing_node",
            "projection_global_limit_requires_supported_join_fragment",
        ) {
            Ok(join) => join,
            Err(decision) => return decision,
        };
        if join_node_execution_capability(join).is_none() {
            return CapabilityDecision::rejected(
                "projection_global_limit_requires_supported_join_fragment",
                [
                    ("node_id", join_id.to_string()),
                    ("join_type", join.spec.join_type.to_string()),
                    ("key_count", join.spec.keys.len().to_string()),
                    ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ],
            );
        }
    }

    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    if let Some(post_limit_projection_id) = shape.post_limit_projection_id {
        required.push(InvariantSelector::LimitBoundedProjection {
            limit_id: shape.global_limit_id,
            projection_id: post_limit_projection_id,
        });
    }
    if shape.join_ids.len() > 1 {
        required.extend([
            InvariantSelector::TopologicalBuildProbeSchedule,
            InvariantSelector::ResidentLifetimeDominatesConsumers,
        ]);
    }

    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_projection_global_limit_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }

    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_projection_scalar_setop_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    _invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let shape = match projection_scalar_setop_shape_decision(plan) {
        Ok(shape) => shape,
        Err(CapabilityDecision::Rejected {
            reason: "projection_scalar_setop_rejects_distinct_aggregate",
            mut facts,
        }) => {
            facts.insert("blocked_reason", "projection_scalar_setop_rejects_distinct_aggregate".to_string());
            facts.insert("required_remediation", "projection_scalar_setop_cardinality_contract".to_string());
            return CapabilityDecision::Rejected { reason: "projection_scalar_setop_contract_not_implemented", facts };
        },
        Err(decision) => return decision,
    };
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    if projection_scalar_setop_scalar_fanout_has_cardinality_proof(plan, &shape)
        || projection_scalar_setop_setop_dedup_scalar_has_contract(plan, &shape)
        || projection_scalar_setop_scalar_join_aggregate_input_has_contract(plan, &shape)
        || projection_scalar_setop_scalar_join_grouped_aggregate_input_has_contract(plan, &shape)
    {
        return CapabilityDecision::Satisfied;
    }
    CapabilityDecision::rejected(
        "projection_scalar_setop_contract_not_implemented",
        [
            ("final_projection_id", shape.final_projection_id.to_string()),
            ("join_node_ids", node_ids_fact(&shape.join_ids)),
            ("aggregate_node_ids", node_ids_fact(&shape.aggregate_ids)),
            ("shape_kind", shape.kind.as_str().to_string()),
            ("required_remediation", "projection_scalar_setop_cardinality_contract".to_string()),
        ],
    )
}

pub(in crate::capability::admission) fn projection_scalar_setop_scalar_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarFanout {
        return false;
    }
    if !shape.aggregate_ids.iter().all(|aggregate_id| {
        matches!(
            plan.node(*aggregate_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
        )
    }) {
        return false;
    }
    projection_scalar_setop_node_output_at_most_one_row(plan, shape.final_projection_id, &mut BTreeSet::new())
}

pub(in crate::capability::admission) fn projection_scalar_setop_node_output_at_most_one_row(
    plan: &QueryPlan,
    node_id: NodeId,
    visiting: &mut BTreeSet<NodeId>,
) -> bool {
    if !visiting.insert(node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        visiting.remove(&node_id);
        return false;
    };
    let proven = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                source.scan_limit.is_some_and(|limit| limit <= 1) || source.max_rows.is_some_and(|rows| rows <= 1)
            },
            SourceNode::LiteralBatch(source) => source.row_count() <= 1,
            SourceNode::GpuResident(_) => false,
        },
        PlanNode::Projection(projection) => {
            projection_scalar_setop_node_output_at_most_one_row(plan, projection.input, visiting)
        },
        PlanNode::Filter(filter) => projection_scalar_setop_node_output_at_most_one_row(plan, filter.input, visiting),
        PlanNode::Aggregate(aggregate) => aggregate.spec.group_exprs.is_empty(),
        PlanNode::Limit(limit) => {
            limit.spec.fetch.is_some_and(|fetch| fetch <= 1)
                || projection_scalar_setop_node_output_at_most_one_row(plan, limit.input, visiting)
        },
        PlanNode::Join(join) => projection_scalar_setop_join_output_at_most_one_row(plan, join, visiting),
        PlanNode::Sort(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => false,
    };
    visiting.remove(&node_id);
    proven
}

pub(in crate::capability::admission) fn projection_scalar_setop_join_output_at_most_one_row(
    plan: &QueryPlan,
    join: &JoinNode,
    visiting: &mut BTreeSet<NodeId>,
) -> bool {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Cross) {
        return false;
    }
    let mut left_visiting = visiting.clone();
    let mut right_visiting = visiting.clone();
    projection_scalar_setop_node_output_at_most_one_row(
        plan,
        join.inputs[join_side_index(JoinSide::Left)],
        &mut left_visiting,
    ) && projection_scalar_setop_node_output_at_most_one_row(
        plan,
        join.inputs[join_side_index(JoinSide::Right)],
        &mut right_visiting,
    )
}

pub(in crate::capability::admission) fn projection_scalar_setop_setop_dedup_scalar_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::SetopDedupScalar {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };

    let mut saw_grouped_dedup_aggregate = false;
    let mut grouped_dedup_aggregate_ids = BTreeSet::new();
    for aggregate_id in &shape.aggregate_ids {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*aggregate_id).map(|node| &node.kind) else {
            return false;
        };
        if *aggregate_id == final_scalar_aggregate_id {
            if !aggregate.spec.group_exprs.is_empty() {
                return false;
            }
            continue;
        }
        if aggregate.spec.grouping_sets.is_some()
            || !aggregate.spec.aggr_exprs.is_empty()
            || aggregate.spec.group_exprs.is_empty()
        {
            return false;
        }
        saw_grouped_dedup_aggregate = true;
        grouped_dedup_aggregate_ids.insert(*aggregate_id);
    }

    let mut saw_left_anti_join = false;
    for join_id in &shape.join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return false;
        };
        match join.spec.join_type {
            JoinType::Inner => {
                if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeyedHash)) {
                    return false;
                }
            },
            JoinType::LeftAnti => {
                if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeyedFilteredSemiAnti))
                {
                    return false;
                }
                let [left, right] = join.inputs;
                if !grouped_dedup_aggregate_ids.contains(&left) || !grouped_dedup_aggregate_ids.contains(&right) {
                    return false;
                }
                saw_left_anti_join = true;
            },
            JoinType::Left | JoinType::Full | JoinType::LeftSemi | JoinType::Cross | JoinType::LeftMark => {
                return false;
            },
        }
    }

    saw_grouped_dedup_aggregate && saw_left_anti_join
}

pub(in crate::capability::admission) fn projection_scalar_setop_scalar_join_aggregate_input_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarJoinAggregateInput {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };
    if shape.aggregate_ids.as_slice() != [final_scalar_aggregate_id] {
        return false;
    }
    shape.join_ids.iter().all(|join_id| {
        matches!(
            plan.node(*join_id).map(|node| &node.kind),
            Some(PlanNode::Join(join))
                if matches!(
                    join_node_execution_capability(join),
                    Some(JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                )
        )
    })
}

pub(in crate::capability::admission) fn projection_scalar_setop_scalar_join_grouped_aggregate_input_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarJoinGroupedAggregateInput {
        return false;
    }
    if shape.join_ids.len() != 1 {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };
    let grouped_aggregate_ids = shape
        .aggregate_ids
        .iter()
        .copied()
        .filter(|aggregate_id| *aggregate_id != final_scalar_aggregate_id)
        .collect::<Vec<_>>();
    if grouped_aggregate_ids.is_empty() {
        return false;
    }
    if !grouped_aggregate_ids.iter().all(|aggregate_id| {
        matches!(
            plan.node(*aggregate_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate))
                if !aggregate.spec.group_exprs.is_empty()
                    && aggregate.spec.grouping_sets.is_none()
                    && !aggregate.spec.aggr_exprs.is_empty()
        )
    }) {
        return false;
    }
    if !grouped_aggregate_ids
        .iter()
        .all(|aggregate_id| projection_scalar_setop_grouped_aggregate_feeds_join(plan, *aggregate_id, &shape.join_ids))
    {
        return false;
    }
    shape.join_ids.iter().all(|join_id| {
        matches!(
            plan.node(*join_id).map(|node| &node.kind),
            Some(PlanNode::Join(join))
                if matches!(
                    join_node_execution_capability(join),
                    Some(
                        JoinExecutionCapability::KeyedHash
                            | JoinExecutionCapability::KeyedResidualMixed
                            | JoinExecutionCapability::KeyedFilteredSemiAnti
                    )
                )
        )
    })
}

pub(in crate::capability::admission) fn projection_scalar_setop_grouped_aggregate_feeds_join(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    join_ids: &[NodeId],
) -> bool {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return false;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    consumers.get(&aggregate_id).is_some_and(|consumer_ids| {
        consumer_ids.iter().any(|consumer_id| {
            projection_scalar_setop_consumer_chain_reaches_join(
                plan,
                &consumers,
                *consumer_id,
                join_ids,
                &mut BTreeSet::new(),
            )
        })
    })
}

pub(in crate::capability::admission) fn projection_scalar_setop_consumer_chain_reaches_join(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    join_ids: &[NodeId],
    visited: &mut BTreeSet<NodeId>,
) -> bool {
    if !visited.insert(node_id) {
        return false;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => join_ids.contains(&node_id),
        Some(PlanNode::Projection(_) | PlanNode::Filter(_)) => consumers.get(&node_id).is_some_and(|next| {
            next.iter().any(|consumer_id| {
                projection_scalar_setop_consumer_chain_reaches_join(plan, consumers, *consumer_id, join_ids, visited)
            })
        }),
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Source(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => false,
    }
}

pub(in crate::capability::admission) fn projection_scalar_setop_final_projection_input_scalar_aggregate_id(
    plan: &QueryPlan,
    final_projection_id: NodeId,
) -> Option<NodeId> {
    let mut current = match plan.node(final_projection_id).map(|node| &node.kind)? {
        PlanNode::Projection(projection) => projection.input,
        _ => return None,
    };
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        match &node.kind {
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => return Some(current),
            PlanNode::Aggregate(_)
            | PlanNode::Source(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Join(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => return None,
        }
    }
}
