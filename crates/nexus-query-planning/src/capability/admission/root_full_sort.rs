use super::*;

pub(in crate::capability) fn analyze_root_full_sort_join_aggregate_admission(
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
    let shape = match root_full_sort_join_aggregate_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };

    if !root_full_sort_subcontract_has_cardinality_proof(plan, &shape) {
        return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
    }
    for join_id in &shape.join_ids {
        let join = match join_node_with_missing_reason(
            plan,
            *join_id,
            "root_full_sort_join_aggregate_missing_node",
            "root_full_sort_join_aggregate_requires_keyed_hash_join",
        ) {
            Ok(join) => join,
            Err(decision) => return decision,
        };
        let Some(capability) = join_node_execution_capability(join) else {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        };
        if !root_full_sort_join_capability_satisfies_subcontract(&shape, *join_id, join, capability) {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        }
    }

    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    for join_id in &shape.join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        };
        if join.spec.residual_predicate.is_some() {
            required.push(InvariantSelector::ResidualUsesBothInputs { join_id: *join_id });
        }
    }
    if shape.aggregate_id != shape.aggregate_projection_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        });
    }
    if matches!(shape.subcontract, RootFullSortJoinAggregateSubcontract::SharedAggregateFanout) {
        required.extend(
            shape
                .shared_aggregate_producer_ids
                .iter()
                .copied()
                .map(|producer| InvariantSelector::SharedResidentSubplan { producer }),
        );
        required.extend(root_full_sort_shared_scalar_aggregate_derivation_invariants(plan, &shape));
    }
    required.extend([
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
    ]);
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
                    "missing_root_full_sort_join_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }

    CapabilityDecision::Satisfied
}

pub(in crate::capability) fn analyze_exact_statistical_join_final_full_sort_admission(
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
    match analyze_root_full_sort_join_aggregate_admission(plan, nodes, edges, invariants) {
        CapabilityDecision::Satisfied => {},
        CapabilityDecision::Rejected { reason, facts } => {
            return CapabilityDecision::Rejected { reason, facts };
        },
        CapabilityDecision::Conditional { waiting_on } => {
            return CapabilityDecision::Conditional { waiting_on };
        },
    }
    let Some(invariant) =
        invariants.iter().find(|invariant| invariant.kind == SubgraphInvariantKind::StatisticalJoinFinalSortExactness)
    else {
        return CapabilityDecision::rejected(
            "exact_statistical_join_final_full_sort_requires_shape",
            [("shape", "absent".to_string())],
        );
    };
    match &invariant.decision {
        CapabilityDecision::Satisfied => {},
        CapabilityDecision::Rejected { reason, facts } => {
            return CapabilityDecision::Rejected { reason, facts: facts.clone() };
        },
        CapabilityDecision::Conditional { waiting_on } => {
            return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
        },
    }
    for required_contract in [
        "host_welford_input_materialization_preflight",
        "host_welford_output_upload_reservation",
        "final_full_sort_output_cardinality_bound",
    ] {
        if !invariant.reservation_requirements.iter().any(|requirement| {
            requirement.required_contract == required_contract && requirement.status == ContractStatus::Modeled
        }) {
            return CapabilityDecision::rejected(
                "exact_statistical_join_final_full_sort_missing_contract_gate",
                [("required_contract", required_contract.to_string())],
            );
        }
    }
    CapabilityDecision::Satisfied
}

pub(in crate::capability::admission) fn root_full_sort_subcontract_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    match shape.subcontract {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate => {
            shape.shared_producer_ids.is_empty()
                && (root_full_sort_aggregate_is_on_sink_path(plan, shape)
                    || root_full_sort_root_join_has_filtered_aggregate_branch(plan, shape))
        },
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin => {
            root_full_sort_keyless_scalar_projection_fanout_has_cardinality_proof(plan, shape)
        },
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout => {
            !shape.shared_producer_ids.is_empty()
                && shape.keyless_or_cross_join_ids.is_empty()
                && root_full_sort_shared_producers_are_projection_fanout(plan, &shape.shared_producer_ids)
        },
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout => {
            root_full_sort_shared_aggregate_fanout_has_cardinality_proof(plan, shape)
        },
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout => false,
    }
}

pub(in crate::capability::admission) fn root_full_sort_aggregate_is_on_sink_path(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    let mut current = shape.final_sort_id;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(node) = plan.node(current) else {
            return false;
        };
        match &node.kind {
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Window(window) if root_full_sort_window_allows_sink_path_grouped_aggregate_search(window) => {
                current = window.input;
            },
            PlanNode::Aggregate(_) => return current == shape.aggregate_id,
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => return false,
        }
    }
}

pub(in crate::capability::admission) fn root_full_sort_root_join_has_filtered_aggregate_branch(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    let Some(PlanNode::Join(join)) = plan.node(shape.root_join_id).map(|node| &node.kind) else {
        return false;
    };
    join.inputs.iter().any(|input| root_full_sort_branch_filters_aggregate(plan, *input, shape.aggregate_id))
}

pub(in crate::capability::admission) fn root_full_sort_branch_filters_aggregate(
    plan: &QueryPlan,
    mut current: NodeId,
    aggregate_id: NodeId,
) -> bool {
    let mut visited = BTreeSet::new();
    let mut saw_filter = false;
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(node) = plan.node(current) else {
            return false;
        };
        match &node.kind {
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => {
                saw_filter = true;
                current = filter.input;
            },
            PlanNode::Aggregate(_) => return current == aggregate_id && saw_filter,
            _ => return false,
        }
    }
}

pub(in crate::capability::admission) fn root_full_sort_keyless_scalar_projection_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    if shape.keyless_or_cross_join_ids.as_slice() != [shape.root_join_id] {
        return false;
    }
    if shape.shared_producer_ids.is_empty()
        || !root_full_sort_shared_producers_are_projection_fanout(plan, &shape.shared_producer_ids)
    {
        return false;
    }
    let Some(PlanNode::Join(join)) = plan.node(shape.root_join_id).map(|node| &node.kind) else {
        return false;
    };
    if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeylessConditional(_))) {
        return false;
    }
    let Some(scalar_side) = root_full_sort_keyless_join_scalar_side(plan, join) else {
        return false;
    };
    join.spec.output_projection.iter().all(|column| column.side != scalar_side)
}

pub(in crate::capability::admission) fn root_full_sort_keyless_join_scalar_side(
    plan: &QueryPlan,
    join: &JoinNode,
) -> Option<JoinSide> {
    [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| root_full_sort_input_is_scalar_aggregate(plan, join.inputs[join_side_index(*side)]))
}

pub(in crate::capability::admission) fn root_full_sort_input_is_scalar_aggregate(
    plan: &QueryPlan,
    node_id: NodeId,
) -> bool {
    let Some((peeled, _)) = peel_resident_unary_chain_for_capability(plan, node_id) else {
        return false;
    };
    matches!(
        plan.node(peeled).map(|node| &node.kind),
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
    )
}

pub(in crate::capability::admission) fn root_full_sort_shared_producers_are_projection_fanout(
    plan: &QueryPlan,
    shared_producer_ids: &[NodeId],
) -> bool {
    shared_producer_ids.iter().all(|node_id| {
        matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Projection(_) | PlanNode::Filter(_)))
    })
}

pub(in crate::capability::admission) fn root_full_sort_shared_aggregate_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    if !shape.unsupported_shared_producer_ids.is_empty()
        || !shape.shared_projection_producer_ids.is_empty()
        || shape.shared_aggregate_producer_ids.is_empty()
        || shape.shared_producer_ids != shape.shared_aggregate_producer_ids
        || shape.shared_aggregate_consumer_join_ids.is_empty()
        || !shape.shared_aggregate_consumer_join_ids.iter().all(|join_id| shape.join_ids.contains(join_id))
        || !root_full_sort_all_shared_aggregate_consumers_feed_joins(plan, shape)
    {
        return false;
    }

    if !shape.keyless_or_cross_join_ids.is_empty()
        && !root_full_sort_keyless_shared_aggregate_fanout_has_cardinality_proof(plan, shape)
    {
        return false;
    }

    shape.shared_aggregate_producer_ids.iter().all(|node_id| {
        matches!(
            plan.node(*node_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
        )
    })
}

pub(in crate::capability::admission) fn root_full_sort_keyless_shared_aggregate_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    if shape.keyless_or_cross_join_ids.as_slice() != [shape.root_join_id] {
        return false;
    }
    let Some(PlanNode::Join(join)) = plan.node(shape.root_join_id).map(|node| &node.kind) else {
        return false;
    };
    if !matches!(
        join_node_execution_capability(join),
        Some(JoinExecutionCapability::KeylessConditional(KeylessConditionalJoinKind::Inner))
    ) || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        return false;
    }
    let Some(scalar_side) = root_full_sort_keyless_join_scalar_side(plan, join) else {
        return false;
    };
    if scalar_side != JoinSide::Right {
        return false;
    }
    let Some((scalar_aggregate_id, _)) =
        peel_resident_unary_chain_for_capability(plan, join.inputs[join_side_index(scalar_side)])
    else {
        return false;
    };
    if !matches!(
        plan.node(scalar_aggregate_id).map(|node| &node.kind),
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
    ) {
        return false;
    }
    root_full_sort_shared_scalar_aggregate_derivation_invariants(plan, shape).into_iter().any(|selector| {
        matches!(
            selector,
            InvariantSelector::AggregateDerivesFromSharedProducer { aggregate_id, .. }
                if aggregate_id == scalar_aggregate_id
        )
    })
}

pub(in crate::capability::admission) fn root_full_sort_all_shared_aggregate_consumers_feed_joins(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return false;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    shape.shared_aggregate_producer_ids.iter().all(|producer| {
        consumers.get(producer).is_some_and(|producer_consumers| {
            producer_consumers.iter().all(|consumer| {
                root_full_sort_consumer_chain_reaches_join(plan, &consumers, *consumer, &mut BTreeSet::new())
            })
        })
    })
}

pub(in crate::capability::admission) fn root_full_sort_consumer_chain_reaches_join(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
) -> bool {
    if !visited.insert(node_id) {
        return false;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => true,
        Some(PlanNode::Projection(_) | PlanNode::Filter(_) | PlanNode::Window(_)) => {
            consumers.get(&node_id).is_some_and(|next_consumers| {
                next_consumers
                    .iter()
                    .all(|consumer| root_full_sort_consumer_chain_reaches_join(plan, consumers, *consumer, visited))
            })
        },
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty() => {
            consumers.get(&node_id).is_some_and(|next_consumers| {
                next_consumers
                    .iter()
                    .all(|consumer| root_full_sort_consumer_chain_reaches_join(plan, consumers, *consumer, visited))
            })
        },
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Source(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => false,
    }
}

pub(in crate::capability::admission) fn root_full_sort_shared_scalar_aggregate_derivation_invariants(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> Vec<InvariantSelector> {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return Vec::new();
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    let mut pairs = BTreeSet::new();
    for shared_producer in &shape.shared_aggregate_producer_ids {
        if let Some(producer_consumers) = consumers.get(shared_producer) {
            for consumer in producer_consumers {
                root_full_sort_collect_shared_scalar_aggregate_derivations(
                    plan,
                    &consumers,
                    *shared_producer,
                    *consumer,
                    &mut BTreeSet::new(),
                    &mut pairs,
                );
            }
        }
    }
    pairs
        .into_iter()
        .map(|(shared_producer, aggregate_id)| InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer,
            aggregate_id,
        })
        .collect()
}

pub(in crate::capability::admission) fn root_full_sort_collect_shared_scalar_aggregate_derivations(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    shared_producer: NodeId,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    pairs: &mut BTreeSet<(NodeId, NodeId)>,
) {
    if !visited.insert(node_id) {
        return;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty() => {
            pairs.insert((shared_producer, node_id));
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_shared_scalar_aggregate_derivations(
                        plan,
                        consumers,
                        shared_producer,
                        *consumer,
                        visited,
                        pairs,
                    );
                }
            }
        },
        Some(PlanNode::Projection(_) | PlanNode::Filter(_) | PlanNode::Window(_)) => {
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_shared_scalar_aggregate_derivations(
                        plan,
                        consumers,
                        shared_producer,
                        *consumer,
                        visited,
                        pairs,
                    );
                }
            }
        },
        Some(PlanNode::Join(_))
        | Some(
            PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Source(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => {},
    }
}

pub(in crate::capability::admission) fn root_full_sort_join_capability_satisfies_subcontract(
    shape: &RootFullSortJoinAggregateShape,
    join_id: NodeId,
    join: &JoinNode,
    capability: JoinExecutionCapability,
) -> bool {
    match shape.subcontract {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate => {
            matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                || root_full_sort_keyed_filtered_semi_anti_satisfies_grouped_aggregate_subcontract(join, capability)
        },
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin => {
            if shape.keyless_or_cross_join_ids.contains(&join_id) {
                matches!(
                    capability,
                    JoinExecutionCapability::KeylessConditional(_)
                        | JoinExecutionCapability::KeylessOuterCross
                        | JoinExecutionCapability::ProjectedCross
                )
            } else {
                matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
            }
        },
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout => {
            matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                || root_full_sort_keyed_filtered_semi_anti_satisfies_grouped_aggregate_subcontract(join, capability)
        },
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout => {
            if shape.keyless_or_cross_join_ids.contains(&join_id) {
                matches!(capability, JoinExecutionCapability::KeylessConditional(KeylessConditionalJoinKind::Inner))
            } else {
                matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
            }
        },
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout => false,
    }
}

pub(in crate::capability::admission) fn root_full_sort_keyed_filtered_semi_anti_satisfies_grouped_aggregate_subcontract(
    join: &JoinNode,
    capability: JoinExecutionCapability,
) -> bool {
    capability == JoinExecutionCapability::KeyedFilteredSemiAnti
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(in crate::capability::admission) fn root_full_sort_unsupported_subcontract_rejection(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> CapabilityDecision {
    CapabilityDecision::Rejected {
        reason: "root_full_sort_join_aggregate_rejects_unsupported_subcontract",
        facts: BTreeMap::from([
            ("subcontract", shape.subcontract.as_str().to_string()),
            ("final_sort_id", shape.final_sort_id.to_string()),
            ("sink_kind", root_full_sort_sink_kind_fact(plan.sink()).to_string()),
            ("sink_input_id", plan.sink().input().to_string()),
            ("host_sink_path", format!("{}->{}", shape.final_sort_id, root_full_sort_sink_kind_fact(plan.sink()))),
            ("root_join_id", shape.root_join_id.to_string()),
            ("join_node_ids", node_ids_fact(&shape.join_ids)),
            ("keyless_or_cross_join_node_ids", node_ids_fact(&shape.keyless_or_cross_join_ids)),
            ("shared_producer_node_ids", node_ids_fact(&shape.shared_producer_ids)),
            ("shared_projection_producer_node_ids", node_ids_fact(&shape.shared_projection_producer_ids)),
            ("shared_aggregate_producer_node_ids", node_ids_fact(&shape.shared_aggregate_producer_ids)),
            ("unsupported_shared_producer_node_ids", node_ids_fact(&shape.unsupported_shared_producer_ids)),
            ("shared_aggregate_consumer_join_node_ids", node_ids_fact(&shape.shared_aggregate_consumer_join_ids)),
            ("aggregate_id", shape.aggregate_id.to_string()),
            ("aggregate_projection_id", shape.aggregate_projection_id.to_string()),
            ("required_remediation", "root_full_sort_subcontract_cardinality_proof".to_string()),
        ]),
    }
}

pub(in crate::capability::admission) fn root_full_sort_sink_kind_fact(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::GpuResident { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    }
}
