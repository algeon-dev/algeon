use super::{edge::*, graph::*, model::*, node::*, shape::*, *};

pub(super) fn analyze_invariants(plan: &QueryPlan, edges: &[EdgeContract]) -> Vec<SubgraphInvariant> {
    let mut invariants = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
        let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
        let invariant_id = InvariantId(invariants.len());
        let decision = solve_build_before_probe_invariant(plan, node.id, build_input, probe_input);
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::BuildBeforeProbe { join_id: node.id },
            nodes: vec![build_input, probe_input, node.id],
            required_schedule: ScheduleConstraint::BuildBeforeProbe,
            required_resources: vec![
                ResidentResource::JoinBuildInput { join_id: node.id, input: build_input },
                ResidentResource::JoinProbeInput { join_id: node.id, input: probe_input },
            ],
            decision,
        });
        if join.spec.residual_predicate.is_some() {
            let invariant_id = InvariantId(invariants.len());
            let decision = solve_residual_uses_both_inputs_invariant(node.id, join);
            invariants.push(SubgraphInvariant {
                invariant_id,
                kind: SubgraphInvariantKind::ResidualUsesBothInputs { join_id: node.id },
                nodes: vec![join.inputs[0], join.inputs[1], node.id],
                required_schedule: ScheduleConstraint::None,
                required_resources: Vec::new(),
                decision,
            });
        }
    }
    append_shared_resident_subplan_invariants(plan, edges, &mut invariants);
    append_aggregate_derivation_invariants(plan, &mut invariants);
    append_projection_transparent_residency_invariants(plan, &mut invariants);
    append_resident_unary_chain_preserves_residency_invariants(plan, &mut invariants);
    append_final_barrier_order_invariants(plan, &mut invariants);
    append_mixed_residual_post_root_stage_order_invariant(plan, &mut invariants);
    append_hash_join_resident_grouped_aggregate_topk_post_root_stage_order_invariant(plan, &mut invariants);
    append_multi_join_aggregate_over_join_subplan_topk_post_root_stage_order_invariant(plan, &mut invariants);

    if reachable_join_count_from_sink(plan).unwrap_or(0) > 1 {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::TopologicalBuildProbeSchedule,
            nodes: plan
                .nodes()
                .iter()
                .filter_map(|node| matches!(node.kind, PlanNode::Join(_)).then_some(node.id))
                .collect(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: solve_topological_build_probe_schedule_invariant(plan),
        });
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
            nodes: collect_reachable_node_ids(plan).map(|nodes| nodes.into_iter().collect()).unwrap_or_default(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: solve_resident_lifetime_invariant(plan, edges),
        });
    }

    if matches!(closed_multi_join_subgraph_structure_decision(plan), CapabilityDecision::Satisfied)
        && let Ok(root_join_id) = post_root_join_id(plan)
    {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ClosedJoinSubtree { root_join_id },
            nodes: collect_reachable_node_ids_from(plan, root_join_id)
                .map(|nodes| nodes.into_iter().collect())
                .unwrap_or_else(|_| vec![root_join_id]),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: CapabilityDecision::Satisfied,
        });
    }
    invariants
}

pub(super) fn append_shared_resident_subplan_invariants(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    for (producer, consumers) in consumer_nodes_for_reachable(plan, &reachable) {
        if consumers.len() <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        let mut nodes = Vec::with_capacity(consumers.len() + 1);
        nodes.push(producer);
        nodes.extend(consumers.iter().copied());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::SharedResidentSubplan,
            nodes,
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(producer)],
            decision: solve_shared_resident_subplan_invariant(plan, edges, producer, &consumers),
        });
    }
}

pub(super) fn append_aggregate_derivation_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if !aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let Some((shared_producer, projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input)
        else {
            continue;
        };
        if projection_count == 0 || consumers.get(&shared_producer).map_or(0, Vec::len) <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::AggregateDerivesFromSharedProducer,
            nodes: vec![shared_producer, aggregate.input, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(shared_producer)],
            decision: solve_aggregate_derives_from_shared_producer_invariant(plan, shared_producer, node.id),
        });
    }
}

pub(super) fn append_projection_transparent_residency_invariants(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    for node in plan.nodes() {
        if !matches!(node.kind, PlanNode::Projection(_)) {
            continue;
        }
        let Some((peeled_producer, projection_count)) = peel_projection_chain_for_capability(plan, node.id) else {
            continue;
        };
        if projection_count == 0 || peeled_producer == node.id {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ProjectionTransparentResidency,
            nodes: vec![peeled_producer, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(peeled_producer)],
            decision: solve_projection_transparent_residency_invariant(plan, peeled_producer, node.id),
        });
    }
}

pub(super) fn append_resident_unary_chain_preserves_residency_invariants(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    for node in plan.nodes() {
        if !matches!(node.kind, PlanNode::Projection(_) | PlanNode::Filter(_)) {
            continue;
        }
        let Some((peeled_producer, chain_ids)) = peel_resident_unary_chain_for_capability(plan, node.id) else {
            continue;
        };
        if chain_ids.is_empty() || peeled_producer == node.id {
            continue;
        }
        let mut invariant_nodes = Vec::with_capacity(chain_ids.len() + 1);
        invariant_nodes.push(peeled_producer);
        invariant_nodes.extend(chain_ids.iter().rev().copied());
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentUnaryChainPreservesResidency,
            nodes: invariant_nodes,
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(peeled_producer)],
            decision: solve_resident_unary_chain_preserves_residency_invariant(plan, peeled_producer, &chain_ids),
        });
    }
}

pub(super) fn append_final_barrier_order_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    for node in plan.nodes() {
        let PlanNode::Sort(sort) = &node.kind else {
            continue;
        };
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::FinalBarrierOrder,
            nodes: vec![sort.input, node.id],
            required_schedule: ScheduleConstraint::Barrier,
            required_resources: vec![ResidentResource::NodeOutput(sort.input)],
            decision: solve_final_barrier_order_invariant(plan, node.id, sort.input),
        });
    }
}

pub(super) fn append_mixed_residual_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: vec![
            shape.root_join_id,
            shape.post_root_projection_id,
            shape.post_root_aggregate_id,
            shape.final_projection_id,
        ],
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn append_hash_join_resident_grouped_aggregate_topk_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: vec![
            shape.root_join_id,
            shape.post_root_projection_id,
            shape.post_root_aggregate_id,
            shape.final_projection_id,
            shape.final_sort_id,
            shape.final_limit_id,
        ],
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn append_multi_join_aggregate_over_join_subplan_topk_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: vec![shape.root_join_id, shape.post_root_projection_id, shape.final_sort_id, shape.final_limit_id],
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn solve_build_before_probe_invariant(
    plan: &QueryPlan,
    join_id: NodeId,
    build_input: NodeId,
    probe_input: NodeId,
) -> CapabilityDecision {
    let build_nodes = match collect_reachable_node_ids_from(plan, build_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    let probe_nodes = match collect_reachable_node_ids_from(plan, probe_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if build_nodes.contains(&join_id) || probe_nodes.contains(&join_id) {
        return CapabilityDecision::rejected(
            "build_before_probe_cycle",
            [
                ("join_id", join_id.to_string()),
                ("build_input", build_input.to_string()),
                ("probe_input", probe_input.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_topological_build_probe_schedule_invariant(plan: &QueryPlan) -> CapabilityDecision {
    let mut scheduler = GraphScheduleProof::new(plan);
    scheduler.schedule_from(plan.sink().input()).map_or_else(|decision| decision, |()| CapabilityDecision::Satisfied)
}

pub(super) fn solve_resident_lifetime_invariant(plan: &QueryPlan, edges: &[EdgeContract]) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    let mut consumers = consumer_nodes_for_reachable(plan, &reachable);
    consumers.entry(plan.sink().input()).or_default();

    for producer in reachable {
        let Some(node) = plan.node(producer) else {
            return CapabilityDecision::rejected("resident_lifetime_missing_node", [("node_id", producer.to_string())]);
        };
        if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
            return CapabilityDecision::rejected(
                "resident_lifetime_requires_resident_frame",
                [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            );
        }
        for edge in edges.iter().filter(|edge| edge.producer == producer) {
            if let EdgeConsumer::Node(consumer) = edge.consumer
                && !consumers.get(&producer).is_some_and(|values| values.contains(&consumer))
            {
                continue;
            }
            if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
                return CapabilityDecision::rejected(
                    "resident_lifetime_edge_rejected",
                    [("producer", producer.to_string()), ("consumer", edge_consumer_fact(edge.consumer))],
                );
            }
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_shared_resident_subplan_invariant(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    producer: NodeId,
    consumers: &[NodeId],
) -> CapabilityDecision {
    let Some(node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_missing_node",
            [("node_id", producer.to_string())],
        );
    };
    if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_requires_resident_frame",
            [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        );
    }
    for consumer in consumers {
        let Some(edge) =
            edges.iter().find(|edge| edge.producer == producer && edge.consumer == EdgeConsumer::Node(*consumer))
        else {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_missing_edge",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        };
        if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_edge_rejected",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_aggregate_derives_from_shared_producer_invariant(
    plan: &QueryPlan,
    shared_producer: NodeId,
    aggregate_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(shared_producer) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_node",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    let Some(aggregate) = plan.node(aggregate_id) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    if !matches!(aggregate.kind, PlanNode::Aggregate(_)) {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_consumer_not_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    }
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_requires_resident_frame",
            [("shared_producer", shared_producer.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_projection_transparent_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    projection_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(peeled_producer) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_producer",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    let Some(projection) = plan.node(projection_id) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    if !matches!(projection.kind, PlanNode::Projection(_)) {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_consumer_not_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame)
        || !produced_properties(projection).contains(&ProducedProperty::ResidentFrame)
    {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_requires_resident_frames",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_resident_unary_chain_preserves_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    chain_ids_from_consumer: &[NodeId],
) -> CapabilityDecision {
    let Some(producer) = plan.node(peeled_producer) else {
        return CapabilityDecision::rejected(
            "resident_unary_chain_missing_producer",
            [("peeled_producer", peeled_producer.to_string())],
        );
    };
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "resident_unary_chain_requires_resident_producer",
            [("peeled_producer", peeled_producer.to_string())],
        );
    }
    let mut expected_input = peeled_producer;
    for chain_id in chain_ids_from_consumer.iter().rev() {
        let Some(chain_node) = plan.node(*chain_id) else {
            return CapabilityDecision::rejected(
                "resident_unary_chain_missing_node",
                [("node_id", chain_id.to_string()), ("peeled_producer", peeled_producer.to_string())],
            );
        };
        let actual_input = match &chain_node.kind {
            PlanNode::Projection(projection) => projection.input,
            PlanNode::Filter(filter) => filter.input,
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_) => {
                return CapabilityDecision::rejected(
                    "resident_unary_chain_requires_filter_projection_nodes",
                    [("node_id", chain_id.to_string()), ("operator", node_operator_fact(&chain_node.kind).to_string())],
                );
            },
        };
        if actual_input != expected_input {
            return CapabilityDecision::rejected(
                "resident_unary_chain_requires_connected_nodes",
                [
                    ("node_id", chain_id.to_string()),
                    ("expected_input", expected_input.to_string()),
                    ("actual_input", actual_input.to_string()),
                ],
            );
        }
        if !produced_properties(chain_node).contains(&ProducedProperty::ResidentFrame) {
            return CapabilityDecision::rejected(
                "resident_unary_chain_requires_resident_nodes",
                [("node_id", chain_id.to_string()), ("peeled_producer", peeled_producer.to_string())],
            );
        }
        expected_input = *chain_id;
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_final_barrier_order_invariant(
    plan: &QueryPlan,
    sort_id: NodeId,
    sort_input: NodeId,
) -> CapabilityDecision {
    let input_nodes = match collect_reachable_node_ids_from(plan, sort_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if input_nodes.contains(&sort_id) {
        return CapabilityDecision::rejected(
            "final_barrier_order_cycle",
            [("sort_id", sort_id.to_string()), ("sort_input", sort_input.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_residual_uses_both_inputs_invariant(join_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    let Some(predicate) = &join.spec.residual_predicate else {
        return CapabilityDecision::Satisfied;
    };
    let references_left = expr_references_input(predicate, 0);
    let references_right = expr_references_input(predicate, 1);
    if references_left && references_right {
        return CapabilityDecision::Satisfied;
    }
    CapabilityDecision::rejected(
        "residual_does_not_reference_both_inputs",
        [
            ("join_id", join_id.to_string()),
            ("references_left", references_left.to_string()),
            ("references_right", references_right.to_string()),
            ("expression", predicate.to_string()),
        ],
    )
}

pub(super) fn expr_references_input(expr: &Expr, input_index: usize) -> bool {
    match expr {
        Expr::Column(column) => {
            matches!(column.qualifier(), Some(ColumnQualifier::Input { index }) if *index == input_index)
        },
        Expr::Literal(_) => false,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::Alias { expr, .. } => expr_references_input(expr, input_index),
        Expr::Binary { left, right, .. } => {
            expr_references_input(left, input_index) || expr_references_input(right, input_index)
        },
        Expr::Like { expr, .. } | Expr::Substring { expr, .. } | Expr::Extract { expr, .. } => {
            expr_references_input(expr, input_index)
        },
        Expr::InList { expr, list, .. } => {
            expr_references_input(expr, input_index) || list.iter().any(|item| expr_references_input(item, input_index))
        },
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                expr_references_input(&branch.when, input_index) || expr_references_input(&branch.then, input_index)
            }) || expr_references_input(else_expr, input_index)
        },
    }
}
