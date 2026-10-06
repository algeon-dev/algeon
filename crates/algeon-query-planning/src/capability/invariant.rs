//! Subgraph-invariant analysis (currently focused on join shape invariants).
//!
//! Inspects edge contracts plus producer/consumer node facts to verify
//! invariants that span more than a single node; for example, "build side of
//! this join must dominate its resident build resource."

use super::{
    ColumnQualifier, Expr, JoinNode, JoinSide, NodeId, PlanNode, QueryPlan,
    edge::{edge_consumer_fact, node_operator_fact},
    graph::{consumer_nodes, peel_projection_chain_for_capability, peel_resident_unary_chain_for_capability},
    lineage::{OutputLineage, expression_read_lineages, output_lineage},
    model::{
        CapabilityDecision, EdgeConsumer, EdgeContract, InvariantId, ResidentResource, ScheduleConstraint,
        SubgraphInvariant, SubgraphInvariantKind,
    },
    rejection::Reason,
};

pub(super) fn analyze_invariants(plan: &QueryPlan, edges: &[EdgeContract]) -> Vec<SubgraphInvariant> {
    let mut invariants = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
        let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
        let invariant_id = InvariantId(invariants.len());
        // Plan construction orders both inputs before the join, so neither
        // input subgraph can contain it.
        let decision = CapabilityDecision::Satisfied;
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

    if plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Join(_))).count() > 1 {
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
            decision: CapabilityDecision::Satisfied,
        });
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
            nodes: plan.nodes().iter().map(|node| node.id).collect(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: solve_resident_lifetime_invariant(plan, edges),
        });
    }

    invariants
}

pub(super) fn append_shared_resident_subplan_invariants(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    invariants: &mut Vec<SubgraphInvariant>,
) {
    for (producer, consumers) in consumer_nodes(plan) {
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
            decision: solve_shared_resident_subplan_invariant(edges, producer, &consumers),
        });
    }
}

pub(super) fn append_aggregate_derivation_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let consumers = consumer_nodes(plan);
    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if !aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let Some((shared_producer, _projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input)
        else {
            continue;
        };
        if consumers.get(&shared_producer).map_or(0, Vec::len) <= 1 {
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

pub(super) fn solve_resident_lifetime_invariant(plan: &QueryPlan, edges: &[EdgeContract]) -> CapabilityDecision {
    let mut consumers = consumer_nodes(plan);
    consumers.entry(plan.sink().input()).or_default();

    for producer in plan.nodes().iter().map(|node| node.id) {
        for edge in edges.iter().filter(|edge| edge.producer == producer) {
            if let EdgeConsumer::Node(consumer) = edge.consumer
                && !consumers.get(&producer).is_some_and(|values| values.contains(&consumer))
            {
                continue;
            }
            if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
                return CapabilityDecision::rejected(
                    Reason::ResidentLifetimeEdgeRejected,
                    [("producer", producer.to_string()), ("consumer", edge_consumer_fact(edge.consumer))],
                );
            }
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_shared_resident_subplan_invariant(
    edges: &[EdgeContract],
    producer: NodeId,
    consumers: &[NodeId],
) -> CapabilityDecision {
    for consumer in consumers {
        let Some(edge) =
            edges.iter().find(|edge| edge.producer == producer && edge.consumer == EdgeConsumer::Node(*consumer))
        else {
            return CapabilityDecision::rejected(
                Reason::SharedResidentSubplanMissingEdge,
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        };
        if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
            return CapabilityDecision::rejected(
                Reason::SharedResidentSubplanEdgeRejected,
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
    if plan.node(shared_producer).is_none() {
        return CapabilityDecision::rejected(
            Reason::AggregateSharedProducerMissingNode,
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    }
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return CapabilityDecision::rejected(
            Reason::AggregateSharedProducerMissingAggregate,
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return CapabilityDecision::rejected(
            Reason::AggregateSharedProducerConsumerNotAggregate,
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };

    let Some((peeled_input, _projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input) else {
        return CapabilityDecision::rejected(
            Reason::AggregateSharedProducerInputLineageUnresolved,
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    if peeled_input != shared_producer {
        return CapabilityDecision::rejected(
            Reason::AggregateSharedProducerRequiresDataflowIdentity,
            [
                ("shared_producer", shared_producer.to_string()),
                ("aggregate_id", aggregate_id.to_string()),
                ("peeled_input", peeled_input.to_string()),
            ],
        );
    }

    let producer_lineages = match output_lineages_for_node(plan, shared_producer) {
        Ok(lineages) => lineages,
        Err(decision) => return decision,
    };
    for (aggr_index, expr) in aggregate.spec.aggr_exprs.iter().enumerate() {
        let read_lineages = match expression_read_lineages(plan, aggregate.input, &expr.expr) {
            Ok(lineages) => lineages,
            Err(decision) => return decision,
        };
        if !read_lineages.iter().all(|lineage| producer_lineages.contains(lineage)) {
            return CapabilityDecision::rejected(
                Reason::AggregateSharedProducerRequiresInputLineage,
                [
                    ("shared_producer", shared_producer.to_string()),
                    ("aggregate_id", aggregate_id.to_string()),
                    ("aggr_index", aggr_index.to_string()),
                ],
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn output_lineages_for_node(plan: &QueryPlan, node_id: NodeId) -> Result<Vec<OutputLineage>, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            Reason::AggregateSharedProducerMissingNode,
            [("shared_producer", node_id.to_string())],
        ));
    };
    (0..node.output_schema().fields().len()).map(|field_index| output_lineage(plan, node_id, field_index)).collect()
}

pub(super) fn solve_projection_transparent_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    projection_id: NodeId,
) -> CapabilityDecision {
    if plan.node(peeled_producer).is_none() {
        return CapabilityDecision::rejected(
            Reason::ProjectionTransparentResidencyMissingProducer,
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    let Some(projection) = plan.node(projection_id) else {
        return CapabilityDecision::rejected(
            Reason::ProjectionTransparentResidencyMissingProjection,
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    if !matches!(projection.kind, PlanNode::Projection(_)) {
        return CapabilityDecision::rejected(
            Reason::ProjectionTransparentResidencyConsumerNotProjection,
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
    if plan.node(peeled_producer).is_none() {
        return CapabilityDecision::rejected(
            Reason::ResidentUnaryChainMissingProducer,
            [("peeled_producer", peeled_producer.to_string())],
        );
    }
    let mut expected_input = peeled_producer;
    for chain_id in chain_ids_from_consumer.iter().rev() {
        let Some(chain_node) = plan.node(*chain_id) else {
            return CapabilityDecision::rejected(
                Reason::ResidentUnaryChainMissingNode,
                [("node_id", chain_id.to_string()), ("peeled_producer", peeled_producer.to_string())],
            );
        };
        let actual_input = match &chain_node.kind {
            PlanNode::Projection(projection) => projection.input,
            PlanNode::Filter(filter) => filter.input,
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_) => {
                return CapabilityDecision::rejected(
                    Reason::ResidentUnaryChainRequiresFilterProjectionNodes,
                    [("node_id", chain_id.to_string()), ("operator", node_operator_fact(&chain_node.kind).to_string())],
                );
            },
        };
        if actual_input != expected_input {
            return CapabilityDecision::rejected(
                Reason::ResidentUnaryChainRequiresConnectedNodes,
                [
                    ("node_id", chain_id.to_string()),
                    ("expected_input", expected_input.to_string()),
                    ("actual_input", actual_input.to_string()),
                ],
            );
        }
        expected_input = *chain_id;
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
        Reason::ResidualDoesNotReferenceBothInputs,
        [
            ("join_id", join_id.to_string()),
            ("references_left", references_left.to_string()),
            ("references_right", references_right.to_string()),
            ("expression", predicate.to_string()),
        ],
    )
}

pub(super) fn expr_references_input(expr: &Expr, input_index: usize) -> bool {
    expr.columns()
        .any(|column| matches!(column.qualifier(), Some(ColumnQualifier::Input { index }) if *index == input_index))
}
