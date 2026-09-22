//! Per-edge capability analysis.
//!
//! Each producer→consumer edge in the plan carries a set of contract kinds
//! (column carriage, key carriage, partition compatibility, etc.). Rejections
//! here are local to the edge; whole-plan admission lives in [`super::admission`].

use std::collections::BTreeSet;

use super::{
    JoinExecutionCapability, JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, ResultSink,
    aggregate::{AggregateStateContract, aggregate_state_contract},
    graph::{collect_reachable_node_ids, consumer_nodes_for_reachable},
    join_node_execution_capability,
    model::{
        CapabilityDecision, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, ProducedProperty, ResultSinkKind,
        VectorInputRole,
    },
    node::{produced_properties, source_format_capability},
    shape::{join_side_index, opposite_join_side},
};

pub(super) fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let contract_kinds = edge_contract_kinds(plan, *producer, consumer, kind, input_role);
            let decision = edge_decision(plan, *producer, consumer);
            let decision = match (&decision, kind) {
                (CapabilityDecision::Satisfied, PlanNode::Join(join)) if join.spec.join_type == JoinType::LeftAnti => {
                    left_anti_dedup_edge_decision(plan, *producer, input_role, join)
                },
                _ => decision,
            };
            EdgeContract { producer: *producer, consumer, contract_kinds, input_role, decision }
        })
        .collect()
}

/// Set-operation guard for left-anti joins.
///
/// An anti build against a dedup aggregate claims EXCEPT DISTINCT, so an
/// aggregate build must really be one: a grouped aggregate with measures or
/// distinct state is EXCEPT ALL, which the bounded engine cannot do with
/// multiplicity. Sources, filters, and joins keep ordinary anti semantics,
/// which the engine executes directly, so only aggregate builds are policed.
/// For an except-shaped join the probe side must be dedup-proven, or a
/// trailing scalar count would count duplicates.
fn left_anti_dedup_edge_decision(
    plan: &QueryPlan,
    producer: NodeId,
    input_role: InputRole,
    join: &JoinNode,
) -> CapabilityDecision {
    let producer_state = plan.node(producer).and_then(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => aggregate_state_contract(&aggregate.spec),
        _ => None,
    });
    match input_role {
        InputRole::JoinBuildInput { .. } => {
            if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Aggregate(_)))
                && producer_state != Some(AggregateStateContract::DedupGroupOnly)
            {
                return CapabilityDecision::rejected(
                    "left_anti_build_requires_dedup_aggregate",
                    [
                        ("producer", producer.to_string()),
                        (
                            "producer_aggregate_state",
                            producer_state.map_or("none", AggregateStateContract::as_str).to_string(),
                        ),
                    ],
                );
            }
            CapabilityDecision::Satisfied
        },
        InputRole::JoinProbeInput { .. } => {
            let build_input = join.inputs[join_side_index(join.spec.build_side)];
            let build_dedup = plan.node(build_input).is_some_and(|node| match &node.kind {
                PlanNode::Aggregate(aggregate) => {
                    aggregate_state_contract(&aggregate.spec) == Some(AggregateStateContract::DedupGroupOnly)
                },
                _ => false,
            });
            if build_dedup && !anti_probe_input_is_dedup_proven(plan, producer, &mut BTreeSet::new()) {
                return CapabilityDecision::rejected(
                    "left_anti_probe_requires_dedup_proven_input",
                    [
                        ("producer", producer.to_string()),
                        (
                            "producer_aggregate_state",
                            producer_state.map_or("none", AggregateStateContract::as_str).to_string(),
                        ),
                    ],
                );
            }
            CapabilityDecision::Satisfied
        },
        _ => CapabilityDecision::Satisfied,
    }
}

/// Structural proof that an anti-join probe input is a distinct relation:
/// either a group-only dedup aggregate, or a keyed filtered left-anti join
/// whose own build input is a dedup aggregate and whose probe input is
/// dedup-proven. Consults no row estimates.
fn anti_probe_input_is_dedup_proven(plan: &QueryPlan, node_id: NodeId, visiting: &mut BTreeSet<NodeId>) -> bool {
    if !visiting.insert(node_id) {
        return false;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Aggregate(aggregate)) => {
            aggregate_state_contract(&aggregate.spec) == Some(AggregateStateContract::DedupGroupOnly)
        },
        Some(PlanNode::Join(join))
            if join.spec.join_type == JoinType::LeftAnti
                && matches!(
                    join_node_execution_capability(join),
                    Some(JoinExecutionCapability::KeyedFilteredSemiAnti)
                ) =>
        {
            let build_input = join.inputs[join_side_index(join.spec.build_side)];
            let probe_input = join.inputs[join_side_index(opposite_join_side(join.spec.build_side))];
            plan.node(build_input).is_some_and(|node| match &node.kind {
                PlanNode::Aggregate(aggregate) => {
                    aggregate_state_contract(&aggregate.spec) == Some(AggregateStateContract::DedupGroupOnly)
                },
                _ => false,
            }) && anti_probe_input_is_dedup_proven(plan, probe_input, visiting)
        },
        _ => false,
    }
}

pub(super) fn edge_contract_kinds(
    plan: &QueryPlan,
    producer: NodeId,
    consumer: EdgeConsumer,
    consumer_kind: &PlanNode,
    input_role: InputRole,
) -> Vec<EdgeContractKind> {
    let mut kinds = Vec::new();
    if let Some(node) = plan.node(producer) {
        if let PlanNode::Source(source) = &node.kind {
            kinds.push(EdgeContractKind::ResidentSourceInput { format: source_format_capability(source) });
        }
        if matches!(node.kind, PlanNode::Join(_)) {
            kinds.push(EdgeContractKind::JoinMaterialization);
        }
    }
    if matches!(consumer, EdgeConsumer::Node(_)) {
        kinds.push(EdgeContractKind::ResidentFrameFlow);
    }
    if matches!(consumer_kind, PlanNode::Aggregate(_)) {
        kinds.push(EdgeContractKind::ResidentAggregateInput);
    }
    if matches!(consumer_kind, PlanNode::Window(_)) {
        kinds.push(EdgeContractKind::WindowInput);
    }
    if matches!(consumer_kind, PlanNode::GraphAlgorithm(_)) {
        kinds.push(EdgeContractKind::GraphAlgorithmInput);
    }
    if let InputRole::VectorAlgorithmInput { role } = input_role {
        kinds.push(EdgeContractKind::VectorAlgorithmInput { role });
    }
    match input_role {
        InputRole::JoinBuildInput { side } => kinds.push(EdgeContractKind::JoinBuildInput { side }),
        InputRole::JoinProbeInput { side } => kinds.push(EdgeContractKind::JoinProbeInput { side }),
        InputRole::UnionInput { input_index } => kinds.push(EdgeContractKind::UnionInput { input_index }),
        _ => {},
    }
    if matches!(consumer_kind, PlanNode::Sort(_)) {
        kinds.push(EdgeContractKind::SortInput);
    }
    kinds
}

pub(super) fn edge_decision(plan: &QueryPlan, producer: NodeId, consumer: EdgeConsumer) -> CapabilityDecision {
    let Some(producer_node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "missing_edge_producer",
            [("producer", producer.to_string()), ("consumer", edge_consumer_fact(consumer))],
        );
    };
    if !produced_properties(producer_node).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "missing_edge_required_property",
            [
                ("producer", producer.to_string()),
                ("consumer", edge_consumer_fact(consumer)),
                ("required_property", "resident_frame".to_string()),
                ("producer_placement", producer_node.facts.placement.to_string()),
                ("producer_operator", node_operator_fact(&producer_node.kind).to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn edge_consumer_fact(consumer: EdgeConsumer) -> String {
    match consumer {
        EdgeConsumer::Node(node) => node.to_string(),
        EdgeConsumer::Sink(ResultSinkKind::HostArrow) => "sink:host_arrow".to_string(),
        EdgeConsumer::Sink(ResultSinkKind::GpuResident) => "sink:gpu_resident".to_string(),
    }
}

pub(super) fn node_operator_fact(kind: &PlanNode) -> &'static str {
    match kind {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
        PlanNode::EdgeNormalize(_) => "edge_normalize",
        PlanNode::GraphAlgorithm(_) => "graph_algorithm",
        PlanNode::VectorAlgorithm(_) => "vector_algorithm",
    }
}

pub(super) fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
    match kind {
        PlanNode::Aggregate(_) => InputRole::AggregateInput,
        PlanNode::Window(_) => InputRole::WindowInput,
        PlanNode::Sort(_) => InputRole::SortInput,
        PlanNode::Limit(_) => InputRole::LimitInput,
        PlanNode::Union(_) => InputRole::UnionInput { input_index },
        PlanNode::EdgeNormalize(_) => InputRole::UnaryInput,
        PlanNode::GraphAlgorithm(_) => InputRole::GraphAlgorithmInput,
        PlanNode::VectorAlgorithm(vector) => {
            InputRole::VectorAlgorithmInput { role: VectorInputRole::for_vector_input(vector, input_index) }
        },
        PlanNode::Join(join) => {
            let side = if input_index == 0 { JoinSide::Left } else { JoinSide::Right };
            if side == join.spec.build_side {
                InputRole::JoinBuildInput { side }
            } else {
                InputRole::JoinProbeInput { side }
            }
        },
        _ => InputRole::UnaryInput,
    }
}

pub(super) fn analyze_sink_edge(plan: &QueryPlan) -> EdgeContract {
    let producer = plan.sink().input();
    let consumer = match plan.sink() {
        ResultSink::HostArrow { .. } => EdgeConsumer::Sink(ResultSinkKind::HostArrow),
        ResultSink::GpuResident { .. } => EdgeConsumer::Sink(ResultSinkKind::GpuResident),
    };
    let mut contract_kinds = Vec::new();
    if let Some(node) = plan.node(producer) {
        if let PlanNode::Source(source) = &node.kind {
            contract_kinds.push(EdgeContractKind::ResidentSourceInput { format: source_format_capability(source) });
        }
        if matches!(node.kind, PlanNode::Join(_)) {
            contract_kinds.push(EdgeContractKind::JoinMaterialization);
        }
    }
    if matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        contract_kinds.push(EdgeContractKind::ExplicitSinkMaterialization);
    }
    let decision = edge_decision(plan, producer, consumer);
    EdgeContract { producer, consumer, contract_kinds, input_role: InputRole::SinkInput, decision }
}

pub(super) fn mark_shared_producer_edge_contracts(plan: &QueryPlan, edges: &mut [EdgeContract]) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    for (producer, producer_consumers) in consumers {
        if producer_consumers.len() <= 1 {
            continue;
        }
        for edge in edges.iter_mut().filter(|edge| {
            edge.producer == producer
                && matches!(edge.consumer, EdgeConsumer::Node(consumer) if producer_consumers.contains(&consumer))
        }) {
            if !edge.contract_kinds.contains(&EdgeContractKind::SharedProducer) {
                edge.contract_kinds.push(EdgeContractKind::SharedProducer);
            }
        }
    }
}
