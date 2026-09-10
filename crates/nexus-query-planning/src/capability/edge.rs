//! Per-edge capability analysis.
//!
//! Each producer→consumer edge in the plan carries a set of contract kinds
//! (column carriage, key carriage, partition compatibility, etc.). Rejections
//! here are local to the edge; whole-plan admission lives in [`super::admission`].

use super::{
    JoinSide, NodeId, PlanNode, QueryPlan,
    graph::{collect_reachable_node_ids, consumer_nodes_for_reachable},
    model::{
        CapabilityDecision, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, ProducedProperty,
        RequiredProperty, ResultSinkKind, SharedProducerIdentity, VectorInputRole,
    },
    node::{produced_properties, source_format_capability},
};

pub(super) fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let contract_kinds = edge_contract_kinds(plan, *producer, consumer, kind, input_role);
            let required_properties = vec![RequiredProperty::ResidentFrame];
            let decision = edge_decision(plan, *producer, consumer, &required_properties);
            EdgeContract { producer: *producer, consumer, contract_kinds, input_role, required_properties, decision }
        })
        .collect()
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

pub(super) fn edge_decision(
    plan: &QueryPlan,
    producer: NodeId,
    consumer: EdgeConsumer,
    required_properties: &[RequiredProperty],
) -> CapabilityDecision {
    let Some(producer_node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "missing_edge_producer",
            [("producer", producer.to_string()), ("consumer", edge_consumer_fact(consumer))],
        );
    };
    let produced = produced_properties(producer_node);
    if let Some(required) =
        required_properties.iter().find(|required| !edge_property_is_satisfied(&produced, **required))
    {
        return CapabilityDecision::rejected(
            "missing_edge_required_property",
            [
                ("producer", producer.to_string()),
                ("consumer", edge_consumer_fact(consumer)),
                ("required_property", required_property_fact(*required).to_string()),
                ("producer_placement", producer_node.facts.placement.to_string()),
                ("producer_operator", node_operator_fact(&producer_node.kind).to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn edge_property_is_satisfied(produced: &[ProducedProperty], required: RequiredProperty) -> bool {
    match required {
        RequiredProperty::ResidentFrame => produced.contains(&ProducedProperty::ResidentFrame),
    }
}

pub(super) fn edge_consumer_fact(consumer: EdgeConsumer) -> String {
    match consumer {
        EdgeConsumer::Node(node) => node.to_string(),
        EdgeConsumer::Sink(ResultSinkKind::HostArrow) => "sink:host_arrow".to_string(),
        EdgeConsumer::Sink(ResultSinkKind::GpuResident) => "sink:gpu_resident".to_string(),
    }
}

pub(super) fn required_property_fact(property: RequiredProperty) -> &'static str {
    match property {
        RequiredProperty::ResidentFrame => "resident_frame",
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
        crate::plan::ResultSink::HostArrow { .. } => EdgeConsumer::Sink(ResultSinkKind::HostArrow),
        crate::plan::ResultSink::GpuResident { .. } => EdgeConsumer::Sink(ResultSinkKind::GpuResident),
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
    if matches!(plan.sink(), crate::plan::ResultSink::HostArrow { .. }) {
        contract_kinds.push(EdgeContractKind::ExplicitSinkMaterialization);
    }
    let required_properties = vec![RequiredProperty::ResidentFrame];
    let decision = edge_decision(plan, producer, consumer, &required_properties);
    EdgeContract { producer, consumer, contract_kinds, input_role: InputRole::SinkInput, required_properties, decision }
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
            if !edge.contract_kinds.iter().any(|kind| {
                matches!(
                    kind,
                    EdgeContractKind::SharedProducer { identity: SharedProducerIdentity::SameNodeIdAfterCse }
                )
            }) {
                edge.contract_kinds
                    .push(EdgeContractKind::SharedProducer { identity: SharedProducerIdentity::SameNodeIdAfterCse });
            }
        }
    }
}
