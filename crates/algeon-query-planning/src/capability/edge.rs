//! Per-edge capability analysis.
//!
//! Each producer→consumer edge in the plan carries the residency contracts the
//! consumer places on the producer (resident source input, join build or probe
//! input, union position, sink materialization) and one decision. Rejections
//! here are local to the edge; whole-plan admission lives in [`crate::admission`].

use super::{
    JoinSide, NodeId, PlanNode, QueryPlan, ResultSink,
    graph::consumer_nodes,
    model::{
        CapabilityDecision, EdgeConsumer, EdgeContract, EdgeContractKind, InputRole, ResultSinkKind, VectorInputRole,
    },
    node::source_format_capability,
    rejection::Reason,
};

/// One edge per input slot of `kind`, in input order.
///
/// Edge analysis infers no set operation from the producers around a join. A
/// native left anti join keeps every probe row with no build match, duplicates
/// included, and reads only key existence from its build input, so its edges
/// carry the base decision like every other join edge.
pub(super) fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let contract_kinds = edge_contract_kinds(plan, *producer, consumer, kind, input_role);
            let decision = edge_decision(plan, *producer, consumer);
            EdgeContract { producer: *producer, consumer, contract_kinds, input_role, decision }
        })
        .collect()
}

/// Contract kinds one edge carries: producer facts first (source format, join
/// materialization), then the consumer's role facts. The shared-producer kind
/// is added later by [`mark_shared_producer_edge_contracts`], once every edge
/// exists.
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
    if let InputRole::MlAlgorithmInput { role } = input_role {
        kinds.push(EdgeContractKind::MlAlgorithmInput { role });
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

/// Base decision for an edge. Plan construction guarantees every producer
/// exists, so a rejection here means the analysis ran over an unvalidated node
/// list; residency itself is judged by the compiled pipeline contracts.
pub(super) fn edge_decision(plan: &QueryPlan, producer: NodeId, consumer: EdgeConsumer) -> CapabilityDecision {
    if plan.node(producer).is_none() {
        return CapabilityDecision::rejected(
            Reason::MissingEdgeProducer,
            [("producer", producer.to_string()), ("consumer", edge_consumer_fact(consumer))],
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
        PlanNode::MlAlgorithm(_) => "ml_algorithm",
    }
}

/// Semantic role of input `input_index` of `kind`. A join's roles follow its
/// `build_side`: input 0 is the left side, and whichever side the spec builds
/// is the build input.
pub(crate) fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
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
        PlanNode::MlAlgorithm(ml) => InputRole::MlAlgorithmInput { role: ml.spec.params.relation_role(input_index) },
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

/// The edge from the root node to the sink. A deferred host sink is analyzed
/// as a GPU-resident consumer because execution treats it as one; only a host
/// Arrow sink carries the explicit materialization contract.
pub(super) fn analyze_sink_edge(plan: &QueryPlan) -> EdgeContract {
    let producer = plan.sink().input();
    let consumer = match plan.sink() {
        ResultSink::HostArrow { .. } => EdgeConsumer::Sink(ResultSinkKind::HostArrow),
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => {
            EdgeConsumer::Sink(ResultSinkKind::GpuResident)
        },
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

/// Adds the shared-producer kind to every node-consumer edge of a producer
/// with more than one consumer. Runs after all input edges exist because it
/// needs the full consumer set; the sink edge is never marked. Idempotent.
pub(super) fn mark_shared_producer_edge_contracts(plan: &QueryPlan, edges: &mut [EdgeContract]) {
    let consumers = consumer_nodes(plan);
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
