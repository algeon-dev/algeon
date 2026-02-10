use super::{graph::*, model::*, node::*, *};

pub(super) fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let contract_kinds = edge_contract_kinds(plan, *producer, consumer, kind, input_role);
            let required_properties = vec![RequiredProperty::ResidentFrame];
            let reservation_requirements = edge_requirements(plan, *producer, kind, input_role);
            let decision = edge_decision(plan, *producer, consumer, &required_properties);
            EdgeContract {
                producer: *producer,
                consumer,
                contract_kinds,
                input_role,
                required_properties,
                reservation_requirements,
                decision,
            }
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
    match input_role {
        InputRole::JoinBuildInput { side } => kinds.push(EdgeContractKind::JoinBuildInput { side }),
        InputRole::JoinProbeInput { side } => kinds.push(EdgeContractKind::JoinProbeInput { side }),
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
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}

pub(super) fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
    match kind {
        PlanNode::Aggregate(_) => InputRole::AggregateInput,
        PlanNode::Sort(_) => InputRole::SortInput,
        PlanNode::Limit(_) => InputRole::LimitInput,
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

pub(super) fn edge_requirements(
    plan: &QueryPlan,
    producer: NodeId,
    consumer_kind: &PlanNode,
    input_role: InputRole,
) -> Vec<ReservationRequirement> {
    let mut requirements = Vec::new();
    append_source_read_edge_requirement(plan, producer, &mut requirements);
    if matches!(input_role, InputRole::JoinBuildInput { .. }) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::BuildSide,
            "join_build_side",
            "query_engine_build_side_source_reservation",
            ContractQuality::ExactPreflight,
        ));
    }
    if matches!(input_role, InputRole::JoinProbeInput { .. }) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::ProbeSide,
            "join_probe_side",
            "query_engine_probe_side_source_reservation",
            ContractQuality::QueryEngineOwnedOrdering,
        ));
    }
    append_producer_output_edge_requirements(plan, producer, &mut requirements);
    if matches!(consumer_kind, PlanNode::Sort(_)) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::FinalSort,
            "join_pipeline_final_sort",
            "final_sort_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
    requirements
}

pub(super) fn append_source_read_edge_requirement(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
    {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::SourceRead,
            "source_read",
            "source_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
}

pub(super) fn append_producer_output_edge_requirements(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if let Some(PlanNode::Join(join)) = plan.node(producer).map(|node| &node.kind) {
        let one_shot_equi = matches!(
            keyed_hash_join_strategy_decision_for_plan(plan, producer, join),
            Some(KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::OneShotEqui, .. })
        );
        let gather_map_requirement = if one_shot_equi {
            ReservationRequirement::modeled(
                ReservationKind::GatherMap,
                "join_one_shot_indices",
                "one_shot_join_indices_observed_retained_bytes",
                ContractQuality::ObservedRetainedState,
            )
        } else {
            ReservationRequirement::modeled(
                ReservationKind::GatherMap,
                "join_gather_map",
                "join_gather_map_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            )
        };
        requirements.push(gather_map_requirement);
        if one_shot_equi {
            requirements.push(ReservationRequirement::modeled(
                ReservationKind::JoinOutput,
                "join_output",
                "one_shot_equality_join_output_size_preflight",
                ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            ));
        } else {
            requirements.push(ReservationRequirement::modeled(
                ReservationKind::JoinOutput,
                "join_output",
                "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            ));
        }
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
    contract_kinds.push(EdgeContractKind::ExplicitSinkMaterialization);
    let required_properties = vec![RequiredProperty::ResidentFrame];
    let mut reservation_requirements = Vec::new();
    append_source_read_edge_requirement(plan, producer, &mut reservation_requirements);
    append_producer_output_edge_requirements(plan, producer, &mut reservation_requirements);
    reservation_requirements.push(ReservationRequirement::modeled(
        ReservationKind::Materialization,
        "host_materialization",
        "query_engine_explicit_sink_materialization_reservation",
        ContractQuality::QueryEngineOwnedBoundary,
    ));
    let decision = edge_decision(plan, producer, consumer, &required_properties);
    EdgeContract {
        producer,
        consumer,
        contract_kinds,
        input_role: InputRole::SinkInput,
        required_properties,
        reservation_requirements,
        decision,
    }
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
