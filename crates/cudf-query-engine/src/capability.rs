use std::collections::BTreeMap;

use crate::{
    bounded::{
        BoundedJoinPlanSupport, analyze_bounded_join_plan_support, bounded_multi_join_compatible,
        bounded_multi_join_mixed_keyed_residual_compatible, bounded_root_hash_join_compatible,
        bounded_root_join_compatible,
    },
    expr::AggregateFunction,
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan},
    runtime::ReservationKind,
    source::SourceNode,
};

/// Whether the engine currently proves a bounded contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractStatus {
    Modeled,
    BlockedByLowerLevel,
    DeferredUntilPrerequisite,
}

/// How precise a modeled bounded contract is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractQuality {
    ExactPreflight,
    MetadataBound,
    HardBudgetedReportWithUnresolvedPreflight,
    ObservedRetainedState,
    QueryEngineOwnedOrdering,
    QueryEngineOwnedBoundary,
}

/// One memory or execution-boundary requirement attached to the most local
/// capability owner that can state it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRequirement {
    pub kind: ReservationKind,
    pub memory_boundary: &'static str,
    pub required_contract: &'static str,
    pub status: ContractStatus,
    pub quality: ContractQuality,
}

impl ReservationRequirement {
    pub fn modeled(
        kind: ReservationKind,
        memory_boundary: &'static str,
        required_contract: &'static str,
        quality: ContractQuality,
    ) -> Self {
        Self { kind, memory_boundary, required_contract, status: ContractStatus::Modeled, quality }
    }
}

/// A symbolic dependency that a conditional capability is waiting for.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintRef {
    pub scope: ConstraintScope,
    pub name: &'static str,
}

impl ConstraintRef {
    pub const fn node(node_id: NodeId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Node(node_id), name }
    }

    pub const fn edge(producer: NodeId, consumer: EdgeConsumer, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Edge { producer, consumer }, name }
    }

    pub const fn invariant(invariant_id: InvariantId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Invariant(invariant_id), name }
    }
}

/// The graph scope where a constraint belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintScope {
    Node(NodeId),
    Edge { producer: NodeId, consumer: EdgeConsumer },
    Invariant(InvariantId),
}

/// Capability proof result for a node, edge, or invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    Satisfied,
    Rejected { reason: &'static str, facts: BTreeMap<&'static str, String> },
    Conditional { waiting_on: Vec<ConstraintRef> },
}

impl CapabilityDecision {
    fn conditional(scope: ConstraintScope, name: &'static str) -> Self {
        Self::Conditional { waiting_on: vec![ConstraintRef { scope, name }] }
    }

    fn rejected(reason: &'static str, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }
}

/// Node-local executable capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub operator: NodeOperator,
    pub executable: CapabilityDecision,
    pub produced_properties: Vec<ProducedProperty>,
    pub local_requirements: Vec<ReservationRequirement>,
}

/// Native operator shape captured without naming a whole bounded family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOperator {
    Source { format: SourceFormatCapability },
    Filter,
    Projection,
    Aggregate { grouping: AggregateGrouping, functions: Vec<AggregateFunction> },
    Sort { ordering: OrderingRequirement },
    Limit,
    Join { join_type: JoinType, key_count: usize, residual: ResidualRequirement, build_side: JoinSide },
}

/// Source format capability known by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormatCapability {
    LocalParquet,
    GpuResident,
}

/// Aggregate grouping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateGrouping {
    Scalar,
    Grouped,
}

/// Whether a node establishes an ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingRequirement {
    Required,
    None,
}

/// Whether a join has a residual predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualRequirement {
    None,
    Required,
}

/// Properties produced by a node for downstream edge contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducedProperty {
    ResidentFrame,
    GroupedFrame,
    ScalarFrame,
    OrderedFrame,
}

/// Producer-consumer capability for one dependency edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeContract {
    pub producer: NodeId,
    pub consumer: EdgeConsumer,
    pub input_role: InputRole,
    pub required_properties: Vec<RequiredProperty>,
    pub reservation_requirements: Vec<ReservationRequirement>,
    pub decision: CapabilityDecision,
}

/// Consumer endpoint for an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeConsumer {
    Node(NodeId),
    Sink(ResultSinkKind),
}

/// Result sink kinds as capability consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResultSinkKind {
    HostArrow,
    GpuResident,
}

/// Semantic role of an input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputRole {
    UnaryInput,
    AggregateInput,
    SortInput,
    LimitInput,
    JoinBuildInput { side: JoinSide },
    JoinProbeInput { side: JoinSide },
    SinkInput,
}

/// Property required from an edge producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequiredProperty {
    ResidentFrame,
}

/// Stable identifier for an invariant inside one analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvariantId(pub usize);

impl InvariantId {
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A non-local statement that cannot be proven from one node or edge alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphInvariant {
    pub invariant_id: InvariantId,
    pub kind: SubgraphInvariantKind,
    pub nodes: Vec<NodeId>,
    pub required_schedule: ScheduleConstraint,
    pub required_resources: Vec<ResidentResource>,
    pub decision: CapabilityDecision,
}

/// Subgraph invariant vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgraphInvariantKind {
    BuildBeforeProbe { join_id: NodeId },
    TopologicalBuildProbeSchedule,
    SharedResidentSubplan,
    AggregateDerivesFromSharedProducer,
    FinalBarrierOrder,
    ProjectionTransparentResidency,
    ResidualUsesBothInputs { join_id: NodeId },
    ResidentLifetimeDominatesConsumers,
}

/// Scheduling proof required by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleConstraint {
    None,
    BuildBeforeProbe,
    Topological,
    Barrier,
}

/// Resident resources whose lifetime or ordering is governed by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResource {
    NodeOutput(NodeId),
    JoinBuildInput { join_id: NodeId, input: NodeId },
    JoinProbeInput { join_id: NodeId, input: NodeId },
}

/// Transitional family projection for report/executor compatibility.
///
/// This is deliberately separate from the node, edge, and invariant primitives:
/// admission should migrate toward capability proofs, while legacy reports and
/// dispatch can continue projecting a family name during the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatibilityFamilyProjection {
    RootJoin,
    SingleJoinPostRootPipeline,
    KeylessConditionalJoinInputSubplanPipeline,
    KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    MultiJoinTreePostRootPipeline,
    MultiJoinAggregateOverJoinSubplanTopKPipeline,
    MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
    MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
    KeyedHashJoinResidentSubplanPipeline,
}

/// Side-channel capability analysis for one lowered native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    pub nodes: Vec<NodeCapability>,
    pub edges: Vec<EdgeContract>,
    pub invariants: Vec<SubgraphInvariant>,
    pub requirements: Vec<ReservationRequirement>,
    pub compatibility_family_projection: Option<CompatibilityFamilyProjection>,
}

impl CapabilityAnalysis {
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }
}

impl QueryPlan {
    /// Runs capability analysis without changing bounded admission or execution.
    pub fn capability_analysis(&self) -> CapabilityAnalysis {
        analyze_query_capabilities(self)
    }
}

/// Runs capability analysis without changing bounded admission or execution.
pub fn analyze_query_capabilities(plan: &QueryPlan) -> CapabilityAnalysis {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node.id, &node.kind)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));

    let invariants = analyze_invariants(plan);
    let requirements = collect_requirements(&nodes, &edges);
    CapabilityAnalysis {
        nodes,
        edges,
        invariants,
        requirements,
        compatibility_family_projection: compatibility_family_projection(plan),
    }
}

fn analyze_node(_plan: &QueryPlan, node_id: NodeId, kind: &PlanNode) -> NodeCapability {
    let operator = node_operator(kind);
    let executable = node_executable_decision(node_id, kind);
    let produced_properties = produced_properties(kind);
    let local_requirements = node_local_requirements(kind);
    NodeCapability { node_id, operator, executable, produced_properties, local_requirements }
}

fn node_operator(kind: &PlanNode) -> NodeOperator {
    match kind {
        PlanNode::Source(source) => NodeOperator::Source { format: source_format_capability(source) },
        PlanNode::Filter(_) => NodeOperator::Filter,
        PlanNode::Projection(_) => NodeOperator::Projection,
        PlanNode::Aggregate(aggregate) => NodeOperator::Aggregate {
            grouping: if aggregate.spec.group_exprs.is_empty() {
                AggregateGrouping::Scalar
            } else {
                AggregateGrouping::Grouped
            },
            functions: aggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect(),
        },
        PlanNode::Sort(sort) => NodeOperator::Sort {
            ordering: if sort.spec.expressions.is_empty() {
                OrderingRequirement::None
            } else {
                OrderingRequirement::Required
            },
        },
        PlanNode::Limit(_) => NodeOperator::Limit,
        PlanNode::Join(join) => NodeOperator::Join {
            join_type: join.spec.join_type,
            key_count: join.spec.keys.len(),
            residual: if join.spec.residual_predicate.is_some() {
                ResidualRequirement::Required
            } else {
                ResidualRequirement::None
            },
            build_side: join.spec.build_side,
        },
    }
}

fn source_format_capability(source: &SourceNode) -> SourceFormatCapability {
    match source {
        SourceNode::LocalParquet(_) => SourceFormatCapability::LocalParquet,
        SourceNode::GpuResident(_) => SourceFormatCapability::GpuResident,
    }
}

fn node_executable_decision(node_id: NodeId, kind: &PlanNode) -> CapabilityDecision {
    match kind {
        PlanNode::Join(join) if join_node_admitted_subset(join) => CapabilityDecision::Satisfied,
        PlanNode::Join(join) => CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ("build_side", join.spec.build_side.to_string()),
            ],
        ),
        PlanNode::Filter(_) | PlanNode::Projection(_) => {
            CapabilityDecision::conditional(ConstraintScope::Node(node_id), "expression_capability_extraction")
        },
        _ => CapabilityDecision::Satisfied,
    }
}

fn join_node_admitted_subset(join: &JoinNode) -> bool {
    bounded_root_join_compatible(join)
        || bounded_multi_join_compatible(join)
        || bounded_multi_join_mixed_keyed_residual_compatible(join)
}

fn produced_properties(kind: &PlanNode) -> Vec<ProducedProperty> {
    let mut properties = vec![ProducedProperty::ResidentFrame];
    match kind {
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            properties.push(ProducedProperty::ScalarFrame);
        },
        PlanNode::Aggregate(_) => properties.push(ProducedProperty::GroupedFrame),
        PlanNode::Sort(_) => properties.push(ProducedProperty::OrderedFrame),
        _ => {},
    }
    properties
}

fn node_local_requirements(kind: &PlanNode) -> Vec<ReservationRequirement> {
    match kind {
        PlanNode::Aggregate(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "aggregate_contract",
            "cudf_rs_hard_budgeted_report_with_unresolved_preflight",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
        PlanNode::Join(join) if bounded_root_hash_join_compatible(join) => vec![ReservationRequirement::modeled(
            ReservationKind::HashTable,
            "hash_join_pre_build_reservation_size_bytes",
            "cudf_rs_pre_build_sizing",
            ContractQuality::ExactPreflight,
        )],
        PlanNode::Join(join) if join.spec.residual_predicate.is_some() => vec![ReservationRequirement::modeled(
            ReservationKind::MixedSizeData,
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            "cudf_rs_retained_join_size_data_observability",
            ContractQuality::ObservedRetainedState,
        )],
        _ => Vec::new(),
    }
}

fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let reservation_requirements = edge_requirements(plan, *producer, kind, input_role);
            let decision = if plan.node(*producer).is_some() {
                CapabilityDecision::Satisfied
            } else {
                CapabilityDecision::rejected(
                    "missing_edge_producer",
                    [("producer", producer.to_string()), ("consumer", consumer_id.to_string())],
                )
            };
            EdgeContract {
                producer: *producer,
                consumer,
                input_role,
                required_properties: vec![RequiredProperty::ResidentFrame],
                reservation_requirements,
                decision,
            }
        })
        .collect()
}

fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
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

fn edge_requirements(
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
            "query_engine_build_side_source_reservation",
            "query_engine_resident_build_input_before_execution",
            ContractQuality::ExactPreflight,
        ));
    }
    append_producer_output_edge_requirements(plan, producer, &mut requirements);
    if matches!(consumer_kind, PlanNode::Sort(_)) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::FinalSort,
            "final_sort_contract",
            "cudf_rs_hard_budgeted_report_with_unresolved_preflight",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
    requirements
}

fn append_source_read_edge_requirement(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
    {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::SourceRead,
            "source_contract",
            "cudf_rs_hard_budgeted_report_with_unresolved_preflight",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
}

fn append_producer_output_edge_requirements(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Join(_))) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::GatherMap,
            "join_gather_map_reservation_size_bytes",
            "cudf_rs_join_gather_map_sizing",
            ContractQuality::ExactPreflight,
        ));
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::JoinOutput,
            "dataframe_gather_output_reservation_size_bytes",
            "cudf_rs_dataframe_gather_output_sizing",
            ContractQuality::ExactPreflight,
        ));
    }
}

fn analyze_sink_edge(plan: &QueryPlan) -> EdgeContract {
    let producer = plan.sink().input();
    let consumer = match plan.sink() {
        crate::plan::ResultSink::HostArrow { .. } => EdgeConsumer::Sink(ResultSinkKind::HostArrow),
        crate::plan::ResultSink::GpuResident { .. } => EdgeConsumer::Sink(ResultSinkKind::GpuResident),
    };
    let mut reservation_requirements = Vec::new();
    append_source_read_edge_requirement(plan, producer, &mut reservation_requirements);
    append_producer_output_edge_requirements(plan, producer, &mut reservation_requirements);
    reservation_requirements.push(ReservationRequirement::modeled(
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
        "explicit_sink_boundary",
        ContractQuality::QueryEngineOwnedBoundary,
    ));
    EdgeContract {
        producer,
        consumer,
        input_role: InputRole::SinkInput,
        required_properties: vec![RequiredProperty::ResidentFrame],
        reservation_requirements,
        decision: CapabilityDecision::Satisfied,
    }
}

fn analyze_invariants(plan: &QueryPlan) -> Vec<SubgraphInvariant> {
    let mut invariants = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
        let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::BuildBeforeProbe { join_id: node.id },
            nodes: vec![build_input, probe_input, node.id],
            required_schedule: ScheduleConstraint::BuildBeforeProbe,
            required_resources: vec![
                ResidentResource::JoinBuildInput { join_id: node.id, input: build_input },
                ResidentResource::JoinProbeInput { join_id: node.id, input: probe_input },
            ],
            decision: CapabilityDecision::conditional(
                ConstraintScope::Invariant(invariant_id),
                "bounded_join_scheduler_build_before_probe_proof",
            ),
        });
        if join.spec.residual_predicate.is_some() {
            let invariant_id = InvariantId(invariants.len());
            invariants.push(SubgraphInvariant {
                invariant_id,
                kind: SubgraphInvariantKind::ResidualUsesBothInputs { join_id: node.id },
                nodes: vec![join.inputs[0], join.inputs[1], node.id],
                required_schedule: ScheduleConstraint::None,
                required_resources: Vec::new(),
                decision: CapabilityDecision::conditional(
                    ConstraintScope::Invariant(invariant_id),
                    "residual_input_coverage_proof",
                ),
            });
        }
    }
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
            decision: CapabilityDecision::conditional(
                ConstraintScope::Invariant(invariant_id),
                "multi_join_topological_scheduler_proof",
            ),
        });
    }
    invariants
}

fn collect_requirements(nodes: &[NodeCapability], edges: &[EdgeContract]) -> Vec<ReservationRequirement> {
    nodes
        .iter()
        .flat_map(|node| node.local_requirements.iter())
        .chain(edges.iter().flat_map(|edge| edge.reservation_requirements.iter()))
        .cloned()
        .collect()
}

fn compatibility_family_projection(plan: &QueryPlan) -> Option<CompatibilityFamilyProjection> {
    match analyze_bounded_join_plan_support(plan) {
        BoundedJoinPlanSupport::RootJoin => Some(CompatibilityFamilyProjection::RootJoin),
        BoundedJoinPlanSupport::SingleJoinPostRootPipeline => {
            Some(CompatibilityFamilyProjection::SingleJoinPostRootPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinInputSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinInputSubplanPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinTreePostRootPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinAggregateOverJoinSubplanTopKPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinAggregateOverJoinSubplanTopKPipeline)
        },
        BoundedJoinPlanSupport::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline => {
            Some(CompatibilityFamilyProjection::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline)
        },
        BoundedJoinPlanSupport::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline => {
            Some(CompatibilityFamilyProjection::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline)
        },
        BoundedJoinPlanSupport::KeyedHashJoinResidentSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline)
        },
        BoundedJoinPlanSupport::Unsupported => None,
    }
}
