use super::*;

/// Whether the engine currently proves a bounded contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractStatus {
    Modeled,
    BlockedByLowerLevel,
    DeferredUntilPrerequisite,
    RejectedByCapability,
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
    CapabilityProof,
}

impl ContractQuality {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactPreflight => "exact_preflight",
            Self::MetadataBound => "metadata_bound",
            Self::HardBudgetedReportWithUnresolvedPreflight => "hard_budgeted_report_with_unresolved_preflight",
            Self::ObservedRetainedState => "observed_retained_state",
            Self::QueryEngineOwnedOrdering => "query_engine_owned_ordering",
            Self::QueryEngineOwnedBoundary => "query_engine_owned_boundary",
            Self::CapabilityProof => "capability_proof",
        }
    }
}

/// The most local graph scope that owns a reservation requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationRequirementScope {
    Node,
    Edge,
    Subgraph,
}

impl ReservationRequirementScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::Subgraph => "subgraph",
        }
    }
}

/// One memory or execution-boundary requirement attached to the most local
/// capability owner that can state it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRequirement {
    pub kind: ReservationKind,
    pub scope: ReservationRequirementScope,
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
        Self {
            kind,
            scope: reservation_requirement_scope(kind),
            memory_boundary,
            required_contract,
            status: ContractStatus::Modeled,
            quality,
        }
    }
}

pub fn lower_level_unresolved_preflight_gap(kind: ReservationKind, required_contract: &str) -> Option<&'static str> {
    if matches!(kind, ReservationKind::SourceRead | ReservationKind::OperatorOutput | ReservationKind::FinalSort) {
        lower_level_unresolved_preflight_gap_for_contract(required_contract)
    } else {
        None
    }
}

pub fn lower_level_unresolved_preflight_gap_for_contract(required_contract: &str) -> Option<&'static str> {
    match required_contract {
        "source_contract" => Some("libcudf_parquet_decoded_output_preflight_unavailable"),
        "aggregate_contract" => Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"),
        "final_sort_contract" => Some("libcudf_final_sort_order_preflight_unavailable"),
        _ => None,
    }
}

pub const fn reservation_requirement_scope(kind: ReservationKind) -> ReservationRequirementScope {
    match kind {
        ReservationKind::OperatorOutput
        | ReservationKind::HashTable
        | ReservationKind::FilteredProbeState
        | ReservationKind::MixedSizeData
        | ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => ReservationRequirementScope::Node,
        ReservationKind::SourceRead
        | ReservationKind::BuildSide
        | ReservationKind::ProbeSide
        | ReservationKind::GatherMap
        | ReservationKind::JoinOutput
        | ReservationKind::FinalSort
        | ReservationKind::Materialization => ReservationRequirementScope::Edge,
        ReservationKind::CapabilityContract => ReservationRequirementScope::Subgraph,
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
    pub(super) fn rejected(reason: &'static str, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }

    pub const fn rejection_reason(&self) -> Option<&'static str> {
        match self {
            Self::Rejected { reason, .. } => Some(reason),
            Self::Satisfied | Self::Conditional { .. } => None,
        }
    }

    pub fn rejection_facts(&self) -> Option<&BTreeMap<&'static str, String>> {
        match self {
            Self::Rejected { facts, .. } => Some(facts),
            Self::Satisfied | Self::Conditional { .. } => None,
        }
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
    pub contract_kinds: Vec<EdgeContractKind>,
    pub input_role: InputRole,
    pub required_properties: Vec<RequiredProperty>,
    pub reservation_requirements: Vec<ReservationRequirement>,
    pub decision: CapabilityDecision,
}

/// Producer-consumer contract classes attached to one edge.
///
/// A single graph edge can carry more than one contract. For example, a
/// local-Parquet source feeding a join build side owns both the source-read
/// resident input contract and the build-side residency contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeContractKind {
    ResidentSourceInput { format: SourceFormatCapability },
    ResidentAggregateInput,
    JoinBuildInput { side: JoinSide },
    JoinProbeInput { side: JoinSide },
    SharedProducer { identity: SharedProducerIdentity },
    ResidentFrameFlow,
    JoinMaterialization,
    SortInput,
    ExplicitSinkMaterialization,
}

/// Identity proof used for resident producers with multiple consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedProducerIdentity {
    SameNodeIdAfterCse,
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
    pub reservation_requirements: Vec<ReservationRequirement>,
    pub decision: CapabilityDecision,
}

/// Subgraph invariant vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgraphInvariantKind {
    BuildBeforeProbe { join_id: NodeId },
    ClosedJoinSubtree { root_join_id: NodeId },
    TopologicalBuildProbeSchedule,
    SharedResidentSubplan,
    AggregateDerivesFromSharedProducer,
    FinalBarrierOrder,
    PostRootStageOrder { root_join_id: NodeId },
    ProjectionTransparentResidency,
    ResidentUnaryChainPreservesResidency,
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

/// Report-facing projection of the first capability rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRejectionProjection {
    pub owner: CapabilityRejectionOwner,
    pub contract_scope: ReservationRequirementScope,
    pub memory_boundary: &'static str,
    pub reason: &'static str,
    pub facts: BTreeMap<&'static str, String>,
}

/// Local capability owner projected into strict bounded contract diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityRejectionOwner {
    Node(NodeId),
    Edge { producer: NodeId, consumer: EdgeConsumer },
    Invariant(InvariantId),
    Admission(&'static str),
}

/// Capability analysis for one lowered native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    pub nodes: Vec<NodeCapability>,
    pub edges: Vec<EdgeContract>,
    pub invariants: Vec<SubgraphInvariant>,
    pub requirements: Vec<ReservationRequirement>,
    pub root_join_admission: CapabilityDecision,
    pub single_join_post_root_pipeline_admission: CapabilityDecision,
    pub closed_multi_join_subgraph_admission: CapabilityDecision,
    pub multi_join_aggregate_over_join_subplan_topk_admission: CapabilityDecision,
    pub mixed_keyed_residual_join_resident_grouped_aggregate_input_admission: CapabilityDecision,
    pub hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission: CapabilityDecision,
    pub keyed_hash_join_resident_subplan_admission: CapabilityDecision,
    pub keyless_conditional_join_input_subplan_admission: CapabilityDecision,
    pub keyless_conditional_join_aggregate_over_join_subplan_inputs_admission: CapabilityDecision,
    pub multi_join_resident_grouped_aggregate_subquery_filter_admission: CapabilityDecision,
}

impl CapabilityAnalysis {
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }

    pub fn admits_closed_multi_join_subgraph(&self) -> bool {
        self.closed_multi_join_subgraph_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_root_join(&self) -> bool {
        self.root_join_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_single_join_post_root_pipeline(&self) -> bool {
        self.single_join_post_root_pipeline_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_multi_join_aggregate_over_join_subplan_topk_pipeline(&self) -> bool {
        self.multi_join_aggregate_over_join_subplan_topk_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(&self) -> bool {
        self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(&self) -> bool {
        self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission
            == CapabilityDecision::Satisfied
    }

    pub fn admits_keyed_hash_join_resident_subplan_pipeline(&self) -> bool {
        self.keyed_hash_join_resident_subplan_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_keyless_conditional_join_input_subplan_pipeline(&self) -> bool {
        self.keyless_conditional_join_input_subplan_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(&self) -> bool {
        self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(&self) -> bool {
        self.multi_join_resident_grouped_aggregate_subquery_filter_admission == CapabilityDecision::Satisfied
    }

    pub fn admits_strict_bounded_join(&self) -> bool {
        !self.has_rejections()
            && (self.admits_root_join()
                || self.admits_single_join_post_root_pipeline()
                || self.admits_keyless_conditional_join_input_subplan_pipeline()
                || self.admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
                || self.admits_closed_multi_join_subgraph()
                || self.admits_multi_join_aggregate_over_join_subplan_topk_pipeline()
                || self.admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline()
                || self.admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
                || self.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline()
                || self.admits_keyed_hash_join_resident_subplan_pipeline())
    }

    pub fn first_rejection_projection(&self) -> Option<CapabilityRejectionProjection> {
        self.nodes
            .iter()
            .find_map(|node| {
                Some(capability_rejection_projection(
                    CapabilityRejectionOwner::Node(node.node_id),
                    ReservationRequirementScope::Node,
                    "node_capability",
                    node.executable.rejection_reason()?,
                    node.executable.rejection_facts()?,
                ))
            })
            .or_else(|| {
                self.edges.iter().find_map(|edge| {
                    Some(capability_rejection_projection(
                        CapabilityRejectionOwner::Edge { producer: edge.producer, consumer: edge.consumer },
                        ReservationRequirementScope::Edge,
                        "edge_capability",
                        edge.decision.rejection_reason()?,
                        edge.decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                self.invariants.iter().find_map(|invariant| {
                    Some(capability_rejection_projection(
                        CapabilityRejectionOwner::Invariant(invariant.invariant_id),
                        ReservationRequirementScope::Subgraph,
                        "subgraph_capability",
                        invariant.decision.rejection_reason()?,
                        invariant.decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                [
                    ("root_join_admission", &self.root_join_admission),
                    ("single_join_post_root_pipeline_admission", &self.single_join_post_root_pipeline_admission),
                    ("closed_multi_join_subgraph_admission", &self.closed_multi_join_subgraph_admission),
                    (
                        "multi_join_aggregate_over_join_subplan_topk_admission",
                        &self.multi_join_aggregate_over_join_subplan_topk_admission,
                    ),
                    (
                        "mixed_keyed_residual_join_resident_grouped_aggregate_input_admission",
                        &self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
                    ),
                    (
                        "hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission",
                        &self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
                    ),
                    ("keyed_hash_join_resident_subplan_admission", &self.keyed_hash_join_resident_subplan_admission),
                    (
                        "keyless_conditional_join_input_subplan_admission",
                        &self.keyless_conditional_join_input_subplan_admission,
                    ),
                    (
                        "keyless_conditional_join_aggregate_over_join_subplan_inputs_admission",
                        &self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
                    ),
                    (
                        "multi_join_resident_grouped_aggregate_subquery_filter_admission",
                        &self.multi_join_resident_grouped_aggregate_subquery_filter_admission,
                    ),
                ]
                .into_iter()
                .find_map(|(admission, decision)| {
                    Some(capability_rejection_projection(
                        CapabilityRejectionOwner::Admission(admission),
                        ReservationRequirementScope::Subgraph,
                        "admission_capability",
                        decision.rejection_reason()?,
                        decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                self.requirements.iter().find_map(|requirement| {
                    (requirement.status == ContractStatus::BlockedByLowerLevel).then(|| CapabilityRejectionProjection {
                        owner: CapabilityRejectionOwner::Admission("lower_level_contract"),
                        contract_scope: ReservationRequirementScope::Subgraph,
                        memory_boundary: "lower_level_contract",
                        reason: "blocked_lower_level_contract",
                        facts: BTreeMap::from([
                            ("reservation_kind", requirement.kind.as_str().to_string()),
                            ("required_contract", requirement.required_contract.to_string()),
                            ("contract_status", requirement.status.as_str().to_string()),
                        ]),
                    })
                })
            })
    }
}

fn capability_rejection_projection(
    owner: CapabilityRejectionOwner,
    contract_scope: ReservationRequirementScope,
    memory_boundary: &'static str,
    reason: &'static str,
    facts: &BTreeMap<&'static str, String>,
) -> CapabilityRejectionProjection {
    CapabilityRejectionProjection { owner, contract_scope, memory_boundary, reason, facts: facts.clone() }
}

impl ContractStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
            Self::RejectedByCapability => "rejected_by_capability",
        }
    }
}
