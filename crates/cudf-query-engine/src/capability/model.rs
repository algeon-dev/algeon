use super::*;

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

    pub fn compatibility_modeled(kind: ReservationKind) -> Self {
        let contract = compatibility_reservation_contract(kind);
        Self::modeled(kind, contract.memory_boundary, contract.required_contract, contract.quality)
    }

    pub fn compatibility_fact_row(&self) -> String {
        format!("{}:{}:{}", self.kind.as_str(), self.required_contract, self.compatibility_detail())
    }

    fn compatibility_detail(&self) -> &'static str {
        match self.kind {
            ReservationKind::SourceRead
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::OperatorOutput
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::BuildSide => "modeled_before_execution",
            ReservationKind::ProbeSide => "modeled_by_ordered_bounded_join_execution_path",
            ReservationKind::HashTable => "modeled_by_cudf_rs_pre_build_sizing",
            ReservationKind::FilteredProbeState => "modeled_by_cudf_rs_filtered_probe_state_observability",
            ReservationKind::MixedSizeData => "modeled_by_cudf_rs_retained_join_size_data_observability",
            ReservationKind::GatherMap => "modeled_by_cudf_rs_join_gather_map_sizing",
            ReservationKind::JoinOutput => "modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing",
            ReservationKind::FinalSort
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::Materialization => "modeled_by_explicit_sink_boundary",
            _ => "modeled_by_capability_analysis",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompatibilityReservationContract {
    memory_boundary: &'static str,
    required_contract: &'static str,
    quality: ContractQuality,
}

const fn compatibility_reservation_contract(kind: ReservationKind) -> CompatibilityReservationContract {
    match kind {
        ReservationKind::SourceRead => CompatibilityReservationContract {
            memory_boundary: "source_read",
            required_contract: "source_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::OperatorOutput => CompatibilityReservationContract {
            memory_boundary: "operator_output",
            required_contract: "aggregate_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::BuildSide => CompatibilityReservationContract {
            memory_boundary: "join_build_side",
            required_contract: "query_engine_build_side_source_reservation",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::ProbeSide => CompatibilityReservationContract {
            memory_boundary: "join_probe_side",
            required_contract: "query_engine_probe_side_source_reservation",
            quality: ContractQuality::QueryEngineOwnedOrdering,
        },
        ReservationKind::HashTable => CompatibilityReservationContract {
            memory_boundary: "join_hash_table",
            required_contract: "hash_join_pre_build_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::FilteredProbeState => CompatibilityReservationContract {
            memory_boundary: "join_filtered_probe_state",
            required_contract: "filtered_join_retained_probe_state",
            quality: ContractQuality::ObservedRetainedState,
        },
        ReservationKind::MixedSizeData => CompatibilityReservationContract {
            memory_boundary: "join_mixed_size_data",
            required_contract: "mixed_keyed_residual_or_conditional_full_retained_size_data",
            quality: ContractQuality::ObservedRetainedState,
        },
        ReservationKind::GatherMap => CompatibilityReservationContract {
            memory_boundary: "join_gather_map",
            required_contract: "join_gather_map_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::JoinOutput => CompatibilityReservationContract {
            memory_boundary: "join_output",
            required_contract: "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::FinalSort => CompatibilityReservationContract {
            memory_boundary: "join_pipeline_final_sort",
            required_contract: "final_sort_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::Materialization => CompatibilityReservationContract {
            memory_boundary: "host_materialization",
            required_contract: "query_engine_explicit_sink_materialization_reservation",
            quality: ContractQuality::QueryEngineOwnedBoundary,
        },
        ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => CompatibilityReservationContract {
            memory_boundary: "non_bounded_join_compatibility_boundary",
            required_contract: "non_bounded_join_compatibility_contract",
            quality: ContractQuality::ExactPreflight,
        },
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

/// Report-facing projection of the first capability rejection.
///
/// This is separate from family projection on purpose: rejected capability
/// analysis should be meaningful even when no legacy family name applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityRejectionProjection {
    pub scope: &'static str,
    pub reason: &'static str,
    pub facts: BTreeMap<&'static str, String>,
}

/// Side-channel capability analysis for one lowered native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    pub nodes: Vec<NodeCapability>,
    pub edges: Vec<EdgeContract>,
    pub invariants: Vec<SubgraphInvariant>,
    pub requirements: Vec<ReservationRequirement>,
    pub closed_multi_join_subgraph_admission: CapabilityDecision,
    pub multi_join_aggregate_over_join_subplan_topk_admission: CapabilityDecision,
    pub mixed_keyed_residual_join_resident_grouped_aggregate_input_admission: CapabilityDecision,
    pub hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission: CapabilityDecision,
    pub keyed_hash_join_resident_subplan_admission: CapabilityDecision,
    pub keyless_conditional_join_input_subplan_admission: CapabilityDecision,
    pub keyless_conditional_join_aggregate_over_join_subplan_inputs_admission: CapabilityDecision,
    pub multi_join_resident_grouped_aggregate_subquery_filter_admission: CapabilityDecision,
    pub compatibility_family_projection: Option<CompatibilityFamilyProjection>,
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

    pub fn compatibility_reservation_requirements(&self) -> Vec<ReservationRequirement> {
        let mut kinds = Vec::new();
        if self.has_requirement_kind(ReservationKind::SourceRead) {
            kinds.push(ReservationKind::SourceRead);
        }
        if self.has_requirement_kind(ReservationKind::OperatorOutput) {
            kinds.push(ReservationKind::OperatorOutput);
        }
        if self.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. })) {
            kinds.extend([
                ReservationKind::BuildSide,
                ReservationKind::HashTable,
                ReservationKind::ProbeSide,
                ReservationKind::FilteredProbeState,
                ReservationKind::MixedSizeData,
                ReservationKind::GatherMap,
                ReservationKind::JoinOutput,
            ]);
        }
        if self.has_requirement_kind(ReservationKind::FinalSort) {
            kinds.push(ReservationKind::FinalSort);
        }
        if self.has_requirement_kind(ReservationKind::Materialization) {
            kinds.push(ReservationKind::Materialization);
        }
        kinds.into_iter().map(ReservationRequirement::compatibility_modeled).collect()
    }

    pub fn compatibility_reservation_kinds_csv(&self) -> String {
        csv_join(self.compatibility_reservation_requirements().iter().map(|requirement| requirement.kind.as_str()))
    }

    pub fn compatibility_reservation_statuses_csv(&self) -> String {
        csv_join(
            self.compatibility_reservation_requirements()
                .iter()
                .map(|requirement| format!("{}:{}", requirement.kind.as_str(), requirement.status.as_str())),
        )
    }

    pub fn compatibility_reservation_requirements_csv(&self) -> String {
        csv_join(
            self.compatibility_reservation_requirements().iter().map(ReservationRequirement::compatibility_fact_row),
        )
    }

    pub fn compatibility_modeled_reservations_csv(&self) -> String {
        csv_join(self.compatibility_reservation_requirements().iter().filter_map(|requirement| {
            (requirement.status == ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }))
    }

    pub fn compatibility_first_rejection_projection(&self) -> Option<CompatibilityRejectionProjection> {
        self.nodes
            .iter()
            .find_map(|node| {
                Some(capability_rejection_projection(
                    "node",
                    node.executable.rejection_reason()?,
                    node.executable.rejection_facts()?,
                ))
            })
            .or_else(|| {
                self.edges.iter().find_map(|edge| {
                    Some(capability_rejection_projection(
                        "edge",
                        edge.decision.rejection_reason()?,
                        edge.decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                self.invariants.iter().find_map(|invariant| {
                    Some(capability_rejection_projection(
                        "invariant",
                        invariant.decision.rejection_reason()?,
                        invariant.decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                [
                    &self.closed_multi_join_subgraph_admission,
                    &self.multi_join_aggregate_over_join_subplan_topk_admission,
                    &self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
                    &self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
                    &self.keyed_hash_join_resident_subplan_admission,
                    &self.keyless_conditional_join_input_subplan_admission,
                    &self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
                    &self.multi_join_resident_grouped_aggregate_subquery_filter_admission,
                ]
                .into_iter()
                .find_map(|decision| {
                    Some(capability_rejection_projection(
                        "admission",
                        decision.rejection_reason()?,
                        decision.rejection_facts()?,
                    ))
                })
            })
            .or_else(|| {
                self.requirements.iter().find_map(|requirement| {
                    (requirement.status == ContractStatus::BlockedByLowerLevel).then(|| {
                        CompatibilityRejectionProjection {
                            scope: "lower_level_contract",
                            reason: "blocked_lower_level_contract",
                            facts: BTreeMap::from([
                                ("reservation_kind", requirement.kind.as_str().to_string()),
                                ("required_contract", requirement.required_contract.to_string()),
                                ("contract_status", requirement.status.as_str().to_string()),
                            ]),
                        }
                    })
                })
            })
    }

    fn has_requirement_kind(&self, kind: ReservationKind) -> bool {
        self.requirements.iter().any(|requirement| requirement.kind == kind)
    }
}

fn capability_rejection_projection(
    scope: &'static str,
    reason: &'static str,
    facts: &BTreeMap<&'static str, String>,
) -> CompatibilityRejectionProjection {
    CompatibilityRejectionProjection { scope, reason, facts: facts.clone() }
}

impl ContractStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
        }
    }
}

fn csv_join<I, T>(values: I) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    let mut output = String::new();
    for value in values {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}
