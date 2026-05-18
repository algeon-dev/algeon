use super::*;
use crate::plan::{EdgeDedupStrategy, EdgeWeightCombiner};

/// Whether the engine currently proves a bounded contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractStatus {
    /// The boundary has an admissible contract for strict bounded execution.
    Modeled,
    /// The boundary is understood, but a required cudf-rs/libcudf contract is
    /// not available yet.
    BlockedByLowerLevel,
    /// The boundary can be decided only after another capability proof lands.
    DeferredUntilPrerequisite,
    /// The owning node, edge, or invariant failed capability analysis.
    RejectedByCapability,
}

/// How precise a modeled bounded contract is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractQuality {
    /// A lower layer provides the bytes or bound before the operation runs.
    ExactPreflight,
    /// Authoritative metadata provides the bound before the operation runs.
    MetadataBound,
    /// A bounded budget is granted before execution, but exact preflight is
    /// waiting on a named lower-level API.
    HardBudgetedReportWithUnresolvedPreflight,
    /// The engine accounts for retained state after a bounded operation runs.
    ObservedRetainedState,
    /// The query engine owns the ordering guarantee rather than a memory size.
    QueryEngineOwnedOrdering,
    /// The query engine owns the boundary semantics rather than a lower layer.
    QueryEngineOwnedBoundary,
    /// The row represents a capability proof failure, not a memory contract.
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
    /// The reservation class enforced by runtime accounting.
    pub kind: ReservationKind,
    /// The most local graph scope that owns this requirement.
    pub scope: ReservationRequirementScope,
    /// Stable report label for the resource or execution boundary.
    pub memory_boundary: &'static str,
    /// Stable report label for the lower-level or query-engine contract.
    pub required_contract: &'static str,
    /// Whether the contract is currently admitted for strict bounded mode.
    pub status: ContractStatus,
    /// How strong the admitted contract evidence is.
    pub quality: ContractQuality,
}

/// A current strict preflight gap that can only be retired by replacing the
/// placeholder contract with a stronger cudf-rs/libcudf contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrictUnresolvedPreflightGap {
    pub kind: ReservationKind,
    pub required_contract: &'static str,
    pub unresolved_gap: &'static str,
    pub retirement_requirement: &'static str,
}

pub const STRICT_UNRESOLVED_PREFLIGHT_GAPS: &[StrictUnresolvedPreflightGap] = &[
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::SourceRead,
        required_contract: "source_contract",
        unresolved_gap: "libcudf_parquet_decoded_output_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf decoded-output preflight for planned row groups and projected columns",
    },
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::OperatorOutput,
        required_contract: "aggregate_contract",
        unresolved_gap: "libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf groupby aggregate output and temporary-workspace preflight",
    },
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::OperatorOutput,
        required_contract: "window_contract",
        unresolved_gap: "libcudf_window_output_temporary_workspace_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf window output and temporary-workspace preflight",
    },
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::OperatorOutput,
        required_contract: "edge_normalize_output_size_preflight",
        unresolved_gap: "libcudf_edge_normalize_output_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf edge normalization output and temporary-workspace preflight",
    },
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::FinalSort,
        required_contract: "final_sort_contract",
        unresolved_gap: "libcudf_final_sort_order_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf final sort or top-k output and temporary-workspace preflight",
    },
    StrictUnresolvedPreflightGap {
        kind: ReservationKind::JoinOutput,
        required_contract: "keyed_equality_join_output_size_preflight",
        unresolved_gap: "libcudf_equality_join_output_size_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf exact output row-count preflight for keyed inner/left/full equality join indices",
    },
];

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

    pub fn blocked_by_lower_level(
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
            status: ContractStatus::BlockedByLowerLevel,
            quality,
        }
    }
}

pub fn lower_level_unresolved_preflight_gap(kind: ReservationKind, required_contract: &str) -> Option<&'static str> {
    strict_unresolved_preflight_gap(kind, required_contract).map(|gap| gap.unresolved_gap)
}

pub fn lower_level_unresolved_preflight_gap_for_contract(required_contract: &str) -> Option<&'static str> {
    STRICT_UNRESOLVED_PREFLIGHT_GAPS
        .iter()
        .find(|gap| gap.required_contract == required_contract)
        .map(|gap| gap.unresolved_gap)
}

pub fn strict_unresolved_preflight_gap(
    kind: ReservationKind,
    required_contract: &str,
) -> Option<&'static StrictUnresolvedPreflightGap> {
    STRICT_UNRESOLVED_PREFLIGHT_GAPS.iter().find(|gap| gap.kind == kind && gap.required_contract == required_contract)
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
    /// The capability is fully proven at its owning scope.
    Satisfied,
    /// The capability failed at this scope; facts should name the local cause.
    Rejected { reason: &'static str, facts: BTreeMap<&'static str, String> },
    /// The capability depends on other graph-scope proofs.
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
    /// The native plan node this capability describes.
    pub node_id: NodeId,
    /// The node's operator-level shape, independent of whole-query families.
    pub operator: NodeOperator,
    /// Whether this node is executable before considering downstream edges.
    pub executable: CapabilityDecision,
    /// Properties this node makes available to consumers.
    pub produced_properties: Vec<ProducedProperty>,
    /// Reservations owned by this node itself.
    pub local_requirements: Vec<ReservationRequirement>,
}

/// Native operator shape captured without naming a whole bounded family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOperator {
    Source { format: SourceFormatCapability },
    Filter,
    Projection,
    Aggregate { grouping: AggregateGrouping, functions: Vec<AggregateFunction> },
    Window { function_count: usize },
    Sort { ordering: OrderingRequirement },
    Limit,
    Join { join_type: JoinType, key_count: usize, residual: ResidualRequirement, build_side: JoinSide },
    Union { input_count: usize },
    EdgeNormalize { add_reverse: bool, dedup: EdgeDedupStrategy, weight_combiner: EdgeWeightCombiner },
    GraphAlgorithm { algorithm: GraphAlgorithm },
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
    GroupingSets,
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
    /// Producer node for this dependency.
    pub producer: NodeId,
    /// Consumer node or final sink for this dependency.
    pub consumer: EdgeConsumer,
    /// Contract classes carried by this producer-consumer relationship.
    pub contract_kinds: Vec<EdgeContractKind>,
    /// Semantic input slot on the consumer.
    pub input_role: InputRole,
    /// Producer properties required by the consumer.
    pub required_properties: Vec<RequiredProperty>,
    /// Reservations owned by this edge boundary.
    pub reservation_requirements: Vec<ReservationRequirement>,
    /// Whether this edge satisfies its consumer contract.
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
    WindowInput,
    JoinBuildInput { side: JoinSide },
    JoinProbeInput { side: JoinSide },
    UnionInput { input_index: usize },
    SharedProducer { identity: SharedProducerIdentity },
    ResidentFrameFlow,
    JoinMaterialization,
    SortInput,
    GraphAlgorithmInput,
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
    WindowInput,
    SortInput,
    UnionInput { input_index: usize },
    GraphAlgorithmInput,
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
    /// Stable identifier within this capability analysis result.
    pub invariant_id: InvariantId,
    /// The kind of non-local graph property being proven.
    pub kind: SubgraphInvariantKind,
    /// Nodes covered by the invariant.
    pub nodes: Vec<NodeId>,
    /// Scheduling property required from the compiled pipeline.
    pub required_schedule: ScheduleConstraint,
    /// Resident resources whose lifetime participates in the proof.
    pub required_resources: Vec<ResidentResource>,
    /// Reservations owned by this subgraph boundary.
    pub reservation_requirements: Vec<ReservationRequirement>,
    /// Whether the invariant is satisfied for this graph.
    pub decision: CapabilityDecision,
}

/// Subgraph invariant vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgraphInvariantKind {
    /// The join's build input can be made resident before the probe side needs
    /// it, without a graph cycle through the join itself.
    BuildBeforeProbe { join_id: NodeId },
    /// The reachable join region below the root is closed over supported join,
    /// projection, filter, and source nodes.
    ClosedJoinSubtree { root_join_id: NodeId },
    /// The compiled graph can schedule all reachable joins in dependency order
    /// so build/probe residency requirements are not violated by join nesting.
    TopologicalBuildProbeSchedule,
    /// One resident producer is shared by multiple consumers and each consumer
    /// edge preserves that producer identity.
    SharedResidentSubplan,
    /// A scalar aggregate can consume a shared resident producer only when every
    /// aggregate expression derives from columns produced by that same shared
    /// node after transparent projections are peeled.
    AggregateDerivesFromSharedProducer,
    /// A final ordering barrier consumes the resident input after all upstream
    /// producers are complete and does not feed back into its input.
    FinalBarrierOrder,
    /// Post-root stages above a root join remain in the specific order required
    /// by the admitted envelope.
    PostRootStageOrder { root_join_id: NodeId },
    /// A projection wrapper can be ignored for residency only when both the
    /// peeled producer and the projection output are resident frames.
    ProjectionTransparentResidency,
    /// A filter/projection chain preserves residency only when every node in
    /// the connected unary chain consumes the previous resident output and also
    /// produces a resident frame.
    ResidentUnaryChainPreservesResidency,
    /// The residual predicate on the join references both join inputs, so the
    /// residual is a true join condition rather than a single-side filter.
    ResidualUsesBothInputs { join_id: NodeId },
    /// Every reachable resident producer remains valid for all of its admitted
    /// consumers, and no rejected producer-consumer edge is hidden by the
    /// broader compiled subgraph admission.
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
///
/// The node, edge, and invariant vectors are the primary proof surface.
/// The `*_admission` decisions are compiled-executor envelope gates derived
/// from those proofs; they must not become report compatibility families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    /// Node-local executable proofs.
    pub nodes: Vec<NodeCapability>,
    /// Producer-consumer edge proofs.
    pub edges: Vec<EdgeContract>,
    /// Non-local scheduling, lifetime, and shared-resource proofs.
    pub invariants: Vec<SubgraphInvariant>,
    /// Flattened reservation requirements used by reports and compiled gates.
    pub requirements: Vec<ReservationRequirement>,
    /// Strict root-join compiled-envelope admission.
    pub root_join_admission: CapabilityDecision,
    /// Strict single-join post-root compiled-envelope admission.
    pub single_join_post_root_pipeline_admission: CapabilityDecision,
    /// Strict closed multi-join compiled-envelope admission.
    pub closed_multi_join_subgraph_admission: CapabilityDecision,
    /// Strict multi-join aggregate/top-k compiled-envelope admission.
    pub multi_join_aggregate_over_join_subplan_topk_admission: CapabilityDecision,
    /// Strict mixed keyed/residual resident-aggregate compiled-envelope admission.
    pub mixed_keyed_residual_join_resident_grouped_aggregate_input_admission: CapabilityDecision,
    /// Strict hash-join resident-aggregate post-join/top-k envelope admission.
    pub hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission: CapabilityDecision,
    /// Strict shared resident-subplan compiled-envelope admission.
    pub keyed_hash_join_resident_subplan_admission: CapabilityDecision,
    /// Strict keyless conditional input-subplan envelope admission.
    pub keyless_conditional_join_input_subplan_admission: CapabilityDecision,
    /// Strict keyless aggregate-over-join-inputs envelope admission.
    pub keyless_conditional_join_aggregate_over_join_subplan_inputs_admission: CapabilityDecision,
    /// Strict multi-join resident aggregate subquery-filter envelope admission.
    pub multi_join_resident_grouped_aggregate_subquery_filter_admission: CapabilityDecision,
    /// Strict union compiled-envelope admission.
    pub union_pipeline_admission: CapabilityDecision,
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
        self.admits_root_join()
            || self.admits_single_join_post_root_pipeline()
            || self.admits_keyless_conditional_join_input_subplan_pipeline()
            || self.admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
            || self.admits_closed_multi_join_subgraph()
            || self.admits_multi_join_aggregate_over_join_subplan_topk_pipeline()
            || self.admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline()
            || self.admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
            || self.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline()
            || self.admits_keyed_hash_join_resident_subplan_pipeline()
            || self.union_pipeline_admission == CapabilityDecision::Satisfied
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
                    ("union_pipeline_admission", &self.union_pipeline_admission),
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
