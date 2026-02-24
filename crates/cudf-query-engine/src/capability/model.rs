//! Capability-analysis model for native query admission.
//!
//! This module is the public vocabulary for local node, edge, and subgraph
//! proofs. Rejection reasons and fact keys are report-facing stable strings;
//! add new labels instead of reusing old whole-plan family names.
#![deny(missing_docs)]

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
    /// Stable snake-case label emitted in contract diagnostics.
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
    /// Requirement owned by one native plan node.
    Node,
    /// Requirement owned by one producer-consumer boundary.
    Edge,
    /// Requirement owned by a non-local graph proof or compiled envelope.
    Subgraph,
}

impl ReservationRequirementScope {
    /// Stable snake-case label emitted in contract diagnostics.
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
    /// Reservation kind affected by the unresolved lower-level contract.
    pub kind: ReservationKind,
    /// Stable contract label that currently relies on a placeholder proof.
    pub required_contract: &'static str,
    /// Stable report code naming the missing cudf-rs/libcudf preflight.
    pub unresolved_gap: &'static str,
    /// Human-readable condition required to remove this gap from strict mode.
    pub retirement_requirement: &'static str,
}

/// Known strict-mode preflight gaps that require lower-level cudf-rs/libcudf work.
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
        required_contract: "regexp_replace_expression_output_contract",
        unresolved_gap: "libcudf_regexp_replace_output_preflight_unavailable",
        retirement_requirement: "cudf-rs/libcudf regexp_replace output and temporary-workspace preflight",
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

/// Rejection code for large grouping-set/rank-window scans whose blocking
/// aggregate, rank-sort, window, and optional final top-k stages still rely on
/// unresolved hard-budget contracts.
pub const BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS: &str =
    "blocking_rollup_rank_full_scan_requires_exact_contracts";

impl ReservationRequirement {
    /// Creates an admitted strict-bounded reservation requirement.
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

    /// Creates a requirement whose strict proof waits on a lower-level contract.
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

/// Returns the stable unresolved-gap code for a reservation contract, if known.
pub fn lower_level_unresolved_preflight_gap(kind: ReservationKind, required_contract: &str) -> Option<&'static str> {
    strict_unresolved_preflight_gap(kind, required_contract).map(|gap| gap.unresolved_gap)
}

/// Returns the stable unresolved-gap code for a contract label, if known.
pub fn lower_level_unresolved_preflight_gap_for_contract(required_contract: &str) -> Option<&'static str> {
    STRICT_UNRESOLVED_PREFLIGHT_GAPS
        .iter()
        .find(|gap| gap.required_contract == required_contract)
        .map(|gap| gap.unresolved_gap)
}

/// Looks up the full strict-mode unresolved preflight descriptor.
pub fn strict_unresolved_preflight_gap(
    kind: ReservationKind,
    required_contract: &str,
) -> Option<&'static StrictUnresolvedPreflightGap> {
    STRICT_UNRESOLVED_PREFLIGHT_GAPS.iter().find(|gap| gap.kind == kind && gap.required_contract == required_contract)
}

/// Maps a reservation kind to the most local capability scope that owns it.
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
    /// Graph scope that owns the prerequisite proof.
    pub scope: ConstraintScope,
    /// Stable local constraint name within that scope.
    pub name: &'static str,
}

impl ConstraintRef {
    /// Creates a prerequisite reference owned by a node.
    pub const fn node(node_id: NodeId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Node(node_id), name }
    }

    /// Creates a prerequisite reference owned by a producer-consumer edge.
    pub const fn edge(producer: NodeId, consumer: EdgeConsumer, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Edge { producer, consumer }, name }
    }

    /// Creates a prerequisite reference owned by a subgraph invariant.
    pub const fn invariant(invariant_id: InvariantId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Invariant(invariant_id), name }
    }
}

/// The graph scope where a constraint belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintScope {
    /// Constraint attached to one native plan node.
    Node(NodeId),
    /// Constraint attached to one producer-consumer boundary.
    Edge {
        /// Producer node on the dependency edge.
        producer: NodeId,
        /// Consumer endpoint on the dependency edge.
        consumer: EdgeConsumer,
    },
    /// Constraint attached to a non-local subgraph invariant.
    Invariant(InvariantId),
}

/// Capability proof result for a node, edge, or invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    /// The capability is fully proven at its owning scope.
    Satisfied,
    /// The capability failed at this scope; facts should name the local cause.
    Rejected {
        /// Stable rejection code for reports and tests.
        reason: &'static str,
        /// Stable fact-key map with values formatted for diagnostics.
        facts: BTreeMap<&'static str, String>,
    },
    /// The capability depends on other graph-scope proofs.
    Conditional {
        /// Prerequisite proofs that must be satisfied before this decision is final.
        waiting_on: Vec<ConstraintRef>,
    },
}

impl CapabilityDecision {
    pub(super) fn rejected(reason: &'static str, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }

    /// Returns the stable rejection code for rejected decisions.
    pub const fn rejection_reason(&self) -> Option<&'static str> {
        match self {
            Self::Rejected { reason, .. } => Some(reason),
            Self::Satisfied | Self::Conditional { .. } => None,
        }
    }

    /// Returns the stable rejection fact map for rejected decisions.
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
    /// Local or resident source scan.
    Source {
        /// Physical input format and residency source.
        format: SourceFormatCapability,
    },
    /// Predicate filter preserving row order and residency when admitted.
    Filter,
    /// Expression projection over a resident input frame.
    Projection,
    /// Aggregate operator with grouping and aggregate-function capability facts.
    Aggregate {
        /// Grouping shape used for capability admission.
        grouping: AggregateGrouping,
        /// Aggregate functions that must be supported by the native engine.
        functions: Vec<AggregateFunction>,
    },
    /// Window operator over a resident input.
    Window {
        /// Number of window functions that participate in output sizing.
        function_count: usize,
    },
    /// Sort or top-k ordering barrier.
    Sort {
        /// Whether the plan shape requires the operator to establish ordering.
        ordering: OrderingRequirement,
    },
    /// Row-count limit over a resident input.
    Limit,
    /// Join operator with local join-shape facts.
    Join {
        /// Logical join type requested by the plan.
        join_type: JoinType,
        /// Number of equality key pairs.
        key_count: usize,
        /// Whether an additional residual predicate must be evaluated.
        residual: ResidualRequirement,
        /// Input side chosen as the resident build side.
        build_side: JoinSide,
    },
    /// Union over multiple resident inputs.
    Union {
        /// Number of input edges expected by the union node.
        input_count: usize,
    },
    /// Graph edge normalization operator before graph construction.
    EdgeNormalize {
        /// Whether reverse edges are inserted for undirected processing.
        add_reverse: bool,
        /// Duplicate-edge handling requested by the graph operation.
        dedup: EdgeDedupStrategy,
        /// Weight-combining rule applied when duplicate edges are collapsed.
        weight_combiner: EdgeWeightCombiner,
    },
    /// cuGraph algorithm execution over a constructed graph.
    GraphAlgorithm {
        /// Algorithm family whose parameter contract must be supported.
        algorithm: GraphAlgorithm,
    },
}

/// Source format capability known by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormatCapability {
    /// Local Parquet files decoded through the native source reader.
    LocalParquet,
    /// Already resident cudf table supplied by an upstream native node.
    GpuResident,
    /// Host Arrow literal batches imported by native execution.
    LiteralBatch,
}

/// Aggregate grouping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateGrouping {
    /// Scalar aggregate with no grouping keys.
    Scalar,
    /// Standard grouped aggregate with explicit grouping keys.
    Grouped,
    /// Grouping sets, rollups, or cubes, which require separate support.
    GroupingSets,
}

/// Whether a node establishes an ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingRequirement {
    /// The operator or edge must establish an ordered output.
    Required,
    /// No ordering proof is required.
    None,
}

/// Whether a join has a residual predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualRequirement {
    /// Join has no residual predicate beyond equality keys.
    None,
    /// Join includes a residual predicate that must be proven supported.
    Required,
}

/// Properties produced by a node for downstream edge contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducedProperty {
    /// Output is a resident cudf frame available to downstream consumers.
    ResidentFrame,
    /// Output is a grouped aggregate frame.
    GroupedFrame,
    /// Output is a one-row scalar aggregate frame.
    ScalarFrame,
    /// Output ordering is established for downstream order-sensitive consumers.
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
    /// Source output must be resident before the consumer runs.
    ResidentSourceInput {
        /// Source format that supplies the resident input.
        format: SourceFormatCapability,
    },
    /// Aggregate input must be a resident frame.
    ResidentAggregateInput,
    /// Window input must be a resident frame.
    WindowInput,
    /// Join build input residency and ordering contract.
    JoinBuildInput {
        /// Join side consumed as the build input.
        side: JoinSide,
    },
    /// Join probe input residency contract.
    JoinProbeInput {
        /// Join side consumed as the probe input.
        side: JoinSide,
    },
    /// Union input position contract.
    UnionInput {
        /// Zero-based union input index.
        input_index: usize,
    },
    /// Shared resident producer identity contract.
    SharedProducer {
        /// Identity proof that lets multiple consumers reuse the producer.
        identity: SharedProducerIdentity,
    },
    /// Generic resident-frame flow between native nodes.
    ResidentFrameFlow,
    /// Materialized join output required by a downstream consumer.
    JoinMaterialization,
    /// Sort input residency contract.
    SortInput,
    /// Graph algorithm input residency contract.
    GraphAlgorithmInput,
    /// Final result materialization boundary.
    ExplicitSinkMaterialization,
}

/// Identity proof used for resident producers with multiple consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedProducerIdentity {
    /// Common-subexpression elimination preserved the same producer node id.
    SameNodeIdAfterCse,
}

/// Consumer endpoint for an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeConsumer {
    /// Another native plan node consumes the edge.
    Node(NodeId),
    /// Final result sink consumes the edge.
    Sink(ResultSinkKind),
}

/// Result sink kinds as capability consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResultSinkKind {
    /// Host Arrow batches are materialized for the caller.
    HostArrow,
    /// Output remains resident on the GPU for a native downstream consumer.
    GpuResident,
}

/// Semantic role of an input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputRole {
    /// Single ordinary input for unary operators.
    UnaryInput,
    /// Input consumed by an aggregate node.
    AggregateInput,
    /// Input consumed by a window node.
    WindowInput,
    /// Input consumed by a sort or top-k node.
    SortInput,
    /// Input consumed at a specific union position.
    UnionInput {
        /// Zero-based union input index.
        input_index: usize,
    },
    /// Input consumed by a graph algorithm node.
    GraphAlgorithmInput,
    /// Input consumed by a limit node.
    LimitInput,
    /// Build-side input consumed by a join node.
    JoinBuildInput {
        /// Join side consumed as build.
        side: JoinSide,
    },
    /// Probe-side input consumed by a join node.
    JoinProbeInput {
        /// Join side consumed as probe.
        side: JoinSide,
    },
    /// Final result sink input.
    SinkInput,
}

/// Property required from an edge producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequiredProperty {
    /// Producer must yield a GPU-resident cudf frame.
    ResidentFrame,
}

/// Stable identifier for an invariant inside one analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvariantId(pub usize);

impl InvariantId {
    /// Returns the zero-based index within one capability analysis.
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
    BuildBeforeProbe {
        /// Join whose build side must be scheduled before its probe side.
        join_id: NodeId,
    },
    /// The reachable join region below the root is closed over supported join,
    /// projection, filter, and source nodes.
    ClosedJoinSubtree {
        /// Root join of the closed resident join region.
        root_join_id: NodeId,
    },
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
    PostRootStageOrder {
        /// Root join whose post-root stages are being ordered.
        root_join_id: NodeId,
    },
    /// A projection wrapper can be ignored for residency only when both the
    /// peeled producer and the projection output are resident frames.
    ProjectionTransparentResidency,
    /// A filter/projection chain preserves residency only when every node in
    /// the connected unary chain consumes the previous resident output and also
    /// produces a resident frame.
    ResidentUnaryChainPreservesResidency,
    /// The residual predicate on the join references both join inputs, so the
    /// residual is a true join condition rather than a single-side filter.
    ResidualUsesBothInputs {
        /// Join whose residual predicate is being checked.
        join_id: NodeId,
    },
    /// A large full-source scan feeds grouping sets and a rank window while
    /// blocking-stage exact preflight/progress contracts are not modeled.
    /// A downstream final top-k may exist but is not required for the guard.
    BlockingRollupRankTopKFullScan,
    /// Every reachable resident producer remains valid for all of its admitted
    /// consumers, and no rejected producer-consumer edge is hidden by the
    /// broader compiled subgraph admission.
    ResidentLifetimeDominatesConsumers,
}

/// Scheduling proof required by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleConstraint {
    /// No non-local scheduling constraint is required.
    None,
    /// Build-side work must complete before probe-side work.
    BuildBeforeProbe,
    /// Nodes must be schedulable in topological dependency order.
    Topological,
    /// A final barrier must wait for all upstream producers.
    Barrier,
}

/// Resident resources whose lifetime or ordering is governed by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResource {
    /// Resident output table produced by one node.
    NodeOutput(NodeId),
    /// Resident join build input retained for a join.
    JoinBuildInput {
        /// Join that owns the resident build input.
        join_id: NodeId,
        /// Producer node supplying the build input.
        input: NodeId,
    },
    /// Resident join probe input retained for a join.
    JoinProbeInput {
        /// Join that owns the resident probe input.
        join_id: NodeId,
        /// Producer node supplying the probe input.
        input: NodeId,
    },
}

/// Report-facing projection of the first capability rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRejectionProjection {
    /// Local owner whose rejection is projected into diagnostics.
    pub owner: CapabilityRejectionOwner,
    /// Contract scope used by strict bounded reports.
    pub contract_scope: ReservationRequirementScope,
    /// Stable diagnostic boundary label.
    pub memory_boundary: &'static str,
    /// Stable rejection code.
    pub reason: &'static str,
    /// Stable fact-key map copied from the rejected capability decision.
    pub facts: BTreeMap<&'static str, String>,
}

/// Local capability owner projected into strict bounded contract diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityRejectionOwner {
    /// Rejection owned by a node-local capability.
    Node(NodeId),
    /// Rejection owned by a producer-consumer edge.
    Edge {
        /// Producer node for the rejected edge.
        producer: NodeId,
        /// Consumer endpoint for the rejected edge.
        consumer: EdgeConsumer,
    },
    /// Rejection owned by a subgraph invariant.
    Invariant(InvariantId),
    /// Rejection owned by a compiled-envelope admission gate.
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
    /// Node-local aggregate strategy capability facts.
    pub aggregate_strategies: Vec<AggregateStrategyCapability>,
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
    /// Returns true when any node, edge, or invariant has a local rejection.
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }

    /// Returns whether the strict closed multi-join envelope is admitted.
    pub fn admits_closed_multi_join_subgraph(&self) -> bool {
        self.closed_multi_join_subgraph_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict root-join envelope is admitted.
    pub fn admits_root_join(&self) -> bool {
        self.root_join_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict single-join post-root envelope is admitted.
    pub fn admits_single_join_post_root_pipeline(&self) -> bool {
        self.single_join_post_root_pipeline_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict multi-join aggregate/top-k envelope is admitted.
    pub fn admits_multi_join_aggregate_over_join_subplan_topk_pipeline(&self) -> bool {
        self.multi_join_aggregate_over_join_subplan_topk_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict mixed keyed/residual aggregate-input envelope is admitted.
    pub fn admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(&self) -> bool {
        self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict hash-join aggregate post-join/top-k envelope is admitted.
    pub fn admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(&self) -> bool {
        self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission
            == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict shared resident-subplan envelope is admitted.
    pub fn admits_keyed_hash_join_resident_subplan_pipeline(&self) -> bool {
        self.keyed_hash_join_resident_subplan_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict keyless conditional input-subplan envelope is admitted.
    pub fn admits_keyless_conditional_join_input_subplan_pipeline(&self) -> bool {
        self.keyless_conditional_join_input_subplan_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict keyless aggregate-over-inputs envelope is admitted.
    pub fn admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(&self) -> bool {
        self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission == CapabilityDecision::Satisfied
    }

    /// Returns whether the strict resident aggregate subquery-filter envelope is admitted.
    pub fn admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(&self) -> bool {
        self.multi_join_resident_grouped_aggregate_subquery_filter_admission == CapabilityDecision::Satisfied
    }

    /// Returns true when any strict bounded join compiled envelope is admitted.
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

    /// Projects the first local rejection or lower-level block into report diagnostics.
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
    /// Stable snake-case label emitted in contract diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
            Self::RejectedByCapability => "rejected_by_capability",
        }
    }
}
