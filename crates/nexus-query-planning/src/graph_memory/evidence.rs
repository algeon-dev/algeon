use serde_json::{Value, json};

use super::{
    GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON, GraphCalibrationSource,
    GraphMemoryClass, GraphPairCardinalityAdmissionMode, GraphPairCardinalitySource, GraphReservationOutcome,
    GraphSeedMode, GraphStatistics, GraphVertexEstimateSource,
    model::{
        GraphDerivedTerms, graph_live_reservation_floor_top_up_bytes, graph_memory_arithmetic_error,
        project_model_bytes, semantic_edge_row_width_bytes,
    },
    side_input::graph_side_input_workspace_peak_bytes,
};
use crate::{
    admission::canonical_execution_scope_fingerprint,
    correlation::QueryId,
    error::{QueryError, Result},
    execution_contracts::MemoryReservationRequest,
    expr::alias_wrapped_column,
    plan::{GraphAlgorithm, GraphConstructionPolicy, NodeId, PlanNode, QueryPlan},
};

#[doc(hidden)]
pub fn graph_output_is_final(plan: &QueryPlan, graph_node_id: NodeId) -> bool {
    let mut node_id = plan.root_id();
    loop {
        if node_id == graph_node_id {
            return true;
        }
        let Some(PlanNode::Projection(projection)) = plan.node(node_id).map(|node| &node.kind) else {
            return false;
        };
        let Some(input_schema) = plan.node(projection.input).map(|node| node.output_schema()) else {
            return false;
        };
        if projection.outputs.iter().any(|output| {
            let Some(column) = alias_wrapped_column(&output.expr) else {
                return true;
            };
            match input_schema.field_with_name(column.name()) {
                Ok(field) => field.data_type() != output.field.data_type(),
                Err(_) => true,
            }
        }) {
            return false;
        }
        node_id = projection.input;
    }
}

/// Typed retained-output contribution from the graph-memory owner.
///
/// The retained output is kept as a typed contribution instead of being
/// recovered from a difference between rounded phase peaks.  That distinction
/// matters when composing graph producers: a producer's output handle remains
/// live after its isolated phase and is therefore an independent owner in the
/// schedule fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryRetainedOutputContribution {
    bytes: u64,
    row_count_upper_bound: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum GraphMemoryOwnerTermRole {
    GraphPersistent,
    ConstructionTransient,
    ExecutionWorkspace,
    GraphOutput,
    SsspUnitWeights,
    GraphLiveReservationFloorTopUp,
    SideInputWorkspace,
    RetainedOutput,
}

impl GraphMemoryOwnerTermRole {
    #[doc(hidden)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GraphPersistent => "graph_persistent",
            Self::ConstructionTransient => "construction_transient",
            Self::ExecutionWorkspace => "execution_workspace",
            Self::GraphOutput => "graph_output",
            Self::SsspUnitWeights => "sssp_unit_weights",
            Self::GraphLiveReservationFloorTopUp => "graph_live_reservation_floor_top_up",
            Self::SideInputWorkspace => "side_input_workspace",
            Self::RetainedOutput => "retained_output",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct GraphMemoryOwnerTermFact {
    pub(super) execution_scope_fingerprint: String,
    pub(super) graph_node_id: NodeId,
    pub(super) role: GraphMemoryOwnerTermRole,
    pub(super) bytes: u64,
    pub(super) row_count_upper_bound: Option<u64>,
}

impl GraphMemoryOwnerTermFact {
    fn new(
        execution_scope_fingerprint: String,
        graph_node_id: NodeId,
        role: GraphMemoryOwnerTermRole,
        bytes: u64,
        row_count_upper_bound: Option<u64>,
    ) -> Result<Self> {
        if bytes == u64::MAX {
            return Err(graph_memory_arithmetic_error(role.as_str(), "saturated_owner_term"));
        }
        Ok(Self { execution_scope_fingerprint, graph_node_id, role, bytes, row_count_upper_bound })
    }

    #[doc(hidden)]
    pub fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    #[doc(hidden)]
    pub const fn graph_node_id(&self) -> NodeId {
        self.graph_node_id
    }

    #[doc(hidden)]
    pub const fn role(&self) -> GraphMemoryOwnerTermRole {
        self.role
    }

    #[doc(hidden)]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[doc(hidden)]
    pub const fn row_count_upper_bound(&self) -> Option<u64> {
        self.row_count_upper_bound
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct GraphMemoryOwnerComponents {
    pub(super) graph_persistent_bytes: u64,
    pub(super) construction_transient_bytes: u64,
    pub(super) execution_workspace_bytes: u64,
    pub(super) sssp_unit_weight_bytes: Option<u64>,
}

impl GraphMemoryRetainedOutputContribution {
    /// Builds a model-owned retained-output contribution.
    #[doc(hidden)]
    pub const fn new(bytes: u64, row_count_upper_bound: u64) -> Self {
        Self { bytes, row_count_upper_bound }
    }

    /// Device bytes retained by the producer after its isolated phase.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Upper bound on rows represented by the retained producer output.
    ///
    /// The row bound is carried with the bytes so an engine-owned consumer
    /// preflight can price a downstream join without asking an adapter to
    /// reverse-engineer graph output cardinality.
    pub const fn row_count_upper_bound(self) -> u64 {
        self.row_count_upper_bound
    }
}

/// Approximate graph memory estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryEstimate {
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub raw_input_edges: Option<u64>,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub pair_rows: Option<u64>,
    pub pair_vertex_width_bytes: Option<u64>,
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
    /// Admission contract that consumed [`Self::pair_cardinality_source`].
    pub pair_cardinality_admission_mode: Option<GraphPairCardinalityAdmissionMode>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub edge_normalize_inserted: bool,
    pub calibration_source: GraphCalibrationSource,
    pub seed_count: Option<u64>,
    pub seed_mode: Option<GraphSeedMode>,
    pub data_dependent: bool,
    /// Peak bytes needed while constructing the native graph representation.
    ///
    /// The execution-only estimate may omit this diagnostic term, but
    /// admission never uses that estimate to select a grant. A resident cache
    /// needs its own accounted reservation before retaining a graph across
    /// attempts.
    pub construction_bytes: u64,
    /// Peak bytes needed by the algorithm after graph construction.
    pub execution_bytes: u64,
    /// Call-local unit-weight property allocated only for unweighted SSSP.
    pub(super) sssp_unit_weight_bytes: Option<u64>,
    /// Typed bytes and row-count bound retained by the producer result handle.
    ///
    /// This is the owner-calculated `output_bytes` contribution. It is not
    /// inferred from `requested_bytes` or `execution_bytes`, both of which may
    /// include rounded workspace and phase terms.
    pub retained_output_bytes: GraphMemoryRetainedOutputContribution,
    /// Cold native-graph peak estimate. For algorithms with distinct native
    /// lifetime phases this is the phase-aware peak, not a naïve sum of
    /// construction and execution estimates. Admission composes any retained
    /// source-stage owner separately when the plan proves one.
    pub requested_bytes: u64,
    pub(super) owner_components: GraphMemoryOwnerComponents,
}

/// Opaque graph-memory model output.
///
/// The model is the only constructor for this token. It contains descriptive
/// bytes only: it cannot form a generic admission requirement or alter the
/// ledger's floor, capability, or fit proof. An engine-issued capability proof
/// may retain its cold estimate as a soft immutable-grant target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum GraphSourceStageLifetime {
    /// The cold graph peak already owns the prepared edge frame emitted by the
    /// native source. The source diagnostic is not a second GPU allocation.
    SharedPreparedEdgeFrame,
    /// Explicit edge normalization retains the source frame while it creates
    /// the normalized graph input, so both owners are live at the cold peak.
    SeparateNormalizedInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryAdmissionEvidence {
    execution_scope_fingerprint: String,
    cold_model_requested_bytes: u64,
    warm_model_requested_bytes: u64,
    retained_output_bytes: GraphMemoryRetainedOutputContribution,
    pub(super) owner_terms: Vec<GraphMemoryOwnerTermFact>,
    source_stage_lifetime: GraphSourceStageLifetime,
    retained_normalization_source_model_bytes: u64,
}

impl GraphMemoryAdmissionEvidence {
    pub(super) fn from_bounded_plan_estimate(
        plan: &QueryPlan,
        node_id: NodeId,
        estimate: &GraphMemoryEstimate,
    ) -> Result<Self> {
        // `requested_bytes` is already phase-aware: persistent graph/input
        // owners survive into execution, while construction-only scratch does not.
        let cold_model_requested_bytes = estimate.construction_bytes.max(estimate.requested_bytes).max(1);
        let execution_scope_fingerprint = canonical_execution_scope_fingerprint(plan);
        let live_reservation_floor_top_up_bytes =
            graph_live_reservation_floor_top_up_bytes(cold_model_requested_bytes)?;
        let mut owner_terms = estimate.owner_components.owner_terms(
            execution_scope_fingerprint.clone(),
            node_id,
            estimate.retained_output_bytes,
            graph_output_is_final(plan, node_id),
            live_reservation_floor_top_up_bytes,
        )?;
        if let Some(side_input_peak_bytes) = graph_side_input_workspace_peak_bytes(plan, node_id, estimate)? {
            owner_terms.push(GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                node_id,
                GraphMemoryOwnerTermRole::SideInputWorkspace,
                side_input_peak_bytes,
                None,
            )?);
        }
        Ok(Self {
            execution_scope_fingerprint,
            cold_model_requested_bytes,
            warm_model_requested_bytes: estimate.execution_bytes,
            retained_output_bytes: estimate.retained_output_bytes,
            owner_terms,
            source_stage_lifetime: if estimate.edge_normalize_inserted {
                GraphSourceStageLifetime::SeparateNormalizedInput
            } else {
                GraphSourceStageLifetime::SharedPreparedEdgeFrame
            },
            retained_normalization_source_model_bytes: if estimate.edge_normalize_inserted {
                estimate
                    .raw_input_edges
                    .unwrap_or(estimate.input_edges)
                    .checked_mul(semantic_edge_row_width_bytes(
                        estimate.vertex_width_bytes,
                        estimate.weight_width_bytes,
                        estimate.has_edge_id,
                    ))
                    .ok_or_else(|| {
                        graph_memory_arithmetic_error("retained_normalization_source_bytes", "multiplication")
                    })?
            } else {
                0
            },
        })
    }

    #[doc(hidden)]
    pub fn matches_execution_scope_fingerprint(&self, fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == fingerprint
    }

    #[doc(hidden)]
    pub fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    #[doc(hidden)]
    pub fn owner_term_facts(&self) -> &[GraphMemoryOwnerTermFact] {
        &self.owner_terms
    }

    /// Returns the construction-inclusive model's descriptive byte estimate.
    ///
    /// This is diagnostic/model output. It may refine an engine-issued graph
    /// capability proof's soft grant target, but is not accepted as a raw
    /// public admission-request input.
    pub fn cold_model_requested_bytes(&self) -> u64 {
        self.cold_model_requested_bytes
    }

    /// Returns the execution-only model's descriptive byte estimate.
    ///
    /// This is a hypothetical cache-price diagnostic only. It is not accepted
    /// by public admission request construction or device placement.
    pub fn warm_model_requested_bytes(&self) -> u64 {
        self.warm_model_requested_bytes
    }

    /// Returns the typed producer output contribution retained through the
    /// composed consumer phase.
    pub fn retained_output(&self) -> GraphMemoryRetainedOutputContribution {
        self.retained_output_bytes
    }

    /// Returns the retained output bytes without exposing a caller-authored
    /// byte input path.
    pub fn retained_output_bytes(&self) -> u64 {
        self.retained_output_bytes.bytes()
    }

    /// Returns the model-owned retained output row bound.
    pub fn retained_output_rows_upper_bound(&self) -> u64 {
        self.retained_output_bytes.row_count_upper_bound()
    }

    #[doc(hidden)]
    pub const fn source_stage_lifetime(&self) -> GraphSourceStageLifetime {
        self.source_stage_lifetime
    }

    #[doc(hidden)]
    pub const fn retained_normalization_source_model_bytes(&self) -> u64 {
        self.retained_normalization_source_model_bytes
    }

    /// Composes the model's cold graph peak with the exact source-stage hint
    /// retained by the matching native edge fragment.
    ///
    /// The lifecycle classification is model-owned: raw input shares the
    /// prepared edge frame already priced by the graph peak, while an explicit
    /// normalize operator keeps the source and its derived graph input live
    /// together. Callers cannot choose this relation or inject a raw graph
    /// model byte value.
    #[doc(hidden)]
    pub fn cold_model_with_source_stage_bytes(&self, source_stage_bytes: u64) -> Result<u64> {
        self.graph_peak_with_source_stage_bytes(self.cold_model_requested_bytes, source_stage_bytes)
    }

    /// Returns the graph-native live reservation after applying the canonical
    /// bounded execution minimum. The model value remains available through
    /// [`Self::cold_model_requested_bytes`] as diagnostics; this lower bound
    /// is a reservation contract, not an observed allocation term.
    #[doc(hidden)]
    pub fn bounded_live_graph_peak_bytes(&self) -> Result<u64> {
        let Some(top_up) = self.graph_live_reservation_floor_top_up_bytes()? else {
            return Ok(self.cold_model_requested_bytes);
        };
        self.cold_model_requested_bytes
            .checked_add(top_up)
            .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation", "addition"))
    }

    /// Composes the bounded graph-native live minimum with a prepared source
    /// frame using the same owner lifetime relation as the raw model.
    #[doc(hidden)]
    pub fn bounded_live_graph_peak_with_source_stage_bytes(&self, source_stage_bytes: u64) -> Result<u64> {
        let Some(top_up) = self.graph_live_reservation_floor_top_up_bytes()? else {
            return self.cold_model_with_source_stage_bytes(source_stage_bytes);
        };
        let graph_peak = self
            .cold_model_requested_bytes
            .checked_add(top_up)
            .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation", "addition"))?;
        self.graph_peak_with_source_stage_bytes(graph_peak, source_stage_bytes)
    }

    fn graph_peak_with_source_stage_bytes(&self, graph_peak_bytes: u64, source_stage_bytes: u64) -> Result<u64> {
        if source_stage_bytes == u64::MAX || graph_peak_bytes == u64::MAX {
            return Err(graph_memory_arithmetic_error("source_stage_graph_peak", "saturated"));
        }
        match self.source_stage_lifetime {
            GraphSourceStageLifetime::SharedPreparedEdgeFrame => Ok(source_stage_bytes.max(graph_peak_bytes)),
            GraphSourceStageLifetime::SeparateNormalizedInput => source_stage_bytes
                .checked_add(graph_peak_bytes)
                .ok_or_else(|| graph_memory_arithmetic_error("source_stage_graph_peak", "addition")),
        }
    }

    #[doc(hidden)]
    pub fn graph_live_reservation_floor_top_up_bytes(&self) -> Result<Option<u64>> {
        let expected = graph_live_reservation_floor_top_up_bytes(self.cold_model_requested_bytes)?;
        let mut facts = self
            .owner_terms
            .iter()
            .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp);
        let actual = facts.next().map(GraphMemoryOwnerTermFact::bytes);
        if facts.next().is_some() || actual != expected {
            return Err(graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "owner_term_mismatch")
                .with_fact("reason", "graph_memory_owner_term_floor_mismatch"));
        }
        Ok(expected)
    }
}

/// Planning-time graph memory estimate for a node in a native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPlanMemoryEstimate {
    pub node_id: NodeId,
    pub estimate: GraphMemoryEstimate,
}

/// Fixed-point `ask / measured_peak` ratio for graph model calibration drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphModelErrorRatio {
    milli_ratio: u64,
}

impl GraphModelErrorRatio {
    /// Build a ratio from model-requested bytes and measured graph peak bytes.
    pub fn from_requested_and_measured(requested_bytes: u64, measured_peak_bytes: u64) -> Option<Self> {
        if measured_peak_bytes == 0 {
            return None;
        }
        let numerator = u128::from(requested_bytes).saturating_mul(1000);
        let rounded = numerator.saturating_add(u128::from(measured_peak_bytes / 2)) / u128::from(measured_peak_bytes);
        Some(Self { milli_ratio: rounded.min(u128::from(u64::MAX)) as u64 })
    }

    /// Stable decimal string with three fractional digits.
    pub fn as_decimal_string(self) -> String {
        format!("{}.{:03}", self.milli_ratio / 1000, self.milli_ratio % 1000)
    }
}

impl std::fmt::Display for GraphModelErrorRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_decimal_string())
    }
}

/// Direct source for one graph attempt's measured device-memory peak.
///
/// This is deliberately separate from [`GraphCalibrationSource`]: that enum
/// describes model inputs, while this one describes the runtime observation
/// used to calibrate the completed graph attempt.  Consumers must not infer a
/// source from a missing value or from an error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMeasuredPeakProvenance {
    /// The graph execution's reservation resource reported its own high-water
    /// mark. This is the preferred graph-stage measurement.
    ReservationHighWater,
    /// The graph execution did not expose a reservation high-water mark, so
    /// the CUDA async-pool high-water delta bounded around that attempt was
    /// used instead.
    PoolDelta,
    /// An exclusive resident-graph cache scope reported its own allocator
    /// high-water mark after the warm graph execution had retired. This is
    /// the only physical peak source for work that runs on the cache runtime
    /// rather than the admitted attempt runtime.
    CacheScopeHighWater,
    /// Neither runtime measurement completed. `measured_peak_bytes` is
    /// unavailable rather than a synthetic zero.
    Unavailable,
}

impl GraphMeasuredPeakProvenance {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReservationHighWater => "reservation_high_water",
            Self::PoolDelta => "pool_delta",
            Self::CacheScopeHighWater => "cache_scope_high_water",
            Self::Unavailable => "unavailable",
        }
    }
}

impl std::fmt::Display for GraphMeasuredPeakProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Engine-owned lifecycle state for one graph model claim.
///
/// A claim exists before graph-memory admission so a missing terminal state is
/// deliberately fail-closed. The two terminal states distinguish a model
/// admission denial from an actual entry into the graph executor; neither is
/// inferred from an error string, an allocator event, or elapsed time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphClaimExecutionState {
    /// The claim was recorded but no exact terminal graph boundary was
    /// observed. This state is not execution evidence.
    Pending,
    /// The native graph algorithm executor was entered.
    AlgorithmStarted,
    /// Graph-memory pricing rejected the request because its tracked request
    /// exceeded the current graph-admission budget, before the executor ran.
    GraphMemoryAdmissionDenied,
}

impl GraphClaimExecutionState {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::AlgorithmStarted => "algorithm_started",
            Self::GraphMemoryAdmissionDenied => "graph_memory_admission_denied",
        }
    }
}

impl std::fmt::Display for GraphClaimExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-shape model-claim versus measured-peak calibration row for one graph
/// attempt.
///
/// The engine creates one of these before it enters the graph executor and
/// fills the runtime measurement when that attempt leaves the executor. It is
/// diagnostic only: the row cannot mint admission evidence or modify a
/// device grant. `node_index` is retained solely as typed executor identity;
/// consumers should use `node_id` for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphClaimCalibrationRow {
    /// Query-wide diagnostic correlation id, when the execution context has
    /// one. The runtime metrics owner assigns this sequence.
    pub query_id: Option<QueryId>,
    /// Stable native graph node id for reports.
    pub node_id: String,
    /// Typed graph node identity used to correlate the post-execution sample
    /// without parsing a report string.
    pub node_index: usize,
    /// The graph algorithm that owned this model claim and measured peak.
    pub algorithm: GraphAlgorithm,
    /// Construction-inclusive bytes requested by the graph memory model.
    pub model_requested_bytes: u64,
    /// Bytes requested from the graph pricing tier after its documented floor
    /// and any live graph-input or similarity-input credit.
    pub requested_bytes: u64,
    /// The graph-pricing budget considered for this attempt, when one was
    /// present. A missing budget is not rendered as a zero-byte budget.
    pub budget_bytes: Option<u64>,
    /// Exact engine state at the graph executor boundary. A pending claim is
    /// intentionally not promoted to physical execution by an adapter.
    pub execution_state: GraphClaimExecutionState,
    /// Measured graph-stage peak bytes when a runtime observation completed.
    pub measured_peak_bytes: Option<u64>,
    /// Source of [`Self::measured_peak_bytes`].
    pub measured_peak_provenance: GraphMeasuredPeakProvenance,
    /// Fixed-point model-ask / measured-peak ratio, when a nonzero measured
    /// peak was available.
    pub graph_model_error_ratio: Option<GraphModelErrorRatio>,
}

impl GraphClaimCalibrationRow {
    /// Serializes this graph claim-versus-peak calibration row as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "node_id": self.node_id,
            "node_index": self.node_index,
            "algorithm": self.algorithm.as_str(),
            "model_requested_bytes": self.model_requested_bytes,
            "requested_bytes": self.requested_bytes,
            "budget_bytes": self.budget_bytes,
            "execution_state": self.execution_state.as_str(),
            "measured_peak_bytes": self.measured_peak_bytes,
            "measured_peak_provenance": self.measured_peak_provenance.as_str(),
            "graph_model_error_ratio": self.graph_model_error_ratio.map(|ratio| ratio.as_decimal_string()),
        })
    }
}

impl GraphClaimCalibrationRow {
    /// Builds the pending calibration row from the typed graph-pricing report.
    ///
    /// The report has no private execution identity, so the caller supplies
    /// the plan node that produced it. Keeping this identity outside the
    /// report avoids changing existing reservation-row semantics.
    pub fn from_reservation_report(report: &GraphReservationReport, node_id: NodeId) -> Self {
        Self {
            query_id: report.query_id.clone(),
            node_id: report.node_id.clone(),
            node_index: node_id.index(),
            algorithm: report.algorithm,
            model_requested_bytes: report.model_requested_bytes,
            requested_bytes: report.requested_bytes,
            budget_bytes: report.budget_bytes,
            execution_state: GraphClaimExecutionState::Pending,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
            graph_model_error_ratio: None,
        }
    }

    /// Records one terminal graph-boundary state without allowing a later
    /// observation to overwrite the first exact boundary.
    #[doc(hidden)]
    pub fn record_execution_state(&mut self, execution_state: GraphClaimExecutionState) {
        if self.execution_state == GraphClaimExecutionState::Pending {
            self.execution_state = execution_state;
        }
    }

    /// Records the one direct peak observation for this graph attempt.
    ///
    /// An unavailable source always clears the measured value and ratio. A
    /// zero high-water mark remains an observed zero, but cannot form a
    /// meaningful model-error ratio.
    pub fn record_measured_peak(
        &mut self,
        measured_peak_bytes: Option<u64>,
        measured_peak_provenance: GraphMeasuredPeakProvenance,
    ) {
        match measured_peak_provenance {
            GraphMeasuredPeakProvenance::Unavailable => {
                self.measured_peak_bytes = None;
                self.graph_model_error_ratio = None;
            },
            GraphMeasuredPeakProvenance::ReservationHighWater
            | GraphMeasuredPeakProvenance::PoolDelta
            | GraphMeasuredPeakProvenance::CacheScopeHighWater => {
                self.measured_peak_bytes = measured_peak_bytes;
                self.graph_model_error_ratio = measured_peak_bytes.and_then(|peak| {
                    GraphModelErrorRatio::from_requested_and_measured(self.model_requested_bytes, peak)
                });
            },
        }
        self.measured_peak_provenance = measured_peak_provenance;
    }
}

/// Public graph memory-tier report row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReservationReport {
    pub query_id: Option<QueryId>,
    pub node_id: String,
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub outcome: GraphReservationOutcome,
    pub raw_input_edges: Option<u64>,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_estimate: u64,
    pub vertex_estimate_source: GraphVertexEstimateSource,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub pair_rows: Option<u64>,
    pub pair_vertex_width_bytes: Option<u64>,
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
    /// Admission contract that consumed explicit-pair cardinality evidence.
    pub pair_cardinality_admission_mode: Option<GraphPairCardinalityAdmissionMode>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub edge_normalize_inserted: bool,
    pub calibration_source: GraphCalibrationSource,
    pub seed_count: Option<u64>,
    pub seed_mode: Option<GraphSeedMode>,
    pub data_dependent: bool,
    pub model_requested_bytes: u64,
    pub requested_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub graph_model_error_ratio: Option<GraphModelErrorRatio>,
    /// Measured graph-stage allocator high-water bytes, when runtime evidence
    /// was available.  This is the denominator of [`Self::graph_model_error_ratio`].
    pub measured_peak_bytes: Option<u64>,
    pub statistics: Option<GraphStatistics>,
    pub reason: Option<&'static str>,
}

impl GraphReservationReport {
    /// Serializes this graph memory-tier reservation row as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "node_id": self.node_id,
            "algorithm": self.algorithm.as_str(),
            "class": self.class.as_str(),
            "outcome": self.outcome.as_str(),
            "raw_input_edges": self.raw_input_edges,
            "input_edges": self.input_edges,
            "vertex_upper_bound": self.vertex_upper_bound,
            "vertex_estimate": self.vertex_estimate,
            "vertex_estimate_source": self.vertex_estimate_source.as_str(),
            "vertex_width_bytes": self.vertex_width_bytes,
            "weight_width_bytes": self.weight_width_bytes,
            "pair_rows": self.pair_rows,
            "pair_vertex_width_bytes": self.pair_vertex_width_bytes,
            "pair_cardinality_source": self.pair_cardinality_source.map(|source| source.as_str()),
            "pair_cardinality_admission_mode": self.pair_cardinality_admission_mode.map(|mode| mode.as_str()),
            "has_edge_id": self.has_edge_id,
            "directed": self.directed,
            "renumber": self.renumber,
            "construction_policy": self.construction_policy.as_str(),
            "edge_normalize_inserted": self.edge_normalize_inserted,
            "calibration_source": self.calibration_source.as_str(),
            "seed_count": self.seed_count,
            "seed_mode": self.seed_mode.map(|mode| mode.as_str()),
            "data_dependent": self.data_dependent,
            "model_requested_bytes": self.model_requested_bytes,
            "requested_bytes": self.requested_bytes,
            "budget_bytes": self.budget_bytes,
            "graph_model_error_ratio": self.graph_model_error_ratio.map(|ratio| ratio.as_decimal_string()),
            "measured_peak_bytes": self.measured_peak_bytes,
            "statistics": self.statistics.map(|statistics| {
                json!({
                    "exact_vertices": statistics.exact_vertices,
                    "degree_max": statistics.degree_max,
                    "degree_p99": statistics.degree_p99,
                    "sum_min_degree_oriented": statistics.sum_min_degree_oriented,
                    "oriented_edge_count": statistics.oriented_edge_count,
                    "peak_chunk_min_degree_sum": statistics.peak_chunk_min_degree_sum,
                    "sum_out_degree": statistics.sum_out_degree,
                })
            }),
            "reason": self.reason,
        })
    }
}

impl GraphReservationReport {
    /// Builds a report from an estimate and optional bounded budget.
    pub fn from_estimate(node_id: impl ToString, estimate: GraphMemoryEstimate, budget_bytes: Option<u64>) -> Self {
        let requested_bytes = if budget_bytes.is_some() {
            estimate.requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
        } else {
            estimate.requested_bytes
        };
        let denied = budget_bytes.is_some_and(|budget| requested_bytes > budget);
        let outcome = match (budget_bytes, denied) {
            (None, _) => GraphReservationOutcome::NotBounded,
            (Some(_), true) => GraphReservationOutcome::Denied,
            (Some(_), false) => GraphReservationOutcome::Admitted,
        };
        Self {
            query_id: None,
            node_id: node_id.to_string(),
            algorithm: estimate.algorithm,
            class: estimate.class,
            outcome,
            raw_input_edges: estimate.raw_input_edges,
            input_edges: estimate.input_edges,
            vertex_upper_bound: estimate.vertex_upper_bound,
            vertex_estimate: estimate.vertex_estimate,
            vertex_estimate_source: estimate.vertex_estimate_source,
            vertex_width_bytes: estimate.vertex_width_bytes,
            weight_width_bytes: estimate.weight_width_bytes,
            pair_rows: estimate.pair_rows,
            pair_vertex_width_bytes: estimate.pair_vertex_width_bytes,
            pair_cardinality_source: estimate.pair_cardinality_source,
            pair_cardinality_admission_mode: estimate.pair_cardinality_admission_mode,
            has_edge_id: estimate.has_edge_id,
            directed: estimate.directed,
            renumber: estimate.renumber,
            construction_policy: estimate.construction_policy,
            edge_normalize_inserted: estimate.edge_normalize_inserted,
            calibration_source: estimate.calibration_source,
            seed_count: estimate.seed_count,
            seed_mode: estimate.seed_mode,
            data_dependent: estimate.data_dependent,
            model_requested_bytes: estimate.requested_bytes,
            requested_bytes,
            budget_bytes,
            graph_model_error_ratio: None,
            measured_peak_bytes: None,
            statistics: None,
            reason: denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON),
        }
    }

    /// Converts the cold whole-peak model into the incremental reservation
    /// needed when its graph-input edge frame is already live in the same query
    /// grant.
    ///
    /// The model includes the normalized semantic edge frame and, after an
    /// explicit edge-normalize stage, its retained source frame. Each live
    /// owner is credited separately and capped at its modeled overlap, so a
    /// wider input cannot erase graph construction, output, or workspace
    /// bytes. The exact normalized-frame owner may use its physical tracker
    /// charge as a tighter cap than semantic column widths.
    #[cfg(any(feature = "cugraph", test))]
    #[doc(hidden)]
    pub fn credit_live_graph_input(
        mut self,
        already_accounted_normalized_frame_bytes: u64,
        exact_normalized_frame_charge_bytes: Option<u64>,
        retained_normalization_source_bytes: u64,
        retained_normalization_source_credit_cap_bytes: Option<u64>,
    ) -> Self {
        let Some(budget_bytes) = self.budget_bytes else {
            return self;
        };
        let credit_cap_bytes =
            exact_normalized_frame_charge_bytes.unwrap_or_else(|| self.modeled_live_graph_input_frame_bytes());
        let normalized_frame_credit_bytes =
            credit_cap_bytes.min(already_accounted_normalized_frame_bytes).min(self.requested_bytes);
        self.requested_bytes -= normalized_frame_credit_bytes;
        let source_credit_cap_bytes = if self.edge_normalize_inserted {
            retained_normalization_source_credit_cap_bytes
                .unwrap_or_else(|| self.modeled_retained_normalization_source_bytes())
        } else {
            0
        };
        let retained_source_credit_bytes =
            source_credit_cap_bytes.min(retained_normalization_source_bytes).min(self.requested_bytes);
        self.requested_bytes -= retained_source_credit_bytes;
        let denied = self.requested_bytes > budget_bytes;
        self.outcome = if denied { GraphReservationOutcome::Denied } else { GraphReservationOutcome::Admitted };
        self.reason = denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON);
        self
    }

    #[cfg(any(feature = "cugraph", test))]
    fn modeled_live_graph_input_frame_bytes(&self) -> u64 {
        let semantic_edge_row_width =
            semantic_edge_row_width_bytes(self.vertex_width_bytes, self.weight_width_bytes, self.has_edge_id);
        self.input_edges.saturating_mul(semantic_edge_row_width)
    }

    #[cfg(any(feature = "cugraph", test))]
    fn modeled_retained_normalization_source_bytes(&self) -> u64 {
        let semantic_edge_row_width =
            semantic_edge_row_width_bytes(self.vertex_width_bytes, self.weight_width_bytes, self.has_edge_id);
        self.raw_input_edges.unwrap_or(self.input_edges).saturating_mul(semantic_edge_row_width)
    }

    /// Converts an isolated similarity model price into the incremental
    /// reservation required while its explicit pair relation is already live.
    ///
    /// The model intentionally includes the pair columns because it is also
    /// used for unbounded and planning-time estimates. Bounded execution must
    /// reserve only the additional graph work when those exact bytes already
    /// consume the same query grant. Credit no more than the modeled pair
    /// overlap so a wider source frame cannot reduce unrelated graph work.
    pub fn credit_live_similarity_pair_input(mut self, already_accounted_bytes: u64) -> Self {
        let Some(budget_bytes) = self.budget_bytes else {
            return self;
        };
        if !self.algorithm.is_similarity() {
            return self;
        }
        let pair_input_overlap_bytes = self
            .pair_rows
            .zip(self.pair_vertex_width_bytes)
            .map(|(rows, width)| rows.saturating_mul(width.saturating_mul(2)))
            .unwrap_or(0);
        let credit_bytes = pair_input_overlap_bytes.min(already_accounted_bytes);
        if credit_bytes == 0 {
            return self;
        }

        self.requested_bytes = self.requested_bytes.saturating_sub(credit_bytes);
        let denied = self.requested_bytes > budget_bytes;
        self.outcome = if denied { GraphReservationOutcome::Denied } else { GraphReservationOutcome::Admitted };
        self.reason = denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON);
        self
    }

    /// Attach runtime graph statistics evidence to this row.
    pub fn with_statistics(mut self, statistics: GraphStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Attach the observed model error ratio for a completed graph execution.
    pub fn record_model_error_ratio(&mut self, measured_peak_bytes: u64) {
        self.measured_peak_bytes = Some(measured_peak_bytes);
        self.graph_model_error_ratio =
            GraphModelErrorRatio::from_requested_and_measured(self.model_requested_bytes, measured_peak_bytes);
    }

    /// Builds the structured local admission error for a denied graph fragment.
    pub fn denied_error(&self, request: &MemoryReservationRequest, granted_bytes: u64) -> QueryError {
        let mut error = request
            .denied_error(granted_bytes)
            .with_fact("reason", GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON)
            .with_fact("algorithm", self.algorithm.as_str())
            .with_fact("graph_memory_class", self.class.as_str())
            .with_fact("input_edges", self.input_edges)
            .with_fact("vertex_upper_bound", self.vertex_upper_bound)
            .with_fact("vertex_estimate", self.vertex_estimate)
            .with_fact("vertex_estimate_source", self.vertex_estimate_source.as_str())
            .with_fact("vertex_width_bytes", self.vertex_width_bytes)
            .with_fact("pair_rows", self.pair_rows.unwrap_or(0))
            .with_fact(
                "pair_cardinality_source",
                self.pair_cardinality_source.map_or("none", GraphPairCardinalitySource::as_str),
            )
            .with_fact("directed", self.directed)
            .with_fact("renumber", self.renumber)
            .with_fact("construction_policy", self.construction_policy.as_str())
            .with_fact("edge_normalize_inserted", self.edge_normalize_inserted)
            .with_fact("calibration_source", self.calibration_source.as_str())
            .with_fact("data_dependent", self.data_dependent);
        if let Some(raw_input_edges) = self.raw_input_edges {
            error = error.with_fact("raw_input_edges", raw_input_edges);
        }
        if let Some(seed_count) = self.seed_count {
            error = error.with_fact("seed_count", seed_count);
        }
        if let Some(seed_mode) = self.seed_mode {
            error = error.with_fact("seed_mode", seed_mode.as_str());
        }
        error
    }
}

/// Approximate model for the graph memory tier.
impl GraphMemoryOwnerComponents {
    pub(super) fn from_derived(derived: GraphDerivedTerms, diagnostic_projection: bool) -> Result<Self> {
        Ok(Self {
            graph_persistent_bytes: project_model_bytes(
                derived.graph_persistent,
                "graph_persistent_bytes",
                diagnostic_projection,
            )?,
            construction_transient_bytes: project_model_bytes(
                derived.construction_transient,
                "construction_transient_bytes",
                diagnostic_projection,
            )?,
            execution_workspace_bytes: project_model_bytes(
                derived.execution_workspace,
                "execution_workspace_bytes",
                diagnostic_projection,
            )?,
            sssp_unit_weight_bytes: derived
                .sssp_unit_weights
                .map(|bytes| project_model_bytes(bytes, "sssp_unit_weight_bytes", diagnostic_projection))
                .transpose()?,
        })
    }

    fn owner_terms(
        self,
        execution_scope_fingerprint: String,
        graph_node_id: NodeId,
        graph_output: GraphMemoryRetainedOutputContribution,
        graph_output_is_final: bool,
        live_reservation_floor_top_up_bytes: Option<u64>,
    ) -> Result<Vec<GraphMemoryOwnerTermFact>> {
        let mut terms = vec![
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::GraphPersistent,
                self.graph_persistent_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::ConstructionTransient,
                self.construction_transient_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::ExecutionWorkspace,
                self.execution_workspace_bytes,
                None,
            )?,
            GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                if graph_output_is_final {
                    GraphMemoryOwnerTermRole::RetainedOutput
                } else {
                    GraphMemoryOwnerTermRole::GraphOutput
                },
                graph_output.bytes(),
                graph_output_is_final.then(|| graph_output.row_count_upper_bound()),
            )?,
        ];
        if let Some(bytes) = self.sssp_unit_weight_bytes {
            terms.push(GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint.clone(),
                graph_node_id,
                GraphMemoryOwnerTermRole::SsspUnitWeights,
                bytes,
                None,
            )?);
        }
        if let Some(bytes) = live_reservation_floor_top_up_bytes {
            terms.push(GraphMemoryOwnerTermFact::new(
                execution_scope_fingerprint,
                graph_node_id,
                GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp,
                bytes,
                None,
            )?);
        }
        Ok(terms)
    }
}
