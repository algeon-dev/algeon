//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{AttemptReservationShape, canonical_execution_scope_fingerprint},
    },
    capability::ReservationRequirement,
    error::{QueryError, QueryErrorCode, Result},
    graph_memory::GraphMemoryAdmissionEvidence,
    plan::{PlanNode, QueryPlan, estimated_local_parquet_source_bytes, estimated_reachable_local_parquet_source_bytes},
    runtime::ReservationKind,
    source::{LocalParquetSource, SourceNode},
};

/// Trustworthiness of the descriptive working-set hint.
///
/// Ordering is `Unknown < Modeled < Measured`; aggregation retains the weakest
/// evidence so a multi-fragment invocation never overstates its estimate.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EstimateQuality {
    #[default]
    Unknown,
    Modeled,
    Measured,
}

/// Resolved, planner-owned capability evidence for one bounded execution attempt.
///
/// Callers can forward this value into admission, but cannot author a floor,
/// hint, or device-capability set.  The installed device profile owns baseline
/// grant sizing; planner evidence may only record a proven lower-layer raise
/// and a descriptive model result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptCapabilityEvidence {
    /// Deployment-wide floor supplied by the installed device profile.  This
    /// remains crate-visible only for synthetic controller tests.
    pub(crate) configured_floor_bytes: u64,
    /// Additional lower-layer workspace contracts proven before admission.
    pub(crate) capability_floor_raise_bytes: u64,
    /// Hard lower bound for the service-resolved physical phase envelope.
    pub(crate) required_phase_floor_bytes: u64,
    /// Descriptive local-Parquet stage, distinct from a whole-attempt target.
    pub(crate) source_stage_bytes: Option<u64>,
    /// Descriptive target only; it never proves fit or admission.
    pub(crate) soft_working_set_hint_bytes: Option<u64>,
    /// Evidence quality for [`Self::soft_working_set_hint_bytes`].
    pub(crate) estimate_quality: EstimateQuality,
    /// Capability contracts required by the admitted fragments.
    pub(crate) required_capabilities: Vec<ReservationRequirement>,
    /// Exact graph plans for which the engine may replace its generic
    /// lower-level graph-workspace blocker with the cuGraph contract.
    ///
    /// This remains part of the engine-owned evidence bundle from admission
    /// through execution. A capability label alone is deliberately not enough
    /// to make that substitution: it would let a combined proof authorize an
    /// unrelated graph plan.
    pub(crate) attested_cugraph_graph_execution_scopes: Vec<AttestedCugraphGraphExecutionScope>,
    /// Private, exact plan scopes resolved only after the service fixes
    /// scheduling and chunk policy for this admission.
    pub(crate) reservation_shapes: Vec<AttemptReservationShape>,
}

/// Collision-free structural identity for one graph plan whose frontend owns
/// construction and workspace lifetimes.
///
/// The identity retains the full canonical execution form rather than a
/// fixed-width hash. These scopes are short-lived admission evidence, so a
/// graph contract cannot widen to a separately materialized plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttestedCugraphGraphExecutionScope {
    execution_scope_fingerprint: String,
}

impl AttestedCugraphGraphExecutionScope {
    fn for_plan(plan: &QueryPlan) -> Self {
        Self { execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan) }
    }

    fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    fn matches_reservation_shape(&self, shape: &AttemptReservationShape) -> bool {
        shape.matches_execution_scope_fingerprint(&self.execution_scope_fingerprint)
    }
}

/// Opaque capability proof issued from a native plan or a canonical graph
/// execution contract.
///
/// Public callers can forward a proof into admission, but cannot author a
/// capability set, a floor, or a working-set target.  In particular, there is
/// deliberately no empty-proof constructor: every proof comes from an exact
/// native plan and carries its private execution scope, even when that plan
/// has no resource-capability requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlanCapabilityProof {
    requirement: AttemptCapabilityEvidence,
}

impl QueryPlanCapabilityProof {
    /// Derives proof from one lowered native query plan.
    ///
    /// The proof always retains the plan's exact private execution scope. A
    /// capability-free native fragment therefore still cannot run under a
    /// grant issued for a different fragment, while adding no resource
    /// capability or grant-floor allowance of its own.
    pub fn for_query_plan(plan: &QueryPlan) -> Option<Self> {
        let requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        Some(Self { requirement })
    }

    /// Derives the proof for the exact native fragment that materializes the
    /// cuGraph edge input. The graph-memory model owns that final resident
    /// frame, so its scope is retained for execution equality but contributes
    /// no second whole-attempt phase.
    pub fn for_cugraph_graph_input_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.mark_cugraph_graph_input_scope();
        Some(Self { requirement })
    }

    /// Derives proof for an exact host handoff of one frontend-attested graph
    /// result. The plan must do no native work: it can only expose the named
    /// GPU-resident graph result to its host sink. Its device frame is still
    /// owned by the graph-execution phase, while its structural scope remains
    /// available for execution-time equality validation.
    pub fn for_attested_graph_result_passthrough_plan(plan: &QueryPlan, graph_result_input_name: &str) -> Option<Self> {
        if !is_attested_graph_result_passthrough_plan(plan, graph_result_input_name) {
            return None;
        }
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.mark_attested_graph_result_passthrough_scope();
        Some(Self { requirement })
    }

    /// Derives proof for the bounded native sort phase that follows one
    /// frontend-attested graph result. The graph result remains the only
    /// external GPU frame; the engine executes the sort through the same
    /// immutable bounded-attempt resource, so it cannot exceed that attempt's
    /// grant while the graph and sort phases remain sequential.
    pub fn for_attested_graph_result_bounded_consumer_plan(
        plan: &QueryPlan,
        graph_result_input_name: &str,
    ) -> Option<Self> {
        if !is_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name) {
            return None;
        }
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.mark_attested_graph_result_bounded_consumer_scope();
        Some(Self { requirement })
    }

    /// Derives a plan-scoped proof for the exact internal plan issued by the
    /// cuGraph frontend.
    ///
    /// The generic graph node analysis deliberately keeps its lower-level
    /// bounded-execution blocker: the engine alone cannot establish that an
    /// embedding graph runtime owns all graph construction and workspace
    /// lifetimes.  cuGraph establishes that contract at its own boundary, so
    /// its engine-issued proof replaces only that blocker with the canonical
    /// cuGraph execution capability. The replacement is bound to a private,
    /// collision-free structural scope for this exact plan; combining proofs
    /// preserves every scope, and execution rechecks it against the plan
    /// about to run. The adapter-only host/deferred-GPU sink boundary is
    /// canonicalized; all execution-relevant graph and operator facts remain
    /// exact.
    pub fn for_cugraph_graph_execution_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        let graph_requirement = generic_graph_execution_blocker();
        if !requirement.required_capabilities.contains(&graph_requirement) {
            return (!requirement.required_capabilities.is_empty()).then_some(Self { requirement });
        }
        requirement.required_capabilities.retain(|candidate| candidate != &graph_requirement);
        requirement.required_capabilities.push(cugraph_graph_execution_capability());
        requirement.attested_cugraph_graph_execution_scopes.push(AttestedCugraphGraphExecutionScope::for_plan(plan));
        Some(Self { requirement })
    }

    /// Combines proofs already issued for native fragments in one invocation.
    ///
    /// Requiring a first proof prevents public callers from using aggregation
    /// as an empty-proof constructor.
    pub fn combine(first: Self, remaining: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut requirements = vec![first.requirement];
        requirements.extend(remaining.into_iter().map(|proof| proof.requirement));
        AttemptCapabilityEvidence::aggregate(requirements).map(|requirement| Self { requirement })
    }

    /// Refines this engine-issued proof with an opaque cold graph-model hint.
    ///
    /// The model-owned graph evidence classifies whether the source frame and
    /// graph input share one owner or are concurrently live through explicit
    /// normalization. It becomes a complete phase only when it can be bound
    /// to one exact cuGraph execution scope; callers cannot supply raw
    /// graph-model bytes or a lifecycle choice.
    pub fn with_graph_memory_admission_evidence(mut self, evidence: &GraphMemoryAdmissionEvidence) -> Self {
        self.requirement.extend_with_graph_memory_admission_evidence(evidence);
        self
    }

    /// Adds the retained source stage for the exact local-Parquet rowset that
    /// a frontend passes directly into a graph execution boundary.
    ///
    /// This accepts the engine-owned source model rather than a caller-supplied
    /// byte count. It is descriptive only and therefore cannot alter the
    /// immutable floor, capability contracts, or ledger fit decision. Native
    /// plans acquire the same hint through [`Self::for_query_plan`]; this
    /// narrow entry point exists for DataFusion scans that a graph frontend
    /// consumes before general native lowering.
    pub fn with_local_parquet_source_stage(mut self, source: &LocalParquetSource) -> Self {
        self.requirement.extend_with_local_parquet_source_stage(source);
        self
    }

    /// Returns the device-capability contracts captured by the engine-issued
    /// proof. This is diagnostic-only and cannot be used to construct a new
    /// proof.
    pub fn required_capabilities(&self) -> &[ReservationRequirement] {
        &self.requirement.required_capabilities
    }

    /// Resolves the proof's immutable floor against an installed profile.
    /// This is read-only inspection and never selects a grant target.
    pub fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        self.requirement.required_floor_bytes_for(configured_floor_bytes)
    }

    /// Returns the proof's descriptive working-set target, if planning
    /// supplied one. This is diagnostic-only; admission remains ledger-owned.
    pub fn soft_working_set_hint_bytes(&self) -> Option<u64> {
        self.requirement.soft_working_set_hint_bytes()
    }

    /// Resolves the proof's private plan shapes against the policy installed
    /// by the query service that is about to admit it.
    pub(crate) fn resolve_reservation_envelope(
        &self,
        policy: AttemptReservationPolicy,
    ) -> Result<AttemptReservationEnvelope> {
        AttemptReservationEnvelope::resolve(&self.requirement.reservation_shapes, policy)
    }

    #[cfg(test)]
    pub(crate) fn into_requirement(self) -> AttemptCapabilityEvidence {
        self.requirement
    }

    pub(crate) fn as_requirement(&self) -> &AttemptCapabilityEvidence {
        &self.requirement
    }
}

impl AttemptCapabilityEvidence {
    /// Returns evidence derived from the engine's capability analysis for one
    /// lowered native plan.  The plan, rather than an embedding caller,
    /// determines the capability set.
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        let source_stage_bytes = estimated_reachable_local_parquet_source_bytes(plan);
        let mut evidence = Self::from_parts(
            0,
            0,
            None,
            source_stage_bytes.map_or(EstimateQuality::Unknown, |_| EstimateQuality::Modeled),
            plan.capability_analysis().requirements,
        );
        evidence.source_stage_bytes = source_stage_bytes;
        evidence.reservation_shapes.push(AttemptReservationShape::for_query_plan(plan));
        evidence
    }

    fn mark_cugraph_graph_input_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_cugraph_graph_input_scope();
        }
    }

    fn mark_attested_graph_result_passthrough_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_attested_graph_result_passthrough_scope();
        }
    }

    fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_attested_graph_result_bounded_consumer_scope();
        }
    }

    /// Returns the descriptive working-set target, if a planner produced one.
    pub(crate) fn soft_working_set_hint_bytes(&self) -> Option<u64> {
        self.soft_working_set_hint_bytes
    }

    /// Returns the provenance quality of the descriptive working-set target.
    #[cfg(test)]
    pub(crate) fn estimate_quality(&self) -> EstimateQuality {
        self.estimate_quality
    }

    /// Resolves this evidence against an already-installed device-profile
    /// floor.  This is read-only inspection; it never selects a grant.
    pub(crate) fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        configured_floor_bytes
            .max(self.configured_floor_bytes)
            .max(self.required_phase_floor_bytes)
            .checked_add(self.capability_floor_raise_bytes)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.admission.requirement",
                    "attempt capability floor overflows u64",
                )
                .with_fact("reason", "attempt_requirement_floor_overflow")
                .with_fact("configured_floor_bytes", configured_floor_bytes)
                .with_fact("required_phase_floor_bytes", self.required_phase_floor_bytes)
                .with_fact("capability_floor_raise_bytes", self.capability_floor_raise_bytes)
            })
    }

    /// Combines evidence from all fragments in one invocation.
    ///
    /// The installed device profile supplies the baseline floor exactly once;
    /// only independently live lower-layer raises are summed here.
    pub(crate) fn aggregate<I: IntoIterator<Item = Self>>(fragments: I) -> Result<Self> {
        Self::aggregate_with_configured_floor(0, fragments)
    }

    pub(crate) fn empty() -> Self {
        Self::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new())
    }

    pub(crate) fn from_parts(
        configured_floor_bytes: u64,
        capability_floor_raise_bytes: u64,
        soft_working_set_hint_bytes: Option<u64>,
        estimate_quality: EstimateQuality,
        required_capabilities: Vec<ReservationRequirement>,
    ) -> Self {
        Self {
            configured_floor_bytes,
            capability_floor_raise_bytes,
            required_phase_floor_bytes: 0,
            source_stage_bytes: None,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
            attested_cugraph_graph_execution_scopes: Vec::new(),
            reservation_shapes: Vec::new(),
        }
    }

    /// Applies a service-resolved envelope immediately before placement. A
    /// graph envelope with an incomplete scope is rejected before this point;
    /// non-graph incomplete shapes retain the ledger's conservative candidate
    /// ceiling rather than a partial phase estimate.
    pub(crate) fn with_reservation_envelope(mut self, envelope: &AttemptReservationEnvelope) -> Self {
        self.required_phase_floor_bytes = self.required_phase_floor_bytes.max(envelope.required_phase_floor_bytes());
        self.soft_working_set_hint_bytes = envelope.complete_phase_peak_bytes();
        self.estimate_quality =
            envelope.complete_phase_peak_bytes().map_or(EstimateQuality::Unknown, |_| EstimateQuality::Modeled);
        self
    }

    /// Combines an engine-issued graph-native cold peak with the matching
    /// native source stage using the graph model's exact source/graph lifetime
    /// classification. Raw graph input shares its prepared frame with the
    /// graph peak; explicit normalization retains the source while producing
    /// a separate graph input and therefore composes additively.
    fn extend_with_graph_memory_admission_evidence(&mut self, evidence: &GraphMemoryAdmissionEvidence) {
        let graph_native_peak_bytes = evidence.cold_model_requested_bytes();
        let whole_attempt_peak_bytes = self.source_stage_bytes.map_or(graph_native_peak_bytes, |source_stage_bytes| {
            evidence.cold_model_with_source_stage_bytes(source_stage_bytes)
        });
        self.soft_working_set_hint_bytes = Some(whole_attempt_peak_bytes);
        self.estimate_quality = EstimateQuality::Modeled;
        let source_stage_bytes = self.source_stage_bytes;
        let Some(graph_shape_index) = self.exact_attested_graph_reservation_shape_index() else {
            return;
        };

        self.reservation_shapes[graph_shape_index].set_graph_memory_phase(evidence, source_stage_bytes);

        self.account_graph_owned_runtime_scopes(graph_shape_index);

        // A logical BFS decode/projection remains a distinct exact runtime
        // scope. It has a typed post-graph peak only for BFS and only for the
        // adapter's private fragment shape; every other GPU-resident scope
        // stays incomplete and therefore clears the whole-attempt target.
        let logical_output_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.is_bfs_logical_output_scope().then_some(index))
            .collect::<Vec<_>>();
        if let (Some(logical_output_peak), [logical_output_scope_index]) =
            (evidence.post_graph_bfs_logical_output_peak_bytes(), logical_output_scope_indices.as_slice())
        {
            self.reservation_shapes[graph_shape_index]
                .set_post_graph_bfs_logical_output_peak_bytes(logical_output_peak);
            self.reservation_shapes[*logical_output_scope_index].account_for_by_graph_phase();
        }
    }

    /// Retains exact scopes for graph-owned frames while treating each one as
    /// part of the already-modeled graph phase. This must run again after
    /// composition, because a prepared graph ticket is refined before an
    /// outer native host-passthrough adds its exact result scope.
    fn account_graph_owned_runtime_scopes(&mut self, graph_shape_index: usize) {
        // The graph phase consumes each exact local-Parquet stage carried by
        // this invocation. Keep those scopes for execution-time equality
        // validation, but do not add their bytes a second time after the
        // graph model already classified their lifetime.
        for (index, shape) in self.reservation_shapes.iter_mut().enumerate() {
            if index != graph_shape_index && shape.has_modeled_source_stage() {
                shape.account_for_by_graph_phase();
            }
        }

        // The adapter identifies exactly one native fragment as the producer
        // of the graph's prepared edge frame. Its output is the graph model's
        // input owner, including literal/DataFusion sources that do not have a
        // local-Parquet stage. Retain the structural scope for execution-time
        // validation while folding that known lifetime into the graph phase.
        let graph_input_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.is_cugraph_graph_input_scope().then_some(index))
            .collect::<Vec<_>>();
        if let [graph_input_scope_index] = graph_input_scope_indices.as_slice() {
            self.reservation_shapes[*graph_input_scope_index].account_for_by_graph_phase();
        }

        // A composed native root may merely expose the graph executor's
        // already-live result frame to its host sink. Only the graph frontend
        // can attest that exact source identity; keep the scope for runtime
        // equality while charging no second phase for the same frame.
        let graph_result_passthrough_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.is_attested_graph_result_passthrough_scope().then_some(index))
            .collect::<Vec<_>>();
        if let [graph_result_passthrough_scope_index] = graph_result_passthrough_scope_indices.as_slice() {
            self.reservation_shapes[*graph_result_passthrough_scope_index].account_for_by_graph_phase();
        }

        // A native full sort over that same graph result is executed only
        // after the graph producer has yielded. Its scratch/output allocations
        // are charged to the engine's existing immutable attempt resource;
        // unlike another graph/input source, it cannot introduce an
        // unbounded external GPU owner. Keep its exact scope for execution
        // validation and account the sequential bounded phase once.
        let graph_result_bounded_consumer_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.is_attested_graph_result_bounded_consumer_scope().then_some(index))
            .collect::<Vec<_>>();
        if let [graph_result_bounded_consumer_scope_index] = graph_result_bounded_consumer_scope_indices.as_slice() {
            self.reservation_shapes[*graph_result_bounded_consumer_scope_index].account_for_by_graph_phase();
        }
    }

    fn extend_with_local_parquet_source_stage(&mut self, source: &LocalParquetSource) {
        let Some(source_stage_bytes) = estimated_local_parquet_source_bytes(source) else {
            return;
        };
        self.source_stage_bytes = Some(match self.source_stage_bytes {
            Some(existing) => existing.saturating_add(source_stage_bytes),
            None => source_stage_bytes,
        });
        if let Some(shape_index) = self.exact_attested_graph_reservation_shape_index() {
            self.reservation_shapes[shape_index].add_source_stage_bytes(source_stage_bytes);
        }
        self.estimate_quality = EstimateQuality::Modeled;
    }

    /// A graph model token has no caller-provided plan identity. It can only
    /// refine a proof that attests exactly one graph execution scope, and it
    /// must attach to the corresponding reservation shape rather than a
    /// generic fragment in a composed proof.
    fn exact_attested_graph_reservation_shape_index(&self) -> Option<usize> {
        let [scope] = self.attested_cugraph_graph_execution_scopes.as_slice() else {
            return None;
        };
        let mut matching_shapes = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| scope.matches_reservation_shape(shape).then_some(index));
        let index = matching_shapes.next()?;
        matching_shapes.next().is_none().then_some(index)
    }

    /// Stable, human-readable identities for structured device-capability
    /// rejection facts.
    pub(crate) fn capability_labels(required_capabilities: &[ReservationRequirement]) -> String {
        required_capabilities
            .iter()
            .map(|capability| format!("{}:{}", capability.kind.as_str(), capability.required_contract))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The non-negotiable attempt floor, rejecting integer overflow.
    pub(crate) fn required_floor_bytes(&self) -> Result<u64> {
        self.required_floor_bytes_for(self.configured_floor_bytes)
    }

    /// Verifies that this admitted requirement covers the plan about to run.
    ///
    /// A ticket may aggregate proofs for several native fragments, so the
    /// admitted set may be a strict superset.  Missing contracts are unsafe:
    /// the execution plan must be rejected before it allocates or dispatches
    /// GPU work.
    pub(crate) fn validate_covers_plan(&self, plan: &QueryPlan) -> Result<()> {
        let required_capabilities = plan.capability_analysis().requirements;
        let cugraph_graph_execution_is_proven =
            self.required_capabilities.contains(&cugraph_graph_execution_capability())
                && self.attested_cugraph_graph_execution_scopes.iter().any(|scope| scope.matches(plan));
        let missing_capabilities = required_capabilities
            .iter()
            .filter(|required| {
                !self.required_capabilities.contains(required)
                    && !(cugraph_graph_execution_is_proven && is_generic_graph_execution_blocker(required))
            })
            .cloned()
            .collect::<Vec<_>>();
        if missing_capabilities.is_empty() {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.requirement",
            "admitted capability proof does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_capability_evidence_mismatch")
        .with_fact("required_capabilities", Self::capability_labels(&required_capabilities))
        .with_fact("admitted_capabilities", Self::capability_labels(&self.required_capabilities))
        .with_fact("missing_capabilities", Self::capability_labels(&missing_capabilities)))
    }

    /// Aggregates all native fragments in one invocation.
    ///
    /// Capability workspace raises are summed deliberately: independently
    /// scheduled fragments can be concurrently live under one attempt grant.
    /// The configured deployment floor is supplied once and is never summed.
    pub(crate) fn aggregate_with_configured_floor<I: IntoIterator<Item = Self>>(
        configured_floor_bytes: u64,
        fragments: I,
    ) -> Result<Self> {
        let mut capability_floor_raise_bytes = 0_u64;
        let mut required_phase_floor_bytes = 0_u64;
        let mut source_stage_bytes = Some(0_u64);
        let mut soft_working_set_hint_bytes = Some(0_u64);
        let mut estimate_quality = EstimateQuality::Measured;
        let mut required_capabilities = Vec::new();
        let mut attested_cugraph_graph_execution_scopes = Vec::new();
        let mut reservation_shapes = Vec::new();
        for fragment in fragments {
            capability_floor_raise_bytes =
                capability_floor_raise_bytes.checked_add(fragment.capability_floor_raise_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.admission.requirement",
                        "aggregated capability floor raise overflows u64",
                    )
                    .with_fact("reason", "attempt_requirement_capability_raise_overflow")
                })?;
            required_phase_floor_bytes = required_phase_floor_bytes.max(fragment.required_phase_floor_bytes);
            source_stage_bytes = match (source_stage_bytes, fragment.source_stage_bytes) {
                (Some(left), Some(right)) => left.checked_add(right),
                _ => None,
            };
            soft_working_set_hint_bytes = match (soft_working_set_hint_bytes, fragment.soft_working_set_hint_bytes) {
                (Some(left), Some(right)) => Some(left.max(right)),
                _ => None,
            };
            estimate_quality = estimate_quality.min(fragment.estimate_quality);
            for capability in fragment.required_capabilities {
                if !required_capabilities.contains(&capability) {
                    required_capabilities.push(capability);
                }
            }
            for scope in fragment.attested_cugraph_graph_execution_scopes {
                if !attested_cugraph_graph_execution_scopes.contains(&scope) {
                    attested_cugraph_graph_execution_scopes.push(scope);
                }
            }
            for shape in fragment.reservation_shapes {
                if !reservation_shapes.contains(&shape) {
                    reservation_shapes.push(shape);
                }
            }
        }
        let mut aggregate = Self::from_parts(
            configured_floor_bytes,
            capability_floor_raise_bytes,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
        );
        aggregate.required_phase_floor_bytes = required_phase_floor_bytes;
        aggregate.source_stage_bytes = source_stage_bytes;
        aggregate.attested_cugraph_graph_execution_scopes = attested_cugraph_graph_execution_scopes;
        aggregate.reservation_shapes = reservation_shapes;
        if let Some(graph_shape_index) = aggregate.exact_attested_graph_reservation_shape_index()
            && aggregate.reservation_shapes[graph_shape_index].has_graph_memory_phase()
        {
            aggregate.account_graph_owned_runtime_scopes(graph_shape_index);
        }
        Ok(aggregate)
    }
}

fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

fn cugraph_graph_execution_capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        crate::capability::ContractQuality::CapabilityProof,
    )
}

fn is_generic_graph_execution_blocker(requirement: &ReservationRequirement) -> bool {
    requirement == &generic_graph_execution_blocker()
}

fn is_attested_graph_result_passthrough_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
    let [node] = plan.nodes() else {
        return false;
    };
    if !plan.unreachable_node_ids().is_empty() || plan.root_id() != node.id {
        return false;
    }
    matches!(
        &node.kind,
        PlanNode::Source(source)
            if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name)
    )
}

fn is_attested_graph_result_bounded_consumer_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
    let mut graph_result_source_count = 0_usize;
    for node in plan.nodes() {
        if plan.unreachable_node_ids().contains(&node.id) {
            continue;
        }
        match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name) =>
            {
                graph_result_source_count += 1;
            },
            PlanNode::Sort(_) => {},
            _ => return false,
        }
    }
    graph_result_source_count == 1
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        capability::{ContractQuality, ReservationRequirement},
        chunk::SourceChunkPlanningOptions,
        graph_memory::{GraphMemoryCalibrationOverride, graph_memory_admission_evidence_for_edge_list},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, QueryPlan, QueryPlanBuilder,
        },
        runtime::{ReservationKind, SourceChunkPlanningMode, SourceSchedulerPolicy},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    fn capability() -> ReservationRequirement {
        ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "test",
            "test_contract",
            ContractQuality::CapabilityProof,
        )
    }

    fn graph_bfs_plan() -> QueryPlan {
        graph_bfs_plan_with_source_vertex(0)
    }

    fn graph_bfs_plan_with_source_vertex(source_vertex: i64) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", schema)));
        let graph = builder.add_graph_algorithm(
            edges,
            GraphAlgorithmSpec::with_stable_schema(
                GraphAlgorithm::Bfs,
                GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
                GraphConstructionOptions::default(),
                GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(source_vertex, None))),
            ),
        );
        builder.finish_with_host_sink(graph)
    }

    fn ordinary_native_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        builder.finish_with_host_sink(source)
    }

    fn reservation_policy() -> AttemptReservationPolicy {
        AttemptReservationPolicy::new(
            64 * 1024,
            1,
            3,
            1,
            SourceSchedulerPolicy::new(2),
            SourceChunkPlanningMode::default(),
            SourceChunkPlanningOptions::default(),
        )
    }

    #[test]
    fn known_local_parquet_sources_keep_a_separate_modeled_source_stage() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/requirement-known.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![100])
                    .with_projected_row_group_bytes(vec![40], vec![96]),
            ],
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_with_host_sink(source);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        assert_eq!(requirement.source_stage_bytes, Some(96));
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.estimate_quality(), EstimateQuality::Modeled);
    }

    #[test]
    fn unknown_local_parquet_sources_issue_no_modeled_source_stage() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/requirement-unknown.parquet",
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_with_host_sink(source);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        assert_eq!(requirement.source_stage_bytes, None);
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.estimate_quality(), EstimateQuality::Unknown);
    }

    #[test]
    fn zero_column_count_source_resolves_a_parquet_read_floor_without_a_partial_target() {
        let source = LocalParquetSource::zero_column_count(vec![
            LocalParquetFile::new("/tmp/requirement-count.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![4_096])
                .with_projected_row_group_bytes(vec![0], vec![0]),
        ]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source);

        let minimum_read_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("sink plan must issue a proof");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        let requirement = proof.into_requirement().with_reservation_envelope(&envelope);

        assert_eq!(requirement.source_stage_bytes, Some(0));
        assert_eq!(envelope.required_phase_floor_bytes(), minimum_read_budget);
        assert_eq!(envelope.complete_phase_peak_bytes(), None);
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.required_floor_bytes_for(1).unwrap(), minimum_read_budget);
    }

    #[test]
    fn graph_evidence_does_not_double_count_its_prepared_local_source_frame() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/retained-graph-source.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            vec!["src".to_owned(), "dst".to_owned()],
        )));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_with_host_sink(graph);
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &spec,
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).soft_working_set_hint_bytes(),
            Some(96_u64.max(evidence.cold_model_requested_bytes())),
            "the graph cold peak already owns the prepared edge frame"
        );
    }

    #[test]
    fn graph_evidence_adds_a_source_stage_that_normalizes_into_a_separate_graph_input() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        )
        .with_edge_normalize_inserted(true);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/normalized-graph-source.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            vec!["src".to_owned(), "dst".to_owned()],
        )));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_with_host_sink(graph);
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &spec,
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).soft_working_set_hint_bytes(),
            Some(96_u64.saturating_add(evidence.cold_model_requested_bytes())),
            "the explicit normalize source and derived graph input are separate live owners"
        );
    }

    #[test]
    fn graph_evidence_without_a_source_stage_remains_the_native_peak() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &spec,
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_bfs_plan())
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).soft_working_set_hint_bytes(),
            Some(evidence.cold_model_requested_bytes())
        );
    }

    #[test]
    fn required_floor_uses_checked_arithmetic() {
        let requirement =
            AttemptCapabilityEvidence::from_parts(u64::MAX, 1, None, EstimateQuality::Unknown, Vec::new());
        assert_eq!(
            requirement.required_floor_bytes().unwrap_err().fact_value("reason"),
            Some("attempt_requirement_floor_overflow")
        );
    }

    #[test]
    fn legacy_band_is_an_interim_floor_and_hint() {
        let requirement = AttemptCapabilityEvidence::from_parts(10, 0, Some(20), EstimateQuality::Modeled, Vec::new());
        assert_eq!(requirement.required_floor_bytes().unwrap(), 10);
        assert_eq!(requirement.soft_working_set_hint_bytes, Some(20));
        assert_eq!(requirement.estimate_quality, EstimateQuality::Modeled);
    }

    #[test]
    fn aggregate_sums_only_raises_and_unions_capabilities() {
        let capability = capability();
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            10,
            [
                AttemptCapabilityEvidence::from_parts(
                    99,
                    2,
                    Some(12),
                    EstimateQuality::Measured,
                    vec![capability.clone()],
                ),
                AttemptCapabilityEvidence::from_parts(88, 3, Some(20), EstimateQuality::Unknown, vec![capability]),
            ],
        )
        .unwrap();
        assert_eq!(result.configured_floor_bytes, 10);
        assert_eq!(result.capability_floor_raise_bytes, 5);
        assert_eq!(result.soft_working_set_hint_bytes, Some(20));
        assert_eq!(result.estimate_quality, EstimateQuality::Unknown);
        assert_eq!(result.required_capabilities.len(), 1);
    }

    #[test]
    fn aggregate_clears_the_target_when_any_fragment_is_unmodeled() {
        let result = AttemptCapabilityEvidence::aggregate([
            AttemptCapabilityEvidence::from_parts(0, 0, Some(12), EstimateQuality::Modeled, Vec::new()),
            AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new()),
        ])
        .unwrap();

        assert_eq!(result.soft_working_set_hint_bytes(), None);
    }

    #[test]
    fn aggregate_rejects_raise_overflow() {
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            0,
            [
                AttemptCapabilityEvidence::empty(),
                AttemptCapabilityEvidence::from_parts(0, u64::MAX, None, EstimateQuality::Unknown, Vec::new()),
                AttemptCapabilityEvidence::from_parts(0, 1, None, EstimateQuality::Unknown, Vec::new()),
            ],
        );
        assert_eq!(result.unwrap_err().fact_value("reason"), Some("attempt_requirement_capability_raise_overflow"));
    }

    #[test]
    fn cugraph_graph_scope_rejects_an_unattested_native_plan_with_stable_facts() {
        let attested_plan = graph_bfs_plan_with_source_vertex(1);
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&attested_plan)
            .expect("attested graph execution has capability requirements");
        let error = proof
            .as_requirement()
            .validate_covers_plan(&graph_bfs_plan())
            .expect_err("a plan-scoped cuGraph proof must not authorize a different native graph plan");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }

    #[test]
    fn exact_cugraph_plan_proof_replaces_only_the_engine_graph_blocker() {
        let plan = graph_bfs_plan();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph execution has capability requirements");

        assert!(proof.required_capabilities().iter().any(|requirement| {
            requirement.kind == ReservationKind::GraphAlgorithm
                && requirement.required_contract == "cugraph_graph_execution_contract"
        }));
        assert!(!proof.required_capabilities().iter().any(is_generic_graph_execution_blocker));
        assert_eq!(proof.as_requirement().attested_cugraph_graph_execution_scopes.len(), 1);
        proof
            .as_requirement()
            .validate_covers_plan(&plan)
            .expect("the issued cuGraph proof must cover its exact internal plan");
    }

    #[test]
    fn combining_cugraph_proofs_retains_each_attested_graph_scope() {
        let native_plan = ordinary_native_plan();
        let first_plan = graph_bfs_plan_with_source_vertex(0);
        let second_plan = graph_bfs_plan_with_source_vertex(1);
        let unscoped_plan = graph_bfs_plan_with_source_vertex(2);
        let proof = QueryPlanCapabilityProof::combine(
            QueryPlanCapabilityProof::for_query_plan(&native_plan)
                .expect("native plan has an explicit sink capability requirement"),
            [
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&first_plan)
                    .expect("first graph plan has capability requirements"),
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan)
                    .expect("second graph plan has capability requirements"),
            ],
        )
        .expect("native and plan-scoped graph proofs must combine");

        let requirement = proof.as_requirement();
        assert_eq!(requirement.attested_cugraph_graph_execution_scopes.len(), 2);
        requirement.validate_covers_plan(&native_plan).expect("combined proof must retain the native plan capability");
        requirement.validate_covers_plan(&first_plan).expect("combined proof must retain the first graph scope");
        requirement.validate_covers_plan(&second_plan).expect("combined proof must retain the second graph scope");
        let error = requirement
            .validate_covers_plan(&unscoped_plan)
            .expect_err("combined proof must not widen its graph scope");
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }
}
