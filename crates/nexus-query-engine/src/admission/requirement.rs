//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

use std::collections::{BTreeSet, HashMap, HashSet};

use arrow::datatypes::{DataType, SchemaRef, TimeUnit};

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{
            AttemptReservationProducerOccurrence, AttemptReservationSchedule, AttemptReservationShape,
            canonical_execution_scope_fingerprint,
        },
    },
    capability::ReservationRequirement,
    error::{QueryError, QueryErrorCode, Result},
    expr::alias_wrapped_column,
    graph_memory::GraphMemoryAdmissionEvidence,
    plan::{
        JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, estimated_local_parquet_source_bytes,
        estimated_reachable_local_parquet_source_bytes,
    },
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

/// Adapter-observed identity and opaque admission proof for one graph producer
/// in a composed attempt.
///
/// The adapter may report an unavailable identity or proof, but it cannot
/// provide graph-memory bytes or construct a reservation shape. The engine
/// issues a composed schedule only when every observation is complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedGraphProducerCandidate {
    runtime_occurrence_id: Option<String>,
    graph_result_input_name: Option<String>,
    capability_proof: Option<QueryPlanCapabilityProof>,
}

impl ComposedGraphProducerCandidate {
    /// Creates one adapter-observed graph-producer candidate in execution
    /// order.
    pub fn new(
        runtime_occurrence_id: Option<String>,
        graph_result_input_name: Option<String>,
        capability_proof: Option<QueryPlanCapabilityProof>,
    ) -> Self {
        Self { runtime_occurrence_id, graph_result_input_name, capability_proof }
    }
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

    /// Combines one consumer proof with graph-producer proofs in the exact
    /// order used by the composed executor.
    ///
    /// This seam carries only opaque, engine-issued proofs and occurrence
    /// identifiers. It never accepts adapter-authored byte arithmetic: graph
    /// output evidence stays attached to each producer proof and the native
    /// consumer's allocations are derived from its exact engine IR here.
    pub fn combine_composed_graph_attempt(
        consumer_plan: &QueryPlan,
        consumer: Self,
        ordered_producers: impl IntoIterator<Item = (String, Self)>,
    ) -> Result<Self> {
        let ordered_producers = ordered_producers.into_iter().collect::<Vec<_>>();
        if ordered_producers.is_empty() {
            return Err(composed_schedule_error("missing_producer_evidence", None));
        }

        let [consumer_shape] = consumer.requirement.reservation_shapes.as_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        if !consumer_shape.matches(consumer_plan) {
            return Err(composed_schedule_error("consumer_scope_coverage_mismatch", None));
        }
        let consumer_scope = consumer_shape.execution_scope_fingerprint().to_owned();

        let mut occurrence_ids = std::collections::HashSet::with_capacity(ordered_producers.len());
        let mut occurrences = Vec::with_capacity(ordered_producers.len());
        let mut producer_outputs = Vec::with_capacity(ordered_producers.len());
        let mut producer_proofs = Vec::with_capacity(ordered_producers.len());
        for (occurrence_id, producer) in ordered_producers {
            if occurrence_id.is_empty() {
                return Err(composed_schedule_error("missing_producer_occurrence_id", None));
            }
            if !occurrence_ids.insert(occurrence_id.clone()) {
                return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&occurrence_id)));
            }
            let [attested_scope] = producer.requirement.attested_cugraph_graph_execution_scopes.as_slice() else {
                return Err(composed_schedule_error("missing_producer_scope", Some(&occurrence_id)));
            };
            let matching_shapes = producer
                .requirement
                .reservation_shapes
                .iter()
                .filter(|shape| attested_scope.matches_reservation_shape(shape))
                .collect::<Vec<_>>();
            let [producer_shape] = matching_shapes.as_slice() else {
                return Err(composed_schedule_error("producer_scope_coverage_mismatch", Some(&occurrence_id)));
            };
            let Some(graph_evidence) = producer_shape.graph_memory_admission_evidence() else {
                return Err(composed_schedule_error("missing_producer_graph_memory_evidence", Some(&occurrence_id)));
            };
            let retained_output_bytes = graph_evidence.retained_output_bytes();
            producer_outputs.push(ComposedGraphProducerOutputEvidence {
                occurrence_id: occurrence_id.clone(),
                retained_output_bytes,
                row_count_upper_bound: graph_evidence.retained_output_rows_upper_bound(),
            });
            occurrences.push(AttemptReservationProducerOccurrence::new(
                occurrence_id,
                producer_shape.execution_scope_fingerprint().to_owned(),
                Some(graph_evidence.cold_model_requested_bytes()),
                Some(retained_output_bytes),
            ));
            producer_proofs.push(producer);
        }

        let consumer_additional_peak_bytes =
            composed_consumer_additional_peak_bytes_for_plan(consumer_plan, &producer_outputs)?;
        let schedule = AttemptReservationSchedule::new(
            // Current bounded composed attempts have no independent
            // attempt-wide device allocation.  Keep this explicit so a future
            // attempt-wide owner must be represented in the schedule rather
            // than silently folded into a producer or consumer phase.
            Some(0),
            occurrences,
            Some(consumer_scope),
            Some(consumer_additional_peak_bytes),
        );
        let mut consumer = consumer;
        let [consumer_shape] = consumer.requirement.reservation_shapes.as_mut_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        consumer_shape.attach_ordered_schedule(schedule)?;

        let first = producer_proofs.remove(0);
        AttemptCapabilityEvidence::aggregate(
            std::iter::once(first.requirement)
                .chain(producer_proofs.into_iter().map(|proof| proof.requirement))
                .chain(std::iter::once(consumer.requirement)),
        )
        .map(|requirement| Self { requirement })
    }

    /// Resolves adapter-observed graph-producer identities into the complete
    /// composed schedule when the engine owns every allocation in the attempt.
    ///
    /// Every producer count, including `N = 1`, uses this generalized path.
    /// A partial schedule cannot establish the live allocation timeline, so it
    /// fails closed with the underlying structured schedule error.
    pub fn resolve_composed_graph_attempt(
        consumer_plan: &QueryPlan,
        consumer: Self,
        ordered_producers: impl IntoIterator<Item = ComposedGraphProducerCandidate>,
    ) -> Result<Self> {
        let ordered_producers = ordered_producers.into_iter().collect::<Vec<_>>();
        let complete_schedule = (|| {
            let ordered_proofs = ordered_producers
                .into_iter()
                .map(|producer| {
                    let occurrence_id = producer
                        .runtime_occurrence_id
                        .ok_or_else(|| composed_schedule_error("runtime_producer_occurrence_missing", None))?;
                    let graph_result_input_name = producer.graph_result_input_name.ok_or_else(|| {
                        composed_schedule_error("producer_graph_evidence_missing", Some(&occurrence_id))
                    })?;
                    if occurrence_id != graph_result_input_name {
                        return Err(composed_schedule_error(
                            "producer_occurrence_identity_mismatch",
                            Some(&occurrence_id),
                        )
                        .with_fact("graph_result_input_name", graph_result_input_name));
                    }
                    let proof = producer.capability_proof.ok_or_else(|| {
                        composed_schedule_error("producer_admission_proof_missing_or_ambiguous", Some(&occurrence_id))
                    })?;
                    Ok((occurrence_id, proof))
                })
                .collect::<Result<Vec<_>>>()?;
            Self::combine_composed_graph_attempt(consumer_plan, consumer, ordered_proofs)
        })();

        complete_schedule
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
                // A reservation shape is an allocation occurrence.  Structural
                // equality is evidence identity, not runtime multiplicity;
                // equal graph producers must remain separate retained outputs.
                reservation_shapes.push(shape);
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

#[derive(Debug, Clone)]
struct ComposedGraphProducerOutputEvidence {
    occurrence_id: String,
    retained_output_bytes: u64,
    row_count_upper_bound: u64,
}

#[derive(Debug, Clone)]
struct ComposedConsumerBranchBound {
    rows: u64,
    schema: SchemaRef,
    /// Bytes allocated by this consumer branch that remain live at its root.
    /// Producer result handles are intentionally excluded: the schedule fold
    /// retains them outside the consumer phase.
    final_owned_bytes: u64,
    /// Maximum additional bytes owned while this branch is materialized.
    additional_peak_bytes: u64,
}

fn composed_consumer_additional_peak_bytes_for_plan(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
) -> Result<u64> {
    if !consumer_plan.unreachable_node_ids().is_empty() {
        return Err(composed_schedule_error("consumer_contains_unreachable_nodes", None));
    }
    let mut outputs = HashMap::with_capacity(producer_outputs.len());
    for output in producer_outputs {
        if output.occurrence_id.is_empty() || outputs.insert(output.occurrence_id.as_str(), output).is_some() {
            return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&output.occurrence_id)));
        }
    }
    let mut used_occurrences = HashSet::with_capacity(outputs.len());
    let root = consumer_plan.root_id();
    let additional_peak_bytes = match &consumer_plan.root_node().kind {
        PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) => {
            composed_consumer_branch_bound(consumer_plan, root, &outputs, &mut used_occurrences)?.additional_peak_bytes
        },
        PlanNode::Union(union) => {
            if union.inputs.len() != producer_outputs.len() {
                return Err(composed_schedule_error("consumer_union_occurrence_count_mismatch", None));
            }
            let mut total_rows = 0_u64;
            let mut branch_peak_bytes = 0_u64;
            for input in &union.inputs {
                let branch = composed_consumer_branch_bound(consumer_plan, *input, &outputs, &mut used_occurrences)?;
                total_rows = checked_consumer_add(total_rows, branch.rows, "consumer_union_rows_overflow")?;
                branch_peak_bytes = checked_consumer_add(
                    branch_peak_bytes,
                    branch.additional_peak_bytes,
                    "consumer_union_branch_peak_overflow",
                )?;
            }
            let union_output = bounded_consumer_frame_bytes(total_rows, consumer_plan.root_node().output_schema())?;
            checked_consumer_add(branch_peak_bytes, union_output, "consumer_union_output_overflow")?
        },
        PlanNode::Join(join) => {
            composed_consumer_join_peak_bytes(consumer_plan, join, &outputs, &mut used_occurrences)?
        },
        _ => return Err(composed_schedule_error("consumer_topology_unmodeled", None)),
    };
    if used_occurrences.len() != producer_outputs.len() {
        return Err(composed_schedule_error("consumer_occurrence_coverage_mismatch", None)
            .with_fact("consumer_occurrence_count", used_occurrences.len())
            .with_fact("producer_occurrence_count", producer_outputs.len()));
    }
    Ok(additional_peak_bytes)
}

fn composed_consumer_branch_bound(
    plan: &QueryPlan,
    node_id: NodeId,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
) -> Result<ComposedConsumerBranchBound> {
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    match &node.kind {
        PlanNode::Source(source) => {
            let SourceNode::GpuResident(source) = source.as_ref() else {
                return Err(composed_schedule_error("consumer_non_graph_source", None));
            };
            let Some(output) = outputs.get(source.input_name.as_str()) else {
                return Err(composed_schedule_error("consumer_source_occurrence_missing", Some(&source.input_name)));
            };
            if !used_occurrences.insert(source.input_name.clone()) {
                return Err(composed_schedule_error("consumer_source_occurrence_reused", Some(&source.input_name)));
            }
            if source.row_count_upper_bound.is_some_and(|rows| rows != output.row_count_upper_bound) {
                return Err(composed_schedule_error("consumer_source_row_bound_mismatch", Some(&source.input_name))
                    .with_fact("producer_row_count_upper_bound", output.row_count_upper_bound)
                    .with_fact("consumer_row_count_upper_bound", source.row_count_upper_bound.unwrap_or_default()));
            }
            // The source handle is retained by the producer schedule, not
            // allocated by the consumer. Its exact typed bytes are therefore
            // intentionally excluded from this branch's additional phase.
            let _typed_retained_output_bytes = output.retained_output_bytes;
            Ok(ComposedConsumerBranchBound {
                rows: output.row_count_upper_bound,
                schema: source.schema.clone(),
                final_owned_bytes: 0,
                additional_peak_bytes: 0,
            })
        },
        PlanNode::Filter(filter) => {
            let input = composed_consumer_branch_bound(plan, filter.input, outputs, used_occurrences)?;
            let output_bytes = bounded_consumer_frame_bytes(input.rows, node.output_schema())?;
            // cuDF filter materializes the selected frame while a Boolean
            // predicate mask is live. Size the mask through the same typed
            // frame-bound primitive rather than treating it as host metadata.
            let predicate_mask_bytes = bounded_consumer_boolean_mask_bytes(input.rows)?;
            let stage_bytes = checked_consumer_add(
                checked_consumer_add(input.final_owned_bytes, output_bytes, "consumer_filter_output_overflow")?,
                predicate_mask_bytes,
                "consumer_filter_mask_overflow",
            )?;
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: output_bytes,
                additional_peak_bytes: input.additional_peak_bytes.max(stage_bytes),
            })
        },
        PlanNode::Projection(projection) => {
            let input = composed_consumer_branch_bound(plan, projection.input, outputs, used_occurrences)?;
            let mut selected_columns = HashSet::with_capacity(projection.outputs.len());
            for output in &projection.outputs {
                let Some(column) = alias_wrapped_column(&output.expr) else {
                    return Err(composed_schedule_error("consumer_projection_expression_workspace_unmodeled", None));
                };
                if !selected_columns.insert(column.name()) {
                    return Err(composed_schedule_error("consumer_projection_source_column_reused", None)
                        .with_fact("column", column.name()));
                }
                let source = input.schema.field_with_name(column.name()).map_err(|_| {
                    composed_schedule_error("consumer_projection_source_column_missing", None)
                        .with_fact("column", column.name())
                })?;
                if source.data_type() != output.field.data_type() {
                    return Err(composed_schedule_error("consumer_projection_output_type_mismatch", None)
                        .with_fact("column", column.name())
                        .with_fact("input_type", source.data_type())
                        .with_fact("output_type", output.field.data_type()));
                }
            }
            // Column selection and rename retain their input columns. The
            // executor takes the same zero-copy path, so the projection owns
            // neither a fresh frame nor a duplicate of branch-owned data.
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: input.final_owned_bytes,
                additional_peak_bytes: input.additional_peak_bytes,
            })
        },
        _ => Err(composed_schedule_error("consumer_branch_topology_unmodeled", None)),
    }
}

fn composed_consumer_join_peak_bytes(
    plan: &QueryPlan,
    join: &JoinNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
) -> Result<u64> {
    if join.spec.join_type != JoinType::Inner || join.spec.residual_predicate.is_some() || join.spec.keys.is_empty() {
        return Err(composed_schedule_error("consumer_join_topology_unmodeled", None));
    }
    let left = composed_consumer_branch_bound(plan, join.inputs[0], outputs, used_occurrences)?;
    let right = composed_consumer_branch_bound(plan, join.inputs[1], outputs, used_occurrences)?;
    let (build, probe, build_keys) = match join.spec.build_side {
        JoinSide::Left => (&left, &right, join.spec.keys.iter().map(|key| &key.left).collect::<Vec<_>>()),
        JoinSide::Right => (&right, &left, join.spec.keys.iter().map(|key| &key.right).collect::<Vec<_>>()),
    };
    let key_type_ids = build_keys
        .iter()
        .map(|key| non_nullable_fixed_width_cudf_type_id(&build.schema, key.name()))
        .collect::<Result<Vec<_>>>()?;
    // Probe keys must retain the same fixed-width non-null contract. This is
    // stricter than the metadata preflight's build requirement and prevents a
    // consumer proof from accepting a join whose runtime key conversion has no
    // typed allocation contract.
    for key in &join.spec.keys {
        let probe_key = match join.spec.build_side {
            JoinSide::Left => &key.right,
            JoinSide::Right => &key.left,
        };
        non_nullable_fixed_width_cudf_type_id(&probe.schema, probe_key.name())?;
    }
    let output_rows = left
        .rows
        .checked_mul(right.rows)
        .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?;
    let output_rows_usize = usize::try_from(output_rows)
        .map_err(|_| composed_schedule_error("consumer_join_output_rows_exceed_usize", None))?;
    let hash_table_bytes =
        cudf::ops::join::hash_join_pre_build_reservation_size_bytes_from_non_nullable_fixed_width_metadata(
            &key_type_ids,
            build.rows,
            cudf::DataFrameHashJoinOptions::new().load_factor_value(),
        )
        .map_err(|error| {
            composed_schedule_error("consumer_join_hash_preflight_unavailable", None)
                .with_fact("cudf_error", error.to_string())
        })?;
    let hash_table_bytes = u64::try_from(hash_table_bytes)
        .map_err(|_| composed_schedule_error("consumer_join_hash_preflight_exceeds_u64", None))?;
    let pair_indices_bytes = cudf::DataFrameHashJoinPairIndices::reservation_size_bytes_for_len(output_rows_usize)
        .map_err(|_| composed_schedule_error("consumer_join_pair_indices_preflight_unavailable", None))?;
    let pair_indices_bytes = u64::try_from(pair_indices_bytes)
        .map_err(|_| composed_schedule_error("consumer_join_pair_indices_exceed_u64", None))?;
    let output_bytes = bounded_consumer_frame_bytes(output_rows, plan.root_node().output_schema())?;
    let build_selected_bytes =
        bounded_selected_join_input_bytes(&build.schema, build.rows, join, join.spec.build_side)?;
    let probe_selected_bytes =
        bounded_selected_join_input_bytes(&probe.schema, probe.rows, join, opposite_join_side(join.spec.build_side))?;
    [
        left.additional_peak_bytes,
        right.additional_peak_bytes,
        build_selected_bytes,
        probe_selected_bytes,
        hash_table_bytes,
        pair_indices_bytes,
        output_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| checked_consumer_add(total, bytes, "consumer_join_peak_overflow"))
}

fn bounded_selected_join_input_bytes(schema: &SchemaRef, rows: u64, join: &JoinNode, side: JoinSide) -> Result<u64> {
    let mut names = BTreeSet::new();
    for key in &join.spec.keys {
        let column = match side {
            JoinSide::Left => &key.left,
            JoinSide::Right => &key.right,
        };
        names.insert(column.name());
    }
    for column in join.spec.output_projection.iter().filter(|column| column.side == side) {
        names.insert(column.column.name());
    }
    if names.is_empty() {
        return Err(composed_schedule_error("consumer_join_selected_input_empty", None));
    }
    let fields = names
        .into_iter()
        .map(|name| {
            schema.field_with_name(name).map(|field| field.as_ref().clone()).map_err(|_| {
                composed_schedule_error("consumer_join_selected_column_missing", None).with_fact("column", name)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    bounded_consumer_frame_bytes(rows, &std::sync::Arc::new(arrow::datatypes::Schema::new(fields)))
}

fn opposite_join_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

fn non_nullable_fixed_width_cudf_type_id(schema: &SchemaRef, field_name: &str) -> Result<cudf::TypeId> {
    let field = schema
        .field_with_name(field_name)
        .map_err(|_| composed_schedule_error("consumer_join_key_missing", None).with_fact("column", field_name))?;
    if field.is_nullable() {
        return Err(composed_schedule_error("consumer_join_nullable_key", None).with_fact("column", field_name));
    }
    arrow_fixed_width_cudf_type_id(field.data_type()).ok_or_else(|| {
        composed_schedule_error("consumer_join_variable_width_key", None).with_fact("column", field_name)
    })
}

fn arrow_fixed_width_cudf_type_id(data_type: &DataType) -> Option<cudf::TypeId> {
    Some(match data_type {
        DataType::Boolean => cudf::TypeId::Bool8,
        DataType::Int8 => cudf::TypeId::Int8,
        DataType::Int16 => cudf::TypeId::Int16,
        DataType::Int32 => cudf::TypeId::Int32,
        DataType::Int64 => cudf::TypeId::Int64,
        DataType::UInt8 => cudf::TypeId::UInt8,
        DataType::UInt16 => cudf::TypeId::UInt16,
        DataType::UInt32 => cudf::TypeId::UInt32,
        DataType::UInt64 => cudf::TypeId::UInt64,
        DataType::Float32 => cudf::TypeId::Float32,
        DataType::Float64 => cudf::TypeId::Float64,
        DataType::Date32 => cudf::TypeId::TimestampDays,
        DataType::Date64 => cudf::TypeId::TimestampMilliseconds,
        DataType::Time32(TimeUnit::Second) => cudf::TypeId::DurationSeconds,
        DataType::Time32(TimeUnit::Millisecond) => cudf::TypeId::DurationMilliseconds,
        DataType::Time64(TimeUnit::Microsecond) => cudf::TypeId::DurationMicroseconds,
        DataType::Time64(TimeUnit::Nanosecond) => cudf::TypeId::DurationNanoseconds,
        DataType::Timestamp(TimeUnit::Second, None) => cudf::TypeId::TimestampSeconds,
        DataType::Timestamp(TimeUnit::Millisecond, None) => cudf::TypeId::TimestampMilliseconds,
        DataType::Timestamp(TimeUnit::Microsecond, None) => cudf::TypeId::TimestampMicroseconds,
        DataType::Timestamp(TimeUnit::Nanosecond, None) => cudf::TypeId::TimestampNanoseconds,
        DataType::Duration(TimeUnit::Second) => cudf::TypeId::DurationSeconds,
        DataType::Duration(TimeUnit::Millisecond) => cudf::TypeId::DurationMilliseconds,
        DataType::Duration(TimeUnit::Microsecond) => cudf::TypeId::DurationMicroseconds,
        DataType::Duration(TimeUnit::Nanosecond) => cudf::TypeId::DurationNanoseconds,
        DataType::Decimal128(_, _) => cudf::TypeId::Decimal128,
        _ => return None,
    })
}

fn bounded_consumer_boolean_mask_bytes(rows: u64) -> Result<u64> {
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "predicate",
        DataType::Boolean,
        false,
    )]));
    bounded_consumer_frame_bytes(rows, &schema)
}

fn bounded_consumer_frame_bytes(rows: u64, schema: &SchemaRef) -> Result<u64> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| fixed_width_consumer_output_column(field.data_type()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| composed_schedule_error("consumer_variable_width_frame_unmodeled", None))?;
    let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns)
        .map_err(|_| composed_schedule_error("consumer_frame_upper_bound_unavailable", None))?;
    // The legacy join-size helper intentionally saturates because it also
    // serves best-effort diagnostics. Admission cannot use a saturated value
    // as a finite allocation contract, so fail closed instead.
    if bytes == u64::MAX {
        return Err(composed_schedule_error("consumer_frame_bytes_overflow", None));
    }
    Ok(bytes)
}

fn fixed_width_consumer_output_column(data_type: &DataType) -> Option<crate::exec::JoinOutputColumnUpperBoundInput> {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::UInt8
            | DataType::Int16
            | DataType::UInt16
            | DataType::Int32
            | DataType::UInt32
            | DataType::Int64
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, None)
            | DataType::Duration(_)
            | DataType::Decimal128(_, _)
    )
    .then(|| crate::exec::JoinOutputColumnUpperBoundInput {
        data_type: data_type.clone(),
        input_max_string_length_bytes: None,
        input_total_string_bytes: None,
    })
}

fn checked_consumer_add(left: u64, right: u64, reason: &'static str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| composed_schedule_error(reason, None))
}

fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

fn composed_schedule_error(schedule_reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "a composed graph attempt lacks a complete ordered memory schedule",
    )
    .with_fact("reason", "graph_attempt_complete_phase_unmodeled")
    .with_fact("schedule_reason", schedule_reason);
    if let Some(occurrence_id) = occurrence_id {
        error = error.with_fact("occurrence_id", occurrence_id);
    }
    error
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
        expr::Expr,
        graph_memory::{
            GraphMemoryCalibrationOverride, GraphVertexEstimate, graph_memory_admission_evidence_for_edge_list,
            graph_memory_admission_evidence_for_edge_list_with_vertex_estimate,
        },
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, ProjectionOutput,
            QueryPlan, QueryPlanBuilder,
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

    fn graph_bfs_proof_with_typed_output(source_vertex: i64) -> QueryPlanCapabilityProof {
        let plan = graph_bfs_plan_with_source_vertex(source_vertex);
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(source_vertex, None))),
        );
        let evidence = graph_memory_admission_evidence_for_edge_list_with_vertex_estimate(
            &spec,
            &schema,
            20,
            None,
            Some(GraphVertexEstimate::graph_metadata(10)),
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("test graph metadata must produce typed admission evidence");
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("test graph plan must issue a capability proof")
            .with_graph_memory_admission_evidence(&evidence)
    }

    fn graph_output_union_consumer(input_names: &[&str]) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let inputs = input_names
            .iter()
            .map(|name| builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(*name, Arc::clone(&schema)))))
            .collect();
        let root = builder.add_union(inputs, schema);
        builder.finish_with_host_sink(root)
    }

    fn graph_output_projection_consumer(expr: Expr, output_type: DataType) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("only", input_schema)));
        let projection = builder
            .add_projection(source, vec![ProjectionOutput::new(expr, Field::new("projected", output_type, false))]);
        builder.finish_with_host_sink(projection)
    }

    fn graph_output_inner_join_consumer(
        key_type: DataType,
        key_nullable: bool,
        include_unary_branch: bool,
    ) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("vertex", key_type.clone(), key_nullable),
            Field::new("value", DataType::Float64, false),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("left_vertex", key_type.clone(), key_nullable),
            Field::new("left_value", DataType::Float64, false),
            Field::new("right_value", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let left =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
        let projected_left = if include_unary_branch {
            let filtered_left = builder.add_filter(left, Expr::column("vertex"), Arc::clone(&input_schema));
            builder.add_projection(
                filtered_left,
                vec![
                    ProjectionOutput::new(Expr::column("vertex"), input_schema.field(0).as_ref().clone()),
                    ProjectionOutput::new(Expr::column("value"), input_schema.field(1).as_ref().clone()),
                ],
            )
        } else {
            left
        };
        let right =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
        let spec = JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                crate::expr::ColumnRef::input(0, "vertex", "vertex"),
                crate::expr::ColumnRef::input(1, "vertex", "vertex"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "left_vertex"),
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "value", "value"), "left_value"),
            JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "value", "value"), "right_value"),
        ]);
        let join = builder.add_join(projected_left, right, spec, output_schema);
        builder.finish_with_host_sink(join)
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

    #[test]
    fn composed_graph_schedule_seam_derives_complete_typed_union_evidence() {
        let consumer_plan = graph_output_union_consumer(&["first-occurrence", "second-occurrence"]);
        let first = graph_bfs_proof_with_typed_output(0);
        let second = graph_bfs_proof_with_typed_output(1);
        let consumer =
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer proof must be engine-issued");

        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [("first-occurrence".to_owned(), first), ("second-occurrence".to_owned(), second)],
        )
        .expect("typed producer and engine consumer evidence must derive one complete schedule");
        assert_eq!(combined.as_requirement().reservation_shapes.len(), 3);
        let envelope = combined
            .resolve_reservation_envelope(reservation_policy())
            .expect("complete typed graph and union evidence must resolve before placement");
        assert!(envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn composed_graph_schedule_prices_the_canonical_join_and_materialized_unary_branch() {
        let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, true);
        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
            ],
        )
        .expect("fixed-width non-null graph join must have complete engine-owned evidence");
        let envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(envelope.complete_phase_peak_bytes().is_some());
        assert!(envelope.required_phase_floor_bytes() > 0);
    }

    #[test]
    fn composed_graph_schedule_routes_n1_and_equal_n3_occurrences_through_one_path() {
        let n1_consumer = graph_output_union_consumer(&["only"]);
        let n1 = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &n1_consumer,
            QueryPlanCapabilityProof::for_query_plan(&n1_consumer).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                Some("only".to_owned()),
                Some(graph_bfs_proof_with_typed_output(0)),
            )],
        )
        .unwrap();
        assert!(n1.resolve_reservation_envelope(reservation_policy()).unwrap().complete_phase_peak_bytes().is_some());

        let n3_consumer = graph_output_union_consumer(&["same-0", "same-1", "same-2"]);
        let equal = graph_bfs_proof_with_typed_output(0);
        let n3 = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &n3_consumer,
            QueryPlanCapabilityProof::for_query_plan(&n3_consumer).unwrap(),
            [("same-0".to_owned(), equal.clone()), ("same-1".to_owned(), equal.clone()), ("same-2".to_owned(), equal)],
        )
        .unwrap();
        let envelope = n3.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(envelope.complete_phase_peak_bytes().is_some());
        assert_eq!(n3.as_requirement().reservation_shapes.len(), 4, "equal producer scopes remain three occurrences");
    }

    #[test]
    fn incomplete_single_graph_candidate_fails_closed_with_structured_schedule_facts() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let incomplete_graph_proof =
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_bfs_plan_with_source_vertex(0))
                .expect("the graph plan issues its scoped capability proof before evidence is hydrated");

        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                Some("only".to_owned()),
                Some(incomplete_graph_proof),
            )],
        )
        .expect_err("an incomplete N=1 candidate must not retain an unscheduled admission proof");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("missing_producer_graph_memory_evidence"));
        assert_eq!(error.fact_value("occurrence_id"), Some("only"));
    }

    #[test]
    fn missing_single_graph_proof_fails_closed_with_structured_schedule_facts() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(Some("only".to_owned()), Some("only".to_owned()), None)],
        )
        .expect_err("missing N=1 proof evidence must fail before admission");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
        assert_eq!(error.fact_value("occurrence_id"), Some("only"));
    }

    #[test]
    fn incomplete_multi_graph_candidate_remains_fail_closed() {
        let consumer_plan = graph_output_union_consumer(&["first", "second"]);
        let complete_graph_proof = graph_bfs_proof_with_typed_output(0);
        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ComposedGraphProducerCandidate::new(
                    Some("first".to_owned()),
                    Some("first".to_owned()),
                    Some(complete_graph_proof),
                ),
                ComposedGraphProducerCandidate::new(Some("second".to_owned()), Some("second".to_owned()), None),
            ],
        )
        .expect_err("two retained graph outputs require complete evidence");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
    }

    #[test]
    fn composed_consumer_alias_column_projection_reuses_its_input_allocation() {
        let consumer_plan = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Int64);
        let output = ComposedGraphProducerOutputEvidence {
            occurrence_id: "only".to_owned(),
            retained_output_bytes: 4_096,
            row_count_upper_bound: 128,
        };
        assert_eq!(
            composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &[output])
                .expect("an alias-only projection must have a bounded zero-copy contract"),
            0,
            "an alias-only projection is a zero-copy select/rename"
        );

        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [("only".to_owned(), graph_bfs_proof_with_typed_output(0))],
        )
        .expect("an exact-type alias projection must issue complete composed evidence");
        assert!(
            combined.resolve_reservation_envelope(reservation_policy()).unwrap().complete_phase_peak_bytes().is_some()
        );
    }

    #[test]
    fn composed_consumer_rejects_projected_column_type_changes_and_computation() {
        let output = [ComposedGraphProducerOutputEvidence {
            occurrence_id: "only".to_owned(),
            retained_output_bytes: 4_096,
            row_count_upper_bound: 128,
        }];
        let type_changed =
            graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Float64);
        let error = composed_consumer_additional_peak_bytes_for_plan(&type_changed, &output)
            .expect_err("a type-changing projection would allocate a new output frame");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_output_type_mismatch"));

        for expression in
            [Expr::cast(Expr::column("value"), DataType::Float64), Expr::literal(crate::expr::LiteralValue::Int64(1))]
        {
            let consumer_plan = graph_output_projection_consumer(expression, DataType::Int64);
            let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &output)
                .expect_err("casts and computed expressions lack a zero-copy allocation contract");
            assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_expression_workspace_unmodeled"));
        }
    }

    #[test]
    fn composed_graph_schedule_rejects_nullable_and_variable_width_join_keys() {
        for (data_type, nullable, expected_reason) in [
            (DataType::Int64, true, "consumer_join_nullable_key"),
            (DataType::Utf8, false, "consumer_join_variable_width_key"),
        ] {
            let consumer_plan = graph_output_inner_join_consumer(data_type, nullable, false);
            let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
                &consumer_plan,
                QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
                [
                    ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                    ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
                ],
            )
            .expect_err("unsupported key evidence must reject before placement");
            assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
            assert_eq!(error.fact_value("schedule_reason"), Some(expected_reason));
        }
    }

    #[test]
    fn composed_consumer_rejects_join_output_row_overflow() {
        let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, false);
        let producer_outputs = [
            ComposedGraphProducerOutputEvidence {
                occurrence_id: "left".to_owned(),
                retained_output_bytes: 1,
                row_count_upper_bound: u64::MAX,
            },
            ComposedGraphProducerOutputEvidence {
                occurrence_id: "right".to_owned(),
                retained_output_bytes: 1,
                row_count_upper_bound: 2,
            },
        ];
        let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &producer_outputs)
            .expect_err("overflowing inner-join output rows cannot become a finite admission contract");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_join_output_rows_overflow"));
    }

    #[test]
    fn composed_consumer_uses_the_linked_cudf_metadata_hash_preflight() {
        assert!(
            cudf::ops::join::hash_join_pre_build_reservation_size_bytes_from_non_nullable_fixed_width_metadata(
                &[cudf::TypeId::Int64],
                10,
                0.5,
            )
            .expect("the linked cuDF metadata preflight must accept a fixed-width non-null key")
                > 0
        );
    }
}
