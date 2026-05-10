//! Policy-resolved reservation envelopes for one admitted attempt.
//!
//! A capability proof is issued while lowering owns only the native plan. The
//! service owns source scheduling and chunk limits, so it resolves the plan
//! facts below immediately before ledger placement. This keeps a descriptive
//! source-size estimate out of immutable-grant sizing while still making
//! policy-created concurrent phases explicit.

use std::collections::HashSet;

use arrow::datatypes::DataType;

use crate::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    exec::{JoinOutputColumnUpperBoundInput, PartialAggregateShape, partial_aggregate_upper_bound_bytes},
    graph_memory::GraphMemoryAdmissionEvidence,
    observability::{
        AttemptComposedScheduleOccurrence, AttemptComposedScheduleProjection, COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1,
    },
    plan::{PlanNode, QueryPlan, estimated_reachable_local_parquet_source_bytes},
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy},
    source::SourceNode,
};

const GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON: &str = "graph_attempt_complete_phase_unmodeled";

/// One ordered graph-producer occurrence in a composed attempt.
///
/// Structural plan scopes are deliberately carried alongside an explicit
/// occurrence identifier.  Equal plans therefore remain distinct allocation
/// owners when their result handles are retained through the consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationProducerOccurrence {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    isolated_peak_bytes: Option<u64>,
    retained_output_bytes: Option<u64>,
}

impl AttemptReservationProducerOccurrence {
    /// Builds one occurrence from engine-owned evidence.  The byte fields are
    /// optional only while a proof is being assembled; resolution rejects any
    /// missing phase evidence before placement.
    pub(crate) fn new(
        occurrence_id: impl Into<String>,
        execution_scope_fingerprint: impl Into<String>,
        isolated_peak_bytes: Option<u64>,
        retained_output_bytes: Option<u64>,
    ) -> Self {
        Self {
            occurrence_id: occurrence_id.into(),
            execution_scope_fingerprint: execution_scope_fingerprint.into(),
            isolated_peak_bytes,
            retained_output_bytes,
        }
    }
}

/// Engine-owned ordered lifetime model for one composed graph attempt.
///
/// The schedule is intentionally opaque to adapter callers: its byte values
/// come from engine-issued capability evidence.  The fold models the executor
/// order exactly: fixed bytes plus retained prior outputs and one producer's
/// isolated peak, followed by retained-output accumulation, then the consumer
/// additional peak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationSchedule {
    attempt_wide_fixed_bytes: Option<u64>,
    producer_occurrences: Vec<AttemptReservationProducerOccurrence>,
    consumer_scope_fingerprint: Option<String>,
    consumer_additional_peak_bytes: Option<u64>,
}

impl AttemptReservationSchedule {
    pub(crate) fn new(
        attempt_wide_fixed_bytes: Option<u64>,
        producer_occurrences: Vec<AttemptReservationProducerOccurrence>,
        consumer_scope_fingerprint: Option<String>,
        consumer_additional_peak_bytes: Option<u64>,
    ) -> Self {
        Self {
            attempt_wide_fixed_bytes,
            producer_occurrences,
            consumer_scope_fingerprint,
            consumer_additional_peak_bytes,
        }
    }

    #[cfg(test)]
    pub(crate) fn producer_occurrences(&self) -> &[AttemptReservationProducerOccurrence] {
        &self.producer_occurrences
    }

    fn scope_counts(&self) -> std::collections::HashMap<&str, usize> {
        let mut counts = std::collections::HashMap::with_capacity(self.producer_occurrences.len() + 1);
        for scope in self
            .producer_occurrences
            .iter()
            .map(|occurrence| occurrence.execution_scope_fingerprint.as_str())
            .chain(self.consumer_scope_fingerprint.as_deref())
        {
            *counts.entry(scope).or_default() += 1;
        }
        counts
    }

    #[cfg(test)]
    pub(crate) fn checked_fold(&self) -> Result<u64> {
        self.checked_fold_with_scopes(&[])
    }

    fn checked_fold_with_scopes(&self, scopes: &[AttemptReservationShape]) -> Result<u64> {
        let fixed = self
            .attempt_wide_fixed_bytes
            .ok_or_else(|| graph_schedule_error("missing_attempt_wide_fixed_bytes", None, None))?;
        if self.producer_occurrences.is_empty() {
            return Err(graph_schedule_error("missing_producer_evidence", None, None));
        }
        if self.consumer_scope_fingerprint.is_none() {
            return Err(graph_schedule_error("missing_consumer_scope", None, None));
        }
        let consumer_peak = self
            .consumer_additional_peak_bytes
            .ok_or_else(|| graph_schedule_error("missing_consumer_additional_peak", None, None))?;
        self.validate_occurrence_ids_and_scopes(scopes)?;

        let mut retained_prior = 0_u64;
        let mut complete_peak = 0_u64;
        for occurrence in &self.producer_occurrences {
            let isolated_peak = occurrence.isolated_peak_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_isolated_peak", Some(&occurrence.occurrence_id), None)
            })?;
            let retained_output = occurrence.retained_output_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_retained_output", Some(&occurrence.occurrence_id), None)
            })?;
            let producer_stage = fixed
                .checked_add(retained_prior)
                .ok_or_else(|| {
                    graph_schedule_error(
                        "producer_stage_fixed_plus_retained_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("producer_stage_fixed_plus_retained"),
                    )
                })?
                .checked_add(isolated_peak)
                .ok_or_else(|| {
                    graph_schedule_error(
                        "producer_stage_isolated_peak_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("producer_stage_isolated_peak"),
                    )
                })?;
            complete_peak = complete_peak.max(producer_stage);
            retained_prior = retained_prior.checked_add(retained_output).ok_or_else(|| {
                graph_schedule_error(
                    "retained_output_accumulation_overflow",
                    Some(&occurrence.occurrence_id),
                    Some("retained_output_accumulation"),
                )
            })?;
        }

        let consumer_stage = fixed
            .checked_add(retained_prior)
            .ok_or_else(|| {
                graph_schedule_error(
                    "consumer_stage_fixed_plus_retained_overflow",
                    None,
                    Some("consumer_stage_fixed_plus_retained"),
                )
            })?
            .checked_add(consumer_peak)
            .ok_or_else(|| {
                graph_schedule_error(
                    "consumer_stage_additional_peak_overflow",
                    None,
                    Some("consumer_stage_additional_peak"),
                )
            })?;
        Ok(complete_peak.max(consumer_stage))
    }

    fn projection(
        &self,
        scopes: &[AttemptReservationShape],
        attempt_id: crate::observability::AttemptId,
        device_ordinal: usize,
        grant_id: u64,
        grant_bytes: u64,
    ) -> Result<AttemptComposedScheduleProjection> {
        // Run the canonical fold first. This validates every required owner
        // contribution and exact scope coverage before any public projection
        // is published.
        let complete_peak = self.checked_fold_with_scopes(scopes)?;
        let fixed = self
            .attempt_wide_fixed_bytes
            .ok_or_else(|| graph_schedule_error("missing_attempt_wide_fixed_bytes", None, None))?;
        let consumer_scope = self
            .consumer_scope_fingerprint
            .clone()
            .ok_or_else(|| graph_schedule_error("missing_consumer_scope", None, None))?;
        let consumer_additional_peak_bytes = self
            .consumer_additional_peak_bytes
            .ok_or_else(|| graph_schedule_error("missing_consumer_additional_peak", None, None))?;

        let mut retained_prior = 0_u64;
        let mut producer_occurrences = Vec::with_capacity(self.producer_occurrences.len());
        for occurrence in &self.producer_occurrences {
            let isolated_peak_bytes = occurrence.isolated_peak_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_isolated_peak", Some(&occurrence.occurrence_id), None)
            })?;
            let retained_output_bytes = occurrence.retained_output_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_retained_output", Some(&occurrence.occurrence_id), None)
            })?;
            let stage_peak_bytes = fixed
                .checked_add(retained_prior)
                .and_then(|value| value.checked_add(isolated_peak_bytes))
                .ok_or_else(|| {
                    graph_schedule_error(
                        "producer_stage_projection_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("producer_stage_peak"),
                    )
                })?;
            producer_occurrences.push(AttemptComposedScheduleOccurrence {
                occurrence_id: occurrence.occurrence_id.clone(),
                execution_scope_fingerprint: occurrence.execution_scope_fingerprint.clone(),
                isolated_peak_bytes,
                retained_output_bytes,
                stage_peak_bytes,
            });
            retained_prior = retained_prior.checked_add(retained_output_bytes).ok_or_else(|| {
                graph_schedule_error(
                    "retained_output_accumulation_overflow",
                    Some(&occurrence.occurrence_id),
                    Some("retained_output_accumulation"),
                )
            })?;
        }
        let consumer_stage_peak_bytes = fixed
            .checked_add(retained_prior)
            .and_then(|value| value.checked_add(consumer_additional_peak_bytes))
            .ok_or_else(|| {
                graph_schedule_error("consumer_stage_projection_overflow", None, Some("consumer_stage_peak"))
            })?;
        if grant_bytes < complete_peak {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("projection"))
                .with_fact("immutable_grant_bytes", grant_bytes)
                .with_fact("complete_phase_peak_bytes", complete_peak));
        }

        let fingerprint = AttemptComposedScheduleProjection::fingerprint_for(
            fixed,
            &producer_occurrences,
            &consumer_scope,
            consumer_additional_peak_bytes,
            consumer_stage_peak_bytes,
            complete_peak,
        );
        Ok(AttemptComposedScheduleProjection {
            schema: COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1.to_owned(),
            attempt_id,
            device_ordinal,
            grant_id,
            grant_bytes,
            attempt_wide_fixed_bytes: fixed,
            producer_occurrences,
            consumer_scope_fingerprint: consumer_scope,
            consumer_additional_peak_bytes,
            consumer_stage_peak_bytes,
            complete_peak_bytes: complete_peak,
            fingerprint,
        })
    }

    fn validate_occurrence_ids_and_scopes(&self, scopes: &[AttemptReservationShape]) -> Result<()> {
        let mut occurrence_ids = HashSet::with_capacity(self.producer_occurrences.len());
        for occurrence in &self.producer_occurrences {
            if occurrence.occurrence_id.is_empty() {
                return Err(graph_schedule_error("missing_producer_occurrence_id", None, None));
            }
            if !occurrence_ids.insert(&occurrence.occurrence_id) {
                return Err(graph_schedule_error(
                    "duplicate_producer_occurrence_id",
                    Some(&occurrence.occurrence_id),
                    None,
                ));
            }
            if occurrence.execution_scope_fingerprint.is_empty() {
                return Err(graph_schedule_error(
                    "missing_producer_occurrence_scope",
                    Some(&occurrence.occurrence_id),
                    None,
                ));
            }
        }

        let Some(consumer_scope) = self.consumer_scope_fingerprint.as_deref() else {
            return Err(graph_schedule_error("missing_consumer_scope", None, None));
        };
        if consumer_scope.is_empty() {
            return Err(graph_schedule_error("missing_consumer_scope", None, None));
        }
        if scopes.is_empty() {
            return Ok(());
        }

        let mut expected_scope_counts = std::collections::HashMap::<&str, usize>::new();
        for scope in self
            .producer_occurrences
            .iter()
            .map(|occurrence| occurrence.execution_scope_fingerprint.as_str())
            .chain(std::iter::once(consumer_scope))
        {
            *expected_scope_counts.entry(scope).or_default() += 1;
        }
        let mut actual_scope_counts = std::collections::HashMap::<&str, usize>::new();
        for scope in scopes.iter().map(|scope| scope.execution_scope_fingerprint.as_str()) {
            *actual_scope_counts.entry(scope).or_default() += 1;
        }
        if expected_scope_counts
            .iter()
            .any(|(scope, expected)| actual_scope_counts.get(scope).copied().unwrap_or_default() < *expected)
        {
            return Err(graph_schedule_error("schedule_scope_coverage_mismatch", None, Some("scope_coverage")));
        }
        Ok(())
    }

    fn validate_runtime(&self, consumer_plan: &QueryPlan, occurrence_ids: &[String]) -> Result<()> {
        let expected_consumer_scope = self
            .consumer_scope_fingerprint
            .as_deref()
            .ok_or_else(|| graph_schedule_error("missing_consumer_scope", None, Some("runtime")))?;
        if expected_consumer_scope != canonical_execution_scope_fingerprint(consumer_plan) {
            return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
        }
        if self.producer_occurrences.len() != occurrence_ids.len() {
            return Err(graph_schedule_error("runtime_occurrence_count_mismatch", None, Some("runtime"))
                .with_fact("scheduled_occurrence_count", self.producer_occurrences.len())
                .with_fact("runtime_occurrence_count", occurrence_ids.len()));
        }
        for (scheduled, runtime) in self.producer_occurrences.iter().zip(occurrence_ids) {
            if scheduled.occurrence_id != *runtime {
                return Err(graph_schedule_error("runtime_occurrence_order_mismatch", Some(runtime), Some("runtime"))
                    .with_fact("scheduled_occurrence_id", scheduled.occurrence_id.as_str()));
            }
        }
        self.validate_occurrence_ids_and_scopes(&[])
    }
}

/// Service-owned policy inputs that affect the bounded attempt's physical
/// liveness. `device_budget_bytes` is deliberately absent: a grant is the
/// result of admission, not an input to its proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttemptReservationPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_chunk_planning_options: SourceChunkPlanningOptions,
}

impl AttemptReservationPolicy {
    pub(crate) const fn new(
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Self {
        Self {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        }
    }

    fn matches_execution(
        self,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> bool {
        let MemoryControlPolicy::Bounded {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            ..
        } = memory_policy
        else {
            return false;
        };
        self.max_source_chunk_bytes == max_source_chunk_bytes
            && self.max_row_groups_per_chunk == max_row_groups_per_chunk
            && self.max_retry_attempts == max_retry_attempts
            && self.min_retry_chunk_row_groups == min_retry_chunk_row_groups
            && self.source_scheduler_policy == source_scheduler_policy
            && self.source_chunk_planning_mode == source_chunk_planning_mode
            && self.source_chunk_planning_options == source_chunk_planning_options
    }

    fn source_read_phase_slots(self, source_chunk_count: u64) -> u64 {
        // `OrderedSourceReadQueue` retains at most the current decoded source
        // plus one optional prefetch. Its public scheduler bound can be larger
        // for future scheduling work, but it cannot create a third concurrent
        // source read in the current bounded executor.
        if source_chunk_count > 1 && self.source_scheduler_policy.enables_prefetch() { 2 } else { 1 }
    }
}

/// Immutable plan facts carried privately by a capability proof until the
/// query service can resolve them against its installed policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationShape {
    execution_scope_fingerprint: String,
    has_local_parquet_source: bool,
    source_stage_bytes: Option<u64>,
    complete_zero_column_global_aggregate: Option<ZeroColumnGlobalAggregateShape>,
    complete_graph_memory_phase: Option<GraphMemoryAdmissionEvidence>,
    complete_cugraph_side_input_phase_bytes: Option<u64>,
    accounted_by_graph_phase: bool,
    is_cugraph_graph_input_scope: bool,
    is_cugraph_side_input_scope: bool,
    is_attested_graph_result_passthrough_scope: bool,
    is_attested_graph_result_bounded_consumer_scope: bool,
    sort_phase_peak_bytes: Option<u64>,
    sort_metadata: Option<crate::admission::requirement::SortMetadataDescriptor>,
    sort_rejection_reason: Option<String>,
    owner_term_rejection_reason: Option<String>,
    ordered_schedule: Option<AttemptReservationSchedule>,
}

impl AttemptReservationShape {
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            has_local_parquet_source: plan_has_reachable_local_parquet_source(plan),
            source_stage_bytes: estimated_reachable_local_parquet_source_bytes(plan),
            complete_zero_column_global_aggregate: zero_column_global_aggregate_shape(plan),
            complete_graph_memory_phase: None,
            complete_cugraph_side_input_phase_bytes: None,
            accounted_by_graph_phase: false,
            is_cugraph_graph_input_scope: false,
            is_cugraph_side_input_scope: false,
            is_attested_graph_result_passthrough_scope: false,
            is_attested_graph_result_bounded_consumer_scope: false,
            sort_phase_peak_bytes: None,
            sort_metadata: None,
            sort_rejection_reason: None,
            owner_term_rejection_reason: None,
            ordered_schedule: None,
        }
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(crate) fn attach_ordered_schedule(&mut self, schedule: AttemptReservationSchedule) -> Result<()> {
        if self.ordered_schedule.is_some() {
            return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule")));
        }
        self.ordered_schedule = Some(schedule);
        Ok(())
    }

    fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    pub(crate) fn matches_execution_scope_fingerprint(&self, execution_scope_fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == execution_scope_fingerprint
    }

    pub(crate) fn set_graph_memory_phase(
        &mut self,
        evidence: &GraphMemoryAdmissionEvidence,
        source_stage_bytes: Option<u64>,
    ) {
        self.complete_graph_memory_phase = Some(evidence.clone());
        // A composed graph proof may obtain its prepared edge frame from a
        // separately lowered native source. The aggregate proof owns that
        // exact stage byte count, so replace this graph scope's local view
        // instead of adding it a second time.
        if let Some(source_stage_bytes) = source_stage_bytes {
            self.source_stage_bytes = Some(source_stage_bytes);
        }
    }

    pub(crate) fn account_for_by_graph_phase(&mut self) {
        self.accounted_by_graph_phase = true;
    }

    #[cfg(test)]
    pub(crate) fn is_accounted_by_graph_phase(&self) -> bool {
        self.accounted_by_graph_phase
    }

    pub(crate) fn mark_cugraph_graph_input_scope(&mut self) {
        self.is_cugraph_graph_input_scope = true;
    }

    pub(crate) fn set_cugraph_side_input_phase_bytes(&mut self, frame_bytes: u64) -> Result<()> {
        if self.complete_cugraph_side_input_phase_bytes.replace(frame_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_side_input_frame_evidence", None, Some("side_input_frame")));
        }
        self.is_cugraph_side_input_scope = true;
        Ok(())
    }

    pub(crate) fn mark_attested_graph_result_passthrough_scope(&mut self) {
        self.is_attested_graph_result_passthrough_scope = true;
    }

    pub(crate) fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
        self.is_attested_graph_result_bounded_consumer_scope = true;
    }

    pub(crate) fn set_sort_phase_peak_bytes(&mut self, peak_bytes: u64) -> Result<()> {
        if self.sort_phase_peak_bytes.replace(peak_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_sort_phase_evidence", None, Some("sort")));
        }
        Ok(())
    }

    pub(crate) fn set_sort_metadata(
        &mut self,
        metadata: crate::admission::requirement::SortMetadataDescriptor,
    ) -> Result<()> {
        if self.sort_metadata.replace(metadata).is_some() {
            return Err(graph_schedule_error("duplicate_sort_metadata", None, Some("sort")));
        }
        Ok(())
    }

    pub(crate) fn set_sort_rejection_reason(&mut self, reason: impl Into<String>) {
        self.sort_rejection_reason = Some(reason.into());
    }

    pub(crate) fn set_owner_term_rejection_reason(&mut self, reason: impl Into<String>) {
        self.owner_term_rejection_reason = Some(reason.into());
    }

    pub(crate) fn sort_metadata(&self) -> Option<&crate::admission::requirement::SortMetadataDescriptor> {
        self.sort_metadata.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn set_sort_stable_for_test(&mut self, stable: bool) {
        if let Some(metadata) = self.sort_metadata.as_mut() {
            metadata.stable = stable;
        }
    }

    pub(crate) fn has_modeled_source_stage(&self) -> bool {
        self.source_stage_bytes.is_some()
    }

    pub(crate) fn has_local_parquet_source(&self) -> bool {
        self.has_local_parquet_source
    }

    pub(crate) fn is_cugraph_graph_input_scope(&self) -> bool {
        self.is_cugraph_graph_input_scope
    }

    pub(crate) fn is_cugraph_side_input_scope(&self) -> bool {
        self.is_cugraph_side_input_scope
    }

    pub(crate) fn is_attested_graph_result_passthrough_scope(&self) -> bool {
        self.is_attested_graph_result_passthrough_scope
    }

    pub(crate) fn has_graph_memory_phase(&self) -> bool {
        self.complete_graph_memory_phase.is_some()
    }

    pub(crate) fn graph_memory_admission_evidence(&self) -> Option<&GraphMemoryAdmissionEvidence> {
        self.complete_graph_memory_phase.as_ref()
    }

    fn resolve(&self, policy: AttemptReservationPolicy) -> Result<ResolvedReservationShape> {
        if let Some(reason) = &self.owner_term_rejection_reason {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "attempt owner-term inventory is unavailable",
            )
            .with_fact("reason", reason.as_str()));
        }
        if self.accounted_by_graph_phase {
            return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: Some(0) });
        }
        if let Some(graph_phase) = &self.complete_graph_memory_phase {
            let graph_peak = match self.source_stage_bytes {
                Some(source_stage_bytes) => graph_phase.cold_model_with_source_stage_bytes(source_stage_bytes)?,
                None => graph_phase.cold_model_requested_bytes(),
            };
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: graph_peak,
                complete_phase_peak_bytes: Some(graph_peak),
            });
        }

        if let Some(frame_bytes) = self.complete_cugraph_side_input_phase_bytes {
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: frame_bytes,
                complete_phase_peak_bytes: Some(frame_bytes),
            });
        }

        if self.is_attested_graph_result_bounded_consumer_scope {
            if let Some(reason) = &self.sort_rejection_reason {
                return Err(graph_schedule_error("sort_metadata_rejected", None, Some("sort"))
                    .with_fact("sort_rejection_reason", reason.as_str()));
            }
            let peak = self
                .sort_phase_peak_bytes
                .ok_or_else(|| graph_schedule_error("sort_preflight_unresolved", None, Some("sort")))?;
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: peak,
                complete_phase_peak_bytes: Some(peak),
            });
        }

        let minimum_source_read_bytes =
            self.has_local_parquet_source.then(minimum_parquet_source_read_budget_bytes).unwrap_or_default();
        let Some(shape) = &self.complete_zero_column_global_aggregate else {
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: minimum_source_read_bytes,
                complete_phase_peak_bytes: None,
            });
        };

        let source_chunk_count = shape.source_chunk_count(policy.max_row_groups_per_chunk)?;
        let source_phase_bytes = minimum_source_read_bytes
            .checked_mul(policy.source_read_phase_slots(source_chunk_count))
            .ok_or_else(|| envelope_overflow_error("source_read_phase"))?;
        let retained_partial_bytes = shape
            .partial_state_upper_bound_bytes
            .checked_mul(source_chunk_count)
            .ok_or_else(|| envelope_overflow_error("retained_partial_state"))?;
        // The scalar aggregate result and its immediate projection can overlap
        // with the retained partial list during finalization. They are bounded
        // independently by the same closed-form device-output contract.
        let final_scalar_phase_bytes =
            shape.maximum_scalar_frame_bytes.checked_mul(2).ok_or_else(|| envelope_overflow_error("final_scalar"))?;
        let peak = source_phase_bytes
            .checked_add(retained_partial_bytes)
            .and_then(|value| value.checked_add(final_scalar_phase_bytes))
            .ok_or_else(|| envelope_overflow_error("zero_column_global_aggregate"))?;
        Ok(ResolvedReservationShape { required_phase_floor_bytes: peak, complete_phase_peak_bytes: Some(peak) })
    }
}

/// Resolved service-admission envelope retained by the admitted context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationEnvelope {
    policy: AttemptReservationPolicy,
    scopes: Vec<AttemptReservationShape>,
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
    ordered_schedule: Option<AttemptReservationSchedule>,
}

impl AttemptReservationEnvelope {
    pub(crate) fn resolve(
        reservation_shapes: &[AttemptReservationShape],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        // Reservation shapes are allocation occurrences, not structural set
        // members.  Equal graph plans may retain two distinct result handles
        // and must remain two entries in the resolved envelope.
        let scopes = reservation_shapes.to_vec();
        let ordered_schedule =
            match scopes.iter().filter_map(AttemptReservationShape::ordered_schedule).collect::<Vec<_>>().as_slice() {
                [] => None,
                [schedule] => Some((*schedule).clone()),
                _ => return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule"))),
            };

        let mut required_phase_floor_bytes = 0_u64;
        let mut complete_phase_peak_bytes = Some(0_u64);
        let has_graph_memory_phase = scopes.iter().any(AttemptReservationShape::has_graph_memory_phase);
        let mut scheduled_scope_counts = ordered_schedule.as_ref().map(AttemptReservationSchedule::scope_counts);
        for scope in &scopes {
            let resolved = scope.resolve(policy)?;
            let is_scheduled_scope = scheduled_scope_counts.as_mut().is_some_and(|counts| {
                let Some(remaining) = counts.get_mut(scope.execution_scope_fingerprint()) else {
                    return false;
                };
                if *remaining == 0 {
                    return false;
                }
                *remaining -= 1;
                true
            });
            if is_scheduled_scope {
                // The ordered fold is the only complete peak for its covered
                // producer/consumer occurrences.  Adding their isolated
                // peaks here would reintroduce the obsolete blind sum.
                continue;
            }
            required_phase_floor_bytes =
                required_phase_floor_bytes.checked_add(resolved.required_phase_floor_bytes).ok_or_else(|| {
                    ordered_schedule.as_ref().map_or_else(
                        || envelope_overflow_error("combined_required_phase"),
                        |_| {
                            graph_schedule_error("combined_required_phase_overflow", None, Some("required_phase_floor"))
                        },
                    )
                })?;
            complete_phase_peak_bytes = match (complete_phase_peak_bytes, resolved.complete_phase_peak_bytes) {
                (Some(current), Some(peak)) => match current.checked_add(peak) {
                    Some(value) => Some(value),
                    None if ordered_schedule.is_some() => {
                        return Err(graph_schedule_error(
                            "auxiliary_complete_phase_overflow",
                            None,
                            Some("auxiliary_complete_phase"),
                        ));
                    },
                    None => None,
                },
                _ => None,
            };
        }

        if let Some(schedule) = &ordered_schedule {
            if complete_phase_peak_bytes.is_none() {
                return Err(graph_schedule_error("auxiliary_scope_unmodeled", None, Some("auxiliary_scope")));
            }
            let complete_peak = schedule.checked_fold_with_scopes(&scopes)?;
            required_phase_floor_bytes = required_phase_floor_bytes.max(complete_peak);
            complete_phase_peak_bytes =
                complete_phase_peak_bytes.map(|auxiliary_peak| auxiliary_peak.max(complete_peak));
        }

        if ordered_schedule.is_none() && has_graph_memory_phase && complete_phase_peak_bytes.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "a graph attempt contains a runtime scope without a complete bounded phase target",
            )
            .with_fact("reason", GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        }

        Ok(Self { policy, scopes, required_phase_floor_bytes, complete_phase_peak_bytes, ordered_schedule })
    }

    pub(crate) fn required_phase_floor_bytes(&self) -> u64 {
        self.required_phase_floor_bytes
    }

    pub(crate) fn complete_phase_peak_bytes(&self) -> Option<u64> {
        self.complete_phase_peak_bytes
    }

    #[cfg(test)]
    pub(crate) fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    pub(crate) fn validate_composed_execution(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
        immutable_grant_bytes: u64,
    ) -> Result<()> {
        let Some(schedule) = &self.ordered_schedule else {
            return Err(graph_schedule_error("missing_admitted_schedule", None, Some("runtime")));
        };
        schedule.validate_runtime(consumer_plan, occurrence_ids)?;
        let required_peak = schedule.checked_fold_with_scopes(&self.scopes)?;
        if immutable_grant_bytes < required_peak {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                .with_fact("complete_phase_peak_bytes", required_peak));
        }
        Ok(())
    }

    /// Publishes the immutable schedule priced by this envelope, bound to the
    /// grant and device selected by the admission transaction.
    pub(crate) fn composed_schedule_projection(
        &self,
        attempt_id: crate::observability::AttemptId,
        device_ordinal: usize,
        grant_id: u64,
        grant_bytes: u64,
    ) -> Result<AttemptComposedScheduleProjection> {
        let Some(schedule) = &self.ordered_schedule else {
            return Err(graph_schedule_error("missing_admitted_schedule", None, Some("projection")));
        };
        schedule.projection(&self.scopes, attempt_id, device_ordinal, grant_id, grant_bytes)
    }

    pub(crate) fn validate_execution(
        &self,
        plan: &QueryPlan,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Result<()> {
        if !self.policy.matches_execution(
            memory_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        ) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "admitted reservation envelope does not match the execution policy",
            )
            .with_fact("reason", "admitted_reservation_policy_mismatch"));
        }
        if self.scopes.iter().any(|scope| scope.matches(plan)) {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.reservation_envelope",
            "admitted reservation envelope does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_reservation_envelope_mismatch"))
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedReservationShape {
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
}

/// Exact plan facts for the specialized zero-column aggregate executor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ZeroColumnGlobalAggregateShape {
    selected_row_groups_per_file: Vec<usize>,
    partial_state_upper_bound_bytes: u64,
    maximum_scalar_frame_bytes: u64,
}

impl ZeroColumnGlobalAggregateShape {
    fn source_chunk_count(&self, max_row_groups_per_chunk: usize) -> Result<u64> {
        let max_row_groups_per_chunk = u64::try_from(max_row_groups_per_chunk.max(1))
            .map_err(|_| envelope_overflow_error("max_row_groups_per_chunk"))?;
        let mut chunks = 0_u64;
        for row_groups in &self.selected_row_groups_per_file {
            let row_groups = u64::try_from(*row_groups).map_err(|_| envelope_overflow_error("selected_row_groups"))?;
            chunks = chunks
                .checked_add(row_groups.div_ceil(max_row_groups_per_chunk))
                .ok_or_else(|| envelope_overflow_error("source_chunk_count"))?;
        }
        Ok(chunks.max(1))
    }
}

fn zero_column_global_aggregate_shape(plan: &QueryPlan) -> Option<ZeroColumnGlobalAggregateShape> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut row_groups_per_file = Vec::new();
    let mut partial_state_upper_bound_bytes = 0_u64;
    let mut maximum_scalar_frame_bytes = 0_u64;
    let mut has_aggregate = false;

    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        maximum_scalar_frame_bytes =
            maximum_scalar_frame_bytes.max(scalar_frame_upper_bound_bytes(node.output_schema())?);
        match &node.kind {
            PlanNode::Source(source) => {
                let SourceNode::LocalParquet(source) = source.as_ref() else {
                    return None;
                };
                if !source.is_zero_column_count_source() || source.files.is_empty() {
                    return None;
                }
                for file in &source.files {
                    row_groups_per_file.push(file.row_groups.as_ref()?.len());
                }
            },
            PlanNode::Aggregate(aggregate) => {
                if !aggregate.spec.group_exprs.is_empty() {
                    return None;
                }
                partial_state_upper_bound_bytes = partial_state_upper_bound_bytes.max(
                    global_partial_state_upper_bound_bytes(node.output_schema(), aggregate.spec.aggr_exprs.len())?,
                );
                has_aggregate = true;
            },
            // This is the exact scalar path selected for COUNT(*) and related
            // zero-column global aggregates. Any widening, join, sort, graph,
            // or source-adjacent operator leaves the complete peak unknown so
            // admission deliberately uses the candidate grant ceiling.
            PlanNode::Projection(_) | PlanNode::Limit(_) => {},
            _ => return None,
        }
    }

    (has_aggregate && !row_groups_per_file.is_empty()).then_some(ZeroColumnGlobalAggregateShape {
        selected_row_groups_per_file: row_groups_per_file,
        partial_state_upper_bound_bytes,
        maximum_scalar_frame_bytes,
    })
}

fn global_partial_state_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef, aggregate_count: usize) -> Option<u64> {
    let mut columns = fixed_width_columns(schema)?;
    // SUM/MIN/MAX partials retain an extra validity count and several aggregate
    // families retain more than their final projection. One fixed-width state
    // per aggregate is a conservative owner-side floor for the scalar
    // zero-column path; unsupported variable-width state declines the model.
    columns.extend((0..aggregate_count).map(|_| fixed_width_column(DataType::UInt64)));
    partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &columns).ok()
}

fn scalar_frame_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef) -> Option<u64> {
    crate::exec::join_output_upper_bound_bytes(1, &fixed_width_columns(schema)?).ok()
}

fn fixed_width_columns(schema: &arrow::datatypes::SchemaRef) -> Option<Vec<JoinOutputColumnUpperBoundInput>> {
    schema.fields().iter().map(|field| fixed_width_column_checked(field.data_type())).collect()
}

fn fixed_width_column_checked(data_type: &DataType) -> Option<JoinOutputColumnUpperBoundInput> {
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
    .then(|| fixed_width_column(data_type.clone()))
}

fn fixed_width_column(data_type: DataType) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput { data_type, input_max_string_length_bytes: None, input_total_string_bytes: None }
}

fn plan_has_reachable_local_parquet_source(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    plan.nodes().iter().any(|node| {
        !unreachable.contains(&node.id)
            && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(source) if !source.files.is_empty()))
    })
}

fn minimum_parquet_source_read_budget_bytes() -> u64 {
    u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap_or(u64::MAX)
}

/// The DataFusion adapter always executes host-facing native plans through a
/// GPU-resident physical sink, retaining the host contract only for the final
/// cursor. Admission must therefore scope the physical plan rather than the
/// wrapper-only host boundary. GPU-resident plans remain unchanged.
fn canonical_execution_scope(plan: &QueryPlan) -> QueryPlan {
    plan.clone_with_deferred_host_materialization_sink()
}

/// Collision-free structural identity for the exact plan shape that enters the
/// engine. Formatting follows the full derived IR value rather than a hash;
/// the scope is short-lived private admission evidence, never an external key.
pub(crate) fn canonical_execution_scope_fingerprint(plan: &QueryPlan) -> String {
    format!("{:#?}", canonical_execution_scope(plan))
}

fn envelope_overflow_error(phase: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.admission.reservation_envelope",
        "attempt reservation envelope overflows u64",
    )
    .with_fact("reason", "attempt_reservation_envelope_overflow")
    .with_fact("phase", phase)
}

fn graph_schedule_error(reason: &'static str, occurrence_id: Option<&str>, phase: Option<&'static str>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "a composed graph attempt lacks a complete ordered memory schedule",
    )
    .with_fact("reason", GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON)
    .with_fact("schedule_reason", reason);
    if let Some(occurrence_id) = occurrence_id {
        error = error.with_fact("occurrence_id", occurrence_id);
    }
    if let Some(phase) = phase {
        error = error.with_fact("phase", phase);
    }
    error
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{
        admission::QueryPlanCapabilityProof,
        chunk::SourceChunkPlanningOptions,
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, SortExpr, SortSpec},
        graph_memory::{
            GraphMemoryAdmissionEvidence, GraphMemoryCalibrationOverride,
            graph_memory_bounded_admission_evidence_for_plan,
        },
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, QueryPlanBuilder,
        },
        source::{
            GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, SourceNode,
            ZERO_COLUMN_COUNT_COLUMN,
        },
    };

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

    fn zero_column_count_plan(row_group_count: usize) -> QueryPlan {
        let row_groups = (0..row_group_count).collect::<Vec<_>>();
        let source = LocalParquetSource::zero_column_count(vec![
            LocalParquetFile::new("/tmp/admission-envelope-count.parquet")
                .with_row_groups(row_groups)
                .with_row_group_bytes(vec![4_096; row_group_count])
                .with_projected_row_group_bytes(vec![0; row_group_count], vec![0; row_group_count]),
        ]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count")],
            ),
            Arc::new(Schema::new(vec![Field::new("count", DataType::UInt64, false)])),
        );
        builder.finish_with_host_sink(aggregate)
    }

    fn unmodeled_native_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "unmodeled",
            Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)])),
        )));
        builder.finish_with_host_sink(source)
    }

    fn graph_edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]))
    }

    fn bfs_spec() -> GraphAlgorithmSpec {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        )
    }

    fn local_graph_source(schema: Arc<Schema>) -> LocalParquetSource {
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/admission-envelope-graph.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![10])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            schema,
            vec!["src".to_owned(), "dst".to_owned()],
        )
    }

    fn graph_bfs_plan_with_local_source(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
        let graph = builder.add_graph_algorithm(source, bfs_spec());
        builder.finish_with_host_sink(graph)
    }

    fn graph_bfs_plan_with_gpu_resident_source(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("prepared-edges", schema).with_row_count_upper_bound(10),
        ));
        let graph = builder.add_graph_algorithm(source, bfs_spec());
        builder.finish_with_host_sink(graph)
    }

    fn graph_memory_evidence(plan: &QueryPlan) -> GraphMemoryAdmissionEvidence {
        graph_memory_bounded_admission_evidence_for_plan(plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("graph plan must produce bounded memory evidence")
    }

    fn source_stage_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
        builder.finish_gpu_resident(source)
    }

    fn unknown_source_stage_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/admission-envelope-graph-unknown.parquet",
            schema,
            vec!["src".to_owned(), "dst".to_owned()],
        )));
        builder.finish_gpu_resident(source)
    }

    fn capability_free_literal_plan(data_type: DataType) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        builder.finish_gpu_resident(source)
    }

    fn one_row_literal_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![1]))])
            .expect("one-row literal must be typed");
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        builder.finish_gpu_resident(source)
    }

    fn placeholder_literal_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::placeholder_row()));
        builder.finish_gpu_resident(source)
    }

    fn nullable_literal_side_input_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![Some(1), None]))])
            .expect("nullable literal side input must be typed");
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        builder.finish_gpu_resident(source)
    }

    fn unknown_row_bound_side_input_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("predicate", schema)));
        builder.finish_gpu_resident(source)
    }

    fn overflow_row_bound_side_input_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("predicate-overflow", schema).with_row_count_upper_bound(u64::MAX),
        ));
        builder.finish_gpu_resident(source)
    }

    fn graph_result_passthrough_plan(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
        builder.finish_with_host_sink(source)
    }

    fn graph_result_sort_consumer_plan(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new(input_name, Arc::clone(&schema)).with_row_count_upper_bound(20),
        ));
        let sort =
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]), schema);
        builder.finish_with_host_sink(sort)
    }

    #[test]
    fn capability_free_native_fragment_keeps_an_exact_reservation_scope() {
        let plan = capability_free_literal_plan(DataType::Int64);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan)
            .expect("an executed native fragment must issue an exact-scope proof");

        assert!(proof.required_capabilities().is_empty(), "the scope-only proof must not add a capability allowance");
        assert_eq!(proof.required_floor_bytes_for(0).unwrap(), 0, "the scope-only proof must not raise admission");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.required_phase_floor_bytes(), 0, "a literal input has no source-read reservation");
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&plan)));

        let different_plan = capability_free_literal_plan(DataType::Int32);
        assert!(
            !envelope.scopes.iter().any(|scope| scope.matches(&different_plan)),
            "a scope-only proof must not authorize a different capability-free native fragment",
        );
    }

    #[test]
    fn one_row_zero_column_literal_keeps_zero_floor() {
        let plan = placeholder_literal_plan();
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
        let envelope = proof
            .resolve_reservation_envelope(reservation_policy())
            .expect("one-row zero-column placeholders do not require a literal byte owner");
        assert_eq!(envelope.required_phase_floor_bytes(), 0);
        assert_eq!(envelope.complete_phase_peak_bytes(), None);
    }

    #[test]
    fn generic_nonempty_literal_rejects_without_a_source_owner_bound() {
        let proof = QueryPlanCapabilityProof::for_query_plan(&one_row_literal_plan()).unwrap();
        let error = proof
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("generic nonempty literals must fail before placement");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_literal_source_bytes_unavailable"));
    }

    #[test]
    fn unreachable_nonempty_literal_does_not_reject_a_reachable_empty_plan() {
        let mut builder = QueryPlanBuilder::new();
        let reachable =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::new(Schema::new(vec![
                Field::new("value", DataType::Int64, false),
            ])))));
        let unreachable_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let unreachable_batch =
            RecordBatch::try_new(Arc::clone(&unreachable_schema), vec![Arc::new(Int64Array::from(vec![1]))])
                .expect("unreachable literal must be typed");
        let _unreachable = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(unreachable_schema, vec![unreachable_batch])));
        let plan = builder.finish_gpu_resident(reachable);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
        proof
            .resolve_reservation_envelope(reservation_policy())
            .expect("unreachable literal sources are not executed and must not reject admission");
    }

    #[test]
    fn cugraph_side_input_frame_is_additive_and_retains_both_exact_scopes() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let graph_peak = evidence.cold_model_requested_bytes();
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");
        let side_input = nullable_literal_side_input_plan();
        let side_proof = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&side_input)
            .expect("the exact nullable predicate fragment must issue a bounded side-input proof");
        let side_frame_bytes = crate::exec::join_output_upper_bound_bytes(
            2,
            &[JoinOutputColumnUpperBoundInput {
                data_type: DataType::Int64,
                input_max_string_length_bytes: None,
                input_total_string_bytes: None,
            }],
        )
        .expect("fixed-width side-input frame is finite");
        assert_eq!(side_proof.as_requirement().side_input_frame_owner_term_bytes(), vec![side_frame_bytes]);
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [side_proof]).expect("proofs compose");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).expect("envelope resolves");

        assert_eq!(envelope.required_phase_floor_bytes(), graph_peak + side_frame_bytes);
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(graph_peak + side_frame_bytes));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&graph_plan)));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&side_input)));

        let generic_side_proof = QueryPlanCapabilityProof::for_query_plan(&nullable_literal_side_input_plan())
            .expect("generic side-input proof issues an exact scope");
        let rejected = QueryPlanCapabilityProof::combine(
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
                .expect("graph plan must issue its cuGraph capability proof")
                .with_graph_memory_admission_evidence(&evidence)
                .expect("finite graph evidence"),
            [generic_side_proof],
        )
        .expect("generic proof composition remains structurally valid")
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("the generic predicate proof must reject its unpriced literal source");
        assert_eq!(rejected.fact_value("reason"), Some("attempt_owner_term_literal_source_bytes_unavailable"));
    }

    #[test]
    fn cugraph_side_input_proof_fails_closed_without_a_finite_fixed_width_frame() {
        let unknown = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&unknown_row_bound_side_input_plan())
            .expect_err("an unknown resident row bound cannot price a side-input frame");
        assert_eq!(unknown.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(unknown.fact_value("schedule_reason"), Some("side_input_row_bound_unavailable"));
        assert_eq!(unknown.fact_value("phase"), Some("side_input_frame"));

        let variable_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Utf8, false)]));
        let mut variable_builder = QueryPlanBuilder::new();
        let variable_source =
            variable_builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(variable_schema)));
        let variable = variable_builder.finish_gpu_resident(variable_source);
        let variable_error = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&variable)
            .expect_err("variable-width predicate output lacks a finite engine bound");
        assert_eq!(variable_error.fact_value("schedule_reason"), Some("side_input_schema_unmodeled"));

        let overflow = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&overflow_row_bound_side_input_plan())
            .expect_err("saturated side-input frame bytes must fail closed");
        assert_eq!(overflow.fact_value("schedule_reason"), Some("side_input_frame_bytes_overflow"));
    }

    #[test]
    fn duplicate_cugraph_side_input_identity_is_rejected() {
        let side_input = nullable_literal_side_input_plan();
        let first = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&side_input).unwrap();
        let second = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&side_input).unwrap();
        let error = QueryPlanCapabilityProof::combine(first, [second])
            .expect_err("one exact predicate identity cannot authorize two ambiguous side-input occurrences");
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("duplicate_predicate_identity"));
        assert_eq!(error.fact_value("phase"), Some("side_input_frame"));
    }

    #[test]
    fn two_row_group_zero_column_count_envelope_covers_a_retained_partial_and_next_source_read() {
        let plan = zero_column_count_plan(2);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("host-sink plan issues a proof");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        let minimum_read = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();

        assert_eq!(envelope.complete_phase_peak_bytes(), Some(envelope.required_phase_floor_bytes()));
        assert!(
            envelope.required_phase_floor_bytes() > minimum_read * 2,
            "the second source read must coexist with the retained first partial and scalar finalization state"
        );
    }

    #[test]
    fn combining_an_unmodeled_fragment_clears_the_complete_phase_target() {
        let known = QueryPlanCapabilityProof::for_query_plan(&zero_column_count_plan(2))
            .expect("zero-column aggregate issues a proof");
        let known_envelope = known.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(known_envelope.complete_phase_peak_bytes().is_some());

        let unmodeled =
            QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
        let combined = QueryPlanCapabilityProof::combine(known, [unmodeled]).unwrap();
        let combined_envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();

        assert_eq!(combined_envelope.complete_phase_peak_bytes(), None);
        assert!(combined_envelope.required_phase_floor_bytes() >= known_envelope.required_phase_floor_bytes());
    }

    #[test]
    fn graph_phase_envelope_is_complete_and_bound_to_its_exact_scope() {
        let schema = graph_edge_schema();
        let plan = graph_bfs_plan_with_local_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&plan);
        let expected_peak = evidence.cold_model_with_source_stage_bytes(96).unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue its cuGraph capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.required_phase_floor_bytes(), expected_peak);
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_peak));

        let unmodeled =
            QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).unwrap();
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_composed_native_source_stage_without_losing_its_exact_scope() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let expected_graph_peak = evidence.cold_model_with_source_stage_bytes(96).unwrap();
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof");
        let source_plan = source_stage_plan(schema);
        let source_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&source_plan)
            .expect("graph-input source plan must issue its graph-input capability proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [source_proof])
            .expect("graph and exact source proofs compose")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_graph_peak));
        assert_eq!(envelope.required_phase_floor_bytes(), expected_graph_peak);
    }

    #[test]
    fn graph_phase_rejects_an_unmodeled_marked_local_graph_input() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof");
        let known_input = source_stage_plan(Arc::clone(&schema));
        let known_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&known_input)
            .expect("known graph-input source must issue its graph-input capability proof");
        let unknown_input = unknown_source_stage_plan(schema);
        let unknown_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&unknown_input)
            .expect("unknown graph-input source must issue its graph-input capability proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [known_input_proof, unknown_input_proof])
            .expect("graph and graph-input proofs compose")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");

        let requirement = proof.as_requirement();
        assert_eq!(requirement.source_stage_bytes, None);
        let unknown_shape = requirement
            .reservation_shapes
            .iter()
            .find(|shape| shape.has_local_parquet_source() && !shape.has_modeled_source_stage())
            .expect("the unknown graph-input source remains an exact reservation scope");
        assert!(!unknown_shape.is_accounted_by_graph_phase());

        let error = proof
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("an unmodeled local graph input must keep the graph envelope incomplete");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_local_source_bytes_unavailable"));
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_literal_graph_input_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof");
        let graph_input = one_row_literal_plan();
        let graph_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&graph_input)
            .expect("the exact graph-input fragment must issue a proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [graph_input_proof])
            .expect("graph and exact graph-input proofs compose")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(evidence.cold_model_requested_bytes()));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&graph_input)));

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_result_passthrough_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof");
        let passthrough = graph_result_passthrough_plan("graph_result");
        let passthrough_proof =
            QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(&passthrough, "graph_result")
                .expect("the exact graph-result source handoff must issue its retained-scope proof");
        let proof = QueryPlanCapabilityProof::combine(
            graph_proof.with_graph_memory_admission_evidence(&evidence).expect("finite graph evidence"),
            [passthrough_proof],
        )
        .expect("a refined graph proof and its graph-result passthrough proof compose");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(evidence.cold_model_requested_bytes()));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&passthrough)));

        let different_input = QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(
            &passthrough,
            "different_graph_result",
        );
        assert!(different_input.is_none(), "the graph frontend must attest the exact GPU-result input name");

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_bounded_sort_consumer_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
        let evidence = graph_memory_evidence(&graph_plan);
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence");
        let sort_consumer = graph_result_sort_consumer_plan("graph_result");
        let sort_consumer_proof =
            QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan(&sort_consumer, "graph_result")
                .expect("the exact bounded graph-result sort must issue its retained-scope proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [sort_consumer_proof])
            .expect("a refined graph proof and its bounded sort proof compose");

        let proof = proof
            .requirement_for_device_with_preflight(0, |_, _| {
                Ok(crate::admission::requirement::SortRadixPreflight::new_for_test(128, 80, 256, 80))
            })
            .expect("the deterministic test preflight resolves the sort owner terms");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(envelope.complete_phase_peak_bytes().unwrap() >= evidence.cold_model_requested_bytes());
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&sort_consumer)));

        let unrelated = graph_result_passthrough_plan("different_graph_result");
        assert!(
            QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan(&unrelated, "graph_result")
                .is_none(),
            "the bounded consumer must consume the exact graph-result input",
        );

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    fn schedule_occurrence(
        occurrence_id: &str,
        scope: &str,
        isolated_peak_bytes: Option<u64>,
        retained_output_bytes: Option<u64>,
    ) -> AttemptReservationProducerOccurrence {
        AttemptReservationProducerOccurrence::new(occurrence_id, scope, isolated_peak_bytes, retained_output_bytes)
    }

    fn schedule(
        fixed_bytes: Option<u64>,
        producers: Vec<AttemptReservationProducerOccurrence>,
        consumer_scope: Option<&str>,
        consumer_peak_bytes: Option<u64>,
    ) -> AttemptReservationSchedule {
        AttemptReservationSchedule::new(fixed_bytes, producers, consumer_scope.map(str::to_owned), consumer_peak_bytes)
    }

    fn schedule_scope(scope: &str) -> AttemptReservationShape {
        AttemptReservationShape {
            execution_scope_fingerprint: scope.to_owned(),
            has_local_parquet_source: false,
            source_stage_bytes: None,
            complete_zero_column_global_aggregate: None,
            complete_graph_memory_phase: None,
            complete_cugraph_side_input_phase_bytes: None,
            accounted_by_graph_phase: false,
            is_cugraph_graph_input_scope: false,
            is_cugraph_side_input_scope: false,
            is_attested_graph_result_passthrough_scope: false,
            is_attested_graph_result_bounded_consumer_scope: false,
            sort_phase_peak_bytes: None,
            sort_metadata: None,
            sort_rejection_reason: None,
            owner_term_rejection_reason: None,
            ordered_schedule: None,
        }
    }

    fn assert_schedule_rejected(schedule: &AttemptReservationSchedule, expected_reason: &str) {
        let error = schedule.checked_fold().expect_err("incomplete schedule must reject closed");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some(expected_reason));
    }

    #[test]
    fn ordered_schedule_n1_folds_fixed_producer_output_and_consumer() {
        let schedule = schedule(
            Some(10),
            vec![schedule_occurrence("p0", "producer", Some(20), Some(5))],
            Some("consumer"),
            Some(7),
        );
        assert_eq!(schedule.checked_fold().unwrap(), 30, "max(10 + 20, 10 + 5 + 7)");
    }

    #[test]
    fn ordered_schedule_folds_heterogeneous_two_producers() {
        let schedule = schedule(
            Some(10),
            vec![
                schedule_occurrence("pagerank", "p", Some(20), Some(5)),
                schedule_occurrence("forceatlas", "f", Some(40), Some(8)),
            ],
            Some("consumer"),
            Some(7),
        );
        assert_eq!(schedule.checked_fold().unwrap(), 55, "the second producer coexists with the retained first output");
        assert_ne!(schedule.checked_fold().unwrap(), 80, "scheduled producers must not be blindly summed");
    }

    #[test]
    fn ordered_schedule_retains_equal_structural_occurrences() {
        let schedule = schedule(
            Some(4),
            vec![
                schedule_occurrence("same-0", "equal", Some(10), Some(6)),
                schedule_occurrence("same-1", "equal", Some(10), Some(6)),
            ],
            Some("consumer"),
            Some(1),
        );
        assert_eq!(schedule.producer_occurrences().len(), 2);
        assert_eq!(schedule.checked_fold().unwrap(), 20);
    }

    #[test]
    fn ordered_schedule_folds_three_producers() {
        let schedule = schedule(
            Some(3),
            vec![
                schedule_occurrence("p0", "p0", Some(4), Some(5)),
                schedule_occurrence("p1", "p1", Some(8), Some(6)),
                schedule_occurrence("p2", "p2", Some(2), Some(7)),
            ],
            Some("consumer"),
            Some(9),
        );
        assert_eq!(schedule.checked_fold().unwrap(), 30);
    }

    #[test]
    fn ordered_schedule_is_order_sensitive() {
        let first = schedule(
            Some(5),
            vec![
                schedule_occurrence("large", "large", Some(40), Some(1)),
                schedule_occurrence("small", "small", Some(10), Some(20)),
            ],
            Some("consumer"),
            Some(1),
        );
        let second = schedule(
            Some(5),
            vec![
                schedule_occurrence("small", "small", Some(10), Some(20)),
                schedule_occurrence("large", "large", Some(40), Some(1)),
            ],
            Some("consumer"),
            Some(1),
        );
        assert_eq!(first.checked_fold().unwrap(), 45);
        assert_eq!(second.checked_fold().unwrap(), 65);
    }

    #[test]
    fn ordered_schedule_rejects_missing_evidence() {
        assert_schedule_rejected(
            &schedule(None, vec![schedule_occurrence("p", "p", Some(1), Some(1))], Some("c"), Some(1)),
            "missing_attempt_wide_fixed_bytes",
        );
        assert_schedule_rejected(&schedule(Some(0), Vec::new(), Some("c"), Some(1)), "missing_producer_evidence");
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), Some(1))], None, Some(1)),
            "missing_consumer_scope",
        );
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), Some(1))], Some("c"), None),
            "missing_consumer_additional_peak",
        );
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("p", "p", None, Some(1))], Some("c"), Some(1)),
            "missing_producer_isolated_peak",
        );
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), None)], Some("c"), Some(1)),
            "missing_producer_retained_output",
        );
    }

    #[test]
    fn ordered_schedule_rejects_duplicate_occurrence_ids() {
        assert_schedule_rejected(
            &schedule(
                Some(0),
                vec![
                    schedule_occurrence("duplicate", "p0", Some(1), Some(1)),
                    schedule_occurrence("duplicate", "p1", Some(1), Some(1)),
                ],
                Some("c"),
                Some(1),
            ),
            "duplicate_producer_occurrence_id",
        );
    }

    #[test]
    fn ordered_schedule_rejects_scope_mismatch_but_allows_auxiliary_scopes() {
        let schedule =
            schedule(Some(0), vec![schedule_occurrence("p", "producer", Some(1), Some(1))], Some("consumer"), Some(1));
        let scopes = vec![schedule_scope("producer"), schedule_scope("consumer"), schedule_scope("auxiliary")];
        assert_eq!(schedule.checked_fold_with_scopes(&scopes).unwrap(), 2);

        let mismatched = vec![schedule_scope("different"), schedule_scope("consumer")];
        let error = schedule.checked_fold_with_scopes(&mismatched).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("schedule_scope_coverage_mismatch"));
    }

    #[test]
    fn envelope_uses_schedule_peak_as_the_composed_floor_without_deduplicating_scopes() {
        let schedule = schedule(
            Some(10),
            vec![schedule_occurrence("p", "producer", Some(20), Some(5))],
            Some("consumer"),
            Some(7),
        );
        let mut producer = schedule_scope("producer");
        producer.attach_ordered_schedule(schedule).unwrap();
        let mut auxiliary = schedule_scope("auxiliary");
        auxiliary.account_for_by_graph_phase();
        let scopes = vec![producer, schedule_scope("consumer"), auxiliary];
        let envelope = AttemptReservationEnvelope::resolve(&scopes, reservation_policy())
            .expect("auxiliary exact scopes remain valid with a complete schedule");
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(30));
        assert_eq!(envelope.required_phase_floor_bytes(), 30);
        assert!(envelope.ordered_schedule().is_some());
        assert_eq!(envelope.scopes.len(), 3, "equal/auxiliary occurrences are not shape-deduplicated");
    }

    #[test]
    fn envelope_does_not_sum_heterogeneous_isolated_peaks_when_schedule_is_present() {
        let schedule = schedule(
            Some(10),
            vec![
                schedule_occurrence("p", "producer-p", Some(20), Some(5)),
                schedule_occurrence("f", "producer-f", Some(40), Some(8)),
            ],
            Some("consumer"),
            Some(7),
        );
        let mut producer = schedule_scope("producer-p");
        producer.attach_ordered_schedule(schedule).unwrap();
        let scopes = vec![producer, schedule_scope("producer-f"), schedule_scope("consumer")];
        let envelope = AttemptReservationEnvelope::resolve(&scopes, reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(55));
        assert_ne!(envelope.complete_phase_peak_bytes(), Some(80));
    }

    fn resolved_runtime_schedule_envelope(consumer_plan: &QueryPlan) -> AttemptReservationEnvelope {
        let mut producer_scope = schedule_scope("producer-scope");
        producer_scope
            .attach_ordered_schedule(schedule(
                Some(0),
                vec![
                    schedule_occurrence("first", "producer-scope", Some(20), Some(5)),
                    schedule_occurrence("second", "producer-scope", Some(30), Some(6)),
                ],
                Some(&canonical_execution_scope_fingerprint(consumer_plan)),
                Some(7),
            ))
            .unwrap();
        AttemptReservationEnvelope::resolve(
            &[
                producer_scope,
                schedule_scope("producer-scope"),
                schedule_scope(&canonical_execution_scope_fingerprint(consumer_plan)),
            ],
            reservation_policy(),
        )
        .expect("complete runtime schedule must resolve")
    }

    #[test]
    fn composed_runtime_validation_rejects_producer_occurrence_order_mismatch() {
        let consumer_plan = graph_result_passthrough_plan("graph-result");
        let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

        let error = envelope
            .validate_composed_execution(
                &consumer_plan,
                &["second".to_owned(), "first".to_owned()],
                envelope.required_phase_floor_bytes(),
            )
            .expect_err("runtime execution must preserve the admitted producer order");

        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
        assert_eq!(error.fact_value("scheduled_occurrence_id"), Some("first"));
    }

    #[test]
    fn composed_runtime_validation_rejects_producer_occurrence_count_mismatch() {
        let consumer_plan = graph_result_passthrough_plan("graph-result");
        let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

        let error = envelope
            .validate_composed_execution(&consumer_plan, &["first".to_owned()], envelope.required_phase_floor_bytes())
            .expect_err("runtime execution must not omit an admitted producer occurrence");

        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_count_mismatch"));
        assert_eq!(error.fact_value("scheduled_occurrence_count"), Some("2"));
        assert_eq!(error.fact_value("runtime_occurrence_count"), Some("1"));
    }

    #[test]
    fn composed_runtime_validation_rejects_consumer_scope_mismatch() {
        let admitted_consumer_plan = graph_result_passthrough_plan("graph-result");
        let envelope = resolved_runtime_schedule_envelope(&admitted_consumer_plan);
        let changed_consumer_plan = graph_result_sort_consumer_plan("graph-result");

        let error = envelope
            .validate_composed_execution(
                &changed_consumer_plan,
                &["first".to_owned(), "second".to_owned()],
                envelope.required_phase_floor_bytes(),
            )
            .expect_err("runtime execution must preserve the admitted consumer scope");

        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("runtime_consumer_scope_mismatch"));
    }

    #[test]
    fn composed_runtime_validation_rejects_immutable_grant_below_schedule_peak() {
        let consumer_plan = graph_result_passthrough_plan("graph-result");
        let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

        let error = envelope
            .validate_composed_execution(&consumer_plan, &["first".to_owned(), "second".to_owned()], 34)
            .expect_err("runtime execution must not begin below its schedule peak");

        assert_eq!(envelope.required_phase_floor_bytes(), 35);
        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("immutable_grant_below_schedule_peak"));
        assert_eq!(error.fact_value("immutable_grant_bytes"), Some("34"));
        assert_eq!(error.fact_value("complete_phase_peak_bytes"), Some("35"));
    }

    #[test]
    fn composed_schedule_projection_is_lossless_and_fingerprint_stable() {
        let consumer_plan = graph_result_passthrough_plan("graph-result");
        let envelope = resolved_runtime_schedule_envelope(&consumer_plan);
        let attempt_id = crate::observability::AttemptId::mint();
        let projection = envelope
            .composed_schedule_projection(attempt_id, 2, 41, envelope.required_phase_floor_bytes())
            .expect("resolved composed schedule must publish a projection");

        assert_eq!(projection.attempt_id, attempt_id);
        assert_eq!(projection.device_ordinal, 2);
        assert_eq!(projection.grant_id, 41);
        assert_eq!(projection.attempt_wide_fixed_bytes, 0);
        assert_eq!(projection.producer_occurrences.len(), 2);
        assert_eq!(projection.producer_occurrences[0].occurrence_id, "first");
        assert_eq!(projection.producer_occurrences[0].stage_peak_bytes, 20);
        assert_eq!(projection.producer_occurrences[1].stage_peak_bytes, 35);
        assert_eq!(projection.consumer_stage_peak_bytes, 18);
        assert_eq!(projection.complete_peak_bytes, 35);
        assert!(projection.fingerprint.starts_with("composed-schedule-fnv1a64:"));
        assert!(projection.fingerprint_matches_contents());

        let mut corrupted = projection.clone();
        corrupted.fingerprint = "composed-schedule-fnv1a64:0000000000000000".to_owned();
        assert!(!corrupted.fingerprint_matches_contents());

        let repeat = envelope
            .composed_schedule_projection(attempt_id, 2, 99, envelope.required_phase_floor_bytes())
            .expect("same schedule must retain its deterministic fingerprint");
        assert_eq!(projection.fingerprint, repeat.fingerprint);
        assert_ne!(projection.grant_id, repeat.grant_id, "placement identity is not part of schedule identity");
    }

    #[test]
    fn composed_schedule_projection_fails_closed_without_an_admitted_schedule() {
        let envelope = AttemptReservationEnvelope::resolve(&[schedule_scope("ordinary")], reservation_policy())
            .expect("ordinary scope resolves without a composed schedule");
        let error = envelope
            .composed_schedule_projection(crate::observability::AttemptId::mint(), 0, 1, 1)
            .expect_err("projection requests must fail closed when schedule evidence is absent");
        assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        assert_eq!(error.fact_value("schedule_reason"), Some("missing_admitted_schedule"));
    }

    #[test]
    fn ordered_schedule_rejects_empty_occurrence_id_and_scope() {
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("", "p", Some(1), Some(1))], Some("c"), Some(1)),
            "missing_producer_occurrence_id",
        );
        assert_schedule_rejected(
            &schedule(Some(0), vec![schedule_occurrence("p", "", Some(1), Some(1))], Some("c"), Some(1)),
            "missing_producer_occurrence_scope",
        );
    }

    #[test]
    fn ordered_schedule_rejects_every_checked_fold_overflow_edge() {
        assert_schedule_rejected(
            &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(1), Some(0))], Some("c"), Some(0)),
            "producer_stage_isolated_peak_overflow",
        );
        assert_schedule_rejected(
            &schedule(
                Some(u64::MAX),
                vec![
                    schedule_occurrence("p0", "p0", Some(0), Some(1)),
                    schedule_occurrence("p1", "p1", Some(0), Some(0)),
                ],
                Some("c"),
                Some(0),
            ),
            "producer_stage_fixed_plus_retained_overflow",
        );
        assert_schedule_rejected(
            &schedule(
                Some(0),
                vec![
                    schedule_occurrence("p0", "p0", Some(0), Some(u64::MAX)),
                    schedule_occurrence("p1", "p1", Some(0), Some(1)),
                ],
                Some("c"),
                Some(0),
            ),
            "retained_output_accumulation_overflow",
        );
        assert_schedule_rejected(
            &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(0), Some(1))], Some("c"), Some(0)),
            "consumer_stage_fixed_plus_retained_overflow",
        );
        assert_schedule_rejected(
            &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(0), Some(0))], Some("c"), Some(1)),
            "consumer_stage_additional_peak_overflow",
        );
    }
}
