//! Policy-resolved reservation envelopes for one admitted attempt.
//!
//! A capability proof is issued while lowering owns only the native plan. The
//! service owns source scheduling and chunk limits, so it resolves the plan
//! facts below immediately before ledger placement. This keeps a descriptive
//! source-size estimate out of immutable-grant sizing while still making
//! policy-created concurrent phases explicit.

use std::collections::{HashMap, HashSet};

use arrow::datatypes::DataType;

use crate::{
    admission::requirement::{
        AllocationOwnerTerm, AttemptBoundDerivation, OwnerTermId, checked_owner_sum, owner_inventory_covers_scopes,
    },
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

fn admitted_manifest_shape<'a>(
    scopes: &'a [AttemptReservationShape],
    schedule: Option<&AttemptReservationSchedule>,
) -> Result<&'a AttemptReservationShape> {
    let manifests = scopes.iter().filter(|scope| !scope.producer_occurrences.is_empty()).collect::<Vec<_>>();
    let [manifest] = manifests.as_slice() else {
        let reason = if manifests.is_empty() { "missing_admitted_manifest" } else { "duplicate_admitted_manifest" };
        return Err(graph_schedule_error(reason, None, Some("manifest")));
    };
    if let Some(schedule) = schedule
        && !manifest.ordered_schedule.as_ref().is_some_and(|attached| attached == schedule)
    {
        return Err(graph_schedule_error("missing_admitted_manifest", None, Some("manifest")));
    }
    Ok(manifest)
}

/// One ordered graph-producer occurrence in the mandatory composed-attempt
/// identity core.
///
/// Equal plans remain distinct allocation owners because occurrence identity
/// is carried separately from the structural execution scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationProducerOccurrence {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    retained_output_term_id: OwnerTermId,
}

impl AttemptReservationProducerOccurrence {
    pub(crate) fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }

    pub(crate) fn retained_output_term_id(&self) -> &OwnerTermId {
        &self.retained_output_term_id
    }

    pub(crate) fn from_term_id(
        occurrence_id: impl Into<String>,
        execution_scope_fingerprint: impl Into<String>,
        retained_output_term_id: OwnerTermId,
    ) -> Self {
        Self {
            occurrence_id: occurrence_id.into(),
            execution_scope_fingerprint: execution_scope_fingerprint.into(),
            retained_output_term_id,
        }
    }

    pub(crate) fn validate_retained_output_term(&self, term: &AllocationOwnerTerm) -> Result<()> {
        if !term.is_retained_output() {
            return Err(graph_schedule_error(
                "retained_output_term_role_invalid",
                Some(&self.occurrence_id),
                Some("retained_output"),
            ));
        }
        let expected_occurrence = format!("composed:{}", self.occurrence_id);
        let actual_occurrence = term.id().occurrence_label();
        if actual_occurrence != expected_occurrence {
            return Err(graph_schedule_error(
                "retained_output_term_occurrence_mismatch",
                Some(&self.occurrence_id),
                Some("retained_output"),
            )
            .with_fact("expected_occurrence", expected_occurrence)
            .with_fact("actual_occurrence", actual_occurrence));
        }
        if term.id().execution_scope_fingerprint() != self.execution_scope_fingerprint {
            return Err(graph_schedule_error(
                "retained_output_term_scope_mismatch",
                Some(&self.occurrence_id),
                Some("retained_output"),
            )
            .with_fact("expected_scope", &self.execution_scope_fingerprint)
            .with_fact("actual_scope", term.id().execution_scope_fingerprint()));
        }
        Ok(())
    }
}

/// Optional ordered phase evidence attached to the mandatory occurrence core.
/// A conservative owner-sum derivation intentionally has no phase rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationProducerPhase {
    isolated_peak_term_groups: Vec<Vec<OwnerTermId>>,
    alternative_term_ids: Vec<OwnerTermId>,
    concurrent_side_input_term_ids: Vec<OwnerTermId>,
    dominated_term_ids: Vec<OwnerTermId>,
    nested_schedule: bool,
    #[cfg(test)]
    isolated_peak_bytes: Option<u64>,
    #[cfg(test)]
    retained_output_bytes: Option<u64>,
}

impl AttemptReservationProducerPhase {
    #[cfg(test)]
    pub(crate) fn from_term_groups(isolated_peak_term_groups: Vec<Vec<OwnerTermId>>) -> Self {
        Self::from_term_groups_with_alternatives_and_side_inputs(isolated_peak_term_groups, Vec::new(), Vec::new())
    }

    pub(crate) fn from_term_groups_with_alternatives_and_side_inputs(
        isolated_peak_term_groups: Vec<Vec<OwnerTermId>>,
        alternative_term_ids: Vec<OwnerTermId>,
        concurrent_side_input_term_ids: Vec<OwnerTermId>,
    ) -> Self {
        Self {
            isolated_peak_term_groups,
            alternative_term_ids,
            concurrent_side_input_term_ids,
            dominated_term_ids: Vec::new(),
            nested_schedule: false,
            #[cfg(test)]
            isolated_peak_bytes: None,
            #[cfg(test)]
            retained_output_bytes: None,
        }
    }

    pub(crate) fn from_nested_term_groups(
        isolated_peak_term_groups: Vec<Vec<OwnerTermId>>,
        alternative_term_ids: Vec<OwnerTermId>,
        dominated_term_ids: Vec<OwnerTermId>,
    ) -> Self {
        Self {
            isolated_peak_term_groups,
            alternative_term_ids,
            concurrent_side_input_term_ids: Vec::new(),
            dominated_term_ids,
            nested_schedule: true,
            #[cfg(test)]
            isolated_peak_bytes: None,
            #[cfg(test)]
            retained_output_bytes: None,
        }
    }

    #[cfg(test)]
    fn new(isolated_peak_bytes: Option<u64>, retained_output_bytes: Option<u64>) -> Self {
        Self {
            isolated_peak_term_groups: Vec::new(),
            alternative_term_ids: Vec::new(),
            concurrent_side_input_term_ids: Vec::new(),
            dominated_term_ids: Vec::new(),
            nested_schedule: false,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttemptReservationConsumerState {
    Passthrough,
    Bounded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationSchedule {
    attempt_wide_fixed_term_ids: Vec<OwnerTermId>,
    producer_phases: Vec<AttemptReservationProducerPhase>,
    consumer_state: AttemptReservationConsumerState,
    consumer_peak_term_groups: Vec<Vec<OwnerTermId>>,
    consumer_dominated_term_ids: Vec<OwnerTermId>,
    #[cfg(test)]
    test_attempt_wide_fixed_bytes: Option<u64>,
    #[cfg(test)]
    test_consumer_additional_peak_bytes: Option<u64>,
}

pub(crate) type NestedProducerPeakTerms = (Vec<Vec<OwnerTermId>>, Vec<OwnerTermId>, Vec<OwnerTermId>);

impl AttemptReservationSchedule {
    #[cfg(test)]
    pub(crate) fn from_phase_term_ids(
        attempt_wide_fixed_term_ids: Vec<OwnerTermId>,
        producer_phases: Vec<AttemptReservationProducerPhase>,
        consumer_state: AttemptReservationConsumerState,
        consumer_peak_term_ids: Vec<OwnerTermId>,
    ) -> Self {
        let consumer_peak_term_groups =
            if consumer_peak_term_ids.is_empty() { Vec::new() } else { vec![consumer_peak_term_ids] };
        Self::from_phase_term_groups_with_dominated(
            attempt_wide_fixed_term_ids,
            producer_phases,
            consumer_state,
            consumer_peak_term_groups,
            Vec::new(),
        )
    }

    pub(crate) fn from_phase_term_groups_with_dominated(
        attempt_wide_fixed_term_ids: Vec<OwnerTermId>,
        producer_phases: Vec<AttemptReservationProducerPhase>,
        consumer_state: AttemptReservationConsumerState,
        consumer_peak_term_groups: Vec<Vec<OwnerTermId>>,
        consumer_dominated_term_ids: Vec<OwnerTermId>,
    ) -> Self {
        Self {
            attempt_wide_fixed_term_ids,
            producer_phases,
            consumer_state,
            consumer_peak_term_groups,
            consumer_dominated_term_ids,
            #[cfg(test)]
            test_attempt_wide_fixed_bytes: Some(0),
            #[cfg(test)]
            test_consumer_additional_peak_bytes: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(
        attempt_wide_fixed_bytes: Option<u64>,
        producer_phases: Vec<AttemptReservationProducerPhase>,
        consumer_additional_peak_bytes: Option<u64>,
    ) -> Self {
        Self {
            attempt_wide_fixed_term_ids: Vec::new(),
            consumer_state: if consumer_additional_peak_bytes == Some(0) {
                AttemptReservationConsumerState::Passthrough
            } else {
                AttemptReservationConsumerState::Bounded
            },
            producer_phases,
            consumer_peak_term_groups: Vec::new(),
            consumer_dominated_term_ids: Vec::new(),
            test_attempt_wide_fixed_bytes: attempt_wide_fixed_bytes,
            #[cfg(test)]
            test_consumer_additional_peak_bytes: consumer_additional_peak_bytes,
        }
    }

    fn producer_phase_count_internal(&self) -> usize {
        self.producer_phases.len()
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn phase_group_counts_for_test(&self) -> (usize, usize) {
        (self.producer_phases.len(), self.consumer_peak_term_groups.len())
    }

    fn concurrent_side_input_scope_fingerprints(&self) -> impl Iterator<Item = &str> {
        self.producer_phases
            .iter()
            .flat_map(|phase| phase.concurrent_side_input_term_ids.iter().map(OwnerTermId::execution_scope_fingerprint))
    }

    fn referenced_scope_fingerprints(&self) -> HashSet<&str> {
        self.attempt_wide_fixed_term_ids
            .iter()
            .chain(self.producer_phases.iter().flat_map(|phase| {
                phase
                    .isolated_peak_term_groups
                    .iter()
                    .flatten()
                    .chain(&phase.alternative_term_ids)
                    .chain(&phase.concurrent_side_input_term_ids)
                    .chain(&phase.dominated_term_ids)
            }))
            .chain(self.consumer_peak_term_groups.iter().flatten())
            .chain(&self.consumer_dominated_term_ids)
            .map(OwnerTermId::execution_scope_fingerprint)
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn attempt_wide_fixed_term_ids(&self) -> &[OwnerTermId] {
        &self.attempt_wide_fixed_term_ids
    }

    #[cfg(test)]
    pub(crate) fn consumer_state(&self) -> AttemptReservationConsumerState {
        self.consumer_state
    }

    #[cfg(test)]
    pub(crate) fn consumer_dominated_term_ids(&self) -> &[OwnerTermId] {
        &self.consumer_dominated_term_ids
    }

    #[cfg(test)]
    pub(crate) fn consumer_peak_term_groups(&self) -> &[Vec<OwnerTermId>] {
        &self.consumer_peak_term_groups
    }

    #[cfg(test)]
    pub(crate) fn consumer_peak_bytes_for_test(&self, owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
        self.consumer_peak_bytes(owner_terms, "test")
    }

    #[cfg(test)]
    pub(crate) fn producer_phase_term_ids(&self) -> Vec<OwnerTermId> {
        self.producer_phases
            .iter()
            .flat_map(|phase| {
                phase
                    .isolated_peak_term_groups
                    .iter()
                    .flatten()
                    .chain(&phase.alternative_term_ids)
                    .chain(&phase.concurrent_side_input_term_ids)
                    .chain(&phase.dominated_term_ids)
            })
            .cloned()
            .collect()
    }

    pub(crate) fn nested_producer_peak_terms(
        &self,
        producer_occurrences: &[AttemptReservationProducerOccurrence],
    ) -> Result<NestedProducerPeakTerms> {
        if self.producer_phases.len() != producer_occurrences.len() {
            return Err(graph_schedule_error("producer_phase_row_count_mismatch", None, Some("nested_schedule")));
        }

        let mut groups = Vec::new();
        let mut retained_prior = Vec::new();
        let mut dominated = Vec::new();
        let mut inner_retained_outputs = Vec::with_capacity(producer_occurrences.len());
        for (occurrence, phase) in producer_occurrences.iter().zip(&self.producer_phases) {
            if phase.isolated_peak_term_groups.is_empty() {
                return Err(graph_schedule_error(
                    "missing_producer_isolated_peak",
                    Some(occurrence.occurrence_id()),
                    Some("nested_schedule"),
                ));
            }
            for isolated in &phase.isolated_peak_term_groups {
                let mut group = Vec::with_capacity(
                    retained_prior.len() + isolated.len() + phase.concurrent_side_input_term_ids.len(),
                );
                group.extend(retained_prior.iter().cloned());
                group.extend(isolated.iter().cloned());
                group.extend(phase.concurrent_side_input_term_ids.iter().cloned());
                groups.push(group);
            }
            dominated.extend(phase.dominated_term_ids.iter().cloned());
            retained_prior.push(occurrence.retained_output_term_id().clone());
            inner_retained_outputs.push(occurrence.retained_output_term_id().clone());
        }

        match self.consumer_state {
            AttemptReservationConsumerState::Passthrough => groups.push(retained_prior),
            AttemptReservationConsumerState::Bounded => {
                if self.consumer_peak_term_groups.is_empty() {
                    return Err(graph_schedule_error("missing_consumer_peak_terms", None, Some("nested_schedule")));
                }
                for consumer in &self.consumer_peak_term_groups {
                    let mut group = Vec::with_capacity(retained_prior.len() + consumer.len());
                    group.extend(retained_prior.iter().cloned());
                    group.extend(consumer.iter().cloned());
                    groups.push(group);
                }
            },
        }
        dominated.extend(self.consumer_dominated_term_ids.iter().cloned());
        Ok((groups, dominated, inner_retained_outputs))
    }

    fn checked_fold_with_scopes(&self, scopes: &[AttemptReservationShape]) -> Result<u64> {
        self.checked_fold_with_inventory(scopes, &[])
    }

    pub(crate) fn checked_fold_with_inventory(
        &self,
        scopes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<u64> {
        let fixed = self.fixed_bytes(owner_terms)?;
        if self.producer_phases.is_empty() {
            return Err(graph_schedule_error("missing_producer_evidence", None, None));
        }
        let manifest = self.manifest_shape(scopes)?;
        manifest.validate_manifest_core()?;
        let actual_scope_counts =
            scopes.iter().fold(std::collections::HashMap::<&str, usize>::new(), |mut counts, scope| {
                *counts.entry(scope.execution_scope_fingerprint()).or_default() += 1;
                counts
            });
        let mut expected_scope_counts = manifest.manifest_scope_counts();
        for scope in self.concurrent_side_input_scope_fingerprints() {
            *expected_scope_counts.entry(scope).or_default() += 1;
        }
        for scope in self.referenced_scope_fingerprints() {
            expected_scope_counts.entry(scope).or_insert(1);
        }
        if expected_scope_counts
            .iter()
            .any(|(scope, expected)| actual_scope_counts.get(scope).copied().unwrap_or_default() < *expected)
        {
            return Err(graph_schedule_error("schedule_scope_coverage_mismatch", None, Some("scope_coverage")));
        }
        if self.producer_phases.len() != manifest.producer_occurrences.len() {
            return Err(graph_schedule_error("producer_phase_row_count_mismatch", None, Some("schedule"))
                .with_fact("producer_phase_row_count", self.producer_phases.len())
                .with_fact("producer_occurrence_count", manifest.producer_occurrences.len()));
        }
        if !owner_terms.is_empty() {
            self.validate_term_references(manifest, owner_terms, scopes)?;
        }
        let consumer_peak = match self.consumer_peak_bytes(owner_terms, manifest.execution_scope_fingerprint()) {
            Ok(bytes) => bytes,
            Err(error)
                if !owner_terms.is_empty()
                    && matches!(
                        error.fact_value("schedule_reason"),
                        Some("missing_consumer_peak_terms" | "missing_consumer_additional_peak")
                    ) =>
            {
                return Err(error.with_fact("ordered_inventory_incomplete", "true"));
            },
            Err(error) => return Err(error),
        };

        let mut retained_prior = 0_u64;
        let mut complete_peak = 0_u64;
        for (occurrence, phase) in manifest.producer_occurrences.iter().zip(&self.producer_phases) {
            let isolated_peak = match self.producer_isolated_peak_bytes(occurrence, phase, owner_terms) {
                Ok(bytes) => bytes,
                Err(error)
                    if !owner_terms.is_empty()
                        && error.fact_value("schedule_reason") == Some("missing_producer_isolated_peak") =>
                {
                    return Err(error.with_fact("ordered_inventory_incomplete", "true"));
                },
                Err(error) => return Err(error),
            };
            let retained_output = self.producer_retained_output_bytes(occurrence, phase, owner_terms)?;
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

    fn fixed_bytes(&self, owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
        if !self.attempt_wide_fixed_term_ids.is_empty() {
            let mut seen = HashSet::with_capacity(self.attempt_wide_fixed_term_ids.len());
            if self.attempt_wide_fixed_term_ids.iter().any(|id| !seen.insert(id)) {
                return Err(graph_schedule_error("duplicate_owner_term_reference", None, Some("attempt_wide_fixed")));
            }
            return self.attempt_wide_fixed_term_ids.iter().try_fold(0_u64, |total, term_id| {
                let term = self.resolve_term(term_id, owner_terms, None, "attempt_wide_fixed")?;
                if !term.is_attempt_wide_fixed() {
                    return Err(graph_schedule_error(
                        "owner_term_role_class_mismatch",
                        None,
                        Some("attempt_wide_fixed"),
                    )
                    .with_fact("owner_term_role", term_id.role_label())
                    .with_fact("owner_term_occurrence", term_id.occurrence_label()));
                }
                let bytes = term.bytes();
                total.checked_add(bytes).ok_or_else(|| {
                    graph_schedule_error("attempt_wide_fixed_sum_overflow", None, Some("owner_term_sum"))
                })
            });
        }

        #[cfg(test)]
        if let Some(fixed) = self.test_attempt_wide_fixed_bytes {
            return Ok(fixed);
        }
        #[cfg(test)]
        return Err(graph_schedule_error("missing_attempt_wide_fixed_bytes", None, None));
        #[cfg(not(test))]
        Ok(0)
    }

    fn producer_isolated_peak_bytes(
        &self,
        occurrence: &AttemptReservationProducerOccurrence,
        phase: &AttemptReservationProducerPhase,
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<u64> {
        if !phase.isolated_peak_term_groups.is_empty() {
            let mut isolated_peak = 0_u64;
            for group in &phase.isolated_peak_term_groups {
                let mut seen = HashSet::with_capacity(group.len());
                if group.iter().any(|id| !seen.insert(id)) {
                    return Err(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_isolated_peak"),
                    ));
                }
                let phase_peak = self.sum_term_ids(
                    group,
                    owner_terms,
                    Some(&occurrence.occurrence_id),
                    "producer_stage_isolated_peak_overflow",
                )?;
                isolated_peak = isolated_peak.max(phase_peak);
            }
            return self.add_concurrent_side_input_peak(occurrence, phase, owner_terms, isolated_peak);
        }
        #[cfg(test)]
        {
            let isolated_peak = phase.isolated_peak_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_isolated_peak", Some(&occurrence.occurrence_id), None)
            })?;
            self.add_concurrent_side_input_peak(occurrence, phase, owner_terms, isolated_peak)
        }
        #[cfg(not(test))]
        {
            Err(graph_schedule_error("missing_producer_isolated_peak", Some(&occurrence.occurrence_id), None))
        }
    }

    fn add_concurrent_side_input_peak(
        &self,
        occurrence: &AttemptReservationProducerOccurrence,
        phase: &AttemptReservationProducerPhase,
        owner_terms: &[AllocationOwnerTerm],
        isolated_peak: u64,
    ) -> Result<u64> {
        if phase.concurrent_side_input_term_ids.is_empty() {
            return Ok(isolated_peak);
        }
        let side_input_peak = self.sum_term_ids(
            &phase.concurrent_side_input_term_ids,
            owner_terms,
            Some(&occurrence.occurrence_id),
            "producer_stage_side_input_overflow",
        )?;
        isolated_peak.checked_add(side_input_peak).ok_or_else(|| {
            graph_schedule_error(
                "producer_stage_side_input_overflow",
                Some(&occurrence.occurrence_id),
                Some("producer_stage_side_input"),
            )
        })
    }

    fn producer_retained_output_bytes(
        &self,
        occurrence: &AttemptReservationProducerOccurrence,
        _phase: &AttemptReservationProducerPhase,
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<u64> {
        #[cfg(test)]
        if owner_terms.is_empty() {
            return _phase.retained_output_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_retained_output", Some(&occurrence.occurrence_id), None)
            });
        }
        let term =
            owner_terms.iter().find(|term| term.id() == &occurrence.retained_output_term_id).ok_or_else(|| {
                graph_schedule_error(
                    "missing_owner_term_reference",
                    Some(&occurrence.occurrence_id),
                    Some("retained_output"),
                )
            })?;
        occurrence.validate_retained_output_term(term)?;
        Ok(term.bytes())
    }

    fn consumer_peak_bytes(&self, owner_terms: &[AllocationOwnerTerm], _consumer_scope: &str) -> Result<u64> {
        match self.consumer_state {
            AttemptReservationConsumerState::Passthrough => {
                if !self.consumer_peak_term_groups.is_empty() {
                    return Err(graph_schedule_error(
                        "passthrough_consumer_has_peak_terms",
                        None,
                        Some("consumer_peak"),
                    ));
                }
                if !self.consumer_dominated_term_ids.is_empty() {
                    return Err(graph_schedule_error(
                        "passthrough_consumer_has_dominated_terms",
                        None,
                        Some("consumer_dominated"),
                    ));
                }
                #[cfg(test)]
                if self.test_consumer_additional_peak_bytes.is_some_and(|bytes| bytes != 0) {
                    return Err(graph_schedule_error(
                        "passthrough_consumer_has_nonzero_peak",
                        None,
                        Some("consumer_peak"),
                    ));
                }
                Ok(0)
            },
            AttemptReservationConsumerState::Bounded => {
                if !self.consumer_peak_term_groups.is_empty() {
                    let mut consumer_peak = 0_u64;
                    for group in &self.consumer_peak_term_groups {
                        let mut seen = HashSet::with_capacity(group.len());
                        if group.iter().any(|id| !seen.insert(id)) {
                            return Err(graph_schedule_error(
                                "duplicate_owner_term_reference",
                                None,
                                Some("consumer_peak"),
                            ));
                        }
                        let group_peak =
                            self.sum_term_ids(group, owner_terms, None, "consumer_stage_additional_peak_overflow")?;
                        consumer_peak = consumer_peak.max(group_peak);
                    }
                    return Ok(consumer_peak);
                }
                #[cfg(test)]
                if let Some(bytes) = self.test_consumer_additional_peak_bytes {
                    return Ok(bytes);
                }
                #[cfg(test)]
                return Err(graph_schedule_error("missing_consumer_additional_peak", None, None));
                #[cfg(not(test))]
                Err(graph_schedule_error("missing_consumer_peak_terms", None, Some("consumer_peak")))
            },
        }
    }

    fn sum_term_ids(
        &self,
        term_ids: &[OwnerTermId],
        owner_terms: &[AllocationOwnerTerm],
        occurrence_id: Option<&str>,
        overflow_reason: &'static str,
    ) -> Result<u64> {
        term_ids.iter().try_fold(0_u64, |total, term_id| {
            let bytes = self.resolve_term_bytes(term_id, owner_terms, occurrence_id, "owner_term")?;
            total
                .checked_add(bytes)
                .ok_or_else(|| graph_schedule_error(overflow_reason, occurrence_id, Some("owner_term_sum")))
        })
    }

    fn resolve_term_bytes(
        &self,
        term_id: &OwnerTermId,
        owner_terms: &[AllocationOwnerTerm],
        occurrence_id: Option<&str>,
        phase: &'static str,
    ) -> Result<u64> {
        self.resolve_term(term_id, owner_terms, occurrence_id, phase).map(AllocationOwnerTerm::bytes)
    }

    fn resolve_term<'a>(
        &self,
        term_id: &OwnerTermId,
        owner_terms: &'a [AllocationOwnerTerm],
        occurrence_id: Option<&str>,
        phase: &'static str,
    ) -> Result<&'a AllocationOwnerTerm> {
        owner_terms.iter().find(|term| term.id() == term_id).ok_or_else(|| {
            graph_schedule_error("missing_owner_term_reference", occurrence_id, Some(phase))
                .with_fact("owner_term_role", term_id.role_label())
                .with_fact("owner_term_occurrence", term_id.occurrence_label())
        })
    }

    pub(crate) fn validate_term_references(
        &self,
        manifest: &AttemptReservationShape,
        owner_terms: &[AllocationOwnerTerm],
        scopes: &[AttemptReservationShape],
    ) -> Result<()> {
        self.validate_term_references_with_totality(manifest, owner_terms, scopes, true)
    }

    pub(crate) fn validate_term_references_for_conservative_sum(
        &self,
        manifest: &AttemptReservationShape,
        owner_terms: &[AllocationOwnerTerm],
        scopes: &[AttemptReservationShape],
    ) -> Result<()> {
        self.validate_term_references_with_totality(manifest, owner_terms, scopes, false)
    }

    fn validate_term_references_with_totality(
        &self,
        manifest: &AttemptReservationShape,
        owner_terms: &[AllocationOwnerTerm],
        scopes: &[AttemptReservationShape],
        require_totality: bool,
    ) -> Result<()> {
        self.fixed_bytes(owner_terms)?;
        manifest.validate_manifest_core()?;
        if self.producer_phases.len() != manifest.producer_occurrences.len() {
            return Err(graph_schedule_error("producer_phase_row_count_mismatch", None, Some("schedule")));
        }
        self.validate_inventory_coverage(manifest, owner_terms, scopes, require_totality)?;
        Ok(())
    }

    fn validate_inventory_coverage(
        &self,
        manifest: &AttemptReservationShape,
        owner_terms: &[AllocationOwnerTerm],
        scopes: &[AttemptReservationShape],
        require_totality: bool,
    ) -> Result<()> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Category {
            Fixed,
            ProducerPhase { producer_index: usize, group_index: usize },
            ConcurrentSideInput { producer_index: usize },
            ProducerDominated { producer_index: usize },
            RetainedOccurrence(usize),
            ConsumerPeak,
            ConsumerDominated,
        }

        let mut references = HashMap::<OwnerTermId, Vec<Category>>::with_capacity(owner_terms.len());
        let mut inventory_ids = HashSet::with_capacity(owner_terms.len());
        for term in owner_terms {
            if !inventory_ids.insert(term.id()) {
                return Err(graph_schedule_error("duplicate_owner_term_inventory", None, Some("inventory"))
                    .with_fact("owner_term_role", term.id().role_label())
                    .with_fact("owner_term_occurrence", term.id().occurrence_label()));
            }
        }
        let mut record = |term_id: &OwnerTermId, category: Category| {
            references.entry(term_id.clone()).or_default().push(category);
        };

        let mut fixed_seen = HashSet::with_capacity(self.attempt_wide_fixed_term_ids.len());
        for term_id in &self.attempt_wide_fixed_term_ids {
            if !fixed_seen.insert(term_id) {
                return Err(graph_schedule_error("duplicate_owner_term_reference", None, Some("attempt_wide_fixed")));
            }
            let term = self.resolve_term(term_id, owner_terms, None, "attempt_wide_fixed")?;
            if !term.is_attempt_wide_fixed() {
                return Err(graph_schedule_error("owner_term_role_class_mismatch", None, Some("attempt_wide_fixed"))
                    .with_fact("owner_term_role", term_id.role_label())
                    .with_fact("owner_term_occurrence", term_id.occurrence_label()));
            }
            record(term_id, Category::Fixed);
        }

        for (phase_index, occurrence) in manifest.producer_occurrences.iter().enumerate() {
            let term = self.resolve_term(
                &occurrence.retained_output_term_id,
                owner_terms,
                Some(&occurrence.occurrence_id),
                "retained_output",
            )?;
            occurrence.validate_retained_output_term(term)?;
            record(&occurrence.retained_output_term_id, Category::RetainedOccurrence(phase_index));
        }

        for (phase_index, (occurrence, phase)) in
            manifest.producer_occurrences.iter().zip(&self.producer_phases).enumerate()
        {
            let mut side_input_seen = HashSet::with_capacity(phase.concurrent_side_input_term_ids.len());
            for term_id in &phase.concurrent_side_input_term_ids {
                if !side_input_seen.insert(term_id) {
                    return Err(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_side_input"),
                    ));
                }
                let term =
                    self.resolve_term(term_id, owner_terms, Some(&occurrence.occurrence_id), "producer_side_input")?;
                self.validate_concurrent_side_input_term(term, occurrence, scopes)?;
                record(term_id, Category::ConcurrentSideInput { producer_index: phase_index });
            }
            let mut alternatives_seen = HashSet::with_capacity(phase.alternative_term_ids.len());
            for term_id in &phase.alternative_term_ids {
                if !alternatives_seen.insert(term_id) {
                    return Err(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_alternative_phase"),
                    ));
                }
                let term = self.resolve_term(
                    term_id,
                    owner_terms,
                    Some(&occurrence.occurrence_id),
                    "producer_alternative_phase",
                )?;
                self.validate_producer_phase_term(term, occurrence, phase.nested_schedule, scopes)?;
                if phase.isolated_peak_term_groups.iter().filter(|group| group.contains(term_id)).count() < 2 {
                    return Err(graph_schedule_error(
                        "producer_alternative_phase_group_invalid",
                        Some(&occurrence.occurrence_id),
                        Some("producer_alternative_phase"),
                    ));
                }
            }
            let mut group_term_ids = HashSet::new();
            for (group_index, group) in phase.isolated_peak_term_groups.iter().enumerate() {
                group_term_ids.clear();
                for term_id in group {
                    if !group_term_ids.insert(term_id) {
                        return Err(graph_schedule_error(
                            "duplicate_owner_term_reference",
                            Some(&occurrence.occurrence_id),
                            Some("producer_isolated_peak"),
                        ));
                    }
                    let term = self.resolve_term(
                        term_id,
                        owner_terms,
                        Some(&occurrence.occurrence_id),
                        "producer_isolated_peak",
                    )?;
                    self.validate_producer_phase_term(term, occurrence, phase.nested_schedule, scopes)?;
                    record(term_id, Category::ProducerPhase { producer_index: phase_index, group_index });
                }
            }
            let mut dominated_seen = HashSet::with_capacity(phase.dominated_term_ids.len());
            for term_id in &phase.dominated_term_ids {
                if !dominated_seen.insert(term_id) {
                    return Err(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_dominated"),
                    ));
                }
                let term =
                    self.resolve_term(term_id, owner_terms, Some(&occurrence.occurrence_id), "producer_dominated")?;
                self.validate_producer_phase_term(term, occurrence, phase.nested_schedule, scopes)?;
                record(term_id, Category::ProducerDominated { producer_index: phase_index });
            }
        }

        for group in &self.consumer_peak_term_groups {
            let mut consumer_peak_seen = HashSet::with_capacity(group.len());
            for term_id in group {
                if !consumer_peak_seen.insert(term_id) {
                    return Err(graph_schedule_error("duplicate_owner_term_reference", None, Some("consumer_peak")));
                }
                let term = self.resolve_term(term_id, owner_terms, None, "consumer_peak")?;
                self.validate_consumer_term(term, manifest.execution_scope_fingerprint(), "consumer_peak")?;
                record(term_id, Category::ConsumerPeak);
            }
        }
        let mut consumer_dominated_seen = HashSet::with_capacity(self.consumer_dominated_term_ids.len());
        for term_id in &self.consumer_dominated_term_ids {
            if !consumer_dominated_seen.insert(term_id) {
                return Err(graph_schedule_error("duplicate_owner_term_reference", None, Some("consumer_dominated")));
            }
            let term = self.resolve_term(term_id, owner_terms, None, "consumer_dominated")?;
            self.validate_consumer_term(term, manifest.execution_scope_fingerprint(), "consumer_dominated")?;
            record(term_id, Category::ConsumerDominated);
        }

        for (term_id, categories) in &references {
            let phase_categories = categories
                .iter()
                .filter_map(|category| match category {
                    Category::ProducerPhase { producer_index, group_index } => Some((*producer_index, *group_index)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let side_input_categories = categories
                .iter()
                .filter_map(|category| match category {
                    Category::ConcurrentSideInput { producer_index } => Some(*producer_index),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let retained_categories = categories
                .iter()
                .filter_map(|category| match category {
                    Category::RetainedOccurrence(index) => Some(*index),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let producer_dominated_categories = categories
                .iter()
                .filter_map(|category| match category {
                    Category::ProducerDominated { producer_index } => Some(*producer_index),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let has_fixed = categories.iter().any(|category| matches!(category, Category::Fixed));
            let has_peak = categories.iter().any(|category| matches!(category, Category::ConsumerPeak));
            let has_dominated = categories.iter().any(|category| matches!(category, Category::ConsumerDominated));
            let has_consumer = has_peak || has_dominated;
            if (has_fixed && categories.len() > 1)
                || (has_consumer
                    && (!phase_categories.is_empty()
                        || !retained_categories.is_empty()
                        || !side_input_categories.is_empty()
                        || !producer_dominated_categories.is_empty()))
                || (!side_input_categories.is_empty() && (categories.len() != 1 || side_input_categories.len() != 1))
                || (!producer_dominated_categories.is_empty()
                    && (categories.len() != 1 || producer_dominated_categories.len() != 1))
            {
                return Err(graph_schedule_error("owner_term_reference_category_conflict", None, Some("inventory"))
                    .with_fact("owner_term_role", term_id.role_label())
                    .with_fact("owner_term_occurrence", term_id.occurrence_label()));
            }
            if has_peak && has_dominated {
                return Err(graph_schedule_error("consumer_peak_dominated_overlap", None, Some("consumer_inventory"))
                    .with_fact("owner_term_role", term_id.role_label())
                    .with_fact("owner_term_occurrence", term_id.occurrence_label()));
            }
            if retained_categories.len() > 1 {
                return Err(graph_schedule_error(
                    "duplicate_retained_occurrence_reference",
                    None,
                    Some("retained_output"),
                ));
            }
            if !phase_categories.is_empty() && phase_categories.len() > 1 {
                let producer_indices =
                    phase_categories.iter().map(|(producer_index, _)| *producer_index).collect::<HashSet<_>>();
                let group_indices =
                    phase_categories.iter().map(|(_, group_index)| *group_index).collect::<HashSet<_>>();
                let allowed = producer_indices.len() == 1
                    && group_indices.len() == phase_categories.len()
                    && phase_categories.iter().all(|(producer_index, _)| {
                        self.producer_phases[*producer_index]
                            .alternative_term_ids
                            .iter()
                            .any(|allowed_id| allowed_id == term_id)
                    });
                if !allowed {
                    return Err(graph_schedule_error(
                        "producer_phase_group_duplicate",
                        None,
                        Some("producer_isolated_peak"),
                    )
                    .with_fact("owner_term_role", term_id.role_label())
                    .with_fact("owner_term_occurrence", term_id.occurrence_label()));
                }
            }
            if !phase_categories.is_empty() && !retained_categories.is_empty() {
                let term = self.resolve_term(term_id, owner_terms, None, "producer_isolated_peak")?;
                let nested_retained = retained_categories.first().is_some_and(|retained_index| {
                    phase_categories.iter().all(|(producer_index, _)| {
                        producer_index == retained_index
                            && self.producer_phases[*producer_index].nested_schedule
                            && self.producer_phases[*producer_index]
                                .alternative_term_ids
                                .iter()
                                .any(|allowed_id| allowed_id == term_id)
                    })
                });
                if !term.is_retained_output() || (phase_categories.len() != 1 && !nested_retained) {
                    return Err(graph_schedule_error(
                        "producer_phase_retained_category_conflict",
                        None,
                        Some("producer_isolated_peak"),
                    ));
                }
            }
        }

        if require_totality {
            let mut scheduled_scopes = manifest.manifest_scope_counts().into_keys().collect::<HashSet<_>>();
            scheduled_scopes.extend(
                scopes
                    .iter()
                    .filter(|scope| scope.is_cugraph_side_input_scope())
                    .map(AttemptReservationShape::execution_scope_fingerprint),
            );
            scheduled_scopes.extend(
                self.producer_phases
                    .iter()
                    .flat_map(|phase| phase.concurrent_side_input_term_ids.iter())
                    .map(OwnerTermId::execution_scope_fingerprint),
            );
            scheduled_scopes.extend(self.referenced_scope_fingerprints());
            if let Some(missing) = owner_terms.iter().find(|term| {
                scheduled_scopes.contains(term.execution_scope_fingerprint()) && !references.contains_key(term.id())
            }) {
                return Err(graph_schedule_error("ordered_inventory_incomplete", None, Some("inventory"))
                    .with_fact("ordered_inventory_incomplete", "true")
                    .with_fact("owner_term_role", missing.id().role_label())
                    .with_fact("owner_term_occurrence", missing.id().occurrence_label()));
            }
        }
        Ok(())
    }

    fn validate_concurrent_side_input_term(
        &self,
        term: &AllocationOwnerTerm,
        occurrence: &AttemptReservationProducerOccurrence,
        scopes: &[AttemptReservationShape],
    ) -> Result<()> {
        if !term.is_side_input_frame() {
            return Err(graph_schedule_error(
                "owner_term_role_class_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_side_input"),
            )
            .with_fact("owner_term_role", term.id().role_label()));
        }
        let expected_occurrence = format!("composed:{}", occurrence.occurrence_id);
        if term.id().occurrence_label() != expected_occurrence {
            return Err(graph_schedule_error(
                "owner_term_occurrence_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_side_input"),
            )
            .with_fact("expected_occurrence", expected_occurrence)
            .with_fact("actual_occurrence", term.id().occurrence_label()));
        }
        if !scopes.iter().any(|scope| {
            scope.is_cugraph_side_input_scope()
                && scope.execution_scope_fingerprint() == term.execution_scope_fingerprint()
        }) {
            return Err(graph_schedule_error(
                "side_input_scope_coverage_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_side_input"),
            )
            .with_fact("side_input_scope", term.execution_scope_fingerprint()));
        }
        Ok(())
    }

    fn validate_producer_phase_term(
        &self,
        term: &AllocationOwnerTerm,
        occurrence: &AttemptReservationProducerOccurrence,
        nested_schedule: bool,
        scopes: &[AttemptReservationShape],
    ) -> Result<()> {
        if nested_schedule {
            if !term.id().is_composed_occurrence() {
                return Err(graph_schedule_error(
                    "owner_term_occurrence_mismatch",
                    Some(&occurrence.occurrence_id),
                    Some("nested_producer"),
                ));
            }
            if !scopes.iter().any(|scope| scope.execution_scope_fingerprint() == term.execution_scope_fingerprint()) {
                return Err(graph_schedule_error(
                    "owner_term_scope_mismatch",
                    Some(&occurrence.occurrence_id),
                    Some("nested_producer"),
                )
                .with_fact("actual_scope", term.execution_scope_fingerprint()));
            }
            return Ok(());
        }
        if term.execution_scope_fingerprint() != occurrence.execution_scope_fingerprint {
            return Err(graph_schedule_error(
                "owner_term_scope_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_isolated_peak"),
            )
            .with_fact("expected_scope", occurrence.execution_scope_fingerprint.as_str())
            .with_fact("actual_scope", term.execution_scope_fingerprint()));
        }
        let expected_occurrence = format!("composed:{}", occurrence.occurrence_id);
        if term.id().occurrence_label() != expected_occurrence {
            return Err(graph_schedule_error(
                "owner_term_occurrence_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_isolated_peak"),
            )
            .with_fact("expected_occurrence", expected_occurrence)
            .with_fact("actual_occurrence", term.id().occurrence_label()));
        }
        if !(term.is_graph_phase_term() || term.is_source_stage() || term.is_retained_output()) {
            return Err(graph_schedule_error(
                "owner_term_role_class_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_isolated_peak"),
            )
            .with_fact("owner_term_role", term.id().role_label()));
        }
        Ok(())
    }

    fn validate_consumer_term(&self, term: &AllocationOwnerTerm, scope: &str, phase: &'static str) -> Result<()> {
        if term.execution_scope_fingerprint() != scope {
            return Err(graph_schedule_error("owner_term_scope_mismatch", None, Some(phase))
                .with_fact("expected_scope", scope)
                .with_fact("actual_scope", term.execution_scope_fingerprint()));
        }
        if !term.is_consumer_schedule_owner() {
            return Err(graph_schedule_error("owner_term_role_class_mismatch", None, Some(phase))
                .with_fact("owner", term.owner_label())
                .with_fact("owner_term_role", term.id().role_label()));
        }
        if !term.has_consumer_schedule_identity() {
            return Err(graph_schedule_error("owner_term_occurrence_mismatch", None, Some(phase))
                .with_fact("owner_term_role", term.id().role_label())
                .with_fact("owner_term_occurrence", term.id().occurrence_label()));
        }
        Ok(())
    }

    fn projection(
        &self,
        scopes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
        attempt_id: crate::observability::AttemptId,
        device_ordinal: usize,
        grant_id: u64,
        grant_bytes: u64,
    ) -> Result<AttemptComposedScheduleProjection> {
        // Run the canonical fold first. This validates every required owner
        // contribution and exact scope coverage before any public projection
        // is published.
        let complete_peak = self.checked_fold_with_inventory(scopes, owner_terms)?;
        let fixed = self.fixed_bytes(owner_terms)?;
        let manifest = self.manifest_shape(scopes)?;
        let consumer_scope = manifest.execution_scope_fingerprint().to_owned();
        let consumer_additional_peak_bytes = self.consumer_peak_bytes(owner_terms, &consumer_scope)?;

        let mut retained_prior = 0_u64;
        let mut producer_occurrences = Vec::with_capacity(manifest.producer_occurrences.len());
        for (occurrence, phase) in manifest.producer_occurrences.iter().zip(&self.producer_phases) {
            let isolated_peak_bytes = self.producer_isolated_peak_bytes(occurrence, phase, owner_terms)?;
            let retained_output_bytes = self.producer_retained_output_bytes(occurrence, phase, owner_terms)?;
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

    fn manifest_shape<'a>(&self, scopes: &'a [AttemptReservationShape]) -> Result<&'a AttemptReservationShape> {
        admitted_manifest_shape(scopes, Some(self))
    }

    fn validate_runtime(
        &self,
        manifest: &AttemptReservationShape,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
    ) -> Result<()> {
        let expected_consumer_scope = manifest.execution_scope_fingerprint();
        if expected_consumer_scope != canonical_execution_scope_fingerprint(consumer_plan) {
            return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
        }
        if manifest.producer_occurrences.len() != occurrence_ids.len() {
            return Err(graph_schedule_error("runtime_occurrence_count_mismatch", None, Some("runtime"))
                .with_fact("scheduled_occurrence_count", manifest.producer_occurrences.len())
                .with_fact("runtime_occurrence_count", occurrence_ids.len()));
        }
        for (scheduled, runtime) in manifest.producer_occurrences.iter().zip(occurrence_ids) {
            if scheduled.occurrence_id != *runtime {
                return Err(graph_schedule_error("runtime_occurrence_order_mismatch", Some(runtime), Some("runtime"))
                    .with_fact("scheduled_occurrence_id", scheduled.occurrence_id.as_str()));
            }
        }
        manifest.validate_manifest_core()
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
    output_column_evidence: crate::admission::string_payload::PlanColumnEvidence,
    output_column_evidence_input_scopes: Vec<(String, String)>,
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
    limit_phase_peak_bytes: Option<u64>,
    sort_metadata: Option<crate::admission::requirement::SortMetadataDescriptor>,
    sort_rejection_reason: Option<String>,
    owner_term_rejection_reason: Option<String>,
    producer_occurrences: Vec<AttemptReservationProducerOccurrence>,
    ordered_schedule: Option<AttemptReservationSchedule>,
}

impl AttemptReservationShape {
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            output_column_evidence: crate::admission::string_payload::PlanColumnEvidence::default(),
            output_column_evidence_input_scopes: Vec::new(),
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
            limit_phase_peak_bytes: None,
            sort_metadata: None,
            sort_rejection_reason: None,
            owner_term_rejection_reason: None,
            producer_occurrences: Vec::new(),
            ordered_schedule: None,
        }
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(super) fn output_column_evidence(&self) -> &crate::admission::string_payload::PlanColumnEvidence {
        &self.output_column_evidence
    }

    pub(super) fn set_output_column_evidence(
        &mut self,
        evidence: crate::admission::string_payload::PlanColumnEvidence,
    ) {
        self.output_column_evidence = evidence;
    }

    pub(super) fn set_output_column_evidence_input_scopes(&mut self, scopes: Vec<(String, String)>) {
        self.output_column_evidence_input_scopes = scopes;
    }

    pub(super) fn output_column_evidence_input_scopes(&self) -> &[(String, String)] {
        &self.output_column_evidence_input_scopes
    }

    pub(crate) fn attach_ordered_schedule(&mut self, schedule: AttemptReservationSchedule) -> Result<()> {
        if self.ordered_schedule.is_some() {
            return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule")));
        }
        self.validate_manifest_core()?;
        if schedule.producer_phase_count_internal() != self.producer_occurrences.len() {
            return Err(graph_schedule_error("producer_phase_row_count_mismatch", None, Some("schedule"))
                .with_fact("producer_phase_row_count", schedule.producer_phase_count_internal())
                .with_fact("producer_occurrence_count", self.producer_occurrences.len()));
        }
        self.ordered_schedule = Some(schedule);
        Ok(())
    }

    pub(crate) fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    pub(crate) fn take_ordered_schedule_for_nesting(
        &mut self,
    ) -> Option<(AttemptReservationSchedule, Vec<AttemptReservationProducerOccurrence>)> {
        let schedule = self.ordered_schedule.take()?;
        Some((schedule, std::mem::take(&mut self.producer_occurrences)))
    }

    pub(crate) fn owner_term_rejection_reason(&self) -> Option<&str> {
        self.owner_term_rejection_reason.as_deref()
    }

    pub(crate) fn set_producer_occurrences(
        &mut self,
        occurrences: impl IntoIterator<Item = AttemptReservationProducerOccurrence>,
    ) -> Result<()> {
        if !self.producer_occurrences.is_empty() {
            return Err(graph_schedule_error("duplicate_producer_occurrence_rows", None, Some("manifest")));
        }
        self.producer_occurrences = occurrences.into_iter().collect();
        self.validate_manifest_core()?;
        Ok(())
    }

    pub(crate) fn producer_occurrences(&self) -> &[AttemptReservationProducerOccurrence] {
        &self.producer_occurrences
    }

    pub(crate) fn validate_manifest_core(&self) -> Result<()> {
        let mut occurrence_ids = HashSet::with_capacity(self.producer_occurrences.len());
        for occurrence in &self.producer_occurrences {
            if occurrence.occurrence_id.is_empty() {
                return Err(graph_schedule_error("missing_producer_occurrence_id", None, Some("manifest")));
            }
            if !occurrence_ids.insert(&occurrence.occurrence_id) {
                return Err(graph_schedule_error(
                    "duplicate_producer_occurrence_id",
                    Some(&occurrence.occurrence_id),
                    Some("manifest"),
                ));
            }
            if occurrence.execution_scope_fingerprint.is_empty() {
                return Err(graph_schedule_error(
                    "missing_producer_occurrence_scope",
                    Some(&occurrence.occurrence_id),
                    Some("manifest"),
                ));
            }
        }
        if self.execution_scope_fingerprint.is_empty() {
            return Err(graph_schedule_error("missing_consumer_scope", None, Some("manifest")));
        }
        Ok(())
    }

    fn manifest_scope_counts(&self) -> std::collections::HashMap<&str, usize> {
        let mut counts = std::collections::HashMap::with_capacity(self.producer_occurrences.len() + 1);
        *counts.entry(self.execution_scope_fingerprint.as_str()).or_default() += 1;
        for occurrence in &self.producer_occurrences {
            *counts.entry(occurrence.execution_scope_fingerprint.as_str()).or_default() += 1;
        }
        counts
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

    pub(crate) fn set_limit_phase_peak_bytes(&mut self, peak_bytes: u64) -> Result<()> {
        if self.limit_phase_peak_bytes.replace(peak_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_limit_phase_evidence", None, Some("limit")));
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
                Some(source_stage_bytes) => {
                    graph_phase.bounded_live_graph_peak_with_source_stage_bytes(source_stage_bytes)?
                },
                None => graph_phase.bounded_live_graph_peak_bytes()?,
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
            let limit_expected = self.sort_metadata.as_ref().is_some_and(|metadata| metadata.limit.is_some());
            match (limit_expected, self.limit_phase_peak_bytes.is_some()) {
                (true, false) => {
                    return Err(graph_schedule_error("limit_preflight_unresolved", None, Some("limit")));
                },
                (false, true) => {
                    return Err(graph_schedule_error("unexpected_limit_phase_evidence", None, Some("limit")));
                },
                _ => {},
            }
            let sort_peak = self.sort_phase_peak_bytes;
            let limit_peak = self.limit_phase_peak_bytes;
            let peak = sort_peak
                .into_iter()
                .chain(limit_peak)
                .max()
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
    bound_derivation: Option<AttemptBoundDerivation>,
    owner_terms: Vec<AllocationOwnerTerm>,
}

impl AttemptReservationEnvelope {
    #[cfg(test)]
    pub(crate) fn resolve(
        reservation_shapes: &[AttemptReservationShape],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        Self::resolve_internal(reservation_shapes, None, policy)
    }

    pub(crate) fn resolve_with_owner_terms(
        reservation_shapes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        Self::resolve_internal(reservation_shapes, Some(owner_terms), policy)
    }

    fn resolve_internal(
        reservation_shapes: &[AttemptReservationShape],
        owner_terms: Option<&[AllocationOwnerTerm]>,
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        // Reservation shapes are allocation occurrences, not structural set
        // members.  Equal graph plans may retain two distinct result handles
        // and must remain two entries in the resolved envelope.
        let scopes = reservation_shapes.to_vec();
        let mut ordered_schedule =
            match scopes.iter().filter_map(AttemptReservationShape::ordered_schedule).collect::<Vec<_>>().as_slice() {
                [] => None,
                [schedule] => Some((*schedule).clone()),
                _ => return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule"))),
            };

        let mut required_phase_floor_bytes = 0_u64;
        let mut complete_phase_peak_bytes = Some(0_u64);
        let has_graph_memory_phase = scopes.iter().any(AttemptReservationShape::has_graph_memory_phase);
        let mut scheduled_scope_counts = ordered_schedule
            .as_ref()
            .map(|schedule| {
                let manifest = admitted_manifest_shape(&scopes, Some(schedule))?;
                let mut counts = manifest.manifest_scope_counts();
                for scope in schedule.concurrent_side_input_scope_fingerprints() {
                    *counts.entry(scope).or_default() += 1;
                }
                for scope in schedule.referenced_scope_fingerprints() {
                    counts.entry(scope).or_insert(1);
                }
                Ok(counts)
            })
            .transpose()?;
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

        let mut selected_derivation = None;
        if let Some(schedule) = &ordered_schedule {
            let ordered_result = match owner_terms {
                Some(owner_terms) => schedule.checked_fold_with_inventory(&scopes, owner_terms),
                None => schedule.checked_fold_with_scopes(&scopes),
            };
            match ordered_result {
                Ok(complete_peak) => {
                    let Some(auxiliary_peak) = complete_phase_peak_bytes else {
                        return Err(graph_schedule_error("auxiliary_scope_unmodeled", None, Some("auxiliary_scope")));
                    };
                    required_phase_floor_bytes = required_phase_floor_bytes.max(complete_peak);
                    complete_phase_peak_bytes = Some(auxiliary_peak.max(complete_peak));
                    selected_derivation = Some(AttemptBoundDerivation::OrderedTimeline);
                },
                Err(ordered_error)
                    if owner_terms.is_some()
                        && ordered_error.fact_value("ordered_inventory_incomplete") == Some("true") =>
                {
                    let conservative = checked_owner_sum(owner_terms.expect("owner terms present"), &scopes);
                    match conservative {
                        Ok(owner_sum)
                            if owner_inventory_covers_scopes(
                                owner_terms.expect("owner terms present"),
                                &scopes,
                                owner_sum,
                                required_phase_floor_bytes,
                                false,
                            ) =>
                        {
                            ordered_schedule = None;
                            required_phase_floor_bytes = owner_sum.max(required_phase_floor_bytes);
                            complete_phase_peak_bytes = Some(owner_sum);
                            selected_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
                        },
                        Ok(_) => {
                            return Err(derivation_unavailable_error(
                                ordered_error,
                                owner_derivation_incomplete_error(),
                            ));
                        },
                        Err(conservative_error) => {
                            return Err(derivation_unavailable_error(ordered_error, conservative_error));
                        },
                    }
                },
                Err(ordered_error) if owner_terms.is_some() => {
                    let conservative_error = checked_owner_sum(owner_terms.expect("owner terms present"), &scopes)
                        .err()
                        .unwrap_or_else(owner_derivation_incomplete_error);
                    return Err(derivation_unavailable_error(ordered_error, conservative_error));
                },
                Err(error) => return Err(error),
            }
        } else if let Some(owner_terms) = owner_terms {
            match checked_owner_sum(owner_terms, &scopes) {
                Ok(owner_sum)
                    if owner_inventory_covers_scopes(
                        owner_terms,
                        &scopes,
                        owner_sum,
                        required_phase_floor_bytes,
                        false,
                    ) =>
                {
                    required_phase_floor_bytes = required_phase_floor_bytes.max(owner_sum);
                    complete_phase_peak_bytes = Some(owner_sum);
                    selected_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
                },
                Ok(_) => {},
                Err(error) => return Err(error),
            }
        }

        if ordered_schedule.is_none() && has_graph_memory_phase && complete_phase_peak_bytes.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "a graph attempt contains a runtime scope without a complete bounded phase target",
            )
            .with_fact("reason", GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        }

        Ok(Self {
            policy,
            scopes,
            required_phase_floor_bytes,
            complete_phase_peak_bytes,
            ordered_schedule,
            bound_derivation: selected_derivation,
            owner_terms: owner_terms.map_or_else(Vec::new, ToOwned::to_owned),
        })
    }

    pub(crate) fn required_phase_floor_bytes(&self) -> u64 {
        self.required_phase_floor_bytes
    }

    pub(crate) fn complete_phase_peak_bytes(&self) -> Option<u64> {
        self.complete_phase_peak_bytes
    }

    pub(crate) fn bound_derivation(&self) -> Option<AttemptBoundDerivation> {
        self.bound_derivation
    }

    fn required_grant_bound_bytes(&self) -> Result<u64> {
        match self.bound_derivation {
            Some(AttemptBoundDerivation::OrderedTimeline) => self
                .ordered_schedule
                .as_ref()
                .ok_or_else(|| graph_schedule_error("ordered_schedule_missing", None, Some("bound")))?
                .checked_fold_with_inventory(&self.scopes, &self.owner_terms),
            Some(AttemptBoundDerivation::ConservativeOwnerSum) => self
                .complete_phase_peak_bytes
                .ok_or_else(|| graph_schedule_error("conservative_complete_phase_peak_missing", None, Some("bound"))),
            None => Ok(self.complete_phase_peak_bytes.unwrap_or(self.required_phase_floor_bytes)),
        }
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
        let manifest = admitted_manifest_shape(&self.scopes, self.ordered_schedule.as_ref())?;
        manifest.validate_manifest_core()?;
        if let Some(schedule) = &self.ordered_schedule {
            let is_manifest_consumer = manifest.matches(consumer_plan);
            let passthrough_matches = self
                .scopes
                .iter()
                .filter(|scope| scope.is_attested_graph_result_passthrough_scope() && scope.matches(consumer_plan))
                .count();
            if is_manifest_consumer {
                schedule.validate_runtime(manifest, consumer_plan, occurrence_ids)?;
            } else if passthrough_matches != 1 {
                return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
            }
        } else {
            if manifest.execution_scope_fingerprint() != canonical_execution_scope_fingerprint(consumer_plan) {
                return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
            }
            if manifest.producer_occurrences().len() != occurrence_ids.len() {
                return Err(graph_schedule_error("runtime_occurrence_count_mismatch", None, Some("runtime"))
                    .with_fact("scheduled_occurrence_count", manifest.producer_occurrences().len())
                    .with_fact("runtime_occurrence_count", occurrence_ids.len()));
            }
            for (scheduled, runtime) in manifest.producer_occurrences().iter().zip(occurrence_ids) {
                if scheduled.occurrence_id() != runtime {
                    return Err(graph_schedule_error(
                        "runtime_occurrence_order_mismatch",
                        Some(runtime),
                        Some("runtime"),
                    )
                    .with_fact("scheduled_occurrence_id", scheduled.occurrence_id()));
                }
            }
        }
        let required_peak = self.required_grant_bound_bytes()?;
        if immutable_grant_bytes < required_peak {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                .with_fact("complete_phase_peak_bytes", required_peak));
        }
        Ok(())
    }

    /// Validates the admitted ordered schedule before optionally publishing
    /// its identity-bound projection. Ordinary contexts have no attempt
    /// identity and therefore validate without publishing an artifact.
    pub(crate) fn validate_and_project_composed_execution(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
        immutable_grant_bytes: u64,
        projection_identity: Option<(crate::observability::AttemptId, usize, u64)>,
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        self.validate_composed_execution(consumer_plan, occurrence_ids, immutable_grant_bytes)?;
        let Some((attempt_id, device_ordinal, grant_id)) = projection_identity else {
            return Ok(None);
        };
        self.composed_schedule_projection(attempt_id, device_ordinal, grant_id, immutable_grant_bytes).map(Some)
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
        schedule.projection(&self.scopes, &self.owner_terms, attempt_id, device_ordinal, grant_id, grant_bytes)
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

    pub(crate) fn validate_execution_with_grant(
        &self,
        plan: &QueryPlan,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
        immutable_grant_bytes: u64,
    ) -> Result<()> {
        self.validate_execution(
            plan,
            memory_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        )?;
        let required_bound = self.required_grant_bound_bytes()?;
        if immutable_grant_bytes < required_bound {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                .with_fact("complete_phase_peak_bytes", required_bound));
        }
        Ok(())
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

fn derivation_unavailable_error(ordered: QueryError, conservative: QueryError) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "no complete bounded attempt derivation is available",
    )
    .with_fact("reason", "attempt_bound_derivation_unavailable")
    .with_fact("ordered_reason", ordered.fact_value("schedule_reason").unwrap_or("ordered_timeline_unavailable"))
    .with_fact("conservative_reason", conservative.fact_value("reason").unwrap_or("conservative_owner_sum_unavailable"))
}

fn owner_derivation_incomplete_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "attempt owner-term inventory does not cover every incomplete scope",
    )
    .with_fact("reason", "attempt_owner_term_coverage_incomplete")
    .with_fact("derivation", "conservative_owner_sum")
}

#[cfg(test)]
mod tests;
