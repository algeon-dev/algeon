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
        AllocationOwnerTerm, AttemptBoundDerivation, OwnerId, OwnerTermId, OwnerTermRole, checked_owner_sum,
        owner_inventory_covers_scopes,
    },
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy},
    graph_memory::GraphMemoryAdmissionEvidence,
    materialization_sizing::{
        JoinOutputColumnUpperBoundInput, PartialAggregateShape, partial_aggregate_upper_bound_bytes,
    },
    plan::{
        PlanNode, QueryPlan, canonical_execution_scope_fingerprint, estimated_reachable_local_parquet_source_bytes,
    },
    schedule_projection::{
        AttemptComposedScheduleOccurrence, AttemptComposedScheduleProjection, COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1,
    },
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
#[doc(hidden)]
pub struct AttemptReservationProducerOccurrence {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    retained_output_term_id: OwnerTermId,
}

impl AttemptReservationProducerOccurrence {
    #[doc(hidden)]
    pub fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }

    #[doc(hidden)]
    pub fn retained_output_term_id(&self) -> &OwnerTermId {
        &self.retained_output_term_id
    }

    #[doc(hidden)]
    pub fn from_term_id(
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

    #[doc(hidden)]
    pub fn validate_retained_output_term(&self, term: &AllocationOwnerTerm) -> Result<()> {
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
            .with_fact("expected_scope", crate::admission::log_safe_fingerprint(&self.execution_scope_fingerprint))
            .with_fact(
                "actual_scope",
                crate::admission::log_safe_fingerprint(term.id().execution_scope_fingerprint()),
            ));
        }
        Ok(())
    }
}

/// Optional ordered phase evidence attached to the mandatory occurrence core.
/// A conservative owner-sum derivation intentionally has no phase rows.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationProducerPhase {
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
    #[doc(hidden)]
    pub fn from_term_groups(isolated_peak_term_groups: Vec<Vec<OwnerTermId>>) -> Self {
        Self::from_term_groups_with_alternatives_and_side_inputs(isolated_peak_term_groups, Vec::new(), Vec::new())
    }

    #[doc(hidden)]
    pub fn from_term_groups_with_alternatives_and_side_inputs(
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

    #[doc(hidden)]
    pub fn from_nested_term_groups(
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
#[doc(hidden)]
pub enum AttemptReservationConsumerState {
    Passthrough,
    Bounded,
    DeferredUntilResidentEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationSchedule {
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

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationStageAudit {
    #[doc(hidden)]
    pub stage: &'static str,
    #[doc(hidden)]
    pub occurrence_id: Option<String>,
    #[doc(hidden)]
    pub fixed_bytes: u64,
    #[doc(hidden)]
    pub retained_prior_bytes: u64,
    #[doc(hidden)]
    pub active_peak_bytes: u64,
    #[doc(hidden)]
    pub stage_total_bytes: u64,
    #[doc(hidden)]
    pub retained_output_bytes: Option<u64>,
    #[doc(hidden)]
    pub dominant_owner: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationFoldAudit {
    #[doc(hidden)]
    pub complete_peak_bytes: u64,
    #[doc(hidden)]
    pub stages: Vec<AttemptReservationStageAudit>,
}

impl AttemptReservationFoldAudit {
    #[doc(hidden)]
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.stages.len() + 1);
        lines.push(format!("schedule_complete_peak_bytes={}", self.complete_peak_bytes));
        lines.extend(self.stages.iter().map(|stage| {
            format!(
                "stage={},occurrence={},fixed_bytes={},retained_prior_bytes={},active_peak_bytes={},stage_total_bytes={},retained_output_bytes={},dominant_owner={}",
                stage.stage,
                stage.occurrence_id.as_deref().unwrap_or("none"),
                stage.fixed_bytes,
                stage.retained_prior_bytes,
                stage.active_peak_bytes,
                stage.stage_total_bytes,
                stage.retained_output_bytes.map_or_else(|| "none".to_owned(), |bytes| bytes.to_string()),
                stage.dominant_owner.as_deref().unwrap_or("none"),
            )
        }));
        lines
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SchedulePeak {
    bytes: u64,
    term_ids: Vec<OwnerTermId>,
}

#[doc(hidden)]
pub type NestedProducerPeakTerms = (Vec<Vec<OwnerTermId>>, Vec<OwnerTermId>, Vec<OwnerTermId>);

impl AttemptReservationSchedule {
    #[doc(hidden)]
    pub fn resolve_deferred_consumer(
        &mut self,
        consumer_state: AttemptReservationConsumerState,
        consumer_peak_term_groups: Vec<Vec<OwnerTermId>>,
        consumer_dominated_term_ids: Vec<OwnerTermId>,
    ) -> Result<()> {
        if self.consumer_state != AttemptReservationConsumerState::DeferredUntilResidentEvidence {
            return Err(graph_schedule_error("consumer_phase_not_deferred", None, Some("consumer_peak")));
        }
        if consumer_state == AttemptReservationConsumerState::DeferredUntilResidentEvidence {
            return Err(graph_schedule_error("consumer_phase_remains_deferred", None, Some("consumer_peak")));
        }
        self.consumer_state = consumer_state;
        self.consumer_peak_term_groups = consumer_peak_term_groups;
        self.consumer_dominated_term_ids = consumer_dominated_term_ids;
        Ok(())
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn from_phase_term_ids(
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

    #[doc(hidden)]
    pub fn from_phase_term_groups_with_dominated(
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
    #[doc(hidden)]
    pub fn new(
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
    #[doc(hidden)]
    pub fn phase_group_counts_for_test(&self) -> (usize, usize) {
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

    fn referenced_composed_occurrence_scope_counts(&self) -> HashMap<&str, usize> {
        let mut occurrences = HashMap::<&str, HashSet<String>>::new();
        for term_id in self
            .attempt_wide_fixed_term_ids
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
            .filter(|term_id| term_id.is_composed_occurrence())
        {
            occurrences.entry(term_id.execution_scope_fingerprint()).or_default().insert(term_id.occurrence_label());
        }
        occurrences.into_iter().map(|(scope, occurrences)| (scope, occurrences.len())).collect()
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn attempt_wide_fixed_term_ids(&self) -> &[OwnerTermId] {
        &self.attempt_wide_fixed_term_ids
    }

    #[doc(hidden)]
    pub fn consumer_state(&self) -> AttemptReservationConsumerState {
        self.consumer_state
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn consumer_dominated_term_ids(&self) -> &[OwnerTermId] {
        &self.consumer_dominated_term_ids
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn consumer_peak_term_groups(&self) -> &[Vec<OwnerTermId>] {
        &self.consumer_peak_term_groups
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn consumer_peak_bytes_for_test(&self, owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
        self.consumer_peak(owner_terms, "test").map(|peak| peak.bytes)
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn producer_phase_term_ids(&self) -> Vec<OwnerTermId> {
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

    #[doc(hidden)]
    pub fn nested_producer_peak_terms(
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
            AttemptReservationConsumerState::DeferredUntilResidentEvidence => groups.push(retained_prior),
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

    #[doc(hidden)]
    pub fn checked_fold_with_inventory(
        &self,
        scopes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<u64> {
        self.fold_audit_with_inventory(scopes, owner_terms).map(|audit| audit.complete_peak_bytes)
    }

    #[doc(hidden)]
    pub fn fold_audit_with_inventory(
        &self,
        scopes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<AttemptReservationFoldAudit> {
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
        for (scope, referenced) in self.referenced_composed_occurrence_scope_counts() {
            let count = expected_scope_counts.entry(scope).or_default();
            *count = (*count).max(referenced);
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
        let consumer_peak = match self.consumer_peak(owner_terms, manifest.execution_scope_fingerprint()) {
            Ok(peak) => peak,
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
        let mut retained_prior_term_ids = Vec::new();
        let mut complete_peak = 0_u64;
        let mut stages = Vec::with_capacity(self.producer_phases.len() + 1);
        for (occurrence, phase) in manifest.producer_occurrences.iter().zip(&self.producer_phases) {
            let isolated_peak = match self.producer_isolated_peak(occurrence, phase, owner_terms) {
                Ok(peak) => peak,
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
                    let error = graph_schedule_error(
                        "producer_stage_fixed_plus_retained_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("producer_stage_fixed_plus_retained"),
                    )
                    .with_fact("fixed_bytes", fixed)
                    .with_fact("retained_prior_bytes", retained_prior);
                    self.with_dominant_owner_facts(
                        error,
                        owner_terms,
                        self.attempt_wide_fixed_term_ids.iter().chain(&retained_prior_term_ids),
                    )
                })?
                .checked_add(isolated_peak.bytes)
                .ok_or_else(|| {
                    let error = graph_schedule_error(
                        "producer_stage_isolated_peak_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("producer_stage_isolated_peak"),
                    )
                    .with_fact("fixed_bytes", fixed)
                    .with_fact("retained_prior_bytes", retained_prior)
                    .with_fact("isolated_peak_bytes", isolated_peak.bytes);
                    self.with_dominant_owner_facts(
                        error,
                        owner_terms,
                        self.attempt_wide_fixed_term_ids
                            .iter()
                            .chain(&retained_prior_term_ids)
                            .chain(&isolated_peak.term_ids),
                    )
                })?;
            complete_peak = complete_peak.max(producer_stage);
            stages.push(AttemptReservationStageAudit {
                stage: "producer",
                occurrence_id: Some(occurrence.occurrence_id.clone()),
                fixed_bytes: fixed,
                retained_prior_bytes: retained_prior,
                active_peak_bytes: isolated_peak.bytes,
                stage_total_bytes: producer_stage,
                retained_output_bytes: Some(retained_output),
                dominant_owner: self
                    .dominant_owner(
                        owner_terms,
                        self.attempt_wide_fixed_term_ids
                            .iter()
                            .chain(&retained_prior_term_ids)
                            .chain(&isolated_peak.term_ids),
                    )
                    .map(AllocationOwnerTerm::audit_line),
            });
            retained_prior = retained_prior.checked_add(retained_output).ok_or_else(|| {
                occurrence.retained_output_term_id.with_error_facts(
                    graph_schedule_error(
                        "retained_output_accumulation_overflow",
                        Some(&occurrence.occurrence_id),
                        Some("retained_output_accumulation"),
                    )
                    .with_fact("retained_prior_bytes", retained_prior)
                    .with_fact("retained_output_bytes", retained_output),
                )
            })?;
            retained_prior_term_ids.push(occurrence.retained_output_term_id.clone());
        }

        let consumer_stage = fixed
            .checked_add(retained_prior)
            .ok_or_else(|| {
                let error = graph_schedule_error(
                    "consumer_stage_fixed_plus_retained_overflow",
                    None,
                    Some("consumer_stage_fixed_plus_retained"),
                )
                .with_fact("fixed_bytes", fixed)
                .with_fact("retained_prior_bytes", retained_prior);
                self.with_dominant_owner_facts(
                    error,
                    owner_terms,
                    self.attempt_wide_fixed_term_ids.iter().chain(&retained_prior_term_ids),
                )
            })?
            .checked_add(consumer_peak.bytes)
            .ok_or_else(|| {
                let error = graph_schedule_error(
                    "consumer_stage_additional_peak_overflow",
                    None,
                    Some("consumer_stage_additional_peak"),
                )
                .with_fact("fixed_bytes", fixed)
                .with_fact("retained_prior_bytes", retained_prior)
                .with_fact("consumer_peak_bytes", consumer_peak.bytes);
                self.with_dominant_owner_facts(
                    error,
                    owner_terms,
                    self.attempt_wide_fixed_term_ids
                        .iter()
                        .chain(&retained_prior_term_ids)
                        .chain(&consumer_peak.term_ids),
                )
            })?;
        stages.push(AttemptReservationStageAudit {
            stage: "consumer",
            occurrence_id: None,
            fixed_bytes: fixed,
            retained_prior_bytes: retained_prior,
            active_peak_bytes: consumer_peak.bytes,
            stage_total_bytes: consumer_stage,
            retained_output_bytes: None,
            dominant_owner: self
                .dominant_owner(
                    owner_terms,
                    self.attempt_wide_fixed_term_ids
                        .iter()
                        .chain(&retained_prior_term_ids)
                        .chain(&consumer_peak.term_ids),
                )
                .map(AllocationOwnerTerm::audit_line),
        });
        Ok(AttemptReservationFoldAudit { complete_peak_bytes: complete_peak.max(consumer_stage), stages })
    }

    fn fixed_bytes(&self, owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
        if !self.attempt_wide_fixed_term_ids.is_empty() {
            let mut seen = HashSet::with_capacity(self.attempt_wide_fixed_term_ids.len());
            if let Some(duplicate) = self.attempt_wide_fixed_term_ids.iter().find(|id| !seen.insert(*id)) {
                return Err(duplicate.with_error_facts(graph_schedule_error(
                    "duplicate_owner_term_reference",
                    None,
                    Some("attempt_wide_fixed"),
                )));
            }
            return self.attempt_wide_fixed_term_ids.iter().try_fold(0_u64, |total, term_id| {
                let term = self.resolve_term(term_id, owner_terms, None, "attempt_wide_fixed")?;
                if !term.is_attempt_wide_fixed() {
                    return Err(term.with_error_facts(graph_schedule_error(
                        "owner_term_role_class_mismatch",
                        None,
                        Some("attempt_wide_fixed"),
                    )));
                }
                let bytes = term.bytes();
                total.checked_add(bytes).ok_or_else(|| {
                    term.with_error_facts(
                        graph_schedule_error("attempt_wide_fixed_sum_overflow", None, Some("owner_term_sum"))
                            .with_fact("lhs_bytes", total)
                            .with_fact("rhs_bytes", bytes),
                    )
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

    fn producer_isolated_peak(
        &self,
        occurrence: &AttemptReservationProducerOccurrence,
        phase: &AttemptReservationProducerPhase,
        owner_terms: &[AllocationOwnerTerm],
    ) -> Result<SchedulePeak> {
        if !phase.isolated_peak_term_groups.is_empty() {
            let mut isolated_peak = SchedulePeak { bytes: 0, term_ids: Vec::new() };
            for group in &phase.isolated_peak_term_groups {
                let mut seen = HashSet::with_capacity(group.len());
                if let Some(duplicate) = group.iter().find(|id| !seen.insert(*id)) {
                    return Err(duplicate.with_error_facts(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_isolated_peak"),
                    )));
                }
                let phase_peak = self.sum_term_ids(
                    group,
                    owner_terms,
                    Some(&occurrence.occurrence_id),
                    "producer_stage_isolated_peak_overflow",
                )?;
                if isolated_peak.term_ids.is_empty() || phase_peak > isolated_peak.bytes {
                    isolated_peak = SchedulePeak { bytes: phase_peak, term_ids: group.clone() };
                }
            }
            return self.add_concurrent_side_input_peak(occurrence, phase, owner_terms, isolated_peak);
        }
        #[cfg(test)]
        {
            let isolated_peak = phase.isolated_peak_bytes.ok_or_else(|| {
                graph_schedule_error("missing_producer_isolated_peak", Some(&occurrence.occurrence_id), None)
            })?;
            self.add_concurrent_side_input_peak(
                occurrence,
                phase,
                owner_terms,
                SchedulePeak { bytes: isolated_peak, term_ids: Vec::new() },
            )
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
        mut isolated_peak: SchedulePeak,
    ) -> Result<SchedulePeak> {
        if phase.concurrent_side_input_term_ids.is_empty() {
            return Ok(isolated_peak);
        }
        let side_input_peak = self.sum_term_ids(
            &phase.concurrent_side_input_term_ids,
            owner_terms,
            Some(&occurrence.occurrence_id),
            "producer_stage_side_input_overflow",
        )?;
        isolated_peak.bytes = isolated_peak.bytes.checked_add(side_input_peak).ok_or_else(|| {
            let error = graph_schedule_error(
                "producer_stage_side_input_overflow",
                Some(&occurrence.occurrence_id),
                Some("producer_stage_side_input"),
            )
            .with_fact("isolated_peak_bytes", isolated_peak.bytes)
            .with_fact("side_input_peak_bytes", side_input_peak);
            self.with_dominant_owner_facts(
                error,
                owner_terms,
                isolated_peak.term_ids.iter().chain(&phase.concurrent_side_input_term_ids),
            )
        })?;
        isolated_peak.term_ids.extend(phase.concurrent_side_input_term_ids.iter().cloned());
        Ok(isolated_peak)
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
                occurrence.retained_output_term_id.with_error_facts(graph_schedule_error(
                    "missing_owner_term_reference",
                    Some(&occurrence.occurrence_id),
                    Some("retained_output"),
                ))
            })?;
        occurrence.validate_retained_output_term(term)?;
        Ok(term.bytes())
    }

    fn consumer_peak(&self, owner_terms: &[AllocationOwnerTerm], _consumer_scope: &str) -> Result<SchedulePeak> {
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
                Ok(SchedulePeak { bytes: 0, term_ids: Vec::new() })
            },
            AttemptReservationConsumerState::DeferredUntilResidentEvidence => {
                if !self.consumer_peak_term_groups.is_empty() || !self.consumer_dominated_term_ids.is_empty() {
                    return Err(graph_schedule_error(
                        "deferred_consumer_has_static_peak_terms",
                        None,
                        Some("consumer_peak"),
                    ));
                }
                Ok(SchedulePeak { bytes: 0, term_ids: Vec::new() })
            },
            AttemptReservationConsumerState::Bounded => {
                if !self.consumer_peak_term_groups.is_empty() {
                    let mut consumer_peak = SchedulePeak { bytes: 0, term_ids: Vec::new() };
                    for group in &self.consumer_peak_term_groups {
                        let mut seen = HashSet::with_capacity(group.len());
                        if let Some(duplicate) = group.iter().find(|id| !seen.insert(*id)) {
                            return Err(duplicate.with_error_facts(graph_schedule_error(
                                "duplicate_owner_term_reference",
                                None,
                                Some("consumer_peak"),
                            )));
                        }
                        let group_peak =
                            self.sum_term_ids(group, owner_terms, None, "consumer_stage_additional_peak_overflow")?;
                        if consumer_peak.term_ids.is_empty() || group_peak > consumer_peak.bytes {
                            consumer_peak = SchedulePeak { bytes: group_peak, term_ids: group.clone() };
                        }
                    }
                    return Ok(consumer_peak);
                }
                #[cfg(test)]
                if let Some(bytes) = self.test_consumer_additional_peak_bytes {
                    return Ok(SchedulePeak { bytes, term_ids: Vec::new() });
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
            let term = self.resolve_term(term_id, owner_terms, occurrence_id, "owner_term")?;
            let bytes = term.bytes();
            total.checked_add(bytes).ok_or_else(|| {
                term.with_error_facts(
                    graph_schedule_error(overflow_reason, occurrence_id, Some("owner_term_sum"))
                        .with_fact("lhs_bytes", total)
                        .with_fact("rhs_bytes", bytes),
                )
            })
        })
    }

    fn dominant_owner<'a, 'b>(
        &self,
        owner_terms: &'a [AllocationOwnerTerm],
        term_ids: impl IntoIterator<Item = &'b OwnerTermId>,
    ) -> Option<&'a AllocationOwnerTerm> {
        let mut dominant = None::<&AllocationOwnerTerm>;
        for term_id in term_ids {
            let Some(term) = owner_terms.iter().find(|term| term.id() == term_id) else {
                continue;
            };
            if dominant.is_none_or(|current| term.bytes() > current.bytes()) {
                dominant = Some(term);
            }
        }
        dominant
    }

    fn with_dominant_owner_facts<'a>(
        &self,
        error: QueryError,
        owner_terms: &[AllocationOwnerTerm],
        term_ids: impl IntoIterator<Item = &'a OwnerTermId>,
    ) -> QueryError {
        match self.dominant_owner(owner_terms, term_ids) {
            Some(term) => term.with_error_facts(error.with_fact("dominant_owner", term.id().owner.diagnostic_label())),
            None => error,
        }
    }

    fn resolve_term<'a>(
        &self,
        term_id: &OwnerTermId,
        owner_terms: &'a [AllocationOwnerTerm],
        occurrence_id: Option<&str>,
        phase: &'static str,
    ) -> Result<&'a AllocationOwnerTerm> {
        owner_terms.iter().find(|term| term.id() == term_id).ok_or_else(|| {
            term_id.with_error_facts(graph_schedule_error("missing_owner_term_reference", occurrence_id, Some(phase)))
        })
    }

    #[doc(hidden)]
    pub fn validate_term_references(
        &self,
        manifest: &AttemptReservationShape,
        owner_terms: &[AllocationOwnerTerm],
        scopes: &[AttemptReservationShape],
    ) -> Result<()> {
        self.validate_term_references_with_totality(manifest, owner_terms, scopes, true)
    }

    #[doc(hidden)]
    pub fn validate_term_references_for_conservative_sum(
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
                return Err(term.with_error_facts(graph_schedule_error(
                    "duplicate_owner_term_inventory",
                    None,
                    Some("inventory"),
                )));
            }
        }
        let mut record = |term_id: &OwnerTermId, category: Category| {
            references.entry(term_id.clone()).or_default().push(category);
        };

        let mut fixed_seen = HashSet::with_capacity(self.attempt_wide_fixed_term_ids.len());
        for term_id in &self.attempt_wide_fixed_term_ids {
            if !fixed_seen.insert(term_id) {
                return Err(term_id.with_error_facts(graph_schedule_error(
                    "duplicate_owner_term_reference",
                    None,
                    Some("attempt_wide_fixed"),
                )));
            }
            let term = self.resolve_term(term_id, owner_terms, None, "attempt_wide_fixed")?;
            if !term.is_attempt_wide_fixed() {
                return Err(term.with_error_facts(graph_schedule_error(
                    "owner_term_role_class_mismatch",
                    None,
                    Some("attempt_wide_fixed"),
                )));
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
                    return Err(term_id.with_error_facts(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_side_input"),
                    )));
                }
                let term =
                    self.resolve_term(term_id, owner_terms, Some(&occurrence.occurrence_id), "producer_side_input")?;
                self.validate_concurrent_side_input_term(term, occurrence, scopes)?;
                record(term_id, Category::ConcurrentSideInput { producer_index: phase_index });
            }
            let mut alternatives_seen = HashSet::with_capacity(phase.alternative_term_ids.len());
            for term_id in &phase.alternative_term_ids {
                if !alternatives_seen.insert(term_id) {
                    return Err(term_id.with_error_facts(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_alternative_phase"),
                    )));
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
                        return Err(term_id.with_error_facts(graph_schedule_error(
                            "duplicate_owner_term_reference",
                            Some(&occurrence.occurrence_id),
                            Some("producer_isolated_peak"),
                        )));
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
                    return Err(term_id.with_error_facts(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        Some(&occurrence.occurrence_id),
                        Some("producer_dominated"),
                    )));
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
                    return Err(term_id.with_error_facts(graph_schedule_error(
                        "duplicate_owner_term_reference",
                        None,
                        Some("consumer_peak"),
                    )));
                }
                let term = self.resolve_term(term_id, owner_terms, None, "consumer_peak")?;
                self.validate_consumer_term(term, manifest.execution_scope_fingerprint(), "consumer_peak")?;
                record(term_id, Category::ConsumerPeak);
            }
        }
        let mut consumer_dominated_seen = HashSet::with_capacity(self.consumer_dominated_term_ids.len());
        for term_id in &self.consumer_dominated_term_ids {
            if !consumer_dominated_seen.insert(term_id) {
                return Err(term_id.with_error_facts(graph_schedule_error(
                    "duplicate_owner_term_reference",
                    None,
                    Some("consumer_dominated"),
                )));
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
                    .filter(|scope| scope.is_resident_side_input_scope())
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
            scope.is_resident_side_input_scope()
                && scope.execution_scope_fingerprint() == term.execution_scope_fingerprint()
        }) {
            return Err(graph_schedule_error(
                "side_input_scope_coverage_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_side_input"),
            )
            .with_fact(
                "side_input_scope",
                crate::admission::log_safe_fingerprint(term.execution_scope_fingerprint()),
            ));
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
                .with_fact(
                    "actual_scope",
                    crate::admission::log_safe_fingerprint(term.execution_scope_fingerprint()),
                ));
            }
            return Ok(());
        }
        if term.execution_scope_fingerprint() != occurrence.execution_scope_fingerprint {
            return Err(graph_schedule_error(
                "owner_term_scope_mismatch",
                Some(&occurrence.occurrence_id),
                Some("producer_isolated_peak"),
            )
            .with_fact(
                "expected_scope",
                crate::admission::log_safe_fingerprint(&occurrence.execution_scope_fingerprint),
            )
            .with_fact("actual_scope", crate::admission::log_safe_fingerprint(term.execution_scope_fingerprint())));
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
                .with_fact("expected_scope", crate::admission::log_safe_fingerprint(scope))
                .with_fact(
                    "actual_scope",
                    crate::admission::log_safe_fingerprint(term.execution_scope_fingerprint()),
                ));
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
        attempt_id: crate::schedule_projection::AttemptId,
        device_ordinal: usize,
        grant_id: u64,
        grant_bytes: u64,
    ) -> Result<AttemptComposedScheduleProjection> {
        // Run the canonical fold first. This validates every required owner
        // contribution and exact scope coverage before any public projection
        // is published.
        let fold = self.fold_audit_with_inventory(scopes, owner_terms)?;
        let complete_peak = fold.complete_peak_bytes;
        let manifest = self.manifest_shape(scopes)?;
        let consumer_scope = manifest.execution_scope_fingerprint().to_owned();
        let (consumer_stage, producer_stages) =
            fold.stages.split_last().expect("canonical schedule fold always emits a consumer stage");
        let fixed = consumer_stage.fixed_bytes;

        let mut producer_occurrences = Vec::with_capacity(manifest.producer_occurrences.len());
        for (occurrence, stage) in manifest.producer_occurrences.iter().zip(producer_stages) {
            let retained_output_bytes =
                stage.retained_output_bytes.expect("producer schedule audit always carries its retained output");
            producer_occurrences.push(AttemptComposedScheduleOccurrence {
                occurrence_id: occurrence.occurrence_id.clone(),
                execution_scope_fingerprint: occurrence.execution_scope_fingerprint.clone(),
                isolated_peak_bytes: stage.active_peak_bytes,
                retained_output_bytes,
                stage_peak_bytes: stage.stage_total_bytes,
            });
        }
        let consumer_additional_peak_bytes = consumer_stage.active_peak_bytes;
        let consumer_stage_peak_bytes = consumer_stage.stage_total_bytes;
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

include!("shape.rs");
include!("envelope.rs");

#[cfg(test)]
mod tests;
