//! The versioned run artifact and its atomic, redaction-checked writer.
//!
//! The measured-run sections are only constructed by the measured run, so the
//! CPU-only feature reaches them from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::Value;

use super::{
    REDACTION_POLICY, RUN_ARTIFACT_SCHEMA, RUN_ARTIFACT_VERSION,
    aggregate::{
        ClauseStatus, DomainExhaustionEvidence, GraphClaimCalibrationEvidence, MeasuredAggregate,
        PhysicalAllocatorPressureEvidence, QuarantineCyclesEvidence, StepAggregate,
    },
    cache_calibration::{CacheCalibrationEvidence, CacheSameKeyProbeEvidence},
    metrics_audit::MetricsAudit,
    operability::{AttributionCompleteness, FeatureOffCugraphEvidence, OtlpExportEvidence},
    preflight::FixturePreflightArtifact,
    profile::{QpsEnvelopeProfile, RunProfile},
    provenance::Provenance,
    redact::ensure_redacted_json,
    schedule::ArrivalPlan,
    timeseries::{
        DeviceSeriesRow, DriftSummary, ObservationPullSummary, OrdinalUtilizationSummary, ProcessSeriesRow, SampleGap,
        VramCalibrationEvidence,
    },
    util::write_new_artifact,
    workload::SelectedQueryDescriptor,
};
use crate::ToolResult;

#[derive(Debug, Serialize)]
pub(super) struct RunArtifact {
    pub(super) schema: &'static str,
    pub(super) version: u32,
    pub(super) status: &'static str,
    pub(super) created_unix_ms: u128,
    pub(super) profile: ArtifactProfile,
    pub(super) execution: ValidationExecution,
    pub(super) provenance: Provenance,
    pub(super) redaction: RedactionEvidence,
}

impl RunArtifact {
    pub(super) fn dry_run(profile: RunProfile, config_sha256: String, provenance: Provenance) -> Self {
        Self::new(profile, config_sha256, provenance, "dry_run", "not_applicable_p6_4", Vec::new())
    }

    pub(super) fn fixture_preflight(
        profile: RunProfile,
        config_sha256: String,
        fixtures: Vec<FixturePreflightArtifact>,
        provenance: Provenance,
    ) -> Self {
        Self::new(profile, config_sha256, provenance, "fixture_preflight", "complete", fixtures)
    }

    /// A completed measured run. `status` distinguishes a run whose scenario
    /// clauses all held from one that recorded a failure, so a non-qualifying
    /// result can never read as a pass.
    pub(super) fn measured_run(
        profile: RunProfile,
        config_sha256: String,
        fixtures: Vec<FixturePreflightArtifact>,
        provenance: Provenance,
        mut measurement: MeasuredRunArtifact,
    ) -> Self {
        let require_overload_outcome = profile.scenario_requirements().require_overload_outcome;
        let composed_schedules = measurement.composed_schedules.clone();
        let failures = qualification_failures(&measurement, &provenance, &composed_schedules, require_overload_outcome);
        if !failures.is_empty() {
            measurement.qualifying = false;
        }
        let qualifying = measurement.qualifying;
        let status = if qualifying { "measured_run" } else { "measured_run_non_qualifying" };
        let mut artifact = Self::new(profile, config_sha256, provenance, status, "complete", fixtures);
        artifact.execution.workload_execution = "complete";
        artifact.execution.attempted_workloads = measurement.selected_queries.len() as u64;
        artifact.execution.attempted_backend_attempts = measurement.aggregate.attempts.len() as u64;
        artifact.execution.qualification_failures = failures;
        artifact.execution.composed_schedules = composed_schedules;
        artifact.execution.measurement = Some(measurement);
        artifact
    }

    /// Publishes the two directly comparable halves of a paired cache
    /// calibration. The caller derives the disabled half from the opt-in
    /// profile, so the artifact's single profile remains the complete
    /// description of their shared workload and controller contract.
    pub(super) fn cache_calibration(
        profile: RunProfile,
        config_sha256: String,
        fixtures: Vec<FixturePreflightArtifact>,
        provenance: Provenance,
        cache_disabled: MeasuredRunArtifact,
        cache_opt_in: MeasuredRunArtifact,
    ) -> Self {
        let require_overload_outcome = profile.scenario_requirements().require_overload_outcome;
        let mut composed_schedules = cache_disabled.composed_schedules.clone();
        composed_schedules.extend(cache_opt_in.composed_schedules.iter().cloned());
        let evidence = CacheCalibrationEvidence::new(cache_disabled, cache_opt_in, profile.cache.stress.clone());
        let status = if evidence.qualifies() { "cache_calibration" } else { "cache_calibration_non_qualifying" };
        let attempted_backend_attempts =
            evidence.cache_disabled.aggregate.attempts.len() + evidence.cache_opt_in.aggregate.attempts.len();
        let attempted_workloads = evidence.cache_opt_in.selected_queries.len() as u64;
        let mut failures = qualification_failures(
            &evidence.cache_disabled,
            &provenance,
            &evidence.cache_disabled.composed_schedules,
            require_overload_outcome,
        );
        failures.extend(qualification_failures(
            &evidence.cache_opt_in,
            &provenance,
            &evidence.cache_opt_in.composed_schedules,
            require_overload_outcome,
        ));
        let status = if failures.is_empty() { status } else { "cache_calibration_non_qualifying" };
        let mut artifact = Self::new(profile, config_sha256, provenance, status, "complete", fixtures);
        artifact.execution.workload_execution = "complete";
        artifact.execution.attempted_workloads = attempted_workloads;
        artifact.execution.attempted_backend_attempts = attempted_backend_attempts as u64;
        artifact.execution.qualification_failures = failures;
        artifact.execution.composed_schedules = composed_schedules;
        artifact.execution.cache_calibration = Some(evidence);
        artifact
    }

    fn new(
        profile: RunProfile,
        config_sha256: String,
        provenance: Provenance,
        status: &'static str,
        fixture_preflight: &'static str,
        fixtures: Vec<FixturePreflightArtifact>,
    ) -> Self {
        Self {
            schema: RUN_ARTIFACT_SCHEMA,
            version: RUN_ARTIFACT_VERSION,
            status,
            created_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_millis()),
            profile: ArtifactProfile { profile, sha256: config_sha256 },
            execution: ValidationExecution {
                fixture_preflight,
                workload_execution: "not_applicable_p6_5",
                selected_workloads: fixtures.len() as u64,
                attempted_workloads: 0,
                attempted_backend_attempts: 0,
                fixtures,
                qualification_failures: Vec::new(),
                composed_schedules: Vec::new(),
                measurement: None,
                cache_calibration: None,
            },
            provenance,
            redaction: RedactionEvidence {
                policy: REDACTION_POLICY,
                raw_config_persisted: false,
                sensitive_keys_rejected: true,
                raw_sql_rejected: true,
                raw_diffs_rejected: true,
                unbounded_errors_persisted: false,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ArtifactProfile {
    #[serde(flatten)]
    pub(super) profile: RunProfile,
    pub(super) sha256: String,
}

#[derive(Debug, Serialize)]
pub(super) struct ValidationExecution {
    pub(super) fixture_preflight: &'static str,
    pub(super) workload_execution: &'static str,
    pub(super) selected_workloads: u64,
    pub(super) attempted_workloads: u64,
    pub(super) attempted_backend_attempts: u64,
    pub(super) fixtures: Vec<FixturePreflightArtifact>,
    /// Machine-readable, closed-vocabulary reasons why composed evidence
    /// cannot qualify. Empty is the only qualifying composed state.
    pub(super) qualification_failures: Vec<QualificationFailure>,
    /// Independent versioned evidence for every composed graph attempt the
    /// runner observed. Each row originates in the sealed engine report.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) composed_schedules: Vec<ComposedScheduleArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) measurement: Option<MeasuredRunArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_calibration: Option<CacheCalibrationEvidence>,
}

#[derive(Debug, Serialize)]
pub(super) struct MeasuredRunArtifact {
    /// False when any scenario clause failed. `not_evidenced` clauses are
    /// declared gaps and leave a run qualifying.
    pub(super) qualifying: bool,
    pub(super) backend: BackendEvidence,
    pub(super) selected_queries: Vec<SelectedQueryDescriptor>,
    pub(super) arrival_plan: ArrivalPlanEvidence,
    pub(super) cache_state: &'static str,
    /// Engine-owned composed schedules collected from sealed reports. They are
    /// rendered once at the outer artifact boundary, rather than duplicated
    /// in the measured payload.
    #[serde(skip)]
    pub(super) composed_schedules: Vec<ComposedScheduleArtifact>,
    /// Dedicated exact-key source/BFS cohorts, when the opt-in profile
    /// asked the runner to establish cache-singleflight evidence separately
    /// from the throughput window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_same_key_probe: Option<CacheSameKeyProbeEvidence>,
    /// Profile-owned QPS envelope contract. It joins the engine-owned
    /// queue high-watermark with sealed attempt throughput rather than
    /// treating the runner's submission loop as queue evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) qps_envelope: Option<QpsEnvelopeEvidence>,
    /// Graph-model calibration gate. It retains typed per-attempt
    /// model/request/budget/peak rows and never derives an algorithm peak from
    /// device-wide telemetry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationEvidence>,
    pub(super) aggregate: MeasuredAggregate,
    /// The shared measured-window deadline and the observed time required to
    /// seal every submitted report. A cooperative deadline cannot preempt a
    /// CUDA call already in flight, so an overrun is retained as explicit
    /// non-qualifying evidence rather than hidden with a task abort.
    pub(super) deadline: RunDeadlineEvidence,
    /// Present only when the caller explicitly requested the public attempt
    /// cancellation barrier. Its result is retained independently from a
    /// deadline so a run cannot mistake timeout for caller cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cancellation_barrier: Option<CancellationBarrierEvidence>,
    /// Present only for the controlled after-admission deadline probe.
    /// It distinguishes a typed deadline at a native safe boundary from
    /// queue expiry or explicit caller cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) deadline_barrier: Option<DeadlineBarrierEvidence>,
    /// Present only for the ordinary domain-cap exhaustion probe. This
    /// is deliberately distinct from a physical allocator failure: the sealed
    /// allocator event must name `reservation_limit` with exact attempt
    /// provenance before the run can qualify.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) domain_exhaustion: Option<DomainExhaustionEvidence>,
    /// Present only for the real external allocator-pressure probe. The
    /// predicate is fail-closed: it requires one sealed exact physical event,
    /// immutable logical caps above its request, verified pressure release,
    /// same-device recovery, and a quiescent backend close.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) physical_allocator_pressure: Option<PhysicalAllocatorPressureEvidence>,
    /// Present only for the repeated-quarantine probe. Each cycle reuses
    /// the controlled-quarantine barrier semantics against a rotation target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) quarantine_cycles: Option<QuarantineCyclesEvidence>,
    /// Present only for the close-under-saturation probe. Close begins at
    /// the measurement-window midpoint with a cohort in flight; queued
    /// waiters must seal `backend_closing`, in-flight attempts must finish,
    /// and every submitted attempt must be accounted for exactly once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) close_under_load: Option<CloseUnderLoadEvidence>,
    /// The serialized public completion-receiver probe. It records the
    /// offered rate against the single sink's drain limit, then proves every
    /// submitted report was collected and the backend closed quiescently.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) completion_sink: Option<CompletionSinkEvidence>,
    /// The feature-off cell. This is present only when the profile asks the
    /// base runner to issue its dedicated schema-only cuGraph UDTF request
    /// beside relational traffic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) feature_off_cugraph: Option<FeatureOffCugraphEvidence>,
    pub(super) metrics_audit: MetricsAudit,
    /// Embedded-library operability evidence. Every field is reduced
    /// from public sealed reports or public snapshot accessors.
    pub(super) operability: OperabilityEvidence,
    /// Compact time series: per-device utilization/ledger/cache rows and
    /// process rows at the sampler cadence, plus their pure summaries. The
    /// raw snapshots below remain the retained evidence; the series is the
    /// queryable projection used by the telemetry, window, and cache checks.
    pub(super) time_series: TimeSeriesEvidence,
    pub(super) snapshots: Vec<BackendSnapshotSample>,
    pub(super) trace: TraceEvidence,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct QualificationFailure {
    pub(super) code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) attempt_id: Option<String>,
}

/// Versioned composed-schedule evidence. Every byte field is copied from one
/// sealed engine attempt report; telemetry supplies only independent physical
/// witnesses and is never used to derive a schedule value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct ComposedScheduleArtifact {
    pub(super) schema: &'static str,
    pub(super) version: u32,
    pub(super) attempt_id: String,
    pub(super) selected_ordinal: usize,
    pub(super) grant_id: u64,
    pub(super) grant_bytes: u64,
    pub(super) schedule_fingerprint: String,
    /// Set only by the engine projection's content verifier before the runner
    /// copies the schedule into this artifact.
    pub(super) engine_fingerprint_matches_contents: bool,
    pub(super) attempt_wide_fixed_bytes: u64,
    pub(super) occurrences: Vec<ComposedScheduleOccurrence>,
    pub(super) consumer_structural_scope: String,
    pub(super) consumer_additional_peak_bytes: u64,
    pub(super) consumer_stage_peak_bytes: u64,
    pub(super) complete_phase_peak_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct ComposedScheduleOccurrence {
    pub(super) occurrence_id: String,
    pub(super) order: usize,
    pub(super) structural_scope: String,
    pub(super) isolated_peak_bytes: u64,
    pub(super) stage_peak_bytes: u64,
    pub(super) retained_output_bytes: u64,
}

impl ComposedScheduleArtifact {
    #[cfg(test)]
    pub(super) fn test_complete() -> Self {
        Self {
            schema: super::COMPOSED_SCHEDULE_SCHEMA,
            version: super::COMPOSED_SCHEDULE_VERSION,
            attempt_id: "attempt-1".to_string(),
            selected_ordinal: 0,
            grant_id: 1,
            grant_bytes: 160,
            schedule_fingerprint: "composed-schedule-fnv1a64:0123456789abcdef".to_string(),
            engine_fingerprint_matches_contents: true,
            attempt_wide_fixed_bytes: 10,
            occurrences: vec![
                ComposedScheduleOccurrence {
                    occurrence_id: "pagerank#0".to_string(),
                    order: 0,
                    structural_scope: "scope-pr".to_string(),
                    isolated_peak_bytes: 90,
                    stage_peak_bytes: 100,
                    retained_output_bytes: 20,
                },
                ComposedScheduleOccurrence {
                    occurrence_id: "forceatlas2#0".to_string(),
                    order: 1,
                    structural_scope: "scope-fa2".to_string(),
                    isolated_peak_bytes: 100,
                    stage_peak_bytes: 130,
                    retained_output_bytes: 30,
                },
            ],
            consumer_structural_scope: "scope-consumer".to_string(),
            consumer_additional_peak_bytes: 10,
            consumer_stage_peak_bytes: 70,
            complete_phase_peak_bytes: 130,
        }
    }

    fn failures(&self) -> Vec<QualificationFailure> {
        let mut failures = Vec::new();
        if self.schema != super::COMPOSED_SCHEDULE_SCHEMA || self.version != super::COMPOSED_SCHEDULE_VERSION {
            failures.push(QualificationFailure {
                code: "composed_schedule_version_mismatch",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if self.attempt_id.is_empty() || self.grant_id == 0 {
            failures.push(QualificationFailure {
                code: "composed_schedule_identity_missing",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if !is_engine_schedule_fingerprint(&self.schedule_fingerprint) {
            failures.push(QualificationFailure {
                code: "composed_schedule_fingerprint_invalid",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if !self.engine_fingerprint_matches_contents {
            failures.push(QualificationFailure {
                code: "composed_schedule_fingerprint_mismatch",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if self.occurrences.is_empty() {
            failures.push(QualificationFailure {
                code: "missing_composed_schedule_row",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        let mut retained = 0_u64;
        let mut derived_peak = 0_u64;
        let mut ids = std::collections::BTreeSet::new();
        for (expected_order, occurrence) in self.occurrences.iter().enumerate() {
            if occurrence.order != expected_order
                || occurrence.occurrence_id.is_empty()
                || occurrence.structural_scope.is_empty()
            {
                failures.push(QualificationFailure {
                    code: "contradictory_composed_schedule",
                    attempt_id: Some(self.attempt_id.clone()),
                });
            }
            if !ids.insert(&occurrence.occurrence_id) {
                failures.push(QualificationFailure {
                    code: "duplicate_composed_schedule_occurrence",
                    attempt_id: Some(self.attempt_id.clone()),
                });
            }
            let Some(stage) = self
                .attempt_wide_fixed_bytes
                .checked_add(retained)
                .and_then(|value| value.checked_add(occurrence.isolated_peak_bytes))
            else {
                failures.push(QualificationFailure {
                    code: "composed_schedule_arithmetic_overflow",
                    attempt_id: Some(self.attempt_id.clone()),
                });
                break;
            };
            if occurrence.stage_peak_bytes != stage {
                failures.push(QualificationFailure {
                    code: "composed_schedule_stage_peak_mismatch",
                    attempt_id: Some(self.attempt_id.clone()),
                });
            }
            derived_peak = derived_peak.max(stage);
            let Some(next_retained) = retained.checked_add(occurrence.retained_output_bytes) else {
                failures.push(QualificationFailure {
                    code: "composed_schedule_arithmetic_overflow",
                    attempt_id: Some(self.attempt_id.clone()),
                });
                break;
            };
            retained = next_retained;
        }
        if self.consumer_structural_scope.is_empty() {
            failures.push(QualificationFailure {
                code: "missing_composed_consumer_scope",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if let Some(consumer_stage) = self
            .attempt_wide_fixed_bytes
            .checked_add(retained)
            .and_then(|value| value.checked_add(self.consumer_additional_peak_bytes))
        {
            if self.consumer_stage_peak_bytes != consumer_stage {
                failures.push(QualificationFailure {
                    code: "composed_schedule_consumer_peak_mismatch",
                    attempt_id: Some(self.attempt_id.clone()),
                });
            }
            derived_peak = derived_peak.max(consumer_stage);
        } else {
            failures.push(QualificationFailure {
                code: "composed_schedule_arithmetic_overflow",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        if self.complete_phase_peak_bytes != derived_peak || self.grant_bytes < self.complete_phase_peak_bytes {
            failures.push(QualificationFailure {
                code: "composed_schedule_complete_peak_mismatch",
                attempt_id: Some(self.attempt_id.clone()),
            });
        }
        failures
    }
}

fn is_engine_schedule_fingerprint(fingerprint: &str) -> bool {
    let Some(hex) = fingerprint.strip_prefix("composed-schedule-fnv1a64:") else {
        return false;
    };
    hex.len() == 16 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn qualification_failures(
    measurement: &MeasuredRunArtifact,
    provenance: &Provenance,
    composed_schedules: &[ComposedScheduleArtifact],
    require_overload_outcome: bool,
) -> Vec<QualificationFailure> {
    let mut failures = Vec::new();
    for repository in [
        &provenance.repositories.root,
        &provenance.repositories.cudf,
        &provenance.repositories.cugraph,
        &provenance.repositories.cuvs,
        &provenance.repositories.cuml,
    ] {
        if repository.revision.is_empty() || repository.dirty_manifest.sha256.is_empty() {
            failures.push(QualificationFailure { code: "missing_source_hash", attempt_id: None });
        }
        if !repository.dirty_manifest.entries.is_empty() {
            failures.push(QualificationFailure { code: "dirty_source", attempt_id: None });
        }
    }
    if provenance.runner.executable.sha256.is_empty() {
        failures.push(QualificationFailure { code: "missing_binary_hash", attempt_id: None });
    }
    if provenance.native_libraries.libcudf_resolved.as_ref().is_none_or(|identity| identity.sha256.is_empty()) {
        failures.push(QualificationFailure { code: "missing_libcudf_hash", attempt_id: None });
    }
    if provenance.native_libraries.libcugraph_resolved.as_ref().is_none_or(|identity| identity.sha256.is_empty()) {
        failures.push(QualificationFailure { code: "missing_libcugraph_hash", attempt_id: None });
    }
    for placement in &measurement.aggregate.placement {
        if placement.attempts_placed == 0 {
            continue;
        }
        let ordinal = placement.configuration.ordinal;
        if !measurement
            .time_series
            .device_rows
            .iter()
            .any(|row| row.device_ordinal == ordinal && row.cuda_free_bytes.is_some() && row.nvml_used_bytes.is_some())
        {
            failures.push(QualificationFailure { code: "missing_physical_memory_sample", attempt_id: None });
        }
    }

    if measurement.selected_queries.iter().any(|query| query.label == "pagerank_forceatlas2_join") {
        let composed_attempts = measurement
            .aggregate
            .attempts
            .iter()
            .filter(|attempt| attempt.query_label == "pagerank_forceatlas2_join")
            .collect::<Vec<_>>();
        if composed_attempts.is_empty() || composed_schedules.is_empty() {
            failures.push(QualificationFailure { code: "missing_composed_schedule", attempt_id: None });
        }
        let mut schedules_by_attempt = std::collections::BTreeMap::new();
        for schedule in composed_schedules {
            if schedules_by_attempt.insert(schedule.attempt_id.as_str(), schedule).is_some() {
                failures.push(QualificationFailure {
                    code: "duplicate_composed_schedule_attempt",
                    attempt_id: Some(schedule.attempt_id.clone()),
                });
            }
            failures.extend(schedule.failures());
        }
        for attempt in composed_attempts {
            let Some(schedule) = schedules_by_attempt.get(attempt.attempt_id.as_str()) else {
                failures.push(QualificationFailure {
                    code: "missing_composed_schedule",
                    attempt_id: Some(attempt.attempt_id.clone()),
                });
                continue;
            };
            if attempt.device_ordinal != Some(schedule.selected_ordinal)
                || attempt.grant_bytes != Some(schedule.grant_bytes)
                || attempt.grant_id != Some(schedule.grant_id)
            {
                failures.push(QualificationFailure {
                    code: "composed_schedule_attempt_binding_mismatch",
                    attempt_id: Some(attempt.attempt_id.clone()),
                });
            }
            if !measurement.aggregate.placement.iter().any(|placement| {
                placement.configuration.ordinal == schedule.selected_ordinal && placement.attempts_placed > 0
            }) {
                failures.push(QualificationFailure {
                    code: "missing_device_witness",
                    attempt_id: Some(attempt.attempt_id.clone()),
                });
            }
        }
        for schedule in composed_schedules {
            if !measurement.aggregate.attempts.iter().any(|attempt| attempt.attempt_id == schedule.attempt_id) {
                failures.push(QualificationFailure {
                    code: "composed_schedule_attempt_not_observed",
                    attempt_id: Some(schedule.attempt_id.clone()),
                });
            }
        }
    }
    if require_overload_outcome
        && !measurement
            .aggregate
            .scenarios
            .overload_fairness
            .as_ref()
            .is_some_and(|evidence| evidence.status == ClauseStatus::Evidenced)
    {
        failures.push(QualificationFailure { code: "missing_required_overload_outcome", attempt_id: None });
    }
    failures
}

/// Evidence for one profile that declares the QPS-envelope contract.
///
/// The queue high-water mark comes from the service-owned public admission
/// snapshot and is recorded cumulatively through each sequential step.  A
/// cumulative value still proves the global queue bound for every preceding
/// phase; it intentionally does not claim an invented per-step resettable
/// controller metric.
#[derive(Debug, Clone, Serialize)]
pub(super) struct QpsEnvelopeEvidence {
    pub(super) status: ClauseStatus,
    pub(super) configured_active_attempt_capacity: u64,
    pub(super) configured_queue_limit: u64,
    pub(super) arrival_plan_truncated_by_attempt_cap: bool,
    pub(super) global_queue_high_watermark: u64,
    pub(super) queue_bound_respected: bool,
    pub(super) gpu_only_successes: bool,
    pub(super) steps: Vec<QpsEnvelopeStepEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recovery: Option<QpsEnvelopeRecoveryEvidence>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct QpsEnvelopeStepEvidence {
    pub(super) step_index: usize,
    pub(super) concurrency: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) successful_completions: u64,
    pub(super) gpu_reaching_completions: u64,
    pub(super) minimum_gpu_reaching_completions: u64,
    /// Service-owned maximum depth observed from backend construction through
    /// the end of this step.  `None` is a hard evidence gap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) queue_high_watermark_through_step: Option<u64>,
    pub(super) queue_bound_respected: bool,
    pub(super) gpu_only_successes: bool,
    pub(super) has_required_gpu_progress: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct QpsEnvelopeRecoveryEvidence {
    pub(super) baseline_step_index: usize,
    pub(super) recovery_step_index: usize,
    pub(super) max_p95_latency_multiplier: u32,
    pub(super) baseline_p95_total_ns: u64,
    pub(super) recovery_p95_total_ns: u64,
    pub(super) baseline_p95_queue_wait_ns: u64,
    pub(super) recovery_p95_queue_wait_ns: u64,
    pub(super) total_latency_within_band: bool,
    pub(super) queue_wait_within_band: bool,
}

impl QpsEnvelopeEvidence {
    pub(super) fn from_measurement(
        profile: QpsEnvelopeProfile,
        aggregate: &MeasuredAggregate,
        configured_active_attempt_capacity: usize,
        configured_queue_limit: usize,
        arrival_plan_truncated_by_attempt_cap: bool,
        queue_high_watermarks: &[(usize, u64)],
    ) -> Self {
        Self::from_steps(
            profile,
            &aggregate.steps,
            configured_active_attempt_capacity,
            configured_queue_limit,
            arrival_plan_truncated_by_attempt_cap,
            queue_high_watermarks,
        )
    }

    fn from_steps(
        profile: QpsEnvelopeProfile,
        aggregate_steps: &[StepAggregate],
        configured_active_attempt_capacity: usize,
        configured_queue_limit: usize,
        arrival_plan_truncated_by_attempt_cap: bool,
        queue_high_watermarks: &[(usize, u64)],
    ) -> Self {
        let queue_limit = configured_queue_limit as u64;
        let minimum = profile.min_gpu_reaching_completions_per_step;
        let steps = aggregate_steps
            .iter()
            .map(|step| QpsEnvelopeStepEvidence::from_aggregate(step, minimum, queue_limit, queue_high_watermarks))
            .collect::<Vec<_>>();
        let global_queue_high_watermark = queue_high_watermarks.iter().map(|(_, peak)| *peak).max().unwrap_or_default();
        let queue_bound_respected = !steps.is_empty()
            && steps.iter().all(|step| step.queue_high_watermark_through_step.is_some() && step.queue_bound_respected);
        let gpu_only_successes = !steps.is_empty() && steps.iter().all(|step| step.gpu_only_successes);
        let all_steps_have_gpu_progress = !steps.is_empty() && steps.iter().all(|step| step.has_required_gpu_progress);
        let recovery = profile.recovery.and_then(|recovery| {
            let baseline = aggregate_steps.iter().find(|step| step.step_index == recovery.baseline_step_index)?;
            let recovered = aggregate_steps.iter().find(|step| step.step_index == recovery.recovery_step_index)?;
            Some(QpsEnvelopeRecoveryEvidence::from_steps(recovery, baseline, recovered))
        });
        let recovery_evidenced =
            profile.recovery.is_none_or(|_| recovery.as_ref().is_some_and(QpsEnvelopeRecoveryEvidence::within_band));
        let status = if !arrival_plan_truncated_by_attempt_cap
            && queue_bound_respected
            && gpu_only_successes
            && all_steps_have_gpu_progress
            && recovery_evidenced
        {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::Failed
        };
        Self {
            status,
            configured_active_attempt_capacity: configured_active_attempt_capacity as u64,
            configured_queue_limit: queue_limit,
            arrival_plan_truncated_by_attempt_cap,
            global_queue_high_watermark,
            queue_bound_respected,
            gpu_only_successes,
            steps,
            recovery,
        }
    }

    pub(super) fn passes(&self) -> bool {
        self.status == ClauseStatus::Evidenced
    }
}

impl QpsEnvelopeStepEvidence {
    fn from_aggregate(
        step: &StepAggregate,
        minimum_gpu_reaching_completions: u64,
        queue_limit: u64,
        queue_high_watermarks: &[(usize, u64)],
    ) -> Self {
        let queue_high_watermark_through_step =
            queue_high_watermarks.iter().find(|(step_index, _)| *step_index == step.step_index).map(|(_, peak)| *peak);
        let successful_completions = step.throughput.successful_completions;
        let gpu_reaching_completions = step.throughput.gpu_reaching_completions;
        Self {
            step_index: step.step_index,
            concurrency: step.concurrency,
            arrival_rate_per_second: step.arrival_rate_per_second,
            successful_completions,
            gpu_reaching_completions,
            minimum_gpu_reaching_completions,
            queue_high_watermark_through_step,
            queue_bound_respected: queue_high_watermark_through_step.is_some_and(|peak| peak <= queue_limit),
            gpu_only_successes: successful_completions == gpu_reaching_completions,
            has_required_gpu_progress: gpu_reaching_completions >= minimum_gpu_reaching_completions,
        }
    }
}

impl QpsEnvelopeRecoveryEvidence {
    fn from_steps(
        recovery: super::profile::QpsEnvelopeRecoveryProfile,
        baseline: &StepAggregate,
        recovered: &StepAggregate,
    ) -> Self {
        let multiplier = u64::from(recovery.max_p95_latency_multiplier);
        let baseline_p95_total_ns = baseline.latency.p95_ns;
        let recovery_p95_total_ns = recovered.latency.p95_ns;
        let baseline_p95_queue_wait_ns = baseline.latency.queue_wait_p95_ns;
        let recovery_p95_queue_wait_ns = recovered.latency.queue_wait_p95_ns;
        Self {
            baseline_step_index: recovery.baseline_step_index,
            recovery_step_index: recovery.recovery_step_index,
            max_p95_latency_multiplier: recovery.max_p95_latency_multiplier,
            baseline_p95_total_ns,
            recovery_p95_total_ns,
            baseline_p95_queue_wait_ns,
            recovery_p95_queue_wait_ns,
            total_latency_within_band: within_latency_band(baseline_p95_total_ns, recovery_p95_total_ns, multiplier),
            queue_wait_within_band: within_latency_band(
                baseline_p95_queue_wait_ns,
                recovery_p95_queue_wait_ns,
                multiplier,
            ),
        }
    }

    fn within_band(&self) -> bool {
        self.total_latency_within_band && self.queue_wait_within_band
    }
}

fn within_latency_band(baseline: u64, recovered: u64, multiplier: u64) -> bool {
    if baseline == 0 {
        return recovered == 0;
    }
    u128::from(recovered) <= u128::from(baseline).saturating_mul(u128::from(multiplier))
}

/// Operability facts an embedder needs in addition to throughput.  The OTLP
/// export cell is attached here once explicitly requested; ordinary runs keep
/// the attribution section without pretending that an exporter was used.
#[derive(Debug, Serialize)]
pub(super) struct OperabilityEvidence {
    pub(super) attribution: AttributionCompleteness,
    pub(super) otlp_export: OtlpExportEvidence,
}

/// The sampled time series and its reductions. `sample_interval_ns` is
/// the cadence the sampler actually used: it adapts to the configured window
/// so a 10-minute plateau stays inside the bounded retention instead of
/// silently truncating mid-window.
#[derive(Debug, Serialize)]
pub(super) struct TimeSeriesEvidence {
    pub(super) sample_interval_ns: u64,
    /// True when the window required a coarser cadence than the 250 ms base
    /// interval to stay inside bounded retention.
    pub(super) retention_downsampled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) maximum_cuda_nvml_sample_gap: Option<SampleGap>,
    pub(super) device_rows: Vec<DeviceSeriesRow>,
    pub(super) process_rows: Vec<ProcessSeriesRow>,
    pub(super) utilization: Vec<OrdinalUtilizationSummary>,
    pub(super) drift: DriftSummary,
    /// Claim-versus-VRAM calibration reduced from the same rows above. It is
    /// absent only when this profile did not opt into a calibration gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) vram_calibration: Option<VramCalibrationEvidence>,
    /// Monitoring cost over exactly the public snapshot pulls recorded
    /// in `process_rows`.
    pub(super) observation_pull: ObservationPullSummary,
}

/// The immutable backend envelope this run measured against.
#[derive(Debug, Serialize)]
pub(super) struct BackendEvidence {
    pub(super) backend_generation: u64,
    pub(super) shared_session_count: usize,
    pub(super) queue_max_queued_attempts: usize,
    pub(super) queue_max_queue_wait_ms: u64,
    pub(super) queue_max_overtakes_per_waiter: u32,
    pub(super) device_startup_records: Value,
    /// The close result retains both pre- and post-close snapshots. It is
    /// never omitted: the close result needs to distinguish a quiescent result from a
    /// timeout or cleanup failure without persisting arbitrary error text.
    pub(super) close: BackendCloseEvidence,
}

/// Bounded timing evidence for the one deadline shared by every attempted
/// query in a measured run.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RunDeadlineEvidence {
    /// Warmup plus measurement time during which the runner may submit work.
    pub(super) measurement_window_ns: u64,
    /// Bounded post-submission allowance for already-admitted work to reach a
    /// safe boundary and seal its report; zero means no configured drain allowance.
    pub(super) completion_drain_grace_ns: u64,
    /// The total completion deadline: `measurement_window_ns` plus the
    /// profile-owned bounded drain grace. The retained pass/fail flag below is
    /// evaluated against this deadline, not against a fabricated extension of
    /// the measurement interval.
    pub(super) configured_window_ns: u64,
    pub(super) elapsed_until_reports_ns: u64,
    pub(super) overrun_ns: u64,
    pub(super) reports_sealed_within_deadline: bool,
}

/// Evidence for the deliberately slow, serialized completion consumer.
/// `offered_rate_exceeds_serial_drain_capacity` is integer arithmetic for
/// `offered_rate_per_second > 1 / sink_delay`; no rounded rate estimate can
/// accidentally turn a non-probe into a pass.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionSinkEvidence {
    pub(super) sink_delay_ms: u64,
    pub(super) offered_rate_per_second: u64,
    pub(super) offered_rate_exceeds_serial_drain_capacity: bool,
    pub(super) submitted_attempts: u64,
    pub(super) sealed_reports: u64,
    pub(super) begin_failures: u64,
    pub(super) join_failures: u64,
    pub(super) drains_started: u64,
    pub(super) drains_completed: u64,
    pub(super) report_accounting_exact: bool,
    pub(super) bounded_drain: bool,
    pub(super) close_quiescent: bool,
}

impl CompletionSinkEvidence {
    pub(super) fn is_evidenced(&self) -> bool {
        self.sink_delay_ms != 0
            && self.offered_rate_exceeds_serial_drain_capacity
            && self.submitted_attempts != 0
            && self.report_accounting_exact
            && self.begin_failures == 0
            && self.join_failures == 0
            && self.drains_started == self.sealed_reports
            && self.drains_completed == self.sealed_reports
            && self.bounded_drain
            && self.close_quiescent
    }
}

/// One explicit public-API cancellation action and the sealed outcomes it
/// produced. The runner never aborts its attempt task.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CancellationBarrierEvidence {
    pub(super) attempt_sequence: usize,
    pub(super) requested_after_ms: u64,
    pub(super) cancellation_requested: bool,
    pub(super) cancelled_reports: u64,
}

impl CancellationBarrierEvidence {
    pub(super) fn is_evidenced(&self) -> bool {
        self.cancellation_requested && self.cancelled_reports > 0
    }
}

/// Evidence for a caller-provided attempt deadline that was first witnessed
/// as an admitted native attempt. The runner waits for the sealed report and
/// then verifies a healthy follow-up attempt; it never aborts a task to force
/// this outcome.
#[derive(Debug, Clone, Serialize)]
pub(super) struct DeadlineBarrierEvidence {
    pub(super) attempt_sequence: usize,
    pub(super) deadline_after_ms: u64,
    pub(super) admitted_witness_offset_ns: u64,
    pub(super) admitted_device_ordinal: usize,
    pub(super) admitted_ledger_sequence: u64,
    pub(super) admitted_live_grants: usize,
    pub(super) admitted_active_query_jobs: usize,
    pub(super) typed_deadline_cancelled_reports: u64,
    pub(super) released_after_deadline: bool,
    pub(super) healthy_reuse_completed: bool,
}

impl DeadlineBarrierEvidence {
    pub(super) fn is_evidenced(&self) -> bool {
        self.admitted_witness_offset_ns < self.deadline_after_ms.saturating_mul(1_000_000)
            && (self.admitted_live_grants > 0 || self.admitted_active_query_jobs > 0)
            && self.typed_deadline_cancelled_reports == 1
            && self.released_after_deadline
            && self.healthy_reuse_completed
    }
}

/// Close-under-saturation evidence. The runner initiates close at the
/// measurement-window midpoint through the public `close()` path and then
/// only observes: every fact below comes from public snapshots or sealed
/// attempt reports, never from runner-side assumptions about who was queued.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CloseUnderLoadEvidence {
    pub(super) close_requested_offset_ns: u64,
    /// True when this call was the one that transitioned the backend to
    /// closing (the public `close()` returns exactly that).
    pub(super) close_initiated: bool,
    /// Public admission snapshot at the moment close began.
    pub(super) admitted_at_close: u64,
    pub(super) queued_at_close: u64,
    /// Attempt tasks the runner had not yet collected when close began.
    pub(super) in_flight_at_close: u64,
    pub(super) submitted_attempts: u64,
    pub(super) sealed_reports: u64,
    pub(super) join_failures: u64,
    /// Sealed reports whose bounded failure class is `BackendClosing`: the
    /// queued waiters close had to account for.
    pub(super) backend_closing_sealed: u64,
    /// In-flight attempts that sealed any terminal report after close began.
    /// Closing is allowed to cancel active work; the probe requires the
    /// in-flight cohort to finish, not to fabricate successful execution.
    pub(super) in_flight_terminal_after_close: u64,
    /// Successful subset of the terminal in-flight cohort, retained as a
    /// diagnostic rather than a close-under-load pass condition.
    pub(super) in_flight_succeeded_after_close: u64,
    /// submitted == sealed + join-failures, exactly.
    pub(super) accounting_exact: bool,
}

impl CloseUnderLoadEvidence {
    pub(super) fn is_evidenced(&self) -> bool {
        self.close_initiated
            && self.accounting_exact
            && self.admitted_at_close > 0
            && self.queued_at_close > 0
            && self.in_flight_at_close > 0
            && self.backend_closing_sealed > 0
            && self.in_flight_terminal_after_close == self.in_flight_at_close
    }
}

/// Structured close result projected from the public backend API.
#[derive(Debug, Clone, Serialize)]
pub(super) struct BackendCloseEvidence {
    pub(super) before: BackendCloseSnapshotEvidence,
    pub(super) outcome: BackendCloseOutcome,
    pub(super) after: BackendCloseSnapshotEvidence,
}

impl BackendCloseEvidence {
    pub(super) fn is_quiescent(&self) -> bool {
        matches!(self.outcome, BackendCloseOutcome::Quiescent { .. })
            && self.after.gpu_resource_quiescent
            && self.after.observation_quiescent
            && self.after.active_gpu_permits == 0
            && self.after.active_observation_permits == 0
            && !self.after.cleanup_failed
    }
}

/// The success case retains the close snapshot returned by the engine; the
/// failure case retains only its stable classification and whitelisted facts.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub(super) enum BackendCloseOutcome {
    Quiescent { snapshot: BackendCloseSnapshotEvidence },
    Failed { failure: BackendCloseFailure },
}

/// CPU-only, versioned projection of the engine close snapshot.
///
/// The runner artifact must remain testable without linking the optional
/// GPU/engine dependency. `measure.rs` copies this bounded public snapshot at
/// the hardware boundary; the serialized shape deliberately matches the
/// engine fields it records, without retaining the engine type in this module.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct BackendCloseSnapshotEvidence {
    pub(super) closing: bool,
    pub(super) gpu_resource_quiescent: bool,
    pub(super) observation_quiescent: bool,
    pub(super) active_gpu_permits: usize,
    pub(super) active_observation_permits: usize,
    pub(super) cleanup_failed: bool,
    pub(super) cleanup_deadline_elapsed_count: u64,
    pub(super) cleanup_late_success_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct BackendCloseFailure {
    pub(super) code: &'static str,
    pub(super) kind: &'static str,
    pub(super) status: &'static str,
    pub(super) operation: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reason: Option<String>,
}

/// The plan minus its per-arrival vector, which the attempt rows already carry.
#[derive(Debug, Serialize)]
pub(super) struct ArrivalPlanEvidence {
    pub(super) arrival_mode: &'static str,
    pub(super) seed: u64,
    pub(super) concurrency: usize,
    pub(super) arrival_rate_per_second: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_rounds: Option<usize>,
    /// True when a closed-loop service profile creates rounds only as the
    /// preceding round drains, rather than serializing a speculative number
    /// of future arrivals into the artifact plan.
    pub(super) duration_paced_concurrency: bool,
    pub(super) rate_overrun: &'static str,
    pub(super) planned_arrivals: usize,
    /// The bound every plan is checked against, recorded so a truncation flag
    /// is auditable against the profile's intended offered process.
    pub(super) max_planned_attempts: usize,
    /// Planned phase split, present only for rate mode. Concurrency mode paces
    /// closed-loop rounds and assigns phases at submission, so a planned split
    /// would be meaningless; the observed split is in the measurement window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) planned_warmup_arrivals: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) planned_measurement_arrivals: Option<usize>,
    /// A plan cut short by the attempt cap is disclosed, never presented as the
    /// profile's intended plan.
    pub(super) truncated_by_attempt_cap: bool,
    /// Sequential phases. Present whenever the plan carried more than one
    /// step; single-phase runs omit the section.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) steps: Vec<StepPlanEvidence>,
    /// Deadline/cancellation storm knobs and armed cancellation count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) storm: Option<StormEvidence>,
    pub(super) dropped_overrun_arrivals: u64,
    /// Arrivals the backend refused to begin, which ends submission early.
    pub(super) begin_failures: u64,
    /// Attempt tasks that panicked or were cancelled, losing only their row.
    pub(super) join_failures: u64,
}

#[derive(Debug, Serialize)]
pub(super) struct StepPlanEvidence {
    pub(super) step_index: usize,
    pub(super) concurrency: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) rate_overrun: &'static str,
    pub(super) start_ns: u64,
    pub(super) measurement_ns: u64,
    pub(super) planned_arrivals: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct StormEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) attempt_deadline_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cancel_every_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cancel_after_ms: Option<u64>,
    pub(super) cancellations_requested: u64,
}

impl ArrivalPlanEvidence {
    pub(super) fn new(
        plan: &ArrivalPlan,
        dropped_overrun_arrivals: u64,
        begin_failures: u64,
        join_failures: u64,
        storm_cancellations_requested: u64,
    ) -> Self {
        let storm = (plan.attempt_deadline_ms.is_some() || plan.cancel_every_n.is_some()).then_some(StormEvidence {
            attempt_deadline_ms: plan.attempt_deadline_ms,
            cancel_every_n: plan.cancel_every_n,
            cancel_after_ms: plan.cancel_after_ms,
            cancellations_requested: storm_cancellations_requested,
        });
        Self {
            arrival_mode: plan.arrival_mode,
            seed: plan.seed,
            concurrency: plan.concurrency,
            arrival_rate_per_second: plan.arrival_rate_per_second,
            max_rounds: plan.max_rounds,
            duration_paced_concurrency: plan.duration_paced_concurrency,
            rate_overrun: plan.rate_overrun,
            planned_arrivals: plan.arrivals.len(),
            max_planned_attempts: super::profile::MAX_PLANNED_ATTEMPTS,
            planned_warmup_arrivals: (plan.arrival_mode == "rate").then_some(plan.warmup_arrivals),
            planned_measurement_arrivals: (plan.arrival_mode == "rate").then_some(plan.measurement_arrivals),
            truncated_by_attempt_cap: plan.truncated_by_attempt_cap,
            steps: if plan.steps.len() > 1 {
                plan.steps
                    .iter()
                    .map(|step| StepPlanEvidence {
                        step_index: step.step_index,
                        concurrency: step.concurrency,
                        arrival_rate_per_second: step.arrival_rate_per_second,
                        rate_overrun: step.rate_overrun,
                        start_ns: step.start_ns,
                        measurement_ns: step.measurement_ns,
                        planned_arrivals: step.planned_arrivals,
                    })
                    .collect()
            } else {
                Vec::new()
            },
            storm,
            dropped_overrun_arrivals,
            begin_failures,
            join_failures,
        }
    }
}

#[cfg(test)]
impl TimeSeriesEvidence {
    pub(super) fn empty() -> Self {
        Self {
            sample_interval_ns: 0,
            retention_downsampled: false,
            maximum_cuda_nvml_sample_gap: None,
            device_rows: Vec::new(),
            process_rows: Vec::new(),
            utilization: Vec::new(),
            drift: super::timeseries::DriftSummary {
                ordinals: Vec::new(),
                generations_monotonic: true,
                ledger_sequences_monotonic: true,
            },
            vram_calibration: None,
            observation_pull: super::timeseries::ObservationPullSummary {
                samples: 0,
                min_ns: 0,
                p50_ns: 0,
                p95_ns: 0,
                max_ns: 0,
                mean_ns: 0,
            },
        }
    }
}

/// One timestamped sampler observation. The nine backend snapshot types are
/// serialized through their own `Serialize`, not re-projected here.
#[derive(Debug, Serialize)]
pub(super) struct BackendSnapshotSample {
    pub(super) offset_ns: u64,
    /// Exact public backend-metric pull retained through the engine-owned serde shape.
    pub(super) backend_metrics_snapshot: Value,
    pub(super) admission: Value,
    pub(super) device_ledger_reports: Value,
    /// Physical cache allocation-domain reports, each paired with its own
    /// canonical ledger sample by the engine before serialization.
    pub(super) device_cache_reports: Value,
    pub(super) device_memory_failure_reports: Value,
    pub(super) gpu_execution_pool_reports: Value,
    pub(super) device_telemetry_snapshots: Value,
    /// Exact-request BFS coalescing counters, when the cugraph build exposes
    /// them.  Feature-off samples retain JSON null instead of inventing zeros.
    pub(super) cugraph_bfs_singleflight_snapshot: Value,
}

/// Nsight trace identity. This runner never captures the trace itself; it
/// records what an external `nsys profile` produced so the artifact can cite
/// the stream-level overlap evidence it does not own.
#[derive(Debug, Serialize)]
pub(super) struct TraceEvidence {
    pub(super) captured_by_runner: bool,
    pub(super) uri: Option<String>,
    pub(super) sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct RedactionEvidence {
    pub(super) policy: &'static str,
    pub(super) raw_config_persisted: bool,
    pub(super) sensitive_keys_rejected: bool,
    pub(super) raw_sql_rejected: bool,
    pub(super) raw_diffs_rejected: bool,
    pub(super) unbounded_errors_persisted: bool,
}

pub(super) fn write_artifact(out_dir: &Path, artifact: &RunArtifact) -> ToolResult<PathBuf> {
    let encoded = serde_json::to_vec_pretty(artifact)?;
    ensure_redacted_json(&serde_json::from_slice(&encoded)?, "")?;
    write_new_artifact(out_dir, &encoded)
}

#[cfg(test)]
mod qps_envelope_tests {
    use super::{QpsEnvelopeEvidence, QpsEnvelopeProfile};
    use crate::single_node_multi_gpu_validate::{
        aggregate::{ClauseStatus, LatencyAggregate, StepAggregate, ThroughputAggregate},
        profile::QpsEnvelopeRecoveryProfile,
    };

    fn step(
        index: usize,
        rate: u64,
        successful: u64,
        gpu_reaching: u64,
        p95_ns: u64,
        queue_p95_ns: u64,
    ) -> StepAggregate {
        StepAggregate {
            step_index: index,
            concurrency: 0,
            arrival_rate_per_second: Some(rate),
            rate_overrun: "delay",
            measurement_ns: 1_000_000_000,
            throughput: ThroughputAggregate {
                measurement_seconds: 1.0,
                completions: successful,
                successful_completions: successful,
                gpu_reaching_completions: gpu_reaching,
                application_qps: successful as f64,
                gpu_reaching_qps: gpu_reaching as f64,
            },
            latency: LatencyAggregate {
                samples: successful,
                min_ns: p95_ns,
                p50_ns: p95_ns,
                p95_ns,
                p99_ns: p95_ns,
                max_ns: p95_ns,
                mean_ns: p95_ns,
                queue_wait_p95_ns: queue_p95_ns,
                admission_wait_samples: successful,
                admission_wait_p50_ns: queue_p95_ns,
                admission_wait_p95_ns: queue_p95_ns,
                admission_wait_p99_ns: queue_p95_ns,
            },
            placements: Vec::new(),
        }
    }

    #[test]
    fn qps_envelope_requires_complete_gpu_progress_bounded_queue_and_recovery() {
        let profile = QpsEnvelopeProfile {
            min_gpu_reaching_completions_per_step: 1,
            recovery: Some(QpsEnvelopeRecoveryProfile {
                baseline_step_index: 0,
                recovery_step_index: 2,
                max_p95_latency_multiplier: 3,
            }),
        };
        let steps = [step(0, 10, 10, 10, 100, 10), step(1, 100, 4, 4, 900, 80), step(2, 10, 8, 8, 250, 30)];
        let evidence = QpsEnvelopeEvidence::from_steps(profile, &steps, 16, 64, false, &[(0, 0), (1, 64), (2, 64)]);
        assert_eq!(evidence.status, ClauseStatus::Evidenced);
        assert!(evidence.passes());
        assert!(evidence.recovery.as_ref().is_some_and(|recovery| recovery.total_latency_within_band));

        let missing_high_water = QpsEnvelopeEvidence::from_steps(profile, &steps, 16, 64, false, &[(0, 0), (1, 64)]);
        assert_eq!(missing_high_water.status, ClauseStatus::Failed);

        let non_gpu_success = QpsEnvelopeEvidence::from_steps(
            profile,
            &[step(0, 10, 10, 10, 100, 10), step(1, 100, 4, 3, 900, 80), step(2, 10, 8, 8, 250, 30)],
            16,
            64,
            false,
            &[(0, 0), (1, 64), (2, 64)],
        );
        assert_eq!(non_gpu_success.status, ClauseStatus::Failed);

        let truncated = QpsEnvelopeEvidence::from_steps(profile, &steps, 16, 64, true, &[(0, 0), (1, 64), (2, 64)]);
        assert_eq!(truncated.status, ClauseStatus::Failed);
    }
}

#[cfg(test)]
mod composed_schedule_tests {
    use serde_json::Value;

    use super::ComposedScheduleArtifact;

    #[test]
    fn complete_schedule_validates_and_serializes_as_versioned_evidence() {
        let schedule = ComposedScheduleArtifact::test_complete();
        assert!(schedule.failures().is_empty());
        let value: Value = serde_json::to_value(schedule).unwrap();
        assert_eq!(value["schema"], "algeon_tools.single_node_multi_gpu.composed_schedule.v1");
        assert_eq!(value["version"], 1);
        assert_eq!(value["occurrences"].as_array().unwrap().len(), 2);
        assert_eq!(value["occurrences"][0]["order"], 0);
        assert_eq!(value["consumer_structural_scope"], "scope-consumer");
    }

    #[test]
    fn missing_or_contradictory_schedule_fails_closed() {
        let mut missing = ComposedScheduleArtifact::test_complete();
        missing.occurrences.clear();
        let codes = missing.failures().into_iter().map(|failure| failure.code).collect::<Vec<_>>();
        assert!(codes.contains(&"missing_composed_schedule_row"));
        assert!(codes.contains(&"composed_schedule_complete_peak_mismatch"));

        let mut contradictory = ComposedScheduleArtifact::test_complete();
        contradictory.occurrences[1].order = 0;
        contradictory.occurrences[1].occurrence_id = contradictory.occurrences[0].occurrence_id.clone();
        contradictory.occurrences[1].stage_peak_bytes = 1;
        contradictory.schedule_fingerprint = "not-an-engine-fingerprint".to_string();
        contradictory.engine_fingerprint_matches_contents = false;
        contradictory.grant_bytes = 1;
        let codes = contradictory.failures().into_iter().map(|failure| failure.code).collect::<Vec<_>>();
        assert!(codes.contains(&"contradictory_composed_schedule"));
        assert!(codes.contains(&"duplicate_composed_schedule_occurrence"));
        assert!(codes.contains(&"composed_schedule_stage_peak_mismatch"));
        assert!(codes.contains(&"composed_schedule_fingerprint_invalid"));
        assert!(codes.contains(&"composed_schedule_fingerprint_mismatch"));
        assert!(codes.contains(&"composed_schedule_complete_peak_mismatch"));
    }
}
