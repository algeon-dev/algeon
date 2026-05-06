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
    aggregate::{DomainExhaustionEvidence, MeasuredAggregate, PhysicalAllocatorPressureEvidence},
    cache_calibration::CacheCalibrationEvidence,
    metrics_audit::MetricsAudit,
    preflight::FixturePreflightArtifact,
    profile::RunProfile,
    provenance::Provenance,
    redact::ensure_redacted_json,
    schedule::ArrivalPlan,
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
        measurement: MeasuredRunArtifact,
    ) -> Self {
        let status = if measurement.qualifying { "measured_run" } else { "measured_run_non_qualifying" };
        let mut artifact = Self::new(profile, config_sha256, provenance, status, "complete", fixtures);
        artifact.execution.workload_execution = "complete";
        artifact.execution.attempted_workloads = measurement.selected_queries.len() as u64;
        artifact.execution.attempted_backend_attempts = measurement.aggregate.attempts.len() as u64;
        artifact.execution.measurement = Some(measurement);
        artifact
    }

    /// Publishes the two directly comparable halves of a P6.8 cache
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
        let evidence = CacheCalibrationEvidence::new(cache_disabled, cache_opt_in);
        let status = if evidence.qualifies() { "cache_calibration" } else { "cache_calibration_non_qualifying" };
        let attempted_backend_attempts =
            evidence.cache_disabled.aggregate.attempts.len() + evidence.cache_opt_in.aggregate.attempts.len();
        let attempted_workloads = evidence.cache_opt_in.selected_queries.len() as u64;
        let mut artifact = Self::new(profile, config_sha256, provenance, status, "complete", fixtures);
        artifact.execution.workload_execution = "complete";
        artifact.execution.attempted_workloads = attempted_workloads;
        artifact.execution.attempted_backend_attempts = attempted_backend_attempts as u64;
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
    /// Present only for the P6.6 controlled after-admission deadline probe.
    /// It distinguishes a typed deadline at a native safe boundary from
    /// queue expiry or explicit caller cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) deadline_barrier: Option<DeadlineBarrierEvidence>,
    /// Present only for the P6.6 ordinary domain-cap exhaustion probe. This
    /// is deliberately distinct from a physical allocator failure: the sealed
    /// allocator event must name `reservation_limit` with exact attempt
    /// provenance before the run can qualify.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) domain_exhaustion: Option<DomainExhaustionEvidence>,
    /// Present only for the P6.6 real external allocator-pressure probe. The
    /// predicate is fail-closed: it requires one sealed exact physical event,
    /// immutable logical caps above its request, verified pressure release,
    /// same-device recovery, and a quiescent backend close.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) physical_allocator_pressure: Option<PhysicalAllocatorPressureEvidence>,
    pub(super) metrics_audit: MetricsAudit,
    pub(super) snapshots: Vec<BackendSnapshotSample>,
    pub(super) trace: TraceEvidence,
}

/// The immutable backend envelope this run measured against.
#[derive(Debug, Serialize)]
pub(super) struct BackendEvidence {
    pub(super) resource_core: u64,
    pub(super) observation_core: u64,
    pub(super) resource_generation: u64,
    pub(super) shared_session_count: usize,
    pub(super) queue_max_queued_attempts: usize,
    pub(super) queue_max_queue_wait_ms: u64,
    pub(super) queue_max_overtakes_per_waiter: u32,
    pub(super) device_startup_records: Value,
    /// The close result retains both pre- and post-close snapshots. It is
    /// never omitted: P6.6 needs to distinguish a quiescent result from a
    /// timeout or cleanup failure without persisting arbitrary error text.
    pub(super) close: BackendCloseEvidence,
}

/// Bounded timing evidence for the one deadline shared by every attempted
/// query in a measured run.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct RunDeadlineEvidence {
    pub(super) configured_window_ns: u64,
    pub(super) elapsed_until_reports_ns: u64,
    pub(super) overrun_ns: u64,
    pub(super) reports_sealed_within_deadline: bool,
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
    pub(super) rate_overrun: &'static str,
    pub(super) planned_arrivals: usize,
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
    pub(super) dropped_overrun_arrivals: u64,
    /// Arrivals the backend refused to begin, which ends submission early.
    pub(super) begin_failures: u64,
    /// Attempt tasks that panicked or were cancelled, losing only their row.
    pub(super) join_failures: u64,
}

impl ArrivalPlanEvidence {
    pub(super) fn new(
        plan: &ArrivalPlan,
        dropped_overrun_arrivals: u64,
        begin_failures: u64,
        join_failures: u64,
    ) -> Self {
        Self {
            arrival_mode: plan.arrival_mode,
            seed: plan.seed,
            concurrency: plan.concurrency,
            arrival_rate_per_second: plan.arrival_rate_per_second,
            max_rounds: plan.max_rounds,
            rate_overrun: plan.rate_overrun,
            planned_arrivals: plan.arrivals.len(),
            planned_warmup_arrivals: (plan.arrival_mode == "rate").then_some(plan.warmup_arrivals),
            planned_measurement_arrivals: (plan.arrival_mode == "rate").then_some(plan.measurement_arrivals),
            truncated_by_attempt_cap: plan.truncated_by_attempt_cap,
            dropped_overrun_arrivals,
            begin_failures,
            join_failures,
        }
    }
}

/// One timestamped sampler observation. The nine backend snapshot types are
/// serialized through their own `Serialize`, not re-projected here.
#[derive(Debug, Serialize)]
pub(super) struct BackendSnapshotSample {
    pub(super) offset_ns: u64,
    /// Exact P4 pull snapshot, retained through the engine-owned serde shape.
    pub(super) backend_metrics_snapshot: Value,
    pub(super) admission: Value,
    pub(super) device_ledger_reports: Value,
    pub(super) device_memory_failure_reports: Value,
    pub(super) gpu_execution_pool_reports: Value,
    pub(super) device_telemetry_snapshots: Value,
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
