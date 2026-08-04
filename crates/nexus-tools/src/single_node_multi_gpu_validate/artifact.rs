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
    REDACTION_POLICY, RUN_ARTIFACT_SCHEMA, RUN_ARTIFACT_VERSION, aggregate::MeasuredAggregate,
    preflight::FixturePreflightArtifact, profile::RunProfile, provenance::Provenance, redact::ensure_redacted_json,
    schedule::ArrivalPlan, util::write_new_artifact, workload::SelectedQueryDescriptor,
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
    pub(super) close: Value,
}

/// The plan minus its per-arrival vector, which the attempt rows already carry.
#[derive(Debug, Serialize)]
pub(super) struct ArrivalPlanEvidence {
    pub(super) arrival_mode: &'static str,
    pub(super) seed: u64,
    pub(super) concurrency: usize,
    pub(super) arrival_rate_per_second: Option<u64>,
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
