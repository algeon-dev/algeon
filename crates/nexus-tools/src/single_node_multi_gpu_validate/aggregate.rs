//! Pure reduction over collected attempt reports and sampled snapshots.
//!
//! Nothing here touches a GPU or a clock, so every scenario assertion is
//! unit-testable. The observation inputs are deliberately narrow structs
//! rather than the engine snapshots themselves: the binary projects the
//! snapshots once, and the reduction stays testable without CUDA.
//!
//! Only the measured run constructs these values, so the CPU-only feature
//! reaches them from tests alone.
#![cfg_attr(not(feature = "single-node-multi-gpu-validate-run"), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

use super::{artifact::ComposedScheduleArtifact, profile::FixtureKind, schedule::SchedulePhase};

/// One attempt's measured outcome, projected from `QueryAttemptReport`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AttemptObservation {
    pub(super) sequence: usize,
    pub(super) phase: SchedulePhase,
    /// Multi-phase step this attempt was submitted under. Single-phase runs are all
    /// step zero.
    pub(super) step_index: usize,
    /// The fixture kind the query came from. Labels collide across kinds
    /// (`q01` exists in TPC-H and TPC-DS), so this remains artifact identity.
    pub(super) query_kind: FixtureKind,
    pub(super) query_label: String,
    pub(super) graph_workload: bool,
    /// The runner owns this expectation at query selection time. A false
    /// value is reserved for an explicit non-executing capability probe whose
    /// own reducer proves its exact terminal contract.
    pub(super) placement_required: bool,
    pub(super) attempt_id: String,
    pub(super) disposition: String,
    pub(super) outcome: String,
    /// Bounded closed-vocabulary classification of the primary failure, when
    /// the attempt failed. Never a raw error string.
    pub(super) failure: Option<AttemptFailureClass>,
    /// Bounded, redacted facts from the canonical sealed report needed for
    /// Dedicated-probe deadline, fault, cleanup, and allocator evidence. This is a
    /// read-only projection, never an independent event store.
    pub(super) terminal_evidence: AttemptTerminalEvidence,
    pub(super) succeeded: bool,
    /// Present only when the attempt reached native GPU execution.
    pub(super) reached_gpu: bool,
    /// One typed graph model-claim snapshot frozen with the sealed report.
    /// It is absent for non-native attempts, attempts that have not reached
    /// their seal point, or an unexpectedly missing native-root projection.
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationSnapshot>,
    /// The exact engine-owned composed schedule attached to this sealed report.
    /// It is retained separately from normal aggregate rows because schedule
    /// qualification needs its full ordered byte contract.
    pub(super) composed_schedule: Option<ComposedScheduleArtifact>,
    pub(super) device_ordinal: Option<usize>,
    pub(super) grant_bytes: Option<u64>,
    pub(super) admission_sequence: Option<u64>,
    pub(super) admission_wait_ns: Option<u64>,
    /// Admission-boundary timestamps on the engine's shared observation
    /// clock. They let the overload scenario prove unrelated-device progress
    /// without relying on runner task order or a host-side clock.
    pub(super) admission_requested_ns: Option<u64>,
    pub(super) admitted_ns: Option<u64>,
    /// Bounded sealed queue/overtake facts emitted by the engine-owned
    /// admission controller. They are not inferred from runner ordering.
    pub(super) admission_wait: Option<AdmissionWaitEvidence>,
    /// Ledger grant identity, when the report carried available resource
    /// evidence. `None` means the evidence field was unavailable, which is a
    /// non-qualifying result rather than a skipped assertion.
    pub(super) grant_id: Option<u64>,
    pub(super) domain_id: Option<u64>,
    pub(super) domain_pool_usage: Option<QueryDomainPoolUsageObservation>,
    pub(super) total_ns: Option<u64>,
    pub(super) queue_wait_ns: Option<u64>,
    pub(super) physical_execution_ns: Option<u64>,
    /// Submit and terminal instants on the engine's shared observation clock.
    /// Both come from the sealed report, so intervals from different attempts
    /// are comparable and the overlap witness can actually fail.
    pub(super) submitted_ns: u64,
    pub(super) outcome_terminal_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QueryDomainPoolUsageObservation {
    pub(super) reserved_high_bytes: u64,
    pub(super) used_high_bytes: u64,
}

/// The report's bounded failure classification, projected for the artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct AttemptFailureClass {
    pub(super) origin: String,
    pub(super) code: String,
    pub(super) status: String,
    pub(super) operation: String,
}

/// Bounded, typed engine graph-calibration projection associated with one
/// sealed attempt. This stays independent of admission and does not parse
/// DataFusion metric labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct GraphClaimCalibrationSnapshot {
    pub(super) truncated: bool,
    pub(super) rows: Vec<GraphClaimCalibrationRow>,
    /// The execution owner explicitly classified this graph attempt before
    /// sealing. Rows do not imply physical execution: a BFS singleflight
    /// follower intentionally has no locally measured graph row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) execution: Option<GraphClaimCalibrationExecution>,
    /// True only when exactly one non-conflicting execution classification
    /// reached the sealed snapshot.
    pub(super) execution_complete: bool,
}

/// Typed execution semantics sealed with one graph calibration snapshot.
///
/// This is the artifact-versioned serde projection. The closed vocabulary
/// itself is owned by the adapter
/// (`datafusion_nexus::attempt::GraphClaimCalibrationExecution`); this
/// wrapper stays `snake_case`-serialized so the stress artifact JSON is
/// byte-identical while the measured run converts via `From`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GraphClaimCalibrationExecution {
    Physical,
    NonPhysical(GraphClaimCalibrationNonPhysicalReason),
}

/// Closed vocabulary for graph attempts that intentionally did not execute
/// a physical graph algorithm locally.
///
/// Artifact serde shape is frozen; variant names are the wire contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GraphClaimCalibrationNonPhysicalReason {
    BfsSingleflightFollower {
        #[serde(skip_serializing_if = "Option::is_none")]
        leader_attempt_id: Option<String>,
    },
    GraphMemoryAdmissionDenied,
}

#[cfg(feature = "single-node-multi-gpu-validate-run")]
impl From<datafusion_nexus::attempt::GraphClaimCalibrationExecution> for GraphClaimCalibrationExecution {
    fn from(value: datafusion_nexus::attempt::GraphClaimCalibrationExecution) -> Self {
        match value {
            datafusion_nexus::attempt::GraphClaimCalibrationExecution::Physical => Self::Physical,
            datafusion_nexus::attempt::GraphClaimCalibrationExecution::NonPhysical(reason) => {
                Self::NonPhysical(reason.into())
            },
        }
    }
}

#[cfg(feature = "single-node-multi-gpu-validate-run")]
impl From<datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason>
    for GraphClaimCalibrationNonPhysicalReason
{
    fn from(value: datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason) -> Self {
        match value {
            datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower {
                leader_attempt_id,
            } => Self::BfsSingleflightFollower { leader_attempt_id: leader_attempt_id.map(|id| id.to_string()) },
            datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason::GraphMemoryAdmissionDenied => {
                Self::GraphMemoryAdmissionDenied
            },
        }
    }
}

/// One engine-owned graph model claim and its correlated runtime peak.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct GraphClaimCalibrationRow {
    pub(super) node_id: String,
    pub(super) algorithm: String,
    pub(super) model_requested_bytes: u64,
    pub(super) requested_bytes: u64,
    pub(super) budget_bytes: Option<u64>,
    pub(super) execution_state: String,
    pub(super) measured_peak_bytes: Option<u64>,
    pub(super) measured_peak_provenance: String,
    pub(super) graph_model_error_ratio: Option<String>,
}

/// Graph-model calibration completeness result for every measurement-phase graph attempt.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GraphClaimCalibrationEvidence {
    pub(super) status: ClauseStatus,
    pub(super) graph_attempts: u64,
    pub(super) attempts: Vec<GraphClaimCalibrationAttemptEvidence>,
}

impl GraphClaimCalibrationEvidence {
    pub(super) fn passes(&self) -> bool {
        self.status == ClauseStatus::Evidenced
    }
}

/// Per-attempt failure is intentionally a closed vocabulary rather than a
/// rendered error. The raw typed rows beside it remain the evidence.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GraphClaimCalibrationAttemptEvidence {
    pub(super) sequence: usize,
    pub(super) status: ClauseStatus,
    pub(super) reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) snapshot: Option<GraphClaimCalibrationSnapshot>,
}

/// Dedicated-probe facts projected directly from one sealed report.
///
/// JSON retains the engine's closed serde contracts while allowing the runner
/// to exclude arbitrary failure text before the artifact redaction guard runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct AttemptTerminalEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) primary_failure: Option<Value>,
    pub(super) cleanup_disposition: Value,
    pub(super) cleanup_failure_count: u64,
    pub(super) cleanup_failure_count_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) first_cleanup_failure: Option<Value>,
    pub(super) capacity_disposition: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) health_transition: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) native_fault_evidence: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) resource_evidence: Option<Value>,
    pub(super) retry_advice: Value,
}

/// Public-artifact projection of one final queue admission fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct AdmissionWaitEvidence {
    pub(super) submission_sequence: u64,
    pub(super) overtake_limit: u32,
    pub(super) overtakes_consumed: u32,
    pub(super) overtakes_remaining: u32,
    pub(super) protection_device_ordinal: Option<usize>,
}

/// Per-device occupancy peaks observed by the sampler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(super) struct DeviceOccupancyPeaks {
    pub(super) peak_active_attempts: usize,
    pub(super) peak_active_query_jobs: usize,
    pub(super) peak_live_grants_bytes: u64,
    pub(super) peak_live_reserved_bytes: u64,
    pub(super) allocation_failure_count: u64,
    /// Minimum observed logical root-stream slot availability. Profile-derived
    /// accounting, not a CUDA stream counter.
    pub(super) min_root_streams_available: usize,
    pub(super) min_root_worker_slots_available: usize,
}

/// The configured envelope of one device, as resolved at backend startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct DeviceConfiguration {
    pub(super) ordinal: usize,
    pub(super) model_name: Option<String>,
    pub(super) uuid: Option<String>,
    pub(super) physical_total_bytes: Option<u64>,
    pub(super) capacity_bytes: u64,
    /// Immutable aggregate resident-cache cap for this ordinal. A zero cap
    /// means no cache domain exists on the device.
    pub(super) cache_cap_bytes: u64,
    /// Whether the resolved backend installed diagnostic telemetry for this
    /// ordinal. It never participates in admission.
    pub(super) install_telemetry: bool,
    pub(super) max_active_attempts: usize,
    pub(super) stream_pool_size: usize,
    pub(super) worker_count: usize,
    pub(super) lane_count: usize,
    pub(super) ownership: String,
    pub(super) device_health_generation: u64,
    pub(super) starting_health_open: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct MeasuredAggregate {
    pub(super) window: MeasurementWindow,
    pub(super) throughput: ThroughputAggregate,
    pub(super) latency: LatencyAggregate,
    pub(super) placement: Vec<DevicePlacement>,
    pub(super) attempts: Vec<AttemptRow>,
    pub(super) scenarios: ScenarioEvidence,
    /// Controlled quarantine/fallback evidence, when the explicit
    /// validation capability was invoked for this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) quarantine_fallback: Option<QuarantineFallbackEvidence>,
    /// Per-step envelope aggregates. Empty for single-phase profiles, so
    /// their artifacts carry no per-step section.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) steps: Vec<StepAggregate>,
    pub(super) report_aggregate_equality: ReportAggregateEquality,
}

/// One step's envelope curve point: throughput and latency computed over
/// exactly the attempts that step measured, against its own window.
#[derive(Debug, Clone, Serialize)]
pub(super) struct StepAggregate {
    pub(super) step_index: usize,
    pub(super) concurrency: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) rate_overrun: &'static str,
    pub(super) measurement_ns: u64,
    pub(super) throughput: ThroughputAggregate,
    pub(super) latency: LatencyAggregate,
    /// Placement counts per ordinal inside this step, for balance reading.
    pub(super) placements: Vec<StepPlacement>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct StepPlacement {
    pub(super) device_ordinal: usize,
    pub(super) attempts_placed: u64,
    pub(super) attempts_succeeded: u64,
}

/// The per-step window the reducer aggregates against, projected from the
/// arrival plan so the pure reduction never sees the plan itself.
#[derive(Debug, Clone, Copy)]
pub(super) struct StepWindow {
    pub(super) step_index: usize,
    pub(super) concurrency: usize,
    pub(super) arrival_rate_per_second: Option<u64>,
    pub(super) rate_overrun: &'static str,
    pub(super) measurement_ns: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct MeasurementWindow {
    pub(super) warmup_ns: u64,
    pub(super) measurement_ns: u64,
    pub(super) warmup_attempts: u64,
    pub(super) measurement_attempts: u64,
}

/// Application QPS counts every completed measurement attempt; GPU-reaching
/// QPS counts only those that reached native execution. They differ whenever a
/// plan falls back to DataFusion CPU.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct ThroughputAggregate {
    pub(super) measurement_seconds: f64,
    pub(super) completions: u64,
    pub(super) successful_completions: u64,
    pub(super) gpu_reaching_completions: u64,
    pub(super) application_qps: f64,
    pub(super) gpu_reaching_qps: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct LatencyAggregate {
    pub(super) samples: u64,
    pub(super) min_ns: u64,
    pub(super) p50_ns: u64,
    pub(super) p95_ns: u64,
    pub(super) p99_ns: u64,
    pub(super) max_ns: u64,
    pub(super) mean_ns: u64,
    pub(super) queue_wait_p95_ns: u64,
    /// Sealed admission-controller waits, separate from execution-plan queue
    /// timing. The sample count keeps a no-admission observation distinct from
    /// a measured zero-wait fast path.
    pub(super) admission_wait_samples: u64,
    pub(super) admission_wait_p50_ns: u64,
    pub(super) admission_wait_p95_ns: u64,
    pub(super) admission_wait_p99_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct DevicePlacement {
    #[serde(flatten)]
    pub(super) configuration: DeviceConfiguration,
    pub(super) attempts_placed: u64,
    pub(super) attempts_succeeded: u64,
    pub(super) distinct_grants: u64,
    pub(super) grant_bytes_min: u64,
    pub(super) grant_bytes_max: u64,
    pub(super) occupancy: DeviceOccupancyPeaks,
    pub(super) made_progress: bool,
}

/// One row per attempt. Bounded by the profile's planned attempt count.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AttemptRow {
    pub(super) sequence: usize,
    pub(super) phase: SchedulePhase,
    pub(super) step_index: usize,
    pub(super) query_label: String,
    pub(super) attempt_id: String,
    pub(super) disposition: String,
    pub(super) outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) failure: Option<AttemptFailureClass>,
    pub(super) terminal_evidence: AttemptTerminalEvidence,
    /// Per-attempt public-report attribution check. This is a
    /// deterministic projection of the sealed facts beside it, not a second
    /// event stream or inference from runner control flow.
    pub(super) attribution: super::operability::AttemptAttribution,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationSnapshot>,
    pub(super) device_ordinal: Option<usize>,
    pub(super) grant_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) admission_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) admission_wait_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) admission_requested_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) admitted_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) admission_wait: Option<AdmissionWaitEvidence>,
    pub(super) grant_id: Option<u64>,
    pub(super) domain_id: Option<u64>,
    pub(super) total_ns: Option<u64>,
    pub(super) queue_wait_ns: Option<u64>,
    pub(super) physical_execution_ns: Option<u64>,
}

/// A single scenario clause and whether the run evidenced it.
///
/// `not_evidenced` is a first-class outcome: a clause that needs a capability
/// this phase does not own is recorded as a declared gap, never as a pass.
#[derive(Debug, Clone, Serialize)]
pub(super) struct ScenarioClause {
    pub(super) clause: &'static str,
    pub(super) status: ClauseStatus,
    pub(super) detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ClauseStatus {
    Evidenced,
    NotEvidenced,
    Failed,
}

/// Requires one unambiguous, typed model-claim/peak row for every graph
/// attempt that entered the measurement phase. A profile opts into this graph-model calibration
/// gate only for graph workload cells; non-graph attempts are deliberately
/// outside its denominator.
pub(super) fn graph_claim_calibration_evidence(observations: &[AttemptObservation]) -> GraphClaimCalibrationEvidence {
    let graph_attempts = observations
        .iter()
        .filter(|attempt| attempt.graph_workload && attempt.phase == SchedulePhase::Measurement)
        .collect::<Vec<_>>();
    let mut attempts_by_id = BTreeMap::<&str, Vec<&AttemptObservation>>::new();
    for attempt in observations.iter().filter(|attempt| attempt.graph_workload) {
        attempts_by_id.entry(&attempt.attempt_id).or_default().push(attempt);
    }
    let attempts = graph_attempts
        .iter()
        .map(|attempt| {
            let Some(snapshot) = attempt.graph_claim_calibration.as_ref() else {
                return GraphClaimCalibrationAttemptEvidence {
                    sequence: attempt.sequence,
                    status: ClauseStatus::NotEvidenced,
                    reason: "missing_sealed_graph_claim_snapshot",
                    snapshot: None,
                };
            };
            let (status, reason) = match graph_claim_calibration_reason(snapshot, attempt, &attempts_by_id) {
                Some(reason) => (ClauseStatus::NotEvidenced, reason),
                None => (ClauseStatus::Evidenced, "complete"),
            };
            GraphClaimCalibrationAttemptEvidence {
                sequence: attempt.sequence,
                status,
                reason,
                snapshot: Some(snapshot.clone()),
            }
        })
        .collect::<Vec<_>>();
    let status = if attempts.is_empty() {
        ClauseStatus::NotEvidenced
    } else if attempts.iter().all(|attempt| attempt.status == ClauseStatus::Evidenced) {
        ClauseStatus::Evidenced
    } else {
        ClauseStatus::NotEvidenced
    };
    GraphClaimCalibrationEvidence { status, graph_attempts: graph_attempts.len() as u64, attempts }
}

/// Returns the closed-vocabulary reason why one measurement-phase graph
/// attempt cannot qualify. `None` means that its physical peak or linked
/// non-physical follower semantics are complete.
fn graph_claim_calibration_reason(
    snapshot: &GraphClaimCalibrationSnapshot,
    attempt: &AttemptObservation,
    attempts_by_id: &BTreeMap<&str, Vec<&AttemptObservation>>,
) -> Option<&'static str> {
    if snapshot.truncated {
        return Some("graph_claim_snapshot_truncated");
    }
    let Some(execution) = snapshot.execution.as_ref() else {
        return Some("graph_execution_classification_missing");
    };
    if !snapshot.execution_complete {
        return Some("graph_execution_classification_conflicted");
    }
    match execution {
        GraphClaimCalibrationExecution::Physical => physical_graph_claim_calibration_reason(snapshot),
        GraphClaimCalibrationExecution::NonPhysical(
            GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower { leader_attempt_id },
        ) => {
            if !snapshot.rows.is_empty() {
                return Some("singleflight_follower_has_graph_claim_rows");
            }
            let Some(leader_attempt_id) = leader_attempt_id.as_deref() else {
                return Some("singleflight_follower_missing_leader_attempt_id");
            };
            let Some([leader]) = attempts_by_id.get(leader_attempt_id).map(Vec::as_slice) else {
                return Some("singleflight_follower_leader_not_evidenced");
            };
            let Some(leader_snapshot) = leader.graph_claim_calibration.as_ref() else {
                return Some("singleflight_follower_leader_not_evidenced");
            };
            if matches!(leader_snapshot.execution.as_ref(), Some(GraphClaimCalibrationExecution::Physical))
                && leader_snapshot.execution_complete
                && physical_graph_claim_calibration_reason(leader_snapshot).is_none()
            {
                None
            } else {
                Some("singleflight_follower_leader_not_evidenced")
            }
        },
        GraphClaimCalibrationExecution::NonPhysical(
            GraphClaimCalibrationNonPhysicalReason::GraphMemoryAdmissionDenied,
        ) => graph_memory_admission_denied_reason(snapshot, attempt),
    }
}

/// A physical execution qualifies only with one direct, non-synthetic peak.
fn physical_graph_claim_calibration_reason(snapshot: &GraphClaimCalibrationSnapshot) -> Option<&'static str> {
    if snapshot.rows.len() != 1 {
        return Some("expected_exactly_one_graph_claim_row");
    }
    let row = &snapshot.rows[0];
    if row.execution_state != "algorithm_started" {
        return Some("physical_graph_execution_state_not_started");
    }
    match (row.measured_peak_bytes, row.measured_peak_provenance.as_str()) {
        (Some(_), "reservation_high_water" | "cache_scope_high_water" | "pool_delta") => None,
        _ => Some("graph_peak_measurement_unavailable"),
    }
}

/// A graph-memory model denial can qualify without a runtime peak only when
/// the engine row proves that the executor never started and the canonical
/// attempt failure has the matching closed native reservation class.
fn graph_memory_admission_denied_reason(
    snapshot: &GraphClaimCalibrationSnapshot,
    attempt: &AttemptObservation,
) -> Option<&'static str> {
    if snapshot.rows.len() != 1 {
        return Some("expected_exactly_one_graph_claim_row");
    }
    let row = &snapshot.rows[0];
    if row.execution_state != "graph_memory_admission_denied" {
        return Some("graph_memory_admission_denied_state_missing");
    }
    if row.measured_peak_bytes.is_some() || row.measured_peak_provenance != "unavailable" {
        return Some("graph_memory_admission_denied_has_graph_peak");
    }
    let Some(budget_bytes) = row.budget_bytes else {
        return Some("graph_memory_admission_denied_budget_missing");
    };
    if row.requested_bytes <= budget_bytes {
        return Some("graph_memory_admission_denied_budget_mismatch");
    }
    match attempt.failure.as_ref() {
        Some(AttemptFailureClass { origin, code, status, operation })
            if origin == "Native"
                && code == "MemoryReservationFailed"
                && status == "Temporary"
                && operation == "Execution" =>
        {
            None
        },
        _ => Some("graph_memory_admission_denied_failure_mismatch"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ScenarioEvidence {
    pub(super) same_device: Vec<ScenarioClause>,
    pub(super) multiple_devices: Vec<ScenarioClause>,
    /// Present only for the explicit bounded-overload probe. Its facts
    /// are all sealed report projections or resolved profile limits; no queue
    /// ordering is reconstructed from runner scheduling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) overload_fairness: Option<OverloadFairnessEvidence>,
    /// Logical slot occupancy is recorded, never asserted: the ledger derives
    /// these counts from the immutable profile, so they cannot witness CUDA
    /// stream identity. Nsight owns stream-level overlap.
    pub(super) logical_slot_occupancy: LogicalSlotOccupancyEvidence,
}

/// Evidence for the explicit public bounded-overload and drain scenario.
///
/// `AttemptAdmissionWaitFact` is controller-owned and sealed with the final
/// report. The release runner uses actual public plans, which can exercise
/// the bounded queue but cannot author the unequal controller requirements
/// needed to manufacture a finite overtake. Exact finite-overtake protection
/// stays with the deterministic owning controller suite rather than being
/// inferred from workload arrival order here.
#[derive(Debug, Clone, Serialize)]
pub(super) struct OverloadFairnessEvidence {
    pub(super) status: ClauseStatus,
    pub(super) configured_active_attempt_capacity: u64,
    pub(super) configured_queue_limit: u64,
    pub(super) configured_overtake_limit: u32,
    pub(super) planned_cohort_attempts: u64,
    pub(super) sealed_attempts: u64,
    pub(super) begin_failures: u64,
    pub(super) join_failures: u64,
    pub(super) dropped_overrun_arrivals: u64,
    pub(super) bounded_overload_shape: bool,
    pub(super) all_planned_attempts_sealed: bool,
    pub(super) all_planned_attempts_admitted_and_completed: bool,
    /// Highest public queue depth observed while the finite cohort was live.
    pub(super) observed_peak_queued_attempts: u64,
    pub(super) queue_limit_reached: bool,
    pub(super) queue_bound_respected: bool,
    pub(super) queued_admission_reports: u64,
    pub(super) wait_facts_exact: bool,
    pub(super) exhausted_waiter_count: u64,
    pub(super) protected_exhausted_waiter_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) protected_waiter_submission_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) protected_waiter_admission_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) protection_device_ordinal: Option<usize>,
    /// Successful native admissions to a different ordinal while the selected
    /// protected waiter's public admission interval remained outstanding.
    pub(super) unrelated_device_progress_count: u64,
    /// The stricter subset whose sealed terminal timestamp also precedes the
    /// protected waiter's admission. This is the liveness witness required by
    /// the scenario.
    pub(super) unrelated_device_completion_count: u64,
    /// The physical public workload must nevertheless complete on more than
    /// one eligible device while the saturated cohort drains.
    pub(super) eligible_device_completion_count: u64,
}

/// Bounded evidence for one explicit controller-level device quarantine.
///
/// The pure reducer accepts only observations made at the public boundary. It
/// never claims a synthetic quarantine was a CUDA fault; the injector's
/// contract is to exercise the controller response to a device-local fault.
#[derive(Debug, Clone, Serialize)]
pub(super) struct QuarantineFallbackEvidence {
    pub(super) target_ordinal: usize,
    /// Offset from the start of the run at which the target's live work was
    /// witnessed. This must precede the controller acknowledgement below.
    pub(super) pre_ack_witness_offset_ns: u64,
    /// The target ledger sequence observed with the live grant/job. Together
    /// with the synchronous injector acknowledgement, this pins the barrier to
    /// controller state rather than the runner's submission order.
    pub(super) pre_ack_ledger_sequence: u64,
    pub(super) pre_ack_live_grants: usize,
    pub(super) pre_ack_active_query_jobs: usize,
    /// Offset at which `quarantine_device` returned successfully. The engine
    /// takes the same fleet-admission gate as placement, so this is the
    /// post-ack admission boundary for the released cohort.
    pub(super) acknowledgement_offset_ns: u64,
    pub(super) device_health_generation: u64,
    pub(super) newly_quarantined: bool,
    pub(super) target_ledger_closed: bool,
    pub(super) post_ack_attempts: u64,
    pub(super) post_ack_target_placements: u64,
    pub(super) post_ack_healthy_gpu_reaching_completions: u64,
    /// A pre-ack grant must return after its attempt has drained. This proves
    /// the transition did not strand already admitted target work.
    pub(super) pre_existing_target_work_released: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_residency: Option<CacheQuarantineEvidence>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(super) require_cache_residency: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheQuarantineEvidence {
    pub(super) pre_ack: serde_json::Value,
    pub(super) post_ack: serde_json::Value,
    pub(super) final_observation: serde_json::Value,
    pub(super) status: ClauseStatus,
}

impl CacheQuarantineEvidence {
    pub(super) fn from_snapshots(
        pre_ack: Option<serde_json::Value>,
        post_ack: Option<serde_json::Value>,
        final_observation: Option<serde_json::Value>,
    ) -> Option<Self> {
        let (Some(pre_ack), Some(post_ack), Some(final_observation)) = (pre_ack, post_ack, final_observation) else {
            return None;
        };
        let valid = [&pre_ack, &post_ack, &final_observation].iter().enumerate().all(|(index, snapshot)| {
            let Some(object) = snapshot.as_object() else { return false };
            let cap = object.get("cache_cap_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0);
            let ledger = object.get("ledger_evictable_resident_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0)
                + object.get("ledger_pinned_in_use_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0);
            let physical = object.get("physical_current_bytes").and_then(serde_json::Value::as_u64).unwrap_or(u64::MAX);
            cap > 0 && (index != 0 || ledger > 0) && ledger <= cap && physical <= ledger && physical <= cap
        });
        Some(Self {
            pre_ack,
            post_ack,
            final_observation,
            status: if valid { ClauseStatus::Evidenced } else { ClauseStatus::Failed },
        })
    }
}

/// Evidence for an ordinary query-domain cap exhaustion collected entirely
/// from sealed attempt reports. A `reservation_limit` event is intentionally
/// not relabelled as a physical allocator failure: it proves the bounded
/// domain contract and its recovery path instead.
#[derive(Debug, Clone, Serialize)]
pub(super) struct DomainExhaustionEvidence {
    /// The predicate requires exactly one failed native attempt with the
    /// canonical `CudfResourceExhausted` failure class.
    pub(super) matched_native_resource_exhaustion_reports: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) failure_sequence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) failure_device_ordinal: Option<usize>,
    pub(super) allocator_event_attempt_exact_reservation_limit: bool,
    pub(super) cleanup_completed: bool,
    pub(super) capacity_released: bool,
    /// The failure report must keep the device healthy and record no
    /// quarantined teardown, proving this ordinary OOM is not a device fault.
    pub(super) remained_healthy_without_quarantine: bool,
    pub(super) same_device_recovery_completed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recovery_sequence: Option<usize>,
}

impl DomainExhaustionEvidence {
    /// Reduces only bounded report facts that the artifact already publishes.
    /// It neither fabricates an allocator error nor infers a native failure
    /// from a runner clock or process exit code.
    pub(super) fn from_observations(observations: &[AttemptObservation]) -> Self {
        let failures = observations
            .iter()
            .filter(|attempt| {
                attempt.outcome == "failed"
                    && attempt.reached_gpu
                    && attempt.device_ordinal.is_some()
                    && attempt
                        .failure
                        .as_ref()
                        .is_some_and(|failure| failure.origin == "Native" && failure.code == "CudfResourceExhausted")
            })
            .collect::<Vec<_>>();
        let failure = (failures.len() == 1).then(|| failures[0]);
        let failure_sequence = failure.map(|attempt| attempt.sequence);
        let failure_device_ordinal = failure.and_then(|attempt| attempt.device_ordinal);
        let allocator_event_attempt_exact_reservation_limit = failure.is_some_and(|attempt| {
            attempt
                .terminal_evidence
                .resource_evidence
                .as_ref()
                .and_then(|evidence| evidence.get("allocator_event"))
                .is_some_and(|event| {
                    event.get("provenance").and_then(Value::as_str) == Some("attempt_exact")
                        && event.get("value").and_then(|value| value.get("source")).and_then(Value::as_str)
                            == Some("reservation_limit")
                })
        });
        let cleanup_completed = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.cleanup_disposition == Value::String("completed".to_string())
                && attempt.terminal_evidence.cleanup_failure_count_complete
                && attempt.terminal_evidence.cleanup_failure_count == 0
        });
        let capacity_released = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.capacity_disposition == Value::String("released".to_string())
        });
        let remained_healthy_without_quarantine = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.health_transition == Some(Value::String("healthy_reuse".to_string()))
                && attempt.terminal_evidence.native_fault_evidence.as_ref().is_some_and(|evidence| {
                    evidence.get("device_health").is_some_and(|health| {
                        health.get("status").and_then(Value::as_str) == Some("healthy")
                            && health.get("quarantined_teardown_count").and_then(Value::as_u64) == Some(0)
                    })
                })
        });
        let recovery = failure.and_then(|failed| {
            observations.iter().find(|attempt| {
                attempt.sequence > failed.sequence
                    && attempt.succeeded
                    && attempt.reached_gpu
                    && attempt.device_ordinal == failed.device_ordinal
            })
        });

        Self {
            matched_native_resource_exhaustion_reports: failures.len() as u64,
            failure_sequence,
            failure_device_ordinal,
            allocator_event_attempt_exact_reservation_limit,
            cleanup_completed,
            capacity_released,
            remained_healthy_without_quarantine,
            same_device_recovery_completed: recovery.is_some(),
            recovery_sequence: recovery.map(|attempt| attempt.sequence),
        }
    }

    pub(super) fn is_evidenced(&self) -> bool {
        self.matched_native_resource_exhaustion_reports == 1
            && self.failure_sequence.is_some()
            && self.failure_device_ordinal.is_some()
            && self.allocator_event_attempt_exact_reservation_limit
            && self.cleanup_completed
            && self.capacity_released
            && self.remained_healthy_without_quarantine
            && self.same_device_recovery_completed
    }
}

/// Bounded runner projection of one real external pressure lease.
///
/// The optional cuDF helper lives behind the measured-run feature. Keeping its
/// facts in this plain DTO lets the proof reducer and artifact schema stay
/// CPU-only and makes the helper's CUDA-sidecar contract explicit at the
/// boundary.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct PhysicalPressureLeaseFacts {
    pub(super) device_ordinal: usize,
    pub(super) method: &'static str,
    pub(super) free_before_bytes: u64,
    pub(super) free_under_pressure_bytes: u64,
    pub(super) bytes_held: u64,
    pub(super) reserve_target_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) free_after_release_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recovered_bytes: Option<u64>,
}

/// Explicit proof for a genuine allocator failure under external CUDA
/// pressure. This is separate from [`DomainExhaustionEvidence`]: a
/// `reservation_limit` result does not establish a physical allocation
/// failure, and conversely a physical event is rejected if the immutable
/// logical caps could have caused it.
#[derive(Debug, Clone, Serialize)]
pub(super) struct PhysicalAllocatorPressureEvidence {
    pub(super) status: ClauseStatus,
    pub(super) target_device_ordinal: usize,
    pub(super) pressure: PhysicalPressureLeaseFacts,
    pub(super) release: PhysicalPressureLeaseFacts,
    /// The driver creates the lease before it starts the public q01 attempt;
    /// this flag stays explicit because report timestamps do not cover an
    /// external CUDA sidecar allocation.
    pub(super) pressure_established_before_q01: bool,
    /// The driver consumes and verifies the lease before it begins recovery.
    pub(super) pressure_released_before_recovery: bool,
    pub(super) external_cuda_sidecar_pressure_established: bool,
    pub(super) pressure_release_verified: bool,
    pub(super) matched_native_resource_exhaustion_reports: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) failure_sequence: Option<usize>,
    pub(super) failure_is_q01_on_pressure_device: bool,
    pub(super) allocator_event_attempt_exact_physical: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) observed_physical_request_bytes: Option<u64>,
    /// Diagnostic comparison against the aggregate CUDA free-memory snapshot.
    /// A false value does not disprove an exact physical allocator decline:
    /// CUDA's total free bytes do not establish contiguous allocatable capacity.
    pub(super) physical_request_exceeds_free_under_pressure: bool,
    pub(super) grant_cap_exceeds_physical_request: bool,
    pub(super) domain_cap_exceeds_physical_request: bool,
    pub(super) manager_cap_exceeds_physical_request: bool,
    pub(super) cleanup_completed: bool,
    pub(super) capacity_released: bool,
    pub(super) remained_healthy_without_quarantine: bool,
    pub(super) same_device_recovery_completed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recovery_sequence: Option<usize>,
    pub(super) final_close_quiescent: bool,
}

impl PhysicalAllocatorPressureEvidence {
    #[expect(
        clippy::too_many_arguments,
        reason = "validation measurement aggregation; each parameter is one independently reported measurement axis"
    )]
    pub(super) fn from_observations(
        pressure: PhysicalPressureLeaseFacts,
        release: PhysicalPressureLeaseFacts,
        pressure_established_before_q01: bool,
        pressure_released_before_recovery: bool,
        observations: &[AttemptObservation],
    ) -> Self {
        let target_device_ordinal = pressure.device_ordinal;
        let external_cuda_sidecar_pressure_established = pressure.method == "cuda_on_sidecar_fixed_chunks"
            && pressure.free_under_pressure_bytes < pressure.free_before_bytes
            && pressure.bytes_held > 0;
        let pressure_facts_match = release.device_ordinal == pressure.device_ordinal
            && release.method == pressure.method
            && release.free_before_bytes == pressure.free_before_bytes
            && release.free_under_pressure_bytes == pressure.free_under_pressure_bytes
            && release.bytes_held == pressure.bytes_held
            && release.reserve_target_bytes == pressure.reserve_target_bytes;
        let pressure_release_verified = pressure_facts_match
            && release
                .free_after_release_bytes
                .is_some_and(|free_after_release| free_after_release > pressure.free_under_pressure_bytes)
            && release.recovered_bytes.is_some_and(|recovered| recovered > 0);

        let failures = observations
            .iter()
            .filter(|attempt| {
                attempt.outcome == "failed"
                    && attempt.reached_gpu
                    && attempt.device_ordinal == Some(target_device_ordinal)
                    && attempt
                        .failure
                        .as_ref()
                        .is_some_and(|failure| failure.origin == "Native" && failure.code == "CudfResourceExhausted")
            })
            .collect::<Vec<_>>();
        let failure = (failures.len() == 1).then(|| failures[0]);
        let failure_sequence = failure.map(|attempt| attempt.sequence);
        let failure_is_q01_on_pressure_device = failure.is_some_and(|attempt| {
            attempt.sequence == 0
                && attempt.query_label == "q01"
                && attempt.device_ordinal == Some(target_device_ordinal)
        });
        let allocator_event = failure
            .and_then(|attempt| attempt.terminal_evidence.resource_evidence.as_ref())
            .and_then(|evidence| evidence.get("allocator_event"));
        let allocator_event_attempt_exact_physical = allocator_event.is_some_and(|event| {
            event.get("provenance").and_then(Value::as_str) == Some("attempt_exact")
                && event.get("value").and_then(|value| value.get("source")).and_then(Value::as_str) == Some("physical")
        });
        let observed_physical_request_bytes = allocator_event_attempt_exact_physical
            .then(|| {
                allocator_event
                    .and_then(|event| event.get("value"))
                    .and_then(|value| value.get("requested_bytes"))
                    .and_then(Value::as_u64)
            })
            .flatten();
        let (grant_cap_bytes, domain_cap_bytes, manager_cap_bytes) = failure
            .and_then(|attempt| attempt.terminal_evidence.resource_evidence.as_ref())
            .map_or((None, None, None), |evidence| {
                (
                    attempt_exact_u64(evidence, "grant_cap_bytes"),
                    attempt_exact_u64(evidence, "domain_cap_bytes"),
                    attempt_exact_u64(evidence, "managed_capacity_bytes"),
                )
            });
        let physical_request_exceeds_free_under_pressure =
            observed_physical_request_bytes.is_some_and(|request| request > pressure.free_under_pressure_bytes);
        let grant_cap_exceeds_physical_request =
            observed_physical_request_bytes.zip(grant_cap_bytes).is_some_and(|(request, cap)| cap > request);
        let domain_cap_exceeds_physical_request =
            observed_physical_request_bytes.zip(domain_cap_bytes).is_some_and(|(request, cap)| cap > request);
        let manager_cap_exceeds_physical_request =
            observed_physical_request_bytes.zip(manager_cap_bytes).is_some_and(|(request, cap)| cap > request);
        let cleanup_completed = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.cleanup_disposition == Value::String("completed".to_string())
                && attempt.terminal_evidence.cleanup_failure_count_complete
                && attempt.terminal_evidence.cleanup_failure_count == 0
        });
        let capacity_released = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.capacity_disposition == Value::String("released".to_string())
        });
        let remained_healthy_without_quarantine = failure.is_some_and(|attempt| {
            attempt.terminal_evidence.health_transition == Some(Value::String("healthy_reuse".to_string()))
                && attempt.terminal_evidence.native_fault_evidence.as_ref().is_some_and(|evidence| {
                    evidence.get("device_health").is_some_and(|health| {
                        health.get("status").and_then(Value::as_str) == Some("healthy")
                            && health.get("quarantined_teardown_count").and_then(Value::as_u64) == Some(0)
                    })
                })
        });
        let recovery = failure.and_then(|failed| {
            observations.iter().find(|attempt| {
                attempt.sequence > failed.sequence
                    && attempt.query_label == "edge_scan"
                    && attempt.succeeded
                    && attempt.reached_gpu
                    && attempt.device_ordinal == Some(target_device_ordinal)
            })
        });

        let mut evidence = Self {
            status: ClauseStatus::Failed,
            target_device_ordinal,
            pressure,
            release,
            pressure_established_before_q01,
            pressure_released_before_recovery,
            external_cuda_sidecar_pressure_established,
            pressure_release_verified,
            matched_native_resource_exhaustion_reports: failures.len() as u64,
            failure_sequence,
            failure_is_q01_on_pressure_device,
            allocator_event_attempt_exact_physical,
            observed_physical_request_bytes,
            physical_request_exceeds_free_under_pressure,
            grant_cap_exceeds_physical_request,
            domain_cap_exceeds_physical_request,
            manager_cap_exceeds_physical_request,
            cleanup_completed,
            capacity_released,
            remained_healthy_without_quarantine,
            same_device_recovery_completed: recovery.is_some(),
            recovery_sequence: recovery.map(|attempt| attempt.sequence),
            final_close_quiescent: false,
        };
        evidence.status = if evidence.is_evidenced() { ClauseStatus::Evidenced } else { ClauseStatus::Failed };
        evidence
    }

    pub(super) fn with_final_close_quiescent(mut self, final_close_quiescent: bool) -> Self {
        self.final_close_quiescent = final_close_quiescent;
        self.status = if self.is_evidenced() { ClauseStatus::Evidenced } else { ClauseStatus::Failed };
        self
    }

    pub(super) fn is_evidenced(&self) -> bool {
        self.pressure_established_before_q01
            && self.pressure_released_before_recovery
            && self.external_cuda_sidecar_pressure_established
            && self.pressure_release_verified
            && self.matched_native_resource_exhaustion_reports == 1
            && self.failure_sequence == Some(0)
            && self.failure_is_q01_on_pressure_device
            && self.allocator_event_attempt_exact_physical
            && self.observed_physical_request_bytes.is_some()
            && self.grant_cap_exceeds_physical_request
            && self.domain_cap_exceeds_physical_request
            && self.manager_cap_exceeds_physical_request
            && self.cleanup_completed
            && self.capacity_released
            && self.remained_healthy_without_quarantine
            && self.same_device_recovery_completed
            && self.recovery_sequence.is_some()
            && self.final_close_quiescent
    }
}

fn attempt_exact_u64(evidence: &Value, field: &str) -> Option<u64> {
    let field = evidence.get(field)?;
    (field.get("provenance").and_then(Value::as_str) == Some("attempt_exact"))
        .then(|| field.get("value").and_then(Value::as_u64))
        .flatten()
}

impl OverloadFairnessEvidence {
    #[expect(
        clippy::too_many_arguments,
        reason = "validation measurement aggregation; each parameter is one independently reported measurement axis"
    )]
    pub(super) fn from_observations(
        observations: &[AttemptObservation],
        devices: &[DeviceConfiguration],
        planned_cohort_attempts: usize,
        configured_queue_limit: usize,
        configured_overtake_limit: u32,
        observed_peak_queued_attempts: usize,
        dropped_overrun_arrivals: u64,
        begin_failures: u64,
        join_failures: u64,
    ) -> Self {
        let configured_active_attempt_capacity =
            devices.iter().fold(0_u64, |total, device| total.saturating_add(device.max_active_attempts as u64));
        let configured_queue_limit_usize = configured_queue_limit;
        let configured_queue_limit = configured_queue_limit_usize as u64;
        let planned_cohort_attempts = planned_cohort_attempts as u64;
        let bounded_overload_shape = planned_cohort_attempts > configured_active_attempt_capacity
            && planned_cohort_attempts <= configured_active_attempt_capacity.saturating_add(configured_queue_limit);
        let sealed_attempts = observations.len() as u64;
        let all_planned_attempts_sealed = sealed_attempts == planned_cohort_attempts
            && dropped_overrun_arrivals == 0
            && begin_failures == 0
            && join_failures == 0;
        let all_planned_attempts_admitted_and_completed = all_planned_attempts_sealed
            && observations.iter().all(|attempt| {
                attempt.succeeded
                    && attempt.reached_gpu
                    && attempt.device_ordinal.is_some()
                    && attempt.admission_sequence.is_some()
            });
        let eligible_device_completion_count = observations
            .iter()
            .filter(|attempt| attempt.succeeded && attempt.reached_gpu)
            .filter_map(|attempt| attempt.device_ordinal)
            .collect::<BTreeSet<_>>()
            .len() as u64;

        let mut submission_sequences = BTreeSet::new();
        let queued = observations
            .iter()
            .filter_map(|attempt| attempt.admission_wait.as_ref().map(|wait| (attempt, wait)))
            .collect::<Vec<_>>();
        let wait_facts_exact = !queued.is_empty()
            && queued.iter().all(|(attempt, wait)| {
                submission_sequences.insert(wait.submission_sequence)
                    && wait.overtake_limit > 0
                    && wait.overtake_limit == configured_overtake_limit
                    && wait.overtakes_consumed <= wait.overtake_limit
                    && wait.overtakes_remaining <= wait.overtake_limit
                    && wait
                        .overtakes_consumed
                        .checked_add(wait.overtakes_remaining)
                        .is_some_and(|total| total == wait.overtake_limit)
                    && attempt.admission_sequence.is_some()
                    && attempt.admission_wait_ns.is_some()
                    && attempt
                        .admission_requested_ns
                        .zip(attempt.admitted_ns)
                        .is_some_and(|(requested, admitted)| requested <= admitted)
                    && wait
                        .protection_device_ordinal
                        .is_none_or(|ordinal| devices.iter().any(|device| device.ordinal == ordinal))
            });
        let exhausted_waiters = queued
            .iter()
            .filter(|(_, wait)| wait.overtakes_consumed == wait.overtake_limit && wait.overtakes_remaining == 0)
            .collect::<Vec<_>>();
        let protected_waiters = exhausted_waiters
            .iter()
            .filter_map(|(attempt, wait)| {
                let protection_ordinal = wait.protection_device_ordinal?;
                (attempt.device_ordinal == Some(protection_ordinal)).then_some((
                    protection_ordinal,
                    wait.submission_sequence,
                    attempt.admission_sequence?,
                    attempt.admission_requested_ns?,
                    attempt.admitted_ns?,
                    *attempt,
                ))
            })
            .collect::<Vec<_>>();
        let protected_waiter =
            protected_waiters.iter().min_by_key(|(_, submission_sequence, _, _, _, _)| *submission_sequence).copied();
        let (protected_waiter_submission_sequence, protected_waiter_admission_sequence, protection_device_ordinal) =
            protected_waiter.map_or((None, None, None), |(ordinal, submission, admission, _, _, _)| {
                (Some(submission), Some(admission), Some(ordinal))
            });
        let (unrelated_device_progress_count, unrelated_device_completion_count) = protected_waiter.map_or(
            (0, 0),
            |(protection_ordinal, _, _, requested_ns, admitted_ns, protected_attempt)| {
                observations
                    .iter()
                    .filter(|candidate| {
                        !std::ptr::eq(*candidate, protected_attempt)
                            && candidate.succeeded
                            && candidate.reached_gpu
                            && candidate.device_ordinal.is_some_and(|ordinal| ordinal != protection_ordinal)
                            && candidate.admitted_ns.is_some_and(|candidate_admitted_ns| {
                                candidate_admitted_ns >= requested_ns && candidate_admitted_ns < admitted_ns
                            })
                    })
                    .fold((0_u64, 0_u64), |(progress, completion), candidate| {
                        (
                            progress.saturating_add(1),
                            completion.saturating_add(u64::from(candidate.outcome_terminal_ns < admitted_ns)),
                        )
                    })
            },
        );

        let mut evidence = Self {
            status: ClauseStatus::Failed,
            configured_active_attempt_capacity,
            configured_queue_limit,
            configured_overtake_limit,
            planned_cohort_attempts,
            sealed_attempts,
            begin_failures,
            join_failures,
            dropped_overrun_arrivals,
            bounded_overload_shape,
            all_planned_attempts_sealed,
            all_planned_attempts_admitted_and_completed,
            observed_peak_queued_attempts: observed_peak_queued_attempts as u64,
            queue_limit_reached: observed_peak_queued_attempts == configured_queue_limit_usize,
            queue_bound_respected: observed_peak_queued_attempts <= configured_queue_limit_usize,
            queued_admission_reports: queued.len() as u64,
            wait_facts_exact,
            exhausted_waiter_count: exhausted_waiters.len() as u64,
            protected_exhausted_waiter_count: protected_waiters.len() as u64,
            protected_waiter_submission_sequence,
            protected_waiter_admission_sequence,
            protection_device_ordinal,
            unrelated_device_progress_count,
            unrelated_device_completion_count,
            eligible_device_completion_count,
        };
        evidence.status = if evidence.is_evidenced() { ClauseStatus::Evidenced } else { ClauseStatus::Failed };
        evidence
    }

    pub(super) fn is_evidenced(&self) -> bool {
        self.bounded_overload_shape
            && self.all_planned_attempts_sealed
            && self.all_planned_attempts_admitted_and_completed
            && self.queue_limit_reached
            && self.queue_bound_respected
            && self.queued_admission_reports > 0
            && self.wait_facts_exact
            && self.eligible_device_completion_count > 1
    }
}

impl QuarantineFallbackEvidence {
    pub(super) fn require_cache_residency(&mut self) {
        self.require_cache_residency = true;
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "validation measurement aggregation; each parameter is one independently reported measurement axis"
    )]
    pub(super) fn from_barrier(
        target_ordinal: usize,
        pre_ack_witness_offset_ns: u64,
        pre_ack_ledger_sequence: u64,
        pre_ack_live_grants: usize,
        pre_ack_active_query_jobs: usize,
        acknowledgement_offset_ns: u64,
        device_health_generation: u64,
        newly_quarantined: bool,
        target_ledger_closed: bool,
        post_ack_sequences: &BTreeSet<usize>,
        pre_existing_target_work_released: bool,
        cache_residency: Option<CacheQuarantineEvidence>,
        observations: &[AttemptObservation],
    ) -> Self {
        let post_ack_attempts =
            observations.iter().filter(|attempt| post_ack_sequences.contains(&attempt.sequence)).count() as u64;
        let post_ack_target_placements = observations
            .iter()
            .filter(|attempt| {
                post_ack_sequences.contains(&attempt.sequence) && attempt.device_ordinal == Some(target_ordinal)
            })
            .count() as u64;
        let post_ack_healthy_gpu_reaching_completions = observations
            .iter()
            .filter(|attempt| {
                post_ack_sequences.contains(&attempt.sequence)
                    && attempt.succeeded
                    && attempt.reached_gpu
                    && attempt.device_ordinal.is_some_and(|ordinal| ordinal != target_ordinal)
            })
            .count() as u64;
        Self {
            target_ordinal,
            pre_ack_witness_offset_ns,
            pre_ack_ledger_sequence,
            pre_ack_live_grants,
            pre_ack_active_query_jobs,
            acknowledgement_offset_ns,
            device_health_generation,
            newly_quarantined,
            target_ledger_closed,
            post_ack_attempts,
            post_ack_target_placements,
            post_ack_healthy_gpu_reaching_completions,
            pre_existing_target_work_released,
            cache_residency,
            require_cache_residency: false,
        }
    }

    fn clause(&self) -> ScenarioClause {
        if self.require_cache_residency
            && self.cache_residency.as_ref().is_none_or(|e| e.status != ClauseStatus::Evidenced)
        {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                "cache residency evidence missing or invalid".to_string(),
            );
        }
        if self.pre_ack_witness_offset_ns >= self.acknowledgement_offset_ns {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                "controlled quarantine acknowledgement did not follow a target live-work witness".to_string(),
            );
        }
        if self.pre_ack_live_grants == 0 && self.pre_ack_active_query_jobs == 0 {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                format!(
                    "ordinal {} had neither a live grant nor a live query job before controlled quarantine",
                    self.target_ordinal
                ),
            );
        }
        if !self.newly_quarantined {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                format!("controlled quarantine did not transition ordinal {} from open to closed", self.target_ordinal),
            );
        }
        if !self.target_ledger_closed {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                format!("ordinal {} was not observed closed after controlled quarantine", self.target_ordinal),
            );
        }
        if self.post_ack_attempts == 0 {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                "no post-acknowledgement cohort was released".to_string(),
            );
        }
        if self.post_ack_target_placements != 0 {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                format!(
                    "{} post-acknowledgement attempts were placed on quarantined ordinal {}",
                    self.post_ack_target_placements, self.target_ordinal
                ),
            );
        }
        if self.post_ack_healthy_gpu_reaching_completions == 0 {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                "no post-acknowledgement attempt completed on a healthy GPU device".to_string(),
            );
        }
        if !self.pre_existing_target_work_released {
            return clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::Failed,
                format!(
                    "pre-acknowledgement work on quarantined ordinal {} did not release before close",
                    self.target_ordinal
                ),
            );
        }
        clause(
            "device_block_or_quarantine_fallback",
            ClauseStatus::Evidenced,
            format!(
                "ordinal {} had {} live grants and {} live query jobs at ledger sequence {}, then closed at health generation {}; {} post-acknowledgement attempts completed on healthy devices",
                self.target_ordinal,
                self.pre_ack_live_grants,
                self.pre_ack_active_query_jobs,
                self.pre_ack_ledger_sequence,
                self.device_health_generation,
                self.post_ack_healthy_gpu_reaching_completions,
            ),
        )
    }
}

/// Repeated-quarantine evidence: each cycle reuses the single-quarantine
/// barrier semantics against a rotation target, and the section evidences
/// only when every configured cycle landed with the fleet stabilizing on the
/// remaining ordinals afterwards.
#[derive(Debug, Clone, Serialize)]
pub(super) struct QuarantineCyclesEvidence {
    pub(super) status: ClauseStatus,
    pub(super) configured_cycles: u64,
    pub(super) completed_cycles: u64,
    pub(super) cycles: Vec<QuarantineCycleRow>,
}

impl QuarantineCyclesEvidence {
    pub(super) fn from_cycle_evidence(configured_cycles: usize, cycles: Vec<QuarantineFallbackEvidence>) -> Self {
        let rows = cycles
            .into_iter()
            .enumerate()
            .map(|(cycle_index, evidence)| {
                let clause = evidence.clause();
                QuarantineCycleRow {
                    cycle_index,
                    target_ordinal: evidence.target_ordinal,
                    device_health_generation: evidence.device_health_generation,
                    status: clause.status,
                    detail: clause.detail,
                    evidence,
                }
            })
            .collect::<Vec<_>>();
        let status = if rows.len() == configured_cycles && rows.iter().all(|row| row.status == ClauseStatus::Evidenced)
        {
            ClauseStatus::Evidenced
        } else {
            ClauseStatus::Failed
        };
        Self {
            status,
            configured_cycles: configured_cycles.min(u64::MAX as usize) as u64,
            completed_cycles: rows.len().min(u64::MAX as usize) as u64,
            cycles: rows,
        }
    }

    pub(super) fn is_evidenced(&self) -> bool {
        self.status == ClauseStatus::Evidenced
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct QuarantineCycleRow {
    pub(super) cycle_index: usize,
    pub(super) target_ordinal: usize,
    /// The target's health generation after its quarantine landed. Each
    /// cycle's target is distinct, so the per-cycle generation increment is
    /// recorded on its own ordinal.
    pub(super) device_health_generation: u64,
    pub(super) status: ClauseStatus,
    pub(super) detail: String,
    pub(super) evidence: QuarantineFallbackEvidence,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct LogicalSlotOccupancyEvidence {
    pub(super) basis: &'static str,
    pub(super) proves_gpu_overlap: bool,
    pub(super) stream_identity_source: &'static str,
    pub(super) per_device: BTreeMap<String, DeviceOccupancyPeaks>,
}

/// Equality between the artifact's published per-attempt rows and the summary
/// aggregates beside them.
///
/// Both sides are counted independently: the `rows_*` figures are recounted
/// from the emitted [`AttemptRow`]s, while the `aggregate_*` figures come from
/// the throughput and placement reductions over the observations. A reduction
/// that dropped, duplicated, or misattributed an attempt breaks the equality
/// instead of hiding inside a single shared expression.
#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct ReportAggregateEquality {
    pub(super) rows_measurement_attempts: u64,
    pub(super) aggregate_measurement_attempts: u64,
    pub(super) rows_succeeded: u64,
    pub(super) aggregate_succeeded: u64,
    pub(super) rows_placed: u64,
    pub(super) aggregate_placed: u64,
    pub(super) rows_total: u64,
    pub(super) aggregate_total: u64,
    pub(super) equal: bool,
}

impl MeasuredAggregate {
    #[cfg(test)]
    pub(super) fn reduce(
        observations: &[AttemptObservation],
        devices: &[DeviceConfiguration],
        occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
        warmup_ns: u64,
        measurement_ns: u64,
    ) -> Self {
        Self::reduce_full(observations, devices, occupancy, warmup_ns, measurement_ns, None, &[])
    }

    /// Reduces ordinary measured observations plus the optional explicit
    /// quarantine transition. Keeping the input as a small observation DTO
    /// preserves CPU-only unit tests and keeps GPU interaction in `measure`.
    #[cfg(test)]
    pub(super) fn reduce_with_quarantine(
        observations: &[AttemptObservation],
        devices: &[DeviceConfiguration],
        occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
        warmup_ns: u64,
        measurement_ns: u64,
        quarantine_fallback: Option<QuarantineFallbackEvidence>,
    ) -> Self {
        Self::reduce_full(observations, devices, occupancy, warmup_ns, measurement_ns, quarantine_fallback, &[])
    }

    /// The complete reduction. `steps` carries the per-step windows; an
    /// empty slice is a single-phase run and emits no per-step section.
    #[expect(
        clippy::too_many_arguments,
        reason = "validation measurement aggregation; each parameter is one independently reported measurement axis"
    )]
    pub(super) fn reduce_full(
        observations: &[AttemptObservation],
        devices: &[DeviceConfiguration],
        occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
        warmup_ns: u64,
        measurement_ns: u64,
        quarantine_fallback: Option<QuarantineFallbackEvidence>,
        steps: &[StepWindow],
    ) -> Self {
        let measured =
            observations.iter().filter(|attempt| attempt.phase == SchedulePhase::Measurement).collect::<Vec<_>>();
        let throughput = throughput(&measured, measurement_ns);
        let latency = latency(&measured);
        let placement = placement(observations, devices, occupancy);
        let attempts = observations.iter().map(row).collect::<Vec<_>>();
        let scenarios =
            scenarios(observations, &measured, devices, &placement, occupancy, quarantine_fallback.as_ref());
        let window = MeasurementWindow {
            warmup_ns,
            measurement_ns,
            warmup_attempts: (observations.len() - measured.len()) as u64,
            measurement_attempts: measured.len() as u64,
        };
        let steps = steps.iter().map(|window| step_aggregate(window, observations)).collect();
        let report_aggregate_equality = equality(&window, &placement, &throughput, &attempts);
        Self {
            window,
            throughput,
            latency,
            placement,
            attempts,
            scenarios,
            quarantine_fallback,
            steps,
            report_aggregate_equality,
        }
    }

    /// Attaches the one explicit overload probe after its driver supplies the
    /// finite-cohort counters. The aggregate remains reusable for ordinary runs
    /// and cache calibrations that did not request this scenario.
    pub(super) fn with_overload_fairness(mut self, evidence: OverloadFairnessEvidence) -> Self {
        self.scenarios.overload_fairness = Some(evidence);
        self
    }
}

/// Per-step throughput/latency over exactly the attempts the step measured.
/// Warmup arrivals belong to no step's measurement; step windows are
/// disjoint measurement phases, so their union is the run's measurement set.
fn step_aggregate(window: &StepWindow, observations: &[AttemptObservation]) -> StepAggregate {
    let measured = observations
        .iter()
        .filter(|attempt| attempt.phase == SchedulePhase::Measurement && attempt.step_index == window.step_index)
        .collect::<Vec<_>>();
    let mut placements: BTreeMap<usize, StepPlacement> = BTreeMap::new();
    for attempt in observations.iter().filter(|attempt| attempt.step_index == window.step_index) {
        let Some(ordinal) = attempt.device_ordinal else {
            continue;
        };
        let entry = placements.entry(ordinal).or_insert(StepPlacement {
            device_ordinal: ordinal,
            attempts_placed: 0,
            attempts_succeeded: 0,
        });
        entry.attempts_placed += 1;
        entry.attempts_succeeded += u64::from(attempt.succeeded);
    }
    StepAggregate {
        step_index: window.step_index,
        concurrency: window.concurrency,
        arrival_rate_per_second: window.arrival_rate_per_second,
        rate_overrun: window.rate_overrun,
        measurement_ns: window.measurement_ns,
        throughput: throughput(&measured, window.measurement_ns),
        latency: latency(&measured),
        placements: placements.into_values().collect(),
    }
}

fn throughput(measured: &[&AttemptObservation], measurement_ns: u64) -> ThroughputAggregate {
    let seconds = measurement_ns as f64 / 1_000_000_000_f64;
    let completions = measured.len() as u64;
    let successful = measured.iter().filter(|attempt| attempt.succeeded).count() as u64;
    let gpu_reaching = measured.iter().filter(|attempt| attempt.succeeded && attempt.reached_gpu).count() as u64;
    let rate = |count: u64| if seconds > 0.0 { count as f64 / seconds } else { 0.0 };
    ThroughputAggregate {
        measurement_seconds: seconds,
        completions,
        successful_completions: successful,
        gpu_reaching_completions: gpu_reaching,
        application_qps: rate(successful),
        gpu_reaching_qps: rate(gpu_reaching),
    }
}

fn latency(measured: &[&AttemptObservation]) -> LatencyAggregate {
    let mut totals = measured.iter().filter_map(|attempt| attempt.total_ns).collect::<Vec<_>>();
    totals.sort_unstable();
    let mut queue_waits = measured.iter().filter_map(|attempt| attempt.queue_wait_ns).collect::<Vec<_>>();
    queue_waits.sort_unstable();
    let mut admission_waits = measured.iter().filter_map(|attempt| attempt.admission_wait_ns).collect::<Vec<_>>();
    admission_waits.sort_unstable();
    let sum: u128 = totals.iter().map(|value| u128::from(*value)).sum();
    LatencyAggregate {
        samples: totals.len() as u64,
        min_ns: totals.first().copied().unwrap_or_default(),
        p50_ns: percentile(&totals, 50.0),
        p95_ns: percentile(&totals, 95.0),
        p99_ns: percentile(&totals, 99.0),
        max_ns: totals.last().copied().unwrap_or_default(),
        mean_ns: if totals.is_empty() { 0 } else { (sum / totals.len() as u128) as u64 },
        queue_wait_p95_ns: percentile(&queue_waits, 95.0),
        admission_wait_samples: admission_waits.len() as u64,
        admission_wait_p50_ns: percentile(&admission_waits, 50.0),
        admission_wait_p95_ns: percentile(&admission_waits, 95.0),
        admission_wait_p99_ns: percentile(&admission_waits, 99.0),
    }
}

/// Nearest-rank percentile over a sorted slice. Raw samples are retained, so
/// this needs no bucket calibration; bucket suitability is checked by the metrics audit.
fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percentile / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn placement(
    observations: &[AttemptObservation],
    devices: &[DeviceConfiguration],
    occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
) -> Vec<DevicePlacement> {
    devices
        .iter()
        .map(|configuration| {
            let placed = observations
                .iter()
                .filter(|attempt| attempt.device_ordinal == Some(configuration.ordinal))
                .collect::<Vec<_>>();
            let succeeded = placed.iter().filter(|attempt| attempt.succeeded).count() as u64;
            let grants = placed.iter().filter_map(|attempt| attempt.grant_bytes).collect::<Vec<_>>();
            DevicePlacement {
                configuration: configuration.clone(),
                attempts_placed: placed.len() as u64,
                attempts_succeeded: succeeded,
                distinct_grants: placed.iter().filter_map(|attempt| attempt.grant_id).collect::<BTreeSet<_>>().len()
                    as u64,
                grant_bytes_min: grants.iter().copied().min().unwrap_or_default(),
                grant_bytes_max: grants.iter().copied().max().unwrap_or_default(),
                occupancy: occupancy.get(&configuration.ordinal).copied().unwrap_or_default(),
                made_progress: succeeded > 0,
            }
        })
        .collect()
}

fn row(attempt: &AttemptObservation) -> AttemptRow {
    AttemptRow {
        sequence: attempt.sequence,
        phase: attempt.phase,
        step_index: attempt.step_index,
        query_label: attempt.query_label.clone(),
        attempt_id: attempt.attempt_id.clone(),
        disposition: attempt.disposition.clone(),
        outcome: attempt.outcome.clone(),
        failure: attempt.failure.clone(),
        terminal_evidence: attempt.terminal_evidence.clone(),
        attribution: super::operability::attribution_for(attempt),
        graph_claim_calibration: attempt.graph_claim_calibration.clone(),
        device_ordinal: attempt.device_ordinal,
        grant_bytes: attempt.grant_bytes,
        admission_sequence: attempt.admission_sequence,
        admission_wait_ns: attempt.admission_wait_ns,
        admission_requested_ns: attempt.admission_requested_ns,
        admitted_ns: attempt.admitted_ns,
        admission_wait: attempt.admission_wait.clone(),
        grant_id: attempt.grant_id,
        domain_id: attempt.domain_id,
        total_ns: attempt.total_ns,
        queue_wait_ns: attempt.queue_wait_ns,
        physical_execution_ns: attempt.physical_execution_ns,
    }
}

fn scenarios(
    observations: &[AttemptObservation],
    measured: &[&AttemptObservation],
    devices: &[DeviceConfiguration],
    placement: &[DevicePlacement],
    occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
    quarantine_fallback: Option<&QuarantineFallbackEvidence>,
) -> ScenarioEvidence {
    ScenarioEvidence {
        same_device: same_device_clauses(observations, placement),
        multiple_devices: multiple_device_clauses(measured, devices, placement, quarantine_fallback),
        overload_fairness: None,
        logical_slot_occupancy: LogicalSlotOccupancyEvidence {
            basis: "profile_derived_logical_slots",
            proves_gpu_overlap: false,
            stream_identity_source: "nsight_systems_trace",
            per_device: occupancy.iter().map(|(ordinal, peaks)| (ordinal.to_string(), *peaks)).collect(),
        },
    }
}

fn same_device_clauses(observations: &[AttemptObservation], placement: &[DevicePlacement]) -> Vec<ScenarioClause> {
    // Scenario 1 asks whether *a* device carried concurrent attempts, so select
    // the ordinal with the highest witnessed occupancy rather than the one that
    // merely placed the most. Ties break on attempts placed and then on ordinal
    // so the selection is deterministic across runs.
    let busiest = placement.iter().max_by_key(|device| {
        (device.occupancy.peak_active_query_jobs, device.attempts_placed, usize::MAX - device.configuration.ordinal)
    });
    let Some(busiest) = busiest.filter(|device| device.attempts_placed >= 2) else {
        return vec![clause(
            "multiple_admitted_attempts_on_one_device",
            ClauseStatus::NotEvidenced,
            "no ordinal placed two or more attempts".to_string(),
        )];
    };
    if busiest.configuration.max_active_attempts < 2 {
        return vec![clause(
            "multiple_admitted_attempts_on_one_device",
            ClauseStatus::NotEvidenced,
            format!("ordinal {} has serial attempt capacity", busiest.configuration.ordinal),
        )];
    }
    let ordinal = busiest.configuration.ordinal;
    let on_device = observations.iter().filter(|attempt| attempt.device_ordinal == Some(ordinal)).collect::<Vec<_>>();

    let mut clauses = vec![clause(
        "multiple_admitted_attempts_on_one_device",
        ClauseStatus::Evidenced,
        format!("ordinal {ordinal} placed {} attempts", busiest.attempts_placed),
    )];
    clauses.push(distinctness_clause("distinct_grants", &on_device, |attempt| attempt.grant_id, ordinal));
    clauses.push(distinctness_clause("distinct_domains", &on_device, |attempt| attempt.domain_id, ordinal));
    clauses.push(overlap_clause(&on_device, ordinal));
    clauses.push(occupancy_witness_clause(busiest, ordinal));
    clauses.push(clause(
        "nsight_confirmed_overlapping_cuda_ranges",
        ClauseStatus::NotEvidenced,
        "an external nsys capture supplies stream-level overlap; this runner records the trace identity only"
            .to_string(),
    ));
    clauses
}

/// Every attempt on the ordinal must carry an available identity and all of
/// them must be pairwise distinct. An unavailable evidence field fails the
/// clause rather than shrinking the sample.
fn distinctness_clause(
    name: &'static str,
    on_device: &[&AttemptObservation],
    identity: impl Fn(&AttemptObservation) -> Option<u64>,
    ordinal: usize,
) -> ScenarioClause {
    let unavailable = on_device.iter().filter(|attempt| identity(attempt).is_none()).count();
    if unavailable > 0 {
        return clause(
            name,
            ClauseStatus::Failed,
            format!("{unavailable} of {} attempts on ordinal {ordinal} carried no identity", on_device.len()),
        );
    }
    let identities = on_device.iter().filter_map(|attempt| identity(attempt)).collect::<BTreeSet<_>>();
    if identities.len() == on_device.len() {
        clause(name, ClauseStatus::Evidenced, format!("{} distinct on ordinal {ordinal}", identities.len()))
    } else {
        clause(
            name,
            ClauseStatus::Failed,
            format!("{} attempts on ordinal {ordinal} shared {} identities", on_device.len(), identities.len()),
        )
    }
}

/// Lifetime overlap between two attempts on the same ordinal, measured on the
/// engine's shared observation clock. A witness of concurrent admission and
/// residency, not proof of GPU kernel overlap.
///
/// Overlap requires strictly positive intersection, so two attempts that merely
/// share a submit instant and then run back to back do not qualify.
fn overlap_clause(on_device: &[&AttemptObservation], ordinal: usize) -> ScenarioClause {
    let overlapping = on_device
        .iter()
        .enumerate()
        .flat_map(|(index, first)| on_device[index + 1..].iter().map(move |second| (first, second)))
        .filter(|(first, second)| {
            let start = first.submitted_ns.max(second.submitted_ns);
            let end = first.outcome_terminal_ns.min(second.outcome_terminal_ns);
            start < end
        })
        .count();
    if overlapping > 0 {
        clause(
            "concurrent_attempt_intervals_on_one_device",
            ClauseStatus::Evidenced,
            format!("{overlapping} overlapping attempt pairs on ordinal {ordinal}"),
        )
    } else {
        clause(
            "concurrent_attempt_intervals_on_one_device",
            ClauseStatus::Failed,
            format!("attempts on ordinal {ordinal} did not overlap in time"),
        )
    }
}

/// `peak_active_query_jobs` only increments when a job actually starts, so it
/// is falsifiable evidence of device-local concurrency.
fn occupancy_witness_clause(device: &DevicePlacement, ordinal: usize) -> ScenarioClause {
    let peak = device.occupancy.peak_active_query_jobs;
    if peak >= 2 {
        clause(
            "witnessed_concurrent_device_occupancy",
            ClauseStatus::Evidenced,
            format!("ordinal {ordinal} reached {peak} concurrent GPU execution jobs"),
        )
    } else {
        clause(
            "witnessed_concurrent_device_occupancy",
            ClauseStatus::Failed,
            format!("ordinal {ordinal} peaked at {peak} concurrent GPU execution jobs"),
        )
    }
}

fn multiple_device_clauses(
    measured: &[&AttemptObservation],
    devices: &[DeviceConfiguration],
    placement: &[DevicePlacement],
    quarantine_fallback: Option<&QuarantineFallbackEvidence>,
) -> Vec<ScenarioClause> {
    let quarantined_ordinal =
        quarantine_fallback.filter(|evidence| evidence.target_ledger_closed).map(|evidence| evidence.target_ordinal);
    let eligible = placement
        .iter()
        .filter(|device| {
            device.configuration.starting_health_open && Some(device.configuration.ordinal) != quarantined_ordinal
        })
        .collect::<Vec<_>>();
    let mut clauses = Vec::new();

    // A caller may intentionally cancel an attempt before it reaches
    // admission. That sealed control outcome has no device by contract and
    // must not be reclassified as a placement defect. An explicit
    // non-executing capability probe also has no placement by design; its
    // separate probe reducer owns the typed rejection proof. Every selected
    // workload attempt that reached admission still requires exactly one
    // ordinal; a typed pre-admission capacity rejection owns no device.
    let pre_admission_cancellations =
        measured.iter().filter(|attempt| attempt.outcome == "cancelled" && attempt.device_ordinal.is_none()).count();
    let not_admitted = measured
        .iter()
        .filter(|attempt| attempt.terminal_evidence.capacity_disposition.as_str() == Some("not_admitted"))
        .count();
    let non_executing_capability_probes = measured.iter().filter(|attempt| !attempt.placement_required).count();
    let placement_required = measured
        .iter()
        .filter(|attempt| {
            attempt.placement_required
                && attempt.terminal_evidence.capacity_disposition.as_str() != Some("not_admitted")
                && !(attempt.outcome == "cancelled" && attempt.device_ordinal.is_none())
        })
        .collect::<Vec<_>>();
    let one_ordinal_each =
        !placement_required.is_empty() && placement_required.iter().all(|attempt| attempt.device_ordinal.is_some());
    clauses.push(if one_ordinal_each {
        clause(
            "one_ordinal_per_attempt",
            ClauseStatus::Evidenced,
            format!(
                "{} placement-required measured attempts each recorded exactly one ordinal; {not_admitted} not-admitted attempts, {non_executing_capability_probes} non-executing capability probes, and {pre_admission_cancellations} explicit pre-admission cancellations were exempt",
                placement_required.len()
            ),
        )
    } else {
        clause(
            "one_ordinal_per_attempt",
            ClauseStatus::Failed,
            "at least one measured attempt recorded no placement ordinal".to_string(),
        )
    });

    let stalled = eligible.iter().filter(|device| !device.made_progress).count();
    clauses.push(if !eligible.is_empty() && stalled == 0 {
        clause(
            "progress_on_all_eligible_devices",
            ClauseStatus::Evidenced,
            format!("{} eligible ordinals each completed at least one attempt", eligible.len()),
        )
    } else {
        clause(
            "progress_on_all_eligible_devices",
            ClauseStatus::Failed,
            format!("{stalled} of {} eligible ordinals completed no attempt", eligible.len()),
        )
    });

    clauses.push(heterogeneity_clause(devices));
    clauses.push(quarantine_fallback.map_or_else(
        || {
            clause(
                "device_block_or_quarantine_fallback",
                ClauseStatus::NotEvidenced,
                "no controlled device-quarantine capability was requested for this run".to_string(),
            )
        },
        QuarantineFallbackEvidence::clause,
    ));
    clauses
}

/// Heterogeneity must be visible in the resolved envelopes, or the clause is
/// unevidenced even when the profile text differs.
fn heterogeneity_clause(devices: &[DeviceConfiguration]) -> ScenarioClause {
    if devices.len() < 2 {
        return clause(
            "heterogeneous_ordinal_profiles",
            ClauseStatus::NotEvidenced,
            format!("{} configured ordinal(s); heterogeneity needs at least two", devices.len()),
        );
    }
    let capacities = devices.iter().map(|device| device.capacity_bytes).collect::<BTreeSet<_>>();
    let attempt_caps = devices.iter().map(|device| device.max_active_attempts).collect::<BTreeSet<_>>();
    let streams = devices.iter().map(|device| device.stream_pool_size).collect::<BTreeSet<_>>();
    if capacities.len() > 1 || attempt_caps.len() > 1 || streams.len() > 1 {
        clause(
            "heterogeneous_ordinal_profiles",
            ClauseStatus::Evidenced,
            format!(
                "{} distinct capacities, {} distinct attempt caps, {} distinct stream pools",
                capacities.len(),
                attempt_caps.len(),
                streams.len()
            ),
        )
    } else {
        clause(
            "heterogeneous_ordinal_profiles",
            ClauseStatus::NotEvidenced,
            "every configured ordinal resolved to an identical envelope".to_string(),
        )
    }
}

fn clause(clause: &'static str, status: ClauseStatus, detail: String) -> ScenarioClause {
    ScenarioClause { clause, status, detail }
}

/// Recounts the published rows and compares them with the summary reductions.
///
/// The row side reads only the emitted [`AttemptRow`]s — the same values a
/// consumer of the artifact would count — so it does not share an expression
/// with the aggregate side.
fn equality(
    window: &MeasurementWindow,
    placement: &[DevicePlacement],
    throughput: &ThroughputAggregate,
    attempts: &[AttemptRow],
) -> ReportAggregateEquality {
    let rows_measurement_attempts =
        attempts.iter().filter(|row| row.phase == SchedulePhase::Measurement).count() as u64;
    let rows_succeeded = attempts
        .iter()
        .filter(|row| row.phase == SchedulePhase::Measurement && row.outcome == SUCCEEDED_OUTCOME)
        .count() as u64;
    let rows_placed = attempts.iter().filter(|row| row.device_ordinal.is_some()).count() as u64;
    let rows_total = attempts.len() as u64;

    let aggregate_placed = placement.iter().map(|device| device.attempts_placed).sum();
    let aggregate_total = window.warmup_attempts + window.measurement_attempts;
    ReportAggregateEquality {
        rows_measurement_attempts,
        aggregate_measurement_attempts: throughput.completions,
        rows_succeeded,
        aggregate_succeeded: throughput.successful_completions,
        rows_placed,
        aggregate_placed,
        rows_total,
        aggregate_total,
        equal: rows_measurement_attempts == throughput.completions
            && rows_succeeded == throughput.successful_completions
            && rows_placed == aggregate_placed
            && rows_total == aggregate_total,
    }
}

/// The single terminal outcome label that counts as a completion.
pub(super) const SUCCEEDED_OUTCOME: &str = "succeeded";

/// True when no scenario clause failed. `not_evidenced` clauses are declared
/// gaps and do not fail the run.
#[cfg(test)]
pub(super) fn scenario_clauses_pass(evidence: &ScenarioEvidence) -> bool {
    evidence
        .same_device
        .iter()
        .chain(evidence.multiple_devices.iter())
        .all(|clause| clause.status != ClauseStatus::Failed)
}

#[cfg(test)]
mod tests;
