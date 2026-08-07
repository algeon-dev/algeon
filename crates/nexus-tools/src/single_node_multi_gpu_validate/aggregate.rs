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

use super::{profile::FixtureKind, schedule::SchedulePhase};

/// One attempt's measured outcome, projected from `QueryAttemptReport`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AttemptObservation {
    pub(super) sequence: usize,
    pub(super) phase: SchedulePhase,
    /// W3 step this attempt was submitted under. Single-phase runs are all
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
    /// P6.6's deadline, fault, cleanup, and allocator evidence. This is a
    /// read-only projection, never an independent event store.
    pub(super) terminal_evidence: AttemptTerminalEvidence,
    pub(super) succeeded: bool,
    /// Present only when the attempt reached native GPU execution.
    pub(super) reached_gpu: bool,
    /// One typed graph model-claim snapshot frozen with the sealed report.
    /// It is absent for non-native attempts, attempts that have not reached
    /// their seal point, or an unexpectedly missing native-root projection.
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationSnapshot>,
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
    pub(super) total_ns: Option<u64>,
    pub(super) queue_wait_ns: Option<u64>,
    pub(super) physical_execution_ns: Option<u64>,
    /// Submit and terminal instants on the engine's shared observation clock.
    /// Both come from the sealed report, so intervals from different attempts
    /// are comparable and the overlap witness can actually fail.
    pub(super) submitted_ns: u64,
    pub(super) outcome_terminal_ns: u64,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GraphClaimCalibrationExecution {
    Physical,
    NonPhysical(GraphClaimCalibrationNonPhysicalReason),
}

/// Closed vocabulary for graph attempts that intentionally did not execute
/// a physical graph algorithm locally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GraphClaimCalibrationNonPhysicalReason {
    BfsSingleflightFollower {
        #[serde(skip_serializing_if = "Option::is_none")]
        leader_attempt_id: Option<String>,
    },
    GraphMemoryAdmissionDenied,
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

/// I11 completeness result for every measurement-phase graph attempt.
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

/// P6.6-relevant facts projected directly from one sealed report.
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
    /// Immutable P5 aggregate resident-cache cap for this ordinal. A zero cap
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
    /// Controlled P6.6 quarantine/fallback evidence, when the explicit
    /// validation capability was invoked for this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) quarantine_fallback: Option<QuarantineFallbackEvidence>,
    /// W3 per-step envelope aggregates. Empty for single-phase profiles, so
    /// their artifacts carry no per-step section.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) steps: Vec<StepAggregate>,
    pub(super) report_aggregate_equality: ReportAggregateEquality,
}

/// One W3 step's envelope curve point: throughput and latency computed over
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
    /// W8's per-attempt public-report attribution check. This is a
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
/// attempt that entered the measurement phase. A profile opts into this I11
/// gate only for graph workload cells; non-graph attempts are deliberately
/// outside its denominator.
pub(super) fn graph_claim_calibration_evidence(observations: &[AttemptObservation]) -> GraphClaimCalibrationEvidence {
    let graph_attempts = observations
        .iter()
        .filter(|attempt| attempt.graph_workload && attempt.phase == SchedulePhase::Measurement)
        .collect::<Vec<_>>();
    let mut attempts_by_id = BTreeMap::<&str, Vec<&AttemptObservation>>::new();
    for attempt in &graph_attempts {
        attempts_by_id.entry(&attempt.attempt_id).or_default().push(*attempt);
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
    /// Present only for the explicit P6.6 bounded-overload probe. Its facts
    /// are all sealed report projections or resolved profile limits; no queue
    /// ordering is reconstructed from runner scheduling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) overload_fairness: Option<OverloadFairnessEvidence>,
    /// Logical slot occupancy is recorded, never asserted: the ledger derives
    /// these counts from the immutable profile, so they cannot witness CUDA
    /// stream identity. Nsight owns stream-level overlap.
    pub(super) logical_slot_occupancy: LogicalSlotOccupancyEvidence,
}

/// Evidence for the explicit P6.6 public bounded-overload and drain scenario.
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
            status: valid.then_some(ClauseStatus::Evidenced).unwrap_or(ClauseStatus::Failed),
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

/// Explicit P6.6 proof for a genuine allocator failure under external CUDA
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
        let configured_active_attempt_capacity = devices.iter().fold(0_u64, |total, device| {
            total.saturating_add(u64::try_from(device.max_active_attempts).unwrap_or(u64::MAX))
        });
        let configured_queue_limit = u64::try_from(configured_queue_limit).unwrap_or(u64::MAX);
        let planned_cohort_attempts = u64::try_from(planned_cohort_attempts).unwrap_or(u64::MAX);
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
            observed_peak_queued_attempts: u64::try_from(observed_peak_queued_attempts).unwrap_or(u64::MAX),
            queue_limit_reached: observed_peak_queued_attempts
                == usize::try_from(configured_queue_limit).unwrap_or(usize::MAX),
            queue_bound_respected: observed_peak_queued_attempts
                <= usize::try_from(configured_queue_limit).unwrap_or(usize::MAX),
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
    #[allow(clippy::too_many_arguments)]
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

/// S6 repeated-quarantine evidence: each cycle reuses the single-quarantine
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

    /// Reduces ordinary P6.5 observations plus the optional explicit P6.6
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

    /// The complete reduction. `steps` carries the W3 per-step windows; an
    /// empty slice is a single-phase run and emits no per-step section.
    #[allow(clippy::too_many_arguments)]
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

    /// Attaches the one explicit P6.6 overload probe after its driver has
    /// supplied the finite-cohort counters. The ordinary aggregate remains
    /// reusable for P6.5 and P6.8 runs that did not request this scenario.
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
        mean_ns: if totals.is_empty() { 0 } else { u64::try_from(sum / totals.len() as u128).unwrap_or(u64::MAX) },
        queue_wait_p95_ns: percentile(&queue_waits, 95.0),
        admission_wait_samples: admission_waits.len() as u64,
        admission_wait_p50_ns: percentile(&admission_waits, 50.0),
        admission_wait_p95_ns: percentile(&admission_waits, 95.0),
        admission_wait_p99_ns: percentile(&admission_waits, 99.0),
    }
}

/// Nearest-rank percentile over a sorted slice. Raw samples are retained, so
/// this needs no bucket calibration; bucket suitability is P6.7's.
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
    // workload attempt still requires exactly one ordinal.
    let pre_admission_cancellations =
        measured.iter().filter(|attempt| attempt.outcome == "cancelled" && attempt.device_ordinal.is_none()).count();
    let non_executing_capability_probes = measured.iter().filter(|attempt| !attempt.placement_required).count();
    let placement_required = measured
        .iter()
        .filter(|attempt| {
            attempt.placement_required && !(attempt.outcome == "cancelled" && attempt.device_ordinal.is_none())
        })
        .collect::<Vec<_>>();
    let one_ordinal_each =
        !placement_required.is_empty() && placement_required.iter().all(|attempt| attempt.device_ordinal.is_some());
    clauses.push(if one_ordinal_each {
        clause(
            "one_ordinal_per_attempt",
            ClauseStatus::Evidenced,
            format!(
                "{} placement-required measured attempts each recorded exactly one ordinal; {non_executing_capability_probes} non-executing capability probes and {pre_admission_cancellations} explicit pre-admission cancellations were exempt",
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
pub(super) fn scenario_clauses_pass(evidence: &ScenarioEvidence) -> bool {
    evidence
        .same_device
        .iter()
        .chain(evidence.multiple_devices.iter())
        .all(|clause| clause.status != ClauseStatus::Failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(ordinal: usize, capacity_bytes: u64, max_active_attempts: usize) -> DeviceConfiguration {
        DeviceConfiguration {
            ordinal,
            model_name: Some("NVIDIA L40S".to_string()),
            uuid: Some(format!("GPU-{ordinal}")),
            physical_total_bytes: Some(46_068 * 1024 * 1024),
            capacity_bytes,
            cache_cap_bytes: 0,
            install_telemetry: true,
            max_active_attempts,
            stream_pool_size: 8,
            worker_count: 2,
            lane_count: 2,
            ownership: "whole_device_exclusive".to_string(),
            device_health_generation: 0,
            starting_health_open: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attempt(
        sequence: usize,
        ordinal: usize,
        grant_id: Option<u64>,
        domain_id: Option<u64>,
        submitted_ns: u64,
        outcome_terminal_ns: u64,
        total_ns: u64,
        phase: SchedulePhase,
    ) -> AttemptObservation {
        AttemptObservation {
            sequence,
            phase,
            step_index: 0,
            query_kind: super::super::profile::FixtureKind::Tpch,
            query_label: "q01".to_string(),
            graph_workload: false,
            placement_required: true,
            attempt_id: format!("attempt-{sequence:032x}"),
            disposition: "Native".to_string(),
            outcome: "succeeded".to_string(),
            failure: None,
            terminal_evidence: terminal_evidence(),
            succeeded: true,
            reached_gpu: true,
            graph_claim_calibration: None,
            device_ordinal: Some(ordinal),
            grant_bytes: Some(1024),
            admission_sequence: Some(sequence as u64),
            admission_wait_ns: Some(10),
            admission_requested_ns: Some(submitted_ns),
            admitted_ns: Some(submitted_ns.saturating_add(1)),
            admission_wait: None,
            grant_id,
            domain_id,
            total_ns: Some(total_ns),
            queue_wait_ns: Some(5),
            physical_execution_ns: Some(total_ns / 2),
            submitted_ns,
            outcome_terminal_ns,
        }
    }

    fn terminal_evidence() -> AttemptTerminalEvidence {
        AttemptTerminalEvidence {
            primary_failure: None,
            cleanup_disposition: serde_json::json!("completed"),
            cleanup_failure_count: 0,
            cleanup_failure_count_complete: true,
            first_cleanup_failure: None,
            capacity_disposition: serde_json::json!("released"),
            health_transition: Some(serde_json::json!("healthy_reuse")),
            native_fault_evidence: None,
            resource_evidence: None,
            retry_advice: serde_json::json!({ "recovery": "do_not_retry", "prerequisite": "none" }),
        }
    }

    fn domain_exhaustion_failure() -> AttemptObservation {
        let mut failure = attempt(0, 0, Some(1), Some(10), 0, 1_000, 1_000, SchedulePhase::Measurement);
        failure.outcome = "failed".to_string();
        failure.succeeded = false;
        failure.failure = Some(AttemptFailureClass {
            origin: "Native".to_string(),
            code: "CudfResourceExhausted".to_string(),
            status: "Temporary".to_string(),
            operation: "Execution".to_string(),
        });
        failure.terminal_evidence.native_fault_evidence = Some(serde_json::json!({
            "device_health": {
                "status": "healthy",
                "quarantined_teardown_count": 0,
            }
        }));
        failure.terminal_evidence.resource_evidence = Some(serde_json::json!({
            "allocator_event": {
                "provenance": "attempt_exact",
                "value": {"source": "reservation_limit"},
            }
        }));
        failure
    }

    fn physical_allocator_pressure_failure() -> AttemptObservation {
        let mut failure = domain_exhaustion_failure();
        failure.terminal_evidence.resource_evidence = Some(serde_json::json!({
            "allocator_event": {
                "provenance": "attempt_exact",
                "value": {
                    "source": "physical",
                    "requested_bytes": 1024,
                },
            },
            "grant_cap_bytes": {"provenance": "attempt_exact", "value": 2048},
            "domain_cap_bytes": {"provenance": "attempt_exact", "value": 2048},
            "managed_capacity_bytes": {"provenance": "attempt_exact", "value": 4096},
        }));
        failure
    }

    fn physical_pressure_facts(released: bool) -> PhysicalPressureLeaseFacts {
        PhysicalPressureLeaseFacts {
            device_ordinal: 0,
            method: "cuda_on_sidecar_fixed_chunks",
            free_before_bytes: 4096,
            free_under_pressure_bytes: 512,
            bytes_held: 3584,
            reserve_target_bytes: 512,
            free_after_release_bytes: released.then_some(4096),
            recovered_bytes: released.then_some(3584),
        }
    }

    fn quarantine_evidence(target_ordinal: usize, observations: &[AttemptObservation]) -> QuarantineFallbackEvidence {
        QuarantineFallbackEvidence::from_barrier(
            target_ordinal,
            10,
            7,
            1,
            0,
            20,
            1,
            true,
            true,
            &observations.iter().map(|observation| observation.sequence).collect(),
            true,
            None,
            observations,
        )
    }

    fn occupancy(peak_jobs: usize) -> BTreeMap<usize, DeviceOccupancyPeaks> {
        BTreeMap::from([(
            0,
            DeviceOccupancyPeaks {
                peak_active_attempts: 2,
                peak_active_query_jobs: peak_jobs,
                peak_live_grants_bytes: 2048,
                peak_live_reserved_bytes: 4096,
                allocation_failure_count: 0,
                min_root_streams_available: 6,
                min_root_worker_slots_available: 1,
            },
        )])
    }

    #[test]
    fn same_device_evidence_requires_distinct_grants_domains_overlap_and_witnessed_occupancy() {
        let observations = vec![
            attempt(0, 0, Some(10), Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
            attempt(1, 0, Some(11), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
        ];
        let aggregate =
            MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
        let status = |name: &str| {
            aggregate
                .scenarios
                .same_device
                .iter()
                .find(|clause| clause.clause == name)
                .map(|clause| clause.status)
                .unwrap()
        };
        assert_eq!(status("multiple_admitted_attempts_on_one_device"), ClauseStatus::Evidenced);
        assert_eq!(status("distinct_grants"), ClauseStatus::Evidenced);
        assert_eq!(status("distinct_domains"), ClauseStatus::Evidenced);
        assert_eq!(status("concurrent_attempt_intervals_on_one_device"), ClauseStatus::Evidenced);
        assert_eq!(status("witnessed_concurrent_device_occupancy"), ClauseStatus::Evidenced);
        // Nsight overlap is never claimed by this runner.
        assert_eq!(status("nsight_confirmed_overlapping_cuda_ranges"), ClauseStatus::NotEvidenced);
        assert!(scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn a_shared_or_unavailable_identity_fails_rather_than_shrinking_the_sample() {
        let shared = vec![
            attempt(0, 0, Some(10), Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
            attempt(1, 0, Some(10), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
        ];
        let aggregate = MeasuredAggregate::reduce(&shared, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
        let grants = aggregate.scenarios.same_device.iter().find(|c| c.clause == "distinct_grants").unwrap();
        assert_eq!(grants.status, ClauseStatus::Failed);
        assert!(!scenario_clauses_pass(&aggregate.scenarios));

        let unavailable = vec![
            attempt(0, 0, None, Some(100), 0, 2_000, 2_000, SchedulePhase::Measurement),
            attempt(1, 0, Some(11), Some(101), 1_000, 3_000, 2_000, SchedulePhase::Measurement),
        ];
        let aggregate =
            MeasuredAggregate::reduce(&unavailable, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
        let grants = aggregate.scenarios.same_device.iter().find(|c| c.clause == "distinct_grants").unwrap();
        assert_eq!(grants.status, ClauseStatus::Failed);
        assert!(grants.detail.contains("carried no identity"), "{}", grants.detail);
    }

    #[test]
    fn serialized_occupancy_is_labelled_profile_derived_and_never_claims_gpu_overlap() {
        let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let aggregate =
            MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
        let value = serde_json::to_value(&aggregate.scenarios.logical_slot_occupancy).unwrap();
        assert_eq!(value["basis"], "profile_derived_logical_slots");
        assert_eq!(value["proves_gpu_overlap"], false);
        assert_eq!(value["stream_identity_source"], "nsight_systems_trace");
        // The word "stream" only appears on the min-availability counter, never
        // as an identity claim.
        assert!(value["per_device"]["0"]["min_root_streams_available"].is_number());
    }

    #[test]
    fn four_device_progress_and_heterogeneity_are_asserted_from_resolved_envelopes() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2), device(2, 2 << 30, 2), device(3, 1 << 30, 4)];
        let observations = (0..4)
            .map(|ordinal| {
                attempt(
                    ordinal,
                    ordinal,
                    Some(ordinal as u64 + 1),
                    Some(ordinal as u64 + 100),
                    0,
                    1_000,
                    1_000,
                    SchedulePhase::Measurement,
                )
            })
            .collect::<Vec<_>>();
        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
        let status = |name: &str| {
            aggregate
                .scenarios
                .multiple_devices
                .iter()
                .find(|clause| clause.clause == name)
                .map(|clause| clause.status)
                .unwrap()
        };
        assert_eq!(status("one_ordinal_per_attempt"), ClauseStatus::Evidenced);
        assert_eq!(status("progress_on_all_eligible_devices"), ClauseStatus::Evidenced);
        assert_eq!(status("heterogeneous_ordinal_profiles"), ClauseStatus::Evidenced);
        // Quarantine fallback is a declared P6.6 gap, not a pass and not a failure.
        assert_eq!(status("device_block_or_quarantine_fallback"), ClauseStatus::NotEvidenced);
        assert!(aggregate.placement.iter().all(|device| device.made_progress));
        assert!(scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn explicit_pre_admission_cancellation_is_not_a_missing_placement() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let mut cancelled = attempt(0, 0, Some(1), Some(10), 0, 10, 10, SchedulePhase::Measurement);
        cancelled.outcome = "cancelled".to_string();
        cancelled.succeeded = false;
        cancelled.reached_gpu = false;
        cancelled.device_ordinal = None;
        cancelled.grant_bytes = None;
        cancelled.grant_id = None;
        cancelled.domain_id = None;
        let observations = vec![
            cancelled,
            attempt(1, 0, Some(2), Some(11), 0, 10, 10, SchedulePhase::Measurement),
            attempt(2, 1, Some(3), Some(12), 0, 10, 10, SchedulePhase::Measurement),
        ];

        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
        let clause = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "one_ordinal_per_attempt")
            .unwrap();
        assert_eq!(clause.status, ClauseStatus::Evidenced);
        assert!(clause.detail.contains("1 explicit pre-admission cancellations"));
    }

    #[test]
    fn explicit_nonexecuting_capability_probe_is_not_a_missing_placement() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let mut capability_probe = attempt(0, 0, None, None, 0, 10, 10, SchedulePhase::Measurement);
        capability_probe.graph_workload = true;
        capability_probe.placement_required = false;
        capability_probe.outcome = "failed".to_string();
        capability_probe.succeeded = false;
        capability_probe.reached_gpu = false;
        capability_probe.device_ordinal = None;
        let observations = vec![
            capability_probe,
            attempt(1, 0, Some(2), Some(11), 0, 10, 10, SchedulePhase::Measurement),
            attempt(2, 1, Some(3), Some(12), 0, 10, 10, SchedulePhase::Measurement),
        ];

        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
        let clause = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "one_ordinal_per_attempt")
            .unwrap();
        assert_eq!(clause.status, ClauseStatus::Evidenced);
        assert!(clause.detail.contains("1 non-executing capability probes"));
    }

    #[test]
    fn controlled_quarantine_evidence_requires_a_closed_target_and_healthy_fallback() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let observations = vec![attempt(0, 1, Some(10), Some(100), 0, 1_000, 1_000, SchedulePhase::Measurement)];
        let evidence = quarantine_evidence(0, &observations);
        let aggregate = MeasuredAggregate::reduce_with_quarantine(
            &observations,
            &devices,
            &BTreeMap::new(),
            0,
            1_000_000_000,
            Some(evidence),
        );
        let clause = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "device_block_or_quarantine_fallback")
            .expect("fallback clause is present");
        assert_eq!(clause.status, ClauseStatus::Evidenced);
        assert_eq!(aggregate.quarantine_fallback.as_ref().map(|evidence| evidence.target_ordinal), Some(0));
    }

    #[test]
    fn controlled_quarantine_evidence_rejects_a_post_injection_target_placement() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let observations = vec![attempt(0, 0, Some(10), Some(100), 0, 1_000, 1_000, SchedulePhase::Measurement)];
        let evidence = quarantine_evidence(0, &observations);
        let aggregate = MeasuredAggregate::reduce_with_quarantine(
            &observations,
            &devices,
            &BTreeMap::new(),
            0,
            1_000_000_000,
            Some(evidence),
        );
        let clause = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "device_block_or_quarantine_fallback")
            .expect("fallback clause is present");
        assert_eq!(clause.status, ClauseStatus::Failed);
        assert!(!scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn domain_exhaustion_requires_exact_reservation_limit_cleanup_and_healthy_reuse() {
        let failure = domain_exhaustion_failure();
        let recovery = attempt(1, 0, Some(2), Some(11), 1_001, 2_000, 999, SchedulePhase::Measurement);
        let evidence = DomainExhaustionEvidence::from_observations(&[failure.clone(), recovery]);

        assert!(evidence.is_evidenced());
        assert_eq!(evidence.matched_native_resource_exhaustion_reports, 1);
        assert_eq!(evidence.failure_sequence, Some(0));
        assert_eq!(evidence.failure_device_ordinal, Some(0));
        assert_eq!(evidence.recovery_sequence, Some(1));

        let mut physical = failure;
        physical.terminal_evidence.resource_evidence = Some(serde_json::json!({
            "allocator_event": {
                "provenance": "attempt_exact",
                "value": {"source": "physical"},
            }
        }));
        let evidence = DomainExhaustionEvidence::from_observations(&[physical]);
        assert!(!evidence.is_evidenced());
        assert!(!evidence.allocator_event_attempt_exact_reservation_limit);
        assert!(!evidence.same_device_recovery_completed);
    }

    #[test]
    fn physical_allocator_pressure_requires_sidecar_pressure_exact_physical_event_caps_release_recovery_and_close() {
        let failure = physical_allocator_pressure_failure();
        let mut recovery = attempt(1, 0, Some(2), Some(11), 1_001, 2_000, 999, SchedulePhase::Measurement);
        recovery.query_label = "edge_scan".to_string();
        let evidence = PhysicalAllocatorPressureEvidence::from_observations(
            physical_pressure_facts(false),
            physical_pressure_facts(true),
            true,
            true,
            &[failure.clone(), recovery.clone()],
        )
        .with_final_close_quiescent(true);

        assert!(evidence.is_evidenced());
        assert_eq!(evidence.status, ClauseStatus::Evidenced);
        assert_eq!(evidence.observed_physical_request_bytes, Some(1024));
        assert_eq!(evidence.failure_sequence, Some(0));
        assert_eq!(evidence.recovery_sequence, Some(1));
        let projection = serde_json::to_value(&evidence).unwrap();
        assert_eq!(projection["pressure"]["method"], "cuda_on_sidecar_fixed_chunks");
        assert_eq!(projection["observed_physical_request_bytes"], 1024);
        assert_eq!(projection["status"], "evidenced");

        let mut non_contiguous_pressure = physical_pressure_facts(false);
        let mut non_contiguous_release = physical_pressure_facts(true);
        non_contiguous_pressure.free_under_pressure_bytes = 2_048;
        non_contiguous_release.free_under_pressure_bytes = 2_048;
        let non_contiguous = PhysicalAllocatorPressureEvidence::from_observations(
            non_contiguous_pressure,
            non_contiguous_release,
            true,
            true,
            &[failure.clone(), recovery.clone()],
        )
        .with_final_close_quiescent(true);
        assert!(!non_contiguous.physical_request_exceeds_free_under_pressure);
        assert!(non_contiguous.is_evidenced());

        let close_required = PhysicalAllocatorPressureEvidence::from_observations(
            physical_pressure_facts(false),
            physical_pressure_facts(true),
            true,
            true,
            &[failure.clone(), recovery.clone()],
        );
        assert!(!close_required.is_evidenced());
        assert_eq!(close_required.status, ClauseStatus::Failed);
        assert!(!close_required.final_close_quiescent);

        let mut too_small_grant = failure;
        too_small_grant.terminal_evidence.resource_evidence.as_mut().unwrap()["grant_cap_bytes"]["value"] =
            serde_json::json!(1024);
        let failed = PhysicalAllocatorPressureEvidence::from_observations(
            physical_pressure_facts(false),
            physical_pressure_facts(true),
            true,
            true,
            &[too_small_grant, recovery],
        )
        .with_final_close_quiescent(true);
        assert!(!failed.is_evidenced());
        assert!(!failed.grant_cap_exceeds_physical_request);

        let missing_close = PhysicalAllocatorPressureEvidence::from_observations(
            physical_pressure_facts(false),
            physical_pressure_facts(true),
            true,
            true,
            &[physical_allocator_pressure_failure()],
        );
        assert!(!missing_close.is_evidenced());
        assert!(!missing_close.final_close_quiescent);
        assert!(!missing_close.same_device_recovery_completed);
    }

    #[test]
    fn overload_evidence_requires_a_bounded_full_queue_and_sealed_public_rows() {
        let devices = vec![device(0, 1 << 30, 1), device(1, 1 << 30, 1)];
        let mut protected = attempt(0, 0, Some(1), Some(10), 0, 2_000, 2_000, SchedulePhase::Measurement);
        protected.admission_sequence = Some(4);
        protected.admission_wait_ns = Some(900);
        protected.admission_requested_ns = Some(100);
        protected.admitted_ns = Some(1_000);
        protected.admission_wait = Some(AdmissionWaitEvidence {
            submission_sequence: 21,
            overtake_limit: 2,
            overtakes_consumed: 2,
            overtakes_remaining: 0,
            protection_device_ordinal: Some(0),
        });

        let mut unrelated = attempt(1, 1, Some(2), Some(11), 200, 500, 300, SchedulePhase::Measurement);
        unrelated.admission_sequence = Some(3);
        unrelated.admission_requested_ns = Some(200);
        unrelated.admitted_ns = Some(300);

        let mut observations = vec![
            protected,
            unrelated,
            attempt(2, 0, Some(3), Some(12), 0, 300, 300, SchedulePhase::Measurement),
            attempt(3, 1, Some(4), Some(13), 0, 400, 400, SchedulePhase::Measurement),
        ];
        observations[2].admission_sequence = Some(1);
        observations[3].admission_sequence = Some(2);

        let evidence = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 2, 0, 0, 0);
        assert!(evidence.is_evidenced());
        assert_eq!(evidence.status, ClauseStatus::Evidenced);
        assert_eq!(evidence.protection_device_ordinal, Some(0));
        assert_eq!(evidence.unrelated_device_progress_count, 1);
        assert_eq!(evidence.unrelated_device_completion_count, 1);

        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000)
            .with_overload_fairness(evidence.clone());
        let artifact_projection = serde_json::to_value(&aggregate).unwrap();
        assert_eq!(artifact_projection["scenarios"]["overload_fairness"]["status"], "evidenced");
        assert_eq!(artifact_projection["attempts"][0]["admission_wait"]["protection_device_ordinal"], 0);
        assert_eq!(artifact_projection["attempts"][0]["admission_requested_ns"], 100);
        assert_eq!(artifact_projection["attempts"][0]["admitted_ns"], 1_000);

        let queue_not_full = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 1, 0, 0, 0);
        assert!(!queue_not_full.is_evidenced());
        assert_eq!(queue_not_full.status, ClauseStatus::Failed);

        let queue_exceeded = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 3, 0, 0, 0);
        assert!(!queue_exceeded.is_evidenced());
        assert_eq!(queue_exceeded.status, ClauseStatus::Failed);

        observations[0].admission_wait.as_mut().unwrap().overtakes_remaining = 1;
        let malformed = OverloadFairnessEvidence::from_observations(&observations, &devices, 4, 2, 2, 2, 0, 0, 0);
        assert!(!malformed.is_evidenced());
        assert_eq!(malformed.status, ClauseStatus::Failed);
        assert!(!malformed.wait_facts_exact);
    }

    #[test]
    fn an_idle_eligible_device_fails_the_progress_clause() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 2 << 30, 2)];
        let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 1_000, 1_000, SchedulePhase::Measurement)];
        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
        let progress = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "progress_on_all_eligible_devices")
            .unwrap();
        assert_eq!(progress.status, ClauseStatus::Failed);
        assert!(!scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn identical_envelopes_leave_heterogeneity_unevidenced_rather_than_passing() {
        let devices = vec![device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let observations = (0..2)
            .map(|ordinal| {
                attempt(
                    ordinal,
                    ordinal,
                    Some(ordinal as u64),
                    Some(ordinal as u64),
                    0,
                    10,
                    10,
                    SchedulePhase::Measurement,
                )
            })
            .collect::<Vec<_>>();
        let aggregate = MeasuredAggregate::reduce(&observations, &devices, &BTreeMap::new(), 0, 1_000_000_000);
        let clause = aggregate
            .scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "heterogeneous_ordinal_profiles")
            .unwrap();
        assert_eq!(clause.status, ClauseStatus::NotEvidenced);
        assert!(scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn warmup_attempts_are_excluded_from_throughput_and_latency() {
        let observations = vec![
            attempt(0, 0, Some(1), Some(10), 0, 100, 5_000_000_000, SchedulePhase::Warmup),
            attempt(1, 0, Some(2), Some(11), 100, 200, 1_000_000_000, SchedulePhase::Measurement),
            attempt(2, 0, Some(3), Some(12), 200, 300, 3_000_000_000, SchedulePhase::Measurement),
        ];
        let aggregate = MeasuredAggregate::reduce(
            &observations,
            &[device(0, 1 << 30, 2)],
            &occupancy(2),
            1_000_000_000,
            2_000_000_000,
        );
        assert_eq!(aggregate.window.warmup_attempts, 1);
        assert_eq!(aggregate.window.measurement_attempts, 2);
        assert_eq!(aggregate.throughput.completions, 2);
        assert_eq!(aggregate.latency.samples, 2);
        // The 5s warmup sample must not enter the distribution.
        assert_eq!(aggregate.latency.max_ns, 3_000_000_000);
        assert!((aggregate.throughput.application_qps - 1.0).abs() < f64::EPSILON);
        // Every attempt, warmup included, still has a report row.
        assert_eq!(aggregate.attempts.len(), 3);
    }

    #[test]
    fn gpu_reaching_qps_excludes_cpu_fallback_while_application_qps_counts_it() {
        let mut fallback = attempt(1, 0, Some(2), Some(11), 0, 10, 1_000_000_000, SchedulePhase::Measurement);
        fallback.reached_gpu = false;
        fallback.disposition = "CpuFallback".to_string();
        fallback.device_ordinal = None;
        fallback.grant_id = None;
        let observations =
            vec![attempt(0, 0, Some(1), Some(10), 0, 10, 1_000_000_000, SchedulePhase::Measurement), fallback];
        let aggregate =
            MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 2_000_000_000);
        assert_eq!(aggregate.throughput.successful_completions, 2);
        assert_eq!(aggregate.throughput.gpu_reaching_completions, 1);
        assert!(aggregate.throughput.gpu_reaching_qps < aggregate.throughput.application_qps);
    }

    #[test]
    fn report_and_aggregate_totals_are_equal_by_construction() {
        let observations = (0..7)
            .map(|index| {
                attempt(
                    index,
                    index % 2,
                    Some(index as u64),
                    Some(index as u64 + 50),
                    index as u64 * 10,
                    index as u64 * 10 + 100,
                    1_000,
                    if index == 0 { SchedulePhase::Warmup } else { SchedulePhase::Measurement },
                )
            })
            .collect::<Vec<_>>();
        let aggregate = MeasuredAggregate::reduce(
            &observations,
            &[device(0, 1 << 30, 2), device(1, 2 << 30, 2)],
            &occupancy(2),
            0,
            1_000_000_000,
        );
        let equality = aggregate.report_aggregate_equality;
        assert!(equality.equal);
        assert_eq!(equality.rows_total, 7);
        assert_eq!(equality.aggregate_total, 7);
        assert_eq!(equality.rows_measurement_attempts, equality.aggregate_measurement_attempts);
        assert_eq!(equality.rows_succeeded, equality.aggregate_succeeded);
        assert_eq!(equality.rows_placed, equality.aggregate_placed);
    }

    #[test]
    fn percentiles_use_nearest_rank_over_retained_samples() {
        let sorted = (1..=100_u64).collect::<Vec<_>>();
        assert_eq!(percentile(&sorted, 50.0), 50);
        assert_eq!(percentile(&sorted, 95.0), 95);
        assert_eq!(percentile(&sorted, 99.0), 99);
        assert_eq!(percentile(&[], 95.0), 0);
        assert_eq!(percentile(&[7], 95.0), 7);
    }

    /// The witness must be falsifiable: two attempts that ran back to back on
    /// one ordinal did not overlap, even though both were placed there.
    #[test]
    fn sequential_attempts_on_one_device_fail_the_overlap_clause() {
        let sequential = vec![
            attempt(0, 0, Some(10), Some(100), 1_000, 2_000, 1_000, SchedulePhase::Measurement),
            attempt(1, 0, Some(11), Some(101), 2_000, 3_000, 1_000, SchedulePhase::Measurement),
        ];
        let aggregate =
            MeasuredAggregate::reduce(&sequential, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
        let overlap = aggregate
            .scenarios
            .same_device
            .iter()
            .find(|clause| clause.clause == "concurrent_attempt_intervals_on_one_device")
            .unwrap();
        assert_eq!(overlap.status, ClauseStatus::Failed);
        assert!(!scenario_clauses_pass(&aggregate.scenarios));

        // A shared submit instant alone is not overlap either: zero-width
        // intersection must not count.
        let touching = vec![
            attempt(0, 0, Some(10), Some(100), 1_000, 1_000, 0, SchedulePhase::Measurement),
            attempt(1, 0, Some(11), Some(101), 1_000, 2_000, 1_000, SchedulePhase::Measurement),
        ];
        let aggregate = MeasuredAggregate::reduce(&touching, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
        assert_eq!(
            aggregate
                .scenarios
                .same_device
                .iter()
                .find(|clause| clause.clause == "concurrent_attempt_intervals_on_one_device")
                .unwrap()
                .status,
            ClauseStatus::Failed
        );
    }

    /// Equality must be falsifiable too: an attempt placed on an ordinal the
    /// backend never reported breaks placement conservation.
    #[test]
    fn an_attempt_placed_on_an_unconfigured_ordinal_breaks_the_equality_check() {
        let stray = vec![
            attempt(0, 0, Some(10), Some(100), 0, 2_000, 1_000, SchedulePhase::Measurement),
            // Ordinal 7 has no device configuration, so `placement` cannot
            // account for it.
            attempt(1, 7, Some(11), Some(101), 0, 2_000, 1_000, SchedulePhase::Measurement),
        ];
        let aggregate = MeasuredAggregate::reduce(&stray, &[device(0, 1 << 30, 2)], &occupancy(2), 0, 1_000_000_000);
        let equality = aggregate.report_aggregate_equality;
        assert!(!equality.equal);
        assert_eq!(equality.rows_placed, 2);
        assert_eq!(equality.aggregate_placed, 1);
    }

    #[test]
    fn a_single_placed_attempt_leaves_same_device_concurrency_unevidenced() {
        let observations = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let aggregate =
            MeasuredAggregate::reduce(&observations, &[device(0, 1 << 30, 2)], &occupancy(1), 0, 1_000_000_000);
        assert_eq!(aggregate.scenarios.same_device.len(), 1);
        assert_eq!(aggregate.scenarios.same_device[0].status, ClauseStatus::NotEvidenced);
        assert!(scenario_clauses_pass(&aggregate.scenarios));
    }

    #[test]
    fn per_step_aggregates_attribute_throughput_and_placement_to_their_step() {
        let observations = vec![
            AttemptObservation {
                step_index: 0,
                total_ns: Some(1_000_000),
                ..attempt(0, 0, Some(1), Some(2), 0, 10, 1_000_000, SchedulePhase::Measurement)
            },
            AttemptObservation {
                step_index: 1,
                total_ns: Some(3_000_000),
                ..attempt(1, 1, Some(3), Some(4), 0, 20, 3_000_000, SchedulePhase::Measurement)
            },
            AttemptObservation {
                step_index: 1,
                total_ns: Some(5_000_000),
                ..attempt(2, 1, Some(5), Some(6), 0, 30, 5_000_000, SchedulePhase::Measurement)
            },
        ];
        let steps = [
            StepWindow {
                step_index: 0,
                concurrency: 1,
                arrival_rate_per_second: None,
                rate_overrun: "delay",
                measurement_ns: 1_000_000_000,
            },
            StepWindow {
                step_index: 1,
                concurrency: 4,
                arrival_rate_per_second: None,
                rate_overrun: "delay",
                measurement_ns: 2_000_000_000,
            },
        ];
        let devices = [device(0, 1 << 30, 2), device(1, 1 << 30, 2)];
        let aggregate =
            MeasuredAggregate::reduce_full(&observations, &devices, &occupancy(2), 0, 3_000_000_000, None, &steps);
        assert_eq!(aggregate.steps.len(), 2);
        let step0 = &aggregate.steps[0];
        assert_eq!(step0.throughput.completions, 1);
        assert_eq!(step0.latency.p50_ns, 1_000_000);
        assert_eq!(step0.placements.iter().map(|placement| placement.device_ordinal).collect::<Vec<_>>(), [0]);
        let step1 = &aggregate.steps[1];
        assert_eq!(step1.throughput.completions, 2);
        assert_eq!(step1.throughput.application_qps, 1.0);
        assert_eq!(step1.latency.p50_ns, 3_000_000);
        assert_eq!(step1.latency.max_ns, 5_000_000);
        assert_eq!(step1.placements.len(), 1);
        assert_eq!(step1.placements[0].attempts_placed, 2);
        // The run-level aggregate still spans every step's measurement set.
        assert_eq!(aggregate.throughput.completions, 3);
    }

    #[test]
    fn latency_keeps_sealed_admission_waits_separate_from_plan_queue_waits() {
        let mut first = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        first.admission_wait_ns = Some(10);
        first.queue_wait_ns = Some(1_000);
        let mut second = attempt(1, 0, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement);
        second.admission_wait_ns = Some(20);
        second.queue_wait_ns = Some(2_000);
        let mut third = attempt(2, 0, Some(5), Some(6), 0, 30, 30, SchedulePhase::Measurement);
        third.admission_wait_ns = Some(90);
        third.queue_wait_ns = Some(3_000);

        let reduced = latency(&[&first, &second, &third]);
        assert_eq!(reduced.admission_wait_samples, 3);
        assert_eq!(reduced.admission_wait_p50_ns, 20);
        assert_eq!(reduced.admission_wait_p95_ns, 90);
        assert_eq!(reduced.admission_wait_p99_ns, 90);
        assert_eq!(reduced.queue_wait_p95_ns, 3_000);
    }

    #[test]
    fn quarantine_cycles_evidence_requires_every_configured_cycle_to_evidence() {
        let observations = vec![
            attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement),
            attempt(1, 2, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement),
        ];
        let first = quarantine_evidence(0, &observations);
        let second = quarantine_evidence(1, &observations[1..]);
        let evidenced = QuarantineCyclesEvidence::from_cycle_evidence(2, vec![first.clone(), second]);
        assert!(evidenced.is_evidenced());
        assert_eq!(evidenced.completed_cycles, 2);
        assert_eq!(evidenced.cycles[0].target_ordinal, 0);
        assert_eq!(evidenced.cycles[1].device_health_generation, 1);

        // Fewer cycles than configured is a failure, not a partial pass.
        let incomplete = QuarantineCyclesEvidence::from_cycle_evidence(3, vec![first.clone()]);
        assert!(!incomplete.is_evidenced());
        assert_eq!(incomplete.completed_cycles, 1);

        // A cycle with post-ack placements on its target fails the section.
        let contaminated = vec![attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let failed = QuarantineCyclesEvidence::from_cycle_evidence(1, vec![quarantine_evidence(0, &contaminated)]);
        assert!(!failed.is_evidenced());
        assert_eq!(failed.cycles[0].status, ClauseStatus::Failed);
    }

    fn cache_snapshot(physical: u64, ledger: u64, cap: u64) -> serde_json::Value {
        serde_json::json!({"cache_cap_bytes": cap, "ledger_evictable_resident_bytes": ledger, "ledger_pinned_in_use_bytes": 0, "physical_current_bytes": physical})
    }

    #[test]
    fn opted_in_cache_quarantine_requires_complete_consistent_snapshots() {
        let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let mut evidence = quarantine_evidence(0, &observations);
        evidence.cache_residency = CacheQuarantineEvidence::from_snapshots(
            Some(cache_snapshot(90, 100, 100)),
            Some(cache_snapshot(0, 0, 100)),
            Some(cache_snapshot(0, 0, 100)),
        );
        evidence.require_cache_residency();
        assert_eq!(evidence.clause().status, ClauseStatus::Evidenced);
    }

    #[test]
    fn opted_in_cache_quarantine_rejects_missing_or_zero_pre_ack() {
        let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let mut evidence = quarantine_evidence(0, &observations);
        evidence.require_cache_residency();
        assert_eq!(evidence.clause().status, ClauseStatus::Failed);
        evidence.cache_residency = CacheQuarantineEvidence::from_snapshots(
            Some(cache_snapshot(0, 0, 100)),
            Some(cache_snapshot(0, 100, 100)),
            Some(cache_snapshot(0, 100, 100)),
        );
        assert_eq!(evidence.clause().status, ClauseStatus::Failed);
    }

    #[test]
    fn opted_in_cache_quarantine_rejects_physical_over_ledger_or_cap() {
        let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        for snapshots in [
            (cache_snapshot(101, 100, 100), cache_snapshot(80, 100, 100), cache_snapshot(80, 100, 100)),
            (cache_snapshot(101, 200, 100), cache_snapshot(80, 100, 100), cache_snapshot(80, 100, 100)),
            (cache_snapshot(80, 100, 100), cache_snapshot(101, 100, 100), cache_snapshot(80, 100, 100)),
            (cache_snapshot(80, 100, 100), cache_snapshot(1, 0, 100), cache_snapshot(80, 100, 100)),
        ] {
            let mut evidence = quarantine_evidence(0, &observations);
            evidence.cache_residency =
                CacheQuarantineEvidence::from_snapshots(Some(snapshots.0), Some(snapshots.1), Some(snapshots.2));
            evidence.require_cache_residency();
            assert_eq!(evidence.clause().status, ClauseStatus::Failed);
        }
    }

    #[test]
    fn legacy_quarantine_remains_evidenced_without_cache_observations() {
        let observations = vec![attempt(0, 1, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement)];
        let evidence = quarantine_evidence(0, &observations);
        assert_eq!(evidence.clause().status, ClauseStatus::Evidenced);
    }

    fn complete_graph_claim_row() -> GraphClaimCalibrationRow {
        GraphClaimCalibrationRow {
            node_id: "n7".to_string(),
            algorithm: "pagerank".to_string(),
            model_requested_bytes: 4_096,
            requested_bytes: 4_096,
            budget_bytes: Some(8_192),
            execution_state: "algorithm_started".to_string(),
            measured_peak_bytes: Some(4_000),
            measured_peak_provenance: "reservation_high_water".to_string(),
            graph_model_error_ratio: Some("1.024".to_string()),
        }
    }

    fn physical_graph_claim_snapshot(rows: Vec<GraphClaimCalibrationRow>) -> GraphClaimCalibrationSnapshot {
        GraphClaimCalibrationSnapshot {
            truncated: false,
            rows,
            execution: Some(GraphClaimCalibrationExecution::Physical),
            execution_complete: true,
        }
    }

    fn bfs_singleflight_follower_snapshot(leader_attempt_id: Option<String>) -> GraphClaimCalibrationSnapshot {
        GraphClaimCalibrationSnapshot {
            truncated: false,
            rows: Vec::new(),
            execution: Some(GraphClaimCalibrationExecution::NonPhysical(
                GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower { leader_attempt_id },
            )),
            execution_complete: true,
        }
    }

    #[test]
    fn graph_claim_calibration_requires_one_complete_typed_row_per_graph_attempt() {
        let mut graph = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        graph.graph_workload = true;
        graph.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![complete_graph_claim_row()]));

        let evidence = graph_claim_calibration_evidence(&[graph.clone()]);
        assert!(evidence.passes());
        assert_eq!(evidence.graph_attempts, 1);
        assert_eq!(evidence.attempts[0].reason, "complete");

        graph.graph_claim_calibration.as_mut().expect("snapshot exists").rows.push(complete_graph_claim_row());
        let duplicate = graph_claim_calibration_evidence(&[graph]);
        assert!(!duplicate.passes());
        assert_eq!(duplicate.attempts[0].reason, "expected_exactly_one_graph_claim_row");
    }

    #[test]
    fn graph_claim_calibration_rejects_missing_or_unmeasured_rows() {
        let mut missing = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        missing.graph_workload = true;
        let missing_evidence = graph_claim_calibration_evidence(&[missing]);
        assert!(!missing_evidence.passes());
        assert_eq!(missing_evidence.attempts[0].reason, "missing_sealed_graph_claim_snapshot");

        let mut unavailable = attempt(1, 0, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement);
        unavailable.graph_workload = true;
        unavailable.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![GraphClaimCalibrationRow {
            measured_peak_bytes: None,
            measured_peak_provenance: "unavailable".to_string(),
            ..complete_graph_claim_row()
        }]));
        let unavailable_evidence = graph_claim_calibration_evidence(&[unavailable]);
        assert!(!unavailable_evidence.passes());
        assert_eq!(unavailable_evidence.attempts[0].reason, "graph_peak_measurement_unavailable");
    }

    #[test]
    fn graph_claim_calibration_accepts_only_the_exact_model_admission_denial_contract() {
        let mut denied = attempt(1, 0, Some(3), Some(4), 0, 20, 20, SchedulePhase::Measurement);
        denied.graph_workload = true;
        denied.outcome = "failed".to_string();
        denied.succeeded = false;
        denied.failure = Some(AttemptFailureClass {
            origin: "Native".to_string(),
            code: "MemoryReservationFailed".to_string(),
            status: "Temporary".to_string(),
            operation: "Execution".to_string(),
        });
        denied.graph_claim_calibration = Some(GraphClaimCalibrationSnapshot {
            truncated: false,
            rows: vec![GraphClaimCalibrationRow {
                model_requested_bytes: 9_000,
                requested_bytes: 9_000,
                budget_bytes: Some(8_000),
                execution_state: "graph_memory_admission_denied".to_string(),
                measured_peak_bytes: None,
                measured_peak_provenance: "unavailable".to_string(),
                graph_model_error_ratio: None,
                ..complete_graph_claim_row()
            }],
            execution: Some(GraphClaimCalibrationExecution::NonPhysical(
                GraphClaimCalibrationNonPhysicalReason::GraphMemoryAdmissionDenied,
            )),
            execution_complete: true,
        });

        let evidence = graph_claim_calibration_evidence(&[denied.clone()]);
        assert!(evidence.passes());
        assert_eq!(evidence.attempts[0].reason, "complete");

        denied.failure = None;
        let mismatch = graph_claim_calibration_evidence(&[denied]);
        assert!(!mismatch.passes());
        assert_eq!(mismatch.attempts[0].reason, "graph_memory_admission_denied_failure_mismatch");
    }

    #[test]
    fn graph_claim_calibration_accepts_a_fully_linked_singleflight_follower() {
        let mut leader = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        leader.graph_workload = true;
        leader.graph_claim_calibration = Some(physical_graph_claim_snapshot(vec![complete_graph_claim_row()]));

        let mut follower = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
        follower.graph_workload = true;
        follower.graph_claim_calibration = Some(bfs_singleflight_follower_snapshot(Some(leader.attempt_id.clone())));

        let evidence = graph_claim_calibration_evidence(&[leader, follower]);
        assert!(evidence.passes());
        assert_eq!(evidence.graph_attempts, 2);
        assert_eq!(evidence.attempts[1].reason, "complete");
    }

    #[test]
    fn graph_claim_calibration_rejects_unlinked_or_row_carrying_singleflight_followers() {
        let mut absent_leader_id = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        absent_leader_id.graph_workload = true;
        absent_leader_id.graph_claim_calibration = Some(bfs_singleflight_follower_snapshot(None));
        let absent_leader_id_evidence = graph_claim_calibration_evidence(&[absent_leader_id]);
        assert!(!absent_leader_id_evidence.passes());
        assert_eq!(absent_leader_id_evidence.attempts[0].reason, "singleflight_follower_missing_leader_attempt_id");

        let mut missing_leader = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        missing_leader.graph_workload = true;
        missing_leader.graph_claim_calibration =
            Some(bfs_singleflight_follower_snapshot(Some("attempt-missing".to_string())));
        let missing_leader_evidence = graph_claim_calibration_evidence(&[missing_leader]);
        assert!(!missing_leader_evidence.passes());
        assert_eq!(missing_leader_evidence.attempts[0].reason, "singleflight_follower_leader_not_evidenced");

        let mut carrying_row = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
        carrying_row.graph_workload = true;
        let mut snapshot = bfs_singleflight_follower_snapshot(Some("attempt-missing".to_string()));
        snapshot.rows.push(complete_graph_claim_row());
        carrying_row.graph_claim_calibration = Some(snapshot);
        let carrying_row_evidence = graph_claim_calibration_evidence(&[carrying_row]);
        assert!(!carrying_row_evidence.passes());
        assert_eq!(carrying_row_evidence.attempts[0].reason, "singleflight_follower_has_graph_claim_rows");
    }

    #[test]
    fn graph_claim_calibration_rejects_missing_or_conflicting_execution_classification() {
        let mut missing = attempt(0, 0, Some(1), Some(2), 0, 10, 10, SchedulePhase::Measurement);
        missing.graph_workload = true;
        missing.graph_claim_calibration = Some(GraphClaimCalibrationSnapshot {
            truncated: false,
            rows: vec![complete_graph_claim_row()],
            execution: None,
            execution_complete: false,
        });
        let missing_evidence = graph_claim_calibration_evidence(&[missing]);
        assert!(!missing_evidence.passes());
        assert_eq!(missing_evidence.attempts[0].reason, "graph_execution_classification_missing");

        let mut conflicting = attempt(1, 0, Some(3), Some(4), 0, 10, 10, SchedulePhase::Measurement);
        conflicting.graph_workload = true;
        conflicting.graph_claim_calibration = Some(GraphClaimCalibrationSnapshot {
            execution_complete: false,
            ..physical_graph_claim_snapshot(vec![complete_graph_claim_row()])
        });
        let conflicting_evidence = graph_claim_calibration_evidence(&[conflicting]);
        assert!(!conflicting_evidence.passes());
        assert_eq!(conflicting_evidence.attempts[0].reason, "graph_execution_classification_conflicted");
    }
}
