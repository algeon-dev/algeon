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

use super::schedule::SchedulePhase;

/// One attempt's measured outcome, projected from `QueryAttemptReport`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AttemptObservation {
    pub(super) sequence: usize,
    pub(super) phase: SchedulePhase,
    pub(super) query_label: String,
    pub(super) graph_workload: bool,
    pub(super) attempt_id: String,
    pub(super) disposition: String,
    pub(super) outcome: String,
    /// Bounded closed-vocabulary classification of the primary failure, when
    /// the attempt failed. Never a raw error string.
    pub(super) failure: Option<AttemptFailureClass>,
    pub(super) succeeded: bool,
    /// Present only when the attempt reached native GPU execution.
    pub(super) reached_gpu: bool,
    pub(super) device_ordinal: Option<usize>,
    pub(super) grant_bytes: Option<u64>,
    pub(super) admission_sequence: Option<u64>,
    pub(super) admission_wait_ns: Option<u64>,
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
    pub(super) report_aggregate_equality: ReportAggregateEquality,
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
    pub(super) query_label: String,
    pub(super) attempt_id: String,
    pub(super) disposition: String,
    pub(super) outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) failure: Option<AttemptFailureClass>,
    pub(super) device_ordinal: Option<usize>,
    pub(super) grant_bytes: Option<u64>,
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

#[derive(Debug, Clone, Serialize)]
pub(super) struct ScenarioEvidence {
    pub(super) same_device: Vec<ScenarioClause>,
    pub(super) multiple_devices: Vec<ScenarioClause>,
    /// Logical slot occupancy is recorded, never asserted: the ledger derives
    /// these counts from the immutable profile, so they cannot witness CUDA
    /// stream identity. Nsight owns stream-level overlap.
    pub(super) logical_slot_occupancy: LogicalSlotOccupancyEvidence,
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
    pub(super) fn reduce(
        observations: &[AttemptObservation],
        devices: &[DeviceConfiguration],
        occupancy: &BTreeMap<usize, DeviceOccupancyPeaks>,
        warmup_ns: u64,
        measurement_ns: u64,
    ) -> Self {
        let measured =
            observations.iter().filter(|attempt| attempt.phase == SchedulePhase::Measurement).collect::<Vec<_>>();
        let throughput = throughput(&measured, measurement_ns);
        let latency = latency(&measured);
        let placement = placement(observations, devices, occupancy);
        let attempts = observations.iter().map(row).collect::<Vec<_>>();
        let scenarios = scenarios(observations, &measured, devices, &placement, occupancy);
        let window = MeasurementWindow {
            warmup_ns,
            measurement_ns,
            warmup_attempts: (observations.len() - measured.len()) as u64,
            measurement_attempts: measured.len() as u64,
        };
        let report_aggregate_equality = equality(&window, &placement, &throughput, &attempts);
        Self { window, throughput, latency, placement, attempts, scenarios, report_aggregate_equality }
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
        query_label: attempt.query_label.clone(),
        attempt_id: attempt.attempt_id.clone(),
        disposition: attempt.disposition.clone(),
        outcome: attempt.outcome.clone(),
        failure: attempt.failure.clone(),
        device_ordinal: attempt.device_ordinal,
        grant_bytes: attempt.grant_bytes,
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
) -> ScenarioEvidence {
    ScenarioEvidence {
        same_device: same_device_clauses(observations, placement),
        multiple_devices: multiple_device_clauses(measured, devices, placement),
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
) -> Vec<ScenarioClause> {
    let eligible = placement.iter().filter(|device| device.configuration.starting_health_open).collect::<Vec<_>>();
    let mut clauses = Vec::new();

    let one_ordinal_each = measured.iter().all(|attempt| attempt.device_ordinal.is_some());
    clauses.push(if one_ordinal_each && !measured.is_empty() {
        clause(
            "one_ordinal_per_attempt",
            ClauseStatus::Evidenced,
            format!("{} measured attempts each recorded exactly one ordinal", measured.len()),
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
    clauses.push(clause(
        "device_block_or_quarantine_fallback",
        ClauseStatus::NotEvidenced,
        "forcing a device block requires the P6.6 fault-injection capability; this phase does not own it".to_string(),
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
            query_label: "q01".to_string(),
            graph_workload: false,
            attempt_id: format!("attempt-{sequence:032x}"),
            disposition: "Native".to_string(),
            outcome: "succeeded".to_string(),
            failure: None,
            succeeded: true,
            reached_gpu: true,
            device_ordinal: Some(ordinal),
            grant_bytes: Some(1024),
            admission_sequence: Some(sequence as u64),
            admission_wait_ns: Some(10),
            grant_id,
            domain_id,
            total_ns: Some(total_ns),
            queue_wait_ns: Some(5),
            physical_execution_ns: Some(total_ns / 2),
            submitted_ns,
            outcome_terminal_ns,
        }
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
}
