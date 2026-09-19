//! The GPU-touching measured run.
//!
//! This module is deliberately thin: it builds the backend from an already
//! validated profile, drives the already planned arrivals, samples the public
//! snapshot accessors, and projects reports into the pure reduction in
//! [`super::aggregate`]. Every decision that can be made without a GPU is made
//! elsewhere.

#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceInstallationBounds, AlgeonGpuDeviceProfile,
        AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    },
    error::DataFusionErrorExt,
    native_exec::AlgeonNativeExecutionConfig,
    planner::DataFusionCpuExecutionPolicy,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{datasource::MemTable, execution::SessionStateBuilder, prelude::SessionContext};
use futures::StreamExt as _;
use query_engine::{
    metrics::{BackendMetric, backend_metric_descriptors},
    observability::{
        AttemptComposedScheduleProjection, AttemptDisposition, COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1,
        QueryAttemptReport,
    },
    runtime::{
        BackendAttemptDisposition, BackendAttemptDuration, BackendAttemptOutcome, BackendHistogramSnapshot,
        BackendMetricsSnapshot,
    },
};
use serde_json::Value;

#[cfg(test)]
use super::aggregate::scenario_clauses_pass;
use super::{
    aggregate::{
        AttemptFailureClass, AttemptObservation, AttemptTerminalEvidence, CacheQuarantineEvidence, DeviceConfiguration,
        DeviceOccupancyPeaks, DomainExhaustionEvidence, GraphClaimCalibrationRow, GraphClaimCalibrationSnapshot,
        MeasuredAggregate, OverloadFairnessEvidence, PhysicalAllocatorPressureEvidence, PhysicalPressureLeaseFacts,
        QuarantineCyclesEvidence, QuarantineFallbackEvidence, StepWindow,
    },
    artifact::{
        ArrivalPlanEvidence, BackendCloseEvidence, BackendCloseFailure, BackendCloseOutcome,
        BackendCloseSnapshotEvidence, BackendEvidence, BackendSnapshotSample, CancellationBarrierEvidence,
        CloseUnderLoadEvidence, CompletionSinkEvidence, ComposedScheduleArtifact, ComposedScheduleOccurrence,
        DeadlineBarrierEvidence, MeasuredRunArtifact, OperabilityEvidence, QpsEnvelopeEvidence, RunDeadlineEvidence,
        TimeSeriesEvidence, TraceEvidence,
    },
    cache_calibration::{CacheSameKeyCohortEvidence, CacheSameKeyProbeEvidence},
    metrics_audit::{MetricSnapshotObservation, MetricsAudit},
    operability::{
        AttributionCompleteness, FEATURE_OFF_CUGRAPH_OPERATION, FEATURE_OFF_CUGRAPH_QUERY_LABEL,
        FeatureOffCugraphEvidence, FeatureOffCugraphSourceError, OtlpExportEvidence,
    },
    profile::{
        CacheMode, CacheSameKeyProbeProfile, CacheSameKeyQueryProfile, DeviceOwnership, RateOverrunPolicy, RunProfile,
        ScenarioRequirementsProfile,
    },
    schedule::{ArrivalPlan, SchedulePhase},
    timeseries::{self, DeviceSeriesRow, ProcessSeriesRow},
    workload::{CompletionSink, CompletionSinkCounters, SelectedQuery, select_queries},
};
use crate::ToolResult;
mod arrival;
mod barriers;
mod evidence;
mod otlp;
mod probes;
mod sampler;

use arrival::*;
use barriers::*;
use evidence::*;
pub(super) use otlp::OtlpExportConfig;
use otlp::OtlpExportWorker;
pub(super) use probes::validate_cugraph_runner_mode;
use probes::*;
use sampler::*;

/// How often the sampler polls the backend snapshot accessors at minimum.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
/// Bound on retained raw snapshot samples. Each carries the full per-device
/// report vectors, so this caps artifact size and peak memory; the per-device
/// peaks the scenario clauses use keep folding past the cap. A final sample
/// always displaces the retained tail when this bound is reached, so terminal
/// backend-metric counters are never audited against a pre-drain snapshot.
const MAX_SNAPSHOT_SAMPLES: usize = 1024;

/// The sampler cadence for a configured run window. Windows short enough to
/// fit the retention bound at the base interval keep 250 ms; longer windows
/// (long plateaus) trade cadence for whole-window coverage instead of
/// truncating the series mid-window.
fn sample_interval_for(window: Duration) -> Duration {
    let window_ns = window.as_nanos().min(u128::from(u64::MAX)) as u64;
    // One slot is reserved for the terminal sample taken after the drive
    // loop, so the steady-state cadence gets MAX_SNAPSHOT_SAMPLES - 1 slots.
    let slots = (MAX_SNAPSHOT_SAMPLES - 1) as u64;
    SAMPLE_INTERVAL.max(Duration::from_nanos(window_ns / slots.max(1)))
}
/// Grace period for the backend close that ends the run.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);
/// How late a planned arrival may be before rate mode treats it as an overrun.
const RATE_OVERRUN_TOLERANCE: Duration = Duration::from_millis(1);
/// The feature-off UDTF validates this catalog entry's schema, then its
/// zero-child feature-disabled execution node returns the structured error.
/// It is never a DBLP fixture and is never scanned or staged on this path.
const FEATURE_OFF_CUGRAPH_EDGE_TABLE: &str = "__algeon_feature_off_cugraph_edges";

fn feature_off_cugraph_sql() -> String {
    format!("SELECT * FROM cugraph_pagerank('{FEATURE_OFF_CUGRAPH_EDGE_TABLE}')")
}

/// The runner's two bounded phases. Submission stops at the configured
/// warmup-plus-measurement boundary; already-admitted work may use only the
/// profile-owned completion grace to reach a native safe boundary and seal.
/// This keeps sustained-service measurement duration distinct from bounded
/// cooperative finalization; the post-submission drain grace is optional.
#[derive(Clone, Copy)]
struct RunTiming {
    started: Instant,
    warmup_deadline: Instant,
    submission_deadline: Instant,
    completion_deadline: Instant,
    measurement_window_ns: u64,
    completion_drain_grace_ns: u64,
}

impl RunTiming {
    fn new(plan: &ArrivalPlan, completion_drain_grace: Duration) -> Self {
        let started = Instant::now();
        let measurement_window_ns = plan.warmup_ns.saturating_add(plan.measurement_ns);
        let warmup_deadline = started + Duration::from_nanos(plan.warmup_ns);
        let submission_deadline = started + Duration::from_nanos(measurement_window_ns);
        let completion_deadline = submission_deadline + completion_drain_grace;
        Self {
            started,
            warmup_deadline,
            submission_deadline,
            completion_deadline,
            measurement_window_ns,
            completion_drain_grace_ns: completion_drain_grace.as_nanos().min(u128::from(u64::MAX)) as u64,
        }
    }

    fn deadline_evidence(self) -> RunDeadlineEvidence {
        run_deadline_evidence(self.measurement_window_ns, self.completion_drain_grace_ns, elapsed_ns(self.started))
    }
}

/// The feature-off graph request is intentionally absent from normal relational
/// round-robin selection.  The runner uses this profile-owned schedule to
/// issue the declared warmup and measurement probes beside live relational
/// attempts without changing the relational cohort's query distribution.

#[derive(Clone)]
pub(super) struct TraceSelection {
    pub(super) uri: Option<String>,
    pub(super) sha256: Option<String>,
}

/// The mutually exclusive scenario controls a measured run may carry. CLI
/// validation owns the pairwise-exclusion contract; this struct keeps the
/// dispatch signature from growing a parameter per probe.
#[derive(Clone, Default)]
pub(super) struct ProbeSelection {
    pub(super) quarantine_device: Option<usize>,
    pub(super) cancel_first_after: Option<Duration>,
    pub(super) deadline_first_after: Option<Duration>,
    pub(super) domain_exhaustion_probe: bool,
    pub(super) overload_fairness: bool,
    pub(super) physical_allocator_pressure: bool,
    /// Repeated quarantine cycles over a rotation of configured ordinals.
    pub(super) quarantine_cycles: Option<usize>,
    /// Close initiated at the midpoint of the measurement window.
    pub(super) close_under_load: bool,
    /// The explicit reference OTLP exporter cell. None means the run records
    /// that no exporter was requested rather than claiming export success.
    pub(super) otlp_export: Option<OtlpExportConfig>,
    /// The intentionally serialized public completion receiver. The sink is
    /// a probe-only consumer delay, not an execution or admission policy.
    pub(super) slow_report_sink: Option<Duration>,
}

#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
fn unix_epoch_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Runs the measurement on a dedicated multi-thread runtime.
///
/// The tool entrypoint pins a current-thread runtime, which would interleave
/// rather than overlap attempts. A scoped thread owns a multi-thread runtime so
/// the measured concurrency is real, without building a nested runtime inside
/// the caller's and without changing every other subcommand's entrypoint.
pub(super) fn run_on_owned_runtime(
    profile: &RunProfile,
    trace: TraceSelection,
    probes: ProbeSelection,
) -> ToolResult<MeasuredRunArtifact> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
                runtime.block_on(run_measured(profile, trace, probes))
            })
            .join()
            .map_err(|_| "the measured-run thread panicked")?
    })
}

/// Builds the backend, executes the plan, and returns the measured artifact.
async fn run_measured(
    profile: &RunProfile,
    trace: TraceSelection,
    probes: ProbeSelection,
) -> ToolResult<MeasuredRunArtifact> {
    let mut queries = select_queries(&profile.workloads)?;
    let relational_query_count = queries.len();
    let feature_off_cugraph_probe = append_feature_off_cugraph_probe(profile, &mut queries)?;
    validate_cugraph_runner_mode(profile, &queries)?;
    let plan = ArrivalPlan::plan(&profile.scheduling, relational_query_count, profile.session.shared_session_count)?;
    let feature_off_cugraph_schedule =
        profile.feature_off_cugraph_probe.zip(feature_off_cugraph_probe).map(|(profile, query_index)| {
            FeatureOffCugraphProbeSchedule {
                query_index,
                warmup_probe_attempts: profile.warmup_probe_attempts,
                measurement_probe_attempts: profile.measurement_probe_attempts,
            }
        });
    if plan.truncated_by_attempt_cap {
        // The artifact records this too; the operator warning keeps a
        // truncated offered process from reading as the profile's intent.
        eprintln!(
            "warning: offered process exceeds the planned-attempt cap {}; the plan was truncated and the artifact records truncated_by_attempt_cap",
            super::profile::MAX_PLANNED_ATTEMPTS
        );
    }
    let backend = build_backend(profile)?;
    let sessions = install_sessions(&backend, profile, &queries).await?;

    let ProbeSelection {
        quarantine_device,
        cancel_first_after,
        deadline_first_after,
        domain_exhaustion_probe,
        overload_fairness,
        physical_allocator_pressure,
        quarantine_cycles,
        close_under_load,
        otlp_export,
        slow_report_sink,
    } = probes;
    if profile.feature_off_cugraph_probe.is_some()
        && (quarantine_device.is_some()
            || cancel_first_after.is_some()
            || deadline_first_after.is_some()
            || domain_exhaustion_probe
            || overload_fairness
            || physical_allocator_pressure
            || quarantine_cycles.is_some()
            || close_under_load
            || otlp_export.is_some()
            || slow_report_sink.is_some())
    {
        return Err(
            "feature_off_cugraph_probe is an S1 mixed-load cell and cannot be combined with another runner probe"
                .into(),
        );
    }
    let completion_sink = slow_report_sink.map(CompletionSink::new);
    if let Some(sink) = completion_sink.as_ref() {
        for query in &mut queries {
            query.attach_completion_sink(sink.clone());
        }
    }
    let completion_drain_grace = Duration::from_secs(profile.completion_drain_grace_seconds);
    let sampler = Sampler::start(
        &backend,
        Duration::from_nanos(plan.warmup_ns.saturating_add(plan.measurement_ns)).saturating_add(completion_drain_grace),
    );
    let otlp_exporter = otlp_export.as_ref().map(|config| OtlpExportWorker::start(&backend, config));
    let outcome = match (
        quarantine_device,
        deadline_first_after,
        domain_exhaustion_probe,
        physical_allocator_pressure,
        overload_fairness,
        quarantine_cycles,
        close_under_load,
    ) {
        (Some(target_ordinal), None, false, false, false, None, false) => {
            drive_concurrent_quarantine(
                &backend,
                &sessions,
                &queries,
                &plan,
                completion_drain_grace,
                cancel_first_after,
                target_ordinal,
            )
            .await
        },
        (None, Some(deadline_after), false, false, false, None, false) => {
            drive_deadline_after_admission(&backend, &sessions, &queries, &plan, completion_drain_grace, deadline_after)
                .await
        },
        (None, None, true, false, false, None, false) => {
            drive_domain_exhaustion_probe(&backend, &sessions, &queries, &plan, completion_drain_grace).await
        },
        (None, None, false, true, false, None, false) => {
            drive_physical_allocator_pressure(
                &backend,
                &sessions,
                &queries,
                &plan,
                completion_drain_grace,
                profile.backend.device_profiles[0].ordinal,
            )
            .await
        },
        (None, None, false, false, true, None, false) => {
            drive_overload_fairness(
                &backend,
                &sessions,
                &queries,
                &plan,
                completion_drain_grace,
                profile.backend.queue.max_queued_attempts,
            )
            .await
        },
        (None, None, false, false, false, Some(cycles), false) => {
            drive_quarantine_cycles(
                &backend,
                &sessions,
                &queries,
                &plan,
                completion_drain_grace,
                cycles,
                &profile.backend.device_profiles.iter().map(|device| device.ordinal).collect::<Vec<_>>(),
            )
            .await
        },
        (None, None, false, false, false, None, true) => {
            drive_close_under_load(&backend, &sessions, &queries, &plan, completion_drain_grace).await
        },
        (None, None, false, false, false, None, false) => {
            drive(
                &backend,
                &sessions,
                &queries,
                &plan,
                completion_drain_grace,
                cancel_first_after,
                feature_off_cugraph_schedule,
            )
            .await
        },
        _ => unreachable!("CLI validation rejects combined scenario controls"),
    };
    let sampler = sampler.stop();
    let otlp_export = otlp_exporter.map(OtlpExportWorker::stop).unwrap_or_else(OtlpExportEvidence::not_requested);
    let driven = outcome?;
    // Close after every submitted completion has been observed, so the
    // attempt rows remain canonical. The evidence retains a timeout or cleanup
    // failure instead of erasing it with `.ok()`.
    let close = close_backend(&backend);
    // The same-key cache proof needs a cold cache on exactly one allocation
    // domain. It therefore runs only after the normal traffic backend closes,
    // on a fresh backend restricted to the profile-owned ordinal.
    let cache_same_key_probe = match profile.cache.stress.as_ref().and_then(|stress| stress.same_key_probe.as_ref()) {
        Some(probe) => Some(run_cache_same_key_probe(profile, probe).await?),
        None => None,
    };

    let devices = device_configurations(&backend);
    let quarantine_fallback = driven.quarantine_barrier.as_ref().map(|barrier| {
        let mut evidence = QuarantineFallbackEvidence::from_barrier(
            barrier.target_ordinal,
            barrier.pre_ack_witness_offset_ns,
            barrier.pre_ack_ledger_sequence,
            barrier.pre_ack_live_grants,
            barrier.pre_ack_active_query_jobs,
            barrier.acknowledgement_offset_ns,
            barrier.device_health_generation,
            barrier.newly_quarantined,
            barrier.target_ledger_closed,
            &barrier.post_ack_sequences,
            barrier.pre_existing_target_work_released,
            CacheQuarantineEvidence::from_snapshots(
                barrier.pre_ack_cache.clone(),
                barrier.post_ack_cache.clone(),
                barrier.final_cache.clone(),
            ),
            &driven.observations,
        );
        if profile.cache.stress.as_ref().is_some_and(|stress| stress.require_resident_quarantine) {
            evidence.require_cache_residency();
        }
        evidence
    });
    let step_windows = if plan.steps.len() > 1 {
        plan.steps
            .iter()
            .map(|step| StepWindow {
                step_index: step.step_index,
                concurrency: step.concurrency,
                arrival_rate_per_second: step.arrival_rate_per_second,
                rate_overrun: step.rate_overrun,
                measurement_ns: step.measurement_ns,
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let aggregate = MeasuredAggregate::reduce_full(
        &driven.observations,
        &devices,
        &sampler.occupancy,
        plan.warmup_ns,
        plan.measurement_ns,
        quarantine_fallback,
        &step_windows,
    );
    let overload_fairness_evidence = overload_fairness.then(|| {
        OverloadFairnessEvidence::from_observations(
            &driven.observations,
            &devices,
            plan.arrivals.len(),
            profile.backend.queue.max_queued_attempts,
            driven.observed_peak_queued_attempts,
            driven.dropped_overrun_arrivals,
            driven.begin_failures,
            driven.join_failures,
        )
    });
    let aggregate = match overload_fairness_evidence.clone() {
        Some(evidence) => aggregate.with_overload_fairness(evidence),
        None => aggregate,
    };
    let domain_exhaustion =
        domain_exhaustion_probe.then(|| DomainExhaustionEvidence::from_observations(&driven.observations));
    let physical_allocator_pressure = driven.physical_allocator_pressure.as_ref().map(|barrier| {
        PhysicalAllocatorPressureEvidence::from_observations(
            barrier.pressure,
            barrier.release,
            barrier.pressure_established_before_q01,
            barrier.pressure_released_before_recovery,
            &driven.observations,
        )
        .with_final_close_quiescent(close.is_quiescent())
    });
    let quarantine_cycles_evidence = quarantine_cycles.map(|configured| {
        QuarantineCyclesEvidence::from_cycle_evidence(
            configured,
            driven
                .quarantine_cycle_barriers
                .iter()
                .map(|barrier| {
                    let mut evidence = QuarantineFallbackEvidence::from_barrier(
                        barrier.target_ordinal,
                        barrier.pre_ack_witness_offset_ns,
                        barrier.pre_ack_ledger_sequence,
                        barrier.pre_ack_live_grants,
                        barrier.pre_ack_active_query_jobs,
                        barrier.acknowledgement_offset_ns,
                        barrier.device_health_generation,
                        barrier.newly_quarantined,
                        barrier.target_ledger_closed,
                        &barrier.post_ack_sequences,
                        barrier.pre_existing_target_work_released,
                        CacheQuarantineEvidence::from_snapshots(
                            barrier.pre_ack_cache.clone(),
                            barrier.post_ack_cache.clone(),
                            barrier.final_cache.clone(),
                        ),
                        &driven.observations,
                    );
                    if profile.cache.stress.as_ref().is_some_and(|stress| stress.require_resident_quarantine) {
                        evidence.require_cache_residency();
                    }
                    evidence
                })
                .collect(),
        )
    });
    let close_under_load_evidence = driven.close_under_load.as_ref().map(|barrier| {
        let sealed_reports = driven.observations.len() as u64;
        CloseUnderLoadEvidence {
            close_requested_offset_ns: barrier.close_requested_offset_ns,
            close_initiated: barrier.close_initiated,
            admitted_at_close: barrier.admitted_at_close as u64,
            queued_at_close: barrier.queued_at_close as u64,
            in_flight_at_close: barrier.in_flight_at_close as u64,
            submitted_attempts: barrier.submitted_attempts,
            sealed_reports,
            join_failures: driven.join_failures,
            backend_closing_sealed: driven
                .observations
                .iter()
                .filter(|observation| is_typed_backend_closing(observation))
                .count() as u64,
            in_flight_terminal_after_close: driven
                .observations
                .iter()
                .filter(|observation| barrier.in_flight_sequences.contains(&observation.sequence))
                .count() as u64,
            in_flight_succeeded_after_close: driven
                .observations
                .iter()
                .filter(|observation| {
                    barrier.in_flight_sequences.contains(&observation.sequence) && observation.succeeded
                })
                .count() as u64,
            accounting_exact: barrier.submitted_attempts == sealed_reports + driven.join_failures,
        }
    });
    let completion_sink_evidence =
        completion_sink.as_ref().map(|sink| completion_sink_evidence(sink.counters(), &plan, &driven, &close));
    let metrics_audit = MetricsAudit::from_snapshots(
        &sampler.metric_snapshots,
        &devices.iter().map(|device| device.ordinal).collect::<Vec<_>>(),
        aggregate.attempts.len(),
        sampler.sample_interval.as_millis().min(u128::from(u64::MAX)) as u64,
        sampler.raw_snapshot_retention_truncated,
    );
    let attribution = AttributionCompleteness::from_observations(&driven.observations);
    let feature_off_cugraph = profile.feature_off_cugraph_probe.map(|probe| {
        FeatureOffCugraphEvidence::from_observations(
            probe,
            &driven.observations,
            &driven.feature_off_cugraph_source_errors,
            driven.begin_failures,
            driven.join_failures,
        )
    });
    let configured_active_attempt_capacity = profile
        .backend
        .device_profiles
        .iter()
        .try_fold(0_usize, |total, device| total.checked_add(device.max_active_attempts))
        .ok_or("configured active-attempt capacity overflows")?;
    let qps_envelope = profile.qps_envelope.map(|contract| {
        QpsEnvelopeEvidence::from_measurement(
            contract,
            &aggregate,
            configured_active_attempt_capacity,
            profile.backend.queue.max_queued_attempts,
            plan.truncated_by_attempt_cap,
            &driven.queue_high_watermarks,
        )
    });
    let vram_calibration = profile.vram_calibration.map(|policy| {
        let configured_ordinals =
            profile.backend.device_profiles.iter().map(|device| device.ordinal).collect::<Vec<_>>();
        let query_domain_pool_claims = driven
            .observations
            .iter()
            .filter_map(|attempt| {
                let pool = attempt.domain_pool_usage?;
                Some(timeseries::QueryDomainPoolClaim {
                    device_ordinal: attempt.device_ordinal?,
                    grant_id: attempt.grant_id?,
                    grant_bytes: attempt.grant_bytes?,
                    reserved_high_bytes: pool.reserved_high_bytes,
                    used_high_bytes: pool.used_high_bytes,
                })
            })
            .collect::<Vec<_>>();
        timeseries::vram_calibration_evidence(
            &sampler.device_series,
            &query_domain_pool_claims,
            &configured_ordinals,
            policy,
        )
    });
    let graph_claim_calibration = profile
        .graph_claim_calibration
        .map(|_| super::aggregate::graph_claim_calibration_evidence(&driven.observations));
    let qualifying = selected_scenario_clauses_pass(
        &aggregate.scenarios,
        deadline_first_after.is_some()
            || domain_exhaustion_probe
            || overload_fairness
            || physical_allocator_pressure.is_some(),
        profile.scenario_requirements(),
    ) && has_measurement_attempts(&aggregate)
        && every_configured_step_has_measurement_attempts(&plan, &aggregate)
        && aggregate.report_aggregate_equality.equal
        && metrics_audit.passes()
        && (completion_sink_evidence.is_some() || driven.deadline.reports_sealed_within_deadline)
        && driven.cancellation_barrier.as_ref().is_none_or(CancellationBarrierEvidence::is_evidenced)
        && driven.deadline_barrier.as_ref().is_none_or(DeadlineBarrierEvidence::is_evidenced)
        && domain_exhaustion.as_ref().is_none_or(DomainExhaustionEvidence::is_evidenced)
        && overload_fairness_evidence.as_ref().is_none_or(OverloadFairnessEvidence::is_evidenced)
        && physical_allocator_pressure.as_ref().is_none_or(PhysicalAllocatorPressureEvidence::is_evidenced)
        && quarantine_cycles_evidence.as_ref().is_none_or(QuarantineCyclesEvidence::is_evidenced)
        && close_under_load_evidence.as_ref().is_none_or(CloseUnderLoadEvidence::is_evidenced)
        && completion_sink_evidence.as_ref().is_none_or(CompletionSinkEvidence::is_evidenced)
        && attribution.passes()
        && feature_off_cugraph.as_ref().is_none_or(FeatureOffCugraphEvidence::passes)
        && qps_envelope.as_ref().is_none_or(QpsEnvelopeEvidence::passes)
        && vram_calibration.as_ref().is_none_or(timeseries::VramCalibrationEvidence::passes)
        && graph_claim_calibration.as_ref().is_none_or(super::aggregate::GraphClaimCalibrationEvidence::passes)
        && otlp_export.passes()
        && close.is_quiescent();
    let backend_generation = backend.backend_metrics_snapshot().backend_generation;
    Ok(MeasuredRunArtifact {
        qualifying,
        backend: BackendEvidence {
            backend_generation,
            shared_session_count: profile.session.shared_session_count,
            queue_max_queued_attempts: profile.backend.queue.max_queued_attempts,
            queue_max_queue_wait_ms: profile.backend.queue.max_queue_wait_ms,
            // Retain the historical artifact field as an explicit zero fact;
            // queue admission no longer has a configurable overtake budget.
            queue_max_overtakes_per_waiter: 0,
            // `GpuBackendDeviceStartupRecord` is not one of the nine
            // `Serialize` snapshot types and transitively holds a cuDF type, so
            // the artifact records the same startup facts through the
            // aggregate's own device configuration rather than widening that
            // frozen JSON contract from here.
            device_startup_records: serde_json::to_value(&devices)?,
            close,
        },
        selected_queries: queries.iter().map(SelectedQuery::descriptor).collect(),
        arrival_plan: ArrivalPlanEvidence::new(
            &plan,
            driven.dropped_overrun_arrivals,
            driven.begin_failures,
            driven.join_failures,
            driven.storm_cancellations_requested,
        ),
        cache_state: match profile.cache.mode {
            CacheMode::Disabled => "disabled",
            CacheMode::Enabled => "enabled",
        },
        composed_schedules: driven
            .observations
            .iter()
            .filter_map(|observation| observation.composed_schedule.clone())
            .collect(),
        cache_same_key_probe,
        qps_envelope,
        graph_claim_calibration,
        aggregate,
        deadline: driven.deadline,
        cancellation_barrier: driven.cancellation_barrier,
        deadline_barrier: driven.deadline_barrier,
        domain_exhaustion,
        physical_allocator_pressure,
        quarantine_cycles: quarantine_cycles_evidence,
        close_under_load: close_under_load_evidence,
        completion_sink: completion_sink_evidence,
        feature_off_cugraph,
        metrics_audit,
        operability: OperabilityEvidence { attribution, otlp_export },
        time_series: TimeSeriesEvidence {
            sample_interval_ns: sampler.sample_interval.as_nanos().min(u128::from(u64::MAX)) as u64,
            retention_downsampled: sampler.sample_interval > SAMPLE_INTERVAL,
            maximum_cuda_nvml_sample_gap: timeseries::maximum_physical_sample_gap(&sampler.device_series),
            utilization: timeseries::utilization_summary(&sampler.device_series),
            drift: timeseries::drift_summary(&sampler.device_series, &sampler.process_series),
            vram_calibration,
            observation_pull: timeseries::observation_pull_summary(&sampler.process_series),
            device_rows: sampler.device_series,
            process_rows: sampler.process_series,
        },
        snapshots: sampler.samples,
        trace: TraceEvidence { captured_by_runner: false, uri: trace.uri, sha256: trace.sha256 },
    })
}

/// A completed warmup cohort is useful for cache conditioning but cannot
/// satisfy a measured-run contract. In particular, duration-paced closed-loop
/// work may spend the entire configured window draining a long warmup round;
/// retain that artifact as non-qualifying rather than reporting zero-QPS data
/// as a passing service measurement.
fn has_measurement_attempts(aggregate: &MeasuredAggregate) -> bool {
    aggregate.window.measurement_attempts != 0
}

/// An explicit multi-phase schedule is an envelope only when every
/// declared phase emitted a measured attempt. A long warmup or prior phase
/// that consumes a later phase's wall-clock interval is a coverage failure,
/// not a zero-QPS datum. A single implicit phase is covered by
/// [`has_measurement_attempts`]; it intentionally has no per-step aggregate.
fn every_configured_step_has_measurement_attempts(plan: &ArrivalPlan, aggregate: &MeasuredAggregate) -> bool {
    if plan.steps.len() <= 1 {
        return true;
    }
    plan.steps.iter().all(|planned| {
        aggregate
            .steps
            .iter()
            .any(|measured| measured.step_index == planned.step_index && measured.throughput.completions != 0)
    })
}

/// Runs the explicit cold-cache coalescing cohorts. The throughput run may
/// span several ordinal-local domains, so it cannot prove that equal keys
/// actually joined one cache. This probe isolates one configured ordinal on a
/// fresh backend and retains only public observations before and after each
/// cohort.
async fn run_cache_same_key_probe(
    profile: &RunProfile,
    probe: &CacheSameKeyProbeProfile,
) -> ToolResult<CacheSameKeyProbeEvidence> {
    let mut probe_profile = profile.clone();
    probe_profile.backend.device_profiles.retain(|device| device.ordinal == probe.target_device_ordinal);
    let backend = build_backend(&probe_profile)?;
    let result = async {
        let queries = select_queries(&probe_profile.workloads)?;
        validate_cugraph_runner_mode(&probe_profile, &queries)?;
        let sessions = install_sessions(&backend, &probe_profile, &queries).await?;
        let source_query_index = same_key_probe_query_index(&queries, &probe.source, "source")?;
        let bfs_query_index = same_key_probe_query_index(&queries, &probe.bfs, "bfs")?;
        let cohort_timeout = Duration::from_secs(probe.max_cohort_seconds);
        let source = run_cache_same_key_cohort(
            &backend,
            &sessions,
            &queries,
            source_query_index,
            probe.target_device_ordinal,
            probe.cohort_width,
            cohort_timeout,
        )
        .await?;
        let bfs = run_cache_same_key_cohort(
            &backend,
            &sessions,
            &queries,
            bfs_query_index,
            probe.target_device_ordinal,
            probe.cohort_width,
            cohort_timeout,
        )
        .await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((source, bfs))
    }
    .await;
    let close = close_backend(&backend);
    let (source, bfs) = result?;
    Ok(CacheSameKeyProbeEvidence {
        target_device_ordinal: probe.target_device_ordinal,
        cohort_width: probe.cohort_width,
        source,
        bfs,
        close_quiescent: close.is_quiescent(),
    })
}

fn same_key_probe_query_index(
    queries: &[SelectedQuery],
    requested: &CacheSameKeyQueryProfile,
    cohort: &str,
) -> ToolResult<usize> {
    let mut matches =
        queries.iter().enumerate().filter(|(_, query)| query.kind == requested.kind && query.label == requested.label);
    let index = matches.next().map(|(index, _)| index).ok_or_else(|| {
        format!(
            "cache same-key {cohort} cohort cannot find selected {} query {}",
            requested.kind.label(),
            requested.label
        )
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "cache same-key {cohort} cohort matches more than one selected {} query {}; a same-key probe requires exactly one fixture selection",
            requested.kind.label(),
            requested.label
        )
        .into());
    }
    Ok(index)
}

async fn run_cache_same_key_cohort(
    backend: &AlgeonGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    query_index: usize,
    target_device_ordinal: usize,
    cohort_width: usize,
    cohort_timeout: Duration,
) -> ToolResult<CacheSameKeyCohortEvidence> {
    let query = queries.get(query_index).ok_or("cache same-key cohort query index is out of bounds")?;
    let cohort_id = format!("s7-same-key-{}-ordinal-{target_device_ordinal}", query.label);
    let expected_correlation_fingerprints = (0..cohort_width)
        .map(|sequence| {
            query_engine::runtime::cache_observation_fingerprint(&format!("{cohort_id}-attempt-{sequence}"))
        })
        .collect::<Vec<_>>();
    let cache_reports_before = cache_reports_snapshot(backend)?;
    let source_cache_events_before = cache_source_event_snapshots(backend)?;
    let bfs_singleflight_before = cugraph_bfs_singleflight_snapshot(backend);
    let started = Instant::now();
    let deadline = started + cohort_timeout;
    let mut tasks = Vec::with_capacity(cohort_width);
    let mut observations = Vec::with_capacity(cohort_width);
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    for sequence in 0..cohort_width {
        let arrival = super::schedule::ScheduledArrival {
            sequence,
            round: 0,
            step_index: 0,
            offset_ns: 0,
            phase: SchedulePhase::Measurement,
            rate_overrun: RateOverrunPolicy::Delay,
            query_index,
            session_index: sequence,
        };
        match spawn_attempt_task_with_query_id(
            backend,
            sessions,
            queries,
            arrival,
            false,
            started,
            deadline,
            None,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
            Some(format!("{cohort_id}-attempt-{sequence}")),
        ) {
            Ok(task) => tasks.push(task),
            Err(()) => begin_failures = begin_failures.saturating_add(1),
        }
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let cache_reports_after = cache_reports_snapshot(backend)?;
    let source_cache_events_after = cache_source_event_snapshots(backend)?;
    let bfs_singleflight_after = cugraph_bfs_singleflight_snapshot(backend);
    let successful_attempts = observations.iter().filter(|observation| observation.succeeded).count() as u64;
    let measurement_attempts =
        observations.iter().filter(|observation| observation.phase == SchedulePhase::Measurement).count() as u64;
    let measurement_successes = observations
        .iter()
        .filter(|observation| observation.phase == SchedulePhase::Measurement && observation.succeeded)
        .count() as u64;
    let target_ordinal_attempts =
        observations.iter().filter(|observation| observation.device_ordinal == Some(target_device_ordinal)).count()
            as u64;
    let other_ordinal_attempts = observations
        .iter()
        .filter(|observation| observation.device_ordinal.is_some_and(|ordinal| ordinal != target_device_ordinal))
        .count() as u64;
    Ok(CacheSameKeyCohortEvidence {
        cohort_id,
        expected_correlation_fingerprints,
        fixture_kind: query.kind,
        query_label: query.label.clone(),
        submitted_attempts: cohort_width as u64,
        begin_failures,
        join_failures,
        measurement_attempts,
        measurement_successes,
        successful_attempts,
        target_ordinal_attempts,
        other_ordinal_attempts,
        cache_reports_before,
        cache_reports_after,
        source_cache_events_before,
        source_cache_events_after,
        bfs_singleflight_before,
        bfs_singleflight_after,
    })
}

fn cache_reports_snapshot(backend: &AlgeonGpuBackend) -> ToolResult<Value> {
    Ok(serde_json::to_value(backend.device_cache_reports()?)?)
}

fn cache_quarantine_observation(backend: &AlgeonGpuBackend, ordinal: usize) -> ToolResult<Value> {
    let report =
        serde_json::to_value(backend.device_cache_reports()?.into_iter().find(|r| r.device_ordinal == ordinal))?;
    let Some(report) = report.as_object() else {
        return Err(format!("missing cache report for target ordinal {ordinal}").into());
    };
    let cap = report.get("cache_cap_bytes").and_then(Value::as_u64).unwrap_or(0);
    let ledger = report.get("ledger_evictable_resident_bytes").and_then(Value::as_u64).unwrap_or(0)
        + report.get("ledger_pinned_in_use_bytes").and_then(Value::as_u64).unwrap_or(0);
    let physical = report.get("physical_current_bytes").and_then(Value::as_u64).unwrap_or(u64::MAX);
    if cap == 0 || ledger > cap || physical > ledger || physical > cap {
        return Err(format!("target ordinal {ordinal} cache residency is absent or inconsistent").into());
    }
    Ok(Value::Object(report.clone()))
}

fn cache_source_event_snapshots(backend: &AlgeonGpuBackend) -> ToolResult<Value> {
    Ok(serde_json::to_value(backend.device_cache_source_event_snapshots()?)?)
}

/// Applies only the ordinary scenario clauses that the requested physical run
/// is able to prove. Dedicated probes have their own strict predicates: the
/// deadline barrier is serialized, while overload uses sealed queue facts
/// rather than ordinary overlap clauses. The complete aggregate remains in
/// the artifact in either case.
fn selected_scenario_clauses_pass(
    scenarios: &super::aggregate::ScenarioEvidence,
    dedicated_p6_6_scenario: bool,
    requirements: ScenarioRequirementsProfile,
) -> bool {
    if dedicated_p6_6_scenario {
        return true;
    }
    let failed = |clauses: &[super::aggregate::ScenarioClause]| {
        clauses.iter().any(|clause| clause.status == super::aggregate::ClauseStatus::Failed)
    };
    let same_device_failed = requirements.require_same_device_overlap && failed(&scenarios.same_device);
    let progress_failed = requirements.require_all_device_progress
        && scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "progress_on_all_eligible_devices")
            .is_some_and(|clause| clause.status == super::aggregate::ClauseStatus::Failed);
    let heterogeneity_failed = requirements.require_heterogeneous_device_profiles
        && scenarios
            .multiple_devices
            .iter()
            .find(|clause| clause.clause == "heterogeneous_ordinal_profiles")
            .is_some_and(|clause| clause.status == super::aggregate::ClauseStatus::Failed);
    !same_device_failed && !progress_failed && !heterogeneity_failed
}

/// Projects the explicit handoff-pressure contract. Every step must offer
/// more completions than one serialized sink can drain; the profile preflight
/// rejects any slower step before fixtures or CUDA are touched. The bounded
/// drain allowance is derived from the actual started-task count and the
/// configured sink delay, while backend close remains bounded by
/// `CLOSE_TIMEOUT` in [`close_backend`].
fn completion_sink_evidence(
    counters: CompletionSinkCounters,
    plan: &ArrivalPlan,
    driven: &DrivenRun,
    close: &BackendCloseEvidence,
) -> CompletionSinkEvidence {
    let offered_rate_per_second =
        plan.steps.iter().filter_map(|step| step.arrival_rate_per_second).min().unwrap_or_default();
    let sink_delay_ms = counters.delay.as_millis().min(u128::from(u64::MAX)) as u64;
    let offered_rate_exceeds_serial_drain_capacity =
        u128::from(offered_rate_per_second).saturating_mul(u128::from(sink_delay_ms)) > 1_000;
    let sealed_reports = driven.observations.len() as u64;
    let submitted_attempts = sealed_reports.saturating_add(driven.join_failures);
    let configured_window_ns = plan.warmup_ns.saturating_add(plan.measurement_ns);
    let sink_budget_ns = counters.delay.as_nanos().saturating_mul(u128::from(submitted_attempts));
    let drain_deadline_ns = u128::from(configured_window_ns).saturating_add(sink_budget_ns);
    CompletionSinkEvidence {
        sink_delay_ms,
        offered_rate_per_second,
        offered_rate_exceeds_serial_drain_capacity,
        submitted_attempts,
        sealed_reports,
        begin_failures: driven.begin_failures,
        join_failures: driven.join_failures,
        drains_started: counters.drains_started,
        drains_completed: counters.drains_completed,
        report_accounting_exact: submitted_attempts == sealed_reports.saturating_add(driven.join_failures)
            && driven.begin_failures == 0,
        bounded_drain: u128::from(driven.deadline.elapsed_until_reports_ns) <= drain_deadline_ns,
        close_quiescent: close.is_quiescent(),
    }
}

/// Flattens an error's source chain into one bounded line.
fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut rendered = error.to_string();
    let mut source = error.source();
    // Bounded so a cyclic or deep chain cannot produce an unbounded artifact
    // or log line.
    for _ in 0..8 {
        let Some(inner) = source else {
            break;
        };
        rendered.push_str(&format!(" | caused by: {inner}"));
        source = inner.source();
    }
    rendered
}

pub(super) fn build_backend(profile: &RunProfile) -> ToolResult<AlgeonGpuBackend> {
    if profile.backend.device_profiles.is_empty() {
        return Err("a measured run requires at least one backend.device_profiles entry".into());
    }
    let native_execution = profile.backend.native_execution;
    let queue = AlgeonGpuQueuePolicy::default()
        .with_max_queued_attempts(profile.backend.queue.max_queued_attempts)
        .with_max_queue_wait(Duration::from_millis(profile.backend.queue.max_queue_wait_ms));
    let backend = AlgeonGpuBackend::builder()
        .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            native_execution.device_budget_bytes,
            native_execution.max_source_chunk_bytes,
            native_execution.max_row_groups_per_chunk,
            native_execution.max_retry_attempts,
            native_execution.min_retry_chunk_row_groups,
        ))
        // Every cell of the backend-service stress program runs GPU-only: a
        // final plan that retains any DataFusion CPU operator fails planning
        // with a structured FinalPlanRequirementUnsatisfied rather than
        // silently falling back.
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .queue_policy(queue)
        .devices(profile.backend.device_profiles.iter().map(|device| {
            let p = AlgeonGpuDeviceProfile::new(device.ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: device.memory_limit_bytes })
                .with_backend_reserve_bytes(device.backend_reserve_bytes)
                .with_cache_cap_bytes(device.cache_cap_bytes)
                .with_default_query_limit_bytes(device.default_query_limit_bytes)
                .with_source_work_per_attempt(device.source_work_per_attempt)
                .with_max_active_attempts(device.max_active_attempts)
                .with_telemetry(device.install_telemetry)
                .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(
                    device.installation.stream_pool_size,
                    device.installation.worker_count,
                    device.installation.lane_count,
                ));
            let ownership = match device.ownership {
                DeviceOwnership::WholeDeviceExclusive => GpuMemoryOwnership::WholeDeviceExclusive,
                DeviceOwnership::StaticPartition { allocation_limit_bytes } => {
                    GpuMemoryOwnership::StaticPartition { allocation_limit_bytes }
                },
            };
            AlgeonGpuDeviceConfig::new(p, ownership)
        }))
        // The engine's device-config rejection carries the field and reason;
        // flattening the source chain keeps that actionable instead of leaving
        // it behind an omitted source.
        .build()
        .map_err(|error| format!("backend build rejected the profile: {}", error_chain(&error)))?;
    Ok(backend)
}

/// The shared session pool, keyed by profile workload position. TPC-H and
/// TPC-DS share table names (`customer`), and several DBLP graph
/// fixtures with those same registration names, so one context pool per
/// fixture workload is the only correct routing boundary.
pub(super) type SessionPool = BTreeMap<usize, Vec<SessionContext>>;

/// `shared_session_count` contexts per selected fixture workload. Sessions are
/// shared only across attempts bound to that same fixture entry.
pub(super) async fn install_sessions(
    backend: &AlgeonGpuBackend,
    profile: &RunProfile,
    queries: &[SelectedQuery],
) -> ToolResult<SessionPool> {
    let needs_cugraph_sql = queries.iter().any(|query| query.graph);
    let needs_feature_off_probe = queries.iter().any(|query| query.label == FEATURE_OFF_CUGRAPH_QUERY_LABEL);
    let mut sessions: SessionPool = BTreeMap::new();
    for (fixture_index, workload) in profile.workloads.iter().enumerate() {
        if workload.queries.is_empty() {
            continue;
        }
        let mut kind_sessions = Vec::with_capacity(profile.session.shared_session_count);
        for _ in 0..profile.session.shared_session_count {
            // Sort-declared fixtures (the source-sorted DBLP layout) only
            // survive to native lowering as a single file group: DataFusion's
            // parallel byte-range grouping cannot reproduce a declared scan
            // ordering. The GPU-only cells never use CPU parallelism, and the
            // native device pipeline parallelizes reads internally.
            let session_config = datafusion::prelude::SessionConfig::new().with_target_partitions(1);
            let builder = SessionStateBuilder::new_with_default_features().with_config(session_config);
            let builder = backend.install_on(builder)?;
            let builder = if needs_cugraph_sql {
                builder.try_with_cugraph_sql(algeon_datafusion::cugraph_sql::CugraphSqlConfig::default())?
            } else {
                builder
            };
            let ctx = SessionContext::new_with_state(builder.build());
            match workload.kind {
                super::profile::FixtureKind::Tpch => {
                    workloads::tpch::register_tpch_tables(&ctx, &workload.path).await?;
                },
                super::profile::FixtureKind::Tpcds => {
                    workloads::tpcds::register_tpcds_tables(&ctx, &workload.path).await?;
                },
                super::profile::FixtureKind::Clickbench => {
                    let layout = workloads::clickbench::discover_clickbench_layout(&workload.path)?;
                    workloads::clickbench::register_clickbench_table(&ctx, &layout).await?;
                },
                super::profile::FixtureKind::Dblp => {
                    workloads::graph::register_dblp_tables(&ctx, &workload.path).await?;
                },
            }
            if needs_feature_off_probe {
                register_feature_off_cugraph_edge_schema(&ctx)?;
            }
            kind_sessions.push(ctx);
        }
        sessions.insert(fixture_index, kind_sessions);
    }
    Ok(sessions)
}

/// Registers only the static edge schema needed by the feature-off provider.
/// The provider validates this catalog entry, then builds a zero-child
/// `CugraphFeatureDisabledExec`; the in-memory batch cannot be scanned,
/// staged, or turned into a graph fixture on that execution path.
fn register_feature_off_cugraph_edge_schema(ctx: &SessionContext) -> ToolResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let provider = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table(FEATURE_OFF_CUGRAPH_EDGE_TABLE, Arc::new(provider))?;
    Ok(())
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_claim_requires_identical_ledger_brackets() {
        let before = CalibrationClaimSnapshot {
            sequence: 7,
            live_grants_bytes: 1_024,
            cache_evictable_resident_bytes: 128,
            cache_pinned_in_use_bytes: 64,
        };
        assert!(calibration_claim_stable_across_telemetry(before, Some(before)));
        assert!(!calibration_claim_stable_across_telemetry(before, None));
        assert!(!calibration_claim_stable_across_telemetry(
            before,
            Some(CalibrationClaimSnapshot { sequence: 8, ..before }),
        ));
        assert!(!calibration_claim_stable_across_telemetry(
            before,
            Some(CalibrationClaimSnapshot { live_grants_bytes: 2_048, ..before }),
        ));
    }

    #[test]
    fn measured_attempts_receive_the_shared_window_deadline() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(attempt_options(deadline).deadline, Some(deadline));
    }

    #[test]
    fn deadline_evidence_fails_closed_on_a_finalizer_overrun() {
        let within = run_deadline_evidence(10, 0, 10);
        assert!(within.reports_sealed_within_deadline);
        assert_eq!(within.overrun_ns, 0);

        let overrun = run_deadline_evidence(10, 0, 14);
        assert!(!overrun.reports_sealed_within_deadline);
        assert_eq!(overrun.overrun_ns, 4);

        let drained = run_deadline_evidence(10, 5, 14);
        assert_eq!(drained.measurement_window_ns, 10);
        assert_eq!(drained.completion_drain_grace_ns, 5);
        assert!(drained.reports_sealed_within_deadline);
    }

    #[test]
    fn cancellation_barrier_requires_both_the_action_and_a_cancelled_report() {
        let no_report = CancellationBarrierEvidence {
            attempt_sequence: 0,
            requested_after_ms: 0,
            cancellation_requested: true,
            cancelled_reports: 0,
        };
        assert!(!no_report.is_evidenced());
        let evidenced = CancellationBarrierEvidence { cancelled_reports: 1, ..no_report };
        assert!(evidenced.is_evidenced());
    }

    #[test]
    fn deadline_after_admission_scope_does_not_require_unrelated_concurrency() {
        let aggregate = MeasuredAggregate::reduce(&[], &[], &BTreeMap::new(), 0, 1);
        assert!(!scenario_clauses_pass(&aggregate.scenarios));
        assert!(selected_scenario_clauses_pass(
            &aggregate.scenarios,
            true,
            ScenarioRequirementsProfile {
                require_same_device_overlap: true,
                require_all_device_progress: true,
                require_heterogeneous_device_profiles: true,
                require_overload_outcome: false,
            },
        ));
        assert!(!selected_scenario_clauses_pass(
            &aggregate.scenarios,
            false,
            ScenarioRequirementsProfile {
                require_same_device_overlap: true,
                require_all_device_progress: true,
                require_heterogeneous_device_profiles: true,
                require_overload_outcome: false,
            },
        ));
        assert!(selected_scenario_clauses_pass(&aggregate.scenarios, false, ScenarioRequirementsProfile::default(),));
    }

    #[test]
    fn sealed_engine_composed_projection_is_copied_without_runner_byte_derivation() {
        let attempt_id = query_engine::observability::AttemptId::mint();
        let mut hasher = query_engine::harness::StableFnv1a64::new();
        hasher.update_field(COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1);
        hasher.update_field("10");
        hasher.update_field("2");
        for (id, scope, isolated, retained, stage) in [
            ("pagerank#0", "scope-pr", 90_u64, 20_u64, 100_u64),
            ("forceatlas2#0", "scope-fa2", 100_u64, 30_u64, 130_u64),
        ] {
            hasher.update_field(id);
            hasher.update_field(scope);
            hasher.update_field(&isolated.to_string());
            hasher.update_field(&retained.to_string());
            hasher.update_field(&stage.to_string());
        }
        hasher.update_field("scope-consumer");
        hasher.update_field("10");
        hasher.update_field("70");
        hasher.update_field("130");
        let fingerprint = format!("composed-schedule-fnv1a64:{:016x}", hasher.finish());
        let mut projection: AttemptComposedScheduleProjection = serde_json::from_value(serde_json::json!({
            "schema": COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1,
            "attempt_id": attempt_id.to_string(),
            "device_ordinal": 3,
            "grant_id": 7,
            "grant_bytes": 160,
            "attempt_wide_fixed_bytes": 10,
            "producer_occurrences": [
                {
                    "occurrence_id": "pagerank#0",
                    "execution_scope_fingerprint": "scope-pr",
                    "isolated_peak_bytes": 90,
                    "retained_output_bytes": 20,
                    "stage_peak_bytes": 100,
                },
                {
                    "occurrence_id": "forceatlas2#0",
                    "execution_scope_fingerprint": "scope-fa2",
                    "isolated_peak_bytes": 100,
                    "retained_output_bytes": 30,
                    "stage_peak_bytes": 130,
                },
            ],
            "consumer_scope_fingerprint": "scope-consumer",
            "consumer_additional_peak_bytes": 10,
            "consumer_stage_peak_bytes": 70,
            "complete_peak_bytes": 130,
            "fingerprint": fingerprint,
        }))
        .unwrap();
        assert!(projection.fingerprint_matches_contents());

        let artifact = project_composed_schedule_projection(&projection);
        assert_eq!(artifact.attempt_id, attempt_id.to_string());
        assert_eq!(artifact.selected_ordinal, 3);
        assert_eq!(artifact.grant_id, 7);
        assert_eq!(artifact.grant_bytes, 160);
        assert_eq!(artifact.occurrences[1].occurrence_id, "forceatlas2#0");
        assert_eq!(artifact.occurrences[1].stage_peak_bytes, 130);
        assert_eq!(artifact.complete_phase_peak_bytes, 130);
        assert!(artifact.engine_fingerprint_matches_contents);

        projection.fingerprint = "composed-schedule-fnv1a64:0000000000000000".to_owned();
        assert!(!project_composed_schedule_projection(&projection).engine_fingerprint_matches_contents);
    }

    #[test]
    fn an_all_warmup_run_cannot_qualify_as_a_measurement() {
        let aggregate = MeasuredAggregate::reduce(&[], &[], &BTreeMap::new(), 1, 1);
        assert!(!has_measurement_attempts(&aggregate));
    }

    #[test]
    fn a_step_schedule_requires_coverage_for_every_declared_phase() {
        let profile =
            RunProfile::parse_bytes(include_str!("../../../fixtures/stress/stress-s2-sweep-scan-sf10.json").as_bytes())
                .unwrap();
        let plan = ArrivalPlan::plan(
            &profile.scheduling,
            profile.selected_query_count(),
            profile.session.shared_session_count,
        )
        .unwrap();
        let aggregate = MeasuredAggregate::reduce(&[], &[], &BTreeMap::new(), 1, 1);
        assert!(!every_configured_step_has_measurement_attempts(&plan, &aggregate));
    }

    #[test]
    fn an_implicit_single_phase_uses_the_measurement_guard_not_step_coverage() {
        let profile = RunProfile::parse_bytes(
            include_str!("../../../fixtures/stress/stress-s3-heterogeneous-sf1.json").as_bytes(),
        )
        .unwrap();
        let plan = ArrivalPlan::plan(
            &profile.scheduling,
            profile.selected_query_count(),
            profile.session.shared_session_count,
        )
        .unwrap();
        let aggregate = MeasuredAggregate::reduce(&[], &[], &BTreeMap::new(), 0, 1);
        assert!(every_configured_step_has_measurement_attempts(&plan, &aggregate));
        assert!(!has_measurement_attempts(&aggregate));
    }

    #[test]
    fn terminal_snapshot_replaces_the_bounded_tail_in_lockstep() {
        let mut samples = (0..MAX_SNAPSHOT_SAMPLES).collect::<Vec<_>>();
        let mut audits = (0..MAX_SNAPSHOT_SAMPLES).collect::<Vec<_>>();
        let mut truncated = false;

        assert!(!reserve_snapshot_slot(&mut samples, &mut audits, &mut truncated, false));
        assert!(truncated);
        assert_eq!(samples.len(), MAX_SNAPSHOT_SAMPLES);
        assert_eq!(audits.len(), MAX_SNAPSHOT_SAMPLES);

        assert!(reserve_snapshot_slot(&mut samples, &mut audits, &mut truncated, true));
        assert_eq!(samples.len(), MAX_SNAPSHOT_SAMPLES - 1);
        assert_eq!(audits.len(), MAX_SNAPSHOT_SAMPLES - 1);
        samples.push(MAX_SNAPSHOT_SAMPLES);
        audits.push(MAX_SNAPSHOT_SAMPLES);
        assert_eq!(samples.last(), audits.last());
    }
}
