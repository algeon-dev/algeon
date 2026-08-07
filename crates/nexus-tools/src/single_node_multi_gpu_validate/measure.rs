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
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use arrow::{
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{datasource::MemTable, execution::SessionStateBuilder, prelude::SessionContext};
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    error::DataFusionErrorExt,
    native_exec::NexusNativeExecutionConfig,
};
use futures::StreamExt as _;
use nexus_query_engine::{
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

/// How often the sampler polls the backend snapshot accessors at minimum.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
/// Bound on retained raw snapshot samples. Each carries the full per-device
/// report vectors, so this caps artifact size and peak memory; the per-device
/// peaks the scenario clauses use keep folding past the cap. A final sample
/// always displaces the retained tail when this bound is reached, so terminal
/// P4 counters are never audited against a pre-drain snapshot.
const MAX_SNAPSHOT_SAMPLES: usize = 1024;

/// The sampler cadence for a configured run window. Windows short enough to
/// fit the retention bound at the base interval keep 250 ms; longer windows
/// (S4 plateaus) trade cadence for whole-window coverage instead of
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
const FEATURE_OFF_CUGRAPH_EDGE_TABLE: &str = "__nexus_feature_off_cugraph_edges";

fn feature_off_cugraph_sql() -> String {
    format!("SELECT * FROM cugraph_pagerank('{FEATURE_OFF_CUGRAPH_EDGE_TABLE}')")
}

/// The runner's two bounded phases. Submission stops at the configured
/// warmup-plus-measurement boundary; already-admitted work may use only the
/// profile-owned completion grace to reach a native safe boundary and seal.
/// This keeps sustained-service measurement duration distinct from bounded
/// cooperative finalization without weakening legacy P6's zero-grace shape.
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

/// S1's graph request is intentionally absent from normal relational
/// round-robin selection.  The runner uses this profile-owned schedule to
/// issue the declared warmup and measurement probes beside live relational
/// attempts without changing the relational cohort's query distribution.
#[derive(Clone, Copy)]
struct FeatureOffCugraphProbeSchedule {
    query_index: usize,
    warmup_probe_attempts: usize,
    measurement_probe_attempts: usize,
}

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
    /// S6 repeated quarantine cycles over a rotation of configured ordinals.
    pub(super) quarantine_cycles: Option<usize>,
    /// S6 close initiated at the midpoint of the measurement window.
    pub(super) close_under_load: bool,
    /// W8's explicit reference OTLP exporter cell. None means the run records
    /// that no exporter was requested rather than claiming export success.
    pub(super) otlp_export: Option<OtlpExportConfig>,
    /// I8's intentionally serialized public completion receiver. The sink is
    /// a probe-only consumer delay, not an execution or admission policy.
    pub(super) slow_report_sink: Option<Duration>,
}

/// Runtime-only OTLP exporter input. Its full endpoint is deliberately never
/// copied into an artifact; only `endpoint_host_port` is retained by W8.
#[derive(Clone)]
pub(super) struct OtlpExportConfig {
    #[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
    endpoint: String,
    endpoint_host_port: String,
    interval: Duration,
}

impl OtlpExportConfig {
    pub(super) fn new(endpoint: &str, interval: Duration) -> ToolResult<Self> {
        let parsed =
            url::Url::parse(endpoint).map_err(|_| "--otlp-metrics-endpoint must be an absolute http(s) URL")?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err("--otlp-metrics-endpoint must use http or https".into());
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err("--otlp-metrics-endpoint must not contain credentials".into());
        }
        let host = parsed.host_str().ok_or("--otlp-metrics-endpoint must include a host")?;
        let port = parsed.port_or_known_default().ok_or("--otlp-metrics-endpoint must include a port")?;
        Ok(Self {
            #[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
            endpoint: endpoint.to_string(),
            endpoint_host_port: format!("{host}:{port}"),
            interval,
        })
    }
}

/// Runs the reference `nexus-examples` OTLP exporter beside live traffic.
/// It owns no execution state and obtains every value through the public
/// `backend_metrics_snapshot()` accessor supplied to the example adapter.
#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
struct OtlpExportWorker {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: std::thread::JoinHandle<OtlpExportEvidence>,
    endpoint_host_port: String,
    interval_ms: u64,
}

#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
impl OtlpExportWorker {
    fn start(backend: &NexusGpuBackend, config: &OtlpExportConfig) -> Self {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let endpoint = config.endpoint.clone();
        let endpoint_host_port = config.endpoint_host_port.clone();
        let fallback_endpoint_host_port = endpoint_host_port.clone();
        let interval = config.interval;
        let interval_ms = interval.as_millis().min(u128::from(u64::MAX)) as u64;
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime,
                Err(_) => return OtlpExportEvidence::requested(endpoint_host_port, interval_ms, 0, 0, 1),
            };
            let meter = opentelemetry::global::meter("datafusion-nexus-stress");
            let snapshot_backend = backend.clone();
            let exporter = nexus_examples::otlp_metrics::BackendMetricsOtlpExporter::new(&meter, move || {
                snapshot_backend.backend_metrics_snapshot()
            });
            let resource = opentelemetry_proto::tonic::resource::v1::Resource::default();
            let start_time_unix_nano = unix_epoch_ns();
            let mut export_attempts = 0_u64;
            let mut successful_exports = 0_u64;
            let mut failed_exports = 0_u64;

            loop {
                export_attempts += 1;
                match runtime.block_on(exporter.export_histograms(
                    &endpoint,
                    &resource,
                    start_time_unix_nano,
                    unix_epoch_ns(),
                )) {
                    Ok(()) => successful_exports += 1,
                    Err(_) => failed_exports += 1,
                }
                let (lock, condition) = &*signal;
                let should_stop = match lock.lock() {
                    Ok(stopped) => match condition.wait_timeout(stopped, interval) {
                        Ok((stopped, _)) => *stopped,
                        Err(_) => true,
                    },
                    Err(_) => true,
                };
                if should_stop {
                    break;
                }
            }
            OtlpExportEvidence::requested(
                endpoint_host_port,
                interval_ms,
                export_attempts,
                successful_exports,
                failed_exports,
            )
        });
        Self { stop, handle, endpoint_host_port: fallback_endpoint_host_port, interval_ms }
    }

    fn stop(self) -> OtlpExportEvidence {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        let endpoint_host_port = self.endpoint_host_port;
        let interval_ms = self.interval_ms;
        self.handle.join().unwrap_or_else(|_| OtlpExportEvidence::requested(endpoint_host_port, interval_ms, 0, 0, 1))
    }
}

#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
fn unix_epoch_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// A requested OTLP cell cannot run unless the explicit optional feature
/// linked the reference exporter. `run_measured_run` rejects that CLI shape
/// before reaching this fallback; retaining it here keeps the failure honest
/// if a future caller bypasses the CLI parser.
#[cfg(not(feature = "single-node-multi-gpu-validate-run-otlp"))]
struct OtlpExportWorker {
    evidence: OtlpExportEvidence,
}

#[cfg(not(feature = "single-node-multi-gpu-validate-run-otlp"))]
impl OtlpExportWorker {
    fn start(_backend: &NexusGpuBackend, config: &OtlpExportConfig) -> Self {
        Self {
            evidence: OtlpExportEvidence::requested(
                config.endpoint_host_port.clone(),
                config.interval.as_millis().min(u128::from(u64::MAX)) as u64,
                0,
                0,
                1,
            ),
        }
    }

    fn stop(self) -> OtlpExportEvidence {
        self.evidence
    }
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
        profile.feature_off_cugraph_probe.clone().zip(feature_off_cugraph_probe).map(|(profile, query_index)| {
            FeatureOffCugraphProbeSchedule {
                query_index,
                warmup_probe_attempts: profile.warmup_probe_attempts,
                measurement_probe_attempts: profile.measurement_probe_attempts,
            }
        });
    if plan.truncated_by_attempt_cap {
        // W6: the artifact records this too; the operator warning keeps a
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
    // S7's same-key proof needs a cold cache on exactly one allocation
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
            profile.backend.queue.max_overtakes_per_waiter,
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
    let feature_off_cugraph = profile.feature_off_cugraph_probe.clone().map(|probe| {
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
        timeseries::vram_calibration_evidence(&sampler.device_series, &configured_ordinals, policy)
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
    let identity = backend.identity();
    Ok(MeasuredRunArtifact {
        qualifying,
        backend: BackendEvidence {
            resource_core: identity.resource_core,
            observation_core: identity.observation_core,
            resource_generation: identity.resource_generation,
            shared_session_count: profile.session.shared_session_count,
            queue_max_queued_attempts: profile.backend.queue.max_queued_attempts,
            queue_max_queue_wait_ms: profile.backend.queue.max_queue_wait_ms,
            queue_max_overtakes_per_waiter: profile.backend.queue.max_overtakes_per_waiter,
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

/// An explicit multi-phase W3 schedule is an envelope only when every
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

/// Runs S7's explicit cold-cache coalescing cohorts.  The throughput run may
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

#[allow(clippy::too_many_arguments)]
async fn run_cache_same_key_cohort(
    backend: &NexusGpuBackend,
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
            nexus_query_engine::runtime::cache_observation_fingerprint(&format!("{cohort_id}-attempt-{sequence}"))
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

fn cache_reports_snapshot(backend: &NexusGpuBackend) -> ToolResult<Value> {
    Ok(serde_json::to_value(backend.device_cache_reports()?)?)
}

fn cache_quarantine_observation(backend: &NexusGpuBackend, ordinal: usize) -> ToolResult<Value> {
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

fn cache_source_event_snapshots(backend: &NexusGpuBackend) -> ToolResult<Value> {
    Ok(serde_json::to_value(backend.device_cache_source_event_snapshots()?)?)
}

/// Adds S1's runner-owned graph request after the relational set.  The caller
/// retains its index for the dedicated probe schedule, while normal arrivals
/// are planned against the pre-append relational count.
fn append_feature_off_cugraph_probe(
    profile: &RunProfile,
    queries: &mut Vec<SelectedQuery>,
) -> ToolResult<Option<usize>> {
    if profile.feature_off_cugraph_probe.is_none() {
        return Ok(None);
    }
    let mut query = queries
        .iter()
        .find(|query| !query.graph)
        .cloned()
        .ok_or("feature_off_cugraph_probe requires one selected relational query")?;
    query.label = FEATURE_OFF_CUGRAPH_QUERY_LABEL.to_string();
    query.sql = feature_off_cugraph_sql();
    query.graph = true;
    query.placement_required = false;
    queries.push(query);
    Ok(Some(queries.len() - 1))
}

/// A normal graph qualification profile must be built with the explicit
/// cugraph-enabled runner feature. The base runner admits exactly its profile
/// owned feature-off probe; accepting arbitrary graph labels there would turn
/// a missing build feature into generic failed workload rows.
pub(super) fn validate_cugraph_runner_mode(
    profile: &RunProfile,
    #[cfg(feature = "single-node-multi-gpu-validate-run-cugraph")] _queries: &[SelectedQuery],
    #[cfg(not(feature = "single-node-multi-gpu-validate-run-cugraph"))] queries: &[SelectedQuery],
) -> ToolResult<()> {
    #[cfg(feature = "single-node-multi-gpu-validate-run-cugraph")]
    if profile.feature_off_cugraph_probe.is_some() {
        return Err(
            "feature_off_cugraph_probe requires the base single-node-multi-gpu-validate-run build without single-node-multi-gpu-validate-run-cugraph"
                .into(),
        );
    }
    #[cfg(not(feature = "single-node-multi-gpu-validate-run-cugraph"))]
    if profile.feature_off_cugraph_probe.is_none() && queries.iter().any(|query| query.graph) {
        return Err(
            "profile selects cuGraph workloads, but the base single-node-multi-gpu-validate-run build has cugraph disabled; rebuild with --features single-node-multi-gpu-validate-run-cugraph or use feature_off_cugraph_probe"
                .into(),
        );
    }
    Ok(())
}

/// Runs the serial, public-SQL P6.6 domain-cap probe. The profile preflight
/// pins the two-attempt shape (`q01` followed by `edge_scan`) and the single
/// static partition. This driver deliberately reuses ordinary submission and
/// drain logic: no runner-created error substitutes for allocator evidence.
async fn drive_domain_exhaustion_probe(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
) -> ToolResult<DrivenRun> {
    if plan.arrivals.len() != 2 {
        return Err("--domain-exhaustion-probe requires exactly two planned attempts".into());
    }
    drive(backend, sessions, queries, plan, completion_drain_grace, None, None).await
}

/// Drives P6.6's one-device physical allocator-pressure probe.
///
/// The cuDF testing lease owns a separate `cuda_on` sidecar and only exposes
/// observed CUDA memory facts. It is established after the service has
/// installed its own manager, before the public q01 attempt exists, and is
/// consumed (synchronized and verified) before public edge_scan recovery. The
/// two SQL attempts otherwise use the ordinary public attempt API; no runner
/// error, allocator callback, or backend mode substitutes for their reports.
async fn drive_physical_allocator_pressure(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    target_ordinal: usize,
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" || plan.arrivals.len() != 2 {
        return Err("--physical-allocator-pressure requires exactly two closed-loop planned attempts".into());
    }
    let first = plan.arrivals[0];
    let recovery = plan.arrivals[1];
    if queries.get(first.query_index).is_none_or(|query| query.label != "q01")
        || queries.get(recovery.query_index).is_none_or(|query| query.label != "edge_scan")
    {
        return Err("--physical-allocator-pressure requires public q01 followed by edge_scan".into());
    }

    let timing = RunTiming::new(plan, completion_drain_grace);
    let warmup_deadline = timing.warmup_deadline;
    let run_deadline = timing.completion_deadline;
    let device = cudf::DeviceId::new(target_ordinal)
        .map_err(|error| format!("--physical-allocator-pressure cannot resolve device {target_ordinal}: {error}"))?;
    let lease = cudf::testing::memory::PhysicalMemoryPressureLease::establish(device).map_err(|error| {
        format!("--physical-allocator-pressure could not establish real external pressure: {error}")
    })?;
    let pressure = project_physical_pressure_facts(lease.facts());

    let mut tasks = Vec::with_capacity(1);
    let mut observations = Vec::with_capacity(2);
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    // Dedicated probes never arm S5 storms; the knob set stays disabled.
    let mut storm_cancellers = Vec::new();
    match spawn_attempt_task(
        backend,
        sessions,
        queries,
        first,
        true,
        warmup_deadline,
        run_deadline,
        None,
        &mut cancellation_barrier,
        StormKnobs::default(),
        &mut storm_cancellers,
    ) {
        Ok(task) => tasks.push(task),
        Err(()) => {
            return match lease.release_and_verify() {
                Ok(_) => Err("--physical-allocator-pressure could not begin the public q01 attempt".into()),
                Err(error) => Err(format!(
                    "--physical-allocator-pressure could not begin public q01 and could not verify external pressure release: {error}"
                )
                .into()),
            };
        },
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;

    // This consuming call synchronizes and drops every sidecar buffer before
    // the recovery attempt begins; a failed verification is a hard runner
    // error rather than a manufactured recovery claim.
    let release = lease.release_and_verify().map_err(|error| {
        format!("--physical-allocator-pressure could not verify external pressure release before recovery: {error}")
    })?;
    let release = project_physical_pressure_facts(release);

    match spawn_attempt_task(
        backend,
        sessions,
        queries,
        recovery,
        true,
        warmup_deadline,
        run_deadline,
        None,
        &mut cancellation_barrier,
        StormKnobs::default(),
        &mut storm_cancellers,
    ) {
        Ok(task) => tasks.push(task),
        Err(()) => begin_failures += 1,
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: Some(PhysicalAllocatorPressureBarrier {
            pressure,
            release,
            pressure_established_before_q01: true,
            pressure_released_before_recovery: true,
        }),
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: None,
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

fn project_physical_pressure_facts(
    facts: cudf::testing::memory::PhysicalMemoryPressureFacts,
) -> PhysicalPressureLeaseFacts {
    PhysicalPressureLeaseFacts {
        device_ordinal: facts.device.ordinal(),
        method: facts.method,
        free_before_bytes: facts.free_before_bytes,
        free_under_pressure_bytes: facts.free_under_pressure_bytes,
        bytes_held: facts.bytes_held,
        reserve_target_bytes: facts.reserve_target_bytes,
        free_after_release_bytes: facts.free_after_release_bytes,
        recovered_bytes: facts.recovered_bytes,
    }
}

/// Applies only the ordinary scenario clauses that the requested physical run
/// is able to prove. Dedicated P6.6 probes have their own strict predicates:
/// the deadline barrier is serialized, while overload uses sealed queue facts
/// rather than P6.5 overlap clauses. The complete aggregate remains in the
/// artifact in either case.
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

/// Projects I8's explicit handoff-pressure contract. Every step must offer
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

pub(super) fn build_backend(profile: &RunProfile) -> ToolResult<NexusGpuBackend> {
    if profile.backend.device_profiles.is_empty() {
        return Err("a measured run requires at least one backend.device_profiles entry".into());
    }
    let native_execution = profile.backend.native_execution;
    let queue = NexusGpuQueuePolicy::default()
        .with_max_queued_attempts(profile.backend.queue.max_queued_attempts)
        .with_max_queue_wait(Duration::from_millis(profile.backend.queue.max_queue_wait_ms))
        .with_max_overtakes_per_waiter(profile.backend.queue.max_overtakes_per_waiter);
    let backend = NexusGpuBackend::builder()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(
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
        .with_no_datafusion_cpu_requirement()
        .queue_policy(queue)
        .device_profiles(profile.backend.device_profiles.iter().map(|device| {
            NexusGpuDeviceProfile::new(device.ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: device.memory_limit_bytes })
                .with_backend_reserve_bytes(device.backend_reserve_bytes)
                .with_cache_cap_bytes(device.cache_cap_bytes)
                .with_attempt_floor_bytes(Some(device.attempt_floor_bytes))
                .with_source_work_per_attempt(device.source_work_per_attempt)
                .with_max_active_attempts(device.max_active_attempts)
                .with_telemetry(device.install_telemetry)
                .with_installation_bounds(
                    device.installation.stream_pool_size,
                    device.installation.worker_count,
                    device.installation.lane_count,
                )
        }))
        .device_memory_ownership(profile.backend.device_profiles.iter().map(|device| {
            (
                device.ordinal,
                match device.ownership {
                    DeviceOwnership::WholeDeviceExclusive => GpuMemoryOwnership::WholeDeviceExclusive,
                    DeviceOwnership::StaticPartition { allocation_limit_bytes } => {
                        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes }
                    },
                },
            )
        }))
        // The engine's device-config rejection carries the field and reason;
        // flattening the source chain keeps that actionable instead of leaving
        // it behind an omitted source.
        .build()
        .map_err(|error| format!("backend build rejected the profile: {}", error_chain(&error)))?;
    Ok(backend)
}

/// The shared session pool, keyed by profile workload position. TPC-H and
/// TPC-DS share table names (`customer`), and W9 permits several DBLP graph
/// fixtures with those same registration names, so one context pool per
/// fixture workload is the only correct routing boundary.
pub(super) type SessionPool = BTreeMap<usize, Vec<SessionContext>>;

/// `shared_session_count` contexts per selected fixture workload. Sessions are
/// shared only across attempts bound to that same fixture entry.
pub(super) async fn install_sessions(
    backend: &NexusGpuBackend,
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
            let builder = if needs_cugraph_sql {
                backend
                    .install_on_with_cugraph_sql(builder, datafusion_nexus::cugraph_sql::CugraphSqlConfig::default())?
            } else {
                backend.install_on(builder)?
            };
            let ctx = SessionContext::new_with_state(builder.build());
            match workload.kind {
                super::profile::FixtureKind::Tpch => {
                    nexus_workloads::tpch::register_tpch_tables(&ctx, &workload.path).await?;
                },
                super::profile::FixtureKind::Tpcds => {
                    nexus_workloads::tpcds::register_tpcds_tables(&ctx, &workload.path).await?;
                },
                super::profile::FixtureKind::Clickbench => {
                    let layout = nexus_workloads::clickbench::discover_clickbench_layout(&workload.path)?;
                    nexus_workloads::clickbench::register_clickbench_table(&ctx, &layout).await?;
                },
                super::profile::FixtureKind::Dblp => {
                    nexus_workloads::graph::register_dblp_tables(&ctx, &workload.path).await?;
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

struct DrivenRun {
    observations: Vec<AttemptObservation>,
    dropped_overrun_arrivals: u64,
    begin_failures: u64,
    join_failures: u64,
    deadline: RunDeadlineEvidence,
    cancellation_barrier: Option<CancellationBarrierEvidence>,
    quarantine_barrier: Option<QuarantineBarrier>,
    deadline_barrier: Option<DeadlineBarrierEvidence>,
    physical_allocator_pressure: Option<PhysicalAllocatorPressureBarrier>,
    observed_peak_queued_attempts: usize,
    /// Service-owned queue high-water marks captured at each sequential step
    /// boundary.  They are cumulative from backend construction, so each one
    /// proves the configured bound through that phase without resetting or
    /// synthesizing a controller counter in the runner.
    queue_high_watermarks: Vec<(usize, u64)>,
    /// S5 cancellation-storm actions the driver armed and confirmed fired.
    storm_cancellations_requested: u64,
    /// S6 repeated-quarantine barriers, one per completed cycle.
    quarantine_cycle_barriers: Vec<QuarantineBarrier>,
    /// S6 close-under-saturation facts captured at the close boundary.
    close_under_load: Option<CloseUnderLoadBarrier>,
    /// Direct, typed feature-off UDTF source errors paired by sequence with
    /// their sealed reports. Ordinary runs leave this empty.
    feature_off_cugraph_source_errors: Vec<(usize, FeatureOffCugraphSourceError)>,
}

/// Runner-local facts captured at the S6 close boundary. The evidence
/// reducer combines these with sealed attempt outcomes; nothing here infers
/// queue membership from runner ordering.
struct CloseUnderLoadBarrier {
    close_requested_offset_ns: u64,
    close_initiated: bool,
    admitted_at_close: usize,
    queued_at_close: usize,
    in_flight_at_close: usize,
    in_flight_sequences: BTreeSet<usize>,
    submitted_attempts: u64,
}

/// The S5 per-attempt storm knobs, resolved from the arrival plan once.
#[derive(Clone, Copy, Default)]
struct StormKnobs {
    attempt_deadline_ms: Option<u64>,
    cancel_every_n: Option<usize>,
    cancel_after_ms: Option<u64>,
}

impl StormKnobs {
    fn of(plan: &ArrivalPlan) -> Self {
        Self {
            attempt_deadline_ms: plan.attempt_deadline_ms,
            cancel_every_n: plan.cancel_every_n,
            cancel_after_ms: plan.cancel_after_ms,
        }
    }
}

/// Runner-local facts sampled around a synchronous controlled quarantine.
/// `QuarantineFallbackEvidence` turns these plus sealed reports into the
/// versioned artifact projection after all tasks have drained.
struct QuarantineBarrier {
    target_ordinal: usize,
    pre_ack_witness_offset_ns: u64,
    pre_ack_ledger_sequence: u64,
    pre_ack_live_grants: usize,
    pre_ack_active_query_jobs: usize,
    acknowledgement_offset_ns: u64,
    device_health_generation: u64,
    newly_quarantined: bool,
    target_ledger_closed: bool,
    post_ack_sequences: BTreeSet<usize>,
    pre_existing_target_work_released: bool,
    pre_ack_cache: Option<serde_json::Value>,
    post_ack_cache: Option<serde_json::Value>,
    final_cache: Option<serde_json::Value>,
}

/// Runner-local lease facts around the one physical-pressure probe. The
/// reducer combines these bounded CUDA snapshots with sealed attempt reports;
/// it never treats the sidecar itself as a synthetic allocator error.
struct PhysicalAllocatorPressureBarrier {
    pressure: PhysicalPressureLeaseFacts,
    release: PhysicalPressureLeaseFacts,
    pressure_established_before_q01: bool,
    pressure_released_before_recovery: bool,
}

struct PendingCancellationBarrier {
    attempt_sequence: usize,
    requested_after: Duration,
    requested: Arc<AtomicBool>,
    task: Option<tokio::task::JoinHandle<()>>,
}

/// Submits every planned arrival at its offset and collects one bounded report
/// per attempt.
async fn drive(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    cancel_first_after: Option<Duration>,
    feature_off_cugraph_probe: Option<FeatureOffCugraphProbeSchedule>,
) -> ToolResult<DrivenRun> {
    drive_with_queue_observation(
        backend,
        sessions,
        queries,
        plan,
        completion_drain_grace,
        cancel_first_after,
        None,
        feature_off_cugraph_probe,
    )
    .await
}

/// Drives the physical bounded-overload cohort through the same public attempt
/// API as an ordinary run, while retaining only the public queue high-water
/// mark. It does not manufacture unequal admission requirements.
async fn drive_overload_fairness(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    queue_limit: usize,
) -> ToolResult<DrivenRun> {
    if queue_limit == 0 {
        return Err("--overload-fairness requires a non-zero configured queue limit".into());
    }
    drive_with_queue_observation(
        backend,
        sessions,
        queries,
        plan,
        completion_drain_grace,
        None,
        Some(queue_limit),
        None,
    )
    .await
}

/// Drives an unbounded-duration closed-loop profile without first allocating a
/// speculative arrival vector. Each round is constructed only after the
/// preceding one drains; the wall-clock step deadline, not an arbitrary
/// planning cap, ends the step. Fixed P6 cohorts continue through the stored
/// plan so their exact attempt shape remains frozen.
async fn drive_duration_paced_concurrency(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    timing: RunTiming,
    cancel_first_after: Option<Duration>,
    feature_off_cugraph_probe: Option<FeatureOffCugraphProbeSchedule>,
) -> ToolResult<DrivenRun> {
    debug_assert!(plan.is_duration_paced_concurrency());
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut feature_off_cugraph_source_errors = Vec::new();
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    let storm = StormKnobs::of(plan);
    let mut sequence = 0_usize;
    let mut warmup_feature_off_probes_submitted = false;
    let mut measurement_feature_off_probes_submitted = false;
    let mut submissions_open = true;
    let mut queue_high_watermarks = Vec::with_capacity(plan.steps.len());

    for step in &plan.steps {
        let step_deadline = timing.started + Duration::from_nanos(step.start_ns.saturating_add(step.measurement_ns));
        let mut round = 0_usize;
        while submissions_open && Instant::now() < step_deadline && Instant::now() < timing.submission_deadline {
            let mut submitted_this_round = 0_usize;
            for _ in 0..step.concurrency {
                if Instant::now() >= step_deadline || Instant::now() >= timing.submission_deadline {
                    break;
                }
                let arrival = duration_paced_arrival(sequence, round, step.step_index, queries, sessions);
                sequence = sequence.saturating_add(1);
                match spawn_attempt_task(
                    backend,
                    sessions,
                    queries,
                    arrival,
                    true,
                    timing.warmup_deadline,
                    timing.completion_deadline,
                    cancel_first_after,
                    &mut cancellation_barrier,
                    storm,
                    &mut storm_cancellers,
                ) {
                    Ok(task) => {
                        tasks.push(task);
                        submitted_this_round += 1;
                    },
                    Err(()) => {
                        begin_failures += 1;
                        submissions_open = false;
                        break;
                    },
                }
            }
            if submitted_this_round == 0 {
                break;
            }
            // The duration-paced relational cohort is the service-rate
            // denominator.  Keep the feature-off control-plane sample exact,
            // but inject it beside a live relational round rather than using
            // a finite round cap to manufacture its phase boundary.
            if let Some(probe) = feature_off_cugraph_probe {
                let probe_count = if !warmup_feature_off_probes_submitted && Instant::now() < timing.warmup_deadline {
                    Some((probe.warmup_probe_attempts, &mut warmup_feature_off_probes_submitted))
                } else if !measurement_feature_off_probes_submitted && Instant::now() >= timing.warmup_deadline {
                    Some((probe.measurement_probe_attempts, &mut measurement_feature_off_probes_submitted))
                } else {
                    None
                };
                if let Some((count, submitted)) = probe_count {
                    if spawn_feature_off_cugraph_probes(
                        backend,
                        sessions,
                        queries,
                        probe,
                        count,
                        &mut sequence,
                        round,
                        step.step_index,
                        timing.warmup_deadline,
                        timing.completion_deadline,
                        &mut cancellation_barrier,
                        storm,
                        &mut storm_cancellers,
                        &mut tasks,
                    )
                    .is_err()
                    {
                        begin_failures += 1;
                        submissions_open = false;
                        break;
                    }
                    *submitted = true;
                }
            }
            drain_with_feature_off_cugraph_capture(
                &mut tasks,
                &mut observations,
                &mut join_failures,
                &mut feature_off_cugraph_source_errors,
            )
            .await;
            round = round.saturating_add(1);
        }
        let queue_high_watermark = backend.admission_snapshot()?.queue_high_watermark;
        queue_high_watermarks.push((step.step_index, u64::try_from(queue_high_watermark).unwrap_or(u64::MAX)));
    }
    drain_with_feature_off_cugraph_capture(
        &mut tasks,
        &mut observations,
        &mut join_failures,
        &mut feature_off_cugraph_source_errors,
    )
    .await;
    let cancellation_barrier = finish_cancellation_barrier(cancellation_barrier, &observations).await;
    let storm_cancellations_requested = finish_storm_cancellers(storm_cancellers).await;
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks,
        storm_cancellations_requested,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: None,
        feature_off_cugraph_source_errors,
    })
}

fn duration_paced_arrival(
    sequence: usize,
    round: usize,
    step_index: usize,
    queries: &[SelectedQuery],
    sessions: &SessionPool,
) -> super::schedule::ScheduledArrival {
    let query_index = sequence % queries.len();
    let session_index = sequence
        % sessions
            .get(&queries[query_index].fixture_index)
            .expect("install_sessions registers every selected fixture workload")
            .len();
    super::schedule::ScheduledArrival {
        sequence,
        round,
        step_index,
        offset_ns: 0,
        phase: SchedulePhase::Warmup,
        rate_overrun: RateOverrunPolicy::Delay,
        query_index,
        session_index,
    }
}

async fn drive_with_queue_observation(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    cancel_first_after: Option<Duration>,
    queue_limit: Option<usize>,
    feature_off_cugraph_probe: Option<FeatureOffCugraphProbeSchedule>,
) -> ToolResult<DrivenRun> {
    let timing = RunTiming::new(plan, completion_drain_grace);
    if plan.is_duration_paced_concurrency() {
        if queue_limit.is_some() {
            return Err("--overload-fairness requires an explicit fixed closed-loop cohort".into());
        }
        return drive_duration_paced_concurrency(
            backend,
            sessions,
            queries,
            plan,
            timing,
            cancel_first_after,
            feature_off_cugraph_probe,
        )
        .await;
    }
    let started = timing.started;
    let closed_loop = plan.arrival_mode == "concurrency";
    let warmup_deadline = timing.warmup_deadline;
    let submission_deadline = timing.submission_deadline;
    let run_deadline = timing.completion_deadline;
    let storm = StormKnobs::of(plan);

    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut feature_off_cugraph_source_errors = Vec::new();
    let mut dropped_overrun_arrivals = 0_u64;
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut current_step = 0_usize;
    let mut current_round = 0;
    let mut step_deadline = step_submission_deadline(started, plan, 0);
    let mut cancellation_barrier = None;
    let mut storm_cancellers: Vec<PendingCancellationBarrier> = Vec::new();
    let mut next_feature_off_sequence = plan.arrivals.len();
    let mut measurement_feature_off_probes_submitted = false;
    let mut submissions_open = true;
    let mut queue_high_watermarks = Vec::with_capacity(plan.steps.len());

    if let Some(probe) = feature_off_cugraph_probe
        && spawn_feature_off_cugraph_probes(
            backend,
            sessions,
            queries,
            probe,
            probe.warmup_probe_attempts,
            &mut next_feature_off_sequence,
            0,
            0,
            warmup_deadline,
            run_deadline,
            &mut cancellation_barrier,
            storm,
            &mut storm_cancellers,
            &mut tasks,
        )
        .is_err()
    {
        begin_failures += 1;
        submissions_open = false;
    }

    for arrival in &plan.arrivals {
        if !submissions_open {
            break;
        }
        // Steps are sequential phases inside one run: a step change drains
        // the previous step's in-flight work before the next step's cohort
        // starts, so per-step aggregates are cleanly attributable.
        if arrival.step_index != current_step {
            drain_with_feature_off_cugraph_capture(
                &mut tasks,
                &mut observations,
                &mut join_failures,
                &mut feature_off_cugraph_source_errors,
            )
            .await;
            let queue_high_watermark = backend.admission_snapshot()?.queue_high_watermark;
            queue_high_watermarks.push((current_step, u64::try_from(queue_high_watermark).unwrap_or(u64::MAX)));
            current_step = arrival.step_index;
            current_round = arrival.round;
            step_deadline = step_submission_deadline(started, plan, current_step);
        }
        // The step window bounds submission; arrivals planned past their
        // step's deadline never run. The run window is the global backstop.
        if Instant::now() >= step_deadline {
            continue;
        }
        if Instant::now() >= submission_deadline {
            break;
        }
        if closed_loop {
            // The step window, not the plan length, ends a closed-loop step:
            // rounds are planned generously and pacing is observed. Drain the
            // previous round so `concurrency` bounds in-flight attempts
            // instead of becoming an offered arrival rate.
            if arrival.round != current_round {
                drain_with_feature_off_cugraph_capture(
                    &mut tasks,
                    &mut observations,
                    &mut join_failures,
                    &mut feature_off_cugraph_source_errors,
                )
                .await;
                current_round = arrival.round;
                if feature_off_cugraph_probe.is_some()
                    && !measurement_feature_off_probes_submitted
                    && Instant::now() < warmup_deadline
                {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(warmup_deadline)).await;
                }
            }
        } else {
            let planned = started + Duration::from_nanos(arrival.offset_ns);
            match Instant::now().checked_duration_since(planned) {
                // Late by more than the tolerance: rate mode may abandon it
                // rather than convert the offered rate into a closed loop.
                // The policy is read from the arrival: rate steps may mix
                // `delay` and `drop` phases inside one run.
                Some(late) if late > RATE_OVERRUN_TOLERANCE => {
                    if arrival.rate_overrun == RateOverrunPolicy::Drop {
                        dropped_overrun_arrivals += 1;
                        continue;
                    }
                },
                Some(_) => {},
                None => tokio::time::sleep_until(tokio::time::Instant::from_std(planned)).await,
            }
        }

        // A begin failure ends submission but must not discard the attempts
        // that already ran; the artifact records how many were refused.
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            closed_loop,
            warmup_deadline,
            run_deadline,
            cancel_first_after,
            &mut cancellation_barrier,
            storm,
            &mut storm_cancellers,
        ) {
            Ok(task) => tasks.push(task),
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
        if let Some(probe) = feature_off_cugraph_probe
            && !measurement_feature_off_probes_submitted
            && Instant::now() >= warmup_deadline
        {
            if spawn_feature_off_cugraph_probes(
                backend,
                sessions,
                queries,
                probe,
                probe.measurement_probe_attempts,
                &mut next_feature_off_sequence,
                arrival.round,
                arrival.step_index,
                warmup_deadline,
                run_deadline,
                &mut cancellation_barrier,
                storm,
                &mut storm_cancellers,
                &mut tasks,
            )
            .is_err()
            {
                begin_failures += 1;
                break;
            }
            measurement_feature_off_probes_submitted = true;
        }
    }

    let observed_peak_queued_attempts = match queue_limit {
        Some(limit) => observe_queue_peak(backend, limit, submission_deadline).await?,
        None => 0,
    };
    drain_with_feature_off_cugraph_capture(
        &mut tasks,
        &mut observations,
        &mut join_failures,
        &mut feature_off_cugraph_source_errors,
    )
    .await;
    let queue_high_watermark = backend.admission_snapshot()?.queue_high_watermark;
    queue_high_watermarks.push((current_step, u64::try_from(queue_high_watermark).unwrap_or(u64::MAX)));
    let cancellation_barrier = finish_cancellation_barrier(cancellation_barrier, &observations).await;
    let storm_cancellations_requested = finish_storm_cancellers(storm_cancellers).await;
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts,
        queue_high_watermarks,
        storm_cancellations_requested,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: None,
        feature_off_cugraph_source_errors,
    })
}

/// The submission deadline of one step. Step zero spans warmup plus its own
/// measurement window, matching the single-phase run deadline.
fn step_submission_deadline(started: Instant, plan: &ArrivalPlan, step_index: usize) -> Instant {
    let step = &plan.steps[step_index];
    started + Duration::from_nanos(step.start_ns.saturating_add(step.measurement_ns))
}

/// Joins every storm canceller and counts the cancellation actions that
/// actually fired. Cancellers own no resources; joining only makes their
/// requested timestamps deterministic.
async fn finish_storm_cancellers(cancellers: Vec<PendingCancellationBarrier>) -> u64 {
    let mut requested = 0_u64;
    for mut canceller in cancellers {
        if let Some(task) = canceller.task.take() {
            let _ = task.await;
        }
        requested += u64::from(canceller.requested.load(Ordering::Acquire));
    }
    requested
}

/// Samples the public aggregate queue count while the initial finite cohort is
/// live. The five-second bound is only an observation window; every submitted
/// attempt is still awaited and sealed by the normal drain below.
async fn observe_queue_peak(backend: &NexusGpuBackend, queue_limit: usize, run_deadline: Instant) -> ToolResult<usize> {
    const POLL_INTERVAL: Duration = Duration::from_millis(1);
    const OBSERVATION_WINDOW: Duration = Duration::from_secs(5);
    let deadline = (Instant::now() + OBSERVATION_WINDOW).min(run_deadline);
    let mut peak = 0;
    loop {
        let snapshot = backend.admission_snapshot()?;
        peak = peak.max(snapshot.queued_queries);
        if peak >= queue_limit || Instant::now() >= deadline {
            return Ok(peak);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        tokio::time::sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// Drives a two-cohort controlled-quarantine scenario through the ordinary
/// public attempt API. The first closed-loop round is intentionally kept live
/// until the target ledger exposes a grant or query job. Only after the
/// synchronous controller acknowledgement do we release the remaining rounds.
///
/// This is not a second scheduler or a backend mode: the sole state change is
/// `GpuBackendFaultInjector::quarantine_device`, which owns the same fleet
/// admission gate as normal placement. The runner merely orders public calls
/// and records the barrier they establish.
async fn drive_concurrent_quarantine(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    cancel_first_after: Option<Duration>,
    target_ordinal: usize,
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" {
        return Err("--quarantine-device requires concurrency scheduling so a pre-ack cohort can remain live".into());
    }
    let Some(first_post_ack_index) = plan.arrivals.iter().position(|arrival| arrival.round > 0) else {
        return Err(
            "--quarantine-device requires at least two closed-loop rounds: set scheduling.max_rounds to at least 2"
                .into(),
        );
    };
    if first_post_ack_index == 0 {
        return Err("--quarantine-device could not construct a non-empty pre-ack cohort".into());
    }

    let timing = RunTiming::new(plan, completion_drain_grace);
    let started = timing.started;
    let warmup_deadline = timing.warmup_deadline;
    let submission_deadline = timing.submission_deadline;
    let run_deadline = timing.completion_deadline;
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let dropped_overrun_arrivals = 0_u64;
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    // Dedicated probes never arm S5 storms; the knob set stays disabled.
    let mut storm_cancellers = Vec::new();

    for arrival in &plan.arrivals[..first_post_ack_index] {
        if Instant::now() >= submission_deadline {
            break;
        }
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            true,
            warmup_deadline,
            run_deadline,
            cancel_first_after,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => tasks.push(task),
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }
    if tasks.is_empty() {
        return Err("--quarantine-device could not begin a pre-acknowledgement cohort".into());
    }

    let witness = wait_for_target_live_work(backend, target_ordinal, started, submission_deadline).await?;
    let pre_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
    let injection = backend.fault_injector().quarantine_device(target_ordinal)?;
    let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
    let acknowledgement_offset_ns = elapsed_ns(started);
    let target_ledger_closed = backend
        .device_ledger_reports()?
        .into_iter()
        .find(|report| report.device_ordinal == target_ordinal)
        .is_some_and(|report| report.health == "closed");

    // The initial transition from round 0 to round 1 deliberately does not
    // drain `tasks`: the post-ack cohort must contend with, and then prove
    // release of, the work admitted before the quarantine acknowledgement.
    let mut current_round = 0;
    let mut post_ack_sequences = BTreeSet::new();
    for arrival in &plan.arrivals[first_post_ack_index..] {
        if Instant::now() >= submission_deadline {
            break;
        }
        if arrival.round != current_round {
            if current_round != 0 {
                drain(&mut tasks, &mut observations, &mut join_failures).await;
            }
            current_round = arrival.round;
        }
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            true,
            warmup_deadline,
            run_deadline,
            cancel_first_after,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => {
                post_ack_sequences.insert(arrival.sequence);
                tasks.push(task);
            },
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let pre_existing_target_work_released = backend
        .device_ledger_reports()?
        .into_iter()
        .find(|report| report.device_ordinal == target_ordinal)
        .is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty());
    let final_cache = cache_quarantine_observation(backend, target_ordinal).ok();
    let cancellation_barrier = finish_cancellation_barrier(cancellation_barrier, &observations).await;
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier,
        quarantine_barrier: Some(QuarantineBarrier {
            target_ordinal,
            pre_ack_witness_offset_ns: witness.offset_ns,
            pre_ack_ledger_sequence: witness.ledger_sequence,
            pre_ack_live_grants: witness.live_grants,
            pre_ack_active_query_jobs: witness.active_query_jobs,
            acknowledgement_offset_ns,
            device_health_generation: injection.device_health_generation(),
            newly_quarantined: injection.newly_quarantined(),
            target_ledger_closed,
            post_ack_sequences,
            pre_existing_target_work_released,
            pre_ack_cache,
            post_ack_cache,
            final_cache,
        }),
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: None,
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

/// Drives the S6 repeated-quarantine scenario: one controlled quarantine per
/// cycle against a rotation of configured ordinals while sustained closed-loop
/// load runs. Each cycle reuses the P6.6 barrier semantics — live-work
/// witness, synchronous injector acknowledgement, zero post-ack placements on
/// the target, healthy progress on the remaining ordinals — so a cycle is
/// exactly as falsifiable as the single-quarantine probe.
///
/// Quarantine is a terminal per-device transition, so the rotation targets a
/// distinct ordinal per cycle; each target's own health-generation increment
/// is recorded on its cycle row. Profile preflight keeps one ordinal open, so
/// stabilization between cycles is real fleet progress, not a drained fleet.
async fn drive_quarantine_cycles(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    cycles: usize,
    rotation: &[usize],
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" {
        return Err("--quarantine-cycles requires concurrency scheduling for its sustained closed-loop load".into());
    }
    if cycles == 0 || cycles > rotation.len().saturating_sub(1) {
        return Err("--quarantine-cycles needs a distinct target per cycle and one ordinal left open".into());
    }
    if plan.is_duration_paced_concurrency() {
        return drive_duration_paced_quarantine_cycles(
            backend,
            sessions,
            queries,
            plan,
            completion_drain_grace,
            cycles,
            rotation,
        )
        .await;
    }

    let timing = RunTiming::new(plan, completion_drain_grace);
    let started = timing.started;
    let warmup_deadline = timing.warmup_deadline;
    let submission_deadline = timing.submission_deadline;
    let run_deadline = timing.completion_deadline;
    // Cycle boundaries at defined window offsets: cycle k fires once the run
    // crosses (k + 1) / (cycles + 1) of the measurement window.
    let cycle_interval_ns = plan.measurement_ns / (cycles as u64 + 1);
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut current_round = 0;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    let mut barriers: Vec<QuarantineBarrier> = Vec::with_capacity(cycles);

    for arrival in &plan.arrivals {
        if Instant::now() >= submission_deadline {
            break;
        }
        if arrival.round != current_round {
            drain(&mut tasks, &mut observations, &mut join_failures).await;
            current_round = arrival.round;
        }
        while barriers.len() < cycles
            && elapsed_ns(started) >= cycle_interval_ns.saturating_mul(barriers.len() as u64 + 1)
        {
            let target_ordinal = rotation[barriers.len()];
            let witness = wait_for_target_live_work(backend, target_ordinal, started, submission_deadline).await?;
            let pre_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
            let injection = backend.fault_injector().quarantine_device(target_ordinal)?;
            let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
            let acknowledgement_offset_ns = elapsed_ns(started);
            let target_ledger_closed = backend
                .device_ledger_reports()?
                .into_iter()
                .find(|report| report.device_ordinal == target_ordinal)
                .is_some_and(|report| report.health == "closed");
            barriers.push(QuarantineBarrier {
                target_ordinal,
                pre_ack_witness_offset_ns: witness.offset_ns,
                pre_ack_ledger_sequence: witness.ledger_sequence,
                pre_ack_live_grants: witness.live_grants,
                pre_ack_active_query_jobs: witness.active_query_jobs,
                acknowledgement_offset_ns,
                device_health_generation: injection.device_health_generation(),
                newly_quarantined: injection.newly_quarantined(),
                target_ledger_closed,
                post_ack_sequences: BTreeSet::new(),
                pre_existing_target_work_released: false,
                pre_ack_cache,
                post_ack_cache,
                final_cache: None,
            });
        }
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            true,
            warmup_deadline,
            run_deadline,
            None,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => {
                // Every later submission is a post-acknowledgement arrival
                // for every cycle that has already landed.
                for barrier in &mut barriers {
                    barrier.post_ack_sequences.insert(arrival.sequence);
                }
                tasks.push(task);
            },
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let ledgers = backend.device_ledger_reports()?;
    for barrier in &mut barriers {
        barrier.final_cache = cache_quarantine_observation(backend, barrier.target_ordinal).ok();
        barrier.pre_existing_target_work_released = ledgers
            .iter()
            .find(|report| report.device_ordinal == barrier.target_ordinal)
            .is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty());
    }
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: barriers,
        close_under_load: None,
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

async fn drive_duration_paced_quarantine_cycles(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    cycles: usize,
    rotation: &[usize],
) -> ToolResult<DrivenRun> {
    let timing = RunTiming::new(plan, completion_drain_grace);
    let cycle_interval_ns = plan.measurement_ns / (cycles as u64 + 1);
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    let mut barriers: Vec<QuarantineBarrier> = Vec::with_capacity(cycles);
    let mut sequence = 0_usize;
    let mut submissions_open = true;

    for step in &plan.steps {
        let step_deadline = timing.started + Duration::from_nanos(step.start_ns.saturating_add(step.measurement_ns));
        let mut round = 0_usize;
        while submissions_open && Instant::now() < step_deadline && Instant::now() < timing.submission_deadline {
            let mut submitted_this_round = 0_usize;
            for _ in 0..step.concurrency {
                if Instant::now() >= step_deadline || Instant::now() >= timing.submission_deadline {
                    break;
                }
                let arrival = duration_paced_arrival(sequence, round, step.step_index, queries, sessions);
                sequence = sequence.saturating_add(1);
                match spawn_attempt_task(
                    backend,
                    sessions,
                    queries,
                    arrival,
                    true,
                    timing.warmup_deadline,
                    timing.completion_deadline,
                    None,
                    &mut cancellation_barrier,
                    StormKnobs::default(),
                    &mut storm_cancellers,
                ) {
                    Ok(task) => {
                        for barrier in &mut barriers {
                            barrier.post_ack_sequences.insert(arrival.sequence);
                        }
                        tasks.push(task);
                        submitted_this_round += 1;
                    },
                    Err(()) => {
                        begin_failures += 1;
                        submissions_open = false;
                        break;
                    },
                }
            }
            if submitted_this_round == 0 {
                break;
            }
            while barriers.len() < cycles
                && elapsed_ns(timing.started) >= cycle_interval_ns.saturating_mul(barriers.len() as u64 + 1)
            {
                let target_ordinal = rotation[barriers.len()];
                let witness =
                    wait_for_target_live_work(backend, target_ordinal, timing.started, timing.submission_deadline)
                        .await?;
                let pre_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
                let injection = backend.fault_injector().quarantine_device(target_ordinal)?;
                let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
                let acknowledgement_offset_ns = elapsed_ns(timing.started);
                let target_ledger_closed = backend
                    .device_ledger_reports()?
                    .into_iter()
                    .find(|report| report.device_ordinal == target_ordinal)
                    .is_some_and(|report| report.health == "closed");
                barriers.push(QuarantineBarrier {
                    target_ordinal,
                    pre_ack_witness_offset_ns: witness.offset_ns,
                    pre_ack_ledger_sequence: witness.ledger_sequence,
                    pre_ack_live_grants: witness.live_grants,
                    pre_ack_active_query_jobs: witness.active_query_jobs,
                    acknowledgement_offset_ns,
                    device_health_generation: injection.device_health_generation(),
                    newly_quarantined: injection.newly_quarantined(),
                    target_ledger_closed,
                    post_ack_sequences: BTreeSet::new(),
                    pre_existing_target_work_released: false,
                    pre_ack_cache,
                    post_ack_cache,
                    final_cache: None,
                });
            }
            drain(&mut tasks, &mut observations, &mut join_failures).await;
            round = round.saturating_add(1);
        }
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let ledgers = backend.device_ledger_reports()?;
    for barrier in &mut barriers {
        barrier.final_cache = cache_quarantine_observation(backend, barrier.target_ordinal).ok();
        barrier.pre_existing_target_work_released = ledgers
            .iter()
            .find(|report| report.device_ordinal == barrier.target_ordinal)
            .is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty());
    }
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: barriers,
        close_under_load: None,
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

/// Drives the S6 close-under-saturation scenario. A closed-loop cohort
/// oversubscribes the fleet; at the measurement-window midpoint the runner
/// stops submitting, snapshots public admission state, and initiates the
/// public close path. Queued waiters must seal `backend_closing`, in-flight
/// attempts must finish, and every submitted attempt must be accounted for
/// exactly once. The runner never aborts a task to force the shape.
async fn drive_close_under_load(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" {
        return Err("--close-under-load requires concurrency scheduling so close lands mid-cohort".into());
    }
    if plan.is_duration_paced_concurrency() {
        return drive_duration_paced_close_under_load(backend, sessions, queries, plan, completion_drain_grace).await;
    }

    let timing = RunTiming::new(plan, completion_drain_grace);
    let started = timing.started;
    let warmup_deadline = timing.warmup_deadline;
    let run_deadline = timing.completion_deadline;
    let close_at = started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns / 2);
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut current_round = 0;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    let mut submitted_attempts = 0_u64;
    let mut live_sequences = BTreeSet::new();

    for arrival in &plan.arrivals {
        if Instant::now() >= close_at || Instant::now() >= run_deadline {
            break;
        }
        if arrival.round != current_round {
            drain(&mut tasks, &mut observations, &mut join_failures).await;
            live_sequences.clear();
            current_round = arrival.round;
        }
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            true,
            warmup_deadline,
            run_deadline,
            None,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => {
                submitted_attempts += 1;
                live_sequences.insert(arrival.sequence);
                tasks.push(task);
            },
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }

    // The close boundary: public admission facts first, then the public
    // close transition. No further submissions happen after this point.
    let (admitted_at_close, queued_at_close) = wait_for_close_under_load_cohort(backend, run_deadline).await?;
    let close_requested_offset_ns = elapsed_ns(started);
    let close_initiated = backend.close();
    let barrier = CloseUnderLoadBarrier {
        close_requested_offset_ns,
        close_initiated,
        admitted_at_close,
        queued_at_close,
        in_flight_at_close: tasks.len(),
        in_flight_sequences: std::mem::take(&mut live_sequences),
        submitted_attempts,
    };
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: Some(barrier),
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

async fn drive_duration_paced_close_under_load(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
) -> ToolResult<DrivenRun> {
    let timing = RunTiming::new(plan, completion_drain_grace);
    let close_at = timing.started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns / 2);
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    let mut storm_cancellers = Vec::new();
    let mut sequence = 0_usize;
    let mut submitted_attempts = 0_u64;
    let mut last_step = &plan.steps[0];

    for step in &plan.steps {
        if Instant::now() >= close_at {
            break;
        }
        last_step = step;
        let step_deadline =
            (timing.started + Duration::from_nanos(step.start_ns.saturating_add(step.measurement_ns))).min(close_at);
        let mut round = 0_usize;
        while Instant::now() < step_deadline {
            let mut submitted_this_round = 0_usize;
            for _ in 0..step.concurrency {
                if Instant::now() >= step_deadline {
                    break;
                }
                let arrival = duration_paced_arrival(sequence, round, step.step_index, queries, sessions);
                sequence = sequence.saturating_add(1);
                match spawn_attempt_task(
                    backend,
                    sessions,
                    queries,
                    arrival,
                    true,
                    timing.warmup_deadline,
                    timing.completion_deadline,
                    None,
                    &mut cancellation_barrier,
                    StormKnobs::default(),
                    &mut storm_cancellers,
                ) {
                    Ok(task) => {
                        tasks.push(task);
                        submitted_attempts += 1;
                        submitted_this_round += 1;
                    },
                    Err(()) => {
                        begin_failures += 1;
                        break;
                    },
                }
            }
            if submitted_this_round == 0 {
                break;
            }
            drain(&mut tasks, &mut observations, &mut join_failures).await;
            round = round.saturating_add(1);
        }
    }

    if Instant::now() < close_at {
        tokio::time::sleep_until(tokio::time::Instant::from_std(close_at)).await;
    }
    let mut in_flight_sequences = BTreeSet::new();
    for _ in 0..last_step.concurrency {
        let arrival = duration_paced_arrival(sequence, 0, last_step.step_index, queries, sessions);
        sequence = sequence.saturating_add(1);
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            arrival,
            true,
            timing.warmup_deadline,
            timing.completion_deadline,
            None,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => {
                in_flight_sequences.insert(arrival.sequence);
                tasks.push(task);
                submitted_attempts += 1;
            },
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }

    let (admitted_at_close, queued_at_close) =
        wait_for_close_under_load_cohort(backend, timing.completion_deadline).await?;
    let close_requested_offset_ns = elapsed_ns(timing.started);
    let close_initiated = backend.close();
    let barrier = CloseUnderLoadBarrier {
        close_requested_offset_ns,
        close_initiated,
        admitted_at_close,
        queued_at_close,
        in_flight_at_close: tasks.len(),
        in_flight_sequences,
        submitted_attempts,
    };
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: Some(barrier),
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

/// Waits until the public admission snapshot proves the close boundary has
/// both executing work and queued waiters. The runner submits no work after
/// this witness; it then closes through the public backend path.
async fn wait_for_close_under_load_cohort(backend: &NexusGpuBackend, deadline: Instant) -> ToolResult<(usize, usize)> {
    loop {
        let admission = backend.admission_snapshot()?;
        if admission.active_queries > 0 && admission.queued_queries > 0 {
            return Ok((admission.active_queries, admission.queued_queries));
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "--close-under-load did not observe an active-and-queued public cohort before its deadline (active {}, queued {})",
                admission.active_queries, admission.queued_queries
            )
            .into());
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// A queued waiter sealed by close carries the bounded `BackendClosing`
/// failure class in its sealed report. The runner reads only that closed
/// vocabulary, never reconstructed queue ordering.
fn is_typed_backend_closing(observation: &AttemptObservation) -> bool {
    observation.failure.as_ref().is_some_and(|failure| failure.code == "BackendClosing")
}

/// Drives one attempt with a shorter caller deadline, but only accepts it as
/// P6.6 evidence after public snapshots first witness an immutable admission
/// envelope. The deadline remains the normal attempt contract: no runner task
/// is aborted and a CUDA call already in flight may finish before the next
/// native safe boundary observes expiry.
async fn drive_deadline_after_admission(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    completion_drain_grace: Duration,
    deadline_after: Duration,
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" {
        return Err("--deadline-first-after-ms requires concurrency scheduling".into());
    }
    if plan.arrivals.len() < 2 {
        return Err("--deadline-first-after-ms requires a follow-up attempt for healthy reuse evidence".into());
    }

    let timing = RunTiming::new(plan, completion_drain_grace);
    let started = timing.started;
    let warmup_deadline = timing.warmup_deadline;
    let submission_deadline = timing.submission_deadline;
    let run_deadline = timing.completion_deadline;
    let deadline = started.checked_add(deadline_after).ok_or("--deadline-first-after-ms overflows Instant")?;
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut join_failures = 0_u64;
    let mut begin_failures = 0_u64;
    let mut cancellation_barrier = None;
    // Dedicated probes never arm S5 storms; the knob set stays disabled.
    let mut storm_cancellers = Vec::new();

    let first = plan.arrivals[0];
    let task = spawn_attempt_task(
        backend,
        sessions,
        queries,
        first,
        true,
        warmup_deadline,
        deadline,
        None,
        &mut cancellation_barrier,
        StormKnobs::default(),
        &mut storm_cancellers,
    )
    .map_err(|_| "--deadline-first-after-ms could not begin the deadline probe attempt")?;
    tasks.push(task);

    let witness = wait_for_any_live_work(backend, started, deadline).await?;
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let released_after_deadline = wait_for_target_release(backend, witness.target_ordinal, run_deadline).await?;
    let typed_deadline_cancelled_reports = observations
        .iter()
        .filter(|observation| observation.sequence == first.sequence && is_typed_deadline_cancellation(observation))
        .count() as u64;

    // The caller deadline applies only to the dedicated first attempt. Every
    // later arrival follows the ordinary measured-window deadline, proving the
    // same healthy controller can admit and complete subsequent native work.
    let mut post_deadline_sequences = BTreeSet::new();
    let mut current_round = first.round;
    for arrival in &plan.arrivals[1..] {
        if Instant::now() >= submission_deadline {
            break;
        }
        if arrival.round != current_round {
            drain(&mut tasks, &mut observations, &mut join_failures).await;
            current_round = arrival.round;
        }
        match spawn_attempt_task(
            backend,
            sessions,
            queries,
            *arrival,
            true,
            warmup_deadline,
            run_deadline,
            None,
            &mut cancellation_barrier,
            StormKnobs::default(),
            &mut storm_cancellers,
        ) {
            Ok(task) => {
                post_deadline_sequences.insert(arrival.sequence);
                tasks.push(task);
            },
            Err(()) => {
                begin_failures += 1;
                break;
            },
        }
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let healthy_reuse_completed = observations.iter().any(|observation| {
        post_deadline_sequences.contains(&observation.sequence)
            && observation.succeeded
            && observation.reached_gpu
            && observation.device_ordinal == Some(witness.target_ordinal)
    });
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: timing.deadline_evidence(),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: Some(DeadlineBarrierEvidence {
            attempt_sequence: first.sequence,
            deadline_after_ms: deadline_after.as_millis().min(u128::from(u64::MAX)) as u64,
            admitted_witness_offset_ns: witness.offset_ns,
            admitted_device_ordinal: witness.target_ordinal,
            admitted_ledger_sequence: witness.ledger_sequence,
            admitted_live_grants: witness.live_grants,
            admitted_active_query_jobs: witness.active_query_jobs,
            typed_deadline_cancelled_reports,
            released_after_deadline,
            healthy_reuse_completed,
        }),
        physical_allocator_pressure: None,
        observed_peak_queued_attempts: 0,
        queue_high_watermarks: Vec::new(),
        storm_cancellations_requested: 0,
        quarantine_cycle_barriers: Vec::new(),
        close_under_load: None,
        feature_off_cugraph_source_errors: Vec::new(),
    })
}

struct TargetLiveWorkWitness {
    target_ordinal: usize,
    offset_ns: u64,
    ledger_sequence: u64,
    live_grants: usize,
    active_query_jobs: usize,
}

/// Waits only on public pull snapshots. A target qualifies once an immutable
/// grant has been published or a native query job is live; the artifact keeps
/// both counts rather than pretending a configured slot proves execution.
async fn wait_for_target_live_work(
    backend: &NexusGpuBackend,
    target_ordinal: usize,
    started: Instant,
    deadline: Instant,
) -> ToolResult<TargetLiveWorkWitness> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let ledgers = backend.device_ledger_reports()?;
        let pools = backend.gpu_execution_pool_reports()?;
        let Some(ledger) = ledgers.into_iter().find(|report| report.device_ordinal == target_ordinal) else {
            return Err(format!("--quarantine-device {target_ordinal} disappeared from public ledger reports").into());
        };
        let active_query_jobs = pools
            .into_iter()
            .find(|report| report.device_ordinal == target_ordinal)
            .map_or(0, |report| report.active_query_jobs);
        if !ledger.grants.is_empty() || active_query_jobs > 0 {
            return Ok(TargetLiveWorkWitness {
                target_ordinal,
                offset_ns: elapsed_ns(started),
                ledger_sequence: ledger.sequence,
                live_grants: ledger.grants.len(),
                active_query_jobs,
            });
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Err(format!(
                "--quarantine-device {target_ordinal} did not expose a live grant or query job before the run deadline"
            )
            .into());
        };
        tokio::time::sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// Picks the actual device that first exposes a live immutable grant or
/// native query job. The deadline scenario intentionally does not introduce
/// caller device affinity; its artifact records the controller's real choice.
async fn wait_for_any_live_work(
    backend: &NexusGpuBackend,
    started: Instant,
    deadline: Instant,
) -> ToolResult<TargetLiveWorkWitness> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let ledgers = backend.device_ledger_reports()?;
        let pools = backend.gpu_execution_pool_reports()?;
        for ledger in ledgers {
            let active_query_jobs = pools
                .iter()
                .find(|report| report.device_ordinal == ledger.device_ordinal)
                .map_or(0, |report| report.active_query_jobs);
            if !ledger.grants.is_empty() || active_query_jobs > 0 {
                return Ok(TargetLiveWorkWitness {
                    target_ordinal: ledger.device_ordinal,
                    offset_ns: elapsed_ns(started),
                    ledger_sequence: ledger.sequence,
                    live_grants: ledger.grants.len(),
                    active_query_jobs,
                });
            }
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Err("--deadline-first-after-ms elapsed before the first attempt exposed admitted work".into());
        };
        tokio::time::sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

async fn wait_for_target_release(
    backend: &NexusGpuBackend,
    target_ordinal: usize,
    deadline: Instant,
) -> ToolResult<bool> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let reports = backend.device_ledger_reports()?;
        let Some(target) = reports.into_iter().find(|report| report.device_ordinal == target_ordinal) else {
            return Err(format!(
                "deadline probe target ordinal {target_ordinal} disappeared from public ledger reports"
            )
            .into());
        };
        if target.active_attempts == 0 && target.grants.is_empty() {
            return Ok(true);
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Ok(false);
        };
        tokio::time::sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// The attempt report preserves `deadline_expired` in its bounded reason
/// vocabulary. Inspect only that JSON projection: never rehydrate arbitrary
/// error text or infer a timeout from the runner clock.
fn is_typed_deadline_cancellation(observation: &AttemptObservation) -> bool {
    observation.outcome == "cancelled"
        && observation.terminal_evidence.primary_failure.as_ref().is_some_and(|failure| {
            failure.get("facts").and_then(Value::as_array).is_some_and(|facts| {
                facts.iter().any(|fact| {
                    fact.get("key").and_then(Value::as_str) == Some("reason")
                        && fact.get("value").and_then(Value::as_str) == Some("deadline_expired")
                })
            })
        })
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn run_deadline_evidence(
    measurement_window_ns: u64,
    completion_drain_grace_ns: u64,
    elapsed_until_reports_ns: u64,
) -> RunDeadlineEvidence {
    let configured_window_ns = measurement_window_ns.saturating_add(completion_drain_grace_ns);
    RunDeadlineEvidence {
        measurement_window_ns,
        completion_drain_grace_ns,
        configured_window_ns,
        elapsed_until_reports_ns,
        overrun_ns: elapsed_until_reports_ns.saturating_sub(configured_window_ns),
        reports_sealed_within_deadline: elapsed_until_reports_ns <= configured_window_ns,
    }
}

/// Produces the public attempt contract for one bounded measured window.
///
/// The engine propagates this deadline through planning, admission, stream
/// acquisition, GPU safe boundaries, and result drain.  We deliberately do
/// not wrap the task in `tokio::time::timeout`: aborting the task could skip
/// the linear attempt finalizer that returns a grant and seals its report.
fn attempt_options(run_deadline: Instant) -> AttemptOptions {
    AttemptOptions { deadline: Some(run_deadline), ..AttemptOptions::default() }
}

/// Per-attempt evidence that is not itself part of the sealed report.
#[derive(Debug, Clone, Default)]
struct OutputCapture {
    feature_off_cugraph_source_error: Option<FeatureOffCugraphSourceError>,
    graph_claim_calibration: Option<GraphClaimCalibrationSnapshot>,
}

type AttemptTask =
    tokio::task::JoinHandle<(super::schedule::ScheduledArrival, SelectedQuery, Arc<QueryAttemptReport>, OutputCapture)>;

/// Schedules each runner-owned S1 request through the ordinary public attempt
/// path.  Only the selected query and sequence differ from a planned
/// relational arrival; no result or error evidence is synthesized here.
#[allow(clippy::too_many_arguments)]
fn spawn_feature_off_cugraph_probes(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    probe: FeatureOffCugraphProbeSchedule,
    count: usize,
    next_sequence: &mut usize,
    round: usize,
    step_index: usize,
    warmup_deadline: Instant,
    run_deadline: Instant,
    cancellation_barrier: &mut Option<PendingCancellationBarrier>,
    storm: StormKnobs,
    storm_cancellers: &mut Vec<PendingCancellationBarrier>,
    tasks: &mut Vec<AttemptTask>,
) -> Result<(), ()> {
    for _ in 0..count {
        let sequence = *next_sequence;
        *next_sequence = (*next_sequence).saturating_add(1);
        let arrival = super::schedule::ScheduledArrival {
            sequence,
            round,
            step_index,
            offset_ns: 0,
            // `spawn_attempt_task` records the observed closed-loop phase at
            // submission, which is the only phase evidence the S1 reducer
            // trusts. This placeholder is therefore never artifact-visible.
            phase: SchedulePhase::Warmup,
            rate_overrun: RateOverrunPolicy::Delay,
            query_index: probe.query_index,
            session_index: 0,
        };
        tasks.push(spawn_attempt_task(
            backend,
            sessions,
            queries,
            arrival,
            true,
            warmup_deadline,
            run_deadline,
            None,
            cancellation_barrier,
            storm,
            storm_cancellers,
        )?);
    }
    Ok(())
}

/// Begins one ordinary public attempt and packages its report collector. The
/// quarantine scenario uses the same submission path as a regular measured
/// run; only the placement of the controller acknowledgement differs.
#[allow(clippy::too_many_arguments)]
fn spawn_attempt_task(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    arrival: super::schedule::ScheduledArrival,
    closed_loop: bool,
    warmup_deadline: Instant,
    attempt_deadline: Instant,
    cancel_first_after: Option<Duration>,
    cancellation_barrier: &mut Option<PendingCancellationBarrier>,
    storm: StormKnobs,
    storm_cancellers: &mut Vec<PendingCancellationBarrier>,
) -> Result<AttemptTask, ()> {
    spawn_attempt_task_with_query_id(
        backend,
        sessions,
        queries,
        arrival,
        closed_loop,
        warmup_deadline,
        attempt_deadline,
        cancel_first_after,
        cancellation_barrier,
        storm,
        storm_cancellers,
        None,
    )
}

/// The normal runner deliberately lets the public attempt API mint its own
/// query id. The S7 exact-cohort probe is the narrow exception: it supplies a
/// unique id whose one-way fingerprint can be matched against cache-owner and
/// BFS-coordinator observations without retaining the raw id in the artifact.
#[allow(clippy::too_many_arguments)]
fn spawn_attempt_task_with_query_id(
    backend: &NexusGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    mut arrival: super::schedule::ScheduledArrival,
    closed_loop: bool,
    warmup_deadline: Instant,
    attempt_deadline: Instant,
    cancel_first_after: Option<Duration>,
    cancellation_barrier: &mut Option<PendingCancellationBarrier>,
    storm: StormKnobs,
    storm_cancellers: &mut Vec<PendingCancellationBarrier>,
    query_id: Option<String>,
) -> Result<AttemptTask, ()> {
    // S5 deadline storm: each attempt carries its own caller deadline, never
    // later than the shared window deadline.
    let attempt_deadline = storm
        .attempt_deadline_ms
        .map_or(attempt_deadline, |ms| attempt_deadline.min(Instant::now() + Duration::from_millis(ms)));
    let attempt =
        backend.begin_attempt(AttemptOptions { query_id, ..attempt_options(attempt_deadline) }).map_err(|_| ())?;
    if cancellation_barrier.is_none()
        && let Some(requested_after) = cancel_first_after
    {
        let requested = Arc::new(AtomicBool::new(false));
        let cancellation = attempt.cancellation();
        let task = if requested_after.is_zero() {
            cancellation.cancel();
            requested.store(true, Ordering::Release);
            None
        } else {
            let requested = Arc::clone(&requested);
            Some(tokio::spawn(async move {
                tokio::time::sleep(requested_after).await;
                cancellation.cancel();
                requested.store(true, Ordering::Release);
            }))
        };
        *cancellation_barrier =
            Some(PendingCancellationBarrier { attempt_sequence: arrival.sequence, requested_after, requested, task });
    }
    // S5 cancellation storm: every Nth attempt gets its own cooperative
    // canceller through the same public cancellation handle.
    if let Some(every_n) = storm.cancel_every_n
        && arrival.sequence % every_n == every_n - 1
    {
        let requested_after = Duration::from_millis(storm.cancel_after_ms.unwrap_or(0));
        let requested = Arc::new(AtomicBool::new(false));
        let cancellation = attempt.cancellation();
        let task = if requested_after.is_zero() {
            cancellation.cancel();
            requested.store(true, Ordering::Release);
            None
        } else {
            let requested = Arc::clone(&requested);
            Some(tokio::spawn(async move {
                tokio::time::sleep(requested_after).await;
                cancellation.cancel();
                requested.store(true, Ordering::Release);
            }))
        };
        storm_cancellers.push(PendingCancellationBarrier {
            attempt_sequence: arrival.sequence,
            requested_after,
            requested,
            task,
        });
    }
    let query = queries[arrival.query_index].clone();
    let fixture_sessions =
        sessions.get(&query.fixture_index).expect("install_sessions registers every selected fixture workload");
    let ctx = fixture_sessions[arrival.session_index % fixture_sessions.len()].clone();
    if closed_loop {
        // Closed-loop arrivals carry no planned offset. Classify at the
        // submission boundary, before any later report drain can cross the
        // warmup deadline.
        arrival.phase =
            if Instant::now() < warmup_deadline { SchedulePhase::Warmup } else { SchedulePhase::Measurement };
    }
    Ok(tokio::spawn(async move {
        // A failing statement is a measured outcome, not a runner error: the
        // sealed report still describes what happened. Qualification drains
        // result streams without inspecting or retaining output rows.
        let feature_off_cugraph_probe = query.label == FEATURE_OFF_CUGRAPH_QUERY_LABEL;
        let mut capture = OutputCapture::default();
        match attempt.sql(&ctx, &query.sql).await {
            Ok(stream) => {
                let mut stream = stream;
                while let Some(batch) = stream.next().await {
                    match batch {
                        Ok(_) => {},
                        Err(error) => {
                            if feature_off_cugraph_probe {
                                capture.feature_off_cugraph_source_error =
                                    Some(feature_off_cugraph_source_error(&error));
                            }
                            break;
                        },
                    }
                }
            },
            Err(error) => {
                if feature_off_cugraph_probe {
                    capture.feature_off_cugraph_source_error = Some(feature_off_cugraph_source_error(&error));
                }
            },
        }
        // I8's optional sink deliberately serializes the public completion
        // receiver. The permit remains live through `.completion().await`,
        // so a saturated sink can exert the real handoff backpressure that a
        // per-task sleep before this point would miss.
        let completion_sink = query.completion_sink().cloned();
        let _completion_sink_permit = if let Some(sink) = completion_sink.as_ref() {
            let permit = sink.acquire().await;
            tokio::time::sleep(sink.delay()).await;
            Some(permit)
        } else {
            None
        };
        let report = attempt.completion().await;
        capture.graph_claim_calibration =
            attempt.sealed_graph_claim_calibration_snapshot().map(project_graph_claim_calibration_snapshot);
        if let Some(sink) = completion_sink.as_ref() {
            sink.mark_drained();
        }
        (arrival, query, report, capture)
    }))
}

/// Reduces the public adapter error by its typed code, operation, and
/// log-safe facts. The artifact receives only equality bits for the stable
/// contract, so neither display text nor an arbitrary source chain can leak
/// into a measured-run artifact.
fn feature_off_cugraph_source_error(error: &datafusion::common::DataFusionError) -> FeatureOffCugraphSourceError {
    let Some(error) = error.find_nexus_error() else {
        return FeatureOffCugraphSourceError { source_error_observed: true, ..FeatureOffCugraphSourceError::default() };
    };
    FeatureOffCugraphSourceError {
        source_error_observed: true,
        unsupported_operator: error.code().as_str() == "unsupported_operator",
        operation_matches: error.operation() == FEATURE_OFF_CUGRAPH_OPERATION,
        reason_matches: error.fact_value("reason") == Some("required_feature_disabled"),
        required_feature_matches: error.fact_value("required_feature") == Some("cugraph"),
        host_fallback_matches: error.fact_value("host_fallback") == Some("disabled"),
    }
}

async fn finish_cancellation_barrier(
    mut barrier: Option<PendingCancellationBarrier>,
    observations: &[AttemptObservation],
) -> Option<CancellationBarrierEvidence> {
    let mut barrier = barrier.take()?;
    if let Some(task) = barrier.task.take() {
        // The canceller has no resource ownership; joining it only makes the
        // requested action's timestamp/result deterministic.
        let _ = task.await;
    }
    Some(CancellationBarrierEvidence {
        attempt_sequence: barrier.attempt_sequence,
        requested_after_ms: barrier.requested_after.as_millis().min(u128::from(u64::MAX)) as u64,
        cancellation_requested: barrier.requested.load(Ordering::Acquire),
        cancelled_reports: observations
            .iter()
            .filter(|observation| {
                observation.sequence == barrier.attempt_sequence && observation.outcome == "cancelled"
            })
            .count() as u64,
    })
}

/// Awaits every in-flight attempt and folds its report into an observation.
async fn drain(tasks: &mut Vec<AttemptTask>, observations: &mut Vec<AttemptObservation>, join_failures: &mut u64) {
    for task in tasks.drain(..) {
        // A panicked or cancelled task loses only its own row. Aborting here
        // would discard a whole measured run's evidence.
        match task.await {
            Ok((arrival, query, report, capture)) => {
                observations.push(observe(
                    arrival.sequence,
                    arrival.phase,
                    arrival.step_index,
                    &query,
                    &report,
                    capture,
                ));
            },
            Err(_) => *join_failures += 1,
        }
    }
}

/// Ordinary stress cells use [`drain`]. The feature-off S1 driver retains one
/// extra bounded source-error projection for its runner-owned UDTF so the
/// artifact can pair exact adapter facts with the same sealed report row.
async fn drain_with_feature_off_cugraph_capture(
    tasks: &mut Vec<AttemptTask>,
    observations: &mut Vec<AttemptObservation>,
    join_failures: &mut u64,
    source_errors: &mut Vec<(usize, FeatureOffCugraphSourceError)>,
) {
    for task in tasks.drain(..) {
        match task.await {
            Ok((arrival, query, report, capture)) => {
                if query.label == FEATURE_OFF_CUGRAPH_QUERY_LABEL {
                    source_errors
                        .push((arrival.sequence, capture.feature_off_cugraph_source_error.unwrap_or_default()));
                }
                observations.push(observe(
                    arrival.sequence,
                    arrival.phase,
                    arrival.step_index,
                    &query,
                    &report,
                    capture,
                ));
            },
            Err(_) => *join_failures += 1,
        }
    }
}

fn observe(
    sequence: usize,
    phase: SchedulePhase,
    step_index: usize,
    query: &SelectedQuery,
    report: &QueryAttemptReport,
    capture: OutputCapture,
) -> AttemptObservation {
    let resource_evidence = report.resource_evidence.as_ref();
    AttemptObservation {
        sequence,
        phase,
        step_index,
        query_kind: query.kind,
        query_label: query.label.clone(),
        graph_workload: query.graph,
        placement_required: query.placement_required,
        attempt_id: report.attempt_id.to_string(),
        disposition: format!("{:?}", report.disposition),
        outcome: report.outcome.stable_label().to_string(),
        failure: report.primary_failure.as_ref().map(|failure| AttemptFailureClass {
            origin: format!("{:?}", failure.origin()),
            code: format!("{:?}", failure.code()),
            status: format!("{:?}", failure.status()),
            operation: format!("{:?}", failure.operation()),
        }),
        terminal_evidence: terminal_evidence(report),
        succeeded: report.outcome.stable_label() == super::aggregate::SUCCEEDED_OUTCOME,
        reached_gpu: matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
        graph_claim_calibration: capture.graph_claim_calibration,
        composed_schedule: report
            .composed_schedule
            .as_ref()
            .filter(|projection| projection.schema == COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1)
            .map(project_composed_schedule_projection),
        device_ordinal: report.admission.as_ref().map(|admission| admission.device_ordinal).or(report.device_ordinal),
        grant_bytes: report.admission.as_ref().map(|admission| admission.grant_bytes),
        admission_sequence: report.admission.as_ref().map(|admission| admission.admission_sequence),
        admission_wait_ns: report.admission.as_ref().map(|admission| admission.admission_wait_ns),
        admission_requested_ns: report.timestamps.admission_requested_ns,
        admitted_ns: report.timestamps.admitted_ns,
        admission_wait: report.admission_wait.map(|fact| super::aggregate::AdmissionWaitEvidence {
            submission_sequence: fact.submission_sequence,
            overtake_limit: fact.overtake_limit,
            overtakes_consumed: fact.overtakes_consumed,
            overtakes_remaining: fact.overtakes_remaining,
            protection_device_ordinal: fact.protection_device_ordinal,
        }),
        grant_id: resource_evidence.and_then(|evidence| evidence.grant_id.value),
        domain_id: resource_evidence.and_then(|evidence| evidence.domain_id.value),
        total_ns: report.total_ns(),
        queue_wait_ns: report.queue_wait_ns(),
        physical_execution_ns: report.physical_execution_ns(),
        // The engine stamps both from one observation clock shared by every
        // attempt, so these intervals are directly comparable. A host-side
        // stamp taken in the collector loop would only record collector order.
        submitted_ns: report.timestamps.submitted_ns,
        outcome_terminal_ns: report.timestamps.outcome_terminal_ns,
    }
}

/// Copies a schedule only from the sealed engine projection. This conversion
/// changes artifact field names but never derives a byte, scope, order, or
/// fingerprint in the runner.
fn project_composed_schedule_projection(projection: &AttemptComposedScheduleProjection) -> ComposedScheduleArtifact {
    ComposedScheduleArtifact {
        schema: super::COMPOSED_SCHEDULE_SCHEMA,
        version: super::COMPOSED_SCHEDULE_VERSION,
        attempt_id: projection.attempt_id.to_string(),
        selected_ordinal: projection.device_ordinal,
        grant_id: projection.grant_id,
        grant_bytes: projection.grant_bytes,
        schedule_fingerprint: projection.fingerprint.clone(),
        engine_fingerprint_matches_contents: projection.fingerprint_matches_contents(),
        attempt_wide_fixed_bytes: projection.attempt_wide_fixed_bytes,
        occurrences: projection
            .producer_occurrences
            .iter()
            .enumerate()
            .map(|(order, occurrence)| ComposedScheduleOccurrence {
                occurrence_id: occurrence.occurrence_id.clone(),
                order,
                structural_scope: occurrence.execution_scope_fingerprint.clone(),
                isolated_peak_bytes: occurrence.isolated_peak_bytes,
                stage_peak_bytes: occurrence.stage_peak_bytes,
                retained_output_bytes: occurrence.retained_output_bytes,
            })
            .collect(),
        consumer_structural_scope: projection.consumer_scope_fingerprint.clone(),
        consumer_additional_peak_bytes: projection.consumer_additional_peak_bytes,
        consumer_stage_peak_bytes: projection.consumer_stage_peak_bytes,
        complete_phase_peak_bytes: projection.complete_peak_bytes,
    }
}

/// Projects the root adapter's immutable typed snapshot into the versioned
/// stress artifact. This intentionally reads engine fields directly rather
/// than reconstructing graph rows from DataFusion metric labels or error text.
fn project_graph_claim_calibration_snapshot(
    snapshot: datafusion_nexus::attempt::SealedAttemptGraphClaimCalibrationSnapshot,
) -> GraphClaimCalibrationSnapshot {
    GraphClaimCalibrationSnapshot {
        truncated: snapshot.truncated(),
        execution: match snapshot.execution() {
            Some(datafusion_nexus::attempt::GraphClaimCalibrationExecution::Physical) => {
                Some(super::aggregate::GraphClaimCalibrationExecution::Physical)
            },
            Some(datafusion_nexus::attempt::GraphClaimCalibrationExecution::NonPhysical(
                datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower {
                    leader_attempt_id,
                },
            )) => Some(super::aggregate::GraphClaimCalibrationExecution::NonPhysical(
                super::aggregate::GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower {
                    leader_attempt_id: leader_attempt_id.as_ref().map(ToString::to_string),
                },
            )),
            Some(datafusion_nexus::attempt::GraphClaimCalibrationExecution::NonPhysical(
                datafusion_nexus::attempt::GraphClaimCalibrationNonPhysicalReason::GraphMemoryAdmissionDenied,
            )) => Some(super::aggregate::GraphClaimCalibrationExecution::NonPhysical(
                super::aggregate::GraphClaimCalibrationNonPhysicalReason::GraphMemoryAdmissionDenied,
            )),
            None => None,
        },
        execution_complete: snapshot.execution_complete(),
        rows: snapshot
            .rows()
            .iter()
            .map(|row| GraphClaimCalibrationRow {
                node_id: row.node_id.clone(),
                algorithm: row.algorithm.as_str().to_string(),
                model_requested_bytes: row.model_requested_bytes,
                requested_bytes: row.requested_bytes,
                budget_bytes: row.budget_bytes,
                execution_state: row.execution_state.as_str().to_string(),
                measured_peak_bytes: row.measured_peak_bytes,
                measured_peak_provenance: row.measured_peak_provenance.as_str().to_string(),
                graph_model_error_ratio: row.graph_model_error_ratio.map(|ratio| ratio.as_decimal_string()),
            })
            .collect(),
    }
}

/// Projects P6.6's report facts without persisting caller correlations or
/// arbitrary failure text. The sealed report remains the sole event record.
fn terminal_evidence(report: &QueryAttemptReport) -> AttemptTerminalEvidence {
    AttemptTerminalEvidence {
        primary_failure: report.primary_failure.as_ref().map(redacted_failure_summary),
        cleanup_disposition: json_value(report.cleanup_disposition),
        cleanup_failure_count: report.cleanup_failure_count,
        cleanup_failure_count_complete: report.cleanup_failure_count_complete,
        first_cleanup_failure: report.first_cleanup_failure.as_ref().map(redacted_cleanup_failure_summary),
        capacity_disposition: json_value(report.capacity_disposition),
        health_transition: report.health_transition.map(json_value),
        native_fault_evidence: report.native_fault_evidence.map(json_value),
        resource_evidence: report.resource_evidence.map(json_value),
        retry_advice: json_value(report.retry_advice),
    }
}

fn redacted_failure_summary(summary: &nexus_query_engine::observability::AttemptFailureSummary) -> Value {
    let mut value = json_value(summary);
    if let Value::Object(fields) = &mut value {
        fields.remove("text");
    }
    value
}

fn redacted_cleanup_failure_summary(summary: &nexus_query_engine::observability::CleanupFailureSummary) -> Value {
    let mut value = json_value(summary);
    if let Value::Object(fields) = &mut value
        && let Some(Value::Object(failure)) = fields.get_mut("failure")
    {
        failure.remove("text");
    }
    value
}

fn json_value<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// Closes the public backend once and retains an auditable bounded projection
/// of either outcome. Do not convert an error to a `null` artifact field: P6.6
/// treats a failed close as a real result that must remain visible.
fn close_backend(backend: &NexusGpuBackend) -> BackendCloseEvidence {
    let before = project_close_snapshot(backend.close_snapshot());
    let outcome = match backend.close_and_wait(CLOSE_TIMEOUT) {
        Ok(snapshot) => BackendCloseOutcome::Quiescent { snapshot: project_close_snapshot(snapshot) },
        Err(error) => BackendCloseOutcome::Failed {
            failure: BackendCloseFailure {
                code: error.code().as_str(),
                kind: error.kind().as_str(),
                status: error.status().as_str(),
                operation: error.operation(),
                // These are the only close facts that carry a closed,
                // non-caller-controlled vocabulary. `cleanup_error` and any
                // source text intentionally never leave the engine.
                reason: error.fact_value("reason").map(str::to_owned),
            },
        },
    };
    let after = project_close_snapshot(backend.close_snapshot());
    BackendCloseEvidence { before, outcome, after }
}

fn project_close_snapshot(
    snapshot: nexus_query_engine::runtime::GpuBackendCloseSnapshot,
) -> BackendCloseSnapshotEvidence {
    BackendCloseSnapshotEvidence {
        closing: snapshot.closing,
        gpu_resource_quiescent: snapshot.gpu_resource_quiescent,
        observation_quiescent: snapshot.observation_quiescent,
        active_gpu_permits: snapshot.active_gpu_permits,
        active_observation_permits: snapshot.active_observation_permits,
        cleanup_failed: snapshot.cleanup_failed,
        cleanup_deadline_elapsed_count: snapshot.cleanup_deadline_elapsed_count,
        cleanup_late_success_count: snapshot.cleanup_late_success_count,
    }
}

fn device_configurations(backend: &NexusGpuBackend) -> Vec<DeviceConfiguration> {
    backend
        .device_startup_records()
        .iter()
        .map(|record| DeviceConfiguration {
            ordinal: record.device_ordinal,
            model_name: record.model_name.clone(),
            uuid: record.uuid.clone(),
            physical_total_bytes: record.physical_total_bytes,
            capacity_bytes: record.capacity_bytes,
            cache_cap_bytes: record.resource_profile.cache_cap_bytes,
            install_telemetry: record.install_telemetry,
            max_active_attempts: record.max_active_queries,
            stream_pool_size: record.stream_pool_size,
            worker_count: record.worker_count,
            lane_count: record.lane_count,
            ownership: format!("{:?}", record.memory_ownership),
            device_health_generation: record.device_health_generation,
            // A device that enumerated a startup record is admissible at t0;
            // P6.6 owns making one become ineligible mid-run.
            starting_health_open: true,
        })
        .collect()
}

/// Polls the public snapshot accessors on the window-aware interval and folds
/// them into per-device peaks plus the W1 compact time series.
struct Sampler {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: std::thread::JoinHandle<SamplerResult>,
}

struct SamplerResult {
    samples: Vec<BackendSnapshotSample>,
    occupancy: BTreeMap<usize, DeviceOccupancyPeaks>,
    metric_snapshots: Vec<MetricSnapshotObservation>,
    raw_snapshot_retention_truncated: bool,
    device_series: Vec<DeviceSeriesRow>,
    process_series: Vec<ProcessSeriesRow>,
    sample_interval: Duration,
}

impl Sampler {
    fn start(backend: &NexusGpuBackend, window: Duration) -> Self {
        let sample_interval = sample_interval_for(window);
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let started = Instant::now();
        // Capture the baseline before the caller can submit work. Spawning a
        // sampler thread and hoping its first poll wins that race makes the
        // resulting CUDA-free delta unsuitable for claim-versus-VRAM
        // calibration.
        let mut initial = SamplerResult::empty(sample_interval);
        sample_once(
            backend,
            started,
            &mut initial.samples,
            &mut initial.occupancy,
            &mut initial.metric_snapshots,
            &mut initial.raw_snapshot_retention_truncated,
            &mut initial.device_series,
            &mut initial.process_series,
            false,
        );
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let mut samples = initial.samples;
            let mut peaks = initial.occupancy;
            let mut metric_snapshots = initial.metric_snapshots;
            let mut raw_snapshot_retention_truncated = initial.raw_snapshot_retention_truncated;
            let mut device_series = initial.device_series;
            let mut process_series = initial.process_series;
            loop {
                // Waiting on the condvar rather than sleeping keeps `stop`
                // prompt, so the last sample lands right after the run ends.
                let (lock, condition) = &*signal;
                let Ok(stopped) = lock.lock() else {
                    break;
                };
                let Ok((stopped, _)) = condition.wait_timeout(stopped, sample_interval) else {
                    break;
                };
                if *stopped {
                    // One final observation after the drive loop finished.
                    drop(stopped);
                    sample_once(
                        &backend,
                        started,
                        &mut samples,
                        &mut peaks,
                        &mut metric_snapshots,
                        &mut raw_snapshot_retention_truncated,
                        &mut device_series,
                        &mut process_series,
                        true,
                    );
                    break;
                }
                sample_once(
                    &backend,
                    started,
                    &mut samples,
                    &mut peaks,
                    &mut metric_snapshots,
                    &mut raw_snapshot_retention_truncated,
                    &mut device_series,
                    &mut process_series,
                    false,
                );
            }
            for entry in peaks.values_mut() {
                if entry.min_root_streams_available == usize::MAX {
                    entry.min_root_streams_available = 0;
                }
                if entry.min_root_worker_slots_available == usize::MAX {
                    entry.min_root_worker_slots_available = 0;
                }
            }
            SamplerResult {
                samples,
                occupancy: peaks,
                metric_snapshots,
                raw_snapshot_retention_truncated,
                device_series,
                process_series,
                sample_interval,
            }
        });
        Self { stop, handle }
    }

    fn stop(self) -> SamplerResult {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        self.handle.join().unwrap_or_else(|_| SamplerResult::empty(SAMPLE_INTERVAL))
    }
}

impl SamplerResult {
    fn empty(sample_interval: Duration) -> Self {
        Self {
            samples: Vec::new(),
            occupancy: BTreeMap::new(),
            metric_snapshots: Vec::new(),
            raw_snapshot_retention_truncated: false,
            device_series: Vec::new(),
            process_series: Vec::new(),
            sample_interval,
        }
    }
}

/// The ledger-owned portion of one I11 observation. Equality across the two
/// reads bracketing telemetry proves that the physical sample is not paired
/// with a claim from a different ledger transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CalibrationClaimSnapshot {
    sequence: u64,
    live_grants_bytes: u64,
    cache_evictable_resident_bytes: u64,
    cache_pinned_in_use_bytes: u64,
}

impl CalibrationClaimSnapshot {
    fn from_report(report: &nexus_query_engine::runtime::DeviceLedgerReport) -> Self {
        Self {
            sequence: report.sequence,
            live_grants_bytes: report.live_grants_bytes,
            cache_evictable_resident_bytes: report.cache_evictable_resident_bytes,
            cache_pinned_in_use_bytes: report.cache_pinned_in_use_bytes,
        }
    }
}

fn calibration_claim_stable_across_telemetry(
    before: CalibrationClaimSnapshot,
    after: Option<CalibrationClaimSnapshot>,
) -> bool {
    after == Some(before)
}

/// One polling pass: folds the peaks, retains a bounded raw sample, and
/// appends the compact W1 series rows. Raw retention may displace its tail
/// for the terminal sample; the compact series instead keeps the whole run,
/// since whole-window drift coverage is its purpose and its rows stay small.
#[allow(clippy::too_many_arguments)]
fn sample_once(
    backend: &NexusGpuBackend,
    started: Instant,
    samples: &mut Vec<BackendSnapshotSample>,
    peaks: &mut BTreeMap<usize, DeviceOccupancyPeaks>,
    metric_snapshots: &mut Vec<MetricSnapshotObservation>,
    raw_snapshot_retention_truncated: &mut bool,
    device_series: &mut Vec<DeviceSeriesRow>,
    process_series: &mut Vec<ProcessSeriesRow>,
    retain_terminal_sample: bool,
) {
    let observation_started = Instant::now();
    // I11 calibrates a logical claim against device telemetry. The public
    // accessors are individually coherent but not one global transaction, so
    // bracket telemetry with two ledger reads. A row whose ledger sequence
    // changed cannot supply the "same-sample" claim and is retained only for
    // non-calibration utilization/drift evidence.
    let ledgers = backend.device_ledger_reports().unwrap_or_default();
    let telemetry = backend.device_telemetry_snapshots().unwrap_or_default();
    let ledgers_after_telemetry = backend.device_ledger_reports().unwrap_or_default();
    let cache_reports = backend.device_cache_reports().unwrap_or_default();
    let pools = backend.gpu_execution_pool_reports().unwrap_or_default();
    let failures = backend.device_memory_failure_reports().unwrap_or_default();
    let cugraph_bfs_singleflight = cugraph_bfs_singleflight_snapshot(backend);
    let admission = backend.admission_snapshot().ok();
    let backend_metrics = backend.backend_metrics_snapshot();
    let observation_pull_ns = observation_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    for ledger in &ledgers {
        let entry = peaks.entry(ledger.device_ordinal).or_insert_with(|| DeviceOccupancyPeaks {
            min_root_streams_available: usize::MAX,
            min_root_worker_slots_available: usize::MAX,
            ..DeviceOccupancyPeaks::default()
        });
        entry.peak_active_attempts = entry.peak_active_attempts.max(ledger.active_attempts);
        entry.peak_live_grants_bytes = entry.peak_live_grants_bytes.max(ledger.live_grants_bytes);
        entry.peak_live_reserved_bytes = entry.peak_live_reserved_bytes.max(ledger.live_reserved_bytes);
        entry.min_root_streams_available = entry.min_root_streams_available.min(ledger.root_streams_available);
        entry.min_root_worker_slots_available =
            entry.min_root_worker_slots_available.min(ledger.root_worker_slots_available);
    }
    for pool in &pools {
        let entry = peaks.entry(pool.device_ordinal).or_default();
        entry.peak_active_query_jobs = entry.peak_active_query_jobs.max(pool.peak_active_query_jobs);
    }
    for failure in &failures {
        let entry = peaks.entry(failure.device_ordinal).or_default();
        entry.allocation_failure_count = entry.allocation_failure_count.max(failure.allocation_failure_count);
    }
    // Raw snapshots and metric-audit DTOs have a bounded artifact retention
    // budget, but the compact W1 series must span the entire load window.
    // Keep sampling the same public observations after the raw budget fills.
    let retain_raw_snapshot =
        reserve_snapshot_slot(samples, metric_snapshots, raw_snapshot_retention_truncated, retain_terminal_sample);
    let offset_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    process_series.push(ProcessSeriesRow {
        offset_ns,
        active_queries: admission.as_ref().map_or(0, |snapshot| snapshot.active_queries),
        queued_queries: admission.as_ref().map_or(0, |snapshot| snapshot.queued_queries),
        backend_generation: backend_metrics.backend_generation,
        metric_generation: backend_metrics.generation,
        observation_pull_ns,
    });
    for ledger in &ledgers {
        let ledger_after_telemetry =
            ledgers_after_telemetry.iter().find(|after| after.device_ordinal == ledger.device_ordinal);
        let claim_snapshot_end_sequence = ledger_after_telemetry.map(|after| after.sequence);
        let claim_snapshot_stable = calibration_claim_stable_across_telemetry(
            CalibrationClaimSnapshot::from_report(ledger),
            ledger_after_telemetry.map(CalibrationClaimSnapshot::from_report),
        );
        let cache_report = cache_reports.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        let device_telemetry = telemetry.iter().find(|snapshot| snapshot.device_ordinal == ledger.device_ordinal);
        let pool = pools.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        let failure = failures.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        device_series.push(DeviceSeriesRow {
            offset_ns,
            device_ordinal: ledger.device_ordinal,
            health_open: ledger.health == "open",
            gpu_utilization_pct: device_telemetry.and_then(|snapshot| snapshot.gpu_utilization_pct),
            memory_utilization_pct: device_telemetry.and_then(|snapshot| snapshot.memory_utilization_pct),
            cuda_free_bytes: device_telemetry.and_then(|snapshot| snapshot.cuda_free_bytes),
            nvml_used_bytes: device_telemetry.and_then(|snapshot| snapshot.nvml_used_bytes),
            active_attempts: ledger.active_attempts,
            live_grants_bytes: ledger.live_grants_bytes,
            live_reserved_bytes: ledger.live_reserved_bytes,
            cache_evictable_resident_bytes: ledger.cache_evictable_resident_bytes,
            cache_pinned_in_use_bytes: ledger.cache_pinned_in_use_bytes,
            cache_physical_current_bytes: cache_report.map(|report| report.physical_current_bytes),
            cache_physical_peak_bytes: cache_report.map(|report| report.physical_peak_bytes),
            allocation_failure_count: failure.map_or(0, |report| report.allocation_failure_count),
            ledger_sequence: ledger.sequence,
            claim_snapshot_end_sequence,
            claim_snapshot_stable,
            active_query_jobs: pool.map_or(0, |report| report.active_query_jobs),
            queued_jobs: pool.map_or(0, |report| report.queued_jobs),
        });
    }
    if !retain_raw_snapshot {
        return;
    }
    metric_snapshots.push(metric_snapshot_observation(offset_ns, &backend_metrics));
    samples.push(BackendSnapshotSample {
        offset_ns,
        backend_metrics_snapshot: serde_json::to_value(backend_metrics).unwrap_or(Value::Null),
        admission: admission.and_then(|snapshot| serde_json::to_value(snapshot).ok()).unwrap_or(Value::Null),
        device_ledger_reports: serde_json::to_value(&ledgers).unwrap_or(Value::Null),
        device_cache_reports: serde_json::to_value(&cache_reports).unwrap_or(Value::Null),
        device_memory_failure_reports: serde_json::to_value(&failures).unwrap_or(Value::Null),
        gpu_execution_pool_reports: serde_json::to_value(&pools).unwrap_or(Value::Null),
        device_telemetry_snapshots: serde_json::to_value(&telemetry).unwrap_or(Value::Null),
        cugraph_bfs_singleflight_snapshot: cugraph_bfs_singleflight,
    });
}

#[cfg(feature = "single-node-multi-gpu-validate-run-cugraph")]
fn cugraph_bfs_singleflight_snapshot(backend: &NexusGpuBackend) -> Value {
    let snapshot = backend.cugraph_bfs_singleflight_snapshot();
    serde_json::json!({
        "eligible_requests": snapshot.eligible_requests,
        "physical_traversals": snapshot.physical_traversals,
        "follower_copies": snapshot.follower_copies,
        "groups_formed": snapshot.groups_formed,
        "overflow_groups": snapshot.overflow_groups,
        "leader_failures": snapshot.leader_failures,
        "cancelled_waiters": snapshot.cancelled_waiters,
        "peak_width": snapshot.peak_width,
        "events": snapshot.events.iter().map(|event| serde_json::json!({
            "sequence": event.sequence,
            "device_ordinal": event.device_ordinal,
            "singleflight_key_fingerprint": event.singleflight_key_fingerprint,
            "correlation_fingerprint": event.correlation_fingerprint,
            "generation": event.generation,
            "action": event.action,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(not(feature = "single-node-multi-gpu-validate-run-cugraph"))]
fn cugraph_bfs_singleflight_snapshot(_backend: &NexusGpuBackend) -> Value {
    Value::Null
}

/// Reserves one bounded raw/audit snapshot slot. The two vectors always move
/// together so an artifact sample and the corresponding P6.7 DTO cannot refer
/// to different P4 pulls.
fn reserve_snapshot_slot<T, U>(
    samples: &mut Vec<T>,
    metric_snapshots: &mut Vec<U>,
    raw_snapshot_retention_truncated: &mut bool,
    retain_terminal_sample: bool,
) -> bool {
    debug_assert_eq!(samples.len(), metric_snapshots.len());
    if samples.len() < MAX_SNAPSHOT_SAMPLES {
        return true;
    }
    *raw_snapshot_retention_truncated = true;
    if !retain_terminal_sample {
        return false;
    }
    // The bounded artifact cannot retain every intermediate snapshot, but P4
    // report equality must use the post-drain pull. Drop the former tail and
    // replace it with the terminal snapshot in lockstep with the audit DTO;
    // cadence remains explicitly not-evidenced after truncation.
    samples.pop();
    metric_snapshots.pop();
    true
}

/// Projects one public P4 snapshot into the pure P6.7 audit DTO. The raw
/// snapshot is retained separately in the artifact; this only records the
/// bounded checks needed to prove its cadence, cardinality, and exact terminal
/// count.
fn metric_snapshot_observation(offset_ns: u64, snapshot: &BackendMetricsSnapshot) -> MetricSnapshotObservation {
    let configured_device_ordinals =
        snapshot.devices.iter().flatten().map(|device| device.device_ordinal).collect::<Vec<_>>();
    let (projected_series_count, expected_series_count) =
        backend_metric_descriptors().iter().fold((Some(0_usize), 0_usize), |(projected, expected), descriptor| {
            let projected = projected.and_then(|count| {
                snapshot.project_backend_metric(descriptor).ok().and_then(|samples| count.checked_add(samples.len()))
            });
            let expected = expected
                .saturating_add(expected_descriptor_series(descriptor.metric, configured_device_ordinals.len()));
            (projected, expected)
        });
    let terminal_completed = BackendAttemptDisposition::ALL
        .into_iter()
        .flat_map(|disposition| {
            BackendAttemptOutcome::ALL.into_iter().map(move |outcome| snapshot.attempts.completed(disposition, outcome))
        })
        .sum();
    let mut total_duration_bucket_counts =
        vec![0_u64; nexus_query_engine::runtime::BACKEND_ATTEMPT_DURATION_BUCKET_COUNT];
    let mut total_duration_count = 0_u64;
    for disposition in BackendAttemptDisposition::ALL {
        for outcome in BackendAttemptOutcome::ALL {
            let histogram = snapshot.attempts.duration(disposition, outcome, BackendAttemptDuration::Total);
            accumulate_histogram(&mut total_duration_bucket_counts, &mut total_duration_count, histogram);
        }
    }
    MetricSnapshotObservation {
        offset_ns,
        backend_generation: snapshot.backend_generation,
        generation: snapshot.generation,
        configured_device_ordinals,
        projected_series_count,
        expected_series_count,
        terminal_completed,
        total_duration_bucket_counts,
        total_duration_count,
    }
}

fn expected_descriptor_series(metric: BackendMetric, configured_device_count: usize) -> usize {
    match metric {
        BackendMetric::attempts_completed => BackendAttemptDisposition::COUNT * BackendAttemptOutcome::COUNT,
        BackendMetric::attempts_duration => {
            BackendAttemptDisposition::COUNT * BackendAttemptOutcome::COUNT * BackendAttemptDuration::COUNT
        },
        _ => configured_device_count,
    }
}

fn accumulate_histogram(total_bucket_counts: &mut [u64], total_count: &mut u64, histogram: BackendHistogramSnapshot) {
    for (total, value) in total_bucket_counts.iter_mut().zip(histogram.bucket_counts) {
        *total = total.saturating_add(value);
    }
    *total_count = total_count.saturating_add(histogram.count);
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
        let attempt_id = nexus_query_engine::observability::AttemptId::mint();
        let mut hasher = nexus_query_engine::harness::StableFnv1a64::new();
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
            RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s2-sweep-scan-sf10.json").as_bytes())
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
        let profile =
            RunProfile::parse_bytes(include_str!("../../fixtures/stress/stress-s3-heterogeneous-sf1.json").as_bytes())
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
