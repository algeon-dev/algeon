//! The GPU-touching measured run.
//!
//! This module is deliberately thin: it builds the backend from an already
//! validated profile, drives the already planned arrivals, samples the public
//! snapshot accessors, and projects reports into the pure reduction in
//! [`super::aggregate`]. Every decision that can be made without a GPU is made
//! elsewhere.

use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    native_exec::NexusNativeExecutionConfig,
};
use futures::StreamExt as _;
use nexus_query_engine::observability::{AttemptDisposition, QueryAttemptReport};
use serde_json::Value;

use super::{
    aggregate::{
        AttemptFailureClass, AttemptObservation, DeviceConfiguration, DeviceOccupancyPeaks, MeasuredAggregate,
        scenario_clauses_pass,
    },
    artifact::{ArrivalPlanEvidence, BackendEvidence, BackendSnapshotSample, MeasuredRunArtifact, TraceEvidence},
    profile::{CacheMode, DeviceOwnership, RateOverrunPolicy, RunProfile},
    schedule::{ArrivalPlan, SchedulePhase},
    workload::{SelectedQuery, select_queries},
};
use crate::ToolResult;

/// How often the sampler polls the backend snapshot accessors.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
/// Bound on retained raw snapshot samples. Each carries the full per-device
/// report vectors, so this caps artifact size and peak memory; the per-device
/// peaks the scenario clauses use keep folding past the cap.
const MAX_SNAPSHOT_SAMPLES: usize = 512;
/// Grace period for the backend close that ends the run.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);
/// How late a planned arrival may be before rate mode treats it as an overrun.
const RATE_OVERRUN_TOLERANCE: Duration = Duration::from_millis(1);

pub(super) struct TraceSelection {
    pub(super) uri: Option<String>,
    pub(super) sha256: Option<String>,
}

/// Runs the measurement on a dedicated multi-thread runtime.
///
/// The tool entrypoint pins a current-thread runtime, which would interleave
/// rather than overlap attempts. A scoped thread owns a multi-thread runtime so
/// the measured concurrency is real, without building a nested runtime inside
/// the caller's and without changing every other subcommand's entrypoint.
pub(super) fn run_on_owned_runtime(profile: &RunProfile, trace: TraceSelection) -> ToolResult<MeasuredRunArtifact> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
                runtime.block_on(run_measured(profile, trace))
            })
            .join()
            .map_err(|_| "the measured-run thread panicked")?
    })
}

/// Builds the backend, executes the plan, and returns the measured artifact.
async fn run_measured(profile: &RunProfile, trace: TraceSelection) -> ToolResult<MeasuredRunArtifact> {
    let queries = select_queries(&profile.workloads)?;
    let plan = ArrivalPlan::plan(&profile.scheduling, queries.len(), profile.session.shared_session_count)?;
    let backend = build_backend(profile)?;
    let sessions = install_sessions(&backend, profile, &queries).await?;

    let sampler = Sampler::start(&backend);
    let outcome = drive(&backend, &sessions, &queries, &plan, profile.scheduling.rate_overrun).await;
    let (samples, occupancy) = sampler.stop();
    // Close before the artifact is built: the close snapshot is part of the
    // envelope evidence, and a failed run must still record it.
    let close = backend.close_and_wait(CLOSE_TIMEOUT).ok();
    let driven = outcome?;

    let devices = device_configurations(&backend);
    let aggregate =
        MeasuredAggregate::reduce(&driven.observations, &devices, &occupancy, plan.warmup_ns, plan.measurement_ns);
    let qualifying = scenario_clauses_pass(&aggregate.scenarios) && aggregate.report_aggregate_equality.equal;
    let identity = backend.identity();
    Ok(MeasuredRunArtifact {
        qualifying,
        backend: BackendEvidence {
            resource_core: identity.resource_core,
            observation_core: identity.observation_core,
            resource_generation: identity.resource_generation,
            shared_session_count: sessions.len(),
            queue_max_queued_attempts: profile.backend.queue.max_queued_attempts,
            queue_max_queue_wait_ms: profile.backend.queue.max_queue_wait_ms,
            queue_max_overtakes_per_waiter: profile.backend.queue.max_overtakes_per_waiter,
            // `GpuBackendDeviceStartupRecord` is not one of the nine
            // `Serialize` snapshot types and transitively holds a cuDF type, so
            // the artifact records the same startup facts through the
            // aggregate's own device configuration rather than widening that
            // frozen JSON contract from here.
            device_startup_records: serde_json::to_value(&devices)?,
            close: close.map(serde_json::to_value).transpose()?.unwrap_or(Value::Null),
        },
        selected_queries: queries.iter().map(SelectedQuery::descriptor).collect(),
        arrival_plan: ArrivalPlanEvidence::new(
            &plan,
            driven.dropped_overrun_arrivals,
            driven.begin_failures,
            driven.join_failures,
        ),
        cache_state: match profile.cache.mode {
            CacheMode::Disabled => "disabled",
            CacheMode::Enabled => "enabled",
        },
        aggregate,
        snapshots: samples,
        trace: TraceEvidence { captured_by_runner: false, uri: trace.uri, sha256: trace.sha256 },
    })
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

fn build_backend(profile: &RunProfile) -> ToolResult<NexusGpuBackend> {
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
        .queue_policy(queue)
        .device_profiles(profile.backend.device_profiles.iter().map(|device| {
            NexusGpuDeviceProfile::new(device.ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: device.memory_limit_bytes })
                .with_backend_reserve_bytes(device.backend_reserve_bytes)
                .with_cache_cap_bytes(device.cache_cap_bytes)
                .with_attempt_floor_bytes(Some(device.attempt_floor_bytes))
                .with_source_work_per_attempt(device.source_work_per_attempt)
                .with_max_active_attempts(device.max_active_attempts)
                .with_installation_bounds(
                    device.installation.stream_pool_size,
                    device.installation.worker_count,
                    device.installation.lane_count,
                )
                .with_telemetry(true)
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

/// One `SessionContext` per shared session, each with every selected fixture
/// registered. Sessions are shared across attempts by design.
async fn install_sessions(
    backend: &NexusGpuBackend,
    profile: &RunProfile,
    queries: &[SelectedQuery],
) -> ToolResult<Vec<SessionContext>> {
    let needs_graph = queries.iter().any(|query| query.kind == super::profile::FixtureKind::Dblp);
    let mut sessions = Vec::with_capacity(profile.session.shared_session_count);
    for _ in 0..profile.session.shared_session_count {
        let builder = SessionStateBuilder::new_with_default_features();
        let builder = if needs_graph {
            backend.install_on_with_cugraph_sql(builder, datafusion_nexus::cugraph_sql::CugraphSqlConfig::default())?
        } else {
            backend.install_on(builder)?
        };
        let ctx = SessionContext::new_with_state(builder.build());
        for workload in &profile.workloads {
            if workload.queries.is_empty() {
                continue;
            }
            match workload.kind {
                super::profile::FixtureKind::Tpch => {
                    nexus_workloads::tpch::register_tpch_tables(&ctx, &workload.path).await?;
                },
                super::profile::FixtureKind::Dblp => {
                    nexus_workloads::graph::register_dblp_tables(&ctx, &workload.path).await?;
                },
            }
        }
        sessions.push(ctx);
    }
    Ok(sessions)
}

struct DrivenRun {
    observations: Vec<AttemptObservation>,
    dropped_overrun_arrivals: u64,
    begin_failures: u64,
    join_failures: u64,
}

/// Submits every planned arrival at its offset and collects one bounded report
/// per attempt.
async fn drive(
    backend: &NexusGpuBackend,
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    rate_overrun: RateOverrunPolicy,
) -> ToolResult<DrivenRun> {
    let started = Instant::now();
    let closed_loop = plan.arrival_mode == "concurrency";
    let warmup_deadline = started + Duration::from_nanos(plan.warmup_ns);
    let run_deadline = started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns);

    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut dropped_overrun_arrivals = 0_u64;
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut current_round = 0;

    for arrival in &plan.arrivals {
        if closed_loop {
            // The measurement window, not the plan length, ends a closed-loop
            // run: rounds are planned generously and pacing is observed.
            if Instant::now() >= run_deadline {
                break;
            }
            // Drain the previous round so `concurrency` bounds in-flight
            // attempts instead of becoming an offered arrival rate.
            if arrival.round != current_round {
                drain(&mut tasks, &mut observations, &mut join_failures).await;
                current_round = arrival.round;
            }
        } else {
            let planned = started + Duration::from_nanos(arrival.offset_ns);
            match Instant::now().checked_duration_since(planned) {
                // Late by more than the tolerance: rate mode may abandon it
                // rather than convert the offered rate into a closed loop.
                Some(late) if late > RATE_OVERRUN_TOLERANCE => {
                    if rate_overrun == RateOverrunPolicy::Drop {
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
        let Ok(attempt) = backend.begin_attempt(AttemptOptions::default()) else {
            begin_failures += 1;
            break;
        };
        let ctx = sessions[arrival.session_index].clone();
        let query = queries[arrival.query_index].clone();
        let mut arrival = *arrival;
        if closed_loop {
            // Closed-loop arrivals carry no planned offset, so the window an
            // attempt belongs to is decided when it is submitted. Judging it at
            // collection time would misattribute a warmup attempt whose round
            // drained after the deadline.
            arrival.phase =
                if Instant::now() < warmup_deadline { SchedulePhase::Warmup } else { SchedulePhase::Measurement };
        }
        tasks.push(tokio::spawn(async move {
            // A failing statement is a measured outcome, not a runner error:
            // the sealed report still describes what happened.
            if let Ok(stream) = attempt.sql(&ctx, &query.sql).await {
                let mut stream = stream;
                while let Some(batch) = stream.next().await {
                    if batch.is_err() {
                        break;
                    }
                }
            }
            let report = attempt.completion().await;
            (arrival, query, report)
        }));
    }

    drain(&mut tasks, &mut observations, &mut join_failures).await;
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun { observations, dropped_overrun_arrivals, begin_failures, join_failures })
}

type AttemptTask = tokio::task::JoinHandle<(super::schedule::ScheduledArrival, SelectedQuery, Arc<QueryAttemptReport>)>;

/// Awaits every in-flight attempt and folds its report into an observation.
async fn drain(tasks: &mut Vec<AttemptTask>, observations: &mut Vec<AttemptObservation>, join_failures: &mut u64) {
    for task in tasks.drain(..) {
        // A panicked or cancelled task loses only its own row. Aborting here
        // would discard a whole measured run's evidence.
        match task.await {
            Ok((arrival, query, report)) => {
                observations.push(observe(arrival.sequence, arrival.phase, &query, &report));
            },
            Err(_) => *join_failures += 1,
        }
    }
}

fn observe(
    sequence: usize,
    phase: SchedulePhase,
    query: &SelectedQuery,
    report: &QueryAttemptReport,
) -> AttemptObservation {
    let resource_evidence = report.resource_evidence.as_ref();
    AttemptObservation {
        sequence,
        phase,
        query_label: query.label.clone(),
        graph_workload: query.graph,
        attempt_id: report.attempt_id.to_string(),
        disposition: format!("{:?}", report.disposition),
        outcome: report.outcome.stable_label().to_string(),
        failure: report.primary_failure.as_ref().map(|failure| AttemptFailureClass {
            origin: format!("{:?}", failure.origin()),
            code: format!("{:?}", failure.code()),
            status: format!("{:?}", failure.status()),
            operation: format!("{:?}", failure.operation()),
        }),
        succeeded: report.outcome.stable_label() == super::aggregate::SUCCEEDED_OUTCOME,
        reached_gpu: matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
        device_ordinal: report.admission.as_ref().map(|admission| admission.device_ordinal).or(report.device_ordinal),
        grant_bytes: report.admission.as_ref().map(|admission| admission.grant_bytes),
        admission_sequence: report.admission.as_ref().map(|admission| admission.admission_sequence),
        admission_wait_ns: report.admission.as_ref().map(|admission| admission.admission_wait_ns),
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

/// Polls the public snapshot accessors on a fixed interval and folds them into
/// per-device peaks.
struct Sampler {
    stop: Arc<(Mutex<bool>, Condvar)>,
    handle: std::thread::JoinHandle<(Vec<BackendSnapshotSample>, BTreeMap<usize, DeviceOccupancyPeaks>)>,
}

impl Sampler {
    fn start(backend: &NexusGpuBackend) -> Self {
        let backend = backend.clone();
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let started = Instant::now();
        let handle = std::thread::spawn(move || {
            let mut samples = Vec::new();
            let mut peaks: BTreeMap<usize, DeviceOccupancyPeaks> = BTreeMap::new();
            loop {
                sample_once(&backend, started, &mut samples, &mut peaks);
                // Waiting on the condvar rather than sleeping keeps `stop`
                // prompt, so the last sample lands right after the run ends.
                let (lock, condition) = &*signal;
                let Ok(stopped) = lock.lock() else {
                    break;
                };
                let Ok((stopped, _)) = condition.wait_timeout(stopped, SAMPLE_INTERVAL) else {
                    break;
                };
                if *stopped {
                    // One final observation after the drive loop finished.
                    drop(stopped);
                    sample_once(&backend, started, &mut samples, &mut peaks);
                    break;
                }
            }
            for entry in peaks.values_mut() {
                if entry.min_root_streams_available == usize::MAX {
                    entry.min_root_streams_available = 0;
                }
                if entry.min_root_worker_slots_available == usize::MAX {
                    entry.min_root_worker_slots_available = 0;
                }
            }
            (samples, peaks)
        });
        Self { stop, handle }
    }

    fn stop(self) -> (Vec<BackendSnapshotSample>, BTreeMap<usize, DeviceOccupancyPeaks>) {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        self.handle.join().unwrap_or_else(|_| (Vec::new(), BTreeMap::new()))
    }
}

/// One polling pass: folds the peaks and retains a bounded raw sample.
fn sample_once(
    backend: &NexusGpuBackend,
    started: Instant,
    samples: &mut Vec<BackendSnapshotSample>,
    peaks: &mut BTreeMap<usize, DeviceOccupancyPeaks>,
) {
    let ledgers = backend.device_ledger_reports().unwrap_or_default();
    let pools = backend.gpu_execution_pool_reports().unwrap_or_default();
    let failures = backend.device_memory_failure_reports().unwrap_or_default();
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
    if samples.len() >= MAX_SNAPSHOT_SAMPLES {
        return;
    }
    samples.push(BackendSnapshotSample {
        offset_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        admission: backend
            .admission_snapshot()
            .ok()
            .and_then(|snapshot| serde_json::to_value(snapshot).ok())
            .unwrap_or(Value::Null),
        device_ledger_reports: serde_json::to_value(&ledgers).unwrap_or(Value::Null),
        device_memory_failure_reports: serde_json::to_value(&failures).unwrap_or(Value::Null),
        gpu_execution_pool_reports: serde_json::to_value(&pools).unwrap_or(Value::Null),
        device_telemetry_snapshots: backend
            .device_telemetry_snapshots()
            .ok()
            .and_then(|snapshots| serde_json::to_value(snapshots).ok())
            .unwrap_or(Value::Null),
    });
}
