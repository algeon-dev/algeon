//! The GPU-touching measured run.
//!
//! This module is deliberately thin: it builds the backend from an already
//! validated profile, drives the already planned arrivals, samples the public
//! snapshot accessors, and projects reports into the pure reduction in
//! [`super::aggregate`]. Every decision that can be made without a GPU is made
//! elsewhere.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    native_exec::NexusNativeExecutionConfig,
};
use futures::StreamExt as _;
use nexus_query_engine::{
    metrics::{BackendMetric, backend_metric_descriptors},
    observability::{AttemptDisposition, QueryAttemptReport},
    runtime::{
        BackendAttemptDisposition, BackendAttemptDuration, BackendAttemptOutcome, BackendHistogramSnapshot,
        BackendMetricsSnapshot,
    },
};
use serde_json::Value;

use super::{
    aggregate::{
        AttemptFailureClass, AttemptObservation, AttemptTerminalEvidence, DeviceConfiguration, DeviceOccupancyPeaks,
        DomainExhaustionEvidence, MeasuredAggregate, OverloadFairnessEvidence, PhysicalAllocatorPressureEvidence,
        PhysicalPressureLeaseFacts, QuarantineFallbackEvidence, scenario_clauses_pass,
    },
    artifact::{
        ArrivalPlanEvidence, BackendCloseEvidence, BackendCloseFailure, BackendCloseOutcome,
        BackendCloseSnapshotEvidence, BackendEvidence, BackendSnapshotSample, CancellationBarrierEvidence,
        DeadlineBarrierEvidence, MeasuredRunArtifact, RunDeadlineEvidence, TraceEvidence,
    },
    metrics_audit::{MetricSnapshotObservation, MetricsAudit},
    profile::{CacheMode, DeviceOwnership, RateOverrunPolicy, RunProfile},
    schedule::{ArrivalPlan, SchedulePhase},
    workload::{SelectedQuery, select_queries},
};
use crate::ToolResult;

/// How often the sampler polls the backend snapshot accessors.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);
/// Bound on retained raw snapshot samples. Each carries the full per-device
/// report vectors, so this caps artifact size and peak memory; the per-device
/// peaks the scenario clauses use keep folding past the cap. A final sample
/// always displaces the retained tail when this bound is reached, so terminal
/// P4 counters are never audited against a pre-drain snapshot.
const MAX_SNAPSHOT_SAMPLES: usize = 1024;
/// Grace period for the backend close that ends the run.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);
/// How late a planned arrival may be before rate mode treats it as an overrun.
const RATE_OVERRUN_TOLERANCE: Duration = Duration::from_millis(1);

#[derive(Clone)]
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
pub(super) fn run_on_owned_runtime(
    profile: &RunProfile,
    trace: TraceSelection,
    quarantine_device: Option<usize>,
    cancel_first_after: Option<Duration>,
    deadline_first_after: Option<Duration>,
    domain_exhaustion_probe: bool,
    overload_fairness: bool,
    physical_allocator_pressure: bool,
) -> ToolResult<MeasuredRunArtifact> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
                runtime.block_on(run_measured(
                    profile,
                    trace,
                    quarantine_device,
                    cancel_first_after,
                    deadline_first_after,
                    domain_exhaustion_probe,
                    overload_fairness,
                    physical_allocator_pressure,
                ))
            })
            .join()
            .map_err(|_| "the measured-run thread panicked")?
    })
}

/// Builds the backend, executes the plan, and returns the measured artifact.
async fn run_measured(
    profile: &RunProfile,
    trace: TraceSelection,
    quarantine_device: Option<usize>,
    cancel_first_after: Option<Duration>,
    deadline_first_after: Option<Duration>,
    domain_exhaustion_probe: bool,
    overload_fairness: bool,
    physical_allocator_pressure: bool,
) -> ToolResult<MeasuredRunArtifact> {
    let queries = select_queries(&profile.workloads)?;
    let plan = ArrivalPlan::plan(&profile.scheduling, queries.len(), profile.session.shared_session_count)?;
    let backend = build_backend(profile)?;
    let sessions = install_sessions(&backend, profile, &queries).await?;

    let sampler = Sampler::start(&backend);
    let outcome = match (quarantine_device, deadline_first_after, domain_exhaustion_probe, physical_allocator_pressure)
    {
        (Some(target_ordinal), None, false, false) => {
            drive_concurrent_quarantine(
                &backend,
                &sessions,
                &queries,
                &plan,
                profile.scheduling.rate_overrun,
                cancel_first_after,
                target_ordinal,
            )
            .await
        },
        (None, Some(deadline_after), false, false) => {
            drive_deadline_after_admission(
                &backend,
                &sessions,
                &queries,
                &plan,
                profile.scheduling.rate_overrun,
                deadline_after,
            )
            .await
        },
        (None, None, true, false) => {
            drive_domain_exhaustion_probe(&backend, &sessions, &queries, &plan, profile.scheduling.rate_overrun).await
        },
        (None, None, false, true) => {
            drive_physical_allocator_pressure(
                &backend,
                &sessions,
                &queries,
                &plan,
                profile.scheduling.rate_overrun,
                profile.backend.device_profiles[0].ordinal,
            )
            .await
        },
        (None, None, false, false) => {
            drive(&backend, &sessions, &queries, &plan, profile.scheduling.rate_overrun, cancel_first_after).await
        },
        _ => unreachable!("CLI validation rejects combined P6.6 scenario controls"),
    };
    let sampler = sampler.stop();
    let driven = outcome?;
    // Close after every submitted completion has been observed, so the
    // attempt rows remain canonical. The evidence retains a timeout or cleanup
    // failure instead of erasing it with `.ok()`.
    let close = close_backend(&backend);

    let devices = device_configurations(&backend);
    let quarantine_fallback = driven.quarantine_barrier.as_ref().map(|barrier| {
        QuarantineFallbackEvidence::from_barrier(
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
            &driven.observations,
        )
    });
    let aggregate = MeasuredAggregate::reduce_with_quarantine(
        &driven.observations,
        &devices,
        &sampler.occupancy,
        plan.warmup_ns,
        plan.measurement_ns,
        quarantine_fallback,
    );
    let overload_fairness_evidence = overload_fairness.then(|| {
        OverloadFairnessEvidence::from_observations(
            &driven.observations,
            &devices,
            plan.arrivals.len(),
            profile.backend.queue.max_queued_attempts,
            profile.backend.queue.max_overtakes_per_waiter,
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
    let metrics_audit = MetricsAudit::from_snapshots(
        &sampler.metric_snapshots,
        &devices.iter().map(|device| device.ordinal).collect::<Vec<_>>(),
        aggregate.attempts.len(),
        SAMPLE_INTERVAL.as_millis().min(u128::from(u64::MAX)) as u64,
        sampler.raw_snapshot_retention_truncated,
    );
    let qualifying = selected_scenario_clauses_pass(
        &aggregate.scenarios,
        deadline_first_after.is_some()
            || domain_exhaustion_probe
            || overload_fairness
            || physical_allocator_pressure.is_some(),
    ) && aggregate.report_aggregate_equality.equal
        && metrics_audit.passes()
        && driven.deadline.reports_sealed_within_deadline
        && driven.cancellation_barrier.as_ref().is_none_or(CancellationBarrierEvidence::is_evidenced)
        && driven.deadline_barrier.as_ref().is_none_or(DeadlineBarrierEvidence::is_evidenced)
        && domain_exhaustion.as_ref().is_none_or(DomainExhaustionEvidence::is_evidenced)
        && overload_fairness_evidence.as_ref().is_none_or(OverloadFairnessEvidence::is_evidenced)
        && physical_allocator_pressure.as_ref().is_none_or(PhysicalAllocatorPressureEvidence::is_evidenced)
        && close.is_quiescent();
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
            close,
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
        deadline: driven.deadline,
        cancellation_barrier: driven.cancellation_barrier,
        deadline_barrier: driven.deadline_barrier,
        domain_exhaustion,
        physical_allocator_pressure,
        metrics_audit,
        snapshots: sampler.samples,
        trace: TraceEvidence { captured_by_runner: false, uri: trace.uri, sha256: trace.sha256 },
    })
}

/// Runs the serial, public-SQL P6.6 domain-cap probe. The profile preflight
/// pins the two-attempt shape (`q01` followed by `edge_scan`) and the single
/// static partition. This driver deliberately reuses ordinary submission and
/// drain logic: no runner-created error substitutes for allocator evidence.
async fn drive_domain_exhaustion_probe(
    backend: &NexusGpuBackend,
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    rate_overrun: RateOverrunPolicy,
) -> ToolResult<DrivenRun> {
    if plan.arrivals.len() != 2 {
        return Err("--domain-exhaustion-probe requires exactly two planned attempts".into());
    }
    drive(backend, sessions, queries, plan, rate_overrun, None).await
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
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    _rate_overrun: RateOverrunPolicy,
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

    let started = Instant::now();
    let warmup_deadline = started + Duration::from_nanos(plan.warmup_ns);
    let run_deadline = started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns);
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
    ) {
        Ok(task) => tasks.push(task),
        Err(()) => begin_failures += 1,
    }
    drain(&mut tasks, &mut observations, &mut join_failures).await;
    let elapsed_until_reports_ns = elapsed_ns(started);
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals: 0,
        begin_failures,
        join_failures,
        deadline: run_deadline_evidence(plan.warmup_ns.saturating_add(plan.measurement_ns), elapsed_until_reports_ns),
        cancellation_barrier: None,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: Some(PhysicalAllocatorPressureBarrier {
            pressure,
            release,
            pressure_established_before_q01: true,
            pressure_released_before_recovery: true,
        }),
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
) -> bool {
    if dedicated_p6_6_scenario { true } else { scenario_clauses_pass(scenarios) }
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
    deadline: RunDeadlineEvidence,
    cancellation_barrier: Option<CancellationBarrierEvidence>,
    quarantine_barrier: Option<QuarantineBarrier>,
    deadline_barrier: Option<DeadlineBarrierEvidence>,
    physical_allocator_pressure: Option<PhysicalAllocatorPressureBarrier>,
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
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    rate_overrun: RateOverrunPolicy,
    cancel_first_after: Option<Duration>,
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
    let mut cancellation_barrier = None;

    for arrival in &plan.arrivals {
        // The run window bounds submission in both scheduling modes.  In
        // particular, rate-mode `delay` must not turn late work into a second
        // unbounded measurement window.
        if Instant::now() >= run_deadline {
            break;
        }
        if closed_loop {
            // The measurement window, not the plan length, ends a closed-loop
            // run: rounds are planned generously and pacing is observed.
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
        let Ok(attempt) = backend.begin_attempt(attempt_options(run_deadline)) else {
            begin_failures += 1;
            break;
        };
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
            cancellation_barrier = Some(PendingCancellationBarrier {
                attempt_sequence: arrival.sequence,
                requested_after,
                requested,
                task,
            });
        }
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
    let elapsed_until_reports_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let cancellation_barrier = if let Some(mut barrier) = cancellation_barrier {
        if let Some(task) = barrier.task.take() {
            // The canceller has no resource ownership; joining it only makes
            // the requested action's timestamp/result deterministic.
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
    } else {
        None
    };
    observations.sort_by_key(|observation| observation.sequence);
    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals,
        begin_failures,
        join_failures,
        deadline: run_deadline_evidence(plan.warmup_ns.saturating_add(plan.measurement_ns), elapsed_until_reports_ns),
        cancellation_barrier,
        quarantine_barrier: None,
        deadline_barrier: None,
        physical_allocator_pressure: None,
    })
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
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    _rate_overrun: RateOverrunPolicy,
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

    let started = Instant::now();
    let warmup_deadline = started + Duration::from_nanos(plan.warmup_ns);
    let run_deadline = started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns);
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let dropped_overrun_arrivals = 0_u64;
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;

    for arrival in &plan.arrivals[..first_post_ack_index] {
        if Instant::now() >= run_deadline {
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

    let witness = wait_for_target_live_work(backend, target_ordinal, started, run_deadline).await?;
    let injection = backend.fault_injector().quarantine_device(target_ordinal)?;
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
        if Instant::now() >= run_deadline {
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
    let elapsed_until_reports_ns = elapsed_ns(started);
    let pre_existing_target_work_released = backend
        .device_ledger_reports()?
        .into_iter()
        .find(|report| report.device_ordinal == target_ordinal)
        .is_some_and(|report| report.active_attempts == 0 && report.grants.is_empty());
    let cancellation_barrier = finish_cancellation_barrier(cancellation_barrier, &observations).await;
    observations.sort_by_key(|observation| observation.sequence);

    Ok(DrivenRun {
        observations,
        dropped_overrun_arrivals,
        begin_failures,
        join_failures,
        deadline: run_deadline_evidence(plan.warmup_ns.saturating_add(plan.measurement_ns), elapsed_until_reports_ns),
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
        }),
        deadline_barrier: None,
        physical_allocator_pressure: None,
    })
}

/// Drives one attempt with a shorter caller deadline, but only accepts it as
/// P6.6 evidence after public snapshots first witness an immutable admission
/// envelope. The deadline remains the normal attempt contract: no runner task
/// is aborted and a CUDA call already in flight may finish before the next
/// native safe boundary observes expiry.
async fn drive_deadline_after_admission(
    backend: &NexusGpuBackend,
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    plan: &ArrivalPlan,
    _rate_overrun: RateOverrunPolicy,
    deadline_after: Duration,
) -> ToolResult<DrivenRun> {
    if plan.arrival_mode != "concurrency" {
        return Err("--deadline-first-after-ms requires concurrency scheduling".into());
    }
    if plan.arrivals.len() < 2 {
        return Err("--deadline-first-after-ms requires a follow-up attempt for healthy reuse evidence".into());
    }

    let started = Instant::now();
    let warmup_deadline = started + Duration::from_nanos(plan.warmup_ns);
    let run_deadline = started + Duration::from_nanos(plan.warmup_ns + plan.measurement_ns);
    let deadline = started.checked_add(deadline_after).ok_or("--deadline-first-after-ms overflows Instant")?;
    let mut tasks = Vec::new();
    let mut observations = Vec::new();
    let mut join_failures = 0_u64;
    let mut begin_failures = 0_u64;
    let mut cancellation_barrier = None;

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
        if Instant::now() >= run_deadline {
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
    let elapsed_until_reports_ns = elapsed_ns(started);
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
        deadline: run_deadline_evidence(plan.warmup_ns.saturating_add(plan.measurement_ns), elapsed_until_reports_ns),
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

fn run_deadline_evidence(configured_window_ns: u64, elapsed_until_reports_ns: u64) -> RunDeadlineEvidence {
    RunDeadlineEvidence {
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

type AttemptTask = tokio::task::JoinHandle<(super::schedule::ScheduledArrival, SelectedQuery, Arc<QueryAttemptReport>)>;

/// Begins one ordinary public attempt and packages its report collector. The
/// quarantine scenario uses the same submission path as a regular measured
/// run; only the placement of the controller acknowledgement differs.
#[allow(clippy::too_many_arguments)]
fn spawn_attempt_task(
    backend: &NexusGpuBackend,
    sessions: &[SessionContext],
    queries: &[SelectedQuery],
    mut arrival: super::schedule::ScheduledArrival,
    closed_loop: bool,
    warmup_deadline: Instant,
    attempt_deadline: Instant,
    cancel_first_after: Option<Duration>,
    cancellation_barrier: &mut Option<PendingCancellationBarrier>,
) -> Result<AttemptTask, ()> {
    let attempt = backend.begin_attempt(attempt_options(attempt_deadline)).map_err(|_| ())?;
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
    let ctx = sessions[arrival.session_index].clone();
    let query = queries[arrival.query_index].clone();
    if closed_loop {
        // Closed-loop arrivals carry no planned offset. Classify at the
        // submission boundary, before any later report drain can cross the
        // warmup deadline.
        arrival.phase =
            if Instant::now() < warmup_deadline { SchedulePhase::Warmup } else { SchedulePhase::Measurement };
    }
    Ok(tokio::spawn(async move {
        // A failing statement is a measured outcome, not a runner error: the
        // sealed report still describes what happened.
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
    }))
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
        terminal_evidence: terminal_evidence(report),
        succeeded: report.outcome.stable_label() == super::aggregate::SUCCEEDED_OUTCOME,
        reached_gpu: matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
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
    handle: std::thread::JoinHandle<SamplerResult>,
}

struct SamplerResult {
    samples: Vec<BackendSnapshotSample>,
    occupancy: BTreeMap<usize, DeviceOccupancyPeaks>,
    metric_snapshots: Vec<MetricSnapshotObservation>,
    raw_snapshot_retention_truncated: bool,
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
            let mut metric_snapshots = Vec::new();
            let mut raw_snapshot_retention_truncated = false;
            loop {
                sample_once(
                    &backend,
                    started,
                    &mut samples,
                    &mut peaks,
                    &mut metric_snapshots,
                    &mut raw_snapshot_retention_truncated,
                    false,
                );
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
                    sample_once(
                        &backend,
                        started,
                        &mut samples,
                        &mut peaks,
                        &mut metric_snapshots,
                        &mut raw_snapshot_retention_truncated,
                        true,
                    );
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
            SamplerResult { samples, occupancy: peaks, metric_snapshots, raw_snapshot_retention_truncated }
        });
        Self { stop, handle }
    }

    fn stop(self) -> SamplerResult {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        self.handle.join().unwrap_or_else(|_| SamplerResult {
            samples: Vec::new(),
            occupancy: BTreeMap::new(),
            metric_snapshots: Vec::new(),
            raw_snapshot_retention_truncated: false,
        })
    }
}

/// One polling pass: folds the peaks and retains a bounded raw sample.
fn sample_once(
    backend: &NexusGpuBackend,
    started: Instant,
    samples: &mut Vec<BackendSnapshotSample>,
    peaks: &mut BTreeMap<usize, DeviceOccupancyPeaks>,
    metric_snapshots: &mut Vec<MetricSnapshotObservation>,
    raw_snapshot_retention_truncated: &mut bool,
    retain_terminal_sample: bool,
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
    if !reserve_snapshot_slot(samples, metric_snapshots, raw_snapshot_retention_truncated, retain_terminal_sample) {
        return;
    }
    let offset_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let backend_metrics = backend.backend_metrics_snapshot();
    metric_snapshots.push(metric_snapshot_observation(offset_ns, &backend_metrics));
    samples.push(BackendSnapshotSample {
        offset_ns,
        backend_metrics_snapshot: serde_json::to_value(backend_metrics).unwrap_or(Value::Null),
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
    fn measured_attempts_receive_the_shared_window_deadline() {
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(attempt_options(deadline).deadline, Some(deadline));
    }

    #[test]
    fn deadline_evidence_fails_closed_on_a_finalizer_overrun() {
        let within = run_deadline_evidence(10, 10);
        assert!(within.reports_sealed_within_deadline);
        assert_eq!(within.overrun_ns, 0);

        let overrun = run_deadline_evidence(10, 14);
        assert!(!overrun.reports_sealed_within_deadline);
        assert_eq!(overrun.overrun_ns, 4);
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
        assert!(selected_scenario_clauses_pass(&aggregate.scenarios, true));
        assert!(!selected_scenario_clauses_pass(&aggregate.scenarios, false));
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
