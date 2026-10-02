use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::diagnostics::memory::{PhysicalMemoryPressureFacts, PhysicalMemoryPressureLease};
use datafusion::common::DataFusionError;
use tokio::{task::JoinHandle, time::sleep};

use super::*;
use crate::single_node_multi_gpu_validate::schedule::ScheduledArrival;

#[derive(Clone, Copy)]
pub(super) struct FeatureOffCugraphProbeSchedule {
    pub(super) query_index: usize,
    pub(super) warmup_probe_attempts: usize,
    pub(super) measurement_probe_attempts: usize,
}

/// Adds the runner-owned feature-off graph request after the relational set. The caller
/// retains its index for the dedicated probe schedule, while normal arrivals
/// are planned against the pre-append relational count.
pub(super) fn append_feature_off_cugraph_probe(
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
pub(in crate::single_node_multi_gpu_validate) fn validate_cugraph_runner_mode(
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

/// Runs the serial, public-SQL domain-cap probe. The profile preflight
/// pins the two-attempt shape (`q01` followed by `edge_scan`) and the single
/// static partition. This driver deliberately reuses ordinary submission and
/// drain logic: no runner-created error substitutes for allocator evidence.
pub(super) async fn drive_domain_exhaustion_probe(
    backend: &AlgeonGpuBackend,
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

/// Drives the one-device physical allocator-pressure probe.
///
/// The cuDF testing lease owns a separate `cuda_on` sidecar and only exposes
/// observed CUDA memory facts. It is established after the service has
/// installed its own manager, before the public q01 attempt exists, and is
/// consumed (synchronized and verified) before public edge_scan recovery. The
/// two SQL attempts otherwise use the ordinary public attempt API; no runner
/// error, allocator callback, or backend mode substitutes for their reports.
pub(super) async fn drive_physical_allocator_pressure(
    backend: &AlgeonGpuBackend,
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
    let lease = PhysicalMemoryPressureLease::establish(device).map_err(|error| {
        format!("--physical-allocator-pressure could not establish real external pressure: {error}")
    })?;
    let pressure = project_physical_pressure_facts(lease.facts());

    let mut tasks = Vec::with_capacity(1);
    let mut observations = Vec::with_capacity(2);
    let mut begin_failures = 0_u64;
    let mut join_failures = 0_u64;
    let mut cancellation_barrier = None;
    // Dedicated probes never arm deadline/cancellation storms; the knob set stays disabled.
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

pub(super) fn project_physical_pressure_facts(facts: PhysicalMemoryPressureFacts) -> PhysicalPressureLeaseFacts {
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

/// Produces the public attempt contract for one bounded measured window.
///
/// The engine propagates this deadline through planning, admission, stream
/// acquisition, GPU safe boundaries, and result drain.  We deliberately do
/// not wrap the task in `tokio::time::timeout`: aborting the task could skip
/// the linear attempt finalizer that returns a grant and seals its report.
pub(super) fn attempt_options(run_deadline: Instant) -> AttemptOptions {
    AttemptOptions { deadline: Some(run_deadline), ..AttemptOptions::default() }
}

/// Per-attempt evidence that is not itself part of the sealed report.
#[derive(Debug, Clone, Default)]
pub(super) struct OutputCapture {
    pub(super) feature_off_cugraph_source_error: Option<FeatureOffCugraphSourceError>,
    pub(super) graph_claim_calibration: Option<GraphClaimCalibrationSnapshot>,
}

pub(super) type AttemptTask = JoinHandle<(ScheduledArrival, SelectedQuery, Arc<QueryAttemptReport>, OutputCapture)>;

/// Schedules each runner-owned feature-off request through the ordinary public attempt
/// path.  Only the selected query and sequence differ from a planned
/// relational arrival; no result or error evidence is synthesized here.
#[expect(clippy::too_many_arguments, reason = "probe-harness expansion; parameters are the independent probe axes")]
pub(super) fn spawn_feature_off_cugraph_probes(
    backend: &AlgeonGpuBackend,
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
        let arrival = ScheduledArrival {
            sequence,
            round,
            step_index,
            offset_ns: 0,
            // `spawn_attempt_task` records the observed closed-loop phase at
            // submission, which is the only phase evidence the feature-off reducer
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
#[expect(clippy::too_many_arguments, reason = "probe-harness expansion; parameters are the independent probe axes")]
pub(super) fn spawn_attempt_task(
    backend: &AlgeonGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    arrival: ScheduledArrival,
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
/// query id. The exact-key cache probe is the narrow exception: it supplies a
/// unique id whose one-way fingerprint can be matched against cache-owner and
/// BFS-coordinator observations without retaining the raw id in the artifact.
#[expect(clippy::too_many_arguments, reason = "probe-harness expansion; parameters are the independent probe axes")]
pub(super) fn spawn_attempt_task_with_query_id(
    backend: &AlgeonGpuBackend,
    sessions: &SessionPool,
    queries: &[SelectedQuery],
    mut arrival: ScheduledArrival,
    closed_loop: bool,
    warmup_deadline: Instant,
    attempt_deadline: Instant,
    cancel_first_after: Option<Duration>,
    cancellation_barrier: &mut Option<PendingCancellationBarrier>,
    storm: StormKnobs,
    storm_cancellers: &mut Vec<PendingCancellationBarrier>,
    query_id: Option<String>,
) -> Result<AttemptTask, ()> {
    // Per-attempt deadline: each attempt carries its own caller deadline, never
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
                sleep(requested_after).await;
                cancellation.cancel();
                requested.store(true, Ordering::Release);
            }))
        };
        *cancellation_barrier =
            Some(PendingCancellationBarrier { attempt_sequence: arrival.sequence, requested_after, requested, task });
    }
    // Per-attempt cancellation: every Nth attempt gets its own cooperative
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
                sleep(requested_after).await;
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
        // The optional sink deliberately serializes the public completion
        // receiver. The permit remains live through `.completion().await`,
        // so a saturated sink can exert the real handoff backpressure that a
        // per-task sleep before this point would miss.
        let completion_sink = query.completion_sink().cloned();
        let _completion_sink_permit = if let Some(sink) = completion_sink.as_ref() {
            let permit = sink.acquire().await;
            sleep(sink.delay()).await;
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
pub(super) fn feature_off_cugraph_source_error(error: &DataFusionError) -> FeatureOffCugraphSourceError {
    let Some(error) = error.find_algeon_error() else {
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

pub(super) async fn finish_cancellation_barrier(
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
pub(super) async fn drain(
    tasks: &mut Vec<AttemptTask>,
    observations: &mut Vec<AttemptObservation>,
    join_failures: &mut u64,
) {
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

/// Ordinary stress cells use [`drain`]. The feature-off driver retains one
/// extra bounded source-error projection for its runner-owned UDTF so the
/// artifact can pair exact adapter facts with the same sealed report row.
pub(super) async fn drain_with_feature_off_cugraph_capture(
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
