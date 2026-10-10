use std::{
    collections::BTreeSet,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use tokio::time::{Instant as TokioInstant, sleep, sleep_until};

use super::*;
use crate::single_node_multi_gpu_validate::schedule::ScheduledArrival;

pub(super) struct DrivenRun {
    pub(super) observations: Vec<AttemptObservation>,
    pub(super) dropped_overrun_arrivals: u64,
    pub(super) begin_failures: u64,
    pub(super) join_failures: u64,
    pub(super) deadline: RunDeadlineEvidence,
    pub(super) cancellation_barrier: Option<CancellationBarrierEvidence>,
    pub(super) quarantine_barrier: Option<QuarantineBarrier>,
    pub(super) deadline_barrier: Option<DeadlineBarrierEvidence>,
    pub(super) physical_allocator_pressure: Option<PhysicalAllocatorPressureBarrier>,
    pub(super) observed_peak_queued_attempts: usize,
    /// Service-owned queue high-water marks captured at each sequential step
    /// boundary.  They are cumulative from backend construction, so each one
    /// proves the configured bound through that phase without resetting or
    /// synthesizing a controller counter in the runner.
    pub(super) queue_high_watermarks: Vec<(usize, u64)>,
    /// Deadline/cancellation storm actions the driver armed and confirmed fired.
    pub(super) storm_cancellations_requested: u64,
    /// Repeated-quarantine barriers, one per completed cycle.
    pub(super) quarantine_cycle_barriers: Vec<QuarantineBarrier>,
    /// Close-under-saturation facts captured at the close boundary.
    pub(super) close_under_load: Option<CloseUnderLoadBarrier>,
    /// Direct, typed feature-off UDTF source errors paired by sequence with
    /// their sealed reports. Ordinary runs leave this empty.
    pub(super) feature_off_cugraph_source_errors: Vec<(usize, FeatureOffCugraphSourceError)>,
}

/// Submits every planned arrival at its offset and collects one bounded report
/// per attempt.
pub(super) async fn drive(
    backend: &AlgeonGpuBackend,
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
pub(super) async fn drive_overload_fairness(
    backend: &AlgeonGpuBackend,
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
/// preceding one drains. The wall-clock step deadline ends the step, and no
/// planning cap does. Fixed cohorts continue through the stored plan so their
/// exact attempt shape remains frozen.
pub(super) async fn drive_duration_paced_concurrency(
    backend: &AlgeonGpuBackend,
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
    let mut workload_sequence = 0_usize;
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
                let arrival = duration_paced_workload_arrival(
                    sequence,
                    &mut workload_sequence,
                    round,
                    step.step_index,
                    queries,
                    sessions,
                    feature_off_cugraph_probe,
                );
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
            // but inject it beside a live relational round. The phase boundary
            // comes from the warmup deadline.
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
        let queue_high_watermark = backend.diagnostics()?.admission.queue_high_watermark;
        queue_high_watermarks.push((step.step_index, queue_high_watermark as u64));
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

/// Dedicated probes consume report sequences; query and session rotation
/// advances only with ordinary workload arrivals.
fn duration_paced_workload_arrival(
    sequence: usize,
    workload_sequence: &mut usize,
    round: usize,
    step_index: usize,
    queries: &[SelectedQuery],
    sessions: &SessionPool,
    feature_off_cugraph_probe: Option<FeatureOffCugraphProbeSchedule>,
) -> ScheduledArrival {
    let workload_query_count = feature_off_cugraph_probe.map_or(queries.len(), |probe| probe.query_index);
    let mut arrival =
        duration_paced_arrival(*workload_sequence, round, step_index, &queries[..workload_query_count], sessions);
    *workload_sequence = workload_sequence.saturating_add(1);
    arrival.sequence = sequence;
    arrival
}

pub(super) fn duration_paced_arrival(
    sequence: usize,
    round: usize,
    step_index: usize,
    queries: &[SelectedQuery],
    sessions: &SessionPool,
) -> ScheduledArrival {
    let query_index = sequence % queries.len();
    let session_index = sequence
        % sessions
            .get(&queries[query_index].fixture_index)
            .expect("install_sessions registers every selected fixture workload")
            .len();
    ScheduledArrival {
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

#[cfg(test)]
mod tests;

#[expect(clippy::too_many_arguments, reason = "validation harness parameters are independent measurement axes")]
pub(super) async fn drive_with_queue_observation(
    backend: &AlgeonGpuBackend,
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
        // Steps are sequential phases inside one run. A step change drains
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
            let queue_high_watermark = backend.diagnostics()?.admission.queue_high_watermark;
            queue_high_watermarks.push((current_step, queue_high_watermark as u64));
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
            // The step window ends a closed-loop step; the plan length does not.
            // Rounds are planned generously and pacing is observed. Drain the
            // previous round so `concurrency` bounds in-flight attempts. The
            // drain keeps `concurrency` from becoming an offered arrival rate.
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
                    sleep_until(TokioInstant::from_std(warmup_deadline)).await;
                }
            }
        } else {
            let planned = started + Duration::from_nanos(arrival.offset_ns);
            match Instant::now().checked_duration_since(planned) {
                // Late by more than the tolerance. Rate mode may abandon it,
                // which keeps the offered rate from turning into a closed loop.
                // The policy is read from each arrival, so rate steps may mix
                // `delay` and `drop` phases inside one run.
                Some(late) if late > RATE_OVERRUN_TOLERANCE => {
                    if arrival.rate_overrun == RateOverrunPolicy::Drop {
                        dropped_overrun_arrivals += 1;
                        continue;
                    }
                },
                Some(_) => {},
                None => sleep_until(TokioInstant::from_std(planned)).await,
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
    let queue_high_watermark = backend.diagnostics()?.admission.queue_high_watermark;
    queue_high_watermarks.push((current_step, queue_high_watermark as u64));
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
pub(super) fn step_submission_deadline(started: Instant, plan: &ArrivalPlan, step_index: usize) -> Instant {
    let step = &plan.steps[step_index];
    started + Duration::from_nanos(step.start_ns.saturating_add(step.measurement_ns))
}

/// Joins every storm canceller and counts the cancellation actions that
/// fired. Cancellers own no resources; joining only makes their
/// requested timestamps deterministic.
pub(super) async fn finish_storm_cancellers(cancellers: Vec<PendingCancellationBarrier>) -> u64 {
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
pub(super) async fn observe_queue_peak(
    backend: &AlgeonGpuBackend,
    queue_limit: usize,
    run_deadline: Instant,
) -> ToolResult<usize> {
    const POLL_INTERVAL: Duration = Duration::from_millis(1);
    const OBSERVATION_WINDOW: Duration = Duration::from_secs(5);
    let deadline = (Instant::now() + OBSERVATION_WINDOW).min(run_deadline);
    let mut peak = 0;
    loop {
        let snapshot = backend.diagnostics()?.admission;
        peak = peak.max(snapshot.queued_queries);
        if peak >= queue_limit || Instant::now() >= deadline {
            return Ok(peak);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// Drives a two-cohort controlled-quarantine scenario through the ordinary
/// public attempt API. The first closed-loop round is intentionally kept live
/// until the target ledger exposes a grant or query job. Only after the
/// synchronous controller acknowledgement are the remaining rounds released.
///
/// This is not a second scheduler or a backend mode. The sole state change is
/// `GpuBackendFaultInjector::quarantine_device`, which owns the same fleet
/// admission gate as normal placement. The runner merely orders public calls
/// and records the barrier they establish.
pub(super) async fn drive_concurrent_quarantine(
    backend: &AlgeonGpuBackend,
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
    // Dedicated probes never arm deadline/cancellation storms; they pass `StormKnobs::default()`.
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
    let injection = backend.quarantine_device_for_qualification(target_ordinal)?;
    let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
    let acknowledgement_offset_ns = elapsed_ns(started);
    let target_ledger_closed = backend
        .diagnostics()?
        .ledgers
        .into_iter()
        .find(|report| report.device_ordinal == target_ordinal)
        .is_some_and(|report| report.health == "closed");

    // The initial transition from round 0 to round 1 deliberately does not
    // drain `tasks`. The post-ack cohort must contend with, and then prove
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
        .diagnostics()?
        .ledgers
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

/// Drives the repeated-quarantine scenario. Each cycle applies one controlled
/// quarantine to a rotation of configured ordinals while sustained closed-loop
/// load runs. Each cycle reuses the single-quarantine barrier semantics: a
/// live-work witness, a synchronous injector acknowledgement, zero post-ack
/// placements on the target, and healthy progress on the remaining ordinals.
/// A cycle is as falsifiable as the single-quarantine probe.
///
/// Quarantine is a terminal per-device transition, so the rotation targets a
/// distinct ordinal per cycle; each target's own health-generation increment
/// is recorded on its cycle row. Profile preflight keeps one ordinal open, so
/// stabilization between cycles is fleet progress. The fleet is not drained.
pub(super) async fn drive_quarantine_cycles(
    backend: &AlgeonGpuBackend,
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
    // Cycle boundaries sit at defined window offsets. Cycle k fires once the
    // run crosses (k + 1) / (cycles + 1) of the measurement window.
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
            let injection = backend.quarantine_device_for_qualification(target_ordinal)?;
            let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
            let acknowledgement_offset_ns = elapsed_ns(started);
            let target_ledger_closed = backend
                .diagnostics()?
                .ledgers
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
    let ledgers = backend.diagnostics()?.ledgers;
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

pub(super) async fn drive_duration_paced_quarantine_cycles(
    backend: &AlgeonGpuBackend,
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
                let injection = backend.quarantine_device_for_qualification(target_ordinal)?;
                let post_ack_cache = cache_quarantine_observation(backend, target_ordinal).ok();
                let acknowledgement_offset_ns = elapsed_ns(timing.started);
                let target_ledger_closed = backend
                    .diagnostics()?
                    .ledgers
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
    let ledgers = backend.diagnostics()?.ledgers;
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

/// Drives the close-under-saturation scenario. A closed-loop cohort
/// oversubscribes the fleet; at the measurement-window midpoint the runner
/// stops submitting, snapshots public admission state, and initiates the
/// public close path. Queued waiters must seal `backend_closing`, in-flight
/// attempts must finish, and every submitted attempt must be accounted for
/// exactly once. The runner never aborts a task to force the shape.
pub(super) async fn drive_close_under_load(
    backend: &AlgeonGpuBackend,
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

    // Public admission facts are read first, then the public close
    // transition runs. No further submissions happen after this point.
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

pub(super) async fn drive_duration_paced_close_under_load(
    backend: &AlgeonGpuBackend,
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
        sleep_until(TokioInstant::from_std(close_at)).await;
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
pub(super) async fn wait_for_close_under_load_cohort(
    backend: &AlgeonGpuBackend,
    deadline: Instant,
) -> ToolResult<(usize, usize)> {
    loop {
        let admission = backend.diagnostics()?.admission;
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
        sleep(Duration::from_millis(1)).await;
    }
}

/// A queued waiter sealed by close carries the bounded `BackendClosing`
/// failure class in its sealed report. The runner reads only that closed
/// vocabulary, never reconstructed queue ordering.
pub(super) fn is_typed_backend_closing(observation: &AttemptObservation) -> bool {
    observation.failure.as_ref().is_some_and(|failure| failure.code == "BackendClosing")
}

/// Drives one attempt with a shorter caller deadline, but only accepts it as
/// evidence after public snapshots first witness an immutable admission
/// envelope. The deadline remains the normal attempt contract: no runner task
/// is aborted and a CUDA call already in flight may finish before the next
/// native safe boundary records the expiry.
pub(super) async fn drive_deadline_after_admission(
    backend: &AlgeonGpuBackend,
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
    // Dedicated probes never arm deadline/cancellation storms; they pass `StormKnobs::default()`.
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

pub(super) struct TargetLiveWorkWitness {
    pub(super) target_ordinal: usize,
    pub(super) offset_ns: u64,
    pub(super) ledger_sequence: u64,
    pub(super) live_grants: usize,
    pub(super) active_query_jobs: usize,
}

/// Waits only on public pull snapshots. A target qualifies once an immutable
/// grant has been published or a native query job is live. The artifact keeps
/// both counts, and a configured slot does not prove execution.
pub(super) async fn wait_for_target_live_work(
    backend: &AlgeonGpuBackend,
    target_ordinal: usize,
    started: Instant,
    deadline: Instant,
) -> ToolResult<TargetLiveWorkWitness> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let diagnostics = backend.diagnostics()?;
        let ledgers = diagnostics.ledgers;
        let pools = diagnostics.execution_pools;
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
        sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// Picks the device that first exposes a live immutable grant or native
/// query job. The deadline scenario intentionally does not introduce caller
/// device affinity. Its artifact records the controller's choice.
pub(super) async fn wait_for_any_live_work(
    backend: &AlgeonGpuBackend,
    started: Instant,
    deadline: Instant,
) -> ToolResult<TargetLiveWorkWitness> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let diagnostics = backend.diagnostics()?;
        let ledgers = diagnostics.ledgers;
        let pools = diagnostics.execution_pools;
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
        sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

pub(super) async fn wait_for_target_release(
    backend: &AlgeonGpuBackend,
    target_ordinal: usize,
    deadline: Instant,
) -> ToolResult<bool> {
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    loop {
        let reports = backend.diagnostics()?.ledgers;
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
        sleep(POLL_INTERVAL.min(remaining)).await;
    }
}

/// The attempt report preserves `deadline_expired` in its bounded reason
/// vocabulary. This check reads only that JSON projection. It never rehydrates
/// arbitrary error text or infers a timeout from the runner clock.
pub(super) fn is_typed_deadline_cancellation(observation: &AttemptObservation) -> bool {
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
