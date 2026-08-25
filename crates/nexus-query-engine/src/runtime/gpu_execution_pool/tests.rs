use std::{
    sync::{
        Condvar,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use futures::task::{ArcWake, waker_ref};

use super::*;

#[derive(Default)]
struct BlockingGate {
    released: Mutex<bool>,
    wake: Condvar,
}

impl BlockingGate {
    fn wait(&self) {
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.wake.wait(released).unwrap();
        }
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

struct DropProbe(Arc<AtomicBool>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

struct WakeFlag(AtomicBool);

impl ArcWake for WakeFlag {
    fn wake_by_ref(flag: &Arc<Self>) {
        flag.0.store(true, Ordering::Release);
    }
}

async fn wait_until(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(Instant::now() < deadline, "condition was not satisfied before the test deadline");
        tokio::task::yield_now().await;
    }
}

fn observed_job_states(owner: &crate::observability::RuntimeObservationOwner) -> Vec<GpuJobLifecycleState> {
    owner
        .live_snapshot()
        .expect("observation remains live")
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::GpuJob(observation) => Some(observation.lifecycle.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn process_exit_joins_an_orphaned_gpu_execution_supervisor() {
    let pool = GpuExecutionPool::new(usize::MAX - 4, 1).unwrap();
    let supervisor = Arc::clone(&pool.inner.supervisor);

    drop(pool);
    supervisor.join_for_process_exit();

    assert_eq!(supervisor.join_disposition.load(Ordering::Acquire), SUPERVISOR_JOINED);
}

#[tokio::test(flavor = "current_thread")]
async fn observed_job_defines_worker_and_records_host_lifecycle() {
    let pool = GpuExecutionPool::new(0, 1).unwrap();
    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-job-success",
        None,
        crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
    );

    let value = pool.run_observed(GpuExecutionJobKind::QueryExecution, owner.handle(), async { 7_u8 }).await.unwrap();

    assert_eq!(value, 7);
    assert_eq!(
        observed_job_states(&owner),
        vec![GpuJobLifecycleState::Enqueued, GpuJobLifecycleState::Started, GpuJobLifecycleState::Finished]
    );
    let snapshot = owner.live_snapshot().unwrap();
    crate::observability::assert_live_topology_references_resolve(&snapshot);
    assert_eq!(snapshot.topology().filter(|row| matches!(row, RuntimeTopologyRecord::JobDefinition(_))).count(), 1);
    assert_eq!(snapshot.topology().filter(|row| matches!(row, RuntimeTopologyRecord::WorkerDefinition(_))).count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn auxiliary_context_setup_uses_auxiliary_capacity_without_changing_its_observation_kind() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(usize::MAX - 8, 2, 1, 1, GpuExecutionQueryTracker::new())
        .unwrap();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let task_release = Arc::clone(&release);
    let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
    let first = {
        let pool = pool.clone();
        tokio::spawn(async move {
            pool.run_observed_scoped_in_class(
                GpuExecutionJobKind::ContextSetup,
                GpuExecutionJobClass::Auxiliary,
                RuntimeObservationHandle::disabled(),
                move |_scope| async move {
                    let _ = started_sender.send(());
                    let _permit = task_release.acquire_owned().await.expect("test release semaphore remains open");
                },
            )
            .await
        })
    };
    started_receiver.await.expect("first auxiliary context setup starts");

    let saturated = pool
        .run_observed_scoped_in_class(
            GpuExecutionJobKind::ContextSetup,
            GpuExecutionJobClass::Auxiliary,
            RuntimeObservationHandle::disabled(),
            |_scope| async {},
        )
        .await
        .expect_err("second auxiliary setup must respect auxiliary capacity");
    assert_eq!(saturated.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(saturated.fact_value("job_kind"), Some("context_setup"));
    assert_eq!(saturated.fact_value("job_class"), Some("auxiliary"));

    pool.run_observed_scoped_in_class(
        GpuExecutionJobKind::ContextSetup,
        GpuExecutionJobClass::Root,
        RuntimeObservationHandle::disabled(),
        |_scope| async {},
    )
    .await
    .expect("root setup stays admissible while auxiliary capacity is full");

    release.add_permits(1);
    first.await.expect("auxiliary setup task must not panic").expect("auxiliary setup completes after release");
}

#[tokio::test(flavor = "current_thread")]
async fn opaque_executor_lease_forwards_only_awaited_execution() {
    let pool = GpuExecutionPool::new(18, 1).unwrap();
    let lease = GpuExecutorLease::from_pool(pool);
    assert_eq!(lease.run(GpuExecutionJobKind::QueryExecution, async { 7_u8 }).await.unwrap(), 7);

    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-executor-lease",
        None,
        crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
    );
    let scope_has_worker = lease
        .run_observed_scoped(GpuExecutionJobKind::QueryExecution, owner.handle(), |scope| async move {
            scope.gpu_job_id.is_some() && scope.gpu_worker_id.is_some()
        })
        .await
        .unwrap();
    assert!(scope_has_worker);
}

#[tokio::test(flavor = "current_thread")]
async fn observed_job_abort_records_one_cancel_terminal() {
    let pool = GpuExecutionPool::new(0, 1).unwrap();
    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-job-cancel",
        None,
        crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
    );
    let task = {
        let pool = pool.clone();
        let observation = owner.handle();
        tokio::spawn(async move {
            pool.run_observed(GpuExecutionJobKind::QueryExecution, observation, std::future::pending::<()>()).await
        })
    };
    wait_until(|| pool.snapshot().active_jobs == 1).await;

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    wait_until(|| pool.snapshot().active_jobs == 0).await;

    let states = observed_job_states(&owner);
    assert_eq!(
        states,
        vec![
            GpuJobLifecycleState::Enqueued,
            GpuJobLifecycleState::Started,
            GpuJobLifecycleState::CancellationRequested,
            GpuJobLifecycleState::Cancelled,
        ]
    );
    assert_eq!(
        states
            .iter()
            .filter(|state| matches!(state, GpuJobLifecycleState::Cancelled | GpuJobLifecycleState::Dropped))
            .count(),
        1
    );
}

#[test]
fn dropping_woken_outer_after_child_finished_emits_no_post_terminal_cancel() {
    let pool = GpuExecutionPool::new(0, 1).unwrap();
    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-job-finished-before-outer-drop",
        None,
        crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
    );
    let gate = Arc::new(BlockingGate::default());
    let child_gate = Arc::clone(&gate);
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    let mut outer = Box::pin(pool.run_observed(GpuExecutionJobKind::QueryExecution, owner.handle(), async move {
        started_sender.send(()).unwrap();
        child_gate.wait();
        7_u8
    }));
    let wake_flag = Arc::new(WakeFlag(AtomicBool::new(false)));
    let waker = waker_ref(&wake_flag);
    let mut context = Context::from_waker(&waker);

    assert!(matches!(outer.as_mut().poll(&mut context), Poll::Pending));
    started_receiver.recv().unwrap();
    wake_flag.0.store(false, Ordering::Release);
    gate.release();

    let deadline = Instant::now() + Duration::from_secs(5);
    while !wake_flag.0.load(Ordering::Acquire)
        || !observed_job_states(&owner).iter().any(|state| matches!(state, GpuJobLifecycleState::Finished))
    {
        assert!(Instant::now() < deadline, "child did not finish and wake the outer future");
        thread::yield_now();
    }
    drop(outer);

    assert_eq!(
        observed_job_states(&owner),
        vec![GpuJobLifecycleState::Enqueued, GpuJobLifecycleState::Started, GpuJobLifecycleState::Finished]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_polled_reentrant_job_records_one_observed_cancellation() {
    let pool = GpuExecutionPool::new(0, 1).unwrap();
    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-inline-cancel",
        None,
        crate::observability::RuntimeObservationConfig::bounded(64, 64 * 1024),
    );
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let task = {
        let outer_pool = pool.clone();
        let nested_pool = pool.clone();
        let outer_observation = owner.handle();
        let nested_observation = owner.handle();
        tokio::spawn(async move {
            outer_pool
                .run_observed(GpuExecutionJobKind::QueryExecution, outer_observation, async move {
                    nested_pool
                        .run_observed(GpuExecutionJobKind::ContextSetup, nested_observation, async move {
                            let _ = started_tx.send(());
                            std::future::pending::<()>().await;
                        })
                        .await
                })
                .await
        })
    };
    started_rx.await.unwrap();

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    wait_until(|| pool.snapshot().active_jobs == 0).await;

    let states = owner
        .live_snapshot()
        .unwrap()
        .events()
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::GpuJob(observation)
                if observation.job_kind == GpuExecutionJobKind::ContextSetup.as_str() =>
            {
                Some(observation.lifecycle.clone())
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        states,
        vec![
            GpuJobLifecycleState::Enqueued,
            GpuJobLifecycleState::Started,
            GpuJobLifecycleState::CancellationRequested,
            GpuJobLifecycleState::Cancelled,
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn panicking_reentrant_job_records_exactly_one_panic_terminal() {
    let pool = GpuExecutionPool::new(0, 1).unwrap();
    let owner = crate::observability::RuntimeObservationOwner::new(
        "gpu-inline-panic",
        None,
        crate::observability::RuntimeObservationConfig::bounded(64, 64 * 1024),
    );
    let nested_pool = pool.clone();
    let nested_observation = owner.handle();
    let nested = pool
        .run(GpuExecutionJobKind::QueryExecution, async move {
            nested_pool
                .run_observed(GpuExecutionJobKind::ContextSetup, nested_observation, async move {
                    panic!("inline job panic")
                })
                .await
        })
        .await
        .unwrap();
    assert_eq!(nested.unwrap_err().code(), QueryErrorCode::Internal);

    let terminals = owner
        .live_snapshot()
        .unwrap()
        .events()
        .iter()
        .filter(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::GpuJob(observation)
                    if observation.job_kind == GpuExecutionJobKind::ContextSetup.as_str()
                        && observation.lifecycle == GpuJobLifecycleState::Panicked
            )
        })
        .count();
    assert_eq!(terminals, 1);
}

#[test]
fn zero_worker_count_is_structured() {
    let error = GpuExecutionPool::new(7, 0).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("device_ordinal"), Some("7"));
    assert_eq!(error.fact_value("worker_count"), Some("0"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn snapshots_identify_the_shared_pool_instance_across_clones() {
    let pool = GpuExecutionPool::new(7, 1).unwrap();
    let clone = pool.clone();
    let distinct_pool = GpuExecutionPool::new(7, 1).unwrap();

    assert_eq!(pool.snapshot().pool_id, clone.snapshot().pool_id);
    assert_ne!(pool.snapshot().pool_id, distinct_pool.snapshot().pool_id);
}

#[tokio::test(flavor = "current_thread")]
async fn fixed_width_tracks_peaks_without_starving_serving_runtime() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 2, 2, 4, GpuExecutionQueryTracker::new()).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let heartbeat = Arc::new(AtomicUsize::new(0));
    let stop_heartbeat = Arc::new(AtomicBool::new(false));

    let heartbeat_task = {
        let heartbeat = Arc::clone(&heartbeat);
        let stop_heartbeat = Arc::clone(&stop_heartbeat);
        tokio::spawn(async move {
            while !stop_heartbeat.load(Ordering::Acquire) {
                heartbeat.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
        })
    };

    let mut jobs = Vec::new();
    for value in 0..4 {
        let pool = pool.clone();
        let gate = Arc::clone(&gate);
        jobs.push(tokio::spawn(async move {
            pool.run(GpuExecutionJobKind::QueryExecution, async move {
                gate.wait();
                value
            })
            .await
        }));
    }

    wait_until(|| pool.snapshot().aux_jobs_available == 0).await;

    let saturated = pool
        .run(GpuExecutionJobKind::QueryExecution, async {})
        .await
        .expect_err("the fifth awaited job must not join an unbounded class waiter queue");
    assert_eq!(saturated.fact_value("reason"), Some("gpu_execution_pool_class_saturated"));

    wait_until(|| {
        let snapshot = pool.snapshot();
        snapshot.active_jobs == 2 && snapshot.queued_jobs == 2 && snapshot.active_query_jobs == 2
    })
    .await;
    assert!(heartbeat.load(Ordering::Relaxed) > 0);
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.active_query_jobs, 2);
    assert_eq!(snapshot.peak_active_jobs, 2);
    assert_eq!(snapshot.peak_active_query_jobs, 2);

    gate.release();
    for job in jobs {
        job.await.unwrap().unwrap();
    }
    stop_heartbeat.store(true, Ordering::Release);
    heartbeat_task.await.unwrap();

    let complete = pool.snapshot();
    assert_eq!(complete.active_jobs, 0);
    assert_eq!(complete.queued_jobs, 0);
    assert_eq!(complete.peak_active_jobs, 2);
    assert!(complete.peak_queued_jobs >= 2);
    assert_eq!(complete.peak_active_query_jobs, 2);
    assert_eq!(pool.query_tracker().snapshot().peak_active_query_jobs, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn width_one_spawned_child_reentrant_run_stays_on_the_same_thread() {
    let pool = GpuExecutionPool::new(1, 1).unwrap();
    let nested_pool = pool.clone();
    let (outer_thread, inner_thread) = pool
        .run(GpuExecutionJobKind::QueryExecution, async move {
            let outer_thread = thread::current().id();
            let inner_thread = tokio::spawn(async move {
                nested_pool.run(GpuExecutionJobKind::ContextSetup, async { thread::current().id() }).await
            })
            .await
            .unwrap()
            .unwrap();
            (outer_thread, inner_thread)
        })
        .await
        .unwrap();

    assert_eq!(outer_thread, inner_thread);
    assert_eq!(pool.snapshot().peak_active_jobs, 1);
    assert_eq!(pool.snapshot().peak_active_query_jobs, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn reentrant_same_class_submission_is_bounded_instead_of_bypassing_the_pool() {
    let pool = GpuExecutionPool::new(1, 1).unwrap();
    let nested_pool = pool.clone();

    let nested = pool
        .run(GpuExecutionJobKind::QueryExecution, async move {
            nested_pool.run(GpuExecutionJobKind::QueryExecution, async { 7_u8 }).await
        })
        .await
        .expect("outer job runs");

    let error = nested.expect_err("nested auxiliary work cannot bypass the outer class permit");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("gpu_execution_pool_class_saturated"));
    assert_eq!(pool.snapshot().peak_active_jobs, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_run_aborts_job_and_releases_counters_and_resources() {
    let pool = GpuExecutionPool::new(2, 1).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let task = {
        let pool = pool.clone();
        let dropped = Arc::clone(&dropped);
        tokio::spawn(async move {
            pool.run(GpuExecutionJobKind::QueryExecution, async move {
                let _probe = DropProbe(dropped);
                std::future::pending::<()>().await;
            })
            .await
        })
    };

    wait_until(|| pool.snapshot().active_jobs == 1).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    wait_until(|| dropped.load(Ordering::Acquire) && pool.snapshot().active_jobs == 0).await;

    let snapshot = pool.snapshot();
    assert_eq!(snapshot.queued_jobs, 0);
    assert_eq!(snapshot.active_query_jobs, 0);
    assert_eq!(pool.query_tracker().snapshot().active_query_jobs, 0);
}

#[test]
fn completion_runs_without_drain_lock_and_precedes_zero_counter_transition() {
    let pool = GpuExecutionPool::new(28, 1).unwrap();
    let drain = Arc::clone(&pool.inner.drain);
    let completion_gate = Arc::new(BlockingGate::default());
    let completion_gate_for_job = Arc::clone(&completion_gate);
    let completion_can_lock_drain = Arc::new(AtomicBool::new(false));
    let completion_can_lock_drain_for_job = Arc::clone(&completion_can_lock_drain);
    let completion_finished = Arc::new(AtomicBool::new(false));
    let completion_finished_for_job = Arc::clone(&completion_finished);
    let (entered_sender, entered_receiver) = mpsc::sync_channel(0);

    pool.spawn_observed_with_completion(
        GpuExecutionJobKind::Cleanup,
        RuntimeObservationHandle::disabled(),
        async {},
        move || {
            let can_lock_drain = {
                let _drain_state = drain.changed.try_lock();
                _drain_state.is_ok()
            };
            completion_can_lock_drain_for_job.store(can_lock_drain, Ordering::Release);
            entered_sender.send(()).expect("test receives completion entry");
            completion_gate_for_job.wait();
            completion_finished_for_job.store(true, Ordering::Release);
        },
    )
    .unwrap();

    entered_receiver.recv_timeout(Duration::from_secs(5)).expect("completion starts");
    let active_during_completion = pool.snapshot().active_jobs;

    completion_gate.release();
    pool.wait_for_drain_until(Instant::now() + Duration::from_secs(5)).unwrap();

    assert!(completion_can_lock_drain.load(Ordering::Acquire));
    assert_eq!(active_during_completion, 1);
    assert!(completion_finished.load(Ordering::Acquire));
}

#[tokio::test(flavor = "current_thread")]
async fn final_pool_drop_is_safe_inside_current_thread_runtime() {
    let pool = GpuExecutionPool::new(3, 1).unwrap();
    pool.run(GpuExecutionJobKind::Cleanup, async {}).await.unwrap();
    drop(pool);
}

#[test]
fn composed_cleanup_closes_drains_exits_and_joins() {
    let pool = GpuExecutionPool::new(29, 1).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    pool.close_and_wait_until(deadline).unwrap();
    pool.join_supervisor().unwrap();
    let closed = pool.spawn(GpuExecutionJobKind::Cleanup, async {}).unwrap_err();
    assert_eq!(closed.fact_value("reason"), Some("gpu_execution_pool_closing"));
}

#[test]
fn supervisor_shutdown_closes_cancels_drains_and_joins() {
    let pool = GpuExecutionPool::new(30, 1).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicBool::new(false));
    let future_started = Arc::clone(&started);
    let future_dropped = Arc::clone(&dropped);
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        let _probe = DropProbe(future_dropped);
        future_started.store(true, Ordering::Release);
        std::future::pending::<()>().await;
    })
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !started.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "pool job did not start");
        thread::yield_now();
    }

    pool.close_submissions().unwrap();
    let closed = pool.spawn(GpuExecutionJobKind::Cleanup, async {}).unwrap_err();
    assert_eq!(closed.code(), QueryErrorCode::Cancelled);
    assert_eq!(closed.fact_value("reason"), Some("gpu_execution_pool_closing"));
    pool.request_supervisor_exit().unwrap();
    pool.wait_for_drain_until(deadline).unwrap();
    pool.wait_for_supervisor_exit_until(deadline).unwrap();
    pool.join_supervisor().unwrap();
    pool.join_supervisor().unwrap();
    assert!(dropped.load(Ordering::Acquire));
}

#[test]
fn blocked_worker_times_out_then_supervisor_joins_after_late_success() {
    let pool = GpuExecutionPool::new(31, 1).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let job_gate = Arc::clone(&gate);
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        started_sender.send(()).unwrap();
        job_gate.wait();
    })
    .unwrap();
    started_receiver.recv().unwrap();

    pool.close_submissions().unwrap();
    pool.request_supervisor_exit().unwrap();
    let timeout = pool.wait_for_drain_until(Instant::now() + Duration::from_millis(10)).unwrap_err();
    assert!(timeout.is_cleanup_deadline_elapsed());
    assert_eq!(timeout.fact_value("reason"), Some("gpu_execution_pool_drain_timeout"));
    let timeout = pool.wait_for_supervisor_exit_until(Instant::now() + Duration::from_millis(10)).unwrap_err();
    assert!(timeout.is_cleanup_deadline_elapsed());
    assert_eq!(timeout.fact_value("reason"), Some("runtime_supervisor_exit_timeout"));

    gate.release();
    let deadline = Instant::now() + Duration::from_secs(5);
    pool.wait_for_drain_until(deadline).unwrap();
    pool.wait_for_supervisor_exit_until(deadline).unwrap();
    pool.join_supervisor().unwrap();
}

#[test]
fn supervisor_teardown_panic_is_permanent_and_still_joinable() {
    let pool = GpuExecutionPool::new(32, 1).unwrap();
    pool.inner.supervisor.control.panic_during_teardown.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = pool.close_and_wait_until(deadline).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("runtime_supervisor_panicked"));
    assert_eq!(error.fact_value("supervisor_stage"), Some("runtime_teardown"));
    pool.join_supervisor().unwrap();
}

#[test]
fn observed_join_panic_wins_after_an_earlier_cleanup_deadline() {
    let pool = GpuExecutionPool::new(33, 1).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let job_gate = Arc::clone(&gate);
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        started_sender.send(()).unwrap();
        job_gate.wait();
    })
    .unwrap();
    started_receiver.recv().unwrap();
    pool.inner.supervisor.control.panic_after_exit.store(true, Ordering::Release);
    pool.close_submissions().unwrap();
    pool.request_supervisor_exit().unwrap();
    let deadline = pool.close_and_wait_until(Instant::now() + Duration::from_millis(10)).unwrap_err();
    assert!(deadline.is_cleanup_deadline_elapsed());
    assert_eq!(deadline.fact_value("reason"), Some("gpu_execution_pool_drain_timeout"));
    assert_eq!(pool.inner.supervisor.join_disposition.load(Ordering::Acquire), SUPERVISOR_NOT_JOINED);

    gate.release();
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = pool.close_and_wait_until(deadline).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("runtime_supervisor_panicked"));
    assert_eq!(error.fact_value("supervisor_stage"), Some("thread_join"));
    let repeated = pool.join_supervisor().unwrap_err();
    assert_eq!(repeated.fact_value("reason"), Some("runtime_supervisor_join_failed"));
}

#[tokio::test(flavor = "current_thread")]
async fn panicked_jobs_are_structured_permanent_internal_errors() {
    let pool = GpuExecutionPool::new(6, 1).unwrap();
    let error = pool.run(GpuExecutionJobKind::QueryExecution, async { panic!("test panic") }).await.unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.kind(), crate::error::QueryErrorKind::Internal);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("device_ordinal"), Some("6"));
    assert_eq!(error.fact_value("job_kind"), Some("query_execution"));
    assert_eq!(error.fact_value("reason"), Some("job_panicked"));
}

#[tokio::test(flavor = "current_thread")]
async fn shared_tracker_records_exact_cross_device_peak() {
    let tracker = GpuExecutionQueryTracker::new();
    let first = GpuExecutionPool::new_with_tracker(4, 1, tracker.clone()).unwrap();
    let second = GpuExecutionPool::new_with_tracker(5, 1, tracker.clone()).unwrap();
    let gate = Arc::new(BlockingGate::default());

    let first_job = {
        let first = first.clone();
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { first.run(GpuExecutionJobKind::QueryExecution, async move { gate.wait() }).await })
    };
    let second_job = {
        let second = second.clone();
        let gate = Arc::clone(&gate);
        tokio::spawn(async move { second.run(GpuExecutionJobKind::QueryExecution, async move { gate.wait() }).await })
    };

    wait_until(|| tracker.snapshot().active_query_jobs == 2).await;
    assert_eq!(tracker.snapshot().peak_active_query_jobs, 2);
    gate.release();
    first_job.await.unwrap().unwrap();
    second_job.await.unwrap().unwrap();
    assert_eq!(tracker.snapshot().active_query_jobs, 0);
    assert_eq!(tracker.snapshot().peak_active_query_jobs, 2);
}
