use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
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
async fn execution_context_setup_uses_execution_capacity_without_changing_its_observation_kind() {
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
                GpuExecutionJobClass::Execution,
                RuntimeObservationHandle::disabled(),
                move |_scope| async move {
                    let _ = started_sender.send(());
                    let _permit = task_release.acquire_owned().await.expect("test release semaphore remains open");
                },
            )
            .await
        })
    };
    started_receiver.await.expect("first execution context setup starts");

    let saturated = pool
        .clone()
        .run_observed_scoped_in_class(
            GpuExecutionJobKind::ContextSetup,
            GpuExecutionJobClass::Execution,
            RuntimeObservationHandle::disabled(),
            |_scope| async {},
        )
        .await
        .expect_err("second execution setup must respect execution capacity");
    assert_eq!(saturated.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(saturated.fact_value("job_kind"), Some("context_setup"));
    assert_eq!(saturated.fact_value("job_class"), Some("auxiliary"));

    pool.run_observed_scoped_in_class(
        GpuExecutionJobKind::ContextSetup,
        GpuExecutionJobClass::Lifecycle,
        RuntimeObservationHandle::disabled(),
        |_scope| async {},
    )
    .await
    .expect("lifecycle setup stays admissible while execution capacity is full");

    release.add_permits(1);
    first.await.expect("execution setup task must not panic").expect("execution setup completes after release");
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
                        .run_observed(GpuExecutionJobKind::AttemptWork, nested_observation, async move {
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
                if observation.job_kind == GpuExecutionJobKind::AttemptWork.as_str() =>
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
                .run_observed(GpuExecutionJobKind::AttemptWork, nested_observation, async move {
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
                    if observation.job_kind == GpuExecutionJobKind::AttemptWork.as_str()
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
            pool.run(GpuExecutionJobKind::AttemptWork, async move {
                gate.wait();
                value
            })
            .await
        }));
    }

    wait_until(|| pool.snapshot().aux_jobs_available == 0).await;

    let saturated = pool
        .run(GpuExecutionJobKind::AttemptWork, async {})
        .await
        .expect_err("the fifth awaited job must not join an unbounded class waiter queue");
    assert_eq!(saturated.fact_value("reason"), Some("gpu_execution_pool_class_saturated"));

    wait_until(|| {
        let snapshot = pool.snapshot();
        snapshot.active_jobs == 2 && snapshot.queued_jobs == 2
    })
    .await;
    assert!(heartbeat.load(Ordering::Relaxed) > 0);
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.peak_active_jobs, 2);
    assert_eq!(
        snapshot.active_query_jobs, 0,
        "the query counter counts root query jobs, never the auxiliary work they issue"
    );

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
    assert_eq!(complete.peak_active_query_jobs, 0);
    assert_eq!(pool.query_tracker().snapshot().peak_active_query_jobs, 0);
}

/// The device query counter tracks root query jobs only: the auxiliary work a
/// root job issues underneath itself must not inflate the service's view of
/// how many queries are executing.
#[tokio::test(flavor = "current_thread")]
async fn query_counters_track_root_query_jobs_and_not_the_auxiliary_work_they_issue() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 2, 2, 4, GpuExecutionQueryTracker::new()).unwrap();
    let nested_pool = pool.clone();
    let observed = pool
        .run(GpuExecutionJobKind::QueryExecution, async move {
            let inside_root = pool_query_counts(&nested_pool);
            let inside_auxiliary = nested_pool
                .run(GpuExecutionJobKind::AttemptWork, {
                    let nested_pool = nested_pool.clone();
                    async move { pool_query_counts(&nested_pool) }
                })
                .await
                .expect("auxiliary work runs under the root job");
            (inside_root, inside_auxiliary)
        })
        .await
        .expect("root query job runs");

    assert_eq!(observed.0, (1, 1), "the root query job is counted once while it runs");
    assert_eq!(observed.1, (1, 1), "auxiliary work under it adds nothing to the query counters");
    let complete = pool.snapshot();
    assert_eq!(complete.active_query_jobs, 0);
    assert_eq!(complete.peak_active_query_jobs, 1);
    assert_eq!(pool.query_tracker().snapshot().peak_active_query_jobs, 1);
}

fn pool_query_counts(pool: &GpuExecutionPool) -> (usize, usize) {
    let snapshot = pool.snapshot();
    (snapshot.active_query_jobs, pool.query_tracker().snapshot().active_query_jobs)
}

fn wait_until_sync(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(Instant::now() < deadline, "condition was not satisfied before the test deadline");
        thread::yield_now();
    }
}

#[test]
fn blocking_submit_saturates_fast_with_typed_error_on_min_pool() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 1, 1, 1, GpuExecutionQueryTracker::new()).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let first = pool
        .submit_blocking(
            GpuExecutionJobKind::AttemptWork,
            crate::observability::RuntimeObservationHandle::disabled(),
            {
                let gate = Arc::clone(&gate);
                move || {
                    gate.wait();
                    7_u8
                }
            },
        )
        .expect("first blocking worker is admitted");
    assert_eq!(pool.snapshot().aux_jobs_available, 0, "the running worker holds the only execution slot");

    let saturated = pool
        .submit_blocking(
            GpuExecutionJobKind::AttemptWork,
            crate::observability::RuntimeObservationHandle::disabled(),
            || 9_u8,
        )
        .expect_err("saturation must fail fast, never queue an unbounded waiter");
    assert_eq!(saturated.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(saturated.fact_value("reason"), Some("gpu_execution_pool_class_saturated"));

    gate.release();
    assert_eq!(first.join().expect("blocking worker completes"), 7_u8);
    let drained = pool.snapshot();
    assert_eq!(drained.aux_jobs_available, 1, "a resolved join implies the slot is reusable");
    assert_eq!(drained.active_jobs, 0);
}

/// A resolved join implies the worker's class slot is already reusable: the
/// worker releases its class permit and job counters before it delivers the
/// completion, so a caller that joins and immediately resubmits is admitted
/// rather than failing on a slot the finished worker still holds.
#[test]
fn blocking_join_resolves_only_after_the_class_slot_is_released() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 1, 1, 1, GpuExecutionQueryTracker::new()).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let worker = pool
        .submit_blocking(
            GpuExecutionJobKind::AttemptWork,
            crate::observability::RuntimeObservationHandle::disabled(),
            {
                let gate = Arc::clone(&gate);
                move || {
                    gate.wait();
                    7_u8
                }
            },
        )
        .expect("blocking worker is admitted");
    wait_until_sync(|| pool.snapshot().active_jobs == 1);
    gate.release();

    assert_eq!(worker.join().expect("a started worker is drained and hands back its value"), 7_u8);
    assert_eq!(pool.snapshot().active_jobs, 0, "the joined worker no longer counts as active");
    pool.submit_blocking(
        GpuExecutionJobKind::AttemptWork,
        crate::observability::RuntimeObservationHandle::disabled(),
        || 9_u8,
    )
    .expect("the only class slot is reusable the moment the previous join resolves")
    .join()
    .expect("the resubmitted worker completes");
}

/// `submit_blocking` must hold the pool's resource epoch clone inside the
/// worker closure, not only across submission: the closure runs on a
/// `spawn_blocking` thread outside the async paths that already move their
/// epoch clone into the task. If the epoch clone were dropped at submission
/// time instead of inside the worker, the epoch would go idle (and a
/// concurrent close could complete) while the worker is still parked.
#[test]
fn blocking_worker_holds_the_resource_epoch_until_the_job_finishes() {
    use crate::runtime::context::{QueryResourceEpoch, QueryServiceEpoch, QueryServiceResourceEpoch};

    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 1, 1, 1, GpuExecutionQueryTracker::new()).unwrap();
    let service_epoch = QueryServiceEpoch::new(None);
    let query_epoch = QueryResourceEpoch::new(None);
    let anchor =
        query_epoch.acquire(Arc::new(()), "test.gpu_execution_pool.resource_epoch").expect("query anchor acquires");
    let resource_epoch =
        QueryServiceResourceEpoch::acquire(&service_epoch, Some(&anchor), "test.gpu_execution_pool.resource_epoch")
            .expect("resource epoch acquires");
    let pool = pool.with_resource_epoch(resource_epoch);

    let gate = Arc::new(BlockingGate::default());
    let worker = pool
        .submit_blocking(
            GpuExecutionJobKind::AttemptWork,
            crate::observability::RuntimeObservationHandle::disabled(),
            {
                let gate = Arc::clone(&gate);
                move || {
                    gate.wait();
                    7_u8
                }
            },
        )
        .expect("blocking worker is admitted");
    wait_until_sync(|| pool.snapshot().active_jobs == 1);

    // Drop every external handle to the pool and its query anchor, then close
    // the service epoch, while the worker is still parked on the gate. The
    // task's own epoch clone must be the only thing keeping the epoch's user
    // count above zero. `wait_for_idle_until` always reports the live user
    // count on a non-completing wait, so checking that fact (rather than
    // whether the call itself errors on an immediate deadline, which can
    // trivially expire either way) deterministically distinguishes a held
    // epoch from an idle one.
    drop(pool);
    drop(anchor);
    service_epoch.invalidate();
    let immediate = Instant::now();
    let still_held = service_epoch
        .wait_for_idle_until(immediate, "test.gpu_execution_pool.resource_epoch")
        .expect_err("the closing epoch cannot complete while the worker's clone is live");
    assert_eq!(
        still_held.fact_value("active_resource_users"),
        Some("1"),
        "the parked worker's resource epoch clone must keep the closing service epoch busy"
    );

    gate.release();
    assert_eq!(worker.join().expect("blocking worker completes"), 7_u8);

    let released_deadline = Instant::now() + Duration::from_secs(5);
    service_epoch
        .wait_for_idle_until(released_deadline, "test.gpu_execution_pool.resource_epoch")
        .expect("the resource epoch becomes idle once the worker's task-local clone drops after the job finishes");
}

#[tokio::test(flavor = "current_thread")]
async fn blocking_completion_survives_release_cleanup_saturation() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(0, 1, 1, 1, GpuExecutionQueryTracker::new()).unwrap();
    let (release_sender, release_receiver) = tokio::sync::oneshot::channel::<()>();
    pool.spawn_release_cleanup_observed_with_completion(
        crate::observability::RuntimeObservationHandle::disabled(),
        async move {
            let _ = release_receiver.await;
        },
        || {},
    )
    .expect("first release cleanup occupies the dedicated slot");
    wait_until(|| pool.snapshot().active_jobs == 1).await;

    let saturated = pool
        .spawn_release_cleanup_observed_with_completion(
            crate::observability::RuntimeObservationHandle::disabled(),
            async {},
            || {},
        )
        .expect_err("cleanup saturation is typed, not a second waiter queue");
    assert_eq!(saturated.fact_value("reason"), Some("gpu_execution_pool_release_cleanup_saturated"));

    let worker = pool
        .submit_blocking(
            GpuExecutionJobKind::AttemptWork,
            crate::observability::RuntimeObservationHandle::disabled(),
            || 7_u8,
        )
        .expect("blocking completion uses its own class slot");
    assert_eq!(worker.join().expect("blocking worker completes during cleanup saturation"), 7_u8);

    let _ = release_sender.send(());
    wait_until(|| pool.snapshot().active_jobs == 0).await;
}

#[tokio::test(flavor = "current_thread")]
async fn width_one_spawned_child_reentrant_run_stays_on_the_same_thread() {
    let pool = GpuExecutionPool::new(1, 1).unwrap();
    let nested_pool = pool.clone();
    let (outer_thread, inner_thread) = pool
        .run(GpuExecutionJobKind::QueryExecution, async move {
            let outer_thread = thread::current().id();
            let inner_thread = tokio::spawn(async move {
                nested_pool.run(GpuExecutionJobKind::AttemptWork, async { thread::current().id() }).await
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
        .run(GpuExecutionJobKind::AttemptWork, async move {
            nested_pool.run(GpuExecutionJobKind::AttemptWork, async { 7_u8 }).await
        })
        .await
        .expect("outer job runs");

    let error = nested.expect_err("nested execution work cannot bypass the outer class permit");
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
    assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
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
    assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
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
    assert_eq!(error.kind(), query_planning::error::QueryErrorKind::Internal);
    assert_eq!(error.status(), query_planning::error::QueryErrorStatus::Permanent);
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
