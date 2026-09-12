use std::{
    sync::{Barrier, mpsc},
    time::Instant,
};

use super::*;

#[test]
fn execution_spawn_is_rejected_when_its_class_is_saturated() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(
        cudf::DeviceId::DEFAULT.ordinal(),
        2,
        1,
        1,
        GpuExecutionQueryTracker::new(),
    )
    .unwrap();
    let barrier = std::sync::Arc::new(Barrier::new(2));
    let (entered_sender, entered_receiver) = mpsc::channel();
    let running_barrier = std::sync::Arc::clone(&barrier);
    pool.spawn(GpuExecutionJobKind::AttemptWork, async move {
        let _ = entered_sender.send(());
        running_barrier.wait();
    })
    .unwrap();
    entered_receiver.recv_timeout(std::time::Duration::from_secs(5)).unwrap();

    let error = pool.spawn(GpuExecutionJobKind::AttemptWork, async {}).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("gpu_execution_pool_class_saturated"));
    assert_eq!(error.fact_value("job_class"), Some("auxiliary"));

    // Lifecycle work still admits: the lifecycle class is untouched.
    pool.spawn(GpuExecutionJobKind::Cleanup, async {}).unwrap();
    barrier.wait();
}

#[test]
fn class_capacities_are_reported_and_release_on_completion() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(
        cudf::DeviceId::DEFAULT.ordinal(),
        4,
        2,
        6,
        GpuExecutionQueryTracker::new(),
    )
    .unwrap();
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.root_job_capacity, 2);
    assert_eq!(snapshot.aux_job_capacity, 6);
    assert_eq!(snapshot.root_jobs_available, 2);
    assert_eq!(snapshot.aux_jobs_available, 6);
    pool.spawn(GpuExecutionJobKind::Cleanup, async {}).unwrap();
    pool.spawn(GpuExecutionJobKind::AttemptWork, async {}).unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    pool.wait_for_drain_until(deadline).unwrap();
    // The class permit drops after the job counters, so poll for the
    // permits to return instead of racing the task teardown.
    while Instant::now() < deadline {
        let snapshot = pool.snapshot();
        if snapshot.root_jobs_available == 2 && snapshot.aux_jobs_available == 6 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.root_jobs_available, 2);
    assert_eq!(snapshot.aux_jobs_available, 6);
}

/// An attempt's root execution job is root-class work, one per attempt, but a
/// second native island of the same attempt executes on an auxiliary fragment
/// context that already owns an attempt auxiliary permit. Its execution job is
/// therefore auxiliary work backed by that permit: charging the root class
/// instead would fail the second island wherever root capacity is one job per
/// admitted attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_island_execution_job_charges_the_class_its_context_owns() {
    let pool = GpuExecutionPool::new_with_tracker_and_classes(
        cudf::DeviceId::DEFAULT.ordinal(),
        4,
        1,
        1,
        GpuExecutionQueryTracker::new(),
    )
    .unwrap();
    let root = GpuExecutorLease::from_pool(pool.clone());
    let second_island = GpuExecutorLease::from_pool_for_auxiliary_fragment(pool.clone());

    let (entered_sender, entered_receiver) = tokio::sync::oneshot::channel();
    let (release_sender, release_receiver) = tokio::sync::oneshot::channel();
    let root_job = tokio::spawn(root.run_observed_scoped(
        GpuExecutionJobKind::QueryExecution,
        crate::observability::RuntimeObservationHandle::disabled(),
        move |_scope| async move {
            let _ = entered_sender.send(());
            let _ = release_receiver.await;
        },
    ));
    entered_receiver.await.expect("the root execution job starts");
    assert_eq!(pool.snapshot().root_jobs_available, 0, "the root job holds the attempt's only root slot");

    second_island
        .run_observed_scoped(
            GpuExecutionJobKind::QueryExecution,
            crate::observability::RuntimeObservationHandle::disabled(),
            move |_scope| async move {},
        )
        .await
        .expect("the second island runs on the auxiliary slot its context's permit reserves");

    let _ = release_sender.send(());
    root_job.await.expect("the root job joins").expect("the root job completes");
}
