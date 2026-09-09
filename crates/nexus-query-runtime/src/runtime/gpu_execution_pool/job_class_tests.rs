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
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = entered_sender.send(());
        running_barrier.wait();
    })
    .unwrap();
    entered_receiver.recv_timeout(std::time::Duration::from_secs(5)).unwrap();

    let error = pool.spawn(GpuExecutionJobKind::QueryExecution, async {}).unwrap_err();
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
    pool.spawn(GpuExecutionJobKind::QueryExecution, async {}).unwrap();
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
