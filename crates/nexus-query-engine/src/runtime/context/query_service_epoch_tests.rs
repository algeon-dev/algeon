use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use super::*;

#[test]
fn invalidate_is_nonblocking_during_idle_state_lock_contention() {
    let epoch = QueryServiceEpoch::new(None);
    let state = epoch.inner.idle_state.lock().unwrap();
    let invalidating_epoch = epoch.clone();
    let (invalidated_tx, invalidated_rx) = mpsc::channel();
    let invalidation = std::thread::spawn(move || {
        invalidating_epoch.invalidate();
        invalidated_tx.send(()).unwrap();
    });

    invalidated_rx
        .recv_timeout(Duration::from_millis(50))
        .expect("epoch invalidation must not wait for the idle-state mutex");
    drop(state);
    invalidation.join().unwrap();
    assert!(!epoch.inner.open.load(Ordering::Acquire));
}

#[test]
fn idle_lock_contention_returns_timeout_before_post_deadline_release() {
    let epoch = QueryServiceEpoch::new(None);
    let mut state = epoch.inner.idle_state.lock().unwrap();
    epoch.invalidate();

    let deadline = Instant::now() + Duration::from_millis(20);
    let waiting_epoch = epoch.clone();
    let (result_tx, result_rx) = mpsc::channel();
    let wait = std::thread::spawn(move || {
        result_tx.send(waiting_epoch.wait_for_idle_until(deadline, "test.epoch.shutdown")).unwrap();
    });

    let result = result_rx.recv_timeout(Duration::from_millis(200));
    state.complete = true;
    epoch.inner.idle_changed.notify_all();
    drop(state);
    wait.join().unwrap();

    let error = result
        .expect("epoch shutdown must return its mutex timeout before the contended lock is released")
        .expect_err("completion after the deadline must not be accepted");
    assert_eq!(error.fact_value("reason"), Some("query_service_resource_epoch_timeout"));
}
