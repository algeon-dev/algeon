use query_planning::error::{QueryErrorKind, QueryErrorStatus};

#[test]
fn query_service_reports_overload_when_queue_is_full() {
    let service = test_service(1, 1);
    let active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (finish_sender, finish_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let handle = queued_service.admit_query().unwrap();
        finish_receiver.recv().unwrap();
        drop(handle);
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let error = service.admit_query().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.kind(), QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(
        QueryAdmissionOutcome::from_error_with_queue_bounds(&error, &service.controller.queue_bounds),
        QueryAdmissionOutcome::Capacity {
            queued_queries: Some(1),
            queue_capacity: Some(1),
            max_queue_wait: Some(Duration::from_millis(500)),
        }
    );

    drop(active);
    finish_sender.send(()).unwrap();
    waiter.join().unwrap();
}
#[test]
fn query_service_reports_admission_snapshot() {
    let service = test_service(1, 8);
    let active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let _handle = queued_service.admit_query().unwrap();
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 1);
    assert_eq!(snapshot.queue_high_watermark, 1);

    drop(active);
    waiter.join().unwrap();
}

#[test]
fn query_service_admission_snapshot_retains_queue_high_watermark_after_drain() {
    const QUEUE_LIMIT: usize = 2;

    let service = test_service(1, QUEUE_LIMIT);
    let active = service.admit_query().unwrap();
    let first = service.enqueue_admission().unwrap();
    let second = service.enqueue_admission().unwrap();

    let saturated = service.admission_snapshot().unwrap();
    assert_eq!(saturated.queued_queries, QUEUE_LIMIT);
    assert_eq!(saturated.queue_high_watermark, QUEUE_LIMIT);
    assert!(saturated.queue_high_watermark <= service.controller.queue_bounds.max_waiters);

    let overflow = service.enqueue_admission().unwrap_err();
    assert_eq!(overflow.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(overflow.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(
        service.admission_snapshot().unwrap().queue_high_watermark,
        QUEUE_LIMIT,
        "a rejected arrival must not inflate the service-owned queue high-watermark"
    );

    drop(first);
    drop(second);

    let drained = service.admission_snapshot().unwrap();
    assert_eq!(drained.queued_queries, 0);
    assert_eq!(drained.queue_high_watermark, QUEUE_LIMIT);
    assert!(drained.queue_high_watermark <= service.controller.queue_bounds.max_waiters);

    drop(active);
}
#[test]
fn query_service_try_admit_does_not_queue_when_saturated() {
    let service = test_service(1, 8);
    let active = service.try_admit_query().unwrap().expect("first query should admit");

    assert!(service.try_admit_query().unwrap().is_none());
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 0);

    drop(active);
    assert!(service.try_admit_query().unwrap().is_some());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_wakes_on_release() {
    let service = test_service(1, 8);
    let active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    wait_for_queued_len(&service, 1);

    drop(active);
    let _handle = time::timeout(Duration::from_millis(200), queued).await.unwrap().unwrap().unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_notifies_after_queue_head_admits() {
    let service = test_service(3, 8);
    let active_1 = service.admit_query().unwrap();
    let active_2 = service.admit_query().unwrap();
    let active_3 = service.admit_query().unwrap();
    let first_service = service.clone();
    let second_service = service.clone();
    let first = tokio::spawn(async move { first_service.admit_async().await });
    let second = tokio::spawn(async move { second_service.admit_async().await });
    wait_for_queued_len(&service, 2);

    drop(active_1);
    drop(active_2);
    drop(active_3);
    let _first = time::timeout(Duration::from_millis(200), first).await.unwrap().unwrap().unwrap();
    let _second = time::timeout(Duration::from_millis(200), second).await.unwrap().unwrap().unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 2);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_cancelled_admit_async_removes_queued_request() {
    let service = test_service(1, 8);
    let active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    wait_for_queued_len(&service, 1);

    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    wait_for_exact_queued_len(&service, 0);
    drop(active);

    assert!(service.try_admit_query().unwrap().is_some());
}

#[test]
fn query_service_rejects_an_already_cancelled_execution_token_before_enqueue() {
    let service = test_service(1, 8);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();

    let error = service
        .enqueue_admission_with_request(QueryAdmissionRequest::test_attempt().with_execution_cancellation(cancellation))
        .expect_err("a cancelled attempt must never enter the admission queue");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0);
    assert_eq!(snapshot.queued_queries, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_execution_token_cancels_a_queued_async_admission() {
    let service = test_service(1, 8);
    let active = service.admit_query().unwrap();
    let cancellation = ExecutionCancellation::new();
    let queued_service = service.clone();
    let queued_cancellation = cancellation.clone();
    let queued = tokio::spawn(async move {
        queued_service
            .admit_async_with_request(
                QueryAdmissionRequest::test_attempt().with_execution_cancellation(queued_cancellation),
            )
            .await
    });
    wait_for_queued_len(&service, 1);

    cancellation.cancel();
    let error = time::timeout(Duration::from_secs(1), queued)
        .await
        .expect("queued admission must observe its execution token promptly")
        .expect("queued admission task must not panic")
        .expect_err("cancelled queued admission must not receive a handle");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(QueryAdmissionOutcome::from_error(&error), QueryAdmissionOutcome::Cancelled);
    wait_for_exact_queued_len(&service, 0);

    drop(active);
    assert!(service.try_admit_query().unwrap().is_some(), "cancellation must not retain capacity");
}

/// A queued async admission wakes on its cancellation token, not a timer.
#[tokio::test(start_paused = true)]
async fn cancelling_a_queued_async_admission_wakes_it_without_a_timer() {
    let service = test_service(1, 8);
    let active = service.admit_query().unwrap();
    let cancellation = ExecutionCancellation::new();
    let queued_service = service.clone();
    let queued_cancellation = cancellation.clone();
    let queued = tokio::spawn(async move {
        queued_service
            .admit_async_with_request(
                QueryAdmissionRequest::test_attempt().with_execution_cancellation(queued_cancellation),
            )
            .await
    });
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        yield_now().await;
    }

    let started = time::Instant::now();
    cancellation.cancel();
    let error = queued
        .await
        .expect("queued admission task must not panic")
        .expect_err("cancelled queued admission must not receive a handle");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(started.elapsed(), Duration::ZERO, "cancellation wakes the waiter without a timer");
    drop(active);
}

#[test]
fn blocking_admission_returns_its_cancelled_terminal_result_after_queue_sweep() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first admission occupies the only slot");
    let cancellation = ExecutionCancellation::new();
    let queued_service = service.clone();
    let queued_cancellation = cancellation.clone();
    let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
    let admission = std::thread::spawn(move || {
        let result = queued_service
            .admit_query_with_request(
                QueryAdmissionRequest::test_attempt().with_execution_cancellation(queued_cancellation),
            );
        result_sender.send(result.map(|_| ())).expect("test receives blocking admission result");
    });

    wait_for_queued_len(&service, 1);
    cancellation.cancel();
    let error = result_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("blocking admission must return promptly after its queue entry is swept")
        .expect_err("cancelled admission cannot receive a handle");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_cancelled"));
    admission.join().expect("blocking admission thread joins");
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);

    drop(active);
    assert!(service.try_admit_query().unwrap().is_some(), "cancelled queue entry must not retain capacity");
}
#[test]
fn query_service_concurrent_budget_sum_never_exceeds_capacity() {
    let capacity_bytes = 12 * 1024 * 1024;
    let max_active_attempts_per_device = 4;
    let service = AdmissionTestController::new_with_ledger(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        test_ledger(capacity_bytes, max_active_attempts_per_device, Default::default()),
    )
    .unwrap();

    let handles = (0..max_active_attempts_per_device)
        .map(|_| service.try_admit_query().unwrap().expect("capacity should admit query"))
        .collect::<Vec<_>>();

    let admitted_budget_sum: u64 = handles.iter().map(AdmissionTestHandle::device_budget_bytes).sum();
    let default_query_limit = first_memory_governor_report(&service).default_query_limit_bytes;
    assert_eq!(admitted_budget_sum, default_query_limit * max_active_attempts_per_device as u64);
    assert!(admitted_budget_sum <= capacity_bytes);
    assert!(service.try_admit_query().unwrap().is_none());
}
#[test]
fn query_service_uses_fixed_default_query_caps() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        100,
        10,
        4,
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second fixed-cap query should admit");
    let third = service.try_admit_query().unwrap().expect("third fixed-cap query should admit");
    let fourth = service.try_admit_query().unwrap().expect("fourth fixed-cap query should admit");

    assert_eq!(first.device_budget_bytes(), 10);
    assert_eq!(second.device_budget_bytes(), 10);
    assert_eq!(third.device_budget_bytes(), 10);
    assert_eq!(fourth.device_budget_bytes(), 10);
    assert!(service.try_admit_query().unwrap().is_none());
    let report = first_memory_governor_report(&service);
    assert_eq!(report.managed_capacity_bytes, 100);
    assert_eq!(report.live_grants_bytes, 40);
    assert_eq!(report.managed_capacity_bytes - report.live_grants_bytes, 60);

    drop(first);
    assert_eq!(
        first_memory_governor_report(&service).managed_capacity_bytes
            - first_memory_governor_report(&service).live_grants_bytes,
        70
    );
}
#[test]
fn query_service_waiting_on_memory_grant_wakes_after_grant_release() {
    // Two explicit caps whose sum exceeds capacity serialize until the first
    // immutable grant returns to the ledger.
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2) },
        250,
        1,
        2,
    );
    let active =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150)).unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service
            .admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150))
            .unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);
    assert_eq!(first_memory_governor_report(&service).live_grants_bytes, 150);

    drop(active);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 150);
    assert_eq!(first_memory_governor_report(&service).live_grants_bytes, 150);
    drop(admitted);
    assert_eq!(first_memory_governor_report(&service).live_grants_bytes, 0);
}
#[test]
fn query_service_waiting_on_memory_grant_wakes_after_retained_grant_release() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2) },
        250,
        1,
        2,
    );
    let active =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150)).unwrap();
    let retained_grant = active.memory_grant();
    drop(active);
    assert_eq!(first_memory_governor_report(&service).live_grants_bytes, 150);

    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service
            .admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150))
            .unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);

    drop(retained_grant);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 150);
}
#[test]
fn query_admission_waiter_times_out_while_selected_but_memory_blocked() {
    // Two explicit caps whose sum exceeds capacity keep the second waiter
    // selected but memory-blocked until its deadline.
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(20) },
        250,
        1,
        2,
    );
    let _active =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150)).unwrap();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(150))
        .unwrap();
    service.expire_waiter(&mut waiter);

    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(error.fact_value("queued_queries"), Some("0"), "the expired waiter is not a queue member");
    assert_eq!(
        QueryAdmissionOutcome::from_error_with_queue_bounds(&error, &service.controller.queue_bounds),
        QueryAdmissionOutcome::Capacity {
            queued_queries: Some(0),
            queue_capacity: Some(8),
            max_queue_wait: Some(Duration::from_millis(20)),
        }
    );
}

#[test]
fn a_queue_timeout_ends_only_its_own_waiter_and_reports_the_members_left() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        100,
        1,
        1,
    );
    let _active = service.admit_query().unwrap();
    let mut first = service.enqueue_admission().unwrap();
    let mut second = service.enqueue_admission().unwrap();
    let mut live = service.enqueue_admission().unwrap();
    service.expire_waiter(&mut first);
    service.expire_waiter(&mut second);

    let error = first.try_admit().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(error.fact_value("queued_queries"), Some("1"), "only the live waiter is a queue member");
    assert_eq!(service.queued_len(), 2, "the other expired waiter stays queued until its own step");

    let error = second.try_admit().unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(error.fact_value("queued_queries"), Some("1"));
    assert!(live.try_admit().expect("the live waiter keeps waiting").is_none());
    assert_eq!(service.queued_len(), 1);
}

#[test]
fn a_max_wait_past_the_clock_range_sets_no_queue_deadline() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::MAX },
        100,
        1,
        1,
    );
    let _active = service.admit_query().unwrap();
    let mut waiter = service.enqueue_admission().expect("the waiter queues");

    assert!(waiter.try_admit().expect("the waiter has no queue deadline").is_none());
    assert_eq!(service.queued_len(), 1);
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn query_admission_waiter_timeout_reports_the_selected_device_requirement() {
    let service = QueryService::new_with_device_ordinals(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2) },
        vec![0, 1],
    )
    .expect("two-device runtime-backed service");
    service
        .inner.admission.close_device_admission_for_test(0, "selected-device-timeout-test")
        .expect("ordinal zero closes while ordinal one remains eligible");
    let active = service.admit_query().expect("ordinal one admits the active attempt");
    assert_eq!(active.device_ordinal(), 1);

    let mut waiter = service.enqueue_admission().expect("request enters the shared queue");
    let entries = DeviceAdmissionEntries::from([(0, requirement()), (1, requirement())]);
    {
        let mut state = service.inner.admission.state.lock().expect("admission state");
        state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == waiter.waiter.admission_request_id)
            .expect("queued request")
            .device_entries = entries;
    }

    // Keep the queued request's service deadline in the future so the sweep
    // retains it, while the waiter deadline drives the selected-but-blocked
    // timeout branch below.
    waiter.waiter.deadline = Some(Instant::now());
    let error = waiter.try_admit().expect_err("saturated ordinal one reports a memory-grant timeout");

    // This service is backed by the real installed devices, so the grant size
    // is whatever ordinal one's profile issued; assert the selected-device
    // scope and the live grant itself rather than a hardware-dependent literal.
    let active_grant = format!("1:{}", active.device_budget_bytes());
    assert_eq!(error.fact_value("reason"), Some("memory_grant_wait_timeout"));
    assert_eq!(error.fact_value("requested_query_cap_bytes"), Some("0"));
    assert_eq!(error.fact_value("device_frozen_resident_bytes"), Some("1:0"));
    assert_eq!(error.fact_value("device_live_grants_bytes"), Some(active_grant.as_str()));
    assert_eq!(error.fact_value("device_admissible_query_cap_bytes"), Some("1:0"));

    active.end_test_attempt();
    drop(active);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn query_service_fifo_keeps_younger_work_behind_memory_blocked_head() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        300,
        10,
        2,
    );
    let active = service.admit_query().unwrap();
    let mut large = service
        .enqueue_admission_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(295))
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();

    assert!(large.try_admit().unwrap().is_none());
    assert!(small.try_admit().unwrap().is_none(), "FIFO forbids the younger query from overtaking");
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 2);
    drop(active);
    let admitted_large = large.try_admit().unwrap().expect("oldest waiter admits first after capacity returns");
    assert_eq!(admitted_large.device_budget_bytes(), 295);
    assert!(small.try_admit().unwrap().is_none());
}

#[test]
fn frozen_resident_growth_revalidates_and_rejects_an_impossible_waiter() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        100,
        10,
        2,
    );
    let active =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(60)).unwrap();
    let grant = active.memory_grant();
    let mut waiter =
        service.enqueue_admission_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(60)).unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    grant.freeze_resident(50).unwrap();

    let error = waiter.try_admit().expect_err("the new resident charge makes the explicit cap permanently impossible");
    assert_eq!(error.code(), QueryErrorCode::QueryBudgetUnavailable);
    assert_eq!(error.fact_value("reason"), Some("query_cap_exceeds_resident_capacity"));
    assert_eq!(error.fact_value("requested_query_cap_bytes"), Some("60"));
    drop(active);
    drop(grant);
}

#[test]
fn frozen_resident_release_revalidates_and_wakes_a_capacity_waiter() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        100,
        10,
        2,
    );
    let resident_owner =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(50)).unwrap();
    let resident_grant = resident_owner.memory_grant();
    resident_grant.freeze_resident(30).unwrap();
    drop(resident_owner);
    let active =
        service.admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(40)).unwrap();
    let mut waiter =
        service.enqueue_admission_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(40)).unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    drop(resident_grant);

    let admitted = waiter.try_admit().unwrap().expect("resident release returns enough capacity to the oldest waiter");
    assert_eq!(admitted.device_budget_bytes(), 40);
    drop(active);
    drop(admitted);
}
#[test]
fn query_service_enqueued_waiter_counts_against_queue_limit_until_drop() {
    let service = test_service(1, 1);
    let _active = service.admit_query().unwrap();
    let waiter = service.enqueue_admission().unwrap();

    let error = service.enqueue_admission().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));

    drop(waiter);
    assert!(service.enqueue_admission().is_ok());
}
#[test]
fn query_service_sweeps_expired_unpolled_waiter_before_queue_and_try_admit_gates() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 1, max_wait: Duration::from_millis(10) },
        100,
        1,
        1,
    );
    let active = service.admit_query().unwrap();
    let stale_waiter = service.enqueue_admission().unwrap();
    let mut stale_waiter = stale_waiter;
    service.expire_waiter(&mut stale_waiter);

    let replacement_waiter = service.enqueue_admission().expect("expired waiter should be swept");
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);
    drop(replacement_waiter);
    drop(active);

    let admitted = service.try_admit_query().unwrap().expect("expired stale waiter must not block try-admit");
    assert_eq!(admitted.device_budget_bytes(), 1);
    drop(stale_waiter);
}
#[test]
fn query_service_close_wakes_waiting_admission() {
    let service = test_service(1, 8);
    let _active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query().unwrap_err()
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    service.close().unwrap();
    let error = waiter.join().unwrap();

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_closed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closing_the_only_device_rejects_a_waiting_admission_without_other_admission_activity() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        64 * 1024 * 1024,
        1024,
        1,
    );
    let _active = service.admit_query().unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    wait_for_queued_len(&service, 1);

    let ledger = Arc::clone(service.controller.devices.ledger(0).expect("fixture ledger"));
    ledger.close_admission_for_unknown_isolation("injected").unwrap();

    let error = time::timeout(Duration::from_secs(2), queued)
        .await
        .expect("closing the waiter's only device ends its wait")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("backend_unhealthy"));
    assert_eq!(error.fact_value("detail"), Some("injected"));
}

#[test]
fn device_local_closure_is_not_backend_unhealthy_whatever_its_text_says() {
    let service = test_service(1, 8);
    let ledger = Arc::clone(service.controller.devices.ledger(0).expect("fixture ledger"));
    // Diagnostic text that once encoded the isolation scope decides nothing.
    ledger.close_admission("unknown-isolation backend failure: device-local text").unwrap();

    let error = service.admit_query().unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    assert_eq!(error.fact_value("detail"), None);
}

#[test]
fn attempt_bound_queued_close_seals_one_terminal_report_without_admission() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first admission occupies the only slot");
    let lifecycle = BackendLifecycle::new();
    let counters = std::sync::Arc::new(AttemptBackendCounters::default());
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion.completion();
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");

    let mut queued = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt_owner.admission_binding()))
        .expect("attempt-bound request queues behind the active admission");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);

    service.close().expect("close wakes the queued attempt");
    let error = queued.try_admit().expect_err("closed admission cannot publish a query handle");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_closed"));

    let report = completion.wait_timeout(Duration::from_secs(2)).expect("queued close seals the bound attempt report");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(report.disposition, AttemptDisposition::Native);
    assert!(report.timestamps.admission_requested_ns.is_some());
    assert!(report.timestamps.admitted_ns.is_none());
    assert!(report.timestamps.physical_execution_started_ns.is_none());
    assert!(report.timestamps.report_ready_ns.is_some());
    assert_eq!(counters.snapshot().reports_ready, 1, "queued close publishes exactly one terminal report");
    assert_eq!(
        service.admission_snapshot().unwrap().active_queries,
        1,
        "the queued close cannot release the live admission"
    );
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0, "the closed queue has no leaked waiter");

    drop(active);
    service
        .close_and_wait_for_quiescence(Duration::from_secs(2))
        .expect("active admission release lets the closed service quiesce");
}

#[test]
fn full_queue_rejects_before_owner_resolution_or_lifecycle_ticket() {
    let service = test_service(1, 1);
    let active = service.admit_query().expect("first admission occupies the only slot");
    let queued = service.enqueue_admission().expect("one real waiter fills the queue");
    let lifecycle_before = service.lifecycle_snapshot();

    let error = service
        .begin_submission(QueryAdmissionRequest::test_attempt())
        .expect_err("queue-full admission must reject before it can resolve device entries");
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(error.fact_value("queued_queries"), Some("1"));
    assert_eq!(error.fact_value("pending_preflight"), Some("0"));
    assert_eq!(service.lifecycle_snapshot(), lifecycle_before, "queue-full rejection creates no provisional lifecycle ticket");

    drop(queued);
    drop(active);
}

#[test]
fn queued_and_pending_reservations_share_one_bound() {
    let service = test_service(1, 2);
    let active = service.admit_query().expect("first admission occupies the only slot");
    let queued = service.enqueue_admission().expect("first waiter enters the queue");
    let pending = service.begin_submission(QueryAdmissionRequest::test_attempt()).expect("second request reserves a pending slot");

    assert_eq!(service.queued_len(), 1);
    assert_eq!(service.pending_preflight_len(), 1);
    let error = service.admit_query().expect_err("queued plus pending reservations fill the bound");
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(error.fact_value("queued_queries"), Some("1"));
    assert_eq!(error.fact_value("pending_preflight"), Some("1"));

    let mut converted = service.finish_submission(pending).expect("the pending request enters the queue");
    assert_eq!(service.pending_preflight_len(), 0);
    drop(queued);
    drop(active);
    assert!(converted.try_admit().expect("released capacity admits the converted waiter").is_some());
}

#[test]
fn pending_reservation_is_a_fifo_barrier_for_younger_queue_work() {
    let service = test_service(1, 2);
    let older = service.begin_submission(QueryAdmissionRequest::test_attempt()).expect("older request reserves a pending slot");

    let mut younger = service.enqueue_admission().expect("younger request enters the queue");
    let younger_id = younger.admission_request_id();
    assert!(
        younger.try_admit().expect("younger waiter remains queued").is_none(),
        "free capacity does not let younger work pass an older pending request"
    );
    assert_eq!(service.pending_preflight_len(), 1);
    assert_eq!(service.queued_len(), 1);

    let mut older = service.finish_submission(older).expect("older request enters the queue");
    assert_eq!(service.pending_preflight_len(), 0);
    let state = service.controller.state.lock().expect("admission state");
    assert_eq!(state.queued.len(), 2);
    assert_eq!(state.queued[1].admission_request_id, younger_id);
    assert!(state.queued[0].submission_seq < state.queued[1].submission_seq);
    drop(state);

    let admitted = older.try_admit().expect("older waiter places").expect("the older request admits first");
    assert!(younger.try_admit().expect("younger waiter remains queued").is_none());
    drop(admitted);
}

#[test]
fn owner_resolution_failure_releases_pending_reservation() {
    let service = test_service(1, 1);
    let submission = service.begin_submission(QueryAdmissionRequest::test_attempt()).expect("submission reserves a pending slot");
    assert_eq!(service.pending_preflight_len(), 1);

    let error = service
        .finish_submission_with(submission, |_ordinal| {
            Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "query_engine.test.entry_resolution",
                "injected device entry resolution failure",
            )
            .with_fact("reason", "injected_entry_resolution_failure"))
        })
        .expect_err("the entry resolution failure is returned");
    assert_eq!(error.fact_value("reason"), Some("injected_entry_resolution_failure"));
    assert_eq!(service.pending_preflight_len(), 0, "entry resolution failure releases its reservation exactly once");
    assert_eq!(service.queued_len(), 0);
}

#[test]
fn cancellation_and_deadline_release_pending_reservations() {
    let service = test_service(1, 2);
    let cancellation = ExecutionCancellation::new();
    let cancelled = service
        .begin_submission(QueryAdmissionRequest::test_attempt().with_execution_cancellation(cancellation.clone()))
        .expect("the request reserves a pending slot");
    cancellation.cancel();
    let error = service.finish_submission(cancelled).expect_err("cancellation wins conversion");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(service.pending_preflight_len(), 0);

    let deadline = Instant::now() + Duration::from_millis(20);
    let expiring = service
        .begin_submission(QueryAdmissionRequest::test_attempt().with_queue_deadline(deadline))
        .expect("the request reserves a pending slot before its deadline");
    while Instant::now() < deadline {
        std::thread::yield_now();
    }
    let error = service.finish_submission(expiring).expect_err("queue deadline wins conversion");
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(service.pending_preflight_len(), 0);

    let error = service
        .begin_submission(
            QueryAdmissionRequest::test_attempt().with_queue_deadline(Instant::now() - Duration::from_millis(1)),
        )
        .expect_err("an already-expired queue deadline rejects before reserving a pending slot");
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(service.pending_preflight_len(), 0);
}

#[test]
fn pending_guard_releases_during_panic_unwind() {
    let service = test_service(1, 1);
    let panic_service = service.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut state = panic_service.controller.lock_admission().expect("admission state");
        let pending = panic_service.controller.reserve_pending_preflight_locked(&mut state).expect("slot available");
        drop(state);
        assert_eq!(panic_service.pending_preflight_len(), 1);
        let _ = &pending;
        panic!("injected pending preflight unwind");
    }));
    assert!(result.is_err());
    assert_eq!(service.pending_preflight_len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pending_guard_releases_when_task_is_dropped() {
    let service = test_service(1, 1);
    let task_service = service.clone();
    let task = tokio::spawn(async move {
        let pending = {
            let mut state = task_service.controller.lock_admission().expect("admission state");
            task_service.controller.reserve_pending_preflight_locked(&mut state).expect("slot available")
        };
        std::future::pending::<()>().await;
        drop(pending);
    });
    wait_for_pending_preflight_len(&service, 1);
    task.abort();
    let _ = task.await;
    assert_eq!(service.pending_preflight_len(), 0);
}

#[test]
fn close_waits_for_pending_owner_resolution_release() {
    let service = test_service(1, 1);
    let pending = service.begin_submission(QueryAdmissionRequest::test_attempt()).expect("the request reserves a pending slot");

    let close_service = service.clone();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let close = std::thread::spawn(move || {
        sender
            .send(close_service.close_and_wait_for_quiescence(Duration::from_secs(2)))
            .expect("close result receiver remains live");
    });
    while service.shutdown_waiter_count() == 0 {
        std::thread::yield_now();
    }
    assert!(matches!(receiver.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));

    let error = service.finish_submission(pending).expect_err("closed service rejects pending conversion");
    assert_eq!(error.fact_value("reason"), Some("admission_closed"));
    assert!(receiver.recv_timeout(Duration::from_secs(2)).expect("close result arrives").is_ok());
    close.join().expect("close thread joins");
    assert_eq!(service.pending_preflight_len(), 0);
}

/// Waiter revalidation excludes closed devices from the impossible-cap
/// decision, but a fully closed fleet stays a health failure. A closed
/// device's capacity must never satisfy a cap, and its closure must never be
/// reported as a budget rejection.
#[test]
fn closed_devices_leave_the_impossible_cap_decision_to_the_fleet_health_error() {
    let service = admission_test_service_with_profile(QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30) },
        100,
        10,
        2,
    );
    let ledger = Arc::clone(service.controller.devices.ledger(0).expect("fixture ledger"));
    ledger.close_admission("test closes the only compatible device").expect("device closes");

    let error = service
        .try_admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(80))
        .expect_err("a closed fleet rejects admission");

    assert_ne!(
        error.code(),
        QueryErrorCode::QueryBudgetUnavailable,
        "a closed device is a health failure, not an impossible query cap: {error}"
    );
}
