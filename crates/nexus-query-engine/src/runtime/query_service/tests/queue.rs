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
    assert_eq!(error.kind(), crate::error::QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(
        QueryAdmissionOutcome::from_error_with_queue_bounds(&error, service.service.queue_bounds()),
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
    assert!(saturated.queue_high_watermark <= service.service.queue_bounds().max_waiters);

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
    assert!(drained.queue_high_watermark <= service.service.queue_bounds().max_waiters);

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
        .enqueue_admission_with_request(QueryAdmissionRequest::new().with_execution_cancellation(cancellation))
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
            .admit_async_with_request(QueryAdmissionRequest::new().with_execution_cancellation(queued_cancellation))
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
            .admit_query_with_request(QueryAdmissionRequest::new().with_execution_cancellation(queued_cancellation));
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
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 4 },
        non_gds_deployment(),
        test_ledger(capacity_bytes, max_active_attempts_per_device, Default::default()),
    )
    .unwrap();

    let handles = (0..max_active_attempts_per_device)
        .map(|_| service.try_admit_query().unwrap().expect("capacity should admit query"))
        .collect::<Vec<_>>();

    let admitted_budget_sum: u64 = handles.iter().map(AdmissionTestHandle::device_budget_bytes).sum();
    assert_eq!(admitted_budget_sum, capacity_bytes);
    assert!(service.try_admit_query().unwrap().is_none());
}
#[test]
fn query_service_uses_variable_memory_governor_grants() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 4 },
        100,
        10,
        4,
    );

    let first = service.try_admit_query().unwrap().expect("first query should admit");
    let second = service.try_admit_query().unwrap().expect("second query should preserve remaining minima");
    let third = service.try_admit_query().unwrap().expect("third query should admit minimum grant");
    let fourth = service.try_admit_query().unwrap().expect("fourth query should admit minimum grant");

    // The normative equation reserves the configured floor for every
    // unprotected empty slot: 100-10*3, then 100-70-10*2, then floors.
    assert_eq!(first.device_budget_bytes(), 70);
    assert_eq!(second.device_budget_bytes(), 10);
    assert_eq!(third.device_budget_bytes(), 10);
    assert_eq!(fourth.device_budget_bytes(), 10);
    assert!(service.try_admit_query().unwrap().is_none());
    let report = first_memory_governor_report(&service);
    assert_eq!(report.managed_capacity_bytes, 100);
    assert_eq!(report.live_grants_bytes, 100);
    assert_eq!(report.managed_capacity_bytes - report.live_grants_bytes, 0);

    drop(first);
    assert_eq!(
        first_memory_governor_report(&service).managed_capacity_bytes
            - first_memory_governor_report(&service).live_grants_bytes,
        70
    );
}
#[test]
fn query_service_waiting_on_memory_grant_wakes_after_grant_release() {
    // A request-level floor raise above the configured floor blocks until the
    // active attempt's immutable grant returns to the ledger.
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        250,
        1,
        2,
    );
    let raised = requirement(150, Some(150));
    let active = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(raised.clone()))
        .unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service
            .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(raised.clone()))
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
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        250,
        1,
        2,
    );
    let raised = requirement(150, Some(150));
    let active = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(raised.clone()))
        .unwrap();
    let retained_grant = active.memory_grant().unwrap();
    drop(active);
    assert_eq!(first_memory_governor_report(&service).live_grants_bytes, 150);

    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service
            .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(raised.clone()))
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
    // Immutable ledger semantics: a same-floor attempt always fits (the
    // equation preserves empty-slot floors), so a memory-blocked waiter
    // requires a request-level floor raise above the configured floor.
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(20), max_overtakes_per_waiter: 4 },
        250,
        1,
        2,
    );
    let _active = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_requirement(requirement(150, Some(150))))
        .unwrap();
    let mut waiter = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(150, Some(150))),
        )
        .unwrap();
    service.expire_waiter(&mut waiter);

    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(
        QueryAdmissionOutcome::from_error_with_queue_bounds(&error, service.service.queue_bounds()),
        QueryAdmissionOutcome::Capacity {
            queued_queries: Some(0),
            queue_capacity: Some(8),
            max_queue_wait: Some(Duration::from_millis(20)),
        }
    );
}

#[test]
fn query_admission_waiter_timeout_reports_the_selected_device_requirement() {
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0, 1],
        non_gds_deployment(),
    )
    .expect("two-device runtime-backed service");
    service
        .close_device_admission_for_test(0, "selected-device-timeout-test")
        .expect("ordinal zero closes while ordinal one remains eligible");
    let active = service.admit_query().expect("ordinal one admits the active attempt");
    assert_eq!(active.device_ordinal(), 1);

    let mut waiter = service.enqueue_admission().expect("request enters the shared queue");
    let entries = DeviceAdmissionEntries::from([
        (
            0,
            DeviceAdmissionEntry {
                requirement: requirement(111, Some(111)),
                admitted_reservation_envelope: None,
            },
        ),
        (
            1,
            DeviceAdmissionEntry {
                requirement: requirement(222, Some(222)),
                admitted_reservation_envelope: None,
            },
        ),
    ]);
    waiter.device_entries = entries.clone();
    {
        let mut state = service.inner.admission.lock().expect("admission state");
        state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == waiter.admission_request_id)
            .expect("queued request")
            .device_entries = entries;
    }

    // Keep the queued request's service deadline in the future so the sweep
    // retains it, while the waiter deadline drives the selected-but-blocked
    // timeout branch below.
    waiter.deadline = Instant::now();
    let error = waiter.try_admit().expect_err("saturated ordinal one reports a memory-grant timeout");

    assert_eq!(error.fact_value("reason"), Some("memory_grant_wait_timeout"));
    assert_eq!(error.fact_value("required_floor_bytes"), Some("222"));
    assert_eq!(error.fact_value("soft_working_set_hint_bytes"), Some("222"));

    active.finish();
    drop(active);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn query_service_work_conserving_admission_skips_memory_blocked_head() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 4 },
        300,
        10,
        2,
    );
    let _active = service.admit_query().unwrap();
    let mut large = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(250, Some(250))),
        )
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();

    assert!(large.try_admit().unwrap().is_none());
    let admitted_small = small.try_admit().unwrap().expect("small query should pass memory-blocked large query");
    assert_eq!(admitted_small.device_budget_bytes(), 10);
    assert_eq!(admitted_small.admission_sequence(), 2);
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 2);
    assert_eq!(snapshot.queued_queries, 1);
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
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 1, max_wait: Duration::from_millis(10), max_overtakes_per_waiter: 4 },
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
    assert_eq!(admitted.device_budget_bytes(), 100);
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

#[test]
fn attempt_bound_queued_close_seals_one_terminal_report_without_admission() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first admission occupies the only slot");
    let lifecycle = crate::runtime::BackendLifecycle::new();
    let counters = std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default());
    let attempt = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..crate::attempt::BeginAttemptRequest::default()
        },
        lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");

    let mut queued = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()),
        )
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
