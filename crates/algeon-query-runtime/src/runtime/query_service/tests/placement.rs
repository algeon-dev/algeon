use query_planning::error::QueryErrorCode;

use super::*;

#[test]
#[ignore = "requires at least 2 GPUs"]
fn queued_placement_folds_a_native_fault_into_the_ledger_before_retargeting() {
    let service = QueryService::new_with_device_ordinals(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2) },
        vec![0, 1],
    )
    .expect("two-device runtime-backed service");
    let device_zero_ledger = std::sync::Arc::clone(
        service.inner.resources.devices.context_by_ordinal(0).expect("device zero context").ledger(),
    );
    let scope = DeviceSafetyTestScope::install([]);
    let unhealthy = injected_device_id(0).expect("isolated device zero health");
    report_device_fault(unhealthy, DeviceFaultKind::FatalDeviceOperation, "injected_queued_placement_fault", None);
    assert!(
        matches!(device_zero_ledger.health().expect("engine ledger health"), DeviceAdmissionHealth::Open),
        "the regression starts with a native fault that the placement boundary must fold into the ledger"
    );

    let mut waiter = service.enqueue_admission().expect("request enters the shared queue");
    let handle = waiter
        .try_admit()
        .expect("physical health reranking succeeds")
        .expect("healthy device one admits the queued request");

    assert_eq!(handle.device_ordinal(), 1);
    assert!(
        matches!(
            device_zero_ledger.health().expect("faulted device ledger health"),
            DeviceAdmissionHealth::Closed { .. }
        ),
        "pre-sweep reconciliation closes the faulted device ledger before selecting a healthy device"
    );
    assert_eq!(scope.health_snapshot(unhealthy).fault_count(), 1);
    handle.end_test_attempt();
    drop(handle);
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .expect("first healthy admission cleanup completes before the next placement");
    assert_eq!(
        service.admission_snapshot().expect("first cleanup admission snapshot").active_queries,
        0,
        "stream-pool idleness must follow the first release commit"
    );

    // Keep the only healthy device occupied. Normal ranking has no runnable
    // candidate, so the fallback must select the healthy saturated ordinal
    // for its memory-wait taxonomy. It must not retry the physically faulty
    // ordinal zero or terminally reject this waiter.
    let mut active_waiter = service.enqueue_admission().expect("active request enters the shared queue");
    let active =
        active_waiter.try_admit().expect("healthy placement succeeds").expect("device one admits the active request");
    assert_eq!(active.device_ordinal(), 1);
    let mut blocked = service.enqueue_admission().expect("blocked request enters the shared queue");
    assert!(
        blocked.try_admit().expect("unhealthy ordinal is candidate-local").is_none(),
        "the waiter remains queued behind the healthy device instead of failing on device zero"
    );
    assert_eq!(service.admission_snapshot().expect("admission snapshot").queued_queries, 1);
    active.end_test_attempt();
    drop(active);
    // `finish` schedules destruction of the device-backed envelope on its
    // selected GPU executor. The immutable grant becomes reusable only after
    // that cleanup commits. The test waits on the service's semantic
    // quiescence fence. It does not race the asynchronous cleanup job.
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .expect("active healthy admission cleanup completes before re-admission");
    let admitted_after_release = blocked
        .try_admit()
        .expect("healthy capacity release wakes the waiter")
        .expect("waiter admits on device one after release");
    assert_eq!(admitted_after_release.device_ordinal(), 1);
    admitted_after_release.end_test_attempt();
    drop(admitted_after_release);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

/// A native fault closes the device's ledger when it is reported, so a waiter
/// whose only device faulted ends with the device error without waiting for
/// other admission activity or its queue deadline.
#[test]
fn a_native_fault_ends_a_waiter_without_other_admission_activity() {
    // The service registers its fault observer in the registry of the thread
    // that builds it, so the isolated registry is installed first.
    let scope = DeviceSafetyTestScope::install([]);
    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5) },
        vec![runtime_test_device_profile(1)],
    )
    .unwrap();
    let device_ordinal = service.device_ledger_reports().unwrap()[0].device_ordinal;
    let device = injected_device_id(device_ordinal as i32).expect("isolated device health owner");
    let active = service.admit_query().expect("the active query occupies the only slot");
    let queued_service = service.clone();
    let waiter = std::thread::spawn(move || queued_service.admit_query().map(|handle| handle.end_test_attempt()));
    let deadline = Instant::now() + Duration::from_secs(1);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "the second admission queues behind the active query");
        std::thread::yield_now();
    }

    report_device_fault(device, DeviceFaultKind::FatalDeviceOperation, "injected_queued_waiter_fault", None);

    let error =
        waiter.join().expect("waiter thread does not panic").expect_err("the faulted device rejects the waiter");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    assert_eq!(scope.health_snapshot(device).fault_count(), 1);
    active.end_test_attempt();
}

#[test]
fn dropping_admission_ticket_wakes_a_close_waiter_after_cleanup() {
    let service = test_service(1, 8);
    let ticket = {
        let state = service.controller.state.lock().expect("admission state");
        state.lifecycle.begin_provisional(std::sync::Arc::downgrade(&service.controller))
    };

    let close_service = service.clone();
    let (close_sender, close_receiver) = std::sync::mpsc::sync_channel(1);
    let close = std::thread::spawn(move || {
        close_sender
            .send(close_service.close_and_wait_for_quiescence(Duration::from_secs(2)))
            .expect("test receives close result");
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while service.shutdown_waiter_count() == 0 {
        assert!(Instant::now() < deadline, "close must wait for the live lifecycle ticket");
        std::thread::yield_now();
    }

    drop(ticket);
    close_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("close wakes after the ticket is removed")
        .expect("ticket cleanup leaves the service quiescent");
    close.join().expect("close thread does not panic");
}

#[test]
fn device_entries_keep_distinct_ordinal_capability_requirements() {
    let service = test_service(1, 8).controller;
    let first = requirement();
    let second = distinct_capability_requirement();
    let (entries, compatible) = service
        .resolve_device_admission_entries_with(&[0, 1], |ordinal| {
            Ok(if ordinal == 0 { first.clone() } else { second.clone() })
        })
        .expect("both concrete candidates resolve");

    assert_eq!(compatible, vec![0, 1]);
    assert_ne!(entries.get(&0).expect("ordinal zero entry"), entries.get(&1).expect("ordinal one entry"));
}

#[test]
fn candidate_local_entry_resolution_prunes_only_the_rejected_ordinal() {
    let service = test_service(1, 8).controller;
    let (entries, compatible) = service
        .resolve_device_admission_entries_with(&[0, 1], |ordinal| {
            if ordinal == 0 {
                Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "query_engine.test.sort_preflight",
                    "candidate sort owner preflight failed",
                )
                .with_fact("reason", "sort_owner_candidate_device_unavailable"))
            } else {
                Ok(requirement())
            }
        })
        .expect("one healthy candidate remains");

    assert_eq!(compatible, vec![1]);
    assert!(!entries.contains_key(&0), "the rejected candidate must not enter placement");
    assert!(entries.contains_key(&1), "the healthy candidate keeps its resolved entry");
}

#[test]
fn all_device_entry_resolution_failures_return_capability_reason_before_queueing() {
    let service = test_service(1, 8).controller;
    let error = service
        .resolve_device_admission_entries_with(&[0, 1], |_ordinal| {
            Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "query_engine.test.sort_capability",
                "sort capability is unavailable",
            )
            .with_fact("reason", "sort_capability_unavailable"))
        })
        .expect_err("all candidate-local failures reject before enqueue");

    assert_eq!(error.fact_value("reason"), Some("sort_capability_unavailable"));
    assert_eq!(error.fact_value("candidate_count"), Some("2"));
    assert_eq!(service.state.lock().expect("admission state").queued.len(), 0);
}
