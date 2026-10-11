use query_planning::error::QueryErrorCode;

use super::*;
// Transactional admission tests cover fault-injection rollback and envelope
// identity evidence.

/// Wait for the controller's complete cleanup boundary. Streams returning to the
/// pool are not sufficient, because the release commit still owns the ledger
/// grant and lane until it publishes the terminal report.
fn wait_for_transaction_cleanup(service: &QueryService, what: &str) {
    service.wait_for_stream_pools_idle(Duration::from_secs(10)).unwrap_or_else(|error| panic!("{what}: {error}"));
}

/// Every injected transaction failure must restore the admission counters,
/// ledger state, attempt lanes, and stream permits to their prior values,
/// leave the ledger sequence in a valid observable state, and keep the service
/// healthy for a later admission.
#[test]
fn injected_fault_at_every_transaction_step_restores_all_counters_exactly() {
    for step in AdmissionStep::ALL {
        let service =
            try_guaranteed_runtime_test_service().expect("runtime-backed service for transaction rollback coverage");
        let resources = &service.inner.resources;
        let device = resources.devices.context_by_ordinal(0).expect("device context");
        let ledger_before = device.ledger().snapshot().expect("ledger snapshot");
        assert_eq!(ledger_before.active_attempts, 0);
        assert_eq!(ledger_before.live_grants_bytes, 0);
        let (_, root_capacity, _, aux_capacity) = device.stream_pool().class_counters();

        let error = service
            .admit_query_with_probe(
                QueryAdmissionRequest::test_attempt(),
                Arc::new(AdmissionTransactionProbe::failing_at(step)),
            )
            .expect_err("injected fault rejects the admission");
        assert_eq!(error.fact_value("reason"), Some("injected_admission_step_fault"), "step {step:?}");
        assert_eq!(error.fact_value("admission_step"), Some(step.as_str()), "step {step:?}");

        let snapshot = service.admission_snapshot().expect("admission snapshot");
        assert_eq!(snapshot.active_queries, 0, "active count restored at {step:?}");
        assert_eq!(snapshot.queued_queries, 0, "queue depth restored at {step:?}");
        // The provisional lane and grant intentionally remain held until the
        // deferred stream cleanup and the following release commit finish.
        wait_for_transaction_cleanup(&service, &format!("transaction capacity restored at {step:?}"));
        let ledger_after = device.ledger().snapshot().expect("ledger snapshot");
        let (root_out, _, aux_out, _) = device.stream_pool().class_counters();
        assert_eq!(ledger_after.active_attempts, 0, "ledger slot restored at {step:?}");
        assert_eq!(ledger_after.live_grants_bytes, 0, "ledger grant restored at {step:?}");
        assert_eq!(device.lane_permits().held_count(), 0, "lane restored at {step:?}");
        assert_eq!((root_out, aux_out), (0, 0), "stream permits restored at {step:?}");
        assert!(
            ledger_after.sequence >= ledger_before.sequence,
            "ledger sequence remains valid (monotonic) at {step:?}"
        );

        // The service stays healthy. A later admission publishes a complete
        // matching envelope.
        let handle = service.admit_query().expect("admission succeeds after rollback");
        let (domain_id, grant_id, worker_scope) =
            handle.lifetime.envelope_evidence().expect("admitted handle carries its envelope");
        assert_eq!(domain_id, Some(grant_id), "domain id matches the immutable grant at {step:?}");
        let worker_scope = worker_scope.expect("device-backed envelope carries a worker scope");
        assert_eq!(worker_scope.device_ordinal(), handle.device_ordinal());
        assert!(worker_scope.lane_index() < ledger_after.max_active_attempts);
        let pool = resources.gpu_execution_pool_for_device(0).expect("device worker pool");
        assert_eq!(worker_scope.pool_id(), pool.snapshot().pool_id);
        assert_eq!(root_capacity, ledger_after.max_active_attempts);
        assert!(aux_capacity >= ledger_after.max_active_attempts);
        handle.end_test_attempt();
        drop(handle);
        wait_for_transaction_cleanup(&service, &format!("release returns the ledger slot at {step:?}"));
        assert_eq!(device.ledger().snapshot().expect("ledger snapshot").active_attempts, 0);
        service
            .close_and_wait_for_quiescence(Duration::from_secs(10))
            .unwrap_or_else(|error| panic!("service releases its live backend after {step:?}: {error}"));
        drop(service);
    }
}

#[test]
fn rollback_retains_lane_and_grant_until_stream_cleanup_quiesces() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for rollback cleanup-fence coverage");
    let resources = &service.inner.resources;
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let cleanup_blocker = block_stream_cleanup_thread_for_test(device.stream_pool());
    let error = service
        .admit_query_with_probe(
            QueryAdmissionRequest::test_attempt(),
            Arc::new(AdmissionTransactionProbe::failing_at(AdmissionStep::AfterRootMember)),
        )
        .expect_err("injected post-stream fault rejects admission");
    assert_eq!(error.fact_value("reason"), Some("injected_admission_step_fault"));
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(device.ledger().snapshot().unwrap().active_attempts, 1, "grant remains fenced by cleanup");
    assert_eq!(device.lane_permits().held_count(), 1, "lane remains fenced by cleanup");
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(root_out, 1, "root stream remains outstanding while cleanup is blocked");

    assert!(
        service.try_admit_query().expect("replacement admission attempt").is_none(),
        "replacement cannot reuse provisional capacity before cleanup quiesces"
    );

    drop(cleanup_blocker);
    wait_for_transaction_cleanup(&service, "rollback cleanup releases every retained capacity component");
    let ledger = device.ledger().snapshot().unwrap();
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(ledger.active_attempts, 0);
    assert_eq!(device.lane_permits().held_count(), 0);
    assert_eq!(root_out, 0);

    let replacement = service.admit_query().expect("capacity becomes reusable after cleanup quiesces");
    replacement.end_test_attempt();
    drop(replacement);
}

/// A complete but unpublished envelope transfers its private admission count
/// to rollback cleanup before the admission mutex is released. If the transfer
/// were deferred, the admission-quiescence phase of close could observe every
/// counter as zero in the narrow interval before `AdmissionEnvelope::drop`
/// installs deferred cleanup.
#[test]
fn rejected_complete_envelope_keeps_close_from_reporting_false_quiescence() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for provisional rollback quiescence");
    let device = service.inner.resources.devices.context_by_ordinal(0).expect("device context");
    let cleanup_blocker = block_stream_cleanup_thread_for_test(device.stream_pool());
    let (probe, rollback_pause) = AdmissionTransactionProbe::pausing_after_rollback_accounting();
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_probe(QueryAdmissionRequest::test_attempt(), Arc::new(probe))
    });

    rollback_pause.wait_until_entered(Duration::from_secs(10));
    {
        let state = service.inner.admission.state.lock().expect("admission state after rollback ownership transfer");
        let lifecycle = state.lifecycle.snapshot();
        assert_eq!(lifecycle.provisional, 0, "publication rejection released the provisional admission ticket");
        assert_eq!(lifecycle.cleanup, 1, "rollback cleanup owns the provisional envelope before mutex unlock");
    }

    let close_service = service.clone();
    let (close_sender, close_receiver) = std::sync::mpsc::sync_channel(1);
    let close = std::thread::spawn(move || {
        close_sender
            .send(close_service.close_and_wait_for_quiescence(Duration::from_secs(10)))
            .expect("test receives close result");
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.inner.admission.shutdown_waiters.load(std::sync::atomic::Ordering::Acquire) == 0 {
        assert!(Instant::now() < deadline, "close must wait for pre-accounted provisional cleanup");
        std::thread::yield_now();
    }
    assert!(
        matches!(close_receiver.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "close must not report quiescence before the rejected envelope begins cleanup"
    );

    rollback_pause.release();
    let error = admission
        .join()
        .expect("admission thread joins")
        .expect_err("a cancellation at publication rejects the complete envelope");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("cancellation_requested"), Some("true"));
    drop(cleanup_blocker);
    close_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("close completes after deferred cleanup")
        .expect("close observes rollback cleanup quiescence");
    close.join().expect("close thread joins");
}

/// After placement the queue deadline no longer bounds the admission; the
/// attempt's own deadline does, and its expiry during the envelope build
/// rolls the complete envelope back.
#[test]
fn attempt_deadline_expiry_after_acquisition_rolls_back_complete_envelope() {
    let service = try_guaranteed_runtime_test_service()
        .expect("runtime-backed service for post-acquisition deadline rollback coverage");
    let resources = &service.inner.resources;
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterRootMember);
    let attempt_deadline = Instant::now() + Duration::from_millis(200);
    let attempt_owner =
        observed_attempt("post-acquisition-deadline", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt_owner.observation_for_test();
    let binding = attempt_owner.admission_binding();
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_probe(
            QueryAdmissionRequest::new(binding)
                .with_execution_cancellation(ExecutionCancellation::with_deadline(Some(attempt_deadline))),
            Arc::new(probe),
        )
    });

    pause.wait_until_entered(Duration::from_secs(10));
    std::thread::sleep(attempt_deadline.saturating_duration_since(Instant::now()));
    pause.release();
    let error = admission.join().expect("admission thread joins").expect_err("an expired attempt cannot publish");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    assert!(
        observed_admission_events(&observation).iter().any(|(_, state)| *state == AdmissionLifecycleState::Cancelled),
        "post-acquisition attempt deadline expiry records the cancellation lifecycle"
    );

    wait_for_transaction_cleanup(&service, "deadline rollback releases the complete provisional envelope");
    let ledger = device.ledger().snapshot().unwrap();
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(ledger.active_attempts, 0);
    assert_eq!(device.lane_permits().held_count(), 0);
    assert_eq!(root_out, 0);
}

/// The queue deadline bounds only the wait for ledger placement: a request
/// placed before it elapses publishes its envelope after it.
#[test]
fn queue_deadline_ends_at_placement() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for post-placement queue deadline");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterRootMember);
    let queue_deadline = Instant::now() + Duration::from_millis(200);
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_probe(
            QueryAdmissionRequest::test_attempt().with_queue_deadline(queue_deadline),
            Arc::new(probe),
        )
    });

    pause.wait_until_entered(Duration::from_secs(10));
    std::thread::sleep(queue_deadline.saturating_duration_since(Instant::now()));
    pause.release();
    let handle =
        admission.join().expect("admission thread joins").expect("a placed request publishes after its queue deadline");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    drop(handle);
}

/// A provisional ledger grant is necessary while the physical envelope is
/// assembled, but it must never be counted in the public admitted-handle
/// snapshot. This pauses after the root member exists and before worker-scope
/// completion, then observes the private transaction count separately.
#[test]
fn provisional_admission_is_not_visible_before_the_complete_envelope_publishes() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for provisional admission visibility");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterRootMember);
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_probe(QueryAdmissionRequest::test_attempt(), Arc::new(probe))
    });

    // The first rendezvous means the provisional grant, lane, stream, domain,
    // and root member have been acquired, but the envelope has not published.
    pause.wait_until_entered(Duration::from_secs(10));
    let snapshot = service.admission_snapshot().expect("public admission snapshot");
    assert_eq!(snapshot.active_queries, 0, "only complete handles are public admissions");
    assert_eq!(snapshot.queued_queries, 0);
    assert_eq!(
        service.inner.admission.state.lock().expect("admission state").lifecycle.snapshot().provisional,
        1,
        "the private transaction count still fences shutdown while the envelope is provisional"
    );

    pause.release();
    let handle = admission
        .join()
        .expect("admission thread must not panic")
        .expect("complete envelope must publish after the pause is released");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    handle.end_test_attempt();
    drop(handle);
    wait_for_transaction_cleanup(&service, "released provisional test handle must drain");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
}

/// A panic under the admission mutex while an envelope is being built makes
/// its publication refuse: the envelope rolls back, returning its grant and
/// lifecycle ticket.
#[test]
fn publication_after_admission_poison_rolls_back_the_envelope() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for poisoned publication");
    let device = service.inner.resources.devices.context_by_ordinal(0).expect("device context");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterRootMember);
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service.admit_query_with_probe(QueryAdmissionRequest::test_attempt(), Arc::new(probe))
    });
    pause.wait_until_entered(Duration::from_secs(10));
    let poisoner = service.clone();
    std::thread::spawn(move || {
        let _state = poisoner.inner.admission.state.lock().unwrap();
        panic!("poison the admission state while the envelope builds");
    })
    .join()
    .expect_err("the poisoning thread panics while holding the admission state");

    pause.release();
    let error = admission
        .join()
        .expect("admission thread must not panic")
        .expect_err("a poisoned admission state refuses publication");
    assert_eq!(error.fact_value("reason"), Some("admission_mutex_poisoned"));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let live_grants_bytes = device.ledger().snapshot().expect("ledger snapshot").live_grants_bytes;
        let quiescent =
            service.inner.admission.state.lock().unwrap_or_else(PoisonError::into_inner).lifecycle.is_quiescent();
        if live_grants_bytes == 0 && quiescent {
            break;
        }
        assert!(Instant::now() < deadline, "the rolled-back envelope releases its grant and lifecycle ticket");
        std::thread::yield_now();
    }
}

/// A published admission always carries the complete physical envelope. The
/// root stream is held from admission time (before any context acquisition),
/// the domain matches the grant, and the lane/worker identity is consistent.
#[test]
fn admitted_attempt_holds_complete_envelope_before_context_acquisition() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for complete admission envelope coverage");
    let resources = &service.inner.resources;
    let device = resources.devices.context_by_ordinal(0).expect("device context");

    let handle = service.admit_query().expect("admission");
    // No context was acquired, yet the attempt already holds its root stream.
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(root_out, 1, "root stream is held from admission time");
    assert_eq!(device.lane_permits().held_count(), 1, "attempt lane is held from admission time");
    let (domain_id, grant_id, worker_scope) = handle.lifetime.envelope_evidence().expect("envelope present");
    assert_eq!(domain_id, Some(grant_id));
    let worker_scope = worker_scope.expect("worker scope present");
    assert_eq!(worker_scope.device_ordinal(), 0);
    assert_eq!(worker_scope.lane_index(), 0, "first attempt takes lane zero");

    // The context consumes the envelope binding and does not create a stream
    // or domain late.
    let context = handle.acquire_context().expect("context consumes the envelope binding");
    assert_eq!(domain_id, Some(grant_id));
    drop(context);

    handle.end_test_attempt();
    drop(handle);
    wait_for_transaction_cleanup(&service, "release returns stream, lane, and ledger slot");
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    let lanes = device.lane_permits().held_count();
    let ledger = device.ledger().snapshot().expect("ledger snapshot");
    assert_eq!(root_out, 0);
    assert_eq!(lanes, 0);
    assert_eq!(ledger.active_attempts, 0);
}

/// An async admission builds its envelope's root stream, allocation domain,
/// and root member on the placed device's GPU pool. The caller's Tokio worker
/// does not run that build.
async fn async_admission_builds_its_envelope_on_the_device_gpu_pool_case() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for envelope placement");
    let probe = Arc::new(AdmissionTransactionProbe::disabled());

    let handle = service
        .admit_async_with_probe(QueryAdmissionRequest::test_attempt(), Arc::clone(&probe))
        .await
        .expect("async admission succeeds");

    let thread = probe.transaction_thread();
    assert!(
        thread.as_deref().is_some_and(|name| name.starts_with("algeon-gpu-")),
        "the envelope transaction ran on {thread:?}"
    );
    drop(handle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_admission_builds_its_envelope_on_the_device_gpu_pool() {
    async_admission_builds_its_envelope_on_the_device_gpu_pool_case().await;
    blocking_admission_builds_its_envelope_on_the_device_gpu_pool();
}

/// A blocking admission builds its envelope on the placed device's GPU pool,
/// as an async admission does. The calling thread does not run the build.
fn blocking_admission_builds_its_envelope_on_the_device_gpu_pool() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for envelope placement");
    let probe = Arc::new(AdmissionTransactionProbe::disabled());

    let handle = service
        .admit_query_with_probe(QueryAdmissionRequest::test_attempt(), Arc::clone(&probe))
        .expect("blocking admission succeeds");

    let thread = probe.transaction_thread();
    assert!(
        thread.as_deref().is_some_and(|name| name.starts_with("algeon-gpu-")),
        "the envelope transaction ran on {thread:?}"
    );
    handle.end_test_attempt();
}
