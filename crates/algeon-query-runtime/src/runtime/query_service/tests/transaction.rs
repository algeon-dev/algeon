// Transactional admission: fault-injection rollback and envelope
// identity evidence.

use crate::runtime::query_service::service::transaction::{AdmissionStep, AdmissionTransactionProbe};

/// Wait for the controller's complete cleanup boundary: returning streams are
/// not sufficient because the release commit still owns the ledger grant and
/// lane until it publishes the terminal report.
fn wait_for_transaction_cleanup(service: &QueryService, what: &str) {
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .unwrap_or_else(|error| panic!("{what}: {error}"));
}

/// Every injected transaction failure must restore the admission counters,
/// ledger state, attempt lanes, and stream permits exactly, leave the ledger
/// sequence in a valid observable state, and keep the service healthy for a
/// later admission.
#[test]
fn injected_fault_at_every_transaction_step_restores_all_counters_exactly() {
    for step in AdmissionStep::ALL {
        let service =
            try_guaranteed_runtime_test_service().expect("runtime-backed service for transaction rollback coverage");
        let resources = service.inner.resources.as_ref().expect("runtime-backed service");
        let device = resources.devices.context_by_ordinal(0).expect("device context");
        let ledger_before = device.ledger().snapshot().expect("ledger snapshot");
        assert_eq!(ledger_before.active_attempts, 0);
        assert_eq!(ledger_before.live_grants_bytes, 0);
        let (_, root_capacity, _, aux_capacity) = device.stream_pool().class_counters();

        service.set_admission_transaction_probe(
            AdmissionTransactionProbe::failing_at(step),
        );
        let error = service.admit_query().expect_err("injected fault rejects the admission");
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

        // The service stays healthy: a later admission publishes a complete
        // matching envelope.
        service.set_admission_transaction_probe(
            AdmissionTransactionProbe::disabled(),
        );
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
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let cleanup_blocker = block_stream_cleanup_thread_for_test(device.stream_pool());
    service.set_admission_transaction_probe(
        AdmissionTransactionProbe::failing_at(AdmissionStep::AfterRootMember),
    );

    let error = service.admit_query().expect_err("injected post-stream fault rejects admission");
    assert_eq!(error.fact_value("reason"), Some("injected_admission_step_fault"));
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(device.ledger().snapshot().unwrap().active_attempts, 1, "grant remains fenced by cleanup");
    assert_eq!(device.lane_permits().held_count(), 1, "lane remains fenced by cleanup");
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(root_out, 1, "root stream remains outstanding while cleanup is blocked");

    service.set_admission_transaction_probe(
        AdmissionTransactionProbe::disabled(),
    );
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
/// to rollback cleanup before the admission mutex is released.  Otherwise an
/// admission-only close could observe every counter as zero in the narrow
/// interval before `AdmissionEnvelope::drop` installs deferred cleanup.
#[test]
fn rejected_complete_envelope_keeps_close_from_reporting_false_quiescence() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for provisional rollback quiescence");
    let device = service
        .inner
        .resources
        .as_ref()
        .expect("runtime resources")
        .devices
        .context_by_ordinal(0)
        .expect("device context");
    let cleanup_blocker = block_stream_cleanup_thread_for_test(device.stream_pool());
    let (probe, rollback_pause) =
        AdmissionTransactionProbe::pausing_after_rollback_accounting();
    service.set_admission_transaction_probe(probe);
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || admission_service.admit_query());

    rollback_pause.wait_until_entered(Duration::from_secs(10));
    {
        let state = service.inner.admission.lock().expect("admission state after rollback ownership transfer");
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
    while service.inner.shutdown_waiters.load(std::sync::atomic::Ordering::Acquire) == 0 {
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
        .expect_err("forced publication expiry rejects the complete envelope");
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    drop(cleanup_blocker);
    close_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("close completes after deferred cleanup")
        .expect("close observes rollback cleanup quiescence");
    close.join().expect("close thread joins");
}

#[test]
fn transaction_deadline_expiry_after_acquisition_rolls_back_complete_envelope() {
    let service = try_guaranteed_runtime_test_service()
        .expect("runtime-backed service for post-acquisition deadline rollback coverage");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterRootMember);
    probe.force_deadline_after_acquisition_step(AdmissionStep::AfterRootMember);
    service.set_admission_transaction_probe(probe);
    let queue_deadline = Instant::now() + Duration::from_secs(10);
    let attempt_owner =
        observed_attempt("post-acquisition-deadline", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt_owner.observation_for_test();
    let binding = attempt_owner.admission_binding();
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service
            .admit_query_with_request(QueryAdmissionRequest::new(binding).with_queue_deadline(queue_deadline))
    });

    pause.wait_until_entered(Duration::from_secs(10));
    pause.release();
    let error = admission.join().expect("admission thread joins").expect_err("expired transaction cannot publish");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(error.fact_value("acquisition_stage"), Some("after_root_member"));
    assert!(
        observed_admission_events(&observation).iter().any(|(_, state)| *state == AdmissionLifecycleState::TimedOut),
        "post-acquisition deadline expiry records the timeout lifecycle"
    );

    wait_for_transaction_cleanup(&service, "deadline rollback releases the complete provisional envelope");
    let ledger = device.ledger().snapshot().unwrap();
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(ledger.active_attempts, 0);
    assert_eq!(device.lane_permits().held_count(), 0);
    assert_eq!(root_out, 0);
}

#[test]
fn deadline_is_rechecked_after_admission_mutex_reacquisition_before_publication() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for pre-publication deadline coverage");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let (probe, pause) = AdmissionTransactionProbe::pausing_at(AdmissionStep::AfterWorkerScope);
    probe.force_deadline_before_publication();
    service.set_admission_transaction_probe(probe);
    let queue_deadline = Instant::now() + Duration::from_secs(10);
    let attempt_owner =
        observed_attempt("pre-publication-deadline", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt_owner.observation_for_test();
    let binding = attempt_owner.admission_binding();
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || {
        admission_service
            .admit_query_with_request(QueryAdmissionRequest::new(binding).with_queue_deadline(queue_deadline))
    });

    pause.wait_until_entered(Duration::from_secs(10));
    let admission_state = service.inner.admission.lock().expect("hold admission publication mutex");
    pause.release();
    drop(admission_state);

    let error = admission.join().expect("admission thread joins").expect_err("expired envelope cannot publish");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(error.fact_value("acquisition_stage"), Some("before_publication"));
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert!(
        observed_admission_events(&observation).iter().any(|(_, state)| *state == AdmissionLifecycleState::TimedOut),
        "pre-publication deadline expiry records the timeout lifecycle"
    );

    wait_for_transaction_cleanup(&service, "pre-publication timeout rolls back the complete envelope");
    let ledger = device.ledger().snapshot().unwrap();
    let (root_out, _, _, _) = device.stream_pool().class_counters();
    assert_eq!(ledger.active_attempts, 0);
    assert_eq!(device.lane_permits().held_count(), 0);
    assert_eq!(root_out, 0);
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
    service.set_admission_transaction_probe(probe);
    let admission_service = service.clone();
    let admission = std::thread::spawn(move || admission_service.admit_query());

    // The first rendezvous means the provisional grant, lane, stream, domain,
    // and root member have been acquired, but the envelope has not published.
    pause.wait_until_entered(Duration::from_secs(10));
    let snapshot = service.admission_snapshot().expect("public admission snapshot");
    assert_eq!(snapshot.active_queries, 0, "only complete handles are public admissions");
    assert_eq!(snapshot.queued_queries, 0);
    assert_eq!(
        service.inner.admission.lock().expect("admission state").lifecycle.snapshot().provisional,
        1,
        "the private transaction count still fences shutdown while the envelope is provisional"
    );

    // Future admissions must not inherit the pause, while the in-flight probe
    // remains held until the second rendezvous releases it.
    service.set_admission_transaction_probe(
        AdmissionTransactionProbe::disabled(),
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

/// A published admission always carries the complete physical envelope: the
/// root stream is held from admission time (before any context acquisition),
/// the domain matches the grant, and the lane/worker identity is consistent.
#[test]
fn admitted_attempt_holds_complete_envelope_before_context_acquisition() {
    let service =
        try_guaranteed_runtime_test_service().expect("runtime-backed service for complete admission envelope coverage");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
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

    // The context consumes the envelope binding rather than creating a
    // stream or domain late.
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
/// and root member on the placed device's GPU pool, not on the caller's Tokio
/// worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_admission_builds_its_envelope_on_the_device_gpu_pool() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service for envelope placement");
    service.set_admission_transaction_probe(AdmissionTransactionProbe::disabled());

    let handle = service.admit_async().await.expect("async admission succeeds");

    let probe = service
        .inner
        .admission_transaction_probe
        .lock()
        .unwrap()
        .clone()
        .expect("installed transaction probe");
    let thread = probe.transaction_thread();
    assert!(
        thread.as_deref().is_some_and(|name| name.starts_with("algeon-gpu-")),
        "the envelope transaction ran on {thread:?}"
    );
    drop(handle);
}
