use crate::observability::{AttemptFailureFactKey, AttemptFailureFactValue, AttemptFailureOperation};

#[test]
fn unhealthy_device_rejects_query_admission_before_grant_or_queue() {
    let service = runtime_test_service(1, 8);
    let scope = DeviceSafetyTestScope::install([]);
    let device_ordinal = service.device_ledger_reports().unwrap()[0].device_ordinal;
    let device = injected_device_id(device_ordinal as i32).expect("isolated device health owner");
    report_device_fault(
        device,
        DeviceFaultKind::FatalDeviceOperation,
        "injected_query_admission_fault",
        None,
    );

    let error = service.try_admit_query().expect_err("unhealthy device must reject query admission");

    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    let expected_device_ordinal = device_ordinal.to_string();
    assert_eq!(error.fact_value("device_ordinal"), Some(expected_device_ordinal.as_str()));
    assert!(
        error
            .fact_value("device_health_detail")
            .is_some_and(|detail| detail.contains("injected_query_admission_fault")),
        "ledger-health diagnostic retains the native fault provenance: {error}"
    );
    let admission = service.admission_snapshot().unwrap();
    assert_eq!(admission.active_queries, 0);
    assert_eq!(admission.queued_queries, 0);
    assert_eq!(scope.health_snapshot(device).fault_count(), 1);
}
#[test]
fn opaque_activity_token_lives_until_query_service_release_finishes() {
    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let service = runtime_test_service(1, 8);
    let drops = Arc::new(AtomicUsize::new(0));
    let attempt = AttemptOwner::native_for_test();
    let handle = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(attempt.admission_binding())
                .with_runtime_observation_activity(QueryActivityLease::new(DropProbe(Arc::clone(&drops)))),
        )
        .expect("activity-bearing query admits");
    assert_eq!(drops.load(Ordering::SeqCst), 0, "admission cannot release the activity token early");

    drop(handle);
    assert_eq!(drops.load(Ordering::SeqCst), 0, "dropping the handle releases nothing");
    drop(attempt);

    // The envelope release, which drops the activity, runs on the device
    // executor after the attempt ends.
    let deadline = Instant::now() + Duration::from_secs(10);
    while drops.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "query-service cleanup releases the activity");
        std::thread::yield_now();
    }
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
    assert_eq!(drops.load(Ordering::SeqCst), 1, "final query-service cleanup releases the activity exactly once");
}

#[test]
fn finished_handle_releases_runtime_service_ownership_before_activity_drop() {
    struct ReplacementProbe(Arc<std::sync::Mutex<Option<std::result::Result<(), String>>>>);

    impl Drop for ReplacementProbe {
        fn drop(&mut self) {
            let result = try_guaranteed_runtime_test_service().map(drop).map_err(|error| error.to_string());
            *self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
        }
    }

    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let replacement = Arc::new(std::sync::Mutex::new(None));
    let attempt = AttemptOwner::native_for_test();
    let handle = service
        .admit_query_with_request(
            QueryAdmissionRequest::new(attempt.admission_binding())
                .with_runtime_observation_activity(QueryActivityLease::new(ReplacementProbe(Arc::clone(&replacement)))),
        )
        .expect("activity-bearing query admits");
    drop(service);

    drop(attempt);

    // The admitted envelope owns GPU resources, so the release;
    // including the activity drop; completes on the device executor. Wait
    // for the probe instead of expecting synchronous completion.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let probe_result = replacement.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(result) = probe_result {
            result.expect("finished handle released the old service observation before activity completion");
            break;
        }
        assert!(std::time::Instant::now() < deadline, "activity probe must run after the release completes");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    drop(handle);
}

#[tokio::test(flavor = "current_thread")]
async fn fallible_activity_acquisition_rejects_without_entering_admission() {
    let service = test_service(1, 8);
    let attempt = AttemptOwner::native_for_test();
    let error = service
        .admit_async_with_request(
            QueryAdmissionRequest::new(attempt.admission_binding()).with_runtime_observation_activity_result(Err(QueryError::new(
                QueryErrorCode::Cancelled,
                "test.query_service.observation_activity",
                "coordinated activity acquisition failed",
            )
            .with_fact("reason", "coordinator_closed"))),
        )
        .await
        .expect_err("fallible activity failure must reject admission");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.operation(), "test.query_service.observation_activity");
    assert_eq!(error.fact_value("reason"), Some("coordinator_closed"));
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0);
    assert_eq!(snapshot.queued_queries, 0);
}
#[test]
fn query_service_observes_enqueue_admit_and_release_under_one_admission_id() {
    let service = runtime_test_service(1, 8);
    let attempt = observed_attempt("external-query", Some("external-trace"), RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission enqueues");
    let handle = waiter.try_admit().expect("admission attempt succeeds").expect("capacity admits query");
    let admission_request_id = handle.admission_request_id();

    // The attempt's end, not the handle's drop, releases the admission.
    drop(handle);
    drop(attempt);
    wait_for_sealed_observation(&observation);

    assert_eq!(observation.query_id().map(|query_id| query_id.as_str()), Some("external-query"));
    assert_eq!(observation.trace_id().map(|trace_id| trace_id.as_str()), Some("external-trace"));
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::Admitted),
            (admission_request_id, AdmissionLifecycleState::Released),
        ]
    );
}
#[tokio::test(flavor = "current_thread")]
async fn async_capacity_contention_uses_one_id_without_false_rejection() {
    let service = runtime_test_service(1, 8);
    let active = service.admit_query().expect("capacity barrier admits first query");
    let attempt = observed_attempt("async-contended-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let queued_service = service.clone();
    let queued_binding = attempt.admission_binding();
    let queued = tokio::spawn(async move {
        queued_service.admit_async_with_request(QueryAdmissionRequest::new(queued_binding)).await
    });

    while service.admission_snapshot().unwrap().queued_queries != 1 {
        yield_now().await;
    }
    let enqueued = observed_admission_events(&observation);
    assert_eq!(enqueued.len(), 1);
    assert!(matches!(enqueued[0].1, AdmissionLifecycleState::Enqueued));

    drop(active);
    let admitted = queued.await.unwrap().unwrap();
    let admission_request_id = admitted.admission_request_id();
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::Admitted),
        ]
    );
    drop(admitted);
    assert!(
        !observed_admission_events(&observation)
            .iter()
            .any(|(_, lifecycle)| matches!(lifecycle, AdmissionLifecycleState::Rejected { .. }))
    );
}
#[tokio::test(flavor = "current_thread")]
async fn async_permanent_rejection_remains_terminal_rejected() {
    let service = test_service(1, 8);
    let attempt = observed_attempt("async-rejected-query", None, RuntimeObservationConfig::bounded(16, 16 * 1024));
    let observation = attempt.observation_for_test();

    // A requested cap above every compatible device's resident-adjusted
    // capacity rejects permanently before enqueue.
    let error = service
        .admit_async_with_request(
            QueryAdmissionRequest::new(attempt.admission_binding())
                .with_query_cap_bytes(128 * 1024 * 1024),
        )
        .await
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::QueryBudgetUnavailable);
    assert_eq!(error.fact_value("reason"), Some("query_cap_exceeds_resident_capacity"));
    let events = observed_admission_events(&observation);
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].1, AdmissionLifecycleState::Rejected { .. }));
}
#[test]
fn shared_owner_seals_only_after_admission_release_and_query_cleanup() {
    let service = runtime_test_service(1, 8);
    let attempt = observed_attempt("shared-owner-query", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
    let observation = attempt.observation_for_test();
    let handle = service
        .try_admit_query_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission attempt succeeds")
        .expect("capacity admits query");
    let lifecycle = RuntimePlanExecutionLifecycle::from_attempt(&attempt.handle(), PlanExecutionId::new("shared-plan"));

    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
    assert!(observation.sealed_snapshot().is_none(), "plan completion must not seal a shared query owner");

    attempt.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(handle);
    let snapshot = wait_for_sealed_observation(&observation);
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    let released = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::Admission(observation)
                    if observation.lifecycle == AdmissionLifecycleState::Released
            )
        })
        .expect("admission release is retained");
    let cleanup_finished = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished }
            )
        })
        .expect("cleanup finish is retained");
    let terminal = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(event.payload, RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Succeeded })
        })
        .expect("query success terminal is retained");
    assert!(released < cleanup_finished);
    assert!(cleanup_finished < terminal);
}
#[test]
fn query_service_observes_queued_waiter_drop_as_cancellation() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query admits");
    let attempt = observed_attempt("cancelled-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id();

    drop(waiter);

    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::CancellationRequested),
            (admission_request_id, AdmissionLifecycleState::Cancelled),
        ]
    );
    assert_eq!(service.admission_snapshot().expect("snapshot").queued_queries, 0);
    drop(active);
}

#[test]
fn queued_waiter_drop_recovers_a_poisoned_admission_guard_before_terminal_completion() {
    let service = test_service(1, 8);
    let attempt = observed_attempt("poisoned-waiter-drop-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission enqueues");
    let admission_request_id = waiter.admission_request_id();

    service.poison_admission();
    drop(waiter);

    assert_eq!(service.queued_len_recovering_poison(), 0, "poison recovery removes the owned queue entry");
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (
                admission_request_id,
                AdmissionLifecycleState::Rejected { reason: "admission_mutex_poisoned".to_string() },
            ),
        ]
    );
    assert!(matches!(
        &wait_for_sealed_observation(&observation).footer().terminal,
        Some(RuntimeTerminalOutcome::Failed { error_code, .. }) if *error_code == AttemptFailureCode::Query(QueryErrorCode::Internal)
    ));
}

#[test]
fn close_drains_queued_completion_from_a_poisoned_admission_guard() {
    let service = test_service(1, 8);
    let attempt = observed_attempt("poisoned-close-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission enqueues");

    service.poison_admission();
    let error = service.close().expect_err("poisoned admission remains a structured close failure");

    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.fact_value("reason"), Some("admission_mutex_poisoned"));
    assert_eq!(service.queued_len_recovering_poison(), 0, "close drains queued completion despite poison");
    assert!(matches!(&wait_for_sealed_observation(&observation).footer().terminal, Some(RuntimeTerminalOutcome::Cancelled)));
    drop(waiter);
}

#[test]
fn close_and_wait_quiesces_after_admission_poison() {
    let service = test_service(1, 8);
    let attempt = observed_attempt("poisoned-close-and-wait-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission enqueues");

    service.poison_admission();
    service
        .close_and_wait_for_quiescence(Duration::from_secs(1))
        .expect("the queue, preflight set, and lifecycle tickets stay valid after poison, so the drain quiesces");

    assert_eq!(service.queued_len_recovering_poison(), 0, "close-and-wait drains queued completion despite poison");
    assert!(matches!(&wait_for_sealed_observation(&observation).footer().terminal, Some(RuntimeTerminalOutcome::Cancelled)));
    drop(waiter);
}

#[test]
fn query_service_observes_pre_enqueue_rejection_without_reclassifying_error() {
    let service = test_service(1, 8);
    let attempt = observed_attempt("rejected-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();

    let error = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new(attempt.admission_binding())
                .with_query_cap_bytes(128 * 1024 * 1024),
        )
        .expect_err("an impossible requirement rejects before enqueue");

    assert_eq!(error.code(), QueryErrorCode::QueryBudgetUnavailable);
    let events = observed_admission_events(&observation);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0].1,
        AdmissionLifecycleState::Rejected { reason } if reason == "query_cap_exceeds_resident_capacity"
    ));
    assert_eq!(service.admission_snapshot().expect("snapshot").queued_queries, 0);
}
#[test]
fn query_service_observes_deterministic_queued_timeout_once() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query admits");
    let attempt = observed_attempt("timed-out-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id();
    service.expire_waiter(&mut waiter);

    let error = waiter.try_admit().expect_err("expired query times out");

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::TimedOut),
        ]
    );
    drop(active);
}

/// An admission snapshot only reads: an expired waiter is no longer counted
/// as queued, but the reader does not finish its request. The waiter's own
/// step does.
#[test]
fn admission_snapshot_leaves_an_expired_waiter_to_its_own_step() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query occupies the only slot");
    let attempt = observed_attempt("snapshot-read-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("second query queues");
    service.expire_waiter(&mut waiter);

    let snapshot = service.admission_snapshot().expect("admission snapshot");
    assert_eq!(snapshot.queued_queries, 0, "an expired waiter is not a queue member");
    assert_eq!(service.queued_len(), 1, "the snapshot leaves the expired request in place");
    assert!(observation.sealed_snapshot().is_none(), "the snapshot does not finish the expired request");

    let error = waiter.try_admit().expect_err("the waiter's own step reports its timeout");
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert_eq!(service.queued_len(), 0);
    drop(active);
}

/// A submission sweep can remove and terminalize a waiter before its observation next polls.
/// The waiter is allowed to take its structured error first, but that handoff
/// must not turn the owned completion payload into a dropped terminal outcome.
#[test]
fn sweep_finishes_timeout_after_waiter_consumes_terminal_error() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query occupies the only slot");
    let attempt = observed_attempt("swept-timeout-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id();
    service.expire_waiter(&mut waiter);

    let mut state = service.controller.lock_admission().expect("admission state");
    let sweep = service.controller.sweep_expired_queued_requests_locked(&mut state);
    drop(state);

    assert_eq!(service.queued_len(), 0, "the sweep removes the expired waiter before it finishes completion");
    assert!(observation.sealed_snapshot().is_none(), "the unfinished sweep still owns terminal completion");

    let error = waiter.try_admit().expect_err("waiter consumes its recorded timeout before sweep finish");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert!(observation.sealed_snapshot().is_none(), "waiter error handoff does not complete or drop the observation");

    sweep.finish();
    let snapshot = service.admission_snapshot().expect("snapshot succeeds");
    assert_eq!(snapshot.queued_queries, 0);

    let sealed = observation.sealed_snapshot().expect("sweep finalizes the queued completion");
    assert!(matches!(
        &sealed.footer().terminal,
        Some(RuntimeTerminalOutcome::Failed { error_code, .. }) if *error_code == AttemptFailureCode::Query(QueryErrorCode::ServiceOverloaded)
    ));
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::TimedOut),
        ]
    );

    drop(active);
}

/// The queue sweep must carry the borrowed native summary through the
/// attempt-owned completion even after the waiter has taken its rich error.
#[test]
fn sweep_preserves_attempt_binding_primary_failure_after_waiter_consumes_timeout() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query occupies the only slot");
    let attempt_lifecycle = BackendLifecycle::new();
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..BeginAttemptRequest::default()
        },
        attempt_lifecycle.try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    );
    attempt.owner.handle().set_disposition(AttemptDisposition::Native).expect("native disposition");
    let mut waiter = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new(attempt.owner.admission_binding()),
        )
        .expect("attempt-bound second query queues");
    service.expire_waiter(&mut waiter);

    let mut state = service.controller.lock_admission().expect("admission state");
    let sweep = service.controller.sweep_expired_queued_requests_locked(&mut state);
    drop(state);

    let error = waiter.try_admit().expect_err("waiter consumes its terminal timeout before sweep completion");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    assert!(attempt.completion.try_report().is_none(), "the unfinished sweep still owns attempt completion");

    sweep.finish();
    let admission = service.admission_snapshot().expect("snapshot succeeds");
    assert_eq!(admission.queued_queries, 0);

    let report = attempt.completion.completion().wait_timeout(Duration::from_secs(1)).expect("queue sweep seals the attempt report");
    assert!(matches!(
        &report.outcome,
        RuntimeTerminalOutcome::Failed { error_code, .. } if *error_code == AttemptFailureCode::Query(QueryErrorCode::ServiceOverloaded)
    ));
    let primary = report.primary_failure.as_ref().expect("queue sweep retains the borrowed primary failure");
    assert_eq!(primary.code(), AttemptFailureCode::Query(QueryErrorCode::ServiceOverloaded));
    assert_eq!(primary.operation(), AttemptFailureOperation::Admission);
    assert_eq!(primary.facts().len(), 1, "the stable report retains one closed admission reason");
    assert_eq!(primary.facts()[0].key(), AttemptFailureFactKey::Reason);
    assert_eq!(primary.facts()[0].value(), AttemptFailureFactValue::QueueWaitElapsed);
    assert_eq!(
        report.retry_advice,
        AttemptRetryAdvice {
            recovery: AttemptRetryRecovery::Backoff,
            prerequisite: AttemptRetryPrerequisite::CapacityAvailable,
        }
    );

    drop(active);
}

/// Health rejection carries both a waiter error and a rejection lifecycle.
/// Both must survive the same handoff race as a timeout.
#[test]
fn sweep_finishes_health_rejection_after_waiter_consumes_terminal_error() {
    let service = test_service(1, 8);
    let active = service.admit_query().expect("first query occupies the only slot");
    let attempt = observed_attempt("swept-health-rejection-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let mut waiter = service
        .enqueue_admission_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id();
    service
        .close_device_admission_for_test(0, "sweep-health-rejection")
        .expect("close the waiter's only compatible device");

    let mut state = service.controller.lock_admission().expect("admission state");
    let sweep = service.controller.sweep_expired_queued_requests_locked(&mut state);
    drop(state);

    assert_eq!(service.queued_len(), 0, "the sweep removes the health-rejected waiter");
    let error = waiter.try_admit().expect_err("waiter consumes its recorded health rejection before sweep finish");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    assert!(observation.sealed_snapshot().is_none(), "waiter error handoff does not drop the observation");

    sweep.finish();
    let snapshot = service.admission_snapshot().expect("snapshot succeeds");
    assert_eq!(snapshot.queued_queries, 0);

    let sealed = observation.sealed_snapshot().expect("sweep finalizes the health-rejected completion");
    assert!(matches!(
        &sealed.footer().terminal,
        Some(RuntimeTerminalOutcome::Failed { error_code, .. }) if *error_code == AttemptFailureCode::Query(QueryErrorCode::CudfPersistentFailure)
    ));
    assert_eq!(
        observed_admission_events(&observation),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (
                admission_request_id,
                AdmissionLifecycleState::Rejected { reason: "no_healthy_compatible_device".to_string() },
            ),
        ]
    );

    drop(active);
}
#[test]
fn query_service_context_uses_pre_admission_observation_identity() {
    let service = runtime_test_service(1, 8);
    let attempt = observed_attempt("external-context-query", Some("external-context-trace"), RuntimeObservationConfig::bounded(32, 32 * 1024));
    let observation = attempt.observation_for_test();
    let handle = service
        .try_admit_query_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission attempt succeeds")
        .expect("query admits");
    let admitted_context = handle.acquire_context().expect("context acquisition succeeds");
    let context = admitted_context.context();

    assert_eq!(context.query_id().map(QueryId::as_str), Some("external-context-query"));
    assert_eq!(context.trace_id().map(|trace_id| trace_id.as_str()), Some("external-context-trace"));
    assert_eq!(
        context.runtime_observation().live_snapshot().expect("context sees observation").manifest().query_id.as_str(),
        "external-context-query"
    );

    drop(admitted_context);
    attempt.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(1)).expect("stream pool becomes idle");
    let snapshot = wait_for_sealed_observation(&observation);
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    let admission_released = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::Admission(observation)
                    if observation.lifecycle == AdmissionLifecycleState::Released
            )
        })
        .expect("admission release is retained");
    assert_eq!(
        snapshot
            .events()
            .iter()
            .filter(|event| {
                matches!(
                    &event.payload,
                    RuntimeObservationPayload::Admission(observation)
                        if observation.lifecycle == AdmissionLifecycleState::Released
                )
            })
            .count(),
        1,
        "query admission must be released exactly once"
    );
    let cleanup_job_finished = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::GpuJob(observation)
                    if observation.job_kind == GpuExecutionJobKind::Cleanup.as_str()
                        && observation.lifecycle == GpuJobLifecycleState::Finished
            )
        })
        .expect("cleanup GPU job terminal is retained");
    let cleanup_finished = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                event.payload,
                RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished }
            )
        })
        .expect("query cleanup finish is retained");
    assert!(cleanup_job_finished < admission_released);
    assert!(admission_released < cleanup_finished);
}
#[test]
fn blocked_stream_cleanup_delays_query_seal_and_service_quiescence() {
    let service = runtime_test_service(1, 8);
    let attempt = observed_attempt("stream-release-before-seal", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
    let observation = attempt.observation_for_test();
    let handle = service
        .try_admit_query_with_request(QueryAdmissionRequest::new(attempt.admission_binding()))
        .expect("admission attempt succeeds")
        .expect("query admits");
    let admitted_context = handle.acquire_context().expect("context acquisition succeeds");
    drop(admitted_context);

    let stream_pool = service.inner.resources.devices.context_by_ordinal(0).unwrap().stream_pool();
    let cleanup_blocker = block_stream_cleanup_thread_for_test(stream_pool);
    attempt.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
    drop(handle);

    assert!(observation.sealed_snapshot().is_none(), "blocked stream cleanup keeps observation live");
    assert!(service.wait_for_stream_pools_idle(Duration::ZERO).is_err(), "pool cleanup remains blocked");
    let shutdown_error = service.close_and_wait_for_quiescence(Duration::ZERO).unwrap_err();
    assert_eq!(shutdown_error.fact_value("reason"), Some("query_service_shutdown_timeout"));

    drop(cleanup_blocker);
    service
        .close_and_wait_for_quiescence(Duration::from_secs(5))
        .expect("service quiescence completes only after blocked stream cleanup");

    let snapshot = wait_for_sealed_observation(&observation);
    assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Succeeded));
    let stream_events = snapshot
        .events()
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match &event.payload {
            RuntimeObservationPayload::StreamLease(observation) => Some((index, observation.lifecycle.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        stream_events.as_slice(),
        [
            (_, StreamLeaseLifecycleState::Enqueued),
            (_, StreamLeaseLifecycleState::Acquired { .. }),
            (_, StreamLeaseLifecycleState::Released),
        ]
    ));
    let stream_released = stream_events[2].0;
    let cleanup_finished = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(
                event.payload,
                RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished }
            )
        })
        .expect("query cleanup finish is retained");
    let terminal = snapshot
        .events()
        .iter()
        .position(|event| {
            matches!(event.payload, RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Succeeded })
        })
        .expect("query terminal is retained");
    assert!(stream_released < cleanup_finished);
    assert!(cleanup_finished < terminal);
}

/// A native attempt observed with `config`, as an adapter begins one before
/// admission.
fn observed_attempt(query_id: &str, trace_id: Option<&str>, config: RuntimeObservationConfig) -> AttemptOwner {
    let attempt = AttemptOwner::begin(
        BeginAttemptRequest {
            query_id: Some(AttemptQueryId::new(query_id).expect("valid attempt query id")),
            trace_context: trace_id.map(|trace_id| AttemptTraceContext::new(trace_id).expect("valid attempt trace id")),
            observation: config,
            ..BeginAttemptRequest::default()
        },
        BackendLifecycle::new().try_begin_observation().expect("attempt observation permit"),
        Arc::new(AttemptBackendCounters::default()),
        None,
    )
    .owner;
    attempt.handle().set_disposition(AttemptDisposition::Native).expect("native disposition");
    attempt
}
