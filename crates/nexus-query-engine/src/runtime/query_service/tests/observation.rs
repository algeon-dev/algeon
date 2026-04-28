#[test]
fn unhealthy_device_rejects_query_admission_before_grant_or_queue() {
    let service = runtime_test_service(1, 8);
    let scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
    let device = cudf::testing::device_safety::injected_device_id(service.config().device_ordinal as i32)
        .expect("isolated device health owner");
    cudf::runtime::report_device_fault(
        device,
        cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
        "injected_query_admission_fault",
        None,
    );

    let error = service.try_admit_query("small").expect_err("unhealthy device must reject query admission");

    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("device_health_rejected_admission"));
    let expected_device_ordinal = service.config().device_ordinal.to_string();
    assert_eq!(error.fact_value("device_ordinal"), Some(expected_device_ordinal.as_str()));
    assert_eq!(error.fact_value("first_device_fault_operation"), Some("injected_query_admission_fault"));
    let admission = service.admission_snapshot().unwrap();
    assert_eq!(admission.active_queries, 0);
    assert_eq!(admission.queued_queries, 0);
    assert_eq!(scope.health_snapshot(device).fault_count(), 1);
}
#[test]
fn coordinator_owned_handle_survives_three_legal_admission_epochs_without_intermediate_seal() {
    let service = test_service(1, 8);
    let owner = RuntimeObservationOwner::new(
        "coordinator-query",
        Some(RuntimeTraceId::new("coordinator-trace")),
        RuntimeObservationConfig::bounded(64, 64 * 1024),
    );
    let observation = owner.handle();

    for _ in 0..3 {
        let handle = service
            .admit_query_with_request(
                "small",
                QueryAdmissionRequest::new().with_runtime_observation_handle(observation.clone()),
            )
            .expect("each legal epoch admits against the shared observation handle");
        drop(handle);
        assert!(owner.sealed_snapshot().is_none(), "query service must not seal a coordinator-owned root");
    }

    let events = observed_admission_events(&owner);
    assert_eq!(
        events.iter().filter(|(_, lifecycle)| matches!(lifecycle, AdmissionLifecycleState::Admitted)).count(),
        3
    );
    assert_eq!(
        events.iter().filter(|(_, lifecycle)| matches!(lifecycle, AdmissionLifecycleState::Released)).count(),
        3
    );
    let snapshot = owner.finish_query(RuntimeTerminalOutcome::Succeeded).unwrap();
    assert_eq!(snapshot.manifest().query_id.as_str(), "coordinator-query");
}
#[test]
fn opaque_activity_token_lives_until_query_service_release_finishes() {
    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let service = test_service(1, 8);
    let drops = Arc::new(AtomicUsize::new(0));
    let handle = service
        .admit_query_with_request(
            "small",
            QueryAdmissionRequest::new()
                .with_runtime_observation_activity(QueryActivityLease::new(DropProbe(Arc::clone(&drops)))),
        )
        .expect("activity-bearing query admits");
    assert_eq!(drops.load(Ordering::SeqCst), 0, "admission cannot release the activity token early");

    drop(handle);

    assert_eq!(drops.load(Ordering::SeqCst), 1, "final query-service cleanup releases the activity exactly once");
}
#[tokio::test(flavor = "current_thread")]
async fn fallible_activity_acquisition_rejects_without_entering_admission() {
    let service = test_service(1, 8);
    let error = service
        .admit_async_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_activity_result(Err(QueryError::new(
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
    let service = test_service(1, 8);
    let owner = RuntimeObservationOwner::new(
        "external-query",
        Some(RuntimeTraceId::new("external-trace")),
        RuntimeObservationConfig::bounded(32, 32 * 1024),
    );
    let mut waiter = service
        .enqueue_admission_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("admission enqueues");
    let handle = waiter.try_admit().expect("admission attempt succeeds").expect("capacity admits query");
    let admission_request_id = handle.admission_request_id();

    drop(handle);

    assert_eq!(owner.query_id().map(|query_id| query_id.as_str()), Some("external-query"));
    assert_eq!(owner.trace_id().map(|trace_id| trace_id.as_str()), Some("external-trace"));
    assert_eq!(
        observed_admission_events(&owner),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::Admitted),
            (admission_request_id, AdmissionLifecycleState::Released),
        ]
    );
}
#[tokio::test(flavor = "current_thread")]
async fn async_capacity_contention_uses_one_id_without_false_rejection() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").expect("capacity barrier admits first query");
    let owner =
        RuntimeObservationOwner::new("async-contended-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let queued_service = service.clone();
    let queued_owner = owner.clone();
    let queued = tokio::spawn(async move {
        queued_service
            .admit_async_with_request(
                "small",
                QueryAdmissionRequest::new().with_runtime_observation_owner(queued_owner),
            )
            .await
    });

    loop {
        if service.inner.admission.lock().unwrap().queued.len() == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    let enqueued = observed_admission_events(&owner);
    assert_eq!(enqueued.len(), 1);
    assert!(matches!(enqueued[0].1, AdmissionLifecycleState::Enqueued));

    drop(active);
    let admitted = queued.await.unwrap().unwrap();
    let admission_request_id = admitted.admission_request_id();
    assert_eq!(
        observed_admission_events(&owner),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::Admitted),
        ]
    );
    drop(admitted);
    assert!(
        !observed_admission_events(&owner)
            .iter()
            .any(|(_, lifecycle)| matches!(lifecycle, AdmissionLifecycleState::Rejected { .. }))
    );
}
#[tokio::test(flavor = "current_thread")]
async fn async_permanent_rejection_remains_terminal_rejected() {
    let service = test_service(1, 8);
    let owner =
        RuntimeObservationOwner::new("async-rejected-query", None, RuntimeObservationConfig::bounded(16, 16 * 1024));

    let error = service
        .admit_async_with_request(
            "unknown-tenant",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .await
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    let events = observed_admission_events(&owner);
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].1, AdmissionLifecycleState::Rejected { .. }));
}
#[test]
fn shared_owner_seals_only_after_admission_release_and_query_cleanup() {
    let service = test_service(1, 8);
    let owner =
        RuntimeObservationOwner::new("shared-owner-query", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
    let handle = service
        .try_admit_query_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("admission attempt succeeds")
        .expect("capacity admits query");
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("shared-plan"));

    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
    assert!(owner.sealed_snapshot().is_none(), "plan completion must not seal a shared query owner");

    drop(handle);
    let snapshot = owner.sealed_snapshot().expect("final handle cleanup seals the shared owner");
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
    let active = service.admit_query("large").expect("first query admits");
    let owner = RuntimeObservationOwner::new("cancelled-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let waiter = service
        .enqueue_admission_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id;

    drop(waiter);

    assert_eq!(
        observed_admission_events(&owner),
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
fn query_service_observes_pre_enqueue_rejection_without_reclassifying_error() {
    let service = test_service(1, 8);
    let owner = RuntimeObservationOwner::new("rejected-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));

    let error = service
        .enqueue_admission_with_request(
            "unknown-tenant",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect_err("unknown tenant rejects before enqueue");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    let events = observed_admission_events(&owner);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0].1,
        AdmissionLifecycleState::Rejected { reason } if reason == QueryErrorCode::InvalidPlan.as_str()
    ));
    assert_eq!(service.admission_snapshot().expect("snapshot").queued_queries, 0);
}
#[test]
fn query_service_observes_deterministic_queued_timeout_once() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").expect("first query admits");
    let owner = RuntimeObservationOwner::new("timed-out-query", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
    let mut waiter = service
        .enqueue_admission_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("second query queues");
    let admission_request_id = waiter.admission_request_id;
    let expired = Instant::now();
    waiter.deadline = expired;
    service
        .inner
        .admission
        .lock()
        .expect("admission state")
        .queued
        .iter_mut()
        .find(|request| request.admission_request_id == admission_request_id)
        .expect("queued request")
        .deadline = expired;

    let error = waiter.try_admit().expect_err("expired query times out");

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(
        observed_admission_events(&owner),
        vec![
            (admission_request_id, AdmissionLifecycleState::Enqueued),
            (admission_request_id, AdmissionLifecycleState::TimedOut),
        ]
    );
    drop(active);
}
#[test]
fn query_service_context_uses_pre_admission_observation_identity() {
    let service = runtime_test_service(1, 8);
    let owner = RuntimeObservationOwner::new(
        "external-context-query",
        Some(RuntimeTraceId::new("external-context-trace")),
        RuntimeObservationConfig::bounded(32, 32 * 1024),
    );
    let handle = service
        .try_admit_query_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("admission attempt succeeds")
        .expect("query admits");
    let tenant_context = handle.acquire_context().expect("context acquisition succeeds");
    let context = tenant_context.context();

    assert_eq!(context.query_id().map(QueryId::as_str), Some("external-context-query"));
    assert_eq!(context.trace_id().map(|trace_id| trace_id.as_str()), Some("external-context-trace"));
    assert_eq!(
        context.runtime_observation().live_snapshot().expect("context sees observation").manifest().query_id.as_str(),
        "external-context-query"
    );

    owner.record_terminal_outcome(RuntimeTerminalOutcome::Succeeded);
    drop(tenant_context);
    handle.finish();
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(1)).expect("stream pool becomes idle");
    let snapshot = wait_for_sealed_observation(&owner);
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
    let owner = RuntimeObservationOwner::new(
        "stream-release-before-seal",
        None,
        RuntimeObservationConfig::bounded(64, 64 * 1024),
    );
    let handle = service
        .try_admit_query_with_request(
            "small",
            QueryAdmissionRequest::new().with_runtime_observation_owner(owner.clone()),
        )
        .expect("admission attempt succeeds")
        .expect("query admits");
    let tenant_context = handle.acquire_context().expect("context acquisition succeeds");
    owner.record_terminal_outcome(RuntimeTerminalOutcome::Succeeded);
    drop(tenant_context);

    let stream_pool = service.inner.resources.as_ref().unwrap().devices.context_by_ordinal(0).unwrap().stream_pool();
    let cleanup_blocker = super::super::service_stream::block_stream_cleanup_thread_for_test(stream_pool);
    handle.finish();
    drop(handle);

    assert!(owner.sealed_snapshot().is_none(), "blocked stream cleanup keeps observation live");
    assert!(service.wait_for_stream_pools_idle(Duration::ZERO).is_err(), "pool cleanup remains blocked");
    let shutdown_error = service.close_and_wait_for_quiescence(Duration::ZERO).unwrap_err();
    assert_eq!(shutdown_error.fact_value("reason"), Some("query_service_shutdown_timeout"));

    drop(cleanup_blocker);
    service
        .close_and_wait_for_quiescence(Duration::from_secs(5))
        .expect("service quiescence completes only after blocked stream cleanup");

    let snapshot = wait_for_sealed_observation(&owner);
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
