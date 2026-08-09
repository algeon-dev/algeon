#[test]
fn strict_query_service_context_uses_hard_limited_query_domain_members() {
    let config = QueryExecutionConfig { ..QueryExecutionConfig::default() };
    let service = runtime_test_service_with_profiles(
        config,
        QueueBounds::default(),
        vec![runtime_test_device_profile_with_limits(0, 64 * 1024 * 1024, 8 * 1024 * 1024, 1, 3, 1, 1)],
    )
    .unwrap();

    // An explicit working set pins the immutable grant for the domain-member
    // assertions below.
    let handle = service
        .admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(8 * 1024 * 1024, Some(8 * 1024 * 1024))),
        )
        .unwrap();
    let grant = handle.lifetime.memory_grant().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let token = admitted_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("test_strict_policy"))
        .unwrap();

    assert_eq!(token.enforcement(), cudf::ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(grant.budget_bytes(), 8 * 1024 * 1024);
    assert_eq!(admitted_context.context().attempt_allocation_domain_id(), Some(grant.grant_id()));
    let usage = admitted_context.context().attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!(usage.domain_id, grant.grant_id());

    // One member per stream for the attempt: sequential stage tokens share
    // it instead of attaching a second member, and their outputs stay valid
    // after the producing token drops.
    let outcome =
        admitted_context.context().runtime().unwrap().series_from_value_with_budget(7i32, 1024, &token).unwrap();
    assert_eq!(outcome.report().status(), cudf::ExecutionBudgetStatus::ReservationEnforced);
    let usage_after_alloc = admitted_context.context().attempt_allocation_domain_usage().unwrap().unwrap();
    assert!(usage_after_alloc.peak_bytes > 0);
    assert!(usage_after_alloc.active_members >= 1);
    drop(token);
    let shared_member_token = admitted_context
        .context()
        .execution_budget_token(
            cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("test_strict_policy_second_stage"),
        )
        .expect("the shared member accepts the next stage token on the same stream");
    let second = admitted_context
        .context()
        .runtime()
        .unwrap()
        .series_from_value_with_budget(9i32, 1024, &shared_member_token)
        .unwrap();
    assert_eq!(second.report().status(), cudf::ExecutionBudgetStatus::ReservationEnforced);
    drop(second);
    drop(outcome);
    drop(shared_member_token);
}
#[test]
fn expired_shutdown_deadline_reports_final_timeout() {
    let config = QueryExecutionConfig { ..QueryExecutionConfig::default() };
    let service = QueryService::new_with_deployment(config, QueueBounds::default(), non_gds_deployment()).unwrap();

    let outcome = service.close_drain_step_until(Instant::now());
    let final_completion = outcome
        .phases()
        .iter()
        .find(|phase| phase.phase() == QueryServiceDrainPhase::FinalCompletion)
        .expect("every drain pass reports final completion");

    assert!(matches!(final_completion, QueryServiceDrainPhaseOutcome::DeadlineElapsed { .. }));
    assert!(outcome.into_result().is_err());
}
#[test]
fn query_service_stream_runtimes_are_distinct_while_both_contexts_are_held() {
    let config = QueryExecutionConfig { ..QueryExecutionConfig::default() };
    let service = runtime_test_service_with_profiles(
        config,
        QueueBounds::default(),
        vec![runtime_test_device_profile(2)],
    )
    .unwrap();

    let first_handle = service.admit_query().unwrap();
    let second_handle = service.admit_query().unwrap();
    let first_context = first_handle.acquire_context().unwrap();
    let second_context = second_handle.acquire_context().unwrap();

    let first_runtime = first_context.context().shared_runtime().unwrap();
    let second_runtime = second_context.context().shared_runtime().unwrap();
    assert!(
        !Arc::ptr_eq(&first_runtime, &second_runtime),
        "holding the first context should force the second stream-pool runtime"
    );
}
#[test]
fn query_service_context_carries_graph_calibration_override() {
    let graph_memory_calibration_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
    let config = QueryExecutionConfig {
        graph_memory_calibration_override,
        // Runtime-backed services contend for process-global governor claims in
        // unit tests; optimistic claims can coexist across test cases.
        ..QueryExecutionConfig::default()
    };
    let service = QueryService::new_with_deployment(config, QueueBounds::default(), non_gds_deployment()).unwrap();

    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();

    assert_eq!(admitted_context.context().graph_memory_calibration_override(), graph_memory_calibration_override);
}

#[test]
fn attempt_bound_context_carries_its_engine_identity_for_gpu_execution_annotations() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let lifecycle = crate::runtime::BackendLifecycle::new();
    let attempt = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest::default(),
        lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default()),
        None,
    );
    let attempt_id = attempt.owner.attempt_id();
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle
        .set_disposition(crate::observability::AttemptDisposition::Native)
        .expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("attempt-bound admission");
    let grant_identity = handle.grant_identity().expect("admitted handle owns its grant identity");
    assert_eq!(grant_identity.attempt_id, attempt_id);
    attempt_handle
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let context = handle.acquire_context().expect("attempt-bound execution context");

    assert_eq!(context.context().attempt_id(), Some(attempt_id));
    assert_eq!(context.context().attempt_allocation_grant_identity(), Some(grant_identity));
    assert_eq!(
        context.context().memory_grant().unwrap().unwrap().identity(),
        context.context().attempt_allocation_grant_identity().unwrap()
    );

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("attempt terminalizes");
    drop(context);
    handle.finish();
    drop(handle);
    completion.wait_timeout(Duration::from_secs(10)).expect("attempt report publishes after cleanup");
}

#[cfg(feature = "cugraph")]
#[test]
fn cache_enabled_context_exposes_source_and_graph_leases_from_one_domain() {
    const CACHE_CAP_BYTES: u64 = 16 * 1024 * 1024;

    let service = runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds::default(),
        vec![runtime_test_device_profile_with_limits(0, 64 * 1024 * 1024, 8 * 1024 * 1024, 1, 3, 1, 1)
            .with_cache_cap_bytes(CACHE_CAP_BYTES)],
    )
    .expect("cache-enabled runtime service");
    let handle = service.admit_query().expect("admission");
    let context = handle.acquire_context().expect("admitted context");
    let source = context.context().resident_source_cache().expect("source cache lease is installed");
    let graph = context.context().resident_graph_cache().expect("graph cache lease is installed");

    assert_eq!(source.allocation_domain_id(), graph.cache_allocation_domain_id());
    assert_eq!(source.allocation_domain_id(), graph.cache_allocation_domain().id());
    assert_eq!(graph.cache_allocation_domain().capacity_bytes(), CACHE_CAP_BYTES);

    drop(context);
    handle.finish();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("cache-enabled service drains");
}
#[test]
fn query_service_released_handle_rejects_context_acquisition() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    handle.finish();

    let error = handle.acquire_context().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
}

#[test]
fn admitted_context_gate_rejects_missing_identity_device_and_scope_proofs() {
    let service = runtime_test_service(2, 8);
    let plan = explicit_sink_plan("input");
    let proof = crate::admission::QueryPlanCapabilityProof::for_query_plan(&plan)
        .expect("the test plan must issue an execution capability proof");
    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
        .expect("proof-backed admission");
    let admitted = handle.acquire_context().expect("service context");
    admitted
        .context()
        .validate_admitted_plan_capabilities(&plan)
        .expect("the immutable post-grant proof matches its context");

    let changed_plan = explicit_sink_plan("different-input");
    let error = admitted
        .context()
        .validate_admitted_plan_capabilities(&changed_plan)
        .expect_err("a changed plan must not reuse an admitted proof");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    let proof = admitted
        .context()
        .admitted_attempt_proof_for_test()
        .expect("the acquired context carries the proof")
        .clone();
    let grant_identity = proof.grant_identity();
    let mut wrong_generation = admitted.context().clone();
    wrong_generation.replace_admitted_attempt_proof_for_test(Some(Arc::new(proof.with_identity_for_test(
        proof.selected_device(),
        crate::runtime::gpu::GrantIdentity { generation: grant_identity.generation + 1, ..grant_identity },
    ))));
    let error = wrong_generation
        .validate_admitted_plan_capabilities(&plan)
        .expect_err("a recycled grant generation must not pass the context gate");
    assert_eq!(error.fact_value("reason"), Some("admitted_attempt_proof_grant_identity_mismatch"));

    let mut wrong_attempt = admitted.context().clone();
    wrong_attempt.replace_admitted_attempt_proof_for_test(Some(Arc::new(proof.with_identity_for_test(
        proof.selected_device(),
        crate::runtime::gpu::GrantIdentity {
            attempt_id: crate::observability::AttemptId::mint(),
            ..grant_identity
        },
    ))));
    let error = wrong_attempt
        .validate_admitted_plan_capabilities(&plan)
        .expect_err("a different attempt identity must not pass the context gate");
    assert_eq!(error.fact_value("reason"), Some("admitted_attempt_proof_grant_identity_mismatch"));

    let mut wrong_selected_device = admitted.context().clone();
    wrong_selected_device.replace_admitted_attempt_proof_for_test(Some(Arc::new(proof.with_identity_for_test(
        proof.selected_device() + 1,
        grant_identity,
    ))));
    let error = wrong_selected_device
        .validate_admitted_plan_capabilities(&plan)
        .expect_err("a proof selected for another device must not pass the context gate");
    assert_eq!(error.fact_value("reason"), Some("admitted_attempt_proof_device_mismatch"));

    let second_handle = service.admit_query().expect("second admission");
    let second_context = second_handle.acquire_context().expect("second context");
    let mut wrong_domain = admitted.context().clone();
    wrong_domain.replace_attempt_allocation_domain_for_test(second_context.context().attempt_allocation_domain_for_test());
    let error = wrong_domain
        .validate_admitted_plan_capabilities(&plan)
        .expect_err("a domain from another grant must not pass the context gate");
    assert_eq!(error.fact_value("reason"), Some("admitted_attempt_proof_domain_identity_mismatch"));

    let direct = ExecutionContext::builder().build().expect("direct context");
    direct
        .validate_admitted_plan_capabilities(&plan)
        .expect("direct contexts remain unrestricted by service proofs");
}

#[test]
fn placement_only_service_context_fails_closed_without_post_grant_proof() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().expect("placement-only admission");
    let context = handle.acquire_context().expect("placement-only context");
    let plan = explicit_sink_plan("input");

    let error = context
        .context()
        .validate_admitted_plan_capabilities(&plan)
        .expect_err("service execution requires post-grant proof evidence");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("admitted_attempt_proof_missing"));
}

#[tokio::test(flavor = "current_thread")]
async fn release_racing_root_context_binding_returns_a_structured_error_without_unwinding() {
    let service = runtime_test_service(1, 8);
    let handle = Arc::new(service.admit_query().expect("admission"));
    let gate = handle.lifetime.arm_root_context_binding_gate();
    let acquiring_handle = Arc::clone(&handle);
    let acquire = tokio::spawn(async move { acquiring_handle.acquire_context_async().await });

    let deadline = Instant::now() + Duration::from_secs(10);
    while !gate.entered() {
        assert!(Instant::now() < deadline, "root context setup consumes its binding");
        tokio::task::yield_now().await;
    }
    handle.finish();
    gate.open();

    let error = acquire
        .await
        .expect("a release racing root-context setup must not unwind")
        .expect_err("release invalidates the in-flight root-context setup");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_queued_root_context_setup_rolls_back_the_in_flight_claim() {
    let service = try_guaranteed_runtime_test_service().expect("single-worker guaranteed service");
    let handle = Arc::new(service.admit_query().expect("admission"));
    let pool = handle.gpu_execution_pool().expect("counted GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);

    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release);
    let (blocker_started_tx, blocker_started_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = blocker_started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("blocker release semaphore remains open");
    })
    .expect("auxiliary blocker submits");
    blocker_started_rx.await.expect("auxiliary blocker starts");

    let acquiring_handle = Arc::clone(&handle);
    let acquisition = tokio::spawn(async move { acquiring_handle.acquire_context_async().await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.snapshot().queued_jobs == 0 {
        assert!(Instant::now() < deadline, "root context setup must queue behind the worker blocker");
        tokio::task::yield_now().await;
    }

    acquisition.abort();
    assert!(acquisition.await.expect_err("aborted root setup task must not complete").is_cancelled());

    release.add_permits(1);
    let admitted_context = tokio::time::timeout(Duration::from_secs(10), handle.acquire_context_async())
        .await
        .expect("a subsequent acquisition must not wait on an abandoned root claim")
        .expect("a subsequent acquisition establishes the root context");

    drop(admitted_context);
    handle.finish();
    drop(pool);
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn expired_deadline_stops_secondary_stream_acquisition_before_consuming_a_permit() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let handle = service.admit_query().expect("admission");
    let root_context = handle.acquire_context().expect("root context");
    let counters_before = device.stream_pool().class_counters();

    let error = handle
        .acquire_context_with_cancellation(ExecutionCancellation::with_deadline(Some(
            Instant::now() - Duration::from_millis(1),
        )))
        .expect_err("an expired attempt deadline must not acquire a secondary stream");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    assert_eq!(device.stream_pool().class_counters(), counters_before, "no auxiliary stream permit was consumed");

    drop(root_context);
    handle.finish();
    drop(handle);
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .expect("expired attempt context cleanup returns the root stream");
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[tokio::test(flavor = "current_thread")]
async fn deadline_expiry_while_context_setup_waits_stops_the_worker_and_cleanup_still_releases() {
    let service = try_guaranteed_runtime_test_service().expect("single-worker runtime service");
    let resources = service.inner.resources.as_ref().expect("runtime-backed service");
    let device = resources.devices.context_by_ordinal(0).expect("device context");
    let handle = Arc::new(service.admit_query().expect("admission"));
    let pool = handle.gpu_execution_pool().expect("counted GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);

    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release);
    let (blocker_started_tx, blocker_started_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = blocker_started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("blocker release semaphore remains open");
    })
    .expect("worker blocker submits");
    blocker_started_rx.await.expect("worker blocker starts");

    let deadline = Instant::now() + Duration::from_millis(100);
    let cancellation = ExecutionCancellation::with_deadline(Some(deadline));
    let acquiring_handle = Arc::clone(&handle);
    let acquisition =
        tokio::spawn(async move { acquiring_handle.acquire_context_with_cancellation_async(cancellation).await });
    let queue_observation_deadline = Instant::now() + Duration::from_secs(10);
    while pool.snapshot().queued_jobs == 0 {
        assert!(Instant::now() < queue_observation_deadline, "context setup must reach the GPU worker queue");
        tokio::task::yield_now().await;
    }

    tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    release.add_permits(1);
    let error = tokio::time::timeout(Duration::from_secs(10), acquisition)
        .await
        .expect("queued context setup returns after the worker is released")
        .expect("queued context setup task does not panic")
        .expect_err("deadline-expired worker setup must not create a native context");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));

    let recovery_context = tokio::time::timeout(Duration::from_secs(10), handle.acquire_context_async())
        .await
        .expect("a fresh context setup does not inherit the expired worker claim")
        .expect("a fresh context setup consumes the original root binding");
    let (root_outstanding, _, auxiliary_outstanding, _) = device.stream_pool().class_counters();
    assert_eq!(
        (root_outstanding, auxiliary_outstanding),
        (1, 0),
        "the expired worker did not consume the root binding or create an auxiliary stream"
    );
    drop(recovery_context);

    handle.finish();
    drop(pool);
    drop(handle);
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .expect("deadline expiry does not cancel admission cleanup");
    let replacement = service.admit_query().expect("cleanup returns the expired attempt's admission capacity");
    replacement.finish();
    drop(replacement);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn query_service_context_keeps_admission_until_context_drop() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    drop(handle);

    assert!(service.try_admit_query().unwrap().is_none());
    drop(admitted_context);
    service.admit_query().expect("cleanup should wake the blocked admission");
}
#[test]
fn finished_query_rejects_context_derivations_until_escaped_context_drops() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let escaped_context = admitted_context.context().clone();
    drop(admitted_context);

    handle.finish();

    for error in [
        escaped_context.runtime().unwrap_err(),
        escaped_context.memory_grant().unwrap_err(),
        escaped_context.gpu_execution_pool().unwrap_err(),
        escaped_context.service_stream_pool().acquire().unwrap_err(),
    ] {
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    }
    assert!(service.try_admit_query().unwrap().is_none());

    drop(escaped_context);
    let next = service.admit_query().expect("the final context anchor releases admission");
    drop(next);
}
#[test]
fn finish_vs_runtime_derivation_is_linearizable_and_releases_once() {
    let service = runtime_test_service(1, 8);
    for _ in 0..16 {
        let handle = Arc::new(service.admit_query().unwrap());
        let admitted_context = handle.acquire_context().unwrap();
        let escaped_context = admitted_context.context().clone();
        drop(admitted_context);
        let barrier = Arc::new(Barrier::new(3));
        let finish_handle = Arc::clone(&handle);
        let finish_barrier = Arc::clone(&barrier);
        let finish = std::thread::spawn(move || {
            finish_barrier.wait();
            finish_handle.finish();
        });
        let derive_barrier = Arc::clone(&barrier);
        let derive = std::thread::spawn(move || {
            derive_barrier.wait();
            escaped_context.runtime()
        });

        barrier.wait();
        finish.join().unwrap();
        let runtime = match derive.join().unwrap() {
            Ok(runtime) => Some(runtime),
            Err(error) => {
                assert_eq!(error.code(), QueryErrorCode::Cancelled);
                assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
                None
            },
        };
        drop(handle);
        if runtime.is_some() {
            assert!(
                service.try_admit_query().unwrap().is_none(),
                "a runtime linearized before finish must retain admission"
            );
        }
        drop(runtime);
        let next = service.admit_query().expect("each raced query releases admission exactly once");
        drop(next);
    }
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
}
#[test]
fn absolute_shutdown_deadline_includes_admission_close_lock_contention() {
    let service = test_service(1, 8);
    let admission = service.hold_admission_lock().unwrap();
    let deadline = Instant::now() + Duration::from_millis(20);
    let waiting_service = service.clone();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let wait = std::thread::spawn(move || {
        result_tx.send(waiting_service.close_and_wait_for_quiescence_until(deadline)).unwrap();
    });

    let result = result_rx.recv_timeout(Duration::from_millis(200));
    drop(admission);
    wait.join().unwrap();
    let error = result
        .expect("shutdown must return its mutex timeout before the contended lock is released")
        .expect_err("admission mutex contention must consume the shutdown deadline");

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(error.fact_value("deadline_stage"), Some("admission_mutex"));
    assert_eq!(
        service.try_admit_query().unwrap_err().fact_value("reason"),
        Some("admission_closed"),
        "admission-lock timeout must leave the service fail-closed"
    );

    service.close_and_wait_for_quiescence(Duration::from_secs(1)).unwrap();
}
#[test]
fn close_signal_is_nonblocking_and_fail_closed_during_admission_lock_contention() {
    let service = test_service(1, 8);
    let admission = service.hold_admission_lock().unwrap();
    let closing_service = service.clone();
    let (closed_tx, closed_rx) = std::sync::mpsc::channel();
    let close = std::thread::spawn(move || closed_tx.send(closing_service.close()).unwrap());

    let close_result = closed_rx.recv_timeout(Duration::from_millis(50));
    drop(admission);
    let close_result = close_result.expect("close signal must not wait for the admission-state mutex");
    close.join().unwrap();
    close_result.unwrap();
    assert_eq!(service.try_admit_query().unwrap_err().fact_value("reason"), Some("admission_closed"));
}
#[test]
fn resource_free_expired_shutdown_completes_epoch_cleanup_but_cannot_succeed() {
    let service = test_service(1, 8);
    let error = service
        .close_and_wait_for_quiescence_until(Instant::now())
        .expect_err("the complete shutdown cannot succeed after its absolute deadline");

    assert_eq!(error.fact_value("reason"), Some("query_service_resource_epoch_timeout"));
    assert_eq!(service.try_admit_query().unwrap_err().fact_value("reason"), Some("admission_closed"));
}
#[test]
fn quiescence_reached_after_absolute_deadline_is_not_accepted() {
    let service = test_service(1, 8);
    service.hold_admission_lock().unwrap().set_active_queries(1);
    let deadline = Instant::now() + Duration::from_millis(100);
    let waiting_service = service.clone();
    let wait = std::thread::spawn(move || waiting_service.close_and_wait_for_quiescence_until(deadline));
    while service.shutdown_waiter_count() == 0 {
        std::thread::yield_now();
    }

    let mut state = service.hold_admission_lock().unwrap();
    let release_at = deadline + Duration::from_millis(20);
    std::thread::sleep(release_at.saturating_duration_since(Instant::now()));
    state.set_active_queries(0);
    service.notify_admission_available();
    drop(state);

    let error = wait.join().unwrap().expect_err("late quiescence must preserve the component timeout");
    assert_eq!(error.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(error.fact_value("deadline_stage"), Some("admission_quiescence"));
}
#[test]
fn query_service_handle_binds_secondary_context_to_dedicated_stream() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    let first = handle.acquire_context().unwrap();
    let second = handle.acquire_context().unwrap();
    let first_runtime = first.context().shared_runtime().unwrap();
    let second_runtime = second.context().shared_runtime().unwrap();

    // Secondary acquires keep the same admitted device while binding a dedicated
    // CUDA stream so concurrent native fragments can each create a live
    // query-domain budget token (cuDF rejects two live tokens on one stream).
    assert_eq!(first_runtime.placement().device, second_runtime.placement().device);
    assert!(!Arc::ptr_eq(&first_runtime, &second_runtime));
    let lease = first_runtime.lease_anchor();
    assert!(!second_runtime.matches_lease_anchor(&lease));
    assert_ne!(first_runtime.placement().stream_debug, second_runtime.placement().stream_debug);

    // Concurrent live query-domain budget tokens must succeed on both contexts.
    let left = first
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("secondary_stream_left"))
        .expect("first fragment budget token");
    let right = second
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("secondary_stream_right"))
        .expect("second fragment budget token");
    drop(left);
    drop(right);
}
#[test]
fn escaped_context_causes_bounded_shutdown_timeout_until_final_anchor_drops() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let retained_context = admitted_context.context().clone();
    drop(admitted_context);
    drop(handle);

    let timeout = service.close_and_wait_for_quiescence(Duration::from_millis(10)).unwrap_err();
    assert_eq!(timeout.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(timeout.fact_value("deadline_stage"), Some("admission_quiescence"));

    for error in [
        retained_context.runtime().unwrap_err(),
        retained_context.memory_grant().unwrap_err(),
        retained_context.gpu_execution_pool().unwrap_err(),
    ] {
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("query_service_epoch_closed"));
    }

    drop(retained_context);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    drop(service);
}
#[test]
fn held_runtime_and_produced_object_delay_service_drain_until_last_provenance_drop() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let retained_context = admitted_context.context().clone();
    let runtime = retained_context.runtime().unwrap();
    let series = runtime.series(&[1_i32, 2, 3]).expect("produce a retained GPU series");
    drop(runtime);
    drop(admitted_context);
    drop(retained_context);
    drop(handle);

    let timeout = service.close_and_wait_for_quiescence(Duration::from_millis(10)).unwrap_err();
    assert_eq!(timeout.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(timeout.fact_value("deadline_stage"), Some("admission_quiescence"));

    drop(series);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    drop(service);
}
#[test]
fn execution_budget_token_retains_query_domain_until_drop() {
    let service = try_guaranteed_runtime_test_service().expect("guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let token = admitted_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1).with_boundary("query_epoch_retention"))
        .expect("query-domain execution budget token");
    drop(admitted_context);
    handle.finish();

    assert!(
        service.try_admit_query().unwrap().is_none(),
        "the budget token's retained runtime must keep query admission"
    );
    drop(token);
    let next = service.admit_query().expect("dropping the budget token releases query admission");
    drop(next);
}
#[test]
fn retained_service_stream_pool_keeps_query_admitted_and_rejects_post_finish_acquire() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let retained_pool = admitted_context.context().service_stream_pool();
    drop(admitted_context);
    handle.finish();

    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    let error = retained_pool.acquire().expect_err("closed query epoch must reject stream acquisition");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    drop(retained_pool);
    let replacement = service.admit_query().expect("final stream-pool anchor releases query admission");
    drop(replacement);
}
#[test]
fn acquired_service_stream_and_derived_runtime_hold_query_until_last_counted_drop() {
    use cudf::testing::prelude::RuntimeTestingExt;

    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let retained_pool = admitted_context.context().service_stream_pool();
    let stream_lease = retained_pool.acquire().expect("service stream acquisition");
    let runtime = stream_lease.runtime().expect("counted derived runtime");
    let shared_runtime = stream_lease.shared_runtime().expect("counted shared derived runtime");

    assert_eq!(runtime.testing_retained_lifetime_count(), 1);
    assert_eq!(shared_runtime.testing_retained_lifetime_count(), 1);
    drop(shared_runtime);
    drop(retained_pool);
    drop(admitted_context);
    handle.finish();

    let error = stream_lease.runtime().expect_err("query finish must reject new runtime derivation from the lease");
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    assert!(service.try_admit_query().unwrap().is_none());

    drop(stream_lease);
    assert!(
        service.try_admit_query().unwrap().is_none(),
        "the runtime derived before finish retains the query after its stream is released"
    );

    drop(runtime);
    let replacement = service.admit_query().expect("the final stream runtime releases query admission");
    drop(replacement);
}
#[test]
fn cloned_memory_grant_and_live_reservation_hold_query_and_reject_post_finish_work() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    // The grant clamps to the request's soft working-set hint, exercising the
    // immutable budget independent of the full candidate maximum.
    let handle = service
        .admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(requirement(8 * 1024 * 1024, Some(8 * 1024 * 1024))),
        )
        .unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let grant = admitted_context.context().memory_grant().unwrap().expect("service context carries a memory grant");
    let retained_grant = grant.clone();
    let retained_reservation = retained_grant
        .retain_live_reservation(1)
        .expect("retention fits the immutable grant")
        .expect("positive reservation is retained");
    drop(grant);
    drop(admitted_context);
    handle.finish();

    assert_eq!(retained_grant.budget_bytes(), 8 * 1024 * 1024, "immutable grant budget survives epoch closure");
    assert!(service.try_admit_query().unwrap().is_none());

    drop(retained_grant);
    assert!(
        service.try_admit_query().unwrap().is_none(),
        "the live reservation's cloned grant retains query admission"
    );

    drop(retained_reservation);
    let replacement = service.admit_query().expect("the final memory-grant anchor releases admission");
    drop(replacement);
}
#[tokio::test(flavor = "current_thread")]
async fn cloned_gpu_pool_job_holds_query_and_rejects_post_finish_submission() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let pool = admitted_context.context().gpu_execution_pool().expect("counted GPU execution pool");
    let retained_pool = pool.clone();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let job_release = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = started_tx.send(());
        let _permit = job_release.acquire_owned().await.expect("job release semaphore remains open");
        let _ = finished_tx.send(());
    })
    .unwrap();
    started_rx.await.expect("GPU job starts");
    drop(pool);
    drop(admitted_context);
    handle.finish();

    let spawn_error = retained_pool
        .spawn(GpuExecutionJobKind::QueryExecution, async {})
        .expect_err("closed query epoch must reject detached GPU work");
    assert_eq!(spawn_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(spawn_error.fact_value("reason"), Some("query_handle_released"));
    let run_error = retained_pool
        .run(GpuExecutionJobKind::QueryExecution, async {})
        .await
        .expect_err("closed query epoch must reject awaited GPU work");
    assert_eq!(run_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(run_error.fact_value("reason"), Some("query_handle_released"));
    drop(retained_pool);

    assert!(service.try_admit_query().unwrap().is_none(), "the in-flight GPU job must retain query admission");

    release.add_permits(1);
    finished_rx.await.expect("GPU job finishes after release");
    let replacement = service.admit_query().expect("the completed GPU job releases query admission");
    drop(replacement);
}
#[tokio::test(flavor = "current_thread")]
async fn queued_gpu_pool_job_retains_query_without_parking_cleanup_on_worker_permit() {
    let service = try_guaranteed_runtime_test_service().expect("single-worker guaranteed service");
    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let pool = admitted_context.context().gpu_execution_pool().expect("counted GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release);
    let (blocker_started_tx, blocker_started_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = blocker_started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("blocker release remains open");
    })
    .unwrap();
    blocker_started_rx.await.expect("blocking GPU job starts");

    let (queued_finished_tx, queued_finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = queued_finished_tx.send(());
    })
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while pool.snapshot().queued_jobs == 0 {
        assert!(Instant::now() < deadline, "second GPU job did not enter the worker queue");
        std::thread::yield_now();
    }

    drop(admitted_context);
    handle.finish();
    drop(pool);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    release.add_permits(1);
    queued_finished_rx.await.expect("queued job runs after the worker permit is released");
    let next = service.admit_query().expect("queued and running anchors drain before cleanup");
    drop(next);
}
#[tokio::test(flavor = "current_thread")]
async fn query_handle_gpu_pool_is_counted_and_rejects_work_after_finish() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let retained_pool = handle.gpu_execution_pool().expect("counted query-handle GPU execution pool");

    handle.finish();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    let spawn_error = retained_pool
        .spawn(GpuExecutionJobKind::QueryExecution, async {})
        .expect_err("closed query epoch must reject detached query-handle GPU work");
    assert_eq!(spawn_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(spawn_error.fact_value("reason"), Some("query_handle_released"));
    let run_error = retained_pool
        .run(GpuExecutionJobKind::QueryExecution, async {})
        .await
        .expect_err("closed query epoch must reject awaited query-handle GPU work");
    assert_eq!(run_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(run_error.fact_value("reason"), Some("query_handle_released"));

    drop(retained_pool);
    let replacement = service.admit_query().expect("the final query-handle GPU pool drop releases admission");
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_without_runtime_context_releases_through_the_device_executor() {
    // Every device-backed admission owns its root stream and domain
    // from admission time, so a handle that never acquired a context still
    // releases through the device executor.
    let service = try_guaranteed_runtime_test_service().expect("single-worker guaranteed service");
    let pool = service
        .inner
        .resources
        .as_ref()
        .unwrap()
        .gpu_execution_pool_for_device(0)
        .expect("service owns the device GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);

    let release_blocker = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release_blocker);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        let _ = started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("cleanup blocker remains open");
        let _ = finished_tx.send(());
    })
    .expect("cleanup blocker queues on the service pool");
    started_rx.await.expect("cleanup blocker starts");

    let direct = service.admit_query().expect("direct admission occupies the only slot");
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(1);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission must queue behind the direct handle");
        tokio::task::yield_now().await;
    }

    direct.finish();
    drop(direct);
    // The release owns the envelope's root stream and domain, so it queues
    // behind the blocked device executor and must not unblock the queued
    // admission yet.
    let mut queued = queued;
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut queued).await.is_err(),
        "envelope-owning release must wait for the busy GPU worker"
    );

    release_blocker.add_permits(1);
    finished_rx.await.expect("cleanup blocker finishes after release");
    let admitted = tokio::time::timeout(Duration::from_secs(10), queued)
        .await
        .expect("queued admission completes once the device executor runs the release")
        .expect("queued admission task must not panic")
        .expect("queued admission succeeds after the direct release");
    drop(admitted);
}

#[tokio::test(flavor = "current_thread")]
async fn root_context_release_waits_for_busy_gpu_worker() {
    let service = try_guaranteed_runtime_test_service().expect("single-worker guaranteed service");
    let handle = service.admit_query().expect("context-backed admission");
    let context = handle.acquire_context().expect("root context acquisition");
    drop(context);

    let pool = service
        .inner
        .resources
        .as_ref()
        .unwrap()
        .gpu_execution_pool_for_device(0)
        .expect("service owns the device GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);
    let release_blocker = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release_blocker);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::Cleanup, async move {
        let _ = started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("cleanup blocker remains open");
    })
    .expect("cleanup blocker queues on the service pool");
    started_rx.await.expect("cleanup blocker starts");

    let queued_service = service.clone();
    let mut queued = tokio::spawn(async move { queued_service.admit_async().await });
    let deadline = Instant::now() + Duration::from_secs(1);
    while service.admission_snapshot().unwrap().queued_queries != 1 {
        assert!(Instant::now() < deadline, "second admission must queue behind the context-backed handle");
        tokio::task::yield_now().await;
    }

    handle.finish();
    drop(handle);
    assert!(
        tokio::time::timeout(Duration::from_millis(250), &mut queued).await.is_err(),
        "root-context cleanup must wait for the busy GPU worker"
    );

    release_blocker.add_permits(1);
    let admitted = tokio::time::timeout(Duration::from_secs(1), queued)
        .await
        .expect("GPU cleanup releases the queued admission")
        .expect("queued admission task must not panic")
        .expect("queued admission succeeds after root-context cleanup");
    drop(admitted);
}

#[tokio::test(flavor = "current_thread")]
async fn query_handle_gpu_job_retains_query_epoch_until_joined() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2).expect("two-worker guaranteed service");
    let handle = service.admit_query().unwrap();
    let retained_pool = handle.gpu_execution_pool().expect("counted query-handle GPU execution pool");
    assert_eq!(retained_pool.snapshot().worker_count, 2);
    let job_pool = retained_pool.clone();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let job_release = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let job = tokio::spawn(async move {
        job_pool
            .run(GpuExecutionJobKind::QueryExecution, async move {
                let _ = started_tx.send(());
                let _permit = job_release.acquire_owned().await.expect("job release semaphore remains open");
            })
            .await
    });
    started_rx.await.expect("GPU job starts");

    handle.finish();
    drop(retained_pool);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    release.add_permits(1);
    job.await.expect("GPU job task joins").expect("GPU job succeeds after release");
    let replacement = service.admit_query().expect("joining the final GPU job releases query admission");
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn close_during_admitted_execution_waits_for_the_job_and_seals_before_quiescence() {
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let backend_lifecycle = crate::runtime::BackendLifecycle::new();
    let counters = std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default());
    let attempt = crate::attempt::AttemptOwner::begin(
        crate::attempt::BeginAttemptRequest {
            observation: RuntimeObservationConfig::bounded(32, 32 * 1024),
            ..crate::attempt::BeginAttemptRequest::default()
        },
        backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
        std::sync::Arc::clone(&counters),
        None,
    );
    let attempt_owner = attempt.owner;
    let completion = attempt.completion;
    let attempt_handle = attempt_owner.handle();
    attempt_handle.record_planning_started().expect("planning starts");
    attempt_handle.record_planning_finished().expect("planning finishes");
    attempt_handle.set_disposition(crate::observability::AttemptDisposition::Native).expect("native disposition");
    attempt_handle.record_admission_requested().expect("admission requested");

    let handle = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(attempt_owner.admission_binding()))
        .expect("attempt-bound admission");
    attempt_handle
        .record_admitted(crate::observability::AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("attempt admission fact");
    let context = handle.acquire_context().expect("admitted execution context");
    attempt_handle.record_physical_execution_started().expect("physical execution starts");
    attempt_handle.record_gpu_execution_started().expect("GPU execution starts");

    let pool = handle.gpu_execution_pool().expect("counted query GPU execution pool");
    let job_pool = pool.clone();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let job_release = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let execution = tokio::spawn(async move {
        job_pool
            .run(GpuExecutionJobKind::QueryExecution, async move {
                let _ = started_tx.send(());
                let _permit = job_release.acquire_owned().await.expect("execution release semaphore remains open");
            })
            .await
    });
    started_rx.await.expect("admitted execution job starts");

    service.close().expect("close stops new admission while admitted execution continues");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1, "live execution retains its admission");
    let timeout = service
        .close_and_wait_for_quiescence(Duration::from_millis(25))
        .expect_err("close cannot quiesce while the admitted execution job is blocked");
    assert_eq!(timeout.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert!(!completion.is_ready(), "no terminal report is published while execution still owns the query epoch");

    attempt_owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).expect("execution commits its terminal outcome");
    drop(context);
    handle.finish();
    drop(handle);
    assert!(!completion.is_ready(), "terminal outcome alone cannot release the running execution job");
    assert_eq!(
        service.admission_snapshot().unwrap().active_queries,
        1,
        "capacity stays unavailable until the job drains"
    );

    release.add_permits(1);
    execution.await.expect("execution job task joins").expect("execution job completes after release");
    drop(pool);

    let report = completion.wait_timeout(Duration::from_secs(10)).expect("post-cleanup attempt report seals");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.timestamps.admitted_ns.is_some());
    assert!(report.timestamps.physical_execution_started_ns.is_some());
    assert!(report.timestamps.cleanup_complete_ns.is_some());
    assert!(report.timestamps.report_ready_ns.is_some());
    assert_eq!(report.capacity_disposition, crate::observability::AttemptCapacityDisposition::Released);
    assert_eq!(counters.snapshot().reports_ready, 1, "admitted close publishes exactly one terminal report");
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("late execution cleanup lets close recover");
}

#[test]
fn active_query_owner_prevents_service_drain_until_cleanup_finishes() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();

    service.close().unwrap();

    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    drop(service);
}
#[test]
fn close_and_wait_fences_delayed_gpu_cleanup_before_service_resource_destruction() {
    struct CleanupBlockerGate {
        semaphore: Arc<tokio::sync::Semaphore>,
    }

    impl Drop for CleanupBlockerGate {
        fn drop(&mut self) {
            self.semaphore.close();
        }
    }

    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query().unwrap();
    let pool = handle.gpu_execution_pool().unwrap();
    let worker_count = pool.snapshot().worker_count;
    let release = CleanupBlockerGate { semaphore: Arc::new(tokio::sync::Semaphore::new(0)) };
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();

    for _ in 0..worker_count {
        let started_tx = started_tx.clone();
        let finished_tx = finished_tx.clone();
        let release = Arc::clone(&release.semaphore);
        pool.spawn(GpuExecutionJobKind::Cleanup, async move {
            started_tx.send(()).expect("cleanup blocker start receiver remains live");
            let _permit = release.acquire_owned().await.expect("cleanup blocker semaphore remains open");
            finished_tx.send(()).expect("cleanup finish receiver remains live");
        })
        .unwrap();
    }
    drop(started_tx);
    drop(finished_tx);
    for _ in 0..worker_count {
        started_rx.recv_timeout(Duration::from_secs(5)).expect("every GPU worker starts a cleanup blocker");
    }

    drop(handle);
    service.close().unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    let wait_entry = Arc::new(Barrier::new(2));
    let waiter_entry = Arc::clone(&wait_entry);
    let waiter_service = service.clone();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        waiter_entry.wait();
        done_tx
            .send(waiter_service.close_and_wait_for_quiescence(Duration::from_secs(5)))
            .expect("close waiter receiver remains live");
    });
    wait_entry.wait();
    assert!(
        matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "close-and-wait cannot finish while GPU cleanup is queued"
    );

    release.semaphore.add_permits(worker_count);
    for _ in 0..worker_count {
        finished_rx.recv().expect("every delayed cleanup job finishes after release");
    }
    assert!(
        matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "close-and-wait remains blocked by the counted pool after cleanup finishes"
    );
    drop(pool);
    done_rx.recv().expect("close waiter completes after the counted pool drops").unwrap();
    waiter.join().expect("close waiter must not panic");

    drop(service);
}
#[test]
fn query_service_waits_for_background_stream_cleanup_quiescence() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    let context = handle.acquire_context().unwrap();

    let error = service.wait_for_stream_pools_idle(Duration::ZERO).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("stream_pool_idle_timeout"));
    assert_eq!(error.fact_value("created_runtimes"), Some("1"));
    assert_eq!(error.fact_value("idle_runtimes"), Some("0"));

    drop(context);
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
}
#[test]
fn query_service_quiescence_allows_declared_long_lived_root_contexts() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query().unwrap();
    let context = handle.acquire_context().unwrap();

    service.wait_for_stream_pools_quiescent(&[(0, 1)], Duration::ZERO).unwrap();

    let error = service.wait_for_stream_pools_quiescent(&[(0, 0)], Duration::ZERO).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("stream_pool_idle_timeout"));
    assert_eq!(error.fact_value("active_leased_runtimes"), Some("1"));
    assert_eq!(error.fact_value("expected_leased_runtimes"), Some("0"));

    drop(context);
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
}
