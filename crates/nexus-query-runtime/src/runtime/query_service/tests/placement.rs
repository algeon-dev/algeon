#[test]
fn proof_device_entries_reapply_request_local_placement_floor() {
    let service = test_service(1, 8);
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "value",
        arrow::datatypes::DataType::Int64,
        false,
    )]));
    let mut builder = nexus_query_planning::plan::QueryPlanBuilder::new();
    let source = builder.add_source(nexus_query_planning::source::SourceNode::GpuResident(
        nexus_query_planning::source::GpuResidentSource::new("input", schema),
    ));
    let plan = builder.finish_with_host_sink(source);
    let proof = nexus_query_planning::admission::QueryPlanCapabilityProof::for_query_plan(&plan)
        .expect("fixture plan must issue a capability proof");
    let request = QueryAdmissionRequest::new()
        .with_query_plan_capability_proof(proof.clone())
        .with_test_placement_requirement(nexus_query_planning::admission::AttemptCapabilityEvidence::from_parts(
            0,
            0,
            Some(4096),
            nexus_query_planning::admission::EstimateQuality::Measured,
            Vec::new(),
        ));

    let handle = service.admit_query_with_request(request).expect("proof-backed fixture admission succeeds");
    let requirement = handle
        .handle
        .lifetime
        .admitted_device_requirement_for_test(0)
        .expect("selected device keeps its immutable admission entry");
    assert_eq!(requirement.required_floor_bytes_for(0).expect("fixture floor is finite"), 4096);

    handle.finish();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(2)).expect("proof-backed fixture cleanup completes");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn queued_placement_folds_a_native_fault_into_the_ledger_before_retargeting() {
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0, 1],
        non_gds_deployment(),
    )
    .expect("two-device runtime-backed service");
    let device_zero_ledger = service
        .inner
        .resources
        .as_ref()
        .expect("runtime resources")
        .devices
        .context_by_ordinal(0)
        .expect("device zero context")
        .ledger()
        .clone();
    let scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
    let unhealthy = cudf::testing::device_safety::injected_device_id(0).expect("isolated device zero health");
    cudf::runtime::report_device_fault(
        unhealthy,
        cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
        "injected_queued_placement_fault",
        None,
    );
    assert!(
        matches!(
            device_zero_ledger.health().expect("engine ledger health"),
            crate::runtime::gpu::DeviceAdmissionHealth::Open
        ),
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
            crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. }
        ),
        "pre-sweep reconciliation closes the faulted device ledger before selecting a healthy device"
    );
    assert_eq!(scope.health_snapshot(unhealthy).fault_count(), 1);
    handle.finish();
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
    // for its memory-wait taxonomy instead of retrying the physically faulty
    // ordinal zero and terminally rejecting this waiter.
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
    active.finish();
    drop(active);
    // `finish` schedules destruction of the device-backed envelope on its
    // selected GPU executor. The immutable grant becomes reusable only after
    // that cleanup commits, so wait on the service's semantic quiescence
    // fence rather than racing the asynchronous cleanup job.
    service
        .wait_for_stream_pools_idle(Duration::from_secs(10))
        .expect("active healthy admission cleanup completes before re-admission");
    let admitted_after_release = blocked
        .try_admit()
        .expect("healthy capacity release wakes the waiter")
        .expect("waiter admits on device one after release");
    assert_eq!(admitted_after_release.device_ordinal(), 1);
    admitted_after_release.finish();
    drop(admitted_after_release);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn direct_placement_missing_resolution_cleans_up_before_waking_a_close_waiter() {
    let service = test_service(1, 8);
    service.service.fail_next_direct_placement_resolution_for_test();
    let pause = service.service.pause_next_direct_provisional_cleanup_for_test();
    let admission_service = service.service.clone();
    let admission = std::thread::spawn(move || admission_service.try_admit_query());
    pause.wait_until_entered(Duration::from_secs(2));

    let close_service = service.service.clone();
    let (close_sender, close_receiver) = std::sync::mpsc::sync_channel(1);
    let close = std::thread::spawn(move || {
        close_sender
            .send(close_service.close_and_wait_for_quiescence(Duration::from_secs(2)))
            .expect("test receives close result");
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while service.shutdown_waiter_count() == 0 {
        assert!(Instant::now() < deadline, "close must wait for the unpublished provisional ticket");
        std::thread::yield_now();
    }
    assert!(
        matches!(close_receiver.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "the initial provisional-ticket notification must not let close report quiescence"
    );

    pause.release();
    let error = admission
        .join()
        .expect("direct placement thread does not panic")
        .expect_err("the injected post-placement resolution failure rejects the direct admission");
    assert_eq!(error.fact_value("reason"), Some("missing_device_specific_admission_resolution"));
    close_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("close wakes after unpublished provisional cleanup")
        .expect("direct placement defensive failure leaves the service quiescent");
    close.join().expect("close thread does not panic");

    assert_eq!(service.admission_snapshot().expect("admission snapshot").active_queries, 0);
    let state = service.service.inner.admission.lock().expect("admission state");
    assert!(state.lifecycle.is_quiescent(), "the unpublished provisional ticket must be cleaned up exactly once");
}

#[test]
fn dropping_admission_ticket_wakes_a_close_waiter_after_cleanup() {
    let service = test_service(1, 8);
    let ticket = {
        let state = service.service.inner.admission.lock().expect("admission state");
        state.lifecycle.begin_provisional(std::sync::Arc::downgrade(&service.service.inner))
    };

    let close_service = service.service.clone();
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
fn direct_placement_reports_a_post_snapshot_fleet_closure_as_a_health_failure() {
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0],
        non_gds_deployment(),
    )
    .expect("one-device runtime-backed service");
    let pause = service.pause_next_direct_placement_after_snapshot_for_test();
    let placement_service = service.clone();
    let placement = std::thread::spawn(move || placement_service.try_admit_query());
    pause.wait_until_entered(Duration::from_secs(2));

    service
        .close_device_admission_for_test(0, "post-snapshot-direct-placement-closure")
        .expect("close the only ledger after it was ranked");
    pause.release();

    let error = placement
        .join()
        .expect("direct placement thread does not panic")
        .expect_err("closed fleet cannot be classified as a capacity shortage");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("no_healthy_compatible_device"));
    assert_ne!(error.fact_value("reason"), Some("required_floor_exceeds_device_capacity"));

    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn unknown_isolation_fleet_closure_excludes_a_mid_closure_admission() {
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0, 1],
        non_gds_deployment(),
    )
    .expect("two-device runtime-backed service");
    let closure_pause = service.pause_unknown_isolation_closure_after_first_ledger_for_test();
    let direct_pause = service.pause_next_direct_placement_after_snapshot_for_test();
    let closing_service = service.clone();
    let closing = std::thread::spawn(move || {
        closing_service.force_backend_unhealthy_for_test("test-mid-fleet-closure");
    });
    closure_pause.wait_until_entered(Duration::from_secs(2));

    let candidate_service = service.clone();
    let candidate = std::thread::spawn(move || candidate_service.try_admit_query());
    let deadline = Instant::now() + Duration::from_secs(2);
    while !matches!(service.inner.admission.try_lock(), Err(std::sync::TryLockError::WouldBlock)) {
        assert!(Instant::now() < deadline, "candidate admission did not reach the fleet gate before the deadline");
        std::thread::yield_now();
    }
    let entered_before_fleet_closure = direct_pause.entered_within(Duration::from_millis(200));

    // Always release both worker rendezvous before asserting so a regression
    // fails cleanly instead of stranding a test thread.
    direct_pause.release();
    closure_pause.release();
    closing.join().expect("fleet closure thread does not panic");
    let result = candidate.join().expect("candidate admission thread does not panic");

    assert!(
        !entered_before_fleet_closure,
        "a candidate must not reach its ledger snapshot while fleet closure holds the admission gate"
    );
    let error = result.expect_err("the fleet closure wins before a later device can admit work");
    assert_eq!(error.code(), QueryErrorCode::CudfPersistentFailure);
    assert_eq!(error.fact_value("reason"), Some("backend_unhealthy"));

    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn protected_waiter_retargets_after_native_fault_before_candidate_ranking() {
    let visible_devices =
        cudf::DeviceId::count().expect("enumerate CUDA devices for protected-waiter placement coverage");
    assert!(
        visible_devices >= 2,
        "protected-waiter fault retargeting requires at least two visible CUDA devices; found {visible_devices}"
    );
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0, 1],
        non_gds_deployment(),
    )
    .expect("two-device runtime-backed service");
    let resources = service.inner.resources.as_ref().expect("runtime resources");
    let device_zero_ledger = resources.devices.context_by_ordinal(0).expect("device zero context").ledger().clone();
    let device_one_ledger = resources.devices.context_by_ordinal(1).expect("device one context").ledger().clone();
    let mut waiter = service.enqueue_admission().expect("request enters the shared queue");
    let admission_request_id = waiter.admission_request_id;
    {
        let mut state = service.inner.admission.lock().expect("admission state");
        let request = state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == admission_request_id)
            .expect("queued request");
        let required_floor = device_zero_ledger
            .required_floor_bytes_for(request.requirement_for_device(0).expect("device-zero admission requirement"))
            .expect("device zero attempt floor");
        assert!(
            device_zero_ledger
                .protect_slot(admission_request_id, required_floor)
                .expect("install deterministic device-zero protection"),
            "device zero is initially available for this waiter's protection"
        );
        request.overtakes_remaining = 0;
        request.protected_device = Some(0);
    }

    let scope = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
    let unhealthy = cudf::testing::device_safety::injected_device_id(0).expect("isolated device zero health");
    cudf::runtime::report_device_fault(
        unhealthy,
        cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
        "injected_protected_waiter_placement_fault",
        None,
    );
    assert!(
        matches!(
            device_zero_ledger.health().expect("engine ledger health"),
            crate::runtime::gpu::DeviceAdmissionHealth::Open
        ),
        "the native fault must lead the scheduler-ledger transition"
    );

    let handle = waiter
        .try_admit()
        .expect("health reconciliation and protection retarget succeed")
        .expect("the protected waiter admits on healthy device one in one drive");

    assert_eq!(handle.device_ordinal(), 1);
    assert!(
        matches!(
            device_zero_ledger.health().expect("faulted device ledger health"),
            crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. }
        ),
        "pre-sweep reconciliation closes the faulted protection target"
    );
    assert!(device_zero_ledger.protection().is_none(), "faulted device retains no stale protection");
    assert!(device_one_ledger.protection().is_none(), "admission atomically consumes the retargeted protection");
    assert_eq!(scope.health_snapshot(unhealthy).fault_count(), 1);
    handle.finish();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn failed_protection_retarget_terminalizes_its_waiter_without_rejecting_the_next_caller() {
    let visible_devices =
        cudf::DeviceId::count().expect("enumerate CUDA devices for protected-waiter retarget failure coverage");
    assert!(
        visible_devices >= 2,
        "protected-waiter retarget failure coverage requires at least two visible CUDA devices; found {visible_devices}"
    );
    let service = QueryService::new_with_device_ordinals_and_deployment(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(2), max_overtakes_per_waiter: 4 },
        vec![0, 1],
        non_gds_deployment(),
    )
    .expect("two-device runtime-backed service");
    let resources = service.inner.resources.as_ref().expect("runtime resources");
    let device_zero_ledger = resources.devices.context_by_ordinal(0).expect("device zero context").ledger().clone();
    let device_one_ledger = resources.devices.context_by_ordinal(1).expect("device one context").ledger().clone();

    let mut affected = service.enqueue_admission().expect("affected waiter enters the shared queue");
    let unaffected = service.enqueue_admission().expect("unrelated waiter enters the shared queue");
    let affected_request_id = affected.admission_request_id;
    {
        let mut state = service.inner.admission.lock().expect("admission state");
        let request = state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == affected_request_id)
            .expect("affected waiter remains queued");
        let required_floor = device_zero_ledger
            .required_floor_bytes_for(request.requirement_for_device(0).expect("device-zero admission requirement"))
            .expect("device zero attempt floor");
        assert!(
            device_zero_ledger
                .protect_slot(affected_request_id, required_floor)
                .expect("install deterministic device-zero protection"),
            "the affected waiter owns the first protection"
        );
        request.overtakes_remaining = 0;
        request.protected_device = Some(0);
    }
    service
        .close_device_admission_for_test(0, "retarget-failure-after-old-protection-release")
        .expect("close the old protection target");
    service.fail_next_protection_retarget_after_release_for_test();

    // An unrelated snapshot is allowed to drive health maintenance. It must
    // terminalize the waiter whose old protection was released, rather than
    // leaving partial queue state for a later caller to inherit.
    assert_eq!(service.admission_snapshot().expect("health sweep completes").queued_queries, 1);
    assert!(device_zero_ledger.protection().is_none(), "closed device retains no stale protection");
    assert!(device_one_ledger.protection().is_none(), "failed replacement did not install a competing protection");

    let error = affected.try_admit().expect_err("the affected waiter receives the injected retarget failure");
    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.fact_value("reason"), Some("injected_protection_retarget_failure"));

    let mut unaffected = unaffected;
    let handle = unaffected
        .try_admit()
        .expect("unrelated waiter is not rejected by retarget maintenance")
        .expect("healthy device one remains available to the unrelated waiter");
    assert_eq!(handle.device_ordinal(), 1);
    handle.finish();
    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).expect("service drains");
}

#[test]
fn device_entries_keep_distinct_ordinal_requirement_and_envelope_values() {
    let service = test_service(1, 8).service;
    let policy = service.inner.config.attempt_reservation_policy();
    let empty_envelope = nexus_query_planning::admission::AttemptReservationEnvelope::resolve(&[], policy)
        .expect("empty test envelope is finite");
    let first = requirement(128, Some(128));
    let second = requirement(256, Some(512));
    let (entries, compatible) = service
        .resolve_device_admission_entries_for_test(&[0, 1], |ordinal| {
            Ok(DeviceAdmissionEntry {
                requirement: if ordinal == 0 { first.clone() } else { second.clone() },
                admitted_reservation_envelope: (ordinal == 1).then(|| empty_envelope.clone()),
            })
        })
        .expect("both concrete candidates resolve");

    assert_eq!(compatible, vec![0, 1]);
    assert_eq!(entries.get(&0).expect("ordinal zero entry").requirement.configured_floor_bytes, 128);
    assert_eq!(entries.get(&1).expect("ordinal one entry").requirement.configured_floor_bytes, 256);
    assert_ne!(
        entries.get(&0).expect("ordinal zero entry").admitted_reservation_envelope,
        entries.get(&1).expect("ordinal one entry").admitted_reservation_envelope,
    );
}

#[test]
fn candidate_local_entry_resolution_prunes_only_the_rejected_ordinal() {
    let service = test_service(1, 8).service;
    let (entries, compatible) = service
        .resolve_device_admission_entries_for_test(&[0, 1], |ordinal| {
            if ordinal == 0 {
                Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "nexus_query_engine.test.sort_preflight",
                    "candidate sort owner preflight failed",
                )
                .with_fact("reason", "sort_owner_candidate_device_unavailable"))
            } else {
                Ok(DeviceAdmissionEntry {
                    requirement: requirement(256, Some(256)),
                    admitted_reservation_envelope: None,
                })
            }
        })
        .expect("one healthy candidate remains");

    assert_eq!(compatible, vec![1]);
    assert!(!entries.contains_key(&0), "the rejected candidate must not enter placement");
    assert_eq!(entries.get(&1).expect("ordinal one entry").requirement.configured_floor_bytes, 256);
}

#[test]
fn all_device_entry_resolution_failures_return_owner_reason_before_queueing() {
    let service = test_service(1, 8).service;
    let error = service
        .resolve_device_admission_entries_for_test(&[0, 1], |_ordinal| {
            Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.test.sort_preflight",
                "sort owner preflight is unavailable",
            )
            .with_fact("reason", "sort_owner_native_preflight_unavailable"))
        })
        .expect_err("all candidate-local failures reject before enqueue");

    assert_eq!(error.fact_value("reason"), Some("sort_owner_native_preflight_unavailable"));
    assert_eq!(error.fact_value("candidate_count"), Some("2"));
    assert_eq!(service.inner.admission.lock().expect("admission state").queued.len(), 0);
}

#[test]
fn missing_selected_entry_is_a_typed_invariant_error() {
    let service = test_service(1, 8).service;
    let mut notifications = PendingDeviceSlotNotifications::default();
    let error = service
        .try_place_query(
            AdmissionRequestId::new(91),
            crate::observability::AttemptId::mint(),
            &DeviceAdmissionEntries::new(),
            &[0],
            &mut notifications,
        )
        .expect_err("placement cannot fall back when the selected entry is absent");

    assert_eq!(error.code(), QueryErrorCode::Internal);
    assert_eq!(error.fact_value("reason"), Some("missing_device_specific_admission_resolution"));
    assert_eq!(error.fact_value("device_ordinal"), Some("0"));
}
