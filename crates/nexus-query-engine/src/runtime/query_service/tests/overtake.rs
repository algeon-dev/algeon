// Finite-overtake liveness protection: exact bypass counts, protection
// target selection, protected slot/bytes at every release, retarget and
// termination, and work conservation. CPU-only via the synthetic ledger.

fn raised_requirement(min: u64, want: u64) -> AttemptRequirement {
    requirement(min, Some(want))
}

#[test]
fn overtakes_count_exact_bypasses_and_install_protection_at_limit() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 2 },
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 100, 10, 3, Default::default()),
    )
    .unwrap();
    let active = service.admit_query().unwrap();
    assert_eq!(active.device_budget_bytes(), 80, "grant clamps to preserve two empty-slot floors");
    let mut big = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(50, 50)),
        )
        .unwrap();
    assert!(big.try_admit().unwrap().is_none(), "raised floor 50 cannot fit the remaining headroom");

    let mut small_one = service.enqueue_admission().unwrap();
    let small_one = small_one.try_admit().unwrap().expect("first small bypass admits");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, Vec::<usize>::new());
    let mut small_two = service.enqueue_admission().unwrap();
    let small_two = small_two.try_admit().unwrap().expect("second small bypass admits");

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.protected_devices, vec![0], "the second bypass exhausts the overtake limit");
    let report = service.device_ledger_reports().unwrap().remove(0);
    assert!(report.protected_slot);
    assert_eq!(report.protected_reserve_bytes, 10, "reserve starts at the configured floor");

    drop(small_one);
    assert_eq!(service.device_ledger_reports().unwrap().remove(0).protected_reserve_bytes, 10);
    drop(small_two);
    assert_eq!(
        service.device_ledger_reports().unwrap().remove(0).protected_reserve_bytes,
        20,
        "freed bytes bank into the protection liveness before any later admission"
    );
    drop(active);
    assert_eq!(service.device_ledger_reports().unwrap().remove(0).protected_reserve_bytes, 50);

    let admitted_big = big.try_admit().unwrap().expect("protected waiter is admitted once releases land");
    assert_eq!(admitted_big.device_budget_bytes(), 50);
    let wait = admitted_big.admission_wait_fact().expect("queued admission retains its final overtake state");
    assert_eq!(wait.overtake_limit, 2);
    assert_eq!(wait.overtakes_consumed, 2);
    assert_eq!(wait.overtakes_remaining, 0);
    assert_eq!(wait.protection_device_ordinal, Some(0));
    let report = service.device_ledger_reports().unwrap().remove(0);
    assert!(!report.protected_slot, "admission consumes the protection");
    assert_eq!(report.protected_reserve_bytes, 0);
    assert_eq!(report.live_grants_bytes, 50);
}

#[test]
fn exhausted_waiter_retries_protection_after_the_previous_waiter_leaves() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 100, 10, 2, Default::default()),
    )
    .unwrap();
    let _active = service.admit_query().unwrap();
    let blocked_one = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let mut blocked_two = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();

    let mut small = service.enqueue_admission().unwrap();
    let admitted = small.try_admit().unwrap().expect("bypass admits behind capacity-blocked heads");

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(
        snapshot.protected_devices,
        vec![0],
        "only the oldest limited waiter protects the device; the second cannot compete"
    );

    // The second waiter had already exhausted its bypass allowance, but the
    // first waiter owned the only device protection. Once the first leaves,
    // the next later placement must reinstall the second waiter's protection
    // before it can consume the freed slot.
    drop(admitted);
    drop(blocked_one);
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, Vec::<usize>::new());

    let mut later = service.enqueue_admission().unwrap();
    assert!(later.try_admit().unwrap().is_none(), "restored protection prevents one extra bypass");
    assert_eq!(
        service.admission_snapshot().unwrap().protected_devices,
        vec![0],
        "the exhausted waiter is protected before a later request can be placed"
    );
    drop(_active);
    let _admitted_blocked_two = blocked_two
        .try_admit()
        .unwrap()
        .expect("the restored protection admits its exhausted waiter after capacity releases");
}

#[test]
fn restore_failure_after_an_earlier_installation_rolls_back_every_new_protection() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        100,
        10,
        1,
    );
    let first = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(10, 100)),
        )
        .expect("first exhausted waiter queues");
    let second = service.enqueue_admission().expect("second exhausted waiter queues");

    {
        let mut state = service.service.inner.admission.lock().expect("admission state");
        for request in &mut state.queued {
            request.overtakes_remaining = 0;
        }
        let second_request = state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == second.admission_request_id())
            .expect("second waiter remains queued");
        // Position zero installs on the real ledger. Position one then hits a
        // hard ledger lookup error, exercising restoration after partial work.
        second_request.compatible_devices = vec![1];

        let error = service
            .service
            .select_next_runnable_locked(&mut state)
            .expect_err("the forced position-one restore failure reaches the caller");
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert!(state.queued.iter().all(|request| request.protected_device.is_none()));

        // Restore the fixture's valid submission shape before dropping the
        // waiters. The assertion above is about the restoration transaction,
        // not a persistent invalid-device request.
        state
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == second.admission_request_id())
            .expect("second waiter remains queued")
            .compatible_devices = vec![0];
    }
    assert!(
        !service.device_ledger_reports().unwrap().remove(0).protected_slot,
        "the partial restore must not strand a protected ledger slot"
    );

    drop(first);
    drop(second);
    let full_capacity = service
        .admit_query_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(10, 100)),
        )
        .expect("the device readmits after the rollback");
    assert_eq!(full_capacity.device_budget_bytes(), 100);
}

#[test]
fn deadline_crossing_during_overtake_commit_rolls_back_allowance_and_protection() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        100,
        10,
        2,
    );
    let older = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let younger = service.enqueue_admission().unwrap();
    let younger_id = younger.admission_request_id();
    let mut state = service.service.inner.admission.lock().unwrap();
    let younger_submission_seq = state
        .queued
        .iter()
        .find(|request| request.admission_request_id == younger_id)
        .expect("younger request remains queued")
        .submission_seq;

    let error = service
        .service
        .record_overtake_locked(&mut state, younger_submission_seq, 0, Instant::now())
        .expect_err("expired publication cannot commit overtake accounting");

    assert_eq!(error.fact_value("reason"), Some("queue_wait_elapsed"));
    let older_state = state
        .queued
        .iter()
        .find(|request| request.admission_request_id == older.admission_request_id())
        .expect("older request remains queued");
    assert_eq!(older_state.overtakes_remaining, 1);
    assert_eq!(older_state.protected_device, None);
    drop(state);
    assert!(!service.device_ledger_reports().unwrap().remove(0).protected_slot);
}

#[test]
fn caller_deadline_remains_a_typed_cancellation_at_admission() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        100,
        10,
        2,
    );
    let cancellation = ExecutionCancellation::with_deadline(Some(Instant::now() - Duration::from_millis(1)));

    let error = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_execution_cancellation(cancellation))
        .expect_err("an expired caller deadline must stop admission before it can publish a grant");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
}

#[test]
fn cancelled_and_expired_waiters_end_their_protection() {
    let queue_bounds = |timeout| QueueBounds { max_waiters: 8, max_wait: timeout, max_overtakes_per_waiter: 1 };
    // Cancellation path: dropping the protected waiter ends the protection.
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        queue_bounds(Duration::from_secs(30)),
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 100, 10, 2, Default::default()),
    )
    .unwrap();
    let _active = service.admit_query().unwrap();
    let blocked = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();
    let _admitted = small.try_admit().unwrap().expect("bypass installs protection");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);
    drop(blocked);
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, Vec::<usize>::new());

    // Deadline path: expiry removes the waiter and its protection.
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        queue_bounds(Duration::from_millis(20)),
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 100, 10, 2, Default::default()),
    )
    .unwrap();
    let _active = service.admit_query().unwrap();
    let mut blocked = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();
    let _admitted = small.try_admit().unwrap().expect("bypass installs protection");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);
    std::thread::sleep(Duration::from_millis(40));
    let _ = blocked.try_admit();
    assert_eq!(
        service.admission_snapshot().unwrap().protected_devices,
        Vec::<usize>::new(),
        "deadline expiry ends the protection without acquiring capacity"
    );
}

/// A third-party reporter may be the first caller to sweep a queued request
/// after cancellation. Cancellation remains the terminal result even if the
/// protected device closes before that sweep.
#[test]
fn third_party_sweep_preserves_cancelled_protected_waiter_over_device_closure() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 1 },
        100,
        10,
        2,
    );
    let active = service.admit_query().expect("initial admission occupies the large grant");
    let cancellation = ExecutionCancellation::new();
    let mut protected = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new()
                .with_attempt_requirement(raised_requirement(60, 60))
                .with_execution_cancellation(cancellation.clone()),
        )
        .expect("large waiter queues");
    assert!(protected.try_admit().expect("large waiter remains queued").is_none());
    let mut bypass = service.enqueue_admission().expect("small bypass queues");
    let bypass = bypass.try_admit().expect("small bypass placement").expect("small bypass admits");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);

    cancellation.cancel();
    service
        .close_device_admission_for_test(0, "test-cancelled-protected-waiter-device-close")
        .expect("device closure is recorded before the unrelated sweep");

    // A snapshot is deliberately unrelated to the waiter's owner. It drives
    // the contested sweep path without giving the cancelled waiter a chance
    // to observe and remove itself first.
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
    let error = protected.try_admit().expect_err("cancelled waiter cannot be reclassified by device closure");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_cancelled"));

    drop(bypass);
    drop(active);
}

#[test]
fn impossible_requirement_is_rejected_at_submit_without_queueing() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 4 },
        100,
        10,
        2,
    );
    let requirement = crate::admission::AttemptRequirement {
        configured_floor_bytes: 10,
        capability_floor_raise_bytes: 200,
        required_phase_floor_bytes: 0,
        source_stage_bytes: None,
        soft_working_set_hint_bytes: None,
        estimate_quality: nexus_query_engine::admission::EstimateQuality::Unknown,
        required_capabilities: Vec::new(),
        attested_cugraph_graph_execution_scopes: Vec::new(),
        reservation_shapes: Vec::new(),
    };
    let error = service
        .admit_query_with_request(
            nexus_query_engine::runtime::QueryAdmissionRequest::new().with_attempt_requirement(requirement),
        )
        .expect_err("a floor above every compatible device rejects immediately");
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("impossible_requirement"));
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0, "impossible requests never queue");
}

#[test]
fn protection_does_not_block_capacity_beyond_the_reserve() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 130, 10, 4, Default::default()),
    )
    .unwrap();
    let _active = service.admit_query().unwrap();
    let _blocked = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();
    let _admitted_small = small.try_admit().unwrap().expect("bypass installs protection");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);

    // Later work may use capacity beyond the protection: protected bytes are
    // not a grant and never revoke live allocations.
    let mut another = service.enqueue_admission().unwrap();
    let admitted = another.try_admit().unwrap().expect("capacity beyond the reserve stays usable");
    assert_eq!(admitted.device_budget_bytes(), 10);
}

#[test]
fn service_close_ends_every_protection_without_reusable_capacity() {
    let service = AdmissionTestController::new_with_ledger(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(5), max_overtakes_per_waiter: 1 },
        non_gds_deployment(),
        test_ledger_with_explicit_profile(0, 100, 10, 2, Default::default()),
    )
    .unwrap();
    let _active = service.admit_query().unwrap();
    let _blocked = service
        .enqueue_admission_with_request(
            QueryAdmissionRequest::new().with_attempt_requirement(raised_requirement(60, 60)),
        )
        .unwrap();
    let mut small = service.enqueue_admission().unwrap();
    let _admitted = small.try_admit().unwrap().expect("bypass installs protection");
    assert_eq!(service.admission_snapshot().unwrap().protected_devices, vec![0]);

    service.close().unwrap();

    assert!(
        !service.device_ledger_reports().unwrap().remove(0).protected_slot,
        "close must end the protection rather than withholding capacity forever"
    );
}

#[test]
fn fallback_driver_never_exceeds_the_per_device_active_limit() {
    let service = admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(20), max_overtakes_per_waiter: 2 },
        100,
        10,
        1,
    );
    let _active = service.admit_query().unwrap();
    let mut waiter = service.enqueue_admission().unwrap();
    // The device limit is saturated. The fallback driver may only drive the
    // timeout taxonomy; it must not place a second attempt.
    assert!(waiter.try_admit().unwrap().is_none(), "saturated device limit keeps the waiter queued");
    std::thread::sleep(Duration::from_millis(40));
    let error = waiter.try_admit().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
}
