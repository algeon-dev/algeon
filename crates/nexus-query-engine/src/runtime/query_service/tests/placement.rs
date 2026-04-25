#[test]
fn query_service_optimistic_slot_guarantee_does_not_reserve_empty_slot_minima() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 60)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should consume remaining memory");

    assert_eq!(first.device_budget_bytes(), 60);
    assert_eq!(second.device_budget_bytes(), 40);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
}
#[test]
fn query_service_reserves_remaining_slots_for_largest_tenant_minimum() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(120),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("low", 10, 100), QueryServiceTenantConfig::new("high", 60)],
        non_gds_deployment(),
    )
    .unwrap();

    let low = service.try_admit_query("low").unwrap().expect("low-min tenant should admit");
    let high = service.try_admit_query("high").unwrap().expect("high-min tenant should retain capacity");

    assert_eq!(low.device_budget_bytes(), 60);
    assert_eq!(high.device_budget_bytes(), 60);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 0);
}
#[test]
fn query_service_rejects_guaranteed_minima_that_exceed_slot_capacity() {
    let error = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("low", 10, 100), QueryServiceTenantConfig::new("high", 60)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("tenant_minima_exceed_guaranteed_slot_capacity"));
    assert_eq!(error.fact_value("required_capacity_bytes"), Some("120"));
    assert_eq!(error.fact_value("device_capacity_bytes"), Some("100"));
}
#[test]
fn guaranteed_min_slot_capacity_uses_per_device_packing_not_total_capacity() {
    let packed_slots = guaranteed_min_slot_capacity_for_capacities([90_u64, 30], 50);

    assert_eq!(packed_slots, 1);
    assert!(packed_slots < 2, "two 50-byte guaranteed slots cannot pack onto 90-byte and 30-byte devices");
}
#[test]
fn guaranteed_min_slot_capacity_respects_max_queries_per_device_cap() {
    let packed_slots = super::config::guaranteed_min_slot_capacity_for_capacities_with_cap([100_u64, 100], 10, Some(3));

    assert_eq!(packed_slots, 6);
}
#[test]
fn guaranteed_min_slot_capacity_excludes_devices_unreachable_by_pinned_placement() {
    let packed_slots = super::config::guaranteed_min_slot_capacity_for_ordinal_capacities_with_policy(
        [(0, 100_u64), (1, 100)],
        60,
        None,
        QueryPlacementPolicy::Pinned { device_ordinal: 0 },
    );

    assert_eq!(packed_slots, 1);
}
#[test]
fn governor_claim_rejects_guaranteed_minimum_overlap_and_releases_on_drop() {
    let ordinal = usize::MAX - 17;
    let first = claim_governor_ordinals(&[ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect("first guaranteed claim should install");
    let error = claim_governor_ordinals(&[ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect_err("overlapping guaranteed claim should reject");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));
    let ordinal_string = ordinal.to_string();
    assert_eq!(error.fact_value("device_ordinal"), Some(ordinal_string.as_str()));

    drop(first);
    let second = claim_governor_ordinals(&[ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect("dropped claim should release the ordinal");
    drop(second);
}
#[test]
fn governor_claim_explicit_release_allows_replacement_while_old_owner_is_retained() {
    let ordinal = usize::MAX - 20;
    let first = claim_governor_ordinals(&[ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect("first guaranteed claim should install");

    first.release();
    let replacement = claim_governor_ordinals(&[ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect("explicitly released closed service claim should allow replacement");

    first.release();
    drop(first);
    drop(replacement);
}
#[test]
fn governor_claim_rejects_optimistic_overlap_with_guaranteed_minimum() {
    let guaranteed_first_ordinal = usize::MAX - 18;
    let guaranteed = claim_governor_ordinals(&[guaranteed_first_ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect("guaranteed claim should install");
    let error = claim_governor_ordinals(&[guaranteed_first_ordinal], SlotGuarantee::Optimistic)
        .expect_err("optimistic service must not share an active guaranteed governor claim");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));
    drop(guaranteed);

    let optimistic_first_ordinal = usize::MAX - 19;
    let optimistic = claim_governor_ordinals(&[optimistic_first_ordinal], SlotGuarantee::Optimistic)
        .expect("optimistic claim should install");
    let error = claim_governor_ordinals(&[optimistic_first_ordinal], SlotGuarantee::GuaranteedMinimum)
        .expect_err("guaranteed service must not build while an optimistic service can consume the governor");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));
    drop(optimistic);
}
#[test]
fn shared_device_waiter_registration_prunes_dead_weak_entries() {
    let service = test_service(1, 2);
    let mut waiters = vec![Weak::<QueryServiceInner>::new(), Arc::downgrade(&service.inner)];

    prune_dead_shared_device_admission_waiters(&mut waiters);

    assert_eq!(waiters.len(), 1);
    let retained = waiters[0].upgrade().expect("live service weak should remain");
    assert!(Arc::ptr_eq(&retained, &service.inner));
}
#[test]
fn shared_device_wakeup_locks_target_admission_mutex_before_notifying() {
    let service = test_service(1, 2);
    let device_ordinal = usize::MAX - 128;
    {
        let mut waiters = shared_device_admission_waiters().lock().unwrap();
        waiters.insert(device_ordinal, vec![Arc::downgrade(&service.inner)]);
    }
    let guard = service.inner.admission.lock().unwrap();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (done_sender, done_receiver) = std::sync::mpsc::channel();
    let notifier = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        notify_shared_device_admission_available(device_ordinal);
        done_sender.send(()).unwrap();
    });

    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(
        done_receiver.recv_timeout(Duration::from_millis(50)).is_err(),
        "shared wakeup must wait for the target admission mutex"
    );

    drop(guard);
    done_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    notifier.join().unwrap();
    shared_device_admission_waiters().lock().unwrap().remove(&device_ordinal);
}
#[test]
fn pending_device_slot_notifications_release_slot_and_defer_shared_wakeup_until_drop() {
    let service = test_service(1, 2);
    let device_ordinal = usize::MAX - 129;
    {
        let mut waiters = shared_device_admission_waiters().lock().unwrap();
        waiters.insert(device_ordinal, vec![Arc::downgrade(&service.inner)]);
    }
    let active_queries = Arc::new(AtomicUsize::new(1));
    let mut notifications = PendingDeviceSlotNotifications::default();
    notifications.release_slot(DeviceActiveQuerySlot::new_for_test(device_ordinal, Arc::clone(&active_queries)));
    assert_eq!(active_queries.load(Ordering::Acquire), 0);

    let guard = service.inner.admission.lock().unwrap();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (done_sender, done_receiver) = std::sync::mpsc::channel();
    let notifier = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        drop(notifications);
        done_sender.send(()).unwrap();
    });

    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(
        done_receiver.recv_timeout(Duration::from_millis(50)).is_err(),
        "pending slot notification must synchronize with the target admission mutex"
    );

    drop(guard);
    done_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    notifier.join().unwrap();
    shared_device_admission_waiters().lock().unwrap().remove(&device_ordinal);
}
