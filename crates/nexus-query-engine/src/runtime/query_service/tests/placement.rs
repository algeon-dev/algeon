#[test]
fn query_service_optimistic_slot_guarantee_does_not_reserve_empty_slot_minima() {
    let service = AdmissionTestController::new(
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
    let service = AdmissionTestController::new(
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
    let error = AdmissionTestController::new(
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
fn shared_device_waiter_registration_prunes_dead_weak_entries() {
    let service = test_service(1, 2);
    assert!(service.prune_dead_shared_waiters_retains_self());
}
#[test]
fn shared_device_wakeup_locks_target_admission_mutex_before_notifying() {
    let service = test_service(1, 2);
    let device_ordinal = usize::MAX - 128;
    service.register_shared_device_waiter(device_ordinal);
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (done_sender, done_receiver) = std::sync::mpsc::channel();
    let notifier = service.with_admission_locked(|| {
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
        notifier
    });
    done_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    notifier.join().unwrap();
    shared_device_admission_waiters().lock().unwrap().remove(&device_ordinal);
}
#[test]
fn pending_device_slot_notifications_release_slot_and_defer_shared_wakeup_until_drop() {
    let service = test_service(1, 2);
    let device_ordinal = usize::MAX - 129;
    service.register_shared_device_waiter(device_ordinal);
    let active_queries = Arc::new(AtomicUsize::new(1));
    let mut notifications = PendingDeviceSlotNotifications::default();
    notifications.release_slot(DeviceActiveQuerySlot::new_for_test(device_ordinal, Arc::clone(&active_queries)));
    assert_eq!(active_queries.load(Ordering::Acquire), 0);

    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (done_sender, done_receiver) = std::sync::mpsc::channel();
    let notifier = service.with_admission_locked(|| {
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
        notifier
    });
    done_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    notifier.join().unwrap();
    shared_device_admission_waiters().lock().unwrap().remove(&device_ordinal);
}
