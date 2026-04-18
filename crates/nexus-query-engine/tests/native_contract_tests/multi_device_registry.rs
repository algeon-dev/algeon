use std::{
    sync::{Mutex, MutexGuard, OnceLock, mpsc},
    time::Duration,
};

use nexus_query_engine::{
    DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryGrantRequest,
    QueryPlacementPolicy, QueryService, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId,
};

const TEST_DEVICE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn service_claim_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().expect("multi-device registry test lock poisoned")
}

#[test]
fn query_service_places_concurrent_runtime_handles_on_distinct_devices_when_available() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::new("tenant", TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    let first_context = first.acquire_context().unwrap();
    let second_context = second.acquire_context().unwrap();
    assert_eq!(first_context.context().runtime().device().ordinal(), first.device_ordinal());
    assert_eq!(second_context.context().runtime().device().ordinal(), second.device_ordinal());

    let reports = service.memory_governor_reports().unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].device_ordinal, 0);
    assert_eq!(reports[1].device_ordinal, 1);
    assert_eq!(reports[0].capacity_bytes, TEST_DEVICE_CAPACITY_BYTES);
    assert_eq!(reports[1].capacity_bytes, TEST_DEVICE_CAPACITY_BYTES);
    assert_eq!(reports[0].outstanding_bytes, first.device_budget_bytes());
    assert_eq!(reports[1].outstanding_bytes, second.device_budget_bytes());
}

#[test]
fn best_fit_placement_packs_before_spreading() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        placement_policy: QueryPlacementPolicy::BestFit,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 0);
}

#[test]
fn max_queries_per_device_forces_spread_and_then_blocks() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 3,
        max_queries_per_device: Some(1),
        max_admission_queue_len: 8,
        stream_pool_size: Some(9),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        placement_policy: QueryPlacementPolicy::BestFit,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
}

#[test]
fn pinned_placement_uses_selected_device() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        placement_policy: QueryPlacementPolicy::Pinned { device_ordinal: 1 },
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 1);
    assert_eq!(second.device_ordinal(), 1);
}

#[test]
fn pinned_guaranteed_minimum_reserves_future_slots_on_pinned_device() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let min_budget = TEST_DEVICE_CAPACITY_BYTES / 2;
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        placement_policy: QueryPlacementPolicy::Pinned { device_ordinal: 0 },
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::with_grant_range("tenant", min_budget, TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 0);
    assert_eq!(first.device_budget_bytes(), min_budget);
    assert_eq!(second.device_budget_bytes(), min_budget);
}

#[test]
fn pinned_placement_rejects_device_outside_registry() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_admission_queue_len: 2,
        stream_pool_size: Some(3),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        placement_policy: QueryPlacementPolicy::Pinned { device_ordinal: 1 },
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("pinned_device_not_configured"));
}

#[test]
fn query_service_memory_failure_callback_ignores_budget_failures() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_admission_queue_len: 2,
        stream_pool_size: Some(3),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();

    let initial_report = service.device_memory_failure_reports().into_iter().next().unwrap();
    if initial_report.callback_install_unavailable {
        return;
    }
    assert!(initial_report.callback_installed);

    let handle = service.try_admit_query("tenant").unwrap().expect("query should admit");
    let context = handle.acquire_context().unwrap();
    let token = context
        .context()
        .runtime()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1).with_boundary("nexus_failure_callback"))
        .unwrap();
    let error = context.context().runtime().series_from_value_with_budget(1i32, 1024, &token).unwrap_err();
    assert_eq!(error.kind(), cudf::ErrorKind::ResourceExhausted);

    let report = service.device_memory_failure_reports().into_iter().next().unwrap();
    assert!(report.callback_installed);
    assert_eq!(report.allocation_failure_count, initial_report.allocation_failure_count);
    assert_eq!(report.total_failed_allocation_bytes, initial_report.total_failed_allocation_bytes);
    assert_eq!(report.largest_failed_allocation_bytes, initial_report.largest_failed_allocation_bytes);
    assert_eq!(report.last_failed_allocation_bytes, initial_report.last_failed_allocation_bytes);
    assert!(report.pool_usage_observation_count >= initial_report.pool_usage_observation_count);
}

#[test]
fn query_services_on_same_device_share_memory_governor_ledger() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_admission_queue_len: 2,
        stream_pool_size: Some(3),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let first_service = QueryService::new_with_device_ordinals_and_deployment(
        config.clone(),
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();
    let second_service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = first_service.try_admit_query("tenant").unwrap().expect("first service should admit");

    assert!(second_service.try_admit_query("tenant").unwrap().is_none());
    assert_eq!(second_service.memory_governor_reports().unwrap()[0].outstanding_bytes, first.device_budget_bytes());

    drop(first);
    let _second = second_service.admit_query("tenant").expect("ordered cleanup should wake memory admission");
}

#[test]
fn max_queries_per_device_cap_is_shared_across_services_on_same_device() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_queries_per_device: Some(1),
        max_admission_queue_len: 2,
        stream_pool_size: Some(3),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let first_service = QueryService::new_with_device_ordinals_and_deployment(
        config.clone(),
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();
    let second_service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = first_service.try_admit_query("tenant").unwrap().expect("first service should admit");

    assert!(second_service.try_admit_query("tenant").unwrap().is_none());

    drop(first);
    let _second = second_service.admit_query("tenant").expect("ordered cleanup should wake per-device admission");
}

#[test]
fn max_queries_per_device_release_wakes_waiter_in_other_service() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_queries_per_device: Some(1),
        max_admission_queue_len: 2,
        stream_pool_size: Some(3),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_secs(2),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let first_service = QueryService::new_with_device_ordinals_and_deployment(
        config.clone(),
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();
    let second_service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = first_service.try_admit_query("tenant").unwrap().expect("first service should admit");
    let waiting_service = second_service.clone();
    let (sender, receiver) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let result = waiting_service
            .admit_query("tenant")
            .map(|handle| handle.device_ordinal())
            .map_err(|error| error.to_string());
        sender.send(result).expect("main thread should receive waiter result");
    });

    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while second_service.admission_snapshot().unwrap().queued_queries == 0 {
        assert!(std::time::Instant::now() < deadline, "second service did not queue before release");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(first);

    let admitted_device = receiver.recv_timeout(Duration::from_secs(3)).expect("waiter should wake before timeout");
    waiter.join().expect("waiter thread should finish");
    assert_eq!(admitted_device.unwrap(), 0);
}

#[test]
fn max_queries_per_device_releases_with_handle_even_when_context_retains_grant() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = TEST_DEVICE_CAPACITY_BYTES / 4;
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_queries_per_device: Some(1),
        max_admission_queue_len: 2,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let tenant_context = first.acquire_context().unwrap();
    let retained_context = tenant_context.context().clone();
    drop(tenant_context);
    first.finish();

    let second = service.admit_query("tenant").expect("ordered cleanup should free per-device cap");
    assert_eq!(second.device_ordinal(), 0);
    drop(retained_context);
}

#[test]
fn guaranteed_min_requested_floor_validation_uses_idle_capacity_not_live_device_slots() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let min_budget = TEST_DEVICE_CAPACITY_BYTES / 16;
    let requested_floor = TEST_DEVICE_CAPACITY_BYTES / 2;
    let config = QueryServiceConfig {
        max_concurrent_queries: 4,
        max_queries_per_device: Some(4),
        max_admission_queue_len: 8,
        stream_pool_size: Some(12),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::GuaranteedMinimum,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::with_grant_range("tenant", min_budget, TEST_DEVICE_CAPACITY_BYTES)],
        non_gds_deployment(),
    )
    .unwrap();

    let _first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let _second = service.try_admit_query("tenant").unwrap().expect("second query should admit");
    let _third = service.try_admit_query("tenant").unwrap().expect("third query should admit");

    let band = service
        .tenant_requested_grant_band_with_grant_request(
            &TenantId::new("tenant"),
            QueryGrantRequest::with_requested_grant_min_bytes(requested_floor),
        )
        .expect("live active slots must not collapse the idle guaranteed-minimum ceiling");

    assert_eq!(band.min_bytes, requested_floor);
}

#[test]
fn multi_device_variable_grants_do_not_reserve_every_future_slot_on_one_device() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::with_grant_range("tenant", min_budget, want_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 1);
    assert_eq!(first.device_budget_bytes(), want_budget);
    assert_eq!(second.device_budget_bytes(), want_budget);
}

#[test]
fn multi_device_variable_grants_reserve_when_remaining_slots_exceed_other_device_capacity() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let _guard = service_claim_test_guard();
    let want_budget = TEST_DEVICE_CAPACITY_BYTES;
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let config = QueryServiceConfig {
        max_concurrent_queries: 3,
        max_admission_queue_len: 8,
        stream_pool_size: Some(9),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::with_grant_range("tenant", min_budget, want_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");
    let third = service.try_admit_query("tenant").unwrap().expect("third query should admit");

    assert_eq!(first.device_budget_bytes(), want_budget);
    assert!(second.device_budget_bytes() >= min_budget);
    assert!(third.device_budget_bytes() >= min_budget);
}

#[test]
fn query_service_treats_transient_device_capacity_exhaustion_as_busy_not_rejected() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = (TEST_DEVICE_CAPACITY_BYTES / 2).saturating_add(1);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(500),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");

    assert!(service.try_admit_query("tenant").unwrap().is_none());
    drop(first);
    let _second = service.admit_query("tenant").expect("ordered cleanup should wake capacity admission");
}

#[test]
fn queued_query_service_admission_times_out_while_device_capacity_is_busy() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let _guard = service_claim_test_guard();
    let query_budget = (TEST_DEVICE_CAPACITY_BYTES / 2).saturating_add(1);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(TEST_DEVICE_CAPACITY_BYTES),
        admission_wait_timeout: Duration::from_millis(1),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0],
        vec![QueryServiceTenantConfig::new("tenant", query_budget)],
        non_gds_deployment(),
    )
    .unwrap();
    let _active = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    std::thread::sleep(Duration::from_millis(5));
    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_wait_timeout"));
}
