use std::time::Duration;

use nexus_query_engine::{
    CudfInventory, DeploymentIoCapability, GdsDriverProbeStatus, GpuInventory, KvikioCompatModeStatus,
    NvidiaFsProbeStatus, QueryService, QueryServiceConfig, QueryServiceTenantConfig, device_budget_bytes_from_facts,
};

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[test]
fn query_service_places_concurrent_runtime_handles_on_distinct_devices_when_available() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let device_capacity_bytes = 1024 * 1024;
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        device_capacity_bytes: Some(device_capacity_bytes),
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_device_ordinals_and_deployment(
        config,
        vec![0, 1],
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
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
    assert_eq!(reports[0].capacity_bytes, device_capacity_bytes);
    assert_eq!(reports[1].capacity_bytes, device_capacity_bytes);
    assert_eq!(reports[0].outstanding_bytes, first.device_budget_bytes());
    assert_eq!(reports[1].outstanding_bytes, second.device_budget_bytes());
}

#[test]
fn multi_device_variable_grants_do_not_reserve_every_future_slot_on_one_device() {
    if cudf::DeviceId::count().unwrap_or(0) < 2 {
        return;
    }
    let capacity0 =
        device_budget_bytes_from_facts(&CudfInventory.facts(0).unwrap(), 500 * 1024 * 1024, 256 * 1024 * 1024);
    let capacity1 =
        device_budget_bytes_from_facts(&CudfInventory.facts(1).unwrap(), 500 * 1024 * 1024, 256 * 1024 * 1024);
    let want_budget = capacity0.min(capacity1);
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
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
    let capacity0 =
        device_budget_bytes_from_facts(&CudfInventory.facts(0).unwrap(), 500 * 1024 * 1024, 256 * 1024 * 1024);
    let capacity1 =
        device_budget_bytes_from_facts(&CudfInventory.facts(1).unwrap(), 500 * 1024 * 1024, 256 * 1024 * 1024);
    let want_budget = capacity0.min(capacity1);
    let min_budget = want_budget / 2;
    assert!(min_budget > 0);
    let config = QueryServiceConfig {
        max_concurrent_queries: 3,
        max_admission_queue_len: 8,
        stream_pool_size: Some(9),
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
    let facts = CudfInventory.facts(0).unwrap();
    let capacity = device_budget_bytes_from_facts(&facts, 500 * 1024 * 1024, 256 * 1024 * 1024);
    let query_budget = (capacity / 2).saturating_add(1);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        admission_wait_timeout: Duration::from_millis(500),
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
    assert!(service.try_admit_query("tenant").unwrap().is_some());
}

#[test]
fn queued_query_service_admission_times_out_while_device_capacity_is_busy() {
    if cudf::DeviceId::count().unwrap_or(0) < 1 {
        return;
    }
    let facts = CudfInventory.facts(0).unwrap();
    let capacity = device_budget_bytes_from_facts(&facts, 500 * 1024 * 1024, 256 * 1024 * 1024);
    let query_budget = (capacity / 2).saturating_add(1);
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        max_admission_queue_len: 8,
        stream_pool_size: Some(6),
        admission_wait_timeout: Duration::from_millis(1),
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
    assert_eq!(error.fact_value("reason"), Some("memory_grant_wait_timeout"));
}
