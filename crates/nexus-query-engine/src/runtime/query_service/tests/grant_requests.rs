#[test]
fn requested_grant_bytes_override_tenant_want_for_one_admission() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(4096),
            max_concurrent_queries: 1,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 1024, 3072)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service
        .try_admit_query_with_qos_and_requested_grant_bytes("tenant", QueryQos::default(), Some(1536))
        .unwrap()
        .expect("request within tenant band should admit");

    assert_eq!(handle.device_budget_bytes(), 1536);
    assert_eq!(
        handle.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1536,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    );
    drop(handle);

    let default_handle = service.admit_query("tenant").unwrap();
    assert_eq!(default_handle.device_budget_bytes(), 3072);
}
#[test]
fn requested_grant_bytes_rejects_outside_tenant_band_and_capacity() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(2048),
            max_concurrent_queries: 1,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 512, 4096)],
        non_gds_deployment(),
    )
    .unwrap();

    let below_min = service
        .tenant_requested_budget(&TenantId::new("tenant"), Some(256))
        .expect_err("requested grant below tenant min should reject");
    assert_eq!(below_min.fact_value("reason"), Some("requested_grant_below_minimum"));

    let above_want = service
        .tenant_requested_budget(&TenantId::new("tenant"), Some(8192))
        .expect_err("requested grant above tenant want should reject");
    assert_eq!(above_want.fact_value("reason"), Some("requested_grant_above_tenant_want"));

    let above_capacity = service
        .tenant_requested_budget(&TenantId::new("tenant"), Some(3072))
        .expect_err("requested grant above device capacity should reject");
    assert_eq!(above_capacity.fact_value("reason"), Some("requested_grant_above_device_capacity"));
    assert_eq!(above_capacity.fact_value("max_device_capacity_bytes"), Some("2048"));
}
#[test]
fn requested_grant_min_bytes_raises_floor_for_one_admission() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 100)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service
        .try_admit_query_with_qos_and_grant_request(
            "tenant",
            QueryQos::default(),
            QueryGrantRequest::with_requested_grant_min_bytes(40).prefer(70),
        )
        .unwrap()
        .expect("request within tenant band should admit");

    assert_eq!(handle.device_budget_bytes(), 70);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 30);
}
#[test]
fn requested_grant_min_bytes_queues_instead_of_admitting_below_floor() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service
        .try_admit_query_with_qos_and_requested_grant_bytes("tenant", QueryQos::default(), Some(70))
        .unwrap()
        .expect("first query should admit");
    let mut waiter = service
        .enqueue_admission_with_qos_and_grant_request(
            "tenant",
            QueryQos::default(),
            QueryGrantRequest::with_requested_grant_min_bytes(40),
        )
        .unwrap();

    assert_eq!(active.device_budget_bytes(), 70);
    assert_eq!(first_memory_governor_report(&service).available_bytes, 30);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);

    drop(active);
    let admitted = waiter.try_admit().unwrap().expect("released capacity should admit queued request");

    assert!(admitted.device_budget_bytes() >= 40);
}
#[test]
fn requested_grant_min_bytes_preserves_configured_slot_minima() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 100)],
        non_gds_deployment(),
    )
    .unwrap();

    let raised_floor = service
        .try_admit_query_with_qos_and_grant_request(
            "tenant",
            QueryQos::default(),
            QueryGrantRequest::with_requested_grant_min_bytes(40),
        )
        .unwrap()
        .expect("raised floor should admit while reserving configured tenant minima");
    let second = service.try_admit_query("tenant").unwrap().expect("configured slot minimum should remain 10");
    let third = service.try_admit_query("tenant").unwrap().expect("configured slot minimum should remain 10");
    let fourth = service.try_admit_query("tenant").unwrap().expect("configured slot minimum should remain 10");

    assert_eq!(raised_floor.device_budget_bytes(), 70);
    assert_eq!(second.device_budget_bytes(), 10);
    assert_eq!(third.device_budget_bytes(), 10);
    assert_eq!(fourth.device_budget_bytes(), 10);
    assert_eq!(first_memory_governor_report(&service).outstanding_bytes, 100);
}
#[test]
fn requested_grant_min_bytes_rejects_above_guaranteed_min_idle_slot_ceiling() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 4,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let tenant_id = TenantId::new("tenant");

    let error = service
        .tenant_requested_budget_with_grant_request(&tenant_id, QueryGrantRequest::with_requested_grant_min_bytes(80))
        .expect_err("floor above idle guaranteed-minimum slot ceiling should reject");

    assert_eq!(error.fact_value("reason"), Some("requested_grant_min_above_guaranteed_min_slot_ceiling"));
    assert_eq!(error.fact_value("requested_grant_min_bytes"), Some("80"));
    assert_eq!(error.fact_value("guaranteed_min_idle_grant_ceiling_bytes"), Some("70"));
    assert_eq!(error.fact_value("max_concurrent_queries"), Some("4"));
    assert_eq!(error.fact_value("max_tenant_min_device_budget_bytes"), Some("10"));
}
#[test]
fn requested_grant_min_bytes_timeout_error_carries_floor_and_band_facts() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 4,
            admission_wait_timeout: Duration::from_millis(1),
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service
        .try_admit_query_with_qos_and_requested_grant_bytes("tenant", QueryQos::default(), Some(70))
        .unwrap()
        .expect("first query should admit");
    let error = service
        .admit_query_with_qos_and_grant_request(
            "tenant",
            QueryQos::default(),
            QueryGrantRequest::with_requested_grant_min_bytes(40),
        )
        .expect_err("ungrantable floor should time out while waiting for memory");

    assert_eq!(active.device_budget_bytes(), 70);
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("memory_grant_wait_timeout"));
    assert_eq!(error.fact_value("requested_grant_min_bytes"), Some("40"));
    assert_eq!(error.fact_value("requested_grant_bytes"), Some("100"));
    assert_eq!(error.fact_value("effective_min_device_budget_bytes"), Some("40"));
    assert_eq!(error.fact_value("effective_want_device_budget_bytes"), Some("100"));
    assert_eq!(error.fact_value("slot_guarantee"), Some("optimistic"));
    assert_eq!(error.fact_value("guaranteed_min_idle_grant_ceiling_bytes"), Some("100"));
}
#[test]
fn requested_grant_min_bytes_rejects_outside_tenant_band_capacity_and_request_range() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(2048),
            max_concurrent_queries: 1,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 512, 4096)],
        non_gds_deployment(),
    )
    .unwrap();
    let tenant_id = TenantId::new("tenant");

    let below_min = service
        .tenant_requested_budget_with_grant_request(&tenant_id, QueryGrantRequest::with_requested_grant_min_bytes(256))
        .expect_err("requested grant min below tenant min should reject");
    assert_eq!(below_min.fact_value("reason"), Some("requested_grant_min_below_tenant_minimum"));

    let above_want = service
        .tenant_requested_budget_with_grant_request(&tenant_id, QueryGrantRequest::with_requested_grant_min_bytes(8192))
        .expect_err("requested grant min above tenant want should reject");
    assert_eq!(above_want.fact_value("reason"), Some("requested_grant_min_above_tenant_want"));

    let above_capacity = service
        .tenant_requested_budget_with_grant_request(&tenant_id, QueryGrantRequest::with_requested_grant_min_bytes(3072))
        .expect_err("requested grant min above device capacity should reject");
    assert_eq!(above_capacity.fact_value("reason"), Some("requested_grant_min_above_device_capacity"));
    assert_eq!(above_capacity.fact_value("max_device_capacity_bytes"), Some("2048"));

    let invalid_request_range = service
        .tenant_requested_budget_with_grant_request(
            &tenant_id,
            QueryGrantRequest::with_requested_grant_min_bytes(1536).prefer(1024),
        )
        .expect_err("requested preferred grant below requested floor should reject");
    assert_eq!(invalid_request_range.fact_value("reason"), Some("requested_grant_below_requested_minimum"));
    assert_eq!(invalid_request_range.fact_value("requested_grant_min_bytes"), Some("1536"));
    assert_eq!(invalid_request_range.fact_value("requested_grant_bytes"), Some("1024"));
}
#[test]
fn query_service_assigns_fixed_per_tenant_bounded_budget() {
    let service = test_service(2, 8);

    let large = service.admit_query("large").unwrap();
    let small = service.admit_query("small").unwrap();

    assert_eq!(large.device_budget_bytes(), 8 * 1024 * 1024);
    assert_eq!(small.device_budget_bytes(), 1024 * 1024);
    assert_eq!(large.device_ordinal(), 0);
    assert_eq!(small.device_ordinal(), 0);
    assert_eq!(
        small.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    );
    assert_eq!(large.admission_sequence(), 1);
    assert_eq!(small.admission_sequence(), 2);
    assert!(large.admission_wait() <= Duration::from_secs(1));
    assert!(small.admission_wait() <= Duration::from_secs(1));
}
