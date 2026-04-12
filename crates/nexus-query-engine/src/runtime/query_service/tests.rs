use super::*;
use crate::runtime::{GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus};

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
    let config = QueryServiceConfig {
        max_concurrent_queries,
        max_admission_queue_len,
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    QueryService::new_for_admission_tests(
        config,
        vec![
            QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("small", 1024 * 1024),
        ],
        non_gds_deployment(),
    )
    .unwrap()
}

fn runtime_test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
    let config = QueryServiceConfig {
        max_concurrent_queries,
        max_admission_queue_len,
        admission_wait_timeout: Duration::from_millis(500),
        ..QueryServiceConfig::default()
    };
    QueryService::new_with_deployment(
        config,
        vec![
            QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
            QueryServiceTenantConfig::new("small", 1024 * 1024),
        ],
        non_gds_deployment(),
    )
    .unwrap()
}

fn wait_for_queued_len(service: &QueryService, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let state = service.inner.admission.lock().unwrap();
        if state.queued.len() >= expected {
            return;
        }
        drop(state);
        assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn wait_for_exact_queued_len(service: &QueryService, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let queued = service.admission_snapshot().unwrap().queued_queries;
        if queued == expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {expected} queued admissions; last queued={queued}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn optimistic_shrink_test_service(native_pipeline_execution_mode: NativePipelineExecutionMode) -> QueryService {
    QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            native_pipeline_execution_mode,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 40, 100)],
        non_gds_deployment(),
    )
    .unwrap()
}

fn wait_for_requested_shrink(grant: &MemoryGrant, expected: Option<u64>) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let actual = grant.requested_shrink_target_bytes().unwrap();
        if actual == expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for shrink target {expected:?}; last target={actual:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

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

#[test]
fn query_service_admission_test_config_can_record_non_default_device_without_runtime_resources() {
    let config = QueryServiceConfig { device_ordinal: 2, ..QueryServiceConfig::default() };
    let service = QueryService::new_admission_only(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();

    assert_eq!(handle.device_ordinal(), 2);
    let error = handle.acquire_context().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
}

#[test]
fn query_service_reports_overload_when_queue_is_full() {
    let service = test_service(1, 1);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (finish_sender, finish_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let handle = queued_service.admit_query("large").unwrap();
        finish_receiver.recv().unwrap();
        drop(handle);
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let error = service.admit_query("small").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.kind(), crate::error::QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
    assert_eq!(error.fact_value("tenant_id"), Some("small"));

    drop(active);
    finish_sender.send(()).unwrap();
    waiter.join().unwrap();
}

#[test]
fn query_service_reports_admission_snapshot_by_tenant_order() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let _handle = queued_service.admit_query("small").unwrap();
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 1);
    assert_eq!(snapshot.next_tenant_id.as_ref().map(TenantId::as_str), Some("small"));
    assert_eq!(snapshot.queued_by_tenant.len(), 2);
    assert_eq!(snapshot.queued_by_tenant[0].tenant_id.as_str(), "large");
    assert_eq!(snapshot.queued_by_tenant[0].queued_queries, 0);
    assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "small");
    assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);

    drop(active);
    waiter.join().unwrap();
}

#[test]
fn query_service_handle_exposes_configured_execution_policies() {
    let dynamic_filter_execution_options = DynamicFilterExecutionOptions::default().with_max_build_rows(10_000);
    let keyed_hash_join_strategy_options =
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
    let aggregate_strategy_execution_options =
        AggregateStrategyExecutionOptions::default().with_encoded_key_partial_min_input_rows(22);
    let source_chunk_planning_options =
        SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(33);
    let config = QueryServiceConfig {
        source_scheduler_policy: SourceSchedulerPolicy::new(4),
        source_chunk_planning_mode: SourceChunkPlanningMode::LocalParquetThroughput,
        native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
        stddev_samp_execution_mode: StddevSampExecutionMode::GpuNativeTolerant,
        dynamic_filter_execution_options,
        keyed_hash_join_strategy_options,
        aggregate_strategy_execution_options,
        source_chunk_planning_options,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();

    assert_eq!(handle.boundary_policy(), ExecutionBoundaryPolicy::default());
    assert_eq!(handle.source_scheduler_policy(), SourceSchedulerPolicy::new(4));
    assert_eq!(handle.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(handle.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    assert_eq!(handle.stddev_samp_execution_mode(), StddevSampExecutionMode::GpuNativeTolerant);
    assert_eq!(handle.dynamic_filter_execution_options(), dynamic_filter_execution_options);
    assert_eq!(handle.keyed_hash_join_strategy_options(), keyed_hash_join_strategy_options);
    assert_eq!(handle.aggregate_strategy_execution_options(), aggregate_strategy_execution_options);
    assert_eq!(handle.source_chunk_planning_options(), source_chunk_planning_options);
}

#[test]
fn query_service_rejects_inconsistent_dynamic_filter_execution_policy() {
    let config = QueryServiceConfig {
        dynamic_filter_execution_options: DynamicFilterExecutionOptions::default()
            .with_max_build_rows(10)
            .with_large_local_parquet_range_producer_rows(11),
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
    assert_eq!(error.fact_value("bound_field"), Some("max_build_rows"));
    assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
}

#[test]
fn query_service_rejects_inconsistent_keyed_hash_join_strategy_policy() {
    let config = QueryServiceConfig {
        keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default()
            .with_large_keyed_hash_build_row_threshold(10),
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("large_field"), Some("large_keyed_hash_build_row_threshold"));
    assert_eq!(error.fact_value("small_field"), Some("small_reusable_hash_build_row_threshold"));
    assert_eq!(error.fact_value("reason"), Some("large_threshold_below_small_threshold"));
}

#[test]
fn query_service_rejects_invalid_aggregate_strategy_execution_policy() {
    let config = QueryServiceConfig {
        aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default()
            .with_prepared_sort_groupby_low_cardinality_threshold_ppm(1_000_001),
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("prepared_sort_groupby_low_cardinality_threshold_ppm"));
    assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
}

#[test]
fn query_service_rejects_invalid_source_chunk_planning_policy() {
    let config = QueryServiceConfig {
        source_chunk_planning_options: SourceChunkPlanningOptions::default()
            .with_local_parquet_throughput_max_row_groups_per_chunk(0),
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn query_service_resources_use_independent_lakehouse_cache_fill_limits() {
    let config = QueryServiceConfig {
        source_fetch: SourceFetchConfig {
            host_fetch_budget_bytes: Some(4096),
            host_fetch_acquire_timeout: Duration::from_secs(9),
            host_fetch_max_concurrent_ops: 5,
            lakehouse_cache_fill_budget_bytes: Some(2048),
            lakehouse_cache_fill_max_concurrent_ops: 2,
            ..SourceFetchConfig::default()
        },
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();
    let resources = service.inner.resources.as_ref().unwrap();

    assert_eq!(resources.host_fetch_budget.budget_bytes(), 4096);
    assert_eq!(resources.source_fetch_concurrency.max_concurrent_ops(), 5);
    assert_eq!(resources.lakehouse_cache_fill_budget.budget_bytes(), 2048);
    assert_eq!(resources.lakehouse_cache_fill_concurrency.max_concurrent_ops(), 2);

    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let context = tenant_context.context();
    assert_eq!(context.host_fetch_budget().budget_bytes(), 4096);
    assert_eq!(context.host_fetch_acquire_timeout(), Duration::from_secs(9));
    assert_eq!(context.host_fetch_max_concurrent_ops(), 5);
    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
    assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 2);

    context.lakehouse_cache_fill_events().record(super::super::LakehouseCacheFillEvent {
        outcome: super::super::LakehouseCacheFillOutcome::Completed,
        bytes: Some(128),
        elapsed: Duration::from_millis(1),
        cache_path_hash: "cache-path-hash".to_string(),
        stable_id_hash: "stable-id-hash".to_string(),
        file_index: None,
        chunk_index: None,
        error: None,
    });
    assert_eq!(context.lakehouse_cache_fill_event_snapshots().len(), 1);

    let other_handle = service.admit_query("tenant").unwrap();
    let other_tenant_context = other_handle.acquire_context().unwrap();
    assert!(other_tenant_context.context().lakehouse_cache_fill_event_snapshots().is_empty());
}

#[test]
fn query_service_released_handle_rejects_context_acquisition() {
    let service = test_service(1, 8);
    let handle = service.admit_query("small").unwrap();
    handle.finish();

    let error = handle.acquire_context().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("tenant_id"), Some("small"));
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
}

#[test]
fn query_service_context_keeps_admission_until_context_drop() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    drop(handle);

    assert!(service.try_admit_query("small").unwrap().is_none());
    drop(tenant_context);
    assert!(service.try_admit_query("small").unwrap().is_some());
}

#[test]
fn query_service_handle_reuses_one_root_context() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let first = handle.acquire_context().unwrap();
    let second = handle.acquire_context().unwrap();

    assert!(Arc::ptr_eq(&first.context().shared_runtime(), &second.context().shared_runtime()));
}

#[test]
fn query_service_rejects_stream_pool_smaller_than_admission_limit() {
    let config =
        QueryServiceConfig { max_concurrent_queries: 2, stream_pool_size: Some(1), ..QueryServiceConfig::default() };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("stream_pool_size"));
    assert_eq!(error.fact_value("reason"), Some("stream_pool_smaller_than_admission_limit"));
}

#[test]
fn query_service_rejects_stream_pool_that_covers_only_root_streams() {
    let config = QueryServiceConfig {
        max_concurrent_queries: 2,
        source_scheduler_policy: SourceSchedulerPolicy::new(1),
        stream_pool_size: Some(2),
        ..QueryServiceConfig::default()
    };

    let error = QueryService::new_for_admission_tests(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("stream_pool_size"));
    assert_eq!(error.fact_value("max_source_reads_in_flight"), Some("1"));
    assert_eq!(error.fact_value("minimum_stream_pool_size"), Some("4"));
}

#[test]
fn query_service_try_admit_does_not_queue_when_saturated() {
    let service = test_service(1, 8);
    let active = service.try_admit_query("large").unwrap().expect("first query should admit");

    assert!(service.try_admit_query("small").unwrap().is_none());
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 0);

    drop(active);
    assert!(service.try_admit_query("small").unwrap().is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_wakes_on_release() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
    wait_for_queued_len(&service, 1);

    drop(active);
    let handle = time::timeout(Duration::from_millis(200), queued).await.unwrap().unwrap().unwrap();

    assert_eq!(handle.tenant_id().as_str(), "small");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_admit_async_notifies_after_queue_head_admits() {
    let service = test_service(3, 8);
    let active_1 = service.admit_query("large").unwrap();
    let active_2 = service.admit_query("large").unwrap();
    let active_3 = service.admit_query("large").unwrap();
    let first_service = service.clone();
    let second_service = service.clone();
    let first = tokio::spawn(async move { first_service.admit_async("small").await });
    let second = tokio::spawn(async move { second_service.admit_async("small").await });
    wait_for_queued_len(&service, 2);

    drop(active_1);
    drop(active_2);
    drop(active_3);
    let first = time::timeout(Duration::from_millis(200), first).await.unwrap().unwrap().unwrap();
    let second = time::timeout(Duration::from_millis(200), second).await.unwrap().unwrap().unwrap();

    assert_eq!(first.tenant_id().as_str(), "small");
    assert_eq!(second.tenant_id().as_str(), "small");
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_service_cancelled_admit_async_removes_queued_request() {
    let service = test_service(1, 8);
    let active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
    wait_for_queued_len(&service, 1);

    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    wait_for_exact_queued_len(&service, 0);
    drop(active);

    assert!(service.try_admit_query("small").unwrap().is_some());
}

#[test]
fn query_service_concurrent_budget_sum_never_exceeds_capacity() {
    let capacity_bytes = 12 * 1024 * 1024;
    let max_concurrent_queries = 4;
    let per_query_budget = capacity_bytes / u64::try_from(max_concurrent_queries).unwrap();
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(capacity_bytes),
            max_concurrent_queries,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", per_query_budget)],
        non_gds_deployment(),
    )
    .unwrap();

    let handles = (0..max_concurrent_queries)
        .map(|_| service.try_admit_query("tenant").unwrap().expect("capacity should admit query"))
        .collect::<Vec<_>>();

    let admitted_budget_sum: u64 = handles.iter().map(QueryHandle::device_budget_bytes).sum();
    assert_eq!(admitted_budget_sum, capacity_bytes);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
}

#[test]
fn query_service_uses_variable_memory_governor_grants() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 60)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should preserve remaining minima");
    let third = service.try_admit_query("tenant").unwrap().expect("third query should admit minimum grant");
    let fourth = service.try_admit_query("tenant").unwrap().expect("fourth query should admit minimum grant");

    assert_eq!(first.device_budget_bytes(), 60);
    assert_eq!(second.device_budget_bytes(), 20);
    assert_eq!(third.device_budget_bytes(), 10);
    assert_eq!(fourth.device_budget_bytes(), 10);
    assert!(service.try_admit_query("tenant").unwrap().is_none());
    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.capacity_bytes, 100);
    assert_eq!(report.outstanding_bytes, 100);
    assert_eq!(report.available_bytes, 0);

    drop(first);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 60);
}

#[test]
fn adaptive_query_service_grows_partial_active_grant_on_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 90);
    assert_eq!(second.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 10);

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 90);
    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.outstanding_bytes, 90);
    assert_eq!(report.available_bytes, 10);
}

#[test]
fn strict_query_service_does_not_grow_partial_active_grant_on_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 10);
    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.outstanding_bytes, 10);
    assert_eq!(report.available_bytes, 90);
}

#[test]
fn adaptive_query_service_preserves_queued_minimum_before_growing_active_grants() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 80)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut queued = service.enqueue_admission("tenant").unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 20);
    let admitted = queued.try_admit().unwrap().expect("queued query should receive released capacity");
    assert_eq!(admitted.device_budget_bytes(), 80);
    assert_eq!(second_grant.budget_bytes(), 20);
}

#[test]
fn adaptive_query_service_grows_after_queue_drains_when_capacity_remains() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("wide", 10, 80), QueryServiceTenantConfig::new("small", 10)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
    let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut queued = service.enqueue_admission("small").unwrap();

    drop(first);

    assert_eq!(second_grant.budget_bytes(), 20);
    let admitted = queued.try_admit().unwrap().expect("small queued query should receive its minimum");
    assert_eq!(admitted.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 80);
}

#[test]
fn adaptive_growth_allocations_cap_and_redistribute_by_weight() {
    let allocations = weighted_adaptive_growth_allocations(
        &[
            AdaptiveGrowthCandidate { query_id: 1, missing_bytes: 10, weight: 10 },
            AdaptiveGrowthCandidate { query_id: 2, missing_bytes: 100, weight: 1 },
            AdaptiveGrowthCandidate { query_id: 3, missing_bytes: 100, weight: 1 },
        ],
        60,
    );

    assert_eq!(allocations, vec![(1, 10), (2, 25), (3, 25)]);
}

#[test]
fn adaptive_query_service_grows_partial_grants_by_budget_weight() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 4,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 70)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
    let high = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(4))
        .unwrap()
        .expect("second query should admit minimum");
    let low_a = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(1))
        .unwrap()
        .expect("third query should admit minimum");
    let low_b = service
        .try_admit_query_with_qos("tenant", QueryQos::new(Priority::Normal).with_budget_weight(1))
        .unwrap()
        .expect("fourth query should admit minimum");
    let high_grant = high.lifetime.memory_grant().unwrap();
    let low_a_grant = low_a.lifetime.memory_grant().unwrap();
    let low_b_grant = low_b.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 70);
    assert_eq!(high_grant.budget_bytes(), 10);
    assert_eq!(low_a_grant.budget_bytes(), 10);
    assert_eq!(low_b_grant.budget_bytes(), 10);

    drop(first);

    assert_eq!(high_grant.budget_bytes(), 50);
    assert_eq!(low_a_grant.budget_bytes(), 20);
    assert_eq!(low_b_grant.budget_bytes(), 20);
    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.outstanding_bytes, 90);
    assert_eq!(report.available_bytes, 10);
}

#[test]
fn adaptive_blocking_query_service_grows_after_queue_drains_when_capacity_remains() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(1),
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("wide", 10, 80), QueryServiceTenantConfig::new("small", 10)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
    let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
    let second_grant = second.lifetime.memory_grant().unwrap();
    let queued_service = service.clone();
    let queued = std::thread::spawn(move || queued_service.admit_query("small").unwrap());

    wait_for_queued_len(&service, 1);
    drop(first);

    let admitted = queued.join().expect("blocking admission thread should not panic");
    assert_eq!(admitted.device_budget_bytes(), 10);
    assert_eq!(second_grant.budget_bytes(), 80);
}

#[test]
fn adaptive_optimistic_query_service_requests_shrink_for_memory_blocked_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();

    assert_eq!(first.device_budget_bytes(), 100);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);

    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
    assert!(waiter.try_admit().unwrap().is_none());

    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after actual refund");

    assert_eq!(first_grant.budget_bytes(), 60);
    assert_eq!(admitted.device_budget_bytes(), 40);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
}

#[test]
fn query_service_release_shrinks_retained_grant_to_live_bytes() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let _retained = first_grant.retain_live_reservation(40).expect("retention should be tracked");

    assert_eq!(first_grant.budget_bytes(), 100);
    assert_eq!(first_grant.live_reserved_bytes(), 40);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);

    drop(first);

    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.outstanding_bytes, 40);
    assert_eq!(report.available_bytes, 60);
    assert_eq!(first_grant.budget_bytes(), 40);
    assert_eq!(first_grant.live_reserved_bytes(), 40);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);

    let second = service.try_admit_query("tenant").unwrap().expect("released slot and refunded budget should admit");
    assert_eq!(second.device_budget_bytes(), 60);
}

#[test]
fn adaptive_optimistic_query_service_requests_all_same_device_shrink_candidates() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(120),
            max_concurrent_queries: 3,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 30, 60)],
        non_gds_deployment(),
    )
    .unwrap();
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let second = service.try_admit_query("tenant").unwrap().expect("second query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let second_grant = second.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert_eq!(first.device_budget_bytes(), 60);
    assert_eq!(second.device_budget_bytes(), 60);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(30));
    assert_eq!(second_grant.requested_shrink_target_bytes().unwrap(), None);

    assert!(first_grant.try_apply_requested_shrink(30).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after the deficit-sized refund");

    assert_eq!(admitted.device_budget_bytes(), 30);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(second_grant.requested_shrink_target_bytes().unwrap(), None);
}

#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_drops() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));

    drop(waiter);

    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
}

#[test]
fn adaptive_optimistic_query_service_keeps_requested_shrink_for_remaining_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut first_waiter = service.enqueue_admission("tenant").unwrap();
    let second_waiter = service.enqueue_admission("tenant").unwrap();

    assert!(first_waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));

    drop(first_waiter);

    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);

    drop(second_waiter);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
}

#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_times_out() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(10),
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range("tenant", 40, 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    std::thread::sleep(Duration::from_millis(20));
    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_wait_timeout"));
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 0);
}

#[test]
fn adaptive_optimistic_query_service_rescinds_requested_shrink_when_waiter_admits() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), Some(60));
    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted = waiter.try_admit().unwrap().expect("waiter should admit after shrink");

    assert_eq!(admitted.device_budget_bytes(), 40);
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
}

#[test]
fn adaptive_blocking_query_service_wakes_after_requested_shrink_refunds_capacity() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::Adaptive);
    let first = service.admit_query("tenant").unwrap();
    let first_grant = first.lifetime.memory_grant().unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (admitted_sender, admitted_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let result = queued_service.admit_query("tenant").map(|handle| handle.device_budget_bytes());
        admitted_sender.send(result.map_err(|error| error.to_string())).unwrap();
    });

    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);
    wait_for_requested_shrink(&first_grant, Some(60));
    assert!(admitted_receiver.recv_timeout(Duration::from_millis(50)).is_err());

    assert!(first_grant.try_apply_requested_shrink(60).unwrap());
    let admitted_budget = admitted_receiver.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();

    assert_eq!(admitted_budget, 40);
    waiter.join().unwrap();
}

#[test]
fn strict_query_service_does_not_request_shrink_for_memory_blocked_waiter() {
    let service = optimistic_shrink_test_service(NativePipelineExecutionMode::StrictBounded);
    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
    let first_grant = first.lifetime.memory_grant().unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();

    assert_eq!(first.device_budget_bytes(), 100);
    assert!(waiter.try_admit().unwrap().is_none());
    assert_eq!(first_grant.requested_shrink_target_bytes().unwrap(), None);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
}

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
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
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
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
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
fn query_service_waiting_on_memory_grant_wakes_after_grant_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("tenant").unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);

    drop(active);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 100);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
    drop(admitted);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 100);
}

#[test]
fn query_service_waiting_on_memory_grant_wakes_after_retained_grant_release() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_secs(2),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let retained_grant = active.lifetime.memory_grant().unwrap();
    drop(active);
    assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);

    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("tenant").unwrap()
    });
    started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    wait_for_queued_len(&service, 1);

    drop(retained_grant);
    let admitted = waiter.join().unwrap();
    assert_eq!(admitted.device_budget_bytes(), 100);
}

#[test]
fn query_admission_waiter_times_out_while_selected_but_memory_blocked() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(20),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let _active = service.admit_query("tenant").unwrap();
    let mut waiter = service.enqueue_admission("tenant").unwrap();
    std::thread::sleep(Duration::from_millis(30));

    let error = waiter.try_admit().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_wait_timeout"));
}

#[test]
fn query_service_work_conserving_admission_skips_memory_blocked_head() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_admission_queue_len: 8,
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![
            QueryServiceTenantConfig::new("holder", 90),
            QueryServiceTenantConfig::new("large", 100),
            QueryServiceTenantConfig::new("small", 10),
        ],
        non_gds_deployment(),
    )
    .unwrap();
    let _active = service.admit_query("holder").unwrap();
    let mut large = service.enqueue_admission("large").unwrap();
    let mut small = service.enqueue_admission("small").unwrap();

    assert!(large.try_admit().unwrap().is_none());
    let admitted_small = small.try_admit().unwrap().expect("small query should pass memory-blocked large query");

    assert_eq!(admitted_small.tenant_id().as_str(), "small");
    assert_eq!(admitted_small.device_budget_bytes(), 10);
    assert_eq!(admitted_small.admission_sequence(), 2);
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 2);
    assert_eq!(snapshot.queued_queries, 1);
    assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "large");
    assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);
}

#[test]
fn query_service_admission_priority_passes_lower_priority_queue_head() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 1,
            max_admission_queue_len: 8,
            ..QueryServiceConfig::default()
        },
        vec![
            QueryServiceTenantConfig::new("holder", 100),
            QueryServiceTenantConfig::new("normal", 10),
            QueryServiceTenantConfig::new("high", 10),
        ],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("holder").unwrap();
    let mut normal = service.enqueue_admission("normal").unwrap();
    let mut high = service.enqueue_admission_with_qos("high", QueryQos::new(Priority::High)).unwrap();
    drop(active);

    assert!(normal.try_admit().unwrap().is_none());
    let admitted_high = high.try_admit().unwrap().expect("high-priority query should pass normal-priority head");

    assert_eq!(admitted_high.tenant_id().as_str(), "high");
    assert_eq!(admitted_high.admission_sequence(), 2);
    let report = service.memory_governor_report().unwrap();
    assert_eq!(report.grants.len(), 1);
    assert_eq!(report.grants[0].priority, "high");
}

#[test]
fn query_service_enqueued_waiter_counts_against_queue_limit_until_drop() {
    let service = test_service(1, 1);
    let _active = service.admit_query("large").unwrap();
    let waiter = service.enqueue_admission("small").unwrap();

    let error = service.enqueue_admission("large").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));

    drop(waiter);
    assert!(service.enqueue_admission("large").is_ok());
}

#[test]
fn query_service_sweeps_expired_unpolled_waiter_before_queue_and_try_admit_gates() {
    let service = QueryService::new_for_admission_tests(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 1,
            max_admission_queue_len: 1,
            admission_wait_timeout: Duration::from_millis(10),
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 100)],
        non_gds_deployment(),
    )
    .unwrap();
    let active = service.admit_query("tenant").unwrap();
    let stale_waiter = service.enqueue_admission("tenant").unwrap();
    std::thread::sleep(Duration::from_millis(20));

    let replacement_waiter = service.enqueue_admission("tenant").expect("expired waiter should be swept");
    assert_eq!(service.admission_snapshot().unwrap().queued_queries, 1);
    drop(replacement_waiter);
    drop(active);

    let admitted = service.try_admit_query("tenant").unwrap().expect("expired stale waiter must not block try-admit");
    assert_eq!(admitted.device_budget_bytes(), 100);
    drop(stale_waiter);
}

#[test]
fn query_service_close_wakes_waiting_admission() {
    let service = test_service(1, 8);
    let _active = service.admit_query("large").unwrap();
    let queued_service = service.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        queued_service.admit_query("small").unwrap_err()
    });
    started_receiver.recv().unwrap();
    wait_for_queued_len(&service, 1);

    service.close().unwrap();
    let error = waiter.join().unwrap();

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("admission_closed"));
}

#[test]
fn query_service_rotates_admission_by_tenant_when_queries_wait() {
    let service = test_service(1, 8);
    let first = service.admit_query("large").unwrap();
    let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
    let (result_sender, result_receiver) = std::sync::mpsc::channel();

    for tenant in ["large", "large", "small"] {
        let service = service.clone();
        let ready_sender = ready_sender.clone();
        let result_sender = result_sender.clone();
        std::thread::spawn(move || {
            ready_sender.send(tenant.to_string()).unwrap();
            let handle = service.admit_query(tenant).unwrap();
            result_sender.send(handle.tenant_id().to_string()).unwrap();
            std::thread::sleep(Duration::from_millis(20));
            drop(handle);
        });
    }
    for _ in 0..3 {
        ready_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    wait_for_queued_len(&service, 3);

    drop(first);
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "small");
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
    assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
}
