#[test]
fn query_service_admission_test_config_can_record_non_default_device_without_runtime_resources() {
    let device_safety = cudf::testing::device_safety::DeviceSafetyTestScope::install([]);
    let config = QueryServiceConfig { device_ordinal: 2, ..QueryServiceConfig::default() };
    let service = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();

    assert_eq!(handle.device_ordinal(), 2);
    assert_eq!(device_safety.remaining_call_count(), 0, "fake-device admission must not issue CUDA device-entry calls");
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
    let service = AdmissionTestController::new(
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

    let error = AdmissionTestController::new(
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

    let error = AdmissionTestController::new(
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

    let error = AdmissionTestController::new(
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

    let error = AdmissionTestController::new(
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
fn query_service_rejects_zero_local_parquet_source_cache_capacity() {
    let config =
        QueryServiceConfig { local_parquet_source_cache_max_retained_bytes: Some(0), ..QueryServiceConfig::default() };

    let error = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_source_cache_max_retained_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}
#[test]
fn query_service_query_memory_domain_policy_defaults_off_and_rejects_zero_multiplier() {
    assert_eq!(QueryServiceConfig::default().query_memory_domain, None);
    assert_eq!(QueryMemoryDomainPolicy::default().ceiling_multiplier, 2);

    let config = QueryServiceConfig {
        query_memory_domain: Some(QueryMemoryDomainPolicy::new().with_ceiling_multiplier(0)),
        ..QueryServiceConfig::default()
    };

    let error = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("query_memory_domain.ceiling_multiplier"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}
#[test]
fn gpu_execution_workers_follow_the_per_device_admission_ceiling() {
    let mut config = QueryServiceConfig { max_concurrent_queries: 16, ..QueryServiceConfig::default() };
    assert_eq!(gpu_execution_worker_count_per_device(&config), 16);

    config.max_queries_per_device = Some(4);
    assert_eq!(gpu_execution_worker_count_per_device(&config), 4);

    config.max_queries_per_device = Some(32);
    assert_eq!(gpu_execution_worker_count_per_device(&config), 16);
}
#[test]
fn query_service_rejects_query_memory_domain_with_durable_gpu_caches() {
    let source_cache_config = QueryServiceConfig {
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        local_parquet_source_cache_max_retained_bytes: Some(4096),
        ..QueryServiceConfig::default()
    };

    let source_cache_error = AdmissionTestController::new(
        source_cache_config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(source_cache_error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(source_cache_error.fact_value("reason"), Some("query_memory_domain_incompatible_with_source_cache"));

    #[cfg(feature = "cugraph")]
    {
        let graph_cache_config = QueryServiceConfig {
            query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
            resident_graph_cache_max_retained_bytes: Some(4096),
            ..QueryServiceConfig::default()
        };

        let graph_cache_error = AdmissionTestController::new(
            graph_cache_config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(graph_cache_error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(graph_cache_error.fact_value("reason"), Some("query_memory_domain_incompatible_with_graph_cache"));
    }
}
#[test]
fn query_service_validates_pool_allocator_and_rejects_private_query_domains() {
    let zero_pool = QueryServiceConfig {
        device_memory_manager_base: cudf::DeviceMemoryManagerBase::Pool {
            initial_pool_size: 0,
            maximum_pool_size: Some(4096),
        },
        ..QueryServiceConfig::default()
    };
    let zero_error = AdmissionTestController::new(
        zero_pool,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(zero_error.fact_value("field"), Some("device_memory_manager_base.initial_pool_size"));
    assert_eq!(zero_error.fact_value("reason"), Some("zero"));

    let private_domain = QueryServiceConfig {
        device_memory_manager_base: cudf::DeviceMemoryManagerBase::Pool {
            initial_pool_size: 4096,
            maximum_pool_size: Some(8192),
        },
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        ..QueryServiceConfig::default()
    };
    let domain_error = AdmissionTestController::new(
        private_domain,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();
    assert_eq!(domain_error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(domain_error.fact_value("reason"), Some("query_memory_domain_requires_cuda_async_manager"));
}
#[cfg(feature = "cugraph")]
#[test]
fn query_service_rejects_zero_resident_graph_cache_capacity() {
    let config =
        QueryServiceConfig { resident_graph_cache_max_retained_bytes: Some(0), ..QueryServiceConfig::default() };

    let error = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("resident_graph_cache_max_retained_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}
#[cfg(not(feature = "cugraph"))]
#[test]
fn query_service_rejects_resident_graph_cache_without_cugraph_feature() {
    let config =
        QueryServiceConfig { resident_graph_cache_max_retained_bytes: Some(4096), ..QueryServiceConfig::default() };

    let error = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("resident_graph_cache_max_retained_bytes"));
    assert_eq!(error.fact_value("reason"), Some("cugraph_feature_disabled"));
}
#[test]
fn local_parquet_source_cache_resources_are_per_device() {
    let caches = local_parquet_source_caches_for_ordinals(&[0, 2], Some(4096));

    let device_0 = caches.get(&0).expect("device 0 cache should exist");
    let device_2 = caches.get(&2).expect("device 2 cache should exist");
    assert_eq!(caches.len(), 2);
    assert_eq!(device_0.max_retained_device_bytes(), 4096);
    assert_eq!(device_2.max_retained_device_bytes(), 4096);
    assert!(!Arc::ptr_eq(device_0, device_2));
    assert!(local_parquet_source_caches_for_ordinals(&[0, 2], None).is_empty());
}
#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_cache_resources_are_per_device() {
    let caches = resident_graph_caches_for_ordinals(&[0, 2], Some(8192), 8);

    let device_0 = caches.get(&0).expect("device 0 cache should exist");
    let device_2 = caches.get(&2).expect("device 2 cache should exist");
    assert_eq!(caches.len(), 2);
    assert_eq!(device_0.max_retained_device_bytes(), 8192);
    assert_eq!(device_2.max_retained_device_bytes(), 8192);
    assert_eq!(device_0.max_copies_per_key(), 8);
    assert_eq!(device_2.max_copies_per_key(), 8);
    assert!(!Arc::ptr_eq(device_0, device_2));
    assert!(resident_graph_caches_for_ordinals(&[0, 2], None, 8).is_empty());
}
#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_cache_replica_limit_matches_per_device_admission_ceiling() {
    let config = QueryServiceConfig { max_concurrent_queries: 16, ..QueryServiceConfig::default() };
    assert_eq!(resident_graph_cache_max_copies_per_key(&config), 16);

    let config = QueryServiceConfig {
        max_concurrent_queries: 16,
        max_queries_per_device: Some(6),
        ..QueryServiceConfig::default()
    };
    assert_eq!(resident_graph_cache_max_copies_per_key(&config), 6);

    let config = QueryServiceConfig {
        max_concurrent_queries: 4,
        max_queries_per_device: Some(6),
        ..QueryServiceConfig::default()
    };
    assert_eq!(resident_graph_cache_max_copies_per_key(&config), 4);
}
#[test]
fn query_service_resources_use_shared_lakehouse_cache_fill_limits() {
    let config = QueryServiceConfig {
        lakehouse_cache_fill: LakehouseCacheFillConfig {
            budget_bytes: Some(2048),
            acquire_timeout: Duration::from_secs(9),
            max_concurrent_ops: 2,
        },
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();
    let resources = service.inner.resources.as_ref().unwrap();

    assert_eq!(resources.lakehouse_cache_fill_budget.budget_bytes(), 2048);
    assert_eq!(resources.lakehouse_cache_fill_concurrency.max_concurrent_ops(), 2);

    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let context = tenant_context.context();
    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
    assert_eq!(context.lakehouse_cache_fill_acquire_timeout(), Duration::from_secs(9));
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
fn query_service_rejects_stream_pool_smaller_than_admission_limit() {
    let config =
        QueryServiceConfig { max_concurrent_queries: 2, stream_pool_size: Some(1), ..QueryServiceConfig::default() };

    let error = AdmissionTestController::new(
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

    let error = AdmissionTestController::new(
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
fn query_service_rejects_zero_max_queries_per_device() {
    let config = QueryServiceConfig { max_queries_per_device: Some(0), ..QueryServiceConfig::default() };

    let error = AdmissionTestController::new(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("max_queries_per_device"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}
#[test]
fn admission_only_query_service_honors_max_queries_per_device_cap() {
    let service = AdmissionTestController::new(
        QueryServiceConfig {
            device_capacity_bytes: Some(100),
            max_concurrent_queries: 2,
            max_queries_per_device: Some(1),
            slot_guarantee: SlotGuarantee::Optimistic,
            max_admission_queue_len: 4,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 10)],
        non_gds_deployment(),
    )
    .unwrap();

    let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");

    assert!(service.try_admit_query("tenant").unwrap().is_none());
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    drop(first);
    assert!(service.try_admit_query("tenant").unwrap().is_some());
}
