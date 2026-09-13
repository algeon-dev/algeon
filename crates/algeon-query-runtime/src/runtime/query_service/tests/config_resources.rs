#[test]
fn query_service_admission_test_config_can_record_non_default_device_without_runtime_resources() {
    let device_safety = DeviceSafetyTestScope::install([]);
    let config = QueryExecutionConfig::default();
    let service = AdmissionTestController::new_with_ledger(
        config,
        QueueBounds::default(),
        test_ledger_for_ordinal(2, 64 * 1024 * 1024, 1, Default::default()),
    )
    .unwrap();

    let handle = service.admit_query().unwrap();

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
    let config = QueryExecutionConfig {
        source_scheduler_policy: SourceSchedulerPolicy::new(4),
        source_chunk_planning_mode: SourceChunkPlanningMode::LocalParquetThroughput,
        statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::GpuNativeTolerant,
        dynamic_filter_execution_options,
        keyed_hash_join_strategy_options,
        aggregate_strategy_execution_options,
        source_chunk_planning_options,
        ..QueryExecutionConfig::default()
    };
    let service = AdmissionTestController::new(config, QueueBounds::default()).unwrap();

    let handle = service.admit_query().unwrap();

    assert_eq!(handle.boundary_policy(), ExecutionBoundaryPolicy::default());
    assert_eq!(handle.source_scheduler_policy(), SourceSchedulerPolicy::new(4));
    assert_eq!(handle.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(handle.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
    assert_eq!(handle.dynamic_filter_execution_options(), dynamic_filter_execution_options);
    assert_eq!(handle.keyed_hash_join_strategy_options(), keyed_hash_join_strategy_options);
    assert_eq!(handle.aggregate_strategy_execution_options(), aggregate_strategy_execution_options);
    assert_eq!(handle.source_chunk_planning_options(), source_chunk_planning_options);
}
#[test]
fn query_service_rejects_inconsistent_dynamic_filter_execution_policy() {
    let config = QueryExecutionConfig {
        dynamic_filter_execution_options: DynamicFilterExecutionOptions::default()
            .with_max_build_rows(10)
            .with_large_local_parquet_range_producer_rows(11),
        ..QueryExecutionConfig::default()
    };

    let error = AdmissionTestController::new(config, QueueBounds::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
    assert_eq!(error.fact_value("bound_field"), Some("max_build_rows"));
    assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
}
#[test]
fn query_service_rejects_inconsistent_keyed_hash_join_strategy_policy() {
    let config = QueryExecutionConfig {
        keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default()
            .with_large_keyed_hash_build_row_threshold(10),
        ..QueryExecutionConfig::default()
    };

    let error = AdmissionTestController::new(config, QueueBounds::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("large_field"), Some("large_keyed_hash_build_row_threshold"));
    assert_eq!(error.fact_value("small_field"), Some("small_reusable_hash_build_row_threshold"));
    assert_eq!(error.fact_value("reason"), Some("large_threshold_below_small_threshold"));
}
#[test]
fn query_service_rejects_invalid_aggregate_strategy_execution_policy() {
    let config = QueryExecutionConfig {
        aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default()
            .with_prepared_sort_groupby_low_cardinality_threshold_ppm(1_000_001),
        ..QueryExecutionConfig::default()
    };

    let error = AdmissionTestController::new(config, QueueBounds::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("prepared_sort_groupby_low_cardinality_threshold_ppm"));
    assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
}
#[test]
fn query_service_rejects_invalid_source_chunk_planning_policy() {
    let config = QueryExecutionConfig {
        source_chunk_planning_options: SourceChunkPlanningOptions::default()
            .with_local_parquet_throughput_max_row_groups_per_chunk(0),
        ..QueryExecutionConfig::default()
    };

    let error = AdmissionTestController::new(config, QueueBounds::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}
#[test]
fn query_service_resources_use_shared_lakehouse_cache_fill_limits() {
    let config = QueryExecutionConfig {
        lakehouse_cache_fill: LakehouseCacheFillConfig {
            budget_bytes: Some(2048),
            acquire_timeout: Duration::from_secs(9),
            max_concurrent_ops: 2,
        },
        ..QueryExecutionConfig::default()
    };
    let service =
        runtime_test_service_with_profiles(config, QueueBounds::default(), vec![runtime_test_device_profile(2)])
            .unwrap();
    let resources = service.inner.resources.as_ref().unwrap();

    assert_eq!(resources.lakehouse_cache_fill_budget.budget_bytes(), 2048);
    assert_eq!(resources.lakehouse_cache_fill_concurrency.max_concurrent_ops(), 2);

    let handle = service.admit_query().unwrap();
    let admitted_context = handle.acquire_context().unwrap();
    let context = admitted_context.context();
    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
    assert_eq!(context.lakehouse_cache_fill_acquire_timeout(), Duration::from_secs(9));
    assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 2);

    context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Completed,
        bytes: Some(128),
        elapsed: Duration::from_millis(1),
        cache_path_hash: "cache-path-hash".to_string(),
        stable_id_hash: "stable-id-hash".to_string(),
        file_index: None,
        chunk_index: None,
        error: None,
    });
    assert_eq!(context.lakehouse_cache_fill_event_snapshots().len(), 1);

    let other_handle = service.admit_query().unwrap();
    let other_admitted_context = other_handle.acquire_context().unwrap();
    assert!(other_admitted_context.context().lakehouse_cache_fill_event_snapshots().is_empty());
}
#[test]
fn admission_only_query_service_honors_per_device_active_attempt_limit() {
    let service = AdmissionTestController::new(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 4, max_wait: Duration::from_secs(30) },
    )
    .unwrap();

    let first = service.try_admit_query().unwrap().expect("first query should admit");

    assert!(service.try_admit_query().unwrap().is_none());
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    drop(first);
    assert!(service.try_admit_query().unwrap().is_some());
}
