#[tokio::test]
async fn iceberg_provider_count_star_lowers_to_table_format_zero_column_count_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_empty_scan_plan()));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    assert_eq!(source.files[0].row_group_rows.as_deref(), Some(&[11][..]));
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:data-0001");
}

#[tokio::test]
async fn iceberg_provider_exact_empty_cpu_delegate_returns_zero_without_native_scan_execution() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_empty_scan_plan()));
    let cpu_provider: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).expect("empty CPU provider must build"));
    let provider =
        Arc::new(NexusIcebergTableProvider::from_cpu_provider_and_native_planner(cpu_provider, None, planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let batches = collect(physical_plan, Arc::new(TaskContext::default())).await.unwrap();

    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    assert_eq!(batches.len(), 1);
    let counts = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(counts.values(), &[0]);
}

#[tokio::test]
async fn iceberg_provider_count_star_reads_fallback_column_when_row_count_metadata_is_missing() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(mixed_row_count_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    assert_eq!(native_plan_union_count(&result.plans[0]), 1);
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    let metadata_source = sources.iter().copied().find(|source| source.is_zero_column_count_source()).unwrap();
    assert_eq!(metadata_source.files.len(), 1);
    assert_eq!(metadata_source.files[0].row_group_rows.as_deref(), Some(&[11][..]));
    let fallback_source = sources.iter().copied().find(|source| !source.is_zero_column_count_source()).unwrap();
    assert_eq!(fallback_source.files.len(), 1);
    assert_eq!(fallback_source.read_columns, vec!["id".to_string()]);
    assert_eq!(fallback_source.read_column_file_indices, vec![0]);
    assert_eq!(fallback_source.projected_output_columns, Vec::<String>::new());
    assert_eq!(fallback_source.pure_filter_columns, vec!["id".to_string()]);
    assert_eq!(fallback_source.schema.fields().len(), 1);
}

#[tokio::test]
async fn iceberg_provider_count_star_reads_deleted_files_even_with_row_count_metadata() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(position_delete_row_count_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(native_plan_union_count(&result.plans[0]), 1);
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    let metadata_source = sources.iter().copied().find(|source| source.is_zero_column_count_source()).unwrap();
    assert_eq!(metadata_source.files.len(), 1);
    assert_eq!(metadata_source.files[0].stable_id, "table-format:iceberg:data-0001");
    let fallback_source = sources.iter().copied().find(|source| !source.is_zero_column_count_source()).unwrap();
    assert_eq!(fallback_source.files.len(), 1);
    assert_eq!(fallback_source.files[0].stable_id, "table-format:iceberg:data-0002");
    assert_eq!(fallback_source.files[0].row_group_rows, None);
    assert_eq!(fallback_source.files[0].position_delete_files.len(), 1);
    assert_eq!(fallback_source.read_columns, vec!["id".to_string()]);
    assert_eq!(fallback_source.projected_output_columns, Vec::<String>::new());
    assert_eq!(fallback_source.pure_filter_columns, vec!["id".to_string()]);
}

#[tokio::test]
async fn iceberg_provider_count_star_probe_rejects_unproven_remote_carrier_column() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(RecordingPlanner::new(remote_mixed_row_count_scan_plan_without_probe_mapping(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceSchemaMapping, .. }
    ));
}

#[tokio::test]
async fn iceberg_provider_count_star_fallback_executes_with_mixed_row_count_metadata() {
    let dir = unique_temp_dir("nexus-iceberg-count-star-fallback");
    let first_path = dir.join("data-0001.parquet");
    let second_path = dir.join("data-0002.parquet");
    let schema = iceberg_test_schema();
    write_iceberg_test_parquet(&first_path, &[1, 2], &[Some("a"), Some("b")], Arc::clone(&schema));
    write_iceberg_test_parquet(&second_path, &[3, 4, 5], &[Some("c"), None, Some("d")], Arc::clone(&schema));

    let planner = Arc::new(RecordingPlanner::new(mixed_row_count_scan_plan_for_files(
        Arc::clone(&schema),
        &first_path,
        &second_path,
    )));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(native_plan_union_count(&result.plans[0]), 1);
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    let fallback_source = sources.iter().copied().find(|source| !source.is_zero_column_count_source()).unwrap();
    assert_eq!(fallback_source.projected_output_columns, Vec::<String>::new());
    assert_eq!(fallback_source.pure_filter_columns, vec!["id".to_string()]);

    let backend = datafusion_nexus::backend::NexusGpuBackend::builder()
        .memory_policy(datafusion_nexus::backend::NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 256 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .device_memory_ownership([(
            0,
            datafusion_nexus::backend::GpuMemoryOwnership::WholeDeviceExclusive,
        )])
        .build()
        .unwrap();
    let handle = backend.admit_query().unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let execution_context: &ExecutionContext = tenant_context.context();
    let result = execute_query_plan(execution_context, &result.plans[0]).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("COUNT(*) plan should materialize a host Arrow batch");
    };
    let values = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(values.value(0), 5);

    drop(result);
    drop(tenant_context);
    handle.finish();
    drop(handle);
    backend.close_and_wait(Duration::from_secs(5)).unwrap();

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn iceberg_provider_count_star_ignores_unused_non_identity_partition_transform_facts() {
    let schema = transformed_partition_test_schema();
    let planner = Arc::new(RecordingPlanner::new(identity_and_day_partition_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(test_iceberg_provider(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());
}
