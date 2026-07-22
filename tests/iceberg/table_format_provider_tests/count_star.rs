#[tokio::test]
async fn iceberg_provider_count_star_lowers_to_table_format_zero_column_count_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_empty_scan_plan()));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    assert_eq!(source.files[0].row_group_rows.as_deref(), Some(&[11][..]));
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:data-0001");
}

#[tokio::test]
async fn iceberg_provider_count_star_reads_fallback_column_when_row_count_metadata_is_missing() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(mixed_row_count_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceSchemaMapping, .. }
    ));
}

#[tokio::test]
async fn iceberg_provider_count_star_fallback_executes_with_mixed_row_count_metadata() {
    let dir = unique_temp_dir("datafusion-nexus-iceberg-count-star-fallback");
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
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(native_plan_union_count(&result.plans[0]), 1);
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    let fallback_source = sources.iter().copied().find(|source| !source.is_zero_column_count_source()).unwrap();
    assert_eq!(fallback_source.projected_output_columns, Vec::<String>::new());
    assert_eq!(fallback_source.pure_filter_columns, vec!["id".to_string()]);

    let execution_context = ExecutionContext::from_runtime(cudf::Runtime::new().unwrap()).unwrap();
    let result = execute_query_plan(&execution_context, &result.plans[0]).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("COUNT(*) plan should materialize a host Arrow batch");
    };
    let values = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(values.value(0), 5);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn iceberg_provider_count_star_ignores_unused_non_identity_partition_transform_facts() {
    let schema = transformed_partition_test_schema();
    let planner = Arc::new(RecordingPlanner::new(identity_and_day_partition_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select count(*) from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&[][..]));
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());
}
