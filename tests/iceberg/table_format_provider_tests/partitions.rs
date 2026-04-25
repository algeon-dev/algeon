#[tokio::test]
async fn iceberg_provider_reads_non_identity_partition_source_column_from_parquet() {
    let schema = transformed_partition_test_schema();
    let planner = Arc::new(RecordingPlanner::new(identity_and_day_partition_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select \"customerId\", event_ts from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(
        result.plans[0].output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["customerId", "event_ts"]
    );
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["customerId", "event_ts"]
    );
    assert_eq!(source.read_columns, vec!["event_ts".to_string()]);
    assert_eq!(source.read_column_file_indices, vec![1]);
    assert_eq!(source.projected_output_columns, vec!["customerId".to_string(), "event_ts".to_string()]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::Partition(0), SourceColumnRef::File(1)]
    );
    assert_eq!(source.partition_columns.len(), 1);
    assert_eq!(source.partition_columns[0].name, "customerId");
    assert_eq!(source.files[0].partition_values, vec![NativeLiteralValue::Utf8("customer-1".to_string())]);
}

#[tokio::test]
async fn iceberg_provider_materializes_identity_partition_only_projection() {
    let schema = transformed_partition_test_schema();
    let planner = Arc::new(RecordingPlanner::new(identity_and_day_partition_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select \"customerId\" from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, Vec::<String>::new());
    assert_eq!(source.projected_output_columns, vec!["customerId".to_string()]);
    assert_eq!(source.projected_output_column_source_refs, vec![SourceColumnRef::Partition(0)]);
    assert_eq!(source.partition_columns.len(), 1);
    assert_eq!(source.files[0].partition_values, vec![NativeLiteralValue::Utf8("customer-1".to_string())]);
    assert_eq!(source.files[0].row_group_rows.as_deref(), Some(&[11][..]));
}

#[tokio::test]
async fn iceberg_provider_materializes_timestamp_and_decimal_identity_partitions() {
    let schema = timestamp_decimal_partition_test_schema();
    let planner = Arc::new(RecordingPlanner::new(timestamp_decimal_identity_partition_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select event_ts, amount from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, Vec::<String>::new());
    assert_eq!(source.partition_columns.len(), 2);
    assert_eq!(
        source.files[0].partition_values,
        vec![
            NativeLiteralValue::Timestamp { raw: 1_700_000_000_123_456, unit: TimeUnit::Microsecond },
            NativeLiteralValue::Decimal128 { raw: 12_345, precision: 10, scale: 2 },
        ]
    );
    assert_eq!(source.files[0].row_group_rows.as_deref(), Some(&[7][..]));
}

#[tokio::test]
async fn iceberg_provider_rejects_mixed_file_and_partition_source_mapping_for_one_column() {
    let schema = transformed_partition_test_schema();
    let plan =
        identity_and_day_partition_scan_plan(Arc::clone(&schema)).with_projected_column(TableFormatColumnMapping::new(
            None,
            "customerId",
            Some("customerId"),
            DataType::Utf8,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select \"customerId\" from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceSchemaMapping, .. }
    ));
}
