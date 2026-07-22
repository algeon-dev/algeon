#[tokio::test]
async fn iceberg_provider_plans_structured_table_format_scan_leaf() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new();
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();

    let scan = find_table_format_scan(physical_plan.as_ref()).expect("plan should contain table-format scan leaf");
    assert_eq!(scan.schema().fields().len(), 1);
    assert_eq!(scan.schema().field(0).name(), "value");
    assert_eq!(scan.scan_plan().format, TableFormat::Iceberg);
    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&["value".to_string()][..]));

    assert_eq!(scan.name(), "NexusTableFormatScanExec");

    let lowering = lower_physical_plan_to_native(Arc::clone(&physical_plan));
    assert_eq!(lowering.rewritten_count(), 1);
    let source = native_local_parquet_source(&lowering.plans[0]);
    assert_eq!(source.read_columns, vec!["value".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["value".to_string()]);
}

#[tokio::test]
async fn iceberg_provider_unprojected_unsupported_map_field_does_not_block_native_projection() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("allRiskEvents", map_utf8_int32_type(), true),
    ]));
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema))
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select id from iceberg_t limit 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].projected_column_names.as_deref(), Some(&["id".to_string()][..]));
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["id".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["id".to_string()]);
}

#[tokio::test]
async fn iceberg_provider_projected_unsupported_map_field_rejects_native_projection() {
    let map_type = map_utf8_int32_type();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("allRiskEvents", map_type.clone(), true),
    ]));
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema))
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "allRiskEvents",
            Some("allRiskEvents"),
            map_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select * from iceberg_t limit 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));
}

#[tokio::test]
async fn iceberg_provider_scan_planning_timeout_returns_structured_error() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(SleepingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(50)));
    let provider = NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
        .with_catalog_type("local")
        .with_scan_planning_timeout(Some(Duration::from_millis(1)));
    let ctx = SessionContext::new();

    let error = provider.scan(&ctx.state(), None, &[], None).await.expect_err("scan planning must time out");
    let adapter_error = adapter_error_from(&error);

    assert_eq!(adapter_error.code(), ErrorCode::PlanningTimeout);
    assert_eq!(adapter_error.status(), ErrorStatus::Permanent);
    assert_eq!(adapter_error.fact_value("table_format"), Some("iceberg"));
    assert_eq!(adapter_error.fact_value("catalog_type"), Some("local"));
    assert_eq!(adapter_error.fact_value("planning_timeout"), Some("true"));
    assert_eq!(adapter_error.fact_value("planning_timeout_ms"), Some("1"));
    assert_eq!(adapter_error.fact_value("planning_timeout_stage"), Some("provider_timeout"));
    assert_eq!(adapter_error.fact_value("timeout_retryable"), Some("false"));
    assert_eq!(adapter_error.fact_value("timeout_budget_increase_required"), Some("true"));
    assert_eq!(adapter_error.fact_value("test_planner_fact"), Some("slow"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn iceberg_provider_scan_planning_timeout_returns_before_non_yielding_planner_finishes() {
    let schema = iceberg_test_schema();
    let planner =
        Arc::new(BusyPlanner::new(native_admissible_scan_plan(Arc::clone(&schema)), Duration::from_millis(30)));
    let provider = NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
        .with_catalog_type("local")
        .with_scan_planning_timeout(Some(Duration::from_millis(1)));
    let ctx = SessionContext::new();

    let started = Instant::now();
    let error = provider.scan(&ctx.state(), None, &[], None).await.expect_err("scan planning must time out");
    let elapsed = started.elapsed();
    let adapter_error = adapter_error_from(&error);

    assert_eq!(adapter_error.code(), ErrorCode::PlanningTimeout);
    assert_eq!(adapter_error.fact_value("planning_timeout_stage"), Some("provider_timeout"));
    assert_eq!(adapter_error.fact_value("test_planner_fact"), Some("busy"));
    assert!(
        elapsed < Duration::from_millis(200),
        "provider timeout should not wait for the busy planner to finish; elapsed={elapsed:?}"
    );
}

#[tokio::test]
async fn iceberg_provider_plumbs_explicit_scan_limit_into_table_format_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new();
    let projection = vec![1];
    let physical_plan = provider.scan(&ctx.state(), Some(&projection), &[], Some(7)).await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].scan_limit, Some(7));
    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(7));
    assert_eq!(
        source.files[0].row_group_rows.as_deref(),
        None,
        "Iceberg file-level row counts must not synthesize per-row-group metadata"
    );
}

#[test]
fn native_lowering_pushes_root_global_limit_into_table_format_source_scan_limit() {
    let schema = iceberg_test_schema();
    let scan: Arc<dyn ExecutionPlan> =
        Arc::new(NexusTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));
    let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 3, Some(7)));

    let result = lower_physical_plan_to_native(limited);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(10), "source reads must cover skip + fetch before the native Limit trims");
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(native_plan_limit_count(&result.plans[0]) >= 1, "native Limit must still preserve root LIMIT semantics");
}
