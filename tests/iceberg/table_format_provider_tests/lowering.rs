#[tokio::test]
async fn iceberg_provider_lowers_q09_style_scalar_aggregate_subqueries_as_one_native_plan() {
    let reason_schema = Arc::new(schema_with_source_unique_keys(
        &Schema::new(vec![Field::new("r_reason_sk", DataType::Int64, false)]),
        &[&["r_reason_sk"]],
    ));
    let store_sales_schema = Arc::new(Schema::new(vec![
        Field::new("ss_quantity", DataType::Int64, false),
        Field::new("ss_ext_discount_amt", DataType::Float64, true),
        Field::new("ss_net_paid", DataType::Float64, true),
    ]));
    let reason_planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan_for_schema(
        Arc::clone(&reason_schema),
        "reason-data-0001",
        "/tmp/iceberg/reason.parquet",
    )));
    let store_sales_planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan_for_schema(
        Arc::clone(&store_sales_schema),
        "store-sales-data-0001",
        "/tmp/iceberg/store_sales.parquet",
    )));
    let reason =
        Arc::new(NexusIcebergTableProvider::from_table_format_planner(Arc::clone(&reason_schema), reason_planner));
    let store_sales = Arc::new(NexusIcebergTableProvider::from_table_format_planner(
        Arc::clone(&store_sales_schema),
        store_sales_planner,
    ));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("reason", reason).unwrap();
    ctx.register_table("store_sales", store_sales).unwrap();

    let df = ctx
        .sql(
            "
            select
              case
                when (select count(*) from store_sales where ss_quantity between 1 and 20) > 74129
                then (select avg(ss_ext_discount_amt) from store_sales where ss_quantity between 1 and 20)
                else (select avg(ss_net_paid) from store_sales where ss_quantity between 1 and 20)
              end as bucket1
            from reason
            where r_reason_sk = 1
            ",
        )
        .await
        .unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.plans.len(), 1);
    assert_eq!(
        result.plans[0].output_schema().fields().len(),
        1,
        "root projection must preserve the single bucket output"
    );
    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 4, "reason plus three scalar aggregate sources should lower into one native plan");
    assert!(
        sources.iter().any(|source| {
            source.files[0].stable_id == "table-format:iceberg:reason-data-0001"
                && source.projected_output_columns.is_empty()
                && source.pure_filter_columns == vec!["r_reason_sk".to_string()]
                && source.max_rows == Some(1)
        }),
        "the reason scan is only a cardinality/filter input, should prove max_rows=1, and should not leak filter columns to its parent"
    );
}

#[test]
fn native_lowering_preserves_fetch_pushed_to_table_format_filter_and_coalesce() {
    let schema = iceberg_test_schema();
    let scan: Arc<dyn ExecutionPlan> =
        Arc::new(NexusTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));
    let scan_schema = scan.schema();
    let predicate =
        binary(col("id", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int64(Some(1))), &scan_schema).unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter = filter.with_fetch(Some(5)).expect("FilterExec should accept fetch");
    let coalesce: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(filter));
    let coalesce = coalesce.with_fetch(Some(5)).expect("CoalescePartitionsExec should accept fetch");

    let result = lower_physical_plan_to_native(coalesce);

    assert_eq!(result.rewritten_count(), 1);
    assert!(native_plan_limit_count(&result.plans[0]) >= 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "fetch above an inexact filter must lower after the native filter");
}

#[test]
fn native_lowering_lowers_table_format_scan_to_local_parquet_source() {
    let schema = iceberg_test_schema();
    let scan = Arc::new(NexusTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("expected table-format scan to be rewritten by Milestone D lowering");
    };
    assert_eq!(candidate.shape, "TableFormatScan(iceberg)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files.len(), 1);
    assert_eq!(source.files[0].path.to_string_lossy(), "/tmp/iceberg/data-0001.parquet");
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:data-0001");
    assert_eq!(source.read_columns, vec!["id".to_string(), "value".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["id".to_string(), "value".to_string()]);
}

#[test]
fn native_lowering_lowers_s3_table_format_scan_to_object_store_parquet_source() {
    let schema = iceberg_test_schema();
    let mut plan = native_admissible_scan_plan(Arc::clone(&schema));
    plan.data_files = vec![
        TableFormatDataFile::new(
            "s3-data-0001",
            "s3://bucket/table/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(4096),
    ];
    let scan = Arc::new(NexusTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("expected S3 table-format scan to be rewritten through remote KvikIO lowering");
    };
    assert_eq!(candidate.shape, "TableFormatScan(iceberg)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files.len(), 1);
    assert_eq!(source.files[0].location.read_capability(), "object_store:s3");
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:s3-data-0001");
    assert_eq!(source.files[0].path.to_string_lossy(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(source.files[0].resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect);
    assert!(matches!(&source.files[0].resolved_source.input, ResolvedParquetInput::RemoteKvikio { .. }));
    assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
}

#[test]
fn native_lowering_preserves_remote_direct_reader_contract_for_dispatch() {
    let schema = iceberg_test_schema();
    let mut plan = native_admissible_scan_plan(Arc::clone(&schema));
    plan.data_files = vec![
        TableFormatDataFile::new(
            "s3-data-0001",
            "s3://bucket/table/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(4096)
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/data-0001.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    ];
    let scan = Arc::new(NexusTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    let file = &source.files[0];
    assert_eq!(file.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect);
    assert!(matches!(&file.resolved_source.input, ResolvedParquetInput::RemoteKvikio { .. }));
    assert_eq!(file.path.to_string_lossy(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
}

#[test]
fn table_format_scan_preserves_unsupported_feature_reason_and_never_executes_cpu_arrow_reader() {
    let schema = iceberg_test_schema();
    let plan =
        native_admissible_scan_plan(Arc::clone(&schema)).with_unsupported_feature(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            "delete files require Milestone 2 row-level semantics",
        ));
    let scan = Arc::new(NexusTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(Arc::clone(&scan) as Arc<dyn ExecutionPlan>);

    let CandidateOutcome::Rejected { reason, .. } = &result.report.outcomes()[0] else {
        panic!("expected rejected native candidate");
    };
    assert_eq!(*reason, FallbackReason::IcebergDeleteFilesUnsupported);

    let err = match scan.execute(0, Arc::new(TaskContext::default())) {
        Ok(_) => panic!("scan leaf should not execute"),
        Err(error) => error,
    };
    let rendered = err.to_string();
    assert!(rendered.contains("does not execute a CPU Arrow reader"));
    let nexus = err.find_nexus_error().expect("table-format execution error must preserve typed Nexus facts");
    assert_eq!(nexus.fact_value("cpu_arrow_reader_fallback"), Some("false"));
    assert_eq!(nexus.fact_value("table_format"), Some("iceberg"));
}
