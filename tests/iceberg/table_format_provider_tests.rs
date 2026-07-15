use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, NullEquality, ScalarValue},
    datasource::TableProvider,
    execution::context::TaskContext,
    logical_expr::{JoinType, Operator},
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        ExecutionPlan,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        joins::{HashJoinExec, PartitionMode},
        limit::GlobalLimitExec,
    },
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    error::{ErrorCode, ErrorStatus},
    native::lower_physical_plan_to_native,
    policy::FallbackReason,
    report::CandidateOutcome,
    source_cardinality::schema_with_source_unique_keys,
    table_format::{
        CudfTableFormatScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
        TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent,
        TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig,
        TableFormatParquetReadStrategyPreference, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
        TableFormatPartitionValue, TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue,
        TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
        iceberg::CudfIcebergTableProvider, resolve_table_format_parquet_source,
        resolve_table_format_parquet_source_with_metadata,
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    exec::{execute_query_plan, output::QueryOutput},
    expr::LiteralValue as NativeLiteralValue,
    plan::PlanNode,
    runtime::ExecutionContext,
    source::{
        CredentialMode, LakehouseCacheStatus, ParquetReadStrategy, ResolvedParquetSource, SourceColumnRef,
        SourceDecodeInput, SourceNode,
    },
};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

#[derive(Debug)]
struct RecordingPlanner {
    plan: TableFormatScanPlan,
    requests: Mutex<Vec<TableFormatScanRequest>>,
}

#[derive(Debug)]
struct SleepingPlanner {
    plan: TableFormatScanPlan,
    delay: Duration,
}

#[derive(Debug)]
struct BusyPlanner {
    plan: TableFormatScanPlan,
    spin_for: Duration,
}

impl SleepingPlanner {
    fn new(plan: TableFormatScanPlan, delay: Duration) -> Self {
        Self { plan, delay }
    }
}

impl BusyPlanner {
    fn new(plan: TableFormatScanPlan, spin_for: Duration) -> Self {
        Self { plan, spin_for }
    }
}

impl TableFormatPlanner for SleepingPlanner {
    fn plan_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(plan)
        })
    }

    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        vec![("test_planner_fact", "slow".to_string())]
    }
}

impl TableFormatPlanner for BusyPlanner {
    fn plan_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        let spin_for = self.spin_for;
        Box::pin(async move {
            let deadline = Instant::now() + spin_for;
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
            Ok(plan)
        })
    }

    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        vec![("test_planner_fact", "busy".to_string())]
    }
}

impl RecordingPlanner {
    fn new(plan: TableFormatScanPlan) -> Self {
        Self { plan, requests: Mutex::new(Vec::new()) }
    }

    fn requests(&self) -> Vec<TableFormatScanRequest> {
        self.requests.lock().expect("request recorder mutex poisoned").clone()
    }
}

impl TableFormatPlanner for RecordingPlanner {
    fn plan_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let scan_limit = request.scan_limit;
        self.requests.lock().expect("request recorder mutex poisoned").push(request);
        let plan = self.plan.clone().with_scan_limit(scan_limit);
        Box::pin(async move { Ok(plan) })
    }
}

#[tokio::test]
async fn iceberg_provider_plans_structured_table_format_scan_leaf() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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

    assert_eq!(scan.name(), "CudfTableFormatScanExec");

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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
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
    let provider = CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
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
    let provider = CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner)
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
async fn iceberg_provider_count_star_lowers_to_table_format_zero_column_count_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_empty_scan_plan()));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner));
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
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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

#[tokio::test]
async fn iceberg_provider_plumbs_explicit_scan_limit_into_table_format_source() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
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
        Arc::new(CudfTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));
    let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 3, Some(7)));

    let result = lower_physical_plan_to_native(limited);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(10), "source reads must cover skip + fetch before the native Limit trims");
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(native_plan_limit_count(&result.plans[0]) >= 1, "native Limit must still preserve root LIMIT semantics");
}

#[tokio::test]
async fn iceberg_provider_accepts_inexact_filter_without_losing_filter_columns() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(planner.requests().len(), 1);
    assert_eq!(planner.requests()[0].scan_limit, None);
    let request_predicate = format!("{:?}", planner.requests()[0].predicate);
    assert!(request_predicate.contains("Comparison"), "{request_predicate}");
    assert!(request_predicate.contains("column: \"id\""), "{request_predicate}");
    assert!(request_predicate.contains("op: Eq"), "{request_predicate}");
    assert!(request_predicate.contains("Int64(1)"), "{request_predicate}");
    assert!(
        planner.requests()[0]
            .projected_column_names
            .as_ref()
            .is_some_and(|columns| { columns.contains(&"id".to_string()) && columns.contains(&"value".to_string()) })
    );
    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "the source must not apply LIMIT before the inexact filter");
    assert!(source.read_columns.contains(&"id".to_string()));
    assert!(source.read_columns.contains(&"value".to_string()));
    assert!(source.reader_predicate.is_some());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "reader_pushdown");
}

#[tokio::test]
async fn iceberg_provider_keeps_filter_native_when_position_deletes_are_present() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(position_delete_projection_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].position_delete_files.len(), 1);
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "absent");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "the native filter must run after position deletes so row ordinals remain file-global"
    );
}

#[tokio::test]
async fn iceberg_provider_pushes_supported_filter_conjuncts_and_keeps_native_filter_for_residual() {
    let schema = iceberg_test_schema();
    let planner = Arc::new(RecordingPlanner::new(native_admissible_scan_plan(Arc::clone(&schema))));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where id = 1 and value not like 'bad%'").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    let reader_predicate = source
        .reader_predicate
        .as_ref()
        .expect("supported conjunct should still push into the table-format reader")
        .to_string();
    assert!(reader_predicate.contains("id = 1"), "{reader_predicate}");
    assert!(!reader_predicate.contains("LIKE"), "{reader_predicate}");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "the native filter must remain because NOT LIKE was only a residual expression"
    );
}

#[tokio::test]
async fn iceberg_provider_keeps_native_filter_for_temporal_table_format_predicate() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ship_date", DataType::Date32, false),
        Field::new("value", DataType::Utf8, true),
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
            "ship_date",
            Some("ship_date"),
            DataType::Date32,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let planner = Arc::new(RecordingPlanner::new(plan));
    let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&schema), planner.clone()));
    let ctx = SessionContext::new_with_config(direct_friendly_session_config());
    ctx.register_table("iceberg_t", provider).unwrap();

    let df = ctx.sql("select value from iceberg_t where ship_date = date '2024-01-01'").await.unwrap();
    let physical_plan = df.create_physical_plan().await.unwrap();
    let result = lower_physical_plan_to_native(physical_plan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.read_columns.contains(&"ship_date".to_string()));
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "absent");
    assert!(
        native_plan_filter_count(&result.plans[0]) >= 1,
        "temporal table-format predicates must remain represented as native filters"
    );
}

#[test]
fn native_lowering_rejects_iceberg_missing_predicate_column_above_hash_join() {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("profile_id", DataType::Int64, false),
    ]));
    let profile_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false),
        Field::new("age", DataType::Int64, true),
    ]));
    let profile_footer_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())])),
    ]));
    let edge_scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(
        native_admissible_scan_plan_for_schema(
            Arc::clone(&edge_schema),
            "edges-data-0001",
            "/tmp/iceberg/edges.parquet",
        ),
        Arc::clone(&edge_schema),
    ));
    let profile_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("glue")
        .with_logical_schema(Arc::clone(&profile_schema))
        .with_native_source_schema(Arc::clone(&profile_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "profiles-data-0001",
                "s3://bucket/profiles/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096)
            .with_row_count(11)
            .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
                "s3://bucket/profiles/data-0001.parquet",
                CredentialMode::SharedProcessEnv,
            ))
            .with_footer_schema(profile_footer_schema),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "profile_id",
            Some("profile_id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "age",
            Some("age"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let profile_scan: Arc<dyn ExecutionPlan> =
        Arc::new(CudfTableFormatScanExec::new(profile_plan, Arc::clone(&profile_schema)));
    let edge_profile_id = col("profile_id", &edge_schema).unwrap();
    let profile_profile_id = col("profile_id", &profile_schema).unwrap();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            edge_scan,
            profile_scan,
            vec![(edge_profile_id, profile_profile_id)],
            None,
            &JoinType::Inner,
            Some(vec![0, 1, edge_schema.fields().len() + profile_schema.index_of("age").unwrap()]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let predicate =
        binary(col("age", &join_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(0))), &join_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, join).unwrap());

    let result = lower_physical_plan_to_native(filter);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        result.report.outcomes().iter().any(|outcome| {
            matches!(
                outcome,
                CandidateOutcome::Rejected { reason: FallbackReason::IcebergFieldIdMappingUnsupported, .. }
            )
        }),
        "predicate-required missing Iceberg field above a join must reject instead of materializing NULLs: {:?}",
        result.report.outcomes()
    );
}

#[test]
fn native_lowering_accepts_iceberg_same_name_signed_widened_predicate_column_above_hash_join() {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("profile_id", DataType::Int64, false),
    ]));
    let profile_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false),
        Field::new("age", DataType::Int32, true),
    ]));
    let profile_footer_schema = Arc::new(Schema::new(vec![
        Field::new("profile_id", DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())])),
        Field::new("age", DataType::Int16, true),
    ]));
    let edge_scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(
        native_admissible_scan_plan_for_schema(
            Arc::clone(&edge_schema),
            "edges-data-0001",
            "/tmp/iceberg/edges.parquet",
        ),
        Arc::clone(&edge_schema),
    ));
    let profile_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("glue")
        .with_logical_schema(Arc::clone(&profile_schema))
        .with_native_source_schema(Arc::clone(&profile_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "profiles-data-0001",
                "s3://bucket/profiles/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096)
            .with_row_count(11)
            .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
                "s3://bucket/profiles/data-0001.parquet",
                CredentialMode::SharedProcessEnv,
            ))
            .with_footer_schema(profile_footer_schema),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "profile_id",
            Some("profile_id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "age",
            Some("age"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ));
    let profile_scan: Arc<dyn ExecutionPlan> =
        Arc::new(CudfTableFormatScanExec::new(profile_plan, Arc::clone(&profile_schema)));
    let edge_profile_id = col("profile_id", &edge_schema).unwrap();
    let profile_profile_id = col("profile_id", &profile_schema).unwrap();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            edge_scan,
            profile_scan,
            vec![(edge_profile_id, profile_profile_id)],
            None,
            &JoinType::Inner,
            Some(vec![0, 1, edge_schema.fields().len() + profile_schema.index_of("age").unwrap()]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let predicate =
        binary(col("age", &join_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &join_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, join).unwrap());

    let result = lower_physical_plan_to_native(filter);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    let sources = native_local_parquet_sources(&result.plans[0]);
    let profile_source = sources
        .into_iter()
        .find(|source| source.read_columns.contains(&"age".to_string()))
        .expect("profile source should read same-name widened age column");
    assert_eq!(profile_source.schema.field_with_name("age").unwrap().data_type(), &DataType::Int32);
    assert!(profile_source.reader_predicate.is_none());
    assert!(
        profile_source.files.iter().any(|file| {
            file.field_id_read_bindings.iter().any(|binding| {
                binding.logical_name == "age" && binding.physical_name == "age" && binding.data_type == DataType::Int16
            })
        }),
        "source should keep footer-proven physical age type for diagnostics/coercion"
    );
}

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
        Arc::new(CudfIcebergTableProvider::from_table_format_planner(Arc::clone(&reason_schema), reason_planner));
    let store_sales = Arc::new(CudfIcebergTableProvider::from_table_format_planner(
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
        Arc::new(CudfTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));
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
    let scan = Arc::new(CudfTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));

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
    let scan = Arc::new(CudfTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("expected S3 table-format scan to be rewritten through SourceFetch lowering");
    };
    assert_eq!(candidate.shape, "TableFormatScan(iceberg)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files.len(), 1);
    assert_eq!(source.files[0].location.read_capability(), "object_store:s3");
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:s3-data-0001");
    assert_eq!(source.files[0].path.to_string_lossy(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(source.files[0].resolved_source.read_strategy, ParquetReadStrategy::RemoteDirect);
    assert_eq!(source.files[0].resolved_source.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
    assert_eq!(source.diagnostics.source_decode_input, "filepath");
}

#[test]
fn table_format_parquet_resolver_reports_local_remote_and_cache_strategies() {
    let config = TableFormatParquetReadStrategyConfig::default();
    let local = resolve_table_format_parquet_source(
        "file:///tmp/table/data.parquet",
        &TableFormatReadCapability::local_file_system(),
        &config,
    )
    .unwrap();
    assert_eq!(local.reader_location, "/tmp/table/data.parquet");
    assert_eq!(local.read_strategy, ParquetReadStrategy::LocalKvikio);
    assert_eq!(local.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(local.credential_mode, CredentialMode::None);

    let remote = resolve_table_format_parquet_source(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &config,
    )
    .unwrap();
    assert_eq!(remote.read_strategy, ParquetReadStrategy::RemoteDirect);
    assert_eq!(remote.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(remote.credential_mode, CredentialMode::SharedProcessEnv);

    let cache_dir = tempfile::tempdir().unwrap();
    let cache_miss_config = TableFormatParquetReadStrategyConfig {
        preference: TableFormatParquetReadStrategyPreference::LocalCache,
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let cache_miss = resolve_table_format_parquet_source(
        "s3://bucket/table/missing.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_miss_config,
    )
    .unwrap();
    assert_eq!(cache_miss.read_strategy, ParquetReadStrategy::RemoteDirect);
    assert_eq!(cache_miss.cache_status, LakehouseCacheStatus::Miss);
    assert_eq!(cache_miss.fallback_reason.as_deref(), Some("lakehouse_cache_miss_fill_disabled"));

    let cache_config = TableFormatParquetReadStrategyConfig {
        preference: TableFormatParquetReadStrategyPreference::LocalCache,
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let cache_metadata = TableFormatParquetSourceMetadata::default().with_size_bytes(4096);
    let cache_path_probe = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &TableFormatParquetReadStrategyConfig {
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
            ..cache_config.clone()
        },
        &cache_metadata,
    )
    .unwrap();
    let cache_file = PathBuf::from(cache_path_probe.reader_location);
    std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
    std::fs::write(&cache_file, vec![0u8; 4096]).unwrap();
    let cached = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &cache_metadata,
    )
    .unwrap();
    assert_eq!(cached.read_strategy, ParquetReadStrategy::LocalCacheKvikio);
    assert_eq!(cached.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(cached.cache_status, LakehouseCacheStatus::Hit);
    assert_eq!(cached.reader_location, cache_file.display().to_string());
}

#[test]
fn table_format_parquet_resolver_selects_fill_before_read_for_local_cache_miss_policy() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        preference: TableFormatParquetReadStrategyPreference::LocalCache,
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        lakehouse_cache_max_bytes: Some(16 * 1024 * 1024),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };

    let resolved = resolve_table_format_parquet_source(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
    )
    .unwrap();

    assert_eq!(resolved.read_strategy, ParquetReadStrategy::LocalCacheKvikio);
    assert_eq!(resolved.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(resolved.cache_status, LakehouseCacheStatus::FillBeforeRead);
    assert_eq!(resolved.original_location, "s3://bucket/table/data.parquet");
    assert_eq!(resolved.lakehouse_cache_root.as_deref(), Some(cache_dir.path().to_str().unwrap()));
    assert_eq!(resolved.lakehouse_cache_max_bytes, Some(16 * 1024 * 1024));
    assert!(resolved.reader_location.contains("/v1/s3/bucket/weak_content_unknown/"));
    assert!(resolved.reader_location.contains("data.parquet__h"));
}

#[test]
fn table_format_parquet_resolver_scopes_local_cache_by_storage_domain_and_content_identity() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        preference: TableFormatParquetReadStrategyPreference::LocalCache,
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(12))
        .with_storage_domain("rustfs:bucket-a")
        .with_content_identity("etag:abc123");
    let resolved = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();
    let cache_path = PathBuf::from(&resolved.reader_location);
    assert!(resolved.reader_location.contains("/v1/s3/rustfs_bucket-a/etag_abc123/"));

    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    std::fs::write(&cache_path, vec![0u8; 4096]).unwrap();
    let cached = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();
    assert_eq!(cached.cache_status, LakehouseCacheStatus::Hit);
    assert_eq!(cached.reader_location, resolved.reader_location);
    assert_eq!(cached.storage_domain.as_deref(), Some("rustfs:bucket-a"));
    assert_eq!(cached.content_identity.as_deref(), Some("etag:abc123"));

    let changed_metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(8192), Some(12))
        .with_storage_domain("rustfs:bucket-a")
        .with_content_identity("etag:def456");
    let changed = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &changed_metadata,
    )
    .unwrap();
    assert_eq!(changed.cache_status, LakehouseCacheStatus::FillBeforeRead);
    assert_ne!(changed.reader_location, cached.reader_location);
}

#[test]
fn table_format_parquet_resolver_scopes_local_cache_by_catalog_endpoint_without_endpoint_secrets() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        preference: TableFormatParquetReadStrategyPreference::LocalCache,
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        storage_endpoint: Some("https://user:secret@rustfs.local:9000/minio?X-Amz-Signature=secret#frag".to_string()),
        storage_profile_id: Some("rest-s3".to_string()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(12));
    let resolved = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();

    assert_eq!(resolved.storage_domain.as_deref(), Some("bucket-a@https://rustfs.local:9000/minio"));
    assert_eq!(resolved.storage_profile_id.as_deref(), Some("rest-s3"));
    assert!(resolved.reader_location.contains("rustfs.local_9000"));
    assert!(!resolved.reader_location.contains("user"));
    assert!(!resolved.reader_location.contains("secret"));
    assert!(!resolved.reader_location.contains("X-Amz-Signature"));
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
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/data-0001.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    ];
    let scan = Arc::new(CudfTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    let file = &source.files[0];
    assert_eq!(file.resolved_source.read_strategy, ParquetReadStrategy::RemoteDirect);
    assert_eq!(file.resolved_source.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(file.path.to_string_lossy(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
    assert_eq!(source.diagnostics.source_decode_input, "filepath");
}

#[test]
fn table_format_scan_preserves_unsupported_feature_reason_and_never_executes_cpu_arrow_reader() {
    let schema = iceberg_test_schema();
    let plan =
        native_admissible_scan_plan(Arc::clone(&schema)).with_unsupported_feature(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            "delete files require Milestone 2 row-level semantics",
        ));
    let scan = Arc::new(CudfTableFormatScanExec::new(plan, schema));

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
    assert!(rendered.contains("cpu_arrow_reader_fallback"));
    assert!(rendered.contains("false"));
}

fn iceberg_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("value", DataType::Utf8, true)]))
}

fn map_utf8_int32_type() -> DataType {
    DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(Field::new("value", DataType::Int32, true)),
            ])),
            false,
        )),
        false,
    )
}

fn direct_friendly_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_physical_uncorrelated_scalar_subquery = false;
    config
}

fn native_admissible_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
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
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

fn position_delete_projection_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0001",
        "data-0001",
        "/tmp/iceberg/delete-0001.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::local_file_system())
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio("/tmp/iceberg/delete-0001.parquet"));
    let mut plan = native_admissible_scan_plan(schema);
    plan.data_files[0] = plan.data_files[0].clone().with_position_delete_files(vec![delete_file.clone()]);
    plan.delete_files.push(delete_file);
    plan
}

fn native_admissible_scan_plan_for_schema(
    schema: Arc<Schema>,
    file_id: impl Into<String>,
    file_path: impl Into<String>,
) -> TableFormatScanPlan {
    schema.fields().iter().enumerate().fold(
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_snapshot_id("12345")
            .with_catalog_type("local")
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(Arc::clone(&schema))
            .with_data_file(
                TableFormatDataFile::new(
                    file_id,
                    file_path,
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::local_file_system(),
                )
                .with_size_bytes(4096)
                .with_row_count(11),
            ),
        |plan, (_index, field)| {
            plan.with_projected_column(TableFormatColumnMapping::new(
                None,
                field.name(),
                Some(field.name()),
                field.data_type().clone(),
                field.is_nullable(),
                TableFormatColumnSource::DirectTopLevelFileColumn,
                TableFormatColumnRole::Projected,
            ))
        },
    )
}

fn native_admissible_empty_scan_plan() -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg).with_snapshot_id("12345").with_catalog_type("local").with_data_file(
        TableFormatDataFile::new(
            "data-0001",
            "/tmp/iceberg/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::local_file_system(),
        )
        .with_size_bytes(4096)
        .with_row_count(11),
    )
}

fn mixed_row_count_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
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
        .with_data_file(
            TableFormatDataFile::new(
                "data-0002",
                "/tmp/iceberg/data-0002.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096),
        )
        .with_pure_filter_column(count_star_probe_carrier_mapping())
}

fn position_delete_row_count_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0002",
        "data-0002",
        "/tmp/iceberg/delete-0002.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::local_file_system())
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio("/tmp/iceberg/delete-0002.parquet"));
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
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
        .with_data_file(
            TableFormatDataFile::new(
                "data-0002",
                "/tmp/iceberg/data-0002.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(7)
            .with_position_delete_files(vec![delete_file.clone()]),
        )
        .with_delete_file(delete_file)
        .with_pure_filter_column(count_star_probe_carrier_mapping())
}

fn remote_mixed_row_count_scan_plan_without_probe_mapping(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("glue")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "s3://bucket/table/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "data-0002",
                "s3://bucket/table/data-0002.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(4096),
        )
}

fn mixed_row_count_scan_plan_for_files(
    schema: Arc<Schema>,
    first_path: &Path,
    second_path: &Path,
) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                first_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(fs::metadata(first_path).unwrap().len())
            .with_row_count(2),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "data-0002",
                second_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(fs::metadata(second_path).unwrap().len()),
        )
        .with_pure_filter_column(count_star_probe_carrier_mapping())
}

fn count_star_probe_carrier_mapping() -> TableFormatColumnMapping {
    TableFormatColumnMapping::new(
        None,
        "id",
        Some("id"),
        DataType::Int64,
        false,
        TableFormatColumnSource::DirectTopLevelFileColumn,
        TableFormatColumnRole::PureFilter,
    )
}

fn write_iceberg_test_parquet(path: &Path, ids: &[i64], values: &[Option<&str>], schema: Arc<Schema>) {
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(ids.to_vec())), Arc::new(StringArray::from(values.to_vec()))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn transformed_partition_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("customerId", DataType::Utf8, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn identity_and_day_partition_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema))
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/events/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11)
            .with_partition_values(vec![
                TableFormatPartitionValue::new(
                    "customerId",
                    DataType::Utf8,
                    false,
                    TableFormatPartitionTransform::Identity,
                    TableFormatScalarValue::Utf8("customer-1".to_string()),
                ),
                TableFormatPartitionValue::new(
                    "event_ts_day",
                    DataType::Int32,
                    true,
                    TableFormatPartitionTransform::Day,
                    TableFormatScalarValue::Int32(19_000),
                ),
            ]),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "customerId",
            Some("customerId"),
            DataType::Utf8,
            false,
            TableFormatColumnSource::IdentityPartitionValue,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "event_ts",
            Some("event_ts"),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "value",
            Some("value"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

fn timestamp_decimal_partition_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("amount", DataType::Decimal128(10, 2), false),
    ]))
}

fn timestamp_decimal_identity_partition_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema))
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/events/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(7)
            .with_partition_values(vec![
                TableFormatPartitionValue::new(
                    "event_ts",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                    TableFormatPartitionTransform::Identity,
                    TableFormatScalarValue::Int64(1_700_000_000_123_456),
                ),
                TableFormatPartitionValue::new(
                    "amount",
                    DataType::Decimal128(10, 2),
                    false,
                    TableFormatPartitionTransform::Identity,
                    TableFormatScalarValue::Int128(12_345),
                ),
            ]),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "event_ts",
            Some("event_ts"),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
            TableFormatColumnSource::IdentityPartitionValue,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "amount",
            Some("amount"),
            DataType::Decimal128(10, 2),
            false,
            TableFormatColumnSource::IdentityPartitionValue,
            TableFormatColumnRole::Projected,
        ))
}

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
}

fn adapter_error_from(error: &DataFusionError) -> &datafusion_nexus::error::Error {
    match error {
        DataFusionError::External(source) => {
            source.downcast_ref::<datafusion_nexus::error::Error>().expect("expected datafusion_nexus::error::Error")
        },
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => adapter_error_from(source),
        other => panic!("expected adapter error, got {other:?}"),
    }
}

fn native_local_parquet_source(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> &nexus_query_engine::source::LocalParquetSource {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .expect("native plan should contain a local parquet source")
}

fn native_local_parquet_sources(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> Vec<&nexus_query_engine::source::LocalParquetSource> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

fn native_plan_limit_count(plan: &nexus_query_engine::plan::QueryPlan) -> usize {
    plan.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Limit(_))).count()
}

fn native_plan_filter_count(plan: &nexus_query_engine::plan::QueryPlan) -> usize {
    plan.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Filter(_))).count()
}

fn native_plan_union_count(plan: &nexus_query_engine::plan::QueryPlan) -> usize {
    plan.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Union(_))).count()
}
