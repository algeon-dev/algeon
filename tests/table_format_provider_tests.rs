use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{
    CredentialMode, LakehouseCacheStatus, ParquetReadStrategy, PlanNode, ResolvedParquetSource, SourceDecodeInput,
    SourceNode,
};
use datafusion::{
    common::ScalarValue,
    datasource::TableProvider,
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{
        ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, filter::FilterExec, limit::GlobalLimitExec,
    },
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CandidateOutcome, CudfTableFormatScanExec, FallbackReason, TableFormat, TableFormatColumnMapping,
    TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat,
    TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig, TableFormatParquetReadStrategyPreference,
    TableFormatParquetSourceMetadata, TableFormatPlanner, TableFormatReadCapability, TableFormatScanPlan,
    TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
    iceberg::CudfIcebergTableProvider, lower_physical_plan_to_native, resolve_table_format_parquet_source,
    resolve_table_format_parquet_source_with_metadata,
};
use futures::future::BoxFuture;

#[derive(Debug)]
struct RecordingPlanner {
    plan: TableFormatScanPlan,
    requests: Mutex<Vec<TableFormatScanRequest>>,
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
    fn plan_scan<'a>(
        &'a self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'a, datafusion_cudf::Result<TableFormatScanPlan>> {
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
    assert_eq!(source.files[0].stable_id, "table-format:iceberg:data-0001");
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
    assert_eq!(source.files[0].row_group_rows.as_deref(), Some(&[11][..]));
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
        panic!("expected table-format scan to be rewritten by Phase D lowering");
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
    assert_eq!(source.files[0].resolved_source.read_strategy, ParquetReadStrategy::RemoteHostBuffer);
    assert_eq!(source.files[0].resolved_source.decode_input, SourceDecodeInput::HostBuffer);
    assert_eq!(source.diagnostics.source_read_strategy, "remote_host_buffer");
    assert_eq!(source.diagnostics.source_decode_input, "host_buffer");
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
    assert_eq!(remote.read_strategy, ParquetReadStrategy::RemoteHostBuffer);
    assert_eq!(remote.decode_input, SourceDecodeInput::HostBuffer);
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
    assert_eq!(cache_miss.read_strategy, ParquetReadStrategy::RemoteHostBuffer);
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
    std::fs::write(&cache_file, b"cached parquet placeholder").unwrap();
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
    assert!(resolved.reader_location.contains("/v1/s3/bucket/content_unknown/"));
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
    std::fs::write(&cache_path, b"cached parquet placeholder").unwrap();
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
fn native_lowering_preserves_remote_kvikio_reader_contract_for_dispatch() {
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
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio_experimental(
            "s3://bucket/table/data-0001.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    ];
    let scan = Arc::new(CudfTableFormatScanExec::new(plan, schema));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.rewritten_count(), 1);
    let source = native_local_parquet_source(&result.plans[0]);
    let file = &source.files[0];
    assert_eq!(file.resolved_source.read_strategy, ParquetReadStrategy::RemoteKvikioExperimental);
    assert_eq!(file.resolved_source.decode_input, SourceDecodeInput::Filepath);
    assert_eq!(file.path.to_string_lossy(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(source.diagnostics.source_read_strategy, "remote_kvikio_experimental");
    assert_eq!(source.diagnostics.source_decode_input, "filepath");
}

#[test]
fn table_format_scan_preserves_unsupported_feature_reason_and_never_executes_cpu_arrow_reader() {
    let schema = iceberg_test_schema();
    let plan =
        native_admissible_scan_plan(Arc::clone(&schema)).with_unsupported_feature(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            "delete files require Phase 2 row-level semantics",
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

fn direct_friendly_session_config() -> SessionConfig {
    SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1)
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
            Some(1),
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
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

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.as_any().downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
}

fn native_local_parquet_source(plan: &cudf_query_engine::QueryPlan) -> &cudf_query_engine::LocalParquetSource {
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

fn native_plan_limit_count(plan: &cudf_query_engine::QueryPlan) -> usize {
    plan.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Limit(_))).count()
}
