use std::sync::{Arc, Mutex};

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{ParquetSourceLocation, PlanNode, SourceNode};
use datafusion::{
    execution::context::TaskContext,
    physical_plan::ExecutionPlan,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CandidateOutcome, CudfTableFormatScanExec, FallbackReason, TableFormat, TableFormatColumnMapping,
    TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat, TableFormatPlanner,
    TableFormatReadCapability, TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature,
    TableFormatUnsupportedFeatureCode, iceberg::CudfIcebergTableProvider, lower_physical_plan_to_native,
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
        self.requests.lock().expect("request recorder mutex poisoned").push(request);
        let plan = self.plan.clone();
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
fn native_lowering_lowers_s3_table_format_scan_to_object_store_location() {
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
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files.len(), 1);
    assert_eq!(
        source.files[0].path.to_string_lossy(),
        "https://bucket.s3.us-east-1.amazonaws.com/table/data-0001.parquet"
    );
    assert_eq!(source.files[0].location.read_capability(), "object_store:s3");
    assert!(matches!(source.files[0].location, ParquetSourceLocation::ObjectStoreUri { .. }));
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
            .with_size_bytes(4096),
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
        .with_size_bytes(4096),
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
            },
            _ => None,
        })
        .expect("native plan should contain a local parquet source")
}
