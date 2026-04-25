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
