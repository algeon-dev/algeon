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
struct SleepingResolutionPlanner {
    delay: Duration,
}

#[derive(Debug)]
struct BusyPlanner {
    plan: TableFormatScanPlan,
    spin_for: Duration,
}

#[derive(Debug)]
struct FailingResolutionPlanner {
    code: ErrorCode,
}

#[derive(Debug)]
struct FailingNativeInspectionPlanner {
    plan: TableFormatScanPlan,
    code: ErrorCode,
}

#[derive(Debug)]
struct FailingInspectionAfterNotSupportedPlanner {
    plan: TableFormatScanPlan,
    inspection_count: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct TrimmingInspectionPlanner {
    resolved_scan: TableFormatScanPlan,
    inspected_scan: TableFormatScanPlan,
    inspection_count: Arc<AtomicUsize>,
}

struct RealIcebergProviderFixture {
    _temp_dir: tempfile::TempDir,
    table_location: PathBuf,
    table: Table,
}

impl RealIcebergProviderFixture {
    fn new() -> Self {
        Self::new_with_metadata_transform(|_| {})
    }

    fn new_nested() -> Self {
        Self::new_with_metadata_transform(|metadata| {
            metadata["last-column-id"] = serde_json::json!(9);
            let current_schema = metadata["schemas"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|schema| schema["schema-id"] == serde_json::json!(1))
                .unwrap();
            current_schema["identifier-field-ids"] = serde_json::json!([1]);
            current_schema["fields"] = serde_json::json!([
                {"id": 1, "name": "x", "required": true, "type": "long"},
                {
                    "id": 2,
                    "name": "names",
                    "required": true,
                    "type": {
                        "type": "list",
                        "element-id": 3,
                        "element-required": true,
                        "element": "string"
                    }
                },
                {
                    "id": 4,
                    "name": "profile",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {"id": 5, "name": "city", "required": true, "type": "string"},
                            {"id": 6, "name": "score", "required": true, "type": "long"}
                        ]
                    }
                },
                {
                    "id": 7,
                    "name": "attributes",
                    "required": true,
                    "type": {
                        "type": "map",
                        "key-id": 8,
                        "key": "string",
                        "value-id": 9,
                        "value-required": true,
                        "value": "long"
                    }
                }
            ]);
            metadata["default-sort-order-id"] = serde_json::json!(0);
            metadata["sort-orders"] = serde_json::json!([{"order-id": 0, "fields": []}]);
        })
    }

    #[cfg(feature = "cuvs")]
    fn new_float32_vector_list() -> Self {
        Self::new_with_metadata_transform(|metadata| {
            metadata["last-column-id"] = serde_json::json!(3);
            let current_schema = metadata["schemas"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|schema| schema["schema-id"] == serde_json::json!(1))
                .unwrap();
            current_schema["identifier-field-ids"] = serde_json::json!([1]);
            current_schema["fields"] = serde_json::json!([
                {"id": 1, "name": "x", "required": true, "type": "long"},
                {
                    "id": 2,
                    "name": "embedding",
                    "required": true,
                    "type": {
                        "type": "list",
                        "element-id": 3,
                        "element-required": true,
                        "element": "float"
                    }
                }
            ]);
            metadata["default-sort-order-id"] = serde_json::json!(0);
            metadata["sort-orders"] = serde_json::json!([{"order-id": 0, "fields": []}]);
        })
    }

    fn new_with_metadata_transform(transform: impl FnOnce(&mut serde_json::Value)) -> Self {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let table_location = temp_dir.path().join("table1");
        let metadata_dir = table_location.join("metadata");
        fs::create_dir_all(&metadata_dir).unwrap();

        let manifest_list1_location = metadata_dir.join("manifests_list_1.avro");
        let manifest_list2_location = metadata_dir.join("manifests_list_2.avro");
        let table_metadata1_location = metadata_dir.join("v1.json");
        let template_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../iceberg-rust/crates/iceberg/testdata/example_table_metadata_v2.json");
        let metadata_json = fs::read_to_string(template_path)
            .unwrap()
            .replace("{{ table_location }}", table_location.to_str().unwrap())
            .replace("{{ manifest_list_1_location }}", manifest_list1_location.to_str().unwrap())
            .replace("{{ manifest_list_2_location }}", manifest_list2_location.to_str().unwrap())
            .replace("{{ table_metadata_1_location }}", table_metadata1_location.to_str().unwrap());
        let mut metadata_json = serde_json::from_str::<serde_json::Value>(&metadata_json).unwrap();
        transform(&mut metadata_json);
        let table_metadata = serde_json::from_value::<TableMetadata>(metadata_json).unwrap();
        let table = Table::builder()
            .metadata(table_metadata)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location(table_metadata1_location.to_str().unwrap())
            .runtime(iceberg::test_utils::test_runtime())
            .build()
            .unwrap();

        Self { _temp_dir: temp_dir, table_location, table }
    }

    async fn write_one_row_and_current_manifest_list(&self) {
        self.write_vector_rows_and_current_manifest_list(&[(100, 200, 300)]).await;
    }

    async fn write_clusterable_vector_rows_and_current_manifest_list(&self) {
        self.write_vector_rows_and_current_manifest_list(&[(100, 0, 0), (100, 1, 0), (100, 10, 10), (100, 11, 10)])
            .await;
    }

    #[cfg(feature = "cuvs")]
    async fn write_float32_vector_list_rows_and_current_manifest_list(&self, rows: &[(i64, &[f32])]) {
        assert!(!rows.is_empty(), "vector-list fixture must contain at least one row");
        assert!(
            rows.iter().all(|(x, embedding)| *x == 100 && embedding.len() == 2),
            "fixture must retain the table's identity partition value and a fixed vector dimension"
        );
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let arrow_schema = Arc::new(iceberg::arrow::schema_to_arrow_schema(current_schema.as_ref()).unwrap());
        let DataType::List(element) = arrow_schema.field(1).data_type() else {
            panic!("embedding must be an Iceberg list");
        };
        assert_eq!(element.data_type(), &DataType::Float32, "embedding values must be Float32");
        let offsets =
            (0..=rows.len()).map(|row| i32::try_from(row * 2).expect("small test List offset")).collect::<Vec<_>>();
        let values = rows.iter().flat_map(|(_, embedding)| embedding.iter().copied()).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(rows.iter().map(|(x, _)| *x).collect::<Vec<_>>())),
                Arc::new(ListArray::new(
                    Arc::clone(element),
                    OffsetBuffer::new(offsets.into()),
                    Arc::new(Float32Array::from(values)),
                    None,
                )),
            ],
        )
        .unwrap();
        let data_path = self.table_location.join("data-0001.parquet");
        let file = File::create(&data_path).unwrap();
        let mut parquet_writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
        parquet_writer.write(&batch).unwrap();
        parquet_writer.close().unwrap();

        let current_partition_spec = self.table.metadata().default_partition_spec();
        let manifest_path = self.table_location.join("metadata/data_manifest.avro");
        let manifest_output = self.table.file_io().new_output(manifest_path.to_str().unwrap()).unwrap();
        let mut writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(current_snapshot.snapshot_id()),
            current_schema,
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(data_path.to_str().unwrap().to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(fs::metadata(&data_path).unwrap().len())
            .record_count(u64::try_from(rows.len()).expect("small fixture row count"))
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .key_metadata(None)
            .build()
            .unwrap();
        writer.add_file(data_file, current_snapshot.sequence_number()).unwrap();
        let data_file_manifest = writer.write_manifest_file().await.unwrap();

        let manifest_list_writer =
            self.table.file_io().new_output(current_snapshot.manifest_list()).unwrap().writer().await.unwrap();
        let mut manifest_list_write = ManifestListWriter::v2(
            manifest_list_writer,
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write.add_manifests(vec![data_file_manifest].into_iter()).unwrap();
        manifest_list_write.close().await.unwrap();
    }

    async fn write_vector_rows_and_current_manifest_list(&self, rows: &[(i64, i64, i64)]) {
        assert!(!rows.is_empty(), "vector fixture must contain at least one row");
        assert!(rows.iter().all(|(x, _, _)| *x == 100), "fixture must retain the table's identity partition value");
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let arrow_schema = Arc::new(iceberg::arrow::schema_to_arrow_schema(current_schema.as_ref()).unwrap());
        let data_path = self.table_location.join("data-0001.parquet");
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(rows.iter().map(|(x, _, _)| *x).collect::<Vec<_>>())),
                Arc::new(Int64Array::from(rows.iter().map(|(_, y, _)| *y).collect::<Vec<_>>())),
                Arc::new(Int64Array::from(rows.iter().map(|(_, _, z)| *z).collect::<Vec<_>>())),
                Arc::new(StringArray::from(vec!["row"; rows.len()])),
                Arc::new(Float64Array::from(vec![1.5; rows.len()])),
                Arc::new(Int32Array::from(vec![32; rows.len()])),
                Arc::new(Int64Array::from(vec![64; rows.len()])),
                Arc::new(BooleanArray::from(vec![true; rows.len()])),
            ],
        )
        .unwrap();
        let file = File::create(&data_path).unwrap();
        let mut parquet_writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
        parquet_writer.write(&batch).unwrap();
        parquet_writer.close().unwrap();

        let current_partition_spec = self.table.metadata().default_partition_spec();
        let manifest_path = self.table_location.join("metadata/data_manifest.avro");
        let manifest_output = self.table.file_io().new_output(manifest_path.to_str().unwrap()).unwrap();
        let mut writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(current_snapshot.snapshot_id()),
            current_schema,
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(data_path.to_str().unwrap().to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(fs::metadata(&data_path).unwrap().len())
            .record_count(u64::try_from(rows.len()).expect("small fixture row count"))
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .key_metadata(None)
            .build()
            .unwrap();
        writer.add_file(data_file, current_snapshot.sequence_number()).unwrap();
        let data_file_manifest = writer.write_manifest_file().await.unwrap();

        let manifest_list_writer =
            self.table.file_io().new_output(current_snapshot.manifest_list()).unwrap().writer().await.unwrap();
        let mut manifest_list_write = ManifestListWriter::v2(
            manifest_list_writer,
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write.add_manifests(vec![data_file_manifest].into_iter()).unwrap();
        manifest_list_write.close().await.unwrap();
    }

    async fn write_one_nested_row_and_current_manifest_list(&self) {
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let arrow_schema = Arc::new(iceberg::arrow::schema_to_arrow_schema(current_schema.as_ref()).unwrap());

        let DataType::List(names_element) = arrow_schema.field(1).data_type() else {
            panic!("names must be an Iceberg list");
        };
        let names: ArrayRef = Arc::new(ListArray::new(
            Arc::clone(names_element),
            OffsetBuffer::new(vec![0_i32, 2].into()),
            Arc::new(StringArray::from(vec!["alice", "bob"])),
            None,
        ));

        let DataType::Struct(profile_fields) = arrow_schema.field(2).data_type() else {
            panic!("profile must be an Iceberg struct");
        };
        let profile: ArrayRef = Arc::new(StructArray::new(
            profile_fields.clone(),
            vec![Arc::new(StringArray::from(vec!["Taipei"])), Arc::new(Int64Array::from(vec![7]))],
            None,
        ));

        let DataType::Map(attribute_entries, ordered) = arrow_schema.field(3).data_type() else {
            panic!("attributes must be an Iceberg map");
        };
        let DataType::Struct(attribute_fields) = attribute_entries.data_type() else {
            panic!("map entries must be a struct");
        };
        let entries = StructArray::new(
            attribute_fields.clone(),
            vec![Arc::new(StringArray::from(vec!["level"])), Arc::new(Int64Array::from(vec![3]))],
            None,
        );
        let attributes: ArrayRef = Arc::new(MapArray::new(
            Arc::clone(attribute_entries),
            OffsetBuffer::new(vec![0_i32, 1].into()),
            entries,
            None,
            *ordered,
        ));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![100])), names, profile, attributes],
        )
        .unwrap();

        let data_path = self.table_location.join("data-0001.parquet");
        let file = File::create(&data_path).unwrap();
        let mut parquet_writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
        parquet_writer.write(&batch).unwrap();
        parquet_writer.close().unwrap();

        let current_partition_spec = self.table.metadata().default_partition_spec();
        let manifest_path = self.table_location.join("metadata/data_manifest.avro");
        let manifest_output = self.table.file_io().new_output(manifest_path.to_str().unwrap()).unwrap();
        let mut writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(current_snapshot.snapshot_id()),
            current_schema,
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(data_path.to_str().unwrap().to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(fs::metadata(&data_path).unwrap().len())
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .key_metadata(None)
            .build()
            .unwrap();
        writer.add_file(data_file, current_snapshot.sequence_number()).unwrap();
        let data_file_manifest = writer.write_manifest_file().await.unwrap();

        let manifest_list_writer =
            self.table.file_io().new_output(current_snapshot.manifest_list()).unwrap().writer().await.unwrap();
        let mut manifest_list_write = ManifestListWriter::v2(
            manifest_list_writer,
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write.add_manifests(vec![data_file_manifest].into_iter()).unwrap();
        manifest_list_write.close().await.unwrap();
    }
}

impl SleepingPlanner {
    fn new(plan: TableFormatScanPlan, delay: Duration) -> Self {
        Self { plan, delay }
    }
}

impl SleepingResolutionPlanner {
    fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

impl BusyPlanner {
    fn new(plan: TableFormatScanPlan, spin_for: Duration) -> Self {
        Self { plan, spin_for }
    }
}

impl TableFormatPlanner for FailingResolutionPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let code = self.code;
        Box::pin(async move {
            Err(Error::new(code, "datafusion_nexus.test.iceberg.resolve_scan", "test Iceberg source resolution failed"))
        })
    }
}

impl TableFormatPlanner for FailingNativeInspectionPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }

    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        _resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let code = self.code;
        Box::pin(async move {
            Err(Error::new(
                code,
                "datafusion_nexus.test.iceberg.inspect_native_scan",
                "test optional native Iceberg probe failed",
            ))
        })
    }
}

impl TableFormatPlanner for FailingInspectionAfterNotSupportedPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }

    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        _resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        self.inspection_count.fetch_add(1, AtomicOrdering::Relaxed);
        Box::pin(async move {
            Err(Error::new(
                ErrorCode::SourceTemporaryFailure,
                "datafusion_nexus.test.iceberg.inspect_native_scan",
                "inspection must not replace a resolved capability rejection",
            ))
        })
    }
}

impl TableFormatPlanner for TrimmingInspectionPlanner {
    fn resolve_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let scan_plan = self.resolved_scan.clone().with_scan_limit(request.scan_limit);
        Box::pin(async move { Ok(scan_plan) })
    }

    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        _resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        self.inspection_count.fetch_add(1, AtomicOrdering::Relaxed);
        let scan_plan = self.inspected_scan.clone();
        Box::pin(async move { Ok(scan_plan) })
    }
}

impl TableFormatPlanner for SleepingResolutionPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(TableFormatScanPlan::new(TableFormat::Iceberg))
        })
    }

    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        vec![("test_planner_fact", "slow_resolution".to_string())]
    }
}

impl TableFormatPlanner for SleepingPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }

    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(resolved_scan)
        })
    }

    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        vec![("test_planner_fact", "slow".to_string())]
    }
}

impl TableFormatPlanner for BusyPlanner {
    fn resolve_scan(
        &self,
        _request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let plan = self.plan.clone();
        Box::pin(async move { Ok(plan) })
    }

    fn inspect_native_scan(
        &self,
        _request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let spin_for = self.spin_for;
        Box::pin(async move {
            let deadline = Instant::now() + spin_for;
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
            Ok(resolved_scan)
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
    fn resolve_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
        let scan_limit = request.scan_limit;
        self.requests.lock().expect("request recorder mutex poisoned").push(request);
        let plan = self.plan.clone().with_scan_limit(scan_limit);
        Box::pin(async move { Ok(plan) })
    }
}

fn test_iceberg_provider<P>(schema: Arc<Schema>, planner: Arc<P>) -> NexusIcebergTableProvider
where P: TableFormatPlanner + Send + Sync + 'static {
    let cpu_provider: Arc<dyn TableProvider> = Arc::new(UnknownStatisticsCpuProvider::new(schema));
    NexusIcebergTableProvider::from_cpu_provider_and_native_planner(cpu_provider, None, planner)
}

#[derive(Debug)]
struct UnknownStatisticsCpuProvider {
    schema: Arc<Schema>,
}

impl UnknownStatisticsCpuProvider {
    fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }
}

#[async_trait::async_trait]
impl TableProvider for UnknownStatisticsCpuProvider {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let schema = project_schema(&self.schema, projection)?;
        Ok(Arc::new(UnknownStatisticsCpuExec::new(schema)))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

#[derive(Debug)]
struct UnknownStatisticsCpuExec {
    inner: EmptyExec,
}

impl UnknownStatisticsCpuExec {
    fn new(schema: Arc<Schema>) -> Self {
        Self { inner: EmptyExec::new(schema) }
    }
}

impl DisplayAs for UnknownStatisticsCpuExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }
}

impl ExecutionPlan for UnknownStatisticsCpuExec {
    fn name(&self) -> &str {
        "UnknownStatisticsCpuExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal("test CPU exec does not accept children".to_string()))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        self.inner.execute(partition, context)
    }
}

fn test_iceberg_scan_exec(scan_plan: TableFormatScanPlan, schema: Arc<Schema>) -> NexusIcebergScanExec {
    let cpu_delegate: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&schema)));
    NexusIcebergScanExec::try_new(cpu_delegate, schema, NexusIcebergNativeScanOutcome::from_scan_plan(scan_plan))
        .unwrap()
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

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&NexusIcebergScanExec> {
    if let Some(scan) = plan.downcast_ref::<NexusIcebergScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
}

fn plan_contains_name(plan: &dyn ExecutionPlan, name: &str) -> bool {
    plan.name() == name || plan.children().into_iter().any(|child| plan_contains_name(child.as_ref(), name))
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
