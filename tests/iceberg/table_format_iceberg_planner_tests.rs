use std::{fs, path::PathBuf, sync::Arc};

use arrow::datatypes::DataType;
use datafusion_nexus::{
    capability::CapabilityReason,
    table_format::{
        TableFormatColumnSource, TableFormatDataFileFormat, TableFormatDeleteFileContent,
        TableFormatDirectDeviceReadDisposition, TableFormatPartitionTransform, TableFormatReadCapability,
        TableFormatScalarValue, TableFormatScanRequest,
        iceberg::{
            IcebergTableFormatPlanner, ResolvedIcebergTable, SourceAccessPolicy,
            iceberg_file_scan_tasks_to_table_format_scan_plan,
        },
    },
};
use iceberg::{
    TableIdent,
    io::FileIO,
    scan::{FileScanTask, FileScanTaskDeleteFile},
    spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestListWriter, ManifestWriterBuilder,
        MappedField, NameMapping, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, TableMetadata, Transform,
        Type,
    },
    table::Table,
};

fn iceberg_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_fields([
                Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))),
                Arc::new(NestedField::optional(2, "shipmode", Type::Primitive(PrimitiveType::String))),
            ])
            .build()
            .unwrap(),
    )
}

fn file_scan_task(
    path: impl Into<String>,
    format: DataFileFormat,
    deletes: Vec<FileScanTaskDeleteFile>,
) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: path.into(),
        data_file_format: format,
        schema: iceberg_schema(),
        project_field_ids: vec![1, 2],
        predicate: None,
        deletes,
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    }
}

fn partitioned_iceberg_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_fields([
                Arc::new(NestedField::required(1, "customerId", Type::Primitive(PrimitiveType::String))),
                Arc::new(NestedField::optional(2, "event_ts", Type::Primitive(PrimitiveType::Timestamp))),
                Arc::new(NestedField::optional(3, "value", Type::Primitive(PrimitiveType::Long))),
            ])
            .build()
            .unwrap(),
    )
}

fn identity_and_day_partition_task(path: impl Into<String>) -> FileScanTask {
    let schema = partitioned_iceberg_schema();
    let partition_spec = PartitionSpec::builder(Arc::clone(&schema))
        .add_partition_field("customerId", "customerId", Transform::Identity)
        .unwrap()
        .add_partition_field("event_ts", "event_ts_day", Transform::Day)
        .unwrap()
        .build()
        .unwrap();
    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: path.into(),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: vec![1, 2, 3],
        predicate: None,
        deletes: Vec::new(),
        partition: Some(Struct::from_iter([Some(Literal::string("customer-1")), Some(Literal::int(19_000))])),
        partition_spec: Some(Arc::new(partition_spec)),
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    }
}

fn timestamp_decimal_partition_task(path: impl Into<String>) -> FileScanTask {
    let schema = Arc::new(
        Schema::builder()
            .with_fields([
                Arc::new(NestedField::optional(1, "event_ts", Type::Primitive(PrimitiveType::Timestamp))),
                Arc::new(NestedField::required(
                    2,
                    "amount",
                    Type::Primitive(PrimitiveType::Decimal { precision: 10, scale: 2 }),
                )),
            ])
            .build()
            .unwrap(),
    );
    let partition_spec = PartitionSpec::builder(Arc::clone(&schema))
        .add_partition_field("event_ts", "event_ts", Transform::Identity)
        .unwrap()
        .add_partition_field("amount", "amount", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: path.into(),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: vec![1, 2],
        predicate: None,
        deletes: Vec::new(),
        partition: Some(Struct::from_iter([
            Some(Literal::timestamp(1_700_000_000_123_456)),
            Some(Literal::decimal(12_345)),
        ])),
        partition_spec: Some(Arc::new(partition_spec)),
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    }
}

fn task_with_partition_spec_source_absent_from_scan_schema(path: impl Into<String>) -> FileScanTask {
    let scan_schema = iceberg_schema();
    let spec_schema = Arc::new(
        Schema::builder()
            .with_fields([
                Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))),
                Arc::new(NestedField::optional(99, "legacy_bucket", Type::Primitive(PrimitiveType::Long))),
            ])
            .build()
            .unwrap(),
    );
    let partition_spec = PartitionSpec::builder(spec_schema)
        .add_partition_field("legacy_bucket", "legacy_bucket", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();

    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: path.into(),
        data_file_format: DataFileFormat::Parquet,
        schema: scan_schema,
        project_field_ids: vec![1],
        predicate: None,
        deletes: Vec::new(),
        partition: Some(Struct::from_iter([Some(Literal::long(7))])),
        partition_spec: Some(Arc::new(partition_spec)),
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    }
}

struct MetadataOnlyIcebergFixture {
    _temp_dir: tempfile::TempDir,
    table_location: PathBuf,
    table: Table,
}

impl MetadataOnlyIcebergFixture {
    fn new() -> Self {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let table_location = temp_dir.path().join("table1");
        let metadata_dir = table_location.join("metadata");
        fs::create_dir_all(&metadata_dir).unwrap();

        let manifest_list1_location = metadata_dir.join("manifests_list_1.avro");
        let manifest_list2_location = metadata_dir.join("manifests_list_2.avro");
        let table_metadata1_location = metadata_dir.join("v1.json");
        let metadata_json = include_str!("../../crates/nexus-iceberg/src/testdata/metadata_v2.json")
            .replace("{{ table_location }}", table_location.to_str().unwrap())
            .replace("{{ manifest_list_1_location }}", manifest_list1_location.to_str().unwrap())
            .replace("{{ manifest_list_2_location }}", manifest_list2_location.to_str().unwrap())
            .replace("{{ table_metadata_1_location }}", table_metadata1_location.to_str().unwrap());
        let table_metadata = serde_json::from_str::<TableMetadata>(&metadata_json).unwrap();
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

    async fn write_current_manifest_list(&self) {
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let current_partition_spec = self.table.metadata().default_partition_spec();
        let manifest_path = self.table_location.join("metadata/data_manifest.avro");
        let manifest_output = self.table.file_io().new_output(manifest_path.to_str().unwrap()).unwrap();
        let mut writer = ManifestWriterBuilder::new(
            manifest_output,
            Some(current_snapshot.snapshot_id()),
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(format!("{}/data-0001.parquet", self.table_location.to_str().unwrap()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(4096)
            .record_count(17)
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

#[test]
fn no_delete_local_parquet_iceberg_tasks_produce_native_admissible_scan_plan() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("12345".to_string()),
        Some("local".to_string()),
        [file_scan_task("/tmp/iceberg/table/data-0001.parquet", DataFileFormat::Parquet, vec![])],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.snapshot_id.as_deref(), Some("12345"));
    assert_eq!(plan.catalog_type.as_deref(), Some("local"));
    assert_eq!(plan.data_files.len(), 1);
    assert_eq!(plan.data_files[0].format, TableFormatDataFileFormat::Parquet);
    assert_eq!(plan.data_files[0].size_bytes, Some(4096));
    assert_eq!(plan.data_files[0].row_count, Some(17));
    assert_eq!(plan.data_files[0].read_capability, TableFormatReadCapability::LocalFileSystem);
    assert!(plan.logical_schema.metadata().is_empty());
    assert!(plan.logical_schema.fields().iter().all(|field| field.metadata().is_empty()));
    assert!(plan.native_source_schema.metadata().is_empty());
    assert!(plan.native_source_schema.fields().iter().all(|field| field.metadata().is_empty()));
    assert_eq!(plan.projected_columns.len(), 2);
    assert_eq!(plan.projected_columns[0].field_id, Some(1));
    assert_eq!(plan.projected_columns[0].logical_name, "id");
    assert!(plan.delete_files.is_empty());
    assert!(plan.unsupported_features.is_empty());

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.table_format, "iceberg");
    assert_eq!(diagnostics.planned_data_file_count, 1);
    assert_eq!(diagnostics.planned_delete_file_count, 0);
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Eligible);
}

#[test]
fn no_delete_s3_parquet_iceberg_tasks_are_native_admissible_through_remote_kvikio() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("23456".to_string()),
        Some("aws_glue".to_string()),
        [file_scan_task(
            "s3://lakehouse/warehouse/tpch_sf1/lineitem/data-0001.parquet",
            DataFileFormat::Parquet,
            vec![],
        )],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.data_files[0].read_capability, TableFormatReadCapability::ObjectStore { scheme: "s3".to_string() });
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.native_source_read_capability, "object_store:s3");
    assert_eq!(diagnostics.admitted_native_source_file_count, 1);
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Rejected);
    assert!(diagnostics.rejected_feature_list.is_empty());
}

#[test]
fn non_identity_partition_transforms_are_retained_as_facts_without_rejecting_native_scan() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("34567".to_string()),
        Some("aws_glue".to_string()),
        [identity_and_day_partition_task("s3://lakehouse/warehouse/events/data-0001.parquet")],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());
    assert_eq!(plan.data_files.len(), 1);
    assert_eq!(plan.data_files[0].partition_values.len(), 2);

    let customer = &plan.data_files[0].partition_values[0];
    assert_eq!(customer.column_name, "customerId");
    assert_eq!(customer.transform, TableFormatPartitionTransform::Identity);
    assert_eq!(customer.value, TableFormatScalarValue::Utf8("customer-1".to_string()));

    let day = &plan.data_files[0].partition_values[1];
    assert_eq!(day.column_name, "event_ts_day");
    assert_eq!(day.transform, TableFormatPartitionTransform::Day);
    assert_eq!(day.value, TableFormatScalarValue::Date32(19_000));

    let customer_mapping = plan.projected_columns.iter().find(|mapping| mapping.logical_name == "customerId").unwrap();
    assert_eq!(customer_mapping.source, TableFormatColumnSource::IdentityPartitionValue);
    let event_mapping = plan.projected_columns.iter().find(|mapping| mapping.logical_name == "event_ts").unwrap();
    assert_eq!(event_mapping.source, TableFormatColumnSource::DirectTopLevelFileColumn);

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.admitted_native_source_file_count, 1);
    assert!(diagnostics.rejected_feature_list.is_empty());
}

#[test]
fn identity_timestamp_and_decimal_partition_values_keep_typed_metadata_literals() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("45678".to_string()),
        Some("aws_glue".to_string()),
        [timestamp_decimal_partition_task("s3://lakehouse/warehouse/events/data-0001.parquet")],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    let timestamp = &plan.data_files[0].partition_values[0];
    assert_eq!(timestamp.column_name, "event_ts");
    assert_eq!(timestamp.data_type, DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None));
    assert_eq!(timestamp.transform, TableFormatPartitionTransform::Identity);
    assert_eq!(timestamp.value, TableFormatScalarValue::Int64(1_700_000_000_123_456));

    let decimal = &plan.data_files[0].partition_values[1];
    assert_eq!(decimal.column_name, "amount");
    assert_eq!(decimal.data_type, DataType::Decimal128(10, 2));
    assert_eq!(decimal.transform, TableFormatPartitionTransform::Identity);
    assert_eq!(decimal.value, TableFormatScalarValue::Int128(12_345));

    assert!(
        plan.projected_columns.iter().all(|mapping| mapping.source == TableFormatColumnSource::IdentityPartitionValue)
    );
}

#[test]
fn partition_spec_source_absent_from_scan_schema_is_structured_rejection() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("45679".to_string()),
        Some("aws_glue".to_string()),
        [task_with_partition_spec_source_absent_from_scan_schema("s3://lakehouse/warehouse/events/data-0001.parquet")],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergPartitionTransformUnsupported]);
    assert_eq!(
        plan.diagnostics().rejected_feature_list,
        vec![CapabilityReason::IcebergPartitionTransformUnsupported.code().to_string()]
    );
}

#[tokio::test]
async fn table_format_planner_carries_plan_files_partition_spec_into_scan_plan() {
    let fixture = MetadataOnlyIcebergFixture::new();
    fixture.write_current_manifest_list().await;
    let resolved = ResolvedIcebergTable::from_table(fixture.table, SourceAccessPolicy::s3_default_chain(None)).unwrap();
    let planner = IcebergTableFormatPlanner::new(resolved);
    let request = TableFormatScanRequest::default();
    let resolved = planner.resolve_table_format_scan(request.clone()).await.unwrap();
    let plan = planner.inspect_table_format_scan(request, resolved).await.unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());
    assert_eq!(plan.data_files.len(), 1);
    assert_eq!(plan.data_files[0].partition_values.len(), 1);
    let partition = &plan.data_files[0].partition_values[0];
    assert_eq!(partition.column_name, "x");
    assert_eq!(partition.transform, TableFormatPartitionTransform::Identity);
    assert_eq!(partition.value, TableFormatScalarValue::Int64(100));

    let mapping = plan.projected_columns.iter().find(|mapping| mapping.logical_name == "x").unwrap();
    assert_eq!(mapping.source, TableFormatColumnSource::IdentityPartitionValue);
}

#[test]
fn position_delete_files_are_preserved_and_admit_native_execution() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "/tmp/iceberg/table/delete-0001.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };

    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("67890".to_string()),
        Some("local".to_string()),
        [file_scan_task("/tmp/iceberg/table/data-0001.parquet", DataFileFormat::Parquet, vec![delete_file])],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.delete_files.len(), 1);
    assert_eq!(plan.delete_files[0].location, "/tmp/iceberg/table/delete-0001.parquet");
    assert_eq!(plan.delete_files[0].format, TableFormatDataFileFormat::Parquet);
    assert_eq!(plan.delete_files[0].content, TableFormatDeleteFileContent::PositionDeletes);
    assert_eq!(plan.delete_files[0].size_bytes, Some(512));
    assert_eq!(plan.data_files[0].position_delete_files.len(), 1);
    assert_eq!(plan.data_files[0].position_delete_files[0].id, plan.delete_files[0].id);
    assert!(plan.unsupported_capability_reasons().is_empty());

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.planned_delete_file_count, 1);
    assert_eq!(diagnostics.admitted_native_source_file_count, 1);
    assert!(diagnostics.rejected_feature_list.is_empty());
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Eligible);
}

#[test]
fn equality_delete_files_are_preserved_but_reject_without_delete_footer_field_id_proof() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "/tmp/iceberg/table/delete-0001.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: Some(vec![1]),
        key_metadata: None,
    };

    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("67890".to_string()),
        Some("local".to_string()),
        [file_scan_task("/tmp/iceberg/table/data-0001.parquet", DataFileFormat::Parquet, vec![delete_file])],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.delete_files.len(), 1);
    assert_eq!(plan.delete_files[0].location, "/tmp/iceberg/table/delete-0001.parquet");
    assert_eq!(plan.delete_files[0].format, TableFormatDataFileFormat::Parquet);
    assert_eq!(plan.delete_files[0].content, TableFormatDeleteFileContent::EqualityDeletes);
    assert_eq!(plan.delete_files[0].size_bytes, Some(512));
    assert_eq!(plan.delete_files[0].equality_ids.as_deref(), Some([1].as_slice()));
    assert_eq!(plan.data_files[0].position_delete_files.len(), 0);
    assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergDeleteFilesUnsupported]);

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.planned_delete_file_count, 1);
    assert_eq!(diagnostics.admitted_native_source_file_count, 0);
    assert_eq!(diagnostics.rejected_feature_list, vec!["iceberg_delete_files_unsupported".to_string()]);
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Rejected);
}

#[test]
fn non_parquet_object_store_data_files_are_rejected_with_structured_reasons() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("24680".to_string()),
        Some("aws_glue".to_string()),
        [file_scan_task("s3://lakehouse/warehouse/tpch_sf1/lineitem/data-0001.orc", DataFileFormat::Orc, vec![])],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergNonParquetDataFile]);
    assert_eq!(plan.data_files[0].format, TableFormatDataFileFormat::Orc);
    assert_eq!(plan.data_files[0].read_capability, TableFormatReadCapability::ObjectStore { scheme: "s3".to_string() });

    let facts = plan.diagnostics().facts();
    assert_eq!(facts["catalog_type"], "aws_glue");
    assert_eq!(facts["native_source_read_capability"], "object_store:s3");
    assert_eq!(facts["rejected_feature_list"], "iceberg_non_parquet_data_file");
    assert!(!facts.values().any(|value| value.contains("local-dev")));
}

#[test]
fn unsupported_object_store_scheme_is_rejected_with_structured_reason() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("24681".to_string()),
        Some("object_store".to_string()),
        [file_scan_task("gs://bucket/table/data-0001.parquet", DataFileFormat::Parquet, vec![])],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergObjectStoreReadUnsupported]);
    assert_eq!(plan.data_files[0].read_capability, TableFormatReadCapability::ObjectStore { scheme: "gs".to_string() });

    let facts = plan.diagnostics().facts();
    assert_eq!(facts["native_source_read_capability"], "object_store");
    assert_eq!(facts["rejected_feature_list"], "iceberg_object_store_read_unsupported");
}

fn name_mapping_task(mapping: NameMapping) -> FileScanTask {
    FileScanTask {
        name_mapping: Some(Arc::new(mapping)),
        ..file_scan_task("s3://lakehouse/warehouse/lineitem/data-0001.parquet", DataFileFormat::Parquet, vec![])
    }
}

fn count_star_name_mapping_task(mapping: NameMapping) -> FileScanTask {
    FileScanTask { project_field_ids: vec![], record_count: None, ..name_mapping_task(mapping) }
}

#[test]
fn identity_name_mapping_is_admitted_natively() {
    // Spark's `add_files` attaches an identity mapping (each schema field name
    // to its own field id) for imported Parquet without embedded field ids;
    // by-name native resolution is exactly what the mapping prescribes.
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [name_mapping_task(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            MappedField::new(Some(2), vec!["shipmode".to_string()], vec![]),
        ]))],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());
}

#[test]
fn non_identity_name_mapping_is_admitted_with_field_id_source_names() {
    // A renamed source column is safe when the physical fallback names are
    // carried into native field-id binding instead of relying on by-name
    // resolution.
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [name_mapping_task(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["identifier".to_string()], vec![]),
            MappedField::new(Some(2), vec!["shipmode".to_string()], vec![]),
        ]))],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());
    let id = plan.projected_columns.iter().find(|mapping| mapping.logical_name == "id").unwrap();
    assert_eq!(id.field_id_source_names, vec!["identifier".to_string()]);
    let shipmode = plan.projected_columns.iter().find(|mapping| mapping.logical_name == "shipmode").unwrap();
    assert_eq!(shipmode.field_id_source_names, vec!["shipmode".to_string()]);
}

#[test]
fn count_star_name_mapping_carrier_is_admitted_with_field_id_source_names() {
    // COUNT(*) scans with missing row-count metadata synthesize a pure-filter
    // row-presence carrier before per-task projected columns are appended; it
    // must still carry Iceberg name-mapping aliases.
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [count_star_name_mapping_task(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["identifier".to_string()], vec![]),
            MappedField::new(Some(2), vec!["shipmode".to_string()], vec![]),
        ]))],
    )
    .unwrap();

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), Vec::<CapabilityReason>::new());
    assert!(plan.projected_columns.is_empty());
    assert_eq!(plan.pure_filter_columns.len(), 1);
    let carrier = &plan.pure_filter_columns[0];
    assert_eq!(carrier.logical_name, "id");
    assert_eq!(carrier.field_id_source_names, vec!["identifier".to_string()]);
}

#[test]
fn count_star_name_mapping_rejects_ambiguous_carrier_aliases() {
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [count_star_name_mapping_task(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["identifier".to_string(), "shipmode".to_string()], vec![]),
            MappedField::new(Some(2), vec!["shipmode".to_string()], vec![]),
        ]))],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergNameMappingUnsupported]);
}

#[test]
fn name_mapping_missing_projected_field_is_rejected() {
    // An unmapped projected field reads as null under mapping semantics but
    // resolves by name in the native scan; that divergence keeps the rejection.
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [name_mapping_task(NameMapping::new(vec![MappedField::new(Some(1), vec!["id".to_string()], vec![])]))],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergNameMappingUnsupported]);
}

#[test]
fn name_mapping_alias_claiming_projected_name_is_rejected() {
    // Field 2 also claims the name "id": mapping semantics could bind the file
    // column "id" to field 2 while by-name resolution binds it to field 1.
    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan(
        Some("13579".to_string()),
        Some("rest".to_string()),
        [name_mapping_task(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            MappedField::new(Some(2), vec!["shipmode".to_string(), "id".to_string()], vec![]),
        ]))],
    )
    .unwrap();

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergNameMappingUnsupported]);
}
