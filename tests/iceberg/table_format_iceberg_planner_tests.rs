use std::sync::Arc;

use datafusion_nexus::{
    FallbackReason, TableFormatDataFileFormat, TableFormatDeleteFileContent, TableFormatDirectDeviceReadDisposition,
    TableFormatReadCapability, iceberg::iceberg_file_scan_tasks_to_table_format_scan_plan,
};
use iceberg::{
    scan::{FileScanTask, FileScanTaskDeleteFile},
    spec::{DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, Struct, Type},
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
fn no_delete_s3_parquet_iceberg_tasks_are_native_admissible_through_host_buffer_fetch() {
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
    assert_eq!(plan.unsupported_fallback_reasons(), Vec::<FallbackReason>::new());

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.native_source_read_capability, "object_store:s3");
    assert_eq!(diagnostics.admitted_native_source_file_count, 1);
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Rejected);
    assert!(diagnostics.rejected_feature_list.is_empty());
}

#[test]
fn delete_files_are_preserved_and_reject_native_admission() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "/tmp/iceberg/table/delete-0001.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
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
    assert_eq!(plan.delete_files[0].content, TableFormatDeleteFileContent::PositionDeletes);
    assert_eq!(plan.delete_files[0].size_bytes, Some(512));
    assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergDeleteFilesUnsupported]);

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.planned_delete_file_count, 1);
    assert_eq!(diagnostics.admitted_native_source_file_count, 0);
    assert_eq!(diagnostics.rejected_feature_list, vec!["iceberg_delete_files_unsupported"]);
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
    assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergNonParquetDataFile]);
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
    assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergObjectStoreReadUnsupported]);
    assert_eq!(plan.data_files[0].read_capability, TableFormatReadCapability::ObjectStore { scheme: "gs".to_string() });

    let facts = plan.diagnostics().facts();
    assert_eq!(facts["native_source_read_capability"], "object_store");
    assert_eq!(facts["rejected_feature_list"], "iceberg_object_store_read_unsupported");
}
