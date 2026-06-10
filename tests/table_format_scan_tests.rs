use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::physical_plan::displayable;
use datafusion_nexus::{
    CudfTableFormatScanExec, FallbackReason, TableFormat, TableFormatColumnMapping, TableFormatColumnRole,
    TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile,
    TableFormatDeleteFileContent, TableFormatDirectDeviceReadDisposition, TableFormatPartitionTransform,
    TableFormatPartitionValue, TableFormatReadCapability, TableFormatScalarValue, TableFormatScanPlan,
    TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
};

#[test]
fn no_delete_iceberg_parquet_scan_facts_are_native_admissible() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("l_shipdate", DataType::Date32, true),
        Field::new("l_shipmode", DataType::Utf8, true),
    ]));
    let partition = TableFormatPartitionValue::new(
        "l_shipmode",
        DataType::Utf8,
        true,
        TableFormatPartitionTransform::Identity,
        TableFormatScalarValue::Utf8("AIR".to_string()),
    );
    let file = TableFormatDataFile::new(
        "lineitem-0001",
        "/tmp/lineitem/data-0001.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::local_file_system(),
    )
    .with_size_bytes(4096)
    .with_row_count(128)
    .with_row_groups(vec![0, 1])
    .with_partition_values(vec![partition]);

    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "l_orderkey",
            Some("l_orderkey"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "l_shipmode",
            Some("l_shipmode"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::IdentityPartitionValue,
            TableFormatColumnRole::Projected,
        ))
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(2),
            "l_shipdate",
            Some("l_shipdate"),
            DataType::Date32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::PureFilter,
        ))
        .with_data_file(file);

    assert!(plan.is_native_admissible());
    assert_eq!(plan.planned_data_file_count(), 1);
    assert_eq!(plan.planned_delete_file_count(), 0);
    assert_eq!(plan.planned_bytes(), Some(4096));
    assert_eq!(plan.admitted_native_source_file_count(), 1);
    assert_eq!(plan.unsupported_fallback_reasons(), Vec::<FallbackReason>::new());

    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.table_format, "iceberg");
    assert_eq!(diagnostics.snapshot_id.as_deref(), Some("12345"));
    assert_eq!(diagnostics.catalog_type.as_deref(), Some("local"));
    assert_eq!(diagnostics.native_source_read_capability, "local_file_system");
    assert_eq!(diagnostics.direct_device_read_disposition, TableFormatDirectDeviceReadDisposition::Eligible);
    assert!(diagnostics.rejected_feature_list.is_empty());
}

#[test]
fn non_identity_partition_value_facts_do_not_reject_scan_admission_by_themselves() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("customerId", DataType::Utf8, false),
        Field::new("event_ts", DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None), true),
    ]));
    let day_partition = TableFormatPartitionValue::new(
        "event_ts_day",
        DataType::Int32,
        true,
        TableFormatPartitionTransform::Day,
        TableFormatScalarValue::Int32(19_000),
    );
    let file = TableFormatDataFile::new(
        "events-0001",
        "s3://lakehouse/events/data-0001.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_size_bytes(4096)
    .with_row_count(128)
    .with_partition_values(vec![day_partition]);

    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("aws_glue")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "event_ts",
            Some("event_ts"),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_data_file(file);

    assert!(plan.is_native_admissible());
    assert_eq!(plan.unsupported_fallback_reasons(), Vec::<FallbackReason>::new());
    let diagnostics = plan.diagnostics();
    assert_eq!(diagnostics.admitted_native_source_file_count, 1);
    assert!(diagnostics.rejected_feature_list.is_empty());
}

#[test]
fn empty_table_format_scan_reports_no_source_fallback_reason() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg).with_catalog_type("local");
    let diagnostics = plan.diagnostics();

    assert_eq!(diagnostics.source_fallback_reason, "none");
    assert_eq!(diagnostics.facts()["source_fallback_reason"], "none");
}

#[test]
fn table_format_scan_display_never_renders_unknown_fallback_reason() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(Arc::clone(&schema));
    let scan = CudfTableFormatScanExec::new(plan, schema);
    let rendered = displayable(&scan).indent(false).to_string();

    assert!(rendered.contains("source_fallback_reason=none"), "{rendered}");
    assert!(!rendered.contains("source_fallback_reason=unknown"), "{rendered}");
    assert!(!rendered.contains("fallback_reason=unknown"), "{rendered}");
}

#[test]
fn delete_file_scan_facts_preserve_rejection_without_reading_credentials() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "data-0001",
        "s3://lakehouse/warehouse/tpch_sf1/lineitem/data-0001.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_size_bytes(2048);
    let delete_file = TableFormatDeleteFile::new(
        "delete-0001",
        "s3://lakehouse/warehouse/tpch_sf1/lineitem/delete-0001.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_size_bytes(128)
    .with_row_count(4);

    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("67890")
        .with_catalog_type("aws_glue")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(data_file)
        .with_delete_file(delete_file)
        .with_unsupported_feature(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            "delete-0001",
            "position delete files require Milestone 2 row-level semantics",
        ))
        .with_unsupported_feature(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported,
            "data-0001",
            "s3 object-store reads are not represented by Milestone A local source facts",
        ));

    assert!(!plan.is_native_admissible());
    assert_eq!(plan.planned_data_file_count(), 1);
    assert_eq!(plan.planned_delete_file_count(), 1);
    assert_eq!(plan.admitted_native_source_file_count(), 0);
    assert_eq!(
        plan.unsupported_fallback_reasons(),
        vec![FallbackReason::IcebergDeleteFilesUnsupported, FallbackReason::IcebergObjectStoreReadUnsupported]
    );

    let diagnostics = plan.diagnostics();
    let facts = diagnostics.facts();
    assert_eq!(facts["table_format"], "iceberg");
    assert_eq!(facts["catalog_type"], "aws_glue");
    assert_eq!(facts["planned_delete_file_count"], "1");
    assert_eq!(facts["admitted_native_source_file_count"], "0");
    assert_eq!(facts["native_source_read_capability"], "object_store:s3");
    assert_eq!(
        facts["rejected_feature_list"],
        "iceberg_delete_files_unsupported,iceberg_object_store_read_unsupported"
    );
    assert_eq!(facts["direct_device_read_disposition"], "rejected");

    let rendered = diagnostics.to_string();
    assert!(rendered.contains("snapshot_id=67890"));
    assert!(rendered.contains("planned_delete_file_count=1"));
    assert!(!rendered.contains("AWS_ACCESS_KEY_ID"));
    assert!(!rendered.contains("AWS_SECRET_ACCESS_KEY"));
    assert!(!rendered.contains("AWS_SESSION_TOKEN"));
    assert!(!rendered.contains("local-dev"));
}

#[test]
fn unsupported_feature_codes_map_to_stable_fallback_reasons() {
    let cases = [
        (TableFormatUnsupportedFeatureCode::TableFormatUnsupported, FallbackReason::TableFormatUnsupported),
        (TableFormatUnsupportedFeatureCode::TableFormatPlanningFailed, FallbackReason::TableFormatPlanningFailed),
        (TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile, FallbackReason::IcebergNonParquetDataFile),
        (
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            FallbackReason::IcebergDeleteFilesUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported,
            FallbackReason::IcebergFieldIdMappingUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported,
            FallbackReason::IcebergNameMappingUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergNestedProjectionUnsupported,
            FallbackReason::IcebergNestedProjectionUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
            FallbackReason::IcebergPartitionTransformUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported,
            FallbackReason::IcebergResidualPredicateUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergRowSelectionUnsupported,
            FallbackReason::IcebergRowSelectionUnsupported,
        ),
        (
            TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported,
            FallbackReason::IcebergObjectStoreReadUnsupported,
        ),
        (TableFormatUnsupportedFeatureCode::IcebergSnapshotUnsupported, FallbackReason::IcebergSnapshotUnsupported),
    ];

    for (code, reason) in cases {
        let feature = TableFormatUnsupportedFeature::new(code, reason.code());
        assert_eq!(feature.fallback_reason(), reason);
        assert_eq!(feature.code(), reason.code());
    }
}
