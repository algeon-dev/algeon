use std::{path::PathBuf, sync::Arc};

use datafusion::logical_expr::{Cast, col, lit};
use iceberg::{
    scan::FileScanTask,
    spec::{DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema as IcebergSchema, Struct, Type},
};
use nexus_iceberg::S3FileIoConfigVerification;

use super::*;
use crate::{
    capability::CapabilityReason,
    table_format::{
        iceberg::planner::{
            predicate::datafusion_filters_to_iceberg_predicate,
            tasks::{
                append_delete_file_facts, iceberg_file_scan_tasks_to_table_format_scan_plan_with_config,
                iceberg_task_content_identity,
            },
        },
        scan::TableFormatLakehouseCachePolicy,
    },
};

fn iceberg_schema() -> Arc<IcebergSchema> {
    Arc::new(
        IcebergSchema::builder()
            .with_fields([Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)))])
            .build()
            .unwrap(),
    )
}

fn fixture_delete_file(
    file_path: &str,
    file_size_in_bytes: u64,
    file_type: DataContentType,
    equality_ids: Option<Vec<i32>>,
) -> FileScanTaskDeleteFile {
    FileScanTaskDeleteFile {
        file_path: file_path.to_string(),
        file_size_in_bytes,
        file_type,
        partition_spec_id: 0,
        equality_ids,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
        record_count: None,
        key_metadata: None,
    }
}

/// Plain single-file task fixture. Row-level and sequence metadata stay
/// absent because these tests exercise identity, limits, and delete facts.
fn fixture_task(data_file_path: &str, record_count: Option<u64>, deletes: Vec<FileScanTaskDeleteFile>) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count,
        first_row_id: None,
        data_sequence_number: None,
        data_file_path: data_file_path.to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes,
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        unified_partition_type: None,
        case_sensitive: true,
        key_metadata: None,
    }
}

#[test]
fn iceberg_task_conversion_uses_injected_parquet_read_strategy_config() {
    let config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(PathBuf::from("/tmp/cudf-iceberg-test-cache")),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        storage_endpoint: Some("http://rustfs.local:9000/minio".to_string()),
        storage_profile_id: Some("rest-s3".to_string()),
        ..Default::default()
    };
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![]);

    let plan = iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
        Some("123".to_string()),
        Some("rest".to_string()),
        [task],
        &config,
    )
    .unwrap();
    let resolved = plan.data_files[0].resolved_parquet_source.as_ref().unwrap();

    assert_eq!(resolved.storage_profile_id(), Some("rest-s3"));
    assert_eq!(resolved.storage_domain.as_deref(), Some("bucket@http://rustfs.local:9000/minio"));
    assert!(resolved.content_identity.as_deref().is_some_and(|identity| {
        identity.starts_with("iceberg_snapshot_123_file_") && identity.ends_with("_size_4096_rows_17")
    }));
    assert_eq!(resolved.cache_status.as_str(), "fill_before_read");
}

#[test]
fn iceberg_snapshot_content_identity_is_complete_object_scoped() {
    let mut first = fixture_task("s3://bucket/table/data.parquet", None, vec![]);
    first.length = 2048;
    let second = FileScanTask { start: 2048, length: 2048, record_count: None, ..first.clone() };

    assert_eq!(iceberg_task_content_identity(Some("123"), &first), iceberg_task_content_identity(Some("123"), &second));
}

#[test]
fn remote_s3_without_file_io_policy_proof_is_native_unsupported_but_local_is_not() {
    let mut remote = TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(TableFormatDataFile::new(
        "remote",
        "s3://bucket/table/data.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    ));
    append_unverified_remote_s3_source_access_rejection(&mut remote, S3FileIoConfigVerification::NoS3Properties);

    assert_eq!(remote.unsupported_capability_reasons(), vec![CapabilityReason::IcebergFileIoSourceAccessUnverified]);
    assert!(!remote.is_native_admissible());

    let mut local = TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(TableFormatDataFile::new(
        "local",
        "file:///tmp/table/data.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::local_file_system(),
    ));
    append_unverified_remote_s3_source_access_rejection(&mut local, S3FileIoConfigVerification::NoS3Properties);

    assert!(local.unsupported_features.is_empty());
}

#[test]
fn remote_s3_with_exact_file_io_policy_remains_native_admissible() {
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(TableFormatDataFile::new(
        "remote",
        "s3://bucket/table/data.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    ));

    append_unverified_remote_s3_source_access_rejection(&mut plan, S3FileIoConfigVerification::ExactPolicy);

    assert!(plan.unsupported_features.is_empty());
    assert!(plan.is_native_admissible());
}

#[test]
fn remote_s3_delete_source_without_file_io_policy_proof_is_native_unsupported() {
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete",
        "local",
        "s3://bucket/table/delete.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    );
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_data_file(
            TableFormatDataFile::new(
                "local",
                "file:///tmp/table/data.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_position_delete_files(vec![delete_file.clone()]),
        )
        .with_delete_file(delete_file);

    append_unverified_remote_s3_source_access_rejection(&mut plan, S3FileIoConfigVerification::NoS3Properties);

    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergFileIoSourceAccessUnverified]);
}

#[test]
fn iceberg_snapshot_content_identity_hashes_object_path() {
    let task = fixture_task("s3://secret-bucket/table/prefix/data.parquet?token=do-not-leak", Some(17), vec![]);

    let identity = iceberg_task_content_identity(Some("123"), &task).unwrap();

    assert!(identity.starts_with("iceberg_snapshot_123_file_"));
    assert!(identity.ends_with("_size_4096_rows_17"));
    assert!(!identity.contains("s3://"));
    assert!(!identity.contains("secret-bucket"));
    assert!(!identity.contains("do-not-leak"));
}

#[test]
fn iceberg_data_file_id_hashes_credential_free_object_path() {
    let task =
        fixture_task("s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=do-not-leak", Some(17), vec![]);

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();
    let file_id = &plan.data_files[0].id;

    assert!(file_id.starts_with("iceberg-data-file-0-h"));
    assert!(!file_id.contains("s3://"));
    assert!(!file_id.contains("secret-bucket"));
    assert!(!file_id.contains("X-Amz-Signature"));
    assert!(!file_id.contains("do-not-leak"));
}

#[test]
fn predicate_free_scan_limit_trims_data_files_before_footer_proof() {
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_scan_limit(Some(11))
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/data-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(5),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "file-1",
                "s3://bucket/table/data-1.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(7),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "file-2",
                "s3://bucket/table/data-2.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(100),
        );

    trim_predicate_free_scan_limit_data_files(&mut plan, false);

    assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-2"]);
}

#[test]
fn predicate_free_scan_limit_prunes_dropped_file_local_unsupported_features() {
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_scan_limit(Some(3))
        .with_data_file(
            TableFormatDataFile::new(
                "dropped-file",
                "s3://bucket/table/dropped.orc",
                TableFormatDataFileFormat::Unknown("orc".to_string()),
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(2),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "retained-file",
                "s3://bucket/table/retained.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(10),
        )
        .with_unsupported_feature(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile,
            "dropped-file",
            "dropped file should not keep blocking the trimmed plan",
        ));

    trim_predicate_free_scan_limit_data_files(&mut plan, false);

    assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["retained-file"]);
    assert!(plan.unsupported_features.is_empty());
}

#[test]
fn predicate_free_scan_limit_keeps_all_files_when_delete_semantics_are_present() {
    let delete_file =
        fixture_delete_file("s3://bucket/table/delete.parquet", 512, DataContentType::PositionDeletes, None);
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![delete_file]);
    let mut plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
            .unwrap()
            .with_scan_limit(Some(1));

    trim_predicate_free_scan_limit_data_files(&mut plan, false);

    assert_eq!(plan.data_files.len(), 1);
    assert_eq!(plan.delete_files.len(), 1);
}

#[test]
fn predicate_free_scan_limit_does_not_trim_equality_delete_files_by_metadata_lower_bound() {
    let delete_0 = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        "s3://bucket/table/delete-0.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::EqualityDeletes,
    )
    .with_row_count(10)
    .with_equality_ids([1]);
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_scan_limit(Some(1))
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/data-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(10)
            .with_equality_delete_files(vec![delete_0.clone()]),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "file-1",
                "s3://bucket/table/data-1.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(1),
        )
        .with_delete_file(delete_0);

    trim_predicate_free_scan_limit_data_files(&mut plan, false);

    assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-0", "file-1"]);
    assert_eq!(plan.delete_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["delete-0"]);
}

#[test]
fn predicate_free_scan_limit_trims_delete_files_when_live_lower_bound_covers_limit() {
    let delete_0 = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        "s3://bucket/table/delete-0.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_row_count(9);
    let delete_1 = TableFormatDeleteFile::for_data_file(
        "delete-1",
        "file-1",
        "s3://bucket/table/delete-1.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_row_count(5);
    let delete_2 = TableFormatDeleteFile::for_data_file(
        "delete-2",
        "file-2",
        "s3://bucket/table/delete-2.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_row_count(1);
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_scan_limit(Some(2))
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                "s3://bucket/table/data-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(10)
            .with_position_delete_files(vec![delete_0.clone()]),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "file-1",
                "s3://bucket/table/data-1.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(10)
            .with_position_delete_files(vec![delete_1.clone()]),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "file-2",
                "s3://bucket/table/data-2.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_row_count(10)
            .with_position_delete_files(vec![delete_2.clone()]),
        );
    plan.delete_files = vec![delete_0, delete_1, delete_2];

    trim_predicate_free_scan_limit_data_files(&mut plan, false);

    assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-2"]);
    assert_eq!(plan.delete_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["delete-2"]);
}

#[test]
fn iceberg_delete_file_id_hashes_credential_free_object_path() {
    let delete_file = fixture_delete_file(
        "s3://secret-bucket/table/delete.parquet?X-Amz-Signature=do-not-leak",
        512,
        DataContentType::PositionDeletes,
        None,
    );
    let task = fixture_task(
        "s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=data-token",
        Some(17),
        vec![delete_file],
    );

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();
    let delete_id = &plan.delete_files[0].id;

    assert!(delete_id.contains("#delete-0-h"));
    assert!(!delete_id.contains("s3://"));
    assert!(!delete_id.contains("secret-bucket"));
    assert!(!delete_id.contains("X-Amz-Signature"));
    assert!(!delete_id.contains("do-not-leak"));
    assert!(!delete_id.contains("data-token"));
}

#[test]
fn iceberg_task_delete_attachment_preserves_data_row_groups() {
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg);
    let mut data_file = TableFormatDataFile::new(
        "data-0",
        "s3://bucket/table/data.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_row_groups(vec![2, 3]);
    let delete_file =
        fixture_delete_file("s3://bucket/table/delete.parquet", 512, DataContentType::PositionDeletes, None);

    append_delete_file_facts(&mut plan, &mut data_file, "data-0", 0, &delete_file).unwrap();

    assert_eq!(data_file.row_groups, Some(vec![2, 3]));
    assert_eq!(data_file.position_delete_files.len(), 1);
    assert_eq!(plan.delete_files.len(), 1);
}

#[test]
fn iceberg_position_delete_files_are_associated_for_native_execution() {
    let delete_file =
        fixture_delete_file("s3://bucket/table/delete.parquet", 512, DataContentType::PositionDeletes, None);
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![delete_file]);

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();

    assert!(plan.unsupported_features.is_empty());
    assert_eq!(plan.delete_files.len(), 1);
    assert_eq!(plan.data_files[0].position_delete_files.len(), 1);
    assert_eq!(plan.delete_files[0].data_file_id, plan.data_files[0].id);
    assert_eq!(plan.delete_files[0].id, plan.data_files[0].position_delete_files[0].id);
    assert!(plan.delete_files[0].is_supported_position_delete());
    assert!(plan.is_native_admissible());
}

#[test]
fn iceberg_equality_delete_files_without_footer_field_id_proof_are_structured_rejections() {
    let delete_file =
        fixture_delete_file("s3://bucket/table/delete.parquet", 512, DataContentType::EqualityDeletes, Some(vec![1]));
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![delete_file]);

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();

    assert_eq!(plan.data_files[0].position_delete_files.len(), 0);
    assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
    assert_eq!(plan.delete_files.len(), 1);
    assert_eq!(plan.delete_files[0].data_file_id, plan.data_files[0].id);
    assert_eq!(plan.delete_files[0].equality_ids.as_deref(), Some([1].as_slice()));
    assert!(plan.delete_files[0].is_supported_equality_delete());
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergDeleteFilesUnsupported]);
    assert_eq!(plan.first_not_supported_reason(), Some(CapabilityReason::IcebergDeleteFilesUnsupported));
    assert!(!plan.is_native_admissible());
}

#[test]
fn iceberg_equality_delete_files_without_key_ids_remain_structured_rejections() {
    let delete_file =
        fixture_delete_file("s3://bucket/table/delete.parquet", 512, DataContentType::EqualityDeletes, None);
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![delete_file]);

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();

    assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
    assert_eq!(plan.unsupported_capability_reasons(), vec![CapabilityReason::IcebergDeleteFilesUnsupported]);
    assert_eq!(plan.first_not_supported_reason(), Some(CapabilityReason::IcebergDeleteFilesUnsupported));
    assert!(!plan.is_native_admissible());
}

#[test]
fn iceberg_task_conversion_uses_weak_identity_without_snapshot() {
    let task = fixture_task("s3://bucket/table/data.parquet", Some(17), vec![]);

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();
    let resolved = plan.data_files[0].resolved_parquet_source.as_ref().unwrap();

    assert_eq!(resolved.content_identity.as_deref(), Some("weak_size_4096_rows_17"));
}

#[test]
fn datafusion_filters_convert_to_iceberg_metadata_predicate() {
    let filters = vec![
        datafusion::logical_expr::binary_expr(
            col("id"),
            Operator::GtEq,
            lit(datafusion::common::ScalarValue::Int64(Some(10))),
        ),
        datafusion::logical_expr::binary_expr(
            lit(datafusion::common::ScalarValue::Utf8(Some("BUILDING".to_string()))),
            Operator::Eq,
            col("status"),
        ),
    ];

    let predicate = datafusion_filters_to_iceberg_predicate(&filters).unwrap();
    let rendered = predicate.to_string();

    assert!(rendered.contains("id >= 10"));
    assert!(rendered.contains("status = \"BUILDING\""));
}

#[test]
fn unsupported_datafusion_filter_does_not_block_supported_filter_pushdown() {
    let filters = vec![datafusion::logical_expr::binary_expr(
        datafusion::logical_expr::binary_expr(
            col("id"),
            Operator::Lt,
            lit(datafusion::common::ScalarValue::Int64(Some(99))),
        ),
        Operator::And,
        lit(datafusion::common::ScalarValue::Boolean(Some(true))),
    )];

    let predicate = datafusion_filters_to_iceberg_predicate(&filters).unwrap();

    assert_eq!(predicate.to_string(), "id < 99");
}

#[test]
fn cast_filters_are_not_pushed_into_iceberg_metadata_predicates() {
    let filters = vec![datafusion::logical_expr::binary_expr(
        Expr::Cast(Cast::new(Box::new(col("float_col")), DataType::Int32)),
        Operator::Eq,
        lit(datafusion::common::ScalarValue::Int32(Some(1))),
    )];

    assert!(datafusion_filters_to_iceberg_predicate(&filters).is_none());
}

#[test]
fn not_of_partially_supported_and_is_not_pushed_into_iceberg_metadata_predicates() {
    let supported = datafusion::logical_expr::binary_expr(
        col("id"),
        Operator::Lt,
        lit(datafusion::common::ScalarValue::Int64(Some(99))),
    );
    let unsupported = datafusion::logical_expr::binary_expr(
        Expr::Cast(Cast::new(Box::new(col("float_col")), DataType::Int32)),
        Operator::Eq,
        lit(datafusion::common::ScalarValue::Int32(Some(1))),
    );
    let filters =
        vec![Expr::Not(Box::new(datafusion::logical_expr::binary_expr(supported, Operator::And, unsupported)))];

    assert!(datafusion_filters_to_iceberg_predicate(&filters).is_none());
}
