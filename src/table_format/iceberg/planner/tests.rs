use std::{path::PathBuf, sync::Arc};

use datafusion::logical_expr::{Cast, col, lit};
use iceberg::{
    scan::FileScanTask,
    spec::{
        DataContentType, DataFileBuilder, DataFileFormat, ManifestStatus, MappedField, NameMapping, NestedField,
        PrimitiveType, Schema as IcebergSchema, Struct, Type,
    },
};
use nexus_query_engine::source::{CredentialMode, ResolvedParquetSource};

use super::*;
use crate::{policy::FallbackReason, table_format::scan::TableFormatLakehouseCachePolicy};

fn iceberg_schema() -> Arc<IcebergSchema> {
    Arc::new(
        IcebergSchema::builder()
            .with_fields([Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)))])
            .build()
            .unwrap(),
    )
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
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

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
fn predicate_free_limit_manifest_entry_task_carries_table_name_mapping() {
    let data_file = DataFileBuilder::default()
        .partition_spec_id(0)
        .content(DataContentType::Data)
        .file_path("s3://bucket/table/data.parquet".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(4096)
        .record_count(17)
        .partition(Struct::empty())
        .key_metadata(None)
        .build()
        .unwrap();
    let entry = ManifestEntry {
        status: ManifestStatus::Added,
        snapshot_id: Some(7),
        sequence_number: Some(11),
        file_sequence_number: Some(13),
        data_file,
    };
    let name_mapping = Arc::new(NameMapping::new(vec![MappedField::new(
        Some(1),
        vec!["id".to_string(), "old_id".to_string()],
        vec![],
    )]));

    let task = file_scan_task_from_manifest_entry(
        &entry,
        iceberg_schema(),
        Arc::new(vec![1]),
        None,
        Some(Arc::clone(&name_mapping)),
        Vec::new(),
        true,
    );

    let task_mapping = task.name_mapping.expect("predicate-free LIMIT task should carry table name mapping");
    assert_eq!(task_mapping.fields(), name_mapping.fields());
}

#[test]
fn iceberg_snapshot_content_identity_is_complete_object_scoped() {
    let first = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 2048,
        record_count: None,
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };
    let second = FileScanTask { start: 2048, length: 2048, record_count: None, ..first.clone() };

    assert_eq!(iceberg_task_content_identity(Some("123"), &first), iceberg_task_content_identity(Some("123"), &second));
}

#[test]
fn iceberg_snapshot_content_identity_hashes_object_path() {
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://secret-bucket/table/prefix/data.parquet?token=do-not-leak".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

    let identity = iceberg_task_content_identity(Some("123"), &task).unwrap();

    assert!(identity.starts_with("iceberg_snapshot_123_file_"));
    assert!(identity.ends_with("_size_4096_rows_17"));
    assert!(!identity.contains("s3://"));
    assert!(!identity.contains("secret-bucket"));
    assert!(!identity.contains("do-not-leak"));
}

#[test]
fn iceberg_data_file_id_hashes_credential_free_object_path() {
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=do-not-leak".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

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
fn iceberg_position_delete_bounds_use_reserved_manifest_fields() {
    assert_eq!(
        iceberg_position_delete_bound(&std::collections::HashMap::from([(
            RESERVED_FIELD_ID_DELETE_FILE_POS,
            Datum::long(7),
        )])),
        Some(7)
    );
    assert_eq!(
        iceberg_position_delete_bound(&std::collections::HashMap::from([(RESERVED_FIELD_ID_POS, Datum::long(9))])),
        Some(9)
    );
    assert_eq!(
        iceberg_position_delete_bound(&std::collections::HashMap::from([(
            RESERVED_FIELD_ID_DELETE_FILE_POS,
            Datum::long(-1),
        )])),
        None
    );
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
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![delete_file],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };
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

    assert_eq!(scan_plan_live_row_lower_bound(&plan), None);

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
fn limited_delete_matching_preserves_iceberg_sequence_partition_and_reference_semantics() {
    let data_file = test_data_file("s3://bucket/table/data.parquet", 7);
    let same_sequence_position_delete = test_limited_delete(DataContentType::PositionDeletes, 7, Some(10), None, None);
    let older_position_delete = test_limited_delete(DataContentType::PositionDeletes, 7, Some(9), None, None);
    let wrong_spec_position_delete = test_limited_delete(DataContentType::PositionDeletes, 8, Some(10), None, None);
    let referenced_position_delete = test_limited_delete(
        DataContentType::PositionDeletes,
        7,
        Some(10),
        None,
        Some("s3://bucket/table/data.parquet"),
    );
    let other_referenced_position_delete = test_limited_delete(
        DataContentType::PositionDeletes,
        7,
        Some(10),
        None,
        Some("s3://bucket/table/other.parquet"),
    );
    let same_sequence_global_equality_delete =
        test_limited_delete(DataContentType::EqualityDeletes, 99, Some(10), Some(vec![1]), None);
    let newer_global_equality_delete =
        test_limited_delete(DataContentType::EqualityDeletes, 99, Some(11), Some(vec![1]), None);

    assert!(same_sequence_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(!older_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(!wrong_spec_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(referenced_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(!other_referenced_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(!same_sequence_global_equality_delete.applies_to_data_file(&data_file, 7, Some(10)));
    assert!(newer_global_equality_delete.applies_to_data_file(&data_file, 7, Some(10)));
}

#[test]
fn iceberg_delete_file_id_hashes_credential_free_object_path() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://secret-bucket/table/delete.parquet?X-Amz-Signature=do-not-leak".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=data-token".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![delete_file],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

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
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };

    append_delete_file_facts(&mut plan, &mut data_file, "data-0", 0, &delete_file, &Default::default()).unwrap();

    assert_eq!(data_file.row_groups, Some(vec![2, 3]));
    assert_eq!(data_file.position_delete_files.len(), 1);
    assert_eq!(plan.delete_files.len(), 1);
}

fn test_data_file(path: &str, partition_spec_id: i32) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .partition_spec_id(partition_spec_id)
        .record_count(10)
        .file_size_in_bytes(1024)
        .build()
        .unwrap()
}

fn test_limited_delete(
    file_type: DataContentType,
    partition_spec_id: i32,
    sequence_number: Option<i64>,
    equality_ids: Option<Vec<i32>>,
    referenced_data_file: Option<&str>,
) -> LimitedDeleteFile {
    LimitedDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type,
        partition_spec_id,
        equality_ids,
        sequence_number,
        record_count: 1,
        partition: Struct::empty(),
        referenced_data_file: referenced_data_file.map(str::to_string),
        position_lower_bound: None,
        position_upper_bound: None,
        key_metadata: None,
    }
}

#[test]
fn attach_manifest_delete_metadata_copies_row_count_and_position_bounds_to_associated_delete_files() {
    let table_delete = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "data-0",
        "s3://bucket/table/delete.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::object_store("s3"))
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/delete.parquet",
        CredentialMode::SharedProcessEnv,
    ));
    let data_file = TableFormatDataFile::new(
        "data-0",
        "s3://bucket/table/data.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_position_delete_files(vec![table_delete.clone()]);
    let mut plan =
        TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(data_file).with_delete_file(table_delete);
    let mut limited_delete = test_limited_delete(
        DataContentType::PositionDeletes,
        0,
        Some(10),
        None,
        Some("s3://bucket/table/data.parquet"),
    );
    limited_delete.record_count = 7;
    limited_delete.position_lower_bound = Some(11);
    limited_delete.position_upper_bound = Some(29);

    attach_manifest_delete_metadata(&mut plan, 0, &[limited_delete]);

    let associated_delete = &plan.data_files[0].position_delete_files[0];
    assert_eq!(associated_delete.row_count, Some(7));
    assert_eq!(associated_delete.referenced_data_file.as_deref(), Some("s3://bucket/table/data.parquet"));
    assert_eq!(associated_delete.position_lower_bound, Some(11));
    assert_eq!(associated_delete.position_upper_bound, Some(29));
    assert_eq!(plan.delete_files[0].row_count, Some(7));
    assert_eq!(plan.delete_files[0].referenced_data_file.as_deref(), Some("s3://bucket/table/data.parquet"));
    assert_eq!(plan.delete_files[0].position_lower_bound, Some(11));
    assert_eq!(plan.delete_files[0].position_upper_bound, Some(29));
}

#[test]
fn iceberg_position_delete_files_are_associated_for_native_execution() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![delete_file],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

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
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: Some(vec![1]),
        key_metadata: None,
    };
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![delete_file],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();

    assert_eq!(plan.data_files[0].position_delete_files.len(), 0);
    assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
    assert_eq!(plan.delete_files.len(), 1);
    assert_eq!(plan.delete_files[0].data_file_id, plan.data_files[0].id);
    assert_eq!(plan.delete_files[0].equality_ids.as_deref(), Some([1].as_slice()));
    assert!(plan.delete_files[0].is_supported_equality_delete());
    assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergDeleteFilesUnsupported]);
    assert_eq!(plan.first_rejection_reason(), Some(FallbackReason::IcebergDeleteFilesUnsupported));
    assert!(!plan.is_native_admissible());
}

#[test]
fn iceberg_equality_delete_files_without_key_ids_remain_structured_rejections() {
    let delete_file = FileScanTaskDeleteFile {
        file_path: "s3://bucket/table/delete.parquet".to_string(),
        file_size_in_bytes: 512,
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        key_metadata: None,
    };
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![delete_file],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

    let plan =
        iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default()).unwrap();

    assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
    assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergDeleteFilesUnsupported]);
    assert_eq!(plan.first_rejection_reason(), Some(FallbackReason::IcebergDeleteFilesUnsupported));
    assert!(!plan.is_native_admissible());
}

#[test]
fn iceberg_task_conversion_uses_weak_identity_without_snapshot() {
    let task = FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: Some(17),
        data_file_path: "s3://bucket/table/data.parquet".to_string(),
        data_file_format: DataFileFormat::Parquet,
        schema: iceberg_schema(),
        project_field_ids: vec![1],
        predicate: None,
        deletes: vec![],
        partition: Some(Struct::empty()),
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        key_metadata: None,
    };

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
