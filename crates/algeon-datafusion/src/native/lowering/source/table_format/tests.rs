use std::{
    collections::{BTreeSet, HashMap},
    fs,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col, lit},
    physical_plan::{empty::EmptyExec, filter::FilterExec},
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_engine::{
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput},
    plan::PlanNode,
    runtime::ExecutionContext,
    source::{self, CredentialMode, ResolvedParquetSource, SourceColumnRef as NativeSourceColumnRef},
};

use super::*;
use crate::table_format::{
    AlgeonIcebergNativeScanOutcome, AlgeonIcebergScanExec, TableFormat, TableFormatDeleteFile,
    TableFormatDeleteFileContent, TableFormatPartitionTransform, TableFormatPartitionValue, TableFormatReadCapability,
    TableFormatScalarValue,
};

fn test_scan_exec(scan_plan: TableFormatScanPlan, schema: Arc<Schema>) -> AlgeonIcebergScanExec {
    let cpu_delegate: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&schema)));
    AlgeonIcebergScanExec::try_new(cpu_delegate, schema, AlgeonIcebergNativeScanOutcome::from_scan_plan(scan_plan))
        .unwrap()
}

fn test_column_requirements() -> NativeParquetColumnRequirements {
    NativeParquetColumnRequirements {
        read_columns: vec!["c0".to_string()],
        read_column_file_indices: vec![0],
        projected_output_columns: vec!["c0".to_string()],
        projected_output_column_source_refs: vec![NativeSourceColumnRef::File(0)],
        pure_filter_columns: Vec::new(),
        pure_filter_column_source_refs: Vec::new(),
        predicate_columns: Vec::new(),
        partition_columns: Vec::new(),
        partition_column_values: Vec::new(),
        partition_column_indices: Vec::new(),
        missing_columns: Vec::new(),
        scan_output_columns: vec!["c0".to_string()],
        requires_physical_row_presence: false,
    }
}

#[test]
fn table_format_local_footer_field_id_mismatch_rejects_direct_file_column() {
    let dir = unique_table_format_temp_dir("field-id-mismatch");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let file_schema = table_format_file_schema(&scan_plan);

    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();
    let result = table_format_source_read_groups(&scan_plan, &requirements, output_schema);

    assert_eq!(result.unwrap_err(), CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_matching_field_id_admits_direct_file_column() {
    let dir = unique_table_format_temp_dir("field-id-match");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let file_schema = table_format_file_schema(&scan_plan);

    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();

    assert_eq!(requirements.read_columns, vec!["status"]);
    assert_eq!(requirements.read_column_file_indices, vec![0]);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_missing_field_id_uses_same_name_direct_file_column() {
    let dir = unique_table_format_temp_dir("field-id-missing-name-match");
    let path = dir.join("part.parquet");
    write_plain_parquet_file_with_nullability(&path, "status", true);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["status"]);
    assert_eq!(groups[0].requirements.read_column_file_indices, vec![0]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "status");
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_missing_field_id_uses_name_mapping_alias() {
    let dir = unique_table_format_temp_dir("field-id-missing-name-mapping-alias");
    let path = dir.join("part.parquet");
    write_plain_parquet_file_with_nullability(&path, "old_status", true);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let mut scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
    scan_plan.projected_columns[0] = scan_plan.projected_columns[0].clone().with_field_id_source_names(["old_status"]);
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
    assert_eq!(groups[0].requirements.read_column_file_indices, vec![0]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
    let (native_source, _) =
        table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None, &[])
            .unwrap();
    assert_eq!(native_source.read_columns, vec!["old_status"]);
    assert_eq!(native_source.projected_output_columns, vec!["status"]);
    assert_eq!(native_source.files[0].field_id_read_bindings[0].physical_name, "old_status");
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_footer_missing_field_id_rejects_ambiguous_name_mapping_aliases() {
    let footer_schema = Arc::new(Schema::new(vec![
        Field::new("status", DataType::Int64, false),
        Field::new("old_status", DataType::Int64, false),
    ]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(
            TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            )
            .with_field_id_source_names(["status", "old_status"]),
        );
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_footer_missing_field_id_rejects_ambiguous_aliases_before_type_filtering() {
    let footer_schema = Arc::new(Schema::new(vec![
        Field::new("status", DataType::Utf8, false),
        Field::new("old_status", DataType::Int64, false),
    ]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(
            TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            )
            .with_field_id_source_names(["status", "old_status"]),
        );
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_footer_alias_with_conflicting_field_id_rejects_name_mapping_fallback() {
    let footer_field = Field::new("old_status", DataType::Int64, false)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "9".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(
            TableFormatColumnMapping::new(
                Some(2),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            )
            .with_field_id_source_names(["old_status"]),
        );
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_footer_malformed_field_id_metadata_rejects_same_name_fallback() {
    let footer_field = Field::new("status", DataType::Int64, false)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "bad-id".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_direct_timestamp_timezone_uses_timezone_free_source_contract() {
    let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "event_ts",
            Some("event_ts"),
            timestamp_type.clone(),
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, Arc::clone(&output_schema)).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(),
        &DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
    );
    assert_eq!(groups[0].output_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
}

#[test]
fn table_format_timestamp_partition_source_contract_keeps_logical_timezone() {
    let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_row_count(1)
    .with_partition_values(vec![TableFormatPartitionValue::new(
        "event_ts",
        timestamp_type.clone(),
        false,
        TableFormatPartitionTransform::Identity,
        TableFormatScalarValue::Int64(123),
    )])
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "event_ts",
            None::<String>,
            timestamp_type.clone(),
            false,
            TableFormatColumnSource::IdentityPartitionValue,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let (native_source, _) =
        table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None, &[])
            .unwrap();

    assert_eq!(groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
}

#[test]
fn table_format_timestamp_missing_source_contract_keeps_logical_timezone() {
    let timestamp_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into()));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", timestamp_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_row_count(1)
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "event_ts",
            None::<String>,
            timestamp_type.clone(),
            true,
            TableFormatColumnSource::MissingNullableColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let (native_source, _) =
        table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None, &[])
            .unwrap();

    assert_eq!(groups[0].source_schema.field_with_name("event_ts").unwrap().data_type(), &timestamp_type);
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
}

#[test]
fn table_format_local_footer_matching_field_id_admits_renamed_physical_column() {
    let dir = unique_table_format_temp_dir("field-id-rename");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "old_status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
    assert_eq!(groups[0].requirements.projected_output_columns, vec!["status"]);
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
    assert_eq!(groups[0].source_schema.field_with_name("old_status").unwrap().data_type(), &DataType::Int64);
    assert_eq!(groups[0].output_schema.field_with_name("status").unwrap().data_type(), &DataType::Int64);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_attached_remote_footer_schema_admits_field_id_mapping() {
    let footer_field = Field::new("old_status", DataType::Int64, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
}

#[test]
fn table_format_attached_remote_footer_same_name_signed_integer_widening_admits() {
    let footer_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int16, true)]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int32, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::all(),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let (native_source, _) = table_format_native_source_for_group(
        &scan_plan,
        &groups[0],
        None,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &[],
    )
    .unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["status"]);
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].data_type, DataType::Int16);
    assert_eq!(groups[0].source_schema.field_with_name("status").unwrap().data_type(), &DataType::Int32);
    assert!(groups[0].has_read_coercions());
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
}

#[test]
fn table_format_attached_remote_footer_same_name_unsafe_type_mismatch_rejects_instead_of_missing_null() {
    let footer_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Utf8, true)]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int32, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
        .expect_err("same-name Parquet columns with unsafe schema mismatch must not materialize as missing nulls");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_attached_remote_footer_absent_nullable_field_id_materializes_missing_null_column() {
    let footer_field = Field::new("other", DataType::Int64, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert!(groups[0].requirements.read_columns.is_empty());
    assert_eq!(groups[0].requirements.missing_columns.len(), 1);
    assert_eq!(groups[0].requirements.missing_columns[0].name, "status");
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::MissingNull(0)]);
}

#[test]
fn table_format_filter_only_same_name_signed_integer_widening_admits() {
    let footer_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int16, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int32, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&output_schema))
        .with_data_file(data_file)
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ));
    let scan_exec: Arc<dyn ExecutionPlan> = Arc::new(test_scan_exec(scan_plan.clone(), Arc::clone(&output_schema)));
    let scan_schema = scan_exec.schema();
    let predicate =
        binary(col("status", &scan_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &scan_schema)
            .unwrap();
    let filter = FilterExec::try_new(predicate, scan_exec).unwrap();
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        output_schema,
        Some(&filter),
        &RequiredFields::only(BTreeSet::new()),
    )
    .unwrap();

    assert_eq!(requirements.pure_filter_columns, vec!["status"]);
    assert_eq!(requirements.pure_filter_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let (native_source, _) = table_format_native_source_for_group(
        &scan_plan,
        &groups[0],
        Some(&filter),
        &file_schema,
        filter.schema(),
        None,
        &[],
    )
    .unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["status"]);
    assert_eq!(groups[0].requirements.predicate_columns, vec!["status"]);
    assert_eq!(groups[0].source_schema.field_with_name("status").unwrap().data_type(), &DataType::Int32);
    assert!(groups[0].has_read_coercions());
    assert!(native_source.reader_predicate.is_none());
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
}

#[test]
fn table_format_bare_boolean_filter_stays_post_decode() {
    let footer_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&output_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "id",
            Some("id"),
            DataType::Int32,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ))
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(2),
            "flag",
            Some("flag"),
            DataType::Boolean,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ));
    let scan_exec = Arc::new(test_scan_exec(scan_plan.clone(), Arc::clone(&output_schema)));
    let scan_plan_exec: Arc<dyn ExecutionPlan> = scan_exec.clone();
    let scan_schema = scan_plan_exec.schema();
    let predicate = col("flag", &scan_schema).unwrap();
    let filter = FilterExec::try_new(predicate, scan_plan_exec).unwrap();
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        output_schema,
        Some(&filter),
        &RequiredFields::all(),
    )
    .unwrap();
    let source_output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, Arc::clone(&source_output_schema)).unwrap();
    let (native_source, reader_pushdown_covers_filter) = table_format_native_source_for_group(
        &scan_plan,
        &groups[0],
        Some(&filter),
        &file_schema,
        filter.schema(),
        None,
        &[],
    )
    .unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["id", "flag"]);
    assert_eq!(groups[0].requirements.predicate_columns, vec!["flag"]);
    assert!(native_source.reader_predicate.is_none());
    assert!(!reader_pushdown_covers_filter);

    let mut builder = NativeQueryPlanBuilder::new();
    let lowered =
        lower_table_format_source_input(scan_exec.as_ref(), Some(&filter), &mut builder, &RequiredFields::all(), None)
            .unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);

    assert!(
        plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Filter(_))),
        "bare boolean table-format filters must execute post-decode"
    );
    assert!(plan.nodes().iter().any(|node| {
        matches!(&node.kind, PlanNode::Source(source) if matches!(
            source.as_ref(),
            NativeSourceNode::LocalParquet(source) if source.reader_predicate.is_none()
        ))
    }));
}

#[test]
fn table_format_filter_only_same_name_unsafe_type_mismatch_rejects_instead_of_missing_null() {
    let footer_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Utf8, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int32, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&output_schema))
        .with_data_file(data_file)
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ));
    let scan_exec: Arc<dyn ExecutionPlan> = Arc::new(test_scan_exec(scan_plan.clone(), Arc::clone(&output_schema)));
    let scan_schema = scan_exec.schema();
    let predicate =
        binary(col("status", &scan_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &scan_schema)
            .unwrap();
    let filter = FilterExec::try_new(predicate, scan_exec).unwrap();
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        output_schema,
        Some(&filter),
        &RequiredFields::only(BTreeSet::new()),
    )
    .unwrap();

    assert_eq!(requirements.pure_filter_columns, vec!["status"]);
    assert_eq!(requirements.pure_filter_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
        .expect_err("filter-only same-name unsafe mismatch must not become a missing-null predicate column");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_filter_only_absent_nullable_field_id_rejects_instead_of_missing_null() {
    let footer_field = Field::new("other", DataType::Int64, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&output_schema))
        .with_data_file(data_file)
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ));
    let scan_exec: Arc<dyn ExecutionPlan> = Arc::new(test_scan_exec(scan_plan.clone(), Arc::clone(&output_schema)));
    let scan_schema = scan_exec.schema();
    let predicate =
        binary(col("status", &scan_schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(0))), &scan_schema)
            .unwrap();
    let filter = FilterExec::try_new(predicate, scan_exec).unwrap();
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        output_schema,
        Some(&filter),
        &RequiredFields::only(BTreeSet::new()),
    )
    .unwrap();

    assert_eq!(requirements.predicate_columns, vec!["status"]);
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
        .expect_err("predicate fields absent from Iceberg Parquet footers must not become all-null filters");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_parent_predicate_absent_nullable_field_id_rejects_instead_of_missing_null() {
    let footer_field = Field::new("other", DataType::Int64, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let required = RequiredFields::only_with_predicates(BTreeSet::from([0]), BTreeSet::from([0]));
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &required).unwrap();

    assert_eq!(requirements.predicate_columns, vec!["status"]);
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
        .expect_err("parent predicate fields absent from Iceberg Parquet footers must not become all-null filters");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_field_id_mapping_allows_list_child_field_name_drift() {
    let footer_list_type = DataType::List(Arc::new(
        Field::new("element", DataType::Utf8, true)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "58".to_string())])),
    ));
    let logical_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let footer_field = Field::new("old_tags", footer_list_type, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", logical_list_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "tags",
            Some("tags"),
            logical_list_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["old_tags"]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "tags");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_tags");
}

#[test]
fn table_format_field_id_mapping_allows_fixed_size_list_child_field_name_drift() {
    let footer_list_type = DataType::FixedSizeList(
        Arc::new(
            Field::new("element", DataType::Int32, true)
                .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "58".to_string())])),
        ),
        2,
    );
    let logical_list_type = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2);
    let footer_field = Field::new("old_features", footer_list_type, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("features", logical_list_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "features",
            Some("features"),
            logical_list_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.read_columns, vec!["old_features"]);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "features");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_features");
}

#[test]
fn table_format_field_id_mapping_rejects_non_parquet_list_child_metadata_drift() {
    let footer_list_type =
        DataType::List(Arc::new(Field::new("element", DataType::Utf8, true).with_metadata(HashMap::from([
            (PARQUET_FIELD_ID_META_KEY.to_string(), "58".to_string()),
            ("extension".to_string(), "a".to_string()),
        ]))));
    let logical_list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let footer_field = Field::new("old_tags", footer_list_type, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", logical_list_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "tags",
            Some("tags"),
            logical_list_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_field_id_mapping_allows_required_footer_for_nullable_logical_field() {
    let footer_field = Field::new("old_status", DataType::Int64, false)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "status");
    assert_eq!(groups[0].field_id_bindings[0][0].physical_name, "old_status");
    assert!(!groups[0].field_id_bindings[0][0].nullable);
}

#[test]
fn table_format_field_id_mapping_rejects_nullable_footer_for_required_logical_field() {
    let footer_field = Field::new("old_status", DataType::Int64, true)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
}

#[test]
fn table_format_field_id_mapping_allows_timestamp_timezone_footer_normalization() {
    let footer_field = Field::new(
        "event_ts",
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("+00:00".into())),
        false,
    )
    .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "8".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let logical_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("event_ts", logical_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(8),
            "event_ts",
            Some("event_ts"),
            logical_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].field_id_bindings[0][0].logical_name, "event_ts");
}

#[test]
fn table_format_missing_nullable_list_column_retains_its_declared_type() {
    let footer_field = Field::new("other", DataType::Int64, false)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]));
    let footer_schema = Arc::new(Schema::new(vec![footer_field]));
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("tags", list_type.clone(), true)]));
    let data_file = TableFormatDataFile::new(
        "file-0",
        "s3://bucket/table/part.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::object_store("s3"),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .with_footer_schema(footer_schema);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "tags",
            Some("tags"),
            list_type.clone(),
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, Arc::clone(&output_schema)).unwrap();

    assert_eq!(requirements.read_columns, ["tags"]);
    assert!(requirements.missing_columns.is_empty());
    assert_eq!(output_schema.field(0).data_type(), &list_type);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].requirements.missing_columns.len(), 1);
    assert_eq!(groups[0].requirements.missing_columns[0].data_type, list_type);
}

#[test]
fn table_format_field_id_missing_nullable_list_executes_as_declared_typed_nulls() {
    let dir = unique_table_format_temp_dir("field-id-missing-list-execution");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "other", 1);
    let child = Arc::new(Field::new("logical_item", DataType::Boolean, true));
    let list_type = DataType::List(Arc::clone(&child));
    let scan_schema = Arc::new(Schema::new(vec![Field::new("flags", list_type.clone(), true)]));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(3),
            "flags",
            Some("physical_flags"),
            list_type,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();
    let lowered =
        lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);
    let context = ExecutionContext::from_shared_runtime(Arc::new(cudf::Runtime::new().unwrap())).unwrap();

    let outcome = execute_query_plan(&context, QueryExecutionRequest::new(&plan));
    let QueryOutput::HostArrow(batch) = outcome.result().unwrap() else {
        panic!("table-format list source should materialize host Arrow output");
    };
    assert_eq!(batch.schema(), scan_schema);
    assert_eq!(batch.num_rows(), 2);
    let flags = batch.column(0).as_any().downcast_ref::<arrow::array::ListArray>().unwrap();
    assert_eq!(flags.null_count(), 2);
    assert_eq!(flags.values().len(), 0);
    assert_eq!(flags.value_type(), DataType::Boolean);
    let DataType::List(field) = flags.data_type() else {
        panic!("table-format output should retain the declared List type");
    };
    assert_eq!(field.name(), "logical_item");
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_field_id_alias_and_position_deletes_lower_into_one_native_source() {
    let dir = unique_table_format_temp_dir("field-id-rename-with-delete");
    let data_path = dir.join("part.parquet");
    let delete_path = dir.join("delete.parquet");
    write_field_id_parquet_file(&data_path, "old_status", 2);
    write_field_id_parquet_file(&delete_path, "pos", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        delete_path.display().to_string(),
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::local_file_system())
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(delete_path.display().to_string()));
    let mut scan_plan = single_projected_file_scan_plan(&data_path, Arc::clone(&scan_schema), 2);
    scan_plan.data_files[0] = scan_plan.data_files[0].clone().with_position_delete_files(vec![delete_file.clone()]);
    scan_plan.delete_files.push(delete_file);
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered =
        lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);
    let source = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .expect("lowered plan should contain one local parquet source");

    assert_eq!(source.read_columns, vec!["old_status".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["status".to_string()]);
    assert_eq!(source.files[0].position_delete_files.len(), 1);
    assert_eq!(source.files[0].field_id_read_bindings.len(), 1);
    assert_eq!(source.files[0].field_id_read_bindings[0].logical_name, "status");
    assert_eq!(source.files[0].field_id_read_bindings[0].physical_name, "old_status");
    assert_eq!(source.diagnostics.schema_mapping_disposition.as_str(), "field_id_bindings");
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_equality_delete_hidden_key_rejects_without_delete_footer_field_id_proof() {
    let dir = unique_table_format_temp_dir("equality-delete-hidden-key");
    let data_path = dir.join("part.parquet");
    let delete_path = dir.join("delete.parquet");
    write_two_field_id_parquet_file(&data_path, ("id", 1), ("payload", 2));
    write_field_id_parquet_file(&delete_path, "id", 1);
    let scan_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        delete_path.display().to_string(),
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::EqualityDeletes,
    )
    .with_read_capability(TableFormatReadCapability::local_file_system())
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(delete_path.display().to_string()))
    .with_equality_ids([1]);
    let data_file = TableFormatDataFile::new(
        "file-0",
        data_path.display().to_string(),
        TableFormatDataFileFormat::Parquet,
        TableFormatReadCapability::local_file_system(),
    )
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(data_path.display().to_string()))
    .with_equality_delete_files(vec![delete_file.clone()]);
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(data_file)
        .with_delete_file(delete_file)
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "payload",
            Some("payload"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ))
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(1),
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ));
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let error = lower_table_format_source_input(
        &scan_exec,
        None,
        &mut builder,
        &RequiredFields::only(BTreeSet::from([1])),
        None,
    )
    .unwrap_err();

    assert_eq!(error, CapabilityReason::IcebergDeleteFilesUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_position_delete_scan_limit_is_pushed_with_upper_limit() {
    let dir = unique_table_format_temp_dir("position-delete-scan-limit");
    let data_path = dir.join("part.parquet");
    let delete_path = dir.join("delete.parquet");
    write_field_id_parquet_file(&data_path, "status", 2);
    write_field_id_parquet_file(&delete_path, "pos", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        delete_path.display().to_string(),
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::local_file_system())
    .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(delete_path.display().to_string()))
    .with_referenced_data_file(data_path.display().to_string())
    .with_position_bounds(Some(0), Some(10));
    let mut scan_plan =
        single_projected_file_scan_plan(&data_path, Arc::clone(&scan_schema), 2).with_scan_limit(Some(1));
    scan_plan.data_files[0] = scan_plan.data_files[0].clone().with_position_delete_files(vec![delete_file.clone()]);
    scan_plan.delete_files.push(delete_file);
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered =
        lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);
    let source = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .expect("lowered plan should contain one local parquet source");

    assert_eq!(source.scan_limit, Some(1));
    assert_eq!(
        source.files[0].position_delete_files[0].referenced_data_file.as_deref(),
        Some(data_path.display().to_string().as_str())
    );
    assert_eq!(source.files[0].position_delete_files[0].position_lower_bound, Some(0));
    assert_eq!(source.files[0].position_delete_files[0].position_upper_bound, Some(10));
    assert!(matches!(plan.root_node().kind, PlanNode::Limit(_)));
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_renamed_physical_columns_lower_as_union_groups() {
    let dir = unique_table_format_temp_dir("field-id-rename-union");
    let old_path = dir.join("part-old.parquet");
    let new_path = dir.join("part-new.parquet");
    write_field_id_parquet_file_with_nullability(&old_path, "old_status", 2, true);
    write_field_id_parquet_file_with_nullability(&new_path, "status", 2, true);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_data_file(
            TableFormatDataFile::new(
                "old-file",
                old_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(old_path.display().to_string())),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "new-file",
                new_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(new_path.display().to_string())),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered =
        lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);

    assert!(matches!(plan.root_node().kind, PlanNode::Union(_)));
    let sources = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 2);
    assert!(sources.iter().any(|source| source.read_columns == vec!["old_status".to_string()]));
    assert!(sources.iter().any(|source| source.read_columns == vec!["status".to_string()]));
    assert!(sources.iter().all(|source| source.output_schema().field_with_name("status").is_ok()));
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_multi_group_scan_limit_is_pushed_into_each_source() {
    let dir = unique_table_format_temp_dir("field-id-rename-union-limit");
    let old_path = dir.join("part-old.parquet");
    let new_path = dir.join("part-new.parquet");
    write_field_id_parquet_file_with_nullability(&old_path, "old_status", 2, true);
    write_field_id_parquet_file_with_nullability(&new_path, "status", 2, true);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&scan_schema))
        .with_native_source_schema(Arc::clone(&scan_schema))
        .with_scan_limit(Some(1))
        .with_data_file(
            TableFormatDataFile::new(
                "old-file",
                old_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(old_path.display().to_string())),
        )
        .with_data_file(
            TableFormatDataFile::new(
                "new-file",
                new_path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(new_path.display().to_string())),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "status",
            Some("status"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ));
    let scan_exec = test_scan_exec(scan_plan, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered =
        lower_table_format_source_input(&scan_exec, None, &mut builder, &RequiredFields::all(), None).unwrap();
    let plan = builder.finish_with_host_sink(lowered.node);
    let scan_limits = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source.scan_limit),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(scan_limits, vec![Some(1), Some(1)]);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_absent_nullable_field_id_materializes_missing_null_column() {
    let dir = unique_table_format_temp_dir("field-id-missing-null");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "other", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();

    assert_eq!(groups.len(), 1);
    assert!(groups[0].requirements.read_columns.is_empty());
    assert_eq!(groups[0].requirements.missing_columns.len(), 1);
    assert_eq!(groups[0].requirements.missing_columns[0].name, "status");
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::MissingNull(0)]);
    assert!(groups[0].output_schema.field_with_name("status").unwrap().is_nullable());
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_local_footer_absent_field_id_materializes_literal_default_column() {
    let dir = unique_table_format_temp_dir("field-id-missing-default");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "other", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_projected_file_scan_plan_with_default(
        &path,
        Arc::clone(&scan_schema),
        2,
        TableFormatScalarValue::Int64(7),
    );
    let file_schema = table_format_file_schema(&scan_plan);
    let requirements =
        table_format_column_requirements(&scan_plan, &file_schema, scan_schema, None, &RequiredFields::all()).unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let native_files = table_format_local_parquet_files_for_group(&scan_plan, &groups[0]).unwrap();

    assert_eq!(groups.len(), 1);
    assert!(groups[0].requirements.read_columns.is_empty());
    assert_eq!(groups[0].requirements.partition_columns.len(), 1);
    assert_eq!(groups[0].requirements.partition_columns[0].name, "status");
    assert_eq!(groups[0].requirements.projected_output_column_source_refs, vec![NativeSourceColumnRef::Partition(0)]);
    assert_eq!(native_files[0].partition_values, vec![NativeLiteralValue::Int64(7)]);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_rejects_field_id_mismatch_on_carrier_column() {
    let dir = unique_table_format_temp_dir("count-star-field-id-mismatch");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let mut builder = NativeQueryPlanBuilder::new();

    let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect_err("field-id mismatch must reject COUNT(*) fallback carrier column");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_rejects_iceberg_implicit_local_carrier_column() {
    let dir = unique_table_format_temp_dir("count-star-implicit-carrier");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_unmapped_file_scan_plan(&path, Arc::clone(&scan_schema));
    let mut builder = NativeQueryPlanBuilder::new();

    let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect_err("Iceberg COUNT(*) fallback must not synthesize a name-based carrier column");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_rejects_missing_nullable_carrier_without_probe_metadata() {
    let dir = unique_table_format_temp_dir("count-star-missing-carrier");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "other", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2).with_scan_limit(Some(1));
    let mut builder = NativeQueryPlanBuilder::new();

    let error = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect_err("COUNT(*) fallback must not treat missing/null materialization as row-presence proof");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_admits_matching_field_id_on_carrier_column() {
    let dir = unique_table_format_temp_dir("count-star-field-id-match");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect("matching field id should admit COUNT(*) fallback carrier column");

    assert_eq!(lowered.schema.field(0).name(), NATIVE_ZERO_COLUMN_COUNT_COLUMN);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_admits_same_name_carrier_without_field_id() {
    let dir = unique_table_format_temp_dir("count-star-field-id-missing-name-match");
    let path = dir.join("part.parquet");
    write_plain_parquet_file_with_nullability(&path, "status", false);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect("same-name carrier without a footer field id should admit COUNT(*) fallback");

    assert_eq!(lowered.schema.field(0).name(), NATIVE_ZERO_COLUMN_COUNT_COLUMN);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_count_star_fallback_admits_name_mapping_alias_carrier_without_field_id() {
    let dir = unique_table_format_temp_dir("count-star-field-id-missing-name-mapping-alias");
    let path = dir.join("part.parquet");
    write_plain_parquet_file_with_nullability(&path, "old_status", false);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let mut scan_plan = single_pure_filter_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    scan_plan.pure_filter_columns[0] =
        scan_plan.pure_filter_columns[0].clone().with_field_id_source_names(["old_status"]);
    let mut builder = NativeQueryPlanBuilder::new();

    let lowered = lower_table_format_count_star_fallback_source_for_plan(&scan_plan, &mut builder)
        .expect("name-mapping alias should admit COUNT(*) fallback carrier column");
    let plan = builder.finish_with_host_sink(lowered.node);
    let sources = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(lowered.schema.field(0).name(), NATIVE_ZERO_COLUMN_COUNT_COLUMN);
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].read_columns, vec!["old_status"]);
    assert_eq!(sources[0].pure_filter_columns, vec!["status"]);
    assert_eq!(sources[0].files[0].field_id_read_bindings[0].logical_name, "status");
    assert_eq!(sources[0].files[0].field_id_read_bindings[0].physical_name, "old_status");
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_empty_required_fields_uses_field_id_carrier_without_projecting_it() {
    let dir = unique_table_format_temp_dir("empty-required-field-id-carrier");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "old_status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_projected_file_scan_plan(&path, Arc::clone(&scan_schema), 2);
    let file_schema = table_format_file_schema(&scan_plan);

    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        Arc::clone(&scan_schema),
        None,
        &RequiredFields::only(BTreeSet::new()),
    )
    .unwrap();

    assert_eq!(requirements.read_columns, vec!["status"]);
    assert_eq!(requirements.read_column_file_indices, vec![0]);
    assert!(requirements.projected_output_columns.is_empty());
    assert_eq!(requirements.pure_filter_columns, vec!["status"]);
    assert_eq!(requirements.pure_filter_column_source_refs, vec![NativeSourceColumnRef::File(0)]);
    assert_eq!(requirements.scan_output_columns, vec!["status"]);
    assert!(requirements.requires_physical_row_presence);

    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();
    let groups = table_format_source_read_groups(&scan_plan, &requirements, output_schema).unwrap();
    let (native_source, _) =
        table_format_native_source_for_group(&scan_plan, &groups[0], None, &file_schema, scan_schema, None, &[])
            .unwrap();

    assert_eq!(groups[0].requirements.read_columns, vec!["old_status"]);
    assert!(groups[0].requirements.projected_output_columns.is_empty());
    assert_eq!(groups[0].requirements.pure_filter_columns, vec!["status"]);
    assert_eq!(native_source.read_columns, vec!["old_status"]);
    assert!(native_source.projected_output_columns.is_empty());
    assert_eq!(native_source.pure_filter_columns, vec!["status"]);
    source::validate_local_parquet_source(&native_source, query_engine::rapids_backend::rapids_analysis_context())
        .unwrap();
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_empty_required_fields_rejects_missing_field_id_carrier_column() {
    let dir = unique_table_format_temp_dir("empty-required-missing-carrier");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "other", 1);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, true)]));
    let scan_plan = single_projected_file_scan_plan_with_nullability(&path, Arc::clone(&scan_schema), 2, true);
    let file_schema = table_format_file_schema(&scan_plan);

    let requirements = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        scan_schema,
        None,
        &RequiredFields::only(BTreeSet::new()),
    )
    .unwrap();
    let output_schema = table_format_source_output_schema(&scan_plan, &requirements, &file_schema).unwrap();

    let error = table_format_source_read_groups(&scan_plan, &requirements, output_schema)
        .expect_err("row-presence carrier must remain backed by a physical file read");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

#[test]
fn table_format_empty_required_fields_rejects_iceberg_implicit_local_carrier_column() {
    let dir = unique_table_format_temp_dir("empty-required-implicit-carrier");
    let path = dir.join("part.parquet");
    write_field_id_parquet_file(&path, "status", 2);
    let scan_schema = Arc::new(Schema::new(vec![Field::new("status", DataType::Int64, false)]));
    let scan_plan = single_unmapped_file_scan_plan(&path, Arc::clone(&scan_schema));
    let file_schema = table_format_file_schema(&scan_plan);

    let error = table_format_column_requirements(
        &scan_plan,
        &file_schema,
        scan_schema,
        None,
        &RequiredFields::only(BTreeSet::new()),
    )
    .expect_err("Iceberg empty required fields must not synthesize a name-based carrier column");

    assert_eq!(error, CapabilityReason::IcebergFieldIdMappingUnsupported);
    fs::remove_dir_all(dir).ok();
}

fn single_projected_file_scan_plan(path: &Path, schema: SchemaRef, field_id: i32) -> TableFormatScanPlan {
    single_projected_file_scan_plan_with_nullability(path, schema, field_id, false)
}

fn single_projected_file_scan_plan_with_nullability(
    path: &Path,
    schema: SchemaRef,
    field_id: i32,
    nullable: bool,
) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(field_id),
            "status",
            Some("status"),
            DataType::Int64,
            nullable,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::Projected,
        ))
}

fn single_projected_file_scan_plan_with_default(
    path: &Path,
    schema: SchemaRef,
    field_id: i32,
    default_value: TableFormatScalarValue,
) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
        )
        .with_projected_column(
            TableFormatColumnMapping::new(
                Some(field_id),
                "status",
                Some("status"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                crate::table_format::TableFormatColumnRole::Projected,
            )
            .with_missing_default_value(default_value),
        )
}

fn single_unmapped_file_scan_plan(path: &Path, schema: SchemaRef) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
        )
}

fn single_pure_filter_file_scan_plan(path: &Path, schema: SchemaRef, field_id: i32) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "file-0",
                path.display().to_string(),
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_resolved_parquet_source(ResolvedParquetSource::local_kvikio(path.display().to_string())),
        )
        .with_pure_filter_column(TableFormatColumnMapping::new(
            Some(field_id),
            "status",
            Some("status"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            crate::table_format::TableFormatColumnRole::PureFilter,
        ))
}

fn write_field_id_parquet_file(path: &Path, column_name: &str, field_id: i32) {
    write_field_id_parquet_file_with_nullability(path, column_name, field_id, false);
}

fn write_field_id_parquet_file_with_nullability(path: &Path, column_name: &str, field_id: i32, nullable: bool) {
    let field = Field::new(column_name, DataType::Int64, nullable)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string())]));
    write_parquet_file_with_field(path, field);
}

fn write_plain_parquet_file_with_nullability(path: &Path, column_name: &str, nullable: bool) {
    write_parquet_file_with_field(path, Field::new(column_name, DataType::Int64, nullable));
}

fn write_parquet_file_with_field(path: &Path, field: Field) {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![10, 20]))]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_two_field_id_parquet_file(path: &Path, first: (&str, i32), second: (&str, i32)) {
    let fields = vec![
        Field::new(first.0, DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), first.1.to_string())])),
        Field::new(second.0, DataType::Int64, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), second.1.to_string())])),
    ];
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_table_format_temp_dir(label: &str) -> PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("algeon-datafusion-table-format-{label}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn table_format_file_row_count_does_not_create_synthetic_row_group_metadata() {
    let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
    scan_plan.data_files.push(
        TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(1024)
        .with_row_count(123)
        .with_partition_values(vec![TableFormatPartitionValue::new(
            "part",
            DataType::Int32,
            false,
            TableFormatPartitionTransform::Identity,
            TableFormatScalarValue::Int32(7),
        )])
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    );

    let files = table_format_local_parquet_files(&scan_plan, &test_column_requirements()).unwrap();

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].row_groups, None);
    assert_eq!(files[0].row_group_rows, None);
    assert_eq!(files[0].selected_row_group_count(), 0);
}

#[test]
fn table_format_position_deletes_preserve_contiguous_native_row_group_reads() {
    let delete_file = TableFormatDeleteFile::for_data_file(
        "delete-0",
        "file-0",
        "s3://bucket/table/delete-0.parquet",
        TableFormatDataFileFormat::Parquet,
        TableFormatDeleteFileContent::PositionDeletes,
    )
    .with_read_capability(TableFormatReadCapability::object_store("s3"))
    .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/delete-0.parquet",
        CredentialMode::SharedProcessEnv,
    ));
    let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
    scan_plan.data_files.push(
        TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(1024)
        .with_row_groups(vec![1, 2])
        .with_position_delete_files(vec![delete_file.clone()])
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    );
    scan_plan.delete_files.push(delete_file);

    let files = table_format_local_parquet_files(&scan_plan, &test_column_requirements()).unwrap();

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].row_groups, Some(vec![1, 2]));
    assert_eq!(files[0].selected_row_group_count(), 2);
    assert_eq!(files[0].position_delete_files.len(), 1);
}

#[test]
fn table_format_partition_only_file_row_count_creates_synthetic_row_group_metadata() {
    let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
    scan_plan.data_files.push(
        TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(1024)
        .with_row_count(123)
        .with_partition_values(vec![TableFormatPartitionValue::new(
            "part",
            DataType::Int32,
            false,
            TableFormatPartitionTransform::Identity,
            TableFormatScalarValue::Int32(7),
        )])
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    );
    let mut requirements = test_column_requirements();
    requirements.read_columns.clear();
    requirements.read_column_file_indices.clear();
    requirements.projected_output_columns = vec!["part".to_string()];
    requirements.projected_output_column_source_refs = vec![NativeSourceColumnRef::Partition(0)];
    requirements.partition_columns = vec![query_engine::source::PartitionColumn::new("part", DataType::Int32, false)];
    requirements.partition_column_indices = vec![0];
    requirements.scan_output_columns = vec!["part".to_string()];

    let files = table_format_local_parquet_files(&scan_plan, &requirements).unwrap();

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].row_group_rows.as_deref(), Some(&[123][..]));
}

#[test]
fn table_format_zero_column_file_row_count_creates_synthetic_row_group_metadata() {
    let mut scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg);
    scan_plan.data_files.push(
        TableFormatDataFile::new(
            "file-0",
            "s3://bucket/table/part-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_size_bytes(1024)
        .with_row_count(123)
        .with_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        )),
    );
    let mut requirements = test_column_requirements();
    requirements.read_columns.clear();
    requirements.read_column_file_indices.clear();
    requirements.projected_output_columns.clear();
    requirements.projected_output_column_source_refs.clear();
    requirements.pure_filter_columns.clear();
    requirements.pure_filter_column_source_refs.clear();
    requirements.partition_columns.clear();
    requirements.partition_column_indices.clear();
    requirements.scan_output_columns.clear();

    let files = table_format_local_parquet_files(&scan_plan, &requirements).unwrap();

    assert_eq!(files.len(), 1);
    assert_eq!(files[0].row_group_rows.as_deref(), Some(&[123][..]));
}
