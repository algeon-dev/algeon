use super::*;

#[path = "source_contracts/chunk_planning.rs"]
mod chunk_planning;

#[test]
fn local_parquet_source_tracks_direct_source_diagnostics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/phase4.parquet")
                .with_stable_id("phase4:3235196281")
                .with_file_size_bytes(3_235_196_281)
                .with_row_groups(vec![0, 2, 4])
                .with_row_group_bytes(vec![10, 20, 30, 40, 50]),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);

    validate_local_parquet_source(&source).unwrap();
    assert_eq!(source.files[0].stable_id, "phase4:3235196281");
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 2);
    assert_eq!(source.read_column_file_indices, vec![0, 1]);
    assert_eq!(source.projected_output_column_count(), 1);
    assert_eq!(source.projected_output_column_source_refs, vec![SourceColumnRef::File(0)]);
    assert_eq!(source.pure_filter_column_count(), 1);
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1)]);
    assert_eq!(source.selected_row_group_count(), 3);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.known_bytes, Some(3_235_196_281));
    assert_eq!(source.diagnostics.selected_row_groups, Some(3));
    assert_eq!(source.diagnostics.read_column_count, 2);
    assert_eq!(source.diagnostics.projected_output_column_count, 1);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert_eq!(source.diagnostics.predicate_disposition, PredicateDisposition::Absent);
    assert_eq!(source.diagnostics.byte_range_disposition, ByteRangeDisposition::FullFiles);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Absent);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::NotRequired);
    assert_eq!(source.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::DirectByName);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_represents_partition_columns_as_synthetic_source_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_partition_values(vec![LiteralValue::Int32(7)])],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["key".to_string(), "part".to_string()], vec!["payload".to_string(), "part".to_string()]);

    validate_local_parquet_source(&source).unwrap();
    assert_eq!(source.partition_column_count(), 1);
    assert_eq!(source.read_columns, vec!["key".to_string(), "payload".to_string()]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::Partition(0)]
    );
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1), SourceColumnRef::Partition(0)]);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_rejects_invalid_partition_column_contracts() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet")],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&source).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_count_mismatch"));
    assert_eq!(error.fact_value("partition_columns"), Some("1"));
    assert_eq!(error.fact_value("partition_values"), Some("0"));

    let wrong_type = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_partition_values(vec![LiteralValue::Int64(7)])],
        schema,
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&wrong_type).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_type_mismatch"));
    assert_eq!(error.fact_value("partition_column"), Some("part"));
}

#[test]
fn local_parquet_source_rejects_null_partition_value_for_non_nullable_column() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/phase4.parquet")
                .with_partition_values(vec![LiteralValue::Null(DataType::Int32)]),
        ],
        schema,
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&source).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.local_parquet_source.validate");
    assert_eq!(error.fact_value("reason"), Some("non_nullable_partition_value_null"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("partition_index"), Some("0"));
    assert_eq!(error.fact_value("partition_column"), Some("part"));
    assert_eq!(error.fact_value("partition_type"), Some("Int32"));
    assert_eq!(error.fact_value("partition_nullable"), Some("false"));
    assert_eq!(error.fact_value("partition_value"), Some("NULL::Int32"));
}

#[test]
fn source_disposition_names_are_stable_manifest_values() {
    let dispositions = [
        (PredicateDisposition::Absent, "absent"),
        (PredicateDisposition::PruningOnly, "pruning_only"),
        (PredicateDisposition::Rejected, "rejected"),
    ];

    for (disposition, expected) in dispositions {
        assert_eq!(disposition.as_str(), expected);
    }

    assert_eq!(ByteRangeDisposition::FullFiles.as_str(), "full_files");
    assert_eq!(ByteRangeDisposition::CoalescedFullFiles.as_str(), "coalesced_full_files");
    assert_eq!(ByteRangeDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Absent.as_str(), "absent");
    assert_eq!(PartitionColumnDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Represented.as_str(), "represented");
    assert_eq!(ZeroColumnCountDisposition::NotRequired.as_str(), "not_required");
    assert_eq!(ZeroColumnCountDisposition::Rejected.as_str(), "rejected");
    assert_eq!(ZeroColumnCountDisposition::Represented.as_str(), "represented");
    assert_eq!(SchemaMappingDisposition::DirectByName.as_str(), "direct_by_name");
    assert_eq!(SchemaMappingDisposition::Rejected.as_str(), "rejected");
}

#[test]
fn local_parquet_source_validation_rejects_invalid_reader_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let empty_columns = LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), Vec::new());

    let error = validate_local_parquet_source(&empty_columns).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("empty_read_columns"));
    assert_eq!(empty_columns.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Rejected);
    assert!(empty_columns.diagnostics.rejected_features.iter().any(|feature| feature.as_str() == "zero_column_count"));

    let overflow_row_group = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_row_groups(vec![usize::MAX])],
        schema,
        vec!["key".to_string()],
    );

    let error = validate_local_parquet_source(&overflow_row_group).unwrap_err();
    let expected_row_group = usize::MAX.to_string();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("row_group"), Some(expected_row_group.as_str()));
}

#[test]
fn zero_column_count_source_contract_is_represented_without_data_columns() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/phase4.parquet")
            .with_stable_id("phase4:count")
            .with_file_size_bytes(300)
            .with_row_groups(vec![0, 2])
            .with_row_group_rows(vec![10, 20, 30]),
    ]);

    validate_local_parquet_source(&source).unwrap();
    assert!(source.is_zero_column_count_source());
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 0);
    assert_eq!(source.projected_output_column_count(), 0);
    assert_eq!(source.pure_filter_column_count(), 0);
    assert!(source.read_column_file_indices.is_empty());
    assert!(source.projected_output_column_source_refs.is_empty());
    assert!(source.pure_filter_column_source_refs.is_empty());
    assert!(source.schema.field_with_name(ZERO_COLUMN_COUNT_COLUMN).is_ok());
    assert_eq!(source.diagnostics.read_column_count, 0);
    assert_eq!(source.diagnostics.projected_output_column_count, 0);
    assert_eq!(source.diagnostics.pure_filter_column_count, 0);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_validation_rejects_unhonorable_source_facts_before_execution() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let base = LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["key".to_string()]);

    let mut rejected_byte_range = base.clone();
    rejected_byte_range.diagnostics.byte_range_disposition = ByteRangeDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_byte_range).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("unsupported_source_contract_fact"));
    assert_eq!(error.fact_value("field"), Some("byte_range"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut inconsistent_partition_columns = base.clone();
    inconsistent_partition_columns.diagnostics.partition_column_disposition = PartitionColumnDisposition::Represented;
    let error = validate_local_parquet_source(&inconsistent_partition_columns).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("empty_partition_columns"));

    let mut rejected_zero_column_count = base.clone();
    rejected_zero_column_count.diagnostics.zero_column_count_disposition = ZeroColumnCountDisposition::Represented;
    let error = validate_local_parquet_source(&rejected_zero_column_count).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("invalid_zero_column_count_columns"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let mut rejected_schema_mapping = base.clone();
    rejected_schema_mapping.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_schema_mapping).unwrap_err();
    assert_eq!(error.fact_value("field"), Some("schema_mapping"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut rejected_feature = base;
    rejected_feature.diagnostics.rejected_features.push(RejectedSourceFeature::RowSelection);
    let error = validate_local_parquet_source(&rejected_feature).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("row_selection"));
}

#[test]
fn local_parquet_source_validation_rejects_inconsistent_column_and_row_group_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let two_column_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::UInt32, false),
    ]));

    let mut missing_projected_read =
        LocalParquetSource::new("/tmp/phase4.parquet", two_column_schema, vec!["key".to_string()])
            .with_column_roles(vec!["payload".to_string()], Vec::new());
    missing_projected_read.diagnostics.rejected_features.clear();
    missing_projected_read.diagnostics.direct_device_read = true;
    missing_projected_read.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::DirectByName;
    missing_projected_read.projected_output_column_source_refs.push(SourceColumnRef::File(1));
    let error = validate_local_parquet_source(&missing_projected_read).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("projected_column_not_available"));
    assert_eq!(error.fact_value("column"), Some("payload"));

    let mut mismatched_diagnostics =
        LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["key".to_string()]);
    mismatched_diagnostics.diagnostics.read_column_count = 2;
    let error = validate_local_parquet_source(&mismatched_diagnostics).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("diagnostic_column_count_mismatch"));
    assert_eq!(error.fact_value("diagnostic_read_columns"), Some("2"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let missing_schema_mapping =
        LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["missing".to_string()]);
    assert_eq!(missing_schema_mapping.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::Rejected);
    let error = validate_local_parquet_source(&missing_schema_mapping).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("schema_mapping"));

    let row_group_outside_contract = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/phase4.parquet")
                .with_row_groups(vec![0, 2])
                .with_row_group_bytes(vec![100, 200]),
        ],
        schema,
        vec!["key".to_string()],
    );
    let error = validate_local_parquet_source(&row_group_outside_contract).unwrap_err();
    assert_eq!(error.fact_value("reason"), None);
    assert_eq!(error.fact_value("row_group"), Some("2"));
    assert_eq!(error.fact_value("max_row_group"), Some("1"));
}
