use cudf::{
    DataType, ErrorCode, TypeId,
    io::source::{
        ByteRangeDisposition, LocalParquetFile, LocalParquetSource, LocalParquetSourceKind, PartitionColumn,
        PartitionColumnDisposition, PartitionValue, PredicateDisposition, RejectedSourceFeature,
        SchemaMappingDisposition, SourceColumn, SourceColumnRef, SourceColumnRole, SourceFormat, SourceOutputColumn,
        SourceRequiredColumn, SourceSchema, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
    },
};

fn dtype(id: TypeId) -> DataType {
    DataType::new(id)
}

fn base_schema() -> SourceSchema {
    SourceSchema::new([
        SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
        SourceColumn::new("payload", dtype(TypeId::String), true).unwrap(),
        SourceColumn::new("payload_alias", dtype(TypeId::String), true).unwrap(),
        SourceColumn::new("filter_only", dtype(TypeId::Int32), true).unwrap(),
        SourceColumn::new("region", dtype(TypeId::String), false).unwrap(),
    ])
    .unwrap()
}

fn partitioned_file() -> LocalParquetFile {
    LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::String("AMER".to_string())])
        .unwrap()
        .with_file_size_bytes(512)
        .with_row_groups(Some(vec![2, 4]))
}

#[test]
fn data_source_derives_diagnostics_and_read_column_refs_are_decoded_ordinals() {
    let source = LocalParquetSource::data(
        [partitioned_file()],
        base_schema(),
        ["id", "payload", "filter_only"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("region", SourceColumnRef::Partition(0)).unwrap(),
            SourceOutputColumn::new("payload_alias", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [SourceRequiredColumn::new("filter_only", SourceColumnRef::ReadColumn(2), SourceColumnRole::PureFilter)
            .unwrap()],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap();

    assert_eq!(source.kind(), LocalParquetSourceKind::Data);
    assert_eq!(source.read_columns(), ["id", "payload", "filter_only"]);
    assert_eq!(source.output_columns()[2].name(), "payload_alias");
    assert_eq!(source.output_columns()[2].source(), SourceColumnRef::ReadColumn(1));
    assert_eq!(source.read_columns()[1], "payload");

    let diagnostics = source.diagnostics();
    assert_eq!(diagnostics.format(), SourceFormat::Parquet);
    assert!(diagnostics.direct_device_read());
    assert_eq!(diagnostics.file_count(), 1);
    assert_eq!(diagnostics.ranged_file_count(), 0);
    assert_eq!(diagnostics.known_bytes(), Some(512));
    assert_eq!(diagnostics.selected_row_groups(), Some(2));
    assert_eq!(diagnostics.predicate_disposition(), PredicateDisposition::Absent);
    assert_eq!(diagnostics.byte_range_disposition(), ByteRangeDisposition::FullFiles);
    assert_eq!(diagnostics.partition_column_disposition(), PartitionColumnDisposition::Represented);
    assert_eq!(diagnostics.zero_column_count_disposition(), ZeroColumnCountDisposition::NotRequired);
    assert_eq!(diagnostics.schema_mapping_disposition(), SchemaMappingDisposition::DirectByName);
    assert_eq!(diagnostics.read_column_count(), 3);
    assert_eq!(diagnostics.output_column_count(), 3);
    assert_eq!(diagnostics.pure_filter_column_count(), 1);
    assert!(diagnostics.rejected_features().is_empty());
}

#[test]
fn source_validation_rejects_duplicate_output_names() {
    let err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
        ],
        [],
        [],
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.detail_value("reason"), Some("duplicate_output_column"));
}

#[test]
fn source_validation_rejects_read_partition_name_collisions() {
    let err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::Int32(7)]).unwrap()],
        SourceSchema::new([SourceColumn::new("region", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["region"],
        [SourceOutputColumn::new("region", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [PartitionColumn::new("region", dtype(TypeId::Int32), false).unwrap()],
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::DuplicateColumnName);
    assert_eq!(err.detail_value("reason"), Some("read_partition_name_collision"));
}

#[test]
fn source_validation_checks_partition_value_count_type_and_supported_dtype() {
    let count_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
        base_schema(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(count_err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(count_err.detail_value("reason"), Some("partition_value_count_mismatch"));

    let type_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::Boolean(true)]).unwrap()],
        base_schema(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(type_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(type_err.detail_value("reason"), Some("partition_value_type_mismatch"));

    let unsupported_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::String("2026-05-04".to_string())])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [PartitionColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(unsupported_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(unsupported_err.detail_value("reason"), Some("unsupported_partition_dtype"));
}

#[test]
fn source_validation_checks_schema_refs_and_source_ref_bounds() {
    let missing_schema_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("missing", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap_err();
    assert_eq!(missing_schema_err.code(), ErrorCode::MissingColumn);
    assert_eq!(missing_schema_err.detail_value("reason"), Some("schema_column_missing"));

    let index_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(1)).unwrap()],
        [],
        [],
    )
    .unwrap_err();
    assert_eq!(index_err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(index_err.detail_value("reason"), Some("read_column_ref_out_of_bounds"));
}

#[test]
fn local_files_are_local_and_sources_must_have_files() {
    let path_err = LocalParquetFile::new("s3://bucket/part-0.parquet", "part-0", []).unwrap_err();
    assert_eq!(path_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(path_err.detail_value("rejected_features"), Some(RejectedSourceFeature::NonLocalSource.as_str()));

    let empty_err = LocalParquetSource::data(
        [],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap_err();
    assert_eq!(empty_err.code(), ErrorCode::EmptyInput);
    assert_eq!(empty_err.detail_value("reason"), Some("empty_source_files"));
}

#[test]
fn zero_column_count_constructor_builds_only_the_synthetic_count_shape() {
    let source = LocalParquetSource::zero_column_count([LocalParquetFile::new("part-0.parquet", "part-0", [])
        .unwrap()
        .with_row_group_metadata(vec![128], vec![42])
        .unwrap()])
    .unwrap();

    assert_eq!(source.kind(), LocalParquetSourceKind::MetadataRowCount);
    assert!(source.read_columns().is_empty());
    assert!(source.partition_columns().is_empty());
    assert_eq!(source.schema().columns().len(), 1);
    assert_eq!(source.schema().columns()[0].name(), ZERO_COLUMN_COUNT_COLUMN);
    assert_eq!(source.schema().columns()[0].data_type(), dtype(TypeId::Int64));
    assert!(!source.schema().columns()[0].nullable());
    assert_eq!(source.output_columns().len(), 1);
    assert_eq!(source.output_columns()[0].name(), ZERO_COLUMN_COUNT_COLUMN);
    assert_eq!(source.output_columns()[0].source(), SourceColumnRef::ZeroColumnCount);

    let diagnostics = source.diagnostics();
    assert!(diagnostics.direct_device_read());
    assert_eq!(diagnostics.zero_column_count_disposition(), ZeroColumnCountDisposition::Represented);
    assert_eq!(diagnostics.read_column_count(), 0);
    assert_eq!(diagnostics.output_column_count(), 1);
}

#[test]
fn metadata_shapes_are_checked() {
    let metadata_err = LocalParquetFile::new("part-0.parquet", "part-0", [])
        .unwrap()
        .with_row_group_metadata(vec![100, 200], vec![10]);
    let metadata_err = metadata_err.unwrap_err();
    assert_eq!(metadata_err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(metadata_err.detail_value("reason"), Some("row_group_metadata_count_mismatch"));

    let count_err = LocalParquetSource::zero_column_count([LocalParquetFile::new(
        "part-0.parquet",
        "part-0",
        [PartitionValue::String("AMER".to_string())],
    )
    .unwrap()])
    .unwrap_err();
    assert_eq!(count_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(count_err.detail_value("reason"), Some("metadata_count_partition_values"));
}

#[test]
fn local_files_preserve_row_group_selection_shape() {
    let all_groups = LocalParquetFile::new("all.parquet", "all", []).unwrap();
    assert_eq!(all_groups.row_groups(), None);

    let empty_selection = LocalParquetFile::new("empty.parquet", "empty", []).unwrap().with_row_groups(Some(vec![]));
    assert_eq!(empty_selection.row_groups(), Some([].as_slice()));

    let selected_groups =
        LocalParquetFile::new("selected.parquet", "selected", []).unwrap().with_row_groups(Some(vec![2, 0]));
    assert_eq!(selected_groups.row_groups(), Some([2, 0].as_slice()));
}

#[test]
fn empty_data_read_projection_does_not_infer_metadata_count() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::String("AMER".to_string())]).unwrap()],
        SourceSchema::new([SourceColumn::new("region", dtype(TypeId::String), false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("region", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap();

    assert_eq!(source.kind(), LocalParquetSourceKind::Data);
    assert_eq!(source.diagnostics().zero_column_count_disposition(), ZeroColumnCountDisposition::NotRequired);
}

#[test]
fn rejected_source_feature_names_are_neutral_source_vocabulary() {
    let names = [
        RejectedSourceFeature::NonLocalSource.as_str(),
        RejectedSourceFeature::UnsupportedPartitionColumns.as_str(),
        RejectedSourceFeature::UnsupportedPredicateExecution.as_str(),
        RejectedSourceFeature::UnsupportedScanLimit.as_str(),
        RejectedSourceFeature::UnsupportedOrderingRequirement.as_str(),
        RejectedSourceFeature::EmptySource.as_str(),
        RejectedSourceFeature::UnsupportedPartitionValues.as_str(),
        RejectedSourceFeature::UnsupportedByteRange.as_str(),
        RejectedSourceFeature::UnsupportedRowSelection.as_str(),
        RejectedSourceFeature::UnsupportedZeroColumnCount.as_str(),
        RejectedSourceFeature::UnsupportedSchemaMapping.as_str(),
    ];

    assert_eq!(names[0], "non_local_source");
    assert!(!names.iter().any(|name| name.contains("datafusion")));
    assert!(!names.iter().any(|name| name.contains("parquet_access_plan")));
    assert!(!names.iter().any(|name| name.contains("expr_adapter")));
    assert!(!names.iter().any(|name| name.contains("preserve_order")));
}
