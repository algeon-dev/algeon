use std::{collections::BTreeMap, sync::Arc};

use cudf::{
    DataType, ErrorCode, TimestampResolution, TypeId,
    io::{
        parquet::{
            RemoteCredentialMode, RemoteParquetBackendBuildStatus, RemoteParquetCapabilityDisposition,
            RemoteParquetProfile, RemoteParquetSource, RemoteReadMode, probe_remote_parquet_datasource,
            remote_parquet_backend_build_status, remote_parquet_capability,
        },
        source::{
            ByteRangeDisposition, LocalParquetFile, LocalParquetSource, LocalParquetSourceKind,
            ParquetVariableWidthEncodingProof, ParquetVariableWidthPayloadEvidence, PartitionColumn,
            PartitionColumnDisposition, PartitionValue, PredicateDisposition, RejectedSourceFeature,
            SchemaMappingDisposition, SourceChunkPlanningPolicy, SourceColumn, SourceColumnRef, SourceColumnRole,
            SourceFormat, SourceOutputColumn, SourceRequiredColumn, SourceSchema, ZERO_COLUMN_COUNT_COLUMN,
            ZeroColumnCountDisposition, plan_local_parquet_source_chunks, source_row_count_to_i64,
            source_row_group_index_to_i32,
        },
    },
    ops::ast::AstExpression,
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
fn empty_variable_width_payload_evidence_propagates_through_source_descriptor() {
    let evidence = ParquetVariableWidthPayloadEvidence::new(1, Vec::new(), 0, ParquetVariableWidthEncodingProof::Empty);
    let file = LocalParquetFile::new("empty.parquet", "empty", [])
        .unwrap()
        .with_row_groups(Some(Vec::new()))
        .with_variable_width_payload_evidence(BTreeMap::from([(1, evidence.clone())]));
    let source = LocalParquetSource::data(
        [file],
        SourceSchema::new([SourceColumn::new("payload", dtype(TypeId::String), true).unwrap()]).unwrap(),
        ["payload"],
        [SourceOutputColumn::new("payload", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();

    let propagated = source.files()[0]
        .variable_width_payload_evidence()
        .get(&1)
        .expect("empty variable-width evidence propagates through the source descriptor");
    assert_eq!(propagated, &evidence);
    assert!(propagated.selected_row_groups.is_empty());
    assert_eq!(propagated.payload_upper_bound_bytes, 0);
    assert_eq!(propagated.encoding_proof, ParquetVariableWidthEncodingProof::Empty);
}

#[test]
fn predicate_disposition_names_cover_reader_pushdown_contract() {
    assert_eq!(PredicateDisposition::Absent.as_str(), "absent");
    assert_eq!(PredicateDisposition::PruningOnly.as_str(), "pruning_only");
    assert_eq!(PredicateDisposition::ReaderPushdown.as_str(), "reader_pushdown");
    assert_eq!(PredicateDisposition::NativePostScanOnly.as_str(), "native_post_scan_only");
    assert_eq!(PredicateDisposition::Rejected.as_str(), "rejected");
}

#[test]
fn data_source_reader_predicate_sets_reader_pushdown_diagnostics() {
    let predicate =
        AstExpression::equal(AstExpression::left_column(0).unwrap(), AstExpression::left_column(0).unwrap()).unwrap();
    let source = LocalParquetSource::data(
        [partitioned_file()],
        base_schema(),
        ["id", "payload", "filter_only"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("payload", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [SourceRequiredColumn::new("filter_only", SourceColumnRef::ReadColumn(2), SourceColumnRole::PureFilter)
            .unwrap()],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap()
    .with_reader_predicate(Arc::new(predicate))
    .unwrap();

    assert!(source.reader_predicate().is_some());
    assert_eq!(source.diagnostics().predicate_disposition(), PredicateDisposition::ReaderPushdown);
}

#[test]
fn data_source_represents_missing_nullable_columns_as_schema_mapping() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [])
            .unwrap()
            .with_row_group_metadata(vec![100], vec![3])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("missing_payload", dtype(TypeId::Int32), true).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("missing_payload", SourceColumnRef::MissingNull).unwrap(),
        ],
        [],
        [],
    )
    .unwrap();

    let diagnostics = source.diagnostics();
    assert!(diagnostics.direct_device_read());
    assert_eq!(diagnostics.schema_mapping_disposition(), SchemaMappingDisposition::MissingNullableColumns);
    assert_eq!(diagnostics.read_column_count(), 1);
    assert_eq!(diagnostics.output_column_count(), 2);
    assert!(diagnostics.rejected_features().is_empty());
}

#[test]
fn data_source_rejects_non_nullable_missing_null_columns() {
    let err = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("missing_payload", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("missing_payload", SourceColumnRef::MissingNull).unwrap(),
        ],
        [],
        [],
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.detail_value("reason"), Some("missing_null_for_non_nullable_column"));
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
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::Int64(7)]).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("unsupported", dtype(TypeId::List), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [PartitionColumn::new("unsupported", dtype(TypeId::List), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(unsupported_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(unsupported_err.detail_value("reason"), Some("unsupported_partition_dtype"));

    let fixed_point_err = DataType::fixed_point(TypeId::Int32, 7).unwrap_err();
    assert_eq!(fixed_point_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(fixed_point_err.operation(), "data_type::fixed_point");
    assert_eq!(fixed_point_err.detail_value("reason"), Some("fixed_point_requires_decimal_type"));

    let duration_days_err = LocalParquetSource::data(
        [LocalParquetFile::new(
            "part-0.parquet",
            "part-0",
            [PartitionValue::Duration { raw: 1, resolution: TimestampResolution::Days }],
        )
        .unwrap()],
        SourceSchema::new([SourceColumn::new("duration", dtype(TypeId::DurationDays), false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("duration", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("duration", dtype(TypeId::DurationDays), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(duration_days_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(duration_days_err.detail_value("reason"), Some("unsupported_partition_dtype"));

    let duration_days_value_err = LocalParquetSource::data(
        [LocalParquetFile::new(
            "part-0.parquet",
            "part-0",
            [PartitionValue::Duration { raw: 1, resolution: TimestampResolution::Days }],
        )
        .unwrap()],
        SourceSchema::new([SourceColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("duration", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(duration_days_value_err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(duration_days_value_err.detail_value("reason"), Some("unsupported_partition_value_dtype"));
    assert_eq!(duration_days_value_err.detail_value("actual_partition_dtype"), Some("duration_days"));
}

#[test]
fn source_validation_accepts_extended_partition_values_and_rejects_mismatches() {
    let decimal_dtype = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();
    let nullable_i32 = dtype(TypeId::Int32);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(
            "part-0.parquet",
            "part-0",
            [
                PartitionValue::Decimal128 { raw: 12345, scale: -2 },
                PartitionValue::Date32(19_481),
                PartitionValue::Timestamp { raw: 1_700_000_123_456, resolution: TimestampResolution::Microseconds },
                PartitionValue::Duration { raw: -42_000_000, resolution: TimestampResolution::Nanoseconds },
                PartitionValue::Null(nullable_i32),
            ],
        )
        .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("amount", decimal_dtype, false).unwrap(),
            SourceColumn::new("date", dtype(TypeId::TimestampDays), false).unwrap(),
            SourceColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap(),
            SourceColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap(),
            SourceColumn::new("maybe_id", nullable_i32, true).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("amount", SourceColumnRef::Partition(0)).unwrap(),
            SourceOutputColumn::new("date", SourceColumnRef::Partition(1)).unwrap(),
            SourceOutputColumn::new("ts", SourceColumnRef::Partition(2)).unwrap(),
            SourceOutputColumn::new("duration", SourceColumnRef::Partition(3)).unwrap(),
            SourceOutputColumn::new("maybe_id", SourceColumnRef::Partition(4)).unwrap(),
        ],
        [],
        [
            PartitionColumn::new("amount", decimal_dtype, false).unwrap(),
            PartitionColumn::new("date", dtype(TypeId::TimestampDays), false).unwrap(),
            PartitionColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap(),
            PartitionColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap(),
            PartitionColumn::new("maybe_id", nullable_i32, true).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(source.partition_columns().len(), 5);
    assert_eq!(source.files()[0].partition_values()[0].data_type(), decimal_dtype);
    assert_eq!(source.files()[0].partition_values()[1].data_type(), dtype(TypeId::TimestampDays));
    assert_eq!(source.files()[0].partition_values()[2].data_type(), dtype(TypeId::TimestampMicroseconds));
    assert_eq!(source.files()[0].partition_values()[3].data_type(), dtype(TypeId::DurationNanoseconds));
    assert_eq!(source.files()[0].partition_values()[4].data_type(), nullable_i32);

    let scale_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-1.parquet", "part-1", [PartitionValue::Decimal128 { raw: 12345, scale: -3 }])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("amount", decimal_dtype, false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("amount", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("amount", decimal_dtype, false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(scale_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(scale_err.detail_value("reason"), Some("partition_value_type_mismatch"));

    let nullability_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-2.parquet", "part-2", [PartitionValue::Null(nullable_i32)]).unwrap()],
        SourceSchema::new([SourceColumn::new("maybe_id", nullable_i32, false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("maybe_id", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("maybe_id", nullable_i32, false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(nullability_err.code(), ErrorCode::NullUnexpected);
    assert_eq!(nullability_err.detail_value("reason"), Some("partition_null_for_non_nullable_column"));

    let schema_truth_err = LocalParquetSource::data(
        [LocalParquetFile::new("part-2.parquet", "part-2", [PartitionValue::Null(nullable_i32)]).unwrap()],
        SourceSchema::new([SourceColumn::new("maybe_id", nullable_i32, false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("maybe_id", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("maybe_id", nullable_i32, true).unwrap()],
    )
    .unwrap_err();
    assert_eq!(schema_truth_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(schema_truth_err.detail_value("reason"), Some("source_ref_nullability_mismatch"));
    assert_eq!(schema_truth_err.detail_value("target_nullable"), Some("false"));
    assert_eq!(schema_truth_err.detail_value("source_nullable"), Some("true"));

    let temporal_err = LocalParquetSource::data(
        [LocalParquetFile::new(
            "part-3.parquet",
            "part-3",
            [PartitionValue::Timestamp { raw: 19_481, resolution: TimestampResolution::Days }],
        )
        .unwrap()],
        SourceSchema::new([SourceColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap()]).unwrap(),
        Vec::<String>::new(),
        [SourceOutputColumn::new("ts", SourceColumnRef::Partition(0)).unwrap()],
        [],
        [PartitionColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap()],
    )
    .unwrap_err();
    assert_eq!(temporal_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(temporal_err.detail_value("reason"), Some("partition_value_type_mismatch"));
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

    let s3_file = LocalParquetFile::object_store_uri("s3://bucket/part-0.parquet", "part-0", []).unwrap();
    assert_eq!(s3_file.location().read_capability(), "object_store:s3");
    assert_eq!(s3_file.reader_path(), "s3://bucket/part-0.parquet");
    assert_eq!(s3_file.path(), std::path::Path::new("s3://bucket/part-0.parquet"));
    assert_eq!(s3_file.remote_source().unwrap().uri(), "s3://bucket/part-0.parquet");

    let uppercase_s3_file = LocalParquetFile::object_store_uri("S3://bucket/part-0.parquet", "part-0-upper", [])
        .expect("S3 URI schemes are case-insensitive");
    assert_eq!(uppercase_s3_file.location().read_capability(), "object_store:s3");
    assert_eq!(uppercase_s3_file.remote_source().unwrap().uri(), "S3://bucket/part-0.parquet");
    assert_eq!(RemoteParquetSource::s3("S3://bucket/part-0.parquet").unwrap().uri(), "S3://bucket/part-0.parquet");

    let unsupported_uri = LocalParquetFile::object_store_uri("gs://bucket/part-0.parquet", "part-0", []).unwrap_err();
    assert_eq!(unsupported_uri.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(unsupported_uri.detail_value("reason"), Some("unsupported_object_store_scheme"));

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
fn object_store_source_chunks_preserve_reader_uri() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::object_store_uri("s3://bucket/part-0.parquet", "part-0", [])
            .unwrap()
            .with_file_size_bytes(512)],
        base_schema(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();

    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::default()).unwrap();
    let chunk = &plan.chunks()[0];

    assert_eq!(chunk.location().read_capability(), "object_store:s3");
    assert_eq!(chunk.reader_path(), "s3://bucket/part-0.parquet");
    assert_eq!(chunk.path(), std::path::Path::new("s3://bucket/part-0.parquet"));
    assert_eq!(chunk.remote_source().unwrap().uri(), "s3://bucket/part-0.parquet");
}

#[test]
fn typed_remote_parquet_source_reports_conservative_capability() {
    let source = RemoteParquetSource::s3("s3://bucket/part-0.parquet").unwrap();
    assert_eq!(source.uri(), "s3://bucket/part-0.parquet");

    let capability = remote_parquet_capability(&source);
    let expected_disposition = match capability.backend_build_status {
        RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            RemoteParquetCapabilityDisposition::PathReaderUnverified
        },
        RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {
            RemoteParquetCapabilityDisposition::KvikioRemoteDatasource
        },
        RemoteParquetBackendBuildStatus::Unknown => RemoteParquetCapabilityDisposition::PathReaderUnverified,
    };
    assert_eq!(capability.disposition, expected_disposition);
    assert_eq!(capability.uri_scheme, "s3");
    assert_eq!(capability.read_mode, RemoteReadMode::PreferDirect);
    assert_eq!(capability.credential_mode, RemoteCredentialMode::SharedProcessEnv);
    assert_eq!(capability.backend_build_status, remote_parquet_backend_build_status());
    assert_eq!(RemoteParquetBackendBuildStatus::Unknown.as_str(), "unknown");
    assert_eq!(RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled.as_str(), "kvikio_remote_io_disabled");
    assert_eq!(RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled.as_str(), "kvikio_remote_io_enabled");

    let required = source.clone().with_read_mode(RemoteReadMode::RequireDirect);
    let capability = remote_parquet_capability(&required);
    let expected_required_disposition = match capability.backend_build_status {
        RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            RemoteParquetCapabilityDisposition::BackendBuildDisabled
        },
        RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {
            RemoteParquetCapabilityDisposition::KvikioRemoteDatasource
        },
        RemoteParquetBackendBuildStatus::Unknown => RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable,
    };
    assert_eq!(capability.disposition, expected_required_disposition);
    assert_eq!(capability.read_mode, RemoteReadMode::RequireDirect);
    assert_eq!(capability.backend_build_status, remote_parquet_backend_build_status());

    let scoped = source.clone().with_profile(
        RemoteParquetProfile::shared_process_env().with_credential_mode(RemoteCredentialMode::ScopedProfile),
    );
    let capability = remote_parquet_capability(&scoped);
    let expected_scoped_disposition = match capability.backend_build_status {
        RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            RemoteParquetCapabilityDisposition::BackendBuildDisabled
        },
        RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {
            RemoteParquetCapabilityDisposition::KvikioRemoteDatasource
        },
        RemoteParquetBackendBuildStatus::Unknown => RemoteParquetCapabilityDisposition::CredentialModeUnsupported,
    };
    assert_eq!(capability.disposition, expected_scoped_disposition);
    assert_eq!(capability.credential_mode, RemoteCredentialMode::ScopedProfile);

    let unsupported = RemoteParquetSource::s3("gs://bucket/part-0.parquet").unwrap_err();
    assert_eq!(unsupported.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(unsupported.detail_value("reason"), Some("unsupported_uri_scheme"));

    let debug_profile = RemoteParquetProfile::shared_process_env()
        .with_profile_id("rest-s3")
        .with_endpoint("https://access:secret@example.invalid");
    let debug_source = source.with_profile(debug_profile);
    let debug = format!("{debug_source:?}");
    assert!(debug.contains("rest-s3"));
    assert!(debug.contains("<configured>"));
    assert!(!debug.contains("https://access"));
    assert!(!debug.contains("access:secret"));
    assert!(!debug.contains("secret"));
    assert!(!debug.contains("bucket/part-0.parquet"));
}

#[test]
fn remote_parquet_datasource_probe_rejects_virtual_hosted_custom_endpoint_before_open() {
    let source = RemoteParquetSource::s3("s3://bucket/part-0.parquet").unwrap().with_profile(
        RemoteParquetProfile::shared_process_env().with_endpoint("http://localhost:9000").with_path_style_access(false),
    );

    let error = probe_remote_parquet_datasource(&source, Some(4096))
        .expect_err("virtual-hosted custom endpoint must fail before libcudf datasource open");
    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("remote_direct_virtual_hosted_endpoint_unsupported"));
}

#[test]
fn remote_parquet_datasource_probe_rejects_secret_bearing_endpoint_without_echoing_it() {
    let source = RemoteParquetSource::s3("s3://bucket/part-0.parquet").unwrap().with_profile(
        RemoteParquetProfile::shared_process_env()
            .with_endpoint("https://access:secret@example.invalid")
            .with_path_style_access(true),
    );

    let error = probe_remote_parquet_datasource(&source, Some(4096))
        .expect_err("secret-bearing endpoint must fail before libcudf datasource open");
    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("endpoint_contains_secret_bearing_components"));
    let message = error.to_string();
    assert!(!message.contains("access:secret"));
    assert!(!message.contains("https://access"));
    assert!(!message.contains("example.invalid"));
}

#[test]
fn remote_parquet_datasource_probe_rejects_oversized_preferred_read_size_before_open() {
    let source = RemoteParquetSource::s3("s3://bucket/part-0.parquet").unwrap();

    let error = probe_remote_parquet_datasource(&source, Some(i64::MAX as u64 + 1))
        .expect_err("oversized preferred read size must fail before libcudf datasource open");
    assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error.detail_value("reason"), Some("preferred_read_size_exceeds_i64_max"));
}

#[test]
fn object_store_file_preserves_typed_remote_profile_and_mode() {
    let profile = RemoteParquetProfile::shared_process_env()
        .with_profile_id("lakehouse-prod")
        .with_region("us-west-2")
        .with_endpoint("https://s3.us-west-2.amazonaws.com")
        .with_path_style_access(true);
    let file = LocalParquetFile::object_store_uri("s3://bucket/part-0.parquet", "part-0", [])
        .unwrap()
        .with_remote_parquet_contract(profile, RemoteReadMode::AllowHostFallback)
        .unwrap();
    let source = file.remote_source().unwrap();

    assert_eq!(source.uri(), "s3://bucket/part-0.parquet");
    assert_eq!(source.read_mode(), RemoteReadMode::AllowHostFallback);
    assert_eq!(source.profile().profile_id(), Some("lakehouse-prod"));
    assert_eq!(source.profile().region(), Some("us-west-2"));
    assert_eq!(source.profile().path_style_access(), Some(true));
    assert_eq!(source.profile().credential_mode(), RemoteCredentialMode::SharedProcessEnv);

    let scoped_profile =
        RemoteParquetProfile::shared_process_env().with_credential_mode(RemoteCredentialMode::ScopedProfile);
    let scoped_file = LocalParquetFile::object_store_uri("s3://bucket/part-0.parquet", "part-0", [])
        .unwrap()
        .with_remote_parquet_contract(scoped_profile, RemoteReadMode::PreferDirect)
        .unwrap();
    assert_eq!(scoped_file.remote_source().unwrap().profile().credential_mode(), RemoteCredentialMode::ScopedProfile);

    let local_err = LocalParquetFile::new("part-0.parquet", "part-0", [])
        .unwrap()
        .with_remote_parquet_contract(RemoteParquetProfile::shared_process_env(), RemoteReadMode::PreferDirect)
        .unwrap_err();
    assert_eq!(local_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(local_err.detail_value("reason"), Some("non_remote_source"));
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

#[test]
fn chunk_planning_preserves_none_row_groups_as_all_without_metadata() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::String("AMER".to_string())])
            .unwrap()
            .with_file_size_bytes(512)],
        base_schema(),
        ["id", "payload"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("region", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap();

    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::default()).unwrap();

    assert_eq!(plan.chunks().len(), 1);
    assert!(!plan.row_group_metadata_available());
    assert_eq!(plan.metadata_unavailable_files(), 1);
    assert!(plan.size_estimate_based());

    let chunk = &plan.chunks()[0];
    assert_eq!(chunk.chunk_index(), 0);
    assert_eq!(chunk.file_index(), 0);
    assert_eq!(chunk.path(), std::path::Path::new("part-0.parquet"));
    assert_eq!(chunk.stable_id(), "part-0");
    assert!(chunk.reads_all_row_groups());
    assert!(chunk.row_group_indexes().is_empty());
    assert_eq!(chunk.known_file_bytes(), Some(512));
    assert_eq!(chunk.known_row_group_bytes(), None);
    assert_eq!(chunk.estimated_chunk_bytes(), Some(512));
    assert_eq!(chunk.read_columns(), ["id", "payload"]);
    assert_eq!(chunk.partition_values(), [PartitionValue::String("AMER".to_string())]);
    assert!(!chunk.zero_column_count());
    assert_eq!(chunk.known_row_count(), None);
}

#[test]
fn chunk_planning_policy_rejects_zero_limits() {
    let byte_err = SourceChunkPlanningPolicy::new(Some(0), 1).unwrap_err();
    assert_eq!(byte_err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(byte_err.detail_value("reason"), Some("max_source_chunk_bytes_zero"));

    let row_group_err = SourceChunkPlanningPolicy::new(None, 0).unwrap_err();
    assert_eq!(row_group_err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(row_group_err.detail_value("reason"), Some("max_row_groups_per_chunk_zero"));
}

#[test]
fn chunk_planning_skips_explicit_empty_row_group_selections_for_data_and_count() {
    let data_source = LocalParquetSource::data(
        [LocalParquetFile::new("empty.parquet", "empty", []).unwrap().with_row_groups(Some(vec![]))],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let data_plan = plan_local_parquet_source_chunks(&data_source, SourceChunkPlanningPolicy::default()).unwrap();
    assert!(data_plan.chunks().is_empty());
    assert_eq!(data_plan.known_total_row_count(), Some(0));
    assert_eq!(data_plan.known_total_row_count_i64().unwrap(), Some(0));

    let count_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("empty-count.parquet", "empty-count", [])
            .unwrap()
            .with_row_groups(Some(vec![]))])
        .unwrap();
    let count_plan = plan_local_parquet_source_chunks(&count_source, SourceChunkPlanningPolicy::default()).unwrap();
    assert!(count_plan.chunks().is_empty());
    assert_eq!(count_plan.known_total_row_count(), Some(0));
    assert_eq!(count_plan.known_total_row_count_i64().unwrap(), Some(0));
}

#[test]
fn chunk_planning_preserves_selected_row_group_order_and_file_facts() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "stable-part-0", [PartitionValue::Int32(7)])
            .unwrap()
            .with_file_size_bytes(900)
            .with_row_groups(Some(vec![2, 0]))
            .with_row_group_metadata(vec![100, 200, 300], vec![10, 20, 30])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("part", dtype(TypeId::Int32), false).unwrap()],
    )
    .unwrap();
    let policy = SourceChunkPlanningPolicy::new(None, 1).unwrap();

    let plan = plan_local_parquet_source_chunks(&source, policy).unwrap();

    assert_eq!(plan.chunks().len(), 2);
    assert!(plan.row_group_metadata_available());
    assert_eq!(plan.metadata_unavailable_files(), 0);
    assert!(!plan.size_estimate_based());
    assert_eq!(plan.known_total_row_count(), Some(40));
    assert_eq!(plan.known_total_row_count_i64().unwrap(), Some(40));

    assert_eq!(plan.chunks()[0].chunk_index(), 0);
    assert_eq!(plan.chunks()[0].file_index(), 0);
    assert_eq!(plan.chunks()[0].stable_id(), "stable-part-0");
    assert_eq!(plan.chunks()[0].row_group_indexes(), [2]);
    assert_eq!(plan.chunks()[0].row_group_indexes_i32().unwrap(), vec![2]);
    assert!(!plan.chunks()[0].reads_all_row_groups());
    assert_eq!(plan.chunks()[0].known_file_bytes(), Some(900));
    assert_eq!(plan.chunks()[0].known_row_group_bytes(), Some(300));
    assert_eq!(plan.chunks()[0].estimated_chunk_bytes(), Some(300));
    assert_eq!(plan.chunks()[0].known_row_count(), Some(30));
    assert_eq!(plan.chunks()[0].known_row_count_i64().unwrap(), Some(30));
    assert_eq!(plan.chunks()[0].read_columns(), ["id"]);
    assert_eq!(plan.chunks()[0].partition_values(), [PartitionValue::Int32(7)]);

    assert_eq!(plan.chunks()[1].chunk_index(), 1);
    assert_eq!(plan.chunks()[1].row_group_indexes(), [0]);
    assert_eq!(plan.chunks()[1].known_row_group_bytes(), Some(100));
    assert_eq!(plan.chunks()[1].estimated_chunk_bytes(), Some(100));
    assert_eq!(plan.chunks()[1].known_row_count(), Some(10));
}

#[test]
fn chunk_planning_expands_all_row_groups_and_honors_byte_policy_with_metadata() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("all.parquet", "all", [])
            .unwrap()
            .with_row_group_metadata(vec![100, 250, 100], vec![1, 2, 3])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let policy = SourceChunkPlanningPolicy::new(Some(300), 10).unwrap();

    let plan = plan_local_parquet_source_chunks(&source, policy).unwrap();

    assert_eq!(plan.chunks().len(), 3);
    assert!(plan.row_group_metadata_available());
    assert_eq!(plan.known_total_row_count(), Some(6));
    assert_eq!(plan.chunks()[0].row_group_indexes(), [0]);
    assert!(!plan.chunks()[0].reads_all_row_groups());
    assert_eq!(plan.chunks()[0].estimated_chunk_bytes(), Some(100));
    assert_eq!(plan.chunks()[1].row_group_indexes(), [1]);
    assert_eq!(plan.chunks()[1].estimated_chunk_bytes(), Some(250));
    assert_eq!(plan.chunks()[2].row_group_indexes(), [2]);
    assert_eq!(plan.chunks()[2].estimated_chunk_bytes(), Some(100));
}

#[test]
fn chunk_planning_uses_projected_uncompressed_bytes_before_read_bytes() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("projected.parquet", "projected", [])
            .unwrap()
            .with_row_group_metadata(vec![1_000, 1_000, 1_000, 1_000], vec![1, 1, 1, 1])
            .unwrap()
            .with_projected_row_group_bytes(vec![50, 50, 50, 50], vec![100, 100, 100, 100])
            .unwrap()
            .with_pure_filter_row_group_bytes(vec![100, 100, 100, 100], vec![400, 400, 400, 400])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("payload", dtype(TypeId::Int64), false).unwrap(),
        ])
        .unwrap(),
        ["id", "payload"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [SourceRequiredColumn::new("payload", SourceColumnRef::ReadColumn(1), SourceColumnRole::PureFilter).unwrap()],
        [],
    )
    .unwrap();
    let policy = SourceChunkPlanningPolicy::new(Some(250), 8).unwrap();

    let plan = plan_local_parquet_source_chunks(&source, policy).unwrap();

    assert_eq!(plan.chunks().len(), 2);
    assert_eq!(plan.chunks()[0].row_group_indexes(), [0, 1]);
    assert_eq!(plan.chunks()[0].known_row_group_bytes(), Some(2_000));
    assert_eq!(plan.chunks()[0].known_projected_compressed_bytes(), Some(100));
    assert_eq!(plan.chunks()[0].known_projected_uncompressed_bytes(), Some(200));
    assert_eq!(plan.chunks()[0].known_pure_filter_compressed_bytes(), Some(200));
    assert_eq!(plan.chunks()[0].known_pure_filter_uncompressed_bytes(), Some(800));
    assert_eq!(plan.chunks()[0].estimated_chunk_bytes(), Some(200));
    assert_eq!(plan.chunks()[0].estimated_source_read_bytes(), Some(1_000));
    assert_eq!(plan.chunks()[1].row_group_indexes(), [2, 3]);
}

#[test]
fn chunk_planning_validates_selected_row_groups_against_metadata() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [])
            .unwrap()
            .with_row_groups(Some(vec![2, 3]))
            .with_row_group_metadata(vec![100, 200, 300], vec![10, 20, 30])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();

    let err = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::default()).unwrap_err();

    assert_eq!(err.code(), ErrorCode::IndexOutOfBounds);
    assert_eq!(err.detail_value("reason"), Some("row_group_index_out_of_bounds"));
    assert_eq!(err.detail_value("stable_id"), Some("part-0"));
    assert_eq!(err.detail_value("file_index"), Some("0"));
    assert_eq!(err.detail_value("row_group"), Some("3"));
    assert_eq!(err.detail_value("row_group_position"), Some("1"));
    assert_eq!(err.detail_value("max_row_group"), Some("2"));
}

#[test]
fn zero_column_count_chunks_require_row_count_metadata_and_carry_known_counts() {
    let missing_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()])
            .unwrap();
    let err = plan_local_parquet_source_chunks(&missing_source, SourceChunkPlanningPolicy::default()).unwrap_err();
    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(err.detail_value("reason"), Some("missing_row_group_row_count_metadata"));
    assert_eq!(err.detail_value("stable_id"), Some("part-0"));

    let count_source = LocalParquetSource::zero_column_count([LocalParquetFile::new("part-1.parquet", "part-1", [])
        .unwrap()
        .with_row_groups(Some(vec![1, 0]))
        .with_row_group_metadata(vec![11, 22], vec![5, 7])
        .unwrap()])
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&count_source, SourceChunkPlanningPolicy::default()).unwrap();

    assert_eq!(plan.chunks().len(), 1);
    assert!(plan.chunks()[0].zero_column_count());
    assert!(plan.chunks()[0].read_columns().is_empty());
    assert_eq!(plan.chunks()[0].row_group_indexes(), [1, 0]);
    assert_eq!(plan.chunks()[0].known_row_count(), Some(12));
    assert_eq!(plan.known_total_row_count(), Some(12));
    assert_eq!(plan.known_total_row_count_i64().unwrap(), Some(12));
}

#[test]
fn row_group_and_row_count_execution_boundary_conversions_are_checked() {
    assert_eq!(source_row_group_index_to_i32(i32::MAX as usize).unwrap(), i32::MAX);
    let row_group_err = source_row_group_index_to_i32(i32::MAX as usize + 1).unwrap_err();
    assert_eq!(row_group_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(row_group_err.detail_value("reason"), Some("row_group_index_exceeds_i32"));

    assert_eq!(source_row_count_to_i64(i64::MAX as u64).unwrap(), i64::MAX);
    let row_count_err = source_row_count_to_i64(i64::MAX as u64 + 1).unwrap_err();
    assert_eq!(row_count_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(row_count_err.detail_value("reason"), Some("row_count_exceeds_i64"));

    let source = LocalParquetSource::data(
        [LocalParquetFile::new("large-rg.parquet", "large-rg", [])
            .unwrap()
            .with_row_groups(Some(vec![i32::MAX as usize + 1]))],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int64), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::default()).unwrap();
    let chunk_err = plan.chunks()[0].row_group_indexes_i32().unwrap_err();
    assert_eq!(chunk_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(chunk_err.detail_value("reason"), Some("row_group_index_exceeds_i32"));

    let count_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("huge-count.parquet", "huge-count", [])
            .unwrap()
            .with_row_group_metadata(vec![1], vec![i64::MAX as u64 + 1])
            .unwrap()])
        .unwrap();
    let count_err = plan_local_parquet_source_chunks(&count_source, SourceChunkPlanningPolicy::default()).unwrap_err();
    assert_eq!(count_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(count_err.detail_value("reason"), Some("row_count_exceeds_i64"));

    let total_overflow_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("overflow-count.parquet", "overflow-count", [])
            .unwrap()
            .with_row_group_metadata(vec![1, 1], vec![u64::MAX, 1])
            .unwrap()])
        .unwrap();
    let total_overflow_err =
        plan_local_parquet_source_chunks(&total_overflow_source, SourceChunkPlanningPolicy::default()).unwrap_err();
    assert_eq!(total_overflow_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(total_overflow_err.detail_value("reason"), Some("row_count_total_overflow"));
}
