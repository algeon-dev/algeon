//! Local Parquet source-planning contracts.
//!
//! This module describes checked, schema-aware local Parquet sources that can
//! be lowered from query planners before any GPU execution happens. The types
//! here are intentionally concrete: they describe local Parquet files,
//! projected decoded read columns, explicit partition constants, final output
//! columns, required execution-frame columns, and derived diagnostics.
//!
//! `LocalParquetSource` is not a generic file-source trait and is not an
//! Iceberg table-scan contract. Richer planners should lower into this type
//! only when the resulting scan is truthfully representable as local Parquet
//! files with explicit read-column mapping, nullable missing-column
//! materialization, and explicit metadata.
//!
//! Future source capabilities should be additive and explicitly named.
//! Predicate pruning, object stores, byte ranges, row-level selection, Iceberg
//! delete files, Iceberg field-id or name mapping, partition transforms, and
//! GDS/KvikIO direct-read policy all need their own source facts before callers
//! may rely on them. Local sources therefore reject or keep those planner facts
//! downstream instead of silently treating them as ordinary local full-file
//! reads.

use std::path::Path;

use arrow::datatypes::DataType as ArrowDataType;
use parquet::{
    basic::{ConvertedType, LogicalType, Type as ParquetPhysicalType},
    file::reader::{FileReader, SerializedFileReader},
    schema::types::ColumnDescriptor,
};

use crate::{
    core::{
        error::{Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_COLUMN_INDEX, FACT_EXPECTED_COUNT, FACT_TYPE_ID, Result},
        types::{DataType, TimestampResolution, TypeId},
    },
    hash::{FastHashSet, fast_hash_set_with_capacity},
    ops::ast::{AstExpression, AstOperator},
    runtime::Runtime,
};

/// Synthetic column name used by metadata-only zero-column count sources.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

const OP_FILE_NEW: &str = "io::source::LocalParquetFile::new";
const OP_FILE_METADATA: &str = "io::source::LocalParquetFile::with_row_group_metadata";
const OP_FILE_REMOTE_CONTRACT: &str = "io::source::LocalParquetFile::with_remote_parquet_contract";
const OP_SOURCE_DATA: &str = "io::source::LocalParquetSource::data";
const OP_SOURCE_COUNT: &str = "io::source::LocalParquetSource::zero_column_count";
const OP_SCHEMA_NEW: &str = "io::source::SourceSchema::new";
const OP_COLUMN_NEW: &str = "io::source::SourceColumn::new";
const OP_OUTPUT_NEW: &str = "io::source::SourceOutputColumn::new";
const OP_REQUIRED_NEW: &str = "io::source::SourceRequiredColumn::new";
const OP_PARTITION_NEW: &str = "io::source::PartitionColumn::new";
const OP_CHUNK_POLICY_NEW: &str = "io::source::SourceChunkPlanningPolicy::new";
const OP_PLAN_CHUNKS: &str = "io::source::plan_local_parquet_source_chunks";
const OP_ROW_GROUP_TO_I32: &str = "io::source::source_row_group_index_to_i32";
const OP_ROW_COUNT_TO_I64: &str = "io::source::source_row_count_to_i64";
const OP_SOURCE_READER_PREDICATE: &str = "io::source::build_local_parquet_reader_predicate";

mod chunk;
mod diagnostics;
mod local_parquet;
mod partition;
mod predicate;
mod schema;

pub use chunk::{
    SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceReadMode, plan_local_parquet_source_chunks,
    source_row_count_to_i64, source_row_group_index_to_i32,
};
pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    SchemaMappingDisposition, SourceDiagnostics, ZeroColumnCountDisposition,
};
pub use local_parquet::{
    LocalParquetFile, LocalParquetSource, LocalParquetSourceKind, ParquetReaderDecimalType, ParquetSourceLocation,
    ParquetVariableWidthEncodingProof, ParquetVariableWidthPayloadEvidence,
};
pub use partition::{PartitionColumn, PartitionValue};
pub use predicate::{
    SourceReaderPredicate, SourceReaderPredicateColumn, SourceReaderPredicateLiteral, SourceReaderPredicateOp,
    build_local_parquet_reader_batch_predicate, build_local_parquet_reader_predicate,
};
pub use schema::{
    SourceColumn, SourceColumnRef, SourceColumnRole, SourceOutputColumn, SourceRequiredColumn, SourceSchema,
};
fn checked_name(name: String, operation: &'static str, resource: &'static str) -> Result<String> {
    if name.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "name must not be empty")
            .with_detail("reason", "empty_name")
            .with_detail("resource", resource));
    }
    Ok(name)
}

fn validate_local_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file path must not be empty")
            .with_detail("reason", "empty_path")
            .with_detail("resource", "path"));
    }

    let path_string = path.to_string_lossy();
    if path_string.contains("://") {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "local Parquet sources require local filesystem paths",
        )
        .with_detail("reason", "non_local_source")
        .with_detail("path", path_string.into_owned())
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    }
    Ok(())
}

fn validate_object_store_uri(uri: &str) -> Result<()> {
    if uri.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "object-store URI must not be empty")
            .with_detail("reason", "empty_object_store_uri")
            .with_detail("resource", "location"));
    }

    let Some(scheme) = uri_scheme(uri) else {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "object-store Parquet sources require an explicit URI scheme",
        )
        .with_detail("reason", "missing_object_store_scheme")
        .with_detail("path", uri)
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    };

    if !scheme.eq_ignore_ascii_case("s3") {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "object-store Parquet source scheme is not supported",
        )
        .with_detail("reason", "unsupported_object_store_scheme")
        .with_detail("scheme", scheme.to_ascii_lowercase())
        .with_detail("path", uri)
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    }

    Ok(())
}

fn uri_scheme(value: &str) -> Option<&str> {
    value.split_once("://").map(|(scheme, _)| scheme).filter(|scheme| !scheme.is_empty())
}

fn validate_unique_strings<'a>(
    names: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
    resource: &'static str,
) -> Result<()> {
    let names = names.into_iter();
    let mut seen = fast_hash_set_with_capacity(names.size_hint().0);
    for (index, name) in names.enumerate() {
        if name.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, operation, "column name must not be empty")
                .with_detail("reason", "empty_name")
                .with_detail("resource", resource)
                .with_detail("position", index.to_string()));
        }
        if !seen.insert(name) {
            return Err(Error::new_with(ErrorCode::DuplicateColumnName, operation, "duplicate read column name")
                .with_column(name.to_string())
                .with_detail("reason", "duplicate_read_column")
                .with_detail("resource", resource)
                .with_detail("position", index.to_string()));
        }
    }
    Ok(())
}

fn validate_unique_named<'a>(
    names: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
    reason: &'static str,
) -> Result<()> {
    let names = names.into_iter();
    let mut seen = fast_hash_set_with_capacity(names.size_hint().0);
    for (index, name) in names.enumerate() {
        if !seen.insert(name) {
            return Err(Error::new_with(ErrorCode::DuplicateColumnName, operation, "duplicate column name")
                .with_column(name.to_string())
                .with_detail("reason", reason)
                .with_detail("position", index.to_string()));
        }
    }
    Ok(())
}

fn validate_read_partition_name_collisions(
    read_columns: &[String],
    partition_columns: &[PartitionColumn],
    operation: &'static str,
) -> Result<()> {
    let read_names: FastHashSet<&str> = read_columns.iter().map(String::as_str).collect();
    for (partition_index, column) in partition_columns.iter().enumerate() {
        if read_names.contains(column.name.as_str()) {
            return Err(Error::new_with(
                ErrorCode::DuplicateColumnName,
                operation,
                "partition column name also appears in read columns",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "read_partition_name_collision")
            .with_detail("partition_index", partition_index.to_string())
            .with_detail("partition_column", column.name.clone()));
        }
    }
    Ok(())
}

fn schema_missing_error(operation: &'static str, name: &str) -> Error {
    Error::new_with(ErrorCode::MissingColumn, operation, "source schema does not contain required column")
        .with_column(name.to_string())
        .with_detail("reason", "schema_column_missing")
}

fn validate_matching_row_group_metadata_lengths(
    operation: &'static str,
    stable_id: &str,
    path: &Path,
    reason: &'static str,
    expected_count: usize,
    actual_count: usize,
) -> Result<()> {
    if expected_count != actual_count {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "row-group metadata vectors must have matching lengths",
        )
        .with_fact(FACT_ACTUAL_COUNT, actual_count)
        .with_fact(FACT_EXPECTED_COUNT, expected_count)
        .with_detail("reason", reason)
        .with_detail("stable_id", stable_id.to_string())
        .with_detail("path", path.display().to_string()));
    }
    Ok(())
}

fn source_chunk_mismatch_error(
    operation: &'static str,
    chunk: &SourceChunk,
    message: &'static str,
    reason: &'static str,
) -> Error {
    Error::new_with(ErrorCode::TypeMismatch, operation, message)
        .with_detail("reason", reason)
        .with_detail("chunk_index", chunk.chunk_index.to_string())
        .with_detail("file_index", chunk.file_index.to_string())
        .with_detail("stable_id", chunk.stable_id.clone())
        .with_detail("path", chunk.path.display().to_string())
}

fn is_supported_partition_dtype(data_type: DataType) -> bool {
    let type_id = data_type.id();
    matches!(
        type_id,
        TypeId::Bool8
            | TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::UInt8
            | TypeId::UInt16
            | TypeId::UInt32
            | TypeId::UInt64
            | TypeId::Float32
            | TypeId::Float64
            | TypeId::String
            | TypeId::Decimal128
            | TypeId::DurationSeconds
            | TypeId::DurationMilliseconds
            | TypeId::DurationMicroseconds
            | TypeId::DurationNanoseconds
    ) || type_id.is_timestamp()
}

fn is_supported_missing_null_dtype(data_type: DataType) -> bool {
    is_supported_partition_dtype(data_type) || matches!(data_type.id(), TypeId::Decimal32 | TypeId::Decimal64)
}

fn is_supported_missing_null_column(column: &SourceColumn) -> bool {
    is_supported_missing_null_dtype(column.data_type)
        || column.data_type.id() == TypeId::List && column.arrow_data_type.is_some()
}

fn partition_value_matches_dtype(value: &PartitionValue, data_type: DataType) -> bool {
    value.data_type() == data_type
}

fn split_row_groups_for_policy(
    selected_row_groups: &[usize],
    row_group_bytes: &[u64],
    policy: SourceChunkPlanningPolicy,
) -> Vec<Vec<usize>> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0u64;

    for &row_group in selected_row_groups {
        let group_bytes = row_group_bytes[row_group];
        let would_exceed_count = current.len() >= policy.max_row_groups_per_chunk;
        let would_exceed_bytes = policy
            .max_source_chunk_bytes
            .is_some_and(|max_bytes| !current.is_empty() && current_bytes.saturating_add(group_bytes) > max_bytes);

        if would_exceed_count || would_exceed_bytes {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 0;
        }

        current.push(row_group);
        current_bytes = current_bytes.saturating_add(group_bytes);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn validate_selected_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    row_group_count: usize,
) -> Result<()> {
    for (row_group_position, &row_group) in row_groups.iter().enumerate() {
        if row_group >= row_group_count {
            let mut err = Error::new_with(
                ErrorCode::IndexOutOfBounds,
                OP_PLAN_CHUNKS,
                "selected row group index is out of bounds for source metadata",
            )
            .with_fact(FACT_COLUMN_INDEX, row_group)
            .with_detail("reason", "row_group_index_out_of_bounds")
            .with_detail("file_index", file_index.to_string())
            .with_detail("stable_id", file.stable_id.clone())
            .with_detail("path", file.path.display().to_string())
            .with_detail("row_group", row_group.to_string())
            .with_detail("row_group_position", row_group_position.to_string());
            if row_group_count > 0 {
                err = err.with_detail("max_row_group", (row_group_count - 1).to_string());
            }
            return Err(err);
        }
    }
    Ok(())
}

fn validate_source_file_metadata_lengths(file: &LocalParquetFile, file_index: usize) -> Result<()> {
    let expected_count = [
        file.row_group_bytes.as_ref(),
        file.projected_row_group_compressed_bytes.as_ref(),
        file.projected_row_group_uncompressed_bytes.as_ref(),
        file.pure_filter_row_group_compressed_bytes.as_ref(),
        file.pure_filter_row_group_uncompressed_bytes.as_ref(),
        file.row_group_rows.as_ref(),
    ]
    .into_iter()
    .flatten()
    .next()
    .map(Vec::len);
    let Some(expected_count) = expected_count else {
        return Ok(());
    };

    for (field, len) in [
        ("row_group_bytes", file.row_group_bytes.as_ref().map(Vec::len)),
        ("projected_row_group_compressed_bytes", file.projected_row_group_compressed_bytes.as_ref().map(Vec::len)),
        ("projected_row_group_uncompressed_bytes", file.projected_row_group_uncompressed_bytes.as_ref().map(Vec::len)),
        ("pure_filter_row_group_compressed_bytes", file.pure_filter_row_group_compressed_bytes.as_ref().map(Vec::len)),
        (
            "pure_filter_row_group_uncompressed_bytes",
            file.pure_filter_row_group_uncompressed_bytes.as_ref().map(Vec::len),
        ),
        ("row_group_rows", file.row_group_rows.as_ref().map(Vec::len)),
    ]
    .into_iter()
    .chain(file.row_group_null_counts.values().map(|null_counts| ("row_group_null_counts", Some(null_counts.len()))))
    {
        if let Some(len) = len
            && len != expected_count
        {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                OP_PLAN_CHUNKS,
                "row-group metadata vectors must have matching lengths",
            )
            .with_fact(FACT_ACTUAL_COUNT, len)
            .with_fact(FACT_EXPECTED_COUNT, expected_count)
            .with_detail("reason", "row_group_metadata_count_mismatch")
            .with_detail("field", field.to_string())
            .with_detail("file_index", file_index.to_string())
            .with_detail("stable_id", file.stable_id.clone())
            .with_detail("path", file.path.display().to_string()));
        }
    }

    Ok(())
}

fn validate_chunk_row_groups_for_source_file(
    chunk: &SourceChunk,
    source_file: &LocalParquetFile,
    operation: &'static str,
) -> Result<()> {
    if chunk.reads_all_row_groups && !chunk.row_group_indexes.is_empty() {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk cannot both read all row groups and carry explicit row groups",
            "source_chunk_reads_all_with_explicit_row_groups",
        ));
    }

    if let Some(selected_row_groups) = source_file.row_groups.as_deref() {
        if chunk.reads_all_row_groups {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk reads all row groups but supplied source has an explicit row-group selection",
                "source_chunk_reads_all_for_selected_row_groups",
            )
            .with_detail("source_row_groups", format!("{:?}", selected_row_groups)));
        }

        for (row_group_position, row_group) in chunk.row_group_indexes.iter().copied().enumerate() {
            if !selected_row_groups.contains(&row_group) {
                return Err(source_chunk_mismatch_error(
                    operation,
                    chunk,
                    "source chunk row group is not selected by supplied source file",
                    "source_chunk_row_group_not_selected",
                )
                .with_fact(FACT_COLUMN_INDEX, row_group)
                .with_detail("row_group", row_group.to_string())
                .with_detail("row_group_position", row_group_position.to_string())
                .with_detail("source_row_groups", format!("{:?}", selected_row_groups)));
            }
        }
    }

    if let Some(row_group_count) = row_group_metadata_count(source_file) {
        for (row_group_position, row_group) in chunk.row_group_indexes.iter().copied().enumerate() {
            if row_group >= row_group_count {
                let mut err = Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    operation,
                    "source chunk row group index is out of bounds for supplied source file metadata",
                )
                .with_fact(FACT_COLUMN_INDEX, row_group)
                .with_detail("reason", "source_chunk_row_group_out_of_bounds")
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string())
                .with_detail("row_group", row_group.to_string())
                .with_detail("row_group_position", row_group_position.to_string());
                if row_group_count > 0 {
                    err = err.with_detail("max_row_group", (row_group_count - 1).to_string());
                }
                return Err(err);
            }
        }
    }

    Ok(())
}

fn validate_chunk_source_file_contract(
    chunk: &SourceChunk,
    source_file: &LocalParquetFile,
    operation: &'static str,
) -> Result<()> {
    if chunk.source_file_row_groups != source_file.row_groups {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from a different source file row-group selection",
            "source_chunk_source_row_groups_mismatch",
        )
        .with_detail("expected_source_row_groups", format!("{:?}", source_file.row_groups))
        .with_detail("actual_source_row_groups", format!("{:?}", chunk.source_file_row_groups)));
    }

    if chunk.source_file_row_group_bytes != source_file.row_group_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different source file row-group byte metadata",
            "source_chunk_row_group_bytes_metadata_mismatch",
        )
        .with_detail("expected_row_group_bytes", format!("{:?}", source_file.row_group_bytes))
        .with_detail("actual_row_group_bytes", format!("{:?}", chunk.source_file_row_group_bytes)));
    }

    if chunk.source_file_projected_compressed_bytes != source_file.projected_row_group_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different projected compressed byte metadata",
            "source_chunk_projected_compressed_metadata_mismatch",
        )
        .with_detail(
            "expected_projected_compressed_bytes",
            format!("{:?}", source_file.projected_row_group_compressed_bytes),
        )
        .with_detail(
            "actual_projected_compressed_bytes",
            format!("{:?}", chunk.source_file_projected_compressed_bytes),
        ));
    }

    if chunk.source_file_projected_uncompressed_bytes != source_file.projected_row_group_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different projected uncompressed byte metadata",
            "source_chunk_projected_uncompressed_metadata_mismatch",
        )
        .with_detail(
            "expected_projected_uncompressed_bytes",
            format!("{:?}", source_file.projected_row_group_uncompressed_bytes),
        )
        .with_detail(
            "actual_projected_uncompressed_bytes",
            format!("{:?}", chunk.source_file_projected_uncompressed_bytes),
        ));
    }

    if chunk.source_file_pure_filter_compressed_bytes != source_file.pure_filter_row_group_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different pure-filter compressed byte metadata",
            "source_chunk_pure_filter_compressed_metadata_mismatch",
        )
        .with_detail(
            "expected_pure_filter_compressed_bytes",
            format!("{:?}", source_file.pure_filter_row_group_compressed_bytes),
        )
        .with_detail(
            "actual_pure_filter_compressed_bytes",
            format!("{:?}", chunk.source_file_pure_filter_compressed_bytes),
        ));
    }

    if chunk.source_file_pure_filter_uncompressed_bytes != source_file.pure_filter_row_group_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different pure-filter uncompressed byte metadata",
            "source_chunk_pure_filter_uncompressed_metadata_mismatch",
        )
        .with_detail(
            "expected_pure_filter_uncompressed_bytes",
            format!("{:?}", source_file.pure_filter_row_group_uncompressed_bytes),
        )
        .with_detail(
            "actual_pure_filter_uncompressed_bytes",
            format!("{:?}", chunk.source_file_pure_filter_uncompressed_bytes),
        ));
    }

    if chunk.source_file_row_group_rows != source_file.row_group_rows {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different source file row-count metadata",
            "source_chunk_row_group_rows_metadata_mismatch",
        )
        .with_detail("expected_row_group_rows", format!("{:?}", source_file.row_group_rows))
        .with_detail("actual_row_group_rows", format!("{:?}", chunk.source_file_row_group_rows)));
    }

    if chunk.reader_decimal_types != source_file.reader_decimal_types {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different reader decimal type metadata",
            "source_chunk_reader_decimal_type_metadata_mismatch",
        )
        .with_detail("expected_reader_decimal_types", format!("{:?}", source_file.reader_decimal_types))
        .with_detail("actual_reader_decimal_types", format!("{:?}", chunk.reader_decimal_types)));
    }

    if chunk.known_file_bytes != source_file.file_size_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk file byte metadata does not match supplied source file",
            "source_chunk_file_bytes_metadata_mismatch",
        )
        .with_detail("expected_file_size_bytes", format!("{:?}", source_file.file_size_bytes))
        .with_detail("actual_file_size_bytes", format!("{:?}", chunk.known_file_bytes)));
    }

    let metadata_available = source_file.row_group_bytes.is_some() && source_file.row_group_rows.is_some();
    let selected_row_groups = chunk_metadata_row_groups(chunk, source_file);
    let expected_known_row_group_bytes = if metadata_available {
        checked_row_group_sum(
            source_file.row_group_bytes.as_deref().expect("metadata_available checked row_group_bytes"),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_row_group_bytes != expected_known_row_group_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk row-group byte metadata does not match supplied source file",
            "source_chunk_known_row_group_bytes_mismatch",
        )
        .with_detail("expected_known_row_group_bytes", format!("{expected_known_row_group_bytes:?}"))
        .with_detail("actual_known_row_group_bytes", format!("{:?}", chunk.known_row_group_bytes)));
    }

    let expected_known_projected_compressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.projected_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_projected_compressed_bytes != expected_known_projected_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk projected compressed byte metadata does not match supplied source file",
            "source_chunk_known_projected_compressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_projected_compressed_bytes",
            format!("{expected_known_projected_compressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_projected_compressed_bytes",
            format!("{:?}", chunk.known_projected_compressed_bytes),
        ));
    }

    let expected_known_projected_uncompressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.projected_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_projected_uncompressed_bytes != expected_known_projected_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk projected uncompressed byte metadata does not match supplied source file",
            "source_chunk_known_projected_uncompressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_projected_uncompressed_bytes",
            format!("{expected_known_projected_uncompressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_projected_uncompressed_bytes",
            format!("{:?}", chunk.known_projected_uncompressed_bytes),
        ));
    }

    let expected_known_pure_filter_compressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.pure_filter_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_pure_filter_compressed_bytes != expected_known_pure_filter_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk pure-filter compressed byte metadata does not match supplied source file",
            "source_chunk_known_pure_filter_compressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_pure_filter_compressed_bytes",
            format!("{expected_known_pure_filter_compressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_pure_filter_compressed_bytes",
            format!("{:?}", chunk.known_pure_filter_compressed_bytes),
        ));
    }

    let expected_known_pure_filter_uncompressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_pure_filter_uncompressed_bytes != expected_known_pure_filter_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk pure-filter uncompressed byte metadata does not match supplied source file",
            "source_chunk_known_pure_filter_uncompressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_pure_filter_uncompressed_bytes",
            format!("{expected_known_pure_filter_uncompressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_pure_filter_uncompressed_bytes",
            format!("{:?}", chunk.known_pure_filter_uncompressed_bytes),
        ));
    }

    let expected_estimated_chunk_bytes = if metadata_available {
        expected_known_projected_uncompressed_bytes.or(expected_known_row_group_bytes)
    } else {
        source_file.file_size_bytes
    };
    if chunk.estimated_chunk_bytes != expected_estimated_chunk_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk byte estimate does not match supplied source file metadata",
            "source_chunk_estimated_bytes_mismatch",
        )
        .with_detail("expected_estimated_chunk_bytes", format!("{expected_estimated_chunk_bytes:?}"))
        .with_detail("actual_estimated_chunk_bytes", format!("{:?}", chunk.estimated_chunk_bytes)));
    }

    let expected_estimated_source_read_bytes = if metadata_available {
        expected_known_projected_uncompressed_bytes
            .zip(expected_known_pure_filter_uncompressed_bytes)
            .and_then(|(projected, filter)| projected.checked_add(filter))
            .or(expected_known_row_group_bytes)
            .or(expected_estimated_chunk_bytes)
    } else {
        source_file.file_size_bytes
    };
    if chunk.estimated_source_read_bytes != expected_estimated_source_read_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk source-read byte estimate does not match supplied source file metadata",
            "source_chunk_estimated_source_read_bytes_mismatch",
        )
        .with_detail("expected_estimated_source_read_bytes", format!("{expected_estimated_source_read_bytes:?}"))
        .with_detail("actual_estimated_source_read_bytes", format!("{:?}", chunk.estimated_source_read_bytes)));
    }

    let expected_size_estimate_based = !metadata_available && source_file.file_size_bytes.is_some();
    if chunk.size_estimate_based != expected_size_estimate_based {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk size-estimate mode does not match supplied source file metadata",
            "source_chunk_size_estimate_based_mismatch",
        )
        .with_detail("expected_size_estimate_based", expected_size_estimate_based.to_string())
        .with_detail("actual_size_estimate_based", chunk.size_estimate_based.to_string()));
    }

    let expected_known_row_count = if metadata_available {
        checked_row_group_sum(
            source_file.row_group_rows.as_deref().expect("metadata_available checked row_group_rows"),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_row_count != expected_known_row_count {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk row-count metadata does not match supplied source file",
            "source_chunk_known_row_count_mismatch",
        )
        .with_detail("expected_known_row_count", format!("{expected_known_row_count:?}"))
        .with_detail("actual_known_row_count", format!("{:?}", chunk.known_row_count)));
    }

    Ok(())
}

fn chunk_metadata_row_groups(chunk: &SourceChunk, source_file: &LocalParquetFile) -> Vec<usize> {
    if !chunk.reads_all_row_groups {
        return chunk.row_group_indexes.clone();
    }

    chunk_metadata_row_groups_for_source_file(source_file)
}

fn chunk_metadata_row_groups_for_source_file(source_file: &LocalParquetFile) -> Vec<usize> {
    row_group_metadata_count(source_file).map_or_else(Vec::new, |count| (0..count).collect())
}

/// Row-group count taken from the first attached per-row-group metadata
/// vector, or `None` when the file carries no row-group metadata.
fn row_group_metadata_count(source_file: &LocalParquetFile) -> Option<usize> {
    source_file
        .row_group_rows
        .as_ref()
        .or(source_file.row_group_bytes.as_ref())
        .or(source_file.projected_row_group_uncompressed_bytes.as_ref())
        .or(source_file.projected_row_group_compressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_compressed_bytes.as_ref())
        .map(Vec::len)
}

/// Sums the selected row-group values; `None` on overflow or when a selected
/// index has no metadata entry.
fn checked_row_group_sum(values: &[u64], row_groups: &[usize]) -> Option<u64> {
    row_groups.iter().try_fold(0u64, |acc, &row_group| acc.checked_add(*values.get(row_group)?))
}

fn optional_checked_row_group_sum(values: Option<&[u64]>, row_groups: &[usize]) -> Option<u64> {
    checked_row_group_sum(values?, row_groups)
}

fn checked_row_group_row_count(
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    row_group_rows: &[u64],
    row_groups: &[usize],
) -> Result<u64> {
    let mut total = 0u64;
    for &row_group in row_groups {
        total = total.checked_add(row_group_rows[row_group]).ok_or_else(|| {
            Error::new_with(ErrorCode::UnsupportedOperation, OP_ROW_COUNT_TO_I64, "source row count exceeds i64 range")
                .with_detail("reason", "row_count_total_overflow")
                .with_detail("chunk_index", chunk_index.to_string())
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string())
        })?;
    }
    source_row_count_to_i64_with_context(total, Some((file, file_index, chunk_index)))?;
    Ok(total)
}

fn known_total_row_count(chunks: &[SourceChunk]) -> Result<Option<u64>> {
    let mut total = 0u64;
    for chunk in chunks {
        let Some(row_count) = chunk.known_row_count else {
            return Ok(None);
        };
        total = total.checked_add(row_count).ok_or_else(|| {
            Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_PLAN_CHUNKS,
                "planned source row count exceeds the supported i64 range",
            )
            .with_detail("reason", "row_count_exceeds_i64")
        })?;
    }
    source_row_count_to_i64_with_context(total, None)?;
    Ok(Some(total))
}

fn missing_row_group_row_count_metadata_error(file: &LocalParquetFile, file_index: usize) -> Error {
    Error::new_with(
        ErrorCode::ForeignInvalidArgument,
        OP_PLAN_CHUNKS,
        "metadata row-count source requires row-group row-count metadata",
    )
    .with_detail("reason", "missing_row_group_row_count_metadata")
    .with_detail("file_index", file_index.to_string())
    .with_detail("stable_id", file.stable_id.clone())
    .with_detail("path", file.path.display().to_string())
}

fn source_row_group_index_to_i32_with_context(
    row_group: usize,
    chunk: Option<&SourceChunk>,
    position: Option<usize>,
) -> Result<i32> {
    i32::try_from(row_group).map_err(|_| {
        let mut err = Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_ROW_GROUP_TO_I32,
            "row group index exceeds libcudf i32 range",
        )
        .with_detail("reason", "row_group_index_exceeds_i32")
        .with_detail("row_group", row_group.to_string());
        if let Some(position) = position {
            err = err.with_detail("row_group_position", position.to_string());
        }
        if let Some(chunk) = chunk {
            err = err
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string());
        }
        err
    })
}

fn source_row_count_to_i64_with_context(
    row_count: u64,
    context: Option<(&LocalParquetFile, usize, usize)>,
) -> Result<i64> {
    i64::try_from(row_count).map_err(|_| {
        let mut err =
            Error::new_with(ErrorCode::UnsupportedOperation, OP_ROW_COUNT_TO_I64, "source row count exceeds i64 range")
                .with_detail("reason", "row_count_exceeds_i64")
                .with_detail("row_count", row_count.to_string());
        if let Some((file, file_index, chunk_index)) = context {
            err = err
                .with_detail("chunk_index", chunk_index.to_string())
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string());
        }
        err
    })
}

fn dtype_label(data_type: DataType) -> String {
    if data_type.id().is_decimal() {
        format!("{}(scale={})", data_type.id(), data_type.scale())
    } else {
        data_type.id().to_string()
    }
}
