//! Source chunk planning for local Parquet scans.
//!
//! Chunk plans split one native source node into file or row-group ranges that
//! can be scheduled under a bounded memory policy. Byte fields are device-side
//! planning estimates unless explicitly named as compressed Parquet bytes or
//! exact source-read preflight bytes.

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    source::{LocalParquetFile, LocalParquetSource},
};

#[doc(hidden)]
pub mod assembly;
#[doc(hidden)]
pub mod decimal;
mod memory;
mod model;
mod planner;
mod policy;
#[doc(hidden)]
pub mod row_group;
#[doc(hidden)]
pub mod scan_limit;

#[doc(hidden)]
pub use decimal::object_store_decimal_reader_predicate_requires_footer_contract;
pub use memory::SourceTaskMemoryModel;
pub use model::{SourceChunk, SourceChunkBatch, SourceChunkPlan, SourceChunkRowPositionRange};
pub use planner::{annotate_source_chunk_batching_reasons, plan_source_chunk_batches};
pub use policy::{
    LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES,
    SourceChunkPlanningOptions, SourceChunkPlanningPolicy,
};

/// Returns true when the source can compute exact fixed-width source-read budgets.
pub fn local_parquet_source_has_exact_read_preflight(source: &LocalParquetSource) -> bool {
    source.files.iter().all(assembly::source_file_has_row_count_metadata)
        && assembly::exact_source_read_preflight_bytes(source, Some(1), source.is_zero_column_count_source()).is_some()
}

/// Returns the source file referenced by `chunk` after validating that the
/// chunk's executable identity still matches the file at `file_index`.
#[doc(hidden)]
pub fn source_file_for_chunk<'a>(
    source: &'a LocalParquetSource,
    chunk: &SourceChunk,
    operation: &'static str,
) -> Result<&'a LocalParquetFile> {
    let file = source.files.get(chunk.file_index).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "local parquet source chunk references a missing source file",
        )
        .with_fact("reason", "source_chunk_file_index_missing")
        .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
        .with_fact("stable_id", crate::source::parquet_source_stable_id_fact(&chunk.location, chunk.stable_id.as_str()))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("source_files", source.files.len())
    })?;
    validate_source_chunk_identity(source, file, chunk, operation)?;
    Ok(file)
}

/// Computes the file-global row-position range for a row-group chunk.
///
/// `None` means the chunk reads the whole file, so decoded row positions are
/// already file-global. Explicit row-group chunks require contiguous row groups
/// and per-row-group row counts.
#[doc(hidden)]
pub fn source_chunk_row_position_range(chunk: &SourceChunk) -> Result<Option<SourceChunkRowPositionRange>> {
    if chunk.reads_all_row_groups {
        return Ok(None);
    }
    if chunk.row_group_indexes.is_empty() {
        return Ok(Some(SourceChunkRowPositionRange { start: 0, end: 0 }));
    }
    let row_group_rows = chunk.source_file_row_group_rows.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.row_position_range",
            "Iceberg position-delete row-group chunks require per-row-group row-count metadata",
        )
        .with_fact("reason", "position_deletes_require_row_group_rows")
        .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
        .with_fact("file_index", chunk.file_index)
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("row_groups", source_chunk_row_groups_fact(chunk))
    })?;
    contiguous_row_group_position_range(&chunk.row_group_indexes, row_group_rows).map(Some).map_err(|error| {
        error
            .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("row_groups", source_chunk_row_groups_fact(chunk))
    })
}

#[doc(hidden)]
pub fn contiguous_row_group_position_range(
    row_group_indexes: &[usize],
    source_file_row_group_rows: &[u64],
) -> Result<SourceChunkRowPositionRange> {
    let Some(first_row_group) = row_group_indexes.first().copied() else {
        return Ok(SourceChunkRowPositionRange { start: 0, end: 0 });
    };
    let mut previous = first_row_group;
    if previous >= source_file_row_group_rows.len() {
        return Err(row_position_range_error(
            "position_delete_row_group_out_of_range",
            previous,
            source_file_row_group_rows.len(),
        ));
    }
    for row_group in row_group_indexes.iter().copied().skip(1) {
        if row_group >= source_file_row_group_rows.len() {
            return Err(row_position_range_error(
                "position_delete_row_group_out_of_range",
                row_group,
                source_file_row_group_rows.len(),
            ));
        }
        if previous.checked_add(1) != Some(row_group) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.row_position_range",
                "Iceberg position-delete row-group chunks must be contiguous",
            )
            .with_fact("reason", "position_delete_row_groups_not_contiguous")
            .with_fact("previous_row_group", previous)
            .with_fact("row_group", row_group));
        }
        previous = row_group;
    }

    let start = source_file_row_group_rows
        .iter()
        .take(first_row_group)
        .try_fold(0u64, |total, rows| total.checked_add(*rows))
        .ok_or_else(row_position_range_overflow_error)?;
    let selected_rows = row_group_indexes
        .iter()
        .try_fold(0u64, |total, row_group| {
            source_file_row_group_rows.get(*row_group).and_then(|rows| total.checked_add(*rows))
        })
        .ok_or_else(row_position_range_overflow_error)?;
    let end = start.checked_add(selected_rows).ok_or_else(row_position_range_overflow_error)?;
    Ok(SourceChunkRowPositionRange { start, end })
}

pub(super) fn row_position_range_error(reason: &'static str, row_group: usize, row_group_count: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk.row_position_range",
        "Iceberg position-delete row group is outside row-count metadata",
    )
    .with_fact("reason", reason)
    .with_fact("row_group", row_group)
    .with_fact("row_group_count", row_group_count)
}

pub(super) fn row_position_range_overflow_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk.row_position_range",
        "Iceberg position-delete row-position range exceeds native row-count range",
    )
    .with_fact("reason", "position_delete_row_position_range_overflow")
}

pub(super) fn source_chunk_row_groups_fact(chunk: &SourceChunk) -> String {
    chunk.row_group_indexes.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

pub(super) fn validate_source_chunk_identity(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    chunk: &SourceChunk,
    operation: &'static str,
) -> Result<()> {
    if chunk.location != file.location {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "location"));
    }
    if chunk.path != file.path {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "reader_path"));
    }
    if chunk.resolved_source != file.resolved_source {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "resolved_source"));
    }
    if chunk.stable_id != file.stable_id {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "stable_id"));
    }
    if chunk.partition_values != file.partition_values {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "partition_values"));
    }
    if chunk.read_columns != source.read_columns {
        return Err(source_chunk_identity_mismatch_error(file, chunk, operation, "read_columns"));
    }
    Ok(())
}

pub(super) fn source_chunk_identity_mismatch_error(
    file: &LocalParquetFile,
    chunk: &SourceChunk,
    operation: &'static str,
    field: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        operation,
        "local parquet source chunk identity does not match its referenced source file",
    )
    .with_fact("reason", "source_chunk_identity_mismatch")
    .with_fact("field", field)
    .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
    .with_fact("source_file_path", crate::source::parquet_file_path_fact(file))
    .with_fact("stable_id", crate::source::parquet_source_stable_id_fact(&chunk.location, chunk.stable_id.as_str()))
    .with_fact("source_file_stable_id", crate::source::parquet_file_stable_id_fact(file))
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("file_index", chunk.file_index)
}
