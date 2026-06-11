//! Source chunk planning for local Parquet scans.
//!
//! Chunk plans split one native source node into file or row-group ranges that
//! can be scheduled under a bounded memory policy. Byte fields are device-side
//! planning estimates unless explicitly named as compressed Parquet bytes or
//! exact source-read preflight bytes.

use std::{collections::BTreeSet, time::Duration};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::NodeId,
    runtime::ExecutionContext,
    source::{LocalParquetFile, LocalParquetSource, ParquetReaderDecimalType},
};

mod decimal;
mod memory;
mod metadata;
mod model;
mod planner;
mod policy;
mod row_group;

pub(crate) use decimal::object_store_decimal_reader_predicate_requires_footer_contract;
use decimal::{
    parquet_reader_decimal_types, reader_decimal_type_metadata_unavailable_error,
    validate_object_store_reader_decimal_types, validate_supplied_reader_decimal_types_match_footer,
};
pub use memory::SourceTaskMemoryModel;
pub use metadata::SourceMetadataCache;
use metadata::{FooterMetadataLoad, SourceMetadataLoadContext};
#[cfg(test)]
use metadata::{SourceMetadataCacheKey, local_parquet_footer_metadata_path, sidecar_content_identity};
pub use model::{SourceChunk, SourceChunkBatch, SourceChunkPlan, SourceChunkRowPositionRange};
pub use planner::{annotate_source_chunk_batching_reasons, plan_source_chunk_batches};
pub use policy::{
    LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES,
    SourceChunkPlanningPolicy,
};
use row_group::{RowGroupMetadata, row_group_metadata_from_footer, source_row_group_metadata};

/// Returns the source file referenced by `chunk` after validating that the
/// chunk's executable identity still matches the file at `file_index`.
pub(crate) fn source_file_for_chunk<'a>(
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
        .with_fact("path", source_chunk_path_fact(chunk))
        .with_fact("stable_id", chunk.location.diagnostic_stable_id(chunk.stable_id.as_str()))
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
pub(crate) fn source_chunk_row_position_range(chunk: &SourceChunk) -> Result<Option<SourceChunkRowPositionRange>> {
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
        .with_fact("path", source_chunk_path_fact(chunk))
        .with_fact("file_index", chunk.file_index)
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("row_groups", source_chunk_row_groups_fact(chunk))
    })?;
    contiguous_row_group_position_range(&chunk.row_group_indexes, row_group_rows).map(Some).map_err(|error| {
        error
            .with_fact("path", source_chunk_path_fact(chunk))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("row_groups", source_chunk_row_groups_fact(chunk))
    })
}

pub(crate) fn contiguous_row_group_position_range(
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

fn row_position_range_error(reason: &'static str, row_group: usize, row_group_count: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk.row_position_range",
        "Iceberg position-delete row group is outside row-count metadata",
    )
    .with_fact("reason", reason)
    .with_fact("row_group", row_group)
    .with_fact("row_group_count", row_group_count)
}

fn row_position_range_overflow_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk.row_position_range",
        "Iceberg position-delete row-position range exceeds native row-count range",
    )
    .with_fact("reason", "position_delete_row_position_range_overflow")
}

fn source_chunk_row_groups_fact(chunk: &SourceChunk) -> String {
    chunk.row_group_indexes.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn validate_source_chunk_identity(
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

fn source_chunk_identity_mismatch_error(
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
    .with_fact("path", source_chunk_path_fact(chunk))
    .with_fact("source_file_path", source_file_path_fact(file))
    .with_fact("stable_id", chunk.location.diagnostic_stable_id(chunk.stable_id.as_str()))
    .with_fact("source_file_stable_id", file.location.diagnostic_stable_id(file.stable_id.as_str()))
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("file_index", chunk.file_index)
}

fn source_chunk_path_fact(chunk: &SourceChunk) -> String {
    if chunk.location.is_local_path() {
        crate::source::redacted_path_diagnostic(&chunk.path)
    } else {
        chunk.location.diagnostic_location()
    }
}

fn source_file_path_fact(file: &LocalParquetFile) -> String {
    if file.location.is_local_path() {
        crate::source::redacted_path_diagnostic(&file.path)
    } else {
        file.location.diagnostic_location()
    }
}

/// Plans chunks for a local Parquet source using a fresh footer metadata cache.
pub fn plan_local_parquet_source_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
) -> Result<SourceChunkPlan> {
    let mut metadata_cache = SourceMetadataCache::default();
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        &mut metadata_cache,
        None,
    )
}

/// Plans chunks using the execution context to opportunistically range-read
/// object-store Parquet footers before falling back to conservative read-all
/// chunks.
pub fn plan_local_parquet_source_chunks_with_execution_context(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    context: &ExecutionContext,
) -> Result<SourceChunkPlan> {
    let mut metadata_cache = SourceMetadataCache::default();
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        &mut metadata_cache,
        Some(SourceMetadataLoadContext { execution_context: context }),
    )
}

/// Plans chunks for a local Parquet source using a caller-owned footer cache.
///
/// The cache is keyed by source path, stable id, and known file size so repeated
/// planning of the same fixture can avoid re-reading footers.
pub fn plan_local_parquet_source_chunks_with_metadata_cache(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    metadata_cache: &mut SourceMetadataCache,
) -> Result<SourceChunkPlan> {
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        metadata_cache,
        None,
    )
}

fn plan_local_parquet_source_chunks_with_metadata_cache_and_context(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    metadata_cache: &mut SourceMetadataCache,
    metadata_load_context: Option<SourceMetadataLoadContext<'_>>,
) -> Result<SourceChunkPlan> {
    validate_chunk_planning_source(source)?;
    validate_chunk_planning_policy(policy)?;

    let mut chunks = Vec::new();
    let mut saw_metadata = false;
    let mut saw_missing_metadata = false;
    let mut metadata_unavailable_files = 0;
    let mut metadata_load_time = Duration::ZERO;
    let mut metadata_load_count = 0u64;
    let metadata_prefetch =
        metadata_cache.prefetch_footer_metadata_for_source(source, metadata_load_context.as_ref())?;
    metadata_load_time += metadata_prefetch.load_time;
    metadata_load_count = metadata_load_count.saturating_add(metadata_prefetch.load_count);

    for (file_index, file) in source.files.iter().enumerate() {
        let metadata_load = read_parquet_metadata(source, file, metadata_cache, metadata_load_context.as_ref())?;
        metadata_load_time += metadata_load.load_time;
        metadata_load_count = metadata_load_count.saturating_add(metadata_load.load_count);
        let reader_decimal_types = metadata_load.reader_decimal_types;
        let reader_decimal_types_footer_verified = metadata_load.reader_decimal_types_footer_verified;
        let metadata = metadata_load.metadata;
        saw_metadata |= metadata.is_some();
        if metadata.is_none() {
            saw_missing_metadata = true;
            metadata_unavailable_files += 1;
        }

        let row_groups = planned_row_groups(file, file_index, metadata.as_ref())?;
        let Some(row_groups) = row_groups else {
            chunks.push(single_all_row_groups_chunk(
                source_node_id,
                source,
                file,
                file_index,
                chunks.len(),
                policy,
                &reader_decimal_types,
                reader_decimal_types_footer_verified,
            ));
            continue;
        };

        append_row_group_chunks(
            source_node_id,
            source,
            file,
            file_index,
            &row_groups,
            metadata.as_ref(),
            policy,
            &reader_decimal_types,
            reader_decimal_types_footer_verified,
            &mut chunks,
        );
    }

    chunks = apply_source_scan_limit_to_chunks(source, chunks)?;

    if source.is_zero_column_count_source()
        && let Some(chunk) = chunks.iter().find(|chunk| chunk.known_row_count.is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "zero-column count source requires parquet row-count metadata for each planned chunk",
        )
        .with_fact("reason", "row_count_metadata_unavailable")
        .with_fact("path", source_chunk_path_fact(chunk))
        .with_fact("file_index", chunk.file_index)
        .with_fact("chunk_index", chunk.chunk_index));
    }

    Ok(SourceChunkPlan {
        chunks,
        source_node_id,
        row_group_metadata_available: saw_metadata && !saw_missing_metadata,
        size_estimate_based: saw_missing_metadata,
        metadata_unavailable_files,
        metadata_load_time,
        metadata_load_count,
    })
}

/// Returns true when the source can compute exact fixed-width source-read budgets.
pub fn local_parquet_source_has_exact_read_preflight(source: &LocalParquetSource) -> bool {
    source.files.iter().all(source_file_has_row_count_metadata)
        && exact_source_read_preflight_bytes(source, Some(1), source.is_zero_column_count_source()).is_some()
}

fn apply_source_scan_limit_to_chunks(
    source: &LocalParquetSource,
    chunks: Vec<SourceChunk>,
) -> Result<Vec<SourceChunk>> {
    let Some(scan_limit) = source.scan_limit else {
        return Ok(chunks);
    };
    if source.files.iter().any(LocalParquetFile::has_position_delete_files) {
        // Raw-row prefix trimming can under-produce visible rows when the
        // prefix is deleted. Execution must apply deletes per chunk before
        // deciding whether a scan limit is satisfied.
        return Ok(chunks);
    }
    if chunks.is_empty() {
        return Ok(chunks);
    }

    let mut remaining = u64::try_from(scan_limit).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.scan_limit",
            "source scan limit exceeds native row-count range",
        )
        .with_fact("scan_limit", scan_limit)
    })?;
    let mut limited = Vec::new();
    for mut chunk in chunks {
        let known_row_count = chunk.known_row_count.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.scan_limit",
                "source scan limit requires row-count metadata for planned chunks",
            )
            .with_fact("path", source_chunk_path_fact(&chunk))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
        })?;

        if remaining == 0 {
            if limited.is_empty() {
                chunk.scan_limit_rows = Some(0);
                chunk.known_row_count = Some(0);
                chunk.exact_source_read_preflight_bytes =
                    exact_source_read_preflight_bytes(source, chunk.known_row_count, chunk.zero_column_count);
                limited.push(chunk);
            }
            break;
        }

        if known_row_count <= remaining {
            remaining = remaining.saturating_sub(known_row_count);
            limited.push(chunk);
            continue;
        }

        limited.push(scan_limited_chunk_prefix(source, chunk, remaining, scan_limit)?);
        break;
    }
    Ok(limited)
}

fn scan_limited_chunk_prefix(
    source: &LocalParquetSource,
    chunk: SourceChunk,
    keep_rows: u64,
    scan_limit: usize,
) -> Result<SourceChunk> {
    let keep_rows_usize = usize::try_from(keep_rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.scan_limit",
            "source scan limit chunk slice exceeds native row-count range",
        )
        .with_fact("scan_limit", scan_limit)
        .with_fact("remaining_rows", keep_rows)
    })?;
    if keep_rows == 0 {
        return Ok(rewrite_scan_limited_chunk_selection(source, chunk, Vec::new(), 0, 0, Some(0)));
    }

    let (selected_row_groups, selected_row_group_rows) = scan_limit_row_group_prefix(&chunk, keep_rows)?;
    Ok(rewrite_scan_limited_chunk_selection(
        source,
        chunk,
        selected_row_groups,
        selected_row_group_rows,
        keep_rows,
        Some(keep_rows_usize),
    ))
}

fn scan_limit_row_group_prefix(chunk: &SourceChunk, keep_rows: u64) -> Result<(Vec<usize>, u64)> {
    let row_group_rows = chunk.source_file_row_group_rows.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.scan_limit",
            "source scan limit requires per-row-group row-count metadata for chunk prefix trimming",
        )
        .with_fact("path", source_chunk_path_fact(chunk))
        .with_fact("file_index", chunk.file_index)
        .with_fact("chunk_index", chunk.chunk_index)
    })?;

    let mut selected = Vec::new();
    let mut selected_rows = 0u64;
    for row_group in &chunk.row_group_indexes {
        let row_count = row_group_rows.get(*row_group).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.scan_limit",
                "source scan limit row group is outside row-count metadata",
            )
            .with_fact("path", source_chunk_path_fact(chunk))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("row_group", row_group)
            .with_fact("row_group_count", row_group_rows.len())
        })?;
        selected.push(*row_group);
        selected_rows = selected_rows.checked_add(*row_count).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.scan_limit",
                "source scan limit row group row counts overflow native range",
            )
            .with_fact("path", source_chunk_path_fact(chunk))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
        })?;
        if selected_rows >= keep_rows {
            return Ok((selected, selected_rows));
        }
    }

    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.source_chunk.scan_limit",
        "planned chunk row groups do not cover the requested source scan limit",
    )
    .with_fact("path", source_chunk_path_fact(chunk))
    .with_fact("file_index", chunk.file_index)
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("requested_rows", keep_rows)
    .with_fact("selected_rows", selected_rows))
}

fn rewrite_scan_limited_chunk_selection(
    source: &LocalParquetSource,
    mut chunk: SourceChunk,
    selected_row_groups: Vec<usize>,
    selected_row_group_rows: u64,
    output_row_count: u64,
    scan_limit_rows: Option<usize>,
) -> SourceChunk {
    let known_row_group_bytes =
        sum_selected_row_group_values(chunk.source_file_row_group_bytes.as_ref(), &selected_row_groups);
    let known_projected_compressed_bytes =
        sum_selected_row_group_values(chunk.source_file_projected_compressed_bytes.as_ref(), &selected_row_groups);
    let known_projected_uncompressed_bytes =
        sum_selected_row_group_values(chunk.source_file_projected_uncompressed_bytes.as_ref(), &selected_row_groups);
    let known_pure_filter_compressed_bytes =
        sum_selected_row_group_values(chunk.source_file_pure_filter_compressed_bytes.as_ref(), &selected_row_groups);
    let known_pure_filter_uncompressed_bytes =
        sum_selected_row_group_values(chunk.source_file_pure_filter_uncompressed_bytes.as_ref(), &selected_row_groups);
    let estimated_chunk_bytes = known_projected_uncompressed_bytes
        .or(known_row_group_bytes)
        .or_else(|| scan_limited_proportional_file_bytes(&chunk, selected_row_groups.len()));
    let estimated_source_read_bytes = source_read_uncompressed_bytes(
        source,
        known_projected_uncompressed_bytes,
        known_pure_filter_uncompressed_bytes,
    )
    .or(known_row_group_bytes)
    .or(estimated_chunk_bytes);

    chunk.row_group_indexes = selected_row_groups;
    chunk.reads_all_row_groups = false;
    chunk.known_row_group_bytes = known_row_group_bytes;
    chunk.known_projected_compressed_bytes = known_projected_compressed_bytes;
    chunk.known_projected_uncompressed_bytes = known_projected_uncompressed_bytes;
    chunk.known_pure_filter_compressed_bytes = known_pure_filter_compressed_bytes;
    chunk.known_pure_filter_uncompressed_bytes = known_pure_filter_uncompressed_bytes;
    chunk.estimated_chunk_bytes = estimated_chunk_bytes;
    chunk.estimated_source_read_bytes = estimated_source_read_bytes;
    chunk.exact_source_read_preflight_bytes =
        exact_source_read_preflight_bytes(source, Some(selected_row_group_rows), chunk.zero_column_count);
    chunk.known_row_count = Some(output_row_count);
    chunk.scan_limit_rows = scan_limit_rows;
    chunk.size_estimate_based = chunk.known_row_group_bytes.is_none();
    chunk
}

fn sum_selected_row_group_values(values: Option<&Vec<u64>>, row_groups: &[usize]) -> Option<u64> {
    let values = values?;
    row_groups.iter().try_fold(0u64, |total, row_group| values.get(*row_group)?.checked_add(total))
}

fn scan_limited_proportional_file_bytes(chunk: &SourceChunk, selected_row_group_count: usize) -> Option<u64> {
    if selected_row_group_count == 0 {
        return Some(0);
    }
    proportional_bytes(chunk.known_file_bytes, selected_row_group_count, source_file_row_group_count_basis(chunk))
}

fn source_file_row_group_count_basis(chunk: &SourceChunk) -> usize {
    chunk
        .source_file_row_group_rows
        .as_ref()
        .or(chunk.source_file_row_group_bytes.as_ref())
        .or(chunk.source_file_projected_compressed_bytes.as_ref())
        .or(chunk.source_file_projected_uncompressed_bytes.as_ref())
        .or(chunk.source_file_pure_filter_compressed_bytes.as_ref())
        .or(chunk.source_file_pure_filter_uncompressed_bytes.as_ref())
        .map_or(chunk.row_group_indexes.len(), Vec::len)
        .max(1)
}

fn validate_chunk_planning_policy(policy: SourceChunkPlanningPolicy) -> Result<()> {
    if policy.max_row_groups_per_chunk == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.policy",
            "source chunk planning requires max_row_groups_per_chunk to be greater than zero",
        )
        .with_fact("field", "max_row_groups_per_chunk")
        .with_fact("reason", "zero"));
    }

    Ok(())
}

fn validate_chunk_planning_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "local parquet source must contain at least one file for chunk planning",
        )
        .with_fact("reason", "empty_files"));
    }

    if source.read_columns.is_empty() && !source.is_zero_column_count_source() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "local parquet source must name at least one required read column for chunk planning",
        )
        .with_fact("reason", "empty_read_columns"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "local parquet source path must not be empty for chunk planning",
            )
            .with_fact("file_index", file_index));
        }
        validate_file_row_group_metadata_lengths(file, file_index)?;
    }

    Ok(())
}

fn validate_file_row_group_metadata_lengths(file: &LocalParquetFile, file_index: usize) -> Result<()> {
    let expected_len = [
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
    let Some(expected_len) = expected_len else {
        return Ok(());
    };

    for (field, values) in [
        ("row_group_bytes", file.row_group_bytes.as_ref()),
        ("projected_row_group_compressed_bytes", file.projected_row_group_compressed_bytes.as_ref()),
        ("projected_row_group_uncompressed_bytes", file.projected_row_group_uncompressed_bytes.as_ref()),
        ("pure_filter_row_group_compressed_bytes", file.pure_filter_row_group_compressed_bytes.as_ref()),
        ("pure_filter_row_group_uncompressed_bytes", file.pure_filter_row_group_uncompressed_bytes.as_ref()),
        ("row_group_rows", file.row_group_rows.as_ref()),
    ] {
        if let Some(values) = values
            && values.len() != expected_len
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "local parquet source row-group metadata vectors must have matching lengths",
            )
            .with_fact("reason", "row_group_metadata_count_mismatch")
            .with_fact("file_index", file_index)
            .with_fact("field", field)
            .with_fact("expected_count", expected_len)
            .with_fact("actual_count", values.len()));
        }
    }
    Ok(())
}

struct SourceMetadataLoad {
    metadata: Option<RowGroupMetadata>,
    load_time: Duration,
    load_count: u64,
    reader_decimal_types: Vec<ParquetReaderDecimalType>,
    reader_decimal_types_footer_verified: bool,
}

fn read_parquet_metadata(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    metadata_cache: &mut SourceMetadataCache,
    metadata_load_context: Option<&SourceMetadataLoadContext<'_>>,
) -> Result<SourceMetadataLoad> {
    let needs_object_store_decimal_contract =
        object_store_decimal_reader_predicate_requires_footer_contract(source, file);
    let footer_load = if source_file_needs_footer_metadata(source, file) {
        metadata_cache.footer_metadata(file, metadata_load_context)?
    } else {
        FooterMetadataLoad { metadata: None, load_time: Duration::ZERO, load_count: 0 }
    };
    let footer_reader_decimal_types =
        footer_load.metadata.as_ref().map(|metadata| parquet_reader_decimal_types(metadata, source));
    let reader_decimal_types = if needs_object_store_decimal_contract {
        let Some(footer_reader_decimal_types) = footer_reader_decimal_types else {
            return Err(reader_decimal_type_metadata_unavailable_error(source, file));
        };
        validate_supplied_reader_decimal_types_match_footer(
            source,
            file,
            &file.reader_decimal_types,
            &footer_reader_decimal_types,
        )?;
        footer_reader_decimal_types
    } else if file.reader_decimal_types.is_empty() {
        footer_reader_decimal_types.unwrap_or_default()
    } else {
        file.reader_decimal_types.clone()
    };
    let reader_decimal_types_footer_verified = needs_object_store_decimal_contract;
    validate_object_store_reader_decimal_types(source, file, &reader_decimal_types)?;
    let footer_metadata =
        footer_load.metadata.as_ref().map(|metadata| row_group_metadata_from_footer(metadata, source));
    Ok(SourceMetadataLoad {
        metadata: source_row_group_metadata(file, footer_metadata),
        load_time: footer_load.load_time,
        load_count: footer_load.load_count,
        reader_decimal_types,
        reader_decimal_types_footer_verified,
    })
}

fn source_file_needs_footer_metadata(source: &LocalParquetSource, file: &LocalParquetFile) -> bool {
    file.row_group_bytes.is_none()
        || file.projected_row_group_compressed_bytes.is_none()
        || file.projected_row_group_uncompressed_bytes.is_none()
        || file.pure_filter_row_group_compressed_bytes.is_none()
        || file.pure_filter_row_group_uncompressed_bytes.is_none()
        || file.row_group_rows.is_none()
        || object_store_decimal_reader_predicate_requires_footer_contract(source, file)
        || (file.reader_decimal_types.is_empty()
            && source
                .reader_predicate
                .as_ref()
                .is_some_and(|predicate| source.reader_predicate_requires_source_contract(predicate)))
}

pub(crate) fn footer_verified_object_store_reader_decimal_types(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    footer_verified_object_store_reader_decimal_types_with_supplied(context, source, file, &file.reader_decimal_types)
}

pub(crate) fn footer_verified_object_store_reader_decimal_types_with_supplied(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    supplied_reader_decimal_types: &[ParquetReaderDecimalType],
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    let mut metadata_cache = SourceMetadataCache::default();
    footer_verified_object_store_reader_decimal_types_with_supplied_cache(
        context,
        source,
        file,
        supplied_reader_decimal_types,
        &mut metadata_cache,
    )
}

pub(crate) fn footer_verified_object_store_reader_decimal_types_with_supplied_cache(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    supplied_reader_decimal_types: &[ParquetReaderDecimalType],
    metadata_cache: &mut SourceMetadataCache,
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    if !object_store_decimal_reader_predicate_requires_footer_contract(source, file) {
        return Ok(None);
    }
    let file_with_supplied_types;
    let file = if supplied_reader_decimal_types == file.reader_decimal_types {
        file
    } else {
        file_with_supplied_types = file.clone().with_reader_decimal_types(supplied_reader_decimal_types.to_vec());
        &file_with_supplied_types
    };
    let metadata_load = read_parquet_metadata(
        source,
        file,
        metadata_cache,
        Some(&SourceMetadataLoadContext { execution_context: context }),
    )?;
    Ok(Some(metadata_load.reader_decimal_types))
}

fn planned_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    metadata: Option<&RowGroupMetadata>,
) -> Result<Option<Vec<usize>>> {
    if let Some(row_groups) = &file.row_groups {
        validate_selected_row_groups(file, file_index, row_groups, metadata)?;
        validate_position_delete_planned_row_groups(file, file_index, row_groups, metadata)?;
        return Ok(Some(row_groups.clone()));
    }

    let Some(metadata) = metadata else {
        return Ok(None);
    };

    let row_groups = (0..metadata.row_group_count()).collect::<Vec<_>>();
    validate_position_delete_planned_row_groups(file, file_index, &row_groups, Some(metadata))?;
    Ok(Some(row_groups))
}

fn validate_selected_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
) -> Result<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };

    for (row_group_position, row_group) in row_groups.iter().enumerate() {
        if *row_group >= metadata.row_group_count() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "selected parquet row group is outside file metadata",
            )
            .with_fact("path", file.location.diagnostic_location())
            .with_fact("stable_id", file.location.diagnostic_stable_id(file.stable_id.as_str()))
            .with_fact("file_index", file_index)
            .with_fact("row_group_position", row_group_position)
            .with_fact("row_group", row_group)
            .with_fact("max_row_group", metadata.row_group_count().saturating_sub(1)));
        }
    }

    Ok(())
}

fn validate_position_delete_planned_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
) -> Result<()> {
    if !file.has_position_delete_files() || row_groups.is_empty() {
        return Ok(());
    }
    let row_group_rows = metadata.and_then(RowGroupMetadata::row_group_rows_vec).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "Iceberg position-delete row-group planning requires row-count metadata",
        )
        .with_fact("reason", "position_deletes_require_row_group_rows")
        .with_fact("path", file.location.diagnostic_location())
        .with_fact("stable_id", file.location.diagnostic_stable_id(file.stable_id.as_str()))
        .with_fact("file_index", file_index)
    })?;
    contiguous_row_group_position_range(row_groups, &row_group_rows).map(|_range| ()).map_err(|error| {
        error
            .with_fact("path", file.location.diagnostic_location())
            .with_fact("stable_id", file.location.diagnostic_stable_id(file.stable_id.as_str()))
            .with_fact("file_index", file_index)
    })
}

fn append_row_group_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
    chunks: &mut Vec<SourceChunk>,
) {
    let max_row_groups = policy.effective_max_row_groups_per_chunk();
    let row_group_count_basis = metadata.map_or(row_groups.len(), RowGroupMetadata::row_group_count).max(1);
    let mut current = PendingChunk::default();

    for row_group in row_groups {
        let row_group_bytes = metadata.and_then(|metadata| metadata.row_group_bytes(*row_group));
        let estimated_row_group_bytes =
            row_group_bytes.or_else(|| proportional_bytes(file.file_size_bytes, 1, row_group_count_basis));
        let projected_compressed_bytes = metadata.and_then(|metadata| metadata.projected_compressed_bytes(*row_group));
        let projected_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.projected_uncompressed_bytes(*row_group));
        let pure_filter_compressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_compressed_bytes(*row_group));
        let pure_filter_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_uncompressed_bytes(*row_group));
        let row_group_rows = metadata.and_then(|metadata| metadata.row_group_rows(*row_group));
        let retained_output_bytes = projected_uncompressed_bytes.or(estimated_row_group_bytes);
        let source_read_planning_bytes =
            source_read_uncompressed_bytes(source, projected_uncompressed_bytes, pure_filter_uncompressed_bytes)
                .or(retained_output_bytes);
        if current.should_flush_before(source_read_planning_bytes, max_row_groups, policy.max_source_chunk_bytes) {
            push_pending_chunk(
                source_node_id,
                source,
                file,
                file_index,
                row_groups.len(),
                metadata,
                policy,
                reader_decimal_types,
                reader_decimal_types_footer_verified,
                &mut current,
                chunks,
            );
        }
        current.push(
            *row_group,
            row_group_bytes,
            source_read_planning_bytes,
            projected_compressed_bytes,
            projected_uncompressed_bytes,
            pure_filter_compressed_bytes,
            pure_filter_uncompressed_bytes,
            row_group_rows,
        );
    }

    push_pending_chunk(
        source_node_id,
        source,
        file,
        file_index,
        row_groups.len(),
        metadata,
        policy,
        reader_decimal_types,
        reader_decimal_types_footer_verified,
        &mut current,
        chunks,
    );
}
fn push_pending_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    planned_row_group_count: usize,
    metadata: Option<&RowGroupMetadata>,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
    pending: &mut PendingChunk,
    chunks: &mut Vec<SourceChunk>,
) {
    if pending.row_groups.is_empty() {
        return;
    }

    let known_row_group_bytes = pending.known_bytes();
    let known_projected_compressed_bytes = pending.known_projected_compressed_bytes();
    let known_projected_uncompressed_bytes = pending.known_projected_uncompressed_bytes();
    let known_pure_filter_compressed_bytes = pending.known_pure_filter_compressed_bytes();
    let known_pure_filter_uncompressed_bytes = pending.known_pure_filter_uncompressed_bytes();
    let known_row_count = pending.known_rows();
    let estimated_chunk_bytes = known_projected_uncompressed_bytes
        .or(known_row_group_bytes)
        .or_else(|| proportional_bytes(file.file_size_bytes, pending.row_groups.len(), planned_row_group_count));
    let estimated_source_read_bytes = source_read_uncompressed_bytes(
        source,
        known_projected_uncompressed_bytes,
        known_pure_filter_uncompressed_bytes,
    )
    .or(known_row_group_bytes)
    .or(estimated_chunk_bytes);
    let exact_source_read_preflight_bytes =
        exact_source_read_preflight_bytes(source, known_row_count, source.is_zero_column_count_source());
    chunks.push(SourceChunk {
        source_node_id,
        chunk_index: chunks.len(),
        file_index,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: std::mem::take(&mut pending.row_groups),
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes,
        known_projected_compressed_bytes,
        known_projected_uncompressed_bytes,
        known_pure_filter_compressed_bytes,
        known_pure_filter_uncompressed_bytes,
        estimated_chunk_bytes,
        estimated_source_read_bytes,
        exact_source_read_preflight_bytes,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: metadata.and_then(RowGroupMetadata::row_group_bytes_vec),
        source_file_projected_compressed_bytes: metadata.and_then(RowGroupMetadata::projected_compressed_bytes_vec),
        source_file_projected_uncompressed_bytes: metadata.and_then(RowGroupMetadata::projected_uncompressed_bytes_vec),
        source_file_pure_filter_compressed_bytes: metadata.and_then(RowGroupMetadata::pure_filter_compressed_bytes_vec),
        source_file_pure_filter_uncompressed_bytes: metadata
            .and_then(RowGroupMetadata::pure_filter_uncompressed_bytes_vec),
        source_file_row_group_rows: metadata.and_then(RowGroupMetadata::row_group_rows_vec),
        reader_decimal_types: reader_decimal_types.to_vec(),
        reader_decimal_types_footer_verified,
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count,
        scan_limit_rows: None,
        size_estimate_based: known_row_group_bytes.is_none(),
        source_batching_reason: None,
        source_chunk_planning_mode: policy.source_chunk_planning_mode(),
        source_chunk_planning_reason: policy.source_chunk_planning_reason().to_string(),
    });
    pending.clear();
}

fn source_read_uncompressed_bytes(
    source: &LocalParquetSource,
    projected_uncompressed_bytes: Option<u64>,
    pure_filter_uncompressed_bytes: Option<u64>,
) -> Option<u64> {
    match (projected_uncompressed_bytes, pure_filter_uncompressed_bytes) {
        (Some(projected), Some(filter)) if projected_and_pure_filter_file_columns_overlap(source) => {
            Some(projected.max(filter))
        },
        (Some(projected), Some(filter)) => projected.checked_add(filter),
        (Some(projected), None) => Some(projected),
        (None, Some(filter)) => Some(filter),
        (None, None) => None,
    }
}

fn exact_source_read_preflight_bytes(
    source: &LocalParquetSource,
    known_row_count: Option<u64>,
    zero_column_count: bool,
) -> Option<u64> {
    let row_count = if zero_column_count { 1 } else { known_row_count? };
    if zero_column_count {
        return exact_source_read_required_budget_bytes(exact_fixed_width_column_upper_bound_bytes(
            &arrow::datatypes::DataType::Int64,
            false,
            row_count,
        )?);
    }
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for column in source.projected_output_columns.iter().chain(&source.pure_filter_columns) {
        if !seen.insert(column.as_str()) {
            continue;
        }
        let field = source.schema.field_with_name(column).ok()?;
        let column_bytes =
            exact_fixed_width_column_upper_bound_bytes(field.data_type(), field.is_nullable(), row_count)?;
        total = total.checked_add(column_bytes)?;
    }
    exact_source_read_required_budget_bytes(total)
}

fn exact_source_read_required_budget_bytes(retained_output_bytes: u64) -> Option<u64> {
    let minimum_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).ok()?;
    Some(retained_output_bytes.max(minimum_budget))
}

fn exact_fixed_width_column_upper_bound_bytes(
    data_type: &arrow::datatypes::DataType,
    nullable: bool,
    row_count: u64,
) -> Option<u64> {
    let element_size = match data_type {
        arrow::datatypes::DataType::Boolean | arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => 1,
        arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 => 2,
        arrow::datatypes::DataType::Int32
        | arrow::datatypes::DataType::UInt32
        | arrow::datatypes::DataType::Float32
        | arrow::datatypes::DataType::Date32 => 4,
        arrow::datatypes::DataType::Int64
        | arrow::datatypes::DataType::UInt64
        | arrow::datatypes::DataType::Float64
        | arrow::datatypes::DataType::Date64
        | arrow::datatypes::DataType::Timestamp(_, _)
        | arrow::datatypes::DataType::Duration(_) => 8,
        arrow::datatypes::DataType::Decimal128(_, _) => 16,
        _ => return None,
    };
    let data_bytes = row_count.checked_mul(element_size)?;
    let mask_bytes = if nullable { null_mask_allocation_size_bytes(row_count) } else { 0 };
    data_bytes.checked_add(mask_bytes)
}

fn null_mask_allocation_size_bytes(row_count: u64) -> u64 {
    if row_count == 0 { 0 } else { row_count.div_ceil(8).div_ceil(64) * 64 }
}

fn source_file_has_row_count_metadata(file: &LocalParquetFile) -> bool {
    let Some(rows) = &file.row_group_rows else {
        return false;
    };
    if let Some(selected) = &file.row_groups {
        selected.iter().all(|row_group| *row_group < rows.len())
    } else {
        !rows.is_empty()
    }
}

fn projected_and_pure_filter_file_columns_overlap(source: &LocalParquetSource) -> bool {
    let projected = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            crate::source::SourceColumnRef::File(index) => Some(*index),
            crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    source.pure_filter_column_source_refs.iter().any(|source_ref| match source_ref {
        crate::source::SourceColumnRef::File(index) => projected.contains(index),
        crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => false,
    })
}

fn single_all_row_groups_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
) -> SourceChunk {
    SourceChunk {
        source_node_id,
        chunk_index,
        file_index,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: Vec::new(),
        reads_all_row_groups: true,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: None,
        known_projected_compressed_bytes: None,
        known_projected_uncompressed_bytes: None,
        known_pure_filter_compressed_bytes: None,
        known_pure_filter_uncompressed_bytes: None,
        estimated_chunk_bytes: file.file_size_bytes,
        estimated_source_read_bytes: file.file_size_bytes,
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: None,
        source_file_projected_compressed_bytes: None,
        source_file_projected_uncompressed_bytes: None,
        source_file_pure_filter_compressed_bytes: None,
        source_file_pure_filter_uncompressed_bytes: None,
        source_file_row_group_rows: None,
        reader_decimal_types: reader_decimal_types.to_vec(),
        reader_decimal_types_footer_verified,
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count: None,
        scan_limit_rows: None,
        size_estimate_based: true,
        source_batching_reason: None,
        source_chunk_planning_mode: policy.source_chunk_planning_mode(),
        source_chunk_planning_reason: policy.source_chunk_planning_reason().to_string(),
    }
}

fn proportional_bytes(total_bytes: Option<u64>, child_row_groups: usize, parent_row_groups: usize) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

#[derive(Debug, Default)]
struct PendingChunk {
    row_groups: Vec<usize>,
    known_bytes: u64,
    known_planning_bytes: u64,
    known_projected_compressed_bytes: u64,
    known_projected_uncompressed_bytes: u64,
    known_pure_filter_compressed_bytes: u64,
    known_pure_filter_uncompressed_bytes: u64,
    known_rows: u64,
    has_unknown_bytes: bool,
    has_unknown_projected_compressed_bytes: bool,
    has_unknown_projected_uncompressed_bytes: bool,
    has_unknown_pure_filter_compressed_bytes: bool,
    has_unknown_pure_filter_uncompressed_bytes: bool,
    has_unknown_rows: bool,
}

impl PendingChunk {
    fn push(
        &mut self,
        row_group: usize,
        row_group_bytes: Option<u64>,
        planning_bytes: Option<u64>,
        projected_compressed_bytes: Option<u64>,
        projected_uncompressed_bytes: Option<u64>,
        pure_filter_compressed_bytes: Option<u64>,
        pure_filter_uncompressed_bytes: Option<u64>,
        row_group_rows: Option<u64>,
    ) {
        self.row_groups.push(row_group);
        if let Some(row_group_bytes) = row_group_bytes {
            self.known_bytes = self.known_bytes.saturating_add(row_group_bytes);
        } else {
            self.has_unknown_bytes = true;
        }
        if let Some(planning_bytes) = planning_bytes {
            self.known_planning_bytes = self.known_planning_bytes.saturating_add(planning_bytes);
        }
        if let Some(projected_compressed_bytes) = projected_compressed_bytes {
            self.known_projected_compressed_bytes =
                self.known_projected_compressed_bytes.saturating_add(projected_compressed_bytes);
        } else {
            self.has_unknown_projected_compressed_bytes = true;
        }
        if let Some(projected_uncompressed_bytes) = projected_uncompressed_bytes {
            self.known_projected_uncompressed_bytes =
                self.known_projected_uncompressed_bytes.saturating_add(projected_uncompressed_bytes);
        } else {
            self.has_unknown_projected_uncompressed_bytes = true;
        }
        if let Some(pure_filter_compressed_bytes) = pure_filter_compressed_bytes {
            self.known_pure_filter_compressed_bytes =
                self.known_pure_filter_compressed_bytes.saturating_add(pure_filter_compressed_bytes);
        } else {
            self.has_unknown_pure_filter_compressed_bytes = true;
        }
        if let Some(pure_filter_uncompressed_bytes) = pure_filter_uncompressed_bytes {
            self.known_pure_filter_uncompressed_bytes =
                self.known_pure_filter_uncompressed_bytes.saturating_add(pure_filter_uncompressed_bytes);
        } else {
            self.has_unknown_pure_filter_uncompressed_bytes = true;
        }
        if let Some(row_group_rows) = row_group_rows {
            self.known_rows = self.known_rows.saturating_add(row_group_rows);
        } else {
            self.has_unknown_rows = true;
        }
    }

    fn should_flush_before(
        &self,
        next_row_group_bytes: Option<u64>,
        max_row_groups: usize,
        max_source_chunk_bytes: Option<u64>,
    ) -> bool {
        if self.row_groups.is_empty() {
            return false;
        }
        if self.row_groups.len() >= max_row_groups {
            return true;
        }

        let Some(max_source_chunk_bytes) = max_source_chunk_bytes else {
            return false;
        };
        let Some(next_row_group_bytes) = next_row_group_bytes else {
            return false;
        };

        self.known_planning_bytes.saturating_add(next_row_group_bytes) > max_source_chunk_bytes
    }

    fn known_bytes(&self) -> Option<u64> {
        (!self.has_unknown_bytes).then_some(self.known_bytes)
    }

    fn known_projected_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_compressed_bytes).then_some(self.known_projected_compressed_bytes)
    }

    fn known_projected_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_uncompressed_bytes).then_some(self.known_projected_uncompressed_bytes)
    }

    fn known_pure_filter_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_compressed_bytes).then_some(self.known_pure_filter_compressed_bytes)
    }

    fn known_pure_filter_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_uncompressed_bytes).then_some(self.known_pure_filter_uncompressed_bytes)
    }

    fn known_rows(&self) -> Option<u64> {
        (!self.has_unknown_rows).then_some(self.known_rows)
    }

    fn clear(&mut self) {
        self.row_groups.clear();
        self.known_bytes = 0;
        self.known_planning_bytes = 0;
        self.known_projected_compressed_bytes = 0;
        self.known_projected_uncompressed_bytes = 0;
        self.known_pure_filter_compressed_bytes = 0;
        self.known_pure_filter_uncompressed_bytes = 0;
        self.known_rows = 0;
        self.has_unknown_bytes = false;
        self.has_unknown_projected_compressed_bytes = false;
        self.has_unknown_projected_uncompressed_bytes = false;
        self.has_unknown_pure_filter_compressed_bytes = false;
        self.has_unknown_pure_filter_uncompressed_bytes = false;
        self.has_unknown_rows = false;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{BinaryOp, Expr, LiteralValue},
        runtime::{MemoryControlPolicy, SourceChunkPlanningMode},
        source::{
            CredentialMode, LakehouseCacheReadMode, LakehouseCacheStatus, ParquetPositionDeleteFile,
            ParquetReadStrategy, ParquetReaderDecimalTypeId, ParquetSourceLocation, RemoteParquetReadMode,
            ResolvedParquetSource,
        },
    };

    fn batchable_chunk(index: usize, file_index: usize, bytes: u64) -> SourceChunk {
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index,
            location: ParquetSourceLocation::LocalPath(PathBuf::from(format!("part-{file_index}.parquet"))),
            path: PathBuf::from(format!("part-{file_index}.parquet")),
            resolved_source: ResolvedParquetSource::local_kvikio(format!("part-{file_index}.parquet")),
            stable_id: format!("file-{file_index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(bytes),
            known_row_group_bytes: Some(bytes),
            known_projected_compressed_bytes: Some(bytes),
            known_projected_uncompressed_bytes: Some(bytes),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(bytes),
            estimated_source_read_bytes: Some(bytes),
            exact_source_read_preflight_bytes: Some(bytes),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![bytes]),
            source_file_projected_compressed_bytes: Some(vec![bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["c0".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }

    fn remote_direct_require_batchable_chunk(index: usize, file_index: usize, bytes: u64) -> SourceChunk {
        let mut chunk = batchable_chunk(index, file_index, bytes);
        let location = format!("s3://bucket/part-{file_index}.parquet");
        chunk.resolved_source =
            ResolvedParquetSource::remote_direct(location.clone(), CredentialMode::SharedProcessEnv)
                .with_remote_read_mode(RemoteParquetReadMode::RequireDirect);
        chunk.location = ParquetSourceLocation::object_store_uri(&location).expect("valid s3 object URI");
        chunk.path = PathBuf::from(location);
        chunk
    }

    fn bounded_policy(
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
    ) -> MemoryControlPolicy {
        MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    }

    fn local_source_file() -> LocalParquetFile {
        LocalParquetFile::new("/tmp/source.parquet")
            .with_row_groups((0..2048).collect())
            .with_row_group_bytes(vec![1024; 2048])
            .with_projected_row_group_bytes(vec![512; 2048], vec![1024; 2048])
            .with_pure_filter_row_group_bytes(vec![0; 2048], vec![0; 2048])
            .with_row_group_rows(vec![1; 2048])
    }

    fn source_from_file(file: LocalParquetFile) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
    }

    fn local_source() -> LocalParquetSource {
        source_from_file(local_source_file())
    }

    fn position_delete_file() -> ParquetPositionDeleteFile {
        ParquetPositionDeleteFile::from_resolved_parquet_source(
            "delete-0",
            ResolvedParquetSource::local_kvikio("/tmp/delete.parquet"),
        )
        .unwrap()
    }

    fn missing_metadata_source() -> LocalParquetSource {
        source_from_file(LocalParquetFile::new("/tmp/missing-metadata.parquet").with_row_groups((0..2048).collect()))
    }

    #[test]
    fn conservative_source_chunk_planning_preserves_configured_row_group_cap() {
        let source = local_source();
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::Conservative,
        );

        assert_eq!(policy.max_row_groups_per_chunk, 1);
        assert_eq!(policy.max_source_chunk_bytes, Some(64 * 1024 * 1024));
    }

    #[test]
    fn local_throughput_source_chunk_planning_raises_row_group_cap_for_large_local_kvikio_source() {
        let source = local_source();
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::LocalParquetThroughput,
        );

        assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
        assert_eq!(policy.max_source_chunk_bytes, Some(64 * 1024 * 1024));
    }

    #[test]
    fn local_throughput_source_chunk_planning_accepts_local_cache_reader_path() {
        let path = std::env::temp_dir().join(format!(
            "nexus-query-engine-local-cache-throughput-{}-{}.parquet",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::write(&path, b"placeholder").unwrap();
        let resolved_source = ResolvedParquetSource::local_cache_kvikio(
            "s3://bucket/table/part-0.parquet",
            path.display().to_string(),
            CredentialMode::SharedProcessEnv,
            LakehouseCacheStatus::Hit,
        );
        let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_row_groups((0..2048).collect())
            .with_row_group_bytes(vec![1024; 2048])
            .with_projected_row_group_bytes(vec![512; 2048], vec![1024; 2048])
            .with_pure_filter_row_group_bytes(vec![0; 2048], vec![0; 2048])
            .with_row_group_rows(vec![1; 2048]);
        let source = source_from_file(file);
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::LocalParquetThroughput,
        );

        assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
        assert_eq!(policy.source_chunk_planning_reason(), "local_parquet_throughput_admitted");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn remote_direct_metadata_path_respects_cache_read_bypass() {
        let path = std::env::temp_dir().join(format!(
            "nexus-query-engine-remote-direct-bypass-{}-{}.parquet",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::write(&path, b"placeholder").unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_path(path.display().to_string())
                .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Bypassed)
                .with_content_identity("content-a");
        let file = LocalParquetFile::from_resolved_parquet_source(resolved_source).unwrap().with_stable_id("stable-a");

        assert!(local_parquet_footer_metadata_path(&file).is_none());
        assert_eq!(sidecar_content_identity(&file), "content-a");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_metadata_cache_key_distinguishes_object_store_profile_and_content_identity() {
        let base_resolved =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_remote_read_mode(RemoteParquetReadMode::RequireDirect);
        let base = LocalParquetFile::from_resolved_parquet_source(base_resolved)
            .expect("valid remote direct source")
            .with_stable_id("part-0")
            .with_file_size_bytes(4096);
        let other_profile = base
            .clone()
            .with_resolved_parquet_source(base.resolved_source.clone().with_storage_profile_id("profile-b"))
            .expect("valid profiled remote direct source");
        let other_content = base
            .clone()
            .with_resolved_parquet_source(base.resolved_source.clone().with_content_identity("content-b"))
            .expect("valid content-scoped remote direct source");
        let other_location = LocalParquetFile::from_resolved_parquet_source(
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-1.parquet", CredentialMode::SharedProcessEnv)
                .with_remote_read_mode(RemoteParquetReadMode::RequireDirect),
        )
        .expect("valid remote direct source")
        .with_stable_id("part-0")
        .with_file_size_bytes(4096);

        assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_profile));
        assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_content));
        assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_location));
    }

    #[test]
    fn remote_direct_decimal_reader_predicate_requires_footer_type_metadata() {
        let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
        let resolved =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_remote_read_mode(RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("part-0")
            .with_file_size_bytes(4096);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);

        let error = plan_local_parquet_source_chunks(
            NodeId::from_index_for_internal_use(7),
            &source,
            SourceChunkPlanningPolicy::new(None, 1),
        )
        .expect_err("remote decimal reader predicates need proven footer dtype metadata");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert_eq!(error.fact_value("missing_reader_decimal_columns"), Some("discount"));
        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_host_buffer_decimal_reader_predicate_requires_footer_type_metadata() {
        let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
        let resolved = ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/table/part-0.parquet",
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        );
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote host-buffer source")
            .with_stable_id("part-0")
            .with_file_size_bytes(4096);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);

        let error = plan_local_parquet_source_chunks(
            NodeId::from_index_for_internal_use(7),
            &source,
            SourceChunkPlanningPolicy::new(None, 1),
        )
        .expect_err("remote host-buffer decimal predicates need proven footer dtype metadata");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_host_buffer"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("allow_host_fallback"));
        assert_eq!(error.fact_value("missing_reader_decimal_columns"), Some("discount"));
        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_direct_decimal_reader_predicate_rejects_supplied_footer_type_mismatch() {
        let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
        let resolved =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_remote_read_mode(RemoteParquetReadMode::RequireDirect);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote direct source")
            .with_stable_id("part-0")
            .with_reader_decimal_types(vec![ParquetReaderDecimalType::new(
                "discount",
                ParquetReaderDecimalTypeId::Decimal32,
                -2,
            )]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);
        let footer_reader_decimal_types =
            vec![ParquetReaderDecimalType::new("discount", ParquetReaderDecimalTypeId::Decimal64, -2)];

        let error = validate_supplied_reader_decimal_types_match_footer(
            &source,
            &source.files[0],
            &source.files[0].reader_decimal_types,
            &footer_reader_decimal_types,
        )
        .expect_err("pre-attached reader decimal types must match footer metadata");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_mismatch"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert!(
            error
                .fact_value("missing_reader_decimal_columns")
                .is_some_and(|value| { value.contains("discount:decimal32:-2!=footer:decimal64:-2") })
        );
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn remote_host_buffer_decimal_reader_predicate_rejects_supplied_footer_type_mismatch() {
        let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
        let resolved = ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/table/part-0.parquet",
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        );
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote host-buffer source")
            .with_stable_id("part-0")
            .with_reader_decimal_types(vec![ParquetReaderDecimalType::new(
                "discount",
                ParquetReaderDecimalTypeId::Decimal32,
                -2,
            )]);
        let predicate = Expr::binary(
            Expr::column("discount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
        );
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
            .with_column_file_indices(vec![0], vec![0], vec![])
            .with_reader_predicate(predicate);
        let footer_reader_decimal_types =
            vec![ParquetReaderDecimalType::new("discount", ParquetReaderDecimalTypeId::Decimal64, -2)];

        let error = validate_supplied_reader_decimal_types_match_footer(
            &source,
            &source.files[0],
            &source.files[0].reader_decimal_types,
            &footer_reader_decimal_types,
        )
        .expect_err("pre-attached host-buffer reader decimal types must match footer metadata");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_mismatch"));
        assert_eq!(error.fact_value("read_strategy"), Some("remote_host_buffer"));
        assert_eq!(error.fact_value("remote_read_mode"), Some("allow_host_fallback"));
        assert!(
            error
                .fact_value("missing_reader_decimal_columns")
                .is_some_and(|value| { value.contains("discount:decimal32:-2!=footer:decimal64:-2") })
        );
        assert!(!error.to_string().contains("s3://bucket"));
    }

    #[test]
    fn local_throughput_source_chunk_planning_keeps_small_budget_conservative() {
        let source = local_source();
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES - 1, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::LocalParquetThroughput,
        );

        assert_eq!(policy.max_row_groups_per_chunk, 1);
    }

    #[test]
    fn local_throughput_source_chunk_planning_requires_represented_source_read_bytes() {
        let source = missing_metadata_source();
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::LocalParquetThroughput,
        );

        assert_eq!(policy.max_row_groups_per_chunk, 1);

        let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();
        assert_eq!(plan.chunk_count(), 2048);
        assert!(plan.chunks.iter().all(|chunk| chunk.row_group_count() == 1));
    }

    #[test]
    fn local_throughput_source_chunk_planning_rejects_object_store_and_host_buffer_sources() {
        let object_store = source_from_file(
            LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet").expect("valid s3 object URI"),
        );
        let mut host_buffer_file = local_source_file();
        host_buffer_file.resolved_source.read_strategy = ParquetReadStrategy::RemoteHostBuffer;
        let host_buffer = source_from_file(host_buffer_file);
        let policy = bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1);

        assert_eq!(
            SourceChunkPlanningPolicy::from_memory_policy_with_mode(
                &object_store,
                policy,
                SourceChunkPlanningMode::LocalParquetThroughput,
            )
            .max_row_groups_per_chunk,
            1
        );
        assert_eq!(
            SourceChunkPlanningPolicy::from_memory_policy_with_mode(
                &host_buffer,
                policy,
                SourceChunkPlanningMode::LocalParquetThroughput,
            )
            .max_row_groups_per_chunk,
            1
        );
    }

    #[test]
    fn object_store_source_without_row_group_metadata_plans_read_all_row_groups_chunk() {
        let source = source_from_file(
            LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet").expect("valid s3 object URI"),
        );
        let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &source,
            bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
            SourceChunkPlanningMode::Conservative,
        );

        let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();

        assert_eq!(plan.chunk_count(), 1);
        assert!(plan.chunks[0].reads_all_row_groups);
        assert!(plan.chunks[0].row_group_indexes.is_empty());
        assert_eq!(plan.chunks[0].source_file_row_group_rows, None);
    }

    #[test]
    fn source_chunk_planning_splits_position_delete_files_into_row_position_chunks() {
        let file = LocalParquetFile::new("/tmp/position-delete-source.parquet")
            .with_row_group_bytes(vec![100, 200, 300])
            .with_projected_row_group_bytes(vec![50, 100, 150], vec![100, 200, 300])
            .with_pure_filter_row_group_bytes(vec![0, 0, 0], vec![0, 0, 0])
            .with_row_group_rows(vec![10, 20, 30])
            .with_position_delete_files(vec![position_delete_file()]);
        let source = source_from_file(file);

        let plan = plan_local_parquet_source_chunks(
            NodeId::from_index_for_internal_use(3),
            &source,
            SourceChunkPlanningPolicy::for_position_delete_scan_limit(),
        )
        .unwrap();

        assert_eq!(plan.chunk_count(), 3);
        assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
        assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
        assert_eq!(
            source_chunk_row_position_range(&plan.chunks[1]).unwrap(),
            Some(SourceChunkRowPositionRange { start: 10, end: 30 })
        );
    }

    #[test]
    fn position_delete_source_scan_limit_does_not_raw_trim_chunk_plan() {
        let mut source = source_from_file(
            LocalParquetFile::new("/tmp/position-delete-source.parquet")
                .with_row_group_bytes(vec![100, 200, 300])
                .with_projected_row_group_bytes(vec![50, 100, 150], vec![100, 200, 300])
                .with_pure_filter_row_group_bytes(vec![0, 0, 0], vec![0, 0, 0])
                .with_row_group_rows(vec![10, 20, 30])
                .with_position_delete_files(vec![position_delete_file()]),
        );
        source.scan_limit = Some(1);

        let plan = plan_local_parquet_source_chunks(
            NodeId::from_index_for_internal_use(3),
            &source,
            SourceChunkPlanningPolicy::for_position_delete_scan_limit(),
        )
        .unwrap();

        assert_eq!(plan.chunk_count(), 3);
        assert!(plan.chunks.iter().all(|chunk| chunk.scan_limit_rows.is_none()));
    }

    #[test]
    fn local_throughput_source_chunk_planning_rejects_partitioned_and_scan_limit_sources() {
        let mut partitioned_file = local_source_file();
        partitioned_file.partition_values = vec![LiteralValue::Int64(1)];
        let partitioned = source_from_file(partitioned_file);
        let mut scan_limited = local_source();
        scan_limited.scan_limit = Some(10);
        let policy = bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1);

        assert_eq!(
            SourceChunkPlanningPolicy::from_memory_policy_with_mode(
                &partitioned,
                policy,
                SourceChunkPlanningMode::LocalParquetThroughput,
            )
            .max_row_groups_per_chunk,
            1
        );
        assert_eq!(
            SourceChunkPlanningPolicy::from_memory_policy_with_mode(
                &scan_limited,
                policy,
                SourceChunkPlanningMode::LocalParquetThroughput,
            )
            .max_row_groups_per_chunk,
            1
        );
    }

    #[test]
    fn source_chunk_batches_group_adjacent_compatible_chunks_within_byte_target() {
        let chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20), batchable_chunk(2, 2, 30)];

        let batches = plan_source_chunk_batches(&chunks, Some(35));

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
        assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn source_chunk_batches_group_adjacent_row_group_chunks_from_same_file() {
        let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 0, 20), batchable_chunk(2, 0, 30)];
        chunks[0].row_group_indexes = vec![0];
        chunks[1].row_group_indexes = vec![1];
        chunks[2].row_group_indexes = vec![2];

        let batches = plan_source_chunk_batches(&chunks, Some(35));

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
        assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn source_chunk_batches_keep_ineligible_chunks_singleton() {
        let mut read_all = batchable_chunk(1, 1, 20);
        read_all.reads_all_row_groups = true;
        read_all.row_group_indexes.clear();
        let mut host_buffer = batchable_chunk(2, 2, 30);
        host_buffer.resolved_source.read_strategy = ParquetReadStrategy::RemoteHostBuffer;
        let remote_direct = remote_direct_require_batchable_chunk(3, 3, 40);

        let batches = plan_source_chunk_batches(
            &[batchable_chunk(0, 0, 10), read_all, host_buffer, remote_direct, batchable_chunk(4, 4, 50)],
            None,
        );

        assert_eq!(batches.len(), 5);
        assert!(batches.iter().all(SourceChunkBatch::is_singleton));
        assert_eq!(batches[3].source_batching_reason, "incompatible_read_strategy");
    }

    #[test]
    fn source_chunk_batches_group_adjacent_remote_direct_chunks() {
        let chunks = vec![
            remote_direct_require_batchable_chunk(0, 0, 10),
            remote_direct_require_batchable_chunk(1, 1, 20),
            remote_direct_require_batchable_chunk(2, 2, 30),
        ];

        let batches = plan_source_chunk_batches(&chunks, Some(35));

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
        assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn source_chunk_batches_keep_prefer_direct_remote_chunks_singleton() {
        let mut prefer_direct = remote_direct_require_batchable_chunk(0, 0, 10);
        prefer_direct.resolved_source.remote_read_mode = RemoteParquetReadMode::PreferDirect;
        let chunks = vec![prefer_direct, remote_direct_require_batchable_chunk(1, 1, 20)];

        let batches = plan_source_chunk_batches(&chunks, None);

        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(SourceChunkBatch::is_singleton));
        assert_eq!(batches[0].source_batching_reason, "ineligible_remote_read_mode");
        assert_eq!(batches[1].source_batching_reason, "trailing_singleton");
    }

    #[test]
    fn annotate_source_chunk_batching_reasons_applies_source_level_gate() {
        let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20)];

        annotate_source_chunk_batching_reasons(&mut chunks, Some("source_batching_disabled"), Some(35));

        assert_eq!(
            chunks.iter().map(|chunk| chunk.source_batching_reason.as_deref()).collect::<Vec<_>>(),
            vec![Some("source_batching_disabled"), Some("source_batching_disabled")]
        );
    }

    #[test]
    fn annotate_source_chunk_batching_reasons_uses_planned_batch_reasons() {
        let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20), batchable_chunk(2, 2, 30)];

        annotate_source_chunk_batching_reasons(&mut chunks, None, Some(35));

        assert_eq!(
            chunks.iter().map(|chunk| chunk.source_batching_reason.as_deref()).collect::<Vec<_>>(),
            vec![Some("batched"), Some("batched"), Some("trailing_singleton")]
        );
    }

    #[test]
    fn source_chunk_batch_split_for_retry_bisects_and_recomputes_estimates() {
        let batch = SourceChunkBatch {
            batch_index: 0,
            chunks: vec![
                batchable_chunk(0, 0, 10),
                batchable_chunk(1, 1, 20),
                batchable_chunk(2, 2, 30),
                batchable_chunk(3, 3, 40),
                batchable_chunk(4, 4, 50),
            ],
            estimated_source_read_bytes: Some(150),
            source_batching_reason: "batched".to_string(),
        };

        let (left, right) = batch.split_for_retry().expect("multi-chunk batch should split");

        assert_eq!(left.batch_index, 0);
        assert_eq!(left.chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(left.estimated_source_read_bytes, Some(30));
        assert_eq!(right.batch_index, 2);
        assert_eq!(right.chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2, 3, 4]);
        assert_eq!(right.estimated_source_read_bytes, Some(120));
    }

    #[test]
    fn source_chunk_batch_split_for_retry_rejects_singletons() {
        let batch = SourceChunkBatch {
            batch_index: 0,
            chunks: vec![batchable_chunk(0, 0, 10)],
            estimated_source_read_bytes: Some(10),
            source_batching_reason: "trailing_singleton".to_string(),
        };

        assert!(batch.split_for_retry().is_none());
    }

    #[test]
    fn source_chunk_detects_whole_known_file_row_group_coverage() {
        let mut chunk = batchable_chunk(0, 0, 10);
        chunk.source_file_row_groups = Some(vec![0, 1, 2]);
        chunk.row_group_indexes = vec![2, 0, 1];

        assert!(chunk.selects_all_known_file_row_groups());

        chunk.row_group_indexes = vec![0, 2];
        assert!(!chunk.selects_all_known_file_row_groups());

        chunk.row_group_indexes = vec![0, 1, 2];
        chunk.source_file_row_groups = None;
        chunk.source_file_row_group_rows = Some(vec![10, 10, 10]);
        chunk.source_file_row_group_bytes = Some(vec![1, 1, 1]);
        chunk.source_file_projected_compressed_bytes = Some(vec![1, 1, 1]);
        chunk.source_file_projected_uncompressed_bytes = Some(vec![1, 1, 1]);
        chunk.source_file_pure_filter_compressed_bytes = Some(vec![0, 0, 0]);
        chunk.source_file_pure_filter_uncompressed_bytes = Some(vec![0, 0, 0]);
        assert!(chunk.selects_all_known_file_row_groups());

        chunk.row_group_indexes = vec![0, 2];
        assert!(!chunk.selects_all_known_file_row_groups());
    }
}
