//! Source scan-limit application for planned chunks.

use super::{
    SourceChunk, SourceReadSizing,
    assembly::{exact_source_read_preflight_bytes, proportional_bytes, source_read_uncompressed_bytes},
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    source::{LocalParquetFile, LocalParquetSource},
};

#[doc(hidden)]
pub fn apply_source_scan_limit_to_chunks(
    source: &LocalParquetSource,
    chunks: Vec<SourceChunk>,
    sizing: SourceReadSizing<'_>,
) -> Result<Vec<SourceChunk>> {
    let Some(scan_limit) = source.scan_limit else {
        return Ok(chunks);
    };
    if source.files.iter().any(LocalParquetFile::has_row_delete_files) {
        // Raw-row prefix trimming can under-produce visible rows when the
        // prefix is deleted. Execution must apply row deletes per chunk
        // before deciding whether a scan limit is satisfied.
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
            .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
        })?;

        if remaining == 0 {
            if limited.is_empty() {
                chunk.scan_limit_rows = Some(0);
                chunk.known_row_count = Some(0);
                chunk.exact_source_read_preflight_bytes =
                    exact_source_read_preflight_bytes(source, chunk.known_row_count, chunk.zero_column_count, sizing);
                limited.push(chunk);
            }
            break;
        }

        if known_row_count <= remaining {
            remaining = remaining.saturating_sub(known_row_count);
            limited.push(chunk);
            continue;
        }

        limited.push(scan_limited_chunk_prefix(source, chunk, remaining, scan_limit, sizing)?);
        break;
    }
    Ok(limited)
}

#[doc(hidden)]
pub fn scan_limited_chunk_prefix(
    source: &LocalParquetSource,
    chunk: SourceChunk,
    keep_rows: u64,
    scan_limit: usize,
    sizing: SourceReadSizing<'_>,
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
        return Ok(rewrite_scan_limited_chunk_selection(source, chunk, Vec::new(), 0, 0, Some(0), sizing));
    }

    let (selected_row_groups, selected_row_group_rows) = scan_limit_row_group_prefix(&chunk, keep_rows)?;
    Ok(rewrite_scan_limited_chunk_selection(
        source,
        chunk,
        selected_row_groups,
        selected_row_group_rows,
        keep_rows,
        Some(keep_rows_usize),
        sizing,
    ))
}

#[doc(hidden)]
pub fn scan_limit_row_group_prefix(chunk: &SourceChunk, keep_rows: u64) -> Result<(Vec<usize>, u64)> {
    let row_group_rows = chunk.source_file_row_group_rows.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.scan_limit",
            "source scan limit requires per-row-group row-count metadata for chunk prefix trimming",
        )
        .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
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
            .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
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
            .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
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
    .with_fact("path", crate::source::parquet_source_path_fact(&chunk.location, &chunk.path))
    .with_fact("file_index", chunk.file_index)
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("requested_rows", keep_rows)
    .with_fact("selected_rows", selected_rows))
}

#[doc(hidden)]
pub fn rewrite_scan_limited_chunk_selection(
    source: &LocalParquetSource,
    mut chunk: SourceChunk,
    selected_row_groups: Vec<usize>,
    selected_row_group_rows: u64,
    output_row_count: u64,
    scan_limit_rows: Option<usize>,
    sizing: SourceReadSizing<'_>,
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
        exact_source_read_preflight_bytes(source, Some(selected_row_group_rows), chunk.zero_column_count, sizing);
    chunk.known_row_count = Some(output_row_count);
    chunk.scan_limit_rows = scan_limit_rows;
    chunk.size_estimate_based = chunk.known_row_group_bytes.is_none();
    chunk
}

#[doc(hidden)]
pub fn sum_selected_row_group_values(values: Option<&Vec<u64>>, row_groups: &[usize]) -> Option<u64> {
    let values = values?;
    row_groups.iter().try_fold(0u64, |total, row_group| values.get(*row_group)?.checked_add(total))
}

#[doc(hidden)]
pub fn scan_limited_proportional_file_bytes(chunk: &SourceChunk, selected_row_group_count: usize) -> Option<u64> {
    if selected_row_group_count == 0 {
        return Some(0);
    }
    proportional_bytes(chunk.known_file_bytes, selected_row_group_count, source_file_row_group_count_basis(chunk))
}

#[doc(hidden)]
pub fn source_file_row_group_count_basis(chunk: &SourceChunk) -> usize {
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
