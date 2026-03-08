//! Footer-first ranged Parquet fetch used when a source carries a `scan_limit`.
//!
//! For limited remote S3 Iceberg/Parquet scans, this avoids fetching the whole
//! object just to return the first few row groups.
//!
//! This module fetches only the selected row groups, at Parquet row-group
//! granularity:
//!
//! 1. Fetch and parse the footer with two `ObjectStore::get_range` calls.
//! 2. The minimum prefix of row groups that covers `scan_limit` rows, or an
//!    explicit row-group subset selected by bounded chunk planning, is selected.
//! 3. The byte ranges spanning selected column chunks in those row groups are
//!    fetched with `ObjectStore::get_ranges`.
//! 4. A compact, legal Parquet file is rewritten in memory: magic header,
//!    selected row-group bytes laid out back-to-back, and a regenerated footer
//!    whose column-chunk offsets point at the new buffer layout. Bloom filter,
//!    column index, and offset index references are cleared because the
//!    payload bytes for those structures are not fetched.
//!
//! libcudf's host-buffer reader needs a complete legal Parquet buffer; sparse
//! row-group bytes with a footer are not enough. The mini-Parquet rewrite keeps
//! ranged fetches inside the existing reader contract.

use std::{sync::Arc, time::Instant};

use bytes::{Bytes, BytesMut};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath};
use parquet::{
    file::{
        FOOTER_SIZE,
        metadata::{
            ColumnChunkMetaData, FooterTail, ParquetMetaDataBuilder, ParquetMetaDataReader, ParquetMetaDataWriter,
            RowGroupMetaData,
        },
    },
    schema::types::{ColumnDescPtr, SchemaDescriptor, Type},
};

use super::{
    object_store_fetch::{HostParquetBuffer, ScheduledHostParquetFetch},
    object_store_safety::{object_store_bucket_fact, source_fetch_location},
};
use crate::{
    chunk::SourceChunk,
    error::{QueryError, QueryErrorCode, Result},
    exec::source::metrics::format_row_groups,
    runtime::{ExecutionContext, S3ObjectStoreKey},
    source::{LocalParquetFile, LocalParquetSource},
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Speculative tail size for the first ranged GET. Large enough to cover the
/// footer of any practical Parquet file in a single round trip while still
/// being a tiny fraction of even a small file.
const FOOTER_PREFETCH_BYTES: u64 = 64 * 1024;

/// Selector for the row-group subset to fetch.
#[derive(Debug, Clone)]
pub(super) enum RangedRowGroupSelector {
    /// Fetch the minimal row-group prefix that covers a row scan limit.
    ScanLimit(u64),
    /// Fetch a caller-chosen set of row groups, in the order given.
    Explicit(Vec<usize>),
}

/// Ranged mini-Parquet request.
#[derive(Debug, Clone)]
pub(super) struct RangedParquetFetchRequest {
    pub(super) row_group_selector: RangedRowGroupSelector,
    pub(super) file_column_indices: Option<Vec<usize>>,
}

impl RangedParquetFetchRequest {
    pub(super) fn new(row_group_selector: RangedRowGroupSelector, file_column_indices: Option<Vec<usize>>) -> Self {
        Self { row_group_selector, file_column_indices }
    }
}

/// A compact, legal Parquet payload derived from one object-store Parquet file.
///
/// `mini_*` metadata describes row groups inside `bytes`.
/// `selected_original_row_groups` is diagnostic/mapping metadata only and must
/// not be passed to libcudf as row-group indices for the mini-Parquet buffer.
#[derive(Debug, Clone)]
pub(super) struct RangedParquetFetch {
    pub(super) bytes: Bytes,
    pub(super) original_file_bytes: u64,
    pub(super) selected_original_row_groups: Vec<usize>,
    pub(super) mini_row_group_rows: Vec<u64>,
    pub(super) mini_row_group_bytes: Vec<u64>,
    pub(super) footer_bytes: u64,
    pub(super) selected_file_column_indices: Vec<usize>,
    pub(super) requested_range_count: usize,
    pub(super) requested_range_bytes: u64,
}

impl RangedParquetFetch {
    pub(super) fn mini_payload_bytes(&self) -> u64 {
        u64::try_from(self.bytes.len()).expect("usize payload length always fits in u64")
    }

    fn validate(self, location: &ObjectStorePath) -> Result<Self> {
        if self.mini_row_group_bytes.len() != self.mini_row_group_rows.len() {
            return Err(invalid_rewrite(location, "mini row-group byte and row-count metadata lengths differ")
                .with_fact("mini_row_group_bytes", self.mini_row_group_bytes.len())
                .with_fact("mini_row_group_rows", self.mini_row_group_rows.len()));
        }
        if self.selected_original_row_groups.len() != self.mini_row_group_rows.len() {
            return Err(invalid_rewrite(
                location,
                "original-to-mini row-group mapping length differs from mini metadata",
            )
            .with_fact("selected_original_row_groups", self.selected_original_row_groups.len())
            .with_fact("mini_row_group_rows", self.mini_row_group_rows.len()));
        }
        let mini_payload_bytes = self.mini_payload_bytes();
        let mini_row_group_bytes = self
            .mini_row_group_bytes
            .iter()
            .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| invalid_rewrite(location, "mini row-group byte metadata overflows u64"))?;
        if mini_row_group_bytes > mini_payload_bytes {
            return Err(invalid_rewrite(location, "mini row-group bytes exceed mini-Parquet payload length")
                .with_fact("mini_row_group_bytes", mini_row_group_bytes)
                .with_fact("mini_payload_bytes", mini_payload_bytes));
        }
        if !self.bytes.starts_with(PARQUET_MAGIC) || !self.bytes.ends_with(PARQUET_MAGIC) {
            return Err(invalid_rewrite(location, "rewritten mini-Parquet payload is missing PAR1 magic bytes"));
        }
        Ok(self)
    }
}

// Diagnostic projection of ranged-fetch metadata without retaining payload bytes.
#[derive(Debug, Clone)]
pub(super) struct RangedParquetFetchDiagnostics {
    pub(super) mini_payload_bytes: u64,
    pub(super) original_file_bytes: u64,
    pub(super) selected_original_row_groups: Vec<usize>,
    pub(super) mini_row_group_rows: Vec<u64>,
    pub(super) mini_row_group_bytes: Vec<u64>,
    pub(super) footer_bytes: u64,
    pub(super) selected_file_column_indices: Vec<usize>,
    pub(super) requested_range_count: usize,
    pub(super) requested_range_bytes: u64,
}

impl From<&RangedParquetFetch> for RangedParquetFetchDiagnostics {
    fn from(fetch: &RangedParquetFetch) -> Self {
        Self {
            mini_payload_bytes: fetch.mini_payload_bytes(),
            original_file_bytes: fetch.original_file_bytes,
            selected_original_row_groups: fetch.selected_original_row_groups.clone(),
            mini_row_group_rows: fetch.mini_row_group_rows.clone(),
            mini_row_group_bytes: fetch.mini_row_group_bytes.clone(),
            footer_bytes: fetch.footer_bytes,
            selected_file_column_indices: fetch.selected_file_column_indices.clone(),
            requested_range_count: fetch.requested_range_count,
            requested_range_bytes: fetch.requested_range_bytes,
        }
    }
}

pub(super) fn file_request(source: &LocalParquetSource, file: &LocalParquetFile) -> Option<RangedParquetFetchRequest> {
    if !file.resolved_source.remote_host_buffer_ranged_enabled {
        return None;
    }
    if file.has_position_delete_files() {
        return None;
    }
    if !matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return None;
    }
    let column_indices = source.read_column_file_indices.clone();
    let column_pruned = file_column_pruning_applies(file, source);
    if let Some(row_groups) = file.row_groups.as_ref().filter(|row_groups| !row_groups.is_empty())
        && super::represented_file_row_group_count(file).is_some_and(|total| row_groups.len() < total)
    {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit(row_groups.clone()),
            Some(column_indices),
        ));
    }
    let scan_limit = source.scan_limit.filter(|limit| *limit > 0).and_then(|limit| u64::try_from(limit).ok());
    if let Some(scan_limit) = scan_limit {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::ScanLimit(scan_limit),
            Some(column_indices),
        ));
    }
    if let Some(row_groups) = file.row_groups.as_ref().filter(|row_groups| !row_groups.is_empty())
        && column_pruned
        && super::represented_file_row_group_count(file).is_some()
    {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit(row_groups.clone()),
            Some(column_indices),
        ));
    }
    if column_pruned && super::represented_file_row_group_count(file).is_some_and(|total| total > 0) {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit(
                (0..super::represented_file_row_group_count(file).expect("checked above")).collect(),
            ),
            Some(column_indices),
        ));
    }
    None
}

pub(super) fn chunk_request(source: &LocalParquetSource, chunk: &SourceChunk) -> Option<RangedParquetFetchRequest> {
    if !chunk.resolved_source.remote_host_buffer_ranged_enabled {
        return None;
    }
    if source.files.get(chunk.file_index).is_some_and(LocalParquetFile::has_position_delete_files) {
        return None;
    }
    if !matches!(chunk.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return None;
    }
    let column_indices = source.read_column_file_indices.clone();
    let column_pruned = chunk_column_pruning_applies(chunk, source);
    if !chunk.row_group_indexes.is_empty()
        && chunk_source_file_row_group_count(chunk).is_some_and(|row_groups| chunk.row_group_indexes.len() < row_groups)
    {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit(chunk.row_group_indexes.clone()),
            Some(column_indices),
        ));
    }
    let scan_limit = source.scan_limit.filter(|limit| *limit > 0).and_then(|limit| u64::try_from(limit).ok());
    if chunk.scan_limit_rows.is_some()
        && let Some(scan_limit) = scan_limit
    {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::ScanLimit(scan_limit),
            Some(column_indices),
        ));
    }
    if !chunk.row_group_indexes.is_empty() && column_pruned && chunk_source_file_row_group_count(chunk).is_some() {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit(chunk.row_group_indexes.clone()),
            Some(column_indices),
        ));
    }
    if column_pruned && let Some(total_row_groups) = chunk_source_file_row_group_count(chunk).filter(|count| *count > 0)
    {
        return Some(RangedParquetFetchRequest::new(
            RangedRowGroupSelector::Explicit((0..total_row_groups).collect()),
            Some(column_indices),
        ));
    }
    None
}

pub(super) fn attach_fetch_facts(
    error: QueryError,
    ranged_fetch: &RangedParquetFetchDiagnostics,
    chunk: &SourceChunk,
) -> QueryError {
    error
        .with_fact("source_fetch_mode", "ranged_parquet")
        .with_fact("mini_parquet_bytes", ranged_fetch.mini_payload_bytes)
        .with_fact("requested_range_count", ranged_fetch.requested_range_count)
        .with_fact("requested_range_bytes", ranged_fetch.requested_range_bytes)
        .with_fact("mini_row_group_bytes", comma_separated_u64(&ranged_fetch.mini_row_group_bytes))
        .with_fact("mini_parquet_footer_bytes", ranged_fetch.footer_bytes)
        .with_fact("original_file_bytes", ranged_fetch.original_file_bytes)
        .with_fact(
            "selected_original_row_groups",
            comma_separated_usize_values(&ranged_fetch.selected_original_row_groups),
        )
        .with_fact(
            "selected_file_column_indices",
            comma_separated_usize_values(&ranged_fetch.selected_file_column_indices),
        )
        .with_fact("mini_row_groups", ranged_fetch.mini_row_group_rows.len())
        .with_fact("original_chunk_row_groups", format_row_groups(chunk))
}

pub(super) fn attach_file_fetch_facts(
    error: QueryError,
    ranged_fetch: &RangedParquetFetchDiagnostics,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> QueryError {
    error
        .with_fact("source_fetch_mode", "ranged_parquet")
        .with_fact("mini_parquet_bytes", ranged_fetch.mini_payload_bytes)
        .with_fact("requested_range_count", ranged_fetch.requested_range_count)
        .with_fact("requested_range_bytes", ranged_fetch.requested_range_bytes)
        .with_fact("mini_row_group_bytes", comma_separated_u64(&ranged_fetch.mini_row_group_bytes))
        .with_fact("mini_parquet_footer_bytes", ranged_fetch.footer_bytes)
        .with_fact("original_file_bytes", ranged_fetch.original_file_bytes)
        .with_fact(
            "selected_original_row_groups",
            comma_separated_usize_values(&ranged_fetch.selected_original_row_groups),
        )
        .with_fact(
            "selected_file_column_indices",
            comma_separated_usize_values(&ranged_fetch.selected_file_column_indices),
        )
        .with_fact("mini_row_groups", ranged_fetch.mini_row_group_rows.len())
        .with_fact("path", super::parquet_file_path_fact(file))
        .with_fact("stable_id", super::parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
        .with_fact("chunk_index", chunk_index)
}

/// Schedules a footer-first ranged Parquet fetch that returns a compact
/// mini-Parquet buffer covering selected row groups and file columns. This is
/// the host-buffer fallback for remote reads that can still avoid full-object
/// S3 transfer before libcudf sees the file.
pub(super) fn start_host_parquet_buffer_ranged_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    request: RangedParquetFetchRequest,
) -> Result<ScheduledHostParquetFetch> {
    context.check_cancellation("nexus_query_engine.source_fetch.host_buffer_ranged")?;
    let fetch_location = source_fetch_location(location, resolved_source)?;
    let host_budget_wait_timer = Instant::now();
    let mut permit = context.host_fetch_budget().acquire(1).map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
            .with_fact_if_absent("source_fetch_mode", "ranged_parquet")
    })?;
    let mut host_budget_wait_elapsed = host_budget_wait_timer.elapsed();
    let concurrency_wait_timer = Instant::now();
    let operation_permit = context.acquire_source_fetch_operation().map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
            .with_fact_if_absent("source_fetch_mode", "ranged_parquet")
    })?;
    let concurrency_wait_elapsed = concurrency_wait_timer.elapsed();
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    })?;
    let key = fetch_location.key.clone();
    let bucket_for_error = fetch_location.bucket.clone();
    let region_for_error = fetch_location.region.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let _fetch_handle = context.spawn_object_store_custom_fetch(
        move || fetch_ranged_parquet(store, key, request),
        move |fetch_result| {
            let fetch_elapsed = fetch_result.elapsed;
            let result = fetch_result
                .value
                .map_err(|error| {
                    error
                        .with_fact_if_absent("source_fetch_substrate", "object_store")
                        .with_fact_if_absent("source_fetch_scheme", "s3")
                        .with_fact_if_absent("source_fetch_mode", "ranged_parquet")
                        .with_fact_if_absent("bucket", object_store_bucket_fact(bucket_for_error.as_str()))
                        .with_fact_if_absent("region", region_for_error.as_str())
                })
                .and_then(|ranged_fetch| {
                    let actual_bytes = ranged_fetch.mini_payload_bytes();
                    let grow_wait_timer = Instant::now();
                    permit.grow_to(actual_bytes.max(1)).map_err(|error| {
                        error
                            .with_fact_if_absent("source_fetch_substrate", "object_store")
                            .with_fact_if_absent("source_fetch_scheme", "s3")
                            .with_fact_if_absent("source_fetch_mode", "ranged_parquet")
                            .with_fact_if_absent("bucket", object_store_bucket_fact(bucket_for_error.as_str()))
                            .with_fact_if_absent("actual_bytes", actual_bytes)
                    })?;
                    host_budget_wait_elapsed += grow_wait_timer.elapsed();
                    let diagnostics = RangedParquetFetchDiagnostics::from(&ranged_fetch);
                    Ok(HostParquetBuffer {
                        bytes: ranged_fetch.bytes,
                        permit: Some(permit),
                        actual_bytes,
                        ranged_fetch: Some(diagnostics),
                        host_budget_wait_elapsed,
                        concurrency_wait_elapsed,
                        fetch_elapsed,
                    })
                });
            let _ = sender.send(result);
            drop(operation_permit);
        },
    );
    Ok(ScheduledHostParquetFetch { receiver })
}

fn file_column_pruning_applies(file: &LocalParquetFile, source: &LocalParquetSource) -> bool {
    column_pruning_likely_applies(source)
        || source_read_bytes_are_strict_subset(
            source_read_bytes_for_all_rows(
                file.projected_row_group_compressed_bytes.as_ref(),
                file.pure_filter_row_group_compressed_bytes.as_ref(),
            ),
            file.row_group_bytes.as_ref().map(|bytes| bytes.iter().sum()),
        )
}

fn chunk_column_pruning_applies(chunk: &SourceChunk, source: &LocalParquetSource) -> bool {
    column_pruning_likely_applies(source)
        || source_read_bytes_are_strict_subset(chunk.estimated_source_read_bytes, chunk.known_file_bytes)
        || source_read_bytes_are_strict_subset(
            source_read_bytes_for_all_rows(
                chunk.source_file_projected_compressed_bytes.as_ref(),
                chunk.source_file_pure_filter_compressed_bytes.as_ref(),
            ),
            chunk.source_file_row_group_bytes.as_ref().map(|bytes| bytes.iter().sum()),
        )
}

fn column_pruning_likely_applies(source: &LocalParquetSource) -> bool {
    if source.read_column_file_indices.is_empty() {
        return false;
    }
    if let Some(file_column_count) = source.file_column_count {
        let mut indices = source.read_column_file_indices.clone();
        indices.sort_unstable();
        indices.dedup();
        return indices.len() < file_column_count;
    }
    !source.read_column_file_indices.iter().copied().eq(0..source.read_column_file_indices.len())
}

fn source_read_bytes_are_strict_subset(selected: Option<u64>, total: Option<u64>) -> bool {
    matches!((selected, total), (Some(selected), Some(total)) if selected > 0 && selected < total)
}

fn source_read_bytes_for_all_rows(projected: Option<&Vec<u64>>, pure_filter: Option<&Vec<u64>>) -> Option<u64> {
    match (projected, pure_filter) {
        (Some(projected), Some(pure_filter)) if projected.len() == pure_filter.len() => {
            projected.iter().zip(pure_filter).try_fold(0u64, |total, (projected, pure_filter)| {
                total.checked_add(projected.saturating_add(*pure_filter))
            })
        },
        (Some(projected), None) => projected.iter().try_fold(0u64, |total, bytes| total.checked_add(*bytes)),
        (None, Some(pure_filter)) => pure_filter.iter().try_fold(0u64, |total, bytes| total.checked_add(*bytes)),
        _ => None,
    }
}

fn chunk_source_file_row_group_count(chunk: &SourceChunk) -> Option<usize> {
    chunk
        .source_file_row_group_rows
        .as_ref()
        .or(chunk.source_file_row_group_bytes.as_ref())
        .or(chunk.source_file_projected_compressed_bytes.as_ref())
        .or(chunk.source_file_projected_uncompressed_bytes.as_ref())
        .or(chunk.source_file_pure_filter_compressed_bytes.as_ref())
        .or(chunk.source_file_pure_filter_uncompressed_bytes.as_ref())
        .map(Vec::len)
}

fn comma_separated_usize_values(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn comma_separated_u64(values: &[u64]) -> String {
    values.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

/// Fetches only the row groups needed to satisfy a scan limit or an explicit
/// row-group selection from an object-store Parquet file and returns a
/// compact mini-Parquet buffer that libcudf can decode directly.
pub(super) async fn fetch_ranged_parquet(
    store: Arc<dyn ObjectStore>,
    location: ObjectStorePath,
    request: RangedParquetFetchRequest,
) -> Result<RangedParquetFetch> {
    match &request.row_group_selector {
        RangedRowGroupSelector::ScanLimit(0) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "ranged Parquet fetch requires a positive scan limit",
            ));
        },
        RangedRowGroupSelector::Explicit(indexes) if indexes.is_empty() => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "ranged Parquet fetch requires a non-empty row-group selection",
            ));
        },
        _ => {},
    }

    let head = store
        .head(&location)
        .await
        .map_err(|error| ranged_fetch_error("failed to HEAD object-store Parquet file", error, &location))?;
    let file_size = head.size;
    if file_size < FOOTER_SIZE as u64 {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store Parquet file is smaller than the Parquet footer",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_fact("file_size", file_size));
    }

    let prefetch = FOOTER_PREFETCH_BYTES.min(file_size);
    let mut requested_range_count = 1usize;
    let mut requested_range_bytes = prefetch;
    let tail_start = file_size - prefetch;
    let tail = store
        .get_range(&location, tail_start..file_size)
        .await
        .map_err(|error| ranged_fetch_error("failed to GET Parquet footer range", error, &location))?;
    if tail.len() < FOOTER_SIZE {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store Parquet tail fetch returned fewer bytes than the footer size",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_fact("file_size", file_size)
        .with_fact("tail_bytes", tail.len()));
    }

    let footer_tail_bytes: [u8; FOOTER_SIZE] =
        tail[tail.len() - FOOTER_SIZE..].try_into().expect("footer tail slice was length-checked");
    let footer_tail = FooterTail::try_from(footer_tail_bytes).map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to decode Parquet footer tail",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_source(error)
    })?;
    if footer_tail.is_encrypted_footer() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "encrypted Parquet footers are not supported by ranged scan-limit fetch",
        )
        .with_fact("path", ranged_path_fact(&location)));
    }

    let metadata_len = footer_tail.metadata_length();
    let total_footer_len = metadata_len + FOOTER_SIZE;
    if (total_footer_len as u64) > file_size {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "Parquet footer metadata length exceeds file size",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_fact("file_size", file_size)
        .with_fact("metadata_length", metadata_len));
    }

    let metadata_bytes = if (total_footer_len as u64) <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        requested_range_count += 1;
        requested_range_bytes = requested_range_bytes.saturating_add(metadata_end.saturating_sub(metadata_start));
        store
            .get_range(&location, metadata_start..metadata_end)
            .await
            .map_err(|error| ranged_fetch_error("failed to GET Parquet footer metadata range", error, &location))?
    };

    let mut reader = ParquetMetaDataReader::new();
    let mut footer_with_tail = BytesMut::with_capacity(total_footer_len);
    footer_with_tail.extend_from_slice(&metadata_bytes);
    footer_with_tail.extend_from_slice(&footer_tail_bytes);
    reader.try_parse_sized(&footer_with_tail.freeze(), file_size).map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to decode Parquet footer metadata",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_source(error)
    })?;
    let metadata = reader.finish().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to finalize Parquet footer metadata",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_source(error)
    })?;

    let row_groups = metadata.row_groups();
    if row_groups.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "Parquet footer reports zero row groups",
        )
        .with_fact("path", ranged_path_fact(&location)));
    }

    let selected_indices: Vec<usize> = match &request.row_group_selector {
        RangedRowGroupSelector::ScanLimit(scan_limit) => {
            let mut needed_rows = *scan_limit;
            let mut selected = Vec::new();
            for (idx, row_group) in row_groups.iter().enumerate() {
                selected.push(idx);
                let rg_rows = u64::try_from(row_group.num_rows().max(0)).unwrap_or(0);
                if rg_rows >= needed_rows {
                    break;
                }
                needed_rows -= rg_rows;
            }
            selected
        },
        RangedRowGroupSelector::Explicit(indexes) => {
            for &idx in indexes {
                if idx >= row_groups.len() {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "explicit row-group selection references an unknown row group",
                    )
                    .with_fact("path", ranged_path_fact(&location))
                    .with_fact("requested_row_group", idx)
                    .with_fact("row_groups_in_file", row_groups.len()));
                }
            }
            indexes.clone()
        },
    };
    let selected_count = selected_indices.len();
    let selected_row_groups: Vec<&RowGroupMetaData> = selected_indices.iter().map(|&idx| &row_groups[idx]).collect();
    let selected_file_column_indices =
        selected_file_column_indices(request.file_column_indices.as_deref(), &metadata, &location)?;
    let mini_row_group_rows = selected_row_groups
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows().max(0)).unwrap_or(0))
        .collect::<Vec<_>>();

    // Compute the byte ranges to fetch (one per selected column chunk in each
    // selected row group), and prepare a parallel list of new offsets after rewrite.
    let mut fetch_ranges: Vec<std::ops::Range<u64>> = Vec::new();
    let mut new_offsets: Vec<u64> = Vec::new();
    let mut row_group_start_offsets: Vec<u64> = Vec::with_capacity(selected_row_groups.len());
    let mut mini_row_group_bytes: Vec<u64> = Vec::with_capacity(selected_row_groups.len());
    let mut next_offset: u64 = PARQUET_MAGIC.len() as u64;
    for row_group in &selected_row_groups {
        row_group_start_offsets.push(next_offset);
        let row_group_start = next_offset;
        for &column_index in &selected_file_column_indices {
            let column = row_group.columns().get(column_index).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "selected Parquet column index is not present in a selected row group",
                )
                .with_fact("path", ranged_path_fact(&location))
                .with_fact("column_index", column_index)
                .with_fact("row_group_columns", row_group.columns().len())
            })?;
            let (start, length) = column.byte_range();
            fetch_ranges.push(
                start..start.checked_add(length).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::CudfDependencyFailure,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "Parquet column-chunk byte range overflows file size",
                    )
                    .with_fact("path", ranged_path_fact(&location))
                    .with_fact("column_start", start)
                    .with_fact("column_length", length)
                })?,
            );
            new_offsets.push(next_offset);
            next_offset = next_offset.checked_add(length).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "rewritten Parquet column-chunk offset overflows u64",
                )
                .with_fact("path", ranged_path_fact(&location))
            })?;
        }
        mini_row_group_bytes.push(next_offset - row_group_start);
    }

    let column_data_total_bytes = next_offset - PARQUET_MAGIC.len() as u64;
    requested_range_count += fetch_ranges.len();
    requested_range_bytes = requested_range_bytes
        .saturating_add(fetch_ranges.iter().map(|range| range.end.saturating_sub(range.start)).sum::<u64>());
    let column_chunks = if fetch_ranges.is_empty() {
        Vec::new()
    } else {
        store
            .get_ranges(&location, &fetch_ranges)
            .await
            .map_err(|error| ranged_fetch_error("failed to GET Parquet column-chunk ranges", error, &location))?
    };
    if column_chunks.len() != fetch_ranges.len() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store ranged GET returned an unexpected chunk count",
        )
        .with_fact("path", ranged_path_fact(&location))
        .with_fact("requested_ranges", fetch_ranges.len())
        .with_fact("returned_chunks", column_chunks.len()));
    }

    // Rebuild metadata with adjusted column-chunk offsets and stripped
    // bloom-filter / column-index / offset-index references.
    let source_file_metadata = metadata.file_metadata();
    let projected_schema = projected_flat_schema_descriptor(&metadata, &selected_file_column_indices, &location)?;
    let projects_all_columns = selected_file_column_indices.len() == source_file_metadata.schema_descr().num_columns()
        && selected_file_column_indices.iter().copied().eq(0..source_file_metadata.schema_descr().num_columns());
    let mut rewritten_row_groups: Vec<RowGroupMetaData> = Vec::with_capacity(selected_row_groups.len());
    let mut column_iter = new_offsets.into_iter();
    for (mini_row_group_index, (row_group, &rg_start)) in
        selected_row_groups.iter().zip(row_group_start_offsets.iter()).enumerate()
    {
        let columns: Vec<ColumnChunkMetaData> = row_group.columns().to_vec();
        let mut rewritten_columns = Vec::with_capacity(selected_file_column_indices.len());
        for (projected_column_index, &column_index) in selected_file_column_indices.iter().enumerate() {
            let column = columns.get(column_index).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "selected Parquet column index is missing during metadata rewrite",
                )
                .with_fact("path", ranged_path_fact(&location))
                .with_fact("column_index", column_index)
                .with_fact("row_group_columns", columns.len())
            })?;
            let new_start = column_iter.next().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "internal error: rewritten column offset list shorter than column count",
                )
            })?;
            let dict_offset = column.dictionary_page_offset();
            let data_offset = column.data_page_offset();
            let new_chunk_start = i64::try_from(new_start).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "rewritten Parquet column-chunk offset exceeds i64 range",
                )
            })?;
            let (new_data_page_offset, new_dictionary_page_offset) = match dict_offset {
                Some(dict) => (new_chunk_start + (data_offset - dict), Some(new_chunk_start)),
                None => (new_chunk_start, None),
            };
            let rewritten = rewrite_column_chunk_metadata(
                column,
                projected_schema.column(projected_column_index),
                new_data_page_offset,
                new_dictionary_page_offset,
            )?;
            rewritten_columns.push(rewritten);
        }
        let rg_file_offset = i64::try_from(rg_start).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet row-group file offset exceeds i64 range",
            )
        })?;
        let total_byte_size = rewritten_columns
            .iter()
            .try_fold(0i64, |total, column| total.checked_add(column.uncompressed_size()).ok_or(()))
            .map_err(|()| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "rewritten Parquet row-group byte size overflows i64",
                )
            })?;
        let mut rg_builder = RowGroupMetaData::builder(projected_schema.clone())
            .set_num_rows(row_group.num_rows())
            .set_total_byte_size(total_byte_size)
            .set_column_metadata(rewritten_columns)
            .set_file_offset(rg_file_offset);
        if projects_all_columns {
            rg_builder = rg_builder.set_sorting_columns(row_group.sorting_columns().cloned());
        }
        if let Ok(ordinal) = i16::try_from(mini_row_group_index) {
            rg_builder = rg_builder.set_ordinal(ordinal);
        }
        let rg_meta = rg_builder.build().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "failed to rebuild rewritten Parquet row-group metadata",
            )
            .with_source(error)
        })?;
        rewritten_row_groups.push(rg_meta);
    }

    let mini_total_rows = selected_row_groups.iter().try_fold(0i64, |total, row_group| {
        total.checked_add(row_group.num_rows()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet file row count overflows i64",
            )
            .with_fact("path", ranged_path_fact(&location))
        })
    })?;
    let key_value_metadata =
        if projects_all_columns { source_file_metadata.key_value_metadata().cloned() } else { None };
    let column_orders = source_file_metadata.column_orders().map(|orders| {
        if projects_all_columns {
            orders.clone()
        } else {
            selected_file_column_indices.iter().filter_map(|index| orders.get(*index).cloned()).collect()
        }
    });
    let rewritten_file_metadata = parquet::file::metadata::FileMetaData::new(
        source_file_metadata.version(),
        mini_total_rows,
        source_file_metadata.created_by().map(ToOwned::to_owned),
        key_value_metadata,
        projected_schema,
        column_orders,
    );
    let rewritten_metadata = ParquetMetaDataBuilder::new(rewritten_file_metadata)
        .set_row_groups(rewritten_row_groups)
        .set_column_index(None)
        .set_offset_index(None)
        .build();

    // Emit: PAR1 + column-chunk payload + thrift footer + footer-length + PAR1.
    // ParquetMetaDataWriter assumes its output starts at offset 0; the offsets
    // already written into rewritten_metadata point at the layout we are about
    // to produce.
    let mut output: Vec<u8> =
        Vec::with_capacity(PARQUET_MAGIC.len() + column_data_total_bytes as usize + metadata_len + FOOTER_SIZE);
    output.extend_from_slice(PARQUET_MAGIC);
    for chunk in &column_chunks {
        output.extend_from_slice(chunk);
    }

    let footer_start = output.len();
    {
        let mut footer_buf: Vec<u8> = Vec::new();
        ParquetMetaDataWriter::new(&mut footer_buf, &rewritten_metadata).finish().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "failed to serialize rewritten Parquet footer metadata",
            )
            .with_source(error)
        })?;
        let footer_len = u32::try_from(footer_buf.len()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet footer length exceeds u32 range",
            )
        })?;
        output.extend_from_slice(&footer_buf);
        output.extend_from_slice(&footer_len.to_le_bytes());
        output.extend_from_slice(PARQUET_MAGIC);
        debug_assert!(output.len() > footer_start);
    }

    let footer_bytes = u64::try_from(output.len() - footer_start).map_err(|_error| {
        invalid_rewrite(&location, "rewritten Parquet footer length exceeds u64 range")
            .with_fact("selected_row_groups", selected_count)
    })?;
    RangedParquetFetch {
        bytes: Bytes::from(output),
        original_file_bytes: file_size,
        selected_original_row_groups: selected_indices,
        mini_row_group_rows,
        mini_row_group_bytes,
        footer_bytes,
        selected_file_column_indices,
        requested_range_count,
        requested_range_bytes,
    }
    .validate(&location)
}

fn selected_file_column_indices(
    requested: Option<&[usize]>,
    metadata: &parquet::file::metadata::ParquetMetaData,
    location: &ObjectStorePath,
) -> Result<Vec<usize>> {
    let total_columns = metadata.file_metadata().schema_descr().num_columns();
    let mut columns = requested
        .filter(|columns| !columns.is_empty())
        .map(|columns| columns.to_vec())
        .unwrap_or_else(|| (0..total_columns).collect());
    if columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "ranged Parquet fetch requires at least one selected file column",
        )
        .with_fact("path", ranged_path_fact(location)));
    }
    columns.sort_unstable();
    columns.dedup();
    if let Some(column) = columns.iter().find(|column| **column >= total_columns) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "selected Parquet file column index is out of bounds",
        )
        .with_fact("path", ranged_path_fact(location))
        .with_fact("column_index", *column)
        .with_fact("file_columns", total_columns));
    }
    Ok(columns)
}

fn projected_flat_schema_descriptor(
    metadata: &parquet::file::metadata::ParquetMetaData,
    selected_file_column_indices: &[usize],
    location: &ObjectStorePath,
) -> Result<std::sync::Arc<SchemaDescriptor>> {
    let schema = metadata.file_metadata().schema_descr();
    let root = schema.root_schema();
    let fields = root.get_fields();
    let total_columns = schema.num_columns();
    if selected_file_column_indices.len() == total_columns
        && selected_file_column_indices.iter().copied().eq(0..total_columns)
    {
        return Ok(metadata.file_metadata().schema_descr_ptr());
    }
    for &column_index in selected_file_column_indices {
        let column = schema.column(column_index);
        let path = column.path().parts();
        if path.len() != 1 {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "projected ranged mini-Parquet rewrite currently supports only flat top-level columns",
            )
            .with_fact("path", ranged_path_fact(location))
            .with_fact("column_index", column_index)
            .with_fact("column_path", path.join(".")));
        }
    }
    let projected_fields = selected_file_column_indices
        .iter()
        .map(|index| {
            fields.get(*index).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "selected Parquet file column index is missing from the root schema",
                )
                .with_fact("path", ranged_path_fact(location))
                .with_fact("column_index", *index)
                .with_fact("root_fields", fields.len())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let projected_schema =
        Type::group_type_builder(root.name()).with_fields(projected_fields).build().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "failed to build projected flat Parquet schema for ranged mini-Parquet rewrite",
            )
            .with_fact("path", ranged_path_fact(location))
            .with_source(error)
        })?;
    Ok(Arc::new(SchemaDescriptor::new(Arc::new(projected_schema))))
}

fn rewrite_column_chunk_metadata(
    column: ColumnChunkMetaData,
    projected_column_descriptor: ColumnDescPtr,
    new_data_page_offset: i64,
    new_dictionary_page_offset: Option<i64>,
) -> Result<ColumnChunkMetaData> {
    let mut builder = ColumnChunkMetaData::builder(projected_column_descriptor)
        .set_encodings(column.encodings().collect())
        .set_num_values(column.num_values())
        .set_compression(column.compression())
        .set_total_compressed_size(column.compressed_size())
        .set_total_uncompressed_size(column.uncompressed_size())
        .set_data_page_offset(new_data_page_offset)
        .set_dictionary_page_offset(new_dictionary_page_offset)
        .set_index_page_offset(None)
        .set_bloom_filter_offset(None)
        .set_bloom_filter_length(None)
        .set_offset_index_offset(None)
        .set_offset_index_length(None)
        .set_column_index_offset(None)
        .set_column_index_length(None)
        .set_unencoded_byte_array_data_bytes(column.unencoded_byte_array_data_bytes())
        .set_repetition_level_histogram(column.repetition_level_histogram().cloned())
        .set_definition_level_histogram(column.definition_level_histogram().cloned());
    if let Some(file_path) = column.file_path() {
        builder = builder.set_file_path(file_path.to_string());
    }
    if let Some(statistics) = column.statistics().cloned() {
        builder = builder.set_statistics(statistics);
    }
    if let Some(encoding_stats) = column.page_encoding_stats().cloned() {
        builder = builder.set_page_encoding_stats(encoding_stats);
    } else if let Some(encoding_stats_mask) = column.page_encoding_stats_mask().cloned() {
        builder = builder.set_page_encoding_stats_mask(encoding_stats_mask);
    }
    builder.build().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to rebuild rewritten Parquet column-chunk metadata",
        )
        .with_source(error)
    })
}

fn ranged_fetch_error(message: &'static str, _error: object_store::Error, location: &ObjectStorePath) -> QueryError {
    QueryError::new(QueryErrorCode::CudfDependencyFailure, "nexus_query_engine.source_fetch.ranged_parquet", message)
        .with_fact("path", ranged_path_fact(location))
}

fn invalid_rewrite(location: &ObjectStorePath, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfDependencyFailure, "nexus_query_engine.source_fetch.ranged_parquet", message)
        .with_fact("path", ranged_path_fact(location))
}

fn ranged_path_fact(location: impl AsRef<str>) -> String {
    crate::source::redacted_object_store_path_diagnostic(location.as_ref())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Int64Array, RecordBatch, StringArray, StructArray},
        datatypes::{DataType, Field, Fields, Schema},
    };
    use bytes::Bytes;
    use object_store::{ObjectStoreExt, memory::InMemory, path::Path as ObjectStorePath};
    use parquet::{
        arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
        file::properties::WriterProperties,
    };

    use super::*;

    fn write_test_parquet(num_row_groups: usize, rows_per_group: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let total_rows = num_row_groups * rows_per_group;
        let ids: Vec<i64> = (0..total_rows as i64).collect();
        let names: Vec<String> = (0..total_rows).map(|i| format!("row-{i}")).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(names))],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    fn write_three_column_test_parquet(num_row_groups: usize, rows_per_group: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
        ]));
        let total_rows = num_row_groups * rows_per_group;
        let ids: Vec<i64> = (0..total_rows as i64).collect();
        let names: Vec<String> = (0..total_rows).map(|i| format!("row-{i}")).collect();
        let amounts: Vec<i64> = (0..total_rows as i64).map(|i| i * 10).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(Int64Array::from(amounts)),
            ],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    fn write_nested_test_parquet(rows_per_group: usize) -> Bytes {
        let child_field = Arc::new(Field::new("x", DataType::Int64, false));
        let struct_type = DataType::Struct(Fields::from(vec![child_field.clone()]));
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("s", struct_type, false)]));
        let ids: ArrayRef = Arc::new(Int64Array::from((0..rows_per_group as i64).collect::<Vec<_>>()));
        let values: ArrayRef = Arc::new(Int64Array::from((0..rows_per_group as i64).collect::<Vec<_>>()));
        let struct_array: ArrayRef = Arc::new(StructArray::from(vec![(child_field, values)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, struct_array]).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    fn collect_int64_column(parquet_bytes: Bytes, column: usize) -> Vec<i64> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().build().unwrap();
        let mut values = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
            values.extend(array.iter().map(|v| v.unwrap()));
        }
        values
    }

    fn collect_string_column(parquet_bytes: Bytes, column: usize) -> Vec<String> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().build().unwrap();
        let mut values = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let array = batch.column(column).as_any().downcast_ref::<StringArray>().unwrap();
            values.extend(array.iter().map(|v| v.unwrap().to_string()));
        }
        values
    }

    fn parquet_file_num_rows(parquet_bytes: Bytes) -> i64 {
        ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().metadata().file_metadata().num_rows()
    }

    fn parquet_schema_field_names(parquet_bytes: Bytes) -> Vec<String> {
        ParquetRecordBatchReaderBuilder::try_new(parquet_bytes)
            .unwrap()
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    fn request(selector: RangedRowGroupSelector) -> RangedParquetFetchRequest {
        RangedParquetFetchRequest::new(selector, None)
    }

    #[test]
    fn ranged_path_fact_redacts_object_store_keys() {
        let location = ObjectStorePath::from("secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak");
        let diagnostic = ranged_path_fact(location);

        assert!(diagnostic.starts_with("object_store_path:fnv64:"));
        for sensitive in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
            assert!(!diagnostic.contains(sensitive), "diagnostic leaked {sensitive}: {diagnostic}");
        }
    }

    #[tokio::test]
    async fn fetches_only_first_row_group_when_limit_fits() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, request(RangedRowGroupSelector::ScanLimit(10))).await.unwrap();

        assert!(mini.bytes.len() < parquet_bytes.len(), "mini-Parquet should be smaller than original");
        assert_eq!(mini.original_file_bytes, parquet_bytes.len() as u64);
        assert_eq!(mini.selected_original_row_groups, vec![0]);
        assert_eq!(mini.mini_row_group_rows, vec![25]);
        assert_eq!(mini.mini_row_group_bytes.len(), 1);
        assert!(mini.footer_bytes > FOOTER_SIZE as u64);
        let ids = collect_int64_column(mini.bytes.clone(), 0);
        let names = collect_string_column(mini.bytes, 1);
        assert_eq!(ids.len(), 25);
        assert_eq!(ids, (0..25).collect::<Vec<_>>());
        assert_eq!(names[0], "row-0");
        assert_eq!(names[24], "row-24");
    }

    #[tokio::test]
    async fn fetches_minimum_row_group_prefix_to_cover_limit() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, request(RangedRowGroupSelector::ScanLimit(30))).await.unwrap();

        assert_eq!(mini.selected_original_row_groups, vec![0, 1]);
        assert_eq!(mini.mini_row_group_rows, vec![25, 25]);
        assert_eq!(parquet_file_num_rows(mini.bytes.clone()), 50);
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids.len(), 50, "30-row limit straddles 25-row groups; expect first two row groups");
        assert_eq!(ids, (0..50).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn small_files_fit_in_footer_prefetch() {
        let parquet_bytes = write_test_parquet(1, 8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("tiny.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, request(RangedRowGroupSelector::ScanLimit(4))).await.unwrap();
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn explicit_selection_keeps_original_mapping_but_mini_groups_are_contiguous() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-1.parquet");
        store.put(&location, parquet_bytes.into()).await.unwrap();

        let mini =
            fetch_ranged_parquet(store, location, request(RangedRowGroupSelector::Explicit(vec![2, 3]))).await.unwrap();

        assert_eq!(mini.selected_original_row_groups, vec![2, 3]);
        assert_eq!(mini.mini_row_group_rows, vec![25, 25]);
        assert_eq!(mini.mini_row_group_bytes.len(), 2);
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids, (50..100).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn projected_flat_columns_rewrite_schema_and_data() {
        let parquet_bytes = write_three_column_test_parquet(3, 10);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-2.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(
            store,
            location,
            RangedParquetFetchRequest::new(RangedRowGroupSelector::Explicit(vec![0, 2]), Some(vec![2, 0])),
        )
        .await
        .unwrap();

        assert!(mini.bytes.len() < parquet_bytes.len(), "projected mini-Parquet should be smaller than original");
        assert_eq!(mini.selected_original_row_groups, vec![0, 2]);
        assert_eq!(mini.selected_file_column_indices, vec![0, 2]);
        assert_eq!(parquet_schema_field_names(mini.bytes.clone()), vec!["id".to_string(), "amount".to_string()]);
        assert_eq!(collect_int64_column(mini.bytes.clone(), 0), (0..10).chain(20..30).collect::<Vec<_>>());
        assert_eq!(
            collect_int64_column(mini.bytes, 1),
            (0..10).chain(20..30).map(|value| value * 10).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn projected_nested_schema_falls_back_to_full_host_buffer() {
        let parquet_bytes = write_nested_test_parquet(8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("nested.parquet");
        store.put(&location, parquet_bytes.into()).await.unwrap();

        let error = fetch_ranged_parquet(
            store,
            location,
            RangedParquetFetchRequest::new(RangedRowGroupSelector::Explicit(vec![0]), Some(vec![1])),
        )
        .await
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    }

    #[tokio::test]
    async fn zero_scan_limit_is_rejected() {
        let parquet_bytes = write_test_parquet(1, 8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("tiny.parquet");
        store.put(&location, parquet_bytes.into()).await.unwrap();

        let error =
            fetch_ranged_parquet(store, location, request(RangedRowGroupSelector::ScanLimit(0))).await.unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    }
}
