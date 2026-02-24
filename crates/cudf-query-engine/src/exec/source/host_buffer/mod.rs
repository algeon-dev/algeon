use std::time::Instant;

use super::{
    cudf_contract::{MiniParquetHostBufferDescriptor, cudf_local_parquet_source_mini_host_buffer_chunk},
    *,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::S3ObjectStoreKey,
};

mod ranged_parquet;

struct SourceFetchLocation {
    bucket: String,
    region: String,
    key: object_store::path::Path,
}

pub(super) fn read_object_store_parquet_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let fetch = start_object_store_parquet_file_fetch(context, file, source.scan_limit)?;
    finish_object_store_parquet_file_fetch(context, source, file, file_index, chunk_index, fetch, metrics)
}

pub(super) struct ScheduledHostParquetFetch {
    receiver: std::sync::mpsc::Receiver<Result<HostParquetBuffer>>,
}

pub(super) fn start_object_store_parquet_file_fetch(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    scan_limit: Option<usize>,
) -> Result<ScheduledHostParquetFetch> {
    if let Some(limit) = scan_limit
        && matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. })
    {
        let selector = file
            .row_groups
            .as_ref()
            .filter(|row_groups| !row_groups.is_empty())
            .map(|row_groups| ranged_parquet::RangedRowGroupSelector::Explicit(row_groups.clone()))
            .unwrap_or(ranged_parquet::RangedRowGroupSelector::ScanLimit(limit as u64));
        return start_host_parquet_buffer_ranged_fetch(context, &file.location, selector);
    }
    start_host_parquet_buffer_fetch(context, &file.location, file.file_size_bytes)
}

pub(super) fn finish_object_store_parquet_file_fetch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    fetch: ScheduledHostParquetFetch,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let mut fetch = fetch.finish()?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let frame = if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        let descriptor = MiniParquetHostBufferDescriptor::from_file(
            file,
            source,
            file_index,
            chunk_index,
            fetch.actual_bytes,
            &ranged_fetch.mini_row_group_bytes,
            &ranged_fetch.mini_row_group_rows,
        );
        let (mini_source, mini_chunk) =
            cudf_local_parquet_source_mini_host_buffer_chunk(context, source, descriptor)
                .map_err(|error| attach_ranged_fetch_file_facts(error, ranged_fetch, file, file_index, chunk_index))?;
        DataFrame::read_parquet_source_chunk_from_host_buffer(
            context.runtime(),
            &mini_source,
            &mini_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
        )
    } else {
        let (cudf_source, cudf_chunk) =
            cudf_local_parquet_source_file_selection(context, source, file, file_index, chunk_index)?;
        DataFrame::read_parquet_source_chunk_from_host_buffer(
            context.runtime(),
            &cudf_source,
            &cudf_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
        )
    }
    .map_err(|error| {
        let error = QueryError::from_cudf(
            "cudf_query_engine.source_decode.host_buffer_file",
            "failed to decode object-store Parquet source file from host buffer",
            error,
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("file_index", file_index)
        .with_fact("read_columns", source.read_columns.len());
        if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
            attach_ranged_fetch_file_facts(error, ranged_fetch, file, file_index, chunk_index)
        } else {
            error
        }
    })?;
    let decode_elapsed = decode_timer.elapsed();
    metrics.record_source_decode_time(decode_elapsed);
    metrics.parquet_read_time += decode_elapsed;
    metrics.source_read_time += fetch.fetch_elapsed;
    metrics.source_read_time += decode_elapsed;
    metrics.parquet_files += 1;
    metrics.parquet_bytes_known += fetch.actual_bytes;
    metrics.row_groups_selected += file.selected_row_group_count() as u64;
    metrics.read_columns += source.read_columns.len() as u64;
    metrics.record_input_batch(frame.num_rows() as u64);
    fetch.release_after_decode();
    Ok(frame)
}

pub(super) fn read_object_store_parquet_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    cudf_source: &cudf::io::source::LocalParquetSource,
    cudf_chunk: &cudf::io::source::SourceChunk,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<BudgetedExecutionOutcome<DataFrame>> {
    let estimated_fetch_bytes = chunk.known_file_bytes.or(chunk.estimated_source_read_bytes);
    let mut fetch = if let Some(selector) = ranged_scan_limit_chunk_selector(source, chunk) {
        start_host_parquet_buffer_ranged_fetch(context, &chunk.location, selector)?.finish()?
    } else {
        start_host_parquet_buffer_fetch(context, &chunk.location, estimated_fetch_bytes)?.finish()?
    };
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let outcome = if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        let descriptor = MiniParquetHostBufferDescriptor::from_chunk(
            chunk,
            fetch.actual_bytes,
            &ranged_fetch.mini_row_group_bytes,
            &ranged_fetch.mini_row_group_rows,
        );
        let (mini_source, mini_chunk) = cudf_local_parquet_source_mini_host_buffer_chunk(context, source, descriptor)
            .map_err(|error| attach_ranged_fetch_facts(error, ranged_fetch, chunk))?;
        DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer(
            context.runtime(),
            &mini_source,
            &mini_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
            token,
        )
    } else {
        DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer(
            context.runtime(),
            cudf_source,
            cudf_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
            token,
        )
    }
    .map_err(|error| {
        let error = QueryError::from_cudf_with_budget_details(
            "cudf_query_engine.source_decode.host_buffer_chunk",
            "failed to decode bounded object-store Parquet source chunk from host buffer",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_fact("row_groups", format_row_groups(chunk));
        if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
            attach_ranged_fetch_facts(error, ranged_fetch, chunk)
        } else {
            error
        }
    })?;
    metrics.record_source_decode_time(decode_timer.elapsed());
    fetch.release_after_decode();
    Ok(outcome)
}

fn ranged_scan_limit_chunk_selector(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
) -> Option<ranged_parquet::RangedRowGroupSelector> {
    let source_scan_limit = source.scan_limit?;
    if source_scan_limit == 0 {
        return None;
    }
    let scan_limit = u64::try_from(source_scan_limit).ok()?;
    if !matches!(chunk.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return None;
    }
    if !chunk.row_group_indexes.is_empty()
        && chunk
            .source_file_row_group_rows
            .as_ref()
            .is_some_and(|row_groups| chunk.row_group_indexes.len() < row_groups.len())
    {
        return Some(ranged_parquet::RangedRowGroupSelector::Explicit(chunk.row_group_indexes.clone()));
    }
    if chunk.scan_limit_rows.is_some()
        && chunk.source_file_row_group_bytes.is_none()
        && chunk.source_file_projected_compressed_bytes.is_none()
        && chunk.source_file_projected_uncompressed_bytes.is_none()
        && chunk.source_file_pure_filter_compressed_bytes.is_none()
        && chunk.source_file_pure_filter_uncompressed_bytes.is_none()
    {
        return Some(ranged_parquet::RangedRowGroupSelector::ScanLimit(scan_limit));
    }
    None
}

fn attach_ranged_fetch_facts(
    error: QueryError,
    ranged_fetch: &RangedParquetFetchDiagnostics,
    chunk: &SourceChunk,
) -> QueryError {
    error
        .with_fact("source_fetch_mode", "ranged_scan_limit")
        .with_fact("mini_parquet_bytes", ranged_fetch.mini_payload_bytes)
        .with_fact("mini_row_group_bytes", comma_separated_u64(&ranged_fetch.mini_row_group_bytes))
        .with_fact("mini_parquet_footer_bytes", ranged_fetch.footer_bytes)
        .with_fact("original_file_bytes", ranged_fetch.original_file_bytes)
        .with_fact(
            "selected_original_row_groups",
            comma_separated_usize_values(&ranged_fetch.selected_original_row_groups),
        )
        .with_fact("mini_row_groups", ranged_fetch.mini_row_group_rows.len())
        .with_fact("original_chunk_row_groups", format_row_groups(chunk))
}

fn attach_ranged_fetch_file_facts(
    error: QueryError,
    ranged_fetch: &RangedParquetFetchDiagnostics,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> QueryError {
    error
        .with_fact("source_fetch_mode", "ranged_scan_limit")
        .with_fact("mini_parquet_bytes", ranged_fetch.mini_payload_bytes)
        .with_fact("mini_row_group_bytes", comma_separated_u64(&ranged_fetch.mini_row_group_bytes))
        .with_fact("mini_parquet_footer_bytes", ranged_fetch.footer_bytes)
        .with_fact("original_file_bytes", ranged_fetch.original_file_bytes)
        .with_fact(
            "selected_original_row_groups",
            comma_separated_usize_values(&ranged_fetch.selected_original_row_groups),
        )
        .with_fact("mini_row_groups", ranged_fetch.mini_row_group_rows.len())
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("file_index", file_index)
        .with_fact("chunk_index", chunk_index)
}

fn comma_separated_usize_values(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn comma_separated_u64(values: &[u64]) -> String {
    values.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

struct HostParquetBuffer {
    bytes: bytes::Bytes,
    permit: Option<crate::runtime::HostFetchPermit>,
    actual_bytes: u64,
    ranged_fetch: Option<RangedParquetFetchDiagnostics>,
    host_budget_wait_elapsed: std::time::Duration,
    concurrency_wait_elapsed: std::time::Duration,
    fetch_elapsed: std::time::Duration,
}

#[derive(Debug, Clone)]
struct RangedParquetFetchDiagnostics {
    mini_payload_bytes: u64,
    original_file_bytes: u64,
    selected_original_row_groups: Vec<usize>,
    mini_row_group_rows: Vec<u64>,
    mini_row_group_bytes: Vec<u64>,
    footer_bytes: u64,
}

impl From<&ranged_parquet::RangedParquetFetch> for RangedParquetFetchDiagnostics {
    fn from(fetch: &ranged_parquet::RangedParquetFetch) -> Self {
        Self {
            mini_payload_bytes: fetch.mini_payload_bytes(),
            original_file_bytes: fetch.original_file_bytes,
            selected_original_row_groups: fetch.selected_original_row_groups.clone(),
            mini_row_group_rows: fetch.mini_row_group_rows.clone(),
            mini_row_group_bytes: fetch.mini_row_group_bytes.clone(),
            footer_bytes: fetch.footer_bytes,
        }
    }
}

impl ScheduledHostParquetFetch {
    fn finish(self) -> Result<HostParquetBuffer> {
        self.receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_fetch.receive",
                "object-store source fetch worker stopped before returning a host buffer",
            )
            .with_source(error)
        })?
    }
}

impl HostParquetBuffer {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn release_after_decode(&mut self) {
        self.permit.take();
    }
}

fn start_host_parquet_buffer_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    estimated_fetch_bytes: Option<u64>,
) -> Result<ScheduledHostParquetFetch> {
    context.check_cancellation("cudf_query_engine.source_fetch.host_buffer")?;
    let fetch_location = source_fetch_location(location)?;
    let reserve_bytes = estimated_fetch_bytes.unwrap_or_else(|| context.host_fetch_budget().budget_bytes()).max(1);
    let host_budget_wait_timer = Instant::now();
    let mut permit = context.host_fetch_budget().acquire(reserve_bytes).map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
    })?;
    let mut host_budget_wait_elapsed = host_budget_wait_timer.elapsed();
    let concurrency_wait_timer = Instant::now();
    let operation_permit = context.acquire_source_fetch_operation().map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
    })?;
    let concurrency_wait_elapsed = concurrency_wait_timer.elapsed();
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
    })?;
    let (sender, receiver) = std::sync::mpsc::channel();
    context.spawn_object_store_fetch(store, fetch_location.key, move |fetch_result| {
        let fetch_elapsed = fetch_result.elapsed;
        let result = fetch_result
            .bytes
            .map_err(|error| {
                error
                    .with_fact_if_absent("source_fetch_substrate", "object_store")
                    .with_fact_if_absent("source_fetch_scheme", "s3")
                    .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
                    .with_fact_if_absent("region", fetch_location.region.as_str())
            })
            .and_then(|bytes| {
                let actual_bytes = u64::try_from(bytes.len()).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "cudf_query_engine.source_fetch.host_buffer_size",
                        "object-store source fetch payload length exceeds u64 range",
                    )
                })?;
                let grow_wait_timer = Instant::now();
                permit.grow_to(actual_bytes.max(1)).map_err(|error| {
                    error
                        .with_fact_if_absent("source_fetch_substrate", "object_store")
                        .with_fact_if_absent("source_fetch_scheme", "s3")
                        .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
                        .with_fact_if_absent("actual_bytes", actual_bytes)
                })?;
                host_budget_wait_elapsed += grow_wait_timer.elapsed();
                Ok(HostParquetBuffer {
                    bytes,
                    permit: Some(permit),
                    actual_bytes,
                    ranged_fetch: None,
                    host_budget_wait_elapsed,
                    concurrency_wait_elapsed,
                    fetch_elapsed,
                })
            });
        let _ = sender.send(result);
        drop(operation_permit);
    });
    Ok(ScheduledHostParquetFetch { receiver })
}

/// Schedules a footer-first ranged Parquet fetch that returns a compact
/// mini-Parquet buffer covering the requested scan limit. Used when the
/// source carries a `scan_limit`, so that `SELECT ... LIMIT N` over remote
/// S3 Parquet does not pull the entire object before libcudf can trim.
fn start_host_parquet_buffer_ranged_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    selector: ranged_parquet::RangedRowGroupSelector,
) -> Result<ScheduledHostParquetFetch> {
    context.check_cancellation("cudf_query_engine.source_fetch.host_buffer_ranged")?;
    let fetch_location = source_fetch_location(location)?;
    let host_budget_wait_timer = Instant::now();
    let mut permit = context.host_fetch_budget().acquire(1).map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
            .with_fact_if_absent("source_fetch_mode", "ranged_scan_limit")
    })?;
    let mut host_budget_wait_elapsed = host_budget_wait_timer.elapsed();
    let concurrency_wait_timer = Instant::now();
    let operation_permit = context.acquire_source_fetch_operation().map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", fetch_location.bucket.as_str())
            .with_fact_if_absent("source_fetch_mode", "ranged_scan_limit")
    })?;
    let concurrency_wait_elapsed = concurrency_wait_timer.elapsed();
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
    })?;
    let key = fetch_location.key.clone();
    let bucket_for_error = fetch_location.bucket.clone();
    let region_for_error = fetch_location.region.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    context.spawn_object_store_custom_fetch(
        move || ranged_parquet::fetch_ranged_parquet(store, key, selector),
        move |fetch_result| {
            let fetch_elapsed = fetch_result.elapsed;
            let result = fetch_result
                .value
                .map_err(|error| {
                    error
                        .with_fact_if_absent("source_fetch_substrate", "object_store")
                        .with_fact_if_absent("source_fetch_scheme", "s3")
                        .with_fact_if_absent("source_fetch_mode", "ranged_scan_limit")
                        .with_fact_if_absent("bucket", bucket_for_error.as_str())
                        .with_fact_if_absent("region", region_for_error.as_str())
                })
                .and_then(|ranged_fetch| {
                    let actual_bytes = ranged_fetch.mini_payload_bytes();
                    let grow_wait_timer = Instant::now();
                    permit.grow_to(actual_bytes.max(1)).map_err(|error| {
                        error
                            .with_fact_if_absent("source_fetch_substrate", "object_store")
                            .with_fact_if_absent("source_fetch_scheme", "s3")
                            .with_fact_if_absent("source_fetch_mode", "ranged_scan_limit")
                            .with_fact_if_absent("bucket", bucket_for_error.as_str())
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

fn source_fetch_location(location: &crate::source::ParquetSourceLocation) -> Result<SourceFetchLocation> {
    let crate::source::ParquetSourceLocation::ObjectStoreUri { scheme, uri } = location else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch.location",
            "SourceFetch requires an object-store URI source location",
        )
        .with_fact("read_capability", location.read_capability()));
    };
    if scheme != "s3" {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.source_fetch.location",
            "object-store source fetch currently supports only S3 URIs",
        )
        .with_fact("scheme", scheme));
    }
    parse_s3_source_uri(uri)
}

fn parse_s3_source_uri(uri: &str) -> Result<SourceFetchLocation> {
    let rest = uri.strip_prefix("s3://").ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch.s3_uri",
            "S3 source URI must start with s3://",
        )
        .with_fact("reason", "missing_s3_scheme")
    })?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch.s3_uri",
            "S3 source URI must include a bucket and key",
        )
        .with_fact("reason", "missing_key")
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch.s3_uri",
            "S3 source URI bucket and key must be non-empty",
        )
        .with_fact("reason", "empty_bucket_or_key"));
    }
    let region = std::env::var("AWS_REGION")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "us-east-1".to_string());
    Ok(SourceFetchLocation { bucket: bucket.to_string(), region, key: object_store::path::Path::from(key) })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::*;
    use crate::{
        plan::QueryPlanBuilder,
        source::{LocalParquetFile, SourceNode},
    };

    fn scan_limited_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("payload", ArrowDataType::Int32, true)]));
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(4096)
            .with_row_group_bytes(vec![1024, 1024, 2048])
            .with_row_group_rows(vec![10, 10, 10]);
        LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string()]).with_scan_limit(Some(15))
    }

    fn source_chunk(source: &LocalParquetSource, row_group_indexes: Vec<usize>) -> SourceChunk {
        let mut builder = QueryPlanBuilder::new();
        let source_node_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
        SourceChunk {
            source_node_id,
            chunk_index: 0,
            file_index: 0,
            location: source.files[0].location.clone(),
            path: source.files[0].path.clone(),
            stable_id: source.files[0].stable_id.clone(),
            row_group_indexes,
            reads_all_row_groups: false,
            known_file_bytes: Some(4096),
            known_row_group_bytes: Some(1024),
            known_projected_compressed_bytes: Some(1024),
            known_projected_uncompressed_bytes: Some(1024),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(1024),
            estimated_source_read_bytes: Some(1024),
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![1024, 1024, 2048]),
            source_file_projected_compressed_bytes: Some(vec![1024, 1024, 2048]),
            source_file_projected_uncompressed_bytes: Some(vec![1024, 1024, 2048]),
            source_file_pure_filter_compressed_bytes: Some(vec![0, 0, 0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0, 0, 0]),
            source_file_row_group_rows: Some(vec![10, 10, 10]),
            read_columns: vec!["payload".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
        }
    }

    #[test]
    fn ranged_scan_limit_trigger_covers_fully_kept_limited_chunks() {
        let source = scan_limited_source();
        let chunk = source_chunk(&source, vec![0]);

        let selector = ranged_scan_limit_chunk_selector(&source, &chunk);
        assert!(matches!(
            selector,
            Some(ranged_parquet::RangedRowGroupSelector::Explicit(row_groups)) if row_groups == vec![0]
        ));
    }

    #[test]
    fn ranged_scan_limit_trigger_requires_source_limit_and_skipped_row_groups() {
        let source = scan_limited_source();
        let chunk = source_chunk(&source, vec![0, 1, 2]);
        assert!(ranged_scan_limit_chunk_selector(&source, &chunk).is_none());

        let source_without_limit = source.clone().with_scan_limit(None);
        let limited_chunk = source_chunk(&source_without_limit, vec![0]);
        assert!(ranged_scan_limit_chunk_selector(&source_without_limit, &limited_chunk).is_none());

        let zero_limit_source = source.with_scan_limit(Some(0));
        let zero_limit_chunk = source_chunk(&zero_limit_source, vec![0]);
        assert!(ranged_scan_limit_chunk_selector(&zero_limit_source, &zero_limit_chunk).is_none());
    }

    #[test]
    fn ranged_scan_limit_trigger_uses_footer_selector_when_only_file_row_count_is_known() {
        let source = scan_limited_source();
        let mut chunk = source_chunk(&source, vec![0]);
        chunk.scan_limit_rows = Some(15);
        chunk.known_row_count = Some(15);
        chunk.source_file_row_group_bytes = None;
        chunk.source_file_projected_compressed_bytes = None;
        chunk.source_file_projected_uncompressed_bytes = None;
        chunk.source_file_pure_filter_compressed_bytes = None;
        chunk.source_file_pure_filter_uncompressed_bytes = None;
        chunk.source_file_row_group_rows = Some(vec![30]);

        let selector = ranged_scan_limit_chunk_selector(&source, &chunk);

        assert!(matches!(selector, Some(ranged_parquet::RangedRowGroupSelector::ScanLimit(15))));
    }
}
