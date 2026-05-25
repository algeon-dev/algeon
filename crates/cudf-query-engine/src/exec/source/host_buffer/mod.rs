use std::time::Instant;

use super::*;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::S3ObjectStoreKey,
};

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
    let fetch = start_object_store_parquet_file_fetch(context, file)?;
    finish_object_store_parquet_file_fetch(context, source, file, file_index, chunk_index, fetch, metrics)
}

pub(super) struct ScheduledHostParquetFetch {
    receiver: std::sync::mpsc::Receiver<Result<HostParquetBuffer>>,
}

pub(super) fn start_object_store_parquet_file_fetch(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<ScheduledHostParquetFetch> {
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
    let (cudf_source, cudf_chunk) =
        cudf_local_parquet_source_file_selection(context, source, file, file_index, chunk_index)?;
    let mut fetch = fetch.finish()?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let frame = DataFrame::read_parquet_source_chunk_from_host_buffer(
        context.runtime(),
        &cudf_source,
        &cudf_chunk,
        SourceReadMode::ExecutionFrame,
        fetch.bytes(),
    )
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.source_decode.host_buffer_file",
            "failed to decode object-store Parquet source file from host buffer",
            error,
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("file_index", file_index)
        .with_fact("read_columns", source.read_columns.len())
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
    chunk: &SourceChunk,
    cudf_source: &cudf::io::source::LocalParquetSource,
    cudf_chunk: &cudf::io::source::SourceChunk,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<BudgetedExecutionOutcome<DataFrame>> {
    let estimated_fetch_bytes = chunk.known_file_bytes.or(chunk.estimated_source_read_bytes);
    let mut fetch = start_host_parquet_buffer_fetch(context, &chunk.location, estimated_fetch_bytes)?.finish()?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let outcome = DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer(
        context.runtime(),
        cudf_source,
        cudf_chunk,
        SourceReadMode::ExecutionFrame,
        fetch.bytes(),
        token,
    )
    .map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "cudf_query_engine.source_decode.host_buffer_chunk",
            "failed to decode bounded object-store Parquet source chunk from host buffer",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    metrics.record_source_decode_time(decode_timer.elapsed());
    fetch.release_after_decode();
    Ok(outcome)
}

struct HostParquetBuffer {
    bytes: bytes::Bytes,
    permit: Option<crate::runtime::HostFetchPermit>,
    actual_bytes: u64,
    host_budget_wait_elapsed: std::time::Duration,
    concurrency_wait_elapsed: std::time::Duration,
    fetch_elapsed: std::time::Duration,
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
