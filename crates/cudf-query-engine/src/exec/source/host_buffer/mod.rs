use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use super::{
    cudf_contract::{MiniParquetHostBufferDescriptor, cudf_local_parquet_source_mini_host_buffer_chunk},
    *,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::S3ObjectStoreKey,
};

mod ranged_parquet;

static LAKEHOUSE_CACHE_READ_PIN_COUNTER: AtomicU64 = AtomicU64::new(0);

struct SourceFetchLocation {
    bucket: String,
    region: String,
    key: object_store::path::Path,
    storage_profile_id: Option<String>,
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

pub(super) fn ensure_local_cache_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    ensure_local_cache_parquet_entry(
        context,
        &file.location,
        &file.resolved_source,
        &file.path,
        file.file_size_bytes,
        file.stable_id.as_str(),
        file.resolved_source.lakehouse_cache_root.as_deref(),
        file.resolved_source.lakehouse_cache_max_bytes,
        None,
        None,
        metrics,
    )
}

pub(super) fn ensure_local_cache_parquet_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    ensure_local_cache_parquet_entry(
        context,
        &chunk.location,
        &chunk.resolved_source,
        &chunk.path,
        chunk.known_file_bytes,
        chunk.stable_id.as_str(),
        chunk.resolved_source.lakehouse_cache_root.as_deref(),
        chunk.resolved_source.lakehouse_cache_max_bytes,
        Some(chunk.file_index),
        Some(chunk.chunk_index),
        metrics,
    )
}

pub(super) struct LakehouseCacheReadPin {
    path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(super) struct LakehouseOptimizedBatchCacheEntry {
    pub(super) path: PathBuf,
    pub(super) stable_id: String,
    cache_root: Option<String>,
    cache_max_bytes: Option<u64>,
}

impl Drop for LakehouseCacheReadPin {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

pub(super) fn acquire_lakehouse_cache_file_read_pin(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(context, &file.path, file.stable_id.as_str(), None, None)
}

pub(super) fn acquire_lakehouse_cache_chunk_read_pin(
    context: &ExecutionContext,
    chunk: &SourceChunk,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(
        context,
        &chunk.path,
        chunk.stable_id.as_str(),
        Some(chunk.file_index),
        Some(chunk.chunk_index),
    )
}

pub(super) fn acquire_lakehouse_optimized_batch_read_pin(
    context: &ExecutionContext,
    entry: &LakehouseOptimizedBatchCacheEntry,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(context, &entry.path, entry.stable_id.as_str(), None, None)
}

pub(super) fn local_cache_optimized_batch_entry(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
) -> Option<LakehouseOptimizedBatchCacheEntry> {
    if !local_cache_optimized_batch_eligible(source, batch) {
        return None;
    }
    let first = batch.chunks.first()?;
    let cache_root = first.resolved_source.lakehouse_cache_root.clone()?;
    let source_node = first.source_node_id.to_string();
    let signature = local_cache_optimized_batch_signature(source, batch);
    let hash = stable_fnv1a64(signature.as_bytes());
    let file_name = format!("{}__batch_v4_{hash:016x}.parquet", sanitize_cache_component(&source_node));
    let path = Path::new(&cache_root).join("optimized-batches").join("v4").join(file_name);
    Some(LakehouseOptimizedBatchCacheEntry {
        path,
        stable_id: format!("optimized-batch-v4-fnv64-{hash:016x}"),
        cache_root: Some(cache_root),
        cache_max_bytes: first.resolved_source.lakehouse_cache_max_bytes,
    })
}

pub(super) fn publish_lakehouse_optimized_batch_frame(
    context: &ExecutionContext,
    entry: &LakehouseOptimizedBatchCacheEntry,
    frame: &DataFrame,
) -> Result<bool> {
    context.check_cancellation("cudf_query_engine.lakehouse_cache.optimized_batch.publish")?;
    if entry.path.is_file() {
        return Ok(false);
    }
    if let Some(parent) = entry.path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            lakehouse_cache_io_error(
                error,
                "cudf_query_engine.lakehouse_cache.optimized_batch.mkdir",
                "failed to create optimized lakehouse cache parent directory",
                &entry.path,
                entry.stable_id.as_str(),
                None,
                None,
            )
        })?;
    }

    let lock_path = lakehouse_cache_lock_path(&entry.path);
    let _lock = acquire_lakehouse_cache_lock(context, &entry.path, &lock_path, entry.stable_id.as_str(), None, None)?;
    if entry.path.is_file() {
        return Ok(false);
    }

    let temp_path = lakehouse_cache_temp_path(&entry.path);
    let row_group_size_rows = optimized_batch_row_group_size_rows(frame.num_rows());
    let write_result = frame.write_parquet_with_options(
        cudf::frame::dataframe::ParquetWriteOptions::builder()
            .filepath(&temp_path)
            .row_group_size_rows(row_group_size_rows)
            .dictionary_policy(cudf::frame::dataframe::ParquetDictionaryPolicy::Adaptive)
            .with_runtime(context.runtime())
            .build(),
    );
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(QueryError::from_cudf(
            "cudf_query_engine.lakehouse_cache.optimized_batch.write",
            "failed to write optimized lakehouse cache batch parquet file",
            error,
        )
        .with_fact("path", entry.path.display())
        .with_fact("temp_path", temp_path.display())
        .with_fact("stable_id", entry.stable_id.as_str()));
    }
    if let Err(error) = fs::rename(&temp_path, &entry.path) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.optimized_batch.publish",
            "failed to atomically publish optimized lakehouse cache batch",
            &entry.path,
            entry.stable_id.as_str(),
            None,
            None,
        )
        .with_fact("temp_path", temp_path.display()));
    }
    evict_lakehouse_cache(entry.cache_root.as_deref(), &entry.path, entry.cache_max_bytes);
    Ok(true)
}

fn acquire_lakehouse_cache_read_pin(
    context: &ExecutionContext,
    cache_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<LakehouseCacheReadPin> {
    context.check_cancellation("cudf_query_engine.lakehouse_cache.read_pin")?;
    if !cache_path.is_file() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "cudf_query_engine.lakehouse_cache.read_pin",
            "lakehouse cache entry is missing before read pin acquisition",
        )
        .with_fact("path", cache_path.display())
        .with_fact("stable_id", stable_id));
    }
    let pin_path = lakehouse_cache_read_pin_path(cache_path);
    let mut pin_file = OpenOptions::new().write(true).create_new(true).open(&pin_path).map_err(|error| {
        lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.read_pin",
            "failed to create lakehouse cache read pin",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("pin_path", pin_path.display())
    })?;
    if let Err(error) = writeln!(pin_file, "pid={}", std::process::id()) {
        let _ = fs::remove_file(&pin_path);
        return Err(lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.read_pin",
            "failed to write lakehouse cache read pin",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("pin_path", pin_path.display()));
    }
    if !cache_path.is_file() {
        let _ = fs::remove_file(&pin_path);
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "cudf_query_engine.lakehouse_cache.read_pin",
            "lakehouse cache entry disappeared during read pin acquisition",
        )
        .with_fact("path", cache_path.display())
        .with_fact("pin_path", pin_path.display())
        .with_fact("stable_id", stable_id));
    }
    Ok(LakehouseCacheReadPin { path: Some(pin_path) })
}

fn local_cache_optimized_batch_eligible(source: &LocalParquetSource, batch: &crate::chunk::SourceChunkBatch) -> bool {
    const MIN_OPTIMIZED_BATCH_FILES: usize = 8;
    const MAX_OPTIMIZED_BATCH_SOURCE_READ_BYTES: u64 = 64 * 1024 * 1024;

    if !std::env::var("CUDF_LAKEHOUSE_OPTIMIZED_BATCH_CACHE").is_ok_and(|value| value == "1")
        || batch.chunk_count() < MIN_OPTIMIZED_BATCH_FILES
        || source.reader_predicate.is_some()
        || source.scan_limit.is_some()
        || !source.partition_columns.is_empty()
        || source.read_columns.is_empty()
        || !optimized_batch_uses_identity_source_frame(source)
        || !source_read_columns_are_fixed_width(source)
    {
        return false;
    }
    let Some(estimated_source_read_bytes) = batch.estimated_source_read_bytes else {
        return false;
    };
    if estimated_source_read_bytes > MAX_OPTIMIZED_BATCH_SOURCE_READ_BYTES {
        return false;
    }

    let Some(first) = batch.chunks.first() else {
        return false;
    };
    let cache_root = first.resolved_source.lakehouse_cache_root.as_deref();
    cache_root.is_some()
        && batch.chunks.iter().all(|chunk| {
            chunk.resolved_source.read_strategy == crate::source::ParquetReadStrategy::LocalCacheKvikio
                && chunk.resolved_source.lakehouse_cache_root.as_deref() == cache_root
                && !chunk.zero_column_count
                && chunk.scan_limit_rows.is_none()
                && !chunk.read_columns.is_empty()
                && chunk.partition_values.is_empty()
                && chunk.selects_all_known_file_row_groups()
        })
}

fn local_cache_optimized_batch_signature(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
) -> String {
    let mut signature = String::from(
        "optimized-local-cache-source-batch-v4-small-fixed-width-rowgroup-normalized-adaptive-dictionary\n",
    );
    signature.push_str(&format!("schema={:?}\n", source.schema));
    signature.push_str(&format!("read_columns={:?}\n", source.read_columns));
    signature.push_str(&format!("read_column_file_indices={:?}\n", source.read_column_file_indices));
    signature.push_str(&format!("projected_output_columns={:?}\n", source.projected_output_columns));
    signature
        .push_str(&format!("projected_output_column_source_refs={:?}\n", source.projected_output_column_source_refs));
    signature.push_str(&format!("pure_filter_columns={:?}\n", source.pure_filter_columns));
    signature.push_str(&format!("pure_filter_column_source_refs={:?}\n", source.pure_filter_column_source_refs));
    signature.push_str(&format!("missing_columns={:?}\n", source.missing_columns));
    let mut chunk_signatures = batch
        .chunks
        .iter()
        .map(|chunk| {
            format!(
                "stable_id={} storage_domain={:?} content_identity={:?} known_file_bytes={:?} row_groups={:?}",
                chunk.stable_id,
                chunk.resolved_source.storage_domain,
                chunk.resolved_source.content_identity,
                chunk.known_file_bytes,
                chunk.row_group_indexes
            )
        })
        .collect::<Vec<_>>();
    chunk_signatures.sort();
    for chunk_signature in chunk_signatures {
        signature.push_str("chunk=");
        signature.push_str(&chunk_signature);
        signature.push('\n');
    }
    signature
}

fn optimized_batch_uses_identity_source_frame(source: &LocalParquetSource) -> bool {
    source.missing_columns.is_empty()
        && source.pure_filter_columns.is_empty()
        && source.projected_output_columns == source.read_columns
        && source.projected_output_column_source_refs.len() == source.read_columns.len()
        && source
            .projected_output_column_source_refs
            .iter()
            .zip(&source.read_column_file_indices)
            .all(|(source_ref, read_index)| {
                matches!(source_ref, crate::source::SourceColumnRef::File(index) if index == read_index)
            })
}

fn source_read_columns_are_fixed_width(source: &LocalParquetSource) -> bool {
    source.read_columns.iter().all(|column| {
        source.schema.field_with_name(column).is_ok_and(|field| optimized_cache_fixed_width_type(field.data_type()))
    })
}

fn optimized_cache_fixed_width_type(data_type: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

fn optimized_batch_row_group_size_rows(row_count: usize) -> i32 {
    const TARGET_ROW_GROUPS: usize = 32;
    const MIN_ROWS_PER_GROUP: usize = 50_000;
    const MAX_ROWS_PER_GROUP: usize = 1_000_000;

    if row_count == 0 {
        return 0;
    }
    let rows = row_count.div_ceil(TARGET_ROW_GROUPS).clamp(MIN_ROWS_PER_GROUP, MAX_ROWS_PER_GROUP);
    i32::try_from(rows).unwrap_or(i32::MAX)
}

fn sanitize_cache_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') { ch } else { '_' })
        .collect::<String>();
    if sanitized.is_empty() { "unknown".to_string() } else { sanitized }
}

fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ensure_local_cache_parquet_entry(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    cache_path: &Path,
    estimated_fetch_bytes: Option<u64>,
    stable_id: &str,
    cache_root: Option<&str>,
    cache_max_bytes: Option<u64>,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    context.check_cancellation("cudf_query_engine.lakehouse_cache.ensure")?;
    if cache_path.is_file() {
        return Ok(());
    }
    if !matches!(location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.lakehouse_cache.location",
            "local lakehouse cache fill requires an object-store source location",
        )
        .with_fact("path", cache_path.display())
        .with_fact("stable_id", stable_id));
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            lakehouse_cache_io_error(
                error,
                "cudf_query_engine.lakehouse_cache.mkdir",
                "failed to create lakehouse cache parent directory",
                cache_path,
                stable_id,
                file_index,
                chunk_index,
            )
        })?;
    }

    let lock_path = lakehouse_cache_lock_path(cache_path);
    let _lock = acquire_lakehouse_cache_lock(context, cache_path, &lock_path, stable_id, file_index, chunk_index)?;
    if cache_path.is_file() {
        return Ok(());
    }

    let mut fetch =
        start_host_parquet_buffer_fetch(context, location, resolved_source, estimated_fetch_bytes)?.finish()?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    metrics.source_read_time += fetch.fetch_elapsed;
    publish_lakehouse_cache_file(cache_path, fetch.bytes(), stable_id, file_index, chunk_index)?;
    evict_lakehouse_cache(cache_root, cache_path, cache_max_bytes);
    fetch.release_after_decode();
    Ok(())
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
        return start_host_parquet_buffer_ranged_fetch(context, &file.location, &file.resolved_source, selector);
    }
    start_host_parquet_buffer_fetch(context, &file.location, &file.resolved_source, file.file_size_bytes)
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
        start_host_parquet_buffer_ranged_fetch(context, &chunk.location, &chunk.resolved_source, selector)?.finish()?
    } else {
        start_host_parquet_buffer_fetch(context, &chunk.location, &chunk.resolved_source, estimated_fetch_bytes)?
            .finish()?
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

struct LakehouseCacheLock {
    path: Option<PathBuf>,
}

impl Drop for LakehouseCacheLock {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

fn acquire_lakehouse_cache_lock(
    context: &ExecutionContext,
    cache_path: &Path,
    lock_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<LakehouseCacheLock> {
    const LOCK_WAIT: Duration = Duration::from_millis(100);
    const MAX_ATTEMPTS: usize = 600;

    for attempt in 0..MAX_ATTEMPTS {
        context.check_cancellation("cudf_query_engine.lakehouse_cache.lock")?;
        match OpenOptions::new().write(true).create_new(true).open(lock_path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                return Ok(LakehouseCacheLock { path: Some(lock_path.to_path_buf()) });
            },
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if cache_path.is_file() {
                    return Ok(LakehouseCacheLock { path: None });
                }
                if attempt + 1 < MAX_ATTEMPTS {
                    thread::sleep(LOCK_WAIT);
                    continue;
                }
                let mut error = QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "cudf_query_engine.lakehouse_cache.lock_timeout",
                    "timed out waiting for a lakehouse cache fill lock",
                )
                .with_fact("path", cache_path.display())
                .with_fact("lock_path", lock_path.display())
                .with_fact("stable_id", stable_id);
                if let Some(file_index) = file_index {
                    error = error.with_fact("file_index", file_index);
                }
                if let Some(chunk_index) = chunk_index {
                    error = error.with_fact("chunk_index", chunk_index);
                }
                return Err(error);
            },
            Err(error) => {
                return Err(lakehouse_cache_io_error(
                    error,
                    "cudf_query_engine.lakehouse_cache.lock",
                    "failed to create lakehouse cache fill lock",
                    cache_path,
                    stable_id,
                    file_index,
                    chunk_index,
                )
                .with_fact("lock_path", lock_path.display()));
            },
        }
    }
    unreachable!("lakehouse cache lock loop returns on timeout")
}

fn publish_lakehouse_cache_file(
    cache_path: &Path,
    bytes: &[u8],
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<()> {
    let temp_path = lakehouse_cache_temp_path(cache_path);
    let mut temp_file = OpenOptions::new().write(true).create_new(true).open(&temp_path).map_err(|error| {
        lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.temp",
            "failed to create lakehouse cache temp file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", temp_path.display())
    })?;
    if let Err(error) = temp_file.write_all(bytes).and_then(|()| temp_file.sync_all()) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.write",
            "failed to write lakehouse cache temp file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", temp_path.display()));
    }
    drop(temp_file);
    if let Err(error) = fs::rename(&temp_path, cache_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "cudf_query_engine.lakehouse_cache.publish",
            "failed to atomically publish lakehouse cache file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", temp_path.display()));
    }
    Ok(())
}

fn lakehouse_cache_lock_path(cache_path: &Path) -> PathBuf {
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!("{file_name}.lock"))
}

fn lakehouse_cache_temp_path(cache_path: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!(".{file_name}.tmp.{}.{}", std::process::id(), nonce))
}

fn lakehouse_cache_read_pin_path(cache_path: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let sequence = LAKEHOUSE_CACHE_READ_PIN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!(".{file_name}.read.{}.{}.{}.pin", std::process::id(), nonce, sequence))
}

#[derive(Debug)]
struct LakehouseCacheEntry {
    path: PathBuf,
    bytes: u64,
    last_used: SystemTime,
}

fn evict_lakehouse_cache(cache_root: Option<&str>, protected_path: &Path, max_bytes: Option<u64>) {
    let Some(max_bytes) = max_bytes.filter(|value| *value > 0) else {
        return;
    };
    let Some(cache_root) = cache_root else {
        return;
    };
    let root = Path::new(cache_root);
    let Ok(mut entries) = collect_lakehouse_cache_entries(root, protected_path) else {
        return;
    };
    let mut retained_bytes = entries.iter().map(|entry| entry.bytes).sum::<u64>();
    if retained_bytes <= max_bytes {
        return;
    }

    entries.sort_by(|left, right| left.last_used.cmp(&right.last_used).then_with(|| left.path.cmp(&right.path)));
    for entry in entries {
        if retained_bytes <= max_bytes {
            break;
        }
        if entry.path == protected_path || lakehouse_cache_entry_has_sidecar_lock(&entry.path) {
            continue;
        }
        if fs::remove_file(&entry.path).is_ok() {
            retained_bytes = retained_bytes.saturating_sub(entry.bytes);
        }
    }
}

fn collect_lakehouse_cache_entries(root: &Path, protected_path: &Path) -> std::io::Result<Vec<LakehouseCacheEntry>> {
    let mut entries = Vec::new();
    collect_lakehouse_cache_entries_inner(root, protected_path, &mut entries)?;
    Ok(entries)
}

fn collect_lakehouse_cache_entries_inner(
    dir: &Path,
    protected_path: &Path,
    entries: &mut Vec<LakehouseCacheEntry>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_lakehouse_cache_entries_inner(&path, protected_path, entries)?;
            continue;
        }
        if !metadata.is_file() || !is_lakehouse_cache_data_file(&path) {
            continue;
        }
        let last_used = if path == protected_path {
            SystemTime::now()
        } else {
            metadata.accessed().or_else(|_| metadata.modified()).unwrap_or(UNIX_EPOCH)
        };
        entries.push(LakehouseCacheEntry { path, bytes: metadata.len(), last_used });
    }
    Ok(())
}

fn is_lakehouse_cache_data_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    !(name.ends_with(".lock")
        || is_lakehouse_cache_read_pin_name(name)
        || (name.starts_with('.') && name.contains(".tmp.")))
}

fn lakehouse_cache_entry_has_sidecar_lock(path: &Path) -> bool {
    lakehouse_cache_lock_path(path).exists() || lakehouse_cache_entry_has_read_pin(path)
}

fn lakehouse_cache_entry_has_read_pin(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let prefix = format!(".{file_name}.read.");
    let Ok(entries) = fs::read_dir(parent) else {
        return true;
    };
    entries.filter_map(std::result::Result::ok).any(|entry| {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            return false;
        };
        name.starts_with(&prefix) && name.ends_with(".pin") && lakehouse_cache_read_pin_is_active(&entry.path(), name)
    })
}

fn is_lakehouse_cache_read_pin_name(name: &str) -> bool {
    name.starts_with('.') && name.contains(".read.") && name.ends_with(".pin")
}

fn lakehouse_cache_read_pin_is_active(pin_path: &Path, pin_name: &str) -> bool {
    let Some(read_suffix) = pin_name.rsplit_once(".read.").map(|(_, suffix)| suffix) else {
        return true;
    };
    let Some(pid_text) = read_suffix.split('.').next().filter(|value| !value.is_empty()) else {
        return true;
    };
    let Ok(pid) = pid_text.parse::<u32>() else {
        return true;
    };
    if pid == std::process::id() {
        return true;
    }
    if !Path::new("/proc").is_dir() {
        return true;
    }
    let active = Path::new("/proc").join(pid.to_string()).exists();
    if !active {
        let _ = fs::remove_file(pin_path);
    }
    active
}

fn lakehouse_cache_io_error(
    error: std::io::Error,
    operation: &'static str,
    message: &'static str,
    cache_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> QueryError {
    let mut error = QueryError::new(QueryErrorCode::CudfDependencyFailure, operation, message)
        .with_fact("path", cache_path.display())
        .with_fact("stable_id", stable_id)
        .with_source(error);
    if let Some(file_index) = file_index {
        error = error.with_fact("file_index", file_index);
    }
    if let Some(chunk_index) = chunk_index {
        error = error.with_fact("chunk_index", chunk_index);
    }
    error
}

fn start_host_parquet_buffer_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
) -> Result<ScheduledHostParquetFetch> {
    context.check_cancellation("cudf_query_engine.source_fetch.host_buffer")?;
    let fetch_location = source_fetch_location(location, resolved_source)?;
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
        storage_profile_id: fetch_location.storage_profile_id.clone(),
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
    resolved_source: &crate::source::ResolvedParquetSource,
    selector: ranged_parquet::RangedRowGroupSelector,
) -> Result<ScheduledHostParquetFetch> {
    context.check_cancellation("cudf_query_engine.source_fetch.host_buffer_ranged")?;
    let fetch_location = source_fetch_location(location, resolved_source)?;
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
        storage_profile_id: fetch_location.storage_profile_id.clone(),
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

fn source_fetch_location(
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
) -> Result<SourceFetchLocation> {
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
    parse_s3_source_uri(uri, resolved_source.storage_profile_id.clone())
}

fn parse_s3_source_uri(uri: &str, storage_profile_id: Option<String>) -> Result<SourceFetchLocation> {
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
    Ok(SourceFetchLocation {
        bucket: bucket.to_string(),
        region,
        key: object_store::path::Path::from(key),
        storage_profile_id,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::*;
    use crate::{
        chunk::SourceChunkBatch,
        plan::QueryPlanBuilder,
        source::{
            CredentialMode, LakehouseCacheStatus, LocalParquetFile, MissingColumn, ParquetSourceLocation,
            ResolvedParquetSource, SourceColumnRef, SourceNode,
        },
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
            resolved_source: source.files[0].resolved_source.clone(),
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
            source_batching_reason: None,
        }
    }

    fn optimized_cache_source() -> LocalParquetSource {
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/source.parquet")],
            Arc::new(Schema::new(vec![
                Field::new("c0", ArrowDataType::Int32, false),
                Field::new("c1", ArrowDataType::Int64, false),
            ])),
            vec!["c0".to_string(), "c1".to_string()],
        )
    }

    fn optimized_cache_batch() -> SourceChunkBatch {
        let chunks = (0..8).map(optimized_cache_chunk).collect::<Vec<_>>();
        SourceChunkBatch {
            batch_index: 0,
            chunks,
            estimated_source_read_bytes: Some(8 * 1024),
            source_batching_reason: "batched".to_string(),
        }
    }

    fn optimized_cache_chunk(index: usize) -> SourceChunk {
        let reader_path = format!("/tmp/cache/file-{index}.parquet");
        let resolved_source = ResolvedParquetSource::local_cache_kvikio(
            format!("s3://bucket/file-{index}.parquet"),
            reader_path.clone(),
            CredentialMode::SharedProcessEnv,
            LakehouseCacheStatus::Hit,
        )
        .with_storage_domain("bucket@endpoint")
        .with_content_identity(format!("size=1024,row_count=10,file={index}"))
        .with_lakehouse_cache_config("/tmp/lakehouse-cache", None);
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index: index,
            location: ParquetSourceLocation::object_store_uri(format!("s3://bucket/file-{index}.parquet")).unwrap(),
            path: PathBuf::from(reader_path),
            resolved_source,
            stable_id: format!("file-{index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(1024),
            known_row_group_bytes: Some(1024),
            known_projected_compressed_bytes: Some(512),
            known_projected_uncompressed_bytes: Some(1024),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(1024),
            estimated_source_read_bytes: Some(1024),
            exact_source_read_preflight_bytes: Some(1024),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![1024]),
            source_file_projected_compressed_bytes: Some(vec![512]),
            source_file_projected_uncompressed_bytes: Some(vec![1024]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![10]),
            read_columns: vec!["c0".to_string(), "c1".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
        }
    }

    #[test]
    fn optimized_batch_entry_requires_identity_source_frame() {
        let source = optimized_cache_source();
        let batch = optimized_cache_batch();
        assert!(local_cache_optimized_batch_entry(&source, &batch).is_none());
        assert!(optimized_batch_uses_identity_source_frame(&source));

        let mut alias_source = source.clone();
        alias_source.projected_output_columns = vec!["c1".to_string()];
        alias_source.projected_output_column_source_refs = vec![SourceColumnRef::File(1)];
        assert!(!optimized_batch_uses_identity_source_frame(&alias_source));
        assert!(local_cache_optimized_batch_entry(&alias_source, &batch).is_none());

        let pure_filter_source = source.clone().with_column_roles(vec!["c0".to_string()], vec!["c1".to_string()]);
        assert!(!optimized_batch_uses_identity_source_frame(&pure_filter_source));
        assert!(local_cache_optimized_batch_entry(&pure_filter_source, &batch).is_none());

        let mut missing_source = source;
        missing_source.missing_columns.push(MissingColumn {
            name: "missing".to_string(),
            data_type: ArrowDataType::Int32,
            nullable: true,
        });
        assert!(!optimized_batch_uses_identity_source_frame(&missing_source));
        assert!(local_cache_optimized_batch_entry(&missing_source, &batch).is_none());
    }

    #[test]
    fn parse_s3_source_uri_preserves_storage_profile_id() {
        let location = parse_s3_source_uri("s3://bucket/prefix/file.parquet", Some("rest-s3".to_string())).unwrap();

        assert_eq!(location.bucket, "bucket");
        assert_eq!(location.key.as_ref(), "prefix/file.parquet");
        assert_eq!(location.storage_profile_id.as_deref(), Some("rest-s3"));
    }

    fn unique_temp_cache_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("cudf-query-engine-{test_name}-{nonce}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_cache_entry(root: &Path, relative: &str, bytes: usize) -> PathBuf {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, vec![b'x'; bytes]).unwrap();
        path
    }

    #[test]
    fn lakehouse_cache_eviction_keeps_protected_entry_and_removes_old_data_files() {
        let root = unique_temp_cache_root("eviction");
        let old_a = write_cache_entry(&root, "v1/s3/bucket/content/a.parquet__h1", 4);
        let old_b = write_cache_entry(&root, "v1/s3/bucket/content/b.parquet__h2", 4);
        let protected = write_cache_entry(&root, "v1/s3/bucket/content/c.parquet__h3", 4);
        let lock_path = lakehouse_cache_lock_path(&old_b);
        fs::write(&lock_path, b"locked").unwrap();

        evict_lakehouse_cache(Some(root.to_str().unwrap()), &protected, Some(4));

        assert!(protected.exists());
        assert!(!old_a.exists());
        assert!(old_b.exists());
        assert!(lock_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lakehouse_cache_eviction_skips_active_read_pins() {
        let root = unique_temp_cache_root("read-pin-eviction");
        let old = write_cache_entry(&root, "v1/s3/bucket/content/old.parquet__h1", 8);
        let active = write_cache_entry(&root, "v1/s3/bucket/content/active.parquet__h2", 4);
        let protected = write_cache_entry(&root, "v1/s3/bucket/content/protected.parquet__h3", 4);
        let pin_path = lakehouse_cache_read_pin_path(&active);
        fs::write(&pin_path, b"pid=123").unwrap();

        let entry_paths = collect_lakehouse_cache_entries(&root, &protected)
            .unwrap()
            .into_iter()
            .map(|entry| entry.path)
            .collect::<Vec<_>>();
        assert!(entry_paths.contains(&old));
        assert!(entry_paths.contains(&active));
        assert!(entry_paths.contains(&protected));
        assert!(!entry_paths.contains(&pin_path));

        evict_lakehouse_cache(Some(root.to_str().unwrap()), &protected, Some(8));

        assert!(!old.exists());
        assert!(active.exists());
        assert!(protected.exists());
        assert!(pin_path.exists());

        let pin = LakehouseCacheReadPin { path: Some(pin_path.clone()) };
        drop(pin);
        assert!(!pin_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lakehouse_cache_read_pin_check_removes_stale_pid_pins_when_proc_is_available() {
        if !Path::new("/proc").is_dir() {
            return;
        }
        let root = unique_temp_cache_root("stale-read-pin");
        let entry = write_cache_entry(&root, "v1/s3/bucket/content/active.parquet__h2", 4);
        let file_name = entry.file_name().unwrap().to_str().unwrap();
        let stale_pin = entry.with_file_name(format!(".{file_name}.read.4294967295.1.0.pin"));
        fs::write(&stale_pin, b"pid=4294967295").unwrap();

        assert!(!lakehouse_cache_entry_has_read_pin(&entry));
        assert!(!stale_pin.exists());
        let _ = fs::remove_dir_all(root);
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
