use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    time::Duration,
};

use super::*;

mod validate;

mod cudf_contract;
mod gpu_resident;
mod host_buffer;
mod metrics;
mod partition;
mod predicate;
mod read;
mod scheduler;
mod schema;
mod zero_column_count;

use cudf_contract::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
pub use gpu_resident::read_gpu_resident_source;
use host_buffer::{
    LakehouseOptimizedBatchCacheEntry, ScheduledHostParquetFetch, acquire_lakehouse_cache_chunk_read_pin,
    acquire_lakehouse_cache_file_read_pin, acquire_lakehouse_optimized_batch_read_pin,
    ensure_local_cache_parquet_chunk, ensure_local_cache_parquet_file, finish_object_store_parquet_file_fetch,
    local_cache_optimized_batch_entry, publish_lakehouse_optimized_batch_frame, read_object_store_parquet_chunk,
    read_object_store_parquet_file, start_object_store_parquet_file_fetch,
};
pub(super) use metrics::{format_row_groups, record_bounded_parquet_source_metrics};
use partition::inject_partition_columns;
use read::read_local_parquet_file;
pub(in crate::exec) use scheduler::{ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput};
pub(in crate::exec) use schema::local_parquet_execution_frame_schema;
use schema::{coerce_local_parquet_source_schema, materialize_missing_columns};
pub use validate::validate_local_parquet_source;
use zero_column_count::{read_zero_column_count_source, zero_column_count_chunk_frame_with_budget};

use crate::exec::types::{GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey};

/// Read a local Parquet source into a GPU dataframe.
///
/// The source is validated before reading. When a compatible source cache is
/// configured on the context, lookup and retained-byte metrics are recorded;
/// otherwise the files are read directly.
pub fn read_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_local_parquet_source(source)?;

    if let Some(cache) = context.local_parquet_source_cache() {
        return read_cacheable_local_parquet_source(context, source, &cache, metrics);
    }

    read_local_parquet_source_uncached(context, source, metrics)
}

/// Import a literal host Arrow source into a GPU dataframe.
pub fn read_literal_batch_source(
    context: &ExecutionContext,
    source: &crate::source::LiteralBatchSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    crate::exec::validate_literal_batch_source(source)?;
    let mut frames = Vec::with_capacity(source.batches.len());
    for batch in &source.batches {
        context.check_cancellation("nexus_query_engine.literal_batch_source.read")?;
        let timer = Instant::now();
        let frame = DataFrame::from_record_batch(context.runtime(), batch).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.literal_batch_source.read",
                "failed to import literal Arrow batch into native cudf",
                error,
            )
            .with_fact("rows", batch.num_rows())
            .with_fact("columns", batch.num_columns())
        })?;
        metrics.operator_time += timer.elapsed();
        metrics.record_input_batch(frame.num_rows() as u64);
        metrics.boundary.record_host_to_device_upload(batch.num_rows() as u64, record_batch_memory_size(batch) as u64);
        frames.push(frame);
    }

    let mut frames = frames.into_iter();
    let frame = frames.next().expect("validated literal source contains at least one batch");
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.literal_batch_source.concat",
            "failed to concatenate native literal batches",
            error,
        )
        .with_fact("batch_count", source.batches.len())
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn read_cacheable_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    cache: &std::sync::Arc<crate::exec::GpuLocalParquetSourceCache>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if datafusion_order_tracking_enabled() {
        return read_local_parquet_source_uncached(context, source, metrics);
    }

    let key = GpuLocalParquetSourceCacheKey::from_source(source);
    let lookup_timer = Instant::now();
    let cached = cache.get(&key, context, &source.schema)?;
    metrics.record_source_cache_lookup_time(lookup_timer.elapsed());
    if let Some((handle, _retained_device_bytes)) = cached {
        metrics.record_source_cache_hit(cache.retained_device_bytes()?);
        return Ok(handle.into_gpu_frame());
    }

    metrics.record_source_cache_miss(cache.retained_device_bytes()?);
    let frame = read_local_parquet_source_uncached(context, source, metrics)?;
    let retained_device_bytes = frame.device_allocated_size_bytes()?;
    let handle = GpuResultHandle::new(context, frame.clone(), Arc::clone(&source.schema))?;
    match cache.insert_if_absent(key, handle, retained_device_bytes)? {
        GpuLocalParquetSourceCacheInsert::Inserted => {
            metrics.record_source_cache_insert(cache.retained_device_bytes()?);
        },
        GpuLocalParquetSourceCacheInsert::AlreadyPresent | GpuLocalParquetSourceCacheInsert::CapacitySkipped => {
            metrics.record_source_cache_insert_skipped(cache.retained_device_bytes()?);
        },
    }
    Ok(frame)
}

fn read_local_parquet_source_uncached(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if source.scan_limit == Some(0) {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, 0)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    }

    if source.is_zero_column_count_source() {
        return read_zero_column_count_source(context, source, metrics);
    }

    let columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    if should_pipeline_object_store_file_fetches(context, source, &columns) {
        return read_object_store_parquet_source_with_fetch_pipeline(context, source, metrics);
    }

    let mut frames = Vec::with_capacity(source.files.len());
    let mut remaining_rows = source.scan_limit;
    for (file_index, file) in source.files.iter().enumerate() {
        context.check_cancellation("nexus_query_engine.local_parquet_source.file")?;
        if file.row_groups.as_ref().is_some_and(Vec::is_empty) {
            continue;
        }
        let row_count_override = if columns.is_empty() { Some(local_parquet_file_row_count(file)?) } else { None };
        let frame = if columns.is_empty() {
            context.runtime().empty_dataframe()
        } else if file.resolved_source.uses_host_buffer() {
            read_object_store_parquet_file(context, source, file, file_index, file_index, metrics)?
        } else {
            let _cache_read_pin =
                if file.resolved_source.read_strategy == crate::source::ParquetReadStrategy::LocalCacheKvikio {
                    ensure_local_cache_parquet_file(context, file, metrics)?;
                    Some(acquire_lakehouse_cache_file_read_pin(context, file)?)
                } else {
                    None
                };
            let (cudf_source, _cudf_chunk) =
                cudf_local_parquet_source_file_selection(context, source, file, file_index, file_index)?;
            let reader_predicate = cudf_source.reader_predicate().cloned();
            read_local_parquet_file(context, file, &columns, reader_predicate.clone(), metrics)?
        };
        let frame = if file.resolved_source.uses_host_buffer() && !columns.is_empty() {
            frame
        } else {
            let frame = inject_partition_columns(context, source, &file.partition_values, frame, metrics)?.into_frame();
            materialize_missing_columns(context, source, frame, row_count_override, metrics)?
        };
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let (frame, remaining) = apply_source_scan_limit_to_frame(context, frame, remaining_rows, metrics)?;
        let frame = append_source_order_columns(context, frame, file_index)?;
        frames.push(frame);
        remaining_rows = remaining;
        if remaining_rows == Some(0) {
            break;
        }
    }

    let mut frames = frames.into_iter();
    let Some(frame) = frames.next() else {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, 0)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    };
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.concat",
            "failed to concatenate native local parquet files",
            error,
        )
        .with_fact("file_count", source.file_count())
        .with_fact("read_columns", source.read_column_count())
    })?;
    metrics.operator_time += timer.elapsed();

    Ok(GpuDataFrame::new(output))
}

fn should_pipeline_object_store_file_fetches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    columns: &[&str],
) -> bool {
    object_store_file_fetch_pipeline_enabled(
        context.source_scheduler_policy().max_source_reads_in_flight(),
        source,
        columns,
    )
}

fn object_store_file_fetch_pipeline_enabled(
    max_source_reads_in_flight: usize,
    source: &LocalParquetSource,
    columns: &[&str],
) -> bool {
    !columns.is_empty()
        && source.scan_limit.is_none()
        && source.files.len() > 1
        && max_source_reads_in_flight > 1
        && source.files.iter().all(|file| file.resolved_source.uses_host_buffer())
}

fn read_object_store_parquet_source_with_fetch_pipeline(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let max_in_flight = context.source_scheduler_policy().max_source_reads_in_flight();
    let mut next_file_index = 0usize;
    let mut pending = VecDeque::new();
    let mut frames = Vec::with_capacity(source.files.len());

    start_object_store_file_fetches(context, source, max_in_flight, &mut next_file_index, &mut pending)?;
    while let Some((file_index, fetch)) = pending.pop_front() {
        context.check_cancellation("nexus_query_engine.local_parquet_source.object_store_fetch_pipeline")?;
        let file = &source.files[file_index];
        let frame =
            finish_object_store_parquet_file_fetch(context, source, file, file_index, file_index, fetch, metrics)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, file_index)?;
        frames.push(frame);
        start_object_store_file_fetches(context, source, max_in_flight, &mut next_file_index, &mut pending)?;
    }

    concatenate_local_parquet_source_frames(context, source, frames, metrics)
}

fn start_object_store_file_fetches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    max_in_flight: usize,
    next_file_index: &mut usize,
    pending: &mut VecDeque<(usize, ScheduledHostParquetFetch)>,
) -> Result<()> {
    while pending.len() < max_in_flight && *next_file_index < source.files.len() {
        let file_index = *next_file_index;
        let fetch = start_object_store_parquet_file_fetch(context, &source.files[file_index], None)?;
        pending.push_back((file_index, fetch));
        *next_file_index += 1;
    }
    Ok(())
}

fn concatenate_local_parquet_source_frames(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    mut frames: Vec<DataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = frames.remove(0);
    if frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.concat",
            "failed to concatenate native local parquet files",
            error,
        )
        .with_fact("file_count", source.file_count())
        .with_fact("read_columns", source.read_column_count())
    })?;
    metrics.operator_time += timer.elapsed();

    Ok(GpuDataFrame::new(output))
}

fn local_parquet_file_row_count(file: &LocalParquetFile) -> Result<usize> {
    let rows = match (&file.row_groups, &file.row_group_rows) {
        (_, None) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source.row_count",
                "local parquet source with no read columns requires row-count metadata",
            )
            .with_fact("path", file.path.display())
            .with_fact("stable_id", file.stable_id.as_str())
            .with_fact("reason", "missing_row_count_metadata"));
        },
        (None, Some(row_group_rows)) => row_group_rows.iter().try_fold(0u64, |total, rows| total.checked_add(*rows)),
        (Some(row_groups), Some(row_group_rows)) => row_groups.iter().try_fold(0u64, |total, row_group| {
            row_group_rows.get(*row_group).and_then(|rows| total.checked_add(*rows))
        }),
    }
    .ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.row_count",
            "local parquet source row-count metadata is invalid or overflows",
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("reason", "invalid_row_count_metadata")
    })?;
    usize::try_from(rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.row_count",
            "local parquet source row count exceeds platform usize range",
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("rows", rows)
    })
}

pub(super) fn read_local_parquet_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    budget_bytes: u64,
    budget_boundary: &'static str,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    read_local_parquet_chunk_with_reader_dynamic_filter_count(
        context,
        source,
        chunk,
        budget_bytes,
        budget_boundary,
        0,
        metrics,
    )
}

pub(super) fn read_local_parquet_chunk_with_reader_dynamic_filter_count(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    budget_bytes: u64,
    budget_boundary: &'static str,
    reader_dynamic_filter_count: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    context.check_cancellation("nexus_query_engine.local_parquet_chunk.read")?;
    if chunk.scan_limit_rows == Some(0) {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    }

    if chunk.zero_column_count {
        let row_count = chunk.known_row_count.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_chunk.zero_column_count",
                "zero-column count chunk requires parquet row-count metadata",
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
        return zero_column_count_chunk_frame_with_budget(context, row_count, budget_bytes, budget_boundary, metrics)
            .map_err(|error| {
                error
                    .with_fact("path", chunk.path.display())
                    .with_fact("chunk_index", chunk.chunk_index)
                    .with_fact("file_index", chunk.file_index)
                    .with_fact("row_groups", format_row_groups(chunk))
            });
    }

    let _cache_read_pin = if chunk.resolved_source.read_strategy == crate::source::ParquetReadStrategy::LocalCacheKvikio
    {
        ensure_local_cache_parquet_chunk(context, chunk, metrics)?;
        Some(acquire_lakehouse_cache_chunk_read_pin(context, chunk)?)
    } else {
        None
    };
    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_chunk(context, source, chunk)?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk.budget",
            "bounded local parquet source chunk budget exceeds platform usize range",
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("budget_bytes", budget_bytes)
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    let token = context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.local_parquet_chunk.budget",
                "failed to create bounded native local parquet source chunk budget token",
                error,
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
    let read_timer = Instant::now();
    let outcome = if chunk.resolved_source.uses_host_buffer() {
        read_object_store_parquet_chunk(context, source, chunk, &cudf_source, &cudf_chunk, &token, metrics)?
    } else {
        DataFrame::read_parquet_source_chunk_with_budget(
            context.runtime(),
            &cudf_source,
            &cudf_chunk,
            SourceReadMode::ExecutionFrame,
            &token,
        )
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.local_parquet_chunk.read",
                "failed to read bounded native local parquet source chunk with cudf",
                error,
            )
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
        })?
    };
    metrics.record_budgeted_execution("nexus_query_engine.local_parquet_chunk.read", outcome.report());
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    let frame = apply_source_chunk_scan_limit(context, chunk, frame, metrics)?;
    let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
    record_source_chunk_read_runtime(
        source,
        chunk,
        frame.num_rows() as u64,
        read_elapsed,
        reader_dynamic_filter_count,
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

pub(super) fn read_local_parquet_chunk_batch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    budget_bytes: u64,
    budget_boundary: &'static str,
    reader_dynamic_filter_count: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source chunk batch has one child");
        return read_local_parquet_chunk_with_reader_dynamic_filter_count(
            context,
            source,
            chunk,
            budget_bytes,
            budget_boundary,
            reader_dynamic_filter_count,
            metrics,
        );
    }
    context.check_cancellation("nexus_query_engine.local_parquet_chunk_batch.read")?;
    validate_local_parquet_chunk_batch(source, batch)?;
    if let Some(entry) = local_cache_optimized_batch_entry(source, batch)
        && entry.path.is_file()
    {
        match read_optimized_local_cache_chunk_batch(
            context,
            source,
            batch,
            &entry,
            budget_bytes,
            budget_boundary,
            reader_dynamic_filter_count,
            metrics,
        ) {
            Ok(frame) => return Ok(frame),
            Err(_error) => {
                let _ = fs::remove_file(&entry.path);
            },
        }
    }
    let optimized_cache_entry = local_cache_optimized_batch_entry(source, batch);
    let mut cache_read_pins = Vec::with_capacity(batch.chunk_count());
    for chunk in &batch.chunks {
        if chunk.resolved_source.read_strategy == crate::source::ParquetReadStrategy::LocalCacheKvikio {
            ensure_local_cache_parquet_chunk(context, chunk, metrics)?;
            cache_read_pins.push(acquire_lakehouse_cache_chunk_read_pin(context, chunk)?);
        }
    }
    let (cudf_source, cudf_chunks) = cudf_local_parquet_source_chunk_batch(context, source, &batch.chunks)?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk_batch.budget",
            "bounded local parquet source chunk batch budget exceeds platform usize range",
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("budget_bytes", budget_bytes)
    })?;
    let token = context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.local_parquet_chunk_batch.budget",
                "failed to create bounded native local parquet source chunk batch budget token",
                error,
            )
            .with_fact("batch_index", batch.batch_index)
            .with_fact("chunk_count", batch.chunk_count())
        })?;

    let read_timer = Instant::now();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget(
        context.runtime(),
        &cudf_source,
        &cudf_chunks,
        SourceReadMode::ExecutionFrame,
        &token,
    )
    .map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "nexus_query_engine.local_parquet_chunk_batch.read",
            "failed to read bounded native local parquet source chunk batch with cudf",
            error,
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("first_chunk_index", batch.chunks[0].chunk_index)
        .with_fact("first_file_index", batch.chunks[0].file_index)
    })?;
    drop(cache_read_pins);
    metrics.record_budgeted_execution("nexus_query_engine.local_parquet_chunk_batch.read", outcome.report());
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    record_source_chunk_batch_read_runtime(
        source,
        batch,
        "chunk_batch",
        frame.num_rows() as u64,
        read_elapsed,
        reader_dynamic_filter_count,
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    if let Some(entry) = optimized_cache_entry.as_ref() {
        let _ = publish_lakehouse_optimized_batch_frame(context, entry, &frame);
    }
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let frame = append_source_order_columns(context, frame, first_chunk.chunk_index)?;
    Ok(GpuDataFrame::new(frame))
}

fn read_optimized_local_cache_chunk_batch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    entry: &LakehouseOptimizedBatchCacheEntry,
    budget_bytes: u64,
    budget_boundary: &'static str,
    reader_dynamic_filter_count: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    context.check_cancellation("nexus_query_engine.local_parquet_chunk_batch.optimized_cache_read")?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk_batch.optimized_cache_budget",
            "bounded optimized local parquet source chunk batch budget exceeds platform usize range",
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("budget_bytes", budget_bytes)
        .with_fact("path", entry.path.display())
    })?;
    let token = context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.local_parquet_chunk_batch.optimized_cache_budget",
                "failed to create bounded optimized local parquet source chunk batch budget token",
                error,
            )
            .with_fact("batch_index", batch.batch_index)
            .with_fact("chunk_count", batch.chunk_count())
            .with_fact("path", entry.path.display())
        })?;
    let read_columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let _cache_read_pin = acquire_lakehouse_optimized_batch_read_pin(context, entry)?;
    let read_timer = Instant::now();
    let outcome = DataFrame::read_parquet_with_budget(
        ParquetReadOptions::builder()
            .filepath(&entry.path)
            .columns(&read_columns)
            .with_runtime(context.runtime())
            .build(),
        &token,
    )
    .map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "nexus_query_engine.local_parquet_chunk_batch.optimized_cache_read",
            "failed to read optimized local cache parquet source chunk batch with cudf",
            error,
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("path", entry.path.display())
        .with_fact("stable_id", entry.stable_id.as_str())
    })?;
    metrics.record_budgeted_execution(
        "nexus_query_engine.local_parquet_chunk_batch.optimized_cache_read",
        outcome.report(),
    );
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    record_source_chunk_batch_read_runtime(
        source,
        batch,
        "optimized_chunk_batch",
        frame.num_rows() as u64,
        read_elapsed,
        reader_dynamic_filter_count,
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let frame = append_source_order_columns(context, frame, first_chunk.chunk_index)?;
    Ok(GpuDataFrame::new(frame))
}

fn record_source_chunk_read_runtime(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    metrics: &mut QueryMetrics,
) {
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: chunk.source_node_id.to_string(),
        read_kind: "chunk".to_string(),
        read_strategy: chunk.resolved_source.read_strategy.as_str().to_string(),
        batch_index: chunk.chunk_index as u64,
        chunk_count: 1,
        first_chunk_index: chunk.chunk_index as u64,
        last_chunk_index: chunk.chunk_index as u64,
        first_file_index: chunk.file_index as u64,
        last_file_index: chunk.file_index as u64,
        file_count: 1,
        row_group_count: chunk.row_group_count() as u64,
        read_columns_per_file: chunk.read_columns.len() as u64,
        read_columns_total: chunk.read_columns.len() as u64,
        known_file_bytes: chunk.known_file_bytes.unwrap_or(0),
        estimated_source_read_bytes: best_chunk_source_read_bytes(chunk),
        known_row_group_bytes: chunk.known_row_group_bytes.unwrap_or(0),
        known_projected_compressed_bytes: chunk.known_projected_compressed_bytes.unwrap_or(0),
        known_projected_uncompressed_bytes: chunk.known_projected_uncompressed_bytes.unwrap_or(0),
        known_pure_filter_compressed_bytes: chunk.known_pure_filter_compressed_bytes.unwrap_or(0),
        known_pure_filter_uncompressed_bytes: chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0),
        output_rows,
        reader_predicate: source.reader_predicate.is_some(),
        reader_dynamic_filter_count: reader_dynamic_filter_count as u64,
        source_batching_reason: chunk.source_batching_reason.clone().unwrap_or_else(|| "not_evaluated".to_string()),
        elapsed,
    });
}

fn record_source_chunk_batch_read_runtime(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    read_kind: &'static str,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    metrics: &mut QueryMetrics,
) {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let last_chunk = batch.chunks.last().expect("source chunk batch is non-empty");
    let file_count = batch.chunks.iter().map(|chunk| chunk.file_index).collect::<BTreeSet<_>>().len();
    let estimated_source_read_bytes = batch
        .estimated_source_read_bytes
        .unwrap_or_else(|| batch.chunks.iter().map(best_chunk_source_read_bytes).sum());
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: first_chunk.source_node_id.to_string(),
        read_kind: read_kind.to_string(),
        read_strategy: first_chunk.resolved_source.read_strategy.as_str().to_string(),
        batch_index: batch.batch_index as u64,
        chunk_count: batch.chunk_count() as u64,
        first_chunk_index: first_chunk.chunk_index as u64,
        last_chunk_index: last_chunk.chunk_index as u64,
        first_file_index: first_chunk.file_index as u64,
        last_file_index: last_chunk.file_index as u64,
        file_count: file_count as u64,
        row_group_count: batch.row_group_count() as u64,
        read_columns_per_file: source.read_column_count() as u64,
        read_columns_total: batch.chunks.iter().map(|chunk| chunk.read_columns.len() as u64).sum(),
        known_file_bytes: batch.chunks.iter().map(|chunk| chunk.known_file_bytes.unwrap_or(0)).sum(),
        estimated_source_read_bytes,
        known_row_group_bytes: batch.chunks.iter().map(|chunk| chunk.known_row_group_bytes.unwrap_or(0)).sum(),
        known_projected_compressed_bytes: batch
            .chunks
            .iter()
            .map(|chunk| chunk.known_projected_compressed_bytes.unwrap_or(0))
            .sum(),
        known_projected_uncompressed_bytes: batch
            .chunks
            .iter()
            .map(|chunk| chunk.known_projected_uncompressed_bytes.unwrap_or(0))
            .sum(),
        known_pure_filter_compressed_bytes: batch
            .chunks
            .iter()
            .map(|chunk| chunk.known_pure_filter_compressed_bytes.unwrap_or(0))
            .sum(),
        known_pure_filter_uncompressed_bytes: batch
            .chunks
            .iter()
            .map(|chunk| chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0))
            .sum(),
        output_rows,
        reader_predicate: source.reader_predicate.is_some(),
        reader_dynamic_filter_count: reader_dynamic_filter_count as u64,
        source_batching_reason: batch.source_batching_reason.clone(),
        elapsed,
    });
}

fn best_chunk_source_read_bytes(chunk: &SourceChunk) -> u64 {
    chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes)
        .unwrap_or(0)
}

fn validate_local_parquet_chunk_batch(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
) -> Result<()> {
    if source.reader_predicate.as_ref().is_some_and(|predicate| !source.reader_predicate_is_supported(predicate)) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk_batch",
            "source chunk batching requires a supported reader predicate",
        )
        .with_fact("reason", "unsupported_reader_predicate"));
    }
    if !source.partition_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk_batch",
            "source chunk batching does not yet materialize per-file partition columns",
        )
        .with_fact("reason", "partition_columns_not_supported")
        .with_fact("partition_column_count", source.partition_columns.len()));
    }
    for chunk in &batch.chunks {
        if chunk.zero_column_count
            || chunk.resolved_source.uses_host_buffer()
            || !matches!(
                chunk.resolved_source.read_strategy,
                crate::source::ParquetReadStrategy::LocalKvikio | crate::source::ParquetReadStrategy::LocalCacheKvikio
            )
            || chunk.scan_limit_rows.is_some()
            || chunk.reads_all_row_groups
            || chunk.row_group_indexes.is_empty()
            || chunk.read_columns.is_empty()
            || !chunk.partition_values.is_empty()
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_chunk_batch",
                "source chunk is not eligible for multi-source Parquet batching",
            )
            .with_fact("reason", "ineligible_chunk")
            .with_fact("path", chunk.path.display())
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("row_groups", format_row_groups(chunk)));
        }
    }
    Ok(())
}

fn empty_local_parquet_execution_frame(context: &ExecutionContext, source: &LocalParquetSource) -> Result<DataFrame> {
    let schema = local_parquet_execution_frame_schema(source)?;
    let columns = schema.fields().iter().map(|field| arrow::array::new_empty_array(field.data_type())).collect();
    let batch = arrow::record_batch::RecordBatch::try_new_with_options(
        std::sync::Arc::clone(&schema),
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.local_parquet_source.empty_scan_limit",
            "failed to construct zero-row local parquet source batch",
        )
        .with_source(error)
    })?;
    DataFrame::from_record_batch(context.runtime(), &batch).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.empty_scan_limit",
            "failed to upload zero-row local parquet source batch",
            error,
        )
    })
}

fn apply_source_scan_limit_to_frame(
    context: &ExecutionContext,
    frame: DataFrame,
    remaining_rows: Option<usize>,
    metrics: &mut QueryMetrics,
) -> Result<(DataFrame, Option<usize>)> {
    let Some(remaining_rows) = remaining_rows else {
        return Ok((frame, None));
    };
    let row_count = frame.num_rows();
    let keep_rows = row_count.min(remaining_rows);
    let remaining_rows = Some(remaining_rows.saturating_sub(keep_rows));
    if keep_rows == row_count {
        return Ok((frame, remaining_rows));
    }

    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let frame = frame.with_execution(&overrides).slice(0, keep_rows).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.scan_limit",
            "failed to apply native source scan limit",
            error,
        )
        .with_fact("input_rows", row_count)
        .with_fact("keep_rows", keep_rows)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok((frame, remaining_rows))
}

fn apply_source_chunk_scan_limit(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let Some(keep_rows) = chunk.scan_limit_rows else {
        return Ok(frame);
    };
    let row_count = frame.num_rows();
    if keep_rows >= row_count {
        return Ok(frame);
    }

    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let frame = frame.with_execution(&overrides).slice(0, keep_rows).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_chunk.scan_limit",
            "failed to apply native source chunk scan limit",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("input_rows", row_count)
        .with_fact("keep_rows", keep_rows)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source(files: Vec<LocalParquetFile>, read_columns: Vec<String>) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c1", ArrowDataType::Int32, true)]));
        LocalParquetSource::from_files(files, schema, read_columns)
    }

    #[test]
    fn object_store_file_fetch_pipeline_requires_multiple_object_store_files_without_scan_limit() {
        let files = vec![
            LocalParquetFile::object_store_uri("s3://bucket/a.parquet").unwrap(),
            LocalParquetFile::object_store_uri("s3://bucket/b.parquet").unwrap(),
        ];
        let source = test_source(files, vec!["c1".to_string()]);

        assert!(object_store_file_fetch_pipeline_enabled(2, &source, &["c1"]));
        assert!(!object_store_file_fetch_pipeline_enabled(1, &source, &["c1"]));
        assert!(!object_store_file_fetch_pipeline_enabled(2, &source, &[]));

        let mut limited = source.clone();
        limited.scan_limit = Some(1);
        assert!(!object_store_file_fetch_pipeline_enabled(2, &limited, &["c1"]));
    }

    #[test]
    fn object_store_file_fetch_pipeline_rejects_mixed_or_single_file_sources() {
        let mixed = test_source(
            vec![
                LocalParquetFile::object_store_uri("s3://bucket/a.parquet").unwrap(),
                LocalParquetFile::new("/tmp/b.parquet"),
            ],
            vec!["c1".to_string()],
        );
        assert!(!object_store_file_fetch_pipeline_enabled(2, &mixed, &["c1"]));

        let single = test_source(
            vec![LocalParquetFile::object_store_uri("s3://bucket/a.parquet").unwrap()],
            vec!["c1".to_string()],
        );
        assert!(!object_store_file_fetch_pipeline_enabled(2, &single, &["c1"]));
    }
}
