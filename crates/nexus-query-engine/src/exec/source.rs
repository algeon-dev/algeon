use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    time::Duration,
};

use parquet::file::reader::FileReader;

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

pub use cudf_contract::{ResolvedRemoteParquetDatasourceProbe, probe_resolved_remote_parquet_datasource};
use cudf_contract::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
pub use gpu_resident::read_gpu_resident_source;
use host_buffer::*;
pub(super) use metrics::{format_row_groups, record_bounded_parquet_source_metrics};
use partition::inject_partition_columns;
use read::read_local_parquet_file;
pub(in crate::exec) use scheduler::{ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput};
pub(in crate::exec) use schema::local_parquet_execution_frame_schema;
use schema::{coerce_local_parquet_source_schema, materialize_missing_columns};
pub use validate::validate_local_parquet_source;
use zero_column_count::{read_zero_column_count_source, zero_column_count_chunk_frame_with_budget};

use crate::{
    exec::types::{GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey},
    metrics::SourceReadParquetEvidenceMetrics,
};

/// Read a local Parquet source into a GPU dataframe.
///
/// The source is validated before reading. When a compatible source cache is
/// configured on the context, lookup and retained-byte metrics are recorded;
/// otherwise the files are read directly.
pub fn read_local_parquet_source(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_local_parquet_source(source)?;

    if let Some(cache) = context.local_parquet_source_cache() {
        return read_cacheable_local_parquet_source(context, source_node_id, source, &cache, metrics);
    }

    read_local_parquet_source_uncached(context, source_node_id, source, metrics)
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
    source_node_id: NodeId,
    source: &LocalParquetSource,
    cache: &std::sync::Arc<crate::exec::GpuLocalParquetSourceCache>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if datafusion_order_tracking_enabled() {
        return read_local_parquet_source_uncached(context, source_node_id, source, metrics);
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
    let frame = read_local_parquet_source_uncached(context, source_node_id, source, metrics)?;
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
    source_node_id: NodeId,
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
        return read_object_store_parquet_source_with_fetch_pipeline(context, source_node_id, source, metrics);
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
            let frame =
                read_object_store_parquet_file(context, source, source_node_id, file, file_index, file_index, metrics)?;
            record_background_cache_file_fill_enqueue(context, source_node_id, source, file, file_index, metrics);
            frame
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
            let frame = read_local_parquet_file(
                context,
                source,
                source_node_id,
                file,
                file_index,
                &cudf_source,
                &_cudf_chunk,
                &columns,
                metrics,
            )?;
            record_background_cache_file_fill_enqueue(context, source_node_id, source, file, file_index, metrics);
            frame
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
    source_node_id: NodeId,
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
        let frame = finish_object_store_parquet_file_fetch(
            context,
            source,
            source_node_id,
            file,
            file_index,
            file_index,
            fetch,
            metrics,
        )?;
        record_background_cache_file_fill_enqueue(context, source_node_id, source, file, file_index, metrics);
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
        let fetch = start_object_store_parquet_file_fetch(context, source, &source.files[file_index])?;
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
            .with_fact("path", file_path_fact(file))
            .with_fact("stable_id", file_stable_id_fact(file))
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
        .with_fact("path", file_path_fact(file))
        .with_fact("stable_id", file_stable_id_fact(file))
        .with_fact("reason", "invalid_row_count_metadata")
    })?;
    usize::try_from(rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.row_count",
            "local parquet source row count exceeds platform usize range",
        )
        .with_fact("path", file_path_fact(file))
        .with_fact("stable_id", file_stable_id_fact(file))
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
            .with_fact("path", chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
        return zero_column_count_chunk_frame_with_budget(context, row_count, budget_bytes, budget_boundary, metrics)
            .map_err(|error| {
                error
                    .with_fact("path", chunk_path_fact(chunk))
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
        .with_fact("path", chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("budget_bytes", budget_bytes)
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    let token = context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            cudf_error_for_chunk_with_budget_details(
                "nexus_query_engine.local_parquet_chunk.budget",
                "failed to create bounded native local parquet source chunk budget token",
                error,
                chunk,
            )
            .with_fact("path", chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("read_columns", chunk.read_columns.len())
            .with_fact("row_groups", format_row_groups(chunk))
        })?;
    let read_timer = Instant::now();
    let (outcome, observed_row_group_count, host_buffer_fetch) = if chunk.resolved_source.uses_host_buffer() {
        let read = read_object_store_parquet_chunk(context, source, chunk, &cudf_source, &cudf_chunk, &token, metrics)?;
        record_background_cache_chunk_fill_enqueue(context, chunk, metrics);
        (
            read.outcome,
            read.observed_row_group_count.and_then(|count| usize::try_from(count).ok()),
            read.host_buffer_fetch,
        )
    } else {
        (
            {
                let outcome = DataFrame::read_parquet_source_chunk_with_budget_with_evidence(
                    context.runtime(),
                    &cudf_source,
                    &cudf_chunk,
                    SourceReadMode::ExecutionFrame,
                    &token,
                )
                .map_err(|error| {
                    cudf_error_for_chunk_with_budget_details(
                        "nexus_query_engine.local_parquet_chunk.read",
                        "failed to read bounded native local parquet source chunk with cudf",
                        error,
                        chunk,
                    )
                    .with_fact("path", chunk_path_fact(chunk))
                    .with_fact("chunk_index", chunk.chunk_index)
                    .with_fact("file_index", chunk.file_index)
                    .with_fact("read_columns", chunk.read_columns.len())
                    .with_fact("row_groups", format_row_groups(chunk))
                })?;
                record_background_cache_chunk_fill_enqueue(context, chunk, metrics);
                outcome
            },
            None,
            HostBufferFetchRuntimeMetrics::default(),
        )
    };
    metrics.record_budgeted_execution("nexus_query_engine.local_parquet_chunk.read", outcome.report());
    let read_evidence = outcome.value().read_evidence().cloned();
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value().into_frame();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    let frame = apply_source_chunk_scan_limit(context, chunk, frame, metrics)?;
    let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
    if !chunk.resolved_source.uses_host_buffer()
        && chunk.reads_all_row_groups
        && chunk.row_group_indexes.is_empty()
        && chunk.known_read_row_group_count().is_none()
        && let Some(row_group_count) = observed_chunk_row_group_count(chunk)
    {
        metrics.row_groups_selected += row_group_count as u64;
    }
    record_source_chunk_read_runtime(
        source,
        chunk,
        observed_row_group_count,
        frame.num_rows() as u64,
        read_elapsed,
        reader_dynamic_filter_count,
        &host_buffer_fetch,
        read_evidence.as_ref(),
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
    let outcome = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        context.runtime(),
        &cudf_source,
        &cudf_chunks,
        SourceReadMode::ExecutionFrame,
        &token,
    )
    .map_err(|error| {
        cudf_error_for_chunk_batch_with_budget_details(
            "nexus_query_engine.local_parquet_chunk_batch.read",
            "failed to read bounded native local parquet source chunk batch with cudf",
            error,
            &batch.chunks,
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("first_chunk_index", batch.chunks[0].chunk_index)
        .with_fact("first_file_index", batch.chunks[0].file_index)
    })?;
    for chunk in &batch.chunks {
        record_background_cache_chunk_fill_enqueue(context, chunk, metrics);
    }
    drop(cache_read_pins);
    metrics.record_budgeted_execution("nexus_query_engine.local_parquet_chunk_batch.read", outcome.report());
    let read_evidence = outcome.value().read_evidence().cloned();
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value().into_frame();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    record_source_chunk_batch_read_runtime(
        source,
        batch,
        "chunk_batch",
        frame.num_rows() as u64,
        read_elapsed,
        reader_dynamic_filter_count,
        read_evidence.as_ref(),
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

fn record_background_cache_file_fill_enqueue(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    metrics: &mut QueryMetrics,
) {
    let outcome = match enqueue_background_lakehouse_cache_file_fill(context, file) {
        Ok(BackgroundLakehouseCacheFillEnqueue::Enqueued) => "enqueued".to_string(),
        Ok(BackgroundLakehouseCacheFillEnqueue::Skipped(reason)) => reason.to_string(),
        Err(error) => {
            format!(
                "enqueue_failed_code_{}_kind_{}_status_{}_operation_{}",
                error.code(),
                error.kind(),
                error.status(),
                error.operation()
            )
        },
    };
    if file.resolved_source.lakehouse_cache_fill_mode == crate::source::LakehouseCacheFillMode::Background {
        record_background_cache_file_fill_enqueue_runtime(source_node_id, source, file, file_index, outcome, metrics);
    }
}

fn record_background_cache_chunk_fill_enqueue(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) {
    let outcome = match enqueue_background_lakehouse_cache_chunk_fill(context, chunk) {
        Ok(BackgroundLakehouseCacheFillEnqueue::Enqueued) => "enqueued".to_string(),
        Ok(BackgroundLakehouseCacheFillEnqueue::Skipped(reason)) => reason.to_string(),
        Err(error) => {
            format!(
                "enqueue_failed_code_{}_kind_{}_status_{}_operation_{}",
                error.code(),
                error.kind(),
                error.status(),
                error.operation()
            )
        },
    };
    if chunk.resolved_source.lakehouse_cache_fill_mode == crate::source::LakehouseCacheFillMode::Background {
        record_background_cache_chunk_fill_enqueue_runtime(chunk, outcome, metrics);
    }
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
        .with_fact("path", optimized_cache_path_fact(entry))
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
            .with_fact("path", optimized_cache_path_fact(entry))
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
        .with_fact("path", optimized_cache_path_fact(entry))
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
        None,
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let frame = append_source_order_columns(context, frame, first_chunk.chunk_index)?;
    Ok(GpuDataFrame::new(frame))
}

fn record_background_cache_file_fill_enqueue_runtime(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    outcome: String,
    metrics: &mut QueryMetrics,
) {
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: source_node_id.to_string(),
        read_kind: "background_cache_fill_file".to_string(),
        read_strategy: file.resolved_source.read_strategy.as_str().to_string(),
        host_buffer_fetch_mode: "none".to_string(),
        host_buffer_payload_bytes: 0,
        host_buffer_requested_range_count: 0,
        host_buffer_requested_range_bytes: 0,
        host_buffer_original_file_bytes: file.file_size_bytes.unwrap_or(0),
        host_buffer_mini_parquet_footer_bytes: 0,
        parquet_read_evidence: SourceReadParquetEvidenceMetrics::default(),
        batch_index: file_index as u64,
        chunk_count: 0,
        first_chunk_index: file_index as u64,
        last_chunk_index: file_index as u64,
        first_file_index: file_index as u64,
        last_file_index: file_index as u64,
        file_count: 1,
        row_group_count: file.row_groups.as_ref().map_or(0, |row_groups| row_groups.len() as u64),
        read_columns_per_file: source.read_columns.len() as u64,
        read_columns_total: source.read_columns.len() as u64,
        known_file_bytes: file.file_size_bytes.unwrap_or(0),
        estimated_source_read_bytes: file.file_size_bytes.unwrap_or(0),
        known_row_group_bytes: selected_file_metadata_sum(file.row_group_bytes.as_deref(), file.row_groups.as_deref()),
        known_projected_compressed_bytes: selected_file_metadata_sum(
            file.projected_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_projected_uncompressed_bytes: selected_file_metadata_sum(
            file.projected_row_group_uncompressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_pure_filter_compressed_bytes: selected_file_metadata_sum(
            file.pure_filter_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_pure_filter_uncompressed_bytes: selected_file_metadata_sum(
            file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        output_rows: 0,
        reader_predicate: false,
        reader_dynamic_filter_count: 0,
        source_batching_reason: outcome,
        source_chunk_planning_mode: "background_cache_fill".to_string(),
        source_chunk_planning_reason: file.resolved_source.lakehouse_cache_fill_mode.as_str().to_string(),
        elapsed: Duration::ZERO,
    });
}

fn record_background_cache_chunk_fill_enqueue_runtime(
    chunk: &SourceChunk,
    outcome: String,
    metrics: &mut QueryMetrics,
) {
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: chunk.source_node_id.to_string(),
        read_kind: "background_cache_fill_chunk".to_string(),
        read_strategy: chunk.resolved_source.read_strategy.as_str().to_string(),
        host_buffer_fetch_mode: "none".to_string(),
        host_buffer_payload_bytes: 0,
        host_buffer_requested_range_count: 0,
        host_buffer_requested_range_bytes: 0,
        host_buffer_original_file_bytes: chunk.known_file_bytes.unwrap_or(0),
        host_buffer_mini_parquet_footer_bytes: 0,
        parquet_read_evidence: SourceReadParquetEvidenceMetrics::default(),
        batch_index: chunk.chunk_index as u64,
        chunk_count: 0,
        first_chunk_index: chunk.chunk_index as u64,
        last_chunk_index: chunk.chunk_index as u64,
        first_file_index: chunk.file_index as u64,
        last_file_index: chunk.file_index as u64,
        file_count: 1,
        row_group_count: chunk.row_group_count() as u64,
        read_columns_per_file: chunk.read_columns.len() as u64,
        read_columns_total: chunk.read_columns.len() as u64,
        known_file_bytes: chunk.known_file_bytes.unwrap_or(0),
        estimated_source_read_bytes: chunk.known_file_bytes.unwrap_or(0),
        known_row_group_bytes: chunk.known_row_group_bytes.unwrap_or(0),
        known_projected_compressed_bytes: chunk.known_projected_compressed_bytes.unwrap_or(0),
        known_projected_uncompressed_bytes: chunk.known_projected_uncompressed_bytes.unwrap_or(0),
        known_pure_filter_compressed_bytes: chunk.known_pure_filter_compressed_bytes.unwrap_or(0),
        known_pure_filter_uncompressed_bytes: chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0),
        output_rows: 0,
        reader_predicate: false,
        reader_dynamic_filter_count: 0,
        source_batching_reason: outcome,
        source_chunk_planning_mode: "background_cache_fill".to_string(),
        source_chunk_planning_reason: chunk.resolved_source.lakehouse_cache_fill_mode.as_str().to_string(),
        elapsed: Duration::ZERO,
    });
}

fn record_source_chunk_read_runtime(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    observed_row_group_count: Option<usize>,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    host_buffer_fetch: &HostBufferFetchRuntimeMetrics,
    read_evidence: Option<&cudf::io::parquet::ParquetReadEvidence>,
    metrics: &mut QueryMetrics,
) {
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: chunk.source_node_id.to_string(),
        read_kind: "chunk".to_string(),
        read_strategy: chunk.resolved_source.read_strategy.as_str().to_string(),
        host_buffer_fetch_mode: if host_buffer_fetch.fetch_mode.is_empty() {
            "none".to_string()
        } else {
            host_buffer_fetch.fetch_mode.clone()
        },
        host_buffer_payload_bytes: host_buffer_fetch.payload_bytes,
        host_buffer_requested_range_count: host_buffer_fetch.requested_range_count,
        host_buffer_requested_range_bytes: host_buffer_fetch.requested_range_bytes,
        host_buffer_original_file_bytes: host_buffer_fetch.original_file_bytes,
        host_buffer_mini_parquet_footer_bytes: host_buffer_fetch.mini_parquet_footer_bytes,
        parquet_read_evidence: source_read_parquet_evidence_metrics(read_evidence, host_buffer_fetch),
        batch_index: chunk.chunk_index as u64,
        chunk_count: 1,
        first_chunk_index: chunk.chunk_index as u64,
        last_chunk_index: chunk.chunk_index as u64,
        first_file_index: chunk.file_index as u64,
        last_file_index: chunk.file_index as u64,
        file_count: 1,
        row_group_count: observed_row_group_count
            .or_else(|| observed_chunk_row_group_count(chunk))
            .unwrap_or_else(|| chunk.row_group_count()) as u64,
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
        source_chunk_planning_mode: chunk.source_chunk_planning_mode.as_str().to_string(),
        source_chunk_planning_reason: chunk.source_chunk_planning_reason.clone(),
        elapsed,
    });
}

fn observed_chunk_row_group_count(chunk: &SourceChunk) -> Option<usize> {
    chunk.known_read_row_group_count().or_else(|| {
        (chunk.reads_all_row_groups && chunk.path.is_file())
            .then(|| {
                parquet::file::reader::SerializedFileReader::try_from(chunk.path.as_path())
                    .ok()
                    .map(|reader| reader.metadata().num_row_groups())
            })
            .flatten()
    })
}

fn record_source_chunk_batch_read_runtime(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    read_kind: &'static str,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    read_evidence: Option<&cudf::io::parquet::ParquetReadEvidence>,
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
        host_buffer_fetch_mode: "none".to_string(),
        host_buffer_payload_bytes: 0,
        host_buffer_requested_range_count: 0,
        host_buffer_requested_range_bytes: 0,
        host_buffer_original_file_bytes: 0,
        host_buffer_mini_parquet_footer_bytes: 0,
        parquet_read_evidence: source_read_parquet_evidence_metrics(
            read_evidence,
            &HostBufferFetchRuntimeMetrics::default(),
        ),
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
        source_chunk_planning_mode: first_chunk.source_chunk_planning_mode.as_str().to_string(),
        source_chunk_planning_reason: first_chunk.source_chunk_planning_reason.clone(),
        elapsed,
    });
}

fn record_source_file_read_runtime(
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    observed_row_group_count: u64,
    output_rows: u64,
    elapsed: Duration,
    host_buffer_fetch: &HostBufferFetchRuntimeMetrics,
    read_evidence: Option<&cudf::io::parquet::ParquetReadEvidence>,
    metrics: &mut QueryMetrics,
) {
    metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        source_node_id: source_node_id.to_string(),
        read_kind: "file".to_string(),
        read_strategy: file.resolved_source.read_strategy.as_str().to_string(),
        host_buffer_fetch_mode: if host_buffer_fetch.fetch_mode.is_empty() {
            "none".to_string()
        } else {
            host_buffer_fetch.fetch_mode.clone()
        },
        host_buffer_payload_bytes: host_buffer_fetch.payload_bytes,
        host_buffer_requested_range_count: host_buffer_fetch.requested_range_count,
        host_buffer_requested_range_bytes: host_buffer_fetch.requested_range_bytes,
        host_buffer_original_file_bytes: host_buffer_fetch.original_file_bytes,
        host_buffer_mini_parquet_footer_bytes: host_buffer_fetch.mini_parquet_footer_bytes,
        parquet_read_evidence: source_read_parquet_evidence_metrics(read_evidence, host_buffer_fetch),
        batch_index: chunk_index as u64,
        chunk_count: 1,
        first_chunk_index: chunk_index as u64,
        last_chunk_index: chunk_index as u64,
        first_file_index: file_index as u64,
        last_file_index: file_index as u64,
        file_count: 1,
        row_group_count: observed_row_group_count,
        read_columns_per_file: source.read_columns.len() as u64,
        read_columns_total: source.read_columns.len() as u64,
        known_file_bytes: file.file_size_bytes.unwrap_or(0),
        estimated_source_read_bytes: best_file_source_read_bytes(file),
        known_row_group_bytes: selected_file_metadata_sum(file.row_group_bytes.as_deref(), file.row_groups.as_deref()),
        known_projected_compressed_bytes: selected_file_metadata_sum(
            file.projected_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_projected_uncompressed_bytes: selected_file_metadata_sum(
            file.projected_row_group_uncompressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_pure_filter_compressed_bytes: selected_file_metadata_sum(
            file.pure_filter_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        known_pure_filter_uncompressed_bytes: selected_file_metadata_sum(
            file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ),
        output_rows,
        reader_predicate: source.reader_predicate.is_some(),
        reader_dynamic_filter_count: 0,
        source_batching_reason: "not_evaluated".to_string(),
        source_chunk_planning_mode: "whole_file".to_string(),
        source_chunk_planning_reason: "whole_file_read".to_string(),
        elapsed,
    });
}

fn source_read_parquet_evidence_metrics(
    evidence: Option<&cudf::io::parquet::ParquetReadEvidence>,
    host_buffer_fetch: &HostBufferFetchRuntimeMetrics,
) -> SourceReadParquetEvidenceMetrics {
    let ranged_mini_parquet_bloom_status =
        (host_buffer_fetch.fetch_mode == "ranged_parquet").then_some("bloom_ineligible_ranged_mini_parquet");
    let Some(evidence) = evidence else {
        let mut metrics = SourceReadParquetEvidenceMetrics::default();
        if let Some(status) = ranged_mini_parquet_bloom_status {
            metrics.bloom_pruning_status = status.to_string();
        }
        return metrics;
    };
    let pruning = evidence.pruning;
    SourceReadParquetEvidenceMetrics {
        input_kind: evidence.input_kind.to_string(),
        remote_read_mode: evidence
            .requested_remote_read_mode
            .map(cudf::io::parquet::RemoteReadMode::as_str)
            .unwrap_or("none")
            .to_string(),
        backend: evidence.backend.to_string(),
        fallback_reason: evidence.fallback_reason.to_string(),
        backend_build_status: evidence.backend_build_status.as_str().to_string(),
        remote_bytes: evidence.remote_bytes,
        remote_range_count: evidence.remote_range_count,
        device_read_support: evidence.device_read_support.to_string(),
        pruning_input_row_groups: pruning.map(|pruning| pruning.input_row_groups),
        row_groups_after_stats_filter: pruning.and_then(|pruning| pruning.row_groups_after_stats_filter),
        row_groups_after_bloom_filter: pruning.and_then(|pruning| pruning.row_groups_after_bloom_filter),
        bloom_pruning_status: ranged_mini_parquet_bloom_status
            .or_else(|| bloom_pruning_status(pruning))
            .unwrap_or("not_reported")
            .to_string(),
    }
}

fn bloom_pruning_status(pruning: Option<cudf::io::parquet::ParquetPruningEvidence>) -> Option<&'static str> {
    let pruning = pruning?;
    match (pruning.row_groups_after_stats_filter, pruning.row_groups_after_bloom_filter) {
        (Some(after_stats), Some(after_bloom)) if after_bloom < after_stats => Some("bloom_pruned"),
        (None, Some(after_bloom)) if after_bloom < pruning.input_row_groups => Some("bloom_pruned"),
        (_, Some(_)) => Some("bloom_not_pruned"),
        (Some(_), None) => Some("bloom_not_reported"),
        (None, None) => Some("no_reader_predicate"),
    }
}

fn best_file_source_read_bytes(file: &LocalParquetFile) -> u64 {
    selected_file_metadata_sum(file.row_group_bytes.as_deref(), file.row_groups.as_deref())
        .max(selected_file_metadata_sum(
            file.projected_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ))
        .max(file.file_size_bytes.unwrap_or(0))
}

fn selected_file_metadata_sum(values: Option<&[u64]>, row_groups: Option<&[usize]>) -> u64 {
    let Some(values) = values else {
        return 0;
    };
    match row_groups {
        Some(row_groups) => row_groups.iter().filter_map(|index| values.get(*index)).sum(),
        None => values.iter().sum(),
    }
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
                crate::source::ParquetReadStrategy::LocalKvikio
                    | crate::source::ParquetReadStrategy::LocalCacheKvikio
                    | crate::source::ParquetReadStrategy::RemoteDirect
            )
            || (matches!(chunk.resolved_source.read_strategy, crate::source::ParquetReadStrategy::RemoteDirect)
                && chunk.resolved_source.remote_read_mode != crate::source::RemoteParquetReadMode::RequireDirect)
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
            .with_fact("path", chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("row_groups", format_row_groups(chunk)));
        }
    }
    Ok(())
}

fn chunk_path_fact(chunk: &SourceChunk) -> String {
    if chunk.location.is_local_path() {
        crate::source::redacted_path_diagnostic(&chunk.path)
    } else {
        chunk.location.diagnostic_location()
    }
}

fn file_path_fact(file: &LocalParquetFile) -> String {
    if file.location.is_local_path() {
        crate::source::redacted_path_diagnostic(&file.path)
    } else {
        file.location.diagnostic_location()
    }
}

fn file_stable_id_fact(file: &LocalParquetFile) -> String {
    file.location.diagnostic_stable_id(file.stable_id.as_str())
}

fn optimized_cache_path_fact(entry: &LakehouseOptimizedBatchCacheEntry) -> String {
    crate::source::redacted_path_diagnostic(&entry.path)
}

fn cudf_error_for_chunk_with_budget_details(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    chunk: &SourceChunk,
) -> QueryError {
    if chunk.location.is_local_path() {
        QueryError::from_cudf_with_budget_details(operation, message, error)
    } else {
        QueryError::from_cudf_without_source_with_budget_details(operation, message, &error)
    }
}

fn cudf_error_for_chunk_batch_with_budget_details(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    chunks: &[SourceChunk],
) -> QueryError {
    if chunks.iter().any(|chunk| !chunk.location.is_local_path()) {
        QueryError::from_cudf_without_source_with_budget_details(operation, message, &error)
    } else {
        QueryError::from_cudf_with_budget_details(operation, message, error)
    }
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
        .with_fact("path", chunk_path_fact(chunk))
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
    use crate::source::{CredentialMode, LakehouseCacheFillMode, MissingColumn, ResolvedParquetSource};

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

    #[test]
    fn remote_direct_chunk_budget_errors_redact_object_store_locations() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let resolved = ResolvedParquetSource::remote_direct(sensitive_uri, CredentialMode::SharedProcessEnv);
        let file = LocalParquetFile::from_resolved_parquet_source(resolved)
            .expect("valid remote source")
            .with_row_groups(vec![0])
            .with_row_group_rows(vec![1]);
        let source = test_source(vec![file], vec!["c1".to_string()]);
        let plan = crate::chunk::plan_local_parquet_source_chunks(
            NodeId::from_index_for_internal_use(1),
            &source,
            crate::chunk::SourceChunkPlanningPolicy::default(),
        )
        .expect("chunk planning succeeds");
        let chunk = plan.chunks.first().expect("one chunk");
        let cudf_error = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::IoFailure,
            "io::parquet::read_parquet_table",
            format!("failed to open {sensitive_uri}"),
        );

        let error = cudf_error_for_chunk_with_budget_details(
            "nexus_query_engine.local_parquet_chunk.read",
            "failed to read bounded native local parquet source chunk with cudf",
            cudf_error,
            chunk,
        )
        .with_fact("path", chunk_path_fact(chunk));
        let rendered = format!("{error}");
        let debug = format!("{error:?}");

        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(error.source_error().is_none());
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in display error: {rendered}");
            assert!(!debug.contains(sensitive), "{sensitive} leaked in debug error: {debug}");
        }
    }

    #[test]
    fn row_count_error_facts_redact_object_store_stable_id() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::object_store_uri(sensitive_uri).unwrap().with_stable_id(sensitive_uri);

        let error = local_parquet_file_row_count(&file).unwrap_err();
        let rendered = format!("{error:?}");

        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(
            error.fact_value("stable_id").is_some_and(|value| value.starts_with("object_store:s3:stable_id_fnv64:"))
        );
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in row-count error: {rendered}");
        }
    }

    #[test]
    fn source_validation_row_count_error_facts_redact_object_store_stable_id() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::object_store_uri(sensitive_uri).unwrap().with_stable_id(sensitive_uri);
        let mut source = test_source(vec![file], Vec::new());
        source.missing_columns.push(MissingColumn::new("missing", ArrowDataType::Int32, true));

        let error = validate_local_parquet_source(&source).unwrap_err();
        let rendered = format!("{error:?}");

        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
        assert!(
            error.fact_value("stable_id").is_some_and(|value| value.starts_with("object_store:s3:stable_id_fnv64:"))
        );
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!rendered.contains(sensitive), "{sensitive} leaked in validation error: {rendered}");
        }
    }

    #[test]
    fn background_cache_fill_enqueue_runtime_does_not_use_source_cache_insert_metrics() {
        let root = std::env::temp_dir().join(format!(
            "nexus-query-engine-background-fill-{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let cache_path = root.join("objects/v1/s3/bucket/content/a.parquet");
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source = ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/a.parquet",
            "s3://bucket/a.parquet",
            CredentialMode::SharedProcessEnv,
        )
        .with_lakehouse_cache_config(root.display().to_string(), None)
        .with_lakehouse_cache_path(cache_path.display().to_string())
        .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/a.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_file_size_bytes(1024);
        let source = test_source(vec![file.clone()], vec!["c1".to_string()]);
        let mut metrics = QueryMetrics::default();

        record_background_cache_file_fill_enqueue(
            &context,
            NodeId::from_index_for_internal_use(1),
            &source,
            &file,
            0,
            &mut metrics,
        );

        assert_eq!(metrics.source_cache_insert_skipped, 0);
        assert_eq!(metrics.source_read_runtime_rows.len(), 1);
        assert_eq!(metrics.source_read_runtime_rows[0].read_kind, "background_cache_fill_file");
        assert_eq!(metrics.source_read_runtime_rows[0].source_batching_reason, "fill_concurrency_saturated");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn parquet_evidence_metrics_report_ranged_mini_parquet_bloom_ineligible() {
        let read_evidence = cudf::io::parquet::ParquetReadEvidence {
            requested_remote_read_mode: None,
            input_kind: "host_buffer",
            backend: "host_buffer",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: cudf::io::parquet::RemoteParquetBackendBuildStatus::Unknown,
            pruning: Some(cudf::io::parquet::ParquetPruningEvidence {
                input_row_groups: 3,
                row_groups_after_stats_filter: None,
                row_groups_after_bloom_filter: None,
            }),
        };
        let host_buffer_fetch =
            HostBufferFetchRuntimeMetrics { fetch_mode: "ranged_parquet".to_string(), ..Default::default() };

        let metrics = source_read_parquet_evidence_metrics(Some(&read_evidence), &host_buffer_fetch);

        assert_eq!(metrics.pruning_input_row_groups, Some(3));
        assert_eq!(metrics.row_groups_after_stats_filter, None);
        assert_eq!(metrics.row_groups_after_bloom_filter, None);
        assert_eq!(metrics.bloom_pruning_status, "bloom_ineligible_ranged_mini_parquet");
    }

    #[test]
    fn parquet_evidence_metrics_compare_bloom_pruning_after_stats_pruning() {
        let read_evidence = cudf::io::parquet::ParquetReadEvidence {
            requested_remote_read_mode: None,
            input_kind: "local_path",
            backend: "libcudf_path_reader",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: cudf::io::parquet::RemoteParquetBackendBuildStatus::Unknown,
            pruning: Some(cudf::io::parquet::ParquetPruningEvidence {
                input_row_groups: 9,
                row_groups_after_stats_filter: Some(6),
                row_groups_after_bloom_filter: Some(6),
            }),
        };

        let metrics =
            source_read_parquet_evidence_metrics(Some(&read_evidence), &HostBufferFetchRuntimeMetrics::default());

        assert_eq!(metrics.pruning_input_row_groups, Some(9));
        assert_eq!(metrics.row_groups_after_stats_filter, Some(6));
        assert_eq!(metrics.row_groups_after_bloom_filter, Some(6));
        assert_eq!(metrics.bloom_pruning_status, "bloom_not_pruned");
    }
}
