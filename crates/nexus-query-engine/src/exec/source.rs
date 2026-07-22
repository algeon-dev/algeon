use std::{borrow::Cow, collections::BTreeSet, fs, time::Duration};

use parquet::file::reader::FileReader;

use super::*;

mod cache_runtime;
mod cudf_contract;
mod gpu_resident;
mod lakehouse_cache;
mod metrics;
mod partition;
mod position_delete;
mod predicate;
mod read;
mod read_runtime_metrics;
mod scan_limit;
mod scheduler;
mod schema;
mod zero_column_count;

use cache_runtime::*;
pub use cudf_contract::{ResolvedRemoteParquetDatasourceProbe, probe_resolved_remote_parquet_datasource};
use cudf_contract::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
pub use gpu_resident::read_gpu_resident_source;
use lakehouse_cache::*;
pub(super) use metrics::{SourceChunkRowGroupFacts, format_row_groups, record_bounded_parquet_source_metrics};
use partition::inject_partition_columns;
use position_delete::*;
use read::read_local_parquet_file;
use read_runtime_metrics::*;
pub(in crate::exec) use scan_limit::apply_source_scan_limit_to_frame;
use scan_limit::*;
pub(in crate::exec) use scheduler::{ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput};
pub(in crate::exec) use schema::local_parquet_execution_frame_schema;
use schema::{coerce_local_parquet_source_schema, materialize_missing_columns};
use zero_column_count::{read_zero_column_count_source, zero_column_count_chunk_frame_with_budget};

pub(in crate::exec) use crate::source::{
    local_parquet_file_row_group_metadata_count, parquet_file_path_fact, parquet_file_stable_id_fact,
};
use crate::{
    metrics::SourceReadParquetEvidenceMetrics,
    runtime::cache::local_parquet::{GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey},
    source::validate_local_parquet_source,
};

pub(in crate::exec) fn parquet_chunk_path_fact(chunk: &SourceChunk) -> String {
    crate::source::parquet_source_path_fact(&chunk.location, &chunk.path)
}

pub(in crate::exec) fn source_read_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(parquet_chunk_path_fact(chunk))
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request
}

pub(in crate::exec) fn source_read_batch_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return source_read_request(
            kind,
            memory_boundary,
            requested_bytes,
            budget_bytes,
            node_id,
            first_chunk,
            attempt,
            max_attempts,
        );
    }
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_attempt(attempt, max_attempts);
    let same_file =
        batch.chunks.iter().all(|chunk| chunk.file_index == first_chunk.file_index && chunk.path == first_chunk.path);
    if same_file {
        request = request.with_file_index(first_chunk.file_index).with_path(parquet_chunk_path_fact(first_chunk));
        if batch.chunks.iter().all(|chunk| !chunk.reads_all_row_groups) {
            let mut row_groups =
                batch.chunks.iter().flat_map(|chunk| chunk.row_group_indexes.iter().copied()).collect::<Vec<_>>();
            row_groups.sort_unstable();
            row_groups.dedup();
            request = request.with_row_groups(row_groups);
        }
    }
    request
}

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
    crate::source::validate_literal_batch_source(source)?;
    let mut frames = Vec::with_capacity(source.batches.len());
    for batch in &source.batches {
        context.check_cancellation("nexus_query_engine.literal_batch_source.read")?;
        let timer = Instant::now();
        let frame = DataFrame::from_record_batch(&context.runtime()?, batch).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.literal_batch_source.read",
                "failed to import literal Arrow batch into nexus native",
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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
    cache: &std::sync::Arc<crate::runtime::cache::local_parquet::GpuLocalParquetSourceCache>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if datafusion_order_tracking_enabled() {
        return read_local_parquet_source_uncached(context, source_node_id, source, metrics);
    }

    let key = GpuLocalParquetSourceCacheKey::from_source(source);
    let lookup_timer = Instant::now();
    let cached = cache.get(&key, context, source.output_schema())?;
    metrics.record_source_cache_lookup_time(lookup_timer.elapsed());
    if let Some((handle, _retained_device_bytes)) = cached {
        metrics.record_source_cache_hit(cache.retained_device_bytes()?);
        return Ok(handle.into_gpu_frame());
    }

    metrics.record_source_cache_miss(cache.retained_device_bytes()?);
    let frame = read_local_parquet_source_uncached(context, source_node_id, source, metrics)?;
    let retained_device_bytes = frame.device_allocated_size_bytes()?;
    let handle = GpuResultHandle::new(context, frame.clone(), Arc::clone(source.output_schema()))?;
    let insert = cache.insert_if_absent_pinned(context, key, handle, retained_device_bytes)?;
    match insert.outcome {
        GpuLocalParquetSourceCacheInsert::Inserted => {
            metrics.record_source_cache_insert(cache.retained_device_bytes()?);
            if let Some(handle) = insert.handle {
                return Ok(handle.into_gpu_frame());
            }
        },
        GpuLocalParquetSourceCacheInsert::AlreadyPresent | GpuLocalParquetSourceCacheInsert::CapacitySkipped => {
            metrics.record_source_cache_insert_skipped(cache.retained_device_bytes()?);
            if let Some(handle) = insert.handle {
                return Ok(handle.into_gpu_frame());
            }
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

    if source_requires_position_delete_chunk_reads(source) {
        return read_position_delete_parquet_source_as_chunks(context, source_node_id, source, metrics);
    }

    let columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut frames = Vec::with_capacity(source.files.len());
    let mut remaining_rows = source.scan_limit;
    for (file_index, file) in source.files.iter().enumerate() {
        context.check_cancellation("nexus_query_engine.local_parquet_source.file")?;
        if file.row_groups.as_ref().is_some_and(Vec::is_empty) {
            continue;
        }
        let row_count_override = if columns.is_empty() { Some(local_parquet_file_row_count(file)?) } else { None };
        let frame = if columns.is_empty() {
            context.runtime()?.empty_dataframe()
        } else {
            let _cache_read_pin =
                if file.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
                    ensure_local_cache_parquet_file(context, file)?;
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
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = apply_position_deletes_to_data_file(
            context,
            source,
            source_node_id,
            file,
            file_index,
            None,
            frame,
            metrics,
        )?;
        if source.is_count_star_probe_source() {
            let row_count = frame.num_rows() as u64;
            let frame = count_star_probe_weight_frame(context, row_count, false, metrics)?.into_frame();
            let frame = append_source_order_columns(context, frame, file_index)?;
            frames.push(frame);
            continue;
        }
        let frame =
            inject_partition_columns(context, source, &file.partition_values, frame, row_count_override, metrics)?
                .into_frame();
        let frame = materialize_missing_columns(context, source, frame, row_count_override, metrics)?;
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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
            .with_fact("path", parquet_file_path_fact(file))
            .with_fact("stable_id", parquet_file_stable_id_fact(file))
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
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("reason", "invalid_row_count_metadata")
    })?;
    usize::try_from(rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.row_count",
            "local parquet source row count exceeds platform usize range",
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
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
            .with_fact("path", parquet_chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_row_group_facts(chunk)
        })?;
        return zero_column_count_chunk_frame_with_budget(context, row_count, budget_bytes, budget_boundary, metrics)
            .map_err(|error| {
                error
                    .with_fact("path", parquet_chunk_path_fact(chunk))
                    .with_fact("chunk_index", chunk.chunk_index)
                    .with_fact("file_index", chunk.file_index)
                    .with_row_group_facts(chunk)
            });
    }

    let _cache_read_pin =
        if chunk.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
            ensure_local_cache_parquet_chunk(context, chunk)?;
            Some(acquire_lakehouse_cache_chunk_read_pin(context, chunk)?)
        } else {
            None
        };
    let file =
        crate::chunk::source_file_for_chunk(source, chunk, "nexus_query_engine.local_parquet_chunk.position_deletes")?;
    let row_position_range =
        if file.has_position_delete_files() { crate::chunk::source_chunk_row_position_range(chunk)? } else { None };
    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_chunk(context, source, chunk)?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_chunk.budget",
            "bounded local parquet source chunk budget exceeds platform usize range",
        )
        .with_fact("path", parquet_chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("budget_bytes", budget_bytes)
        .with_row_group_facts(chunk)
    })?;
    let token = context
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            error
                .with_fact("path", parquet_chunk_path_fact(chunk))
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("read_columns", chunk.read_columns.len())
                .with_row_group_facts(chunk)
        })?;
    let _io_lease = context.acquire_io_lease("nexus_query_engine.local_parquet_chunk.read")?;
    let read_timer = Instant::now();
    let outcome = DataFrame::read_parquet_source_chunk_with_budget_with_evidence(
        &context.runtime()?,
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
        .with_fact("path", parquet_chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_row_group_facts(chunk)
    })?;
    record_background_cache_chunk_fill_enqueue(context, chunk, metrics);
    let observed_row_group_count = None;
    metrics.record_budgeted_execution("nexus_query_engine.local_parquet_chunk.read", outcome.report());
    let read_evidence = outcome.value().read_evidence().cloned();
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value().into_frame();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    let frame = apply_position_deletes_to_data_file(
        context,
        source,
        chunk.source_node_id,
        file,
        chunk.file_index,
        row_position_range,
        frame,
        metrics,
    )?;
    if source.is_count_star_probe_source() {
        let row_count = frame.num_rows() as u64;
        record_source_chunk_read_runtime(
            source,
            chunk,
            observed_row_group_count,
            row_count,
            read_elapsed,
            reader_dynamic_filter_count,
            read_evidence.as_ref(),
            metrics,
        );
        return count_star_probe_weight_frame(context, row_count, true, metrics);
    }
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    let frame = apply_source_chunk_scan_limit(context, chunk, frame, metrics)?;
    let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
    if chunk.reads_all_row_groups
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
        read_evidence.as_ref(),
        metrics,
    );
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

fn count_star_probe_weight_frame(
    context: &ExecutionContext,
    row_count: u64,
    record_input_rows: bool,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let count = i64::try_from(row_count).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.local_parquet_source.count_star_probe",
            "parquet row count exceeds native Int64 count representation",
        )
        .with_fact("row_count", row_count)
        .with_fact("max_row_count", i64::MAX)
    })?;
    let timer = Instant::now();
    let series = context.runtime()?.series(&[count]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.count_star_probe",
            "failed to construct native count-star probe weight input",
            error,
        )
    })?;
    let frame = DataFrame::from_series([(crate::source::ZERO_COLUMN_COUNT_COLUMN, series)]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.count_star_probe",
            "failed to construct native count-star probe weight frame",
            error,
        )
    })?;
    metrics.source_read_time += timer.elapsed();
    if record_input_rows {
        metrics.record_input_batch(row_count);
    }
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
        if chunk.resolved_source.read_strategy() == crate::source::ParquetReadStrategy::LocalCacheKvikio {
            ensure_local_cache_parquet_chunk(context, chunk)?;
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
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            error.with_fact("batch_index", batch.batch_index).with_fact("chunk_count", batch.chunk_count())
        })?;

    let _io_lease = context.acquire_io_lease("nexus_query_engine.local_parquet_chunk_batch.read")?;
    let read_timer = Instant::now();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        &context.runtime()?,
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
    if source.is_count_star_probe_source() {
        let row_count = frame.num_rows() as u64;
        record_source_chunk_batch_read_runtime(
            source,
            batch,
            "chunk_batch",
            row_count,
            read_elapsed,
            reader_dynamic_filter_count,
            read_evidence.as_ref(),
            metrics,
        );
        let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
        let frame = count_star_probe_weight_frame(context, row_count, true, metrics)?.into_frame();
        let frame = append_source_order_columns(context, frame, first_chunk.chunk_index)?;
        return Ok(GpuDataFrame::new(frame));
    }
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
            || !matches!(
                chunk.resolved_source.read_strategy(),
                crate::source::ParquetReadStrategy::LocalKvikio
                    | crate::source::ParquetReadStrategy::LocalCacheKvikio
                    | crate::source::ParquetReadStrategy::RemoteDirect
            )
            || chunk.scan_limit_rows.is_some()
            || chunk.reads_all_row_groups
            || chunk.row_group_indexes.is_empty()
            || chunk.read_columns.is_empty()
            || !chunk.partition_values.is_empty()
            || source.files.get(chunk.file_index).is_some_and(LocalParquetFile::has_row_delete_files)
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_chunk_batch",
                "source chunk is not eligible for multi-source Parquet batching",
            )
            .with_fact("reason", "ineligible_chunk")
            .with_fact("path", parquet_chunk_path_fact(chunk))
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_row_group_facts(chunk));
        }
    }
    Ok(())
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

fn cudf_error_for_chunk_read(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    chunk: &SourceChunk,
) -> QueryError {
    if chunk.location.is_local_path() {
        QueryError::from_cudf(operation, message, error)
    } else {
        QueryError::from_cudf_without_source(operation, message, &error)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{
        CredentialMode, LakehouseCacheFillMode, MissingColumn, ParquetEqualityDeleteFile, ResolvedParquetSource,
    };

    fn test_source(files: Vec<LocalParquetFile>, read_columns: Vec<String>) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c1", ArrowDataType::Int32, true)]));
        LocalParquetSource::from_files(files, schema, read_columns)
    }

    #[test]
    fn source_read_batch_request_preserves_shared_file_row_group_context() {
        let source = test_source(
            vec![
                LocalParquetFile::new("/tmp/source-read-request.parquet")
                    .with_row_groups(vec![0, 1])
                    .with_row_group_rows(vec![10, 20]),
            ],
            vec!["c1".to_string()],
        );
        let node_id = NodeId::from_index_for_internal_use(7);
        let plan = crate::chunk::plan_local_parquet_source_chunks(
            node_id,
            &source,
            crate::chunk::SourceChunkPlanningPolicy::new(None, 1),
        )
        .expect("chunk planning should split the file by row group");
        assert_eq!(plan.chunks.len(), 2);

        let batch = SourceChunkBatch {
            batch_index: 0,
            chunks: plan.chunks,
            estimated_source_read_bytes: Some(30),
            source_batching_reason: "batched".to_string(),
        };
        let request =
            source_read_batch_request(ReservationKind::SourceRead, "source_read", 30, 100, node_id, &batch, 2, 5);

        assert_eq!(request.node_id.as_deref(), Some("n7"));
        assert_eq!(request.stage.as_deref(), Some("source_read"));
        assert_eq!(request.file_index, Some(0));
        assert_eq!(request.chunk_index, None);
        assert_eq!(request.path.as_deref(), Some("/tmp/source-read-request.parquet"));
        assert_eq!(request.row_groups, Some(vec![0, 1]));
        assert_eq!(request.attempt, Some(2));
        assert_eq!(request.max_attempts, Some(5));
    }

    #[test]
    fn local_parquet_chunk_batch_rejects_equality_delete_files() {
        let delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
            "delete-000",
            ResolvedParquetSource::local_kvikio("/tmp/delete-000.parquet"),
            vec![1],
            vec!["c1".to_string()],
        )
        .unwrap();
        let source = test_source(
            vec![
                LocalParquetFile::new("/tmp/source-read-equality-delete.parquet")
                    .with_row_groups(vec![0, 1])
                    .with_row_group_bytes(vec![10, 10])
                    .with_projected_row_group_bytes(vec![10, 10], vec![10, 10])
                    .with_row_group_rows(vec![10, 10])
                    .with_equality_delete_files(vec![delete_file]),
            ],
            vec!["c1".to_string()],
        );
        let node_id = NodeId::from_index_for_internal_use(7);
        let plan = crate::chunk::plan_local_parquet_source_chunks(
            node_id,
            &source,
            crate::chunk::SourceChunkPlanningPolicy::new(None, 1),
        )
        .expect("chunk planning should split the file by row group");
        assert_eq!(plan.chunks.len(), 2);
        let batch = SourceChunkBatch {
            batch_index: 0,
            chunks: vec![plan.chunks[0].clone()],
            estimated_source_read_bytes: Some(10),
            source_batching_reason: "batched".to_string(),
        };

        let error = validate_local_parquet_chunk_batch(&source, &batch).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("ineligible_chunk"));
    }

    #[test]
    fn position_delete_referenced_data_file_matches_local_file_uri_variants() {
        let delete_file = crate::source::ParquetPositionDeleteFile::from_resolved_parquet_source(
            "delete-1",
            ResolvedParquetSource::local_kvikio("/tmp/delete.parquet"),
        )
        .unwrap()
        .with_referenced_data_file("file:///tmp/data.parquet");

        assert!(position_delete_file_matches_referenced_data_file(&delete_file, "/tmp/data.parquet"));

        let localhost_delete_file = delete_file.with_referenced_data_file("file://localhost/tmp/data.parquet");
        assert!(position_delete_file_matches_referenced_data_file(&localhost_delete_file, "/tmp/data.parquet"));
        assert!(!position_delete_file_matches_referenced_data_file(&localhost_delete_file, "/tmp/other.parquet"));
    }

    #[test]
    fn position_delete_file_path_filter_values_include_local_file_uri_variants_without_referenced_file() {
        let delete_file = crate::source::ParquetPositionDeleteFile::from_resolved_parquet_source(
            "delete-1",
            ResolvedParquetSource::local_kvikio("/tmp/delete.parquet"),
        )
        .unwrap();

        let values = position_delete_file_path_filter_values(&delete_file, "/tmp/data.parquet");

        assert!(values.contains(&"/tmp/data.parquet".to_string()));
        assert!(values.contains(&"file:/tmp/data.parquet".to_string()));
        assert!(values.contains(&"file:///tmp/data.parquet".to_string()));
        assert!(values.contains(&"file://localhost/tmp/data.parquet".to_string()));
    }

    #[test]
    fn remote_direct_chunk_budget_errors_redact_object_store_locations() {
        let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let resolved = ResolvedParquetSource::remote_kvikio(sensitive_uri, CredentialMode::SharedProcessEnv);
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
        .with_fact("path", parquet_chunk_path_fact(chunk));
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
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/a.parquet", CredentialMode::SharedProcessEnv)
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

        let metrics = source_read_parquet_evidence_metrics(Some(&read_evidence));

        assert_eq!(metrics.pruning_input_row_groups, Some(9));
        assert_eq!(metrics.row_groups_after_stats_filter, Some(6));
        assert_eq!(metrics.row_groups_after_bloom_filter, Some(6));
        assert_eq!(metrics.bloom_pruning_status, "bloom_not_pruned");
    }
}
