use std::collections::VecDeque;

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

use cudf_contract::{cudf_local_parquet_source_chunk, cudf_local_parquet_source_file_selection};
pub use gpu_resident::read_gpu_resident_source;
use host_buffer::{
    ScheduledHostParquetFetch, finish_object_store_parquet_file_fetch, read_object_store_parquet_chunk,
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
        context.check_cancellation("cudf_query_engine.literal_batch_source.read")?;
        let timer = Instant::now();
        let frame = DataFrame::from_record_batch(context.runtime(), batch).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.literal_batch_source.read",
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
            "cudf_query_engine.literal_batch_source.concat",
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
        context.check_cancellation("cudf_query_engine.local_parquet_source.file")?;
        let row_count_override = if columns.is_empty() { Some(local_parquet_file_row_count(file)?) } else { None };
        let frame = if columns.is_empty() {
            context.runtime().empty_dataframe()
        } else if matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
            read_object_store_parquet_file(context, source, file, file_index, file_index, metrics)?
        } else {
            let (cudf_source, _cudf_chunk) =
                cudf_local_parquet_source_file_selection(context, source, file, file_index, file_index)?;
            let reader_predicate = cudf_source.reader_predicate().cloned();
            read_local_parquet_file(context, file, &columns, reader_predicate.clone(), metrics)?
        };
        let frame = if matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. })
            && !columns.is_empty()
        {
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
    let frame = frames.next().expect("validated local parquet source contains at least one file");
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.concat",
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
        && source
            .files
            .iter()
            .all(|file| matches!(file.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }))
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
        context.check_cancellation("cudf_query_engine.local_parquet_source.object_store_fetch_pipeline")?;
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
        let fetch = start_object_store_parquet_file_fetch(context, &source.files[file_index])?;
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
            "cudf_query_engine.local_parquet_source.concat",
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
                "cudf_query_engine.local_parquet_source.row_count",
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
            "cudf_query_engine.local_parquet_source.row_count",
            "local parquet source row-count metadata is invalid or overflows",
        )
        .with_fact("path", file.path.display())
        .with_fact("stable_id", file.stable_id.as_str())
        .with_fact("reason", "invalid_row_count_metadata")
    })?;
    usize::try_from(rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.row_count",
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
    context.check_cancellation("cudf_query_engine.local_parquet_chunk.read")?;
    if chunk.zero_column_count {
        let row_count = chunk.known_row_count.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_chunk.zero_column_count",
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

    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_chunk(context, source, chunk)?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_chunk.budget",
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
                "cudf_query_engine.local_parquet_chunk.budget",
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
    let outcome = if matches!(chunk.location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        read_object_store_parquet_chunk(context, chunk, &cudf_source, &cudf_chunk, &token, metrics)?
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
                "cudf_query_engine.local_parquet_chunk.read",
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
    metrics.record_budgeted_execution("cudf_query_engine.local_parquet_chunk.read", outcome.report());
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    let frame = outcome.into_value();
    let frame = coerce_local_parquet_source_schema(frame, source, Some(&token), metrics)?;
    let frame = apply_source_chunk_scan_limit(context, chunk, frame, metrics)?;
    let frame = append_source_order_columns(context, frame, chunk.chunk_index)?;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
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
            "cudf_query_engine.local_parquet_source.scan_limit",
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
            "cudf_query_engine.local_parquet_chunk.scan_limit",
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
