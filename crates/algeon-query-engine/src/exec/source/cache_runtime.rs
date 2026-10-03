use std::time::Instant;

use query_planning::source::LakehouseCacheFillMode;

use super::{
    BackgroundLakehouseCacheFillEnqueue, DataFrame, ExecutionBudgetRequest, ExecutionContext, ExecutionMetrics,
    GpuDataFrame, LakehouseOptimizedBatchCacheEntry, LocalParquetFile, LocalParquetSource, NodeId, ParquetReadOptions,
    QueryError, QueryErrorCode, Result, SourceChunk, SourceReadFact, acquire_lakehouse_optimized_batch_read_pin,
    append_source_order_columns, coerce_local_parquet_source_schema, cudf_query_error,
    enqueue_background_lakehouse_cache_chunk_fill, enqueue_background_lakehouse_cache_file_fill,
    optimized_cache_path_fact, record_background_cache_chunk_fill_enqueue_runtime,
    record_background_cache_file_fill_enqueue_runtime, record_source_chunk_batch_read_runtime, record_source_read,
};
use crate::chunk::SourceChunkBatch;

pub(super) fn record_background_cache_file_fill_enqueue(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
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
    if file.resolved_source.lakehouse_cache_fill_mode == LakehouseCacheFillMode::Background {
        record_background_cache_file_fill_enqueue_runtime(source_node_id, source, file, file_index, outcome, context);
    }
}

pub(super) fn record_background_cache_chunk_fill_enqueue(context: &ExecutionContext, chunk: &SourceChunk) {
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
    if chunk.resolved_source.lakehouse_cache_fill_mode == LakehouseCacheFillMode::Background {
        record_background_cache_chunk_fill_enqueue_runtime(chunk, outcome, context);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
)]
pub(super) fn read_optimized_local_cache_chunk_batch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    batch: &SourceChunkBatch,
    entry: &LakehouseOptimizedBatchCacheEntry,
    budget_bytes: u64,
    budget_boundary: &'static str,
    reader_dynamic_filter_count: usize,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    context.check_cancellation("query_engine.local_parquet_chunk_batch.optimized_cache_read")?;
    let budget_bytes = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_chunk_batch.optimized_cache_budget",
            "bounded optimized local parquet source chunk batch budget exceeds platform usize range",
        )
        .with_fact("batch_index", batch.batch_index)
        .with_fact("chunk_count", batch.chunk_count())
        .with_fact("budget_bytes", budget_bytes)
        .with_fact("path", optimized_cache_path_fact(entry))
    })?;
    let token = context
        .execution_budget_token(ExecutionBudgetRequest::new(budget_bytes).with_boundary(budget_boundary))
        .map_err(|error| {
            error
                .with_fact("batch_index", batch.batch_index)
                .with_fact("chunk_count", batch.chunk_count())
                .with_fact("path", optimized_cache_path_fact(entry))
        })?;
    let read_columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let _cache_read_pin = acquire_lakehouse_optimized_batch_read_pin(context, entry)?;
    let _io_lease = context.acquire_io_lease("query_engine.local_parquet_chunk_batch.optimized_cache_read")?;
    let read_timer = Instant::now();
    let runtime = context.runtime()?;
    let outcome = token
        .run("dataframe::read_parquet", |exec| {
            DataFrame::read_parquet(
                ParquetReadOptions::builder()
                    .filepath(&entry.path)
                    .columns(&read_columns)
                    .with_runtime(&runtime)
                    .build(),
                exec,
            )
        })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_chunk_batch.optimized_cache_read",
                "failed to read optimized local cache parquet source chunk batch with cudf",
                error,
            )
            .with_fact("batch_index", batch.batch_index)
            .with_fact("chunk_count", batch.chunk_count())
            .with_fact("path", optimized_cache_path_fact(entry))
            .with_fact("stable_id", entry.stable_id.as_str())
        })?;
    metrics.record_budgeted_execution("query_engine.local_parquet_chunk_batch.optimized_cache_read", outcome.report());
    let read_elapsed = read_timer.elapsed();
    record_source_read(context, SourceReadFact::ParquetReadTime(read_elapsed));
    record_source_read(context, SourceReadFact::SourceReadTime(read_elapsed));
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
        context,
    );
    record_source_read(context, SourceReadFact::InputBatch { rows: frame.num_rows() as u64 });
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let frame = append_source_order_columns(context, frame, first_chunk.chunk_index)?;
    Ok(GpuDataFrame::new(frame))
}
