use std::time::Instant;

use super::{
    DataFrame, ExecutionBudgetRequest, ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetSource, NodeId,
    QueryError, QueryErrorCode, Result, SourceChunkPlanningPolicy, ZERO_COLUMN_COUNT_COLUMN, attach_row_group_facts,
    cudf_query_error, parquet_chunk_path_fact, plan_local_parquet_source_chunks_with_execution_context,
    record_bounded_parquet_source_metrics,
};

pub(super) fn read_zero_column_count_source(
    context: &ExecutionContext,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        source_node_id,
        source,
        SourceChunkPlanningPolicy::default(),
        context,
    )?;
    let counts = chunk_plan
        .chunks
        .iter()
        .map(|chunk| {
            chunk.known_row_count.ok_or_else(|| {
                attach_row_group_facts(
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.zero_column_count.read",
                        "zero-column count source requires parquet row-count metadata",
                    )
                    .with_fact("path", parquet_chunk_path_fact(chunk))
                    .with_fact("chunk_index", chunk.chunk_index)
                    .with_fact("file_index", chunk.file_index),
                    chunk,
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    zero_column_count_frame(context, &counts, metrics)
}

pub(super) fn zero_column_count_frame(
    context: &ExecutionContext,
    row_counts: &[u64],
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let counts = row_counts
        .iter()
        .map(|row_count| {
            i64::try_from(*row_count).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.zero_column_count.frame",
                    "parquet row count exceeds native Int64 count representation",
                )
                .with_fact("row_count", row_count)
                .with_fact("max_row_count", i64::MAX)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let timer = Instant::now();
    let series = context.runtime()?.series(counts.as_slice()).map_err(|error| {
        cudf_query_error(
            "query_engine.zero_column_count.frame",
            "failed to construct native zero-column count input",
            error,
        )
    })?;
    let frame = DataFrame::from_series([(ZERO_COLUMN_COUNT_COLUMN, series)]).map_err(|error| {
        cudf_query_error(
            "query_engine.zero_column_count.frame",
            "failed to construct native zero-column count frame",
            error,
        )
    })?;
    let read_elapsed = timer.elapsed();
    metrics.add_source_read_time(read_elapsed);
    metrics.record_input_batch(row_counts.iter().copied().sum());
    Ok(GpuDataFrame::new(frame))
}

pub(super) fn zero_column_count_chunk_frame_with_budget(
    context: &ExecutionContext,
    row_count: u64,
    budget_bytes: u64,
    budget_boundary: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let count = i64::try_from(row_count).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_chunk.zero_column_count",
            "parquet row count exceeds native Int64 count representation",
        )
        .with_fact("row_count", row_count)
        .with_fact("max_row_count", i64::MAX)
    })?;
    let budget = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_chunk.zero_column_count_budget",
            "bounded zero-column count source chunk budget exceeds platform usize range",
        )
        .with_fact("budget_boundary", budget_boundary)
        .with_fact("budget_bytes", budget_bytes)
    })?;
    let token = context
        .execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(budget_boundary))
        .map_err(|error| {
            error
                .with_fact_if_absent("budget_boundary", budget_boundary)
                .with_fact_if_absent("budget_bytes", budget_bytes)
        })?;

    let timer = Instant::now();
    let runtime = context.runtime()?;
    let outcome = token
        .run("runtime::series_from_value", |exec| runtime.on(exec)?.series_from_value(count, 1))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_chunk.zero_column_count",
                "failed to construct bounded native zero-column count input",
                error,
            )
        })?;
    metrics.record_budgeted_execution("query_engine.local_parquet_chunk.zero_column_count", outcome.report());
    let series = outcome.into_value();
    let frame = DataFrame::from_series([(ZERO_COLUMN_COUNT_COLUMN, series)]).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_chunk.zero_column_count",
            "failed to construct bounded native zero-column count frame",
            error,
        )
    })?;
    let read_elapsed = timer.elapsed();
    metrics.add_source_read_time(read_elapsed);
    metrics.record_input_batch(row_count);
    Ok(GpuDataFrame::new(frame))
}
