use std::time::Instant;

use arrow::{
    array::new_empty_array,
    record_batch::{RecordBatch, RecordBatchOptions},
};

use super::{
    DataFrame, ExecutionContext, ExecutionMetrics, LocalParquetSource, QueryError, QueryErrorCode, Result, SourceChunk,
    cudf_query_error, local_parquet_execution_frame_schema, parquet_chunk_path_fact,
};

/// Builds a zero-row device frame with the execution-frame schema, by uploading an empty Arrow
/// batch. It stands in for a read that is known to return no rows, such as a scan limit of 0 or a
/// source whose files were all pruned.
pub(super) fn empty_local_parquet_execution_frame(
    context: &ExecutionContext,
    source: &LocalParquetSource,
) -> Result<DataFrame> {
    let schema = local_parquet_execution_frame_schema(source)?;
    let columns = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect();
    let batch = RecordBatch::try_new_with_options(
        std::sync::Arc::clone(&schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.local_parquet_source.empty_scan_limit",
            "failed to construct zero-row local parquet source batch",
        )
        .with_source(error)
    })?;
    DataFrame::from_record_batch(&context.runtime()?, &batch).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.empty_scan_limit",
            "failed to upload zero-row local parquet source batch",
            error,
        )
    })
}

/// Truncates `frame` to the rows left in a source-level scan limit and returns the rows still
/// allowed afterwards. `None` means no limit. Rows are kept from the start of the frame, so with
/// several files the limit fills from the first file read. No slice is made when the frame
/// already fits.
pub(in crate::exec) fn apply_source_scan_limit_to_frame(
    context: &ExecutionContext,
    frame: DataFrame,
    remaining_rows: Option<usize>,
    metrics: &ExecutionMetrics,
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

    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let frame = frame.on(overrides).slice(0, keep_rows).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.scan_limit",
            "failed to apply native source scan limit",
            error,
        )
        .with_fact("input_rows", row_count)
        .with_fact("keep_rows", keep_rows)
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok((frame, remaining_rows))
}

/// Truncates one chunk's frame to `chunk.scan_limit_rows`, the row allowance the chunk planner
/// assigned to this chunk. A chunk with no allowance, or one already within it, is returned
/// unchanged.
pub(super) fn apply_source_chunk_scan_limit(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    frame: DataFrame,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let Some(keep_rows) = chunk.scan_limit_rows else {
        return Ok(frame);
    };
    let row_count = frame.num_rows();
    if keep_rows >= row_count {
        return Ok(frame);
    }

    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let frame = frame.on(overrides).slice(0, keep_rows).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_chunk.scan_limit",
            "failed to apply native source chunk scan limit",
            error,
        )
        .with_fact("path", parquet_chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("input_rows", row_count)
        .with_fact("keep_rows", keep_rows)
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(frame)
}
