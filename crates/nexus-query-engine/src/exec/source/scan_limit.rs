use super::*;

pub(super) fn empty_local_parquet_execution_frame(
    context: &ExecutionContext,
    source: &LocalParquetSource,
) -> Result<DataFrame> {
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
    DataFrame::from_record_batch(&context.runtime()?, &batch).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_source.empty_scan_limit",
            "failed to upload zero-row local parquet source batch",
            error,
        )
    })
}

pub(in crate::exec) fn apply_source_scan_limit_to_frame(
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

    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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

pub(super) fn apply_source_chunk_scan_limit(
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

    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let frame = frame.with_execution(&overrides).slice(0, keep_rows).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.local_parquet_chunk.scan_limit",
            "failed to apply native source chunk scan limit",
            error,
        )
        .with_fact("path", parquet_chunk_path_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("input_rows", row_count)
        .with_fact("keep_rows", keep_rows)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(frame)
}
