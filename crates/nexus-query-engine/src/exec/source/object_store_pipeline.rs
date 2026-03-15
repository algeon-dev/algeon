use super::*;

pub(super) fn should_pipeline_object_store_file_fetches(
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

pub(super) fn object_store_file_fetch_pipeline_enabled(
    max_source_reads_in_flight: usize,
    source: &LocalParquetSource,
    columns: &[&str],
) -> bool {
    if columns.is_empty() || source.scan_limit.is_some() || max_source_reads_in_flight <= 1 {
        return false;
    }
    let mut candidate_count = 0usize;
    for file in &source.files {
        if file_has_explicit_empty_row_group_selection(file) {
            continue;
        }
        candidate_count += 1;
        if !file.resolved_source.uses_host_buffer() || !object_store_file_fetch_pipeline_compatible(source, file) {
            return false;
        }
    }
    candidate_count > 1
}

pub(super) fn read_object_store_parquet_source_with_fetch_pipeline(
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
            start_object_store_file_fetches(context, source, max_in_flight, &mut next_file_index, &mut pending)?;
            continue;
        }
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, file_index)?;
        frames.push(frame);
        start_object_store_file_fetches(context, source, max_in_flight, &mut next_file_index, &mut pending)?;
    }

    concatenate_local_parquet_source_frames(context, source, frames, metrics)
}

pub(super) fn start_object_store_file_fetches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    max_in_flight: usize,
    next_file_index: &mut usize,
    pending: &mut VecDeque<(usize, ScheduledHostParquetFetch)>,
) -> Result<()> {
    while pending.len() < max_in_flight && *next_file_index < source.files.len() {
        let file_index = *next_file_index;
        if file_has_explicit_empty_row_group_selection(&source.files[file_index]) {
            *next_file_index += 1;
            continue;
        }
        let fetch = if pending.is_empty() {
            Some(start_object_store_parquet_file_pipeline_fetch(context, source, &source.files[file_index])?)
        } else {
            try_start_object_store_parquet_file_pipeline_fetch(context, source, &source.files[file_index])?
        };
        let Some(fetch) = fetch else {
            break;
        };
        pending.push_back((file_index, fetch));
        *next_file_index += 1;
    }
    Ok(())
}

pub(super) fn concatenate_local_parquet_source_frames(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    mut frames: Vec<DataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if frames.is_empty() {
        let frame = empty_local_parquet_execution_frame(context, source)?;
        let frame = coerce_local_parquet_source_schema(frame, source, None, metrics)?;
        let frame = append_source_order_columns(context, frame, 0)?;
        metrics.record_input_batch(0);
        return Ok(GpuDataFrame::new(frame));
    }
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
