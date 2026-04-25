use parquet::file::reader::FileReader;

use super::*;

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_required_only_with_observation(
        context,
        source,
        chunks,
        reservations,
        metrics,
        memory_boundary,
        true,
    )
}

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only_with_observation(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_serial(
        context,
        source,
        chunks,
        reservations,
        metrics,
        memory_boundary,
        grow_source_to_observed,
    )
}
pub(in crate::exec::bounded) fn concatenate_source_read_outputs(
    context: &ExecutionContext,
    node_id: NodeId,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    concat_output_boundary: &'static str,
    operation: &'static str,
    message: &'static str,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let ScheduledLocalParquetReadOutput { frame, reservation, .. } =
            chunks.pop().expect("single source chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, chunk| total.saturating_add(chunk.reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, output_reservation_bytes, reservations.budget())
            .with_memory_boundary(concat_output_boundary)
            .with_node_id(node_id)
            .with_stage(concat_output_boundary)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_source_read_outputs_reverse(reservations, chunks, metrics);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        frames.push(chunk.frame.into_frame());
        input_reservations.push(chunk.reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(operation, message, error)
                .with_fact("node_id", node_id)
                .with_fact("chunk_count", chunk_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, reservations.budget())
                .with_memory_boundary(concat_output_boundary)
                .with_node_id(node_id)
                .with_stage(concat_output_boundary)
                .with_attempt(1, 1),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_source_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn read_local_parquet_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    if source.has_position_delete_visible_scan_limit() {
        return read_local_parquet_chunks_serial_with_position_delete_visible_limit(
            context,
            source,
            chunks,
            reservations,
            metrics,
            memory_boundary,
            grow_source_to_observed,
        );
    }

    let batches = generic_source_chunk_batches(context, source, chunks);
    let mut outputs = Vec::with_capacity(batches.len());
    for batch in batches {
        if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk") {
            release_source_read_outputs_reverse(reservations, outputs, metrics);
            return Err(error);
        }
        match read_local_parquet_source_batch_or_split(
            context,
            source,
            batch,
            reservations,
            metrics,
            memory_boundary,
            grow_source_to_observed,
        ) {
            Ok(mut batch_outputs) => outputs.append(&mut batch_outputs),
            Err(error) => {
                release_source_read_outputs_reverse(reservations, outputs, metrics);
                return Err(error);
            },
        }
    }
    Ok(outputs)
}

fn read_local_parquet_chunks_serial_with_position_delete_visible_limit(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut outputs = Vec::with_capacity(chunks.len());
    let mut remaining_rows = source.scan_limit;
    for chunk in chunks {
        if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk") {
            release_source_read_outputs_reverse(reservations, outputs, metrics);
            return Err(error);
        }
        match read_local_parquet_chunk_required_only(
            context,
            source,
            chunk,
            reservations,
            metrics,
            memory_boundary,
            grow_source_to_observed,
            false,
            Some(&mut remaining_rows),
        ) {
            Ok(output) => {
                record_position_delete_visible_limit_chunk_metrics(&output.chunk, metrics);
                outputs.push(output);
            },
            Err(error) => {
                release_source_read_outputs_reverse(reservations, outputs, metrics);
                return Err(error);
            },
        }
        if remaining_rows == Some(0) {
            break;
        }
    }
    Ok(outputs)
}

fn record_position_delete_visible_limit_chunk_metrics(chunk: &SourceChunk, metrics: &mut QueryMetrics) {
    metrics.parquet_files = metrics.parquet_files.saturating_add(1);
    metrics.parquet_bytes_known =
        metrics.parquet_bytes_known.saturating_add(chunk.known_row_group_bytes.or(chunk.known_file_bytes).unwrap_or(0));
    metrics.read_columns = metrics.read_columns.saturating_add(chunk.read_columns.len() as u64);
    if let Some(row_group_count) = chunk.known_read_row_group_count() {
        metrics.row_groups_selected = metrics.row_groups_selected.saturating_add(row_group_count as u64);
    }
}

fn read_local_parquet_source_batch_or_split(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    batch: crate::chunk::SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk_batch_retry")?;
    let (max_attempts, min_retry_chunk_row_groups) = source_read_retry_policy(context);
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source batch has one chunk").clone();
        return read_local_parquet_chunk_required_only_or_split(
            context,
            source,
            chunk,
            reservations,
            metrics,
            memory_boundary,
            grow_source_to_observed,
            1,
            max_attempts,
            min_retry_chunk_row_groups,
        );
    }
    match read_local_parquet_chunk_batch_required_only_no_commit(
        context,
        source,
        &batch,
        reservations,
        metrics,
        memory_boundary,
        grow_source_to_observed,
    ) {
        Ok(output) => {
            metrics.record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
            Ok(vec![output])
        },
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                context,
                "nexus_query_engine.exec.bounded.source_read_batch_split_retry",
            )?;
            let mut outputs = read_local_parquet_source_batch_or_split(
                context,
                source,
                left,
                reservations,
                metrics,
                memory_boundary,
                grow_source_to_observed,
            )?;
            match read_local_parquet_source_batch_or_split(
                context,
                source,
                right,
                reservations,
                metrics,
                memory_boundary,
                grow_source_to_observed,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_source_read_outputs_reverse(reservations, outputs, metrics);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn source_read_retry_policy(context: &ExecutionContext) -> (usize, usize) {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } => {
            (max_retry_attempts.max(1), min_retry_chunk_row_groups.max(1))
        },
        MemoryControlPolicy::Unbounded => (1, 1),
    }
}

fn read_local_parquet_chunk_required_only_or_split(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.serial_source_chunk_retry")?;
    match read_local_parquet_chunk_required_only_attempt(
        context,
        source,
        &chunk,
        reservations,
        metrics,
        memory_boundary,
        grow_source_to_observed,
        true,
        None,
        attempt,
        max_attempts,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            if attempt < max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
            {
                metrics.record_source_chunk_retry();
                crate::exec::retry::backoff_before_retry(
                    context,
                    "nexus_query_engine.exec.bounded.serial_source_chunk_split_retry",
                )?;
                let mut outputs = read_local_parquet_chunk_required_only_or_split(
                    context,
                    source,
                    left,
                    reservations,
                    metrics,
                    memory_boundary,
                    grow_source_to_observed,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?;
                match read_local_parquet_chunk_required_only_or_split(
                    context,
                    source,
                    right,
                    reservations,
                    metrics,
                    memory_boundary,
                    grow_source_to_observed,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(mut right_outputs) => {
                        outputs.append(&mut right_outputs);
                        Ok(outputs)
                    },
                    Err(error) => {
                        release_source_read_outputs_reverse(reservations, outputs, metrics);
                        Err(error)
                    },
                }
            } else {
                Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
            }
        },
        Err(error) => Err(error),
    }
}

fn read_local_parquet_chunk_required_only(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    cache_chunks: bool,
    visible_limit_remaining: Option<&mut Option<usize>>,
) -> Result<ScheduledLocalParquetReadOutput> {
    read_local_parquet_chunk_required_only_attempt(
        context,
        source,
        chunk,
        reservations,
        metrics,
        memory_boundary,
        grow_source_to_observed,
        cache_chunks,
        visible_limit_remaining,
        1,
        1,
    )
}

fn read_local_parquet_chunk_required_only_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
    cache_chunks: bool,
    visible_limit_remaining: Option<&mut Option<usize>>,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetReadOutput> {
    metrics.record_source_chunk_attempt();
    if cache_chunks
        && let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)?
    {
        metrics.record_source_chunk_commit(committed_source_chunk_row_groups(
            chunk,
            metrics.row_groups_selected,
            metrics,
        ));
        return Ok(output);
    }
    let selected_row_groups_before_read = metrics.row_groups_selected;
    let source_request = source_read_reservation_request(
        context.memory_control_policy(),
        ReservationKind::SourceRead,
        memory_boundary,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let mut reservation = reservations.try_reserve(source_request, metrics)?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk(context, source, chunk, source_token_budget, memory_boundary, metrics) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_local_parquet_chunk_with_corrected_budget_after_error(
                context,
                source,
                chunk,
                &mut reservation,
                reservations,
                metrics,
                memory_boundary,
                &error,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    reservations.release(reservation, metrics);
                    return Err(error);
                },
                Err(retry_error) => {
                    reservations.release(reservation, metrics);
                    return Err(retry_error);
                },
            }
        },
    };
    let frame = if let Some(remaining_rows) = visible_limit_remaining {
        match apply_source_scan_limit_to_frame(context, frame.into_frame(), *remaining_rows, metrics) {
            Ok((frame, remaining)) => {
                *remaining_rows = remaining;
                GpuDataFrame::new(frame)
            },
            Err(error) => {
                reservations.release(reservation, metrics);
                return Err(error);
            },
        }
    } else {
        frame
    };
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        reservation,
        observed_bytes: 0,
        observed_row_group_count: None,
    };
    output.observed_bytes = match if grow_source_to_observed {
        grow_source_reservation_to_observed(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
        )
    } else {
        shrink_source_reservation_to_observed_without_grow(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
        )
    } {
        Ok(observed_bytes) => observed_bytes,
        Err(error) => {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
            return Err(error);
        },
    };
    if grow_source_to_observed && cache_chunks {
        insert_bounded_local_parquet_source_cache(
            context,
            source,
            &output.chunk,
            &output.frame,
            &mut output.reservation,
            reservations,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
            output.observed_bytes,
        )?;
    }
    metrics.record_source_chunk_commit(committed_source_chunk_row_groups(
        chunk,
        selected_row_groups_before_read,
        metrics,
    ));
    Ok(output)
}

fn committed_source_chunk_row_groups(
    chunk: &SourceChunk,
    selected_row_groups_before_read: u64,
    metrics: &QueryMetrics,
) -> u64 {
    if let Some(row_groups) = chunk.known_read_row_group_count() {
        return row_groups as u64;
    }
    let observed_delta = metrics.row_groups_selected.saturating_sub(selected_row_groups_before_read);
    if observed_delta > 0 {
        return observed_delta;
    }
    if chunk.reads_all_row_groups && chunk.path.is_file() {
        return parquet::file::reader::SerializedFileReader::try_from(chunk.path.as_path())
            .ok()
            .map(|reader| reader.metadata().num_row_groups() as u64)
            .unwrap_or(0);
    }
    chunk.row_group_count() as u64
}

pub(in crate::exec::bounded) fn read_local_parquet_chunk_batch_required_only_no_commit(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    for _chunk in &batch.chunks {
        metrics.record_source_chunk_attempt();
    }
    let selected_row_groups_before_read = metrics.row_groups_selected;
    let mut reservation = reservations.try_reserve(
        source_read_batch_reservation_request(
            context.memory_control_policy(),
            ReservationKind::SourceRead,
            memory_boundary,
            reservations.budget(),
            batch,
            1,
            1,
        ),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_batch(
        context,
        source,
        batch,
        source_token_budget,
        memory_boundary,
        0,
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(error);
        },
    };
    let observed_bytes = match if grow_source_to_observed {
        grow_source_batch_reservation_to_observed(
            &frame,
            &mut reservation,
            reservations,
            batch,
            metrics,
            memory_boundary,
            1,
            1,
        )
    } else {
        shrink_source_batch_reservation_to_observed_without_grow(
            &frame,
            &mut reservation,
            reservations,
            batch,
            metrics,
            memory_boundary,
            1,
            1,
        )
    } {
        Ok(observed_bytes) => observed_bytes,
        Err(error) => {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        },
    };
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    Ok(ScheduledLocalParquetReadOutput {
        chunk: first_chunk.clone(),
        frame,
        reservation,
        observed_bytes,
        observed_row_group_count: source_chunk_batch_read_row_groups(batch, selected_row_groups_before_read, metrics),
    })
}

fn generic_source_chunk_batches(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> Vec<crate::chunk::SourceChunkBatch> {
    if context.local_parquet_source_cache().is_some() || !generic_source_chunk_batching_supported(source) {
        return singleton_source_chunk_batches(chunks);
    }
    let target_bytes = BoundedSourceBatchingPolicy::default().generic_source_batch_target_bytes(
        context.memory_control_policy(),
        source_chunk_batch_decode_scratch_risk(source, chunks),
    );
    plan_source_chunk_batches(chunks, target_bytes)
}

fn generic_source_chunk_batching_supported(source: &LocalParquetSource) -> bool {
    source.partition_columns.is_empty()
        && !source.files.iter().any(LocalParquetFile::has_row_delete_files)
        && source.scan_limit.is_none()
        && source.reader_predicate.as_ref().is_none_or(|predicate| source.reader_predicate_is_supported(predicate))
}

fn singleton_source_chunk_batches(chunks: &[SourceChunk]) -> Vec<crate::chunk::SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| crate::chunk::SourceChunkBatch {
            batch_index,
            chunks: vec![chunk.clone()],
            estimated_source_read_bytes: chunk
                .estimated_source_read_bytes
                .or(chunk.known_row_group_bytes)
                .or(chunk.estimated_chunk_bytes)
                .or(chunk.known_file_bytes),
            source_batching_reason: "source_gate:generic_batching_disabled".to_string(),
        })
        .collect()
}

fn source_chunk_batch_decode_scratch_risk(source: &LocalParquetSource, chunks: &[SourceChunk]) -> bool {
    source.reader_predicate.is_some()
        && chunks.iter().any(|chunk| {
            chunk.known_pure_filter_compressed_bytes.unwrap_or(0) > 0
                || chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0) > 0
        })
}

pub(in crate::exec::bounded) fn release_source_read_outputs_reverse(
    reservations: &mut ReservationTracker,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    metrics: &mut QueryMetrics,
) {
    while let Some(chunk) = chunks.pop() {
        drop(chunk.frame);
        reservations.release(chunk.reservation, metrics);
    }
}

fn release_source_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut active: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation, metrics);
    }
}

pub(in crate::exec::bounded) fn try_read_cached_local_parquet_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<ScheduledLocalParquetReadOutput>> {
    if datafusion_order_tracking_enabled() {
        return Ok(None);
    }
    let Some(cache) = context.local_parquet_source_cache() else {
        return Ok(None);
    };
    let Some(observed_row_group_count) = source_chunk_static_read_row_groups(chunk) else {
        return Ok(None);
    };
    let key = GpuLocalParquetSourceCacheKey::from_source_chunk(source, chunk);
    let schema = local_parquet_execution_frame_schema(source)?;
    let lookup_timer = Instant::now();
    let cached = cache.get(&key, context, &schema)?;
    metrics.record_source_cache_lookup_time(lookup_timer.elapsed());
    if let Some((handle, retained_device_bytes)) = cached {
        metrics.record_source_cache_hit(cache.retained_device_bytes()?);
        let frame = handle.into_gpu_frame();
        let source_cache_pin = frame.source_cache_pin();
        return Ok(Some(ScheduledLocalParquetReadOutput {
            chunk: chunk.clone(),
            frame,
            reservation: reservations.borrow_accounted_source_cache(
                key,
                retained_device_bytes,
                source_cache_pin,
                metrics,
            ),
            observed_bytes: retained_device_bytes,
            observed_row_group_count: Some(observed_row_group_count),
        }));
    }
    metrics.record_source_cache_miss(cache.retained_device_bytes()?);
    Ok(None)
}
pub(in crate::exec::bounded) fn insert_bounded_local_parquet_source_cache(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    retained_device_bytes: u64,
) -> Result<()> {
    if datafusion_order_tracking_enabled() {
        return Ok(());
    }
    let Some(cache) = context.local_parquet_source_cache() else {
        return Ok(());
    };
    if source_chunk_static_read_row_groups(chunk).is_none() {
        return Ok(());
    }
    let key = GpuLocalParquetSourceCacheKey::from_source_chunk(source, chunk);
    let schema = local_parquet_execution_frame_schema(source)?;
    let handle = GpuResultHandle::new(context, frame.clone(), schema)?;
    let protected_keys = reservations.source_cache_protected_keys_with(&key);
    let insert =
        cache.insert_if_absent_protecting(context, key.clone(), handle, retained_device_bytes, &protected_keys)?;
    if !cache.is_governor_backed() {
        reservations.release_source_cache_baseline_bytes(insert.evicted_device_bytes, metrics);
    }
    match insert.outcome {
        GpuLocalParquetSourceCacheInsert::Inserted => {
            metrics.record_source_cache_insert(insert.retained_device_bytes);
            let request = chunk_reservation_request(
                ReservationKind::SourceCacheRetained,
                "source_cache_retained",
                retained_device_bytes,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            )
            .with_stage(memory_boundary);
            reservations.promote_source_cache_insert(
                source_reservation,
                retained_device_bytes,
                request,
                metrics,
                &key,
            )?;
        },
        GpuLocalParquetSourceCacheInsert::AlreadyPresent | GpuLocalParquetSourceCacheInsert::CapacitySkipped => {
            metrics.record_source_cache_insert_skipped(insert.retained_device_bytes);
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{BinaryOp, Expr, LiteralValue},
        runtime::SourceChunkPlanningMode,
        source::{LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
    };

    #[test]
    fn generic_source_batches_cap_decode_scratch_risk_only() {
        let context = ExecutionContext::builder()
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .build()
            .expect("valid bounded context");
        let chunk_bytes = BoundedSourceBatchingPolicy::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES / 2 + 1;
        let chunks = vec![test_chunk(0, chunk_bytes, false), test_chunk(1, chunk_bytes, false)];

        let no_predicate_batches = generic_source_chunk_batches(&context, &test_source(false), &chunks);
        assert_eq!(no_predicate_batches.len(), 1);
        assert_eq!(no_predicate_batches[0].chunk_count(), 2);

        let scratch_risk_chunks = vec![test_chunk(0, chunk_bytes, true), test_chunk(1, chunk_bytes, true)];
        let predicate_batches = generic_source_chunk_batches(&context, &test_source(true), &scratch_risk_chunks);
        assert_eq!(predicate_batches.len(), 2);
        assert!(predicate_batches.iter().all(SourceChunkBatch::is_singleton));
    }

    #[test]
    fn serial_source_chunk_retry_policy_uses_bounded_context() {
        let context = ExecutionContext::builder()
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 64 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024 * 1024,
                max_row_groups_per_chunk: 1024,
                max_retry_attempts: 4,
                min_retry_chunk_row_groups: 8,
            })
            .build()
            .expect("valid bounded context");

        assert_eq!(source_read_retry_policy(&context), (4, 8));
    }

    fn test_source(with_reader_predicate: bool) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        let source = LocalParquetSource::from_files(Vec::new(), schema, vec!["c0".to_string()]);
        if with_reader_predicate {
            return source.with_reader_predicate(Expr::binary(
                Expr::column("c0"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Int64(0)),
            ));
        }
        source
    }

    fn test_chunk(index: usize, bytes: u64, pure_filter_bytes: bool) -> SourceChunk {
        let pure_filter_bytes = if pure_filter_bytes { bytes } else { 0 };
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index: index,
            location: ParquetSourceLocation::LocalPath(PathBuf::from(format!("part-{index}.parquet"))),
            path: PathBuf::from(format!("part-{index}.parquet")),
            resolved_source: ResolvedParquetSource::local_kvikio(format!("part-{index}.parquet")),
            stable_id: format!("file-{index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(bytes),
            known_row_group_bytes: Some(bytes),
            known_projected_compressed_bytes: Some(bytes),
            known_projected_uncompressed_bytes: Some(bytes),
            known_pure_filter_compressed_bytes: Some(pure_filter_bytes),
            known_pure_filter_uncompressed_bytes: Some(pure_filter_bytes),
            estimated_chunk_bytes: Some(bytes),
            estimated_source_read_bytes: Some(bytes),
            exact_source_read_preflight_bytes: Some(bytes),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![bytes]),
            source_file_projected_compressed_bytes: Some(vec![bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![pure_filter_bytes]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["c0".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }
}
