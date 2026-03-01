use parquet::file::reader::FileReader;

use super::*;

pub(super) fn read_join_leaf_source_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let batches = join_leaf_source_chunk_batches(
        source,
        source_chunks,
        reader_dynamic_filters,
        source_chunk_batch_target_bytes(context.memory_control_policy()),
    );
    read_join_leaf_source_batches_serial(
        context,
        source,
        node_id,
        batches,
        reader_dynamic_filters,
        reservations,
        metrics,
        max_attempts,
        min_retry_chunk_row_groups,
        grow_source_to_observed,
    )
}

pub(super) fn read_join_leaf_source_batches_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    batches: Vec<SourceChunkBatch>,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let source_chunk_count = batches.iter().map(SourceChunkBatch::chunk_count).sum();
    let mut chunks = Vec::with_capacity(source_chunk_count);
    for batch in batches {
        if batch.is_singleton() {
            let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
            match read_join_leaf_source_chunk_with_retry(
                context,
                source,
                reservations,
                metrics,
                node_id,
                chunk,
                reader_dynamic_filters,
                1,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            ) {
                Ok(mut outputs) => chunks.append(&mut outputs),
                Err(error) => {
                    release_join_source_chunks_reverse(reservations, chunks, metrics);
                    return Err(error);
                },
            }
            continue;
        }
        match read_join_leaf_source_batch_or_split(
            context,
            source,
            reservations,
            metrics,
            node_id,
            batch,
            reader_dynamic_filters,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        ) {
            Ok(mut outputs) => chunks.append(&mut outputs),
            Err(error) => {
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            },
        }
    }

    Ok(chunks)
}

pub(super) fn read_join_leaf_source_batch_or_split(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    batch: SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk,
            reader_dynamic_filters,
            1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        );
    }
    match read_join_leaf_source_batch_attempt(
        context,
        source,
        reservations,
        metrics,
        node_id,
        &batch,
        reader_dynamic_filters,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            let mut outputs = read_join_leaf_source_batch_or_split(
                context,
                source,
                reservations,
                metrics,
                node_id,
                left,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            )?;
            match read_join_leaf_source_batch_or_split(
                context,
                source,
                reservations,
                metrics,
                node_id,
                right,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_join_source_chunks_reverse(reservations, outputs, metrics);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}
pub(super) fn read_join_leaf_source_batch_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch")?;
    for _chunk in &batch.chunks {
        metrics.record_source_chunk_attempt();
    }
    let mut reservation = reservations.try_reserve(
        join_source_read_batch_reservation_request(
            context.memory_control_policy(),
            reservations.budget(),
            node_id,
            batch,
            1,
            1,
        )
        .with_stage("join_source_read_batch"),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_batch(
        context,
        source,
        batch,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(error);
        },
    };
    if grow_source_to_observed {
        if let Err(error) = grow_join_source_batch_reservation_to_observed_frame(
            &frame,
            &mut reservation,
            reservations,
            node_id,
            batch,
            metrics,
            1,
            1,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, metrics) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_batch_rows(reader_dynamic_filters, batch, &frame, metrics);
    metrics.record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
    Ok((frame, reservation))
}

pub(super) fn read_join_leaf_source_chunk_with_retry(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    match read_join_leaf_source_chunk_attempt(
        context,
        source,
        reservations,
        metrics,
        node_id,
        &chunk,
        reader_dynamic_filters,
        attempt,
        max_attempts,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_chunk_after_error(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk,
            reader_dynamic_filters,
            error,
            attempt,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        ),
        Err(error) => Err(error),
    }
}
pub(super) fn retry_join_leaf_source_chunk_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    error: QueryError,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    record_chunk_retry_error(&error, metrics);
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        let mut outputs = read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            left,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?;
        outputs.extend(read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            right,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?);
        return Ok(outputs);
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}
pub(super) fn read_join_leaf_source_chunk_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: &SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk")?;
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &output.frame, metrics);
        metrics.record_source_chunk_commit(committed_join_source_chunk_row_groups(
            chunk,
            metrics.row_groups_selected,
            metrics,
        ));
        return Ok((output.frame, output.reservation));
    }
    let selected_row_groups_before_read = metrics.row_groups_selected;
    let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
    let mut reservation = reservations.try_reserve(
        join_source_read_reservation_request(
            context.memory_control_policy(),
            requested_bytes,
            reservations.budget(),
            node_id,
            chunk,
            attempt,
            max_attempts,
        ),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_with_reader_dynamic_filter_count(
        context,
        source,
        chunk,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
                context,
                source,
                chunk,
                node_id,
                &mut reservation,
                reservations,
                metrics,
                &error,
                reader_dynamic_filters,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    reservations.release(reservation, metrics);
                    return Err(attach_join_source_read_resource_facts(error, node_id, chunk, attempt, max_attempts));
                },
                Err(retry_error) => {
                    reservations.release(reservation, metrics);
                    return Err(attach_join_source_read_resource_facts(
                        retry_error,
                        node_id,
                        chunk,
                        attempt.saturating_add(1),
                        max_attempts,
                    ));
                },
            }
        },
    };
    if grow_source_to_observed {
        if let Err(error) = grow_join_source_reservation_to_observed_frame(
            &frame,
            &mut reservation,
            reservations,
            node_id,
            chunk,
            metrics,
            attempt,
            max_attempts,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
        let retained_device_bytes = reservation.bytes();
        if let Err(error) = insert_bounded_local_parquet_source_cache(
            context,
            source,
            chunk,
            &frame,
            &mut reservation,
            reservations,
            metrics,
            "join_source_read",
            attempt,
            max_attempts,
            retained_device_bytes,
        ) {
            drop(frame);
            reservations.release(reservation, metrics);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, metrics) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &frame, metrics);
    metrics.record_source_chunk_commit(committed_join_source_chunk_row_groups(
        chunk,
        selected_row_groups_before_read,
        metrics,
    ));
    Ok((frame, reservation))
}

fn committed_join_source_chunk_row_groups(
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

pub(super) fn try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    error: &QueryError,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(error, reservations, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, metrics);
    metrics.record_source_chunk_retry();
    let request = join_source_read_reservation_request_for_boundary(
        context.memory_control_policy(),
        0,
        reservations.budget(),
        node_id,
        chunk,
        "join_source_read",
        attempt + 1,
        max_attempts,
    );
    try_grow_source_read_reservation_for_budget_correction(reservations, reservation, correction, request, metrics)?;
    metrics.record_source_chunk_attempt();
    read_local_parquet_chunk_with_reader_dynamic_filter_count(
        context,
        source,
        chunk,
        correction.corrected_budget_bytes(),
        "join_source_read",
        reader_dynamic_filters.len(),
        metrics,
    )
    .map(Some)
    .map_err(|error| correction.attach_to_error(error))
}
