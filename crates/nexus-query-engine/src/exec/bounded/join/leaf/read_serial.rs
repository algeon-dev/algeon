use parquet::file::reader::FileReader;

use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn read_join_leaf_source_chunks_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let batches = join_leaf_source_chunk_batches_for_execution(
        source,
        source_chunks,
        reader_dynamic_filters,
        source_chunk_batch_target_bytes(execution.context().memory_control_policy()),
    );
    read_join_leaf_source_batches_serial(
        execution,
        source,
        node_id,
        batches,
        reader_dynamic_filters,
        max_attempts,
        min_retry_chunk_row_groups,
        grow_source_to_observed,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn read_join_leaf_source_batches_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    batches: Vec<SourceChunkBatch>,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
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
                execution,
                source,
                node_id,
                chunk,
                reader_dynamic_filters,
                1,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            ) {
                Ok(mut outputs) => {
                    chunks.append(&mut outputs);
                },
                Err(error) => {
                    release_join_source_chunks_reverse(execution, chunks);
                    return Err(error);
                },
            }
            continue;
        }
        match read_join_leaf_source_batch_or_split(
            execution,
            source,
            node_id,
            batch,
            reader_dynamic_filters,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        ) {
            Ok(mut outputs) => {
                chunks.append(&mut outputs);
            },
            Err(error) => {
                release_join_source_chunks_reverse(execution, chunks);
                return Err(error);
            },
        }
    }

    Ok(chunks)
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn read_join_leaf_source_batch_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    batch: SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return read_join_leaf_source_chunk_with_retry(
            execution,
            source,
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
        execution,
        source,
        node_id,
        &batch,
        reader_dynamic_filters,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            execution.metrics_mut().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "nexus_query_engine.exec.bounded.join_leaf_source_batch_split_retry",
            )?;
            let mut outputs = read_join_leaf_source_batch_or_split(
                execution,
                source,
                node_id,
                left,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
                grow_source_to_observed,
            )?;
            match read_join_leaf_source_batch_or_split(
                execution,
                source,
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
                    release_join_source_chunks_reverse(execution, outputs);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}
pub(super) fn read_join_leaf_source_batch_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    batch: &SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk_batch")?;
    for _chunk in &batch.chunks {
        execution.metrics_mut().record_source_chunk_attempt();
    }
    let mut reservation = execution.try_reserve(
        join_source_read_batch_reservation_request(
            execution.context().memory_control_policy(),
            execution.reservations().budget(),
            node_id,
            batch,
            1,
            1,
        )
        .with_stage("join_source_read_batch"),
    )?;
    let source_token_budget = execution.reservations().available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_batch(
        execution.context(),
        source,
        batch,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(reservation);
            return Err(error);
        },
    };
    if grow_source_to_observed {
        if let Err(error) = grow_join_source_batch_reservation_to_observed_frame(
            execution,
            &frame,
            &mut reservation,
            node_id,
            batch,
            1,
            1,
        ) {
            execution.release(reservation);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        execution.release(reservation);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_batch_rows(reader_dynamic_filters, batch, &frame, execution.metrics_mut());
    execution
        .metrics_mut()
        .record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
    Ok((frame, reservation))
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn read_join_leaf_source_chunk_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    match read_join_leaf_source_chunk_attempt(
        execution,
        source,
        node_id,
        &chunk,
        reader_dynamic_filters,
        attempt,
        max_attempts,
        grow_source_to_observed,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_chunk_after_error(
            execution,
            source,
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
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn retry_join_leaf_source_chunk_after_error(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    error: QueryError,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    record_chunk_retry_error(&error, execution.metrics_mut());
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        execution.metrics_mut().record_source_chunk_retry();
        crate::exec::retry::backoff_before_retry(
            execution.context(),
            "nexus_query_engine.exec.bounded.join_leaf_source_split_retry",
        )?;
        let mut outputs = read_join_leaf_source_chunk_with_retry(
            execution,
            source,
            node_id,
            left,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
            grow_source_to_observed,
        )?;
        outputs.extend(read_join_leaf_source_chunk_with_retry(
            execution,
            source,
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

    Err(chunk_retry_facts(&chunk, execution.metrics_mut()).attach_to_error(error))
}
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn read_join_leaf_source_chunk_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: &SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_chunk")?;
    execution.metrics_mut().record_source_chunk_attempt();
    let selected_row_groups_before_read = execution.metrics_mut().row_groups_selected;
    let requested_bytes = chunk_requested_bytes(execution.context().memory_control_policy(), chunk);
    let mut reservation = execution.try_reserve(join_source_read_reservation_request(
        execution.context().memory_control_policy(),
        requested_bytes,
        execution.reservations().budget(),
        node_id,
        chunk,
        attempt,
        max_attempts,
    ))?;
    let source_token_budget = execution.reservations().available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk_with_reader_dynamic_filter_count(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            match try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
                execution,
                source,
                chunk,
                node_id,
                &mut reservation,
                &error,
                reader_dynamic_filters,
                attempt,
                max_attempts,
            ) {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    execution.release(reservation);
                    return Err(attach_join_source_read_resource_facts(error, node_id, chunk, attempt, max_attempts));
                },
                Err(retry_error) => {
                    execution.release(reservation);
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
            execution,
            &frame,
            &mut reservation,
            node_id,
            chunk,
            attempt,
            max_attempts,
        ) {
            execution.release(reservation);
            return Err(error);
        }
    } else if let Err(error) = observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        execution.release(reservation);
        return Err(error);
    }
    record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &frame, execution.metrics_mut());
    let committed_row_groups =
        committed_join_source_chunk_row_groups(chunk, selected_row_groups_before_read, execution.metrics());
    execution.metrics_mut().record_source_chunk_commit(committed_row_groups);
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

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn try_read_join_leaf_source_chunk_with_corrected_budget_after_error(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservation: &mut ActiveReservation,
    error: &QueryError,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(execution, error, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, execution.metrics_mut());
    execution.metrics_mut().record_source_chunk_retry();
    crate::exec::retry::backoff_before_retry(
        execution.context(),
        "nexus_query_engine.exec.bounded.join_leaf_source_budget_retry",
    )?;
    let request = join_source_read_reservation_request_for_boundary(
        execution.context().memory_control_policy(),
        0,
        execution.reservations().budget(),
        node_id,
        chunk,
        "join_source_read",
        attempt + 1,
        max_attempts,
    );
    try_grow_source_read_reservation_for_budget_correction(execution, reservation, correction, request)?;
    execution.metrics_mut().record_source_chunk_attempt();
    read_local_parquet_chunk_with_reader_dynamic_filter_count(
        execution.context(),
        source,
        chunk,
        correction.corrected_budget_bytes(),
        "join_source_read",
        reader_dynamic_filters.len(),
        execution.metrics_mut(),
    )
    .map(Some)
    .map_err(|error| correction.attach_to_error(error))
}
