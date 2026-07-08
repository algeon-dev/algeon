use super::*;

pub(super) fn stream_join_leaf_source_chunks_serial<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    stream_join_leaf_source_chunks_serial_from(context, stream, &stream.chunks, reservations, metrics, consume)
}

pub(super) fn stream_join_leaf_source_chunks_serial_from<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let batches = join_leaf_source_chunk_batches(
        &stream.source,
        chunks,
        &stream.reader_dynamic_filters,
        source_chunk_batch_target_bytes(context.memory_control_policy()),
    );
    for batch in batches {
        if batch.is_singleton() {
            let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
            stream_join_leaf_source_chunk_with_consumer_retry(
                context,
                stream,
                chunk,
                1,
                reservations,
                metrics,
                consume,
            )?;
            reservations.acknowledge_requested_grant_shrink()?;
            continue;
        }
        match stream_join_leaf_source_batch_or_split_with_consumer(
            context,
            stream,
            batch,
            reservations,
            metrics,
            consume,
        ) {
            Ok(()) => {
                reservations.acknowledge_requested_grant_shrink()?;
            },
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(super) fn stream_join_leaf_source_batch_or_split_with_consumer<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            chunk,
            1,
            reservations,
            metrics,
            consume,
        );
    }
    match stream_join_leaf_source_batch_with_consumer(context, stream, &batch, reservations, metrics, consume) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                context,
                "nexus_query_engine.exec.bounded.join_leaf_stream_batch_split_retry",
            )?;
            stream_join_leaf_source_batch_or_split_with_consumer(
                context,
                stream,
                left,
                reservations,
                metrics,
                consume,
            )?;
            stream_join_leaf_source_batch_or_split_with_consumer(context, stream, right, reservations, metrics, consume)
        },
        Err(error) => Err(error),
    }
}

pub(super) fn stream_join_leaf_source_batch_with_consumer<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: &SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let grow_source_to_observed = stream.stages.is_empty();
    let (frame, reservation) = read_join_leaf_source_batch_attempt(
        context,
        &stream.source,
        reservations,
        metrics,
        stream.node_id,
        batch,
        &stream.reader_dynamic_filters,
        grow_source_to_observed,
    )?;
    let input = ResidentGpuFrame { frame, reservation };
    let output = apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics)?;
    consume(output, reservations, metrics)
}

pub(super) fn stream_join_leaf_source_chunk_with_consumer_retry<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream")?;
    let grow_source_to_observed = stream.stages.is_empty();
    let (frame, reservation) = match read_join_leaf_source_chunk_attempt(
        context,
        &stream.source,
        reservations,
        metrics,
        stream.node_id,
        &chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
        grow_source_to_observed,
    ) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(
                context,
                stream,
                chunk,
                error,
                attempt,
                reservations,
                metrics,
                consume,
            );
        },
        Err(error) => return Err(error),
    };
    let input = ResidentGpuFrame { frame, reservation };
    let output = match apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(
                context,
                stream,
                chunk,
                error,
                attempt,
                reservations,
                metrics,
                consume,
            );
        },
        Err(error) => return Err(error),
    };
    match consume(output, reservations, metrics) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_stream_chunk_after_error(
            context,
            stream,
            chunk,
            error,
            attempt,
            reservations,
            metrics,
            consume,
        ),
        Err(error) => Err(error),
    }
}

pub(super) fn retry_join_leaf_source_stream_chunk_after_error<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    error: QueryError,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    record_chunk_retry_error(&error, metrics);
    if attempt < stream.max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, stream.min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        crate::exec::retry::backoff_before_retry(
            context,
            "nexus_query_engine.exec.bounded.join_leaf_stream_source_split_retry",
        )?;
        stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            left,
            attempt + 1,
            reservations,
            metrics,
            consume,
        )?;
        reservations.acknowledge_requested_grant_shrink()?;
        stream_join_leaf_source_chunk_with_consumer_retry(
            context,
            stream,
            right,
            attempt + 1,
            reservations,
            metrics,
            consume,
        )?;
        reservations.acknowledge_requested_grant_shrink()?;
        return Ok(());
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}

pub(super) fn stream_join_leaf_source_chunks_with_scheduler<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let grow_source_to_observed = stream.stages.is_empty();
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")?;
    start_next_required_join_leaf_read(
        context,
        &stream.source,
        &stream.chunks,
        stream.node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
        stream.max_attempts,
    )?;

    for (index, chunk) in stream.chunks.iter().enumerate() {
        if let Err(error) =
            context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")
        {
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                if is_prefetch {
                    metrics.record_source_read_prefetch_abandoned();
                }
                abandon_join_leaf_reads(pending, reservations, metrics);
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            },
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(attach_join_source_read_resource_facts(
                    error,
                    stream.node_id,
                    chunk,
                    1,
                    stream.max_attempts,
                ));
            },
        };
        if is_prefetch {
            metrics.record_source_read_prefetch_completed();
        }
        if grow_source_to_observed {
            if let Err(error) = grow_join_source_reservation_to_observed_frame(
                &output.frame,
                &mut output.reservation,
                reservations,
                stream.node_id,
                chunk,
                metrics,
                1,
                stream.max_attempts,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            }
            let retained_device_bytes = output.reservation.bytes();
            if let Err(error) = insert_bounded_local_parquet_source_cache(
                context,
                &stream.source,
                chunk,
                &output.frame,
                &mut output.reservation,
                reservations,
                metrics,
                "join_source_read",
                1,
                stream.max_attempts,
                retained_device_bytes,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            }
        } else if let Err(error) = observe_device_memory_bytes(&output.frame, metrics) {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        record_dynamic_filter_reader_predicate_rows(
            &stream.reader_dynamic_filters,
            &output.chunk,
            &output.frame,
            metrics,
        );
        metrics.record_source_chunk_commit(
            output
                .observed_row_group_count
                .or_else(|| chunk.known_read_row_group_count().map(|row_groups| row_groups as u64))
                .unwrap_or_else(|| chunk.row_group_count() as u64),
        );
        let input = ResidentGpuFrame { frame: output.frame, reservation: output.reservation };
        let output = match apply_join_leaf_source_stream_stages(context, stream, input, reservations, metrics) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            },
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                return Err(error);
            },
        };
        if let Err(error) = consume(output, reservations, metrics) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            if is_retryable_chunk_error(&error) {
                return stream_join_leaf_source_chunks_serial_from(
                    context,
                    stream,
                    &stream.chunks[index..],
                    reservations,
                    metrics,
                    consume,
                );
            }
            return Err(error);
        }
        if let Err(error) = reservations.acknowledge_requested_grant_shrink() {
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }

        if let Err(error) = fill_optional_join_leaf_reads(
            context,
            &stream.source,
            &stream.chunks,
            stream.node_id,
            reservations,
            metrics,
            &mut pending,
            &mut next_chunk_index,
            stream.max_attempts,
        ) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < stream.chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                context,
                &stream.source,
                &stream.chunks,
                stream.node_id,
                reservations,
                metrics,
                &mut pending,
                &mut next_chunk_index,
                stream.max_attempts,
            )
        {
            return Err(error);
        }
    }

    Ok(())
}
