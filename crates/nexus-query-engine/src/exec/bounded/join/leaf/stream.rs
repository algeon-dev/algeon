use std::collections::VecDeque;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, QueryError, ResidentGpuFrame, Result,
    SourceChunk, SourceChunkBatch, abandon_join_leaf_reads, apply_join_leaf_source_stream_stages,
    attach_join_source_read_resource_facts, chunk_retry_facts, fill_optional_join_leaf_reads, is_retryable_chunk_error,
    join_leaf_source_chunk_batches_for_execution, observe_device_memory_bytes, read_join_leaf_source_batch_attempt,
    read_join_leaf_source_chunk_attempt, record_chunk_retry_error, record_dynamic_filter_reader_predicate_rows,
    source_chunk_batch_target_bytes, split_retryable_chunk, start_next_required_join_leaf_read,
};

pub(super) fn stream_join_leaf_source_chunks_serial<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    stream_join_leaf_source_chunks_serial_from(execution, stream, &stream.chunks, consume)
}

pub(super) fn stream_join_leaf_source_chunks_serial_from<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunks: &[SourceChunk],
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let batches = join_leaf_source_chunk_batches_for_execution(
        &stream.source,
        chunks,
        &stream.reader_dynamic_filters,
        source_chunk_batch_target_bytes(execution.context().memory_control_policy()),
    );
    for batch in batches {
        if batch.is_singleton() {
            let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
            stream_join_leaf_source_chunk_with_consumer_retry(execution, stream, chunk, 1, consume)?;
            continue;
        }
        match stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, batch, consume) {
            Ok(()) => {},
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(super) fn stream_join_leaf_source_batch_or_split_with_consumer<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: SourceChunkBatch,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return stream_join_leaf_source_chunk_with_consumer_retry(execution, stream, chunk, 1, consume);
    }
    match stream_join_leaf_source_batch_with_consumer(execution, stream, &batch, consume) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            execution.metrics_mut().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "nexus_query_engine.exec.bounded.join_leaf_stream_batch_split_retry",
            )?;
            stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, left, consume)?;
            stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, right, consume)
        },
        Err(error) => Err(error),
    }
}

pub(super) fn stream_join_leaf_source_batch_with_consumer<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: &SourceChunkBatch,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let (frame, reservation) =
        read_join_leaf_source_batch_attempt(execution, &stream.source, batch, &stream.reader_dynamic_filters)?;
    let input = ResidentGpuFrame::with_device_bytes(frame, reservation);
    let output = apply_join_leaf_source_stream_stages(execution, stream, input)?;
    consume(output, execution)
}

pub(super) fn stream_join_leaf_source_chunk_with_consumer_retry<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream")?;
    let (frame, reservation) = match read_join_leaf_source_chunk_attempt(
        execution,
        &stream.source,
        stream.node_id,
        &chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
    ) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(execution, stream, chunk, error, attempt, consume);
        },
        Err(error) => return Err(error),
    };
    let input = ResidentGpuFrame::with_device_bytes(frame, reservation);
    let output = match apply_join_leaf_source_stream_stages(execution, stream, input) {
        Ok(output) => output,
        Err(error) if is_retryable_chunk_error(&error) => {
            return retry_join_leaf_source_stream_chunk_after_error(execution, stream, chunk, error, attempt, consume);
        },
        Err(error) => return Err(error),
    };
    match consume(output, execution) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            retry_join_leaf_source_stream_chunk_after_error(execution, stream, chunk, error, attempt, consume)
        },
        Err(error) => Err(error),
    }
}

pub(super) fn retry_join_leaf_source_stream_chunk_after_error<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    error: QueryError,
    attempt: usize,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    record_chunk_retry_error(&error, execution.metrics_mut());
    if attempt < stream.max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, stream.min_retry_chunk_row_groups)
    {
        execution.metrics_mut().record_source_chunk_retry();
        crate::exec::retry::backoff_before_retry(
            execution.context(),
            "nexus_query_engine.exec.bounded.join_leaf_stream_source_split_retry",
        )?;
        stream_join_leaf_source_chunk_with_consumer_retry(execution, stream, left, attempt + 1, consume)?;
        stream_join_leaf_source_chunk_with_consumer_retry(execution, stream, right, attempt + 1, consume)?;
        return Ok(());
    }

    Err(chunk_retry_facts(&chunk, execution.metrics_mut()).attach_to_error(error))
}

pub(super) fn stream_join_leaf_source_chunks_with_scheduler<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")?;
    start_next_required_join_leaf_read(
        execution,
        &stream.source,
        &stream.chunks,
        &mut pending,
        &mut next_chunk_index,
        stream.max_attempts,
    )?;

    for (index, chunk) in stream.chunks.iter().enumerate() {
        if let Err(error) =
            execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_stream_scheduler")
        {
            abandon_join_leaf_reads(execution, pending);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let output = match scheduled.finish(execution) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                if is_prefetch {
                    execution.metrics_mut().record_source_read_prefetch_abandoned();
                }
                abandon_join_leaf_reads(execution, pending);
                return stream_join_leaf_source_chunks_serial_from(execution, stream, &stream.chunks[index..], consume);
            },
            Err(error) => {
                abandon_join_leaf_reads(execution, pending);
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
            execution.metrics_mut().record_source_read_prefetch_completed();
        }
        let observed_bytes = match observe_device_memory_bytes(&output.frame, execution.metrics_mut()) {
            Ok(bytes) => bytes,
            Err(error) => {
                abandon_join_leaf_reads(execution, pending);
                return Err(error);
            },
        };
        record_dynamic_filter_reader_predicate_rows(
            &stream.reader_dynamic_filters,
            &output.chunk,
            &output.frame,
            execution.metrics_mut(),
        );
        execution.metrics_mut().record_source_chunk_commit(
            output
                .observed_row_group_count
                .or_else(|| chunk.known_read_row_group_count().map(|row_groups| row_groups as u64))
                .unwrap_or_else(|| chunk.row_group_count() as u64),
        );
        let input = ResidentGpuFrame::with_device_bytes(output.frame, observed_bytes);
        let output = match apply_join_leaf_source_stream_stages(execution, stream, input) {
            Ok(output) => output,
            Err(error) if is_retryable_chunk_error(&error) => {
                abandon_join_leaf_reads(execution, pending);
                return stream_join_leaf_source_chunks_serial_from(execution, stream, &stream.chunks[index..], consume);
            },
            Err(error) => {
                abandon_join_leaf_reads(execution, pending);
                return Err(error);
            },
        };
        if let Err(error) = consume(output, execution) {
            abandon_join_leaf_reads(execution, pending);
            if is_retryable_chunk_error(&error) {
                return stream_join_leaf_source_chunks_serial_from(execution, stream, &stream.chunks[index..], consume);
            }
            return Err(error);
        }
        if let Err(error) = fill_optional_join_leaf_reads(
            execution,
            &stream.source,
            &stream.chunks,
            &mut pending,
            &mut next_chunk_index,
            stream.max_attempts,
        ) {
            abandon_join_leaf_reads(execution, pending);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < stream.chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                execution,
                &stream.source,
                &stream.chunks,
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
