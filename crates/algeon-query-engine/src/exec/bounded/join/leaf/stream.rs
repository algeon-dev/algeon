use std::collections::VecDeque;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame, Result, SourceChunk,
    SourceChunkBatch, abandon_join_leaf_reads, apply_join_leaf_source_stream_stages,
    attach_join_source_read_resource_facts, chunk_retry_facts, fill_optional_join_leaf_reads, is_retryable_chunk_error,
    join_leaf_source_chunk_batches_for_execution, observe_device_memory_bytes, read_join_leaf_source_batch_attempt,
    read_join_leaf_source_chunk_attempt, record_chunk_retry_error, record_dynamic_filter_reader_predicate_rows,
    source_chunk_batch_target_bytes, source_chunk_batches, split_retryable_chunk, start_next_required_join_leaf_read,
};
use crate::exec::bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource};

/// Exposes a deferred Parquet leaf as a pull-based resident frame stream.
///
/// Each pull reads one planned source batch (with the same split retry as the
/// serial push path) and yields its frames, so a consumer that streams its own
/// output holds at most one leaf batch. Consumer failures cannot split the
/// batch they were handed; consumers with their own retry shape (partitioned
/// probes, gather-map slices) recover on their side.
pub(in crate::exec::bounded) fn bounded_join_leaf_local_parquet_source_frame_stream(
    execution: &BoundedExecutionAttempt<'_>,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> ResidentFrameStreamPayload {
    // Batches follow the planned chunk cap, like the push path's batching
    // decision, so a pull never reads more than one planned batch at a time.
    let batches = source_chunk_batches(execution.context(), &stream);
    let max_per_frame_reservation_bytes =
        batches.iter().filter_map(|batch| batch.estimated_source_read_bytes).max().unwrap_or(0);
    let aggregate_reservation_bytes = stream.estimated_source_read_bytes().unwrap_or(0);
    let aggregate_num_rows = stream.estimated_source_rows().unwrap_or(0);
    let num_columns = stream.output_column_count();
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Box::new(JoinLeafSourceFrameStreamSource {
            stream,
            batches: batches.into_iter().map(|batch| (batch, 1)).collect(),
        }),
    )
}

struct JoinLeafSourceFrameStreamSource {
    stream: BoundedJoinLeafLocalParquetSourceStream,
    batches: VecDeque<(SourceChunkBatch, usize)>,
}

impl ResidentFrameStreamSource for JoinLeafSourceFrameStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            let (batch, attempt) = self.batches.pop_front()?;
            let output = if batch.is_singleton() {
                read_join_leaf_source_chunk_with_stages(execution, &self.stream, &batch.chunks[0], attempt)
            } else {
                read_join_leaf_source_batch_with_stages(execution, &self.stream, &batch)
            };
            match output {
                Ok(frame) => return Some(Ok(frame)),
                Err(error) if is_retryable_chunk_error(&error) => {
                    let (left, right, next_attempt) = if batch.is_singleton() {
                        let chunk = &batch.chunks[0];
                        record_chunk_retry_error(&error, execution.metrics());
                        let children = if attempt < self.stream.max_attempts {
                            split_retryable_chunk(chunk, self.stream.min_retry_chunk_row_groups)
                        } else {
                            None
                        };
                        let Some((left, right)) = children else {
                            self.batches.clear();
                            return Some(Err(chunk_retry_facts(chunk, execution).attach_to_error(error)));
                        };
                        (singleton_retry_batch(left), singleton_retry_batch(right), attempt + 1)
                    } else {
                        let (left, right) = batch.split_for_retry().expect("non-singleton source batch splits in two");
                        (left, right, 1)
                    };
                    execution.metrics().record_source_chunk_retry();
                    if let Err(error) = crate::exec::retry::backoff_before_retry(
                        execution.context(),
                        "query_engine.exec.bounded.join_leaf_stream_source_split_retry",
                    ) {
                        self.batches.clear();
                        return Some(Err(error));
                    }
                    // Retry children are work, not retained outputs: yield one before allocating its sibling.
                    self.batches.push_front((right, next_attempt));
                    self.batches.push_front((left, next_attempt));
                },
                Err(error) => {
                    self.batches.clear();
                    return Some(Err(error));
                },
            }
        }
    }

    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {
        self.batches.clear();
    }
}

fn singleton_retry_batch(chunk: SourceChunk) -> SourceChunkBatch {
    SourceChunkBatch {
        batch_index: chunk.chunk_index,
        estimated_source_read_bytes: chunk.estimated_source_read_bytes,
        source_batching_reason: "retry_split".to_string(),
        chunks: vec![chunk],
    }
}

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
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        return stream_join_leaf_source_chunk_with_consumer_retry(execution, stream, chunk, 1, consume);
    }
    match stream_join_leaf_source_batch_with_consumer(execution, stream, &batch, consume) {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            execution.metrics().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "query_engine.exec.bounded.join_leaf_stream_batch_split_retry",
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
    let output = read_join_leaf_source_batch_with_stages(execution, stream, batch)?;
    consume(output, execution)
}

fn read_join_leaf_source_batch_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: &SourceChunkBatch,
) -> Result<ResidentGpuFrame> {
    let (frame, reservation) =
        read_join_leaf_source_batch_attempt(execution, &stream.source, batch, &stream.reader_dynamic_filters)?;
    let input = ResidentGpuFrame::with_device_bytes(frame, reservation);
    apply_join_leaf_source_stream_stages(execution, stream, input)
}

/// Retry stays inside this private work boundary. A successful decode does
/// not publish source commitment before its consumer has produced output.
pub(super) fn execute_uncommitted_source_chunk<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    consume: &mut F,
) -> Result<u64>
where
    F: FnMut(&SourceChunk, usize, ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    execute_source_chunk_with_consumer_retry(execution, stream, chunk, attempt, consume, false)
}

fn execute_source_chunk_with_consumer_retry<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    consume: &mut F,
    commit_decode: bool,
) -> Result<u64>
where
    F: FnMut(&SourceChunk, usize, ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let result = super::read_serial::read_join_leaf_source_chunk_attempt_uncommitted(
        execution,
        &stream.source,
        stream.node_id,
        &chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
        if commit_decode { "join_source_read" } else { "source_chunk_read" },
    )
    .and_then(|(frame, bytes)| {
        // Materialized/pull join callers commit each decoded retry piece;
        // aggregate work defers this fact until its private partials commit.
        if commit_decode {
            execution
                .metrics()
                .record_source_chunk_commit(super::read_serial::committed_join_source_chunk_row_groups(&chunk));
        }
        apply_join_leaf_source_stream_stages(execution, stream, ResidentGpuFrame::with_device_bytes(frame, bytes))
    })
    .and_then(|frame| consume(&chunk, attempt, frame, execution));
    match result {
        Ok(()) => Ok(super::read_serial::committed_join_source_chunk_row_groups(&chunk)),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, execution.metrics());
            if attempt < stream.max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, stream.min_retry_chunk_row_groups)
            {
                execution.metrics().record_source_chunk_retry();
                crate::exec::retry::backoff_before_retry(
                    execution.context(),
                    "query_engine.exec.bounded.join_leaf_stream_split_retry",
                )?;
                let left = execute_source_chunk_with_consumer_retry(
                    execution,
                    stream,
                    left,
                    attempt + 1,
                    consume,
                    commit_decode,
                )?;
                let right = execute_source_chunk_with_consumer_retry(
                    execution,
                    stream,
                    right,
                    attempt + 1,
                    consume,
                    commit_decode,
                )?;
                return Ok(left.saturating_add(right));
            }
            Err(chunk_retry_facts(&chunk, execution).attach_to_error(error))
        },
        Err(error) => Err(error),
    }
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
    execute_source_chunk_with_consumer_retry(
        execution,
        stream,
        chunk,
        attempt,
        &mut |_, _, frame, execution| consume(frame, execution),
        true,
    )?;
    Ok(())
}

fn read_join_leaf_source_chunk_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: &SourceChunk,
    attempt: usize,
) -> Result<ResidentGpuFrame> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream")?;
    let (frame, reservation) = read_join_leaf_source_chunk_attempt(
        execution,
        &stream.source,
        stream.node_id,
        chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
    )?;
    apply_join_leaf_source_stream_stages(execution, stream, ResidentGpuFrame::with_device_bytes(frame, reservation))
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
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream_scheduler")?;
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
            execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream_scheduler")
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
                    execution.metrics().record_source_read_prefetch_abandoned();
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
            execution.metrics().record_source_read_prefetch_completed();
        }
        let observed_bytes = match observe_device_memory_bytes(&output.frame, execution.metrics()) {
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
            execution.metrics(),
        );
        execution.metrics().record_source_chunk_commit(
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
