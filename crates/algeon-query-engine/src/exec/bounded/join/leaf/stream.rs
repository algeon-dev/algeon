//! Consumers of a deferred Parquet join leaf: a pull stream and push loops.
//!
//! A `BoundedJoinLeafLocalParquetSourceStream` carries planned `SourceChunk`s. This module
//! turns it into frames for a join input. The pull stream yields one planned batch per
//! call. The push loops call a consumer closure with each frame, either serially in batch
//! order or through the read scheduler.
//!
//! Where work runs. Serial reads, post-decode stages and the consumer run on the calling
//! thread (the attempt's coordinator or, inside lane work, that lane's thread). The
//! scheduler loop starts reads on attempt work leases (an auxiliary permit plus a leased
//! stream, executed as a pool job); the calling thread then blocks in `finish` on the worker
//! join and on the chunk's readiness event. A thread already running under a work lease reads
//! inline and never takes a second permit.
//!
//! Retry. A chunk error is retryable only when it is a query-cap rejection
//! (`QueryMemoryLimitExceeded`, via `is_retryable_chunk_error`); physical device exhaustion and
//! every other code return unchanged. A retryable failure of a single chunk splits it by row
//! groups after a backoff, up to `max_attempts` and down to `min_retry_chunk_row_groups`;
//! when no split is left the error carries `chunk_retry_facts`. A multi-chunk batch splits in
//! two without counting attempts. The retried unit includes the post-decode stages and, in
//! the push loops, the consumer. A consumer can be called again with the frames of a split
//! chunk and must publish nothing from a call that returns an error.

use std::collections::VecDeque;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame, Result, SourceChunk,
    SourceChunkBatch, abandon_join_leaf_reads, apply_join_leaf_source_stream_stages,
    attach_join_source_read_resource_facts, chunk_retry_facts, fill_optional_join_leaf_reads, is_retryable_chunk_error,
    join_leaf_source_chunk_batches_for_execution, observe_device_memory_bytes, read_join_leaf_source_batch_attempt,
    read_join_leaf_source_chunk_attempt, record_chunk_retry_error, record_dynamic_filter_reader_predicate_rows,
    source_chunk_batch_target_bytes, source_chunk_batches, split_retryable_chunk, start_next_required_join_leaf_read,
};
use crate::exec::{
    bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource},
    retry::backoff_before_retry,
};

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
    let max_frame_bytes_hint = batches.iter().filter_map(|batch| batch.estimated_source_read_bytes).max();
    let total_bytes_hint = stream.estimated_source_read_bytes();
    let aggregate_num_rows = stream.estimated_source_rows().unwrap_or(0);
    let num_columns = stream.output_column_count();
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(JoinLeafSourceFrameStreamSource {
            stream,
            batches: batches.into_iter().map(|batch| (batch, 1)).collect(),
        }),
    )
}

/// Pull source over the planned batches of one leaf stream.
///
/// `batches` pairs each pending batch with its attempt number. A retry split pushes both
/// children to the front. The next pull yields the first child, and its sibling is not
/// read until the consumer asks again. The source owns no device frames: a yielded frame
/// belongs to the consumer, and `release` only discards pending work.
struct JoinLeafSourceFrameStreamSource {
    stream: BoundedJoinLeafLocalParquetSourceStream,
    batches: VecDeque<(SourceChunkBatch, usize)>,
}

impl ResidentFrameStreamSource for JoinLeafSourceFrameStreamSource {
    /// Reads, decodes and applies the stages for the next pending batch on the calling
    /// thread. `None` means every batch was yielded or an earlier pull failed.
    ///
    /// A retryable error splits the batch (a singleton chunk counts an attempt) and loops to
    /// read the first child. A terminal error, or a singleton chunk with no split left,
    /// clears the pending queue and returns the error; later pulls return `None`.
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
                    if let Err(error) = backoff_before_retry(
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

/// Wraps a retry child chunk in a one-chunk batch labeled `retry_split` in the metrics.
fn singleton_retry_batch(chunk: SourceChunk) -> SourceChunkBatch {
    SourceChunkBatch {
        batch_index: chunk.chunk_index,
        estimated_source_read_bytes: chunk.estimated_source_read_bytes,
        source_batching_reason: "retry_split".to_string(),
        chunks: vec![chunk],
    }
}

/// Pushes every chunk of `stream` to `consume`, one batch at a time on the calling thread.
///
/// Batches are cut against the memory-control batching target. A singleton batch takes the
/// chunk retry; a multi-chunk batch takes the batch-split retry. `consume` follows the
/// retry contract in the module docs. Returns the first error that survives retry; frames
/// already consumed stay with the consumer.
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

/// Reads one batch and calls `consume` with its frame, splitting the batch on a retryable
/// error from the read, the stages or `consume` itself.
///
/// After a split the first half is read and consumed before the second; `consume` can see
/// the first half's frame and then fail on the second half. A batch that cannot split
/// returns its error without retry facts. A cancelled attempt ends the retry at the backoff.
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
            backoff_before_retry(execution.context(), "query_engine.exec.bounded.join_leaf_stream_batch_split_retry")?;
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
    let (frame, device_bytes) =
        read_join_leaf_source_batch_attempt(execution, &stream.source, batch, &stream.reader_dynamic_filters)?;
    let input = ResidentGpuFrame::with_device_bytes(frame, device_bytes);
    apply_join_leaf_source_stream_stages(execution, stream, input)
}

/// Retry stays inside this private work boundary. A successful decode does
/// not publish source commitment before its consumer has produced output.
///
/// `consume` receives each chunk piece with its attempt number. Returns the row groups the
/// pieces cover, summed across any retry split, for the caller to commit once its own
/// partials are committed. An error means no piece may be treated as committed; pieces that
/// `consume` already accepted before a later piece failed are the consumer's to discard.
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

/// Reads `chunk`, applies the stages and calls `consume`; on a retryable error splits the
/// chunk and runs both halves through this function at `attempt + 1`.
///
/// `commit_decode` chooses who records the source chunk commit. When true, the commit is
/// recorded right after each decode, before the stages and `consume` run; a retry of that
/// piece records its row groups again. When false the caller records the commit after
/// its own partials are committed. With no split left, the error gains `chunk_retry_facts`.
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
                backoff_before_retry(execution.context(), "query_engine.exec.bounded.join_leaf_stream_split_retry")?;
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

/// Single-chunk push with the split retry; the commit is recorded at decode time.
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

/// One attempt at one chunk for the pull stream: cancellation check, read, then stages.
/// The read records the source chunk commit; retry belongs to the caller.
fn read_join_leaf_source_chunk_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: &SourceChunk,
    attempt: usize,
) -> Result<ResidentGpuFrame> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream")?;
    let (frame, device_bytes) = read_join_leaf_source_chunk_attempt(
        execution,
        &stream.source,
        stream.node_id,
        chunk,
        &stream.reader_dynamic_filters,
        attempt,
        stream.max_attempts,
    )?;
    apply_join_leaf_source_stream_stages(execution, stream, ResidentGpuFrame::with_device_bytes(frame, device_bytes))
}

/// Pushes the chunks to `consume` in order, reading each through a scheduled source read.
///
/// Reads run on attempt work leases; this thread starts them and blocks in `finish` for
/// each chunk, and every chunk's read is started before it is awaited. When
/// `max_source_reads_in_flight` exceeds 1 and the chunk passes
/// `join_leaf_prefetch_admitted`, one further read is started after `consume` returns; a
/// refused prefetch is counted as skipped. The next read therefore overlaps the consumer
/// only through GPU work the consumer enqueued and has not finished.
///
/// The source chunk commit for a chunk is recorded as soon as its read completes, before
/// the stages and `consume` run. The scheduler never splits a chunk. A retryable error from
/// the read, the stages or `consume` abandons the pending reads (each is joined and its
/// frame dropped) and re-runs the remaining chunks from the failing one through the serial
/// loop, where the split retry applies and that chunk's commit is recorded again. Any other
/// error abandons the pending reads and returns.
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
    start_next_required_join_leaf_read(execution, &stream.source, &stream.chunks, &mut pending, &mut next_chunk_index)?;

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
        let output = match scheduled.finish() {
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
            )
        {
            return Err(error);
        }
    }

    Ok(())
}
