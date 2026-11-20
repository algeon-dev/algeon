//! Consumers of a deferred Parquet join leaf: a pull stream and push loops.
//!
//! A `BoundedJoinLeafLocalParquetSourceStream` carries planned `SourceChunk`s. This module
//! turns it into frames for a join input. The pull stream yields one planned batch per
//! call. The push loops call a consumer closure with each frame in batch order.
//!
//! Where work runs. Reads, post-decode stages and the consumer run on the calling thread
//! (the attempt's coordinator or, inside lane work, that lane's thread). A pull stream or push
//! loop that keeps a read ahead (`chunks::read_ahead`) starts the next batch's read on an
//! attempt work lease (an auxiliary permit plus a leased stream, executed as a pool job) once
//! the current batch is decoded, and the next pull or batch blocks in `finish` on the worker
//! join and on that read's readiness event. A thread already running under a work lease keeps
//! no read ahead and never takes a second permit.
//!
//! Retry. A chunk error is retryable only when it is a query-cap rejection
//! (`QueryMemoryLimitExceeded`, via `is_retryable_chunk_error`); physical device exhaustion and
//! every other code return unchanged. A retryable failure of a single chunk splits it by row
//! groups after a backoff, up to `max_attempts` and down to `min_retry_chunk_row_groups`;
//! when no split is left the error carries `chunk_retry_facts`. A multi-chunk batch splits in
//! two without counting attempts. The retried unit includes the post-decode stages and, in
//! the push loops, the consumer. A consumer can be called again with the frames of a split
//! chunk and must publish nothing from a call that returns an error.

use std::{collections::VecDeque, slice};

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, GpuDataFrame, JOIN_LEAF_READ_AHEAD_BOUNDARY,
    ReadAheadStep, ResidentGpuFrame, Result, SourceChunk, SourceChunkBatch, SourceReadAhead,
    apply_join_leaf_source_stream_stages, chunk_retry_facts, is_retryable_chunk_error,
    join_leaf_source_chunk_batches_for_execution, observe_device_memory_bytes, read_join_leaf_source_batch_attempt,
    record_chunk_retry_error, record_dynamic_filter_reader_predicate_batch_rows,
    record_dynamic_filter_reader_predicate_rows, source_chunk_batch_target_bytes, source_chunk_batches,
    split_retryable_chunk,
};
use crate::exec::{
    bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource},
    retry::backoff_before_retry,
};

/// Exposes a deferred Parquet leaf as a pull-based resident frame stream.
///
/// Each pull reads one planned source batch (with the same split retry as the
/// serial push path) and yields its frames. When `source_read_ahead_applies`, each pull also starts the read of the next batch before it
/// yields, so a consumer that streams its own output holds at most one leaf
/// batch plus the one read ahead of it. Consumer failures cannot split the
/// batch they were handed; consumers with their own retry shape (partitioned
/// probes, gather-map slices) recover on their side, and their query-cap
/// retries release the read ahead first (`backoff_before_retry`).
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
            read_ahead: SourceReadAhead::for_stream(execution, &stream, JOIN_LEAF_READ_AHEAD_BOUNDARY),
            stream,
            batches: batches.into_iter().map(|batch| (batch, 1)).collect(),
        }),
    )
}

/// Pull source over the planned batches of one leaf stream.
///
/// `batches` pairs each pending batch with its attempt number. A retry split pushes both
/// children to the front. The next pull yields the first child, and its sibling is not
/// read until the consumer asks again. A yielded frame belongs to the consumer. The source
/// owns at most the read of the next batch; `release`, and dropping the source, join and
/// discard it.
struct JoinLeafSourceFrameStreamSource {
    stream: BoundedJoinLeafLocalParquetSourceStream,
    batches: VecDeque<(SourceChunkBatch, usize)>,
    read_ahead: Option<SourceReadAhead>,
}

impl ResidentFrameStreamSource for JoinLeafSourceFrameStreamSource {
    /// Reads, decodes and applies the stages for the next pending batch on the calling
    /// thread. `None` means every batch was yielded or an earlier pull failed.
    ///
    /// A memory failure while the next batch's read is pending releases that read and reads
    /// the batch again with no read ahead. A retryable error splits the batch (a singleton
    /// chunk counts an attempt) and loops to read the first child. A terminal error, or a
    /// singleton chunk with no split left, clears the pending queue and returns the error;
    /// later pulls return `None`.
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            let (batch, attempt) = self.batches.pop_front()?;
            let step = self.read_ahead.as_ref().map(|read_ahead| ReadAheadStep {
                read_ahead,
                next: self.batches.front().map(|(next, _)| next.clone()),
            });
            let output = if batch.is_singleton() {
                read_join_leaf_source_chunk_with_stages(
                    execution,
                    &self.stream,
                    &batch.chunks[0],
                    attempt,
                    step.as_ref(),
                )
            } else {
                read_join_leaf_source_batch_with_stages(execution, &self.stream, &batch, step.as_ref())
            };
            if let Err(error) = &output
                && step.as_ref().is_some_and(|step| step.release_after_failure(error))
            {
                self.batches.push_front((batch, attempt));
                continue;
            }
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
                            self.clear_pending();
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
                        self.clear_pending();
                        return Some(Err(error));
                    }
                    // Retry children are pending work and hold no retained output. Yield one
                    // before allocating its sibling.
                    self.batches.push_front((right, next_attempt));
                    self.batches.push_front((left, next_attempt));
                },
                Err(error) => {
                    self.clear_pending();
                    return Some(Err(error));
                },
            }
        }
    }

    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {
        self.clear_pending();
    }
}

impl JoinLeafSourceFrameStreamSource {
    /// Drops the pending batches and joins and discards the read ahead, if one is pending.
    fn clear_pending(&mut self) {
        self.batches.clear();
        if let Some(read_ahead) = &self.read_ahead {
            read_ahead.release();
        }
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
/// already consumed stay with the consumer. No read is kept ahead.
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
    stream_join_leaf_source_batches(execution, stream, batches, None, consume)
}

/// Pushes `batches` to `consume` in order, with the retries of
/// `stream_join_leaf_source_chunks_serial`.
///
/// With `read_ahead`, each batch's work starts the read of the batch after it once its own
/// read has completed, so that read runs while the stages and `consume` run. A memory failure
/// while that read is pending releases it and reruns the failed batch with no read ahead, and
/// the released read ahead starts no further reads.
pub(super) fn stream_join_leaf_source_batches<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batches: Vec<SourceChunkBatch>,
    read_ahead: Option<&SourceReadAhead>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let mut batches = batches.into_iter().peekable();
    while let Some(batch) = batches.next() {
        let step = read_ahead.map(|read_ahead| ReadAheadStep { read_ahead, next: batches.peek().cloned() });
        stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, batch, step.as_ref(), consume)?;
    }
    Ok(())
}

/// Reads one batch and calls `consume` with its frame, splitting the batch on a retryable
/// error from the read, the stages or `consume` itself.
///
/// After a split the first half is read and consumed before the second; `consume` can see
/// the first half's frame and then fail on the second half. A batch that cannot split
/// returns its error without retry facts. A cancelled attempt ends the retry at the backoff.
///
/// `read_ahead` applies to this call's batch only, as in `stream_join_leaf_source_batches`;
/// the halves of a split run with none.
pub(super) fn stream_join_leaf_source_batch_or_split_with_consumer<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: SourceChunkBatch,
    read_ahead: Option<&ReadAheadStep<'_>>,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one chunk");
        execute_source_chunk_with_consumer_retry(
            execution,
            stream,
            chunk,
            1,
            &mut |_, _, frame, execution| consume(frame, execution),
            true,
            read_ahead,
        )?;
        return Ok(());
    }
    let result = read_join_leaf_source_batch_with_stages(execution, stream, &batch, read_ahead)
        .and_then(|output| consume(output, execution));
    // The failed attempt ran next to a pending read ahead, so a memory failure may not be this
    // batch's own; the batch reruns with no read ahead.
    if let Err(error) = &result
        && read_ahead.is_some_and(|step| step.release_after_failure(error))
    {
        return stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, batch, None, consume);
    }
    match result {
        Ok(()) => Ok(()),
        Err(error) if is_retryable_chunk_error(&error) => {
            execution.metrics().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            backoff_before_retry(execution.context(), "query_engine.exec.bounded.join_leaf_stream_batch_split_retry")?;
            stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, left, None, consume)?;
            stream_join_leaf_source_batch_or_split_with_consumer(execution, stream, right, None, consume)
        },
        Err(error) => Err(error),
    }
}

/// One attempt at a multi-chunk batch: the read (or the read started for it with
/// `read_ahead`), which records the batch's commit, then the stages. With `read_ahead`, the
/// step's next read starts once the batch is decoded.
fn read_join_leaf_source_batch_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    batch: &SourceChunkBatch,
    read_ahead: Option<&ReadAheadStep<'_>>,
) -> Result<ResidentGpuFrame> {
    let prefetched = match read_ahead {
        Some(step) => step.read_ahead.finish(&batch.chunks)?,
        None => None,
    };
    let (frame, device_bytes) = match prefetched {
        Some(output) => {
            let device_bytes = observe_device_memory_bytes(&output.frame, execution.metrics())?;
            record_dynamic_filter_reader_predicate_batch_rows(
                &stream.reader_dynamic_filters,
                batch,
                &output.frame,
                execution.metrics(),
            );
            execution
                .metrics()
                .record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
            (output.frame, device_bytes)
        },
        None => read_join_leaf_source_batch_attempt(execution, &stream.source, batch, &stream.reader_dynamic_filters)?,
    };
    if let Some(step) = read_ahead {
        step.start_next(execution, &stream.source, stream.reader_dynamic_filters.len())?;
    }
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
    read_ahead: Option<ReadAheadStep<'_>>,
    consume: &mut F,
) -> Result<u64>
where
    F: FnMut(&SourceChunk, usize, ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    execute_source_chunk_with_consumer_retry(execution, stream, chunk, attempt, consume, false, read_ahead.as_ref())
}

/// Reads `chunk`, applies the stages and calls `consume`; on a retryable error splits the
/// chunk and runs both halves through this function at `attempt + 1`.
///
/// `commit_decode` chooses who records the source chunk commit. When true, the commit is
/// recorded right after each decode, before the stages and `consume` run; a retry of that
/// piece records its row groups again. When false the caller records the commit after
/// its own partials are committed. With no split left, the error gains `chunk_retry_facts`.
///
/// With `read_ahead`, the chunk is taken from the read started for it when there is one, and
/// the step's next read starts once this chunk is decoded. A memory failure while that next
/// read is still pending releases it and runs this chunk again, unsplit and with no read
/// ahead; the released read ahead starts no further reads, so the rest of the stream runs
/// serially.
fn execute_source_chunk_with_consumer_retry<F>(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: SourceChunk,
    attempt: usize,
    consume: &mut F,
    commit_decode: bool,
    read_ahead: Option<&ReadAheadStep<'_>>,
) -> Result<u64>
where
    F: FnMut(&SourceChunk, usize, ResidentGpuFrame, &mut BoundedExecutionAttempt<'_>) -> Result<()>,
{
    let result = read_source_chunk_for_execution(
        execution,
        stream,
        &chunk,
        attempt,
        if commit_decode { "join_source_read" } else { "source_chunk_read" },
        read_ahead,
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
    // The failed attempt ran next to a pending read ahead, so a memory failure may not be this
    // chunk's own; the chunk reruns as the scheduler's serial fallback would.
    if let Err(error) = &result
        && read_ahead.is_some_and(|step| step.release_after_failure(error))
    {
        return execute_source_chunk_with_consumer_retry(
            execution,
            stream,
            chunk,
            attempt,
            consume,
            commit_decode,
            None,
        );
    }
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
                    None,
                )?;
                let right = execute_source_chunk_with_consumer_retry(
                    execution,
                    stream,
                    right,
                    attempt + 1,
                    consume,
                    commit_decode,
                    None,
                )?;
                return Ok(left.saturating_add(right));
            }
            Err(chunk_retry_facts(&chunk, execution).attach_to_error(error))
        },
        Err(error) => Err(error),
    }
}

/// One read of `chunk` for `execute_source_chunk_with_consumer_retry`, returning the frame and
/// its observed device bytes.
///
/// With `read_ahead`, the read started for `chunk` supplies the frame when it succeeded, and
/// the chunk is read here otherwise. Once the frame is decoded, the step's next read starts.
fn read_source_chunk_for_execution(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: &SourceChunk,
    attempt: usize,
    budget_boundary: &'static str,
    read_ahead: Option<&ReadAheadStep<'_>>,
) -> Result<(GpuDataFrame, u64)> {
    let prefetched = match read_ahead {
        Some(step) => step.read_ahead.finish(slice::from_ref(chunk))?,
        None => None,
    };
    let output = match prefetched {
        Some(output) => {
            let bytes = observe_device_memory_bytes(&output.frame, execution.metrics())?;
            record_dynamic_filter_reader_predicate_rows(
                &stream.reader_dynamic_filters,
                chunk,
                &output.frame,
                execution.metrics(),
            );
            (output.frame, bytes)
        },
        None => super::read_serial::read_join_leaf_source_chunk_attempt_uncommitted(
            execution,
            &stream.source,
            stream.node_id,
            chunk,
            &stream.reader_dynamic_filters,
            attempt,
            stream.max_attempts,
            budget_boundary,
        )?,
    };
    if let Some(step) = read_ahead {
        step.start_next(execution, &stream.source, stream.reader_dynamic_filters.len())?;
    }
    Ok(output)
}

/// One attempt at one chunk for the pull stream: cancellation check, read (or the read
/// started for it with `read_ahead`), then stages. The read records the source chunk commit;
/// retry belongs to the caller.
fn read_join_leaf_source_chunk_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    chunk: &SourceChunk,
    attempt: usize,
    read_ahead: Option<&ReadAheadStep<'_>>,
) -> Result<ResidentGpuFrame> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_stream")?;
    let (frame, device_bytes) =
        read_source_chunk_for_execution(execution, stream, chunk, attempt, "join_source_read", read_ahead)?;
    execution.metrics().record_source_chunk_commit(super::read_serial::committed_join_source_chunk_row_groups(chunk));
    apply_join_leaf_source_stream_stages(execution, stream, ResidentGpuFrame::with_device_bytes(frame, device_bytes))
}
