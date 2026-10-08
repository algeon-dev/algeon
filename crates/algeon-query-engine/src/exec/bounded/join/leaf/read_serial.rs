//! Serial reads of join leaf chunks and batches that return frames to the caller.
//!
//! These functions run on the calling thread and carry the split retry for the
//! materializing path. That path has no consumer. A retryable (query-cap) error splits a batch
//! in two, or a single chunk by row groups, and reads the halves in order. A failed call
//! returns no frames, because a half that succeeded before a later half failed is dropped
//! with the partial result. Source commit metrics are recorded when a decode succeeds; a
//! failed attempt records none.

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafSourceStreamDynamicFilterStage, GpuDataFrame, LocalParquetSource, NodeId,
    QueryError, Result, SourceChunk, SourceChunkBatch, attach_join_source_read_resource_facts, chunk_retry_facts,
    is_retryable_chunk_error, join_leaf_source_chunk_batches_for_execution, observe_device_memory_bytes,
    read_local_parquet_chunk_batch, read_local_parquet_chunk_with_reader_dynamic_filter_count,
    record_chunk_retry_error, record_dynamic_filter_reader_predicate_batch_rows,
    record_dynamic_filter_reader_predicate_rows, source_chunk_batch_target_bytes, split_retryable_chunk,
};
use crate::exec::{retry::backoff_before_retry, source::observed_chunk_row_group_count};

/// Cuts `source_chunks` into batches against the memory-control target and reads them
/// in order. See `read_join_leaf_source_batches_serial`.
pub(super) fn read_join_leaf_source_chunks_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, u64)>> {
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
    )
}

/// Reads each batch in order and returns every frame with its observed device bytes.
///
/// A singleton batch takes the chunk retry and a larger batch takes the batch-split retry.
/// The first error that survives retry is returned and the frames read so far are dropped.
pub(super) fn read_join_leaf_source_batches_serial(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    batches: Vec<SourceChunkBatch>,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, u64)>> {
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
            ) {
                Ok(mut outputs) => {
                    chunks.append(&mut outputs);
                },
                Err(error) => {
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
        ) {
            Ok(mut outputs) => {
                chunks.append(&mut outputs);
            },
            Err(error) => {
                return Err(error);
            },
        }
    }

    Ok(chunks)
}

/// Reads one batch as a single call, splitting it in two on a query-cap rejection.
///
/// A batch split is not counted against `max_attempts`; recursion ends at singleton
/// batches, and those use the chunk retry. A batch that cannot split returns its error
/// without retry facts. Each split waits for the retry backoff, and cancellation ends the
/// wait.
pub(super) fn read_join_leaf_source_batch_or_split(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    batch: SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, u64)>> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_chunk_batch_retry")?;
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
        );
    }
    match read_join_leaf_source_batch_attempt(execution, source, &batch, reader_dynamic_filters) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => {
            execution.metrics().record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            backoff_before_retry(execution.context(), "query_engine.exec.bounded.join_leaf_source_batch_split_retry")?;
            let mut outputs = read_join_leaf_source_batch_or_split(
                execution,
                source,
                node_id,
                left,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
            )?;
            match read_join_leaf_source_batch_or_split(
                execution,
                source,
                node_id,
                right,
                reader_dynamic_filters,
                max_attempts,
                min_retry_chunk_row_groups,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => Err(error),
            }
        },
        Err(error) => Err(error),
    }
}
/// One read of a batch: records an attempt per chunk, decodes, observes device bytes, and
/// records the commit for all chunks of the batch at once.
pub(super) fn read_join_leaf_source_batch_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    batch: &SourceChunkBatch,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
) -> Result<(GpuDataFrame, u64)> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_chunk_batch")?;
    for _chunk in &batch.chunks {
        execution.metrics().record_source_chunk_attempt();
    }
    let source_token_budget = execution.device_budget_bytes();
    let frame = match read_local_parquet_chunk_batch(
        execution.context(),
        source,
        batch,
        source_token_budget,
        "join_source_read",
        reader_dynamic_filters.len(),
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error);
        },
    };
    let observed_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    record_dynamic_filter_reader_predicate_batch_rows(reader_dynamic_filters, batch, &frame, execution.metrics());
    execution.metrics().record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
    Ok((frame, observed_bytes))
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
/// Reads `chunk` at `attempt`; a query-cap rejection goes to
/// `retry_join_leaf_source_chunk_after_error`. Other errors return unchanged.
pub(super) fn read_join_leaf_source_chunk_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, u64)>> {
    match read_join_leaf_source_chunk_attempt(
        execution,
        source,
        node_id,
        &chunk,
        reader_dynamic_filters,
        attempt,
        max_attempts,
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
        ),
        Err(error) => Err(error),
    }
}
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
/// Splits `chunk` by row groups and reads both halves at `attempt + 1`, in row-group order.
///
/// Splits only while `attempt < max_attempts` and the chunk has more row groups than
/// `min_retry_chunk_row_groups`. Otherwise the original error returns with
/// `chunk_retry_facts` (`fail_min_chunk` or `retry_attempts_exhausted`). The first half
/// finishes, with its own retries, before the second half starts.
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
) -> Result<Vec<(GpuDataFrame, u64)>> {
    record_chunk_retry_error(&error, execution.metrics());
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        execution.metrics().record_source_chunk_retry();
        backoff_before_retry(execution.context(), "query_engine.exec.bounded.join_leaf_source_split_retry")?;
        let mut outputs = read_join_leaf_source_chunk_with_retry(
            execution,
            source,
            node_id,
            left,
            reader_dynamic_filters,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
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
        )?);
        return Ok(outputs);
    }

    Err(chunk_retry_facts(&chunk, execution).attach_to_error(error))
}
/// One read of a chunk that records the source chunk commit once the decode succeeds.
pub(super) fn read_join_leaf_source_chunk_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: &SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
) -> Result<(GpuDataFrame, u64)> {
    let output = read_join_leaf_source_chunk_attempt_uncommitted(
        execution,
        source,
        node_id,
        chunk,
        reader_dynamic_filters,
        attempt,
        max_attempts,
        "join_source_read",
    )?;
    execution.metrics().record_source_chunk_commit(committed_join_source_chunk_row_groups(chunk));
    Ok(output)
}

/// One read of a chunk that leaves the commit to the caller.
///
/// Checks cancellation, records an attempt, reads the chunk under a budget token labeled
/// `budget_boundary`, and observes device bytes. A read error gains the join-source-read
/// resource facts for `attempt` of `max_attempts`.
#[expect(clippy::too_many_arguments, reason = "source retry preserves caller-owned commitment and budget boundary")]
pub(super) fn read_join_leaf_source_chunk_attempt_uncommitted(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    chunk: &SourceChunk,
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    attempt: usize,
    max_attempts: usize,
    budget_boundary: &'static str,
) -> Result<(GpuDataFrame, u64)> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_chunk")?;
    execution.metrics().record_source_chunk_attempt();
    let source_token_budget = execution.device_budget_bytes();
    let frame = read_local_parquet_chunk_with_reader_dynamic_filter_count(
        execution.context(),
        source,
        chunk,
        source_token_budget,
        budget_boundary,
        reader_dynamic_filters.len(),
        execution.metrics(),
    )
    .map_err(|error| attach_join_source_read_resource_facts(error, node_id, chunk, attempt, max_attempts))?;
    let observed_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, chunk, &frame, execution.metrics());
    Ok((frame, observed_bytes))
}

/// Row groups to report as committed for `chunk`: the observed count when known, else the
/// planned count. When the plan holds no count for a local chunk that reads all row groups,
/// the observed count opens the file footer on the calling thread.
pub(super) fn committed_join_source_chunk_row_groups(chunk: &SourceChunk) -> u64 {
    observed_chunk_row_group_count(chunk).unwrap_or_else(|| chunk.row_group_count()) as u64
}
