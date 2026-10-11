//! Scheduled source reads for the materializing join leaf path, plus the start, prefetch
//! and abandon helpers the streaming scheduler loop in `stream` shares.
//!
//! Each scheduled read is a `ScheduledLocalParquetRead`: a worker job on an attempt work
//! lease (one auxiliary permit and one leased stream) that decodes one chunk, or an inline
//! read when the calling thread already runs under such a lease. The calling thread owns
//! the queue of started reads and must settle every one, by `finish` or `abandon`, before it
//! returns, because a started read holds a permit and a stream until it is joined.

use std::collections::VecDeque;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafSourceStreamDynamicFilterStage, GpuDataFrame, LocalParquetSource, NodeId,
    Result, ScheduledLocalParquetRead, SourceChunk, attach_join_source_read_resource_facts,
    join_leaf_prefetch_admitted, observe_device_memory_bytes, record_dynamic_filter_reader_predicate_rows,
};

/// Reads every chunk through scheduled reads and returns the frames in chunk order.
///
/// The calling thread blocks in `finish` for each chunk. There is no split retry here: any
/// error abandons the pending reads, drops the frames read so far, and returns, with
/// join-source-read resource facts attached to a read error. The caller reruns the whole
/// chunk list serially on a retryable error and records commit metrics again for the chunks
/// that finished before the failure.
pub(super) fn read_join_leaf_source_chunks_with_scheduler(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_attempts: usize,
) -> Result<Vec<(GpuDataFrame, u64)>> {
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_scheduler")?;
    start_next_required_join_leaf_read(execution, source, source_chunks, &mut pending, &mut next_chunk_index)?;

    let mut chunks = Vec::with_capacity(source_chunks.len());
    for (index, chunk) in source_chunks.iter().enumerate() {
        if let Err(error) =
            execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_scheduler")
        {
            abandon_join_leaf_reads(execution, pending);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let output = match scheduled.finish() {
            Ok(output) => output,
            Err(error) => {
                abandon_join_leaf_reads(execution, pending);
                return Err(attach_join_source_read_resource_facts(error, node_id, chunk, 1, max_attempts));
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
            reader_dynamic_filters,
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
        chunks.push((output.frame, observed_bytes));

        if let Err(error) =
            fill_optional_join_leaf_reads(execution, source, source_chunks, &mut pending, &mut next_chunk_index)
        {
            abandon_join_leaf_reads(execution, pending);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < source_chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                execution,
                source,
                source_chunks,
                &mut pending,
                &mut next_chunk_index,
            )
        {
            return Err(error);
        }
    }

    Ok(chunks)
}

/// Starts the read of the next unstarted chunk as a required read, and does nothing once
/// every chunk has started. Callers use it when no read is pending.
pub(super) fn start_next_required_join_leaf_read(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
) -> Result<()> {
    if let Some(chunk) = chunks.get(*next_chunk_index) {
        pending.push_back(start_join_leaf_scheduled_source_read(
            execution,
            source,
            chunk,
            "join_source_read",
            false,
            1,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
/// Starts at most one prefetch read when `max_source_reads_in_flight` exceeds 1 and no read
/// is pending.
///
/// A chunk whose minimum decode budget fails `join_leaf_prefetch_admitted` is counted as
/// skipped and left unstarted; the loop's required-read start picks it up later.
pub(super) fn fill_optional_join_leaf_reads(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
) -> Result<()> {
    // Keep one optional read ahead so decoded output cannot starve the next required read.
    let prefetch_target = usize::from(execution.context().source_scheduler_policy().max_source_reads_in_flight() > 1);
    while pending.len() < prefetch_target {
        let Some(chunk) = chunks.get(*next_chunk_index) else {
            break;
        };
        if !join_leaf_prefetch_admitted(execution.context().memory_control_policy(), chunk) {
            execution.metrics().record_source_read_prefetch_skipped();
            break;
        }
        execution.metrics().record_source_read_prefetch_started();
        pending.push_back(start_join_leaf_scheduled_source_read(
            execution,
            source,
            chunk,
            "join_source_read_prefetch",
            true,
            1,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
/// Records one source chunk attempt and starts a scheduled read of `chunk`.
///
/// Waits, cancellably, for an attempt auxiliary permit when the calling thread is not
/// already under a work lease; under one, the read runs to completion inside this call.
/// `prefetch` labels the read in the metrics and `attempt` names it in observations.
pub(super) fn start_join_leaf_scheduled_source_read(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    prefetch: bool,
    attempt: usize,
) -> Result<ScheduledLocalParquetRead> {
    execution.context().check_cancellation("query_engine.exec.bounded.join_leaf_source_scheduler_start")?;
    execution.metrics().record_source_chunk_attempt();
    ScheduledLocalParquetRead::spawn(
        execution.context().clone(),
        execution.metrics().clone(),
        source.clone(),
        chunk.clone(),
        execution.device_budget_bytes(),
        memory_boundary,
        prefetch,
        attempt,
    )
}

/// Joins and discards every pending read, newest first. Blocks until each worker has
/// finished, drops the decoded frames, and records a late decode failure as a cleanup
/// failure of the attempt instead of raising it.
pub(super) fn abandon_join_leaf_reads(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut pending: VecDeque<ScheduledLocalParquetRead>,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(execution);
    }
}
