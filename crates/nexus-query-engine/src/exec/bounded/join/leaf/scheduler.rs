use std::collections::VecDeque;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafSourceStreamDynamicFilterStage, GpuDataFrame, LocalParquetSource, NodeId,
    Result, ScheduledLocalParquetRead, SourceChunk, attach_join_source_read_resource_facts,
    join_leaf_prefetch_budget_bytes, observe_device_memory_bytes, record_dynamic_filter_reader_predicate_rows,
    release_join_source_chunks_reverse, source_read_budget_bytes,
};

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
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler")?;
    start_next_required_join_leaf_read(
        execution,
        source,
        source_chunks,
        &mut pending,
        &mut next_chunk_index,
        max_attempts,
    )?;

    let mut chunks = Vec::with_capacity(source_chunks.len());
    for (index, chunk) in source_chunks.iter().enumerate() {
        if let Err(error) =
            execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler")
        {
            abandon_join_leaf_reads(execution, pending);
            release_join_source_chunks_reverse(chunks);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let output = match scheduled.finish(execution) {
            Ok(output) => output,
            Err(error) => {
                abandon_join_leaf_reads(execution, pending);
                release_join_source_chunks_reverse(chunks);
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
                release_join_source_chunks_reverse(chunks);
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

        if let Err(error) = fill_optional_join_leaf_reads(
            execution,
            source,
            source_chunks,
            &mut pending,
            &mut next_chunk_index,
            max_attempts,
        ) {
            abandon_join_leaf_reads(execution, pending);
            release_join_source_chunks_reverse(chunks);
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
                max_attempts,
            )
        {
            release_join_source_chunks_reverse(chunks);
            return Err(error);
        }
    }

    Ok(chunks)
}

pub(super) fn start_next_required_join_leaf_read(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    if let Some(chunk) = chunks.get(*next_chunk_index) {
        let requested_bytes = source_read_budget_bytes(execution.context().memory_control_policy(), chunk);
        pending.push_back(start_join_leaf_scheduled_source_read(
            execution,
            source,
            chunk,
            requested_bytes,
            "join_source_read",
            false,
            false,
            1,
            max_attempts,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
pub(super) fn fill_optional_join_leaf_reads(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    // Keep one optional read ahead so decoded output cannot starve the next required read.
    let prefetch_target = usize::from(execution.context().source_scheduler_policy().max_source_reads_in_flight() > 1);
    while pending.len() < prefetch_target {
        let Some(chunk) = chunks.get(*next_chunk_index) else {
            break;
        };
        let Some(requested_bytes) = join_leaf_prefetch_budget_bytes(execution.context().memory_control_policy(), chunk)
        else {
            execution.metrics().record_source_read_prefetch_skipped();
            break;
        };
        execution.metrics().record_source_read_prefetch_started();
        pending.push_back(start_join_leaf_scheduled_source_read(
            execution,
            source,
            chunk,
            requested_bytes,
            "join_source_read_prefetch",
            false,
            true,
            1,
            max_attempts,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn start_join_leaf_scheduled_source_read(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    requested_bytes: u64,
    memory_boundary: &'static str,
    hard_token_budget: bool,
    prefetch: bool,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetRead> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler_start")?;
    execution.metrics().record_source_chunk_attempt();
    let source_token_budget = if hard_token_budget { requested_bytes } else { execution.device_budget_bytes() };
    ScheduledLocalParquetRead::spawn(
        execution.context().clone(),
        execution.metrics().clone(),
        source.clone(),
        chunk.clone(),
        source_token_budget,
        memory_boundary,
        prefetch,
        attempt,
        max_attempts,
    )
}

pub(super) fn abandon_join_leaf_reads(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut pending: VecDeque<ScheduledLocalParquetRead>,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(execution);
    }
}
