use super::*;

pub(super) fn read_join_leaf_source_chunks_with_scheduler(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler")?;
    start_next_required_join_leaf_read(
        context,
        source,
        source_chunks,
        node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
        max_attempts,
    )?;

    let mut chunks = Vec::with_capacity(source_chunks.len());
    for (index, chunk) in source_chunks.iter().enumerate() {
        if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler") {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(attach_join_source_read_resource_facts(error, node_id, chunk, 1, max_attempts));
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
                node_id,
                chunk,
                metrics,
                1,
                max_attempts,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            }
            let retained_device_bytes = output.reservation.bytes();
            if let Err(error) = insert_bounded_local_parquet_source_cache(
                context,
                source,
                chunk,
                &output.frame,
                &mut output.reservation,
                reservations,
                metrics,
                "join_source_read",
                1,
                max_attempts,
                retained_device_bytes,
            ) {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            }
        } else if let Err(error) = observe_device_memory_bytes(&output.frame, metrics) {
            drop(output.frame);
            reservations.release(output.reservation, metrics);
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        record_dynamic_filter_reader_predicate_rows(reader_dynamic_filters, &output.chunk, &output.frame, metrics);
        metrics.record_source_chunk_commit(
            output
                .observed_row_group_count
                .or_else(|| chunk.known_read_row_group_count().map(|row_groups| row_groups as u64))
                .unwrap_or_else(|| chunk.row_group_count() as u64),
        );
        chunks.push((output.frame, output.reservation));
        if let Err(error) = reservations.acknowledge_requested_grant_shrink() {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }

        if let Err(error) = fill_optional_join_leaf_reads(
            context,
            source,
            source_chunks,
            node_id,
            reservations,
            metrics,
            &mut pending,
            &mut next_chunk_index,
            max_attempts,
        ) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
        if pending.is_empty()
            && index + 1 < source_chunks.len()
            && let Err(error) = start_next_required_join_leaf_read(
                context,
                source,
                source_chunks,
                node_id,
                reservations,
                metrics,
                &mut pending,
                &mut next_chunk_index,
                max_attempts,
            )
        {
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        }
    }

    Ok(chunks)
}

pub(super) fn start_next_required_join_leaf_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    if let Some(chunk) = chunks.get(*next_chunk_index) {
        let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
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
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
    max_attempts: usize,
) -> Result<()> {
    // Keep only one unobserved optional source read ahead. The source decoded
    // output preflight is still an unresolved lower-level contract, so deeper
    // speculative reads can starve the next front-of-queue read when it grows
    // to observed bytes.
    let prefetch_target = usize::from(context.source_scheduler_policy().max_source_reads_in_flight() > 1);
    while pending.len() < prefetch_target {
        let Some(chunk) = chunks.get(*next_chunk_index) else {
            break;
        };
        let Some(requested_bytes) = join_leaf_prefetch_requested_bytes(context.memory_control_policy(), chunk) else {
            metrics.record_source_read_prefetch_skipped();
            break;
        };
        if !reservations.can_reserve(requested_bytes) {
            metrics.record_source_read_prefetch_skipped();
            break;
        }
        metrics.record_source_read_prefetch_started();
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
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
pub(super) fn start_join_leaf_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    requested_bytes: u64,
    memory_boundary: &'static str,
    strict_token_budget: bool,
    prefetch: bool,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetRead> {
    context.check_cancellation("nexus_query_engine.exec.bounded.join_leaf_source_scheduler_start")?;
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(ScheduledLocalParquetRead::ready(output, memory_boundary, prefetch));
    }
    let request = join_source_read_reservation_request_for_boundary(
        context.memory_control_policy(),
        requested_bytes,
        reservations.budget(),
        node_id,
        chunk,
        memory_boundary,
        attempt,
        max_attempts,
    );
    let reservation = reservations.try_reserve(request, metrics)?;
    let source_token_budget =
        if strict_token_budget { reservation.bytes() } else { reservations.available_bytes_including(&reservation) };
    ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        memory_boundary,
        prefetch,
        attempt,
        max_attempts,
    )
}

pub(super) fn abandon_join_leaf_reads(
    mut pending: VecDeque<ScheduledLocalParquetRead>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(reservations, metrics);
    }
}
