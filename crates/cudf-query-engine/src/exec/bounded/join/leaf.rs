use std::collections::VecDeque;

use super::*;
use crate::exec::{
    bounded::chunks::{insert_bounded_local_parquet_source_cache, try_read_cached_local_parquet_chunk},
    source::{ScheduledLocalParquetRead, format_row_groups, read_local_parquet_chunk},
};

#[derive(Clone)]
pub(in crate::exec::bounded) struct BoundedJoinLeafLocalParquetSourceStream {
    source: LocalParquetSource,
    node_id: NodeId,
    chunks: Vec<SourceChunk>,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
}

impl BoundedJoinLeafLocalParquetSourceStream {
    pub(in crate::exec::bounded) fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

pub(in crate::exec::bounded) fn execute_bounded_join_leaf_source(
    context: &ExecutionContext,
    source: &crate::source::SourceNode,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            execute_bounded_join_leaf_local_parquet_source(context, source, node_id, reservations, metrics)
        },
        crate::source::SourceNode::GpuResident(source) => {
            let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
            let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
            let reservation = reservations.try_reserve(
                MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
                    .with_memory_boundary("gpu_resident_join_input")
                    .with_node_id(node_id)
                    .with_stage("gpu_resident_join_input")
                    .with_attempt(1, 1),
                metrics,
            )?;
            Ok(ResidentGpuFrame { frame, reservation })
        },
    }
}

pub(in crate::exec::bounded) fn plan_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    validate_local_parquet_source(source)?;
    let chunk_plan = plan_local_parquet_source_chunks(
        node_id,
        source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.source",
            "bounded join leaf local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    let (max_attempts, min_retry_chunk_row_groups) = join_source_retry_policy(context);
    Ok(BoundedJoinLeafLocalParquetSourceStream {
        source: source.clone(),
        node_id,
        chunks: chunk_plan.chunks,
        max_attempts,
        min_retry_chunk_row_groups,
    })
}

pub(in crate::exec::bounded) fn materialize_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let chunks = read_join_leaf_source_stream_chunks(context, &stream, reservations, metrics)?;
    concatenate_join_source_chunks(context, stream.node_id, chunks, reservations, metrics)
}

pub(in crate::exec::bounded) fn stream_bounded_join_leaf_local_parquet_source<F>(
    context: &ExecutionContext,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    if context.source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        stream_join_leaf_source_chunks_with_scheduler(context, &stream, reservations, metrics, &mut consume)
    } else {
        stream_join_leaf_source_chunks_serial(context, &stream, reservations, metrics, &mut consume)
    }
}

fn execute_bounded_join_leaf_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let stream = plan_bounded_join_leaf_local_parquet_source_stream(context, source, node_id, metrics)?;
    let chunks = read_join_leaf_source_stream_chunks(context, &stream, reservations, metrics)?;
    concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics)
}

fn read_join_leaf_source_stream_chunks(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let chunks = if context.source_scheduler_policy().enables_prefetch() && stream.chunks.len() > 1 {
        match read_join_leaf_source_chunks_with_scheduler(
            context,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            reservations,
            metrics,
            stream.max_attempts,
        ) {
            Ok(chunks) => chunks,
            Err(error) if is_retryable_chunk_error(&error) => read_join_leaf_source_chunks_serial(
                context,
                &stream.source,
                stream.node_id,
                &stream.chunks,
                reservations,
                metrics,
                stream.max_attempts,
                stream.min_retry_chunk_row_groups,
            )?,
            Err(error) => return Err(error),
        }
    } else {
        read_join_leaf_source_chunks_serial(
            context,
            &stream.source,
            stream.node_id,
            &stream.chunks,
            reservations,
            metrics,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
        )?
    };
    Ok(chunks)
}

fn join_source_retry_policy(context: &ExecutionContext) -> (usize, usize) {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } => {
            (max_retry_attempts.max(1), min_retry_chunk_row_groups.max(1))
        },
        MemoryControlPolicy::Unbounded => (1, 1),
    }
}

#[allow(clippy::too_many_arguments)]
fn read_join_leaf_source_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut chunks = Vec::with_capacity(source_chunks.len());
    for chunk in source_chunks {
        match read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk.clone(),
            1,
            max_attempts,
            min_retry_chunk_row_groups,
        ) {
            Ok(mut outputs) => chunks.append(&mut outputs),
            Err(error) => {
                release_join_source_chunks_reverse(reservations, chunks, metrics);
                return Err(error);
            },
        }
    }

    Ok(chunks)
}

#[allow(clippy::too_many_arguments)]
fn read_join_leaf_source_chunk_with_retry(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    match read_join_leaf_source_chunk_attempt(
        context,
        source,
        reservations,
        metrics,
        node_id,
        &chunk,
        attempt,
        max_attempts,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if is_retryable_chunk_error(&error) => retry_join_leaf_source_chunk_after_error(
            context,
            source,
            reservations,
            metrics,
            node_id,
            chunk,
            error,
            attempt,
            max_attempts,
            min_retry_chunk_row_groups,
        ),
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn retry_join_leaf_source_chunk_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: SourceChunk,
    error: QueryError,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    record_chunk_retry_error(&error, metrics);
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        let mut outputs = read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            left,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?;
        outputs.extend(read_join_leaf_source_chunk_with_retry(
            context,
            source,
            reservations,
            metrics,
            node_id,
            right,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?);
        return Ok(outputs);
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}

#[allow(clippy::too_many_arguments)]
fn read_join_leaf_source_chunk_attempt(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> Result<(GpuDataFrame, ActiveReservation)> {
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        return Ok((output.frame, output.reservation));
    }
    let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
    let mut reservation = reservations.try_reserve(
        join_source_read_reservation_request(
            requested_bytes,
            reservations.budget(),
            node_id,
            chunk,
            attempt,
            max_attempts,
        ),
        metrics,
    )?;
    let source_token_budget = reservations.available_bytes_including(&reservation);
    let frame = match read_local_parquet_chunk(context, source, chunk, source_token_budget, "join_source_read", metrics)
    {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(attach_join_source_read_resource_facts(error, node_id, chunk, attempt, max_attempts));
        },
    };
    if let Err(error) = grow_join_source_reservation_to_observed_frame(
        &frame,
        &mut reservation,
        reservations,
        node_id,
        chunk,
        metrics,
        attempt,
        max_attempts,
    ) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    let retained_device_bytes = reservation.bytes();
    if let Err(error) = insert_bounded_local_parquet_source_cache(
        context,
        source,
        chunk,
        &frame,
        &mut reservation,
        reservations,
        metrics,
        "join_source_read",
        attempt,
        max_attempts,
        retained_device_bytes,
    ) {
        drop(frame);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    Ok((frame, reservation))
}

fn read_join_leaf_source_chunks_with_scheduler(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    max_attempts: usize,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
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
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        chunks.push((output.frame, output.reservation));

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

fn stream_join_leaf_source_chunks_serial<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    for chunk in &stream.chunks {
        let outputs = read_join_leaf_source_chunk_with_retry(
            context,
            &stream.source,
            reservations,
            metrics,
            stream.node_id,
            chunk.clone(),
            1,
            stream.max_attempts,
            stream.min_retry_chunk_row_groups,
        )?;
        for (frame, reservation) in outputs {
            consume(ResidentGpuFrame { frame, reservation }, reservations, metrics)?;
        }
    }
    Ok(())
}

fn stream_join_leaf_source_chunks_with_scheduler<F>(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    consume: &mut F,
) -> Result<()>
where
    F: FnMut(ResidentGpuFrame, &mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
{
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
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
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
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
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        consume(ResidentGpuFrame { frame: output.frame, reservation: output.reservation }, reservations, metrics)?;

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

#[allow(clippy::too_many_arguments)]
fn start_next_required_join_leaf_read(
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

#[allow(clippy::too_many_arguments)]
fn fill_optional_join_leaf_reads(
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
        let requested_bytes = prefetch_requested_bytes(context.memory_control_policy(), chunk);
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
            true,
            true,
            1,
            max_attempts,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn start_join_leaf_scheduled_source_read(
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
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(ScheduledLocalParquetRead::ready(output, memory_boundary, prefetch));
    }
    let request = join_source_read_reservation_request_for_boundary(
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
    Ok(ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        memory_boundary,
        prefetch,
    ))
}

fn abandon_join_leaf_reads(
    mut pending: VecDeque<ScheduledLocalParquetRead>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(reservations, metrics);
    }
}

fn attach_join_source_read_resource_facts(
    error: QueryError,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> QueryError {
    error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", attempt)
        .with_fact_if_absent("max_attempts", max_attempts)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", chunk.path.display())
        .with_fact_if_absent("row_groups", format_row_groups(chunk))
}

fn join_source_read_reservation_request(
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    join_source_read_reservation_request_for_boundary(
        requested_bytes,
        budget_bytes,
        node_id,
        chunk,
        "join_source_read",
        attempt,
        max_attempts,
    )
}

fn join_source_read_reservation_request_for_boundary(
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request
}

#[allow(clippy::too_many_arguments)]
fn grow_join_source_reservation_to_observed_frame(
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    node_id: NodeId,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, metrics)?;
    if observed_bytes <= reservation.bytes() {
        reservations.shrink_to_observed(reservation, observed_bytes, metrics);
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    reservations.try_grow(
        reservation,
        join_source_read_reservation_request(delta, reservations.budget(), node_id, chunk, attempt, max_attempts),
        metrics,
    )
}

fn concatenate_join_source_chunks(
    context: &ExecutionContext,
    node_id: NodeId,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let (frame, reservation) = chunks.pop().expect("single chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, (_frame, reservation)| total.saturating_add(reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, output_reservation_bytes, reservations.budget())
            .with_memory_boundary("join_source_read")
            .with_node_id(node_id)
            .with_stage("join_source_read")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_join_source_chunks_reverse(reservations, chunks, metrics);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for (frame, reservation) in chunks {
        frames.push(frame.into_frame());
        input_reservations.push(reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.source_concat",
                "failed to concatenate bounded join leaf local parquet source chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, reservations.budget())
                .with_memory_boundary("join_source_read")
                .with_node_id(node_id)
                .with_stage("join_source_read")
                .with_attempt(1, 1),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_join_source_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn release_join_source_chunks_reverse(
    reservations: &mut ReservationTracker,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
    metrics: &mut QueryMetrics,
) {
    while let Some((frame, reservation)) = chunks.pop() {
        drop(frame);
        reservations.release(reservation, metrics);
    }
}

fn release_join_source_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut active: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation, metrics);
    }
}
