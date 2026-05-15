use std::collections::VecDeque;

use super::{
    super::source::{
        ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput, local_parquet_execution_frame_schema,
        read_local_parquet_chunk,
    },
    *,
};
use crate::exec::types::{GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey};

pub(super) fn execute_bounded_chunk_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        unreachable!("execute_bounded_chunk_with_retry is called only for bounded policies");
    };

    execute_bounded_chunk_retry_attempt(
        context,
        bounded,
        chunk.clone(),
        reservations,
        metrics,
        1,
        max_retry_attempts.max(1),
        min_retry_chunk_row_groups.max(1),
    )
}

pub(super) fn execute_bounded_chunks_with_source_scheduler(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        unreachable!("execute_bounded_chunks_with_source_scheduler is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let min_retry_chunk_row_groups = min_retry_chunk_row_groups.max(1);
    if !context.source_scheduler_policy().enables_prefetch() || chunks.len() <= 1 {
        let mut partials = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            partials.extend(execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics)?);
        }
        return Ok(partials);
    }

    let mut partials = Vec::with_capacity(chunks.len());
    let mut queue = OrderedSourceReadQueue::start(
        context,
        bounded.source,
        chunks,
        reservations,
        metrics,
        max_attempts,
        context.source_scheduler_policy().max_source_reads_in_flight(),
    )?;

    for (index, chunk) in chunks.iter().enumerate() {
        let scheduled =
            queue.pop_front().expect("source scheduler should always have a current read before processing a chunk");
        let source = match finish_scheduled_source_read(
            scheduled,
            context,
            bounded.source,
            reservations,
            metrics,
            1,
            max_attempts,
            !bounded.has_source_adjacent_stages(),
        ) {
            Ok(source) => source,
            Err(error) if is_prefetch_chunk_error(&error) && is_retryable_chunk_error(&error) => {
                metrics.record_source_read_prefetch_abandoned();
                queue.abandon_all(index + 1, reservations, metrics);
                partials.extend(execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics)?);
                queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)?;
                continue;
            },
            Err(error) if is_retryable_chunk_error(&error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                partials.extend(retry_after_chunk_error(
                    context,
                    bounded,
                    chunk.clone(),
                    error,
                    reservations,
                    metrics,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?);
                queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)?;
                continue;
            },
            Err(error) => return Err(error),
        };

        let prefetch_during_operator = !bounded.has_source_adjacent_stages();
        let result = execute_bounded_chunk_operator_from_source(
            context,
            bounded,
            source,
            reservations,
            metrics,
            1,
            max_attempts,
            |reservations, metrics| {
                if prefetch_during_operator {
                    queue.fill_optional(context, bounded.source, chunks, reservations, metrics, max_attempts)
                } else {
                    Ok(())
                }
            },
        );

        match result {
            Ok(partial) => partials.push(partial),
            Err(error) if is_retryable_chunk_error(&error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                partials.extend(retry_after_chunk_error(
                    context,
                    bounded,
                    chunk.clone(),
                    error,
                    reservations,
                    metrics,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?);
                queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)?;
            },
            Err(error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                return Err(error);
            },
        }
        queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)?;
    }

    Ok(partials)
}

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_required_only_with_observation(
        context,
        source,
        chunks,
        reservations,
        metrics,
        memory_boundary,
        true,
    )
}

pub(in crate::exec::bounded) fn read_local_parquet_chunks_required_only_with_observation(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    read_local_parquet_chunks_serial(
        context,
        source,
        chunks,
        reservations,
        metrics,
        memory_boundary,
        grow_source_to_observed,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn concatenate_source_read_outputs(
    context: &ExecutionContext,
    node_id: NodeId,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    concat_output_boundary: &'static str,
    operation: &'static str,
    message: &'static str,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let ScheduledLocalParquetReadOutput { frame, reservation, .. } =
            chunks.pop().expect("single source chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, chunk| total.saturating_add(chunk.reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, output_reservation_bytes, reservations.budget())
            .with_memory_boundary(concat_output_boundary)
            .with_node_id(node_id)
            .with_stage(concat_output_boundary)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_source_read_outputs_reverse(reservations, chunks, metrics);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        frames.push(chunk.frame.into_frame());
        input_reservations.push(chunk.reservation);
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
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(operation, message, error)
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
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, reservations.budget())
                .with_memory_boundary(concat_output_boundary)
                .with_node_id(node_id)
                .with_stage(concat_output_boundary)
                .with_attempt(1, 1),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_source_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_source_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn read_local_parquet_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut outputs = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        metrics.record_source_chunk_attempt();
        if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
            metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
            outputs.push(output);
            continue;
        }
        let source_request = source_read_reservation_request(
            context.memory_control_policy(),
            ReservationKind::SourceRead,
            memory_boundary,
            reservations.budget(),
            chunk,
            1,
            1,
        );
        let reservation = match reservations.try_reserve(source_request, metrics) {
            Ok(reservation) => reservation,
            Err(error) => {
                release_source_read_outputs_reverse(reservations, outputs, metrics);
                return Err(error);
            },
        };
        let source_token_budget = reservations.available_bytes_including(&reservation);
        let frame =
            match read_local_parquet_chunk(context, source, chunk, source_token_budget, memory_boundary, metrics) {
                Ok(frame) => frame,
                Err(error) => {
                    reservations.release(reservation, metrics);
                    release_source_read_outputs_reverse(reservations, outputs, metrics);
                    return Err(error);
                },
            };
        let mut output =
            ScheduledLocalParquetReadOutput { chunk: chunk.clone(), frame, reservation, observed_bytes: 0 };
        output.observed_bytes = match if grow_source_to_observed {
            grow_source_reservation_to_observed(
                &output.frame,
                &mut output.reservation,
                reservations,
                &output.chunk,
                metrics,
                memory_boundary,
                1,
                1,
            )
        } else {
            shrink_source_reservation_to_observed_without_grow(
                &output.frame,
                &mut output.reservation,
                reservations,
                &output.chunk,
                metrics,
                memory_boundary,
                1,
                1,
            )
        } {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                release_source_read_outputs_reverse(reservations, outputs, metrics);
                return Err(error);
            },
        };
        if grow_source_to_observed {
            insert_bounded_local_parquet_source_cache(
                context,
                source,
                &output.chunk,
                &output.frame,
                &mut output.reservation,
                reservations,
                metrics,
                memory_boundary,
                1,
                1,
                output.observed_bytes,
            )?;
        }
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        outputs.push(output);
    }
    Ok(outputs)
}

pub(in crate::exec::bounded) fn release_source_read_outputs_reverse(
    reservations: &mut ReservationTracker,
    mut chunks: Vec<ScheduledLocalParquetReadOutput>,
    metrics: &mut QueryMetrics,
) {
    while let Some(chunk) = chunks.pop() {
        drop(chunk.frame);
        reservations.release(chunk.reservation, metrics);
    }
}

fn release_source_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut active: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation, metrics);
    }
}

struct OrderedSourceReadQueue {
    pending: VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: usize,
    max_in_flight: usize,
}

impl OrderedSourceReadQueue {
    fn start(
        context: &ExecutionContext,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
        max_attempts: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        let mut queue = Self { pending: VecDeque::new(), next_chunk_index: 0, max_in_flight: max_in_flight.max(1) };
        queue.ensure_current_required(context, source, chunks, reservations, metrics, max_attempts)?;
        Ok(queue)
    }

    fn pop_front(&mut self) -> Option<ScheduledLocalParquetRead> {
        self.pending.pop_front()
    }

    fn ensure_current_required(
        &mut self,
        context: &ExecutionContext,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
        max_attempts: usize,
    ) -> Result<()> {
        if self.pending.is_empty()
            && let Some(chunk) = chunks.get(self.next_chunk_index)
        {
            self.pending.push_back(start_required_scheduled_source_read(
                context,
                source,
                chunk,
                reservations,
                metrics,
                1,
                max_attempts,
            )?);
            self.next_chunk_index += 1;
        }
        Ok(())
    }

    fn fill_optional(
        &mut self,
        context: &ExecutionContext,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
        max_attempts: usize,
    ) -> Result<()> {
        while self.pending.len() < optional_prefetch_target(self.max_in_flight) {
            let Some(chunk) = chunks.get(self.next_chunk_index) else {
                break;
            };
            let Some(prefetch) = try_start_optional_scheduled_source_read(
                context,
                source,
                Some(chunk),
                reservations,
                metrics,
                1,
                max_attempts,
            )?
            else {
                break;
            };
            self.pending.push_back(prefetch);
            self.next_chunk_index += 1;
        }
        Ok(())
    }

    fn abandon_all(
        &mut self,
        next_chunk_index: usize,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
        while let Some(prefetch) = self.pending.pop_back() {
            prefetch.abandon(reservations, metrics);
        }
        self.next_chunk_index = next_chunk_index;
    }
}

fn optional_prefetch_target(max_in_flight: usize) -> usize {
    // libcudf does not expose exact decoded Parquet output preflight sizing.
    // Keep only one unobserved optional source read ahead so that a later
    // front-of-queue read can grow to its observed output without being
    // starved by deeper speculative reservations.
    usize::from(max_in_flight > 1)
}

#[allow(clippy::too_many_arguments)]
fn start_required_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetRead> {
    let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
    start_scheduled_source_read(
        context,
        source,
        chunk,
        reservations,
        metrics,
        requested_bytes,
        "source_chunk_read",
        attempt,
        max_attempts,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_start_optional_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: Option<&SourceChunk>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<ScheduledLocalParquetRead>> {
    let Some(chunk) = chunk else {
        return Ok(None);
    };
    let requested_bytes = prefetch_requested_bytes(context.memory_control_policy(), chunk);
    if !reservations.can_reserve(requested_bytes) {
        metrics.record_source_read_prefetch_skipped();
        return Ok(None);
    }
    metrics.record_source_read_prefetch_started();
    start_scheduled_source_read(
        context,
        source,
        chunk,
        reservations,
        metrics,
        requested_bytes,
        "source_chunk_prefetch",
        attempt,
        max_attempts,
        true,
        true,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn start_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    requested_bytes: u64,
    budget_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    strict_token_budget: bool,
    prefetch: bool,
) -> Result<ScheduledLocalParquetRead> {
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(ScheduledLocalParquetRead::ready(output, budget_boundary, prefetch));
    }
    let contract = source_read_reservation_contract(context.memory_control_policy(), chunk);
    let request = chunk_reservation_request(
        ReservationKind::SourceRead,
        budget_boundary,
        requested_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes);
    let reservation = reservations.try_reserve(request, metrics)?;
    let source_token_budget =
        if strict_token_budget { reservation.bytes() } else { reservations.available_bytes_including(&reservation) };
    Ok(ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        budget_boundary,
        prefetch,
    ))
}

fn finish_scheduled_source_read(
    scheduled: ScheduledLocalParquetRead,
    context: &ExecutionContext,
    source: &LocalParquetSource,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    let is_prefetch = scheduled.is_prefetch();
    let memory_boundary = scheduled.budget_boundary();
    let mut output = scheduled.finish(reservations, metrics)?;
    if grow_source_to_observed {
        output.observed_bytes = grow_source_reservation_to_observed(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
        )?;
        insert_bounded_local_parquet_source_cache(
            context,
            source,
            &output.chunk,
            &output.frame,
            &mut output.reservation,
            reservations,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
            output.observed_bytes,
        )?;
    } else {
        output.observed_bytes = match shrink_source_reservation_to_observed_without_grow(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            memory_boundary,
            attempt,
            max_attempts,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                return Err(error);
            },
        };
    }
    if is_prefetch {
        metrics.record_source_read_prefetch_completed();
    }
    Ok(output)
}

pub(in crate::exec::bounded) fn try_read_cached_local_parquet_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<ScheduledLocalParquetReadOutput>> {
    let Some(cache) = context.local_parquet_source_cache() else {
        return Ok(None);
    };
    let key = GpuLocalParquetSourceCacheKey::from_source_chunk(source, chunk);
    let schema = local_parquet_execution_frame_schema(source)?;
    let lookup_timer = Instant::now();
    let cached = cache.get(&key, context, &schema)?;
    metrics.record_source_cache_lookup_time(lookup_timer.elapsed());
    if let Some((handle, retained_device_bytes)) = cached {
        metrics.record_source_cache_hit(cache.retained_device_bytes()?);
        return Ok(Some(ScheduledLocalParquetReadOutput {
            chunk: chunk.clone(),
            frame: handle.into_gpu_frame(),
            reservation: reservations.borrow_accounted_source_cache(key, retained_device_bytes, metrics),
            observed_bytes: retained_device_bytes,
        }));
    }
    metrics.record_source_cache_miss(cache.retained_device_bytes()?);
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn insert_bounded_local_parquet_source_cache(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    retained_device_bytes: u64,
) -> Result<()> {
    let Some(cache) = context.local_parquet_source_cache() else {
        return Ok(());
    };
    let key = GpuLocalParquetSourceCacheKey::from_source_chunk(source, chunk);
    let schema = local_parquet_execution_frame_schema(source)?;
    let handle = GpuResultHandle::new(context, frame.clone(), schema)?;
    let protected_keys = reservations.source_cache_protected_keys_with(&key);
    let insert = cache.insert_if_absent_protecting(key.clone(), handle, retained_device_bytes, &protected_keys)?;
    reservations.release_source_cache_baseline_bytes(insert.evicted_device_bytes, metrics);
    match insert.outcome {
        GpuLocalParquetSourceCacheInsert::Inserted => {
            metrics.record_source_cache_insert(insert.retained_device_bytes);
            let request = chunk_reservation_request(
                ReservationKind::SourceCacheRetained,
                "source_cache_retained",
                retained_device_bytes,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            )
            .with_stage(memory_boundary);
            reservations.promote_source_cache_insert(
                source_reservation,
                retained_device_bytes,
                request,
                metrics,
                &key,
            )?;
        },
        GpuLocalParquetSourceCacheInsert::AlreadyPresent | GpuLocalParquetSourceCacheInsert::CapacitySkipped => {
            metrics.record_source_cache_insert_skipped(insert.retained_device_bytes);
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bounded_chunk_retry_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    match execute_bounded_chunk_attempt(context, bounded, &chunk, reservations, metrics, attempt, max_attempts) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => retry_after_chunk_error(
            context,
            bounded,
            chunk,
            error,
            reservations,
            metrics,
            attempt,
            max_attempts,
            min_retry_chunk_row_groups,
        ),
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
fn retry_after_chunk_error(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    error: QueryError,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    record_chunk_retry_error(&error, metrics);
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        let mut partials = execute_bounded_chunk_retry_attempt(
            context,
            bounded,
            left,
            reservations,
            metrics,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?;
        partials.extend(execute_bounded_chunk_retry_attempt(
            context,
            bounded,
            right,
            reservations,
            metrics,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?);
        return Ok(partials);
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}

pub(super) fn execute_bounded_chunk_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let source = read_bounded_chunk_source(
        context,
        bounded.source,
        chunk,
        reservations,
        metrics,
        attempt,
        max_attempts,
        !bounded.has_source_adjacent_stages(),
    )
    .map_err(|error| attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics))?;
    execute_bounded_chunk_operator_from_source(
        context,
        bounded,
        source,
        reservations,
        metrics,
        attempt,
        max_attempts,
        |_reservations, _metrics| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
fn read_bounded_chunk_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(output);
    }

    let source_request = source_read_reservation_request(
        context.memory_control_policy(),
        ReservationKind::SourceRead,
        "source_chunk_read",
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let source_reservation = reservations.try_reserve(source_request, metrics)?;
    let source_token_budget = reservations.available_bytes_including(&source_reservation);
    let frame =
        match read_local_parquet_chunk(context, source, chunk, source_token_budget, "source_chunk_read", metrics) {
            Ok(frame) => frame,
            Err(error) => {
                reservations.release(source_reservation, metrics);
                return Err(error);
            },
        };
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        reservation: source_reservation,
        observed_bytes: 0,
    };
    if grow_source_to_observed {
        output.observed_bytes = grow_source_reservation_to_observed(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
        )?;
        insert_bounded_local_parquet_source_cache(
            context,
            source,
            &output.chunk,
            &output.frame,
            &mut output.reservation,
            reservations,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
            output.observed_bytes,
        )?;
    } else {
        output.observed_bytes = match shrink_source_reservation_to_observed_without_grow(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                return Err(error);
            },
        };
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn execute_bounded_chunk_operator_from_source(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    source: ScheduledLocalParquetReadOutput,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    before_operator: impl FnOnce(&mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
) -> Result<CommittedPartial> {
    let ScheduledLocalParquetReadOutput { chunk, frame, reservation: source_reservation, observed_bytes: _ } = source;
    let mut input = CommittedPartial { frame, reservation: source_reservation };
    input = execute_source_adjacent_chunk_stages(
        context,
        bounded,
        &chunk,
        input,
        reservations,
        metrics,
        attempt,
        max_attempts,
    )?;
    let CommittedPartial { frame, reservation: source_reservation } = input;

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_reservation.bytes(),
        reservations.budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(frame);
            reservations.release(source_reservation, metrics);
            return Err(error);
        },
    };
    let operator_token_budget = reservations.available_bytes_including(&operator_reservation);
    let operator_token = match execution_budget_token_for_boundary(
        context,
        operator_token_budget,
        "chunk_operator",
        "cudf_query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, &chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };
    if let Err(error) = before_operator(reservations, metrics) {
        reservations.release(operator_reservation, metrics);
        reservations.release(source_reservation, metrics);
        return Err(error);
    }

    let chunk_operator_timer = Instant::now();
    let chunk_output = execute_partial_aggregate_with_budget(
        context,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        frame,
        &operator_token,
        metrics,
    );
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, &chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    reservations.release(operator_reservation, metrics);
    reservations.release(source_reservation, metrics);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, &chunk, "partial_state", attempt, max_attempts, metrics))?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    metrics.record_partial_aggregate(frame.num_rows() as u64);

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

#[allow(clippy::too_many_arguments)]
fn execute_source_adjacent_chunk_stages(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    mut input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    for (projection, output_schema) in &bounded.pre_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        )?;
    }
    if let Some(filter) = bounded.filter {
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_filter",
            "source_chunk_filter_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_filter_with_budget(context, filter, input, token, metrics),
        )?;
    }
    for (projection, output_schema) in &bounded.post_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        )?;
    }
    Ok(input)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &mut QueryMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let input_reserved_bytes = input.reservation.bytes();
    let mut work_reservation = match reservations.try_reserve_work(
        chunk_reservation_request(
            ReservationKind::ChunkOperator,
            stage,
            input_reserved_bytes,
            reservations.budget(),
            chunk,
            attempt,
            max_attempts,
        )
        .with_node_id(node_id)
        .with_stage(stage),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input.frame);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "cudf_query_engine.source_chunk_unary.budget_token",
        "failed to create bounded native source-adjacent chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };
    let result = execute(context, input.frame, &token, metrics);
    let frame = match result {
        Ok(frame) => {
            reservations.release(input.reservation, metrics);
            frame
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, output_boundary, attempt, max_attempts, metrics));
        },
    };
    let output_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        output_boundary,
        output_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    )
    .with_node_id(node_id)
    .with_stage(stage);
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut work_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut work_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
fn grow_source_reservation_to_observed(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = match observe_device_memory_bytes(frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, memory_boundary, attempt, max_attempts, metrics));
        },
    };
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_source_read_reservation_request(
                memory_boundary,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            ),
            metrics,
        );
        return Ok(source_observed_bytes);
    }

    let source_delta = source_observed_bytes.saturating_sub(source_reservation.bytes());
    let grow_request =
        observed_source_read_reservation_request(memory_boundary, reservations.budget(), chunk, attempt, max_attempts)
            .with_requested_bytes(source_delta);
    reservations.try_grow(source_reservation, grow_request, metrics)?;
    Ok(source_observed_bytes)
}

#[allow(clippy::too_many_arguments)]
fn shrink_source_reservation_to_observed_without_grow(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = observe_device_memory_bytes(frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, memory_boundary, attempt, max_attempts, metrics))?;
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_source_read_reservation_request(
                memory_boundary,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            ),
            metrics,
        );
    }
    Ok(source_observed_bytes)
}

pub(super) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    source_read_reservation_contract(policy, chunk).requested_bytes
}

pub(super) fn prefetch_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let contract = source_read_reservation_contract(policy, chunk);
    match (policy, contract.quality) {
        (MemoryControlPolicy::Unbounded, _) => contract.requested_bytes,
        (
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. },
            SourceReadReservationContractQuality::ConservativeEstimate,
        ) => contract.requested_bytes.max(max_source_chunk_bytes),
        (MemoryControlPolicy::Bounded { .. }, _) => contract.requested_bytes,
    }
}

/// Precision of the byte contract used when reserving memory for one bounded
/// source-read chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SourceReadReservationContractQuality {
    /// A lower layer supplied an exact preflight byte contract, so bounded
    /// source reads can reserve the required bytes before decoding starts.
    #[allow(dead_code)]
    ExactLowerLevelContract,
    /// The read is admitted with an initial reservation and then reconciled to
    /// observed device bytes by growing or shrinking after decoding.
    ObservedGrowShrink,
    /// The reservation is based on file statistics or configured chunk caps;
    /// it may be retried with smaller row-group chunks if execution exceeds it.
    ConservativeEstimate,
}

impl SourceReadReservationContractQuality {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::ExactLowerLevelContract => "exact_lower_level_contract",
            Self::ObservedGrowShrink => "observed_grow_shrink",
            Self::ConservativeEstimate => "conservative_estimate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceReadReservationContract {
    pub(super) requested_bytes: u64,
    pub(super) planning_bytes: Option<u64>,
    pub(super) estimated_bytes: Option<u64>,
    pub(super) quality: SourceReadReservationContractQuality,
}

pub(super) fn source_read_reservation_contract(
    policy: MemoryControlPolicy,
    chunk: &SourceChunk,
) -> SourceReadReservationContract {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    let planning_bytes = chunk
        .estimated_chunk_bytes
        .or(chunk.known_projected_uncompressed_bytes)
        .or(chunk.known_row_group_bytes)
        .or(chunk.known_file_bytes);
    let estimated_bytes = chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes);
    let conservative_bytes = estimated_bytes.unwrap_or(fallback);
    let requested_bytes = match policy {
        MemoryControlPolicy::Unbounded => conservative_bytes,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => conservative_bytes.min(max_source_chunk_bytes),
    };
    SourceReadReservationContract {
        requested_bytes,
        planning_bytes,
        estimated_bytes,
        quality: SourceReadReservationContractQuality::ConservativeEstimate,
    }
}

pub(super) fn chunk_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(chunk.source_node_id)
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

pub(super) fn source_read_reservation_request(
    policy: MemoryControlPolicy,
    kind: ReservationKind,
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let contract = source_read_reservation_contract(policy, chunk);
    chunk_reservation_request(
        kind,
        memory_boundary,
        contract.requested_bytes,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

pub(super) fn observed_source_read_reservation_request(
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    chunk_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}

pub(super) fn is_retryable_chunk_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn is_prefetch_chunk_error(error: &QueryError) -> bool {
    error.fact_value("budget_boundary") == Some("source_chunk_prefetch")
        || error.fact_value("memory_boundary") == Some("source_chunk_prefetch")
}

pub(super) fn record_chunk_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn split_retryable_chunk(
    chunk: &SourceChunk,
    min_retry_chunk_row_groups: usize,
) -> Option<(SourceChunk, SourceChunk)> {
    if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= min_retry_chunk_row_groups {
        return None;
    }

    let split_at = chunk.row_group_indexes.len() / 2;
    if split_at == 0 || split_at == chunk.row_group_indexes.len() {
        return None;
    }

    let left_row_groups = chunk.row_group_indexes[..split_at].to_vec();
    let right_row_groups = chunk.row_group_indexes[split_at..].to_vec();
    let mut left = retry_child_chunk(chunk, left_row_groups);
    let mut right = retry_child_chunk(chunk, right_row_groups);
    apply_parent_scan_limit_to_retry_children(chunk, &mut left, &mut right);
    Some((left, right))
}

fn apply_parent_scan_limit_to_retry_children(parent: &SourceChunk, left: &mut SourceChunk, right: &mut SourceChunk) {
    let Some(parent_limit) = parent.scan_limit_rows else {
        return;
    };
    let Ok(mut remaining) = u64::try_from(parent_limit) else {
        return;
    };
    apply_retry_child_scan_limit(left, &mut remaining);
    apply_retry_child_scan_limit(right, &mut remaining);
}

fn apply_retry_child_scan_limit(child: &mut SourceChunk, remaining: &mut u64) {
    let Some(row_count) = child.known_row_count else {
        child.scan_limit_rows = usize::try_from(*remaining).ok();
        return;
    };
    if *remaining == 0 {
        child.scan_limit_rows = Some(0);
        child.known_row_count = Some(0);
        return;
    }
    if row_count <= *remaining {
        *remaining = remaining.saturating_sub(row_count);
        return;
    }
    if let Ok(keep_rows) = usize::try_from(*remaining) {
        child.scan_limit_rows = Some(keep_rows);
        child.known_row_count = Some(*remaining);
    }
    *remaining = 0;
}

pub(super) fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
    let parent_row_group_count = parent.row_group_indexes.len();
    let known_row_group_bytes =
        checked_row_group_sum(parent.source_file_row_group_bytes.as_deref(), &row_group_indexes).or_else(|| {
            proportional_bytes(parent.known_row_group_bytes, row_group_indexes.len(), parent_row_group_count)
        });
    let known_projected_compressed_bytes = checked_row_group_sum(
        parent.source_file_projected_compressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_projected_compressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_projected_uncompressed_bytes = checked_row_group_sum(
        parent.source_file_projected_uncompressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_projected_uncompressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_pure_filter_compressed_bytes = checked_row_group_sum(
        parent.source_file_pure_filter_compressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_pure_filter_compressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_pure_filter_uncompressed_bytes =
        checked_row_group_sum(parent.source_file_pure_filter_uncompressed_bytes.as_deref(), &row_group_indexes)
            .or_else(|| {
                proportional_bytes(
                    parent.known_pure_filter_uncompressed_bytes,
                    row_group_indexes.len(),
                    parent_row_group_count,
                )
            });
    let known_file_bytes = if parent.known_row_group_bytes.is_some() {
        parent.known_file_bytes
    } else {
        proportional_bytes(parent.known_file_bytes, row_group_indexes.len(), parent_row_group_count)
    };
    let estimated_chunk_bytes = proportional_bytes(
        parent.estimated_chunk_bytes.or(parent.known_row_group_bytes).or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );
    let estimated_source_read_bytes = proportional_bytes(
        parent
            .estimated_source_read_bytes
            .or(parent.known_row_group_bytes)
            .or(parent.estimated_chunk_bytes)
            .or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );
    let known_row_count = checked_row_group_sum(parent.source_file_row_group_rows.as_deref(), &row_group_indexes)
        .or_else(|| proportional_bytes(parent.known_row_count, row_group_indexes.len(), parent_row_group_count));

    SourceChunk {
        source_node_id: parent.source_node_id,
        chunk_index: parent.chunk_index,
        file_index: parent.file_index,
        path: parent.path.clone(),
        stable_id: parent.stable_id.clone(),
        row_group_indexes,
        reads_all_row_groups: false,
        known_file_bytes,
        known_row_group_bytes,
        known_projected_compressed_bytes,
        known_projected_uncompressed_bytes,
        known_pure_filter_compressed_bytes,
        known_pure_filter_uncompressed_bytes,
        estimated_chunk_bytes,
        estimated_source_read_bytes,
        source_file_row_groups: parent.source_file_row_groups.clone(),
        source_file_row_group_bytes: parent.source_file_row_group_bytes.clone(),
        source_file_projected_compressed_bytes: parent.source_file_projected_compressed_bytes.clone(),
        source_file_projected_uncompressed_bytes: parent.source_file_projected_uncompressed_bytes.clone(),
        source_file_pure_filter_compressed_bytes: parent.source_file_pure_filter_compressed_bytes.clone(),
        source_file_pure_filter_uncompressed_bytes: parent.source_file_pure_filter_uncompressed_bytes.clone(),
        source_file_row_group_rows: parent.source_file_row_group_rows.clone(),
        read_columns: parent.read_columns.clone(),
        partition_values: parent.partition_values.clone(),
        zero_column_count: parent.zero_column_count,
        known_row_count,
        scan_limit_rows: None,
        size_estimate_based: true,
    }
}

fn checked_row_group_sum(values: Option<&[u64]>, row_group_indexes: &[usize]) -> Option<u64> {
    row_group_indexes.iter().try_fold(0u64, |total, row_group| total.checked_add(*values?.get(*row_group)?))
}

pub(super) fn proportional_bytes(
    total_bytes: Option<u64>,
    child_row_groups: usize,
    parent_row_groups: usize,
) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

pub(super) fn chunk_retry_facts(chunk: &SourceChunk, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= 1 {
        "fail_min_chunk"
    } else {
        "retry_attempts_exhausted"
    };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("source_chunk")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}
