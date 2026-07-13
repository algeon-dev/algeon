use super::*;

pub(in crate::exec::bounded) fn execute_bounded_chunks_with_source_scheduler(
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
    if bounded_source_adjacent_chunk_batching_enabled(context, bounded, chunks) {
        return execute_bounded_source_adjacent_chunk_batches_serial(context, bounded, chunks, reservations, metrics);
    }
    if !context.source_scheduler_policy().enables_prefetch() || chunks.len() <= 1 {
        let mut partials = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            context.check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
            let chunk_partials = match execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics)
            {
                Ok(partials) => partials,
                Err(error) => {
                    release_committed_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
            };
            partials.extend(chunk_partials);
            if let Err(error) = reservations.acknowledge_requested_grant_shrink() {
                release_committed_partials_reverse(reservations, partials, metrics);
                return Err(error);
            }
        }
        return Ok(partials);
    }

    let mut partials = Vec::with_capacity(chunks.len());
    context.check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
    let mut queue = match OrderedSourceReadQueue::start(
        context,
        bounded.source,
        chunks,
        reservations,
        metrics,
        max_attempts,
        context.source_scheduler_policy().max_source_reads_in_flight(),
    ) {
        Ok(queue) => queue,
        Err(error) if is_retryable_chunk_error(&error) => {
            if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                context,
                bounded,
                chunks,
                0,
                error,
                reservations,
                metrics,
                &mut partials,
                max_attempts,
                min_retry_chunk_row_groups,
            ) {
                release_committed_partials_reverse(reservations, partials, metrics);
                return Err(error);
            }
            return Ok(partials);
        },
        Err(error) => {
            release_committed_partials_reverse(reservations, partials, metrics);
            return Err(error);
        },
    };

    for (index, chunk) in chunks.iter().enumerate() {
        if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler") {
            queue.abandon_all(index, reservations, metrics);
            return Err(error);
        }
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
            retain_decoded_source_for_cache(context, bounded),
        ) {
            Ok(source) => source,
            Err(error) if is_prefetch_chunk_error(&error) && is_retryable_chunk_error(&error) => {
                metrics.record_source_read_prefetch_abandoned();
                queue.abandon_all(index + 1, reservations, metrics);
                let chunk_partials =
                    match execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics) {
                        Ok(partials) => partials,
                        Err(error) => {
                            release_committed_partials_reverse(reservations, partials, metrics);
                            return Err(error);
                        },
                    };
                partials.extend(chunk_partials);
                acknowledge_chunk_boundary_or_release(&mut queue, index + 1, reservations, &mut partials, metrics)?;
                if let Err(error) =
                    queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)
                {
                    queue.abandon_all(index + 1, reservations, metrics);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            context,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            reservations,
                            metrics,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(reservations, partials, metrics);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                }
                continue;
            },
            Err(error) if is_retryable_chunk_error(&error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                let chunk_partials = match retry_after_chunk_error(
                    context,
                    bounded,
                    chunk.clone(),
                    error,
                    reservations,
                    metrics,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(partials) => partials,
                    Err(error) => {
                        release_committed_partials_reverse(reservations, partials, metrics);
                        return Err(error);
                    },
                };
                partials.extend(chunk_partials);
                acknowledge_chunk_boundary_or_release(&mut queue, index + 1, reservations, &mut partials, metrics)?;
                if let Err(error) =
                    queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)
                {
                    queue.abandon_all(index + 1, reservations, metrics);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            context,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            reservations,
                            metrics,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(reservations, partials, metrics);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                }
                continue;
            },
            Err(error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                release_committed_partials_reverse(reservations, partials, metrics);
                return Err(error);
            },
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
                let chunk_partials = match retry_after_chunk_error(
                    context,
                    bounded,
                    chunk.clone(),
                    error,
                    reservations,
                    metrics,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(partials) => partials,
                    Err(error) => {
                        release_committed_partials_reverse(reservations, partials, metrics);
                        return Err(error);
                    },
                };
                partials.extend(chunk_partials);
                acknowledge_chunk_boundary_or_release(&mut queue, index + 1, reservations, &mut partials, metrics)?;
                if let Err(error) =
                    queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)
                {
                    queue.abandon_all(index + 1, reservations, metrics);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            context,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            reservations,
                            metrics,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(reservations, partials, metrics);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                }
                continue;
            },
            Err(error) => {
                queue.abandon_all(index + 1, reservations, metrics);
                release_committed_partials_reverse(reservations, partials, metrics);
                return Err(error);
            },
        }
        acknowledge_chunk_boundary_or_release(&mut queue, index + 1, reservations, &mut partials, metrics)?;
        if let Err(error) =
            queue.ensure_current_required(context, bounded.source, chunks, reservations, metrics, max_attempts)
        {
            queue.abandon_all(index + 1, reservations, metrics);
            if is_retryable_chunk_error(&error) {
                if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                    context,
                    bounded,
                    chunks,
                    index + 1,
                    error,
                    reservations,
                    metrics,
                    &mut partials,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    release_committed_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                }
                return Ok(partials);
            }
            release_committed_partials_reverse(reservations, partials, metrics);
            return Err(error);
        }
    }

    Ok(partials)
}

fn acknowledge_chunk_boundary_or_release(
    queue: &mut OrderedSourceReadQueue,
    next_chunk_index: usize,
    reservations: &mut ReservationTracker,
    partials: &mut Vec<CommittedPartial>,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if let Err(error) = reservations.acknowledge_requested_grant_shrink() {
        queue.abandon_all(next_chunk_index, reservations, metrics);
        release_committed_partials_reverse(reservations, std::mem::take(partials), metrics);
        return Err(error);
    }
    Ok(())
}

fn execute_remaining_chunks_serial_after_scheduler_error(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    next_chunk_index: usize,
    error: QueryError,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    partials: &mut Vec<CommittedPartial>,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<()> {
    let Some(first_chunk) = chunks.get(next_chunk_index) else {
        return Err(error);
    };
    partials.extend(retry_after_chunk_error(
        context,
        bounded,
        first_chunk.clone(),
        error,
        reservations,
        metrics,
        1,
        max_attempts,
        min_retry_chunk_row_groups,
    )?);
    reservations.acknowledge_requested_grant_shrink()?;

    for chunk in &chunks[next_chunk_index + 1..] {
        context.check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
        partials.extend(execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics)?);
        reservations.acknowledge_requested_grant_shrink()?;
    }
    Ok(())
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
    context.check_cancellation("nexus_query_engine.exec.bounded.source_scheduler_start")?;
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
    ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        budget_boundary,
        prefetch,
        attempt,
        max_attempts,
    )
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
