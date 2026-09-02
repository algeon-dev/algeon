use super::*;

pub(in crate::exec::bounded) fn execute_bounded_chunks_with_source_scheduler(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        execution.context().memory_control_policy()
    else {
        unreachable!("execute_bounded_chunks_with_source_scheduler is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let min_retry_chunk_row_groups = min_retry_chunk_row_groups.max(1);
    if bounded_source_adjacent_chunk_batching_enabled(execution.context(), bounded, chunks) {
        return execute_bounded_source_adjacent_chunk_batches_serial(execution, bounded, chunks);
    }
    if !execution.context().source_scheduler_policy().enables_prefetch() || chunks.len() <= 1 {
        let mut partials = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
            let chunk_partials = match execute_bounded_chunk_with_retry(execution, bounded, chunk) {
                Ok(partials) => partials,
                Err(error) => {
                    release_committed_partials_reverse(execution, partials);
                    return Err(error);
                },
            };
            partials.extend(chunk_partials);
        }
        return Ok(partials);
    }

    let mut partials = Vec::with_capacity(chunks.len());
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
    let mut queue = match OrderedSourceReadQueue::start(
        execution,
        bounded.source,
        chunks,
        max_attempts,
        execution.context().source_scheduler_policy().max_source_reads_in_flight(),
    ) {
        Ok(queue) => queue,
        Err(error) if is_retryable_chunk_error(&error) => {
            if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                execution,
                bounded,
                chunks,
                0,
                error,
                &mut partials,
                max_attempts,
                min_retry_chunk_row_groups,
            ) {
                release_committed_partials_reverse(execution, partials);
                return Err(error);
            }
            return Ok(partials);
        },
        Err(error) => {
            release_committed_partials_reverse(execution, partials);
            return Err(error);
        },
    };

    for (index, chunk) in chunks.iter().enumerate() {
        if let Err(error) = execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler") {
            queue.abandon_all(execution, index);
            return Err(error);
        }
        let scheduled =
            queue.pop_front().expect("source scheduler should always have a current read before processing a chunk");
        let source = match finish_scheduled_source_read(
            execution,
            scheduled,
            bounded.source,
            1,
            max_attempts,
            retain_decoded_source(bounded),
        ) {
            Ok(source) => source,
            Err(error) if is_prefetch_chunk_error(&error) && is_retryable_chunk_error(&error) => {
                execution.metrics_mut().record_source_read_prefetch_abandoned();
                queue.abandon_all(execution, index + 1);
                let chunk_partials = match execute_bounded_chunk_with_retry(execution, bounded, chunk) {
                    Ok(partials) => partials,
                    Err(error) => {
                        release_committed_partials_reverse(execution, partials);
                        return Err(error);
                    },
                };
                partials.extend(chunk_partials);
                if let Err(error) = queue.ensure_current_required(execution, bounded.source, chunks, max_attempts) {
                    queue.abandon_all(execution, index + 1);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            execution,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(execution, partials);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(execution, partials);
                    return Err(error);
                }
                continue;
            },
            Err(error) if is_retryable_chunk_error(&error) => {
                queue.abandon_all(execution, index + 1);
                let chunk_partials = match retry_after_chunk_error(
                    execution,
                    bounded,
                    chunk.clone(),
                    error,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(partials) => partials,
                    Err(error) => {
                        release_committed_partials_reverse(execution, partials);
                        return Err(error);
                    },
                };
                partials.extend(chunk_partials);
                if let Err(error) = queue.ensure_current_required(execution, bounded.source, chunks, max_attempts) {
                    queue.abandon_all(execution, index + 1);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            execution,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(execution, partials);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(execution, partials);
                    return Err(error);
                }
                continue;
            },
            Err(error) => {
                queue.abandon_all(execution, index + 1);
                release_committed_partials_reverse(execution, partials);
                return Err(error);
            },
        };

        let prefetch_during_operator = !bounded.has_source_adjacent_stages();
        let result =
            execute_bounded_chunk_operator_from_source(execution, bounded, source, 1, max_attempts, |execution| {
                if prefetch_during_operator {
                    queue.fill_optional(execution, bounded.source, chunks, max_attempts)
                } else {
                    Ok(())
                }
            });

        match result {
            Ok(partial) => partials.push(partial),
            Err(error) if is_retryable_chunk_error(&error) => {
                queue.abandon_all(execution, index + 1);
                let chunk_partials = match retry_after_chunk_error(
                    execution,
                    bounded,
                    chunk.clone(),
                    error,
                    1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    Ok(partials) => partials,
                    Err(error) => {
                        release_committed_partials_reverse(execution, partials);
                        return Err(error);
                    },
                };
                partials.extend(chunk_partials);
                if let Err(error) = queue.ensure_current_required(execution, bounded.source, chunks, max_attempts) {
                    queue.abandon_all(execution, index + 1);
                    if is_retryable_chunk_error(&error) {
                        if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                            execution,
                            bounded,
                            chunks,
                            index + 1,
                            error,
                            &mut partials,
                            max_attempts,
                            min_retry_chunk_row_groups,
                        ) {
                            release_committed_partials_reverse(execution, partials);
                            return Err(error);
                        }
                        return Ok(partials);
                    }
                    release_committed_partials_reverse(execution, partials);
                    return Err(error);
                }
                continue;
            },
            Err(error) => {
                queue.abandon_all(execution, index + 1);
                release_committed_partials_reverse(execution, partials);
                return Err(error);
            },
        }
        if let Err(error) = queue.ensure_current_required(execution, bounded.source, chunks, max_attempts) {
            queue.abandon_all(execution, index + 1);
            if is_retryable_chunk_error(&error) {
                if let Err(error) = execute_remaining_chunks_serial_after_scheduler_error(
                    execution,
                    bounded,
                    chunks,
                    index + 1,
                    error,
                    &mut partials,
                    max_attempts,
                    min_retry_chunk_row_groups,
                ) {
                    release_committed_partials_reverse(execution, partials);
                    return Err(error);
                }
                return Ok(partials);
            }
            release_committed_partials_reverse(execution, partials);
            return Err(error);
        }
    }

    Ok(partials)
}

fn execute_remaining_chunks_serial_after_scheduler_error(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    next_chunk_index: usize,
    error: QueryError,
    partials: &mut Vec<CommittedPartial>,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<()> {
    let Some(first_chunk) = chunks.get(next_chunk_index) else {
        return Err(error);
    };
    partials.extend(retry_after_chunk_error(
        execution,
        bounded,
        first_chunk.clone(),
        error,
        1,
        max_attempts,
        min_retry_chunk_row_groups,
    )?);
    for chunk in &chunks[next_chunk_index + 1..] {
        execution.context().check_cancellation("nexus_query_engine.exec.bounded.chunk_scheduler")?;
        partials.extend(execute_bounded_chunk_with_retry(execution, bounded, chunk)?);
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
        execution: &mut BoundedExecutionAttempt<'_>,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        max_attempts: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        let mut queue = Self { pending: VecDeque::new(), next_chunk_index: 0, max_in_flight: max_in_flight.max(1) };
        queue.ensure_current_required(execution, source, chunks, max_attempts)?;
        Ok(queue)
    }

    fn pop_front(&mut self) -> Option<ScheduledLocalParquetRead> {
        self.pending.pop_front()
    }

    fn ensure_current_required(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        max_attempts: usize,
    ) -> Result<()> {
        if self.pending.is_empty()
            && let Some(chunk) = chunks.get(self.next_chunk_index)
        {
            self.pending.push_back(start_required_scheduled_source_read(execution, source, chunk, 1, max_attempts)?);
            self.next_chunk_index += 1;
        }
        Ok(())
    }

    fn fill_optional(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        max_attempts: usize,
    ) -> Result<()> {
        while self.pending.len() < optional_prefetch_target(self.max_in_flight) {
            let Some(chunk) = chunks.get(self.next_chunk_index) else {
                break;
            };
            let Some(prefetch) =
                try_start_optional_scheduled_source_read(execution, source, Some(chunk), 1, max_attempts)?
            else {
                break;
            };
            self.pending.push_back(prefetch);
            self.next_chunk_index += 1;
        }
        Ok(())
    }

    fn abandon_all(&mut self, execution: &mut BoundedExecutionAttempt<'_>, next_chunk_index: usize) {
        while let Some(prefetch) = self.pending.pop_back() {
            prefetch.abandon(execution);
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
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> Result<ScheduledLocalParquetRead> {
    let requested_bytes = chunk_requested_bytes(execution.context().memory_control_policy(), chunk);
    start_scheduled_source_read(
        execution,
        source,
        chunk,
        requested_bytes,
        "source_chunk_read",
        attempt,
        max_attempts,
        false,
        false,
    )
}
fn try_start_optional_scheduled_source_read(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: Option<&SourceChunk>,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<ScheduledLocalParquetRead>> {
    let Some(chunk) = chunk else {
        return Ok(None);
    };
    let requested_bytes = prefetch_requested_bytes(execution.context().memory_control_policy(), chunk);
    if !execution.reservations().can_reserve(requested_bytes) {
        execution.metrics_mut().record_source_read_prefetch_skipped();
        return Ok(None);
    }
    execution.metrics_mut().record_source_read_prefetch_started();
    start_scheduled_source_read(
        execution,
        source,
        chunk,
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
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    requested_bytes: u64,
    budget_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    hard_token_budget: bool,
    prefetch: bool,
) -> Result<ScheduledLocalParquetRead> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.source_scheduler_start")?;
    execution.metrics_mut().record_source_chunk_attempt();
    let contract = source_read_reservation_contract(execution.context().memory_control_policy(), chunk);
    let request = chunk_reservation_request(
        ReservationKind::SourceRead,
        budget_boundary,
        requested_bytes,
        execution.reservations().budget(),
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes);
    let reservation = execution.try_reserve(request)?;
    let source_token_budget = if hard_token_budget {
        reservation.bytes()
    } else {
        execution.reservations().available_bytes_including(&reservation)
    };
    ScheduledLocalParquetRead::spawn(
        execution.context().clone(),
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
    execution: &mut BoundedExecutionAttempt<'_>,
    scheduled: ScheduledLocalParquetRead,
    _source: &LocalParquetSource,
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    let is_prefetch = scheduled.is_prefetch();
    let memory_boundary = scheduled.budget_boundary();
    let mut output = scheduled.finish(execution)?;
    if grow_source_to_observed {
        output.observed_bytes = grow_source_reservation_to_observed(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            memory_boundary,
            attempt,
            max_attempts,
        )?;
    } else {
        output.observed_bytes = match shrink_source_reservation_to_observed_without_grow(
            execution,
            &output.frame,
            &mut output.reservation,
            &output.chunk,
            memory_boundary,
            attempt,
            max_attempts,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                execution.release(output.reservation);
                return Err(error);
            },
        };
    }
    if is_prefetch {
        execution.metrics_mut().record_source_read_prefetch_completed();
    }
    Ok(output)
}
