use super::{super::source::read_local_parquet_chunk, *};

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
        Err(error) if is_retryable_chunk_error(&error) => {
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
        },
        Err(error) => Err(error),
    }
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
    metrics.record_source_chunk_attempt();

    let source_request = chunk_reservation_request(
        ReservationKind::SourceRead,
        "source_chunk_read",
        chunk_requested_bytes(context.memory_control_policy(), chunk),
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let mut source_reservation = reservations.try_reserve(source_request, metrics)?;
    let source_token_budget = reservations.available_bytes_including(&source_reservation);
    let mut frame = match read_local_parquet_chunk(
        context,
        bounded.source,
        chunk,
        source_token_budget,
        "source_chunk_read",
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
        },
    };
    let source_observed_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
        },
    };
    if source_observed_bytes > source_reservation.bytes() {
        let source_delta = source_observed_bytes.saturating_sub(source_reservation.bytes());
        let grow_request = chunk_reservation_request(
            ReservationKind::SourceRead,
            "source_chunk_read",
            source_delta,
            reservations.budget(),
            chunk,
            attempt,
            max_attempts,
        );
        if let Err(error) = reservations.try_grow(&mut source_reservation, grow_request, metrics) {
            drop(frame);
            reservations.release(source_reservation);
            return Err(error);
        }
    }

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_observed_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(frame);
            reservations.release(source_reservation);
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
            reservations.release(operator_reservation);
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    let chunk_operator_timer = Instant::now();
    let chunk_output = (|| {
        for (projection, output_schema) in &bounded.pre_filter_projections {
            debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
            frame = execute_projection_with_budget(context, projection, frame, &operator_token, metrics)?;
        }
        if let Some(filter) = bounded.filter {
            frame = execute_filter_with_budget(context, filter, frame, &operator_token, metrics)?;
        }
        for (projection, output_schema) in &bounded.post_filter_projections {
            debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
            frame = execute_projection_with_budget(context, projection, frame, &operator_token, metrics)?;
        }
        execute_partial_aggregate_with_budget(
            context,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            frame,
            &operator_token,
            metrics,
        )
    })();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation);
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    reservations.release(operator_reservation);
    reservations.release(source_reservation);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, "partial_state", attempt, max_attempts, metrics))?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    metrics.record_partial_aggregate(frame.num_rows() as u64);

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

pub(super) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(fallback)
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

pub(super) fn is_retryable_chunk_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
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
    let left = retry_child_chunk(chunk, left_row_groups);
    let right = retry_child_chunk(chunk, right_row_groups);
    Some((left, right))
}

pub(super) fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
    let parent_row_group_count = parent.row_group_indexes.len();
    let known_row_group_bytes =
        proportional_bytes(parent.known_row_group_bytes, row_group_indexes.len(), parent_row_group_count);
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
    let known_row_count = proportional_bytes(parent.known_row_count, row_group_indexes.len(), parent_row_group_count);

    SourceChunk {
        source_node_id: parent.source_node_id,
        chunk_index: parent.chunk_index,
        file_index: parent.file_index,
        path: parent.path.clone(),
        row_group_indexes,
        reads_all_row_groups: false,
        known_file_bytes,
        known_row_group_bytes,
        estimated_chunk_bytes,
        read_columns: parent.read_columns.clone(),
        partition_values: parent.partition_values.clone(),
        zero_column_count: parent.zero_column_count,
        known_row_count,
        size_estimate_based: true,
    }
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
