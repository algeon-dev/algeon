use super::*;

pub(in crate::exec::bounded::chunks) fn bounded_source_adjacent_chunk_batching_enabled(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    bounded.has_source_adjacent_stages()
        && bounded.source.partition_columns.is_empty()
        && !bounded.source.files.iter().any(LocalParquetFile::has_row_delete_files)
        && bounded.source.scan_limit.is_none()
        && bounded
            .source
            .reader_predicate
            .as_ref()
            .is_none_or(|predicate| bounded.source.reader_predicate_is_supported(predicate))
        && bounded_source_adjacent_chunk_batches(context, bounded, chunks).iter().any(|batch| batch.chunk_count() > 1)
}

pub(in crate::exec::bounded::chunks) fn bounded_source_adjacent_chunk_batches(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    if bounded.source.files.iter().any(LocalParquetFile::has_row_delete_files) {
        return singleton_source_chunk_batches(chunks);
    }
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let max_batch_source_read_bytes = batching_policy.source_adjacent_raw_source_batch_target_bytes(
        context.memory_control_policy(),
        source_adjacent_source_requires_remote_direct_read(bounded.source, chunks),
        is_source_adjacent_grouped_aggregate(bounded),
    );
    plan_source_chunk_batches(chunks, max_batch_source_read_bytes)
}

#[cfg(test)]
pub(in crate::exec::bounded::chunks) fn source_adjacent_raw_source_batch_max_bytes(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    source_adjacent_grouped_aggregate: bool,
) -> u64 {
    BoundedSourceBatchingPolicy::default().source_adjacent_raw_source_batch_max_bytes(
        source_adjacent_source_requires_remote_direct_read(source, chunks),
        source_adjacent_grouped_aggregate,
    )
}

pub(in crate::exec::bounded::chunks) fn source_adjacent_source_requires_remote_direct_read(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> bool {
    source.scan_limit.is_none()
        && source.partition_columns.is_empty()
        && chunks.iter().all(source_chunk_requires_remote_direct_read)
}

pub(in crate::exec::bounded::chunks) fn source_chunk_requires_remote_direct_read(chunk: &SourceChunk) -> bool {
    matches!(chunk.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect)
}

pub(in crate::exec::bounded::chunks) fn singleton_source_chunk_batches(
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| SourceChunkBatch {
            batch_index,
            chunks: vec![chunk.clone()],
            estimated_source_read_bytes: chunk
                .estimated_source_read_bytes
                .or(chunk.known_row_group_bytes)
                .or(chunk.estimated_chunk_bytes)
                .or(chunk.known_file_bytes),
            source_batching_reason: "source_gate:position_deletes".to_string(),
        })
        .collect()
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batches_serial(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let batches = bounded_source_adjacent_chunk_batches(context, bounded, chunks);
    let mut partials = Vec::with_capacity(batches.len());
    for batch in batches {
        context.check_cancellation("nexus_query_engine.exec.bounded.source_adjacent_chunk_batch")?;
        match execute_bounded_source_adjacent_chunk_batch_or_split(context, bounded, batch, reservations, metrics) {
            Ok(mut batch_partials) => {
                partials.append(&mut batch_partials);
            },
            Err(error) => {
                release_committed_partials_reverse(reservations, partials, metrics);
                return Err(error);
            },
        }
    }
    Ok(partials)
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batch_or_split(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.source_adjacent_chunk_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.first().expect("singleton source chunk batch has one child");
        return execute_bounded_chunk_with_retry(context, bounded, chunk, reservations, metrics);
    }
    match execute_bounded_source_adjacent_chunk_batch_attempt(context, bounded, &batch, reservations, metrics) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            crate::exec::retry::backoff_before_retry(
                context,
                "nexus_query_engine.exec.bounded.source_adjacent_chunk_batch_split_retry",
            )?;
            let mut partials =
                execute_bounded_source_adjacent_chunk_batch_or_split(context, bounded, left, reservations, metrics)?;
            match execute_bounded_source_adjacent_chunk_batch_or_split(context, bounded, right, reservations, metrics) {
                Ok(mut right_partials) => {
                    partials.append(&mut right_partials);
                    Ok(partials)
                },
                Err(error) => {
                    release_committed_partials_reverse(reservations, partials, metrics);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_source_adjacent_chunk_batch_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output = read_local_parquet_chunk_batch_required_only_no_commit(
        context,
        bounded.source,
        batch,
        reservations,
        metrics,
        "source_chunk_read",
        true,
    )?;
    execute_bounded_chunk_batch_operator_from_source(context, bounded, batch, output, reservations, metrics)
}

pub(in crate::exec::bounded::chunks) fn execute_bounded_chunk_batch_operator_from_source(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    batch: &SourceChunkBatch,
    source: ScheduledLocalParquetReadOutput,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let ScheduledLocalParquetReadOutput { frame, reservation: source_reservation, .. } = source;
    if let Err(error) = context.check_cancellation("nexus_query_engine.exec.bounded.chunk_batch_operator") {
        reservations.release(source_reservation, metrics);
        return Err(error);
    }
    let mut input = CommittedPartial { frame, reservation: source_reservation };
    input = execute_source_adjacent_chunk_stages(context, bounded, first_chunk, input, reservations, metrics, 1, 1)?;
    let CommittedPartial { frame, reservation: source_reservation } = input;

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_reservation.bytes().min(reservations.remaining_bytes()),
        reservations.budget(),
        first_chunk,
        1,
        1,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(source_reservation, metrics);
            return Err(error);
        },
    };
    let operator_token_budget = reservations.available_bytes_including(&operator_reservation);
    let operator_token = match execution_budget_token_for_boundary(
        context,
        operator_token_budget,
        "chunk_operator",
        "nexus_query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk batch operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, first_chunk, "chunk_operator", 1, 1, metrics));
        },
    };

    let partial_input_rows = frame.num_rows() as u64;
    let input_evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(&batch.chunks),
        source_adjacent_partial: true,
        source_chunk_row_group_count: batch.row_group_count() as u64,
        source_chunk_reads_all_row_groups: batch.chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: batch
            .estimated_source_read_bytes
            .or_else(|| sum_optional_u64(batch.chunks.iter().map(|chunk| chunk.estimated_source_read_bytes))),
        source_chunk_size_estimate_based: batch.chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: "none",
        source_adjacent_partition_count: None,
        source_adjacent_non_empty_partition_count: None,
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: Some(batch.chunk_count() as u64),
    };
    let chunk_operator_timer = Instant::now();
    let chunk_output = execute_partial_aggregate_with_budget(
        context,
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        frame,
        &operator_token,
        Some(input_evidence),
        metrics,
    );
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, first_chunk, "chunk_operator", 1, 1, metrics));
        },
    };

    reservations.release(operator_reservation, metrics);
    reservations.release(source_reservation, metrics);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, first_chunk, "partial_state", 1, 1, metrics))?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        first_chunk,
        1,
        1,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit_count(batch.chunk_count() as u64, batch.row_group_count() as u64);
    metrics.record_partial_aggregate(partial_input_rows, frame.num_rows() as u64, partial_reservation.bytes());

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

pub(in crate::exec::bounded::chunks) fn execute_source_adjacent_chunk_stages(
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
        input = execute_source_adjacent_chunk_projection(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            projection,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
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
        input = execute_source_adjacent_chunk_projection(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            projection,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
        )?;
    }
    Ok(input)
}

pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    projection: &ProjectionNode,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let default_work_bytes =
        source_adjacent_chunk_unary_work_request_bytes(input.reservation.bytes(), reservations.remaining_bytes());
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_source_adjacent_chunk_unary_with_work_bytes(
            context,
            node_id,
            stage,
            output_boundary,
            chunk,
            input,
            default_work_bytes,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        );
    }

    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let mut work_reservation = match reservations.try_reserve_work(
        chunk_reservation_request(
            ReservationKind::ChunkOperator,
            stage,
            default_work_bytes.max(1),
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
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    // Prepare without a stream reservation so retained prepared state keeps the
    // input frame domain. Mint a single execute token only after reconciling the
    // work reservation to the exact prepared budget.
    let prepared = match projection_regexp_replace_prepare(context, projection, &input_frame, metrics) {
        Ok(Some(prepared)) => prepared,
        Ok(None) => {
            let token = match execution_budget_token_for_boundary(
                context,
                reservations.available_bytes_including(&work_reservation),
                stage,
                "nexus_query_engine.source_chunk_unary.budget_token",
                "failed to create bounded native source-adjacent chunk operator budget token",
            ) {
                Ok(token) => token,
                Err(error) => {
                    reservations.release(work_reservation, metrics);
                    reservations.release(input_reservation, metrics);
                    return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
                },
            };
            let result = execute_projection_with_budget(context, projection, input_frame, &token, metrics);
            return finish_source_adjacent_chunk_unary(
                context,
                node_id,
                stage,
                output_boundary,
                chunk,
                input_reservation,
                work_reservation,
                result,
                reservations,
                metrics,
                attempt,
                max_attempts,
            );
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };

    let required = prepared.required_budget_bytes().max(default_work_bytes).max(1);
    if required > work_reservation.bytes() {
        let delta = required - work_reservation.bytes();
        if let Err(error) = reservations.try_grow(
            &mut work_reservation,
            chunk_reservation_request(
                ReservationKind::ChunkOperator,
                stage,
                delta,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            )
            .with_node_id(node_id)
            .with_stage(stage),
            metrics,
        ) {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        }
    }
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.source_chunk_unary.budget_token",
        "failed to create bounded native source-adjacent chunk operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };

    let result =
        execute_projection_with_budget_and_prepared(context, projection, input_frame, &token, Some(prepared), metrics);
    finish_source_adjacent_chunk_unary(
        context,
        node_id,
        stage,
        output_boundary,
        chunk,
        input_reservation,
        work_reservation,
        result,
        reservations,
        metrics,
        attempt,
        max_attempts,
    )
}

fn finish_source_adjacent_chunk_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    input_reservation: crate::exec::bounded::reservation::ActiveReservation,
    mut work_reservation: crate::exec::bounded::reservation::ActiveReservation,
    result: Result<GpuDataFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let frame = match result {
        Ok(frame) => {
            reservations.release(input_reservation, metrics);
            frame
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
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
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(work_reservation, metrics);
        return Err(attach_chunk_resource_facts(error, chunk, output_boundary, attempt, max_attempts, metrics));
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    let _ = context;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    intermediate_input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &mut QueryMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let work_request_bytes = source_adjacent_chunk_unary_work_request_bytes(
        intermediate_input.reservation.bytes(),
        reservations.remaining_bytes(),
    );
    execute_source_adjacent_chunk_unary_with_work_bytes(
        context,
        node_id,
        stage,
        output_boundary,
        chunk,
        intermediate_input,
        work_request_bytes,
        reservations,
        metrics,
        attempt,
        max_attempts,
        execute,
    )
}

fn execute_source_adjacent_chunk_unary_with_work_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    input: CommittedPartial,
    work_request_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &mut QueryMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let mut work_reservation = match reservations.try_reserve_work(
        chunk_reservation_request(
            ReservationKind::ChunkOperator,
            stage,
            work_request_bytes,
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
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.source_chunk_unary.budget_token",
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
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut work_reservation, output_bytes, metrics);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded::chunks) fn source_adjacent_chunk_unary_work_request_bytes(
    input_reserved_bytes: u64,
    remaining_bytes: u64,
) -> u64 {
    BoundedSourceReservationPolicy::default()
        .source_adjacent_unary_work_request_bytes(input_reserved_bytes, remaining_bytes)
}
pub(in crate::exec::bounded::chunks) fn grow_source_reservation_to_observed(
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

pub(in crate::exec::bounded::chunks) fn grow_source_batch_reservation_to_observed(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    batch: &SourceChunkBatch,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = match observe_device_memory_bytes(frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(attach_source_batch_resource_facts(
                error,
                batch,
                memory_boundary,
                attempt,
                max_attempts,
                metrics,
            ));
        },
    };
    let observed_request = observed_source_read_batch_reservation_request(
        memory_boundary,
        reservations.budget(),
        batch,
        attempt,
        max_attempts,
    );
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_request,
            metrics,
        );
        return Ok(source_observed_bytes);
    }

    let source_delta = source_observed_bytes.saturating_sub(source_reservation.bytes());
    reservations.try_grow(source_reservation, observed_request.with_requested_bytes(source_delta), metrics)?;
    Ok(source_observed_bytes)
}

pub(in crate::exec::bounded::chunks) fn shrink_source_reservation_to_observed_without_grow(
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

pub(in crate::exec::bounded::chunks) fn shrink_source_batch_reservation_to_observed_without_grow(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    batch: &SourceChunkBatch,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = observe_device_memory_bytes(frame, metrics).map_err(|error| {
        attach_source_batch_resource_facts(error, batch, memory_boundary, attempt, max_attempts, metrics)
    })?;
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_source_read_batch_reservation_request(
                memory_boundary,
                reservations.budget(),
                batch,
                attempt,
                max_attempts,
            ),
            metrics,
        );
    }
    Ok(source_observed_bytes)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        runtime::SourceChunkPlanningMode,
        source::{CredentialMode, LocalParquetFile, ParquetSourceLocation, ResolvedParquetSource},
    };

    #[test]
    fn remote_direct_grouped_source_adjacent_batching_uses_grouped_aggregate_cap() {
        let source = remote_direct_source();
        let chunks = vec![remote_direct_chunk(0, 128 * 1024 * 1024), remote_direct_chunk(1, 128 * 1024 * 1024)];

        assert_eq!(
            source_adjacent_raw_source_batch_max_bytes(&source, &chunks, true),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_GROUPED_AGGREGATE_TARGET_CHUNK_BYTES
        );
        assert_eq!(
            source_adjacent_raw_source_batch_max_bytes(&source, &chunks, false),
            BoundedSourceBatchingPolicy::DEFAULT_REMOTE_DIRECT_SOURCE_ADJACENT_RAW_SOURCE_BATCH_MAX_BYTES
        );
    }

    fn remote_direct_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, false)]));
        let file = LocalParquetFile::from_resolved_parquet_source(remote_direct_resolved_source()).unwrap();
        LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
    }

    fn remote_direct_chunk(chunk_index: usize, estimated_source_read_bytes: u64) -> SourceChunk {
        let resolved_source = remote_direct_resolved_source();
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(0),
            chunk_index,
            file_index: 0,
            location: ParquetSourceLocation::object_store_uri("s3://bucket/table.parquet").unwrap(),
            path: PathBuf::from("s3://bucket/table.parquet"),
            resolved_source,
            stable_id: "s3://bucket/table.parquet".to_string(),
            row_group_indexes: vec![chunk_index],
            reads_all_row_groups: false,
            known_file_bytes: Some(estimated_source_read_bytes),
            known_row_group_bytes: Some(estimated_source_read_bytes),
            known_projected_compressed_bytes: Some(estimated_source_read_bytes),
            known_projected_uncompressed_bytes: Some(estimated_source_read_bytes),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(estimated_source_read_bytes),
            estimated_source_read_bytes: Some(estimated_source_read_bytes),
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: Some(vec![chunk_index]),
            source_file_row_group_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_projected_compressed_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![estimated_source_read_bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: true,
            read_columns: vec!["c0".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        }
    }

    fn remote_direct_resolved_source() -> ResolvedParquetSource {
        ResolvedParquetSource::remote_kvikio("s3://bucket/table.parquet", CredentialMode::SharedProcessEnv)
    }
}
