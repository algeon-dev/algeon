use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;
use crate::exec::operators::{
    execute_gpu_native_statistical_final_merge_with_budget,
    execute_gpu_native_statistical_partial_aggregate_with_budget,
    should_execute_gpu_native_statistical_partial_aggregate,
};

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    mut inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let result = (|| -> Result<ResidentGpuFrame> {
        if inputs.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                reservations,
                metrics,
            );
        }
        if should_execute_gpu_native_statistical_partial_aggregate(context, &aggregate.spec) {
            return execute_compiled_bounded_statistical_aggregate_from_stream(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &mut inputs,
                reservations,
                metrics,
            );
        }
        if aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
            || aggregate_spec_has_median(&aggregate.spec)
        {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    context,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    reservations,
                    metrics,
                );
            }
            let input = execute_bounded_generic_union(context, node_id, frames, reservations, metrics)?;
            return execute_compiled_bounded_aggregate_consuming_frame(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                reservations,
                metrics,
            );
        }
        if aggregate_spec_has_distinct(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    context,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    reservations,
                    metrics,
                );
            }
            return execute_compiled_bounded_distinct_aggregate_from_stream(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                frames,
                reservations,
                metrics,
            );
        }

        let mut partials = Vec::with_capacity(inputs.len());
        while let Some(input) = inputs.next_yielded_frame(reservations, metrics) {
            let mut input = match input {
                Ok(input) => input,
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
            };
            let input = input.take_for_release();
            match execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                reservations,
                metrics,
            ) {
                Ok(chunk_partials) => partials.extend(chunk_partials),
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    inputs.release_remaining_frames(reservations, metrics);
                    return Err(error);
                },
            }
        }

        if partials.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                reservations,
                metrics,
            );
        }

        execute_compiled_bounded_final_aggregate_merge(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partials,
            reservations,
            metrics,
        )
    })();
    inputs.release_producer_reservations(reservations, metrics);
    result
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_source_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if aggregate_spec_requires_whole_input(&aggregate.spec) {
        let input = materialize_bounded_join_leaf_local_parquet_source_stream(context, stream, reservations, metrics)?;
        return execute_compiled_bounded_aggregate_consuming_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            reservations,
            metrics,
        );
    }
    if should_execute_gpu_native_statistical_partial_aggregate(context, &aggregate.spec) {
        return execute_compiled_bounded_statistical_aggregate_from_source_stream(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            stream,
            reservations,
            metrics,
        );
    }

    let mut partials = Vec::with_capacity(stream.chunk_count());
    let stream_result = stream_bounded_join_leaf_local_parquet_source_chunks(
        context,
        stream,
        reservations,
        metrics,
        |input, reservations, metrics| {
            let chunk_partials = execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                reservations,
                metrics,
            )?;
            partials.extend(chunk_partials);
            Ok(())
        },
    );
    if let Err(error) = stream_result {
        release_partials_reverse(reservations, partials, metrics);
        return Err(error);
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            reservations,
            metrics,
        );
    }

    execute_compiled_bounded_final_aggregate_merge(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        reservations,
        metrics,
    )
}

fn execute_compiled_bounded_statistical_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: &mut ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(inputs.len());
    while let Some(input) = inputs.next_yielded_frame(reservations, metrics) {
        let mut input = match input {
            Ok(input) => input,
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                return Err(error);
            },
        };
        let input = input.take_for_release();
        match execute_compiled_bounded_statistical_partial_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            reservations,
            metrics,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                inputs.release_remaining_frames(reservations, metrics);
                return Err(error);
            },
        }
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            reservations,
            metrics,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        reservations,
        metrics,
    )
}

fn execute_compiled_bounded_statistical_aggregate_from_source_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(stream.chunk_count());
    let stream_result = stream_bounded_join_leaf_local_parquet_source_chunks(
        context,
        stream,
        reservations,
        metrics,
        |input, reservations, metrics| {
            let partial = execute_compiled_bounded_statistical_partial_from_frame(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                reservations,
                metrics,
            )?;
            partials.push(partial);
            Ok(())
        },
    );
    if let Err(error) = stream_result {
        release_partials_reverse(reservations, partials, metrics);
        return Err(error);
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            reservations,
            metrics,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        reservations,
        metrics,
    )
}

fn execute_compiled_bounded_statistical_final_merge_from_partials(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let merged = execute_bounded_generic_union(context, node_id, partials, reservations, metrics)?;
    let input_reserved_bytes = merged.reservation.bytes();
    let input_reservation = merged.reservation;
    let result = execute_bounded_generic_unary_from_frame(
        context,
        node_id,
        "compiled_statistical_aggregate_final_merge",
        "compiled_statistical_aggregate_output",
        merged.frame,
        input_reserved_bytes,
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_gpu_native_statistical_final_merge_with_budget(
                context,
                Some(node_id),
                input.into_frame(),
                &aggregate.spec,
                input_schema,
                output_schema,
                token.expect("budget token is created for compiled statistical final aggregate stages"),
                metrics,
            )
        },
    );
    reservations.release(input_reservation, metrics);
    result
}

fn execute_compiled_bounded_statistical_partial_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let upper_bound_bytes = match statistical_partial_aggregate_work_reservation_upper_bound_bytes(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let work_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, upper_bound_bytes, reservations.budget())
            .with_memory_boundary("compiled_statistical_aggregate_partial_state_work")
            .with_node_id(node_id)
            .with_stage("compiled_statistical_aggregate_partial");
    let mut work_reservation = match reservations.try_reserve_work(work_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input_frame);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "compiled_statistical_aggregate_partial_state_work",
        "nexus_query_engine.bounded.aggregate.statistical_partial.budget_token",
        "failed to create bounded native statistical partial aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let output = execute_gpu_native_statistical_partial_aggregate_with_budget(
        context,
        Some(node_id),
        input_frame.into_frame(),
        &aggregate.spec,
        input_schema,
        &token,
        metrics,
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    reservations.release(input_reservation, metrics);
    let partial_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error);
        },
    };
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, reservations.budget())
            .with_memory_boundary("compiled_statistical_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_statistical_aggregate_partial");
    if partial_bytes > work_reservation.bytes() {
        let error = partial_aggregate_upper_bound_violation_error(node_id, work_reservation.bytes(), partial_bytes);
        drop(frame);
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    reservations.shrink_to_observed_with_request(
        &mut work_reservation,
        partial_bytes,
        partial_request.clone(),
        metrics,
    );
    let partial_reservation = reservations.transfer(work_reservation, partial_request, metrics);
    metrics.record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn execute_compiled_bounded_distinct_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut frames = inputs;
    let input_reserved_bytes = frames.iter().map(|frame| frame.reservation.bytes()).sum::<u64>();
    if frames.len() == 1 {
        let input = frames.pop().expect("single frame checked");
        return execute_compiled_bounded_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            input.reservation.bytes(),
            reservations,
            metrics,
        );
    }

    let request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_distinct_stream_concat");
    let mut concat_reservation = match reservations.try_reserve_work(request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(frames);
            return Err(error);
        },
    };
    let first = frames.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&concat_reservation),
            "compiled_aggregate_distinct_stream_concat",
            "nexus_query_engine.bounded.aggregate.distinct_stream_concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(concat_reservation, metrics);
                drop(first_frame);
                return Err(error);
            },
        };
        let other_frame_refs = frames.iter().map(|frame| frame.frame.frame()).collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.bounded.aggregate.distinct_stream_concat",
                    "failed to concatenate distinct aggregate stream inputs",
                    error,
                ),
                &token,
            )
        })
    };
    for frame in frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation, metrics);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > concat_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            concatenated_bytes.saturating_sub(concat_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_distinct_stream_concat");
        if let Err(error) = reservations.try_grow(&mut concat_reservation, grow_request, metrics) {
            drop(concatenated);
            reservations.release(concat_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut concat_reservation, concatenated_bytes, metrics);
    }

    execute_bounded_generic_unary(
        context,
        node_id,
        "compiled_aggregate_distinct",
        "compiled_aggregate_distinct_output",
        ResidentGpuFrame { frame: concatenated, reservation: concat_reservation },
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_aggregate_with_budget(
                context,
                aggregate,
                input_schema,
                input,
                output_schema,
                token.expect("budget token is created for compiled bounded distinct aggregate stages"),
                metrics,
            )
        },
    )
}

pub(in crate::exec::bounded::aggregate) fn execute_compiled_bounded_partial_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        upper_bound_bytes,
        reservations,
        metrics,
    )
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        compiled_aggregate_frame_retry_budget(context),
        reservations,
        metrics,
    )
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    retry_budget: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    if should_split_compiled_aggregate_stream_frame(&input, upper_bound_bytes, retry_budget, reservations) {
        metrics.record_retry_reservation_failure();
        crate::exec::retry::backoff_before_retry(
            context,
            "nexus_query_engine.exec.bounded.compiled_aggregate_stream_frame_retry",
        )?;
        let (left, right) = split_compiled_aggregate_stream_frame(context, node_id, input, reservations, metrics)?;
        let mut partials = match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            left,
            retry_budget.saturating_sub(1),
            reservations,
            metrics,
        ) {
            Ok(partials) => partials,
            Err(error) => {
                drop(right.frame);
                reservations.release(right.reservation, metrics);
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            right,
            retry_budget.saturating_sub(1),
            reservations,
            metrics,
        ) {
            Ok(right_partials) => {
                partials.extend(right_partials);
                return Ok(partials);
            },
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                return Err(error);
            },
        }
    }
    execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        upper_bound_bytes,
        reservations,
        metrics,
    )
    .map(|partial| vec![partial])
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    upper_bound_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let work_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, upper_bound_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial_state_work")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    let mut work_reservation = match reservations.try_reserve_work(work_request, metrics) {
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
        "compiled_aggregate_partial_state_work",
        "nexus_query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
        "failed to create bounded native compiled stream aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let output = execute_partial_aggregate_with_budget(
        context,
        Some(node_id),
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        None,
        metrics,
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let partial_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    if partial_bytes > work_reservation.bytes() {
        let error = partial_aggregate_upper_bound_violation_error(node_id, work_reservation.bytes(), partial_bytes);
        drop(frame);
        reservations.release(work_reservation, metrics);
        reservations.release(input.reservation, metrics);
        return Err(error);
    }
    reservations.shrink_to_observed_with_request(
        &mut work_reservation,
        partial_bytes,
        partial_request.clone(),
        metrics,
    );
    let partial_reservation = reservations.transfer(work_reservation, partial_request, metrics);
    reservations.release(input.reservation, metrics);
    metrics.record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn should_split_compiled_aggregate_stream_frame(
    input: &ResidentGpuFrame,
    upper_bound_bytes: u64,
    retry_budget: usize,
    reservations: &ReservationTracker,
) -> bool {
    retry_budget > 0 && input.frame.num_rows() > 1 && upper_bound_bytes > reservations.remaining_bytes()
}

fn compiled_aggregate_frame_retry_budget(context: &ExecutionContext) -> usize {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, .. } => max_retry_attempts.saturating_sub(1),
        MemoryControlPolicy::Unbounded => 0,
    }
}

fn split_compiled_aggregate_stream_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<(ResidentGpuFrame, ResidentGpuFrame)> {
    let rows = input.frame.num_rows();
    let split_at = rows / 2;
    if split_at == 0 || split_at == rows {
        return Err(unsupported_bounded_shape(
            "compiled aggregate stream frame cannot be split further",
            "compiled_aggregate_stream_frame_retry_unsplittable",
        )
        .with_fact("node_id", node_id)
        .with_fact("input_rows", rows));
    }
    let source_reservation = input.reservation;
    let source_frame = input.frame.into_frame();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let left = match source_frame.with_execution(&overrides).slice(0, split_at) {
        Ok(left) => left,
        Err(error) => {
            reservations.release(source_reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.aggregate.stream_frame_split",
                "failed to split compiled aggregate stream frame",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("split_side", "left")
            .with_fact("input_rows", rows)
            .with_fact("split_at", split_at));
        },
    };
    let right_len = rows.saturating_sub(split_at);
    let right = match source_frame.with_execution(&overrides).slice(split_at, right_len) {
        Ok(right) => right,
        Err(error) => {
            reservations.release(source_reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.aggregate.stream_frame_split",
                "failed to split compiled aggregate stream frame",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("split_side", "right")
            .with_fact("input_rows", rows)
            .with_fact("split_at", split_at));
        },
    };
    drop(source_frame);
    reservations.release(source_reservation, metrics);
    let left = GpuDataFrame::new(left);
    let right = GpuDataFrame::new(right);
    let left_bytes = observe_device_memory_bytes(&left, metrics)?;
    let right_bytes = observe_device_memory_bytes(&right, metrics)?;
    let left_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, left_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_stream_frame_split")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_stream_frame_split")
            .with_attempt(1, 1),
        metrics,
    )?;
    let right_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, right_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_stream_frame_split")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_stream_frame_split")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(left_reservation, metrics);
            return Err(error);
        },
    };
    Ok((
        ResidentGpuFrame { frame: left, reservation: left_reservation },
        ResidentGpuFrame { frame: right, reservation: right_reservation },
    ))
}

/// Correct execution path for grouped aggregates that cannot merge chunk-local
/// partial states. Per-chunk `NUnique` values cannot be summed, and
/// `STDDEV_SAMP` needs either one DataFusion-exact host Welford pass or one
/// native tolerant libcudf std pass over the complete input.
///
/// This function reads every chunk's raw data (source read + filter +
/// projections, but NO partial aggregate), concatenates all raw frames into a
/// single in-memory frame, then runs one full groupby aggregate.
pub(in crate::exec::bounded) fn execute_bounded_whole_input_aggregate_from_raw_chunks(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let mut raw_frames: Vec<CommittedPartial> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match execute_bounded_raw_source_chunk(context, bounded, chunk, reservations, metrics) {
            Ok(raw) => {
                metrics.record_source_chunk_commit(raw.source_row_group_count);
                raw_frames.push(raw.partial);
            },
            Err(error) => {
                release_partials_reverse(reservations, raw_frames, metrics);
                return Err(error);
            },
        }
    }

    if raw_frames.len() == 1 {
        let raw = raw_frames.pop().expect("single frame checked");
        let input_reserved_bytes = raw.reservation.bytes();
        let input_reservation = raw.reservation;
        let result = execute_compiled_bounded_aggregate_from_frame(
            context,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            raw.frame,
            input_reserved_bytes,
            reservations,
            metrics,
        );
        reservations.release(input_reservation, metrics);
        return result;
    }

    let total_reserved_bytes = raw_frames.iter().map(|f| f.reservation.bytes()).sum::<u64>();
    let mut concat_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, total_reserved_bytes, reservations.budget())
            .with_memory_boundary("distinct_aggregate_concat")
            .with_node_id(bounded.aggregate_id)
            .with_stage("distinct_aggregate_concat"),
        metrics,
    ) {
        Ok(r) => r,
        Err(error) => {
            release_partials_reverse(reservations, raw_frames, metrics);
            return Err(error);
        },
    };
    let first = raw_frames.remove(0);
    let mut first_reservation = Some(first.reservation);
    let first_df = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&concat_reservation),
            "distinct_aggregate_concat",
            "nexus_query_engine.bounded.distinct_aggregate.concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(concat_reservation, metrics);
                reservations.release(first_reservation.take().expect("first reservation is present"), metrics);
                release_partials_reverse(reservations, raw_frames, metrics);
                return Err(error);
            },
        };
        let other_frame_refs: Vec<_> = raw_frames.iter().map(|f| f.frame.frame()).collect();
        first_df.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.bounded.distinct_aggregate.concat",
                    "failed to concatenate raw frames for bounded distinct aggregate",
                    error,
                ),
                &token,
            )
        })
    };
    for frame in raw_frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation.take().expect("first reservation is present"), metrics);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concat_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concat_bytes > concat_reservation.bytes() {
        let delta = concat_bytes.saturating_sub(concat_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut concat_reservation,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, delta, reservations.budget())
                .with_memory_boundary("distinct_aggregate_concat")
                .with_node_id(bounded.aggregate_id)
                .with_stage("distinct_aggregate_concat"),
            metrics,
        ) {
            drop(concatenated);
            reservations.release(concat_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut concat_reservation, concat_bytes, metrics);
    }

    let result = execute_compiled_bounded_aggregate_from_frame(
        context,
        bounded.aggregate_id,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        concatenated,
        concat_reservation.bytes(),
        reservations,
        metrics,
    );
    reservations.release(concat_reservation, metrics);
    result
}
