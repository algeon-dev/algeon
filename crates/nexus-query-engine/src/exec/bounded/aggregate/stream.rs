use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;
use crate::exec::operators::{
    execute_gpu_native_statistical_final_merge_with_budget,
    execute_gpu_native_statistical_partial_aggregate_with_budget,
    should_execute_gpu_native_statistical_partial_aggregate,
};

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    mut inputs: ResidentFrameStreamPayload,
) -> Result<ResidentGpuFrame> {
    let result = (|| -> Result<ResidentGpuFrame> {
        if inputs.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
            );
        }
        if should_execute_gpu_native_statistical_partial_aggregate(execution.context(), &aggregate.spec) {
            return execute_compiled_bounded_statistical_aggregate_from_stream(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &mut inputs,
            );
        }
        if aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
            || aggregate_spec_has_median(&aggregate.spec)
        {
            let frames = inputs.collect_remaining_frames_owned(execution)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                );
            }
            let input = execute_bounded_generic_union(execution, node_id, frames)?;
            return execute_compiled_bounded_aggregate_consuming_frame(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
            );
        }
        if aggregate_spec_has_distinct(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(execution)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    execution,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                );
            }
            return execute_compiled_bounded_distinct_aggregate_from_stream(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                frames,
            );
        }

        let mut partials = Vec::with_capacity(inputs.len());
        while let Some(input) = inputs.next_yielded_frame(execution) {
            let mut input = match input {
                Ok(input) => input,
                Err(error) => {
                    release_partials_reverse(execution, partials);
                    return Err(error);
                },
            };
            let input = input.take_for_release();
            match execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
            ) {
                Ok(chunk_partials) => partials.extend(chunk_partials),
                Err(error) => {
                    release_partials_reverse(execution, partials);
                    inputs.release_remaining_frames(execution);
                    return Err(error);
                },
            }
        }

        if partials.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
            );
        }

        execute_compiled_bounded_final_aggregate_merge(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partials,
        )
    })();
    inputs.release_producer_reservations(execution);
    result
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_source_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    if aggregate_spec_requires_whole_input(&aggregate.spec) {
        let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
        return execute_compiled_bounded_aggregate_consuming_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        );
    }
    if should_execute_gpu_native_statistical_partial_aggregate(execution.context(), &aggregate.spec) {
        return execute_compiled_bounded_statistical_aggregate_from_source_stream(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            stream,
        );
    }

    let mut partials = Vec::with_capacity(stream.chunk_count());
    let stream_result = stream_bounded_join_leaf_local_parquet_source_chunks(execution, stream, |input, execution| {
        let chunk_partials = execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        )?;
        partials.extend(chunk_partials);
        Ok(())
    });
    if let Err(error) = stream_result {
        release_partials_reverse(execution, partials);
        return Err(error);
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }

    execute_compiled_bounded_final_aggregate_merge(execution, node_id, aggregate, input_schema, output_schema, partials)
}

fn execute_compiled_bounded_statistical_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: &mut ResidentFrameStreamPayload,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(inputs.len());
    while let Some(input) = inputs.next_yielded_frame(execution) {
        let mut input = match input {
            Ok(input) => input,
            Err(error) => {
                release_partials_reverse(execution, partials);
                return Err(error);
            },
        };
        let input = input.take_for_release();
        match execute_compiled_bounded_statistical_partial_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(execution, partials);
                inputs.release_remaining_frames(execution);
                return Err(error);
            },
        }
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
    )
}

fn execute_compiled_bounded_statistical_aggregate_from_source_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<ResidentGpuFrame> {
    let mut partials = Vec::with_capacity(stream.chunk_count());
    let stream_result = stream_bounded_join_leaf_local_parquet_source_chunks(execution, stream, |input, execution| {
        let partial = execute_compiled_bounded_statistical_partial_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
        )?;
        partials.push(partial);
        Ok(())
    });
    if let Err(error) = stream_result {
        release_partials_reverse(execution, partials);
        return Err(error);
    }

    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }

    execute_compiled_bounded_statistical_final_merge_from_partials(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
    )
}

fn execute_compiled_bounded_statistical_final_merge_from_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: Vec<CommittedPartial>,
) -> Result<ResidentGpuFrame> {
    let merged = execute_bounded_generic_union(execution, node_id, partials)?;
    let input_reserved_bytes = merged.reservation.bytes();
    let input_reservation = merged.reservation;
    let result = execute_bounded_generic_unary_from_frame(
        execution,
        node_id,
        "compiled_statistical_aggregate_final_merge",
        "compiled_statistical_aggregate_output",
        merged.frame,
        input_reserved_bytes,
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
    execution.release(input_reservation);
    result
}

fn execute_compiled_bounded_statistical_partial_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let upper_bound_bytes = match statistical_partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let work_request = MemoryReservationRequest::new(
        ReservationKind::PartialState,
        upper_bound_bytes,
        execution.reservations().budget(),
    )
    .with_memory_boundary("compiled_statistical_aggregate_partial_state_work")
    .with_node_id(node_id)
    .with_stage("compiled_statistical_aggregate_partial");
    let mut work_reservation = match execution.try_reserve_work(work_request) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&work_reservation),
        "compiled_statistical_aggregate_partial_state_work",
        "nexus_query_engine.bounded.aggregate.statistical_partial.budget_token",
        "failed to create bounded native statistical partial aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error);
        },
    };
    let output = execute_gpu_native_statistical_partial_aggregate_with_budget(
        execution.context(),
        Some(node_id),
        input_frame.into_frame(),
        &aggregate.spec,
        input_schema,
        &token,
        execution.metrics_mut(),
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error);
        },
    };
    execution.release(input_reservation);
    let partial_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error);
        },
    };
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_statistical_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_statistical_aggregate_partial");
    if partial_bytes > work_reservation.bytes() {
        let error = partial_aggregate_upper_bound_violation_error(node_id, work_reservation.bytes(), partial_bytes);
        execution.release(work_reservation);
        return Err(error);
    }
    execution.shrink_to_observed_with_request(&mut work_reservation, partial_bytes, partial_request.clone());
    let partial_reservation = execution.transfer(work_reservation, partial_request);
    execution.metrics_mut().record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn execute_compiled_bounded_distinct_aggregate_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    inputs: Vec<ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    let mut frames = inputs;
    let input_reserved_bytes = frames.iter().map(|frame| frame.reservation.bytes()).sum::<u64>();
    if frames.len() == 1 {
        let input = frames.pop().expect("single frame checked");
        return execute_compiled_bounded_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            input.reservation.bytes(),
        );
    }

    let request = MemoryReservationRequest::new(
        ReservationKind::OperatorOutput,
        input_reserved_bytes,
        execution.reservations().budget(),
    )
    .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
    .with_node_id(node_id)
    .with_stage("compiled_aggregate_distinct_stream_concat");
    let mut concat_reservation = match execution.try_reserve_work(request) {
        Ok(reservation) => reservation,
        Err(error) => {
            return Err(error);
        },
    };
    let first = frames.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            execution.reservations().available_bytes_including(&concat_reservation),
            "compiled_aggregate_distinct_stream_concat",
            "nexus_query_engine.bounded.aggregate.distinct_stream_concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                execution.release(concat_reservation);
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
        execution.release(frame.reservation);
    }
    execution.release(first_reservation);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, execution.metrics_mut())?;
    if concatenated_bytes > concat_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            concatenated_bytes.saturating_sub(concat_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_distinct_stream_concat");
        if let Err(error) = execution.try_grow(&mut concat_reservation, grow_request) {
            execution.release(concat_reservation);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut concat_reservation, concatenated_bytes);
    }

    execute_bounded_generic_unary(
        execution,
        node_id,
        "compiled_aggregate_distinct",
        "compiled_aggregate_distinct_output",
        ResidentGpuFrame { frame: concatenated, reservation: concat_reservation },
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
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<CommittedPartial> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        upper_bound_bytes,
    )
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
) -> Result<Vec<CommittedPartial>> {
    execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        compiled_aggregate_frame_retry_budget(execution.context()),
    )
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    retry_budget: usize,
) -> Result<Vec<CommittedPartial>> {
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    if should_split_compiled_aggregate_stream_frame(execution, &input, upper_bound_bytes, retry_budget) {
        execution.metrics_mut().record_retry_reservation_failure();
        crate::exec::retry::backoff_before_retry(
            execution.context(),
            "nexus_query_engine.exec.bounded.compiled_aggregate_stream_frame_retry",
        )?;
        let (left, right) = split_compiled_aggregate_stream_frame(execution, node_id, input)?;
        let mut partials = match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            left,
            retry_budget.saturating_sub(1),
        ) {
            Ok(partials) => partials,
            Err(error) => {
                execution.release(right.reservation);
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame_with_retry_budget(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            right,
            retry_budget.saturating_sub(1),
        ) {
            Ok(right_partials) => {
                partials.extend(right_partials);
                return Ok(partials);
            },
            Err(error) => {
                release_partials_reverse(execution, partials);
                return Err(error);
            },
        }
    }
    execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input,
        upper_bound_bytes,
    )
    .map(|partial| vec![partial])
}

fn execute_compiled_bounded_partial_aggregate_from_frame_with_upper_bound(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    upper_bound_bytes: u64,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let work_request = MemoryReservationRequest::new(
        ReservationKind::PartialState,
        upper_bound_bytes,
        execution.reservations().budget(),
    )
    .with_memory_boundary("compiled_aggregate_partial_state_work")
    .with_node_id(node_id)
    .with_stage("compiled_aggregate");
    let mut work_reservation = match execution.try_reserve_work(work_request) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&work_reservation),
        "compiled_aggregate_partial_state_work",
        "nexus_query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
        "failed to create bounded native compiled stream aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let output = execute_partial_aggregate_with_budget(
        execution.context(),
        Some(node_id),
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        None,
        execution.metrics_mut(),
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let partial_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    if partial_bytes > work_reservation.bytes() {
        let error = partial_aggregate_upper_bound_violation_error(node_id, work_reservation.bytes(), partial_bytes);
        execution.release(work_reservation);
        execution.release(input.reservation);
        return Err(error);
    }
    execution.shrink_to_observed_with_request(&mut work_reservation, partial_bytes, partial_request.clone());
    let partial_reservation = execution.transfer(work_reservation, partial_request);
    execution.release(input.reservation);
    execution.metrics_mut().record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn should_split_compiled_aggregate_stream_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: &ResidentGpuFrame,
    upper_bound_bytes: u64,
    retry_budget: usize,
) -> bool {
    retry_budget > 0 && input.frame.num_rows() > 1 && upper_bound_bytes > execution.reservations().remaining_bytes()
}

fn compiled_aggregate_frame_retry_budget(context: &ExecutionContext) -> usize {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, .. } => max_retry_attempts.saturating_sub(1),
        MemoryControlPolicy::Unbounded => 0,
    }
}

fn split_compiled_aggregate_stream_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    input: ResidentGpuFrame,
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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let left = match source_frame.with_execution(&overrides).slice(0, split_at) {
        Ok(left) => left,
        Err(error) => {
            execution.release(source_reservation);
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
            execution.release(source_reservation);
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
    execution.release(source_reservation);
    let left = GpuDataFrame::new(left);
    let right = GpuDataFrame::new(right);
    let left_bytes = observe_device_memory_bytes(&left, execution.metrics_mut())?;
    let right_bytes = observe_device_memory_bytes(&right, execution.metrics_mut())?;
    let left_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, left_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_stream_frame_split")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_stream_frame_split")
            .with_attempt(1, 1),
    )?;
    let right_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, right_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_stream_frame_split")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_stream_frame_split")
            .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(left_reservation);
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
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Result<CommittedPartial> {
    let mut raw_frames: Vec<CommittedPartial> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match execute_bounded_raw_source_chunk(execution, bounded, chunk) {
            Ok(raw) => {
                execution.metrics_mut().record_source_chunk_commit(raw.source_row_group_count);
                raw_frames.push(raw.partial);
            },
            Err(error) => {
                release_partials_reverse(execution, raw_frames);
                return Err(error);
            },
        }
    }

    if raw_frames.len() == 1 {
        let raw = raw_frames.pop().expect("single frame checked");
        let input_reserved_bytes = raw.reservation.bytes();
        let input_reservation = raw.reservation;
        let result = execute_compiled_bounded_aggregate_from_frame(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            raw.frame,
            input_reserved_bytes,
        );
        execution.release(input_reservation);
        return result;
    }

    let total_reserved_bytes = raw_frames.iter().map(|f| f.reservation.bytes()).sum::<u64>();
    let mut concat_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            total_reserved_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("distinct_aggregate_concat")
        .with_node_id(bounded.aggregate_id)
        .with_stage("distinct_aggregate_concat"),
    ) {
        Ok(r) => r,
        Err(error) => {
            release_partials_reverse(execution, raw_frames);
            return Err(error);
        },
    };
    let first = raw_frames.remove(0);
    let mut first_reservation = Some(first.reservation);
    let first_df = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            execution.reservations().available_bytes_including(&concat_reservation),
            "distinct_aggregate_concat",
            "nexus_query_engine.bounded.distinct_aggregate.concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                execution.release(concat_reservation);
                execution.release(first_reservation.take().expect("first reservation is present"));
                release_partials_reverse(execution, raw_frames);
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
        execution.release(frame.reservation);
    }
    execution.release(first_reservation.take().expect("first reservation is present"));
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concat_bytes = observe_device_memory_bytes(&concatenated, execution.metrics_mut())?;
    if concat_bytes > concat_reservation.bytes() {
        let delta = concat_bytes.saturating_sub(concat_reservation.bytes());
        if let Err(error) = execution.try_grow(
            &mut concat_reservation,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, delta, execution.reservations().budget())
                .with_memory_boundary("distinct_aggregate_concat")
                .with_node_id(bounded.aggregate_id)
                .with_stage("distinct_aggregate_concat"),
        ) {
            execution.release(concat_reservation);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut concat_reservation, concat_bytes);
    }

    let result = execute_compiled_bounded_aggregate_from_frame(
        execution,
        bounded.aggregate_id,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        concatenated,
        concat_reservation.bytes(),
    );
    execution.release(concat_reservation);
    result
}
