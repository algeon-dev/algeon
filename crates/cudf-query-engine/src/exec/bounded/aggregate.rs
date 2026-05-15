use super::*;

mod final_merge;
mod plan;

pub(super) use final_merge::{execute_final_aggregate_merge, reduce_merge_fan_in};
pub(super) use plan::BoundedPlan;

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if should_partition_compiled_grouped_aggregate(context, aggregate, input_reserved_bytes) {
        return execute_compiled_bounded_partitioned_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        );
    }

    execute_bounded_generic_unary_from_frame(
        context,
        node_id,
        "compiled_aggregate",
        "compiled_aggregate_output",
        input,
        input_reserved_bytes,
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_aggregate_with_budget(
                context,
                aggregate,
                input,
                output_schema,
                token.expect("budget token is created for compiled bounded aggregate stages"),
                metrics,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_compiled_bounded_partitioned_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    let partition_count = partitioned_aggregate_partition_count(context, input_reserved_bytes);
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_hash_partition"),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&partition_reservation),
        "compiled_aggregate_hash_partition",
        "cudf_query_engine.bounded.aggregate.compiled_partitioned_hash.budget_token",
        "failed to create bounded native compiled partitioned aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            drop(input);
            return Err(error);
        },
    };

    let frame = input.into_frame();
    let partitioned = frame
        .with_execution(&token.execution_overrides())
        .hash_partition_by(lowered_keys.key_columns.iter().map(String::as_str), partition_count)
        .map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.aggregate.hash_partition",
                    "failed to hash-partition compiled bounded aggregate input",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("key_count", lowered_keys.key_columns.len())
                .with_fact("partition_count", partition_count),
                &token,
            )
        });
    drop(frame);
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_hash_partition");
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame = match slice_compiled_aggregate_partition(
            context,
            node_id,
            &partitioned_frame,
            &offsets,
            partition,
            reservations,
            metrics,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_frame,
            reservations,
            metrics,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);

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

fn slice_compiled_aggregate_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    partition: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let overrides = execution_overrides(context);
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.hash_partition_slice",
            "failed to slice compiled bounded hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_partition"),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame: gpu, reservation })
}

fn should_partition_compiled_grouped_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_reserved_bytes: u64,
) -> bool {
    !aggregate.spec.group_exprs.is_empty() && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
}

fn partitioned_aggregate_partition_count(context: &ExecutionContext, input_bytes: u64) -> usize {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return 1;
    };
    let target = partitioned_aggregate_target_partition_bytes(device_budget_bytes);
    let partitions = input_bytes.max(1).saturating_add(target.saturating_sub(1)) / target.max(1);
    usize::try_from(partitions.clamp(1, 1024)).unwrap_or(1024)
}

fn partitioned_aggregate_target_partition_bytes(device_budget_bytes: u64) -> u64 {
    const MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    const MAX_TARGET_BYTES: u64 = 1024 * 1024 * 1024;
    (device_budget_bytes / 32).clamp(MIN_TARGET_BYTES, MAX_TARGET_BYTES)
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.aggregate.compiled_stream",
            "compiled aggregate stream requires at least one input frame",
        )
        .with_fact("node_id", node_id));
    }

    let mut partials = Vec::with_capacity(inputs.len());
    for input in inputs {
        match execute_compiled_bounded_partial_aggregate_from_frame(
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
                return Err(error);
            },
        }
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

fn execute_compiled_bounded_partial_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_reserved_bytes = input.reservation.bytes();
    let input_rows = input.frame.num_rows() as u64;
    let operator_request =
        MemoryReservationRequest::new(ReservationKind::ChunkOperator, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    let operator_reservation = match reservations.try_reserve_work(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input.frame);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&operator_reservation),
        "compiled_aggregate_partial",
        "cudf_query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
        "failed to create bounded native compiled stream aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let output = execute_partial_aggregate_with_budget(
        context,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        metrics,
    );
    reservations.release(operator_reservation, metrics);
    reservations.release(input.reservation, metrics);
    let frame = output?;
    let partial_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn execute_compiled_bounded_final_aggregate_merge(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    mut partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if partials.len() == 1 {
        let partial = partials.pop().expect("single partial checked");
        return execute_compiled_final_aggregate_projection(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partial,
            reservations,
            metrics,
        );
    }

    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_compiled_bounded_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);

    while partials.len() > 1 {
        let mut merge_fan_in = initial_compiled_final_merge_fan_in(context, &partials);
        let mut attempt = 1;

        loop {
            match execute_compiled_final_merge_level_attempt(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &partials,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    metrics.record_final_merge_level();
                    partials = commit_compiled_final_merge_level(partials, level_output, reservations, metrics);
                    break;
                },
                Err(error) if is_retryable_compiled_final_merge_error(&error) => {
                    record_compiled_final_merge_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
            }
        }
    }

    let partial = partials.pop().expect("compiled final merge leaves exactly one partial");
    execute_compiled_final_aggregate_projection(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partial,
        reservations,
        metrics,
    )
}

fn initial_compiled_final_merge_fan_in(context: &ExecutionContext, partials: &[CommittedPartial]) -> usize {
    let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } = context.memory_control_policy() else {
        unreachable!("initial_compiled_final_merge_fan_in is called only for bounded policies");
    };
    let threshold_bytes = max_source_chunk_bytes.max(1);
    let total_bytes = partials.iter().map(compiled_partial_merge_basis_bytes).sum::<u64>();
    if total_bytes <= threshold_bytes {
        return partials.len();
    }

    let mut fan_in = 0usize;
    let mut selected_bytes = 0u64;
    for partial in partials {
        let partial_bytes = compiled_partial_merge_basis_bytes(partial);
        if fan_in >= 2 && selected_bytes.saturating_add(partial_bytes) > threshold_bytes {
            break;
        }
        selected_bytes = selected_bytes.saturating_add(partial_bytes);
        fan_in += 1;
    }

    fan_in.clamp(2, partials.len())
}

fn compiled_partial_merge_basis_bytes(partial: &CommittedPartial) -> u64 {
    partial.reservation.bytes().max(partial.frame.num_rows() as u64)
}

#[allow(clippy::too_many_arguments)]
fn execute_compiled_final_merge_level_attempt(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CompiledMergeLevelItem>> {
    metrics.record_final_merge_attempt(merge_fan_in);
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(CompiledMergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_compiled_final_merge_group_attempt(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            group,
            attempt,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(partial) => output.push(CompiledMergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                release_compiled_merge_level_items(output, reservations, metrics);
                return Err(error);
            },
        }
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn execute_compiled_final_merge_group_attempt(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let mut merge_reservation = reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_merge")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
        metrics,
    )?;
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&merge_reservation),
        "compiled_aggregate_final_merge",
        "cudf_query_engine.bounded.aggregate.compiled_stream_final_merge.budget_token",
        "failed to create bounded native compiled stream final-aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(merge_reservation, metrics);
            return Err(error);
        },
    };

    let timer = Instant::now();
    let result = merge_compiled_final_aggregate_partials(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        &token,
        &mut merge_reservation,
        reservations,
        attempt,
        max_attempts,
        metrics,
    );
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(error);
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_merge_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
        metrics,
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
fn merge_compiled_final_aggregate_partials(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    attempt: usize,
    max_attempts: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let concatenated = first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
    drop(other_frame_refs);
    drop(other_frames);
    drop(first_frame);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.compiled_stream_concat_partials",
            "failed to concatenate compiled stream aggregate partials",
            error,
        )
    })?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_final_merge")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_final_merge")
        .with_attempt(attempt, max_attempts);
        reservations.try_grow(merge_reservation, grow_request, metrics)?;
    } else {
        reservations.shrink_to_observed(merge_reservation, concatenated_bytes, metrics);
    }

    execute_compiled_final_aggregate(context, aggregate, input_schema, output_schema, concatenated, token, metrics)
}

enum CompiledMergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

fn commit_compiled_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<CompiledMergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Vec<CommittedPartial> {
    let mut partials = partials.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_partials = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            CompiledMergeLevelItem::Merged(partial) => next_partials.push(*partial),
            CompiledMergeLevelItem::Carry { partial_index } => {
                let partial = partials[partial_index].take().expect("carry index references a source partial");
                next_partials.push(partial);
            },
        }
    }

    for partial in partials.into_iter().flatten() {
        reservations.release(partial.reservation, metrics);
    }

    next_partials
}

fn release_compiled_merge_level_items(
    items: Vec<CompiledMergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    for item in items {
        if let CompiledMergeLevelItem::Merged(partial) = item {
            let partial = *partial;
            reservations.release(partial.reservation, metrics);
        }
    }
}

fn is_retryable_compiled_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_compiled_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn execute_compiled_final_aggregate_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let work_reservation = reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_projection")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_projection"),
        metrics,
    )?;
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "compiled_aggregate_final_projection",
        "cudf_query_engine.bounded.aggregate.compiled_stream_final_projection.budget_token",
        "failed to create bounded native compiled stream final-projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let frame =
        execute_compiled_final_aggregate(context, aggregate, input_schema, output_schema, input.frame, &token, metrics)
            .and_then(|frame| {
                finalize_compiled_final_aggregate_output(context, aggregate, output_schema, frame, &token, metrics)
            });
    reservations.release(work_reservation, metrics);
    reservations.release(input.reservation, metrics);
    let frame = frame?;
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate"),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn finalize_compiled_final_aggregate_output(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&aggregate.spec) {
        finalize_avg_output_columns_with_budget(
            context,
            input.into_frame(),
            &aggregate.spec,
            output_schema,
            token,
            metrics,
        )
    } else {
        Ok(input.into_frame())
    };
    result
        .and_then(|frame| {
            frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.compiled_final_projection.select",
                    "failed to select bounded native compiled final aggregate output columns",
                    error,
                )
            })
        })
        .map(GpuDataFrame::new)
}

fn execute_compiled_final_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let bounded = CompiledAggregatePlan { aggregate, input_schema, output_schema };
    execute_final_aggregate_for_plan(context, &bounded, input, token, metrics)
}

struct CompiledAggregatePlan<'a> {
    aggregate: &'a AggregateNode,
    input_schema: &'a arrow::datatypes::SchemaRef,
    output_schema: &'a arrow::datatypes::SchemaRef,
}

fn execute_final_aggregate_for_plan(
    context: &ExecutionContext,
    bounded: &CompiledAggregatePlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.spec);
        let output_schema = if aggregate_spec_has_global_final_state(&bounded.aggregate.spec) {
            final_global_aggregate_state_output_schema(
                &bounded.aggregate.spec,
                bounded.input_schema,
                bounded.output_schema,
            )?
        } else {
            Arc::clone(bounded.output_schema)
        };
        return execute_global_aggregate_with_budget(context, &final_spec, input, &output_schema, token, metrics);
    }

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.output_schema,
            "cudf_query_engine.final_merge.validate",
            "final aggregate output names must match output schema fields",
        )?;
    }
    let expected_output_names = if aggregate_spec_has_avg(&bounded.aggregate.spec) {
        aggregate.output_names.clone()
    } else {
        bounded.output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>()
    };

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.spec, bounded.output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    Ok(GpuDataFrame::new(output))
}

fn release_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partials: Vec<CommittedPartial>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partial) = partials.pop() {
        drop(partial.frame);
        reservations.release(partial.reservation, metrics);
    }
}

pub(super) fn execute_post_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some((projection, output_schema)) = bounded.post_aggregate_projection else {
        return Ok(input);
    };
    debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let mut output_reservation = match reservations.try_reserve(output_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    // Input remains live while cuDF evaluates the projection. The token may spend only the explicit
    // work/output reservations plus truly unreserved bytes, so it does not borrow from live input.
    let token_budget = reservations.available_bytes_including(&work_reservation);
    let token_budget = token_budget.saturating_add(output_reservation.bytes());
    let token = match execution_budget_token_for_boundary(
        context,
        token_budget,
        "post_aggregate_projection",
        "cudf_query_engine.post_aggregate_projection.budget_token",
        "failed to create bounded native post-aggregate projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(output_reservation, metrics);
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(RuntimeResourceFacts {
                memory_boundary: "post_aggregate_projection",
                stage: "post_aggregate_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "post_aggregate_projection",
                metrics,
            }
            .attach_to(error));
        },
    };
    let result = execute_projection_with_budget(context, projection, input.frame, &token, metrics);
    reservations.release(work_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(output_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(RuntimeResourceFacts {
                memory_boundary: "post_aggregate_projection",
                stage: "post_aggregate_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "post_aggregate_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation, metrics);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "post_aggregate_projection_output",
            stage: "post_aggregate_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "post_aggregate_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    if output_bytes > output_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            output_bytes.saturating_sub(output_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("post_aggregate_projection_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("post_aggregate_projection");
        if let Err(error) = reservations.try_grow(&mut output_reservation, grow_request, metrics) {
            drop(frame);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes, metrics);
    }
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_final_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if !aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
        return Ok(input);
    }

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("final_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let expected_output_names =
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        let token_budget = reservations.available_bytes_including(&work_reservation);
        let token = match execution_budget_token_for_boundary(
            context,
            token_budget,
            "final_aggregate_projection",
            "cudf_query_engine.final_projection.budget_token",
            "failed to create bounded native final aggregate projection budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(work_reservation, metrics);
                reservations.release(input.reservation, metrics);
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_aggregate_projection",
                    stage: "final_projection",
                    node_id: bounded.aggregate_id,
                    attempt: 1,
                    max_attempts: 1,
                    resume_boundary: "final_projection",
                    metrics,
                }
                .attach_to(error));
            },
        };
        finalize_avg_output_columns_with_budget(
            context,
            input.frame.into_frame(),
            &bounded.aggregate.0.spec,
            bounded.aggregate.1,
            &token,
            metrics,
        )
    } else {
        Ok(input.frame.into_frame())
    }
    .and_then(|frame| {
        frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_projection.select",
                "failed to select bounded native final aggregate output columns",
                error,
            )
        })
    })
    .map(GpuDataFrame::new);
    reservations.release(work_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_projection",
                stage: "final_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "final_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation, metrics);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_projection_output",
            stage: "final_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "final_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection");
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.0.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.0.spec);
        let output_schema = if aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
            final_global_aggregate_state_output_schema(
                &bounded.aggregate.0.spec,
                bounded.aggregate_input_schema,
                bounded.aggregate.1,
            )?
        } else {
            Arc::clone(bounded.aggregate.1)
        };
        return execute_global_aggregate_with_budget(context, &final_spec, input, &output_schema, token, metrics);
    }

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.aggregate.1,
            "cudf_query_engine.final_merge.validate",
            "final aggregate output names must match output schema fields",
        )?;
    }
    let expected_output_names = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        aggregate.output_names.clone()
    } else {
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>()
    };

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.0.spec, bounded.aggregate.1, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    Ok(GpuDataFrame::new(output))
}

pub(super) fn final_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Count => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, state.sum_expr(), state.sum().to_string()),
                        AggregateExpr::new(AggregateFunction::Sum, state.count_expr(), state.count().to_string()),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                    let valid_count_alias = global_valid_count_state_alias(expr.alias.as_str());
                    vec![
                        AggregateExpr::new(expr.function, Expr::column(expr.alias.clone()), expr.alias.clone()),
                        AggregateExpr::new(
                            AggregateFunction::Sum,
                            Expr::column(valid_count_alias.clone()),
                            valid_count_alias,
                        ),
                    ]
                },
            })
            .collect(),
    )
}

pub(super) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}
