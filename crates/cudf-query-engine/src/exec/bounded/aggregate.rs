use super::{compiled::ResidentFrameStreamPayload, *};

mod final_merge;
mod plan;
pub mod sizing;

pub(super) use final_merge::{
    execute_compiled_bounded_final_aggregate_merge, execute_final_aggregate_merge, reduce_merge_fan_in,
};
pub(super) use plan::{BoundedPlan, BoundedUnaryStage};
pub use sizing::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};

use sizing::{partial_aggregate_upper_bound_violation_error, partial_aggregate_work_reservation_upper_bound_bytes};
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
                input_schema,
                input,
                output_schema,
                token.expect("budget token is created for compiled bounded aggregate stages"),
                metrics,
            )
        },
    )
}
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
    let mut lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    if aggregate.spec.grouping_sets.is_some() {
        lowered_keys.key_columns.truncate(aggregate.spec.group_exprs.len());
    }
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
    !aggregate_spec_has_stddev_samp(&aggregate.spec)
        && !aggregate.spec.group_exprs.is_empty()
        && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
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
    mut inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let result = (|| -> Result<ResidentGpuFrame> {
        if inputs.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.bounded.aggregate.compiled_stream",
                "compiled aggregate stream requires at least one input frame",
            )
            .with_fact("node_id", node_id));
        }
        if aggregate_spec_has_stddev_samp(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
            let input = execute_bounded_generic_union(context, node_id, frames, reservations, metrics)?;
            return execute_bounded_generic_unary(
                context,
                node_id,
                "compiled_stddev_aggregate",
                "compiled_stddev_aggregate_output",
                input,
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
                        token.expect("budget token is created for compiled bounded stddev aggregate stages"),
                        metrics,
                    )
                },
            );
        }
        if aggregate_spec_has_distinct(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
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
                    inputs.release_remaining_frames(reservations, metrics);
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
    })();
    inputs.release_producer_reservations(reservations, metrics);
    result
}

fn execute_compiled_bounded_distinct_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
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
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&concat_reservation),
        "compiled_aggregate_distinct_stream_concat",
        "cudf_query_engine.bounded.aggregate.distinct_stream_concat.budget_token",
        "failed to create bounded native distinct aggregate concat budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            drop(frames);
            return Err(error);
        },
    };

    let first = frames.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let other_frame_refs = frames.iter().map(|frame| frame.frame.frame()).collect::<Vec<_>>();
    let concatenated = first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
    for frame in frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation, metrics);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "cudf_query_engine.bounded.aggregate.distinct_stream_concat",
                "failed to concatenate distinct aggregate stream inputs",
                error,
            ),
            &token,
        )
    })?;
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
    let input_rows = input.frame.num_rows() as u64;
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
        "cudf_query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
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
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
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

pub(super) fn execute_post_aggregate_stages(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_aggregate_stages, input, reservations, metrics)
}

pub(super) fn execute_post_sort_projections(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_sort_projections, input, reservations, metrics)
}

pub(super) fn execute_standalone_limit(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if bounded.sort.is_some() {
        return Ok(input);
    }
    let Some((node_id, limit)) = bounded.limit else {
        return Ok(input);
    };
    execute_post_final_stage(
        context,
        node_id,
        "post_aggregate_limit",
        "post_aggregate_limit_output",
        input,
        reservations,
        metrics,
        |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
        false,
    )
}

fn execute_post_final_stages(
    context: &ExecutionContext,
    stages: &[BoundedUnaryStage<'_>],
    mut input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    for stage in stages {
        input = match stage {
            BoundedUnaryStage::Filter { node_id, node } => execute_post_final_stage(
                context,
                *node_id,
                "post_aggregate_filter",
                "post_aggregate_filter_output",
                input,
                reservations,
                metrics,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        node,
                        input,
                        token.expect("budget token is created for bounded post-aggregate filter stages"),
                        metrics,
                    )
                },
                true,
            )?,
            BoundedUnaryStage::Projection { node_id, node, output_schema } => {
                debug_assert_eq!(node.output_schema().as_ref(), output_schema.as_ref());
                execute_post_final_stage(
                    context,
                    *node_id,
                    "post_aggregate_projection",
                    "post_aggregate_projection_output",
                    input,
                    reservations,
                    metrics,
                    |context, input, token, metrics| {
                        execute_projection_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for bounded post-aggregate projection stages"),
                            metrics,
                        )
                    },
                    true,
                )?
            },
        };
    }
    Ok(input)
}

fn execute_post_final_stage(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
    use_budget_token: bool,
) -> Result<CommittedPartial> {
    execute_bounded_generic_unary_consuming_input_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::FinalMerge,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
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
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
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
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn legacy_bounded_final_merge_task_id(node_id: NodeId, stage: &str) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.{stage}")
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
                AggregateFunction::StddevSamp => vec![expr.clone()],
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

/// Correct execution path for grouped `COUNT(DISTINCT)` when the source spans
/// multiple chunks.  Per-chunk `NUnique` values cannot be correctly merged with
/// `Sum` because the same value may appear in multiple chunks for the same
/// group key, which would be counted multiple times.
///
/// This function reads every chunk's raw data (source read + filter +
/// projections, but NO partial aggregate), concatenates all raw frames into a
/// single in-memory frame, then runs one full groupby aggregate — matching the
/// semantics of `execute_compiled_bounded_distinct_aggregate_from_stream` for
/// the legacy bounded path.
pub(in crate::exec::bounded) fn execute_bounded_distinct_aggregate_from_raw_chunks(
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
                metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
                raw_frames.push(raw);
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
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&concat_reservation),
        "distinct_aggregate_concat",
        "cudf_query_engine.bounded.distinct_aggregate.concat.budget_token",
        "failed to create bounded native distinct aggregate concat budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            release_partials_reverse(reservations, raw_frames, metrics);
            return Err(error);
        },
    };

    let first = raw_frames.remove(0);
    let first_reservation = first.reservation;
    let first_df = first.frame.into_frame();
    let other_frame_refs: Vec<_> = raw_frames.iter().map(|f| f.frame.frame()).collect();
    let concatenated = first_df.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
    for frame in raw_frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation, metrics);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "cudf_query_engine.bounded.distinct_aggregate.concat",
                "failed to concatenate raw frames for bounded distinct aggregate",
                error,
            ),
            &token,
        )
    })?;
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
