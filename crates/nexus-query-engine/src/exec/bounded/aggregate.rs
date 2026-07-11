use arrow::{
    array::new_empty_array,
    datatypes::{DataType as ArrowDataType, SchemaRef as ArrowSchemaRef},
    record_batch::RecordBatchOptions,
};

use super::{compiled::ResidentFrameStreamPayload, *};
use crate::exec::{operators::final_global_aggregate_spec, source::record_bounded_parquet_source_metrics};

mod final_merge;
mod final_merge_policy;
mod partitioned;
mod plan;
mod post_stages;
pub mod sizing;
mod source_adjacent;
mod stream;

pub(super) use final_merge::{
    execute_compiled_bounded_final_aggregate_merge, execute_final_aggregate_merge, reduce_merge_fan_in,
};
use partitioned::{
    execute_compiled_bounded_partitioned_aggregate_from_frame, partitioned_aggregate_partition_count,
    should_partition_compiled_grouped_aggregate,
};
pub(super) use plan::{BoundedPlan, BoundedUnaryStage};
use post_stages::{aggregate_spec_has_global_final_state, execute_final_aggregate, execute_final_aggregate_projection};
pub(super) use post_stages::{execute_post_aggregate_stages, execute_post_sort_projections, execute_standalone_limit};
pub use sizing::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
use sizing::{
    partial_aggregate_upper_bound_violation_error, partial_aggregate_work_reservation_upper_bound_bytes,
    statistical_partial_aggregate_work_reservation_upper_bound_bytes,
};
pub(in crate::exec::bounded) use source_adjacent::execute_bounded_partitioned_source_adjacent_aggregate_from_raw_chunks;
pub(in crate::exec::bounded) use stream::{
    execute_bounded_whole_input_aggregate_from_raw_chunks, execute_compiled_bounded_aggregate_from_source_stream,
    execute_compiled_bounded_aggregate_from_stream,
};

fn aggregate_source_chunk_planning_policy(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
) -> SourceChunkPlanningPolicy {
    let batching_policy = BoundedSourceBatchingPolicy::default();
    let mut policy = SourceChunkPlanningPolicy::for_source_task_with_mode_and_options(
        bounded.source,
        context.memory_control_policy(),
        SourceTaskMemoryModel::aggregate_pipeline(bounded.has_source_adjacent_stages()),
        context.source_chunk_planning_mode(),
        context.source_chunk_planning_options(),
    );
    if bounded.has_source_adjacent_stages() {
        let target_chunk_bytes = batching_policy.source_adjacent_aggregate_target_chunk_bytes();
        policy.max_source_chunk_bytes =
            Some(policy.max_source_chunk_bytes.unwrap_or(target_chunk_bytes).min(target_chunk_bytes));
        policy.mark_downstream_cap("source_adjacent_aggregate_downstream_cap");
    }
    if is_source_adjacent_grouped_aggregate(bounded) && source_adjacent_grouped_aggregate_has_string_key(bounded) {
        let target_chunk_bytes = batching_policy.source_adjacent_grouped_aggregate_target_chunk_bytes();
        policy.max_source_chunk_bytes =
            Some(policy.max_source_chunk_bytes.unwrap_or(target_chunk_bytes).min(target_chunk_bytes));
        policy.max_row_groups_per_chunk = policy
            .max_row_groups_per_chunk
            .min(batching_policy.source_adjacent_grouped_aggregate_max_operator_row_groups());
        policy.mark_downstream_cap("source_adjacent_grouped_aggregate_downstream_cap");
    }
    policy
}

pub(in crate::exec::bounded) fn is_source_adjacent_grouped_aggregate(bounded: &BoundedPlan<'_>) -> bool {
    bounded.has_source_adjacent_stages()
        && !bounded.aggregate.0.spec.group_exprs.is_empty()
        && bounded.aggregate.0.spec.grouping_sets.is_none()
}

pub(in crate::exec::bounded) fn execute_bounded_aggregate_to_frame(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    context.check_cancellation("nexus_query_engine.exec.bounded.aggregate")?;
    validate_local_parquet_source(bounded.source)?;
    bounded.validate_capabilities()?;

    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        bounded.source_id,
        bounded.source,
        aggregate_source_chunk_planning_policy(context, bounded),
        context,
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.plan",
            "bounded execution requires at least one planned source chunk",
        )
        .with_fact("source_node_id", bounded.source_id));
    }
    record_bounded_parquet_source_metrics(bounded.source, &chunk_plan, metrics);

    if aggregate_spec_requires_whole_input(&bounded.aggregate.0.spec) {
        return execute_bounded_whole_input_aggregate_from_raw_chunks(
            context,
            bounded,
            &chunk_plan.chunks,
            reservations,
            metrics,
        );
    }
    if should_execute_partitioned_source_adjacent_grouped_aggregate(context, bounded, &chunk_plan) {
        return execute_bounded_partitioned_source_adjacent_aggregate_from_raw_chunks(
            context,
            bounded,
            &chunk_plan.chunks,
            reservations,
            metrics,
        );
    }

    let partials =
        execute_bounded_chunks_with_source_scheduler(context, bounded, &chunk_plan.chunks, reservations, metrics)?;
    execute_final_aggregate_merge(context, bounded, partials, reservations, metrics)
}

fn should_execute_partitioned_source_adjacent_grouped_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk_plan: &SourceChunkPlan,
) -> bool {
    bounded.has_source_adjacent_stages()
        && !bounded.aggregate.0.spec.group_exprs.is_empty()
        && !aggregate_spec_has_floating_statistical_aggregate(&bounded.aggregate.0.spec)
        && !aggregate_spec_has_median(&bounded.aggregate.0.spec)
        && bounded.aggregate.0.spec.grouping_sets.is_none()
        && chunk_plan.chunks.len() > 1
        && source_adjacent_partitioned_chunks_are_row_group_morsels(&chunk_plan.chunks)
        && source_adjacent_partitioned_raw_retention_fits_budget(context, &chunk_plan.chunks)
        && source_adjacent_group_key_partitioning_fits_rows(bounded, &chunk_plan.chunks)
        && source_adjacent_group_key_partitioning_fits_chunk_count(context, bounded, &chunk_plan.chunks)
}

fn source_adjacent_partitioned_chunks_are_row_group_morsels(chunks: &[SourceChunk]) -> bool {
    chunks.iter().all(|chunk| chunk.row_group_count() <= 1)
}

fn source_adjacent_partitioned_raw_retention_fits_budget(context: &ExecutionContext, chunks: &[SourceChunk]) -> bool {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return true;
    };

    let mut retained_bytes = 0u64;
    let mut largest_chunk_bytes = 0u64;
    for chunk in chunks {
        let Some(chunk_bytes) = chunk.estimated_chunk_bytes.or(chunk.estimated_source_read_bytes) else {
            return false;
        };
        retained_bytes = match retained_bytes.checked_add(chunk_bytes) {
            Some(bytes) => bytes,
            None => return false,
        };
        largest_chunk_bytes = largest_chunk_bytes.max(chunk_bytes);
    }

    retained_bytes.saturating_add(largest_chunk_bytes) <= device_budget_bytes
}

fn source_adjacent_group_key_partitioning_fits_rows(bounded: &BoundedPlan<'_>, chunks: &[SourceChunk]) -> bool {
    if !source_adjacent_grouped_aggregate_has_string_key(bounded) {
        return true;
    }
    chunks.iter().all(|chunk| {
        BoundedPartitionRowPolicy::default().source_adjacent_group_key_partitioning_fits_rows(chunk.known_row_count)
    })
}

fn source_adjacent_group_key_partitioning_fits_chunk_count(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> bool {
    if !source_adjacent_grouped_aggregate_has_string_key(bounded)
        || matches!(context.memory_control_policy(), MemoryControlPolicy::Unbounded)
    {
        return true;
    }
    BoundedPartitionRowPolicy::default().source_adjacent_group_key_partitioning_fits_chunk_count(chunks.len())
}

fn source_adjacent_grouped_aggregate_has_string_key(bounded: &BoundedPlan<'_>) -> bool {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&bounded.aggregate.0.spec) else {
        return false;
    };
    lowered_keys.key_columns.iter().any(|key_column| {
        bounded.aggregate_input_schema.field_with_name(key_column).is_ok_and(|field| {
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
        })
    })
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if should_execute_datafusion_exact_stddev(context, &aggregate.spec) {
        return execute_compiled_bounded_exact_host_aggregate_from_frame(
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
            execute_aggregate_with_budget_for_node(
                context,
                Some(node_id),
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

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_consuming_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let input_reservation = input.reservation;
    let result = execute_compiled_bounded_aggregate_from_frame(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        input_reserved_bytes,
        reservations,
        metrics,
    );
    reservations.release(input_reservation, metrics);
    result
}

fn execute_compiled_bounded_exact_host_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let stage = "compiled_exact_host_welford_aggregate";
    let mut work_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
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
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded.aggregate.exact_host_welford.budget_token",
        "failed to create bounded exact host-Welford aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            drop(input);
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let (prepared, spec, prepared_input_schema) = match prepare_filtered_aggregate_frame(
        context,
        input.into_frame(),
        input_schema,
        &aggregate.spec,
        Some(&token),
        metrics,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let prepared = GpuDataFrame::new(prepared);
    let (exact_input, exact_input_schema) =
        match prepare_datafusion_exact_host_aggregate_input_frame(context, prepared, &prepared_input_schema, metrics) {
            Ok(input) => input,
            Err(error) => {
                reservations.release(work_reservation, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
    let exact_input_bytes = match observe_device_memory_bytes(&exact_input, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(exact_input);
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let work_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, exact_input_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if exact_input_bytes > work_reservation.bytes() {
        let delta = exact_input_bytes.saturating_sub(work_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut work_reservation, work_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(exact_input);
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(&mut work_reservation, exact_input_bytes, work_request, metrics);
    }

    let input_preflight_request =
        MemoryReservationRequest::new(ReservationKind::Materialization, exact_input_bytes, reservations.budget())
            .with_memory_boundary("host_welford_input_materialization_preflight")
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    let input_preflight = match reservations.try_reserve_work(input_preflight_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(exact_input);
            reservations.release(work_reservation, metrics);
            return Err(error);
        },
    };
    let batch = match materialize_datafusion_exact_host_aggregate_input(context, exact_input, exact_input_schema) {
        Ok(batch) => batch,
        Err(error) => {
            reservations.release(input_preflight, metrics);
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    reservations.release(input_preflight, metrics);
    reservations.release(work_reservation, metrics);

    let output_batch = execute_datafusion_exact_host_aggregate_batch(
        context,
        Some(node_id),
        &spec,
        &batch,
        output_schema,
        metrics,
        None,
    )?;
    let output_upload_bytes = u64::try_from(record_batch_memory_size(&output_batch)).unwrap_or(u64::MAX);
    let output_upload_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_upload_bytes, reservations.budget())
            .with_memory_boundary("host_welford_output_upload_reservation")
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    let mut output_reservation = reservations.try_reserve_work(output_upload_request.clone(), metrics)?;
    let frame = match upload_datafusion_exact_host_aggregate_output(context, &output_batch).map(GpuDataFrame::new) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(output_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(output_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_resident_request = output_upload_request.clone().with_requested_bytes(output_bytes);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut output_reservation,
        output_bytes,
        &output_resident_request,
        metrics,
    ) {
        drop(frame);
        reservations.release(output_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(output_reservation, output_resident_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded::aggregate) fn execute_compiled_bounded_aggregate_from_empty_input(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input = empty_resident_frame_for_schema(
        context,
        node_id,
        input_schema,
        ReservationKind::OperatorOutput,
        "compiled_aggregate_empty_input",
        "compiled_aggregate_empty_input",
        reservations,
        metrics,
    )?;
    let input_reserved_bytes = input.reservation.bytes();
    let input_reservation = input.reservation;
    let result = execute_compiled_bounded_aggregate_from_frame(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        input_reserved_bytes,
        reservations,
        metrics,
    );
    reservations.release(input_reservation, metrics);
    result
}

fn empty_resident_frame_for_schema(
    context: &ExecutionContext,
    node_id: NodeId,
    schema: &ArrowSchemaRef,
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    stage: &'static str,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let arrays = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(schema),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.bounded.aggregate.empty_input",
            "failed to construct zero-row aggregate input batch",
        )
        .with_fact("node_id", node_id)
        .with_source(error)
    })?;
    let frame = DataFrame::from_record_batch(context.runtime(), &batch).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.aggregate.empty_input",
            "failed to upload zero-row aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
    })?;
    let frame = GpuDataFrame::new(frame);
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
            .with_memory_boundary(memory_boundary)
            .with_node_id(node_id)
            .with_stage(stage),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}

pub(in crate::exec::bounded::aggregate) fn release_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partials: Vec<CommittedPartial>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partial) = partials.pop() {
        drop(partial.frame);
        reservations.release(partial.reservation, metrics);
    }
}

pub(in crate::exec::bounded::aggregate) fn release_nested_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partitions: Vec<Vec<CommittedPartial>>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partition) = partitions.pop() {
        release_partials_reverse(reservations, partition, metrics);
    }
}
