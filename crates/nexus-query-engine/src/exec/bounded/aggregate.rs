use arrow::{array::new_empty_array, datatypes::SchemaRef as ArrowSchemaRef, record_batch::RecordBatchOptions};

use super::{compiled::ResidentFrameStreamPayload, *};

mod final_merge;
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
use post_stages::{
    aggregate_spec_has_global_final_state, execute_final_aggregate, execute_final_aggregate_projection,
    final_global_aggregate_spec,
};
pub(super) use post_stages::{execute_post_aggregate_stages, execute_post_sort_projections, execute_standalone_limit};
pub use sizing::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
use sizing::{partial_aggregate_upper_bound_violation_error, partial_aggregate_work_reservation_upper_bound_bytes};
pub(in crate::exec::bounded) use source_adjacent::execute_bounded_partitioned_source_adjacent_aggregate_from_raw_chunks;
pub(in crate::exec::bounded) use stream::{
    execute_bounded_whole_input_aggregate_from_raw_chunks, execute_compiled_bounded_aggregate_from_stream,
};

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
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            output_upload_request.clone().with_requested_bytes(delta),
            metrics,
        ) {
            drop(frame);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut output_reservation,
            output_bytes,
            output_resident_request.clone(),
            metrics,
        );
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
