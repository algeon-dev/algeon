use arrow::{array::new_empty_array, record_batch::RecordBatchOptions};

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

pub(in crate::exec::bounded::aggregate) fn execute_compiled_bounded_aggregate_from_empty_input(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
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
    schema: &arrow::datatypes::SchemaRef,
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
            "cudf_query_engine.bounded.aggregate.empty_input",
            "failed to construct zero-row aggregate input batch",
        )
        .with_fact("node_id", node_id)
        .with_source(error)
    })?;
    let frame = DataFrame::from_record_batch(context.runtime(), &batch).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.empty_input",
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
