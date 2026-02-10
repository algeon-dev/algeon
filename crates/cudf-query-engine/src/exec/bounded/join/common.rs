use super::*;

pub(in crate::exec::bounded) struct BoundedJoinSideFrame {
    pub(in crate::exec::bounded) frame: DataFrame,
    pub(in crate::exec::bounded) _reservation: ActiveReservation,
    pub(in crate::exec::bounded) rows: u64,
    pub(in crate::exec::bounded) bytes: u64,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_frame(
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    let reservation_kind =
        if input_side == build_side { ReservationKind::BuildSide } else { ReservationKind::ProbeSide };
    let memory_boundary = if input_side == build_side { "join_build_side" } else { "join_probe_side" };
    let work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("join_input_projection")
            .with_node_id(node_id)
            .with_stage("bounded_join")
            .with_attempt(1, 1),
        metrics,
    )?;
    let selected = match select_join_input(input.into_frame(), columns, join_side_label(input_side)) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error);
        },
    };
    let rows = selected.num_rows() as u64;
    let bytes = match observe_device_memory_bytes(&GpuDataFrame::new(selected.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error);
        },
    };
    let reservation = match reservations.try_reserve(
        bounded_join_request(reservation_kind, memory_boundary, bytes, reservations.budget(), node_id),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error);
        },
    };
    reservations.release(work_reservation, metrics);
    Ok(BoundedJoinSideFrame { frame: selected, _reservation: reservation, rows, bytes })
}

fn join_side_label(side: JoinSide) -> &'static str {
    match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    }
}

pub(super) fn bounded_join_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
) -> MemoryReservationRequest {
    MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage("bounded_join")
        .with_attempt(1, 1)
}

pub(super) fn join_side_output_reservation_bytes(
    frame: &DataFrame,
    indices: &cudf::DataFrameJoinIndices,
    columns: &[String],
    overrides: &ExecutionOverrides<'_>,
    side: JoinSide,
) -> Result<u64> {
    if columns.is_empty() {
        return Ok(0);
    }
    frame
        .with_execution(overrides)
        .materialize_join_indices_reservation_size_bytes(indices, columns)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.output_reservation",
                "failed to size bounded native join output materialization",
                error,
            )
            .with_fact("side", side)
        })
}
