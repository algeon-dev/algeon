use super::*;

pub(in crate::exec::bounded) struct BoundedJoinSideFrame {
    pub(in crate::exec::bounded) frame: DataFrame,
    pub(in crate::exec::bounded) _reservation: ActiveReservation,
    pub(in crate::exec::bounded) rows: u64,
    pub(in crate::exec::bounded) bytes: u64,
}

pub(in crate::exec::bounded) fn release_bounded_join_side_frame(
    side: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedJoinSideFrame { frame, _reservation: reservation, .. } = side;
    drop(frame);
    reservations.release(reservation, metrics);
}

enum JoinInputReservation<'a> {
    Shared { reservation: &'a ActiveReservation },
    Movable { reservation: Box<ActiveReservation> },
}

pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_resident_frame(
    input: ResidentGpuFrame,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(
        input.frame,
        JoinInputReservation::Movable { reservation: Box::new(input.reservation) },
        columns,
        input_side,
        node_id,
        build_side,
        reservations,
        metrics,
    )
}
pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_shared_frame(
    input: GpuDataFrame,
    source_reservation: &ActiveReservation,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(
        input,
        JoinInputReservation::Shared { reservation: source_reservation },
        columns,
        input_side,
        node_id,
        build_side,
        reservations,
        metrics,
    )
}
fn prepare_bounded_join_side_input(
    input: GpuDataFrame,
    input_reservation: JoinInputReservation<'_>,
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
    let (input_reserved_bytes, source_reservation, mut movable_input_reservation) = match input_reservation {
        JoinInputReservation::Shared { reservation } => (reservation.bytes(), Some(reservation), None),
        JoinInputReservation::Movable { reservation } => (reservation.bytes(), None, Some(*reservation)),
    };
    let input_frame = input.into_frame();
    let columns = join_input_columns_with_order(&input_frame, columns);
    let select_memory_effect = match input_frame.select_memory_effect(&columns) {
        Ok(effect) => effect,
        Err(error) => {
            release_movable_join_input_reservation(&mut movable_input_reservation, reservations, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.select_memory_effect",
                "failed to determine native join input selection memory effect",
                error,
            )
            .with_fact("side", join_side_label(input_side))
            .with_fact("columns", columns.join(",")));
        },
    };
    let mut work_reservation = if select_memory_effect.is_alias_of_input() {
        None
    } else {
        Some(
            reservations.try_reserve_work(
                MemoryReservationRequest::new(
                    ReservationKind::OperatorOutput,
                    input_reserved_bytes,
                    reservations.budget(),
                )
                .with_memory_boundary("join_input_projection")
                .with_node_id(node_id)
                .with_stage("bounded_join")
                .with_attempt(1, 1),
                metrics,
            )?,
        )
    };
    let selected = match select_join_input(input_frame, &columns, join_side_label(input_side)) {
        Ok(frame) => frame,
        Err(error) => {
            if let Some(work_reservation) = work_reservation {
                reservations.release(work_reservation, metrics);
            }
            release_movable_join_input_reservation(&mut movable_input_reservation, reservations, metrics);
            return Err(error);
        },
    };
    let rows = selected.num_rows() as u64;
    let bytes = match observe_device_memory_bytes(&GpuDataFrame::new(selected.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            if let Some(work_reservation) = work_reservation {
                reservations.release(work_reservation, metrics);
            }
            release_movable_join_input_reservation(&mut movable_input_reservation, reservations, metrics);
            return Err(error);
        },
    };
    let reservation = if select_memory_effect.is_alias_of_input() {
        if let Some(input_reservation) = movable_input_reservation.take() {
            let request = bounded_join_request(
                reservation_kind,
                memory_boundary,
                input_reserved_bytes,
                reservations.budget(),
                node_id,
            );
            if input_reservation.is_alias() {
                reservations.relabel_alias_reservation(input_reservation, request, metrics)
            } else {
                reservations.transfer(input_reservation, request, metrics)
            }
        } else if let Some(source_reservation) = source_reservation {
            reservations.alias_reservation(
                source_reservation,
                bounded_join_request(reservation_kind, memory_boundary, bytes, reservations.budget(), node_id),
                metrics,
            )
        } else {
            match reservations.try_reserve(
                bounded_join_request(reservation_kind, memory_boundary, bytes, reservations.budget(), node_id),
                metrics,
            ) {
                Ok(reservation) => reservation,
                Err(error) => {
                    if let Some(work_reservation) = work_reservation {
                        reservations.release(work_reservation, metrics);
                    }
                    return Err(error);
                },
            }
        }
    } else {
        release_movable_join_input_reservation(&mut movable_input_reservation, reservations, metrics);
        let mut work_reservation =
            work_reservation.take().expect("non-alias join input selection should reserve projection work");
        let output_request =
            bounded_join_request(reservation_kind, memory_boundary, bytes, reservations.budget(), node_id);
        if bytes > work_reservation.bytes() {
            let delta = bytes.saturating_sub(work_reservation.bytes());
            if let Err(error) = reservations.try_grow(
                &mut work_reservation,
                output_request.clone().with_requested_bytes(delta),
                metrics,
            ) {
                reservations.release(work_reservation, metrics);
                return Err(error);
            }
        } else {
            reservations.shrink_to_observed_with_request(&mut work_reservation, bytes, output_request.clone(), metrics);
        }
        reservations.transfer(work_reservation, output_request, metrics)
    };
    if let Some(work_reservation) = work_reservation {
        reservations.release(work_reservation, metrics);
    }
    release_movable_join_input_reservation(&mut movable_input_reservation, reservations, metrics);
    Ok(BoundedJoinSideFrame { frame: selected, _reservation: reservation, rows, bytes })
}

fn release_movable_join_input_reservation(
    reservation: &mut Option<ActiveReservation>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    if let Some(reservation) = reservation.take() {
        reservations.release(reservation, metrics);
    }
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

// Diagnostic trace surface for the bounded-join output-materialize preflight.
// Gated by `CUDF_BOUNDED_JOIN_MATERIALIZE_TRACE=1`. Emits per-call entry /
// success / failure events on stderr with the reservation-tracker snapshot
// (budget, reserved, remaining) at the moment the libcudf preflight runs.
// Used to localize unreserved-allocation hazards in the join-materialize
// chokepoint without changing TSV report schemas.
pub(in crate::exec::bounded) fn trace_preflight_enabled() -> bool {
    std::env::var("CUDF_BOUNDED_JOIN_MATERIALIZE_TRACE").map(|v| v == "1").unwrap_or(false)
}

pub(in crate::exec::bounded) fn trace_preflight_entry(
    caller: &'static str,
    execution_strategy: &str,
    node_id: NodeId,
    reservations: &ReservationTracker,
    gather_map_bytes: u64,
    expected_output_rows: u64,
    left_rows: u64,
    left_bytes: u64,
    right_rows: u64,
    right_bytes: u64,
) {
    if !trace_preflight_enabled() {
        return;
    }
    eprintln!(
        "[bounded-join-materialize-trace] phase=preflight_entry caller={} execution_strategy={} node_id={} budget_bytes={} reserved_bytes={} remaining_bytes={} gather_map_bytes={} expected_output_rows={} left_rows={} left_bytes={} right_rows={} right_bytes={}",
        caller,
        execution_strategy,
        node_id,
        reservations.budget(),
        reservations.reserved_bytes(),
        reservations.remaining_bytes(),
        gather_map_bytes,
        expected_output_rows,
        left_rows,
        left_bytes,
        right_rows,
        right_bytes,
    );
}

pub(in crate::exec::bounded) fn trace_preflight_success(
    caller: &'static str,
    node_id: NodeId,
    reservations: &ReservationTracker,
    output_reservation_bytes: u64,
) {
    if !trace_preflight_enabled() {
        return;
    }
    eprintln!(
        "[bounded-join-materialize-trace] phase=preflight_success caller={} node_id={} output_reservation_bytes={} reserved_bytes_now={} remaining_bytes_now={}",
        caller,
        node_id,
        output_reservation_bytes,
        reservations.reserved_bytes(),
        reservations.remaining_bytes(),
    );
}

pub(in crate::exec::bounded) fn trace_preflight_failed(
    caller: &'static str,
    node_id: NodeId,
    reservations: &ReservationTracker,
    error: &QueryError,
) {
    if !trace_preflight_enabled() {
        return;
    }
    eprintln!(
        "[bounded-join-materialize-trace] phase=preflight_failed caller={} node_id={} reserved_bytes_at_failure={} remaining_bytes_at_failure={} budget_bytes={} error_code={:?}",
        caller,
        node_id,
        reservations.reserved_bytes(),
        reservations.remaining_bytes(),
        reservations.budget(),
        error.code(),
    );
}

pub(super) fn join_side_output_reservation_bytes(
    frame: &DataFrame,
    indices: &cudf::DataFrameJoinIndices,
    columns: &[String],
    overrides: &ExecutionOverrides<'_>,
    side: JoinSide,
) -> Result<u64> {
    let mut columns = columns.to_vec();
    join_side_projection_with_order(frame, &mut columns);
    if columns.is_empty() {
        return Ok(0);
    }
    frame
        .with_execution(overrides)
        .materialize_join_indices_reservation_size_bytes(indices, &columns)
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
