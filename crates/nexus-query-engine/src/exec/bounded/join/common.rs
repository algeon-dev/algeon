use std::collections::BTreeMap;

use arrow::datatypes::DataType as ArrowDataType;

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
                "nexus_query_engine.bounded.join.select_memory_effect",
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

pub(super) fn reserve_full_join_size_preflight_workspace(
    node_id: NodeId,
    node: &JoinNode,
    build_rows: u64,
    left_join_output_rows: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<ActiveReservation>> {
    if node.spec.join_type != JoinType::Full {
        return Ok(None);
    }
    let bytes = full_join_size_preflight_workspace_reservation_bytes(node, build_rows, left_join_output_rows)?;
    if bytes == 0 {
        return Ok(None);
    }
    reservations
        .try_reserve(
            bounded_join_request(
                ReservationKind::GatherMap,
                "join_full_size_preflight_workspace",
                bytes,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )
        .map(Some)
}

fn full_join_size_preflight_workspace_reservation_bytes(
    node: &JoinNode,
    build_rows: u64,
    left_join_output_rows: u64,
) -> Result<u64> {
    let temporary_index_rows = left_join_output_rows.checked_add(build_rows).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.bounded.join.full_size_preflight_workspace",
            "full join size preflight temporary index-row upper bound overflowed",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_rows", build_rows)
        .with_fact("left_join_output_rows", left_join_output_rows)
    })?;
    let temporary_index_rows = usize::try_from(temporary_index_rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.bounded.join.full_size_preflight_workspace",
            "full join size preflight temporary index-row upper bound exceeds platform usize range",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_rows", build_rows)
        .with_fact("left_join_output_rows", left_join_output_rows)
        .with_fact("temporary_index_rows", temporary_index_rows)
    })?;
    cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(temporary_index_rows).map(|bytes| bytes as u64).map_err(
        |error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.full_size_preflight_workspace",
                "failed to size full join size-preflight temporary index workspace",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_rows", build_rows)
            .with_fact("left_join_output_rows", left_join_output_rows)
            .with_fact("temporary_index_rows", temporary_index_rows)
        },
    )
}

// Diagnostic trace surface for the bounded-join output-materialize preflight.
// Gated by `NEXUS_BOUNDED_JOIN_MATERIALIZE_TRACE=1`. Emits per-call entry /
// success / failure events on stderr with the reservation-tracker snapshot
// (budget, reserved, remaining) at the moment the libcudf preflight runs.
// Used to localize unreserved-allocation hazards in the join-materialize
// chokepoint without changing TSV report schemas.
pub(in crate::exec::bounded) fn trace_preflight_enabled() -> bool {
    std::env::var("NEXUS_BOUNDED_JOIN_MATERIALIZE_TRACE").map(|v| v == "1").unwrap_or(false)
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
        "[bounded-join-materialize-trace] stage=preflight_entry caller={} execution_strategy={} node_id={} budget_bytes={} reserved_bytes={} remaining_bytes={} gather_map_bytes={} expected_output_rows={} left_rows={} left_bytes={} right_rows={} right_bytes={}",
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
        "[bounded-join-materialize-trace] stage=preflight_success caller={} node_id={} output_reservation_bytes={} reserved_bytes_now={} remaining_bytes_now={}",
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
        "[bounded-join-materialize-trace] stage=preflight_failed caller={} node_id={} reserved_bytes_at_failure={} remaining_bytes_at_failure={} budget_bytes={} error_code={:?}",
        caller,
        node_id,
        reservations.reserved_bytes(),
        reservations.remaining_bytes(),
        reservations.budget(),
        error.code(),
    );
}

pub(super) fn join_side_output_upper_bound_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_rows: u64,
    columns: &[String],
    side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<u64> {
    let bounds = join_side_output_upper_bound_bytes_for_row_counts(
        context,
        node_id,
        frame,
        &[expected_output_rows],
        columns,
        side,
        reservations,
        metrics,
    )?;
    Ok(bounds.into_iter().next().unwrap_or(0))
}

pub(super) fn join_side_output_upper_bound_bytes_for_row_counts(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_row_counts: &[u64],
    columns: &[String],
    side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<u64>> {
    let mut columns = columns.to_vec();
    join_side_projection_with_order(frame, &mut columns);
    if columns.is_empty() {
        return Ok(vec![0; expected_output_row_counts.len()]);
    }

    let mut upper_bound_columns = Vec::with_capacity(columns.len());
    let mut column_cache: BTreeMap<String, JoinOutputColumnUpperBoundInput> = BTreeMap::new();
    for column in columns {
        let upper_bound = if let Some(upper_bound) = column_cache.get(column.as_str()) {
            upper_bound.clone()
        } else {
            let upper_bound = join_output_upper_bound_column_for_frame_column(
                context,
                node_id,
                frame,
                column.as_str(),
                side,
                reservations,
                metrics,
            )?;
            column_cache.insert(column.clone(), upper_bound.clone());
            upper_bound
        };
        upper_bound_columns.push(upper_bound);
    }
    expected_output_row_counts
        .iter()
        .map(|expected_output_rows| {
            join_output_upper_bound_bytes(*expected_output_rows, &upper_bound_columns).map_err(|error| {
                error
                    .with_fact_if_absent("node_id", node_id)
                    .with_fact_if_absent("side", side)
                    .with_fact_if_absent("expected_output_rows", *expected_output_rows)
            })
        })
        .collect()
}

pub(super) fn projected_output_columns_for_retained_sizing(
    projection: &[JoinOutputColumn],
    side: JoinSide,
) -> Vec<String> {
    projection.iter().filter(|column| column.side == side).map(|column| column.column.name().to_string()).collect()
}

fn join_output_upper_bound_column_for_frame_column(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<JoinOutputColumnUpperBoundInput> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.output_upper_bound",
            "failed to access join output source column for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
    })?;
    let cudf_dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.output_upper_bound",
            "failed to inspect join output source column dtype for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
    })?;
    let data_type = cudf::interop::arrow_host::arrow_type_from_cudf_dtype(cudf_dtype).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.output_upper_bound",
            "failed to convert join output source column dtype for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
        .with_fact("cudf_dtype", format!("{cudf_dtype:?}"))
    })?;
    let input_max_string_length_bytes = if join_output_dtype_needs_string_max(&data_type) {
        Some(max_string_byte_length_for_join_output_sizing(
            context,
            node_id,
            frame,
            column,
            side,
            reservations,
            metrics,
        )?)
    } else {
        None
    };
    Ok(JoinOutputColumnUpperBoundInput { data_type, input_max_string_length_bytes, input_total_string_bytes: None })
}

fn max_string_byte_length_for_join_output_sizing(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<u64> {
    if frame.num_rows() == 0 {
        return Ok(0);
    }
    let byte_count_bound = join_output_upper_bound_bytes(
        frame.num_rows() as u64,
        &[JoinOutputColumnUpperBoundInput {
            data_type: ArrowDataType::Int32,
            input_max_string_length_bytes: None,
            input_total_string_bytes: None,
        }],
    )?;
    let mut reservation = reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::ChunkOperator, byte_count_bound, reservations.budget())
            .with_memory_boundary("join_output_string_length_sizing")
            .with_node_id(node_id)
            .with_stage("join_output_string_length_sizing"),
        metrics,
    )?;
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&reservation),
        "join_output_string_length_sizing",
        "nexus_query_engine.bounded.join.output_string_length_sizing.budget_token",
        "failed to create bounded native join output string-length sizing budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(error.with_fact("node_id", node_id).with_fact("side", side).with_fact("column", column));
        },
    };

    let series = match frame.series(column) {
        Ok(series) => series,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.output_string_length_sizing",
                "failed to access string join output source column for sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    let byte_counts = match series.str().byte_count_in(&token.execution_overrides()) {
        Ok(byte_counts) => byte_counts,
        Err(error) => {
            reservations.release(reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.output_string_length_sizing",
                "failed to compute string byte counts for join output sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    let observed = match byte_counts.device_allocated_size_bytes() {
        Ok(bytes) => bytes as u64,
        Err(error) => {
            drop(byte_counts);
            reservations.release(reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.output_string_length_sizing",
                "failed to observe string byte-count sizing memory",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    if observed > reservation.bytes() {
        let error = join_output_upper_bound_violation_error(node_id, reservation.bytes(), observed)
            .with_fact("stage", "join_output_string_length_sizing")
            .with_fact("side", side)
            .with_fact("column", column);
        drop(byte_counts);
        reservations.release(reservation, metrics);
        return Err(error);
    }
    reservations.shrink_to_observed(&mut reservation, observed, metrics);
    let max_len = match byte_counts.max_value_in::<i32>(&token.execution_overrides()) {
        Ok(max_len) => max_len,
        Err(error) => {
            drop(byte_counts);
            reservations.release(reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.output_string_length_sizing",
                "failed to reduce string byte-count sizing column",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    drop(byte_counts);
    reservations.release(reservation, metrics);
    Ok(max_len.unwrap_or(0).max(0) as u64)
}

fn join_output_dtype_needs_string_max(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

pub(super) fn reconcile_join_output_upper_bound(
    node_id: NodeId,
    output_reservation: &mut ActiveReservation,
    output_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if output_bytes > output_reservation.bytes() {
        return Err(join_output_upper_bound_violation_error(node_id, output_reservation.bytes(), output_bytes));
    }
    reservations.shrink_to_observed(output_reservation, output_bytes, metrics);
    Ok(())
}

fn join_output_upper_bound_violation_error(node_id: NodeId, upper_bound_bytes: u64, observed_bytes: u64) -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        "nexus_query_engine.bounded.join.output_upper_bound",
        "join output materialization exceeded its closed-form upper bound",
    )
    .with_fact("node_id", node_id)
    .with_fact("upper_bound_bytes", upper_bound_bytes)
    .with_fact("observed_bytes", observed_bytes)
}
