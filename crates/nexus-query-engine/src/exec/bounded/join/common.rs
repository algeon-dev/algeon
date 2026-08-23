use std::collections::BTreeMap;

use arrow::datatypes::DataType as ArrowDataType;

use super::{
    ActiveReservation, BoundedExecutionAttempt, DataFrame, GpuDataFrame, JoinNode, JoinOutputColumn,
    JoinOutputColumnUpperBoundInput, JoinSide, JoinType, MemoryReservationRequest, NodeId, QueryError, QueryErrorCode,
    ReservationKind, ResidentGpuFrame, Result, execution_budget_token_for_boundary, join_input_columns_with_order,
    join_output_upper_bound_bytes, join_side_projection_with_order, observe_device_memory_bytes, select_join_input,
};

pub(in crate::exec::bounded) struct BoundedJoinSideFrame {
    pub(in crate::exec::bounded) frame: DataFrame,
    pub(in crate::exec::bounded) _reservation: ActiveReservation,
    pub(in crate::exec::bounded) rows: u64,
    pub(in crate::exec::bounded) bytes: u64,
}

pub(in crate::exec::bounded) fn release_bounded_join_side_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    side: BoundedJoinSideFrame,
) {
    let BoundedJoinSideFrame { _reservation: reservation, .. } = side;
    execution.release(reservation);
}

enum JoinInputReservation<'a> {
    Shared { reservation: &'a ActiveReservation },
    Movable { reservation: Box<ActiveReservation> },
}

pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: ResidentGpuFrame,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(
        execution,
        input.frame,
        JoinInputReservation::Movable { reservation: Box::new(input.reservation) },
        columns,
        input_side,
        node_id,
        build_side,
    )
}
pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_shared_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: GpuDataFrame,
    source_reservation: &ActiveReservation,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(
        execution,
        input,
        JoinInputReservation::Shared { reservation: source_reservation },
        columns,
        input_side,
        node_id,
        build_side,
    )
}
fn prepare_bounded_join_side_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: GpuDataFrame,
    input_reservation: JoinInputReservation<'_>,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
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
            release_movable_join_input_reservation(execution, &mut movable_input_reservation);
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
            execution.try_reserve_work(
                MemoryReservationRequest::new(
                    ReservationKind::OperatorOutput,
                    input_reserved_bytes,
                    execution.reservations().budget(),
                )
                .with_memory_boundary("join_input_projection")
                .with_node_id(node_id)
                .with_stage("bounded_join")
                .with_attempt(1, 1),
            )?,
        )
    };
    let selected = match select_join_input(input_frame, &columns, join_side_label(input_side)) {
        Ok(frame) => frame,
        Err(error) => {
            if let Some(work_reservation) = work_reservation {
                execution.release(work_reservation);
            }
            release_movable_join_input_reservation(execution, &mut movable_input_reservation);
            return Err(error);
        },
    };
    let rows = selected.num_rows() as u64;
    let bytes = match observe_device_memory_bytes(&GpuDataFrame::new(selected.clone()), execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            if let Some(work_reservation) = work_reservation {
                execution.release(work_reservation);
            }
            release_movable_join_input_reservation(execution, &mut movable_input_reservation);
            return Err(error);
        },
    };
    let reservation = if select_memory_effect.is_alias_of_input() {
        if let Some(input_reservation) = movable_input_reservation.take() {
            let request = bounded_join_request(
                reservation_kind,
                memory_boundary,
                input_reserved_bytes,
                execution.reservations().budget(),
                node_id,
            );
            if input_reservation.is_alias() {
                execution.relabel_alias_reservation(input_reservation, request)
            } else {
                execution.transfer(input_reservation, request)
            }
        } else if let Some(source_reservation) = source_reservation {
            execution.alias_reservation(
                source_reservation,
                bounded_join_request(
                    reservation_kind,
                    memory_boundary,
                    bytes,
                    execution.reservations().budget(),
                    node_id,
                ),
            )
        } else {
            match execution.try_reserve(bounded_join_request(
                reservation_kind,
                memory_boundary,
                bytes,
                execution.reservations().budget(),
                node_id,
            )) {
                Ok(reservation) => reservation,
                Err(error) => {
                    if let Some(work_reservation) = work_reservation {
                        execution.release(work_reservation);
                    }
                    return Err(error);
                },
            }
        }
    } else {
        release_movable_join_input_reservation(execution, &mut movable_input_reservation);
        let mut work_reservation =
            work_reservation.take().expect("non-alias join input selection should reserve projection work");
        let output_request =
            bounded_join_request(reservation_kind, memory_boundary, bytes, execution.reservations().budget(), node_id);
        if bytes > work_reservation.bytes() {
            let delta = bytes.saturating_sub(work_reservation.bytes());
            if let Err(error) =
                execution.try_grow(&mut work_reservation, output_request.clone().with_requested_bytes(delta))
            {
                execution.release(work_reservation);
                return Err(error);
            }
        } else {
            execution.shrink_to_observed_with_request(&mut work_reservation, bytes, output_request.clone());
        }
        execution.transfer(work_reservation, output_request)
    };
    if let Some(work_reservation) = work_reservation {
        execution.release(work_reservation);
    }
    release_movable_join_input_reservation(execution, &mut movable_input_reservation);
    Ok(BoundedJoinSideFrame { frame: selected, _reservation: reservation, rows, bytes })
}

fn release_movable_join_input_reservation(
    execution: &mut BoundedExecutionAttempt<'_>,
    reservation: &mut Option<ActiveReservation>,
) {
    if let Some(reservation) = reservation.take() {
        execution.release(reservation);
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
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build_rows: u64,
    left_join_output_rows: u64,
) -> Result<Option<ActiveReservation>> {
    if node.spec.join_type != JoinType::Full {
        return Ok(None);
    }
    let bytes = full_join_size_preflight_workspace_reservation_bytes(node, build_rows, left_join_output_rows)?;
    if bytes == 0 {
        return Ok(None);
    }
    execution
        .try_reserve(bounded_join_request(
            ReservationKind::GatherMap,
            "join_full_size_preflight_workspace",
            bytes,
            execution.reservations().budget(),
            node_id,
        ))
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

pub(super) fn join_side_output_upper_bound_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_rows: u64,
    columns: &[String],
    side: JoinSide,
) -> Result<u64> {
    let bounds = join_side_output_upper_bound_bytes_for_row_counts(
        execution,
        node_id,
        frame,
        &[expected_output_rows],
        columns,
        side,
    )?;
    Ok(bounds.into_iter().next().unwrap_or(0))
}

pub(super) fn join_side_output_upper_bound_bytes_for_row_counts(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_row_counts: &[u64],
    columns: &[String],
    side: JoinSide,
) -> Result<Vec<u64>> {
    let mut columns = columns.to_vec();
    join_side_projection_with_order(frame, &mut columns);
    if columns.is_empty() {
        return Ok(vec![0; expected_output_row_counts.len()]);
    }

    let mut upper_bound_columns = Vec::with_capacity(columns.len());
    let mut column_cache: BTreeMap<String, JoinOutputColumnBound> = BTreeMap::new();
    for column in columns {
        let upper_bound = if let Some(upper_bound) = column_cache.get(column.as_str()) {
            upper_bound.clone()
        } else {
            let upper_bound =
                join_output_upper_bound_column_for_frame_column(execution, node_id, frame, column.as_str(), side)?;
            column_cache.insert(column.clone(), upper_bound.clone());
            upper_bound
        };
        upper_bound_columns.push(upper_bound);
    }
    expected_output_row_counts
        .iter()
        .map(|expected_output_rows| {
            join_output_columns_upper_bound_bytes(*expected_output_rows, &upper_bound_columns).map_err(|error| {
                error
                    .with_fact_if_absent("node_id", node_id)
                    .with_fact_if_absent("side", side)
                    .with_fact_if_absent("expected_output_rows", *expected_output_rows)
            })
        })
        .collect()
}

#[derive(Clone)]
enum JoinOutputColumnBound {
    Standard(JoinOutputColumnUpperBoundInput),
    List { input_column_bytes: u64 },
}

fn join_output_columns_upper_bound_bytes(expected_output_rows: u64, columns: &[JoinOutputColumnBound]) -> Result<u64> {
    columns.iter().try_fold(0u64, |total, column| {
        let column_bytes = match column {
            JoinOutputColumnBound::Standard(column) => {
                join_output_upper_bound_bytes(expected_output_rows, std::slice::from_ref(column))?
            },
            JoinOutputColumnBound::List { input_column_bytes } => {
                // A gather output row cannot retain more child payload than the entire input list column.
                let gathered = expected_output_rows.checked_mul(*input_column_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "nexus_query_engine.bounded.join.output_upper_bound",
                        "list join output byte bound overflowed",
                    )
                    .with_fact("reason", "list_output_bound_overflow")
                })?;
                let offsets =
                    expected_output_rows.checked_add(1).and_then(|rows| rows.checked_mul(4)).ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::SizeOverflow,
                            "nexus_query_engine.bounded.join.output_upper_bound",
                            "list join output offset bound overflowed",
                        )
                        .with_fact("reason", "list_output_offset_bound_overflow")
                    })?;
                let validity = if expected_output_rows == 0 {
                    0
                } else {
                    expected_output_rows.div_ceil(8).div_ceil(64).saturating_mul(64)
                };
                gathered.checked_add(offsets).and_then(|bytes| bytes.checked_add(validity)).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "nexus_query_engine.bounded.join.output_upper_bound",
                        "list join output structural byte bound overflowed",
                    )
                    .with_fact("reason", "list_output_structural_bound_overflow")
                })?
            },
        };
        total.checked_add(column_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::SizeOverflow,
                "nexus_query_engine.bounded.join.output_upper_bound",
                "join output column byte bounds overflowed",
            )
            .with_fact("reason", "join_output_bound_overflow")
        })
    })
}

pub(super) fn projected_output_columns_for_retained_sizing(
    projection: &[JoinOutputColumn],
    side: JoinSide,
) -> Vec<String> {
    projection.iter().filter(|column| column.side == side).map(|column| column.column.name().to_string()).collect()
}

fn join_output_upper_bound_column_for_frame_column(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
) -> Result<JoinOutputColumnBound> {
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
    if cudf_dtype.id() == cudf::TypeId::List {
        let input_column_bytes = u64::try_from(series.device_allocated_size_bytes().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.output_upper_bound",
                "failed to inspect list join output source bytes for closed-form sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column)
        })?)
        .map_err(|_| {
            QueryError::new(
                QueryErrorCode::SizeOverflow,
                "nexus_query_engine.bounded.join.output_upper_bound",
                "list join output source bytes exceed the reservation ledger range",
            )
            .with_fact("reason", "list_source_bytes_exceed_u64")
        })?;
        return Ok(JoinOutputColumnBound::List { input_column_bytes });
    }
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
        Some(max_string_byte_length_for_join_output_sizing(execution, node_id, frame, column, side)?)
    } else {
        None
    };
    Ok(JoinOutputColumnBound::Standard(JoinOutputColumnUpperBoundInput {
        data_type,
        input_max_string_length_bytes,
        input_total_string_bytes: None,
    }))
}

fn max_string_byte_length_for_join_output_sizing(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
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
    let mut reservation = execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::ChunkOperator,
            byte_count_bound,
            execution.reservations().budget(),
        )
        .with_memory_boundary("join_output_string_length_sizing")
        .with_node_id(node_id)
        .with_stage("join_output_string_length_sizing"),
    )?;
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&reservation),
        "join_output_string_length_sizing",
        "nexus_query_engine.bounded.join.output_string_length_sizing.budget_token",
        "failed to create bounded native join output string-length sizing budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(reservation);
            return Err(error.with_fact("node_id", node_id).with_fact("side", side).with_fact("column", column));
        },
    };

    let series = match frame.series(column) {
        Ok(series) => series,
        Err(error) => {
            execution.release(reservation);
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
            execution.release(reservation);
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
            execution.release(reservation);
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
        execution.release(reservation);
        return Err(error);
    }
    execution.shrink_to_observed(&mut reservation, observed);
    let max_len = match byte_counts.max_value_in::<i32>(&token.execution_overrides()) {
        Ok(max_len) => max_len,
        Err(error) => {
            execution.release(reservation);
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
    execution.release(reservation);
    Ok(max_len.unwrap_or(0).max(0) as u64)
}

fn join_output_dtype_needs_string_max(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

pub(super) fn reconcile_join_output_upper_bound(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    output_reservation: &mut ActiveReservation,
    output_bytes: u64,
) -> Result<()> {
    if output_bytes > output_reservation.bytes() {
        return Err(join_output_upper_bound_violation_error(node_id, output_reservation.bytes(), output_bytes));
    }
    execution.shrink_to_observed(output_reservation, output_bytes);
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
