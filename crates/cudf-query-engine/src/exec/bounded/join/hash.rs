use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_upper_bound_bytes,
        prepare_bounded_join_side_input_from_resident_frame, prepare_bounded_join_side_input_from_shared_frame,
        projected_output_columns_for_retained_sizing, reconcile_join_output_upper_bound,
        release_bounded_join_side_frame,
    },
    *,
};
use crate::capability::keyed_hash_join_capable;

pub(in crate::exec::bounded) struct BoundedRootHashJoinBuildState {
    build_side: JoinSide,
    probe_side: JoinSide,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_columns: Vec<String>,
    right_columns: Vec<String>,
    build: BoundedJoinSideFrame,
    hash_join: GpuHashJoinBuildHandle,
    hash_table_reservation: ActiveReservation,
    hash_table_bytes: u64,
    left_indices_from_probe: bool,
    build_keys_unique: bool,
    timer: Instant,
}

pub(in crate::exec::bounded) struct BoundedRootHashJoinProbeState {
    build: BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_reservation: ActiveReservation,
    gather_map_bytes: u64,
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_build_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedRootHashJoinBuildState> {
    validate_root_hash_join(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build_side = node.spec.build_side;
    let build_columns = match build_side {
        JoinSide::Left => &left_columns,
        JoinSide::Right => &right_columns,
    };
    let build_keys = match build_side {
        JoinSide::Left => &left_keys,
        JoinSide::Right => &right_keys,
    };
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let build = prepare_bounded_join_side_input_from_shared_frame(
        input,
        input_reservation,
        build_columns,
        build_side,
        node_id,
        build_side,
        reservations,
        metrics,
    )?;
    let build_keys_unique = single_key_build_is_unique(&build.frame, build_keys, &overrides);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
        context,
        node_id,
        node,
        &build.frame,
        build_keys,
        hash_options,
        build_side,
        reservations,
        metrics,
        &overrides,
    )?;
    Ok(BoundedRootHashJoinBuildState {
        build_side,
        probe_side: probe_side(build_side),
        left_keys,
        right_keys,
        left_columns,
        right_columns,
        build,
        hash_join,
        hash_table_reservation,
        hash_table_bytes,
        left_indices_from_probe: build_side == JoinSide::Right,
        build_keys_unique,
        timer,
    })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_build_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedRootHashJoinBuildState> {
    validate_root_hash_join(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build_side = node.spec.build_side;
    let build_columns = match build_side {
        JoinSide::Left => &left_columns,
        JoinSide::Right => &right_columns,
    };
    let build_keys = match build_side {
        JoinSide::Left => &left_keys,
        JoinSide::Right => &right_keys,
    };
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let build = prepare_bounded_join_side_input_from_resident_frame(
        input,
        build_columns,
        build_side,
        node_id,
        build_side,
        reservations,
        metrics,
    )?;
    let build_keys_unique = single_key_build_is_unique(&build.frame, build_keys, &overrides);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
        context,
        node_id,
        node,
        &build.frame,
        build_keys,
        hash_options,
        build_side,
        reservations,
        metrics,
        &overrides,
    )?;
    Ok(BoundedRootHashJoinBuildState {
        build_side,
        probe_side: probe_side(build_side),
        left_keys,
        right_keys,
        left_columns,
        right_columns,
        build,
        hash_join,
        hash_table_reservation,
        hash_table_bytes,
        left_indices_from_probe: build_side == JoinSide::Right,
        build_keys_unique,
        timer,
    })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedRootHashJoinBuildState,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedRootHashJoinProbeState> {
    validate_root_hash_join(node)?;
    let probe_columns = match build.probe_side {
        JoinSide::Left => &build.left_columns,
        JoinSide::Right => &build.right_columns,
    };
    let probe = prepare_bounded_join_side_input_from_shared_frame(
        input,
        input_reservation,
        probe_columns,
        build.probe_side,
        node_id,
        build.build_side,
        reservations,
        metrics,
    )?;
    let probe_keys = match build.probe_side {
        JoinSide::Left => &build.left_keys,
        JoinSide::Right => &build.right_keys,
    };
    let overrides = execution_overrides(context);
    if reusable_hash_join_single_pass_inner_probe_supported(node, &build) {
        let (probe, indices, gather_map_reservation, gather_map_bytes) =
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?;
        return Ok(BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes });
    }

    let output_size_result = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Left => build.hash_join.hash_join().left_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Full => build.hash_join.hash_join().full_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    };
    let output_size = match output_size_result {
        Ok(output_size) => output_size,
        Err(error) => {
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(join_index_error(error, node));
        },
    };
    let gather_map_reservation_bytes = build
        .hash_join
        .hash_join()
        .probe_gather_map_reservation_size_bytes(output_size)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_reservation",
                "failed to size bounded native reusable hash join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build.build_side)
        })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    let indices_result = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Left => build.hash_join.hash_join().left_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Full => build.hash_join.hash_join().full_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    };
    let indices = match indices_result {
        Ok(indices) => indices,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(join_index_error(error, node));
        },
    };
    let gather_map_bytes = match indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_bytes",
                "failed to observe bounded native join gather-map memory",
                error,
            ));
        },
    };
    Ok(BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes })
}

fn execute_bounded_root_hash_join_single_pass_inner_probe_indices(
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_keys: &[String],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    overrides: &ExecutionOverrides<'_>,
) -> Result<(BoundedJoinSideFrame, cudf::DataFrameHashJoinPairIndices, ActiveReservation, u64)> {
    let output_size_upper_bound = usize::try_from(probe.rows).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.single_pass_probe",
            "single-pass reusable hash join probe row upper bound exceeds platform usize range",
        )
        .with_fact("node_id", node_id)
        .with_fact("probe_rows", probe.rows)
    })?;
    let gather_map_reservation_bytes = build
        .hash_join
        .hash_join()
        .probe_gather_map_reservation_size_bytes(output_size_upper_bound)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_reservation",
                "failed to size bounded native reusable hash join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build.build_side)
            .with_fact("single_pass_probe", true)
        })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    let indices = match build.hash_join.hash_join().inner_probe_in(&probe.frame, probe_keys, overrides) {
        Ok(indices) => indices,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(join_index_error(error, node));
        },
    };
    let gather_map_bytes = match indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_bytes",
                "failed to observe bounded native join gather-map memory",
                error,
            )
            .with_fact("single_pass_probe", true));
        },
    };
    if gather_map_bytes > gather_map_reservation.bytes() {
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        release_bounded_join_side_frame(probe, reservations, metrics);
        return Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "cudf_query_engine.bounded.join.single_pass_probe",
            "single-pass reusable hash join probe exceeded its reserved gather-map upper bound",
        )
        .with_fact("node_id", node_id)
        .with_fact("reserved_bytes", gather_map_reservation_bytes)
        .with_fact("observed_bytes", gather_map_bytes)
        .with_fact("probe_rows", output_size_upper_bound));
    }
    Ok((probe, indices, gather_map_reservation, gather_map_bytes))
}

fn reusable_hash_join_single_pass_inner_probe_supported(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
) -> bool {
    node.spec.join_type == JoinType::Inner
        && node.spec.keys.len() == 1
        && node.spec.residual_predicate.is_none()
        && build.build_keys_unique
}

fn reusable_hash_join_execution_strategy(node: &JoinNode, build: &BoundedRootHashJoinBuildState) -> &'static str {
    if reusable_hash_join_single_pass_inner_probe_supported(node, build) {
        "bounded_reusable_hash_join_single_pass_probe"
    } else {
        "bounded_reusable_hash_join"
    }
}

fn single_key_build_is_unique(frame: &DataFrame, build_keys: &[String], overrides: &ExecutionOverrides<'_>) -> bool {
    let [key] = build_keys else {
        return false;
    };
    let Ok(series) = frame.series(key) else {
        return false;
    };
    series.nunique_in(overrides).ok().is_some_and(|unique_count| unique_count == frame.num_rows())
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedRootHashJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedRootHashJoinProbeState> {
    validate_root_hash_join(node)?;
    let probe_columns = match build.probe_side {
        JoinSide::Left => &build.left_columns,
        JoinSide::Right => &build.right_columns,
    };
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        input,
        probe_columns,
        build.probe_side,
        node_id,
        build.build_side,
        reservations,
        metrics,
    )?;
    let probe_keys = match build.probe_side {
        JoinSide::Left => &build.left_keys,
        JoinSide::Right => &build.right_keys,
    };
    let overrides = execution_overrides(context);
    if reusable_hash_join_single_pass_inner_probe_supported(node, &build) {
        let (probe, indices, gather_map_reservation, gather_map_bytes) =
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?;
        return Ok(BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes });
    }

    let output_size = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Left => build.hash_join.hash_join().left_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Full => build.hash_join.hash_join().full_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_reservation_bytes = build
        .hash_join
        .hash_join()
        .probe_gather_map_reservation_size_bytes(output_size)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_reservation",
                "failed to size bounded native reusable hash join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build.build_side)
        })?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Left => build.hash_join.hash_join().left_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Full => build.hash_join.hash_join().full_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_bytes",
            "failed to observe bounded native join gather-map memory",
            error,
        )
    })?;
    Ok(BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ResidentGpuFrame>> {
    validate_root_hash_join(node)?;
    let probe_columns = match build.probe_side {
        JoinSide::Left => &build.left_columns,
        JoinSide::Right => &build.right_columns,
    };
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        input,
        probe_columns,
        build.probe_side,
        node_id,
        build.build_side,
        reservations,
        metrics,
    )?;
    let probe_keys = match build.probe_side {
        JoinSide::Left => &build.left_keys,
        JoinSide::Right => &build.right_keys,
    };
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    if reusable_hash_join_single_pass_inner_probe_supported(node, build) {
        let (probe, indices, gather_map_reservation, gather_map_bytes) =
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(
                node_id,
                node,
                build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?;
        return materialize_bounded_root_hash_join_probe_stream(
            context,
            node_id,
            node,
            build,
            probe,
            indices,
            gather_map_reservation,
            gather_map_bytes,
            timer,
            reservations,
            metrics,
        );
    }

    let output_size = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Left => build.hash_join.hash_join().left_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::Full => build.hash_join.hash_join().full_probe_size_in(&probe.frame, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_reservation_bytes = build
        .hash_join
        .hash_join()
        .probe_gather_map_reservation_size_bytes(output_size)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.gather_map_reservation",
                "failed to size bounded native reusable hash join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build.build_side)
        })?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Left => build.hash_join.hash_join().left_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::Full => build.hash_join.hash_join().full_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            &overrides,
        ),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_bytes",
            "failed to observe bounded native join gather-map memory",
            error,
        )
    })?;

    materialize_bounded_root_hash_join_probe_stream(
        context,
        node_id,
        node,
        build,
        probe,
        indices,
        gather_map_reservation,
        gather_map_bytes,
        timer,
        reservations,
        metrics,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedRootHashJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_root_hash_join(node)?;
    let timer = state.build.timer;
    let BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes } = state;
    let output = materialize_bounded_root_hash_join_probe_chunk(
        context,
        node_id,
        node,
        &build,
        probe,
        indices,
        gather_map_reservation,
        gather_map_bytes,
        timer,
        reservations,
        metrics,
    )?;
    release_bounded_root_hash_join_build_state(build, reservations, metrics);
    Ok(output)
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedRootHashJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ResidentGpuFrame>> {
    validate_root_hash_join(node)?;
    let timer = state.build.timer;
    let BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, gather_map_bytes } = state;
    match materialize_bounded_root_hash_join_probe_stream(
        context,
        node_id,
        node,
        &build,
        probe,
        indices,
        gather_map_reservation,
        gather_map_bytes,
        timer,
        reservations,
        metrics,
    ) {
        Ok(outputs) => {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
            Ok(outputs)
        },
        Err(error) => {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
            Err(error)
        },
    }
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_build_state(
    build: BoundedRootHashJoinBuildState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedRootHashJoinBuildState { build: build_frame, hash_join, hash_table_reservation, .. } = build;
    drop(hash_join);
    reservations.release(hash_table_reservation, metrics);
    release_bounded_join_side_frame(build_frame, reservations, metrics);
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_probe_state(
    probe: BoundedRootHashJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedRootHashJoinProbeState { build, probe, indices, gather_map_reservation, .. } = probe;
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    release_bounded_root_hash_join_build_state(build, reservations, metrics);
    release_bounded_join_side_frame(probe, reservations, metrics);
}
fn materialize_bounded_root_hash_join_probe_chunk(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_reservation: ActiveReservation,
    gather_map_bytes: u64,
    timer: Instant,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let overrides = execution_overrides(context);
    let build_side = build.build_side;
    let probe_side = build.probe_side;
    let hash_table_bytes = build.hash_table_bytes;
    let (left, right) = match build_side {
        JoinSide::Left => (&build.build, &probe),
        JoinSide::Right => (&probe, &build.build),
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);
    let (left_indices, right_indices) = if build.left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    super::common::trace_preflight_entry(
        "execute_bounded_root_hash_join_materialize",
        "root_hash_join",
        node_id,
        reservations,
        0,
        indices.build().len() as u64,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let expected_output_rows = indices.build().len() as u64;
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            super::common::trace_preflight_failed(
                "execute_bounded_root_hash_join_materialize:left",
                node_id,
                reservations,
                &error,
            );
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            super::common::trace_preflight_failed(
                "execute_bounded_root_hash_join_materialize:right",
                node_id,
                reservations,
                &error,
            );
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    super::common::trace_preflight_success(
        "execute_bounded_root_hash_join_materialize",
        node_id,
        reservations,
        left_output_bytes.saturating_add(right_output_bytes),
    );
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let mut output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let output = match materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
        Ok(output) => output,
        Err(error) => {
            reservations.release(output_reservation, metrics);
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            reservations.release(output_reservation, metrics);
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        drop(left_indices);
        drop(right_indices);
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        drop(probe);
        return Err(error);
    }
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: reusable_hash_join_execution_strategy(node, build).to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));

    // `left_indices`/`right_indices` are `Series::clone`s of `indices.probe()` /
    // `indices.build()`; the underlying columns are Arc-shared, so device memory
    // is only released once every Arc holder drops. All three must be released
    // before `reservations.release(gather_map_reservation)` for the budget to
    // reflect actual residency.
    drop(left_indices);
    drop(right_indices);
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(probe);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn materialize_bounded_root_hash_join_probe_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_reservation: ActiveReservation,
    gather_map_bytes: u64,
    timer: Instant,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ResidentGpuFrame>> {
    let overrides = execution_overrides(context);
    let build_side = build.build_side;
    let probe_side = build.probe_side;
    let hash_table_bytes = build.hash_table_bytes;
    let (left, right) = match build_side {
        JoinSide::Left => (&build.build, &probe),
        JoinSide::Right => (&probe, &build.build),
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);
    let (left_indices, right_indices) = if build.left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let output_rows = indices.build().len() as u64;
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &left.frame,
        output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(left_indices);
            drop(right_indices);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let mut outputs = Vec::new();
    let mut cumulative_output_reservation_bytes = 0u64;
    let mut cumulative_output_bytes = 0u64;
    let mut emitted_rows = 0u64;
    let mut offset = 0usize;
    let output_len = indices.build().len();
    if output_len == 0 {
        match materialize_bounded_root_hash_join_index_slice(
            context,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            0,
            0,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        cumulative_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    outputs.push(output.frame);
                }
            },
            Err(error) => {
                drop(left_indices);
                drop(right_indices);
                drop(indices);
                reservations.release(gather_map_reservation, metrics);
                drop(probe);
                return Err(error);
            },
        }
    }
    while offset < output_len {
        let remaining_rows = u64::try_from(output_len - offset).unwrap_or(u64::MAX);
        let remaining_output_reservation_bytes =
            proportional_join_stream_bytes(output_reservation_bytes, remaining_rows, output_rows);
        let chunk_target_bytes = reusable_hash_join_stream_target_output_bytes(context, reservations);
        let chunk_rows = reusable_hash_join_stream_chunk_rows(
            remaining_rows,
            remaining_output_reservation_bytes,
            chunk_target_bytes,
        );
        let take = chunk_rows.min(output_len - offset);
        match materialize_bounded_root_hash_join_index_slice(
            context,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            offset,
            take,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        cumulative_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    emitted_rows = emitted_rows.saturating_add(output.output_rows);
                    outputs.push(output.frame);
                }
                offset += take;
            },
            Err(error) => {
                release_root_hash_join_stream_outputs(outputs, reservations, metrics);
                drop(left_indices);
                drop(right_indices);
                drop(indices);
                reservations.release(gather_map_reservation, metrics);
                drop(probe);
                return Err(error);
            },
        }
    }
    metrics.operator_time += timer.elapsed();
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: reusable_hash_join_execution_strategy(node, build).to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes: cumulative_output_reservation_bytes,
            output_rows: emitted_rows,
            output_bytes: cumulative_output_bytes,
        },
    ));
    drop(left_indices);
    drop(right_indices);
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(probe);
    Ok(outputs)
}

struct RootHashJoinStreamChunkOutput {
    frame: ResidentGpuFrame,
    output_reservation_bytes: u64,
    output_bytes: u64,
    output_rows: u64,
}

fn materialize_bounded_root_hash_join_index_slice(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    left_indices: &cudf::DataFrameJoinIndices,
    right_indices: &cudf::DataFrameJoinIndices,
    offset: usize,
    len: usize,
    overrides: &ExecutionOverrides<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<RootHashJoinStreamChunkOutput>> {
    match materialize_bounded_root_hash_join_index_slice_once(
        context,
        node_id,
        node,
        left,
        right,
        left_indices,
        right_indices,
        offset,
        len,
        overrides,
        reservations,
        metrics,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if len > 1 && is_retryable_join_stream_slice_error(&error) => {
            record_join_stream_slice_retry_error(&error, metrics);
            let left_len = len / 2;
            let right_len = len - left_len;
            let mut outputs = materialize_bounded_root_hash_join_index_slice(
                context,
                node_id,
                node,
                left,
                right,
                left_indices,
                right_indices,
                offset,
                left_len,
                overrides,
                reservations,
                metrics,
            )?;
            match materialize_bounded_root_hash_join_index_slice(
                context,
                node_id,
                node,
                left,
                right,
                left_indices,
                right_indices,
                offset + left_len,
                right_len,
                overrides,
                reservations,
                metrics,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_root_hash_join_stream_outputs(
                        outputs.into_iter().map(|output| output.frame).collect(),
                        reservations,
                        metrics,
                    );
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn materialize_bounded_root_hash_join_index_slice_once(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    left_indices: &cudf::DataFrameJoinIndices,
    right_indices: &cudf::DataFrameJoinIndices,
    offset: usize,
    len: usize,
    overrides: &ExecutionOverrides<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RootHashJoinStreamChunkOutput> {
    let slice_gather_map_bytes = cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(len)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.reusable_hash_stream_slice_reservation",
                "failed to size reusable hash join stream gather-map slices",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
    let slice_gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map_stream_slice",
            slice_gather_map_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let left_slice = match left_indices.indices().slice(offset, len) {
        Ok(slice) => cudf::DataFrameJoinIndices::from_series(slice),
        Err(error) => {
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.reusable_hash_stream_left_indices",
                "failed to slice reusable hash join left gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };
    let right_slice = match right_indices.indices().slice(offset, len) {
        Ok(slice) => cudf::DataFrameJoinIndices::from_series(slice),
        Err(error) => {
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.reusable_hash_stream_right_indices",
                "failed to slice reusable hash join right gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };
    let expected_output_rows = len as u64;
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(error);
        },
    };
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let mut output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output_stream_chunk",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = match materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_slice,
        &right_slice,
        overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
        Ok(output) => output,
        Err(error) => {
            reservations.release(output_reservation, metrics);
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(error);
        },
    };
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            reservations.release(output_reservation, metrics);
            reservations.release(slice_gather_map_reservation, metrics);
            return Err(error);
        },
    };
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        reservations.release(output_reservation, metrics);
        reservations.release(slice_gather_map_reservation, metrics);
        return Err(error);
    }
    reservations.release(slice_gather_map_reservation, metrics);
    Ok(RootHashJoinStreamChunkOutput {
        frame: ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation },
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

fn is_retryable_join_stream_slice_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_join_stream_slice_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn reusable_hash_join_stream_chunk_rows(output_rows: u64, output_reservation_bytes: u64, target_bytes: u64) -> usize {
    if output_rows == 0 || output_reservation_bytes <= target_bytes {
        return usize::try_from(output_rows.max(1)).unwrap_or(usize::MAX);
    }
    let rows = target_bytes.saturating_mul(output_rows).checked_div(output_reservation_bytes).unwrap_or(1).max(1);
    usize::try_from(rows).unwrap_or(usize::MAX)
}

fn proportional_join_stream_bytes(total_bytes: u64, rows: u64, total_rows: u64) -> u64 {
    if total_rows == 0 {
        return total_bytes;
    }
    ((u128::from(total_bytes) * u128::from(rows)).div_ceil(u128::from(total_rows))).min(u128::from(u64::MAX)) as u64
}

fn reusable_hash_join_stream_target_output_bytes(context: &ExecutionContext, reservations: &ReservationTracker) -> u64 {
    const MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    const MAX_TARGET_BYTES: u64 = 512 * 1024 * 1024;
    let policy_target = match context.memory_control_policy() {
        crate::runtime::MemoryControlPolicy::Bounded { device_budget_bytes, .. } => {
            (device_budget_bytes / 8).clamp(MIN_TARGET_BYTES, MAX_TARGET_BYTES)
        },
        _ => MAX_TARGET_BYTES,
    };
    let live_target = reservations.remaining_bytes().saturating_div(16);
    if live_target == 0 { policy_target } else { policy_target.min(live_target) }
}

fn release_root_hash_join_stream_outputs(
    mut outputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    while let Some(output) = outputs.pop() {
        reservations.release(output.reservation, metrics);
        drop(output.frame);
    }
}

fn validate_root_hash_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)?;
    if !keyed_hash_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled hash join execution requires keyed inner, left, or full equi join capability",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }
    Ok(())
}
fn reserve_and_build_bounded_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &DataFrame,
    build_keys: &[String],
    hash_options: cudf::DataFrameHashJoinOptions,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    overrides: &ExecutionOverrides<'_>,
) -> Result<(GpuHashJoinBuildHandle, ActiveReservation, u64)> {
    let hash_table_reservation_bytes = build
        .with_execution(overrides)
        .hash_join_pre_build_reservation_size_bytes(build_keys, hash_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.hash_table_reservation",
                "failed to size bounded native reusable hash join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            "join_hash_table",
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let hash_join = build.with_execution(overrides).build_hash_join(build_keys, hash_options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.build_hash_join",
            "failed to build bounded native reusable hash join",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("build_side", build_side)
    })?;
    let hash_join = GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality).map_err(|error| {
        error
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side)
    })?;
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
    Ok((hash_join, hash_table_reservation, hash_table_bytes))
}
