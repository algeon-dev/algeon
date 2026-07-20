use super::*;

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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let (probe, indices, gather_map_reservation, gather_map_bytes) =
        if reusable_hash_join_single_pass_inner_probe_supported(node, &build) {
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?
        } else {
            execute_bounded_root_hash_join_multi_pass_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?
        };
    let output_reservation_bounds = match root_hash_join_output_reservation_bounds(
        context,
        node_id,
        node,
        &build,
        &probe,
        &indices,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    Ok(BoundedRootHashJoinProbeState {
        build,
        probe,
        indices,
        gather_map_reservation,
        gather_map_bytes,
        output_reservation_bytes: output_reservation_bounds.output_reservation_bytes,
        minimum_output_reservation_bytes: output_reservation_bounds.minimum_output_reservation_bytes,
    })
}

pub(in crate::exec::bounded::join::hash) fn execute_bounded_root_hash_join_single_pass_inner_probe_indices(
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
            "nexus_query_engine.bounded.join.single_pass_probe",
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
                "nexus_query_engine.bounded.join.gather_map_reservation",
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
                "nexus_query_engine.bounded.join.gather_map_bytes",
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
            "nexus_query_engine.bounded.join.single_pass_probe",
            "single-pass reusable hash join probe exceeded its reserved gather-map upper bound",
        )
        .with_fact("node_id", node_id)
        .with_fact("reserved_bytes", gather_map_reservation_bytes)
        .with_fact("observed_bytes", gather_map_bytes)
        .with_fact("probe_rows", output_size_upper_bound));
    }
    Ok((probe, indices, gather_map_reservation, gather_map_bytes))
}

pub(in crate::exec::bounded::join::hash) fn execute_bounded_root_hash_join_multi_pass_probe_indices(
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_keys: &[String],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    overrides: &ExecutionOverrides<'_>,
) -> Result<(BoundedJoinSideFrame, cudf::DataFrameHashJoinPairIndices, ActiveReservation, u64)> {
    let left_join_output_rows =
        match root_hash_join_left_probe_size_for_full_size_preflight(node, build, &probe, probe_keys, overrides) {
            Ok(rows) => rows as u64,
            Err(error) => {
                release_bounded_join_side_frame(probe, reservations, metrics);
                return Err(error);
            },
        };
    let full_size_preflight_workspace = match reserve_full_join_size_preflight_workspace(
        node_id,
        node,
        build.build.rows,
        left_join_output_rows,
        reservations,
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    let output_size = match root_hash_join_multi_pass_probe_size(node, build, &probe, probe_keys, overrides) {
        Ok(output_size) => {
            if let Some(reservation) = full_size_preflight_workspace {
                reservations.release(reservation, metrics);
            }
            output_size
        },
        Err(error) => {
            if let Some(reservation) = full_size_preflight_workspace {
                reservations.release(reservation, metrics);
            }
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    let gather_map_reservation_bytes = match build
        .hash_join
        .hash_join()
        .probe_gather_map_reservation_size_bytes(output_size)
        .map(|bytes| bytes as u64)
    {
        Ok(bytes) => bytes,
        Err(error) => {
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.gather_map_reservation",
                "failed to size bounded native reusable hash join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build.build_side));
        },
    };
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
    let indices =
        match root_hash_join_multi_pass_probe_with_output_size(node, build, &probe, probe_keys, output_size, overrides)
        {
            Ok(indices) => indices,
            Err(error) => {
                reservations.release(gather_map_reservation, metrics);
                release_bounded_join_side_frame(probe, reservations, metrics);
                return Err(error);
            },
        };
    let gather_map_bytes = match indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.gather_map_bytes",
                "failed to observe bounded native join gather-map memory",
                error,
            ));
        },
    };
    Ok((probe, indices, gather_map_reservation, gather_map_bytes))
}

pub(in crate::exec::bounded::join::hash) fn root_hash_join_left_probe_size_for_full_size_preflight(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: &BoundedJoinSideFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<usize> {
    if node.spec.join_type != JoinType::Full {
        return Ok(0);
    }
    build
        .hash_join
        .hash_join()
        .left_probe_size_in(&probe.frame, probe_keys, overrides)
        .map_err(|error| join_index_error(error, node))
}

pub(in crate::exec::bounded::join::hash) fn root_hash_join_multi_pass_probe_size(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: &BoundedJoinSideFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<usize> {
    match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_size_in(&probe.frame, probe_keys, overrides),
        JoinType::Left => build.hash_join.hash_join().left_probe_size_in(&probe.frame, probe_keys, overrides),
        JoinType::Full => build.hash_join.hash_join().full_probe_size_in(&probe.frame, probe_keys, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))
}

pub(in crate::exec::bounded::join::hash) fn root_hash_join_multi_pass_probe_with_output_size(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: &BoundedJoinSideFrame,
    probe_keys: &[String],
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
) -> Result<cudf::DataFrameHashJoinPairIndices> {
    match node.spec.join_type {
        JoinType::Inner => build.hash_join.hash_join().inner_probe_with_output_size_in(
            &probe.frame,
            probe_keys,
            output_size,
            overrides,
        ),
        JoinType::Left => {
            build.hash_join.hash_join().left_probe_with_output_size_in(&probe.frame, probe_keys, output_size, overrides)
        },
        JoinType::Full => {
            build.hash_join.hash_join().full_probe_with_output_size_in(&probe.frame, probe_keys, output_size, overrides)
        },
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))
}

pub(in crate::exec::bounded::join::hash) fn reusable_hash_join_single_pass_inner_probe_supported(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
) -> bool {
    node.spec.join_type == JoinType::Inner
        && node.spec.keys.len() == 1
        && node.spec.residual_predicate.is_none()
        && build.build_keys_unique
}

pub(in crate::exec::bounded::join::hash) fn reusable_hash_join_execution_strategy(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
) -> &'static str {
    if reusable_hash_join_single_pass_inner_probe_supported(node, build) {
        "bounded_reusable_hash_join_single_pass_probe"
    } else {
        "bounded_reusable_hash_join"
    }
}

pub(in crate::exec::bounded::join::hash) struct RootHashJoinOutputReservationBounds {
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: u64,
    pub(in crate::exec::bounded::join::hash) minimum_output_reservation_bytes: u64,
}

pub(in crate::exec::bounded::join::hash) fn root_hash_join_output_reservation_bounds(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: &BoundedJoinSideFrame,
    indices: &cudf::DataFrameHashJoinPairIndices,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RootHashJoinOutputReservationBounds> {
    let (left, right) = match build.build_side {
        JoinSide::Left => (&build.build, probe),
        JoinSide::Right => (probe, &build.build),
    };
    let expected_output_rows = indices.build().len() as u64;
    let row_counts = if expected_output_rows == 0 { vec![0] } else { vec![expected_output_rows, 1] };
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes = join_side_output_upper_bound_bytes_for_row_counts(
        context,
        node_id,
        &left.frame,
        &row_counts,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    )?;
    let right_output_bytes = join_side_output_upper_bound_bytes_for_row_counts(
        context,
        node_id,
        &right.frame,
        &row_counts,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let output_reservation_bytes = left_output_bytes[0].saturating_add(right_output_bytes[0]);
    let minimum_output_reservation_bytes =
        if expected_output_rows == 0 { 0 } else { left_output_bytes[1].saturating_add(right_output_bytes[1]) };
    Ok(RootHashJoinOutputReservationBounds { output_reservation_bytes, minimum_output_reservation_bytes })
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let (probe, indices, gather_map_reservation, gather_map_bytes) =
        if reusable_hash_join_single_pass_inner_probe_supported(node, &build) {
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?
        } else {
            execute_bounded_root_hash_join_multi_pass_probe_indices(
                node_id,
                node,
                &build,
                probe,
                probe_keys,
                reservations,
                metrics,
                &overrides,
            )?
        };
    let output_reservation_bounds = match root_hash_join_output_reservation_bounds(
        context,
        node_id,
        node,
        &build,
        &probe,
        &indices,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            release_bounded_join_side_frame(probe, reservations, metrics);
            return Err(error);
        },
    };
    Ok(BoundedRootHashJoinProbeState {
        build,
        probe,
        indices,
        gather_map_reservation,
        gather_map_bytes,
        output_reservation_bytes: output_reservation_bounds.output_reservation_bytes,
        minimum_output_reservation_bytes: output_reservation_bounds.minimum_output_reservation_bytes,
    })
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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

    let (probe, indices, gather_map_reservation, gather_map_bytes) =
        execute_bounded_root_hash_join_multi_pass_probe_indices(
            node_id,
            node,
            build,
            probe,
            probe_keys,
            reservations,
            metrics,
            &overrides,
        )?;

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
