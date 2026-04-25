use super::*;

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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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
        "join_hash_table",
        true,
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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
        "join_hash_table",
        true,
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

pub(in crate::exec::bounded::join) fn single_key_build_is_unique(
    frame: &DataFrame,
    build_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> bool {
    let [key] = build_keys else {
        return false;
    };
    let Ok(series) = frame.series(key) else {
        return false;
    };
    series.nunique_in(overrides).ok().is_some_and(|unique_count| unique_count == frame.num_rows())
}

pub(in crate::exec::bounded::join) fn reserve_and_build_bounded_hash_join(
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
    hash_table_memory_boundary: &'static str,
    record_hash_join_metrics: bool,
) -> Result<(GpuHashJoinBuildHandle, ActiveReservation, u64)> {
    let hash_table_reservation_bytes = build
        .with_execution(overrides)
        .hash_join_pre_build_reservation_size_bytes(build_keys, hash_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.hash_table_reservation",
                "failed to size bounded native reusable hash join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            hash_table_memory_boundary,
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let hash_join = match build.with_execution(overrides).build_hash_join(build_keys, hash_options) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            reservations.release(hash_table_reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.build_hash_join",
                "failed to build bounded native reusable hash join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side));
        },
    };
    let hash_join = match GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            reservations.release(hash_table_reservation, metrics);
            return Err(error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", build_side));
        },
    };
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    if record_hash_join_metrics {
        metrics.record_hash_join_build();
        metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
    }
    Ok((hash_join, hash_table_reservation, hash_table_bytes))
}
