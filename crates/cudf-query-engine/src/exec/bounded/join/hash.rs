use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_reservation_bytes,
        prepare_bounded_join_side_input_from_resident_frame, prepare_bounded_join_side_input_from_shared_frame,
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

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_chunk_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
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

    materialize_bounded_root_hash_join_probe_chunk(
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
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
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
    let left_output_bytes = match join_side_output_reservation_bytes(
        &left.frame,
        &left_indices,
        &left_projection,
        &overrides,
        JoinSide::Left,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            super::common::trace_preflight_failed(
                "execute_bounded_root_hash_join_materialize:left",
                node_id,
                reservations,
                &error,
            );
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_reservation_bytes(
        &right.frame,
        &right_indices,
        &right_projection,
        &overrides,
        JoinSide::Right,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            super::common::trace_preflight_failed(
                "execute_bounded_root_hash_join_materialize:right",
                node_id,
                reservations,
                &error,
            );
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
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    )?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_hash_join".to_string(),
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
