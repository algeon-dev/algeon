use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState,
    ExecutionOverrides, Instant, JoinNode, JoinSide, JoinType, NodeId, ResidentGpuFrame, Result, cudf_query_error,
    execution_overrides, join_index_error, join_side_output_upper_bound_bytes_for_row_counts,
    prepare_bounded_join_side_input_from_resident_frame, projected_output_columns_for_retained_sizing,
    release_bounded_join_side_frame, state::validate_root_hash_join,
    stream::materialize_bounded_root_hash_join_probe_stream, sum_bounds,
};

pub(in crate::exec::bounded::join::hash) fn execute_bounded_root_hash_join_single_pass_inner_probe_indices(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<(BoundedJoinSideFrame, cudf::DataFrameHashJoinPairIndices, u64)> {
    let indices = match build.hash_join.hash_join().inner_probe_in(&probe.frame, probe_keys, overrides) {
        Ok(indices) => indices,
        Err(error) => {
            release_bounded_join_side_frame(probe);
            return Err(join_index_error(error, node));
        },
    };
    let gather_map_bytes = match indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_bounded_join_side_frame(probe);
            return Err(cudf_query_error(
                "query_engine.bounded.join.gather_map_bytes",
                "failed to observe bounded native join gather-map memory",
                error,
            )
            .with_fact("single_pass_probe", true));
        },
    };
    Ok((probe, indices, gather_map_bytes))
}

pub(in crate::exec::bounded::join::hash) fn execute_bounded_root_hash_join_multi_pass_probe_indices(
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<(BoundedJoinSideFrame, cudf::DataFrameHashJoinPairIndices, u64)> {
    let output_size = match root_hash_join_multi_pass_probe_size(node, build, &probe, probe_keys, overrides) {
        Ok(output_size) => output_size,
        Err(error) => {
            release_bounded_join_side_frame(probe);
            return Err(error);
        },
    };
    let indices =
        match root_hash_join_multi_pass_probe_with_output_size(node, build, &probe, probe_keys, output_size, overrides)
        {
            Ok(indices) => indices,
            Err(error) => {
                release_bounded_join_side_frame(probe);
                return Err(error);
            },
        };
    let gather_map_bytes = match indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_bounded_join_side_frame(probe);
            return Err(cudf_query_error(
                "query_engine.bounded.join.gather_map_bytes",
                "failed to observe bounded native join gather-map memory",
                error,
            ));
        },
    };
    Ok((probe, indices, gather_map_bytes))
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
        JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } | JoinType::Cross => {
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
        JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } | JoinType::Cross => {
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
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: Option<u64>,
    pub(in crate::exec::bounded::join::hash) minimum_output_reservation_bytes: Option<u64>,
}

pub(in crate::exec::bounded::join::hash) fn root_hash_join_output_reservation_bounds(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: &BoundedJoinSideFrame,
    indices: &cudf::DataFrameHashJoinPairIndices,
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
        execution,
        node_id,
        &left.frame,
        &row_counts,
        &left_projection,
        JoinSide::Left,
    )?;
    let right_output_bytes = join_side_output_upper_bound_bytes_for_row_counts(
        execution,
        node_id,
        &right.frame,
        &row_counts,
        &right_projection,
        JoinSide::Right,
    )?;
    let output_reservation_bytes = sum_bounds(left_output_bytes[0], right_output_bytes[0]);
    let minimum_output_reservation_bytes =
        if expected_output_rows == 0 { Some(0) } else { sum_bounds(left_output_bytes[1], right_output_bytes[1]) };
    Ok(RootHashJoinOutputReservationBounds { output_reservation_bytes, minimum_output_reservation_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedRootHashJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<BoundedRootHashJoinProbeState> {
    validate_root_hash_join(node)?;
    let probe_columns = match build.probe_side {
        JoinSide::Left => &build.left_columns,
        JoinSide::Right => &build.right_columns,
    };
    let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, probe_columns, build.probe_side)?;
    let probe_keys = match build.probe_side {
        JoinSide::Left => &build.left_keys,
        JoinSide::Right => &build.right_keys,
    };
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let (probe, indices, gather_map_bytes) = if reusable_hash_join_single_pass_inner_probe_supported(node, &build) {
        execute_bounded_root_hash_join_single_pass_inner_probe_indices(node, &build, probe, probe_keys, &overrides)?
    } else {
        execute_bounded_root_hash_join_multi_pass_probe_indices(node, &build, probe, probe_keys, &overrides)?
    };
    let output_reservation_bounds =
        match root_hash_join_output_reservation_bounds(execution, node_id, node, &build, &probe, &indices) {
            Ok(bytes) => bytes,
            Err(error) => {
                release_bounded_join_side_frame(probe);
                return Err(error);
            },
        };
    Ok(BoundedRootHashJoinProbeState {
        build,
        probe,
        indices,
        gather_map_bytes,
        output_reservation_bytes: output_reservation_bounds.output_reservation_bytes,
        minimum_output_reservation_bytes: output_reservation_bounds.minimum_output_reservation_bytes,
    })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<Vec<ResidentGpuFrame>> {
    validate_root_hash_join(node)?;
    let probe_columns = match build.probe_side {
        JoinSide::Left => &build.left_columns,
        JoinSide::Right => &build.right_columns,
    };
    let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, probe_columns, build.probe_side)?;
    let probe_keys = match build.probe_side {
        JoinSide::Left => &build.left_keys,
        JoinSide::Right => &build.right_keys,
    };
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    if reusable_hash_join_single_pass_inner_probe_supported(node, build) {
        let (probe, indices, gather_map_bytes) =
            execute_bounded_root_hash_join_single_pass_inner_probe_indices(node, build, probe, probe_keys, &overrides)?;
        return materialize_bounded_root_hash_join_probe_stream(
            execution,
            node_id,
            node,
            build,
            probe,
            indices,
            gather_map_bytes,
            timer,
        );
    }

    let (probe, indices, gather_map_bytes) =
        execute_bounded_root_hash_join_multi_pass_probe_indices(node, build, probe, probe_keys, &overrides)?;

    materialize_bounded_root_hash_join_probe_stream(
        execution,
        node_id,
        node,
        build,
        probe,
        indices,
        gather_map_bytes,
        timer,
    )
}
