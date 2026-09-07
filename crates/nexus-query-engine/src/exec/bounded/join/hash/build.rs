use super::{
    BoundedExecutionAttempt, BoundedRootHashJoinBuildState, DataFrame, ExecutionOverrides, GpuDataFrame,
    GpuHashJoinBuildHandle, Instant, JoinNode, JoinSide, ResidentGpuFrame, Result, cudf_query_error,
    execution_overrides, lower_null_equality, prepare_bounded_join_side_input_from_resident_frame,
    prepare_bounded_join_side_input_from_shared_frame, probe_side, projected_input_columns,
    state::validate_root_hash_join,
};

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_build_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    input: GpuDataFrame,
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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let build = prepare_bounded_join_side_input_from_shared_frame(execution, input, build_columns, build_side)?;
    let build_keys_unique = single_key_build_is_unique(&build.frame, build_keys, &overrides);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_bytes) =
        build_bounded_hash_join(execution, node, &build.frame, build_keys, hash_options, build_side, &overrides, true)?;
    Ok(BoundedRootHashJoinBuildState {
        build_side,
        probe_side: probe_side(build_side),
        left_keys,
        right_keys,
        left_columns,
        right_columns,
        build,
        hash_join,
        hash_table_bytes,
        left_indices_from_probe: build_side == JoinSide::Right,
        build_keys_unique,
        timer,
    })
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_build_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    input: ResidentGpuFrame,
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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let build = prepare_bounded_join_side_input_from_resident_frame(execution, input, build_columns, build_side)?;
    let build_keys_unique = single_key_build_is_unique(&build.frame, build_keys, &overrides);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_bytes) =
        build_bounded_hash_join(execution, node, &build.frame, build_keys, hash_options, build_side, &overrides, true)?;
    Ok(BoundedRootHashJoinBuildState {
        build_side,
        probe_side: probe_side(build_side),
        left_keys,
        right_keys,
        left_columns,
        right_columns,
        build,
        hash_join,
        hash_table_bytes,
        left_indices_from_probe: build_side == JoinSide::Right,
        build_keys_unique,
        timer,
    })
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

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded::join) fn build_bounded_hash_join(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: &DataFrame,
    build_keys: &[String],
    hash_options: cudf::DataFrameHashJoinOptions,
    build_side: JoinSide,
    overrides: &ExecutionOverrides<'_>,
    record_hash_join_metrics: bool,
) -> Result<(GpuHashJoinBuildHandle, u64)> {
    let hash_join = match build.with_execution(overrides).build_hash_join(build_keys, hash_options) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            return Err(cudf_query_error(
                "nexus_query_engine.bounded.join.build_hash_join",
                "failed to build bounded native reusable hash join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side));
        },
    };
    let hash_join = match GpuHashJoinBuildHandle::new(execution.context(), hash_join, node.spec.null_equality) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            return Err(error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", build_side));
        },
    };
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    if record_hash_join_metrics {
        execution.metrics_mut().record_hash_join_build();
        execution.metrics_mut().record_hash_join_hash_table_bytes(hash_table_bytes);
    }
    Ok((hash_join, hash_table_bytes))
}
