use super::{
    BoundedExecutionAttempt, BoundedRootHashJoinBuildState, DataFrame, Exec, GpuHashJoinBuildHandle, Instant, JoinNode,
    JoinSide, ResidentGpuFrame, Result, cudf_query_error, lower_null_equality,
    prepare_bounded_join_side_input_from_resident_frame, probe_side, projected_input_columns,
    state::validate_root_hash_join,
};

/// Narrows `input` to the build columns and builds the reusable hash table over its keys.
///
/// Consumes `input`. The returned state owns the build frame and the table; on error the
/// input and any partial build are dropped; a failed call leaves nothing for the caller to
/// release. The join type and build side are checked by `validate_root_hash_join`,
/// which accepts only the node shapes the root hash join can execute.
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
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let build = prepare_bounded_join_side_input_from_resident_frame(execution, input, build_columns, build_side)?;
    let build_keys_unique = single_key_build_is_unique(&build.frame, build_keys, overrides);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_bytes) =
        build_bounded_hash_join(execution, node, &build.frame, build_keys, hash_options, build_side, overrides, true)?;
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

/// Whether a one-column build key has no duplicate values, by comparing the distinct count
/// with the row count. Any other key shape, a missing column or a failed reduction reports
/// `false`, and the multi-pass probe is used; it is correct for any key distribution.
fn single_key_build_is_unique(frame: &DataFrame, build_keys: &[String], overrides: Exec<'_>) -> bool {
    let [key] = build_keys else {
        return false;
    };
    let Ok(series) = frame.series(key) else {
        return false;
    };
    series.on(overrides).nunique().ok().is_some_and(|unique_count| unique_count == frame.num_rows())
}

/// Builds the cuDF hash table over `build_keys` of `build` and wraps it in a handle checked
/// against the execution context's device.
///
/// Returns the handle and the table's device bytes. `record_hash_join_metrics` is true for
/// the reusable root hash join; it counts one hash-join build and its table bytes. The
/// compiled equality preflight builds a table only to size a probe and passes `false`.
/// Errors carry `join_type`, `key_count` and `build_side` facts.
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
    overrides: Exec<'_>,
    record_hash_join_metrics: bool,
) -> Result<(GpuHashJoinBuildHandle, u64)> {
    let hash_join = match build.on(overrides).build_hash_join(build_keys, hash_options) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.build_hash_join",
                "failed to build bounded native reusable hash join",
                error,
            )
            .with_fact("join_type", &node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side));
        },
    };
    let hash_join = match GpuHashJoinBuildHandle::new(execution.context(), hash_join, node.spec.null_equality) {
        Ok(hash_join) => hash_join,
        Err(error) => {
            return Err(error
                .with_fact("join_type", &node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", build_side));
        },
    };
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    if record_hash_join_metrics {
        execution.metrics().record_hash_join_build();
        execution.metrics().record_hash_join_hash_table_bytes(hash_table_bytes);
    }
    Ok((hash_join, hash_table_bytes))
}
