use super::{
    DataFrame, DataFrameHashJoin, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices, EquiJoinIndices,
    ExecutionContext, ExecutionMetrics, ExecutionOverrides, ExecutionScratch, GpuHashJoinBuildHandle, HashJoinCacheKey,
    JoinIndexInput, JoinIndexResult, JoinNode, JoinSide, JoinType, QueryError, QueryErrorCode, Result,
    cudf_query_error, join_index_error_base, lower_null_equality,
};

struct JoinKeyDiagnostics {
    key_pairs: String,
    left_key_dtypes: String,
    right_key_dtypes: String,
}

impl JoinKeyDiagnostics {
    fn new(left: &DataFrame, right: &DataFrame, left_keys: &[String], right_keys: &[String]) -> Self {
        Self {
            key_pairs: format_join_key_pairs(left_keys, right_keys),
            left_key_dtypes: format_join_key_dtypes(left, left_keys),
            right_key_dtypes: format_join_key_dtypes(right, right_keys),
        }
    }
}

struct BuiltHashJoin {
    hash_join: GpuHashJoinBuildHandle,
    hash_table_bytes: u64,
}

pub(super) fn execute_reusable_hash_join_indices(
    input: &JoinIndexInput<'_, '_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &ExecutionMetrics,
) -> Result<JoinIndexResult> {
    match node.spec.build_side {
        JoinSide::Right => {
            let build =
                build_hash_join(input.context, input.right, input.right_keys, input.overrides, node, scratch, metrics)?;
            let diagnostics = JoinKeyDiagnostics::new(input.left, input.right, input.left_keys, input.right_keys);
            let indices = probe_hash_join(
                build.hash_join.hash_join(),
                input.left,
                input.left_keys,
                input.overrides,
                node,
                &diagnostics,
            )?;
            Ok(JoinIndexResult::with_hash_table(
                EquiJoinIndices::Pair { left: indices.probe().clone(), right: indices.build().clone() },
                build.hash_table_bytes,
            ))
        },
        JoinSide::Left => {
            if node.spec.join_type == JoinType::Left {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "query_engine.join.execute",
                    "native reusable hash left join execution requires the right side to be the build side",
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", node.spec.build_side));
            }
            let build =
                build_hash_join(input.context, input.left, input.left_keys, input.overrides, node, scratch, metrics)?;
            let diagnostics = JoinKeyDiagnostics::new(input.left, input.right, input.left_keys, input.right_keys);
            let indices = probe_hash_join(
                build.hash_join.hash_join(),
                input.right,
                input.right_keys,
                input.overrides,
                node,
                &diagnostics,
            )?;
            Ok(JoinIndexResult::with_hash_table(
                EquiJoinIndices::Pair { left: indices.build().clone(), right: indices.probe().clone() },
                build.hash_table_bytes,
            ))
        },
    }
}

fn build_hash_join(
    context: &ExecutionContext,
    build: &DataFrame,
    build_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &ExecutionMetrics,
) -> Result<BuiltHashJoin> {
    let build_node = match node.spec.build_side {
        JoinSide::Left => node.inputs[0],
        JoinSide::Right => node.inputs[1],
    };
    let cache_key = HashJoinCacheKey::new(build_node, build_keys.to_vec(), node.spec.null_equality);
    if let Some(hash_join) = scratch.hash_join_builds.get(&cache_key) {
        metrics.record_hash_join_cache_hit();
        return Ok(BuiltHashJoin {
            hash_join: hash_join.clone(),
            hash_table_bytes: hash_join.device_allocated_size_bytes(),
        });
    }

    metrics.record_hash_join_cache_miss();
    let hash_join = build
        .with_execution(overrides)
        .build_hash_join(
            build_keys,
            DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality)),
        )
        .map_err(|error| {
            cudf_query_error(
                "query_engine.join.build_hash_join",
                "failed to build algeon native reusable hash join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("null_equality", node.spec.null_equality)
            .with_fact("build_side", node.spec.build_side)
        })?;
    let hash_join = GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality).map_err(|error| {
        error
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", node.spec.build_side)
    })?;
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
    scratch.hash_join_builds.insert(cache_key, hash_join.clone());
    Ok(BuiltHashJoin { hash_join, hash_table_bytes })
}

fn probe_hash_join(
    hash_join: &DataFrameHashJoin,
    probe: &DataFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
    diagnostics: &JoinKeyDiagnostics,
) -> Result<DataFrameHashJoinPairIndices> {
    let output_size = match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_size_in(probe, probe_keys, overrides),
        JoinType::Left => hash_join.left_probe_size_in(probe, probe_keys, overrides),
        JoinType::Full => hash_join.full_probe_size_in(probe, probe_keys, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.join.execute",
                "native join execution currently supports only inner, left, full, left semi, and left anti equi joins",
            )
            .with_fact("join_type", node.spec.join_type));
        },
    }
    .map_err(|error| join_index_error_with_diagnostics(error, node, diagnostics))?;

    let result = match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Left => hash_join.left_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Full => hash_join.full_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.join.execute",
                "native join execution currently supports only inner, left, full, left semi, and left anti equi joins",
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };

    result.map_err(|error| join_index_error_with_diagnostics(error, node, diagnostics))
}

fn join_index_error_with_diagnostics(
    error: cudf::Error,
    node: &JoinNode,
    diagnostics: &JoinKeyDiagnostics,
) -> QueryError {
    join_index_error_base(error, node)
        .with_fact("join_key_pairs", diagnostics.key_pairs.as_str())
        .with_fact("left_key_dtypes", diagnostics.left_key_dtypes.as_str())
        .with_fact("right_key_dtypes", diagnostics.right_key_dtypes.as_str())
}

fn format_join_key_pairs(left_keys: &[String], right_keys: &[String]) -> String {
    left_keys.iter().zip(right_keys.iter()).map(|(left, right)| format!("{left}={right}")).collect::<Vec<_>>().join(",")
}

fn format_join_key_dtypes(frame: &DataFrame, keys: &[String]) -> String {
    keys.iter()
        .map(|key| {
            let dtype = frame
                .series(key.as_str())
                .and_then(|series| series.dtype())
                .map(|dtype| format!("{dtype:?}"))
                .unwrap_or_else(|error| format!("error:{error}"));
            format!("{key}:{dtype}")
        })
        .collect::<Vec<_>>()
        .join(",")
}
