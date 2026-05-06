use std::collections::{BTreeMap, BTreeSet};

use cudf::{
    DataFrame, DataFrameCrossJoinOptions, DataFrameHashJoin, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices,
    DataFrameJoinIndices, DataFrameJoinOptions, DataFrameMixedJoinOptions, DataFramePredicateJoinOptions,
    NullEquality as CudfNullEquality,
    ops::ast::{AstExpression, AstOperator},
};

use super::{types::gpu_hash_join_build_columns, *};
use crate::{
    metrics::{JoinOperatorMetrics, OperatorMetrics},
    plan::{JoinNullEquality, JoinOutputColumn, JoinSide, JoinType},
};

enum EquiJoinIndices {
    Pair { left: DataFrameJoinIndices, right: DataFrameJoinIndices },
    LeftOnly(DataFrameJoinIndices),
}

impl EquiJoinIndices {
    fn device_allocated_size_bytes(&self) -> Result<u64> {
        match self {
            Self::Pair { left, right } => {
                let left_bytes = left.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.join.gather_map_bytes",
                        "failed to observe native cudf join left gather-map memory",
                        error,
                    )
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.join.gather_map_bytes",
                        "failed to observe native cudf join right gather-map memory",
                        error,
                    )
                })?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.join.gather_map_bytes",
                        "failed to observe native cudf join gather-map memory",
                        error,
                    )
                })
            },
        }
    }
}

struct JoinIndexResult {
    indices: EquiJoinIndices,
    hash_table_bytes: u64,
    gather_map_bytes: u64,
}

impl JoinIndexResult {
    fn without_hash_table(indices: EquiJoinIndices) -> Self {
        Self { indices, hash_table_bytes: 0, gather_map_bytes: 0 }
    }

    fn with_hash_table(indices: EquiJoinIndices, hash_table_bytes: u64) -> Self {
        Self { indices, hash_table_bytes, gather_map_bytes: 0 }
    }

    fn with_observed_gather_maps(mut self) -> Result<Self> {
        self.gather_map_bytes = self.indices.device_allocated_size_bytes()?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
struct JoinInputStats {
    left_rows: u64,
    right_rows: u64,
    left_bytes: u64,
    right_bytes: u64,
}

struct JoinIndexInput<'a, 'exec> {
    context: &'a ExecutionContext,
    left: &'a DataFrame,
    right: &'a DataFrame,
    left_keys: &'a [String],
    right_keys: &'a [String],
    residual_columns: &'a JoinResidualColumns,
    overrides: &'a ExecutionOverrides<'exec>,
}

struct BuiltHashJoin {
    hash_join: GpuHashJoinBuildHandle,
    hash_table_bytes: u64,
}

pub(super) fn execute_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_executable_join_subset(node)?;

    let left = left.into_frame();
    let right = right.into_frame();
    let left_rows = left.num_rows() as u64;
    let right_rows = right.num_rows() as u64;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;
    let stats = JoinInputStats { left_rows, right_rows, left_bytes, right_bytes };

    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    if node.spec.join_type == JoinType::Cross {
        return execute_cross_join(context, node_id, node, left, right, stats, metrics);
    }

    let residual_columns = residual_input_columns(node)?;
    let left_columns = projected_input_columns(
        &left_keys,
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Left).iter().map(String::as_str),
        JoinSide::Left,
    );
    let right_columns = projected_input_columns(
        &right_keys,
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Right).iter().map(String::as_str),
        JoinSide::Right,
    );

    let left = select_join_input(left, &left_columns, "left")?;
    let right = select_join_input(right, &right_columns, "right")?;
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let input = JoinIndexInput {
        context,
        left: &left,
        right: &right,
        left_keys: &left_keys,
        right_keys: &right_keys,
        residual_columns: &residual_columns,
        overrides: &overrides,
    };
    let indices = execute_join_indices(input, node, scratch, metrics)?.with_observed_gather_maps()?;
    let output = materialize_join_output(&left, &right, &indices.indices, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: node.spec.residual_predicate.is_some(),
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, stats.left_rows, stats.right_rows),
            build_bytes: side_metric(node.spec.build_side, stats.left_bytes, stats.right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), stats.left_rows, stats.right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), stats.left_bytes, stats.right_bytes),
            hash_table_bytes: indices.hash_table_bytes,
            gather_map_bytes: indices.gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    Ok(GpuDataFrame::new(output))
}

pub(super) fn validate_executable_join_subset(node: &JoinNode) -> Result<()> {
    if !matches!(
        node.spec.join_type,
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross
    ) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "native join execution currently supports inner, left, full, left semi, left anti, and cross joins",
        )
        .with_fact("join_type", node.spec.join_type));
    }
    if node.spec.output_projection.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.execute",
            "native join execution requires an explicit output projection",
        ));
    }
    if node.spec.join_type == JoinType::Left && node.spec.build_side != JoinSide::Right {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "native reusable hash left join execution requires the right side to be the build side",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    if node.spec.join_type == JoinType::Cross && node.spec.residual_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "native cross join execution does not support residual predicates; use conditional join IR instead",
        )
        .with_fact("join_type", node.spec.join_type));
    }
    Ok(())
}

fn execute_cross_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: DataFrame,
    right: DataFrame,
    stats: JoinInputStats,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if !node.spec.keys.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.execute_cross",
            "native cross join must not carry equality keys",
        )
        .with_fact("key_count", node.spec.keys.len()));
    }

    let left_projection = join_side_projection(&node.spec.output_projection, JoinSide::Left);
    let right_projection = join_side_projection(&node.spec.output_projection, JoinSide::Right);
    let options = DataFrameCrossJoinOptions::new()
        .left_projection(left_projection.iter().map(|column| (column.column.name(), column.output_name.as_str())))
        .right_projection(right_projection.iter().map(|column| (column.column.name(), column.output_name.as_str())));
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = left.with_execution(&overrides).cross_join_projected(&right, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.join.cross", "failed to execute native cudf cross join", error)
    })?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            key_count: 0,
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, stats.left_rows, stats.right_rows),
            build_bytes: side_metric(node.spec.build_side, stats.left_bytes, stats.right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), stats.left_rows, stats.right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), stats.left_bytes, stats.right_bytes),
            hash_table_bytes: 0,
            gather_map_bytes: 0,
            output_rows,
            output_bytes,
        },
    ));
    Ok(GpuDataFrame::new(output))
}

fn execute_join_indices(
    input: JoinIndexInput<'_, '_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<JoinIndexResult> {
    if node.spec.keys.is_empty() {
        return execute_conditional_join_indices(&input, node).map(JoinIndexResult::without_hash_table);
    }
    if node.spec.residual_predicate.is_some() {
        return execute_mixed_join_indices(&input, node, scratch, metrics);
    }

    match node.spec.join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full => {
            execute_reusable_hash_join_indices(&input, node, scratch, metrics)
        },
        JoinType::LeftSemi => {
            let options = dataframe_join_options(input.left_keys, input.right_keys, node.spec.null_equality);
            input
                .left
                .with_execution(input.overrides)
                .left_semi_join_indices(input.right, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::LeftAnti => {
            let options = dataframe_join_options(input.left_keys, input.right_keys, node.spec.null_equality);
            input
                .left
                .with_execution(input.overrides)
                .left_anti_join_indices(input.right, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::Cross => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "cross joins are not join-index producing operations",
        )
        .with_fact("join_type", node.spec.join_type)),
    }
}

fn execute_reusable_hash_join_indices(
    input: &JoinIndexInput<'_, '_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<JoinIndexResult> {
    match node.spec.build_side {
        JoinSide::Right => {
            let build =
                build_hash_join(input.context, input.right, input.right_keys, input.overrides, node, scratch, metrics)?;
            let indices =
                probe_hash_join(build.hash_join.hash_join(), input.left, input.left_keys, input.overrides, node)?;
            Ok(JoinIndexResult::with_hash_table(
                EquiJoinIndices::Pair { left: indices.probe().clone(), right: indices.build().clone() },
                build.hash_table_bytes,
            ))
        },
        JoinSide::Left => {
            if node.spec.join_type == JoinType::Left {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "cudf_query_engine.join.execute",
                    "native reusable hash left join execution requires the right side to be the build side",
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", node.spec.build_side));
            }
            let build =
                build_hash_join(input.context, input.left, input.left_keys, input.overrides, node, scratch, metrics)?;
            let indices =
                probe_hash_join(build.hash_join.hash_join(), input.right, input.right_keys, input.overrides, node)?;
            Ok(JoinIndexResult::with_hash_table(
                EquiJoinIndices::Pair { left: indices.build().clone(), right: indices.probe().clone() },
                build.hash_table_bytes,
            ))
        },
    }
}

fn execute_conditional_join_indices(input: &JoinIndexInput<'_, '_>, node: &JoinNode) -> Result<EquiJoinIndices> {
    let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
    match node.spec.join_type {
        JoinType::Inner => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_inner_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::Left => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_left_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::Full => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_full_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::LeftSemi => input
            .left
            .with_execution(input.overrides)
            .conditional_left_semi_join_indices(
                input.right,
                &predicate,
                dataframe_predicate_join_options(input.residual_columns),
            )
            .map(EquiJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        JoinType::LeftAnti => input
            .left
            .with_execution(input.overrides)
            .conditional_left_anti_join_indices(
                input.right,
                &predicate,
                dataframe_predicate_join_options(input.residual_columns),
            )
            .map(EquiJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        JoinType::Cross => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "cross joins are not conditional join-index operations",
        )
        .with_fact("join_type", node.spec.join_type)),
    }
}

fn execute_mixed_join_indices(
    input: &JoinIndexInput<'_, '_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<JoinIndexResult> {
    match node.spec.join_type {
        JoinType::Inner | JoinType::Left => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let equi = execute_reusable_hash_join_indices(input, node, scratch, metrics)?;
            let EquiJoinIndices::Pair { left: left_indices, right: right_indices } = equi.indices else {
                unreachable!("inner/left reusable hash joins produce two-sided indices");
            };
            let options = dataframe_predicate_join_options(input.residual_columns);
            let join_type = lower_pair_join_type(node.spec.join_type)?;
            let filtered = input
                .left
                .with_execution(input.overrides)
                .filter_join_indices(input.right, &left_indices, &right_indices, &predicate, options, join_type)
                .map_err(|error| join_index_error(error, node))?;
            Ok(JoinIndexResult::with_hash_table(
                EquiJoinIndices::Pair { left: filtered.left().clone(), right: filtered.right().clone() },
                equi.hash_table_bytes,
            ))
        },
        JoinType::Full => {
            let predicate = lower_join_match_predicate(input.context, node, input.residual_columns)?;
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_full_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(JoinIndexResult::without_hash_table(EquiJoinIndices::Pair {
                left: pairs.left().clone(),
                right: pairs.right().clone(),
            }))
        },
        JoinType::LeftSemi => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let options = dataframe_mixed_join_options(
                input.left_keys,
                input.right_keys,
                input.residual_columns,
                node.spec.null_equality,
            );
            input
                .left
                .with_execution(input.overrides)
                .mixed_left_semi_join_indices(input.right, &predicate, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::LeftAnti => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let options = dataframe_mixed_join_options(
                input.left_keys,
                input.right_keys,
                input.residual_columns,
                node.spec.null_equality,
            );
            input
                .left
                .with_execution(input.overrides)
                .mixed_left_anti_join_indices(input.right, &predicate, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::Cross => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "cross joins are not mixed join-index operations",
        )
        .with_fact("join_type", node.spec.join_type)),
    }
}

fn build_hash_join(
    context: &ExecutionContext,
    build: &DataFrame,
    build_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
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
    let durable_build_artifact_id = durable_build_artifact_id(context, node)?;
    let durable_build_columns =
        if durable_build_artifact_id.is_some() { Some(gpu_hash_join_build_columns(build)?) } else { None };
    if let Some(artifact_id) = &durable_build_artifact_id
        && let Some(cache) = scratch.durable_hash_join_builds.as_ref()
    {
        match cache.get(artifact_id)? {
            Some(hash_join) => {
                hash_join.validate_for_reuse(
                    context,
                    build_keys,
                    durable_build_columns.as_deref().unwrap_or(&[]),
                    node.spec.null_equality,
                )?;
                let hash_table_bytes = hash_join.device_allocated_size_bytes();
                metrics.record_hash_join_durable_cache_hit();
                metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
                scratch.hash_join_builds.insert(cache_key, hash_join.clone());
                return Ok(BuiltHashJoin { hash_join, hash_table_bytes });
            },
            None => metrics.record_hash_join_durable_cache_miss(),
        }
    }
    let hash_join = build
        .with_execution(overrides)
        .build_hash_join(
            build_keys,
            DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality)),
        )
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.join.build_hash_join",
                "failed to build native cudf reusable hash join",
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
    if let Some(artifact_id) = durable_build_artifact_id
        && let Some(cache) = scratch.durable_hash_join_builds.as_ref()
        && cache.insert_if_absent(artifact_id, hash_join.clone())?
    {
        metrics.record_hash_join_durable_cache_insert();
    }
    Ok(BuiltHashJoin { hash_join, hash_table_bytes })
}

fn durable_build_artifact_id(
    context: &ExecutionContext,
    node: &JoinNode,
) -> Result<Option<crate::plan::HashJoinBuildArtifactId>> {
    if !matches!(
        context.boundary_policy().build_materialization(),
        crate::runtime::BuildMaterializationBoundary::DurableGpuHashJoin
    ) {
        return Ok(None);
    }

    let artifact_id = node.spec.build_artifact_id.clone().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.build_hash_join",
            "durable native GPU hash-join build materialization requires an explicit build artifact id",
        )
        .with_fact("reason", "missing_build_artifact_id")
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("build_side", node.spec.build_side)
    })?;
    Ok(Some(artifact_id))
}

fn probe_hash_join(
    hash_join: &DataFrameHashJoin,
    probe: &DataFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<DataFrameHashJoinPairIndices> {
    let output_size = match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_size_in(probe, probe_keys, overrides),
        JoinType::Left => hash_join.left_probe_size_in(probe, probe_keys, overrides),
        JoinType::Full => hash_join.full_probe_size_in(probe, probe_keys, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "cudf_query_engine.join.execute",
                "native join execution currently supports only inner, left, full, left semi, and left anti equi joins",
            )
            .with_fact("join_type", node.spec.join_type));
        },
    }
    .map_err(|error| join_index_error(error, node))?;

    let result = match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Left => hash_join.left_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Full => hash_join.full_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "cudf_query_engine.join.execute",
                "native join execution currently supports only inner, left, full, left semi, and left anti equi joins",
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };

    result.map_err(|error| join_index_error(error, node))
}

fn dataframe_join_options(
    left_keys: &[String],
    right_keys: &[String],
    null_equality: JoinNullEquality,
) -> DataFrameJoinOptions {
    DataFrameJoinOptions::new()
        .left_on(left_keys)
        .right_on(right_keys)
        .null_equality(lower_null_equality(null_equality))
}

fn dataframe_predicate_join_options(residual_columns: &JoinResidualColumns) -> DataFramePredicateJoinOptions {
    DataFramePredicateJoinOptions::new()
        .left_columns(residual_columns.names(JoinSide::Left).iter().map(String::as_str))
        .right_columns(residual_columns.names(JoinSide::Right).iter().map(String::as_str))
}

fn dataframe_mixed_join_options(
    left_keys: &[String],
    right_keys: &[String],
    residual_columns: &JoinResidualColumns,
    null_equality: JoinNullEquality,
) -> DataFrameMixedJoinOptions {
    DataFrameMixedJoinOptions::new()
        .left_on(left_keys.iter().map(String::as_str))
        .right_on(right_keys.iter().map(String::as_str))
        .left_predicate_columns(residual_columns.names(JoinSide::Left).iter().map(String::as_str))
        .right_predicate_columns(residual_columns.names(JoinSide::Right).iter().map(String::as_str))
        .null_equality(lower_null_equality(null_equality))
}

fn lower_pair_join_type(join_type: JoinType) -> Result<cudf::JoinType> {
    match join_type {
        JoinType::Inner => Ok(cudf::JoinType::Inner),
        JoinType::Left => Ok(cudf::JoinType::Left),
        JoinType::Full => Ok(cudf::JoinType::Full),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.lower_join_type",
            "join-index pair filtering supports only inner, left, and full joins",
        )
        .with_fact("join_type", join_type)),
    }
}

pub(super) fn join_index_error(error: cudf::Error, node: &JoinNode) -> QueryError {
    QueryError::from_cudf("cudf_query_engine.join.execute", "failed to execute native cudf join indices", error)
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("null_equality", node.spec.null_equality)
        .with_fact("build_side", node.spec.build_side)
}

fn materialize_join_output(
    left: &DataFrame,
    right: &DataFrame,
    indices: &EquiJoinIndices,
    overrides: &ExecutionOverrides<'_>,
    projection: &[JoinOutputColumn],
) -> Result<DataFrame> {
    match indices {
        EquiJoinIndices::Pair { left: left_indices, right: right_indices } => {
            materialize_join_output_projection(left, right, left_indices, right_indices, overrides, projection)
        },
        EquiJoinIndices::LeftOnly(indices) => {
            materialize_left_only_join_output_projection(left, indices, overrides, projection)
        },
    }
}

#[derive(Debug, Clone)]
struct ResidualColumnLayout {
    names: Vec<String>,
    positions: BTreeMap<String, usize>,
}

impl ResidualColumnLayout {
    fn new(names: Vec<String>) -> Self {
        let positions = names.iter().enumerate().map(|(index, name)| (name.clone(), index)).collect();
        Self { names, positions }
    }

    fn empty() -> Self {
        Self::new(Vec::new())
    }

    fn names(&self) -> &[String] {
        &self.names
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.positions.get(name).copied()
    }
}

#[derive(Debug, Clone)]
struct JoinResidualColumns {
    left: ResidualColumnLayout,
    right: ResidualColumnLayout,
}

impl Default for JoinResidualColumns {
    fn default() -> Self {
        Self { left: ResidualColumnLayout::empty(), right: ResidualColumnLayout::empty() }
    }
}

impl JoinResidualColumns {
    fn new(left: Vec<String>, right: Vec<String>) -> Self {
        Self { left: ResidualColumnLayout::new(left), right: ResidualColumnLayout::new(right) }
    }

    fn names(&self, side: JoinSide) -> &[String] {
        match side {
            JoinSide::Left => self.left.names(),
            JoinSide::Right => self.right.names(),
        }
    }

    fn index_of(&self, side: JoinSide, name: &str) -> Option<usize> {
        match side {
            JoinSide::Left => self.left.index_of(name),
            JoinSide::Right => self.right.index_of(name),
        }
    }
}

fn residual_input_columns(node: &JoinNode) -> Result<JoinResidualColumns> {
    let Some(predicate) = &node.spec.residual_predicate else {
        return Ok(JoinResidualColumns::default());
    };
    let mut left = BTreeSet::new();
    let mut right = BTreeSet::new();
    collect_join_expr_columns(predicate, &mut left, &mut right)?;
    for key in &node.spec.keys {
        left.insert(key.left.name().to_string());
        right.insert(key.right.name().to_string());
    }
    if left.is_empty() || right.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.residual_columns",
            "native join residual predicates must reference at least one column from each input",
        )
        .with_fact("predicate", predicate));
    }
    Ok(JoinResidualColumns::new(left.into_iter().collect(), right.into_iter().collect()))
}

fn collect_join_expr_columns(expr: &Expr, left: &mut BTreeSet<String>, right: &mut BTreeSet<String>) -> Result<()> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(crate::expr::ColumnQualifier::Input { index: 0 }) => {
                left.insert(column.name().to_string());
                Ok(())
            },
            Some(crate::expr::ColumnQualifier::Input { index: 1 }) => {
                right.insert(column.name().to_string());
                Ok(())
            },
            qualifier => Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.residual_columns",
                "native join residual predicates require input-qualified column references",
            )
            .with_fact("column", column.name())
            .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(_) => Ok(()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => collect_join_expr_columns(inner, left, right),
        Expr::Binary { left: lhs, right: rhs, .. } => {
            collect_join_expr_columns(lhs, left, right)?;
            collect_join_expr_columns(rhs, left, right)
        },
        Expr::InList { expr, list, .. } => {
            collect_join_expr_columns(expr, left, right)?;
            for item in list {
                collect_join_expr_columns(item, left, right)?;
            }
            Ok(())
        },
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.residual_columns",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_join_residual_predicate(
    context: &ExecutionContext,
    node: &JoinNode,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    let Some(predicate) = &node.spec.residual_predicate else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.lower_residual",
            "join residual lowering requires a residual predicate",
        ));
    };
    lower_join_ast_expr(context, predicate, residual_columns)
}

fn lower_join_match_predicate(
    context: &ExecutionContext,
    node: &JoinNode,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    let mut predicate = node.spec.residual_predicate.clone().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.lower_residual",
            "join match predicate lowering requires a residual predicate",
        )
    })?;

    for key in &node.spec.keys {
        let equality = join_key_match_expr(key, node.spec.null_equality);
        predicate = Expr::binary(equality, BinaryOp::And, predicate);
    }

    lower_join_ast_expr(context, &predicate, residual_columns)
}

fn join_key_match_expr(key: &crate::plan::JoinKey, null_equality: JoinNullEquality) -> Expr {
    let left = Expr::Column(key.left.clone());
    let right = Expr::Column(key.right.clone());
    let equality = Expr::binary(left.clone(), BinaryOp::Eq, right.clone());
    match null_equality {
        JoinNullEquality::NullsDoNotMatch => equality,
        JoinNullEquality::NullsMatch => {
            Expr::binary(equality, BinaryOp::Or, Expr::binary(Expr::is_null(left), BinaryOp::And, Expr::is_null(right)))
        },
    }
}

fn lower_join_ast_expr(
    context: &ExecutionContext,
    expr: &Expr,
    residual_columns: &JoinResidualColumns,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let (side, index) = match column.qualifier() {
                Some(crate::expr::ColumnQualifier::Input { index: 0 }) => {
                    (JoinSide::Left, residual_columns.index_of(JoinSide::Left, column.name()))
                },
                Some(crate::expr::ColumnQualifier::Input { index: 1 }) => {
                    (JoinSide::Right, residual_columns.index_of(JoinSide::Right, column.name()))
                },
                qualifier => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.lower_residual",
                        "join residual column must be input-qualified",
                    )
                    .with_fact("column", column.name())
                    .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string())));
                },
            };
            let Some(index) = index else {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.join.lower_residual",
                    "join residual column was not included in predicate columns",
                )
                .with_fact("side", side)
                .with_fact("column", column.name()));
            };
            let index = i32::try_from(index).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.lower_residual",
                    "join residual predicate column index exceeds i32 range",
                )
                .with_fact("side", side)
                .with_fact("column", column.name())
                .with_fact("index", index)
            })?;
            let expression = match side {
                JoinSide::Left => AstExpression::left_column(index),
                JoinSide::Right => AstExpression::right_column(index),
            };
            expression.map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST column reference",
                    error,
                )
            })
        },
        Expr::Literal(value) => lower_join_ast_literal(context, value),
        Expr::Not(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::Not, input).map_err(|error| {
                QueryError::from_cudf("cudf_query_engine.join.lower_residual", "failed to build cudf AST NOT", error)
            })
        },
        Expr::IsNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })
        },
        Expr::IsNotNull(input) => {
            let input = lower_join_ast_expr(context, input, residual_columns)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })?;
            AstExpression::unary(AstOperator::Not, is_null).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IS NOT NULL",
                    error,
                )
            })
        },
        Expr::Cast { expr, target_type } => {
            let input = lower_join_ast_expr(context, expr, residual_columns)?;
            let expression = match target_type {
                ArrowDataType::Int64 => AstExpression::cast_to_int64(input),
                ArrowDataType::UInt64 => AstExpression::cast_to_uint64(input),
                ArrowDataType::Float64 => AstExpression::cast_to_float64(input),
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.lower_residual",
                        "native join residual cast is not supported by the cudf AST contract",
                    )
                    .with_fact("target_type", format!("{target_type:?}")));
                },
            };
            expression.map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST cast expression",
                    error,
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        Expr::Binary { left, op, right } => {
            let left = lower_join_ast_expr(context, left, residual_columns)?;
            let right = lower_join_ast_expr(context, right, residual_columns)?;
            let op = lower_join_ast_operator(*op)?;
            AstExpression::binary(op, left, right).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST binary expression",
                    error,
                )
            })
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.lower_residual",
                    "native join residual IN list must not be empty",
                ));
            }
            let input = lower_join_ast_expr(context, expr, residual_columns)?;
            let values = list
                .iter()
                .map(|value| {
                    if !matches!(value, Expr::Literal(_)) {
                        return Err(QueryError::new(
                            QueryErrorCode::UnsupportedExpression,
                            "cudf_query_engine.join.lower_residual",
                            "native join residual IN list values must be literals",
                        )
                        .with_fact("value", value));
                    }
                    lower_join_ast_expr(context, value, residual_columns)
                })
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.join.lower_residual",
                    "failed to build cudf AST IN list expression",
                    error,
                )
                .with_fact("negated", negated.to_string())
                .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => lower_join_ast_expr(context, expr, residual_columns),
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.lower_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_join_ast_operator(op: BinaryOp) -> Result<AstOperator> {
    match op {
        BinaryOp::Add => Ok(AstOperator::Add),
        BinaryOp::Subtract => Ok(AstOperator::Sub),
        BinaryOp::Multiply => Ok(AstOperator::Mul),
        BinaryOp::Divide => Ok(AstOperator::Div),
        BinaryOp::Modulo => Ok(AstOperator::Mod),
        BinaryOp::Eq => Ok(AstOperator::Equal),
        BinaryOp::NotEq => Ok(AstOperator::NotEqual),
        BinaryOp::Lt => Ok(AstOperator::Less),
        BinaryOp::LtEq => Ok(AstOperator::LessEqual),
        BinaryOp::Gt => Ok(AstOperator::Greater),
        BinaryOp::GtEq => Ok(AstOperator::GreaterEqual),
        BinaryOp::And => Ok(AstOperator::LogicalAnd),
        BinaryOp::Or => Ok(AstOperator::LogicalOr),
    }
}

fn lower_join_ast_literal(context: &ExecutionContext, value: &LiteralValue) -> Result<AstExpression> {
    let runtime = context.runtime();
    let scalar = match value {
        LiteralValue::Boolean(value) => runtime.scalar(*value),
        LiteralValue::Int8(value) => runtime.scalar(*value),
        LiteralValue::Int16(value) => runtime.scalar(*value),
        LiteralValue::Int32(value) => runtime.scalar(*value),
        LiteralValue::Int64(value) => runtime.scalar(*value),
        LiteralValue::UInt8(value) => runtime.scalar(*value),
        LiteralValue::UInt16(value) => runtime.scalar(*value),
        LiteralValue::UInt32(value) => runtime.scalar(*value),
        LiteralValue::UInt64(value) => runtime.scalar(*value),
        LiteralValue::Float32(value) => runtime.scalar(*value),
        LiteralValue::Float64(value) => runtime.scalar(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => runtime.string_scalar(value),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            runtime.decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        LiteralValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), cudf::TimestampResolution::Days),
        LiteralValue::Timestamp { raw, unit } => {
            runtime.timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            runtime.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.lower_residual",
                "join residual AST literals do not support interval values",
            )
            .with_fact("literal", value));
        },
        LiteralValue::Null(data_type) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.lower_residual",
                "join residual AST literals do not yet support typed null values",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.lower_residual",
            "failed to build cudf scalar for join residual AST literal",
            error,
        )
        .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.join.lower_residual", "failed to build cudf AST literal", error)
    })
}

fn join_side_projection(projection: &[JoinOutputColumn], side: JoinSide) -> Vec<JoinOutputColumn> {
    projection.iter().filter(|column| column.side == side).cloned().collect()
}

pub(super) fn projected_input_columns<'a>(
    key_columns: &[String],
    output_projection: &[JoinOutputColumn],
    residual_columns: impl IntoIterator<Item = &'a str>,
    side: JoinSide,
) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut columns = Vec::new();
    for column in key_columns {
        if seen.insert(column.clone()) {
            columns.push(column.clone());
        }
    }
    for column in output_projection.iter().filter(|column| column.side == side) {
        let name = column.column.name().to_string();
        if seen.insert(name.clone()) {
            columns.push(name);
        }
    }
    for column in residual_columns {
        if seen.insert(column.to_string()) {
            columns.push(column.to_string());
        }
    }
    columns
}

pub(super) fn select_join_input(frame: DataFrame, columns: &[String], side: &'static str) -> Result<DataFrame> {
    frame.select(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.select_input",
            "failed to select native join input columns",
            error,
        )
        .with_fact("side", side)
        .with_fact("columns", columns.join(","))
    })
}

pub(super) fn materialize_join_output_projection(
    left: &DataFrame,
    right: &DataFrame,
    left_indices: &DataFrameJoinIndices,
    right_indices: &DataFrameJoinIndices,
    overrides: &ExecutionOverrides<'_>,
    projection: &[JoinOutputColumn],
) -> Result<DataFrame> {
    let left_projection = projected_output_columns(projection, JoinSide::Left);
    let right_projection = projected_output_columns(projection, JoinSide::Right);
    let left = materialize_join_side_projection(left, left_indices, left_projection, JoinSide::Left, overrides)?;
    let right = materialize_join_side_projection(right, right_indices, right_projection, JoinSide::Right, overrides)?;

    let mut columns = Vec::with_capacity(projection.len());
    for output in projection {
        let frame = match output.side {
            JoinSide::Left => left.as_ref(),
            JoinSide::Right => right.as_ref(),
        };
        let frame = frame.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.join.output_projection",
                "join output projection references a side with no materialized columns",
            )
            .with_fact("side", output.side)
            .with_fact("column", output.column.name())
            .with_fact("output_name", output.output_name.as_str())
        })?;
        let series = frame.series(output.column.name()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.join.output_projection",
                "failed to access native join output column",
                error,
            )
            .with_fact("side", output.side)
            .with_fact("column", output.column.name())
            .with_fact("output_name", output.output_name.as_str())
        })?;
        columns.push((output.output_name.clone(), series));
    }

    DataFrame::from_series(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.output_projection",
            "failed to construct native join output frame",
            error,
        )
    })
}

fn materialize_left_only_join_output_projection(
    left: &DataFrame,
    indices: &DataFrameJoinIndices,
    overrides: &ExecutionOverrides<'_>,
    projection: &[JoinOutputColumn],
) -> Result<DataFrame> {
    if let Some(output) = projection.iter().find(|output| output.side == JoinSide::Right) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.output_projection",
            "single-side join output projection cannot reference right-side columns",
        )
        .with_fact("side", output.side)
        .with_fact("column", output.column.name())
        .with_fact("output_name", output.output_name.as_str()));
    }

    let left_projection = projected_output_columns(projection, JoinSide::Left);
    let Some(left) = materialize_join_side_projection(left, indices, left_projection, JoinSide::Left, overrides)?
    else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.output_projection",
            "single-side join output projection requires at least one left-side output column",
        ));
    };

    let mut columns = Vec::with_capacity(projection.len());
    for output in projection {
        let series = left.series(output.column.name()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.join.output_projection",
                "failed to access native join output column",
                error,
            )
            .with_fact("side", output.side)
            .with_fact("column", output.column.name())
            .with_fact("output_name", output.output_name.as_str())
        })?;
        columns.push((output.output_name.clone(), series));
    }

    DataFrame::from_series(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.output_projection",
            "failed to construct native join output frame",
            error,
        )
    })
}

pub(super) fn projected_output_columns(projection: &[JoinOutputColumn], side: JoinSide) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut columns = Vec::new();
    for column in projection.iter().filter(|column| column.side == side) {
        let name = column.column.name().to_string();
        if seen.insert(name.clone()) {
            columns.push(name);
        }
    }
    columns
}

fn materialize_join_side_projection(
    frame: &DataFrame,
    indices: &DataFrameJoinIndices,
    columns: Vec<String>,
    side: JoinSide,
    overrides: &ExecutionOverrides<'_>,
) -> Result<Option<DataFrame>> {
    if columns.is_empty() {
        return Ok(None);
    }

    frame.with_execution(overrides).materialize_join_indices(indices, columns).map(Some).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.join.output_projection",
            "failed to materialize native join output columns",
            error,
        )
        .with_fact("side", side)
    })
}

pub(super) fn lower_null_equality(null_equality: JoinNullEquality) -> CudfNullEquality {
    match null_equality {
        JoinNullEquality::NullsDoNotMatch => CudfNullEquality::Unequal,
        JoinNullEquality::NullsMatch => CudfNullEquality::Equal,
    }
}

pub(super) fn probe_side(build_side: JoinSide) -> JoinSide {
    match build_side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

pub(super) fn side_metric(side: JoinSide, left: u64, right: u64) -> u64 {
    match side {
        JoinSide::Left => left,
        JoinSide::Right => right,
    }
}
