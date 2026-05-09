use cudf::{
    DataFrame, DataFrameHashJoin, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices, DataFrameJoinIndices,
    DataFrameJoinOptions, DataFrameMixedJoinOptions, DataFramePredicateJoinOptions, NullEquality as CudfNullEquality,
};

use super::{types::gpu_hash_join_build_columns, *};
use crate::{
    metrics::{JoinOperatorMetrics, OperatorMetrics},
    plan::{JoinNullEquality, JoinSide, JoinType},
};

mod conditional;
mod hash;
mod mixed;
mod output;
mod residual;

use conditional::execute_conditional_join_indices;
use hash::execute_reusable_hash_join_indices;
use mixed::execute_mixed_join_indices;
use output::materialize_join_output;
pub(super) use output::{
    dataframe_cross_join_options, materialize_join_output_projection, materialize_left_only_join_output_projection,
    projected_input_columns, projected_output_columns, reorder_projected_cross_join_output, select_join_input,
};
pub(super) use residual::{
    JoinResidualColumns, lower_join_match_predicate, lower_join_residual_predicate, residual_input_columns,
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
    mixed_size_data_bytes: u64,
    gather_map_bytes: u64,
}

impl JoinIndexResult {
    fn without_hash_table(indices: EquiJoinIndices) -> Self {
        Self { indices, hash_table_bytes: 0, mixed_size_data_bytes: 0, gather_map_bytes: 0 }
    }

    fn with_hash_table(indices: EquiJoinIndices, hash_table_bytes: u64) -> Self {
        Self { indices, hash_table_bytes, mixed_size_data_bytes: 0, gather_map_bytes: 0 }
    }

    fn with_mixed_size_data(mut self, mixed_size_data_bytes: u64) -> Self {
        self.mixed_size_data_bytes = mixed_size_data_bytes;
        self
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
            execution_strategy: join_execution_strategy(node).to_string(),
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
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: indices.mixed_size_data_bytes,
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

    let options = dataframe_cross_join_options(&node.spec.output_projection);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = left.with_execution(&overrides).cross_join_projected(&right, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.join.cross", "failed to execute native cudf cross join", error)
    })?;
    let output = reorder_projected_cross_join_output(output, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "cross_join".to_string(),
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
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: 0,
            output_rows,
            output_bytes,
        },
    ));
    Ok(GpuDataFrame::new(output))
}

fn join_execution_strategy(node: &JoinNode) -> &'static str {
    if node.spec.keys.is_empty() {
        "conditional_join"
    } else if node.spec.residual_predicate.is_some() {
        "mixed_join"
    } else {
        "hash_join"
    }
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

pub(super) fn dataframe_predicate_join_options(
    residual_columns: &JoinResidualColumns,
) -> DataFramePredicateJoinOptions {
    DataFramePredicateJoinOptions::new()
        .left_columns(residual_columns.names(JoinSide::Left).iter().map(String::as_str))
        .right_columns(residual_columns.names(JoinSide::Right).iter().map(String::as_str))
}

pub(super) fn dataframe_mixed_join_options(
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

pub(super) fn join_index_error(error: cudf::Error, node: &JoinNode) -> QueryError {
    join_index_error_base(error, node)
}

pub(super) fn join_index_error_base(error: cudf::Error, node: &JoinNode) -> QueryError {
    QueryError::from_cudf("cudf_query_engine.join.execute", "failed to execute native cudf join indices", error)
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("null_equality", node.spec.null_equality)
        .with_fact("build_side", node.spec.build_side)
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
