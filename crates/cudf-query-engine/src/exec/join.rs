use std::collections::BTreeSet;

use cudf::{DataFrame, DataFrameJoinIndices, DataFrameJoinOptions, NullEquality as CudfNullEquality};

use super::*;
use crate::{
    metrics::{JoinOperatorMetrics, OperatorMetrics},
    plan::{JoinNullEquality, JoinOutputColumn, JoinSide, JoinType},
};

pub(super) fn execute_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_executable_join_subset(node)?;

    let left = left.into_frame();
    let right = right.into_frame();
    let left_rows = left.num_rows() as u64;
    let right_rows = right.num_rows() as u64;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;

    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, JoinSide::Right);

    let left = select_join_input(left, &left_columns, "left")?;
    let right = select_join_input(right, &right_columns, "right")?;
    let options = DataFrameJoinOptions::new()
        .left_on(left_keys)
        .right_on(right_keys)
        .null_equality(lower_null_equality(node.spec.null_equality));

    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let indices = left.with_execution(&overrides).inner_join_indices(&right, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.join.execute", "failed to execute native cudf join indices", error)
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("null_equality", node.spec.null_equality)
            .with_fact("build_side", node.spec.build_side)
    })?;
    let output = materialize_join_output_projection(
        &left,
        &right,
        indices.left(),
        indices.right(),
        &overrides,
        &node.spec.output_projection,
    )?;
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
            build_rows: side_metric(node.spec.build_side, left_rows, right_rows),
            build_bytes: side_metric(node.spec.build_side, left_bytes, right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), left_rows, right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), left_bytes, right_bytes),
            hash_table_bytes: 0,
            output_rows,
            output_bytes,
        },
    ));

    Ok(GpuDataFrame::new(output))
}

fn validate_executable_join_subset(node: &JoinNode) -> Result<()> {
    if node.spec.join_type != JoinType::Inner {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "native join execution currently supports only inner equi joins",
        )
        .with_fact("join_type", node.spec.join_type));
    }
    if node.spec.residual_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "native join execution does not yet support residual predicates",
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
    Ok(())
}

fn projected_input_columns(
    key_columns: &[String],
    output_projection: &[JoinOutputColumn],
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
    columns
}

fn select_join_input(frame: DataFrame, columns: &[String], side: &'static str) -> Result<DataFrame> {
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

fn materialize_join_output_projection(
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

fn projected_output_columns(projection: &[JoinOutputColumn], side: JoinSide) -> Vec<String> {
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

fn lower_null_equality(null_equality: JoinNullEquality) -> CudfNullEquality {
    match null_equality {
        JoinNullEquality::NullsDoNotMatch => CudfNullEquality::Unequal,
        JoinNullEquality::NullsMatch => CudfNullEquality::Equal,
    }
}

fn probe_side(build_side: JoinSide) -> JoinSide {
    match build_side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

fn side_metric(side: JoinSide, left: u64, right: u64) -> u64 {
    match side {
        JoinSide::Left => left,
        JoinSide::Right => right,
    }
}
