use cudf::{DataFrame, DataFrameMixedJoinOptions, DataFramePredicateJoinOptions, NullEquality as CudfNullEquality};

use super::{
    ArrowDataType, BinaryOp, ExecutionContext, ExecutionOverrides, Expr, JoinNode, LiteralValue, QueryError,
    QueryErrorCode, Result, col, cudf_query_error, timestamp_unit_to_cudf_resolution,
};
use crate::plan::{JoinNullEquality, JoinSide, JoinType};

mod output;
mod residual;

pub(super) use output::{
    dataframe_cross_join_options, materialize_join_output_projection, materialize_left_only_join_output_projection,
    materialize_mark_join_output_projection, projected_input_columns, projected_output_columns,
    reorder_projected_cross_join_output, select_join_input,
};
pub(super) use residual::{JoinResidualColumns, lower_join_residual_predicate, residual_input_columns};

pub(super) fn validate_executable_join_subset(node: &JoinNode) -> Result<()> {
    if !matches!(
        node.spec.join_type,
        JoinType::Inner
            | JoinType::Left
            | JoinType::Full
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::Cross
            | JoinType::LeftMark
    ) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "query_engine.join.execute",
            "native join execution currently supports inner, left, full, left semi, left anti, left mark, and cross joins",
        )
        .with_fact("join_type", node.spec.join_type));
    }
    if node.spec.output_projection.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.join.execute",
            "native join execution requires an explicit output projection",
        ));
    }
    if node.spec.join_type == JoinType::Left && node.spec.build_side != JoinSide::Right {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "query_engine.join.execute",
            "native reusable hash left join execution requires the right side to be the build side",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    if node.spec.join_type == JoinType::Cross && node.spec.residual_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "query_engine.join.execute",
            "native cross join execution does not support residual predicates; use conditional join IR instead",
        )
        .with_fact("join_type", node.spec.join_type));
    }
    Ok(())
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
    cudf_query_error("query_engine.join.execute", "failed to execute algeon native join indices", error)
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

/// How a null-aware `LeftAnti` join (SQL `NOT IN`) restricts its left probe
/// input before an ordinary anti join. The decision depends only on the
/// complete right input, so every probe chunk applies the same rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NullAwareAntiProbe {
    /// Not null-aware, or the right input is empty: `x NOT IN (empty)` holds for every row.
    Unchanged,
    /// A NULL right key makes every comparison unknown, so no left row qualifies.
    Empty,
    /// A non-empty right input without NULL keys: a NULL left key is unknown and drops out.
    DropNullKeys,
}

pub(super) fn null_aware_anti_probe(
    node: &JoinNode,
    right: &DataFrame,
    right_keys: &[String],
) -> Result<NullAwareAntiProbe> {
    if !node.spec.null_aware || right.num_rows() == 0 {
        return Ok(NullAwareAntiProbe::Unchanged);
    }
    if frame_has_null_key(right, right_keys)? {
        Ok(NullAwareAntiProbe::Empty)
    } else {
        Ok(NullAwareAntiProbe::DropNullKeys)
    }
}

/// Returns the restricted left probe input, or `None` when it is unchanged.
pub(super) fn null_aware_anti_probe_input(
    left: &DataFrame,
    left_keys: &[String],
    probe: NullAwareAntiProbe,
    overrides: &ExecutionOverrides<'_>,
) -> Result<Option<DataFrame>> {
    let output = match probe {
        NullAwareAntiProbe::Unchanged => return Ok(None),
        NullAwareAntiProbe::Empty => left.with_execution(overrides).slice(0, 0),
        NullAwareAntiProbe::DropNullKeys if !frame_has_null_key(left, left_keys)? => return Ok(None),
        NullAwareAntiProbe::DropNullKeys => left
            .with_execution(overrides)
            .filter_all(left_keys.iter().map(|key| col(key.as_str()).is_not_null()).collect::<Vec<_>>()),
    };
    output.map(Some).map_err(|error| {
        cudf_query_error(
            "query_engine.join.null_aware_anti_probe",
            "failed to restrict the null-aware anti join probe input",
            error,
        )
    })
}

fn frame_has_null_key(frame: &DataFrame, keys: &[String]) -> Result<bool> {
    for key in keys {
        let series = frame.series(key).map_err(|error| {
            cudf_query_error(
                "query_engine.join.null_aware_anti_probe",
                "failed to inspect null-aware anti join key nulls",
                error,
            )
            .with_fact("column", key.as_str())
        })?;
        if series.null_count() > 0 {
            return Ok(true);
        }
    }
    Ok(false)
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
