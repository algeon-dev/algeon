//! Output projection of native joins: which input columns each side keeps, and the gather of
//! the projected columns from join indices.
//!
//! Join output columns are named by the plan's `JoinOutputColumn` projection, in projection
//! order. Hidden DataFusion order columns (see `exec::order`) ride along on each side and are
//! appended after the projected columns under fresh sequential names.

use std::collections::BTreeSet;

use cudf::{DataFrame, DataFrameCrossJoinOptions, DataFrameJoinIndices, Exec};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{JoinOutputColumn, JoinSide, JoinType},
};
use query_runtime::runtime::ExecutionContext;

use super::cudf_query_error;
use crate::exec::{
    append_datafusion_join_order_columns, join_input_columns_with_order, join_side_projection_with_order,
};

/// Cross join options that project each side as `(source column, output name)` pairs.
pub(in crate::exec) fn dataframe_cross_join_options(projection: &[JoinOutputColumn]) -> DataFrameCrossJoinOptions {
    let left_projection = projection.iter().filter_map(|column| {
        (column.side == JoinSide::Left).then_some((column.column.name(), column.output_name.as_str()))
    });
    let right_projection = projection.iter().filter_map(|column| {
        (column.side == JoinSide::Right).then_some((column.column.name(), column.output_name.as_str()))
    });
    DataFrameCrossJoinOptions::new().left_projection(left_projection).right_projection(right_projection)
}

/// Reorders a cross join result to the projection's output order. The cross join call takes
/// separate left and right projections; an interleaved plan order has to be restored here.
pub(in crate::exec) fn reorder_projected_cross_join_output(
    output: DataFrame,
    projection: &[JoinOutputColumn],
) -> Result<DataFrame> {
    let output_order = projection.iter().map(|column| column.output_name.as_str()).collect::<Vec<_>>();
    output.select(output_order).map_err(|error| {
        cudf_query_error(
            "query_engine.join.projected_cross_output_order",
            "failed to reorder native projected cross join output",
            error,
        )
    })
}

/// Columns one input must keep: key columns first, then the side's output-projected columns,
/// then residual predicate columns, each name once in that first-seen order.
pub(in crate::exec) fn projected_input_columns<'a>(
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

/// Selects `columns` from `frame`, adding any hidden order columns it carries. `side` labels
/// the error facts.
pub(in crate::exec) fn select_join_input(
    frame: DataFrame,
    columns: &[String],
    side: &'static str,
) -> Result<DataFrame> {
    let columns = join_input_columns_with_order(&frame, columns);
    frame.select(&columns).map_err(|error| {
        cudf_query_error("query_engine.join.select_input", "failed to select native join input columns", error)
            .with_fact("side", side)
            .with_fact("columns", columns.join(","))
    })
}

/// Gathers the projected columns of both sides by their join indices and returns one frame
/// in projection order, followed by the hidden order columns.
///
/// `left_indices` and `right_indices` must have equal length (the output row count).
/// A side with no projected or order columns is not gathered; a projection entry that names
/// such a side is `InvalidPlan`. `join_type` and `preserved_side` decide the order in which
/// the sides' hidden order columns are appended (see `append_datafusion_join_order_columns`).
/// cuDF failures map to `query_engine.join.output_projection` with the side as a fact.
#[expect(
    clippy::too_many_arguments,
    reason = "join materialization passes build/probe/output inputs explicitly beside the attempt state"
)]
pub(in crate::exec) fn materialize_join_output_projection(
    left: &DataFrame,
    right: &DataFrame,
    left_indices: &DataFrameJoinIndices,
    right_indices: &DataFrameJoinIndices,
    overrides: Exec<'_>,
    projection: &[JoinOutputColumn],
    join_type: JoinType,
    preserved_side: JoinSide,
) -> Result<DataFrame> {
    let mut left_projection = projected_output_columns(projection, JoinSide::Left);
    let mut right_projection = projected_output_columns(projection, JoinSide::Right);
    join_side_projection_with_order(left, &mut left_projection);
    join_side_projection_with_order(right, &mut right_projection);
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
                "query_engine.join.output_projection",
                "join output projection references a side with no materialized columns",
            )
            .with_fact("side", output.side)
            .with_fact("column", output.column.name())
            .with_fact("output_name", output.output_name.as_str())
        })?;
        let series = frame.series(output.column.name()).map_err(|error| {
            cudf_query_error("query_engine.join.output_projection", "failed to access native join output column", error)
                .with_fact("side", output.side)
                .with_fact("column", output.column.name())
                .with_fact("output_name", output.output_name.as_str())
        })?;
        columns.push((output.output_name.clone(), series));
    }
    append_datafusion_join_order_columns(&mut columns, left.as_ref(), right.as_ref(), join_type, preserved_side)?;

    DataFrame::from_series(columns).map_err(|error| {
        cudf_query_error("query_engine.join.output_projection", "failed to construct native join output frame", error)
    })
}

/// Gathers the left columns for a join whose output has only left columns (semi and anti
/// joins). A projection that names a right-side column is `InvalidPlan`, and so is one with
/// no left column at all.
pub(in crate::exec) fn materialize_left_only_join_output_projection(
    left: &DataFrame,
    indices: &DataFrameJoinIndices,
    overrides: Exec<'_>,
    projection: &[JoinOutputColumn],
    join_type: JoinType,
) -> Result<DataFrame> {
    if let Some(output) = projection.iter().find(|output| output.side == JoinSide::Right) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.join.output_projection",
            "single-side join output projection cannot reference right-side columns",
        )
        .with_fact("side", output.side)
        .with_fact("column", output.column.name())
        .with_fact("output_name", output.output_name.as_str()));
    }

    let mut left_projection = projected_output_columns(projection, JoinSide::Left);
    join_side_projection_with_order(left, &mut left_projection);
    let Some(left) = materialize_join_side_projection(left, indices, left_projection, JoinSide::Left, overrides)?
    else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.join.output_projection",
            "single-side join output projection requires at least one left-side output column",
        ));
    };

    let mut columns = Vec::with_capacity(projection.len());
    for output in projection {
        let series = left.series(output.column.name()).map_err(|error| {
            cudf_query_error("query_engine.join.output_projection", "failed to access native join output column", error)
                .with_fact("side", output.side)
                .with_fact("column", output.column.name())
                .with_fact("output_name", output.output_name.as_str())
        })?;
        columns.push((output.output_name.clone(), series));
    }
    append_datafusion_join_order_columns(&mut columns, Some(&left), None, join_type, JoinSide::Left)?;

    DataFrame::from_series(columns).map_err(|error| {
        cudf_query_error("query_engine.join.output_projection", "failed to construct native join output frame", error)
    })
}

/// Materializes a mark-join output. The output holds all left-side columns from the
/// probe frame (not gathered, because every probe row appears) plus a Boolean mark column.
///
/// `matching_indices` are the left-semi indices returned by the filtered-join
/// probe. They identify which *probe* rows have a match. The mark column is
/// constructed by testing whether each probe row index appears in the matching
/// set via `SeriesOps::isin` (hash-set membership, O(N + M)).
pub(in crate::exec) fn materialize_mark_join_output_projection(
    context: &ExecutionContext,
    probe: &DataFrame,
    matching_indices: &DataFrameJoinIndices,
    projection: &[JoinOutputColumn],
    mark_column_name: &str,
) -> Result<DataFrame> {
    let probe_rows = probe.num_rows();
    let runtime = context.runtime()?;

    // The probe frame may live on a separate Runtime bundle from
    // `context.runtime()?` because the bounded compiled pipeline allocates
    // per-task bundles for the filtered-join probe state. Pre-relocate the
    // probe onto the construction runtime so the resulting DataFrame is built
    // from a single retained provenance. This is a no-op when the probe is
    // already in the construction runtime's provenance.
    let probe = probe.copy_into_runtime(&runtime).map_err(|error| {
        cudf_query_error(
            "query_engine.join.mark_output",
            "failed to relocate probe frame into mark-output runtime",
            error,
        )
    })?;

    // The Boolean mark column is built in two steps.
    // 1. Create a sequential Int32 Series [0, 1, ..., probe_rows-1].
    // 2. Test membership against matching_indices (hash-set, O(N+M)).
    let mark_series = if matching_indices.is_empty() {
        let false_scalar = runtime.scalar(false).map_err(|error| {
            cudf_query_error("query_engine.join.mark_output", "failed to create false scalar for mark column", error)
        })?;
        runtime.series_from_scalar(&false_scalar, probe_rows).map_err(|error| {
            cudf_query_error("query_engine.join.mark_output", "failed to create all-false mark series", error)
        })?
    } else {
        let probe_indices = runtime.sequence(0i32, 1i32, probe_rows).map_err(|error| {
            cudf_query_error(
                "query_engine.join.mark_output",
                "failed to create sequential probe index series for mark column",
                error,
            )
        })?;
        probe_indices.on(Exec::RETAINED).isin(matching_indices.indices()).map_err(|error| {
            cudf_query_error(
                "query_engine.join.mark_output",
                "failed to compute mark column via isin membership test",
                error,
            )
        })?
    };

    let mut columns = Vec::with_capacity(projection.len() + 1);
    for output in projection {
        let series = probe.series(output.column.name()).map_err(|error| {
            cudf_query_error(
                "query_engine.join.mark_output",
                "failed to access probe column for mark join output",
                error,
            )
            .with_fact("column", output.column.name())
            .with_fact("output_name", output.output_name.as_str())
        })?;
        columns.push((output.output_name.clone(), series));
    }
    columns.push((mark_column_name.to_string(), mark_series));

    DataFrame::from_series(columns).map_err(|error| {
        cudf_query_error("query_engine.join.mark_output", "failed to construct native mark join output frame", error)
    })
}

/// Source column names projected from `side`, each once, in first-seen order.
pub(in crate::exec) fn projected_output_columns(projection: &[JoinOutputColumn], side: JoinSide) -> Vec<String> {
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

/// Gathers `columns` of `frame` at `indices`, or returns `None` when the side has no columns.
fn materialize_join_side_projection(
    frame: &DataFrame,
    indices: &DataFrameJoinIndices,
    columns: Vec<String>,
    side: JoinSide,
    overrides: Exec<'_>,
) -> Result<Option<DataFrame>> {
    if columns.is_empty() {
        return Ok(None);
    }

    frame.on(overrides).materialize_join_indices(indices, columns).map(Some).map_err(|error| {
        cudf_query_error(
            "query_engine.join.output_projection",
            "failed to materialize native join output columns",
            error,
        )
        .with_fact("side", side)
    })
}
