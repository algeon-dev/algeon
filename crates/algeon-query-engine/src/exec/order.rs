//! Hidden DataFusion order columns.
//!
//! DataFusion accumulates floating-point statistical aggregates in row order; results that
//! must match it bit for bit depend on the order rows reach the aggregate. When
//! `statistical_aggregate_execution_mode` is DataFusion-exact and the plan contains such an
//! aggregate, `execute_query_plan` enables order tracking for the whole query. Source reads
//! then append two Int64 columns holding a major ordinal (file or chunk index) and the row
//! position, joins renumber and carry the columns of their inputs, and
//! `sort_by_datafusion_order_if_present` restores the order before the aggregate.
//!
//! The columns are recognized by name prefix alone and survive the source-read worker
//! threads. Without tracking, none of these functions finds or creates a
//! column and each reduces to a no-op or a copy of its input list.

use std::{sync::Arc, time::Instant};

use arrow::datatypes::{DataType as ArrowDataType, Field, Schema, SchemaRef};
use cudf::{DataFrame, SortOptions as CudfSortOptions};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    naming::{DATAFUSION_ORDER_COLUMN_PREFIX, is_datafusion_order_column_name},
    plan::{JoinSide, JoinType, PlanNode, QueryPlan},
};
use query_runtime::runtime::ExecutionContext;

use super::{ExecutionMetrics, cudf_query_error};

const SOURCE_ORDER_MAJOR_COLUMN: &str = "__algeon_native_df_order_0";
const SOURCE_ORDER_ROW_COLUMN: &str = "__algeon_native_df_order_1";

/// Whether any aggregate in `plan` is a floating-point statistical function whose result
/// depends on input row order.
pub(in crate::exec) fn plan_requires_datafusion_exact_statistical_order(plan: &QueryPlan) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => {
            aggregate.spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
        },
        _ => false,
    })
}

/// Lists the hidden DataFusion order columns materialized on a frame.
/// Detection is purely by column name so frames keep their order facts across
/// the source-read worker thread boundary; whether a query may create such
/// columns is governed by [`ExecutionContext::datafusion_order_tracking`].
pub(in crate::exec) fn datafusion_order_columns(frame: &DataFrame) -> Vec<String> {
    frame
        .column_names()
        .into_iter()
        .filter(|name| is_datafusion_order_column_name(name))
        .map(ToString::to_string)
        .collect()
}

/// Adds the two hidden order columns to a freshly read source frame: `major_ordinal` as a
/// constant and the row position from zero.
///
/// Returns the frame unchanged when tracking is off or the columns already exist. Otherwise
/// the frame is first relocated into the context's runtime. Failures map to
/// `query_engine.datafusion_order.source`; an ordinal beyond Int64 is `InvalidPlan`.
pub(in crate::exec) fn append_source_order_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    major_ordinal: usize,
) -> Result<DataFrame> {
    if !context.datafusion_order_tracking() || frame.contains(SOURCE_ORDER_MAJOR_COLUMN) {
        return Ok(frame);
    }
    let frame = frame.copy_into_runtime(&context.runtime()?).map_err(|error| {
        cudf_query_error(
            "query_engine.datafusion_order.source",
            "failed to relocate native source frame before appending hidden order columns",
            error,
        )
    })?;
    let rows = frame.num_rows();
    let major = i64::try_from(major_ordinal).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.datafusion_order.source",
            "source order major ordinal exceeds Int64 range",
        )
        .with_fact("major_ordinal", major_ordinal)
    })?;
    let major = context.runtime()?.sequence::<i64>(major, 0, rows).map_err(|error| {
        cudf_query_error(
            "query_engine.datafusion_order.source",
            "failed to materialize native source order major column",
            error,
        )
        .with_fact("rows", rows)
    })?;
    let row = context.runtime()?.sequence::<i64>(0, 1, rows).map_err(|error| {
        cudf_query_error(
            "query_engine.datafusion_order.source",
            "failed to materialize native source order row column",
            error,
        )
        .with_fact("rows", rows)
    })?;
    frame
        .with_column(SOURCE_ORDER_MAJOR_COLUMN, major)
        .and_then(|frame| frame.with_column(SOURCE_ORDER_ROW_COLUMN, row))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.datafusion_order.source",
                "failed to append hidden native source order columns",
                error,
            )
        })
}

/// `columns` plus the frame's hidden order columns that are not already listed.
pub(in crate::exec) fn join_input_columns_with_order(frame: &DataFrame, columns: &[String]) -> Vec<String> {
    let mut output = columns.to_vec();
    for order_column in datafusion_order_columns(frame) {
        if !output.iter().any(|column| column == &order_column) {
            output.push(order_column);
        }
    }
    output
}

pub(in crate::exec) fn join_side_projection_with_order(frame: &DataFrame, columns: &mut Vec<String>) {
    for order_column in datafusion_order_columns(frame) {
        if !columns.iter().any(|column| column == &order_column) {
            columns.push(order_column);
        }
    }
}

/// Appends the join inputs' hidden order columns to `columns` under fresh sequential names.
///
/// The side listed first gets the lower numbers and is the major key of a later sort by
/// these columns. An inner join lists the preserved side first, matching which input
/// DataFusion's hash join streams in order; every other join type lists left, then right.
/// A `None` frame contributes nothing.
pub(in crate::exec) fn append_datafusion_join_order_columns(
    columns: &mut Vec<(String, cudf::Series)>,
    left: Option<&DataFrame>,
    right: Option<&DataFrame>,
    join_type: JoinType,
    preserved_side: JoinSide,
) -> Result<()> {
    // Order inputs are detected by column name: join sides only carry such
    // columns when the query's execution context enabled order tracking at
    // the sources, so there is nothing to propagate otherwise.
    let mut next_index = 0usize;
    match join_type {
        // DataFusion HashJoinExec preserves its probe stream for inner joins.
        // Lowering records which native side corresponds to that stream because
        // native build/probe orientation can differ from DataFusion's.
        JoinType::Inner => match preserved_side {
            JoinSide::Left => {
                append_side_order_columns(columns, left, &mut next_index, "left")?;
                append_side_order_columns(columns, right, &mut next_index, "right")?;
            },
            JoinSide::Right => {
                append_side_order_columns(columns, right, &mut next_index, "right")?;
                append_side_order_columns(columns, left, &mut next_index, "left")?;
            },
        },
        JoinType::Left
        | JoinType::LeftSemi
        | JoinType::LeftAnti { .. }
        | JoinType::LeftMark { .. }
        | JoinType::Full
        | JoinType::Cross => {
            append_side_order_columns(columns, left, &mut next_index, "left")?;
            append_side_order_columns(columns, right, &mut next_index, "right")?;
        },
    }
    Ok(())
}

/// Stably sorts `frame` by its hidden order columns, or returns it unchanged when it has
/// none. The sort time is added to the operator time; a cuDF failure maps to
/// `query_engine.datafusion_order.sort`.
pub(in crate::exec) fn sort_by_datafusion_order_if_present(
    context: &ExecutionContext,
    frame: DataFrame,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let order_columns = datafusion_order_columns(&frame);
    if order_columns.is_empty() {
        return Ok(frame);
    }
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let output = frame
        .on(overrides)
        .sort_values(order_columns.iter().map(String::as_str), CudfSortOptions::new().with_stable(true))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.datafusion_order.sort",
                "failed to sort native stddev aggregate input by DataFusion order columns",
                error,
            )
            .with_fact("order_columns", order_columns.join(","))
        })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(output)
}

/// Returns `schema` extended with the hidden order columns of `frame`, with
/// fields in the frame's column order. Aggregate input preparation appends its
/// columns after the order columns, so the order columns do not always trail.
pub(in crate::exec) fn schema_with_datafusion_order_columns(frame: &DataFrame, schema: &SchemaRef) -> SchemaRef {
    if datafusion_order_columns(frame).is_empty() {
        return Arc::clone(schema);
    }
    let fields = frame
        .column_names()
        .into_iter()
        .filter_map(|name| {
            if is_datafusion_order_column_name(name) {
                Some(Field::new(name, ArrowDataType::Int64, false))
            } else {
                schema.field_with_name(name).ok().cloned()
            }
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

fn append_side_order_columns(
    columns: &mut Vec<(String, cudf::Series)>,
    frame: Option<&DataFrame>,
    next_index: &mut usize,
    side: &'static str,
) -> Result<()> {
    let Some(frame) = frame else {
        return Ok(());
    };
    for input_name in datafusion_order_columns(frame) {
        let output_name = format!("{DATAFUSION_ORDER_COLUMN_PREFIX}{next_index}");
        *next_index += 1;
        let series = frame.series(input_name.as_str()).map_err(|error| {
            cudf_query_error(
                "query_engine.datafusion_order.join",
                "failed to access materialized native join order column",
                error,
            )
            .with_fact("side", side)
            .with_fact("column", input_name.as_str())
        })?;
        columns.push((output_name, series));
    }
    Ok(())
}
