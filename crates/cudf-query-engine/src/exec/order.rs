use std::cell::Cell;

use arrow::datatypes::{DataType as ArrowDataType, Field, Schema, SchemaRef};
use cudf::{DataFrame, SortOptions as CudfSortOptions};

use super::*;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::AggregateFunction,
    plan::{JoinType, PlanNode, QueryPlan},
    runtime::ExecutionContext,
};

const DATAFUSION_ORDER_COLUMN_PREFIX: &str = "__cudf_df_order_";
const SOURCE_ORDER_MAJOR_COLUMN: &str = "__cudf_df_order_0";
const SOURCE_ORDER_ROW_COLUMN: &str = "__cudf_df_order_1";

thread_local! {
    static DATAFUSION_ORDER_TRACKING_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::exec) struct DataFusionOrderTrackingGuard {
    enabled: bool,
}

impl DataFusionOrderTrackingGuard {
    pub(in crate::exec) fn new(enabled: bool) -> Self {
        if enabled {
            DATAFUSION_ORDER_TRACKING_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        }
        Self { enabled }
    }
}

impl Drop for DataFusionOrderTrackingGuard {
    fn drop(&mut self) {
        if self.enabled {
            DATAFUSION_ORDER_TRACKING_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        }
    }
}

pub(in crate::exec) fn datafusion_order_tracking_enabled() -> bool {
    DATAFUSION_ORDER_TRACKING_DEPTH.with(|depth| depth.get() > 0)
}

pub(in crate::exec) fn plan_requires_datafusion_exact_stddev_order(plan: &QueryPlan) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => {
            aggregate.spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::StddevSamp)
        },
        _ => false,
    })
}

pub(in crate::exec) fn is_datafusion_order_column_name(name: &str) -> bool {
    name.starts_with(DATAFUSION_ORDER_COLUMN_PREFIX)
}

pub(in crate::exec) fn datafusion_order_columns(frame: &DataFrame) -> Vec<String> {
    if !datafusion_order_tracking_enabled() {
        return Vec::new();
    }
    frame
        .column_names()
        .into_iter()
        .filter(|name| is_datafusion_order_column_name(name))
        .map(ToString::to_string)
        .collect()
}

pub(in crate::exec) fn append_source_order_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    major_ordinal: usize,
) -> Result<DataFrame> {
    if !datafusion_order_tracking_enabled() || frame.contains(SOURCE_ORDER_MAJOR_COLUMN) {
        return Ok(frame);
    }
    let frame = frame.copy_into_runtime(context.runtime()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.datafusion_order.source",
            "failed to relocate native source frame before appending hidden order columns",
            error,
        )
    })?;
    let rows = frame.num_rows();
    let major = i64::try_from(major_ordinal).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.datafusion_order.source",
            "source order major ordinal exceeds Int64 range",
        )
        .with_fact("major_ordinal", major_ordinal)
    })?;
    let major = context.runtime().sequence::<i64>(major, 0, rows).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.datafusion_order.source",
            "failed to materialize native source order major column",
            error,
        )
        .with_fact("rows", rows)
    })?;
    let row = context.runtime().sequence::<i64>(0, 1, rows).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.datafusion_order.source",
            "failed to materialize native source order row column",
            error,
        )
        .with_fact("rows", rows)
    })?;
    frame
        .with_column(SOURCE_ORDER_MAJOR_COLUMN, major)
        .and_then(|frame| frame.with_column(SOURCE_ORDER_ROW_COLUMN, row))
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.datafusion_order.source",
                "failed to append hidden native source order columns",
                error,
            )
        })
}

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

pub(in crate::exec) fn append_datafusion_join_order_columns(
    columns: &mut Vec<(String, cudf::Series)>,
    left: Option<&DataFrame>,
    right: Option<&DataFrame>,
    join_type: JoinType,
) -> Result<()> {
    if !datafusion_order_tracking_enabled() {
        return Ok(());
    }

    let mut next_index = 0usize;
    match join_type {
        // DataFusion HashJoinExec currently preserves the right/probe side for
        // inner joins. The left/build side order breaks ties between multiple
        // matches for the same probe row.
        JoinType::Inner => {
            append_side_order_columns(columns, right, &mut next_index, "right")?;
            append_side_order_columns(columns, left, &mut next_index, "left")?;
        },
        JoinType::Left
        | JoinType::LeftSemi
        | JoinType::LeftAnti
        | JoinType::LeftMark
        | JoinType::Full
        | JoinType::Cross => {
            append_side_order_columns(columns, left, &mut next_index, "left")?;
            append_side_order_columns(columns, right, &mut next_index, "right")?;
        },
    }
    Ok(())
}

pub(in crate::exec) fn sort_by_datafusion_order_if_present(
    context: &ExecutionContext,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let order_columns = datafusion_order_columns(&frame);
    if order_columns.is_empty() {
        return Ok(frame);
    }
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame
        .with_execution(&overrides)
        .sort_values(order_columns.iter().map(String::as_str), CudfSortOptions::new().with_stable(true))
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.datafusion_order.sort",
                "failed to sort native stddev aggregate input by DataFusion order columns",
                error,
            )
            .with_fact("order_columns", order_columns.join(","))
        })?;
    metrics.operator_time += timer.elapsed();
    Ok(output)
}

pub(in crate::exec) fn schema_with_datafusion_order_columns(frame: &DataFrame, schema: &SchemaRef) -> SchemaRef {
    let order_columns = datafusion_order_columns(frame);
    if order_columns.is_empty() {
        return Arc::clone(schema);
    }
    let mut fields = schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    fields.extend(order_columns.iter().map(|name| Field::new(name, ArrowDataType::Int64, false)).collect::<Vec<_>>());
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
            QueryError::from_cudf(
                "cudf_query_engine.datafusion_order.join",
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
