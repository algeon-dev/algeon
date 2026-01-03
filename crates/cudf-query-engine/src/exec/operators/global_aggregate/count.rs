use super::*;

pub(in crate::exec) fn global_count_series(
    context: &ExecutionContext,
    frame: &DataFrame,
    expr: &AggregateExpr,
) -> Result<cudf::Series> {
    global_count_series_impl(context, frame, expr, None)
}

pub(in crate::exec) fn global_count_series_with_budget(
    context: &ExecutionContext,
    frame: &DataFrame,
    expr: &AggregateExpr,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<cudf::Series> {
    global_count_series_impl(context, frame, expr, Some((token, metrics)))
}

fn global_count_series_impl(
    context: &ExecutionContext,
    frame: &DataFrame,
    expr: &AggregateExpr,
    token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<cudf::Series> {
    let count = if global_count_star_expr(&expr.expr) {
        frame.num_rows()
    } else {
        let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.count_expr")?;
        let series = frame.series(input_column.as_str()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.count_input",
                "failed to access native global count input column",
                error,
            )
            .with_fact("column", input_column.as_str())
        })?;
        series.len().saturating_sub(series.null_count())
    };
    let count = i64::try_from(count).map_err(|_| {
        QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.global_aggregate.count",
            "global count exceeds i64 output range",
        )
        .with_fact("alias", expr.alias.as_str())
        .with_fact("count", count)
    })?;
    match token {
        Some((token, metrics)) => context
            .runtime()
            .series_from_value_with_budget(count, 1, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.global_aggregate.count_output",
                    "failed to construct bounded native global count output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact_if_absent("budget_operation", "runtime::series_from_value_with_budget")
            })
            .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.global_aggregate.count_output")),
        None => {
            let scalar = context.runtime().scalar(count).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.global_aggregate.count",
                    "failed to construct native global count scalar",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            context.runtime().series_from_scalar(&scalar, 1).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.global_aggregate.count_output",
                    "failed to construct native global count output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
            })
        },
    }
}

pub(in crate::exec) fn global_count_star_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal(
            LiteralValue::Boolean(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::Int64(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
                | LiteralValue::UInt64(_)
                | LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Utf8(_)
                | LiteralValue::LargeUtf8(_)
                | LiteralValue::Decimal128 { .. }
                | LiteralValue::Date32(_)
                | LiteralValue::Timestamp { .. }
                | LiteralValue::Duration { .. }
                | LiteralValue::IntervalMonthDayNano { .. },
        )
    )
}
