use super::{
    AggregateExpr, AggregateFunction, DataFrame, ExecutionBudgetToken, ExecutionContext, QueryError, QueryErrorCode,
    QueryMetrics, Result, aggregate_expr_is_exact_nunique, budgeted_value, global_count_star_expr, lower_column_expr,
};

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
    mut token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<cudf::Series> {
    let count = if expr.function == AggregateFunction::Count && !expr.distinct && global_count_star_expr(&expr.expr) {
        frame.num_rows()
    } else {
        let input_column = lower_column_expr(&expr.expr, "nexus_query_engine.global_aggregate.count_expr")?;
        let series = frame.series(input_column.as_str()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.global_aggregate.count_input",
                "failed to access native global count input column",
                error,
            )
            .with_fact("column", input_column.as_str())
        })?;
        if aggregate_expr_is_exact_nunique(expr) {
            let prepared = series
                .prepare_scalar_reduction(cudf::ReduceAgg::NUnique { null_policy: cudf::NullPolicy::Exclude })
                .map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.global_aggregate.count_distinct.prepare",
                        "failed to prepare native global distinct reduction",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("column", input_column.as_str())
                })?;
            let scalar = match token.as_mut() {
                Some((token, metrics)) => prepared.execute_with_budget(token).map(|outcome| {
                    budgeted_value(outcome, metrics, "nexus_query_engine.global_aggregate.count_distinct").into_scalar()
                }),
                None => prepared.execute().map(cudf::ScalarReductionOutput::into_scalar),
            }
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.global_aggregate.count_distinct",
                    "failed to count native global distinct values",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("column", input_column.as_str())
            })?;
            if !scalar.is_valid().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.global_aggregate.count_distinct",
                    "failed to read native global distinct scalar validity",
                    error,
                )
            })? {
                0
            } else {
                let count = scalar.value::<i32>().map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.global_aggregate.count_distinct",
                        "failed to read native global distinct scalar value",
                        error,
                    )
                })?;
                usize::try_from(count).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.global_aggregate.count_distinct",
                        "native global distinct count was negative",
                    )
                    .with_fact("distinct_count", count)
                })?
            }
        } else {
            series.len().saturating_sub(series.null_count())
        }
    };
    let count = i64::try_from(count).map_err(|_| {
        QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.global_aggregate.count",
            "global count exceeds i64 output range",
        )
        .with_fact("alias", expr.alias.as_str())
        .with_fact("count", count)
    })?;
    match token {
        Some((token, metrics)) => context
            .runtime()?
            .series_from_value_with_budget(count, 1, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.global_aggregate.count_output",
                    "failed to construct bounded native global count output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact_if_absent("budget_operation", "runtime::series_from_value_with_budget")
            })
            .map(|outcome| budgeted_value(outcome, metrics, "nexus_query_engine.global_aggregate.count_output")),
        None => {
            let scalar = context.runtime()?.scalar(count).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.global_aggregate.count",
                    "failed to construct native global count scalar",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            context.runtime()?.series_from_scalar(&scalar, 1).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.global_aggregate.count_output",
                    "failed to construct native global count output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
            })
        },
    }
}
