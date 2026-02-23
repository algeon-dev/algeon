use super::*;

mod avg;
mod casts;
mod count;
mod state;

pub(in crate::exec) use avg::{
    avg_sum_state_data_type, finalize_avg_output_columns, finalize_avg_output_columns_with_budget,
    validate_avg_output_fields, validate_avg_state_alias_collisions,
};
pub(in crate::exec) use casts::{
    cast_count_output_columns, cast_count_output_columns_with_budget, cast_reduction_output_columns,
    cast_reduction_output_columns_with_budget,
};
pub(in crate::exec) use count::{global_count_series, global_count_series_with_budget, global_count_star_expr};
pub(in crate::exec) use state::{
    final_global_aggregate_state_output_schema, global_valid_count_state_alias, partial_global_aggregate_output_schema,
    partial_global_aggregate_spec,
};

pub(in crate::exec) fn execute_global_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_global_aggregate_impl(context, spec, input, output_schema, None, metrics)
}

pub(in crate::exec) fn execute_global_aggregate_with_budget(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_global_aggregate_impl(context, spec, input, output_schema, Some(token), metrics)
}

fn execute_global_aggregate_impl(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_global_aggregate_capability(spec, output_schema)?;
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut columns = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "cudf_query_engine.global_aggregate.avg_state_type",
                        "avg aggregate output field is missing from output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                })?;
                let input_type = global_aggregate_frame_input_data_type(&frame, &expr.expr)?;
                let sum_state_type = avg_sum_state_data_type(&input_type, field.data_type())?;
                columns.push((
                    state.sum().to_string(),
                    reduce_global_aggregate_column_impl(
                        context,
                        &frame,
                        &overrides,
                        expr,
                        AggregateFunction::Sum,
                        Some(&sum_state_type),
                        token,
                        metrics,
                    )?,
                ));
                let count = match token {
                    Some(token) => global_count_series_with_budget(context, &frame, expr, token, metrics)?,
                    None => global_count_series(context, &frame, expr)?,
                };
                columns.push((state.count().to_string(), count));
            },
            AggregateFunction::Count | AggregateFunction::NUnique => {
                let count = match token {
                    Some(token) => global_count_series_with_budget(context, &frame, expr, token, metrics)?,
                    None => global_count_series(context, &frame, expr)?,
                };
                columns.push((expr.alias.clone(), count));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                columns.push((
                    expr.alias.clone(),
                    reduce_global_aggregate_column_impl(
                        context,
                        &frame,
                        &overrides,
                        expr,
                        expr.function,
                        output_schema.field_with_name(expr.alias.as_str()).ok().map(|field| field.data_type()),
                        token,
                        metrics,
                    )?,
                ));
            },
            AggregateFunction::StddevSamp => {
                unreachable!("global stddev_samp uses the DataFusion-exact Welford aggregate path")
            },
        }
    }
    let mut output = DataFrame::from_series(columns.iter().map(|(name, series)| (name.as_str(), series.clone())))
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.output",
                "failed to construct native global aggregate output frame",
                error,
            )
        })?;
    output = match token {
        Some(token) => cast_reduction_output_columns_with_budget(output, spec, output_schema, token, metrics)?,
        None => cast_reduction_output_columns(output, spec, output_schema)?,
    };
    output = match token {
        Some(token) => finalize_avg_output_columns_with_budget(context, output, spec, output_schema, token, metrics)?,
        None => finalize_avg_output_columns(context, output, spec, output_schema)?,
    };
    output = match token {
        Some(token) => cast_count_output_columns_with_budget(output, spec, output_schema, token, metrics)?,
        None => cast_count_output_columns(output, spec, output_schema)?,
    };
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.select",
            "failed to select native global aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}
fn reduce_global_aggregate_column_impl(
    context: &ExecutionContext,
    frame: &DataFrame,
    overrides: &ExecutionOverrides<'_>,
    expr: &AggregateExpr,
    function: AggregateFunction,
    output_type: Option<&ArrowDataType>,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<cudf::Series> {
    let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.lower_expr")?;
    let series = frame.series(input_column.as_str()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.input",
            "failed to access native global aggregate input column",
            error,
        )
        .with_fact("column", input_column.as_str())
    })?;
    if matches!(function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
        && series.len() == series.null_count()
        && let Some(output_type) = output_type
    {
        return match token {
            Some(token) => {
                global_null_output_series_with_budget(context, output_type, expr.alias.as_str(), token, metrics)
            },
            None => global_null_output_series(context, output_type, expr.alias.as_str()),
        };
    }
    let series = if matches!(function, AggregateFunction::Sum)
        && let Some(output_type) = output_type
    {
        let target_type = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.global_aggregate.cast_input",
                "global aggregate input cast target type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{output_type:?}"))
        })?;
        let casted = match token {
            Some(token) => budgeted_value(
                frame.cast_column_with_budget(input_column.as_str(), target_type, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "cudf_query_engine.global_aggregate.cast_input",
                        "failed to cast bounded native global aggregate input column before reduction",
                        error,
                    )
                    .with_fact("column", input_column.as_str())
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{output_type:?}"))
                })?,
                metrics,
                "cudf_query_engine.global_aggregate.cast_input",
            ),
            None => frame.cast_column(input_column.as_str(), target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.global_aggregate.cast_input",
                    "failed to cast native global aggregate input column before reduction",
                    error,
                )
                .with_fact("column", input_column.as_str())
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{output_type:?}"))
            })?,
        };
        casted.series(input_column.as_str()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.cast_input",
                "failed to access casted native global aggregate input column",
                error,
            )
            .with_fact("column", input_column.as_str())
            .with_fact("alias", expr.alias.as_str())
        })?
    } else {
        series
    };
    let scalar = match token {
        Some(token) => execute_scalar_reduction_with_budget(&series, function, token, input_column.as_str(), metrics)?,
        None => execute_scalar_reduction(&series, overrides, function, input_column.as_str())?,
    };
    match token {
        Some(token) => context
            .runtime()
            .series_from_scalar_with_budget(&scalar, 1, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.global_aggregate.output",
                    "failed to construct bounded native global aggregate output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact_if_absent("budget_operation", "runtime::series_from_scalar_with_budget")
            })
            .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.global_aggregate.output")),
        None => context.runtime().series_from_scalar(&scalar, 1).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.output",
                "failed to construct native global aggregate output column",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
        }),
    }
}

fn execute_scalar_reduction(
    series: &cudf::Series,
    overrides: &ExecutionOverrides<'_>,
    function: AggregateFunction,
    input_column: &str,
) -> Result<CudfScalar> {
    match function {
        AggregateFunction::Sum => series.sum_in(overrides),
        AggregateFunction::Min => series.min_in(overrides),
        AggregateFunction::Max => series.max_in(overrides),
        AggregateFunction::Avg => unreachable!("global AVG is lowered through explicit sum/count state"),
        AggregateFunction::Count | AggregateFunction::NUnique => {
            unreachable!("global count-like aggregates are not scalar reductions")
        },
        AggregateFunction::StddevSamp => {
            unreachable!("global STDDEV_SAMP uses the DataFusion-exact Welford aggregate path")
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.reduce",
            "failed to execute native global aggregate reduction",
            error,
        )
        .with_fact("column", input_column)
        .with_fact("function", format!("{function:?}"))
    })
}

fn execute_scalar_reduction_with_budget(
    series: &cudf::Series,
    function: AggregateFunction,
    token: &ExecutionBudgetToken,
    input_column: &str,
    metrics: &mut QueryMetrics,
) -> Result<CudfScalar> {
    let reduce_agg = match function {
        AggregateFunction::Sum => ReduceAgg::Sum,
        AggregateFunction::Min => ReduceAgg::Min,
        AggregateFunction::Max => ReduceAgg::Max,
        AggregateFunction::Avg => unreachable!("global AVG is lowered through explicit sum/count state"),
        AggregateFunction::Count | AggregateFunction::NUnique => {
            unreachable!("global count-like aggregates are not scalar reductions")
        },
        AggregateFunction::StddevSamp => {
            unreachable!("global STDDEV_SAMP uses the DataFusion-exact Welford aggregate path")
        },
    };
    let prepared = series.prepare_scalar_reduction(reduce_agg).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.prepare_reduce",
            "failed to prepare native global aggregate reduction",
            error,
        )
        .with_fact("column", input_column)
        .with_fact("function", format!("{function:?}"))
    })?;
    prepared
        .execute_with_budget(token)
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "cudf_query_engine.global_aggregate.reduce",
                "failed to execute bounded native global aggregate reduction",
                error,
            )
            .with_fact("column", input_column)
            .with_fact("function", format!("{function:?}"))
        })
        .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.global_aggregate.reduce").into_scalar())
}

fn global_null_output_series(
    context: &ExecutionContext,
    output_type: &ArrowDataType,
    alias: &str,
) -> Result<cudf::Series> {
    global_null_output_series_impl(context, output_type, alias, None)
}

fn global_null_output_series_with_budget(
    context: &ExecutionContext,
    output_type: &ArrowDataType,
    alias: &str,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<cudf::Series> {
    global_null_output_series_impl(context, output_type, alias, Some((token, metrics)))
}

fn global_null_output_series_impl(
    context: &ExecutionContext,
    output_type: &ArrowDataType,
    alias: &str,
    token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<cudf::Series> {
    let dtype = cudf_dtype_from_arrow_type(output_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.global_aggregate.null_output",
            "native global aggregate cannot construct a typed null output for this type",
        )
        .with_fact("alias", alias)
        .with_fact("data_type", format!("{output_type:?}"))
    })?;
    match token {
        Some((token, metrics)) => context
            .runtime()
            .nulls_with_budget(dtype, 1, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.global_aggregate.null_output",
                    "failed to construct bounded native global aggregate null output column",
                    error,
                )
                .with_fact("alias", alias)
                .with_fact("data_type", format!("{output_type:?}"))
                .with_fact_if_absent("budget_operation", "runtime::nulls_with_budget")
            })
            .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.global_aggregate.null_output")),
        None => context.runtime().nulls(dtype, 1).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.null_output",
                "failed to construct native global aggregate null output column",
                error,
            )
            .with_fact("alias", alias)
            .with_fact("data_type", format!("{output_type:?}"))
        }),
    }
}

fn global_aggregate_frame_input_data_type(frame: &DataFrame, expr: &Expr) -> Result<ArrowDataType> {
    let input_column = lower_column_expr(expr, "cudf_query_engine.global_aggregate.avg_state_type")?;
    frame_column_arrow_data_type(frame, input_column.as_str(), "cudf_query_engine.global_aggregate.avg_state_type")
}

fn frame_column_arrow_data_type(frame: &DataFrame, column: &str, operation: &'static str) -> Result<ArrowDataType> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(operation, "failed to access native column while inspecting its type", error)
            .with_fact("column", column)
    })?;
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(operation, "failed to inspect native column type", error).with_fact("column", column)
    })?;
    cudf::interop::arrow_host::arrow_type_from_cudf_dtype(dtype).map_err(|error| {
        QueryError::from_cudf(operation, "failed to map native column type to Arrow", error).with_fact("column", column)
    })
}
