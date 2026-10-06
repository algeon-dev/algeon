use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use query_planning::prepared::avg_final_expr;

use super::{
    AggregateFunction, AggregateSpec, AvgStateAliases, DataFrame, ExecutionBudgetToken, ExecutionContext,
    ExecutionMetrics, QueryError, QueryErrorCode, Result, aggregate_spec_has_avg, budgeted_value, cudf_query_error,
    lower_expr, validate_avg_output_fields,
};

pub(in crate::exec) fn finalize_avg_output_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, None)
}

pub(in crate::exec) fn finalize_avg_output_columns_with_budget(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, Some((token, metrics)))
}

fn finalize_avg_output_columns_impl(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: Option<(&ExecutionBudgetToken, &ExecutionMetrics)>,
) -> Result<DataFrame> {
    if !aggregate_spec_has_avg(spec) {
        return Ok(frame);
    }

    validate_avg_output_fields(spec, output_schema)?;
    let avg_exprs = spec
        .aggr_exprs
        .iter()
        .enumerate()
        .filter(|(_, expr)| expr.function == AggregateFunction::Avg)
        .map(|(index, expr)| {
            let state = AvgStateAliases::for_aggregate_index(index);
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.aggregate.finalize_avg",
                    "avg aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            let avg = avg_final_expr(state.sum_expr(), state.count_expr(), field.data_type())?;
            lower_expr(&avg).map(|lowered| (expr.alias.clone(), lowered))
        })
        .collect::<Result<Vec<_>>>()?;
    match token {
        Some((token, metrics)) => token
            .run("dataframe::with_columns", |exec| frame.on(exec).with_columns(avg_exprs))
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.finalize_avg",
                    "failed to finalize bounded native avg aggregate output columns",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "query_engine.aggregate.finalize_avg")),
        None => {
            let exec_runtime = context.runtime()?;
            let overrides = exec_runtime.exec();
            frame.on(overrides).with_columns(avg_exprs).map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.finalize_avg",
                    "failed to finalize native avg aggregate output columns",
                    error,
                )
            })
        },
    }
}
