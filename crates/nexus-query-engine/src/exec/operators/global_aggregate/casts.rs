use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{
    AggregateFunction, AggregateSpec, DataFrame, ExecutionBudgetToken, ExecutionMetrics, QueryError, QueryErrorCode,
    Result, budgeted_value, cudf_dtype_from_arrow_type, cudf_query_error, cudf_query_error_with_budget_details,
};

pub(in crate::exec) fn cast_count_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<DataFrame> {
    cast_count_output_columns_impl(frame, spec, output_schema, None)
}

pub(in crate::exec) fn cast_count_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    cast_count_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_count_output_columns_impl(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    mut token: Option<(&ExecutionBudgetToken, &ExecutionMetrics)>,
) -> Result<DataFrame> {
    for expr in spec
        .aggr_exprs
        .iter()
        .filter(|expr| matches!(expr.function, AggregateFunction::Count | AggregateFunction::NUnique))
    {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.cast_count",
                "count aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.aggregate.cast_count",
                "count aggregate output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(expr.alias.as_str()).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.aggregate.cast_count",
                "failed to inspect native count aggregate output column type",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        if actual_type == target_type {
            continue;
        }
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    cudf_query_error_with_budget_details(
                        "nexus_query_engine.aggregate.cast_count",
                        "failed to cast bounded native count aggregate output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                metrics,
                "nexus_query_engine.aggregate.cast_count",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.aggregate.cast_count",
                    "failed to cast native count aggregate output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?,
        };
    }
    Ok(frame)
}

pub(in crate::exec) fn cast_reduction_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<DataFrame> {
    cast_reduction_output_columns_impl(frame, spec, output_schema, None)
}

pub(in crate::exec) fn cast_reduction_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    cast_reduction_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_reduction_output_columns_impl(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    mut token: Option<(&ExecutionBudgetToken, &ExecutionMetrics)>,
) -> Result<DataFrame> {
    for expr in spec.aggr_exprs.iter().filter(|expr| {
        matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            || matches!(expr.function, AggregateFunction::Median)
            || expr.function.is_floating_statistical()
    }) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.cast_reduction",
                "reduction output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.aggregate.cast_reduction",
                "reduction output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(expr.alias.as_str()).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.aggregate.cast_reduction",
                "failed to inspect native reduction output column type",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        if actual_type == target_type {
            continue;
        }
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    cudf_query_error_with_budget_details(
                        "nexus_query_engine.aggregate.cast_reduction",
                        "failed to cast bounded native reduction output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                metrics,
                "nexus_query_engine.aggregate.cast_reduction",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.aggregate.cast_reduction",
                    "failed to cast native reduction output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?,
        };
    }
    Ok(frame)
}
