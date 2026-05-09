use super::*;

pub(in crate::exec) fn cast_count_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_count_output_columns_impl(frame, spec, output_schema, None)
}

pub(in crate::exec) fn cast_count_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    cast_count_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_count_output_columns_impl(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    mut token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<DataFrame> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Count) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.cast_count",
                "count aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.cast_count",
                "count aggregate output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "cudf_query_engine.aggregate.cast_count",
                        "failed to cast bounded native count aggregate output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                metrics,
                "cudf_query_engine.aggregate.cast_count",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.cast_count",
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

pub(in crate::exec) fn cast_avg_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_avg_output_columns_impl(frame, spec, output_schema, None)
}

pub(in crate::exec) fn cast_avg_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    cast_avg_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_avg_output_columns_impl(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    mut token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<DataFrame> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.cast_avg",
                "avg aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.cast_avg",
                "avg aggregate output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "cudf_query_engine.aggregate.cast_avg",
                        "failed to cast bounded native avg aggregate output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                metrics,
                "cudf_query_engine.aggregate.cast_avg",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.cast_avg",
                    "failed to cast native avg aggregate output column",
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
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_reduction_output_columns_impl(frame, spec, output_schema, None)
}

pub(in crate::exec) fn cast_reduction_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    cast_reduction_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_reduction_output_columns_impl(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    mut token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
) -> Result<DataFrame> {
    for expr in spec.aggr_exprs.iter().filter(|expr| {
        matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
    }) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.cast_reduction",
                "reduction output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.cast_reduction",
                "reduction output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "cudf_query_engine.aggregate.cast_reduction",
                        "failed to cast bounded native reduction output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                metrics,
                "cudf_query_engine.aggregate.cast_reduction",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.cast_reduction",
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
