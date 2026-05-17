use super::*;

pub(in crate::exec) fn validate_projection_capability(
    node: &ProjectionNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    let output_field_count = node.outputs.len();
    if output_field_count != output_schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection output count must match mirrored output schema field count",
        )
        .with_fact("projection_output_count", output_field_count)
        .with_fact("output_field_count", output_schema.fields().len()));
    }

    if node.output_schema().as_ref() != output_schema.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection output schema facts must mirror typed projection outputs",
        )
        .with_fact("projection_output_schema", format!("{:?}", node.output_schema()))
        .with_fact("facts_output_schema", format!("{output_schema:?}")));
    }

    for output in &node.outputs {
        lower_projection_output_expr(output)?;
    }

    Ok(())
}

pub(in crate::exec) fn execute_projection(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.into_frame();
    let output_names = node.output_fields().map(|field| field.name().clone()).collect::<Vec<_>>();
    let exprs = node
        .outputs
        .iter()
        .map(|output| lower_projection_output_expr(output).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let mut output_columns = output_names;
    output_columns.extend(datafusion_order_columns(&frame));
    let output_columns = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output =
        frame.with_execution(&overrides).with_columns(exprs).and_then(|frame| frame.select(output_columns)).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.projection.execute",
                    "failed to execute native cudf projection",
                    error,
                )
            },
        )?;
    output = coerce_projection_output_schema(output, node, None, metrics)?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_projection_with_budget(
    _context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.into_frame();
    let output_names = node.output_fields().map(|field| field.name().clone()).collect::<Vec<_>>();
    let exprs = node
        .outputs
        .iter()
        .map(|output| lower_projection_output_expr(output).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let mut output_columns = output_names;
    output_columns.extend(datafusion_order_columns(&frame));
    let output_columns = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let timer = Instant::now();
    let mut output = budgeted_value(
        frame.with_columns_with_budget(exprs, token).map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "cudf_query_engine.projection.execute",
                "failed to execute bounded native cudf projection",
                error,
            )
        })?,
        metrics,
        "cudf_query_engine.projection.execute",
    )
    .select(output_columns)
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.projection.execute",
            "failed to select bounded native cudf projection columns",
            error,
        )
    })?;
    output = coerce_projection_output_schema(output, node, Some(token), metrics)?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn coerce_projection_output_schema(
    mut frame: DataFrame,
    node: &ProjectionNode,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    for (field_index, field) in node.output_fields().enumerate() {
        let field_name = field.name();
        let actual_dtype =
            frame_column_cudf_data_type(&frame, field_name, "cudf_query_engine.projection.schema_coerce")?;
        let actual_type = cudf::interop::arrow_host::arrow_type_from_cudf_dtype(actual_dtype).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.projection.schema_coerce",
                "failed to map native projection column type to Arrow",
                error,
            )
            .with_fact("field_name", field_name)
            .with_fact("field_index", field_index)
        })?;
        let expected_type = field.data_type();
        let target_type = cudf_dtype_from_arrow_type(expected_type).ok_or_else(|| {
            projection_schema_mismatch_error_with_budget(
                token,
                field_index,
                field_name,
                &actual_type,
                expected_type,
                "expected_type_not_supported",
            )
        })?;
        if actual_dtype == target_type {
            continue;
        }
        if !is_supported_arrow_implicit_cast(&actual_type, expected_type) {
            return Err(projection_schema_mismatch_error_with_budget(
                token,
                field_index,
                field_name,
                &actual_type,
                expected_type,
                "implicit_cast_not_supported",
            ));
        }
        frame = match token {
            Some(token) => budgeted_value(
                frame.cast_column_with_budget(field_name, target_type, token).map_err(|error| {
                    attach_budget_token_failure_facts(
                        QueryError::from_cudf_with_budget_details(
                            "cudf_query_engine.projection.schema_coerce",
                            "failed to cast bounded native projection output column to declared schema type",
                            error,
                        ),
                        token,
                    )
                    .with_fact_if_absent("reason", "projection_output_schema_mismatch")
                    .with_fact("field_name", field_name)
                    .with_fact("field_index", field_index)
                    .with_fact("actual_type", format!("{actual_type:?}"))
                    .with_fact("expected_type", format!("{expected_type:?}"))
                })?,
                metrics,
                "cudf_query_engine.projection.schema_coerce",
            ),
            None => frame.cast_column(field_name, target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.projection.schema_coerce",
                    "failed to cast native projection output column to declared schema type",
                    error,
                )
                .with_fact("reason", "projection_output_schema_mismatch")
                .with_fact("field_name", field_name)
                .with_fact("field_index", field_index)
                .with_fact("actual_type", format!("{actual_type:?}"))
                .with_fact("expected_type", format!("{expected_type:?}"))
            })?,
        };
    }
    Ok(frame)
}

fn frame_column_cudf_data_type(frame: &DataFrame, column: &str, operation: &'static str) -> Result<cudf::DataType> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(operation, "failed to access native column while inspecting its type", error)
            .with_fact("column", column)
    })?;
    series.dtype().map_err(|error| {
        QueryError::from_cudf(operation, "failed to inspect native column type", error).with_fact("column", column)
    })
}

fn projection_schema_mismatch_error_with_budget(
    token: Option<&ExecutionBudgetToken>,
    field_index: usize,
    field_name: &str,
    actual_type: &ArrowDataType,
    expected_type: &ArrowDataType,
    reason_detail: &'static str,
) -> QueryError {
    let error = projection_schema_mismatch_error(field_index, field_name, actual_type, expected_type)
        .with_fact("reason_detail", reason_detail);
    match token {
        Some(token) => attach_budget_token_failure_facts(error, token),
        None => error,
    }
}

fn projection_schema_mismatch_error(
    field_index: usize,
    field_name: &str,
    actual_type: &ArrowDataType,
    expected_type: &ArrowDataType,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "cudf_query_engine.projection.schema_coerce",
        "native projection output cannot be coerced to its declared schema type",
    )
    .with_fact("reason", "projection_output_schema_mismatch")
    .with_fact("field_name", field_name)
    .with_fact("field_index", field_index)
    .with_fact("actual_type", format!("{actual_type:?}"))
    .with_fact("expected_type", format!("{expected_type:?}"))
}
