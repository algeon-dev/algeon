use super::*;
use crate::expr::ColumnRef;

pub(in crate::exec) fn validate_projection_capability(
    node: &ProjectionNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    let output_field_count = node.outputs.len();
    if output_field_count != output_schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.projection.validate",
            "projection output count must match mirrored output schema field count",
        )
        .with_fact("projection_output_count", output_field_count)
        .with_fact("output_field_count", output_schema.fields().len()));
    }

    if node.output_schema().as_ref() != output_schema.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.projection.validate",
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
    let frame = input.frame().clone();
    let timer = Instant::now();
    if let Some(output) = try_execute_column_only_projection(node, frame.clone(), None, metrics)? {
        metrics.operator_time += timer.elapsed();
        return GpuDataFrame::try_new_with_source_residency(output, &input);
    }
    let PreparedProjectionColumns { exprs, output_columns } = prepare_projection_columns(node, &frame)?;
    let output_columns = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let mut output =
        frame.with_execution(&overrides).with_columns(exprs).and_then(|frame| frame.select(output_columns)).map_err(
            |error| {
                QueryError::from_cudf(
                    "nexus_query_engine.projection.execute",
                    "failed to execute native cudf projection",
                    error,
                )
            },
        )?;
    output = coerce_projection_output_schema(output, node, None, metrics)?;
    metrics.operator_time += timer.elapsed();
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

pub(in crate::exec) fn execute_projection_with_budget(
    _context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.frame().clone();
    let timer = Instant::now();
    if let Some(output) = try_execute_column_only_projection(node, frame.clone(), Some(token), metrics)? {
        metrics.operator_time += timer.elapsed();
        return GpuDataFrame::try_new_with_source_residency(output, &input);
    }
    let PreparedProjectionColumns { exprs, output_columns } = prepare_projection_columns(node, &frame)?;
    let output_columns = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output = budgeted_value(
        frame.with_columns_with_budget(exprs, token).map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.projection.execute",
                "failed to execute bounded native cudf projection",
                error,
            )
        })?,
        metrics,
        "nexus_query_engine.projection.execute",
    )
    .select(output_columns)
    .map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.projection.execute",
            "failed to select bounded native cudf projection columns",
            error,
        )
    })?;
    output = coerce_projection_output_schema(output, node, Some(token), metrics)?;
    metrics.operator_time += timer.elapsed();
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

struct PreparedProjectionColumns<'a> {
    exprs: Vec<(&'a str, cudf::Expr)>,
    output_columns: Vec<String>,
}

fn prepare_projection_columns<'a>(
    node: &'a ProjectionNode,
    frame: &DataFrame,
) -> Result<PreparedProjectionColumns<'a>> {
    let exprs = node
        .outputs
        .iter()
        .map(|output| lower_projection_output_expr(output).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let mut output_columns = node.output_fields().map(|field| field.name().clone()).collect::<Vec<_>>();
    output_columns.extend(datafusion_order_columns(frame));
    Ok(PreparedProjectionColumns { exprs, output_columns })
}

fn try_execute_column_only_projection(
    node: &ProjectionNode,
    frame: DataFrame,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<Option<DataFrame>> {
    let mut selected_columns = Vec::with_capacity(node.outputs.len());
    let mut renames = Vec::new();
    for output in &node.outputs {
        let Some(column) = projection_output_source_column(&output.expr) else {
            return Ok(None);
        };
        let source_name = column.name().to_string();
        if selected_columns.iter().any(|selected| selected == &source_name) {
            return Ok(None);
        }
        let output_name = output.field.name().to_string();
        if source_name != output_name {
            renames.push((source_name.clone(), output_name));
        }
        selected_columns.push(source_name);
    }
    for order_column in datafusion_order_columns(&frame) {
        let order_column = order_column.to_string();
        if selected_columns.iter().any(|selected| selected == &order_column) {
            return Ok(None);
        }
        selected_columns.push(order_column);
    }
    let selected_refs = selected_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output = frame.select(selected_refs).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.projection.select",
            "failed to select native column-only projection",
            error,
        )
    })?;
    if !renames.is_empty() {
        let rename_refs = renames.iter().map(|(old, new)| (old.as_str(), new.as_str())).collect::<Vec<_>>();
        output = output.rename_columns(rename_refs).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.projection.rename",
                "failed to rename native column-only projection",
                error,
            )
        })?;
    }
    coerce_projection_output_schema(output, node, token, metrics).map(Some)
}

fn projection_output_source_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => projection_output_source_column(expr),
        _ => None,
    }
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
            frame_column_cudf_data_type(&frame, field_name, "nexus_query_engine.projection.schema_coerce")?;
        let expected_type = field.data_type();
        let target_type = cudf_dtype_from_arrow_type(expected_type).ok_or_else(|| {
            projection_schema_mismatch_error_with_budget(
                token,
                field_index,
                field_name,
                expected_type,
                expected_type,
                "expected_type_not_supported",
            )
        })?;
        if actual_dtype == target_type {
            continue;
        }
        let actual_type = cudf::interop::arrow_host::arrow_type_from_cudf_dtype(actual_dtype).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.projection.schema_coerce",
                "failed to map native projection column type to Arrow",
                error,
            )
            .with_fact("field_name", field_name)
            .with_fact("field_index", field_index)
        })?;
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
                            "nexus_query_engine.projection.schema_coerce",
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
                "nexus_query_engine.projection.schema_coerce",
            ),
            None => frame.cast_column(field_name, target_type).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.projection.schema_coerce",
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
        "nexus_query_engine.projection.schema_coerce",
        "native projection output cannot be coerced to its declared schema type",
    )
    .with_fact("reason", "projection_output_schema_mismatch")
    .with_fact("field_name", field_name)
    .with_fact("field_index", field_index)
    .with_fact("actual_type", format!("{actual_type:?}"))
    .with_fact("expected_type", format!("{expected_type:?}"))
}
