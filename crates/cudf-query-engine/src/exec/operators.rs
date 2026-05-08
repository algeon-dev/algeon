use super::*;

pub(super) fn validate_projection_capability(
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

    for expr in node.expressions() {
        lower_expr(expr)?;
    }

    Ok(())
}

pub(super) fn validate_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    Ok(())
}

pub(super) fn validate_final_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    let aggregate = lower_final_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_state_alias_collisions(&node.spec, "cudf_query_engine.final_merge.validate_avg_state_alias")?;
        validate_avg_output_fields(&node.spec, output_schema)?;
        return Ok(());
    }
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.final_merge.validate",
        "final aggregate output names must match output schema fields",
    )
}

pub(super) fn validate_global_aggregate_capability(
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if aggregate_spec_has_avg(spec) {
        validate_avg_state_alias_collisions(spec, "cudf_query_engine.global_aggregate.validate_avg_state_alias")?;
    }
    let expected_output_names = spec.aggr_exprs.iter().map(|expr| expr.alias.clone()).collect::<Vec<_>>();
    let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if expected_output_names != schema_output_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.global_aggregate.validate",
            "global aggregate output names must match output schema fields",
        )
        .with_fact("aggregate_output_names", expected_output_names.join(","))
        .with_fact("schema_output_names", schema_output_names.join(",")));
    }
    for expr in &spec.aggr_exprs {
        if expr.function == AggregateFunction::Avg {
            validate_avg_output_fields(spec, output_schema)?;
        }
        validate_global_aggregate_input_expr(expr)?;
    }
    Ok(())
}

pub(super) fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if global_count_star_expr(&expr.expr) => Ok(()),
        _ => lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.validate_expr").map(|_| ()),
    }
}

pub(super) fn execution_budget_token_for_boundary(
    context: &ExecutionContext,
    budget_bytes: u64,
    budget_boundary: &'static str,
    operation: &'static str,
    message: &'static str,
) -> Result<ExecutionBudgetToken> {
    let budget = usize::try_from(budget_bytes).map_err(|_error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("budget_boundary", budget_boundary)
            .with_fact("budget_bytes", budget_bytes)
    })?;
    context
        .runtime()
        .execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(budget_boundary))
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(operation, message, error)
                .with_fact_if_absent("budget_boundary", budget_boundary)
                .with_fact_if_absent("budget_bytes", budget_bytes)
        })
}

pub(super) fn attach_budget_token_failure_facts(error: QueryError, token: &ExecutionBudgetToken) -> QueryError {
    error
        .with_fact_if_absent("budget_boundary", token.request().boundary())
        .with_fact_if_absent("budget_bytes", token.request().budget_bytes())
        .with_fact_if_absent("budget_enforcement", ExecutionBudgetEnforcement::HardMemoryResourceLimit.as_str())
        .with_fact_if_absent("budget_status", "exceeded_or_lower_layer_rejected")
}

fn budgeted_value<T>(outcome: BudgetedExecutionOutcome<T>, metrics: &mut QueryMetrics, operation: &'static str) -> T {
    let (value, report) = outcome.into_parts();
    metrics.record_budgeted_execution(operation, &report);
    value
}

pub(super) fn validate_aggregate_output_names(
    aggregate: &LoweredAggregate,
    output_schema: &arrow::datatypes::SchemaRef,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if aggregate.output_names != expected_output_names {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("aggregate_output_names", aggregate.output_names.join(","))
            .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    Ok(())
}

pub(super) fn execute_filter(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let predicate = lower_expr(&node.predicate)?;
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.filter.execute", "failed to execute native cudf filter", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_filter_with_budget(
    _context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let predicate = lower_expr(&node.predicate)?;
    let frame = input.into_frame();
    let overrides = token.execution_overrides();
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf("cudf_query_engine.filter.execute", "failed to execute native cudf filter", error),
            token,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_projection(
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
        .map(|output| lower_expr(&output.expr).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let output_columns = output_names.iter().map(String::as_str).collect::<Vec<_>>();
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

pub(super) fn execute_projection_with_budget(
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
        .map(|output| lower_expr(&output.expr).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let output_columns = output_names.iter().map(String::as_str).collect::<Vec<_>>();
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

pub(super) fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        return execute_global_aggregate(context, &node.spec, input, output_schema, metrics);
    }

    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "cudf_query_engine.aggregate.execute")?;
    output = cast_count_output_columns(output, &node.spec, output_schema)?;
    output = cast_avg_output_columns(output, &node.spec, output_schema)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        return execute_global_aggregate_with_budget(context, &node.spec, input, output_schema, token, metrics);
    }

    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute",
        "failed to execute bounded native cudf aggregate",
    )?;
    output = cast_count_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
    output = cast_avg_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        let partial_spec = partial_global_aggregate_spec(&node.spec);
        let output_schema = partial_global_aggregate_output_schema(&node.spec, input_schema, aggregate_output_schema)?;
        return execute_global_aggregate_with_budget(context, &partial_spec, input, &output_schema, token, metrics);
    }

    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    let expected_output_names = aggregate.output_names.clone();
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute_partial",
        "failed to execute bounded native partial cudf aggregate",
    )?;
    output = cast_count_output_columns_with_budget(output, &node.spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_partial",
            "failed to select native partial aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_groupby_aggregate(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct(overrides, frame, aggregate, operation);
    }

    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .with_execution(overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| QueryError::from_cudf(operation, "failed to execute native cudf aggregate", error))?;
    rename_groupby_key_columns(&mut output, aggregate, "cudf_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

pub(super) fn execute_groupby_aggregate_with_budget(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct_with_budget(overrides, frame, aggregate, token, operation, message);
    }

    let mut output = budgeted_value(
        frame
            .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
            .options(GroupByOptions::builder().include_null_keys(true).build())
            .agg_with_budget(&aggregate.aggregations, token)
            .map_err(|error| QueryError::from_cudf_with_budget_details(operation, message, error))?,
        metrics,
        operation,
    );
    rename_groupby_key_columns(&mut output, aggregate, "cudf_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

fn rename_groupby_key_columns(
    output: &mut DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<()> {
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        *output = output
            .rename_columns(key_renames)
            .map_err(|error| QueryError::from_cudf(operation, "failed to rename native cudf aggregate keys", error))?;
    }
    Ok(())
}

fn execute_groupby_distinct(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output = frame
        .with_execution(overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to execute native cudf group-only aggregate", error)
        })?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    output.select(aggregate.output_key_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_group_only_keys",
            "failed to select native group-only aggregate keys",
            error,
        )
    })
}

fn execute_groupby_distinct_with_budget(
    _overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let budget_overrides = token.execution_overrides();
    let mut output = frame
        .with_execution(&budget_overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| attach_budget_token_failure_facts(QueryError::from_cudf(operation, message, error), token))?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    output.select(aggregate.output_key_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_group_only_keys",
            "failed to select native group-only aggregate keys",
            error,
        )
    })
}

pub(super) fn execute_global_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_global_aggregate_impl(context, spec, input, output_schema, None, metrics)
}

pub(super) fn execute_global_aggregate_with_budget(
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
            AggregateFunction::Count => {
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
        Some(token) => cast_global_reduction_output_columns_with_budget(output, spec, output_schema, token, metrics)?,
        None => cast_global_reduction_output_columns(output, spec, output_schema)?,
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
        AggregateFunction::Count => unreachable!("global COUNT is not a scalar reduction"),
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
        AggregateFunction::Count => unreachable!("global COUNT is not a scalar reduction"),
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

pub(super) fn global_count_series(
    context: &ExecutionContext,
    frame: &DataFrame,
    expr: &AggregateExpr,
) -> Result<cudf::Series> {
    global_count_series_impl(context, frame, expr, None)
}

pub(super) fn global_count_series_with_budget(
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

pub(super) fn global_count_star_expr(expr: &Expr) -> bool {
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

pub(super) fn partial_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, expr.expr.clone(), state.sum().to_string()),
                        AggregateExpr::new(AggregateFunction::Count, expr.expr.clone(), state.count().to_string()),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => vec![
                    expr.clone(),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        expr.expr.clone(),
                        global_valid_count_state_alias(expr.alias.as_str()),
                    ),
                ],
                _ => vec![expr.clone()],
            })
            .collect(),
    )
}

pub(super) fn partial_global_aggregate_output_schema(
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
) -> Result<Arc<Schema>> {
    let mut fields = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                let input_type = global_aggregate_input_data_type(&expr.expr, input_schema)?;
                let output_field = global_aggregate_output_field(expr.alias.as_str(), aggregate_output_schema)?;
                let sum_state_type = avg_sum_state_data_type(&input_type, output_field.data_type())?;
                fields.push(Field::new(state.sum(), sum_state_type, true));
                fields.push(Field::new(state.count(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Count => {
                fields.push(Field::new(expr.alias.as_str(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                fields.push(global_aggregate_output_field(expr.alias.as_str(), aggregate_output_schema)?);
                fields.push(Field::new(
                    global_valid_count_state_alias(expr.alias.as_str()),
                    ArrowDataType::Int64,
                    false,
                ));
            },
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn final_global_aggregate_state_output_schema(
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
) -> Result<Arc<Schema>> {
    partial_global_aggregate_output_schema(spec, input_schema, aggregate_output_schema)
}

fn global_aggregate_input_data_type(expr: &Expr, input_schema: &arrow::datatypes::SchemaRef) -> Result<ArrowDataType> {
    let input_column = lower_column_expr(expr, "cudf_query_engine.global_aggregate.state_schema")?;
    let field = input_schema.field_with_name(input_column.as_str()).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.global_aggregate.state_schema",
            "global aggregate state references a column missing from its input schema",
        )
        .with_fact("column", input_column.as_str())
        .with_fact("arrow_error", error.to_string())
    })?;
    Ok(field.data_type().clone())
}

fn global_aggregate_output_field(alias: &str, output_schema: &arrow::datatypes::SchemaRef) -> Result<Field> {
    output_schema.field_with_name(alias).cloned().map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.global_aggregate.state_schema",
            "global aggregate state references a column missing from its output schema",
        )
        .with_fact("alias", alias)
        .with_fact("arrow_error", error.to_string())
    })
}

pub(super) fn cast_count_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_count_output_columns_impl(frame, spec, output_schema, None)
}

pub(super) fn cast_count_output_columns_with_budget(
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
                *metrics,
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

pub(super) fn cast_avg_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_avg_output_columns_impl(frame, spec, output_schema, None)
}

pub(super) fn cast_avg_output_columns_with_budget(
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
                *metrics,
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

pub(super) fn cast_global_reduction_output_columns(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    cast_global_reduction_output_columns_impl(frame, spec, output_schema, None)
}

pub(super) fn cast_global_reduction_output_columns_with_budget(
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    cast_global_reduction_output_columns_impl(frame, spec, output_schema, Some((token, metrics)))
}

fn cast_global_reduction_output_columns_impl(
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
                "global reduction output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.cast_reduction",
                "global reduction output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        frame = match token.as_mut() {
            Some((token, metrics)) => budgeted_value(
                frame.cast_column_with_budget(expr.alias.as_str(), target_type, token).map_err(|error| {
                    QueryError::from_cudf_with_budget_details(
                        "cudf_query_engine.aggregate.cast_reduction",
                        "failed to cast bounded native global reduction output column",
                        error,
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                })?,
                *metrics,
                "cudf_query_engine.aggregate.cast_reduction",
            ),
            None => frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.cast_reduction",
                    "failed to cast native global reduction output column",
                    error,
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?,
        };
    }
    Ok(frame)
}

pub(super) fn finalize_avg_output_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, None)
}

pub(super) fn finalize_avg_output_columns_with_budget(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, Some((token, metrics)))
}

fn finalize_avg_output_columns_impl(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
    token: Option<(&ExecutionBudgetToken, &mut QueryMetrics)>,
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
                    "cudf_query_engine.aggregate.finalize_avg",
                    "avg aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            let state_sum_type =
                frame_column_arrow_data_type(&frame, state.sum(), "cudf_query_engine.aggregate.finalize_avg")?;
            let division_type = avg_division_data_type(&state_sum_type, field.data_type())?;
            let sum = Expr::cast(state.sum_expr(), division_type.clone());
            let count = Expr::cast(state.count_expr(), avg_count_output_data_type(&division_type));
            let alias = expr.alias.clone();
            let count_is_zero = Expr::binary(state.count_expr(), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(0)));
            let avg = Expr::cast(Expr::binary(sum, BinaryOp::Divide, count), field.data_type().clone());
            let nullable_value = Expr::case(
                vec![crate::expr::CaseWhen::new(
                    count_is_zero,
                    Expr::literal(LiteralValue::Null(field.data_type().clone())),
                )],
                avg,
            );
            lower_expr(&nullable_value).map(|lowered| (alias, lowered))
        })
        .collect::<Result<Vec<_>>>()?;
    match token {
        Some((token, metrics)) => frame
            .with_columns_with_budget(avg_exprs, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.aggregate.finalize_avg",
                    "failed to finalize bounded native avg aggregate output columns",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.aggregate.finalize_avg")),
        None => {
            let overrides = execution_overrides(context);
            frame.with_execution(&overrides).with_columns(avg_exprs).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.finalize_avg",
                    "failed to finalize native avg aggregate output columns",
                    error,
                )
            })
        },
    }
}

pub(super) fn global_valid_count_state_alias(alias: &str) -> String {
    format!("{GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX}{alias}")
}

const GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX: &str = "__cudf_global_valid_count_";

fn avg_sum_state_data_type(input_type: &ArrowDataType, avg_output_type: &ArrowDataType) -> Result<ArrowDataType> {
    match avg_output_type {
        ArrowDataType::Float32 | ArrowDataType::Float64 => Ok(ArrowDataType::Float64),
        ArrowDataType::Decimal128(_, output_scale) => match input_type {
            ArrowDataType::Decimal128(_, input_scale) => {
                Ok(ArrowDataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *input_scale))
            },
            ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64 => {
                Ok(ArrowDataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale))
            },
            _ => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.avg_state_type",
                "avg sum state does not support this input type for Decimal128 output",
            )
            .with_fact("input_type", format!("{input_type:?}"))
            .with_fact("output_type", format!("{avg_output_type:?}"))),
        },
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.avg_state_type",
            "avg sum state output schema must be Float32, Float64, or Decimal128",
        )
        .with_fact("input_type", format!("{input_type:?}"))
        .with_fact("output_type", format!("{avg_output_type:?}"))),
    }
}

fn avg_division_data_type(state_sum_type: &ArrowDataType, avg_output_type: &ArrowDataType) -> Result<ArrowDataType> {
    match avg_output_type {
        ArrowDataType::Float32 | ArrowDataType::Float64 => Ok(ArrowDataType::Float64),
        ArrowDataType::Decimal128(_, output_scale) => match state_sum_type {
            ArrowDataType::Decimal128(_, _)
            | ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64 => {
                Ok(ArrowDataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale))
            },
            _ => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.avg_division_type",
                "avg division does not support this sum state type for Decimal128 output",
            )
            .with_fact("state_sum_type", format!("{state_sum_type:?}"))
            .with_fact("output_type", format!("{avg_output_type:?}"))),
        },
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.avg_division_type",
            "avg division output schema must be Float32, Float64, or Decimal128",
        )
        .with_fact("state_sum_type", format!("{state_sum_type:?}"))
        .with_fact("output_type", format!("{avg_output_type:?}"))),
    }
}

fn avg_count_output_data_type(avg_output_type: &ArrowDataType) -> ArrowDataType {
    match avg_output_type {
        ArrowDataType::Decimal128(_, _) => ArrowDataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, 0),
        _ => avg_output_type.clone(),
    }
}

pub(super) fn validate_avg_state_alias_collisions(spec: &AggregateSpec, operation: &'static str) -> Result<()> {
    if !aggregate_spec_has_avg(spec) {
        return Ok(());
    }

    let reserved_aliases = spec
        .aggr_exprs
        .iter()
        .enumerate()
        .filter(|(_, expr)| expr.function == AggregateFunction::Avg)
        .flat_map(|(index, _)| AvgStateAliases::for_aggregate_index(index).output_names())
        .collect::<Vec<_>>();
    let visible_aliases = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, operation).map(|(_, alias)| alias))
        .chain(spec.aggr_exprs.iter().map(|expr| Ok(expr.alias.clone())))
        .collect::<Result<Vec<_>>>()?;
    for alias in visible_aliases {
        if reserved_aliases.iter().any(|reserved| reserved == &alias) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "visible aggregate output alias conflicts with reserved internal AVG state column",
            )
            .with_fact("alias", alias)
            .with_fact("reserved_aliases", reserved_aliases.join(",")));
        }
    }

    Ok(())
}

pub(super) fn validate_avg_output_fields(
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.validate_avg",
                "avg aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        if !matches!(
            field.data_type(),
            ArrowDataType::Float32 | ArrowDataType::Float64 | ArrowDataType::Decimal128(_, _)
        ) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.validate_avg",
                "avg aggregate output schema must be Float32, Float64, or Decimal128",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    Ok(())
}

pub(super) fn execute_sort(
    context: &ExecutionContext,
    node: &SortNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_sort_on_frame(context, node, input.into_frame(), metrics)
}

pub(super) fn execute_sort_on_frame(
    context: &ExecutionContext,
    node: &SortNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&node.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).sort_values(sort.columns, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.sort.execute", "failed to execute native cudf sort", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_limit(
    context: &ExecutionContext,
    node: &LimitNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, node.spec)? {
        return Ok(input);
    }

    execute_limit_on_frame(context, node, input.into_frame(), metrics)
}

pub(super) fn execute_limit_on_frame(
    context: &ExecutionContext,
    node: &LimitNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = frame.num_rows();
    execute_limit_spec_on_frame(context, node.spec, frame, row_count, metrics)
}

pub(super) fn execute_limit_spec_on_frame(
    context: &ExecutionContext,
    spec: LimitSpec,
    frame: DataFrame,
    row_count: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (offset, length) = limit_slice(row_count, spec)?;
    if offset == 0 && length == row_count {
        return Ok(GpuDataFrame::new(frame));
    }

    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).slice(offset, length).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.limit.execute", "failed to execute native cudf limit", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn limit_keeps_all_rows(row_count: usize, spec: LimitSpec) -> Result<bool> {
    let (offset, length) = limit_slice(row_count, spec)?;
    Ok(offset == 0 && length == row_count)
}
