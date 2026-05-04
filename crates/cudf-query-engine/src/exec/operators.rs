use super::*;

pub(super) fn validate_projection_capability(
    node: &ProjectionNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    let output_field_count = output_schema.fields().len();
    if node.expressions.len() != output_field_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection expression count must match output schema field count",
        )
        .with_fact("expression_count", node.expressions.len())
        .with_fact("output_field_count", output_field_count));
    }

    for expr in &node.expressions {
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
    )
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
        if expr.function != AggregateFunction::Sum {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "cudf_query_engine.global_aggregate.validate",
                "native global aggregate currently supports metadata-backed sum only",
            )
            .with_fact("function", format!("{:?}", expr.function)));
        }
        lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.validate_expr")?;
    }
    Ok(())
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

pub(super) fn execute_projection(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let output_field_count = output_schema.fields().len();
    if node.expressions.len() != output_field_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection expression count must match output schema field count",
        )
        .with_fact("expression_count", node.expressions.len())
        .with_fact("output_field_count", output_field_count));
    }

    let frame = input.into_frame();
    let output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let exprs = output_names
        .iter()
        .zip(&node.expressions)
        .map(|(name, expr)| lower_expr(expr).map(|expr| (name.as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let output_columns = output_names.iter().map(String::as_str).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output =
        frame.with_execution(&overrides).with_columns(exprs).and_then(|frame| frame.select(output_columns)).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.projection.execute",
                    "failed to execute native cudf projection",
                    error,
                )
            },
        )?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
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

pub(super) fn execute_partial_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        let output_schema = Arc::new(Schema::new(
            node.spec
                .aggr_exprs
                .iter()
                .map(|expr| Field::new(expr.alias.as_str(), ArrowDataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        return execute_global_aggregate(context, &node.spec, input, &output_schema, metrics);
    }

    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    let expected_output_names = aggregate.output_names.clone();
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output =
        execute_groupby_aggregate(&overrides, frame, &aggregate, "cudf_query_engine.aggregate.execute_partial")?
            .select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>())
            .map_err(|error| {
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
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .with_execution(overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| QueryError::from_cudf(operation, "failed to execute native cudf aggregate", error))?;
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
                "cudf_query_engine.aggregate.rename_keys",
                "failed to rename native cudf aggregate keys",
                error,
            )
        })?;
    }
    Ok(output)
}

pub(super) fn execute_global_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_global_aggregate_capability(spec, output_schema)?;
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut columns = Vec::with_capacity(spec.aggr_exprs.len());
    for expr in &spec.aggr_exprs {
        let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.lower_expr")?;
        let series = frame.series(input_column.as_str()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.input",
                "failed to access native global aggregate input column",
                error,
            )
            .with_fact("column", input_column.as_str())
        })?;
        let scalar = series.sum_in(&overrides).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.sum",
                "failed to execute native global sum aggregate",
                error,
            )
            .with_fact("column", input_column.as_str())
        })?;
        let output = context.runtime().series_from_scalar(&scalar, 1).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.global_aggregate.output",
                "failed to construct native global aggregate output column",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        columns.push((expr.alias.as_str(), output));
    }
    let output = DataFrame::from_series(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.output",
            "failed to construct native global aggregate output frame",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn cast_count_output_columns(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
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
        frame = frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.cast_count",
                "failed to cast native count aggregate output column",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
    }
    Ok(frame)
}

pub(super) fn finalize_avg_output_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
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
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.aggregate.finalize_avg",
                    "avg aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            let sum = Expr::cast(Expr::column(avg_sum_state_alias(index)), field.data_type().clone());
            let count = Expr::cast(Expr::column(avg_count_state_alias(index)), field.data_type().clone());
            let alias = expr.alias.clone();
            lower_expr(&Expr::binary(sum, BinaryOp::Divide, count)).map(|lowered| (alias, lowered))
        })
        .collect::<Result<Vec<_>>>()?;
    let overrides = execution_overrides(context);
    frame.with_execution(&overrides).with_columns(avg_exprs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.finalize_avg",
            "failed to finalize native avg aggregate output columns",
            error,
        )
    })
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
        if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.validate_avg",
                "avg aggregate output schema must be Float32 or Float64",
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

pub(super) fn execute_optional_limit(
    context: &ExecutionContext,
    node: Option<&LimitNode>,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    match node {
        Some(node) => execute_limit(context, node, input, metrics),
        None => Ok(input),
    }
}

pub(super) fn execute_limit_spec(
    context: &ExecutionContext,
    spec: LimitSpec,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, spec)? {
        return Ok(input);
    }

    execute_limit_spec_on_frame(context, spec, input.into_frame(), row_count, metrics)
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
