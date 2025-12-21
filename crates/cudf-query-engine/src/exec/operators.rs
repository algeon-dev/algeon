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
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        let partial_spec = partial_global_aggregate_spec(&node.spec);
        let output_schema = partial_global_aggregate_output_schema(&node.spec, input_schema, aggregate_output_schema)?;
        return execute_global_aggregate(context, &partial_spec, input, &output_schema, metrics);
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
        .options(GroupByOptions::builder().include_null_keys(true).build())
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
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let sum_alias = avg_sum_state_alias(index);
                let count_alias = avg_count_state_alias(index);
                columns.push((
                    sum_alias,
                    reduce_global_aggregate_column(context, &frame, &overrides, expr, AggregateFunction::Sum, None)?,
                ));
                columns.push((count_alias, global_count_series(context, &frame, expr)?));
            },
            AggregateFunction::Count => {
                columns.push((expr.alias.clone(), global_count_series(context, &frame, expr)?));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                columns.push((
                    expr.alias.clone(),
                    reduce_global_aggregate_column(
                        context,
                        &frame,
                        &overrides,
                        expr,
                        expr.function,
                        output_schema.field_with_name(expr.alias.as_str()).ok().map(|field| field.data_type()),
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
    output = cast_global_reduction_output_columns(output, spec, output_schema)?;
    output = finalize_avg_output_columns(context, output, spec, output_schema)?;
    output = cast_count_output_columns(output, spec, output_schema)?;
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

pub(super) fn reduce_global_aggregate_column(
    context: &ExecutionContext,
    frame: &DataFrame,
    overrides: &ExecutionOverrides<'_>,
    expr: &AggregateExpr,
    function: AggregateFunction,
    output_type: Option<&ArrowDataType>,
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
        return global_null_output_series(context, output_type, expr.alias.as_str());
    }
    let scalar = match function {
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
        .with_fact("column", input_column.as_str())
        .with_fact("function", format!("{function:?}"))
    })?;
    context.runtime().series_from_scalar(&scalar, 1).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.output",
            "failed to construct native global aggregate output column",
            error,
        )
        .with_fact("alias", expr.alias.as_str())
    })
}

fn global_null_output_series(
    context: &ExecutionContext,
    output_type: &ArrowDataType,
    alias: &str,
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
    context.runtime().nulls(dtype, 1).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.global_aggregate.null_output",
            "failed to construct native global aggregate null output column",
            error,
        )
        .with_fact("alias", alias)
        .with_fact("data_type", format!("{output_type:?}"))
    })
}

pub(super) fn global_count_series(
    context: &ExecutionContext,
    frame: &DataFrame,
    expr: &AggregateExpr,
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
                AggregateFunction::Avg => vec![
                    AggregateExpr::new(AggregateFunction::Sum, expr.expr.clone(), avg_sum_state_alias(index)),
                    AggregateExpr::new(AggregateFunction::Count, expr.expr.clone(), avg_count_state_alias(index)),
                ],
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
                let input_type = global_aggregate_input_data_type(&expr.expr, input_schema)?;
                fields.push(Field::new(avg_sum_state_alias(index), input_type, true));
                fields.push(Field::new(avg_count_state_alias(index), ArrowDataType::Int64, false));
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

pub(super) fn cast_global_reduction_output_columns(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
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
        frame = frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.cast_reduction",
                "failed to cast native global reduction output column",
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
            let count =
                Expr::cast(Expr::column(avg_count_state_alias(index)), avg_count_output_data_type(field.data_type()));
            let alias = expr.alias.clone();
            let count_is_zero = Expr::binary(
                Expr::column(avg_count_state_alias(index)),
                BinaryOp::Eq,
                Expr::literal(LiteralValue::Int64(0)),
            );
            let nullable_value = Expr::case(
                vec![crate::expr::CaseWhen::new(
                    count_is_zero,
                    Expr::literal(LiteralValue::Null(field.data_type().clone())),
                )],
                Expr::binary(sum, BinaryOp::Divide, count),
            );
            lower_expr(&nullable_value).map(|lowered| (alias, lowered))
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

pub(super) fn global_valid_count_state_alias(alias: &str) -> String {
    format!("{GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX}{alias}")
}

const GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX: &str = "__cudf_global_valid_count_";

fn avg_count_output_data_type(avg_output_type: &ArrowDataType) -> ArrowDataType {
    match avg_output_type {
        ArrowDataType::Decimal128(_, _) => ArrowDataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, 0),
        _ => avg_output_type.clone(),
    }
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
