use super::*;

pub(in crate::exec) fn validate_aggregate_capability(
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

pub(in crate::exec) fn validate_final_aggregate_capability(
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

pub(in crate::exec) fn validate_global_aggregate_capability(
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

pub(in crate::exec) fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if global_count_star_expr(&expr.expr) => Ok(()),
        _ => lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.validate_expr").map(|_| ()),
    }
}

pub(in crate::exec) fn validate_aggregate_output_names(
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

pub(in crate::exec) fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        return execute_global_aggregate(context, &node.spec, input, output_schema, metrics);
    }

    // Lowering AVG through libcudf's groupby `MEAN` (`GroupByAgg::Mean`) keeps
    // the result scale equal to the source scale on `fixed_point` columns and
    // truncates the division remainder, so a `Decimal128(7, 2)` input averaged
    // to a `Decimal128(11, 6)` output rounds e.g. `34.65 / 2` to `17.32`
    // instead of `17.325`. Route AVG through the same explicit `sum + count +
    // finalize` state used by the partial/final and global aggregate paths so
    // the division happens at the output scale.
    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "cudf_query_engine.aggregate.execute")?;
    output = finalize_avg_output_columns(context, output, &node.spec, output_schema)?;
    output = cast_reduction_output_columns(output, &node.spec, output_schema)?;
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

pub(in crate::exec) fn execute_aggregate_with_budget(
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

    // See `execute_aggregate` for why AVG must not lower to libcudf groupby
    // `MEAN`; the same finalize-at-output-scale contract applies here.
    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
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
    output = finalize_avg_output_columns_with_budget(context, output, &node.spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
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

pub(in crate::exec) fn execute_partial_aggregate_with_budget(
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

pub(in crate::exec) fn execute_groupby_aggregate(
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

pub(in crate::exec) fn execute_groupby_aggregate_with_budget(
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
