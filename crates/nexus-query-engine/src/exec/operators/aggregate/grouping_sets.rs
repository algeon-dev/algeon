use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;

pub(super) fn execute_grouping_sets_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_floating_statistical_aggregate(spec) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.aggregate.grouping_sets",
            "grouping sets do not currently support floating statistical aggregates",
        ));
    }
    let aggregate = lower_partial_aggregate_spec(spec)?;
    if aggregate_spec_has_avg(spec) {
        validate_avg_output_fields(spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let timer = Instant::now();
    let frame = input.into_frame();
    let mut output = execute_grouping_sets_groupby(
        context,
        spec,
        input_schema,
        output_schema,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.grouping_sets",
    )?;
    output = match token {
        Some(token) => finalize_avg_output_columns_with_budget(context, output, spec, output_schema, token, metrics)?,
        None => finalize_avg_output_columns(context, output, spec, output_schema)?,
    };
    output = match token {
        Some(token) => cast_reduction_output_columns_with_budget(output, spec, output_schema, token, metrics)?,
        None => cast_reduction_output_columns(output, spec, output_schema)?,
    };
    output = match token {
        Some(token) => cast_count_output_columns_with_budget(output, spec, output_schema, token, metrics)?,
        None => cast_count_output_columns(output, spec, output_schema)?,
    };
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.grouping_sets_select",
            "failed to select native grouping-sets aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_grouping_sets_partial_aggregate_with_budget(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_floating_statistical_aggregate(spec) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.aggregate.grouping_sets_partial",
            "grouping sets do not currently support floating statistical aggregates",
        ));
    }
    let aggregate = lower_partial_aggregate_spec(spec)?;
    let expected_output_names = aggregate.output_names.clone();
    let timer = Instant::now();
    let frame = input.into_frame();
    let mut output = execute_grouping_sets_groupby(
        context,
        spec,
        input_schema,
        aggregate_output_schema,
        frame,
        &aggregate,
        Some(token),
        metrics,
        "nexus_query_engine.aggregate.grouping_sets_partial",
    )?;
    output = cast_count_output_columns_with_budget(output, spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.grouping_sets_select_partial",
            "failed to select native partial grouping-sets aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_grouping_sets_groupby(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let grouping_sets = spec.grouping_sets.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "grouping-sets execution requires a grouping-sets aggregate spec",
        )
    })?;
    if rollup_prefix_fast_path_supported(spec)
        && let Some(levels) = prefix_rollup_levels(&grouping_sets.groups, spec.group_exprs.len())
    {
        return execute_prefix_rollup_grouping_sets_groupby(
            context,
            input_schema,
            output_schema,
            frame,
            aggregate,
            grouping_sets.grouping_id_alias.as_str(),
            &levels,
            token,
            metrics,
            operation,
        );
    }
    let expected_output_names = aggregate.output_names.clone();
    let mut outputs = Vec::with_capacity(grouping_sets.groups.len());
    for group in &grouping_sets.groups {
        if group.len() != spec.group_exprs.len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "grouping-set mask length must match group expression count",
            )
            .with_fact("mask_len", group.len())
            .with_fact("group_expr_count", spec.group_exprs.len()));
        }
        let output = execute_single_grouping_set_groupby(
            context,
            input_schema,
            output_schema,
            frame.clone(),
            aggregate,
            grouping_sets.grouping_id_alias.as_str(),
            group,
            token,
            metrics,
            operation,
        )?
        .select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>())
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to select native grouping-set aggregate output columns", error)
        })?;
        outputs.push(output);
    }
    concat_grouping_set_outputs(context, outputs, token, operation)
}

fn execute_prefix_rollup_grouping_sets_groupby(
    context: &ExecutionContext,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    grouping_id_alias: &str,
    levels: &[usize],
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let key_count = levels.len().saturating_sub(1);
    let expected_output_names = aggregate.output_names.clone();
    let mut outputs_by_level = (0..=key_count).map(|_| None).collect::<Vec<Option<DataFrame>>>();

    let full_group = prefix_rollup_group(key_count, key_count);
    let full_output = execute_single_grouping_set_groupby(
        context,
        input_schema,
        output_schema,
        frame,
        aggregate,
        grouping_id_alias,
        &full_group,
        token,
        metrics,
        operation,
    )?
    .select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>())
    .map_err(|error| {
        QueryError::from_cudf(operation, "failed to select native prefix-rollup detail output columns", error)
    })?;
    outputs_by_level[key_count] = Some(full_output);

    let final_aggregate = lower_final_aggregate_spec_for_prefix_rollup(aggregate)?;
    for level in (0..key_count).rev() {
        let source = outputs_by_level[level + 1]
            .as_ref()
            .expect("finer rollup level must be produced before coarser level")
            .clone();
        let group = prefix_rollup_group(key_count, level);
        let source = if level == 0 {
            with_grouping_id_column(
                context,
                output_schema,
                source,
                grouping_id_alias,
                grouping_id_value(&group),
                token,
                metrics,
                operation,
            )?
        } else {
            source
        };
        let rollup_aggregate = aggregate_for_prefix_rollup_level(&final_aggregate, level, grouping_id_alias);
        let grouped = match token {
            Some(token) => {
                let overrides = token.execution_overrides();
                execute_groupby_aggregate_with_budget(
                    &overrides,
                    source,
                    &rollup_aggregate,
                    token,
                    metrics,
                    operation,
                    "failed to execute bounded native prefix-rollup aggregate",
                )?
            },
            None => {
                let overrides_lease = execution_overrides(context)?;
                let overrides = overrides_lease.overrides();
                execute_groupby_aggregate(&overrides, source, &rollup_aggregate, operation)?
            },
        };
        let output = complete_grouping_set_output(
            context,
            output_schema,
            grouped,
            aggregate,
            grouping_id_alias,
            &group,
            token,
            metrics,
            operation,
        )?
        .select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>())
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to select native prefix-rollup aggregate output columns", error)
        })?;
        outputs_by_level[level] = Some(output);
    }

    let mut outputs = Vec::with_capacity(levels.len());
    for level in levels {
        outputs.push(outputs_by_level[*level].take().expect("all requested prefix rollup levels must be produced"));
    }
    concat_grouping_set_outputs(context, outputs, token, operation)
}

fn execute_single_grouping_set_groupby(
    context: &ExecutionContext,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    grouping_id_alias: &str,
    group: &[bool],
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let grouping_set_aggregate = aggregate_for_grouping_set(aggregate, group);
    if grouping_set_aggregate.key_columns.is_empty() {
        let working = prepare_grouping_set_frame(
            context,
            input_schema,
            output_schema,
            frame,
            aggregate,
            grouping_id_alias,
            group,
            token,
            metrics,
            operation,
        )?;
        return match token {
            Some(token) => {
                let overrides = token.execution_overrides();
                execute_groupby_aggregate_with_budget(
                    &overrides,
                    working,
                    aggregate,
                    token,
                    metrics,
                    operation,
                    "failed to execute bounded native grouping-sets aggregate",
                )
            },
            None => {
                let overrides_lease = execution_overrides(context)?;
                let overrides = overrides_lease.overrides();
                execute_groupby_aggregate(&overrides, working, aggregate, operation)
            },
        };
    }

    let grouped = match token {
        Some(token) => {
            let overrides = token.execution_overrides();
            execute_groupby_aggregate_with_budget(
                &overrides,
                frame,
                &grouping_set_aggregate,
                token,
                metrics,
                operation,
                "failed to execute bounded native grouping-set aggregate",
            )?
        },
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            execute_groupby_aggregate(&overrides, frame, &grouping_set_aggregate, operation)?
        },
    };
    complete_grouping_set_output(
        context,
        output_schema,
        grouped,
        aggregate,
        grouping_id_alias,
        group,
        token,
        metrics,
        operation,
    )
}

fn aggregate_for_grouping_set(aggregate: &LoweredAggregate, group: &[bool]) -> LoweredAggregate {
    let mut key_columns = Vec::with_capacity(group.len());
    let mut output_key_names = Vec::with_capacity(group.len());
    for (index, is_null) in group.iter().enumerate() {
        if !*is_null {
            key_columns.push(aggregate.key_columns[index].clone());
            output_key_names.push(aggregate.output_key_names[index].clone());
        }
    }
    let aggregate_output_names =
        aggregate.output_names.iter().skip(aggregate.output_key_names.len()).cloned().collect::<Vec<_>>();
    let output_names = output_key_names.iter().cloned().chain(aggregate_output_names).collect();
    LoweredAggregate { key_columns, output_key_names, aggregations: aggregate.aggregations.clone(), output_names }
}

fn complete_grouping_set_output(
    context: &ExecutionContext,
    output_schema: &ArrowSchemaRef,
    output: DataFrame,
    aggregate: &LoweredAggregate,
    grouping_id_alias: &str,
    group: &[bool],
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let mut expressions = Vec::<(String, cudf::Expr)>::new();
    for (index, is_null) in group.iter().enumerate() {
        if *is_null {
            let key_column = &aggregate.output_key_names[index];
            let field = output_schema.field_with_name(key_column.as_str()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    operation,
                    "grouping-set output key column is missing from aggregate output schema",
                )
                .with_fact("column", key_column.as_str())
                .with_fact("arrow_error", error.to_string())
            })?;
            let dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    operation,
                    "grouping-set null output key column type is not supported by native execution",
                )
                .with_fact("column", key_column.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            expressions.push((key_column.clone(), lit_null(dtype)));
        }
    }

    if !output.contains(grouping_id_alias) {
        let grouping_id_field = output_schema.field_with_name(grouping_id_alias).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "grouping-id column is missing from grouping-sets output schema",
            )
            .with_fact("grouping_id_alias", grouping_id_alias)
            .with_fact("arrow_error", error.to_string())
        })?;
        let grouping_id = grouping_id_value(group);
        expressions.push((
            grouping_id_alias.to_string(),
            grouping_id_literal_expr(grouping_id_field.data_type(), grouping_id, operation)?,
        ));
    }

    match token {
        Some(token) => output
            .with_columns_with_budget(expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    operation,
                    "failed to complete bounded grouping-set aggregate output columns",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, operation)),
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            output.with_execution(&overrides).with_columns(expressions).map_err(|error| {
                QueryError::from_cudf(operation, "failed to complete grouping-set aggregate output columns", error)
            })
        },
    }
}

fn with_grouping_id_column(
    context: &ExecutionContext,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    grouping_id_alias: &str,
    grouping_id: u64,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let grouping_id_field = output_schema.field_with_name(grouping_id_alias).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "grouping-id column is missing from grouping-sets output schema",
        )
        .with_fact("grouping_id_alias", grouping_id_alias)
        .with_fact("arrow_error", error.to_string())
    })?;
    let expressions = vec![(
        grouping_id_alias.to_string(),
        grouping_id_literal_expr(grouping_id_field.data_type(), grouping_id, operation)?,
    )];
    match token {
        Some(token) => frame
            .with_columns_with_budget(expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    operation,
                    "failed to set bounded prefix-rollup grouping-id column",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, operation)),
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            frame.with_execution(&overrides).with_columns(expressions).map_err(|error| {
                QueryError::from_cudf(operation, "failed to set prefix-rollup grouping-id column", error)
            })
        },
    }
}

fn rollup_prefix_fast_path_supported(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().all(|expr| {
        !expr.distinct
            && expr.filter.is_none()
            && matches!(
                expr.function,
                AggregateFunction::Sum | AggregateFunction::Count | AggregateFunction::Min | AggregateFunction::Max
            )
    })
}

fn prefix_rollup_levels(groups: &[Vec<bool>], key_count: usize) -> Option<Vec<usize>> {
    if groups.len() != key_count + 1 {
        return None;
    }
    let mut seen = vec![false; key_count + 1];
    let mut levels = Vec::with_capacity(groups.len());
    for group in groups {
        if group.len() != key_count {
            return None;
        }
        let level = group.iter().take_while(|is_null| !**is_null).count();
        if group.iter().skip(level).any(|is_null| !*is_null) || seen[level] {
            return None;
        }
        seen[level] = true;
        levels.push(level);
    }
    seen.into_iter().all(|present| present).then_some(levels)
}

fn prefix_rollup_group(key_count: usize, level: usize) -> Vec<bool> {
    (0..key_count).map(|index| index >= level).collect()
}

fn lower_final_aggregate_spec_for_prefix_rollup(aggregate: &LoweredAggregate) -> Result<LoweredAggregate> {
    let aggregations = aggregate
        .aggregations
        .iter()
        .map(|aggregation| {
            let output_name = aggregation.resolved_output_name();
            let function = match aggregation.agg() {
                GroupByAgg::Sum | GroupByAgg::Count { .. } => GroupByAgg::Sum,
                GroupByAgg::Min => GroupByAgg::Min,
                GroupByAgg::Max => GroupByAgg::Max,
                other => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "nexus_query_engine.aggregate.prefix_rollup",
                        "prefix rollup does not support this aggregate function",
                    )
                    .with_fact("function", format!("{other:?}")));
                },
            };
            Ok(CudfAggSpec::new(output_name.as_str(), function).with_alias(output_name.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoweredAggregate {
        key_columns: aggregate.key_columns.clone(),
        output_key_names: aggregate.output_key_names.clone(),
        aggregations,
        output_names: aggregate.output_names.clone(),
    })
}

fn aggregate_for_prefix_rollup_level(
    aggregate: &LoweredAggregate,
    level: usize,
    grouping_id_alias: &str,
) -> LoweredAggregate {
    let mut key_columns = aggregate.key_columns.iter().take(level).cloned().collect::<Vec<_>>();
    let mut output_key_names = aggregate.output_key_names.iter().take(level).cloned().collect::<Vec<_>>();
    if level == 0 {
        key_columns.push(grouping_id_alias.to_string());
        output_key_names.push(grouping_id_alias.to_string());
    }
    let aggregate_output_names =
        aggregate.output_names.iter().skip(aggregate.output_key_names.len()).cloned().collect::<Vec<_>>();
    let output_names = output_key_names.iter().cloned().chain(aggregate_output_names).collect();
    LoweredAggregate { key_columns, output_key_names, aggregations: aggregate.aggregations.clone(), output_names }
}

fn prepare_grouping_set_frame(
    context: &ExecutionContext,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    grouping_id_alias: &str,
    group: &[bool],
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> Result<DataFrame> {
    let mut expressions = Vec::<(String, cudf::Expr)>::new();
    for (key_column, is_null) in aggregate.key_columns.iter().take(group.len()).zip(group) {
        if *is_null {
            let field = input_schema.field_with_name(key_column.as_str()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    operation,
                    "grouping-set key column is missing from aggregate input schema",
                )
                .with_fact("column", key_column.as_str())
                .with_fact("arrow_error", error.to_string())
            })?;
            let dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    operation,
                    "grouping-set null key column type is not supported by native execution",
                )
                .with_fact("column", key_column.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            expressions.push((key_column.clone(), lit_null(dtype)));
        }
    }

    let grouping_id_field = output_schema.field_with_name(grouping_id_alias).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "grouping-id column is missing from grouping-sets output schema",
        )
        .with_fact("grouping_id_alias", grouping_id_alias)
        .with_fact("arrow_error", error.to_string())
    })?;
    let grouping_id = grouping_id_value(group);
    expressions.push((
        grouping_id_alias.to_string(),
        grouping_id_literal_expr(grouping_id_field.data_type(), grouping_id, operation)?,
    ));

    match token {
        Some(token) => frame
            .with_columns_with_budget(expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    operation,
                    "failed to prepare bounded grouping-set aggregate columns",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, operation)),
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            frame.with_execution(&overrides).with_columns(expressions).map_err(|error| {
                QueryError::from_cudf(operation, "failed to prepare grouping-set aggregate columns", error)
            })
        },
    }
}

fn grouping_id_value(group: &[bool]) -> u64 {
    group.iter().fold(0u64, |acc, is_null| (acc << 1) | u64::from(*is_null))
}

fn grouping_id_literal_expr(data_type: &ArrowDataType, value: u64, operation: &'static str) -> Result<cudf::Expr> {
    match data_type {
        ArrowDataType::UInt8 => Ok(lit(u8::try_from(value).map_err(|_error| {
            QueryError::new(QueryErrorCode::InvalidPlan, operation, "grouping id does not fit UInt8")
                .with_fact("grouping_id", value)
        })?)),
        ArrowDataType::UInt16 => Ok(lit(u16::try_from(value).map_err(|_error| {
            QueryError::new(QueryErrorCode::InvalidPlan, operation, "grouping id does not fit UInt16")
                .with_fact("grouping_id", value)
        })?)),
        ArrowDataType::UInt32 => Ok(lit(u32::try_from(value).map_err(|_error| {
            QueryError::new(QueryErrorCode::InvalidPlan, operation, "grouping id does not fit UInt32")
                .with_fact("grouping_id", value)
        })?)),
        ArrowDataType::UInt64 => Ok(lit(value)),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "grouping-id output type is not supported by native execution",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn concat_grouping_set_outputs(
    context: &ExecutionContext,
    mut outputs: Vec<DataFrame>,
    token: Option<&ExecutionBudgetToken>,
    operation: &'static str,
) -> Result<DataFrame> {
    if outputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "grouping-sets aggregate requires at least one grouping set",
        ));
    }
    if outputs.len() == 1 {
        return Ok(outputs.remove(0));
    }
    let first = outputs.remove(0);
    let other_refs = outputs.iter().collect::<Vec<_>>();
    match token {
        Some(token) => first.with_execution(&token.execution_overrides()).concat_many(&other_refs),
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            first.with_execution(&overrides).concat_many(&other_refs)
        },
    }
    .map_err(|error| QueryError::from_cudf(operation, "failed to concatenate grouping-set aggregate outputs", error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_rollup_levels_accepts_complete_rollup_masks_in_plan_order() {
        let groups =
            vec![vec![true, true, true], vec![false, true, true], vec![false, false, true], vec![false, false, false]];

        assert_eq!(prefix_rollup_levels(&groups, 3), Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn prefix_rollup_levels_rejects_non_prefix_or_incomplete_masks() {
        assert_eq!(prefix_rollup_levels(&[vec![false, true, false], vec![false, false, false]], 3), None);
        assert_eq!(prefix_rollup_levels(&[vec![true, true], vec![false, true]], 2), None);
    }

    #[test]
    fn prefix_rollup_level_aggregate_uses_active_prefix_keys_and_preserves_aggregate_outputs() {
        let aggregate = LoweredAggregate {
            key_columns: vec!["k0".into(), "k1".into(), "k2".into(), "__grouping_id".into()],
            output_key_names: vec!["k0".into(), "k1".into(), "k2".into(), "__grouping_id".into()],
            aggregations: vec![CudfAggSpec::new("sales", GroupByAgg::Sum).with_alias("sum_sales")],
            output_names: vec!["k0".into(), "k1".into(), "k2".into(), "__grouping_id".into(), "sum_sales".into()],
        };

        let level_two = aggregate_for_prefix_rollup_level(&aggregate, 2, "__grouping_id");
        assert_eq!(level_two.key_columns, vec!["k0", "k1"]);
        assert_eq!(level_two.output_names, vec!["k0", "k1", "sum_sales"]);

        let grand_total = aggregate_for_prefix_rollup_level(&aggregate, 0, "__grouping_id");
        assert_eq!(grand_total.key_columns, vec!["__grouping_id"]);
        assert_eq!(grand_total.output_names, vec!["__grouping_id", "sum_sales"]);
    }
}
