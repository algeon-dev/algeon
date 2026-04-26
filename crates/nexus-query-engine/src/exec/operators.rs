use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{lowering_policy::ExpressionLoweringPolicy, *};

mod aggregate;
mod global_aggregate;
mod projection;
pub(super) use self::{aggregate::*, global_aggregate::*, projection::*};

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
    context.execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(budget_boundary)).map_err(
        |error| {
            error
                .with_fact_if_absent("budget_boundary", budget_boundary)
                .with_fact_if_absent("budget_bytes", budget_bytes)
        },
    )
}

pub(super) fn attach_budget_token_failure_facts(error: QueryError, token: &ExecutionBudgetToken) -> QueryError {
    error
        .with_fact_if_absent("budget_boundary", token.request().boundary())
        .with_fact_if_absent("budget_bytes", token.request().budget_bytes())
        .with_fact_if_absent("budget_enforcement", token.enforcement().as_str())
        .with_fact_if_absent("budget_status", "exceeded_or_lower_layer_rejected")
}

pub(super) fn budgeted_value<T>(
    outcome: BudgetedExecutionOutcome<T>,
    metrics: &mut QueryMetrics,
    operation: &'static str,
) -> T {
    let (value, report) = outcome.into_parts();
    metrics.record_budgeted_execution(operation, &report);
    value
}

pub(super) fn execute_filter(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.frame().clone();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = if let Some((column, values)) = literal_in_list_filter_series(context, &node.predicate, &overrides)? {
        frame.with_execution(&overrides).filter_isin(&column, &values).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.filter.execute",
                "failed to execute nexus native IN-list filter",
                error,
            )
        })?
    } else {
        let predicate = lower_filter_predicate_expr(&node.predicate)?;
        frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
            QueryError::from_cudf("nexus_query_engine.filter.execute", "failed to execute nexus native filter", error)
        })?
    };
    metrics.operator_time += timer.elapsed();
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

pub(super) fn execute_filter_with_budget(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.frame().clone();
    let overrides = token.execution_overrides();
    let timer = Instant::now();
    let output = if let Some((column, values)) = literal_in_list_filter_series(context, &node.predicate, &overrides)? {
        frame.with_execution(&overrides).filter_isin(&column, &values).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.filter.execute",
                    "failed to execute nexus native IN-list filter",
                    error,
                ),
                token,
            )
        })?
    } else {
        let predicate = lower_filter_predicate_expr(&node.predicate)?;
        frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.filter.execute",
                    "failed to execute nexus native filter",
                    error,
                ),
                token,
            )
        })?
    };
    metrics.operator_time += timer.elapsed();
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

fn literal_in_list_filter_series(
    context: &ExecutionContext,
    predicate: &Expr,
    overrides: &ExecutionOverrides<'_>,
) -> Result<Option<(String, cudf::Series)>> {
    let Expr::InList { expr, list, negated: false } = predicate else {
        return Ok(None);
    };
    if !ExpressionLoweringPolicy::default().filter_in_list_should_use_specialized_series(list.len()) {
        return Ok(None);
    }
    let Expr::Column(column) = expr.as_ref() else {
        return Ok(None);
    };
    let Some(first) = list.first().and_then(|expr| match expr {
        Expr::Literal(value) => Some(value),
        _ => None,
    }) else {
        return Ok(None);
    };
    let series = match first {
        LiteralValue::Int32(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::Int32(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.series_in(&values, overrides)
        },
        LiteralValue::Int64(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::Int64(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.series_in(&values, overrides)
        },
        LiteralValue::UInt32(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::UInt32(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.series_in(&values, overrides)
        },
        LiteralValue::UInt64(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::UInt64(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.series_in(&values, overrides)
        },
        _ => return Ok(None),
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.filter.literal_in_list_values",
            "failed to upload native filter IN-list values",
            error,
        )
    })?;
    Ok(Some((column.name().to_string(), series)))
}

fn collect_in_list_literals<T: Copy>(
    list: &[Expr],
    mut convert: impl FnMut(&LiteralValue) -> Option<T>,
) -> Option<Vec<T>> {
    list.iter()
        .map(|expr| match expr {
            Expr::Literal(value) => convert(value),
            _ => None,
        })
        .collect()
}

pub(super) fn execute_sort(
    context: &ExecutionContext,
    node: &SortNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let output = execute_sort_on_frame(context, node, input.frame().clone(), metrics)?;
    GpuDataFrame::try_new_with_source_residency(output.into_frame(), &input)
}

pub(super) fn execute_sort_on_frame(
    context: &ExecutionContext,
    node: &SortNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&node.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).sort_values(sort.columns, options).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.sort.execute", "failed to execute nexus native sort", error)
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

    let output = execute_limit_on_frame(context, node, input.frame().clone(), metrics)?;
    GpuDataFrame::try_new_with_source_residency(output.into_frame(), &input)
}

pub(super) fn execute_union(
    context: &ExecutionContext,
    node_id: NodeId,
    inputs: Vec<GpuDataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if inputs.len() == 1 {
        return Ok(inputs.into_iter().next().expect("single-input union has one frame"));
    }
    let mut frames = inputs.iter().map(|input| input.frame().clone()).collect::<Vec<_>>();
    if frames.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.union.execute",
            "union execution requires at least one input",
        )
        .with_fact("node_id", node_id));
    }

    let input_count = frames.len();
    let first = frames.remove(0);
    let rest = frames.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = first.with_execution(&overrides).concat_many(&rest).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.union.concat", "failed to concatenate native union inputs", error)
            .with_fact("node_id", node_id)
            .with_fact("input_count", input_count)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn execute_window(
    context: &ExecutionContext,
    node: &WindowNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.functions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.window.execute",
            "window node requires at least one function",
        ));
    }

    let mut frame = input.frame().clone();
    if node.spec.functions.iter().any(|function| {
        matches!(function, crate::plan::WindowFunction::Aggregate { frame: WindowFrame::WholePartition, .. })
    }) {
        frame = execute_partition_window_aggregate(context, node, input_schema, frame, output_schema, metrics)?;
    }
    if node.spec.functions.iter().any(|function| {
        matches!(
            function,
            crate::plan::WindowFunction::Aggregate { frame: WindowFrame::RowsUnboundedPrecedingToCurrent, .. }
        )
    }) {
        frame = execute_cumulative_window_aggregate(context, node, frame, metrics)?;
    }
    for function in &node.spec.functions {
        if let crate::plan::WindowFunction::Rank { alias } = function {
            frame = execute_rank_window(context, node, frame, alias.as_str(), metrics)?;
        }
    }
    for function in &node.spec.functions {
        if let crate::plan::WindowFunction::RowNumber { alias } = function {
            frame = execute_row_number_window(context, node, frame, alias.as_str(), metrics)?;
        }
    }
    for function in &node.spec.functions {
        if let crate::plan::WindowFunction::LagLead { expr, offset, default, alias, direction } = function {
            frame = execute_lag_lead_window(
                context,
                node,
                output_schema,
                frame,
                expr,
                *offset,
                default.as_ref(),
                alias.as_str(),
                *direction,
                metrics,
            )?;
        }
    }

    let output_names = output_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    let output = frame.select(output_names).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.select",
            "failed to select native window output columns",
            error,
        )
    })?;
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

fn execute_partition_window_aggregate(
    context: &ExecutionContext,
    node: &WindowNode,
    input_schema: &ArrowSchemaRef,
    frame: DataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    if node.spec.partition_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.window.aggregate",
            "native partition aggregate windows currently require at least one partition key",
        ));
    }
    if !node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.window.aggregate",
            "native partition aggregate windows require a constant-in-partition frame without ORDER BY",
        ));
    }

    let group_exprs = node
        .spec
        .partition_by
        .iter()
        .map(|expr| {
            let name = lower_column_expr(expr, "nexus_query_engine.window.aggregate.partition")?;
            Ok(Expr::column(name.clone()).alias(name))
        })
        .collect::<Result<Vec<_>>>()?;
    let aggr_exprs = node
        .spec
        .functions
        .iter()
        .filter_map(|function| match function {
            crate::plan::WindowFunction::Aggregate { function, expr, alias, frame: WindowFrame::WholePartition } => {
                Some(AggregateExpr::new(*function, expr.clone(), alias.clone()))
            },
            crate::plan::WindowFunction::Aggregate { .. }
            | crate::plan::WindowFunction::Rank { .. }
            | crate::plan::WindowFunction::RowNumber { .. }
            | crate::plan::WindowFunction::LagLead { .. } => None,
        })
        .collect::<Vec<_>>();
    let aggregate_spec = AggregateSpec::new(group_exprs, aggr_exprs);
    let aggregate_schema = window_aggregate_schema(input_schema, output_schema, node)?;
    let aggregate = AggregateNode { input: node.input, spec: aggregate_spec };
    let aggregate_frame = execute_aggregate(
        context,
        &aggregate,
        input_schema,
        GpuDataFrame::new(frame.clone()),
        &aggregate_schema,
        metrics,
    )?
    .into_frame();

    let keys = node
        .spec
        .partition_by
        .iter()
        .map(|expr| lower_column_expr(expr, "nexus_query_engine.window.aggregate.merge_key"))
        .collect::<Result<Vec<_>>>()?;
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let output = frame
        .with_execution(&overrides)
        .merge(
            &aggregate_frame,
            cudf::MergeOptions::new().how(cudf::JoinType::Left).on(keys.iter().map(String::as_str)),
        )
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.aggregate.merge",
                "failed to align native partition aggregate window outputs",
                error,
            )
        })?;
    Ok(output)
}

fn window_aggregate_schema(
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    node: &WindowNode,
) -> Result<ArrowSchemaRef> {
    let mut fields = Vec::with_capacity(node.spec.partition_by.len() + node.spec.functions.len());
    for expr in &node.spec.partition_by {
        let name = lower_column_expr(expr, "nexus_query_engine.window.aggregate.schema_key")?;
        fields.push(input_schema.field_with_name(name.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.aggregate.schema_key",
                "partition key field is missing from input schema",
            )
            .with_fact("field", name)
        })?);
    }
    for function in &node.spec.functions {
        let crate::plan::WindowFunction::Aggregate { alias, frame: WindowFrame::WholePartition, .. } = function else {
            continue;
        };
        fields.push(output_schema.field_with_name(alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.aggregate.schema_output",
                "window aggregate output field is missing from output schema",
            )
            .with_fact("field", alias.as_str())
        })?);
    }
    Ok(Arc::new(Schema::new(fields.into_iter().cloned().collect::<Vec<_>>())))
}

fn execute_cumulative_window_aggregate(
    context: &ExecutionContext,
    node: &WindowNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    if node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.window.cumulative",
            "native cumulative aggregate windows require ORDER BY",
        ));
    }

    let timer = Instant::now();
    let mut temp_columns = Vec::new();
    let row_index_name = unique_temp_column_name(&frame, "__nexus_native_window_input_row_index");
    let row_index_series = frame.row_positions().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.cumulative.row_index",
            "failed to create cumulative window row index",
            error,
        )
    })?;
    let mut working = frame.with_column(row_index_name.as_str(), row_index_series).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.cumulative.row_index",
            "failed to append cumulative window row index",
            error,
        )
    })?;
    temp_columns.push(row_index_name.clone());

    let (with_partition_key, partition_keys) =
        rank_partition_keys(context, working, &node.spec.partition_by, &mut temp_columns)?;
    working = with_partition_key;

    let mut sort_columns = partition_keys.clone();
    let mut descending = vec![false; partition_keys.len()];
    let mut nulls_first = vec![false; partition_keys.len()];
    for sort_expr in &node.spec.order_by {
        sort_columns.push(lower_column_expr(&sort_expr.expr, "nexus_query_engine.window.cumulative.order_key")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    let sort_options = CudfSortOptions::new().with_descending(descending).with_nulls_first(nulls_first);
    working = working.with_execution(&overrides).sort_values(sort_columns, sort_options).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.cumulative.sort",
            "failed to sort cumulative window input",
            error,
        )
    })?;

    for (function_index, function) in node.spec.functions.iter().enumerate() {
        let crate::plan::WindowFunction::Aggregate {
            function,
            expr,
            alias,
            frame: WindowFrame::RowsUnboundedPrecedingToCurrent,
        } = function
        else {
            continue;
        };
        let scan_agg = cumulative_scan_agg(*function)?;
        let value_name =
            unique_temp_column_name(&working, &format!("__nexus_native_window_cumulative_value_{function_index}"));
        let value_expr = lower_expr(expr)?.alias(value_name.as_str());
        working = working.with_execution(&overrides).with_columns([value_expr]).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.value",
                "failed to materialize cumulative window input expression",
                error,
            )
        })?;
        temp_columns.push(value_name.clone());

        let scan_name =
            unique_temp_column_name(&working, &format!("__nexus_native_window_cumulative_scan_{function_index}"));
        let scanned = working
            .group_scan_sorted(partition_keys.iter().map(String::as_str), value_name.as_str(), scan_agg)
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.window.cumulative.scan",
                    "failed to execute grouped cumulative window scan",
                    error,
                )
                .with_fact("function", format!("{function:?}"))
            })?;
        working = working.with_column(scan_name.as_str(), scanned).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.output",
                "failed to append cumulative window scan output",
                error,
            )
            .with_fact("scan_column", scan_name.as_str())
        })?;
        temp_columns.push(scan_name.clone());

        let valid_name =
            unique_temp_column_name(&working, &format!("__nexus_native_window_cumulative_valid_{function_index}"));
        let valid_expr = when(col(value_name.as_str()).is_null()).then(0i32).otherwise(1i32).alias(valid_name.as_str());
        working = working.with_columns([valid_expr]).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.validity",
                "failed to append cumulative window validity indicator",
                error,
            )
        })?;
        temp_columns.push(valid_name.clone());

        let seen_name =
            unique_temp_column_name(&working, &format!("__nexus_native_window_cumulative_seen_{function_index}"));
        let seen = working
            .group_scan_sorted(partition_keys.iter().map(String::as_str), valid_name.as_str(), ScanAgg::Sum)
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.window.cumulative.validity_scan",
                    "failed to execute grouped cumulative window validity scan",
                    error,
                )
            })?;
        working = working.with_column(seen_name.as_str(), seen).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.validity_scan",
                "failed to append cumulative window validity scan output",
                error,
            )
            .with_fact("seen_column", seen_name.as_str())
        })?;
        temp_columns.push(seen_name.clone());

        let filled_name =
            unique_temp_column_name(&working, &format!("__nexus_native_window_cumulative_filled_{function_index}"));
        let filled = working.series(scan_name.as_str()).and_then(|series| series.ffill()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.fill",
                "failed to forward-fill cumulative window scan output",
                error,
            )
            .with_fact("scan_column", scan_name.as_str())
        })?;
        working = working.with_column(filled_name.as_str(), filled).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.fill",
                "failed to append forward-filled cumulative window scan output",
                error,
            )
            .with_fact("filled_column", filled_name.as_str())
        })?;
        temp_columns.push(filled_name.clone());

        let output_expr = when(col(seen_name.as_str()).gt(0i32))
            .then(col(filled_name.as_str()))
            .otherwise(col(scan_name.as_str()))
            .alias(alias.as_str());
        working = working.with_columns([output_expr]).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.output",
                "failed to append cumulative window output",
                error,
            )
            .with_fact("alias", alias.as_str())
        })?;
    }

    working = working
        .sort_values([row_index_name.as_str()], CudfSortOptions::new().with_stable(true))
        .and_then(|frame| frame.without_columns(temp_columns.iter().map(String::as_str)))
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.cumulative.cleanup",
                "failed to restore cumulative window input order",
                error,
            )
        })?;
    metrics.operator_time += timer.elapsed();
    Ok(working)
}

fn cumulative_scan_agg(function: AggregateFunction) -> Result<ScanAgg> {
    match function {
        AggregateFunction::Sum => Ok(ScanAgg::Sum),
        AggregateFunction::Min => Ok(ScanAgg::Min),
        AggregateFunction::Max => Ok(ScanAgg::Max),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.window.cumulative.function",
            "native cumulative aggregate window function is unsupported",
        )
        .with_fact("function", format!("{function:?}"))),
    }
}

fn execute_rank_window(
    context: &ExecutionContext,
    node: &WindowNode,
    frame: DataFrame,
    alias: &str,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    if node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "nexus_query_engine.window.rank",
            "native rank windows require ORDER BY",
        ));
    }
    let timer = Instant::now();
    let mut output = frame;
    let mut temp_columns = Vec::new();
    let (with_partition_key, partition_keys) =
        rank_partition_keys(context, output, &node.spec.partition_by, &mut temp_columns)?;
    output = with_partition_key;
    let mut peer_keys = partition_keys.clone();
    for sort_expr in &node.spec.order_by {
        peer_keys.push(lower_column_expr(&sort_expr.expr, "nexus_query_engine.window.rank.order_key")?);
    }

    let partition_rank_name = unique_temp_column_name(&output, "__nexus_native_window_partition_rank");
    let peer_rank_name = unique_temp_column_name(&output, "__nexus_native_window_peer_rank");
    let partition_rank = output.group_cumcount(partition_keys.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.rank.partition",
            "failed to compute partition row numbers",
            error,
        )
    })?;
    let peer_rank = output.group_cumcount(peer_keys.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.window.rank.peer", "failed to compute rank peer offsets", error)
    })?;
    output = output
        .with_column(partition_rank_name.as_str(), partition_rank)
        .and_then(|frame| frame.with_column(peer_rank_name.as_str(), peer_rank))
        .map_err(|error| {
            QueryError::from_cudf("nexus_query_engine.window.rank.columns", "failed to append rank work columns", error)
        })?;

    let uint64 = cudf::DataType::new(cudf::TypeId::UInt64);
    let rank_expr = (col(partition_rank_name.as_str()).cast(uint64) + lit(1u64)
        - col(peer_rank_name.as_str()).cast(uint64))
    .alias(alias);
    output = output.with_execution(&overrides).with_columns([rank_expr]).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.window.rank.output", "failed to append rank output column", error)
    })?;
    temp_columns.push(partition_rank_name);
    temp_columns.push(peer_rank_name);
    output = output.without_columns(temp_columns.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.window.rank.cleanup", "failed to remove rank work columns", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(output)
}

fn execute_row_number_window(
    context: &ExecutionContext,
    node: &WindowNode,
    frame: DataFrame,
    alias: &str,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let mut output = frame;
    let mut temp_columns = Vec::new();
    let (with_partition_key, partition_keys) =
        rank_partition_keys(context, output, &node.spec.partition_by, &mut temp_columns)?;
    output = with_partition_key;

    let partition_rank_name = unique_temp_column_name(&output, "__nexus_native_window_row_number_partition_rank");
    let partition_rank = output.group_cumcount(partition_keys.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.row_number.partition",
            "failed to compute partition row numbers",
            error,
        )
    })?;
    output = output.with_column(partition_rank_name.as_str(), partition_rank).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.row_number.columns",
            "failed to append row_number work column",
            error,
        )
    })?;

    let uint64 = cudf::DataType::new(cudf::TypeId::UInt64);
    let row_number_expr = (col(partition_rank_name.as_str()).cast(uint64) + lit(1u64)).alias(alias);
    output = output.with_execution(&overrides).with_columns([row_number_expr]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.row_number.output",
            "failed to append row_number output column",
            error,
        )
    })?;
    temp_columns.push(partition_rank_name);
    output = output.without_columns(temp_columns.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.row_number.cleanup",
            "failed to remove row_number work columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(output)
}

fn execute_lag_lead_window(
    context: &ExecutionContext,
    node: &WindowNode,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    value_expr: &Expr,
    offset: u32,
    default: Option<&crate::expr::LiteralValue>,
    alias: &str,
    direction: crate::plan::ShiftDirection,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();

    let output_field = output_schema.field_with_name(alias).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.window.lag_lead",
            "LAG / LEAD output column not found in window schema",
        )
        .with_fact("alias", alias)
        .with_fact("arrow_error", error.to_string())
    })?;
    let output_arrow_dtype = output_field.data_type().clone();

    // 1. Append a stable row_index column so we can recover input row order
    //    after libcudf's group-defined output order.
    let row_index_name = unique_temp_column_name(&frame, "__nexus_native_window_lag_lead_row_index");
    let row_index_series = frame.row_positions().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.row_index",
            "failed to construct LAG / LEAD row index column",
            error,
        )
    })?;
    let working = frame.with_column(row_index_name.as_str(), row_index_series).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.row_index",
            "failed to append LAG / LEAD row index column",
            error,
        )
    })?;

    // 2. Materialize partition keys (or a synthesized global key).
    let mut temp_columns: Vec<String> = Vec::new();
    let (working, partition_keys) = rank_partition_keys(context, working, &node.spec.partition_by, &mut temp_columns)?;

    // 3. Sort working by (partition_keys, order_keys). row_index rides along.
    let mut sort_columns = partition_keys.clone();
    let mut descending = vec![false; partition_keys.len()];
    let mut nulls_first = vec![false; partition_keys.len()];
    for sort_expr in &node.spec.order_by {
        sort_columns.push(lower_column_expr(&sort_expr.expr, "nexus_query_engine.window.lag_lead.order_key")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    let sort_options = CudfSortOptions::new().with_descending(descending).with_nulls_first(nulls_first);
    let sorted = working.with_execution(&overrides).sort_values(sort_columns, sort_options).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.sort",
            "failed to sort LAG / LEAD window input",
            error,
        )
    })?;

    // 4. Materialize the shifted value column on the sorted frame.
    let value_name = unique_temp_column_name(&sorted, "__nexus_native_window_lag_lead_value");
    let value_expr_cudf = lower_expr(value_expr)?.alias(value_name.as_str());
    let sorted = sorted.with_execution(&overrides).with_columns([value_expr_cudf]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.value",
            "failed to materialize LAG / LEAD value column",
            error,
        )
    })?;

    // 5. Build fill scalars. The shifted value's fill is the user-supplied
    //    default (or NULL of the output type). The row_index column is shifted
    //    by 0, so its fill value is never observed; we still construct a
    //    matching Int32 scalar so libcudf sees a well-typed pair.
    let value_fill = build_lag_lead_value_fill_scalar(context, default, &output_arrow_dtype)?;
    let row_index_fill = context.runtime()?.scalar(0i32).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.row_index_fill",
            "failed to construct LAG / LEAD row index fill scalar",
            error,
        )
    })?;

    let signed_offset: i32 = match direction {
        crate::plan::ShiftDirection::Lag => i32::try_from(offset).map_err(|_| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.lag_lead",
                "LAG offset exceeds the i32 range supported by libcudf",
            )
            .with_fact("offset", offset.to_string())
        })?,
        crate::plan::ShiftDirection::Lead => {
            let positive = i32::try_from(offset).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.window.lag_lead",
                    "LEAD offset exceeds the i32 range supported by libcudf",
                )
                .with_fact("offset", offset.to_string())
            })?;
            positive.checked_neg().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.window.lag_lead",
                    "LEAD offset overflows when negated to libcudf's signed shift contract",
                )
                .with_fact("offset", offset.to_string())
            })?
        },
    };

    // 6. Per-group shift the value column. The row_index column rides along
    //    with offset 0 so we can re-sort by it to recover input order.
    //
    // SQL `PARTITION BY` groups NULL keys together, so we have to request
    // `include_null_keys` from cudf-rs explicitly: `DataFrame::groupby`
    // defaults to excluding null keys, which would drop those rows from the
    // shift output and break the row-index based reattach below.
    let shift_specs = [
        CudfShiftSpec::new(value_name.as_str(), signed_offset, value_fill),
        CudfShiftSpec::new(row_index_name.as_str(), 0, row_index_fill),
    ];
    let groupby_options = GroupByOptions::builder().include_null_keys(true).build();
    let shifted = sorted
        .groupby(partition_keys.iter().map(String::as_str))
        .options(groupby_options)
        .shift_sorted(&shift_specs)
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.lag_lead.shift",
                "failed to execute grouped LAG / LEAD shift",
                error,
            )
            .with_fact("alias", alias)
        })?;

    // 7. Sort the shift output by the shifted row index to recover input
    //    row order, then peel off the shifted value column.
    let shifted_row_index_column = format!("{row_index_name}_shift");
    let shifted_value_column = format!("{value_name}_shift");
    let restored = shifted
        .with_execution(&overrides)
        .sort_values(
            [shifted_row_index_column.as_str()],
            CudfSortOptions::new().with_descending(vec![false]).with_nulls_first(vec![false]).with_stable(true),
        )
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.lag_lead.restore_order",
                "failed to restore LAG / LEAD row order",
                error,
            )
        })?;
    let shifted_series = restored.series(shifted_value_column.as_str()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.extract",
            "failed to extract shifted LAG / LEAD column",
            error,
        )
    })?;

    let result = frame.with_column(alias, shifted_series).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.window.lag_lead.attach",
            "failed to attach LAG / LEAD output column to input frame",
            error,
        )
        .with_fact("alias", alias)
    })?;

    let _ = temp_columns;
    metrics.operator_time += timer.elapsed();
    Ok(result)
}

fn build_lag_lead_value_fill_scalar(
    context: &ExecutionContext,
    default: Option<&crate::expr::LiteralValue>,
    output_arrow_dtype: &ArrowDataType,
) -> Result<CudfScalar> {
    use crate::expr::LiteralValue::*;
    let runtime = context.runtime()?;
    let null_for_output = || {
        let cudf_dtype = crate::datatypes::cudf_dtype_from_arrow_type(output_arrow_dtype).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.window.lag_lead.fill_null",
                "LAG / LEAD output Arrow type is not supported by the native engine",
            )
            .with_fact("dtype", format!("{output_arrow_dtype:?}"))
        })?;
        runtime.null_scalar_with_data_type(cudf_dtype).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.lag_lead.fill_null",
                "failed to construct LAG / LEAD null fill scalar",
                error,
            )
            .with_fact("dtype", format!("{output_arrow_dtype:?}"))
        })
    };
    match default {
        None => null_for_output(),
        Some(value) => match value {
            Boolean(v) => runtime.scalar(*v),
            Int8(v) => runtime.scalar(*v),
            Int16(v) => runtime.scalar(*v),
            Int32(v) => runtime.scalar(*v),
            Int64(v) => runtime.scalar(*v),
            UInt8(v) => runtime.scalar(*v),
            UInt16(v) => runtime.scalar(*v),
            UInt32(v) => runtime.scalar(*v),
            UInt64(v) => runtime.scalar(*v),
            Float32(v) => runtime.scalar(*v),
            Float64(v) => runtime.scalar(*v),
            Utf8(s) | LargeUtf8(s) => runtime.string_scalar(s),
            Null(_) => return null_for_output(),
            Decimal128 { .. }
            | Date32(_)
            | Date64(_)
            | Time32 { .. }
            | Time64 { .. }
            | Timestamp { .. }
            | Duration { .. }
            | IntervalMonthDayNano { .. } => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "nexus_query_engine.window.lag_lead.fill",
                    "LAG / LEAD default literal type is not yet supported by native shift",
                )
                .with_fact("literal", format!("{value:?}")));
            },
        }
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.window.lag_lead.fill_value",
                "failed to construct LAG / LEAD fill scalar",
                error,
            )
            .with_fact("literal", format!("{value:?}"))
        }),
    }
}

fn rank_partition_keys(
    _context: &ExecutionContext,
    frame: DataFrame,
    partition_by: &[Expr],
    temp_columns: &mut Vec<String>,
) -> Result<(DataFrame, Vec<String>)> {
    if !partition_by.is_empty() {
        let keys = partition_by
            .iter()
            .map(|expr| lower_column_expr(expr, "nexus_query_engine.window.rank.partition_key"))
            .collect::<Result<Vec<_>>>()?;
        return Ok((frame, keys));
    }
    let key = unique_temp_column_name(&frame, "__nexus_native_window_global_partition");
    let with_key = frame.with_columns([lit(0i32).alias(key.as_str())]).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.window.rank.global_key", "failed to append rank partition key", error)
    })?;
    temp_columns.push(key.clone());
    Ok((with_key, vec![key]))
}

fn unique_temp_column_name(frame: &DataFrame, base: &str) -> String {
    let names = frame.column_names();
    crate::naming::unique_name_with_suffix_from(base, 0, |candidate| names.contains(&candidate))
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

    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).slice(offset, length).map_err(|error| {
        QueryError::from_cudf("nexus_query_engine.limit.execute", "failed to execute nexus native limit", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(super) fn limit_keeps_all_rows(row_count: usize, spec: LimitSpec) -> Result<bool> {
    let (offset, length) = limit_slice(row_count, spec)?;
    Ok(offset == 0 && length == row_count)
}
