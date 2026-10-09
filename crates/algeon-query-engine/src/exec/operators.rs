use std::{sync::Arc, time::Instant};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use cudf::ScalarValue;
use query_planning::{
    expression_lowering_policy::ExpressionLoweringPolicy,
    naming::unique_name_with_suffix_from,
    plan::{ShiftDirection, WindowFunction},
};

use super::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateSpec, ArrowDataType, AvgStateAliases, BinaryOp,
    BudgetedExecutionOutcome, CaseWhen, CudfAggSpec, CudfScalar, CudfShiftSpec, CudfSortOptions, DataFrame,
    DropDuplicatesOptions, DuplicateKeepOption, Exec, ExecutionBudgetRequest, ExecutionBudgetToken, ExecutionContext,
    ExecutionMetrics, Expr, Field, FilterNode, GpuDataFrame, GpuResultHandle, GroupByAgg, GroupByOptions, LimitNode,
    LimitSpec, LiteralValue, LoweredAggregate, NodeId, ProjectionNode, QueryError, QueryErrorCode, RecordBatch,
    ReduceAgg, Result, ScanAgg, Schema, WindowFrame, WindowNode, aggregate_expr_is_exact_nunique,
    aggregate_spec_has_avg, aggregate_spec_has_distinct, aggregate_spec_has_filters,
    aggregate_spec_has_floating_statistical_aggregate, col, cudf_dtype_from_arrow_type, cudf_query_error,
    cudf_unsupported_type_error, datafusion_order_columns, global_count_star_expr, global_valid_count_state_alias,
    is_supported_arrow_implicit_cast, limit_slice, lit, lit_null, lower_aggregate_function, lower_aggregate_spec_keys,
    lower_column_expr, lower_expr, lower_filter_predicate_expr, lower_final_aggregate_spec, lower_named_column_expr,
    lower_partial_aggregate_spec, lower_projection_output_expr, lower_single_stage_grouped_aggregate_spec,
    memory_failure, projection_schema_mismatch_error, schema_with_datafusion_order_columns,
    sort_by_datafusion_order_if_present, validate_avg_output_fields, validate_global_aggregate_capability, when,
};

mod aggregate;
mod global_aggregate;
mod projection;
pub(super) use aggregate::{
    PartialAggregateInputEvidence, execute_aggregate, execute_aggregate_with_budget,
    execute_aggregate_with_budget_for_node, execute_datafusion_exact_host_aggregate_batch,
    execute_gpu_native_statistical_final_merge_with_budget,
    execute_gpu_native_statistical_partial_aggregate_with_budget, execute_groupby_aggregate_with_budget_observed,
    execute_partial_aggregate_with_budget, materialize_datafusion_exact_host_aggregate_input,
    partial_aggregate_source_chunk_fingerprint, prepare_aggregate_input_frame,
    prepare_datafusion_exact_host_aggregate_input_frame, record_default_grouped_aggregate_strategy_decision,
    record_grouped_aggregate_strategy_decision, should_execute_datafusion_exact_floating_statistical_aggregate,
    should_execute_gpu_native_statistical_partial_aggregate, upload_datafusion_exact_host_aggregate_output,
};
pub(super) use global_aggregate::{
    cast_count_output_columns, cast_count_output_columns_with_budget, cast_reduction_output_columns,
    cast_reduction_output_columns_with_budget, execute_global_aggregate, execute_global_aggregate_with_budget,
    final_global_aggregate_spec, final_global_aggregate_state_output_schema, finalize_avg_output_columns,
    finalize_avg_output_columns_with_budget, partial_global_aggregate_output_schema, partial_global_aggregate_spec,
};
pub(super) use projection::{
    execute_projection_with_budget, execute_projection_with_budget_and_prepared, projection_regexp_replace_prepare,
};

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
            .with_fact("stage_request_bytes", budget_bytes)
    })?;
    context.execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(budget_boundary)).map_err(
        |error| {
            error
                .with_fact_if_absent("budget_boundary", budget_boundary)
                .with_fact_if_absent("stage_request_bytes", budget_bytes)
        },
    )
}

/// Records the token's budget scope on a failure from work that ran under its
/// overrides; a scope the cuDF call already attached wins.
pub(super) fn attach_budget_token_failure_facts(error: QueryError, token: &ExecutionBudgetToken) -> QueryError {
    let detail = token.budget_detail(error.operation());
    error.with_memory_failure(memory_failure(&detail))
}

pub(super) fn budgeted_value<T>(
    outcome: BudgetedExecutionOutcome<T>,
    metrics: &ExecutionMetrics,
    operation: &'static str,
) -> T {
    let (value, report) = outcome.into_parts();
    metrics.record_budgeted_execution(operation, &report);
    value
}

pub(super) fn execute_filter_with_budget(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.frame().clone();
    let overrides = token.exec();
    let timer = Instant::now();
    let output = if let Some((column, values)) = literal_in_list_filter_series(context, &node.predicate, overrides)? {
        frame.on(overrides).filter_isin(&column, &values).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.filter.execute",
                    "failed to execute algeon native IN-list filter",
                    error,
                ),
                token,
            )
        })?
    } else {
        let predicate = lower_filter_predicate_expr(&node.predicate)?;
        frame.on(overrides).filter_expr(predicate).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error("query_engine.filter.execute", "failed to execute algeon native filter", error),
                token,
            )
        })?
    };
    metrics.add_operator_time(timer.elapsed());
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

fn literal_in_list_filter_series(
    context: &ExecutionContext,
    predicate: &Expr,
    overrides: Exec<'_>,
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
            context.runtime()?.on(overrides).and_then(|runtime| runtime.series(&values))
        },
        LiteralValue::Int64(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::Int64(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.on(overrides).and_then(|runtime| runtime.series(&values))
        },
        LiteralValue::UInt32(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::UInt32(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.on(overrides).and_then(|runtime| runtime.series(&values))
        },
        LiteralValue::UInt64(_) => {
            let Some(values) = collect_in_list_literals(list, |value| match value {
                LiteralValue::UInt64(value) => Some(*value),
                _ => None,
            }) else {
                return Ok(None);
            };
            context.runtime()?.on(overrides).and_then(|runtime| runtime.series(&values))
        },
        _ => return Ok(None),
    }
    .map_err(|error| {
        cudf_query_error(
            "query_engine.filter.literal_in_list_values",
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

pub(super) fn execute_limit(
    context: &ExecutionContext,
    node: &LimitNode,
    input: GpuDataFrame,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, node.spec)? {
        return Ok(input);
    }

    let output = execute_limit_on_frame(context, node, input.frame().clone(), metrics)?;
    GpuDataFrame::try_new_with_source_residency(output.into_frame(), &input)
}

pub(super) fn execute_window(
    context: &ExecutionContext,
    node: &WindowNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.functions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.window.execute",
            "window node requires at least one function",
        ));
    }

    let mut frame = input.frame().clone();
    if node
        .spec
        .functions
        .iter()
        .any(|function| matches!(function, WindowFunction::Aggregate { frame: WindowFrame::WholePartition, .. }))
    {
        frame = execute_partition_window_aggregate(context, node, input_schema, frame, output_schema, metrics)?;
    }
    if node.spec.functions.iter().any(|function| {
        matches!(function, WindowFunction::Aggregate { frame: WindowFrame::RowsUnboundedPrecedingToCurrent, .. })
    }) {
        frame = execute_cumulative_window_aggregate(context, node, frame, metrics)?;
    }
    for function in &node.spec.functions {
        if let WindowFunction::Rank { output } = function {
            frame = execute_rank_window(context, node, frame, output.name(), metrics)?;
        }
    }
    for function in &node.spec.functions {
        if let WindowFunction::RowNumber { output } = function {
            frame = execute_row_number_window(context, node, frame, output.name(), output.data_type(), metrics)?;
        }
    }
    for function in &node.spec.functions {
        if let WindowFunction::LagLead { expr, offset, default, output, direction } = function {
            let alias = output.name();
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
        cudf_query_error("query_engine.window.select", "failed to select native window output columns", error)
    })?;
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

fn execute_partition_window_aggregate(
    context: &ExecutionContext,
    node: &WindowNode,
    input_schema: &ArrowSchemaRef,
    frame: DataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    if node.spec.partition_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.window.aggregate",
            "native partition aggregate windows currently require at least one partition key",
        ));
    }
    if !node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.window.aggregate",
            "native partition aggregate windows require a constant-in-partition frame without ORDER BY",
        ));
    }

    let group_exprs = node
        .spec
        .partition_by
        .iter()
        .map(|expr| {
            let name = lower_column_expr(expr, "query_engine.window.aggregate.partition")?;
            Ok(Expr::column(name.clone()).alias(name))
        })
        .collect::<Result<Vec<_>>>()?;
    let aggr_exprs = node
        .spec
        .functions
        .iter()
        .filter_map(|function| match function {
            WindowFunction::Aggregate { function, expr, output, frame: WindowFrame::WholePartition } => {
                Some(AggregateExpr::new(*function, expr.clone(), output.name().clone()))
            },
            WindowFunction::Aggregate { .. }
            | WindowFunction::Rank { .. }
            | WindowFunction::RowNumber { .. }
            | WindowFunction::LagLead { .. } => None,
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
        .map(|expr| lower_column_expr(expr, "query_engine.window.aggregate.merge_key"))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_aliases = node
        .spec
        .functions
        .iter()
        .filter_map(|function| match function {
            WindowFunction::Aggregate { output, frame: WindowFrame::WholePartition, .. } => Some(output.name().clone()),
            WindowFunction::Aggregate { .. }
            | WindowFunction::Rank { .. }
            | WindowFunction::RowNumber { .. }
            | WindowFunction::LagLead { .. } => None,
        })
        .collect::<Vec<_>>();
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    // The join that broadcasts each partition's aggregate back to its rows
    // emits them in hash-table order, so a row index restores input order.
    // Only the keys and the row index join, so payload columns are neither
    // gathered by the join nor reordered by the restoring sort.
    let row_index_name = unique_temp_column_name(&frame, "__algeon_native_window_aggregate_row_index");
    let row_index_series = frame.on(Exec::RETAINED).row_positions().map_err(|error| {
        cudf_query_error(
            "query_engine.window.aggregate.row_index",
            "failed to create partition aggregate window row index",
            error,
        )
    })?;
    let indexed = frame
        .select(distinct_names(keys.iter().cloned()))
        .and_then(|keyed| keyed.with_column(row_index_name.as_str(), row_index_series))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.aggregate.row_index",
                "failed to append partition aggregate window row index",
                error,
            )
        })?;
    let merged = indexed
        .on(overrides)
        .merge(
            &aggregate_frame,
            cudf::MergeOptions::new().how(cudf::JoinType::Left).on(keys.iter().map(String::as_str)),
        )
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.aggregate.merge",
                "failed to align native partition aggregate window outputs",
                error,
            )
        })?;
    let aligned = merged
        .select(std::iter::once(row_index_name.as_str()).chain(aggregate_aliases.iter().map(String::as_str)))
        .and_then(|outputs| {
            outputs.on(overrides).sort_values([row_index_name.as_str()], CudfSortOptions::new().with_stable(true))
        })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.aggregate.restore_order",
                "failed to restore partition aggregate window input order",
                error,
            )
        })?;
    attach_window_outputs(
        frame,
        &aligned,
        &aggregate_aliases,
        overrides,
        "query_engine.window.aggregate.attach",
        "failed to attach partition aggregate window outputs to input frame",
    )
}

fn window_aggregate_schema(
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    node: &WindowNode,
) -> Result<ArrowSchemaRef> {
    let mut fields = Vec::with_capacity(node.spec.partition_by.len() + node.spec.functions.len());
    for expr in &node.spec.partition_by {
        let name = lower_column_expr(expr, "query_engine.window.aggregate.schema_key")?;
        fields.push(input_schema.field_with_name(name.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.window.aggregate.schema_key",
                "partition key field is missing from input schema",
            )
            .with_fact("field", name)
        })?);
    }
    for function in &node.spec.functions {
        let WindowFunction::Aggregate { output, frame: WindowFrame::WholePartition, .. } = function else {
            continue;
        };
        let alias = output.name();
        fields.push(output_schema.field_with_name(alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.window.aggregate.schema_output",
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
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    if node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.window.cumulative",
            "native cumulative aggregate windows require ORDER BY",
        ));
    }

    let timer = Instant::now();
    let cumulative = node
        .spec
        .functions
        .iter()
        .enumerate()
        .filter_map(|(function_index, function)| match function {
            WindowFunction::Aggregate {
                function,
                expr,
                output,
                frame: WindowFrame::RowsUnboundedPrecedingToCurrent,
            } => Some((function_index, *function, expr, output.name())),
            WindowFunction::Aggregate { .. }
            | WindowFunction::Rank { .. }
            | WindowFunction::RowNumber { .. }
            | WindowFunction::LagLead { .. } => None,
        })
        .collect::<Vec<_>>();

    let row_index_name = unique_temp_column_name(&frame, "__algeon_native_window_input_row_index");
    let row_index_series = frame.on(Exec::RETAINED).row_positions().map_err(|error| {
        cudf_query_error(
            "query_engine.window.cumulative.row_index",
            "failed to create cumulative window row index",
            error,
        )
    })?;
    // The work frame below is dropped after the scans, so a synthesized
    // global key needs no cleanup.
    let (keyed, partition_keys) =
        rank_partition_keys(context, frame.clone(), &node.spec.partition_by, &mut Vec::new())?;
    let order_names = node
        .spec
        .order_by
        .iter()
        .map(|sort_expr| lower_column_expr(&sort_expr.expr, "query_engine.window.cumulative.order_key"))
        .collect::<Result<Vec<_>>>()?;

    // The scans read only the keys and the value columns. Evaluating the value
    // expressions before the sort keeps every payload column of `frame` out of
    // the sort and the order-restoring sort below.
    let value_names = cumulative
        .iter()
        .map(|(function_index, ..)| {
            unique_temp_column_name(&keyed, &format!("__algeon_native_window_cumulative_value_{function_index}"))
        })
        .collect::<Vec<_>>();
    let value_exprs = cumulative
        .iter()
        .zip(&value_names)
        .map(|((_, _, expr, _), value_name)| Ok(lower_expr(expr)?.alias(value_name.as_str())))
        .collect::<Result<Vec<_>>>()?;
    let input_names =
        distinct_names(partition_keys.iter().chain(&order_names).cloned().chain(
            cumulative.iter().flat_map(|(_, _, expr, _)| expr.columns().map(|column| column.name().to_string())),
        ));
    let valued = keyed
        .select(input_names)
        .and_then(|inputs| inputs.with_column(row_index_name.as_str(), row_index_series))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.row_index",
                "failed to append cumulative window row index",
                error,
            )
        })?
        .on(overrides)
        .with_columns(value_exprs)
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.value",
                "failed to materialize cumulative window input expression",
                error,
            )
        })?;

    let mut sort_columns = partition_keys.clone();
    let mut descending = vec![false; partition_keys.len()];
    let mut nulls_first = vec![false; partition_keys.len()];
    for (name, sort_expr) in order_names.iter().zip(&node.spec.order_by) {
        sort_columns.push(name.clone());
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    let sort_options = CudfSortOptions::new().with_descending(descending).with_nulls_first(nulls_first);
    let sort_names =
        distinct_names(sort_columns.iter().cloned().chain(value_names.iter().cloned()).chain([row_index_name.clone()]));
    let mut working = valued
        .select(sort_names)
        .and_then(|sort_input| sort_input.on(overrides).sort_values(sort_columns, sort_options))
        .map_err(|error| {
            cudf_query_error("query_engine.window.cumulative.sort", "failed to sort cumulative window input", error)
        })?;

    for ((function_index, function, _, alias), value_name) in cumulative.iter().zip(&value_names) {
        let scan_agg = cumulative_scan_agg(*function)?;

        let scan_name =
            unique_temp_column_name(&working, &format!("__algeon_native_window_cumulative_scan_{function_index}"));
        let scanned = working
            .on(Exec::RETAINED)
            .group_scan_sorted(partition_keys.iter().map(String::as_str), value_name.as_str(), scan_agg)
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.window.cumulative.scan",
                    "failed to execute grouped cumulative window scan",
                    error,
                )
                .with_fact("function", format!("{function:?}"))
            })?;
        working = working.with_column(scan_name.as_str(), scanned).map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.output",
                "failed to append cumulative window scan output",
                error,
            )
            .with_fact("scan_column", scan_name.as_str())
        })?;

        let valid_name =
            unique_temp_column_name(&working, &format!("__algeon_native_window_cumulative_valid_{function_index}"));
        let valid_expr = when(col(value_name.as_str()).is_null()).then(0i32).otherwise(1i32).alias(valid_name.as_str());
        working = working.on(Exec::RETAINED).with_columns([valid_expr]).map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.validity",
                "failed to append cumulative window validity indicator",
                error,
            )
        })?;

        let seen_name =
            unique_temp_column_name(&working, &format!("__algeon_native_window_cumulative_seen_{function_index}"));
        let seen = working
            .on(Exec::RETAINED)
            .group_scan_sorted(partition_keys.iter().map(String::as_str), valid_name.as_str(), ScanAgg::Sum)
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.window.cumulative.validity_scan",
                    "failed to execute grouped cumulative window validity scan",
                    error,
                )
            })?;
        working = working.with_column(seen_name.as_str(), seen).map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.validity_scan",
                "failed to append cumulative window validity scan output",
                error,
            )
            .with_fact("seen_column", seen_name.as_str())
        })?;

        let filled_name =
            unique_temp_column_name(&working, &format!("__algeon_native_window_cumulative_filled_{function_index}"));
        let filled = working.series(scan_name.as_str()).and_then(|series| series.on(Exec::RETAINED).ffill()).map_err(
            |error| {
                cudf_query_error(
                    "query_engine.window.cumulative.fill",
                    "failed to forward-fill cumulative window scan output",
                    error,
                )
                .with_fact("scan_column", scan_name.as_str())
            },
        )?;
        working = working.with_column(filled_name.as_str(), filled).map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.fill",
                "failed to append forward-filled cumulative window scan output",
                error,
            )
            .with_fact("filled_column", filled_name.as_str())
        })?;

        let output_expr = when(col(seen_name.as_str()).gt(0i32))
            .then(col(filled_name.as_str()))
            .otherwise(col(scan_name.as_str()))
            .alias(alias.as_str());
        working = working.on(Exec::RETAINED).with_columns([output_expr]).map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.output",
                "failed to append cumulative window output",
                error,
            )
            .with_fact("alias", alias.as_str())
        })?;
    }

    let aliases = cumulative.iter().map(|(_, _, _, alias)| alias.to_string()).collect::<Vec<_>>();
    let restored = working
        .select(std::iter::once(row_index_name.as_str()).chain(aliases.iter().map(String::as_str)))
        .and_then(|outputs| {
            outputs.on(Exec::RETAINED).sort_values([row_index_name.as_str()], CudfSortOptions::new().with_stable(true))
        })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.cumulative.cleanup",
                "failed to restore cumulative window input order",
                error,
            )
        })?;
    let output = attach_window_outputs(
        frame,
        &restored,
        &aliases,
        overrides,
        "query_engine.window.cumulative.attach",
        "failed to attach cumulative window output columns to input frame",
    )?;
    metrics.add_operator_time(timer.elapsed());
    Ok(output)
}

fn cumulative_scan_agg(function: AggregateFunction) -> Result<ScanAgg> {
    match function {
        AggregateFunction::Sum => Ok(ScanAgg::Sum),
        AggregateFunction::Min => Ok(ScanAgg::Min),
        AggregateFunction::Max => Ok(ScanAgg::Max),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.window.cumulative.function",
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
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    if node.spec.order_by.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.window.rank",
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
        peer_keys.push(lower_column_expr(&sort_expr.expr, "query_engine.window.rank.order_key")?);
    }

    let partition_rank_name = unique_temp_column_name(&output, "__algeon_native_window_partition_rank");
    let peer_rank_name = unique_temp_column_name(&output, "__algeon_native_window_peer_rank");
    let partition_rank = output.on(Exec::RETAINED).group_cumcount(&partition_keys).map_err(|error| {
        cudf_query_error("query_engine.window.rank.partition", "failed to compute partition row numbers", error)
    })?;
    let peer_rank = output.on(Exec::RETAINED).group_cumcount(&peer_keys).map_err(|error| {
        cudf_query_error("query_engine.window.rank.peer", "failed to compute rank peer offsets", error)
    })?;
    output = output
        .with_column(partition_rank_name.as_str(), partition_rank)
        .and_then(|frame| frame.with_column(peer_rank_name.as_str(), peer_rank))
        .map_err(|error| {
            cudf_query_error("query_engine.window.rank.columns", "failed to append rank work columns", error)
        })?;

    let uint64 = cudf::DataType::new(cudf::TypeId::UInt64);
    let rank_expr = (col(partition_rank_name.as_str()).cast(uint64) + lit(1u64)
        - col(peer_rank_name.as_str()).cast(uint64))
    .alias(alias);
    output = output.on(overrides).with_columns([rank_expr]).map_err(|error| {
        cudf_query_error("query_engine.window.rank.output", "failed to append rank output column", error)
    })?;
    temp_columns.push(partition_rank_name);
    temp_columns.push(peer_rank_name);
    output = output.without_columns(temp_columns.iter().map(String::as_str)).map_err(|error| {
        cudf_query_error("query_engine.window.rank.cleanup", "failed to remove rank work columns", error)
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(output)
}

/// Rejects inputs with more rows than the signed `row_number` output type can number.
fn ensure_row_number_capacity(output_type: &ArrowDataType, rows: usize) -> Result<()> {
    let max_rows = match output_type {
        ArrowDataType::Int32 => usize::try_from(i32::MAX).unwrap_or(usize::MAX),
        ArrowDataType::Int64 => usize::try_from(i64::MAX).unwrap_or(usize::MAX),
        _ => usize::MAX,
    };
    if rows > max_rows {
        return Err(QueryError::new(
            QueryErrorCode::SizeOverflow,
            "query_engine.window.row_number.output",
            "row_number output type cannot represent the input row count",
        )
        .with_fact("output_type", format!("{output_type:?}"))
        .with_fact("input_rows", rows)
        .with_fact("max_rows", max_rows));
    }
    Ok(())
}

fn execute_row_number_window(
    context: &ExecutionContext,
    node: &WindowNode,
    frame: DataFrame,
    alias: &str,
    output_type: &ArrowDataType,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    if node.spec.partition_by.is_empty() {
        ensure_row_number_capacity(output_type, frame.num_rows())?;
        let output = match output_type {
            ArrowDataType::UInt64 => frame.on(overrides).with_sequence_column(alias, 1_u64, 1_u64),
            ArrowDataType::Int64 => frame.on(overrides).with_sequence_column(alias, 1_i64, 1_i64),
            ArrowDataType::Int32 => frame.on(overrides).with_sequence_column(alias, 1_i32, 1_i32),
            _ => unreachable!("row_number output type validated before execution"),
        }
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.row_number.output",
                "failed to append global row_number output column",
                error,
            )
        })?;
        metrics.add_operator_time(timer.elapsed());
        return Ok(output);
    }
    let mut output = frame;
    let mut temp_columns = Vec::new();
    let (with_partition_key, partition_keys) =
        rank_partition_keys(context, output, &node.spec.partition_by, &mut temp_columns)?;
    output = with_partition_key;

    let partition_rank_name = unique_temp_column_name(&output, "__algeon_native_window_row_number_partition_rank");
    let partition_rank = output.on(Exec::RETAINED).group_cumcount(&partition_keys).map_err(|error| {
        cudf_query_error("query_engine.window.row_number.partition", "failed to compute partition row numbers", error)
    })?;
    output = output.with_column(partition_rank_name.as_str(), partition_rank).map_err(|error| {
        cudf_query_error("query_engine.window.row_number.columns", "failed to append row_number work column", error)
    })?;

    ensure_row_number_capacity(output_type, output.num_rows())?;
    let row_number_expr = match output_type {
        ArrowDataType::UInt64 => {
            let uint64 = cudf::DataType::new(cudf::TypeId::UInt64);
            (col(partition_rank_name.as_str()).cast(uint64) + lit(1u64)).alias(alias)
        },
        ArrowDataType::Int64 => {
            let int64 = cudf::DataType::new(cudf::TypeId::Int64);
            (col(partition_rank_name.as_str()).cast(int64) + lit(1i64)).alias(alias)
        },
        ArrowDataType::Int32 => {
            let int32 = cudf::DataType::new(cudf::TypeId::Int32);
            (col(partition_rank_name.as_str()).cast(int32) + lit(1i32)).alias(alias)
        },
        _ => unreachable!("row_number output type validated before execution"),
    };
    output = output.on(overrides).with_columns([row_number_expr]).map_err(|error| {
        cudf_query_error("query_engine.window.row_number.output", "failed to append row_number output column", error)
    })?;
    temp_columns.push(partition_rank_name);
    output = output.without_columns(temp_columns.iter().map(String::as_str)).map_err(|error| {
        cudf_query_error("query_engine.window.row_number.cleanup", "failed to remove row_number work columns", error)
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(output)
}

#[expect(clippy::too_many_arguments, reason = "execution inputs are distinct domain values that share no owner")]
fn execute_lag_lead_window(
    context: &ExecutionContext,
    node: &WindowNode,
    output_schema: &ArrowSchemaRef,
    frame: DataFrame,
    value_expr: &Expr,
    offset: u32,
    default: Option<&LiteralValue>,
    alias: &str,
    direction: ShiftDirection,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();

    let output_field = output_schema.field_with_name(alias).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.window.lag_lead",
            "LAG / LEAD output column not found in window schema",
        )
        .with_fact("alias", alias)
        .with_fact("arrow_error", error.to_string())
    })?;
    let output_arrow_dtype = output_field.data_type().clone();

    // The shift reads only the keys, the value expression and a row index, so
    // the sort below runs on those columns and `frame` keeps its payload
    // columns untouched. libcudf's grouped shift returns rows in group order;
    // the row index restores input order.
    let row_index_name = unique_temp_column_name(&frame, "__algeon_native_window_lag_lead_row_index");
    let row_index_series = frame.on(Exec::RETAINED).row_positions().map_err(|error| {
        cudf_query_error(
            "query_engine.window.lag_lead.row_index",
            "failed to construct LAG / LEAD row index column",
            error,
        )
    })?;

    // The work frame is dropped after the shift, so a synthesized global key
    // needs no cleanup.
    let (keyed, partition_keys) =
        rank_partition_keys(context, frame.clone(), &node.spec.partition_by, &mut Vec::new())?;
    let order_names = node
        .spec
        .order_by
        .iter()
        .map(|sort_expr| lower_column_expr(&sort_expr.expr, "query_engine.window.lag_lead.order_key"))
        .collect::<Result<Vec<_>>>()?;

    let value_name = unique_temp_column_name(&keyed, "__algeon_native_window_lag_lead_value");
    let value_expr_cudf = lower_expr(value_expr)?.alias(value_name.as_str());
    let input_names = distinct_names(
        partition_keys
            .iter()
            .chain(&order_names)
            .cloned()
            .chain(value_expr.columns().map(|column| column.name().to_string())),
    );
    let valued = keyed
        .select(input_names)
        .and_then(|inputs| inputs.with_column(row_index_name.as_str(), row_index_series))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.lag_lead.row_index",
                "failed to append LAG / LEAD row index column",
                error,
            )
        })?
        .on(overrides)
        .with_columns([value_expr_cudf])
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.lag_lead.value",
                "failed to materialize LAG / LEAD value column",
                error,
            )
        })?;

    // Sort by (partition_keys, order_keys). row_index rides along.
    let mut sort_columns = partition_keys.clone();
    let mut descending = vec![false; partition_keys.len()];
    let mut nulls_first = vec![false; partition_keys.len()];
    for (name, sort_expr) in order_names.iter().zip(&node.spec.order_by) {
        sort_columns.push(name.clone());
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    let sort_options = CudfSortOptions::new().with_descending(descending).with_nulls_first(nulls_first);
    let sort_names = distinct_names(sort_columns.iter().cloned().chain([value_name.clone(), row_index_name.clone()]));
    let sorted = valued
        .select(sort_names)
        .and_then(|sort_input| sort_input.on(overrides).sort_values(sort_columns, sort_options))
        .map_err(|error| {
            cudf_query_error("query_engine.window.lag_lead.sort", "failed to sort LAG / LEAD window input", error)
        })?;

    // Build fill scalars. The shifted value's fill is the user-supplied
    // default (or NULL of the output type). The row_index column is shifted
    // by 0, so its fill value is never observed; we still construct a
    // matching Int32 scalar so libcudf sees a well-typed pair.
    let value_fill = build_lag_lead_value_fill_scalar(context, default, &output_arrow_dtype)?;
    let row_index_fill = context.runtime()?.scalar(0i32).map_err(|error| {
        cudf_query_error(
            "query_engine.window.lag_lead.row_index_fill",
            "failed to construct LAG / LEAD row index fill scalar",
            error,
        )
    })?;

    let signed_offset: i32 = match direction {
        ShiftDirection::Lag => i32::try_from(offset).map_err(|_| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.window.lag_lead",
                "LAG offset exceeds the i32 range supported by libcudf",
            )
            .with_fact("offset", offset.to_string())
        })?,
        ShiftDirection::Lead => {
            let positive = i32::try_from(offset).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.window.lag_lead",
                    "LEAD offset exceeds the i32 range supported by libcudf",
                )
                .with_fact("offset", offset.to_string())
            })?;
            positive.checked_neg().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.window.lag_lead",
                    "LEAD offset overflows when negated to libcudf's signed shift contract",
                )
                .with_fact("offset", offset.to_string())
            })?
        },
    };

    // Per-group shift of the value column. The row_index column rides along
    // with offset 0 so we can re-sort by it to recover input order.
    //
    // SQL `PARTITION BY` groups NULL keys together, so we have to request
    // `include_null_keys` from cudf explicitly: `DataFrame::groupby`
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
        .on(Exec::RETAINED)
        .shift_sorted(&shift_specs)
        .map_err(|error| {
            cudf_query_error("query_engine.window.lag_lead.shift", "failed to execute grouped LAG / LEAD shift", error)
                .with_fact("alias", alias)
        })?;

    // Sort the shifted row index and value alone to recover input row order,
    // then peel off the shifted value column.
    let shifted_row_index_column = format!("{row_index_name}_shift");
    let shifted_value_column = format!("{value_name}_shift");
    let restored = shifted
        .select([shifted_row_index_column.as_str(), shifted_value_column.as_str()])
        .and_then(|outputs| {
            outputs.on(overrides).sort_values(
                [shifted_row_index_column.as_str()],
                CudfSortOptions::new().with_descending(vec![false]).with_nulls_first(vec![false]).with_stable(true),
            )
        })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.lag_lead.restore_order",
                "failed to restore LAG / LEAD row order",
                error,
            )
        })?;
    let shifted_series = restored.series(shifted_value_column.as_str()).map_err(|error| {
        cudf_query_error("query_engine.window.lag_lead.extract", "failed to extract shifted LAG / LEAD column", error)
    })?;

    // The shifted series lives on the runtime's stream and memory resource,
    // while `frame` keeps the provenance of whatever produced it (a source
    // chunk read under a stage budget token, for example). `with_column`
    // without a context rejects that mix.
    let result = frame.on(overrides).with_column(alias, shifted_series).map_err(|error| {
        cudf_query_error(
            "query_engine.window.lag_lead.attach",
            "failed to attach LAG / LEAD output column to input frame",
            error,
        )
        .with_fact("alias", alias)
    })?;

    metrics.add_operator_time(timer.elapsed());
    Ok(result)
}

fn build_lag_lead_value_fill_scalar(
    context: &ExecutionContext,
    default: Option<&LiteralValue>,
    output_arrow_dtype: &ArrowDataType,
) -> Result<CudfScalar> {
    let runtime = context.runtime()?;
    let null_for_output = || {
        let cudf_dtype = cudf_dtype_from_arrow_type(output_arrow_dtype).map_err(|error| {
            cudf_unsupported_type_error(
                "query_engine.window.lag_lead.fill_null",
                "LAG / LEAD output Arrow type is not supported by the native engine",
                error,
            )
            .with_fact("dtype", format!("{output_arrow_dtype:?}"))
        })?;
        runtime.scalar(ScalarValue::Null(cudf_dtype)).map_err(|error| {
            cudf_query_error(
                "query_engine.window.lag_lead.fill_null",
                "failed to construct LAG / LEAD null fill scalar",
                error,
            )
            .with_fact("dtype", format!("{output_arrow_dtype:?}"))
        })
    };
    match default {
        None => null_for_output(),
        Some(value) => match value {
            LiteralValue::Boolean(v) => runtime.scalar(*v),
            LiteralValue::Int8(v) => runtime.scalar(*v),
            LiteralValue::Int16(v) => runtime.scalar(*v),
            LiteralValue::Int32(v) => runtime.scalar(*v),
            LiteralValue::Int64(v) => runtime.scalar(*v),
            LiteralValue::UInt8(v) => runtime.scalar(*v),
            LiteralValue::UInt16(v) => runtime.scalar(*v),
            LiteralValue::UInt32(v) => runtime.scalar(*v),
            LiteralValue::UInt64(v) => runtime.scalar(*v),
            LiteralValue::Float32(v) => runtime.scalar(*v),
            LiteralValue::Float64(v) => runtime.scalar(*v),
            LiteralValue::Utf8(s) | LiteralValue::LargeUtf8(s) => runtime.scalar(s),
            LiteralValue::Null(_) => return null_for_output(),
            LiteralValue::Decimal128 { .. }
            | LiteralValue::Date32(_)
            | LiteralValue::Date64(_)
            | LiteralValue::Time32 { .. }
            | LiteralValue::Time64 { .. }
            | LiteralValue::Timestamp { .. }
            | LiteralValue::Duration { .. }
            | LiteralValue::IntervalMonthDayNano { .. } => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "query_engine.window.lag_lead.fill",
                    "LAG / LEAD default literal type is not yet supported by native shift",
                )
                .with_fact("literal", format!("{value:?}")));
            },
        }
        .map_err(|error| {
            cudf_query_error(
                "query_engine.window.lag_lead.fill_value",
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
            .map(|expr| lower_column_expr(expr, "query_engine.window.rank.partition_key"))
            .collect::<Result<Vec<_>>>()?;
        return Ok((frame, keys));
    }
    let key = unique_temp_column_name(&frame, "__algeon_native_window_global_partition");
    let with_key = frame.on(Exec::RETAINED).with_columns([lit(0i32).alias(key.as_str())]).map_err(|error| {
        cudf_query_error("query_engine.window.rank.global_key", "failed to append rank partition key", error)
    })?;
    temp_columns.push(key.clone());
    Ok((with_key, vec![key]))
}

/// Drops repeated names, keeping each first occurrence in place.
fn distinct_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut distinct: Vec<String> = Vec::new();
    for name in names {
        if !distinct.contains(&name) {
            distinct.push(name);
        }
    }
    distinct
}

/// Adds the `aliases` columns of `outputs`, which must be row-aligned with
/// `frame`, to `frame`. `frame` can carry a different stream or memory resource
/// than `outputs` (a source chunk read under a stage budget token, for
/// example), so the attach goes through the execution context, which moves the
/// frame's columns into the output domain once.
fn attach_window_outputs(
    frame: DataFrame,
    outputs: &DataFrame,
    aliases: &[String],
    overrides: Exec<'_>,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    aliases.iter().try_fold(frame, |attached, alias| {
        outputs
            .series(alias.as_str())
            .and_then(|series| attached.on(overrides).with_column(alias.as_str(), series))
            .map_err(|error| cudf_query_error(operation, message, error).with_fact("alias", alias.as_str()))
    })
}

fn unique_temp_column_name(frame: &DataFrame, base: &str) -> String {
    let names = frame.column_names();
    unique_name_with_suffix_from(base, 0, |candidate| names.contains(&candidate))
}

pub(super) fn execute_limit_on_frame(
    context: &ExecutionContext,
    node: &LimitNode,
    frame: DataFrame,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let row_count = frame.num_rows();
    execute_limit_spec_on_frame(context, node.spec, frame, row_count, metrics)
}

pub(super) fn execute_limit_spec_on_frame(
    context: &ExecutionContext,
    spec: LimitSpec,
    frame: DataFrame,
    row_count: usize,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let (offset, length) = limit_slice(row_count, spec)?;
    if offset == 0 && length == row_count {
        return Ok(GpuDataFrame::new(frame));
    }

    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let output = frame.on(overrides).slice(offset, length).map_err(|error| {
        cudf_query_error("query_engine.limit.execute", "failed to execute algeon native limit", error)
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(GpuDataFrame::new(output))
}

pub(super) fn limit_keeps_all_rows(row_count: usize, spec: LimitSpec) -> Result<bool> {
    let (offset, length) = limit_slice(row_count, spec)?;
    Ok(offset == 0 && length == row_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy cuGraph algorithms number their vertex codebook as Int32, so
    /// a window input with more rows than Int32 holds must fail with a typed
    /// size error instead of wrapping to negative ids.
    #[test]
    fn row_number_capacity_follows_the_signed_output_type() {
        let int32_limit = usize::try_from(i32::MAX).unwrap();
        assert!(ensure_row_number_capacity(&ArrowDataType::Int32, int32_limit).is_ok());
        let error = ensure_row_number_capacity(&ArrowDataType::Int32, int32_limit + 1).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::SizeOverflow);
        assert_eq!(error.fact_value("max_rows"), Some(i32::MAX.to_string().as_str()));

        assert!(ensure_row_number_capacity(&ArrowDataType::Int64, int32_limit + 1).is_ok());
        assert!(ensure_row_number_capacity(&ArrowDataType::UInt64, usize::MAX).is_ok());
    }
}
