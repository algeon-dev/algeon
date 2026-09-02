use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

use super::{
    AggregateExecutionStage, AggregateExpr, AggregateFunction, AggregateSpec, AggregateStrategyCapabilityDecision,
    AggregateStrategyExecutionOptions, ArrowDataType, ArrowSchemaRef, AvgStateAliases, BinaryOp, CaseWhen, CudfAggSpec,
    DataFrame, ExecutionBudgetToken, ExecutionContext, Expr, GpuDataFrame, GroupByAgg, LiteralValue, LoweredAggregate,
    NodeId, QueryError, QueryErrorCode, QueryMetrics, Result, StatisticalAggregateExecutionMode,
    add_floating_statistical_aggregate_contract_facts, aggregate_spec_has_distinct, aggregate_spec_has_filters,
    aggregate_spec_has_floating_statistical_aggregate, aggregate_strategy_capability_decision, budgeted_value,
    cast_count_output_columns, cast_count_output_columns_with_budget, cast_reduction_output_columns,
    cast_reduction_output_columns_with_budget, execute_groupby_aggregate,
    execute_groupby_aggregate_with_budget_observed, execution_overrides, finalize_avg_output_columns,
    finalize_avg_output_columns_with_budget, lower_aggregate_function, lower_aggregate_spec_keys, lower_column_expr,
    lower_expr, record_grouped_aggregate_strategy_decision,
    record_single_grouped_aggregate_strategy_decision_with_metrics,
    select_partial_aggregate_strategy_decision_with_options,
};

const STATISTICAL_MOMENTS_FACT: &str = "algebraic_moments";

#[derive(Debug, Clone)]
struct StatisticalStateAliases {
    value: String,
    square: String,
    sum: String,
    count: String,
    sum_squares: String,
}

struct StatisticalAggregatePlan {
    aggregate: LoweredAggregate,
    pre_group_exprs: Vec<(String, cudf::Expr)>,
    final_exprs: Vec<(String, cudf::Expr)>,
    expected_output_names: Vec<String>,
}

struct StatisticalPartialAggregatePlan {
    aggregate: LoweredAggregate,
    pre_group_exprs: Vec<(String, cudf::Expr)>,
}

struct StatisticalFinalMergePlan {
    aggregate: LoweredAggregate,
    final_exprs: Vec<(String, cudf::Expr)>,
    expected_output_names: Vec<String>,
}

pub(super) fn should_execute_gpu_native_statistical_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
) -> bool {
    aggregate_spec_has_floating_statistical_aggregate(spec)
        && !context.statistical_aggregate_execution_mode().is_datafusion_exact()
        && !spec.group_exprs.is_empty()
        && spec.grouping_sets.is_none()
}

pub(in crate::exec) fn should_execute_gpu_native_statistical_partial_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
) -> bool {
    should_execute_gpu_native_statistical_aggregate(context, spec)
        && !aggregate_spec_has_distinct(spec)
        && !aggregate_spec_has_filters(spec)
        && spec
            .aggr_exprs
            .iter()
            .all(|expr| expr.function.is_floating_statistical() || expr.function == AggregateFunction::Avg)
}

pub(super) fn execute_gpu_native_statistical_aggregate(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let plan = lower_gpu_native_statistical_aggregate_plan(&frame, &spec, output_schema)?;
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let decision = gpu_native_statistical_decision(
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        AggregateExecutionStage::Single,
        context.aggregate_strategy_execution_options(),
        metrics,
    );
    let selection_elapsed = selection_timer.elapsed();
    let total_timer = Instant::now();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();

    let prepare_timer = Instant::now();
    let frame = if plan.pre_group_exprs.is_empty() {
        frame
    } else {
        frame.with_execution(&overrides).with_columns(plan.pre_group_exprs).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.statistical_moments.prepare",
                "failed to prepare native statistical aggregate moment columns",
                error,
            )
        })?
    };
    let prepare_elapsed = prepare_timer.elapsed();

    let groupby_timer = Instant::now();
    let mut output = execute_groupby_aggregate(
        &overrides,
        frame,
        &plan.aggregate,
        "nexus_query_engine.aggregate.statistical_moments.groupby",
    )?;
    let groupby_elapsed = groupby_timer.elapsed();

    let finalize_timer = Instant::now();
    if !plan.final_exprs.is_empty() {
        output = output.with_execution(&overrides).with_columns(plan.final_exprs).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.statistical_moments.finalize",
                "failed to finalize native statistical aggregate output columns",
                error,
            )
        })?;
    }
    output = finalize_avg_output_columns(context, output, &spec, output_schema)?;
    output = cast_reduction_output_columns(output, &spec, output_schema)?;
    output = cast_count_output_columns(output, &spec, output_schema)?;
    output =
        output.select(plan.expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.statistical_moments.select",
                "failed to select native statistical aggregate output columns",
                error,
            )
        })?;
    let finalize_elapsed = finalize_timer.elapsed();

    let elapsed = total_timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_gpu_native_statistical_decision(
        metrics,
        context,
        node_id,
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        prepare_elapsed,
        groupby_elapsed,
        finalize_elapsed,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(super) fn execute_gpu_native_statistical_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let plan = lower_gpu_native_statistical_aggregate_plan(&frame, &spec, output_schema)?;
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let decision = gpu_native_statistical_decision(
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        AggregateExecutionStage::Single,
        context.aggregate_strategy_execution_options(),
        metrics,
    );
    let selection_elapsed = selection_timer.elapsed();
    let total_timer = Instant::now();
    let overrides = token.execution_overrides();

    let prepare_timer = Instant::now();
    let frame = if plan.pre_group_exprs.is_empty() {
        frame
    } else {
        budgeted_value(
            frame.with_columns_with_budget(plan.pre_group_exprs, token).map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.aggregate.statistical_moments.prepare",
                    "failed to prepare bounded native statistical aggregate moment columns",
                    error,
                )
            })?,
            metrics,
            "nexus_query_engine.aggregate.statistical_moments.prepare",
        )
    };
    let prepare_elapsed = prepare_timer.elapsed();

    let groupby_timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &plan.aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.statistical_moments.groupby",
        "failed to execute bounded native statistical aggregate moments groupby",
    )?;
    let groupby_elapsed = groupby_timer.elapsed();

    let finalize_timer = Instant::now();
    if !plan.final_exprs.is_empty() {
        output = budgeted_value(
            output.with_columns_with_budget(plan.final_exprs, token).map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.aggregate.statistical_moments.finalize",
                    "failed to finalize bounded native statistical aggregate output columns",
                    error,
                )
            })?,
            metrics,
            "nexus_query_engine.aggregate.statistical_moments.finalize",
        );
    }
    output = finalize_avg_output_columns_with_budget(context, output, &spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output =
        output.select(plan.expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.statistical_moments.select",
                "failed to select native statistical aggregate output columns",
                error,
            )
        })?;
    let finalize_elapsed = finalize_timer.elapsed();

    let elapsed = total_timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_gpu_native_statistical_decision(
        metrics,
        context,
        node_id,
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        prepare_elapsed,
        groupby_elapsed,
        finalize_elapsed,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_gpu_native_statistical_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_names = input_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let plan = lower_gpu_native_statistical_partial_aggregate_plan(&frame, spec, &input_names)?;
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let mut decision = gpu_native_statistical_decision(
        node_id,
        spec,
        input_schema.as_ref(),
        input_rows,
        AggregateExecutionStage::Partial,
        context.aggregate_strategy_execution_options(),
        metrics,
    );
    decision.facts.insert("selected_strategy", "floating_statistical_gpu_native_tolerant_partial_moments".to_string());
    let selection_elapsed = selection_timer.elapsed();
    let total_timer = Instant::now();
    let overrides = token.execution_overrides();

    let prepare_timer = Instant::now();
    let frame = if plan.pre_group_exprs.is_empty() {
        frame
    } else {
        budgeted_value(
            frame.with_columns_with_budget(plan.pre_group_exprs, token).map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.aggregate.statistical_moments.partial_prepare",
                    "failed to prepare bounded native statistical partial aggregate moment columns",
                    error,
                )
            })?,
            metrics,
            "nexus_query_engine.aggregate.statistical_moments.partial_prepare",
        )
    };
    let prepare_elapsed = prepare_timer.elapsed();

    let groupby_timer = Instant::now();
    let output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &plan.aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.statistical_moments.partial_groupby",
        "failed to execute bounded native statistical partial aggregate moments groupby",
    )?;
    let groupby_elapsed = groupby_timer.elapsed();

    let elapsed = total_timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Partial,
        context.memory_control_policy().as_str(),
        context.aggregate_strategy_execution_options(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        prepare_elapsed,
        groupby_elapsed,
        Duration::ZERO,
        elapsed,
        None,
    );
    Ok(GpuDataFrame::new(output))
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(in crate::exec) fn execute_gpu_native_statistical_final_merge_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: &AggregateSpec,
    original_input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_gpu_native_statistical_output_fields(spec, output_schema)?;
    let input_names = original_input_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let plan = lower_gpu_native_statistical_final_merge_plan(spec, output_schema, &input_names)?;
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let mut decision = gpu_native_statistical_decision(
        node_id,
        spec,
        original_input_schema.as_ref(),
        input_rows,
        AggregateExecutionStage::FinalMerge,
        context.aggregate_strategy_execution_options(),
        metrics,
    );
    decision.facts.insert("selected_strategy", "floating_statistical_gpu_native_tolerant_final_moments".to_string());
    let selection_elapsed = selection_timer.elapsed();
    let total_timer = Instant::now();
    let overrides = token.execution_overrides();

    let groupby_timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &plan.aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.statistical_moments.final_merge_groupby",
        "failed to execute bounded native statistical final aggregate moment merge",
    )?;
    let groupby_elapsed = groupby_timer.elapsed();

    let finalize_timer = Instant::now();
    if !plan.final_exprs.is_empty() {
        output = budgeted_value(
            output.with_columns_with_budget(plan.final_exprs, token).map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.aggregate.statistical_moments.final_merge_finalize",
                    "failed to finalize bounded native statistical final aggregate output columns",
                    error,
                )
            })?,
            metrics,
            "nexus_query_engine.aggregate.statistical_moments.final_merge_finalize",
        );
    }
    output = finalize_avg_output_columns_with_budget(context, output, spec, output_schema, token, metrics)?;
    output =
        output.select(plan.expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.statistical_moments.final_merge_select",
                "failed to select native statistical final aggregate output columns",
                error,
            )
        })?;
    let finalize_elapsed = finalize_timer.elapsed();

    let elapsed = total_timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::FinalMerge,
        context.memory_control_policy().as_str(),
        context.aggregate_strategy_execution_options(),
        spec,
        original_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        Duration::ZERO,
        groupby_elapsed,
        finalize_elapsed,
        elapsed,
        None,
    );
    Ok(GpuDataFrame::new(output))
}

fn gpu_native_statistical_decision(
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::Schema,
    input_rows: u64,
    stage: AggregateExecutionStage,
    aggregate_strategy_options: AggregateStrategyExecutionOptions,
    metrics: &QueryMetrics,
) -> AggregateStrategyCapabilityDecision {
    let mut decision = select_partial_aggregate_strategy_decision_with_options(
        aggregate_strategy_capability_decision(spec, input_schema, stage),
        node_id,
        spec,
        input_schema,
        input_rows,
        None,
        aggregate_strategy_options,
        metrics,
    );
    decision.facts.insert("statistical_aggregate_gpu_native_formula", STATISTICAL_MOMENTS_FACT.to_string());
    add_floating_statistical_aggregate_contract_facts(
        &mut decision.facts,
        spec,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    );
    decision
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_gpu_native_statistical_decision(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    prepare_elapsed: Duration,
    groupby_elapsed: Duration,
    finalize_elapsed: Duration,
    elapsed: Duration,
) {
    record_single_grouped_aggregate_strategy_decision_with_metrics(
        metrics,
        context,
        node_id.map(|id| id.to_string()),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        prepare_elapsed,
        groupby_elapsed,
        finalize_elapsed,
        elapsed,
    );
}

fn lower_gpu_native_statistical_aggregate_plan(
    frame: &DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<StatisticalAggregatePlan> {
    validate_gpu_native_statistical_output_fields(spec, output_schema)?;
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut pre_group_exprs = Vec::new();
    let mut final_exprs = Vec::new();
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(3));
    let mut output_names = aggregate.output_key_names.clone();
    let expected_output_names = aggregate
        .output_key_names
        .iter()
        .chain(spec.aggr_exprs.iter().map(|expr| &expr.alias))
        .cloned()
        .collect::<Vec<_>>();
    let mut statistical_states = BTreeMap::<String, StatisticalStateAliases>::new();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column =
            lower_column_expr(&expr.expr, "nexus_query_engine.aggregate.statistical_moments.lower_expr")?;
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(state.count()),
                );
                output_names.extend(state.output_names());
            },
            function if function.is_floating_statistical() => {
                let state = match statistical_states.get(input_column.as_str()) {
                    Some(state) => state.clone(),
                    None => {
                        let state = statistical_state_aliases(frame, spec, index);
                        pre_group_exprs.extend(statistical_state_input_exprs(&input_column, &state)?);
                        aggregations
                            .push(CudfAggSpec::new(state.value.as_str(), GroupByAgg::Sum).with_alias(&state.sum));
                        aggregations.push(
                            CudfAggSpec::new(state.value.as_str(), GroupByAgg::Count { include_nulls: false })
                                .with_alias(&state.count),
                        );
                        aggregations.push(
                            CudfAggSpec::new(state.square.as_str(), GroupByAgg::Sum).with_alias(&state.sum_squares),
                        );
                        statistical_states.insert(input_column.clone(), state.clone());
                        state
                    },
                };
                final_exprs.push(finalize_statistical_expr(expr, &state, output_schema)?);
                output_names.push(expr.alias.clone());
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(input_column, lower_aggregate_function(expr)).with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    aggregate.aggregations = aggregations;
    aggregate.output_names = output_names;
    Ok(StatisticalAggregatePlan { aggregate, pre_group_exprs, final_exprs, expected_output_names })
}

fn lower_gpu_native_statistical_partial_aggregate_plan(
    frame: &DataFrame,
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
) -> Result<StatisticalPartialAggregatePlan> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut pre_group_exprs = Vec::new();
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(3));
    let mut output_names = aggregate.output_key_names.clone();
    let mut statistical_states = BTreeMap::<String, StatisticalStateAliases>::new();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column =
            lower_column_expr(&expr.expr, "nexus_query_engine.aggregate.statistical_moments.partial_lower_expr")?;
        match expr.function {
            function if function.is_floating_statistical() => {
                ensure_partial_statistical_state(
                    frame,
                    spec,
                    input_names,
                    &input_column,
                    index,
                    &mut statistical_states,
                    &mut pre_group_exprs,
                    &mut aggregations,
                    &mut output_names,
                )?;
            },
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(state.count()),
                );
                output_names.extend(state.output_names());
            },
            _ => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "nexus_query_engine.aggregate.statistical_moments.partial_lower_expr",
                    "gpu-native statistical partial aggregate supports only floating statistical aggregates and avg",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("function", expr.function.as_str()));
            },
        }
    }

    aggregate.aggregations = aggregations;
    aggregate.output_names = output_names;
    Ok(StatisticalPartialAggregatePlan { aggregate, pre_group_exprs })
}

fn lower_gpu_native_statistical_final_merge_plan(
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    input_names: &BTreeSet<String>,
) -> Result<StatisticalFinalMergePlan> {
    let keys = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(3));
    let mut output_names = keys.output_key_names.clone();
    let expected_output_names =
        keys.output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect::<Vec<_>>();
    let mut statistical_states = BTreeMap::<String, StatisticalStateAliases>::new();
    let mut final_exprs = Vec::with_capacity(spec.aggr_exprs.len());

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column =
            lower_column_expr(&expr.expr, "nexus_query_engine.aggregate.statistical_moments.final_merge_lower_expr")?;
        match expr.function {
            function if function.is_floating_statistical() => {
                let state = ensure_final_merge_statistical_state(
                    spec,
                    input_names,
                    &input_column,
                    index,
                    &mut statistical_states,
                    &mut aggregations,
                    &mut output_names,
                );
                final_exprs.push(finalize_statistical_expr(expr, &state, output_schema)?);
            },
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(state.sum(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(CudfAggSpec::new(state.count(), GroupByAgg::Sum).with_alias(state.count()));
                output_names.extend(state.output_names());
            },
            _ => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "nexus_query_engine.aggregate.statistical_moments.final_merge_lower_expr",
                    "gpu-native statistical final aggregate supports only floating statistical aggregates and avg",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("function", expr.function.as_str()));
            },
        }
    }

    Ok(StatisticalFinalMergePlan {
        aggregate: LoweredAggregate {
            key_columns: keys.output_key_names.clone(),
            output_key_names: keys.output_key_names,
            aggregations,
            output_names,
        },
        final_exprs,
        expected_output_names,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
fn ensure_partial_statistical_state(
    frame: &DataFrame,
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
    input_column: &str,
    index: usize,
    states: &mut BTreeMap<String, StatisticalStateAliases>,
    pre_group_exprs: &mut Vec<(String, cudf::Expr)>,
    aggregations: &mut Vec<CudfAggSpec>,
    output_names: &mut Vec<String>,
) -> Result<()> {
    if states.contains_key(input_column) {
        return Ok(());
    }
    let state = statistical_state_aliases_for_input_names(frame, spec, input_names, index);
    pre_group_exprs.extend(statistical_state_input_exprs(input_column, &state)?);
    aggregations.push(CudfAggSpec::new(state.value.as_str(), GroupByAgg::Sum).with_alias(&state.sum));
    aggregations.push(
        CudfAggSpec::new(state.value.as_str(), GroupByAgg::Count { include_nulls: false }).with_alias(&state.count),
    );
    aggregations.push(CudfAggSpec::new(state.square.as_str(), GroupByAgg::Sum).with_alias(&state.sum_squares));
    output_names.extend([state.sum.clone(), state.count.clone(), state.sum_squares.clone()]);
    states.insert(input_column.to_string(), state);
    Ok(())
}

fn ensure_final_merge_statistical_state(
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
    input_column: &str,
    index: usize,
    states: &mut BTreeMap<String, StatisticalStateAliases>,
    aggregations: &mut Vec<CudfAggSpec>,
    output_names: &mut Vec<String>,
) -> StatisticalStateAliases {
    if let Some(state) = states.get(input_column) {
        return state.clone();
    }
    let state = statistical_state_aliases_for_names(spec, input_names, index);
    aggregations.push(CudfAggSpec::new(state.sum.as_str(), GroupByAgg::Sum).with_alias(&state.sum));
    aggregations.push(CudfAggSpec::new(state.count.as_str(), GroupByAgg::Sum).with_alias(&state.count));
    aggregations.push(CudfAggSpec::new(state.sum_squares.as_str(), GroupByAgg::Sum).with_alias(&state.sum_squares));
    output_names.extend([state.sum.clone(), state.count.clone(), state.sum_squares.clone()]);
    states.insert(input_column.to_string(), state.clone());
    state
}

fn statistical_state_aliases(frame: &DataFrame, spec: &AggregateSpec, index: usize) -> StatisticalStateAliases {
    let input_names = frame.column_names().into_iter().map(str::to_string).collect::<BTreeSet<_>>();
    statistical_state_aliases_for_input_names(frame, spec, &input_names, index)
}

fn statistical_state_aliases_for_input_names(
    frame: &DataFrame,
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
    index: usize,
) -> StatisticalStateAliases {
    let aliases = statistical_state_aliases_for_names(spec, input_names, index);
    StatisticalStateAliases {
        value: unique_internal_column_name(frame, spec, aliases.value),
        square: unique_internal_column_name(frame, spec, aliases.square),
        sum: unique_internal_column_name(frame, spec, aliases.sum),
        count: unique_internal_column_name(frame, spec, aliases.count),
        sum_squares: unique_internal_column_name(frame, spec, aliases.sum_squares),
    }
}

fn statistical_state_aliases_for_names(
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
    index: usize,
) -> StatisticalStateAliases {
    let aliases = statistical_state_aliases_with_suffix(index, None);
    if statistical_aliases_available(spec, input_names, &aliases) {
        return aliases;
    }
    let suffix = crate::naming::first_available_numeric_suffix_from(1, |suffix| {
        let aliases = statistical_state_aliases_with_suffix(index, Some(suffix));
        !statistical_aliases_available(spec, input_names, &aliases)
    });
    statistical_state_aliases_with_suffix(index, Some(suffix))
}

fn statistical_state_aliases_with_suffix(index: usize, suffix: Option<usize>) -> StatisticalStateAliases {
    let suffix = suffix.map(|suffix| format!("_{suffix}")).unwrap_or_default();
    StatisticalStateAliases {
        value: format!("__nexus_native_stat_{index}_value{suffix}"),
        square: format!("__nexus_native_stat_{index}_square{suffix}"),
        sum: format!("__nexus_native_stat_{index}_sum{suffix}"),
        count: format!("__nexus_native_stat_{index}_count{suffix}"),
        sum_squares: format!("__nexus_native_stat_{index}_sum_squares{suffix}"),
    }
}

fn statistical_aliases_available(
    spec: &AggregateSpec,
    input_names: &BTreeSet<String>,
    aliases: &StatisticalStateAliases,
) -> bool {
    [
        aliases.value.as_str(),
        aliases.square.as_str(),
        aliases.sum.as_str(),
        aliases.count.as_str(),
        aliases.sum_squares.as_str(),
    ]
    .into_iter()
    .all(|candidate| {
        !input_names.contains(candidate)
            && !spec.aggr_exprs.iter().any(|expr| expr.alias == candidate)
            && !spec.group_exprs.iter().any(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == candidate))
    })
}

fn unique_internal_column_name(frame: &DataFrame, spec: &AggregateSpec, base: String) -> String {
    crate::naming::unique_alias_with_suffix(base.as_str(), |candidate| {
        !internal_column_name_available(frame, spec, candidate)
    })
}

fn internal_column_name_available(frame: &DataFrame, spec: &AggregateSpec, candidate: &str) -> bool {
    !frame.contains(candidate)
        && !spec.aggr_exprs.iter().any(|expr| expr.alias == candidate)
        && !spec.group_exprs.iter().any(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == candidate))
}

fn statistical_state_input_exprs(
    input_column: &str,
    state: &StatisticalStateAliases,
) -> Result<Vec<(String, cudf::Expr)>> {
    let value = Expr::cast(Expr::column(input_column), ArrowDataType::Float64);
    let square =
        Expr::binary_with_output_type(value.clone(), BinaryOp::Multiply, value.clone(), ArrowDataType::Float64);
    Ok(vec![(state.value.clone(), lower_expr(&value)?), (state.square.clone(), lower_expr(&square)?)])
}

fn finalize_statistical_expr(
    expr: &AggregateExpr,
    state: &StatisticalStateAliases,
    output_schema: &ArrowSchemaRef,
) -> Result<(String, cudf::Expr)> {
    let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.aggregate.statistical_moments.finalize",
            "floating statistical aggregate output field is missing from output schema",
        )
        .with_fact("alias", expr.alias.as_str())
    })?;
    let ddof = expr.function.statistical_ddof().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.aggregate.statistical_moments.finalize",
            "non-statistical aggregate reached statistical finalization",
        )
        .with_fact("alias", expr.alias.as_str())
        .with_fact("function", expr.function.as_str())
    })?;
    let count = Expr::column(state.count.as_str());
    let count_f64 = Expr::cast(count.clone(), ArrowDataType::Float64);
    let sum = Expr::column(state.sum.as_str());
    let sum_squares = Expr::column(state.sum_squares.as_str());
    let sum_times_sum = Expr::binary_with_output_type(sum.clone(), BinaryOp::Multiply, sum, ArrowDataType::Float64);
    let squared_mean_component =
        Expr::binary_with_output_type(sum_times_sum, BinaryOp::Divide, count_f64.clone(), ArrowDataType::Float64);
    let numerator =
        Expr::binary_with_output_type(sum_squares, BinaryOp::Subtract, squared_mean_component, ArrowDataType::Float64);
    let denominator = Expr::binary_with_output_type(
        count_f64,
        BinaryOp::Subtract,
        Expr::literal(LiteralValue::Float64(ddof as f64)),
        ArrowDataType::Float64,
    );
    let variance = Expr::binary_with_output_type(numerator, BinaryOp::Divide, denominator, ArrowDataType::Float64);
    let negative_variance = Expr::binary(variance.clone(), BinaryOp::Lt, Expr::literal(LiteralValue::Float64(0.0)));
    let safe_variance =
        Expr::case(vec![CaseWhen::new(negative_variance, Expr::literal(LiteralValue::Float64(0.0)))], variance);
    let value = match expr.function {
        AggregateFunction::StddevSamp | AggregateFunction::StddevPop => {
            Expr::sqrt(safe_variance, ArrowDataType::Float64)
        },
        AggregateFunction::VarSamp | AggregateFunction::VarPop => safe_variance,
        _ => unreachable!("statistical finalization only handles statistical aggregate functions"),
    };
    let count_too_low = Expr::binary(count, BinaryOp::LtEq, Expr::literal(LiteralValue::Int64(ddof as i64)));
    let nullable = Expr::case(
        vec![CaseWhen::new(count_too_low, Expr::literal(LiteralValue::Null(field.data_type().clone())))],
        Expr::cast(value, field.data_type().clone()),
    );
    lower_expr(&nullable).map(|lowered| (expr.alias.clone(), lowered))
}

fn validate_gpu_native_statistical_output_fields(spec: &AggregateSpec, output_schema: &ArrowSchemaRef) -> Result<()> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function.is_floating_statistical()) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.statistical_moments.validate",
                "floating statistical aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.aggregate.statistical_moments.validate",
                "gpu-native statistical aggregate output must be Float32 or Float64",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("function", expr.function.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    Ok(())
}
