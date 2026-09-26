use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use query_planning::util::fnv1a64;

use super::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateSpec, ArrowDataType, AvgStateAliases, BinaryOp, CaseWhen,
    CudfAggSpec, DataFrame, DropDuplicatesOptions, DuplicateKeepOption, ExecutionBudgetToken, ExecutionContext,
    ExecutionMetrics, ExecutionOverrides, Expr, Field, GpuDataFrame, GpuResultHandle, GroupByAgg, GroupByOptions,
    LiteralValue, LoweredAggregate, NodeId, QueryError, QueryErrorCode, RecordBatch, Result, Schema,
    aggregate_spec_has_avg, aggregate_spec_has_distinct, aggregate_spec_has_filters,
    aggregate_spec_has_floating_statistical_aggregate, attach_budget_token_failure_facts, avg_sum_state_data_type,
    budgeted_value, cast_count_output_columns, cast_count_output_columns_with_budget, cast_reduction_output_columns,
    cast_reduction_output_columns_with_budget, cudf_dtype_from_arrow_type, cudf_query_error,
    cudf_query_error_with_budget_details, cudf_unsupported_type_error, execute_global_aggregate,
    execute_global_aggregate_with_budget, execution_overrides, final_global_aggregate_spec,
    final_global_aggregate_state_output_schema, finalize_avg_output_columns, finalize_avg_output_columns_with_budget,
    global_count_star_expr, global_valid_count_state_alias, lit, lit_null, lower_aggregate_function,
    lower_aggregate_spec_keys, lower_column_expr, lower_expr, lower_final_aggregate_spec, lower_named_column_expr,
    lower_partial_aggregate_spec, lower_single_stage_grouped_aggregate_spec, partial_global_aggregate_output_schema,
    partial_global_aggregate_spec, schema_with_datafusion_order_columns, sort_by_datafusion_order_if_present,
    validate_avg_output_fields,
};
use crate::{
    capability::{
        AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
        AggregateStrategyCapabilityDecision, AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision,
    },
    chunk::SourceChunk,
    exec::retry::is_retryable_resource_error,
    materialization_sizing::{PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE},
    mode::StatisticalAggregateExecutionMode,
};

mod filter_inputs;
mod filtered_common;
mod filtered_global;
mod filtered_grouped;
mod filtered_segment_policy;
mod groupby;
mod grouping_sets;
mod host_exact;
mod partial_execute;
mod partial_strategy;
mod statistical;
#[cfg(test)]
mod tests;

pub(in crate::exec) use filter_inputs::prepare_filtered_aggregate_frame;
use filtered_global::{
    execute_segmented_filtered_global_aggregate_with_budget,
    execute_segmented_filtered_global_partial_aggregate_with_budget, record_filtered_global_aggregate_single_pass,
    record_filtered_global_partial_single_pass, select_filtered_global_aggregate_strategy,
};
use filtered_grouped::{
    apply_filtered_grouped_materialization_facts, execute_segmented_filtered_grouped_aggregate_with_budget,
    execute_segmented_filtered_grouped_partial_aggregate_with_budget, record_filtered_grouped_segmented_failure,
    record_filtered_grouped_single_pass, select_filtered_grouped_aggregate_strategy,
};
pub(in crate::exec) use groupby::{
    execute_groupby_aggregate, execute_groupby_aggregate_with_budget, execute_groupby_aggregate_with_budget_observed,
};
use grouping_sets::{execute_grouping_sets_aggregate, execute_grouping_sets_partial_aggregate_with_budget};
use host_exact::execute_datafusion_exact_host_aggregate;
pub(in crate::exec) use host_exact::{
    execute_datafusion_exact_host_aggregate_batch, materialize_datafusion_exact_host_aggregate_input,
    prepare_datafusion_exact_host_aggregate_input_frame, upload_datafusion_exact_host_aggregate_output,
};
use partial_execute::{
    execute_encoded_key_partial_aggregate_with_budget, execute_prepared_sort_groupby_partial_aggregate_with_budget,
    execute_prepared_sort_groupby_single_aggregate_with_budget, prepare_prepared_sort_groupby_with_budget,
};
use partial_strategy::{
    add_prepared_sort_groupby_facts, prepared_sort_groupby_should_select_for_runtime_facts,
    reject_prepared_sort_groupby_for_runtime_facts, select_partial_aggregate_strategy_decision_with_options,
    select_prepared_sort_groupby_for_runtime_facts, source_adjacent_prepared_sort_groupby_runtime_candidate,
    source_chunk_structural_part,
};
pub(in crate::exec) use partial_strategy::{
    record_default_grouped_aggregate_strategy_decision, record_grouped_aggregate_strategy_decision,
};
use statistical::{
    execute_gpu_native_statistical_aggregate, execute_gpu_native_statistical_aggregate_with_budget,
    should_execute_gpu_native_statistical_aggregate,
};
pub(in crate::exec) use statistical::{
    execute_gpu_native_statistical_final_merge_with_budget,
    execute_gpu_native_statistical_partial_aggregate_with_budget,
    should_execute_gpu_native_statistical_partial_aggregate,
};

/// Source/chunk evidence available at a partial aggregate strategy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct PartialAggregateInputEvidence {
    /// Stable fingerprint of the source chunks that fed this partial aggregate.
    pub(in crate::exec) source_chunk_fingerprint: Option<u64>,
    /// True when this partial aggregate consumes output from source-local stages.
    pub(in crate::exec) source_adjacent_partial: bool,
    /// Row groups assigned to the source chunk feeding this partial aggregate.
    pub(in crate::exec) source_chunk_row_group_count: u64,
    /// True when the source chunk represents a whole file with unknown row groups.
    pub(in crate::exec) source_chunk_reads_all_row_groups: bool,
    /// Known source row count for the chunk before source-adjacent stages.
    pub(in crate::exec) source_chunk_known_row_count: Option<u64>,
    /// Best available source-read byte estimate for this chunk.
    pub(in crate::exec) source_chunk_estimated_source_read_bytes: Option<u64>,
    /// True when chunk sizing relied on file-size estimates.
    pub(in crate::exec) source_chunk_size_estimate_based: bool,
    /// Source-adjacent partitioning strategy applied before this partial aggregate.
    pub(in crate::exec) source_adjacent_partition_strategy: &'static str,
    /// Number of source-adjacent partitions planned for the aggregate.
    pub(in crate::exec) source_adjacent_partition_count: Option<u64>,
    /// Number of source-adjacent partitions that retained at least one input row.
    pub(in crate::exec) source_adjacent_non_empty_partition_count: Option<u64>,
    /// Number of grouping-key columns used by source-adjacent partitioning.
    pub(in crate::exec) source_adjacent_partition_key_count: Option<u64>,
    /// Largest number of chunk fragments feeding one source-adjacent partition.
    pub(in crate::exec) source_adjacent_max_partition_fan_in: Option<u64>,
}

pub(in crate::exec) fn partial_aggregate_source_chunk_fingerprint(chunks: &[SourceChunk]) -> Option<u64> {
    if chunks.is_empty() {
        return None;
    }
    let signature = chunks.iter().map(source_chunk_structural_part).collect::<Vec<_>>().join("|");
    Some(fnv1a64(signature.as_bytes()))
}

pub(in crate::exec::operators::aggregate) fn add_floating_statistical_aggregate_contract_facts(
    facts: &mut BTreeMap<&'static str, String>,
    spec: &AggregateSpec,
    mode: StatisticalAggregateExecutionMode,
) {
    let statistical_functions = spec
        .aggr_exprs
        .iter()
        .filter(|expr| expr.function.is_floating_statistical())
        .map(|expr| expr.function.as_str())
        .collect::<Vec<_>>();
    let statistical_ddof = spec
        .aggr_exprs
        .iter()
        .filter(|expr| expr.function.is_floating_statistical())
        .filter_map(|expr| expr.function.statistical_ddof().map(|ddof| format!("{}:{ddof}", expr.function.as_str())))
        .collect::<Vec<_>>();
    facts.insert("floating_statistical_aggregate", "true".to_string());
    facts.insert("statistical_aggregate_execution_mode", mode.as_str().to_string());
    facts.insert("statistical_aggregate_tolerance_contract", mode.tolerance_contract().as_str().to_string());
    facts.insert("statistical_aggregate_functions", statistical_functions.join(","));
    facts.insert("statistical_aggregate_ddof", statistical_ddof.join(","));
}

pub(in crate::exec) fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let (frame, spec, prepared_input_schema) =
        prepare_filtered_aggregate_frame(context, input.into_frame(), input_schema, &node.spec, None, metrics)?;
    if should_execute_datafusion_exact_floating_statistical_aggregate(context, &spec) {
        return execute_datafusion_exact_host_aggregate(
            context,
            None,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
            output_schema,
            metrics,
            None,
        );
    }
    if should_execute_gpu_native_statistical_aggregate(context, &spec) {
        return execute_gpu_native_statistical_aggregate(
            context,
            None,
            frame,
            spec,
            prepared_input_schema,
            output_schema,
            metrics,
        );
    }
    if spec.group_exprs.is_empty() {
        return execute_global_aggregate(context, &spec, GpuDataFrame::new(frame), output_schema, metrics);
    }
    if spec.grouping_sets.is_some() {
        return execute_grouping_sets_aggregate(
            context,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
            output_schema,
            None,
            metrics,
        );
    }

    // Lowering AVG through libcudf's groupby `MEAN` (`GroupByAgg::Mean`) keeps
    // the result scale equal to the source scale on `fixed_point` columns and
    // truncates the division remainder, so a `Decimal128(7, 2)` input averaged
    // to a `Decimal128(11, 6)` output rounds e.g. `34.65 / 2` to `17.32`
    // instead of `17.325`. Route AVG through explicit `sum + count + finalize`
    // state even in single-stage grouped execution, while keeping non-mergeable
    // aggregates such as MEDIAN on their true single-stage cuDF operators.
    let aggregate = lower_single_stage_grouped_aggregate_spec(&spec)?;
    if aggregate_spec_has_avg(&spec) {
        validate_avg_output_fields(&spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let input_rows = frame.num_rows() as u64;
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "query_engine.aggregate.execute")?;
    output = finalize_avg_output_columns(context, output, &spec, output_schema)?;
    output = cast_reduction_output_columns(output, &spec, output_schema)?;
    output = cast_count_output_columns(output, &spec, output_schema)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.aggregate.select",
            "failed to select algeon native aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.add_operator_time(elapsed);
    record_single_grouped_aggregate_strategy_decision(
        metrics,
        context,
        None,
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        aggregate_strategy_capability_decision(&spec, prepared_input_schema.as_ref(), AggregateExecutionStage::Single),
        Duration::ZERO,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    execute_aggregate_with_budget_for_node(context, None, node, input_schema, input, output_schema, token, metrics)
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(in crate::exec) fn execute_aggregate_with_budget_for_node(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let filtered_global_strategy = if node.spec.group_exprs.is_empty() && aggregate_spec_has_filters(&node.spec) {
        let decision = select_filtered_global_aggregate_strategy(
            &node.spec,
            input_schema,
            input_rows,
            token.request().budget_bytes() as u64,
            AggregateExecutionStage::Single,
        )?;
        if decision.use_segmented {
            return execute_segmented_filtered_global_aggregate_with_budget(
                context,
                node_id,
                &node.spec,
                input_schema,
                input,
                output_schema,
                token,
                decision,
                metrics,
            );
        }
        Some(decision)
    } else {
        None
    };
    let filtered_grouped_strategy = if !node.spec.group_exprs.is_empty()
        && node.spec.grouping_sets.is_none()
        && aggregate_spec_has_filters(&node.spec)
    {
        let decision = select_filtered_grouped_aggregate_strategy(
            &node.spec,
            input_schema,
            input_rows,
            token.request().budget_bytes() as u64,
            AggregateExecutionStage::Single,
        )?;
        if decision.use_segmented {
            let failure_decision = decision.clone();
            return match execute_segmented_filtered_grouped_aggregate_with_budget(
                context,
                node_id,
                &node.spec,
                input_schema,
                input,
                output_schema,
                token,
                decision,
                metrics,
            ) {
                Ok(output) => Ok(output),
                Err(error) => {
                    record_filtered_grouped_segmented_failure(
                        metrics,
                        context,
                        node_id,
                        AggregateExecutionStage::Single,
                        &node.spec,
                        input_schema,
                        input_rows,
                        None,
                        failure_decision,
                    );
                    Err(error)
                },
            };
        }
        Some(decision)
    } else {
        None
    };
    let filtered_global_exact_host = filtered_global_strategy.is_some()
        && should_execute_datafusion_exact_floating_statistical_aggregate(context, &node.spec);
    let (frame, spec, prepared_input_schema) = match prepare_filtered_aggregate_frame(
        context,
        input.into_frame(),
        input_schema,
        &node.spec,
        Some(token),
        metrics,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            if let Some(decision) = filtered_global_strategy {
                record_filtered_global_aggregate_single_pass(
                    metrics,
                    context,
                    node_id,
                    &node.spec,
                    input_schema,
                    input_rows,
                    0,
                    decision,
                );
            }
            if let Some(decision) = filtered_grouped_strategy {
                record_filtered_grouped_single_pass(
                    metrics,
                    context,
                    node_id,
                    AggregateExecutionStage::Single,
                    &node.spec,
                    input_schema,
                    input_rows,
                    0,
                    None,
                    decision,
                );
            }
            return Err(error);
        },
    };
    if should_execute_datafusion_exact_floating_statistical_aggregate(context, &spec) {
        let output = match execute_datafusion_exact_host_aggregate(
            context,
            node_id,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
            output_schema,
            metrics,
            filtered_global_strategy.as_ref().map(|decision| &decision.decision.facts),
        ) {
            Ok(output) => output,
            Err(error) => {
                if filtered_global_exact_host && let Some(decision) = filtered_global_strategy {
                    record_filtered_global_aggregate_single_pass(
                        metrics,
                        context,
                        node_id,
                        &node.spec,
                        input_schema,
                        input_rows,
                        0,
                        decision,
                    );
                }
                return Err(error);
            },
        };
        return Ok(output);
    }
    if should_execute_gpu_native_statistical_aggregate(context, &spec) {
        return execute_gpu_native_statistical_aggregate_with_budget(
            context,
            node_id,
            frame,
            spec,
            prepared_input_schema,
            output_schema,
            token,
            metrics,
        );
    }
    if spec.group_exprs.is_empty() {
        let output = match execute_global_aggregate_with_budget(
            context,
            &spec,
            GpuDataFrame::new(frame),
            output_schema,
            token,
            metrics,
        ) {
            Ok(output) => output,
            Err(error) => {
                if let Some(decision) = filtered_global_strategy {
                    record_filtered_global_aggregate_single_pass(
                        metrics,
                        context,
                        node_id,
                        &node.spec,
                        input_schema,
                        input_rows,
                        0,
                        decision,
                    );
                }
                return Err(error);
            },
        };
        if let Some(decision) = filtered_global_strategy {
            record_filtered_global_aggregate_single_pass(
                metrics,
                context,
                node_id,
                &node.spec,
                input_schema,
                input_rows,
                output.num_rows() as u64,
                decision,
            );
        }
        return Ok(output);
    }
    if spec.grouping_sets.is_some() {
        return execute_grouping_sets_aggregate(
            context,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
            output_schema,
            Some(token),
            metrics,
        );
    }

    // See `execute_aggregate` for why AVG must not lower to libcudf groupby
    // `MEAN`; the same finalize-at-output-scale contract applies here.
    let aggregate = match lower_single_stage_grouped_aggregate_spec(&spec) {
        Ok(aggregate) => aggregate,
        Err(error) => {
            if let Some(decision) = filtered_grouped_strategy {
                record_filtered_grouped_single_pass(
                    metrics,
                    context,
                    node_id,
                    AggregateExecutionStage::Single,
                    &node.spec,
                    input_schema,
                    input_rows,
                    0,
                    None,
                    decision,
                );
            }
            return Err(error);
        },
    };
    if aggregate_spec_has_avg(&spec) {
        validate_avg_output_fields(&spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let mut capability_decision =
        aggregate_strategy_capability_decision(&spec, prepared_input_schema.as_ref(), AggregateExecutionStage::Single);
    if let Some(filtered_grouped_strategy) = &filtered_grouped_strategy {
        apply_filtered_grouped_materialization_facts(&mut capability_decision, filtered_grouped_strategy);
    }
    let aggregate_strategy_options = context.aggregate_strategy_execution_options();
    let decision = select_partial_aggregate_strategy_decision_with_options(
        capability_decision,
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        None,
        aggregate_strategy_options,
        metrics,
    );
    let selection_elapsed = selection_timer.elapsed();
    if decision.selected_strategy == AggregateExecutionStrategy::PreparedSortGroupby {
        return execute_prepared_sort_groupby_single_aggregate_with_budget(
            context,
            node_id,
            frame,
            spec,
            prepared_input_schema,
            output_schema,
            aggregate,
            expected_output_names,
            decision,
            selection_elapsed,
            token,
            metrics,
        );
    }
    let timer = Instant::now();
    let mut output = match execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "query_engine.aggregate.execute",
        "failed to execute bounded algeon native aggregate",
    ) {
        Ok(output) => output,
        Err(error) => {
            record_single_grouped_aggregate_strategy_decision(
                metrics,
                context,
                node_id.map(|id| id.to_string()),
                &spec,
                &prepared_input_schema,
                input_rows,
                0,
                decision,
                selection_elapsed,
                timer.elapsed(),
            );
            return Err(error);
        },
    };
    output = finalize_avg_output_columns_with_budget(context, output, &spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.aggregate.select",
            "failed to select algeon native aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.add_operator_time(elapsed);
    record_single_grouped_aggregate_strategy_decision(
        metrics,
        context,
        node_id.map(|id| id.to_string()),
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn should_execute_datafusion_exact_floating_statistical_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
) -> bool {
    aggregate_spec_has_floating_statistical_aggregate(spec)
        && context.statistical_aggregate_execution_mode().is_datafusion_exact()
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_single_grouped_aggregate_strategy_decision(
    metrics: &ExecutionMetrics,
    context: &ExecutionContext,
    node_id: Option<String>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    elapsed: Duration,
) {
    record_single_grouped_aggregate_strategy_decision_with_metrics(
        metrics,
        context,
        node_id,
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
pub(in crate::exec::operators::aggregate) fn record_single_grouped_aggregate_strategy_decision_with_metrics(
    metrics: &ExecutionMetrics,
    context: &ExecutionContext,
    node_id: Option<String>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    mut decision: AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    prepare_elapsed: Duration,
    groupby_elapsed: Duration,
    decode_elapsed: Duration,
    elapsed: Duration,
) {
    if aggregate_spec_has_floating_statistical_aggregate(spec) {
        let mode = context.statistical_aggregate_execution_mode();
        let selected_strategy = if spec
            .aggr_exprs
            .iter()
            .filter(|expr| expr.function.is_floating_statistical())
            .all(|expr| expr.function == AggregateFunction::StddevSamp)
        {
            format!("stddev_samp_{}", mode.as_str())
        } else {
            format!("floating_statistical_{}", mode.as_str())
        };
        decision.facts.insert("selected_strategy", selected_strategy);
        add_floating_statistical_aggregate_contract_facts(&mut decision.facts, spec, mode);
    }
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id,
        AggregateExecutionStage::Single,
        context.aggregate_strategy_execution_options(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        prepare_elapsed,
        groupby_elapsed,
        decode_elapsed,
        elapsed,
        None,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(in crate::exec) fn execute_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    if node.spec.group_exprs.is_empty() && aggregate_spec_has_filters(&node.spec) {
        let decision = select_filtered_global_aggregate_strategy(
            &node.spec,
            input_schema,
            input_rows,
            token.request().budget_bytes() as u64,
            AggregateExecutionStage::Partial,
        )?;
        if decision.use_segmented {
            return execute_segmented_filtered_global_partial_aggregate_with_budget(
                context,
                node_id,
                &node.spec,
                input_schema,
                aggregate_output_schema,
                input,
                token,
                input_evidence,
                decision,
                metrics,
            );
        }
        let output = match execute_partial_aggregate_with_budget_single_pass(
            context,
            node_id,
            node,
            input_schema,
            aggregate_output_schema,
            input,
            token,
            input_evidence,
            None,
            metrics,
        ) {
            Ok(output) => output,
            Err(error) => {
                record_filtered_global_partial_single_pass(
                    metrics,
                    context,
                    node_id,
                    &node.spec,
                    input_schema,
                    0,
                    input_evidence,
                    decision,
                );
                return Err(error);
            },
        };
        record_filtered_global_partial_single_pass(
            metrics,
            context,
            node_id,
            &node.spec,
            input_schema,
            output.num_rows() as u64,
            input_evidence,
            decision,
        );
        return Ok(output);
    }
    let filtered_grouped_strategy = if !node.spec.group_exprs.is_empty()
        && node.spec.grouping_sets.is_none()
        && aggregate_spec_has_filters(&node.spec)
    {
        let decision = select_filtered_grouped_aggregate_strategy(
            &node.spec,
            input_schema,
            input_rows,
            token.request().budget_bytes() as u64,
            AggregateExecutionStage::Partial,
        )?;
        if decision.use_segmented {
            let failure_decision = decision.clone();
            return match execute_segmented_filtered_grouped_partial_aggregate_with_budget(
                context,
                node_id,
                &node.spec,
                input_schema,
                aggregate_output_schema,
                input,
                token,
                input_evidence,
                decision,
                metrics,
            ) {
                Ok(output) => Ok(output),
                Err(error) => {
                    record_filtered_grouped_segmented_failure(
                        metrics,
                        context,
                        node_id,
                        AggregateExecutionStage::Partial,
                        &node.spec,
                        input_schema,
                        input_rows,
                        input_evidence,
                        failure_decision,
                    );
                    Err(error)
                },
            };
        }
        Some(decision)
    } else {
        None
    };
    execute_partial_aggregate_with_budget_single_pass(
        context,
        node_id,
        node,
        input_schema,
        aggregate_output_schema,
        input,
        token,
        input_evidence,
        filtered_grouped_strategy.as_ref(),
        metrics,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
fn execute_partial_aggregate_with_budget_single_pass(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    filtered_grouped_strategy: Option<&filtered_grouped::FilteredGroupedAggregateStrategyDecision>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let original_input_rows = input.num_rows() as u64;
    let (frame, spec, prepared_input_schema) = match prepare_filtered_aggregate_frame(
        context,
        input.into_frame(),
        input_schema,
        &node.spec,
        Some(token),
        metrics,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            if let Some(decision) = filtered_grouped_strategy.cloned() {
                record_filtered_grouped_single_pass(
                    metrics,
                    context,
                    node_id,
                    AggregateExecutionStage::Partial,
                    &node.spec,
                    input_schema,
                    original_input_rows,
                    0,
                    input_evidence,
                    decision,
                );
            }
            return Err(error);
        },
    };
    if spec.group_exprs.is_empty() {
        // Schema derivation returns the structured unsupported error for MEDIAN; the spec lowering treats it as
        // unreachable, so the schema must be derived first.
        let output_schema =
            partial_global_aggregate_output_schema(&spec, &prepared_input_schema, aggregate_output_schema)?;
        let partial_spec = partial_global_aggregate_spec(&spec);
        return execute_global_aggregate_with_budget(
            context,
            &partial_spec,
            GpuDataFrame::new(frame),
            &output_schema,
            token,
            metrics,
        );
    }
    if spec.grouping_sets.is_some() {
        return execute_grouping_sets_partial_aggregate_with_budget(
            context,
            &spec,
            &prepared_input_schema,
            aggregate_output_schema,
            GpuDataFrame::new(frame),
            token,
            metrics,
        );
    }

    let aggregate = match lower_partial_aggregate_spec(&spec) {
        Ok(aggregate) => aggregate,
        Err(error) => {
            if let Some(decision) = filtered_grouped_strategy.cloned() {
                record_filtered_grouped_single_pass(
                    metrics,
                    context,
                    node_id,
                    AggregateExecutionStage::Partial,
                    &node.spec,
                    input_schema,
                    original_input_rows,
                    0,
                    input_evidence,
                    decision,
                );
            }
            return Err(error);
        },
    };
    let expected_output_names = aggregate.output_names.clone();
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let mut capability_decision =
        aggregate_strategy_capability_decision(&spec, prepared_input_schema.as_ref(), AggregateExecutionStage::Partial);
    if let Some(filtered_grouped_strategy) = filtered_grouped_strategy {
        apply_filtered_grouped_materialization_facts(&mut capability_decision, filtered_grouped_strategy);
    }
    let aggregate_strategy_options = context.aggregate_strategy_execution_options();
    let mut decision = select_partial_aggregate_strategy_decision_with_options(
        capability_decision,
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        input_evidence,
        aggregate_strategy_options,
        metrics,
    );
    let selection_elapsed = selection_timer.elapsed();
    let mut prepared_partial_input = None;
    let mut prepared_preflight_elapsed = Duration::ZERO;
    if source_adjacent_prepared_sort_groupby_runtime_candidate(
        &decision,
        input_rows,
        input_evidence,
        aggregate_strategy_options,
    ) {
        let preflight_timer = Instant::now();
        match prepare_prepared_sort_groupby_with_budget(
            context,
            &frame,
            &aggregate,
            token,
            metrics,
            "query_engine.aggregate.prepare_partial.prepared_sort_groupby_runtime_preflight",
        ) {
            Ok((prepared, preflight_elapsed)) => {
                let facts = prepared.prepared_facts();
                add_prepared_sort_groupby_facts(
                    &mut decision,
                    facts.num_groups,
                    facts.grouped_key_output_bytes,
                    facts.max_group_size,
                );
                if prepared_sort_groupby_should_select_for_runtime_facts(
                    &decision,
                    input_rows,
                    facts.num_groups,
                    aggregate_strategy_options,
                ) {
                    decision = select_prepared_sort_groupby_for_runtime_facts(decision);
                    prepared_partial_input = Some((prepared, preflight_elapsed));
                } else {
                    decision = reject_prepared_sort_groupby_for_runtime_facts(decision);
                    prepared_preflight_elapsed = preflight_elapsed;
                }
            },
            Err(error) => {
                let Some(declined) =
                    decline_prepared_sort_groupby_runtime_preflight_for_resource_error(decision, &error)
                else {
                    return Err(error);
                };
                decision = declined;
                prepared_preflight_elapsed = preflight_timer.elapsed();
            },
        }
    }
    if decision.selected_strategy == AggregateExecutionStrategy::EncodedKeyPartial {
        return execute_encoded_key_partial_aggregate_with_budget(
            context,
            node_id,
            frame,
            spec,
            prepared_input_schema,
            aggregate_output_schema,
            aggregate,
            expected_output_names,
            decision,
            selection_elapsed,
            input_evidence,
            token,
            metrics,
        );
    }
    if decision.selected_strategy == AggregateExecutionStrategy::PreparedSortGroupby {
        return execute_prepared_sort_groupby_partial_aggregate_with_budget(
            context,
            node_id,
            frame,
            spec,
            prepared_input_schema,
            aggregate_output_schema,
            aggregate,
            decision,
            selection_elapsed,
            input_evidence,
            prepared_partial_input,
            token,
            metrics,
        );
    }
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let groupby_timer = Instant::now();
    let mut output = match execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "query_engine.aggregate.execute_partial",
        "failed to execute bounded native partial cudf aggregate",
    ) {
        Ok(output) => output,
        Err(error) => {
            let elapsed = timer.elapsed().saturating_add(prepared_preflight_elapsed);
            let groupby_elapsed = groupby_timer.elapsed().saturating_add(prepared_preflight_elapsed);
            record_grouped_aggregate_strategy_decision(
                metrics,
                node_id.map(|id| id.to_string()),
                AggregateExecutionStage::Partial,
                context.aggregate_strategy_execution_options(),
                &spec,
                &prepared_input_schema,
                input_rows,
                0,
                decision.clone(),
                selection_elapsed,
                Duration::ZERO,
                groupby_elapsed,
                Duration::ZERO,
                elapsed,
                input_evidence,
            );
            return Err(error);
        },
    };
    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.aggregate.select_partial",
            "failed to select native partial aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed().saturating_add(prepared_preflight_elapsed);
    let groupby_elapsed = groupby_timer.elapsed().saturating_add(prepared_preflight_elapsed);
    let output_rows = output.num_rows() as u64;
    metrics.add_operator_time(elapsed);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Partial,
        context.aggregate_strategy_execution_options(),
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        Duration::ZERO,
        groupby_elapsed,
        Duration::ZERO,
        elapsed,
        input_evidence,
    );
    Ok(GpuDataFrame::new(output))
}

fn decline_prepared_sort_groupby_runtime_preflight_for_resource_error(
    mut decision: AggregateStrategyCapabilityDecision,
    error: &QueryError,
) -> Option<AggregateStrategyCapabilityDecision> {
    if !is_retryable_resource_error(error) {
        return None;
    }
    decision.facts.insert("prepared_sort_groupby_runtime_preflight_status", "declined_resource_error".to_string());
    decision.facts.insert("prepared_sort_groupby_runtime_preflight_error_code", error.code().as_str().to_string());
    Some(decision)
}
