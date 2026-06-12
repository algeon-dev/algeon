use std::{collections::BTreeMap, time::Duration};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;
use crate::{
    capability::{
        AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
        AggregateStrategyCapabilityDecision, aggregate_strategy_capability_decision,
    },
    chunk::SourceChunk,
    mode::StddevSampExecutionMode,
};

mod filter_inputs;
mod filtered_global;
mod filtered_grouped;
mod groupby;
mod grouping_sets;
mod host_exact;
mod partial_execute;
mod partial_strategy;
#[cfg(test)]
mod tests;
mod validate;

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
pub(in crate::exec) use groupby::{execute_groupby_aggregate, execute_groupby_aggregate_with_budget};
use grouping_sets::{execute_grouping_sets_aggregate, execute_grouping_sets_partial_aggregate_with_budget};
use host_exact::{execute_datafusion_exact_host_aggregate, host_group_expr_output_name};
pub(in crate::exec) use host_exact::{
    execute_datafusion_exact_host_aggregate_batch, materialize_datafusion_exact_host_aggregate_input,
    prepare_datafusion_exact_host_aggregate_input_frame, upload_datafusion_exact_host_aggregate_output,
};
use partial_execute::{
    execute_encoded_key_partial_aggregate_with_budget, execute_prepared_sort_groupby_partial_aggregate_with_budget,
    execute_prepared_sort_groupby_single_aggregate_with_budget,
};
use partial_strategy::{
    add_exact_grouped_key_cardinality_probe_facts, add_exact_single_key_cardinality_probe_facts,
    exact_cardinality_probe_should_select_prepared_sort_groupby, exact_grouped_key_cardinality_probe,
    exact_grouped_key_cardinality_probe_candidate,
    exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby, exact_single_key_cardinality_probe,
    exact_single_key_cardinality_probe_candidate, reject_prepared_sort_groupby_for_exact_cardinality,
    select_partial_aggregate_strategy_decision, select_prepared_sort_groupby_for_exact_cardinality,
    source_chunk_structural_part, stable_fnv1a64,
};
pub(in crate::exec) use partial_strategy::{
    record_default_grouped_aggregate_strategy_decision, record_grouped_aggregate_strategy_decision,
};
pub(in crate::exec) use validate::{
    validate_aggregate_capability, validate_aggregate_output_names, validate_final_aggregate_capability,
    validate_global_aggregate_capability,
};

/// Source/chunk evidence available at a partial aggregate strategy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct PartialAggregateInputEvidence {
    /// Stable fingerprint of the source chunks that fed this partial aggregate.
    pub source_chunk_fingerprint: Option<u64>,
    /// True when this partial aggregate consumes output from source-local stages.
    pub source_adjacent_partial: bool,
    /// Row groups assigned to the source chunk feeding this partial aggregate.
    pub source_chunk_row_group_count: u64,
    /// True when the source chunk represents a whole file with unknown row groups.
    pub source_chunk_reads_all_row_groups: bool,
    /// Known source row count for the chunk before source-adjacent stages.
    pub source_chunk_known_row_count: Option<u64>,
    /// Best available source-read byte estimate for this chunk.
    pub source_chunk_estimated_source_read_bytes: Option<u64>,
    /// True when chunk sizing relied on file-size estimates.
    pub source_chunk_size_estimate_based: bool,
    /// Source-adjacent partitioning strategy applied before this partial aggregate.
    pub source_adjacent_partition_strategy: &'static str,
    /// Number of source-adjacent partitions planned for the aggregate.
    pub source_adjacent_partition_count: Option<u64>,
    /// Number of source-adjacent partitions that retained at least one input row.
    pub source_adjacent_non_empty_partition_count: Option<u64>,
    /// Number of grouping-key columns used by source-adjacent partitioning.
    pub source_adjacent_partition_key_count: Option<u64>,
    /// Largest number of chunk fragments feeding one source-adjacent partition.
    pub source_adjacent_max_partition_fan_in: Option<u64>,
}

pub(in crate::exec) fn partial_aggregate_source_chunk_fingerprint(chunks: &[SourceChunk]) -> Option<u64> {
    if chunks.is_empty() {
        return None;
    }
    let signature = chunks.iter().map(source_chunk_structural_part).collect::<Vec<_>>().join("|");
    Some(stable_fnv1a64(signature.as_bytes()))
}

pub(in crate::exec::operators::aggregate) fn add_floating_statistical_aggregate_contract_facts(
    facts: &mut BTreeMap<&'static str, String>,
    spec: &AggregateSpec,
    mode: StddevSampExecutionMode,
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
    if spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::StddevSamp) {
        facts.insert("stddev_samp_execution_mode", mode.as_str().to_string());
        facts.insert("stddev_samp_ddof", "1".to_string());
    }
}

pub(in crate::exec) fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (frame, spec, prepared_input_schema) =
        prepare_filtered_aggregate_frame(context, input.into_frame(), input_schema, &node.spec, None, metrics)?;
    if should_execute_datafusion_exact_stddev(context, &spec) {
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
    // instead of `17.325`. Route AVG through the same explicit `sum + count +
    // finalize` state used by the partial/final and global aggregate paths so
    // the division happens at the output scale.
    let aggregate = lower_partial_aggregate_spec(&spec)?;
    if aggregate_spec_has_avg(&spec) {
        validate_avg_output_fields(&spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let overrides = execution_overrides(context);
    let input_rows = frame.num_rows() as u64;
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "nexus_query_engine.aggregate.execute")?;
    output = finalize_avg_output_columns(context, output, &spec, output_schema)?;
    output = cast_reduction_output_columns(output, &spec, output_schema)?;
    output = cast_count_output_columns(output, &spec, output_schema)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
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
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_aggregate_with_budget_for_node(context, None, node, input_schema, input, output_schema, token, metrics)
}

pub(in crate::exec) fn execute_aggregate_with_budget_for_node(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
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
    let filtered_global_exact_host =
        filtered_global_strategy.is_some() && should_execute_datafusion_exact_stddev(context, &node.spec);
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
    if should_execute_datafusion_exact_stddev(context, &spec) {
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
    let aggregate = match lower_partial_aggregate_spec(&spec) {
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

    let overrides = execution_overrides(context);
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let mut capability_decision =
        aggregate_strategy_capability_decision(&spec, prepared_input_schema.as_ref(), AggregateExecutionStage::Single);
    if let Some(filtered_grouped_strategy) = &filtered_grouped_strategy {
        apply_filtered_grouped_materialization_facts(&mut capability_decision, filtered_grouped_strategy);
    }
    let mut decision = select_partial_aggregate_strategy_decision(
        capability_decision,
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        None,
        metrics,
    );
    let mut selection_elapsed = selection_timer.elapsed();
    if let Some(key_column) =
        exact_single_key_cardinality_probe_candidate(&decision, &spec, prepared_input_schema.as_ref(), input_rows)
    {
        let exact_probe = exact_single_key_cardinality_probe(&frame, key_column, input_rows, token, metrics)?;
        selection_elapsed = selection_elapsed.saturating_add(exact_probe.elapsed);
        metrics.operator_time += exact_probe.elapsed;
        add_exact_single_key_cardinality_probe_facts(&mut decision, &exact_probe);
        if exact_cardinality_probe_should_select_prepared_sort_groupby(&decision, &exact_probe) {
            decision = select_prepared_sort_groupby_for_exact_cardinality(decision);
        } else {
            decision = reject_prepared_sort_groupby_for_exact_cardinality(decision);
        }
    } else if let Some(key_columns) =
        exact_grouped_key_cardinality_probe_candidate(&decision, &spec, prepared_input_schema.as_ref(), input_rows)
    {
        let exact_probe = exact_grouped_key_cardinality_probe(&frame, &key_columns, input_rows, token)?;
        selection_elapsed = selection_elapsed.saturating_add(exact_probe.elapsed);
        metrics.operator_time += exact_probe.elapsed;
        add_exact_grouped_key_cardinality_probe_facts(&mut decision, &exact_probe);
        if exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby(&decision, &exact_probe) {
            decision = select_prepared_sort_groupby_for_exact_cardinality(decision);
        } else {
            decision = reject_prepared_sort_groupby_for_exact_cardinality(decision);
        }
    }
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
    let mut output = match execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.execute",
        "failed to execute bounded native cudf aggregate",
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
                decision.clone(),
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
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
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

pub(in crate::exec) fn should_execute_datafusion_exact_stddev(
    context: &ExecutionContext,
    spec: &AggregateSpec,
) -> bool {
    aggregate_spec_has_floating_statistical_aggregate(spec)
        && context.stddev_samp_execution_mode().is_datafusion_exact()
}

fn record_single_grouped_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
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

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::operators::aggregate) fn record_single_grouped_aggregate_strategy_decision_with_metrics(
    metrics: &mut QueryMetrics,
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
        let mode = context.stddev_samp_execution_mode();
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
        context.memory_control_policy().as_str(),
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

pub(in crate::exec) fn execute_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &mut QueryMetrics,
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

#[allow(clippy::too_many_arguments)]
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
    metrics: &mut QueryMetrics,
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
        let partial_spec = partial_global_aggregate_spec(&spec);
        let output_schema =
            partial_global_aggregate_output_schema(&spec, &prepared_input_schema, aggregate_output_schema)?;
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
    let mut decision = select_partial_aggregate_strategy_decision(
        capability_decision,
        node_id,
        &spec,
        prepared_input_schema.as_ref(),
        input_rows,
        input_evidence,
        metrics,
    );
    let mut selection_elapsed = selection_timer.elapsed();
    if let Some(key_column) =
        exact_single_key_cardinality_probe_candidate(&decision, &spec, prepared_input_schema.as_ref(), input_rows)
    {
        let exact_probe = exact_single_key_cardinality_probe(&frame, key_column, input_rows, token, metrics)?;
        selection_elapsed = selection_elapsed.saturating_add(exact_probe.elapsed);
        metrics.operator_time += exact_probe.elapsed;
        add_exact_single_key_cardinality_probe_facts(&mut decision, &exact_probe);
        if exact_cardinality_probe_should_select_prepared_sort_groupby(&decision, &exact_probe) {
            decision = select_prepared_sort_groupby_for_exact_cardinality(decision);
        } else {
            decision = reject_prepared_sort_groupby_for_exact_cardinality(decision);
        }
    } else if let Some(key_columns) =
        exact_grouped_key_cardinality_probe_candidate(&decision, &spec, prepared_input_schema.as_ref(), input_rows)
    {
        let exact_probe = exact_grouped_key_cardinality_probe(&frame, &key_columns, input_rows, token)?;
        selection_elapsed = selection_elapsed.saturating_add(exact_probe.elapsed);
        metrics.operator_time += exact_probe.elapsed;
        add_exact_grouped_key_cardinality_probe_facts(&mut decision, &exact_probe);
        if exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby(&decision, &exact_probe) {
            decision = select_prepared_sort_groupby_for_exact_cardinality(decision);
        } else {
            decision = reject_prepared_sort_groupby_for_exact_cardinality(decision);
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
            expected_output_names,
            decision,
            selection_elapsed,
            input_evidence,
            token,
            metrics,
        );
    }
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = match execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.execute_partial",
        "failed to execute bounded native partial cudf aggregate",
    ) {
        Ok(output) => output,
        Err(error) => {
            record_grouped_aggregate_strategy_decision(
                metrics,
                node_id.map(|id| id.to_string()),
                AggregateExecutionStage::Partial,
                context.memory_control_policy().as_str(),
                &spec,
                &prepared_input_schema,
                input_rows,
                0,
                decision.clone(),
                selection_elapsed,
                Duration::ZERO,
                timer.elapsed(),
                Duration::ZERO,
                timer.elapsed(),
                input_evidence,
            );
            return Err(error);
        },
    };
    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.select_partial",
            "failed to select native partial aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Partial,
        context.memory_control_policy().as_str(),
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        decision,
        selection_elapsed,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        input_evidence,
    );
    Ok(GpuDataFrame::new(output))
}
