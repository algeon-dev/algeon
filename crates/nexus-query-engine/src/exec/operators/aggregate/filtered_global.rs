use std::{
    cmp,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{filter_inputs::filtered_aggregate_input_type, *};

// Single-pass materialization keeps the original frame and aggregate state live, so require headroom.
const SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR: u64 = 2;
// Segment scratch target that amortizes cuDF calls while staying below the per-query budget clamp.
const SEGMENT_TARGET_TEMPORARY_BYTES: u64 = 512 * 1024 * 1024;
const TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT: &str = "__nexus_filtered_global_segment_target_bytes";
// Degenerate width estimates must still let segmented execution make forward progress.
const MIN_SEGMENT_ROWS: u64 = 1;
// Retained predicate masks need alignment and temporary headroom beyond bitmap storage.
const FILTER_PREDICATE_BYTES_PER_ROW: u64 = 2;

// Admission decision plus segmentation plan for filtered global aggregates.
#[derive(Debug, Clone)]
pub(super) struct FilteredGlobalAggregateStrategyDecision {
    pub(super) use_segmented: bool,
    pub(super) decision: AggregateStrategyCapabilityDecision,
    rows_per_chunk: u64,
    filtered_inputs_per_segment: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct FilteredGlobalSegmentRuntimeStats {
    pub(super) chunk_count: u64,
    pub(super) segment_count: u64,
    pub(super) max_segment_rows: u64,
    pub(super) state_rows_merged: u64,
    pub(super) observed_temporary_peak_bytes: u64,
    pub(super) observed_retained_peak_bytes: u64,
    pub(super) unresolved_contracts: BTreeSet<String>,
}

struct FilteredGlobalPreparedChunk {
    frame: DataFrame,
    schema: ArrowSchemaRef,
    predicate_aliases: BTreeMap<String, String>,
}

pub(super) fn select_filtered_global_aggregate_strategy(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    budget_bytes: u64,
    stage: AggregateExecutionStage,
) -> Result<FilteredGlobalAggregateStrategyDecision> {
    let footprint = estimate_filtered_global_materialization_footprint(spec, input_schema, input_rows)?;
    let unsupported_stateful_merge = spec.aggr_exprs.iter().any(|expr| {
        expr.distinct
            || matches!(
                expr.function,
                AggregateFunction::NUnique
                    | AggregateFunction::StddevSamp
                    | AggregateFunction::StddevPop
                    | AggregateFunction::VarSamp
                    | AggregateFunction::VarPop
            )
    });
    let threshold = budget_bytes / SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR;
    let single_pass_exceeds_threshold =
        footprint.filtered_count > 0 && footprint.estimated_materialization_bytes > threshold;
    if single_pass_exceeds_threshold && unsupported_stateful_merge {
        return Err(unsupported_large_filtered_global_stateful_merge_error(
            &footprint,
            input_rows,
            budget_bytes,
            threshold,
        ));
    }
    let use_segmented = footprint.filtered_count > 0 && single_pass_exceeds_threshold && !unsupported_stateful_merge;
    let target_temporary_bytes = filtered_global_segment_target_temporary_bytes(input_schema, budget_bytes);
    let retained_predicate_bytes_per_row = u64::try_from(footprint.unique_filter_predicate_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(FILTER_PREDICATE_BYTES_PER_ROW);
    let rows_per_chunk = if use_segmented {
        rows_per_chunk_for_filtered_global_segment(
            footprint.max_segment_input_bytes_per_row.saturating_add(retained_predicate_bytes_per_row),
            target_temporary_bytes,
        )
    } else {
        input_rows.max(1)
    };
    let filtered_inputs_per_segment = if use_segmented {
        filtered_inputs_per_segment_for_budget(
            footprint.max_segment_input_bytes_per_row,
            rows_per_chunk,
            retained_predicate_bytes_per_row,
            budget_bytes,
            target_temporary_bytes,
        )
    } else {
        footprint.filtered_count.max(1)
    };
    let mut decision = aggregate_strategy_capability_decision(spec, input_schema.as_ref(), stage);
    let selected_strategy =
        if use_segmented { "filtered_global_segmented_state" } else { "filtered_global_single_pass_materialized" };
    decision.facts.insert("selected_strategy", selected_strategy.to_string());
    decision.facts.insert("filtered_aggregate_scope", "global".to_string());
    decision.facts.insert(
        "filtered_input_materialization_strategy",
        if use_segmented { "row_chunked_segmented_state" } else { "single_pass_with_columns" }.to_string(),
    );
    decision.facts.insert(
        "filtered_input_materialization_estimated_bytes",
        footprint.estimated_materialization_bytes.to_string(),
    );
    decision.facts.insert("filtered_input_materialization_budget_bytes", budget_bytes.to_string());
    decision.facts.insert("filtered_input_materialization_threshold_bytes", threshold.to_string());
    decision.facts.insert(
        "filtered_input_materialization_budget_fraction",
        format!("1/{SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR}"),
    );
    decision.facts.insert("filtered_input_value_bytes", footprint.value_bytes.to_string());
    decision.facts.insert("filtered_input_validity_bytes", footprint.validity_bytes.to_string());
    decision.facts.insert("filtered_input_predicate_bytes", footprint.predicate_bytes.to_string());
    decision
        .facts
        .insert("filtered_input_original_frame_retained_bytes", footprint.original_frame_retained_bytes.to_string());
    decision.facts.insert("filtered_input_unsupported_stateful_merge", unsupported_stateful_merge.to_string());
    decision.facts.insert("filtered_input_unique_predicate_count", footprint.unique_filter_predicate_count.to_string());
    decision
        .facts
        .insert("filtered_global_retained_predicate_bytes_per_row", retained_predicate_bytes_per_row.to_string());
    decision.facts.insert("filtered_global_strategy_input_rows", input_rows.to_string());
    decision.facts.insert("filtered_global_rows_per_chunk", rows_per_chunk.to_string());
    decision.facts.insert("filtered_global_inputs_per_segment", filtered_inputs_per_segment.to_string());
    decision.facts.insert("filtered_global_segment_target_temporary_bytes", target_temporary_bytes.to_string());
    decision.facts.insert(
        "rejected_candidate_strategies",
        if use_segmented {
            "filtered_global_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold"
        } else if unsupported_stateful_merge {
            "filtered_global_segmented_state:unsupported_stateful_filtered_merge"
        } else {
            "filtered_global_segmented_state:single_pass_estimate_within_threshold"
        }
        .to_string(),
    );

    Ok(FilteredGlobalAggregateStrategyDecision { use_segmented, decision, rows_per_chunk, filtered_inputs_per_segment })
}

pub(super) fn execute_segmented_filtered_global_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    decision: FilteredGlobalAggregateStrategyDecision,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let selection_elapsed = Duration::ZERO;
    let budget_event_start = metrics.budgeted_execution_events.len();
    let timer = Instant::now();
    let mut stats = FilteredGlobalSegmentRuntimeStats::default();
    let result = (|| -> Result<(GpuDataFrame, u64, u64)> {
        let state_schema = partial_global_aggregate_output_schema(spec, input_schema, output_schema)?;
        let partial_state = execute_segmented_filtered_global_partial_state(
            context,
            spec,
            input_schema,
            input,
            output_schema,
            &state_schema,
            token,
            &decision,
            &mut stats,
            metrics,
        )?;
        let partial_rows = partial_state.num_rows() as u64;
        let merge_spec = final_global_aggregate_spec(spec);
        let merge_schema = final_global_aggregate_state_output_schema(spec, input_schema, output_schema)?;
        let mut output =
            execute_global_aggregate_with_budget(context, &merge_spec, partial_state, &merge_schema, token, metrics)?
                .into_frame();
        output = finalize_avg_output_columns_with_budget(context, output, spec, output_schema, token, metrics)?;
        output = output
            .select(output_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>())
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.filtered_global_aggregate.select",
                    "failed to select segmented filtered global aggregate output columns",
                    error,
                )
            })?;
        let output = GpuDataFrame::new(output);
        let output_rows = output.num_rows() as u64;
        Ok((output, output_rows, partial_rows))
    })();
    add_segmented_budgeted_execution_stats(&mut stats, metrics, budget_event_start);
    match result {
        Ok((output, output_rows, partial_rows)) => {
            record_segmented_filtered_global_aggregate_strategy(
                metrics,
                context,
                node_id,
                spec,
                input_schema,
                input_rows,
                output_rows,
                partial_rows,
                stats,
                decision,
                selection_elapsed,
                timer.elapsed(),
            );
            Ok(output)
        },
        Err(error) => {
            record_filtered_global_segmented_failure(
                metrics,
                context,
                node_id,
                AggregateExecutionStage::Single,
                spec,
                input_schema,
                input_rows,
                None,
                decision,
                stats,
                &error,
                timer.elapsed(),
            );
            Err(error)
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_segmented_filtered_global_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    decision: FilteredGlobalAggregateStrategyDecision,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let budget_event_start = metrics.budgeted_execution_events.len();
    let timer = Instant::now();
    let mut stats = FilteredGlobalSegmentRuntimeStats::default();
    let result = (|| -> Result<(GpuDataFrame, u64, u64)> {
        let state_schema = partial_global_aggregate_output_schema(spec, input_schema, aggregate_output_schema)?;
        let partial_state = execute_segmented_filtered_global_partial_state(
            context,
            spec,
            input_schema,
            input,
            aggregate_output_schema,
            &state_schema,
            token,
            &decision,
            &mut stats,
            metrics,
        )?;
        let partial_rows = partial_state.num_rows() as u64;
        let merge_spec = final_global_aggregate_spec(spec);
        let output =
            execute_global_aggregate_with_budget(context, &merge_spec, partial_state, &state_schema, token, metrics)?;
        let output_rows = output.num_rows() as u64;
        Ok((output, output_rows, partial_rows))
    })();
    add_segmented_budgeted_execution_stats(&mut stats, metrics, budget_event_start);
    match result {
        Ok((output, output_rows, partial_rows)) => {
            record_segmented_filtered_global_partial_strategy(
                metrics,
                context,
                node_id,
                spec,
                input_schema,
                input_rows,
                output_rows,
                partial_rows,
                stats,
                input_evidence,
                decision,
                timer.elapsed(),
            );
            Ok(output)
        },
        Err(error) => {
            record_filtered_global_segmented_failure(
                metrics,
                context,
                node_id,
                AggregateExecutionStage::Partial,
                spec,
                input_schema,
                input_rows,
                input_evidence,
                decision,
                stats,
                &error,
                timer.elapsed(),
            );
            Err(error)
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_filtered_global_partial_single_pass(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    output_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    decision: FilteredGlobalAggregateStrategyDecision,
) {
    let input_rows = decision
        .decision
        .facts
        .get("filtered_global_strategy_input_rows")
        .and_then(|rows| rows.parse::<u64>().ok())
        .unwrap_or(0);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Partial,
        context.memory_control_policy().as_str(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision.decision,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        input_evidence,
    );
}

pub(super) fn record_filtered_global_aggregate_single_pass(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: FilteredGlobalAggregateStrategyDecision,
) {
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Single,
        context.memory_control_policy().as_str(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision.decision,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_filtered_global_segmented_failure(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    stage: AggregateExecutionStage,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    mut decision: FilteredGlobalAggregateStrategyDecision,
    mut stats: FilteredGlobalSegmentRuntimeStats,
    error: &QueryError,
    elapsed: Duration,
) {
    add_segmented_failure_error_facts(&mut decision.decision.facts, &mut stats, error);
    if stats.unresolved_contracts.is_empty() {
        stats.unresolved_contracts.insert("filtered_global_segmented_failure_budget_report_unavailable".to_string());
    }
    decision.decision.facts.insert("filtered_global_segmented_failure", "true".to_string());
    add_segmented_runtime_facts(&mut decision.decision.facts, stats.state_rows_merged, stats);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        stage,
        context.memory_control_policy().as_str(),
        spec,
        input_schema,
        input_rows,
        0,
        decision.decision,
        Duration::ZERO,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        input_evidence,
    );
}

fn add_segmented_failure_error_facts(
    facts: &mut BTreeMap<&'static str, String>,
    stats: &mut FilteredGlobalSegmentRuntimeStats,
    error: &QueryError,
) {
    facts.insert("filtered_global_segmented_failure_operation", error.operation().to_string());
    facts.insert("filtered_global_segmented_failure_error_code", error.code().to_string());
    facts.insert("filtered_global_segmented_failure_error_kind", error.kind().to_string());
    facts.insert("filtered_global_segmented_failure_error_status", error.status().to_string());
    for (key, value) in error.facts().iter() {
        if key.starts_with("budget_") {
            facts.insert(key, value.to_string());
        }
    }
    for key in ["budget_peak_bytes", "budget_peak_aligned_bytes"] {
        if let Some(bytes) = error.fact_value(key).and_then(|value| value.parse::<u64>().ok()) {
            stats.observed_temporary_peak_bytes = stats.observed_temporary_peak_bytes.max(bytes);
        }
    }
    for key in ["budget_current_bytes", "budget_current_aligned_bytes"] {
        if let Some(bytes) = error.fact_value(key).and_then(|value| value.parse::<u64>().ok()) {
            stats.observed_retained_peak_bytes = stats.observed_retained_peak_bytes.max(bytes);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_segmented_filtered_global_partial_state(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    aggregate_output_schema: &ArrowSchemaRef,
    state_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    decision: &FilteredGlobalAggregateStrategyDecision,
    stats: &mut FilteredGlobalSegmentRuntimeStats,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows();
    let frame = input.into_frame();
    let rows_per_chunk = usize::try_from(decision.rows_per_chunk).unwrap_or(usize::MAX).max(MIN_SEGMENT_ROWS as usize);
    let mut partial_frames = Vec::new();
    let mut offset = 0usize;
    loop {
        let length = if input_rows == 0 { 0 } else { rows_per_chunk.min(input_rows.saturating_sub(offset)) };
        let chunk = if length == input_rows {
            frame.clone()
        } else {
            frame.with_execution(&token.execution_overrides()).slice(offset, length).map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "nexus_query_engine.filtered_global_aggregate.slice",
                        "failed to slice input for segmented filtered global aggregate",
                        error,
                    ),
                    token,
                )
                .with_fact("chunk_offset", offset)
                .with_fact("chunk_rows", length)
            })?
        };
        let chunk_frame = execute_filtered_global_partial_state_for_chunk(
            context,
            spec,
            input_schema,
            chunk,
            aggregate_output_schema,
            state_schema,
            token,
            decision.filtered_inputs_per_segment,
            length,
            stats,
            metrics,
        )?;
        stats.chunk_count = stats.chunk_count.saturating_add(1);
        stats.max_segment_rows = stats.max_segment_rows.max(length as u64);
        partial_frames.push(chunk_frame);
        if input_rows == 0 || offset + length >= input_rows {
            break;
        }
        offset += length;
    }
    let frame = concat_partial_state_frames(partial_frames, token)?;
    stats.state_rows_merged = frame.num_rows() as u64;
    Ok(frame)
}

#[allow(clippy::too_many_arguments)]
fn execute_filtered_global_partial_state_for_chunk(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    chunk: DataFrame,
    aggregate_output_schema: &ArrowSchemaRef,
    state_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    filtered_inputs_per_segment: usize,
    chunk_rows: usize,
    stats: &mut FilteredGlobalSegmentRuntimeStats,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let prepared_chunk = prepare_filtered_global_predicates_for_chunk(&chunk, input_schema, spec, token, metrics)?;
    let mut output_series = Vec::with_capacity(state_schema.fields().len());
    let mut segment = Vec::<(usize, AggregateExpr)>::new();
    let mut segment_filtered_count = 0usize;
    for (index, expr) in spec.aggr_exprs.iter().cloned().enumerate() {
        let is_filtered = expr.filter.is_some();
        if is_filtered && segment_filtered_count >= filtered_inputs_per_segment && !segment.is_empty() {
            append_filtered_global_partial_segment(
                context,
                input_schema,
                &prepared_chunk,
                aggregate_output_schema,
                &segment,
                &mut output_series,
                token,
                metrics,
            )?;
            stats.segment_count = stats.segment_count.saturating_add(1);
            stats.max_segment_rows = stats.max_segment_rows.max(chunk_rows as u64);
            segment.clear();
            segment_filtered_count = 0;
        }
        segment_filtered_count += usize::from(is_filtered);
        segment.push((index, expr));
    }
    if !segment.is_empty() {
        append_filtered_global_partial_segment(
            context,
            input_schema,
            &prepared_chunk,
            aggregate_output_schema,
            &segment,
            &mut output_series,
            token,
            metrics,
        )?;
        stats.segment_count = stats.segment_count.saturating_add(1);
        stats.max_segment_rows = stats.max_segment_rows.max(chunk_rows as u64);
    }
    let mut output_by_name = output_series.into_iter().collect::<BTreeMap<_, _>>();
    let ordered = state_schema
        .fields()
        .iter()
        .map(|field| {
            output_by_name.remove(field.name().as_str()).map(|series| (field.name().as_str(), series)).ok_or_else(
                || {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.filtered_global_aggregate.partial_state",
                        "segmented filtered global aggregate did not produce an expected state column",
                    )
                    .with_fact("column", field.name().as_str())
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let frame = DataFrame::from_series(ordered).map(GpuDataFrame::new).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.filtered_global_aggregate.partial_state",
            "failed to construct segmented filtered global aggregate partial state frame",
            error,
        )
    })?;
    Ok(frame)
}

fn append_filtered_global_partial_segment(
    context: &ExecutionContext,
    input_schema: &ArrowSchemaRef,
    prepared_chunk: &FilteredGlobalPreparedChunk,
    aggregate_output_schema: &ArrowSchemaRef,
    indexed_exprs: &[(usize, AggregateExpr)],
    output_series: &mut Vec<(String, cudf::Series)>,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    let (prepared, prepared_indexed, prepared_schema) =
        prepare_filtered_global_partial_segment(prepared_chunk, input_schema, indexed_exprs, token, metrics)?;
    let partial_spec = indexed_partial_global_aggregate_spec(&prepared_indexed);
    let partial_schema =
        indexed_partial_global_aggregate_output_schema(&prepared_indexed, &prepared_schema, aggregate_output_schema)?;
    let partial = execute_global_aggregate_with_budget(
        context,
        &partial_spec,
        GpuDataFrame::new(prepared),
        &partial_schema,
        token,
        metrics,
    )?
    .into_frame();
    for field in partial_schema.fields() {
        let series = partial.series(field.name().as_str()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.filtered_global_aggregate.partial_segment",
                "failed to access segmented filtered global aggregate partial state column",
                error,
            )
            .with_fact("column", field.name().as_str())
        })?;
        output_series.push((field.name().clone(), series));
    }
    Ok(())
}

fn prepare_filtered_global_predicates_for_chunk(
    chunk: &DataFrame,
    input_schema: &ArrowSchemaRef,
    spec: &AggregateSpec,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<FilteredGlobalPreparedChunk> {
    let timer = Instant::now();
    let mut frame = chunk.clone();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut predicate_aliases = BTreeMap::<String, String>::new();
    let mut predicate_expressions = Vec::<(String, cudf::Expr)>::new();
    for aggregate_expr in spec.aggr_exprs.iter() {
        let Some(filter) = &aggregate_expr.filter else {
            continue;
        };
        let predicate_key = filter.to_string();
        if predicate_aliases.contains_key(predicate_key.as_str()) {
            continue;
        }
        let alias = unique_filtered_global_segment_alias("__cudf_filtered_global_predicate", &fields);
        predicate_aliases.insert(predicate_key, alias.clone());
        predicate_expressions.push((alias.clone(), lower_expr(filter)?));
        fields.push(Field::new(alias, ArrowDataType::Boolean, true));
    }
    if !predicate_expressions.is_empty() {
        frame = frame
            .with_columns_with_budget(predicate_expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.filtered_global_aggregate.filter_predicates",
                    "failed to materialize segmented filtered global aggregate predicates",
                    error,
                )
            })
            .map(|outcome| {
                budgeted_value(outcome, metrics, "nexus_query_engine.filtered_global_aggregate.filter_predicates")
            })?;
    }
    metrics.operator_time += timer.elapsed();
    Ok(FilteredGlobalPreparedChunk { frame, schema: Arc::new(Schema::new(fields)), predicate_aliases })
}

#[allow(clippy::type_complexity)]
fn prepare_filtered_global_partial_segment(
    prepared_chunk: &FilteredGlobalPreparedChunk,
    input_schema: &ArrowSchemaRef,
    indexed_exprs: &[(usize, AggregateExpr)],
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<(DataFrame, Vec<(usize, AggregateExpr)>, ArrowSchemaRef)> {
    let timer = Instant::now();
    let mut frame = prepared_chunk.frame.clone();
    let mut fields = prepared_chunk.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut masked_expressions = Vec::<(String, cudf::Expr)>::new();
    let mut prepared_indexed = Vec::with_capacity(indexed_exprs.len());
    for (index, aggregate_expr) in indexed_exprs.iter() {
        let Some(filter) = &aggregate_expr.filter else {
            prepared_indexed.push((*index, aggregate_expr.clone()));
            continue;
        };
        let input_type = filtered_aggregate_input_type(aggregate_expr, input_schema)?;
        let value_expr = filtered_global_aggregate_value_expr(aggregate_expr);
        let alias = unique_filtered_global_segment_alias("__cudf_filtered_global_arg", &fields);
        let predicate_alias = prepared_chunk.predicate_aliases.get(filter.to_string().as_str()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.filtered_global_aggregate.filter_predicates",
                "segmented filtered global aggregate predicate alias is missing",
            )
            .with_fact("filter", filter.to_string())
        })?;
        let masked = Expr::case(
            vec![CaseWhen::new(Expr::column(predicate_alias.clone()), value_expr)],
            Expr::literal(LiteralValue::Null(input_type.clone())),
        );
        masked_expressions.push((alias.clone(), lower_expr(&masked)?));
        fields.push(Field::new(alias.as_str(), input_type, true));
        prepared_indexed.push((
            *index,
            AggregateExpr::new(aggregate_expr.function, Expr::column(alias), aggregate_expr.alias.clone())
                .with_distinct(aggregate_expr.distinct),
        ));
    }
    if !masked_expressions.is_empty() {
        frame = frame
            .with_columns_with_budget(masked_expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.filtered_global_aggregate.filter_inputs",
                    "failed to materialize segmented filtered global aggregate inputs",
                    error,
                )
            })
            .map(|outcome| {
                budgeted_value(outcome, metrics, "nexus_query_engine.filtered_global_aggregate.filter_inputs")
            })?;
    }
    metrics.operator_time += timer.elapsed();
    Ok((frame, prepared_indexed, Arc::new(Schema::new(fields))))
}

fn filtered_global_aggregate_value_expr(aggregate_expr: &AggregateExpr) -> Expr {
    if aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && global_count_star_expr(&aggregate_expr.expr)
    {
        Expr::literal(LiteralValue::Int64(1))
    } else {
        aggregate_expr.expr.clone()
    }
}

fn unique_filtered_global_segment_alias(base: &str, fields: &[Field]) -> String {
    if fields.iter().all(|field| field.name() != base) {
        return base.to_string();
    }
    for suffix in 1usize.. {
        let alias = format!("{base}_{suffix}");
        if fields.iter().all(|field| field.name() != alias.as_str()) {
            return alias;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn indexed_partial_global_aggregate_spec(indexed_exprs: &[(usize, AggregateExpr)]) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        indexed_exprs
            .iter()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(*index);
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
                AggregateFunction::Count | AggregateFunction::NUnique => vec![expr.clone()],
                AggregateFunction::StddevSamp
                | AggregateFunction::StddevPop
                | AggregateFunction::VarSamp
                | AggregateFunction::VarPop => vec![expr.clone()],
            })
            .collect(),
    )
}

fn indexed_partial_global_aggregate_output_schema(
    indexed_exprs: &[(usize, AggregateExpr)],
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
) -> Result<Arc<Schema>> {
    let mut fields = Vec::new();
    for (index, expr) in indexed_exprs {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(*index);
                let input_type = filtered_global_expr_input_type(&expr.expr, input_schema)?;
                let output_field = aggregate_output_schema.field_with_name(expr.alias.as_str()).map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.filtered_global_aggregate.state_schema",
                        "avg aggregate output field is missing from output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("arrow_error", error.to_string())
                })?;
                fields.push(Field::new(
                    state.sum(),
                    avg_sum_state_data_type(&input_type, output_field.data_type())?,
                    true,
                ));
                fields.push(Field::new(state.count(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Count | AggregateFunction::NUnique => {
                fields.push(Field::new(expr.alias.as_str(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                fields.push(aggregate_output_schema.field_with_name(expr.alias.as_str()).cloned().map_err(
                    |error| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "nexus_query_engine.filtered_global_aggregate.state_schema",
                            "aggregate output field is missing from output schema",
                        )
                        .with_fact("alias", expr.alias.as_str())
                        .with_fact("arrow_error", error.to_string())
                    },
                )?);
                fields.push(Field::new(
                    global_valid_count_state_alias(expr.alias.as_str()),
                    ArrowDataType::Int64,
                    false,
                ));
            },
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop => {
                fields.push(aggregate_output_schema.field_with_name(expr.alias.as_str()).cloned().map_err(
                    |error| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "nexus_query_engine.filtered_global_aggregate.state_schema",
                            "statistical aggregate output field is missing from output schema",
                        )
                        .with_fact("alias", expr.alias.as_str())
                        .with_fact("arrow_error", error.to_string())
                    },
                )?);
            },
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn concat_partial_state_frames(
    mut partial_frames: Vec<GpuDataFrame>,
    token: &ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    if partial_frames.len() == 1 {
        return Ok(partial_frames.pop().expect("single partial frame checked"));
    }
    let first = partial_frames.remove(0).into_frame();
    let rest = partial_frames.iter().map(GpuDataFrame::frame).collect::<Vec<_>>();
    first.with_execution(&token.execution_overrides()).concat_many(&rest).map(GpuDataFrame::new).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "nexus_query_engine.filtered_global_aggregate.concat_partial_state",
                "failed to concatenate segmented filtered global aggregate partial states",
                error,
            ),
            token,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn record_segmented_filtered_global_aggregate_strategy(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    partial_rows: u64,
    stats: FilteredGlobalSegmentRuntimeStats,
    mut decision: FilteredGlobalAggregateStrategyDecision,
    selection_elapsed: Duration,
    elapsed: Duration,
) {
    add_segmented_runtime_facts(&mut decision.decision.facts, partial_rows, stats);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Single,
        context.memory_control_policy().as_str(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision.decision,
        selection_elapsed,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_segmented_filtered_global_partial_strategy(
    metrics: &mut QueryMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    partial_rows: u64,
    stats: FilteredGlobalSegmentRuntimeStats,
    input_evidence: Option<PartialAggregateInputEvidence>,
    mut decision: FilteredGlobalAggregateStrategyDecision,
    elapsed: Duration,
) {
    add_segmented_runtime_facts(&mut decision.decision.facts, partial_rows, stats);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Partial,
        context.memory_control_policy().as_str(),
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision.decision,
        Duration::ZERO,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        input_evidence,
    );
}

fn add_segmented_runtime_facts(
    facts: &mut BTreeMap<&'static str, String>,
    partial_rows: u64,
    stats: FilteredGlobalSegmentRuntimeStats,
) {
    facts.insert("filtered_global_state_rows_merged", partial_rows.to_string());
    facts.insert("filtered_global_actual_chunk_count", stats.chunk_count.to_string());
    facts.insert("filtered_global_actual_segment_count", stats.segment_count.to_string());
    facts.insert("filtered_global_max_segment_rows", stats.max_segment_rows.to_string());
    facts.insert("filtered_global_observed_temporary_peak_bytes", stats.observed_temporary_peak_bytes.to_string());
    facts.insert("filtered_global_observed_retained_peak_bytes", stats.observed_retained_peak_bytes.to_string());
    facts.insert(
        "filtered_input_lower_level_unresolved_contracts",
        if stats.unresolved_contracts.is_empty() {
            "none".to_string()
        } else {
            stats.unresolved_contracts.into_iter().collect::<Vec<_>>().join("|")
        },
    );
}

fn add_segmented_budgeted_execution_stats(
    stats: &mut FilteredGlobalSegmentRuntimeStats,
    metrics: &QueryMetrics,
    budget_event_start: usize,
) {
    for event in metrics.budgeted_execution_events.iter().skip(budget_event_start) {
        if let Some(bytes) = event.allocator_high_water_bytes {
            stats.observed_temporary_peak_bytes = stats.observed_temporary_peak_bytes.max(bytes);
        }
        if let Some(bytes) = event.retained_bytes {
            stats.observed_retained_peak_bytes = stats.observed_retained_peak_bytes.max(bytes);
        }
        if let Some(contract) = &event.unresolved_contract {
            stats.unresolved_contracts.insert(contract.clone());
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FilteredGlobalMaterializationFootprint {
    filtered_count: usize,
    unique_filter_predicate_count: usize,
    estimated_materialization_bytes: u64,
    original_frame_retained_bytes: u64,
    value_bytes: u64,
    validity_bytes: u64,
    predicate_bytes: u64,
    max_segment_input_bytes_per_row: u64,
}

fn unsupported_large_filtered_global_stateful_merge_error(
    footprint: &FilteredGlobalMaterializationFootprint,
    input_rows: u64,
    budget_bytes: u64,
    threshold: u64,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.aggregate.filtered_global_strategy",
        "large filtered global aggregate requires segmented state execution, but one or more aggregate states cannot be merged safely",
    )
    .with_fact("filtered_aggregate_scope", "global")
    .with_fact("selected_strategy", "none")
    .with_fact(
        "rejected_candidate_strategies",
        "filtered_global_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold;filtered_global_segmented_state:unsupported_stateful_filtered_merge",
    )
    .with_fact("filtered_input_materialization_strategy", "rejected")
    .with_fact("filtered_input_materialization_estimated_bytes", footprint.estimated_materialization_bytes)
    .with_fact("filtered_input_materialization_budget_bytes", budget_bytes)
    .with_fact("filtered_input_materialization_threshold_bytes", threshold)
    .with_fact(
        "filtered_input_materialization_budget_fraction",
        format!("1/{SINGLE_PASS_BUDGET_FRACTION_DENOMINATOR}"),
    )
    .with_fact("filtered_input_unsupported_stateful_merge", true)
    .with_fact("filtered_input_unique_predicate_count", footprint.unique_filter_predicate_count)
    .with_fact("filtered_global_strategy_input_rows", input_rows)
}

fn estimate_filtered_global_materialization_footprint(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
) -> Result<FilteredGlobalMaterializationFootprint> {
    let original_frame_retained_bytes = input_schema
        .fields()
        .iter()
        .map(|field| estimate_column_bytes(field.data_type(), field.is_nullable(), input_rows))
        .try_fold(0u64, |acc, bytes| bytes.map(|bytes| acc.saturating_add(bytes)))?;
    let mut filtered_count = 0usize;
    let mut value_bytes = 0u64;
    let mut validity_bytes = 0u64;
    let mut predicate_bytes = 0u64;
    let mut max_segment_input_bytes_per_row = 1u64;
    let mut unique_filter_predicates = BTreeSet::new();
    for aggregate_expr in spec.aggr_exprs.iter().filter(|expr| expr.filter.is_some()) {
        filtered_count += 1;
        if let Some(filter) = &aggregate_expr.filter {
            unique_filter_predicates.insert(filter.to_string());
        }
        let input_type = filtered_aggregate_input_type(aggregate_expr, input_schema)?;
        let element_width = fixed_width_dtype_size(&input_type).unwrap_or(STRING_OFFSET_BYTES_CONSERVATIVE);
        let column_value_bytes = input_rows.saturating_mul(element_width);
        let column_validity_bytes = null_mask_bytes(input_rows);
        let column_predicate_bytes = input_rows;
        value_bytes = value_bytes.saturating_add(column_value_bytes);
        validity_bytes = validity_bytes.saturating_add(column_validity_bytes);
        predicate_bytes = predicate_bytes.saturating_add(column_predicate_bytes);
        max_segment_input_bytes_per_row = max_segment_input_bytes_per_row.max(element_width.saturating_add(2));
    }
    let estimated_materialization_bytes = original_frame_retained_bytes
        .saturating_add(value_bytes)
        .saturating_add(validity_bytes)
        .saturating_add(predicate_bytes);
    Ok(FilteredGlobalMaterializationFootprint {
        filtered_count,
        unique_filter_predicate_count: unique_filter_predicates.len(),
        estimated_materialization_bytes,
        original_frame_retained_bytes,
        value_bytes,
        validity_bytes,
        predicate_bytes,
        max_segment_input_bytes_per_row,
    })
}

fn estimate_column_bytes(data_type: &ArrowDataType, nullable: bool, rows: u64) -> Result<u64> {
    let width = fixed_width_dtype_size(data_type).unwrap_or(STRING_OFFSET_BYTES_CONSERVATIVE);
    let validity = if nullable { null_mask_bytes(rows) } else { 0 };
    Ok(rows.saturating_mul(width).saturating_add(validity).saturating_add(PER_COLUMN_ALIGNMENT_PADDING_BYTES))
}

fn filtered_inputs_per_segment_for_budget(
    max_segment_input_bytes_per_row: u64,
    rows_per_chunk: u64,
    retained_predicate_bytes_per_row: u64,
    budget_bytes: u64,
    target_temporary_bytes: u64,
) -> usize {
    let target = cmp::max(1, cmp::min(target_temporary_bytes, budget_bytes / 8));
    let retained_predicate_bytes = retained_predicate_bytes_per_row.saturating_mul(rows_per_chunk);
    let available_for_masked_inputs = target.saturating_sub(retained_predicate_bytes).max(1);
    let per_input_bytes = max_segment_input_bytes_per_row.saturating_mul(rows_per_chunk);
    usize::try_from(cmp::max(1, available_for_masked_inputs / per_input_bytes.max(1))).unwrap_or(usize::MAX)
}

fn rows_per_chunk_for_filtered_global_segment(
    max_segment_input_bytes_per_row: u64,
    target_temporary_bytes: u64,
) -> u64 {
    cmp::max(MIN_SEGMENT_ROWS, target_temporary_bytes / max_segment_input_bytes_per_row.max(1))
}

fn filtered_global_segment_target_temporary_bytes(input_schema: &ArrowSchemaRef, budget_bytes: u64) -> u64 {
    let configured = input_schema
        .metadata()
        .get(TEST_SEGMENT_TARGET_TEMPORARY_BYTES_FACT)
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(SEGMENT_TARGET_TEMPORARY_BYTES);
    cmp::max(1, cmp::min(configured, cmp::max(1, budget_bytes / 8)))
}

fn fixed_width_dtype_size(data_type: &ArrowDataType) -> Option<u64> {
    match data_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some(2),
        ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_) => Some(4),
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_) => Some(8),
        ArrowDataType::Decimal128(_, _) => Some(16),
        ArrowDataType::Decimal256(_, _) => Some(32),
        _ => None,
    }
}

fn null_mask_bytes(rows: u64) -> u64 {
    if rows == 0 { 0 } else { rows.div_ceil(8).div_ceil(64) * 64 }
}

fn filtered_global_expr_input_type(expr: &Expr, input_schema: &ArrowSchemaRef) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => {
            input_schema.field_with_name(column.name()).map(|field| field.data_type().clone()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.filtered_global_aggregate.state_schema",
                    "aggregate state references a column missing from its input schema",
                )
                .with_fact("column", column.name())
                .with_fact("arrow_error", error.to_string())
            })
        },
        Expr::Alias { expr, .. } => filtered_global_expr_input_type(expr, input_schema),
        Expr::Literal(value) => Ok(value.data_type()),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.filtered_global_aggregate.state_schema",
            "filtered global aggregate state input must be a prepared column or literal",
        )
        .with_fact("expression", expr.to_string())),
    }
}
