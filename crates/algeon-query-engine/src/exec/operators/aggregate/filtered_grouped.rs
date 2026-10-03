use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{
    cudf_query_error,
    filter_inputs::{filtered_aggregate_input_type, filtered_aggregate_value_expr},
    filtered_common::{
        add_segmented_budgeted_execution_stats, estimate_column_bytes, filtered_fixed_width_dtype_size,
        filtered_inputs_per_segment_for_budget, null_mask_bytes, unsupported_stateful_filtered_merge,
    },
    filtered_segment_policy::{FilteredAggregateScope, FilteredAggregateSegmentationPolicy},
    *,
};
use crate::naming::unique_alias_with_suffix;

// Admission decision plus segmentation plan for filtered grouped aggregates.
#[derive(Debug, Clone)]
pub(super) struct FilteredGroupedAggregateStrategyDecision {
    pub(super) use_segmented: bool,
    pub(super) decision: AggregateStrategyCapabilityDecision,
    rows_per_chunk: u64,
    filtered_inputs_per_segment: usize,
}

#[derive(Debug, Clone, Default)]
struct FilteredGroupedSegmentRuntimeStats {
    chunk_count: u64,
    segment_count: u64,
    max_segment_rows: u64,
    state_rows_merged: u64,
    observed_temporary_peak_bytes: u64,
    observed_retained_peak_bytes: u64,
}

struct SegmentedFilteredGroupedPartialState {
    frame: GpuDataFrame,
    stats: FilteredGroupedSegmentRuntimeStats,
}

struct FilteredGroupedPreparedChunk {
    frame: DataFrame,
    schema: ArrowSchemaRef,
    predicate_aliases: BTreeMap<String, String>,
}

pub(super) fn select_filtered_grouped_aggregate_strategy(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    budget_bytes: u64,
    stage: AggregateExecutionStage,
) -> Result<FilteredGroupedAggregateStrategyDecision> {
    let policy = FilteredAggregateSegmentationPolicy::default();
    let footprint = estimate_filtered_grouped_materialization_footprint(spec, input_schema, input_rows)?;
    let unsupported_stateful_merge = unsupported_stateful_filtered_merge(spec);
    let threshold = policy.single_pass_threshold_bytes(budget_bytes);
    let single_pass_exceeds_threshold =
        footprint.filtered_count > 0 && footprint.estimated_materialization_bytes > threshold;
    if single_pass_exceeds_threshold && unsupported_stateful_merge {
        return Err(unsupported_large_filtered_grouped_stateful_merge_error(
            &footprint,
            input_rows,
            budget_bytes,
            threshold,
        ));
    }
    let use_segmented = footprint.filtered_count > 0 && single_pass_exceeds_threshold && !unsupported_stateful_merge;
    let target_temporary_bytes =
        policy.segment_target_temporary_bytes(FilteredAggregateScope::Grouped, input_schema, budget_bytes);
    let retained_predicate_bytes_per_row =
        policy.retained_filter_predicate_bytes_per_row(footprint.unique_filter_predicate_count);
    let rows_per_chunk = if use_segmented {
        policy.rows_per_segment(
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
        if use_segmented { "filtered_grouped_segmented_state" } else { "filtered_grouped_single_pass_materialized" };
    decision.facts.insert("selected_strategy", selected_strategy.to_string());
    decision.facts.insert("filtered_aggregate_scope", "grouped".to_string());
    decision.facts.insert(
        "filtered_input_materialization_strategy",
        if use_segmented { "row_chunked_grouped_segmented_state" } else { "single_pass_with_columns" }.to_string(),
    );
    decision.facts.insert(
        "filtered_input_materialization_estimated_bytes",
        footprint.estimated_materialization_bytes.to_string(),
    );
    decision.facts.insert("filtered_input_materialization_budget_bytes", budget_bytes.to_string());
    decision.facts.insert("filtered_input_materialization_threshold_bytes", threshold.to_string());
    decision.facts.insert("filtered_input_materialization_budget_fraction", policy.single_pass_budget_fraction_label());
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
        .insert("filtered_grouped_retained_predicate_bytes_per_row", retained_predicate_bytes_per_row.to_string());
    decision.facts.insert("filtered_grouped_strategy_input_rows", input_rows.to_string());
    decision.facts.insert("filtered_grouped_rows_per_chunk", rows_per_chunk.to_string());
    decision.facts.insert("filtered_grouped_inputs_per_segment", filtered_inputs_per_segment.to_string());
    decision.facts.insert("filtered_grouped_segment_target_temporary_bytes", target_temporary_bytes.to_string());
    decision.facts.insert("filtered_grouped_group_key_count", spec.group_exprs.len().to_string());
    decision
        .facts
        .insert("filtered_grouped_group_key_retained_bytes_estimate", footprint.group_key_retained_bytes.to_string());
    decision.facts.insert("filtered_grouped_high_cardinality_admission", "chunk_local_partial_final_merge".to_string());
    decision.facts.insert(
        "rejected_candidate_strategies",
        if use_segmented {
            "filtered_grouped_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold"
        } else if unsupported_stateful_merge {
            "filtered_grouped_segmented_state:unsupported_stateful_filtered_merge"
        } else {
            "filtered_grouped_segmented_state:single_pass_estimate_within_threshold"
        }
        .to_string(),
    );
    Ok(FilteredGroupedAggregateStrategyDecision {
        use_segmented,
        decision,
        rows_per_chunk,
        filtered_inputs_per_segment,
    })
}

pub(super) fn apply_filtered_grouped_materialization_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    filtered: &FilteredGroupedAggregateStrategyDecision,
) {
    for key in [
        "aggregate_filter_count",
        "filtered_aggregate_scope",
        "filtered_input_materialization_strategy",
        "filtered_input_materialization_estimated_bytes",
        "filtered_input_materialization_budget_bytes",
        "filtered_input_materialization_threshold_bytes",
        "filtered_input_materialization_budget_fraction",
        "filtered_input_value_bytes",
        "filtered_input_validity_bytes",
        "filtered_input_predicate_bytes",
        "filtered_input_original_frame_retained_bytes",
        "filtered_input_unsupported_stateful_merge",
        "filtered_input_unique_predicate_count",
        "filtered_grouped_retained_predicate_bytes_per_row",
        "filtered_grouped_strategy_input_rows",
        "filtered_grouped_rows_per_chunk",
        "filtered_grouped_inputs_per_segment",
        "filtered_grouped_segment_target_temporary_bytes",
        "filtered_grouped_group_key_count",
        "filtered_grouped_group_key_retained_bytes_estimate",
        "filtered_grouped_high_cardinality_admission",
        "rejected_candidate_strategies",
    ] {
        if let Some(value) = filtered.decision.facts.get(key) {
            decision.facts.insert(key, value.clone());
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
pub(super) fn record_filtered_grouped_segmented_failure(
    metrics: &ExecutionMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    stage: AggregateExecutionStage,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    decision: FilteredGroupedAggregateStrategyDecision,
) {
    let stats = FilteredGroupedSegmentRuntimeStats::default();
    record_segmented_filtered_grouped_strategy(
        metrics,
        context,
        node_id,
        stage,
        spec,
        input_schema,
        input_rows,
        0,
        0,
        stats,
        input_evidence,
        decision,
        Duration::ZERO,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
pub(super) fn record_filtered_grouped_single_pass(
    metrics: &ExecutionMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    stage: AggregateExecutionStage,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    decision: FilteredGroupedAggregateStrategyDecision,
) {
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        stage,
        context.aggregate_strategy_execution_options(),
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

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(super) fn execute_segmented_filtered_grouped_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    decision: FilteredGroupedAggregateStrategyDecision,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let budget_window = metrics.open_budget_window();
    let timer = Instant::now();
    let partial =
        execute_segmented_filtered_grouped_partial_state(spec, input_schema, input, token, &decision, metrics)?;
    let partial_state = partial.frame;
    let mut stats = partial.stats;
    let partial_rows = partial_state.num_rows() as u64;
    let final_aggregate = lower_final_aggregate_spec(spec)?;
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let overrides = token.exec();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        overrides,
        partial_state.into_frame(),
        &final_aggregate,
        token,
        metrics,
        "query_engine.filtered_grouped_aggregate.final_merge",
        "failed to execute segmented filtered grouped final aggregate merge",
    )?;
    output = finalize_avg_output_columns_with_budget(context, output, spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.filtered_grouped_aggregate.select",
            "failed to select segmented filtered grouped aggregate output columns",
            error,
        )
    })?;
    let output = GpuDataFrame::new(output);
    add_segmented_budgeted_execution_stats(
        &mut stats.observed_temporary_peak_bytes,
        &mut stats.observed_retained_peak_bytes,
        metrics.take_budget_window(budget_window),
    );
    record_segmented_filtered_grouped_strategy(
        metrics,
        context,
        node_id,
        AggregateExecutionStage::Single,
        spec,
        input_schema,
        input_rows,
        output.num_rows() as u64,
        partial_rows,
        stats,
        None,
        decision,
        timer.elapsed(),
    );
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(super) fn execute_segmented_filtered_grouped_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    decision: FilteredGroupedAggregateStrategyDecision,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let budget_window = metrics.open_budget_window();
    let timer = Instant::now();
    let partial =
        execute_segmented_filtered_grouped_partial_state(spec, input_schema, input, token, &decision, metrics)?;
    let mut stats = partial.stats;
    let output = cast_count_output_columns_with_budget(
        partial.frame.into_frame(),
        spec,
        aggregate_output_schema,
        token,
        metrics,
    )
    .map(GpuDataFrame::new)?;
    let output_rows = output.num_rows() as u64;
    add_segmented_budgeted_execution_stats(
        &mut stats.observed_temporary_peak_bytes,
        &mut stats.observed_retained_peak_bytes,
        metrics.take_budget_window(budget_window),
    );
    record_segmented_filtered_grouped_strategy(
        metrics,
        context,
        node_id,
        AggregateExecutionStage::Partial,
        spec,
        input_schema,
        input_rows,
        output_rows,
        output_rows,
        stats,
        input_evidence,
        decision,
        timer.elapsed(),
    );
    Ok(output)
}

fn execute_segmented_filtered_grouped_partial_state(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    decision: &FilteredGroupedAggregateStrategyDecision,
    metrics: &ExecutionMetrics,
) -> Result<SegmentedFilteredGroupedPartialState> {
    let input_rows = input.num_rows();
    let frame = input.into_frame();
    let rows_per_chunk = usize::try_from(decision.rows_per_chunk)
        .unwrap_or(usize::MAX)
        .max(FilteredAggregateSegmentationPolicy::default().min_segment_rows_as_usize());
    let mut partial_frames = Vec::new();
    let mut stats = FilteredGroupedSegmentRuntimeStats::default();
    let mut offset = 0usize;
    loop {
        let length = if input_rows == 0 { 0 } else { rows_per_chunk.min(input_rows.saturating_sub(offset)) };
        let chunk = if length == input_rows {
            frame.clone()
        } else {
            frame.on(token.exec()).slice(offset, length).map_err(|error| {
                attach_budget_token_failure_facts(
                    cudf_query_error(
                        "query_engine.filtered_grouped_aggregate.slice",
                        "failed to slice input for segmented filtered grouped aggregate",
                        error,
                    ),
                    token,
                )
                .with_fact("chunk_offset", offset)
                .with_fact("chunk_rows", length)
            })?
        };
        let chunk_output = execute_filtered_grouped_partial_state_for_chunk(
            spec,
            input_schema,
            chunk,
            token,
            decision.filtered_inputs_per_segment,
            metrics,
        )?;
        stats.chunk_count = stats.chunk_count.saturating_add(1);
        stats.segment_count = stats.segment_count.saturating_add(chunk_output.stats.segment_count);
        stats.max_segment_rows = stats.max_segment_rows.max(length as u64);
        partial_frames.push(chunk_output.frame);
        if input_rows == 0 || offset + length >= input_rows {
            break;
        }
        offset += length;
    }
    let frame = concat_partial_state_frames(partial_frames, token)?;
    stats.state_rows_merged = frame.num_rows() as u64;
    Ok(SegmentedFilteredGroupedPartialState { frame, stats })
}

fn execute_filtered_grouped_partial_state_for_chunk(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    chunk: DataFrame,
    token: &ExecutionBudgetToken,
    filtered_inputs_per_segment: usize,
    metrics: &ExecutionMetrics,
) -> Result<SegmentedFilteredGroupedPartialState> {
    let prepared_chunk = prepare_filtered_grouped_predicates_for_chunk(&chunk, input_schema, spec, token, metrics)?;
    let mut segment = Vec::<(usize, AggregateExpr)>::new();
    let mut segment_filtered_count = 0usize;
    let mut segment_frames = Vec::new();
    let mut stats = FilteredGroupedSegmentRuntimeStats::default();
    for (index, expr) in spec.aggr_exprs.iter().cloned().enumerate() {
        let is_filtered = expr.filter.is_some();
        if is_filtered && segment_filtered_count >= filtered_inputs_per_segment && !segment.is_empty() {
            segment_frames.push(execute_filtered_grouped_partial_segment(
                input_schema,
                &prepared_chunk,
                spec,
                &segment,
                token,
                metrics,
            )?);
            stats.segment_count = stats.segment_count.saturating_add(1);
            segment.clear();
            segment_filtered_count = 0;
        }
        segment_filtered_count += usize::from(is_filtered);
        segment.push((index, expr));
    }
    if !segment.is_empty() {
        segment_frames.push(execute_filtered_grouped_partial_segment(
            input_schema,
            &prepared_chunk,
            spec,
            &segment,
            token,
            metrics,
        )?);
        stats.segment_count = stats.segment_count.saturating_add(1);
    }
    let frame = merge_filtered_grouped_segment_frames(spec, segment_frames, token)?;
    Ok(SegmentedFilteredGroupedPartialState { frame, stats })
}

fn execute_filtered_grouped_partial_segment(
    input_schema: &ArrowSchemaRef,
    prepared_chunk: &FilteredGroupedPreparedChunk,
    spec: &AggregateSpec,
    indexed_exprs: &[(usize, AggregateExpr)],
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let (prepared, prepared_indexed) =
        prepare_filtered_grouped_partial_segment(prepared_chunk, input_schema, indexed_exprs, token, metrics)?;
    let aggregate = indexed_partial_grouped_aggregate(spec, &prepared_indexed)?;
    let output = execute_groupby_aggregate_with_budget(
        token.exec(),
        prepared,
        &aggregate,
        token,
        metrics,
        "query_engine.filtered_grouped_aggregate.partial_segment",
        "failed to execute segmented filtered grouped partial aggregate segment",
    )?;
    let output =
        output.select(aggregate.output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            cudf_query_error(
                "query_engine.filtered_grouped_aggregate.partial_segment_select",
                "failed to select segmented filtered grouped partial aggregate segment columns",
                error,
            )
        })?;
    Ok(GpuDataFrame::new(output))
}

fn merge_filtered_grouped_segment_frames(
    spec: &AggregateSpec,
    mut segment_frames: Vec<GpuDataFrame>,
    token: &ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    let mut merged = segment_frames.remove(0).into_frame();
    let key_names = grouped_key_output_names(spec)?;
    for frame in segment_frames {
        merged = merged
            .on(token.exec())
            .merge(
                frame.frame(),
                cudf::MergeOptions::new()
                    .on(key_names.iter().map(String::as_str).collect::<Vec<_>>())
                    .how_inner()
                    .null_equality(cudf::NullEquality::Equal),
            )
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    cudf_query_error(
                        "query_engine.filtered_grouped_aggregate.merge_segments",
                        "failed to merge segmented filtered grouped aggregate partial states",
                        error,
                    ),
                    token,
                )
            })?;
    }
    let output_names = indexed_partial_grouped_output_names(spec, spec.aggr_exprs.iter().cloned().enumerate())?;
    let merged = merged.select(output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.filtered_grouped_aggregate.merge_select",
            "failed to select segmented filtered grouped merged partial state columns",
            error,
        )
    })?;
    Ok(GpuDataFrame::new(merged))
}

fn prepare_filtered_grouped_predicates_for_chunk(
    chunk: &DataFrame,
    input_schema: &ArrowSchemaRef,
    spec: &AggregateSpec,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<FilteredGroupedPreparedChunk> {
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
        let alias = unique_filtered_grouped_segment_alias("__algeon_native_filtered_grouped_predicate", &fields);
        predicate_aliases.insert(predicate_key, alias.clone());
        predicate_expressions.push((alias.clone(), lower_expr(filter)?));
        fields.push(Field::new(alias, ArrowDataType::Boolean, true));
    }
    if !predicate_expressions.is_empty() {
        frame = token
            .run("dataframe::with_columns", |exec| frame.on(exec).with_columns(predicate_expressions))
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.filtered_grouped_aggregate.filter_predicates",
                    "failed to materialize segmented filtered grouped aggregate predicates",
                    error,
                )
            })
            .map(|outcome| {
                budgeted_value(outcome, metrics, "query_engine.filtered_grouped_aggregate.filter_predicates")
            })?;
    }
    metrics.add_operator_time(timer.elapsed());
    Ok(FilteredGroupedPreparedChunk { frame, schema: Arc::new(Schema::new(fields)), predicate_aliases })
}

fn prepare_filtered_grouped_partial_segment(
    prepared_chunk: &FilteredGroupedPreparedChunk,
    input_schema: &ArrowSchemaRef,
    indexed_exprs: &[(usize, AggregateExpr)],
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<(DataFrame, Vec<(usize, AggregateExpr)>)> {
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
        let value_expr = filtered_aggregate_value_expr(aggregate_expr);
        let alias = unique_filtered_grouped_segment_alias("__algeon_native_filtered_grouped_arg", &fields);
        let predicate_alias = prepared_chunk.predicate_aliases.get(filter.to_string().as_str()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.filtered_grouped_aggregate.filter_predicates",
                "segmented filtered grouped aggregate predicate alias is missing",
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
        frame = token
            .run("dataframe::with_columns", |exec| frame.on(exec).with_columns(masked_expressions))
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.filtered_grouped_aggregate.filter_inputs",
                    "failed to materialize segmented filtered grouped aggregate inputs",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "query_engine.filtered_grouped_aggregate.filter_inputs"))?;
    }
    metrics.add_operator_time(timer.elapsed());
    Ok((frame, prepared_indexed))
}

fn indexed_partial_grouped_aggregate(
    spec: &AggregateSpec,
    indexed_exprs: &[(usize, AggregateExpr)],
) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(indexed_exprs.len().saturating_mul(2));
    let mut output_names = aggregate.output_key_names.clone();
    for (index, expr) in indexed_exprs {
        let input_column = lower_column_expr(&expr.expr, "query_engine.filtered_grouped_aggregate.lower_partial_expr")?;
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(*index);
                aggregations.push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(state.count()),
                );
                output_names.extend(state.output_names());
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
    Ok(aggregate)
}

fn grouped_key_output_names(spec: &AggregateSpec) -> Result<Vec<String>> {
    lower_aggregate_spec_keys(spec).map(|aggregate| aggregate.output_key_names)
}

fn indexed_partial_grouped_output_names(
    spec: &AggregateSpec,
    indexed_exprs: impl IntoIterator<Item = (usize, AggregateExpr)>,
) -> Result<Vec<String>> {
    let mut output_names = grouped_key_output_names(spec)?;
    for (index, expr) in indexed_exprs {
        if expr.function == AggregateFunction::Avg {
            output_names.extend(AvgStateAliases::for_aggregate_index(index).output_names());
        } else {
            output_names.push(expr.alias);
        }
    }
    Ok(output_names)
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
    first.on(token.exec()).concat_many(&rest).map(GpuDataFrame::new).map_err(|error| {
        attach_budget_token_failure_facts(
            cudf_query_error(
                "query_engine.filtered_grouped_aggregate.concat_partial_state",
                "failed to concatenate segmented filtered grouped aggregate partial states",
                error,
            ),
            token,
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_segmented_filtered_grouped_strategy(
    metrics: &ExecutionMetrics,
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    stage: AggregateExecutionStage,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    partial_rows: u64,
    stats: FilteredGroupedSegmentRuntimeStats,
    input_evidence: Option<PartialAggregateInputEvidence>,
    mut decision: FilteredGroupedAggregateStrategyDecision,
    elapsed: Duration,
) {
    add_segmented_runtime_facts(&mut decision.decision.facts, partial_rows, stats);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        stage,
        context.aggregate_strategy_execution_options(),
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
    stats: FilteredGroupedSegmentRuntimeStats,
) {
    facts.insert("filtered_grouped_state_rows_merged", partial_rows.to_string());
    facts.insert("filtered_grouped_actual_chunk_count", stats.chunk_count.to_string());
    facts.insert("filtered_grouped_actual_segment_count", stats.segment_count.to_string());
    facts.insert("filtered_grouped_max_segment_rows", stats.max_segment_rows.to_string());
    facts.insert("filtered_grouped_observed_temporary_peak_bytes", stats.observed_temporary_peak_bytes.to_string());
    facts.insert("filtered_grouped_observed_retained_peak_bytes", stats.observed_retained_peak_bytes.to_string());
}

#[derive(Debug, Clone, Copy)]
struct FilteredGroupedMaterializationFootprint {
    filtered_count: usize,
    unique_filter_predicate_count: usize,
    estimated_materialization_bytes: u64,
    original_frame_retained_bytes: u64,
    group_key_retained_bytes: u64,
    value_bytes: u64,
    validity_bytes: u64,
    predicate_bytes: u64,
    max_segment_input_bytes_per_row: u64,
}

fn unsupported_large_filtered_grouped_stateful_merge_error(
    footprint: &FilteredGroupedMaterializationFootprint,
    input_rows: u64,
    budget_bytes: u64,
    threshold: u64,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "query_engine.aggregate.filtered_grouped_strategy",
        "large filtered grouped aggregate requires segmented state execution, but one or more aggregate states cannot be merged safely",
    )
    .with_fact("filtered_aggregate_scope", "grouped")
    .with_fact("selected_strategy", "none")
    .with_fact(
        "rejected_candidate_strategies",
        "filtered_grouped_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold;filtered_grouped_segmented_state:unsupported_stateful_filtered_merge",
    )
    .with_fact("filtered_input_materialization_strategy", "rejected")
    .with_fact("filtered_input_materialization_estimated_bytes", footprint.estimated_materialization_bytes)
    .with_fact("filtered_input_materialization_budget_bytes", budget_bytes)
    .with_fact("filtered_input_materialization_threshold_bytes", threshold)
    .with_fact(
        "filtered_input_materialization_budget_fraction",
        FilteredAggregateSegmentationPolicy::default().single_pass_budget_fraction_label(),
    )
    .with_fact("filtered_input_unsupported_stateful_merge", true)
    .with_fact("filtered_input_unique_predicate_count", footprint.unique_filter_predicate_count)
    .with_fact("filtered_grouped_strategy_input_rows", input_rows)
}

fn estimate_filtered_grouped_materialization_footprint(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
) -> Result<FilteredGroupedMaterializationFootprint> {
    let original_frame_retained_bytes = input_schema
        .fields()
        .iter()
        .map(|field| estimate_column_bytes(field.data_type(), field.is_nullable(), input_rows))
        .try_fold(0u64, |acc, bytes| bytes.map(|bytes| acc.saturating_add(bytes)))?;
    let group_key_retained_bytes = spec
        .group_exprs
        .iter()
        .map(|expr| {
            let column = lower_column_expr(expr, "query_engine.filtered_grouped_aggregate.footprint_group_key")?;
            let field = input_schema.field_with_name(column.as_str()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.filtered_grouped_aggregate.footprint_group_key",
                    "filtered grouped aggregate key references a missing column",
                )
                .with_fact("column", column.as_str())
                .with_fact("arrow_error", error.to_string())
            })?;
            estimate_column_bytes(field.data_type(), field.is_nullable(), input_rows)
        })
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
        let element_width = filtered_fixed_width_dtype_size(&input_type).unwrap_or(STRING_OFFSET_BYTES_CONSERVATIVE);
        value_bytes = value_bytes.saturating_add(input_rows.saturating_mul(element_width));
        validity_bytes = validity_bytes.saturating_add(null_mask_bytes(input_rows));
        predicate_bytes = predicate_bytes.saturating_add(input_rows);
        max_segment_input_bytes_per_row = max_segment_input_bytes_per_row.max(element_width.saturating_add(2));
    }
    let estimated_materialization_bytes = original_frame_retained_bytes
        .saturating_add(group_key_retained_bytes)
        .saturating_add(value_bytes)
        .saturating_add(validity_bytes)
        .saturating_add(predicate_bytes);
    Ok(FilteredGroupedMaterializationFootprint {
        filtered_count,
        unique_filter_predicate_count: unique_filter_predicates.len(),
        estimated_materialization_bytes,
        original_frame_retained_bytes,
        group_key_retained_bytes,
        value_bytes,
        validity_bytes,
        predicate_bytes,
        max_segment_input_bytes_per_row,
    })
}

fn unique_filtered_grouped_segment_alias(base: &str, fields: &[Field]) -> String {
    unique_alias_with_suffix(base, |alias| fields.iter().any(|field| field.name() == alias))
}
