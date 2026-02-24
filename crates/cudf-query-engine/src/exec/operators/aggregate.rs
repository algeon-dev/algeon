use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray, UInt64Array,
};

use super::*;
use crate::capability::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateStrategyCandidateRejection,
    AggregateStrategyCapabilityDecision, aggregate_strategy_capability_decision,
};

const ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM: u64 = 100_000;
const SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN: u64 = 64;

/// Source/chunk evidence available at a partial aggregate strategy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct PartialAggregateInputEvidence {
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

pub(in crate::exec) fn validate_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_stddev_samp(&node.spec) {
        if aggregate_spec_has_distinct(&node.spec) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "cudf_query_engine.aggregate.validate_stddev_exact",
                "DataFusion-exact stddev aggregate path does not support distinct aggregate expressions",
            ));
        }
        let expected_output_names = node
            .spec
            .group_exprs
            .iter()
            .map(host_group_expr_output_name)
            .chain(node.spec.aggr_exprs.iter().map(|expr| expr.alias.clone()))
            .collect::<Vec<_>>();
        let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
        if expected_output_names != schema_output_names {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.validate_stddev_exact",
                "DataFusion-exact stddev aggregate output names must match output schema fields",
            )
            .with_fact("aggregate_output_names", expected_output_names.join(","))
            .with_fact("schema_output_names", schema_output_names.join(",")));
        }
        return Ok(());
    }
    let validation_spec = aggregate_validation_spec(&node.spec);
    let aggregate = lower_aggregate_spec(&validation_spec)?;
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

fn aggregate_validation_spec(spec: &AggregateSpec) -> AggregateSpec {
    let mut validation = spec.clone();
    for (index, expr) in validation.aggr_exprs.iter_mut().enumerate() {
        if expr.filter.is_some()
            && expr.function == AggregateFunction::Count
            && !expr.distinct
            && global_count_star_expr(&expr.expr)
        {
            expr.expr = Expr::column(format!("__cudf_filtered_count_star_validation_{index}"));
        }
    }
    validation
}

pub(in crate::exec) fn validate_final_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_stddev_samp(&node.spec) {
        if aggregate_spec_has_distinct(&node.spec) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "cudf_query_engine.final_merge.validate_stddev_exact",
                "DataFusion-exact stddev aggregate path does not support distinct aggregate expressions",
            ));
        }
        return validate_aggregate_capability(node, output_schema);
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
        if expr.function == AggregateFunction::StddevSamp {
            if expr.distinct {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "cudf_query_engine.global_aggregate.validate_stddev_exact",
                    "DataFusion-exact stddev aggregate path does not support distinct aggregate expressions",
                )
                .with_fact("alias", expr.alias.as_str()));
            }
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.global_aggregate.validate_stddev_exact",
                    "stddev_samp aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            if !matches!(field.data_type(), ArrowDataType::Float64) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.global_aggregate.validate_stddev_exact",
                    "stddev_samp aggregate output must be Float64",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type())));
            }
        }
        validate_global_aggregate_input_expr(expr)?;
    }
    Ok(())
}

pub(in crate::exec) fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if !expr.distinct && global_count_star_expr(&expr.expr) => Ok(()),
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
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (frame, spec, prepared_input_schema) =
        prepare_filtered_aggregate_frame(context, input.into_frame(), input_schema, &node.spec, None, metrics)?;
    if aggregate_spec_has_stddev_samp(&spec) {
        return execute_datafusion_exact_host_aggregate(
            context,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
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
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "cudf_query_engine.aggregate.execute")?;
    output = finalize_avg_output_columns(context, output, &spec, output_schema)?;
    output = cast_reduction_output_columns(output, &spec, output_schema)?;
    output = cast_count_output_columns(output, &spec, output_schema)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_default_grouped_aggregate_strategy_decision(
        metrics,
        None,
        AggregateExecutionStage::Single,
        context.memory_control_policy().as_str(),
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_aggregate_with_budget_for_node(context, None, node, input_schema, input, output_schema, token, metrics)
}

pub(in crate::exec) fn execute_aggregate_with_budget_for_node(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (frame, spec, prepared_input_schema) =
        prepare_filtered_aggregate_frame(context, input.into_frame(), input_schema, &node.spec, Some(token), metrics)?;
    if aggregate_spec_has_stddev_samp(&spec) {
        return execute_datafusion_exact_host_aggregate(
            context,
            &spec,
            &prepared_input_schema,
            GpuDataFrame::new(frame),
            output_schema,
            metrics,
        );
    }
    if spec.group_exprs.is_empty() {
        return execute_global_aggregate_with_budget(
            context,
            &spec,
            GpuDataFrame::new(frame),
            output_schema,
            token,
            metrics,
        );
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
    let aggregate = lower_partial_aggregate_spec(&spec)?;
    if aggregate_spec_has_avg(&spec) {
        validate_avg_output_fields(&spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let overrides = execution_overrides(context);
    let input_rows = frame.num_rows() as u64;
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
    output = finalize_avg_output_columns_with_budget(context, output, &spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_default_grouped_aggregate_strategy_decision(
        metrics,
        node_id.map(|id| id.to_string()),
        AggregateExecutionStage::Single,
        context.memory_control_policy().as_str(),
        &spec,
        &prepared_input_schema,
        input_rows,
        output_rows,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (frame, spec, prepared_input_schema) =
        prepare_filtered_aggregate_frame(context, input.into_frame(), input_schema, &node.spec, Some(token), metrics)?;
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

    let aggregate = lower_partial_aggregate_spec(&spec)?;
    let expected_output_names = aggregate.output_names.clone();
    let input_rows = frame.num_rows() as u64;
    let selection_timer = Instant::now();
    let capability_decision =
        aggregate_strategy_capability_decision(&spec, prepared_input_schema.as_ref(), AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(capability_decision, node_id, input_rows, input_evidence, metrics);
    let selection_elapsed = selection_timer.elapsed();
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
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute_partial",
        "failed to execute bounded native partial cudf aggregate",
    )?;
    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_partial",
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

fn select_partial_aggregate_strategy_decision(
    mut decision: AggregateStrategyCapabilityDecision,
    node_id: Option<NodeId>,
    input_rows: u64,
    input_evidence: Option<PartialAggregateInputEvidence>,
    metrics: &QueryMetrics,
) -> AggregateStrategyCapabilityDecision {
    add_prepared_sort_groupby_gate_facts(&mut decision, input_rows);
    decision.facts.insert("encoded_key_partial_cost_model", "prior_partial_reduction_ratio".to_string());
    decision.facts.insert(
        "encoded_key_partial_reduction_ratio_threshold_ppm",
        ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM.to_string(),
    );
    decision.facts.insert("encoded_key_partial_current_input_rows", input_rows.to_string());
    add_partial_aggregate_input_evidence_facts(&mut decision.facts, input_evidence);
    if decision.selected_strategy != AggregateExecutionStrategy::EncodedKeyPartial {
        decision.facts.insert("encoded_key_partial_cost_model_status", "not_applicable".to_string());
        decision.facts.insert("encoded_key_partial_selection_reason", "capability_rejected".to_string());
        return decision;
    }
    if let Some(evidence) = input_evidence
        && evidence.source_adjacent_partial
        && prepared_sort_groupby_contract_available(&decision)
    {
        if source_adjacent_prepared_sort_groupby_allowed(evidence) {
            return select_prepared_sort_groupby_for_source_adjacent(decision);
        }
        decision = reject_prepared_sort_groupby_for_cost_model(
            decision,
            "prepared_sort_groupby_requires_group_key_source_partition",
            "source_adjacent_row_balanced_partition",
        );
    }

    let Some(evidence) = prior_partial_cardinality_evidence(metrics, node_id) else {
        let decision = reject_encoded_key_partial_for_cost_model(
            decision,
            "encoded_key_partial_cost_model_no_prior_cardinality_evidence",
            "no_prior_cardinality_evidence",
            None,
        );
        return reject_prepared_sort_groupby_for_cost_model(
            decision,
            "prepared_sort_groupby_cost_model_no_prior_cardinality_evidence",
            "no_prior_cardinality_evidence",
        );
    };
    add_prior_cardinality_facts(&mut decision, &evidence);
    if evidence.reduction_ratio_ppm <= ENCODED_KEY_PARTIAL_LOW_CARDINALITY_THRESHOLD_PPM {
        decision = reject_prepared_sort_groupby_for_cost_model(
            decision,
            "prepared_sort_groupby_requires_source_adjacent_partial",
            "not_source_adjacent_partial",
        );
        decision.facts.insert("encoded_key_partial_cost_model_status", "selected_low_cardinality".to_string());
        decision
            .facts
            .insert("encoded_key_partial_selection_reason", "prior_reduction_ratio_at_or_below_threshold".to_string());
        return decision;
    }

    let decision = reject_encoded_key_partial_for_cost_model(
        decision,
        "encoded_key_partial_cost_model_high_cardinality",
        "prior_reduction_ratio_above_threshold",
        Some(evidence),
    );
    reject_prepared_sort_groupby_for_cost_model(
        decision,
        "prepared_sort_groupby_cost_model_high_cardinality",
        "prior_reduction_ratio_above_threshold",
    )
}

fn add_partial_aggregate_input_evidence_facts(
    facts: &mut BTreeMap<&'static str, String>,
    input_evidence: Option<PartialAggregateInputEvidence>,
) {
    let Some(input_evidence) = input_evidence else {
        facts.insert("source_adjacent_partial", "unknown".to_string());
        facts.insert("source_chunk_row_group_count", "unknown".to_string());
        facts.insert("source_chunk_reads_all_row_groups", "unknown".to_string());
        facts.insert("source_chunk_known_row_count", "unknown".to_string());
        facts.insert("source_chunk_estimated_source_read_bytes", "unknown".to_string());
        facts.insert("source_chunk_size_estimate_based", "unknown".to_string());
        facts.insert("source_adjacent_partition_strategy", "unknown".to_string());
        facts.insert("source_adjacent_partition_count", "unknown".to_string());
        facts.insert("source_adjacent_non_empty_partition_count", "unknown".to_string());
        facts.insert("source_adjacent_partition_key_count", "unknown".to_string());
        facts.insert("source_adjacent_max_partition_fan_in", "unknown".to_string());
        return;
    };
    facts.insert("source_adjacent_partial", input_evidence.source_adjacent_partial.to_string());
    facts.insert("source_chunk_row_group_count", input_evidence.source_chunk_row_group_count.to_string());
    facts.insert("source_chunk_reads_all_row_groups", input_evidence.source_chunk_reads_all_row_groups.to_string());
    facts.insert(
        "source_chunk_known_row_count",
        optional_strategy_fact_u64(input_evidence.source_chunk_known_row_count),
    );
    facts.insert(
        "source_chunk_estimated_source_read_bytes",
        optional_strategy_fact_u64(input_evidence.source_chunk_estimated_source_read_bytes),
    );
    facts.insert("source_chunk_size_estimate_based", input_evidence.source_chunk_size_estimate_based.to_string());
    facts.insert("source_adjacent_partition_strategy", input_evidence.source_adjacent_partition_strategy.to_string());
    facts.insert(
        "source_adjacent_partition_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_count),
    );
    facts.insert(
        "source_adjacent_non_empty_partition_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_non_empty_partition_count),
    );
    facts.insert(
        "source_adjacent_partition_key_count",
        optional_strategy_fact_u64(input_evidence.source_adjacent_partition_key_count),
    );
    facts.insert(
        "source_adjacent_max_partition_fan_in",
        optional_strategy_fact_u64(input_evidence.source_adjacent_max_partition_fan_in),
    );
}

fn optional_strategy_fact_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

#[derive(Debug, Clone)]
struct PriorAggregateCardinalityEvidence {
    input_rows: u64,
    output_group_count: u64,
    reduction_ratio_ppm: u64,
    selected_strategy: String,
}

fn prior_partial_cardinality_evidence(
    metrics: &QueryMetrics,
    node_id: Option<NodeId>,
) -> Option<PriorAggregateCardinalityEvidence> {
    let node_id = node_id?.to_string();
    metrics.aggregate_strategy_decisions.iter().rev().find_map(|decision| {
        if decision.node_id != node_id || decision.facts.get("execution_stage").map(String::as_str) != Some("partial") {
            return None;
        }
        let input_rows = parse_strategy_fact_u64(decision.facts.get("input_rows")?)?;
        if input_rows == 0 {
            return None;
        }
        Some(PriorAggregateCardinalityEvidence {
            input_rows,
            output_group_count: parse_strategy_fact_u64(decision.facts.get("output_group_count")?)?,
            reduction_ratio_ppm: parse_strategy_fact_u64(decision.facts.get("reduction_ratio_ppm")?)?,
            selected_strategy: decision.facts.get("selected_strategy")?.clone(),
        })
    })
}

fn parse_strategy_fact_u64(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn add_prepared_sort_groupby_gate_facts(decision: &mut AggregateStrategyCapabilityDecision, input_rows: u64) {
    decision.facts.insert("prepared_sort_groupby_cost_model", "exact_grouped_key_facts_required".to_string());
    decision.facts.insert("prepared_sort_groupby_current_input_rows", input_rows.to_string());
    decision.facts.insert("prepared_sort_groupby_num_groups", "unknown".to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", "unknown".to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", "unknown".to_string());
    if prepared_sort_groupby_contract_available(decision) {
        decision.facts.insert("prepared_sort_groupby_cost_model_status", "pending_runtime_evidence".to_string());
        decision
            .facts
            .insert("prepared_sort_groupby_selection_reason", "runtime_cardinality_evidence_required".to_string());
    } else {
        decision.facts.insert("prepared_sort_groupby_cost_model_status", "not_applicable".to_string());
        decision.facts.insert("prepared_sort_groupby_selection_reason", "capability_rejected".to_string());
    }
}

fn prepared_sort_groupby_contract_available(decision: &AggregateStrategyCapabilityDecision) -> bool {
    decision.facts.get("prepared_sort_groupby_contract_status").map(String::as_str) == Some("available")
}

fn source_adjacent_prepared_sort_groupby_allowed(evidence: PartialAggregateInputEvidence) -> bool {
    evidence.source_adjacent_partition_strategy == "group_key"
        && evidence
            .source_adjacent_max_partition_fan_in
            .is_some_and(|fan_in| fan_in <= SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_FAN_IN)
}

fn select_prepared_sort_groupby_for_source_adjacent(
    mut decision: AggregateStrategyCapabilityDecision,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::PreparedSortGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::PreparedSortGroupby.as_str().to_string());
    decision.facts.insert("prepared_sort_groupby_rejected_reason", "none".to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "selected_source_adjacent_execution".to_string());
    decision
        .facts
        .insert("prepared_sort_groupby_selection_reason", "source_adjacent_partial_prepared_execution".to_string());
    decision.facts.insert(
        "encoded_key_partial_rejected_reason",
        "encoded_key_partial_superseded_by_prepared_sort_groupby".to_string(),
    );
    decision.facts.insert("encoded_key_partial_cost_model_status", "not_selected".to_string());
    decision.facts.insert("encoded_key_partial_selection_reason", "prepared_sort_groupby_selected".to_string());
    decision
        .rejected_candidates
        .retain(|rejection| rejection.strategy != AggregateExecutionStrategy::PreparedSortGroupby);
    decision.rejected_candidates.insert(
        0,
        AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::EncodedKeyPartial,
            reason: "encoded_key_partial_superseded_by_prepared_sort_groupby",
        },
    );
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn reject_prepared_sort_groupby_for_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    reason: &'static str,
    selection_reason: &'static str,
) -> AggregateStrategyCapabilityDecision {
    if !prepared_sort_groupby_contract_available(&decision) {
        return decision;
    }
    decision.facts.insert("prepared_sort_groupby_rejected_reason", reason.to_string());
    decision.facts.insert("prepared_sort_groupby_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert("prepared_sort_groupby_selection_reason", selection_reason.to_string());
    if let Some(rejection) = decision
        .rejected_candidates
        .iter_mut()
        .find(|rejection| rejection.strategy == AggregateExecutionStrategy::PreparedSortGroupby)
    {
        rejection.reason = reason;
    } else {
        decision.rejected_candidates.push(AggregateStrategyCandidateRejection {
            strategy: AggregateExecutionStrategy::PreparedSortGroupby,
            reason,
        });
    }
    update_rejected_candidate_summary(&mut decision);
    decision
}

#[allow(clippy::too_many_arguments)]
fn execute_prepared_sort_groupby_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    aggregate: LoweredAggregate,
    expected_output_names: Vec<String>,
    mut decision: crate::capability::AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    input_evidence: Option<PartialAggregateInputEvidence>,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = frame.num_rows() as u64;
    let total_timer = Instant::now();
    let prepare_timer = Instant::now();
    let prepared = budgeted_value(
        frame
            .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
            .options(GroupByOptions::builder().include_null_keys(true).build())
            .prepare_sort_groupby_with_budget(token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.aggregate.prepare_partial.prepared_sort_groupby",
                    "failed to prepare bounded native cudf sort groupby",
                    error,
                )
            })?,
        metrics,
        "cudf_query_engine.aggregate.prepare_partial.prepared_sort_groupby",
    );
    let prepare_elapsed = prepare_timer.elapsed();
    let facts = prepared.prepared_facts();
    decision.facts.insert("prepared_sort_groupby_num_groups", facts.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", facts.grouped_key_output_bytes.to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", facts.max_group_size.to_string());

    let aggregate_timer = Instant::now();
    let mut output = budgeted_value(
        prepared.agg_with_budget(&aggregate.aggregations, token).map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "cudf_query_engine.aggregate.execute_partial.prepared_sort_groupby",
                "failed to execute bounded native cudf prepared sort groupby aggregate",
                error,
            )
        })?,
        metrics,
        "cudf_query_engine.aggregate.execute_partial.prepared_sort_groupby",
    );
    let aggregate_elapsed = aggregate_timer.elapsed();
    rename_groupby_key_columns(
        &mut output,
        &aggregate,
        "cudf_query_engine.aggregate.prepared_sort_groupby.rename_keys",
    )?;
    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.prepared_sort_groupby.select_partial",
            "failed to select native prepared sort groupby partial aggregate output columns",
            error,
        )
    })?;
    let elapsed = total_timer.elapsed();
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
        prepare_elapsed,
        prepare_elapsed.saturating_add(aggregate_elapsed),
        Duration::ZERO,
        elapsed,
        input_evidence,
    );
    Ok(GpuDataFrame::new(output))
}

fn reject_encoded_key_partial_for_cost_model(
    mut decision: AggregateStrategyCapabilityDecision,
    reason: &'static str,
    selection_reason: &'static str,
    evidence: Option<PriorAggregateCardinalityEvidence>,
) -> AggregateStrategyCapabilityDecision {
    decision.selected_strategy = AggregateExecutionStrategy::LibcudfGroupby;
    decision.facts.insert("selected_strategy", AggregateExecutionStrategy::LibcudfGroupby.as_str().to_string());
    decision.facts.insert("encoded_key_partial_rejected_reason", reason.to_string());
    decision.facts.insert("encoded_key_partial_cost_model_status", "rejected_by_cost_model".to_string());
    decision.facts.insert("encoded_key_partial_selection_reason", selection_reason.to_string());
    if let Some(evidence) = evidence {
        add_prior_cardinality_facts(&mut decision, &evidence);
    }
    decision.rejected_candidates.insert(
        0,
        AggregateStrategyCandidateRejection { strategy: AggregateExecutionStrategy::EncodedKeyPartial, reason },
    );
    update_rejected_candidate_summary(&mut decision);
    decision
}

fn add_prior_cardinality_facts(
    decision: &mut AggregateStrategyCapabilityDecision,
    evidence: &PriorAggregateCardinalityEvidence,
) {
    decision.facts.insert("encoded_key_partial_prior_input_rows", evidence.input_rows.to_string());
    decision.facts.insert("encoded_key_partial_prior_output_group_count", evidence.output_group_count.to_string());
    decision.facts.insert("encoded_key_partial_prior_reduction_ratio_ppm", evidence.reduction_ratio_ppm.to_string());
    decision.facts.insert("encoded_key_partial_prior_strategy", evidence.selected_strategy.clone());
}

fn update_rejected_candidate_summary(decision: &mut AggregateStrategyCapabilityDecision) {
    decision.facts.insert(
        "rejected_candidate_strategies",
        decision
            .rejected_candidates
            .iter()
            .map(|rejection| format!("{}:{}", rejection.strategy.as_str(), rejection.reason))
            .collect::<Vec<_>>()
            .join(";"),
    );
}

#[allow(clippy::too_many_arguments)]
fn execute_encoded_key_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    mut frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    mut aggregate: LoweredAggregate,
    expected_output_names: Vec<String>,
    decision: crate::capability::AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    input_evidence: Option<PartialAggregateInputEvidence>,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = frame.num_rows() as u64;
    let overrides = token.execution_overrides();
    let total_timer = Instant::now();
    let mut encode_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;
    let mut encoded_key_decoders = HashMap::<String, cudf::frame::factorization::Codebook>::new();

    for index in 0..aggregate.key_columns.len() {
        let key_column = aggregate.key_columns[index].clone();
        let field = prepared_input_schema.field_with_name(key_column.as_str()).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.encoded_key_partial",
                "encoded-key partial aggregate references a missing group key column",
            )
            .with_fact("group_key", key_column.as_str())
            .with_fact("arrow_error", error.to_string())
        })?;
        if !encoded_partial_key_dtype_requires_encoding(field.data_type()) {
            continue;
        }
        let encoded_key_column = unique_encoded_group_key_name(&frame, &aggregate, index);
        let encode_timer = Instant::now();
        let codebook = frame
            .series(key_column.as_str())
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.encoded_key_partial.codebook_input",
                    "failed to access encoded-key partial aggregate key column",
                    error,
                )
            })?
            .codebook_in(&overrides)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "cudf_query_engine.aggregate.encoded_key_partial.codebook",
                        "failed to build encoded-key partial aggregate codebook",
                        error,
                    ),
                    token,
                )
            })?;
        frame = frame
            .with_execution(&overrides)
            .with_encoded_column(encoded_key_column.as_str(), key_column.as_str(), &codebook)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "cudf_query_engine.aggregate.encoded_key_partial.encode",
                        "failed to encode grouped aggregate key column",
                        error,
                    ),
                    token,
                )
            })?;
        encode_elapsed += encode_timer.elapsed();
        let output_key_name = aggregate.output_key_names[index].clone();
        aggregate.key_columns[index] = encoded_key_column;
        encoded_key_decoders.insert(output_key_name, codebook);
    }

    let groupby_timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute_partial.encoded_key",
        "failed to execute bounded native encoded-key partial cudf aggregate",
    )?;
    let groupby_elapsed = groupby_timer.elapsed();

    for output_key_name in &aggregate.output_key_names {
        let Some(codebook) = encoded_key_decoders.get(output_key_name) else {
            continue;
        };
        let decode_timer = Instant::now();
        output = output
            .with_execution(&overrides)
            .with_decoded_column(output_key_name.as_str(), output_key_name.as_str(), codebook)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "cudf_query_engine.aggregate.encoded_key_partial.decode",
                        "failed to decode grouped aggregate key output column",
                        error,
                    ),
                    token,
                )
            })?;
        decode_elapsed += decode_timer.elapsed();
    }

    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_partial",
            "failed to select native encoded-key partial aggregate output columns",
            error,
        )
    })?;
    let elapsed = total_timer.elapsed();
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
        encode_elapsed,
        groupby_elapsed,
        decode_elapsed,
        elapsed,
        input_evidence,
    );
    Ok(GpuDataFrame::new(output))
}

fn encoded_partial_key_dtype_requires_encoding(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn unique_encoded_group_key_name(frame: &DataFrame, aggregate: &LoweredAggregate, key_index: usize) -> String {
    let base = format!("__cudf_encoded_group_key_{key_index}");
    if !frame.contains(base.as_str()) && !aggregate.output_names.iter().any(|name| name == &base) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !frame.contains(candidate.as_str()) && !aggregate.output_names.iter().any(|name| name == &candidate) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

pub(in crate::exec) fn record_default_grouped_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    memory_mode: &'static str,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    input_rows: u64,
    output_rows: u64,
    elapsed: Duration,
) {
    let decision = aggregate_strategy_capability_decision(spec, input_schema.as_ref(), stage);
    record_grouped_aggregate_strategy_decision(
        metrics,
        node_id,
        stage,
        memory_mode,
        spec,
        input_schema,
        input_rows,
        output_rows,
        decision,
        Duration::ZERO,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec) fn record_grouped_aggregate_strategy_decision(
    metrics: &mut QueryMetrics,
    node_id: Option<String>,
    stage: AggregateExecutionStage,
    memory_mode: &'static str,
    _spec: &AggregateSpec,
    _input_schema: &arrow::datatypes::SchemaRef,
    input_rows: u64,
    output_rows: u64,
    decision: crate::capability::AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    encode_elapsed: Duration,
    groupby_elapsed: Duration,
    decode_elapsed: Duration,
    strategy_elapsed: Duration,
    input_evidence: Option<PartialAggregateInputEvidence>,
) {
    let mut facts = decision.facts;
    add_partial_aggregate_input_evidence_facts(&mut facts, input_evidence);
    facts.insert("memory_mode", memory_mode.to_string());
    facts.insert("input_rows", input_rows.to_string());
    facts.insert("chunk_row_count", if stage == AggregateExecutionStage::Partial { input_rows } else { 0 }.to_string());
    facts.insert("chunk_count", metrics.source_chunks_planned.to_string());
    facts.insert("partial_aggregate_count", metrics.partial_aggregate_count.to_string());
    facts.insert("output_group_count", output_rows.to_string());
    facts.insert("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_rows).to_string());
    facts.insert("selection_elapsed_ns", duration_ns(selection_elapsed).to_string());
    facts.insert("encode_elapsed_ns", duration_ns(encode_elapsed).to_string());
    facts.insert("groupby_elapsed_ns", duration_ns(groupby_elapsed).to_string());
    facts.insert("decode_elapsed_ns", duration_ns(decode_elapsed).to_string());
    facts.insert("strategy_operator_elapsed_ns", duration_ns(strategy_elapsed).to_string());
    metrics
        .record_aggregate_strategy_decision(node_id.unwrap_or_else(|| format!("aggregate.{}", stage.as_str())), &facts);
}

fn reduction_ratio_ppm(input_rows: u64, output_rows: u64) -> u64 {
    if input_rows == 0 {
        return 0;
    }
    output_rows.saturating_mul(1_000_000).saturating_div(input_rows)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn prepare_filtered_aggregate_frame(
    context: &ExecutionContext,
    frame: DataFrame,
    input_schema: &arrow::datatypes::SchemaRef,
    spec: &AggregateSpec,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<(DataFrame, AggregateSpec, arrow::datatypes::SchemaRef)> {
    if !aggregate_spec_has_filters(spec) {
        return Ok((frame, spec.clone(), Arc::clone(input_schema)));
    }

    let mut expressions = Vec::<(String, cudf::Expr)>::new();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut prepared_aggr_exprs = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, aggregate_expr) in spec.aggr_exprs.iter().enumerate() {
        let Some(filter) = &aggregate_expr.filter else {
            prepared_aggr_exprs.push(aggregate_expr.clone());
            continue;
        };
        let input_type = filtered_aggregate_input_type(aggregate_expr, input_schema)?;
        let value_expr = filtered_aggregate_value_expr(aggregate_expr);
        let alias = unique_filtered_aggregate_alias(index, &fields);
        let masked = Expr::case(
            vec![CaseWhen::new(filter.clone(), value_expr)],
            Expr::literal(LiteralValue::Null(input_type.clone())),
        );
        expressions.push((alias.clone(), lower_expr(&masked)?));
        fields.push(Field::new(alias.as_str(), input_type, true));
        prepared_aggr_exprs.push(
            AggregateExpr::new(aggregate_expr.function, Expr::column(alias), aggregate_expr.alias.clone())
                .with_distinct(aggregate_expr.distinct),
        );
    }

    let timer = Instant::now();
    let frame = match token {
        Some(token) => frame
            .with_columns_with_budget(expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.aggregate.filter_inputs",
                    "failed to materialize bounded native aggregate filter inputs",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "cudf_query_engine.aggregate.filter_inputs"))?,
        None => {
            let overrides = execution_overrides(context);
            frame.with_execution(&overrides).with_columns(expressions).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.aggregate.filter_inputs",
                    "failed to materialize native aggregate filter inputs",
                    error,
                )
            })?
        },
    };
    metrics.operator_time += timer.elapsed();
    let mut prepared_spec = AggregateSpec::new(spec.group_exprs.clone(), prepared_aggr_exprs);
    prepared_spec.grouping_sets = spec.grouping_sets.clone();
    Ok((frame, prepared_spec, Arc::new(Schema::new(fields))))
}

fn filtered_aggregate_value_expr(aggregate_expr: &AggregateExpr) -> Expr {
    if aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && global_count_star_expr(&aggregate_expr.expr)
    {
        Expr::literal(LiteralValue::Int64(1))
    } else {
        aggregate_expr.expr.clone()
    }
}

fn filtered_aggregate_input_type(
    aggregate_expr: &AggregateExpr,
    input_schema: &arrow::datatypes::SchemaRef,
) -> Result<ArrowDataType> {
    if aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && global_count_star_expr(&aggregate_expr.expr)
    {
        return Ok(ArrowDataType::Int64);
    }
    aggregate_expr_type(&aggregate_expr.expr, input_schema)
}

fn aggregate_expr_type(expr: &Expr, input_schema: &arrow::datatypes::SchemaRef) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => {
            input_schema.field_with_name(column.name()).map(|field| field.data_type().clone()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.aggregate.filter_input_type",
                    "aggregate filter input references a missing column",
                )
                .with_fact("column", column.name())
                .with_fact("arrow_error", error.to_string())
            })
        },
        Expr::Alias { expr, .. } => aggregate_expr_type(expr, input_schema),
        Expr::Literal(value) => Ok(value.data_type()),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.aggregate.filter_input_type",
            "filtered aggregate input must be a prepared column or literal",
        )
        .with_fact("expression", expr.to_string())),
    }
}

fn unique_filtered_aggregate_alias(index: usize, fields: &[Field]) -> String {
    let base = format!("__cudf_filtered_aggregate_arg_{index}");
    if fields.iter().all(|field| field.name() != base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let alias = format!("{base}_{suffix}");
        if fields.iter().all(|field| field.name() != alias.as_str()) {
            return alias;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn execute_grouping_sets_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_stddev_samp(spec) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.grouping_sets",
            "grouping sets do not currently support stddev_samp",
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
        "cudf_query_engine.aggregate.grouping_sets",
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
            "cudf_query_engine.aggregate.grouping_sets_select",
            "failed to select native grouping-sets aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_grouping_sets_partial_aggregate_with_budget(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_stddev_samp(spec) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.grouping_sets_partial",
            "grouping sets do not currently support stddev_samp",
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
        "cudf_query_engine.aggregate.grouping_sets_partial",
    )?;
    output = cast_count_output_columns_with_budget(output, spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.grouping_sets_select_partial",
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
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
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
        let working = prepare_grouping_set_frame(
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
        )?;
        let output = match token {
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
                )?
            },
            None => {
                let overrides = execution_overrides(context);
                execute_groupby_aggregate(&overrides, working, aggregate, operation)?
            },
        };
        outputs.push(output);
    }
    concat_grouping_set_outputs(context, outputs, token, operation)
}

#[allow(clippy::too_many_arguments)]
fn prepare_grouping_set_frame(
    context: &ExecutionContext,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
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
            let overrides = execution_overrides(context);
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
        None => first.with_execution(&execution_overrides(context)).concat_many(&other_refs),
    }
    .map_err(|error| QueryError::from_cudf(operation, "failed to concatenate grouping-set aggregate outputs", error))
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
    let output = frame
        .with_execution(overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to execute native cudf group-only aggregate", error)
        })?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
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
    let output = frame
        .with_execution(&budget_overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| attach_budget_token_failure_facts(QueryError::from_cudf(operation, message, error), token))?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
}

fn select_and_rename_groupby_distinct_keys(output: DataFrame, aggregate: &LoweredAggregate) -> Result<DataFrame> {
    let mut output =
        output.select(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.select_group_only_keys",
                "failed to select native group-only aggregate keys",
                error,
            )
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
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HostGroupKeyValue {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    UInt64(u64),
    Float64(u64),
    Utf8(String),
}

#[derive(Debug, Clone)]
enum HostAggregateState {
    Count { count: u64 },
    Sum { sum: f64, seen: bool },
    Min { value: f64, seen: bool },
    Max { value: f64, seen: bool },
    Avg { sum: f64, count: u64 },
    StddevSamp { count: u64, mean: f64, m2: f64 },
}

impl HostAggregateState {
    fn new(function: AggregateFunction) -> Self {
        match function {
            AggregateFunction::Count | AggregateFunction::NUnique => Self::Count { count: 0 },
            AggregateFunction::Sum => Self::Sum { sum: 0.0, seen: false },
            AggregateFunction::Min => Self::Min { value: 0.0, seen: false },
            AggregateFunction::Max => Self::Max { value: 0.0, seen: false },
            AggregateFunction::Avg => Self::Avg { sum: 0.0, count: 0 },
            AggregateFunction::StddevSamp => Self::StddevSamp { count: 0, mean: 0.0, m2: 0.0 },
        }
    }

    fn update(&mut self, value: Option<f64>) {
        match self {
            Self::Count { count } => {
                if value.is_some() {
                    *count += 1;
                }
            },
            Self::Sum { sum, seen } => {
                if let Some(value) = value {
                    *sum += value;
                    *seen = true;
                }
            },
            Self::Min { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value < *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Max { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value > *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Avg { sum, count } => {
                if let Some(value) = value {
                    *sum += value;
                    *count += 1;
                }
            },
            Self::StddevSamp { count, mean, m2 } => {
                if let Some(value) = value {
                    let new_count = *count + 1;
                    let delta1 = value - *mean;
                    let new_mean = delta1 / new_count as f64 + *mean;
                    let delta2 = value - new_mean;
                    *m2 += delta1 * delta2;
                    *mean = new_mean;
                    *count = new_count;
                }
            },
        }
    }

    fn evaluate_f64(&self) -> Option<f64> {
        match self {
            Self::Count { count } => Some(*count as f64),
            Self::Sum { sum, seen } => seen.then_some(*sum),
            Self::Min { value, seen } | Self::Max { value, seen } => seen.then_some(*value),
            Self::Avg { sum, count } => (*count > 0).then_some(*sum / *count as f64),
            Self::StddevSamp { count, m2, .. } => {
                if *count <= 1 {
                    None
                } else {
                    Some((*m2 / (*count - 1) as f64).sqrt())
                }
            },
        }
    }

    fn evaluate_i64(&self) -> Option<i64> {
        match self {
            Self::Count { count } => Some(*count as i64),
            _ => self.evaluate_f64().map(|value| value as i64),
        }
    }
}

struct HostAggregateGroup {
    keys: Vec<HostGroupKeyValue>,
    aggregates: Vec<HostAggregateState>,
}

fn execute_datafusion_exact_host_aggregate(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let timer = Instant::now();
    let frame = sort_by_datafusion_order_if_present(context, input.into_frame(), metrics)?;
    let schema = schema_with_datafusion_order_columns(&frame, input_schema);
    let handle = GpuResultHandle::new(context, GpuDataFrame::new(frame), schema)?;
    let batch = handle.to_host_record_batch()?;
    let output = execute_datafusion_exact_host_aggregate_batch(context, spec, &batch, output_schema)?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_datafusion_exact_host_aggregate_batch(
    context: &ExecutionContext,
    spec: &AggregateSpec,
    batch: &RecordBatch,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    let key_columns = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, "cudf_query_engine.aggregate.stddev_exact.lower_key"))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_columns = spec.aggr_exprs.iter().map(aggregate_input_column).collect::<Result<Vec<_>>>()?;
    let mut group_indices = HashMap::<Vec<HostGroupKeyValue>, usize>::new();
    let mut groups = Vec::<HostAggregateGroup>::new();
    let row_count = batch.num_rows();

    for row in 0..row_count {
        let key = key_columns
            .iter()
            .map(|(input_name, _)| host_group_key_value(batch, input_name, row))
            .collect::<Result<Vec<_>>>()?;
        let group_index = match group_indices.get(&key) {
            Some(index) => *index,
            None => {
                let index = groups.len();
                group_indices.insert(key.clone(), index);
                groups.push(HostAggregateGroup {
                    keys: key,
                    aggregates: spec.aggr_exprs.iter().map(|expr| HostAggregateState::new(expr.function)).collect(),
                });
                index
            },
        };
        for ((expr, input_column), state) in
            spec.aggr_exprs.iter().zip(&aggregate_columns).zip(&mut groups[group_index].aggregates)
        {
            let value = match input_column {
                HostAggregateInput::Column(name) => host_numeric_value(batch, name, row)?,
                HostAggregateInput::NonNullLiteral => Some(1.0),
                HostAggregateInput::NullLiteral => None,
            };
            if matches!(expr.function, AggregateFunction::Count | AggregateFunction::NUnique) {
                state.update(value.map(|_| 1.0));
            } else {
                state.update(value);
            }
        }
    }

    if key_columns.is_empty() && groups.is_empty() {
        groups.push(HostAggregateGroup {
            keys: Vec::new(),
            aggregates: spec.aggr_exprs.iter().map(|expr| HostAggregateState::new(expr.function)).collect(),
        });
    }

    let mut arrays = Vec::<ArrayRef>::with_capacity(output_schema.fields().len());
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if output_index < key_columns.len() {
            arrays.push(host_key_output_array(&groups, output_index, field.data_type())?);
        } else {
            let aggregate_index = output_index - key_columns.len();
            arrays.push(host_aggregate_output_array(&groups, aggregate_index, field.data_type())?);
        }
    }
    let output = RecordBatch::try_new(Arc::clone(output_schema), arrays).map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.aggregate.stddev_exact.output",
            "failed to construct DataFusion-exact stddev aggregate Arrow output",
        )
        .with_source(error)
    })?;
    DataFrame::from_record_batch(context.runtime(), &output).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.stddev_exact.output",
            "failed to upload DataFusion-exact stddev aggregate output",
            error,
        )
    })
}

fn host_group_expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias { alias, .. } => alias.clone(),
        Expr::Column(column) => column.display_name().to_string(),
        _ => expr.to_string(),
    }
}

enum HostAggregateInput {
    Column(String),
    NonNullLiteral,
    NullLiteral,
}

fn aggregate_input_column(expr: &AggregateExpr) -> Result<HostAggregateInput> {
    if matches!(expr.function, AggregateFunction::Count) && global_count_star_expr(&expr.expr) {
        return Ok(HostAggregateInput::NonNullLiteral);
    }
    aggregate_input_expr(&expr.expr)
}

fn aggregate_input_expr(expr: &Expr) -> Result<HostAggregateInput> {
    match expr {
        Expr::Column(column) => Ok(HostAggregateInput::Column(column.name().to_string())),
        Expr::Alias { expr, .. } => aggregate_input_expr(expr),
        Expr::Literal(LiteralValue::Null(_)) => Ok(HostAggregateInput::NullLiteral),
        Expr::Literal(_) => Ok(HostAggregateInput::NonNullLiteral),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.aggregate.stddev_exact.lower_expr",
            "DataFusion-exact stddev aggregate requires prepared column aggregate inputs",
        )
        .with_fact("expression", expr.to_string())),
    }
}

fn host_group_key_value(batch: &RecordBatch, column: &str, row: usize) -> Result<HostGroupKeyValue> {
    let array = batch.column(batch.schema().index_of(column).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.aggregate.stddev_exact.key",
            "group key column is missing from aggregate input batch",
        )
        .with_fact("column", column)
        .with_source(error)
    })?);
    if array.is_null(row) {
        return Ok(HostGroupKeyValue::Null);
    }
    match array.data_type() {
        ArrowDataType::Boolean => Ok(HostGroupKeyValue::Boolean(
            array.as_any().downcast_ref::<BooleanArray>().expect("validated BooleanArray").value(row),
        )),
        ArrowDataType::Int32 => Ok(HostGroupKeyValue::Int32(
            array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array").value(row),
        )),
        ArrowDataType::Date32 => Ok(HostGroupKeyValue::Int32(
            array.as_any().downcast_ref::<Date32Array>().expect("validated Date32Array").value(row),
        )),
        ArrowDataType::Int64 => Ok(HostGroupKeyValue::Int64(
            array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array").value(row),
        )),
        ArrowDataType::UInt64 => Ok(HostGroupKeyValue::UInt64(
            array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array").value(row),
        )),
        ArrowDataType::Float64 => Ok(HostGroupKeyValue::Float64(
            array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array").value(row).to_bits(),
        )),
        ArrowDataType::Utf8 => Ok(HostGroupKeyValue::Utf8(
            array.as_any().downcast_ref::<StringArray>().expect("validated StringArray").value(row).to_string(),
        )),
        ArrowDataType::LargeUtf8 => Ok(HostGroupKeyValue::Utf8(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("validated LargeStringArray")
                .value(row)
                .to_string(),
        )),
        ArrowDataType::Utf8View => Ok(HostGroupKeyValue::Utf8(
            array.as_any().downcast_ref::<StringViewArray>().expect("validated StringViewArray").value(row).to_string(),
        )),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.key",
            "DataFusion-exact stddev aggregate does not support this group key type",
        )
        .with_fact("column", column)
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_numeric_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<f64>> {
    let array = batch.column(batch.schema().index_of(column).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.aggregate.stddev_exact.value",
            "aggregate input column is missing from aggregate input batch",
        )
        .with_fact("column", column)
        .with_source(error)
    })?);
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        ArrowDataType::Int32 => {
            Ok(Some(array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array").value(row) as f64))
        },
        ArrowDataType::Int64 => {
            Ok(Some(array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array").value(row) as f64))
        },
        ArrowDataType::UInt64 => {
            Ok(Some(array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array").value(row) as f64))
        },
        ArrowDataType::Float64 => {
            Ok(Some(array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array").value(row)))
        },
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.value",
            "DataFusion-exact stddev aggregate requires numeric aggregate inputs",
        )
        .with_fact("column", column)
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_key_output_array(
    groups: &[HostAggregateGroup],
    key_index: usize,
    data_type: &ArrowDataType,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Boolean => Ok(Arc::new(BooleanArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Boolean(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int32 => Ok(Arc::new(Int32Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int32(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Date32 => Ok(Arc::new(Date32Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int32(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int64(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::UInt64 => Ok(Arc::new(UInt64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::UInt64(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Float64(value)) => Some(f64::from_bits(*value)),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8 => Ok(Arc::new(StringArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.clone()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::LargeUtf8 => Ok(Arc::new(LargeStringArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.clone()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8View => Ok(Arc::new(StringViewArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.as_str()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.key_output",
            "DataFusion-exact stddev aggregate does not support this output group key type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_aggregate_output_array(
    groups: &[HostAggregateGroup],
    aggregate_index: usize,
    data_type: &ArrowDataType,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_i64))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_f64))
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.aggregate_output",
            "DataFusion-exact stddev aggregate does not support this aggregate output type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}
