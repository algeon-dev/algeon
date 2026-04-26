use std::collections::HashMap;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{groupby::rename_groupby_key_columns, *};

pub(in crate::exec::operators::aggregate) fn execute_prepared_sort_groupby_single_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    aggregate: LoweredAggregate,
    expected_output_names: Vec<String>,
    mut decision: crate::capability::AggregateStrategyCapabilityDecision,
    selection_elapsed: Duration,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = frame.num_rows() as u64;
    let total_timer = Instant::now();
    let prepare_timer = Instant::now();
    let (observation, plan_execution_id, parent_span_id) = crate::observability::stage_context_from_execution(context);
    let prepare_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation.clone(),
        plan_execution_id.clone(),
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::PreparedGroupbyPrepare,
        crate::observability::RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
    );
    let prepared = match frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .prepare_sort_groupby_with_budget(token)
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.aggregate.prepare_single.prepared_sort_groupby",
                "failed to prepare bounded nexus native sort groupby",
                error,
            )
        }) {
        Ok(outcome) => {
            let prepared =
                budgeted_value(outcome, metrics, "nexus_query_engine.aggregate.prepare_single.prepared_sort_groupby");
            prepare_guard.finish(Some(prepared.prepared_facts().num_groups as u64), None);
            prepared
        },
        Err(error) => {
            prepare_guard.fail(&error);
            return Err(error);
        },
    };
    let prepare_elapsed = prepare_timer.elapsed();
    let facts = prepared.prepared_facts();
    decision.facts.insert("prepared_sort_groupby_num_groups", facts.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", facts.grouped_key_output_bytes.to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", facts.max_group_size.to_string());

    let aggregate_timer = Instant::now();
    let aggregate_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::PreparedGroupbyAggregate,
        crate::observability::RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
    );
    let mut output = match prepared.agg_with_budget(&aggregate.aggregations, token).map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "nexus_query_engine.aggregate.execute_single.prepared_sort_groupby",
            "failed to execute bounded nexus native prepared sort groupby aggregate",
            error,
        )
    }) {
        Ok(outcome) => {
            let output =
                budgeted_value(outcome, metrics, "nexus_query_engine.aggregate.execute_single.prepared_sort_groupby");
            aggregate_guard.finish(Some(output.num_rows() as u64), None);
            output
        },
        Err(error) => {
            aggregate_guard.fail(&error);
            return Err(error);
        },
    };
    let aggregate_elapsed = aggregate_timer.elapsed();
    rename_groupby_key_columns(
        &mut output,
        &aggregate,
        "nexus_query_engine.aggregate.prepared_sort_groupby.rename_keys",
    )?;
    output = finalize_avg_output_columns_with_budget(context, output, &spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.prepared_sort_groupby.select_single",
            "failed to select native prepared sort groupby aggregate output columns",
            error,
        )
    })?;
    let elapsed = total_timer.elapsed();
    let output_rows = output.num_rows() as u64;
    metrics.operator_time += elapsed;
    record_single_grouped_aggregate_strategy_decision_with_metrics(
        metrics,
        context,
        node_id.map(|id| id.to_string()),
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
    );
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec::operators::aggregate) fn execute_prepared_sort_groupby_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
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
    let (observation, plan_execution_id, parent_span_id) = crate::observability::stage_context_from_execution(context);
    let prepare_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation.clone(),
        plan_execution_id.clone(),
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::PreparedGroupbyPrepare,
        crate::observability::RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
    );
    let prepared = match frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .prepare_sort_groupby_with_budget(token)
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.aggregate.prepare_partial.prepared_sort_groupby",
                "failed to prepare bounded nexus native sort groupby",
                error,
            )
        }) {
        Ok(outcome) => {
            let prepared =
                budgeted_value(outcome, metrics, "nexus_query_engine.aggregate.prepare_partial.prepared_sort_groupby");
            prepare_guard.finish(Some(prepared.prepared_facts().num_groups as u64), None);
            prepared
        },
        Err(error) => {
            prepare_guard.fail(&error);
            return Err(error);
        },
    };
    let prepare_elapsed = prepare_timer.elapsed();
    let facts = prepared.prepared_facts();
    decision.facts.insert("prepared_sort_groupby_num_groups", facts.num_groups.to_string());
    decision.facts.insert("prepared_sort_groupby_grouped_key_output_bytes", facts.grouped_key_output_bytes.to_string());
    decision.facts.insert("prepared_sort_groupby_max_group_size", facts.max_group_size.to_string());

    let aggregate_timer = Instant::now();
    let aggregate_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::PreparedGroupbyAggregate,
        crate::observability::RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
    );
    let mut output = match prepared.agg_with_budget(&aggregate.aggregations, token).map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "nexus_query_engine.aggregate.execute_partial.prepared_sort_groupby",
            "failed to execute bounded nexus native prepared sort groupby aggregate",
            error,
        )
    }) {
        Ok(outcome) => {
            let output =
                budgeted_value(outcome, metrics, "nexus_query_engine.aggregate.execute_partial.prepared_sort_groupby");
            aggregate_guard.finish(Some(output.num_rows() as u64), None);
            output
        },
        Err(error) => {
            aggregate_guard.fail(&error);
            return Err(error);
        },
    };
    let aggregate_elapsed = aggregate_timer.elapsed();
    rename_groupby_key_columns(
        &mut output,
        &aggregate,
        "nexus_query_engine.aggregate.prepared_sort_groupby.rename_keys",
    )?;
    output = cast_count_output_columns_with_budget(output, &spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.aggregate.prepared_sort_groupby.select_partial",
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
        context.aggregate_strategy_execution_options(),
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

pub(in crate::exec::operators::aggregate) fn execute_encoded_key_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node_id: Option<NodeId>,
    mut frame: DataFrame,
    spec: AggregateSpec,
    prepared_input_schema: ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
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
                "nexus_query_engine.aggregate.encoded_key_partial",
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
                    "nexus_query_engine.aggregate.encoded_key_partial.codebook_input",
                    "failed to access encoded-key partial aggregate key column",
                    error,
                )
            })?
            .codebook_in(&overrides)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "nexus_query_engine.aggregate.encoded_key_partial.codebook",
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
                        "nexus_query_engine.aggregate.encoded_key_partial.encode",
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
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.aggregate.execute_partial.encoded_key",
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
                        "nexus_query_engine.aggregate.encoded_key_partial.decode",
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
            "nexus_query_engine.aggregate.select_partial",
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
        context.aggregate_strategy_execution_options(),
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
    let base = format!("__nexus_native_encoded_group_key_{key_index}");
    crate::naming::unique_alias_with_suffix(base.as_str(), |candidate| {
        frame.contains(candidate) || aggregate.output_names.iter().any(|name| name == candidate)
    })
}
