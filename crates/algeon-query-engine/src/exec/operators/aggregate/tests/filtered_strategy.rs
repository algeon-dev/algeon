#[test]
fn filtered_global_strategy_keeps_small_single_pass_materialization() {
    let (schema, spec) = filtered_global_q09_like_spec(3);
    let decision = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        1_024,
        2 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .unwrap();

    assert!(!decision.use_segmented);
    assert_eq!(
        decision.decision.facts.get("selected_strategy").map(String::as_str),
        Some("filtered_global_single_pass_materialized")
    );
    assert_eq!(
        decision.decision.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("single_pass_with_columns")
    );
}

#[test]
fn filtered_global_strategy_segments_large_unsafe_materialization() {
    let (schema, spec) = filtered_global_q09_like_spec(15);
    let decision = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        287_997_024,
        80 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .unwrap();

    assert!(decision.use_segmented);
    assert_eq!(
        decision.decision.facts.get("selected_strategy").map(String::as_str),
        Some("filtered_global_segmented_state")
    );
    assert_eq!(
        decision.decision.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(decision.decision.facts.get("aggregate_filter_count").map(String::as_str), Some("15"));
    assert_eq!(decision.decision.facts.get("filtered_input_unique_predicate_count").map(String::as_str), Some("1"));
}

#[test]
fn filtered_global_strategy_segments_exact_tpcds_q09_sf100_shape() {
    let (schema, spec) = filtered_global_tpcds_q09_sf100_spec();
    let decision = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        287_997_024,
        80 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .unwrap();

    assert!(decision.use_segmented);
    assert_eq!(
        decision.decision.facts.get("selected_strategy").map(String::as_str),
        Some("filtered_global_segmented_state")
    );
    assert_eq!(
        decision.decision.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(decision.decision.facts.get("aggregate_filter_count").map(String::as_str), Some("15"));
    assert_eq!(spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg).count(), 10);
    assert_eq!(spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Count).count(), 5);
    assert_eq!(decision.decision.facts.get("filtered_input_unique_predicate_count").map(String::as_str), Some("5"));
    assert_eq!(
        decision.decision.facts.get("filtered_global_retained_predicate_bytes_per_row").map(String::as_str),
        Some("10")
    );
    let rows_per_chunk = decision
        .decision
        .facts
        .get("filtered_global_rows_per_chunk")
        .and_then(|rows| rows.parse::<u64>().ok())
        .expect("rows per chunk fact should be present");
    assert!(
        rows_per_chunk < 287_997_024,
        "SF100 q09 should be row-chunked even when the native execution budget is large"
    );
    let inputs_per_segment = decision
        .decision
        .facts
        .get("filtered_global_inputs_per_segment")
        .and_then(|inputs| inputs.parse::<usize>().ok())
        .expect("inputs per segment fact should be present");
    assert!(
        inputs_per_segment < 15,
        "SF100 q09 should segment masked aggregate inputs instead of materializing all 15 together"
    );
}

#[test]
fn filtered_global_strategy_caps_row_chunks_by_budget() {
    let (schema, spec) = filtered_global_q09_like_spec(15);
    let decision = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        287_997_024,
        32 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .unwrap();

    assert!(decision.use_segmented);
    let expected_target = (32 * 1024 * 1024 / 8).to_string();
    assert_eq!(
        decision.decision.facts.get("filtered_global_segment_target_temporary_bytes").map(String::as_str),
        Some(expected_target.as_str())
    );
    let rows_per_chunk = decision
        .decision
        .facts
        .get("filtered_global_rows_per_chunk")
        .and_then(|rows| rows.parse::<u64>().ok())
        .expect("rows per chunk fact should be present");
    assert!(rows_per_chunk < 1_000_000, "row chunk sizing should follow the small budget");
}

#[test]
fn filtered_global_strategy_rejects_large_distinct_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let spec = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "distinct_values")
                .with_distinct(true)
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let error = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        300_000_000,
        8 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .expect_err("large unsupported filtered state should be rejected");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("filtered_input_unsupported_stateful_merge"), Some("true"));
    assert_eq!(error.fact_value("filtered_input_materialization_strategy"), Some("rejected"));
    assert_eq!(
        error.fact_value("rejected_candidate_strategies"),
        Some(
            "filtered_global_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold;filtered_global_segmented_state:unsupported_stateful_filtered_merge"
        )
    );
}

#[test]
fn filtered_global_strategy_rejects_large_statistical_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let spec = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("value"), "stddev_value")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let error = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        300_000_000,
        8 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .expect_err("large unsupported statistical filtered state should be rejected");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("filtered_aggregate_scope"), Some("global"));
    assert_eq!(error.fact_value("filtered_input_unsupported_stateful_merge"), Some("true"));
    assert_eq!(error.fact_value("filtered_input_materialization_strategy"), Some("rejected"));
}

#[test]
fn filtered_global_strategy_rejects_large_median_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let spec = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let error = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        300_000_000,
        8 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .expect_err("large filtered median state should be rejected instead of segmented");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("filtered_aggregate_scope"), Some("global"));
    assert_eq!(error.fact_value("filtered_input_unsupported_stateful_merge"), Some("true"));
    assert_eq!(error.fact_value("filtered_input_materialization_strategy"), Some("rejected"));
    assert_eq!(
        error.fact_value("rejected_candidate_strategies"),
        Some(
            "filtered_global_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold;filtered_global_segmented_state:unsupported_stateful_filtered_merge"
        )
    );
}

#[test]
fn filtered_global_segmented_failure_records_error_budget_progress_and_partial_evidence() {
    let (schema, spec) = filtered_global_q09_like_spec(15);
    let decision = select_filtered_global_aggregate_strategy(
        &spec,
        &schema,
        60_000_000,
        32 * 1024 * 1024,
        AggregateExecutionStage::Partial,
    )
    .expect("q09-like spec should select a strategy");
    assert_eq!(
        decision.decision.facts.get("selected_strategy").map(String::as_str),
        Some("filtered_global_segmented_state")
    );

    let context = ExecutionContext::standalone(StandaloneConfig {
        memory_control_policy: MemoryControlPolicy {
            device_budget_bytes: 32 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        ..StandaloneConfig::default()
    })
    .expect("execution context builds");
    let metrics = ExecutionMetrics::detached("aggregate-test");
    let node_id = NodeId::from_index_for_test(7);
    let stats = FilteredGlobalSegmentRuntimeStats {
        chunk_count: 2,
        segment_count: 5,
        max_segment_rows: 10_000,
        state_rows_merged: 2,
        observed_temporary_peak_bytes: 4096,
        observed_retained_peak_bytes: 512,
    };
    let error = QueryError::new(
        QueryErrorCode::QueryMemoryLimitExceeded,
        "query_engine.filtered_global_aggregate.filter_inputs",
        "failed while testing segmented filtered global failure reporting",
    )
    .with_memory_failure(MemoryFailure {
        boundary: "test_boundary".into(),
        operation: "dataframe::with_columns".into(),
        usage: Some(MemoryUsage {
            limit_bytes: 32 * 1024 * 1024,
            peak_bytes: 8192,
            peak_aligned_bytes: 8192,
            current_bytes: 2048,
            current_aligned_bytes: 2048,
            failed_allocation_count: 1,
            largest_failed_allocation_bytes: 65_536,
        }),
    });
    let input_evidence = Some(source_adjacent_group_key_evidence(4, 3));

    record_filtered_global_segmented_failure(
        &metrics,
        &context,
        Some(node_id),
        AggregateExecutionStage::Partial,
        &spec,
        &schema,
        60_000_000,
        input_evidence,
        decision,
        stats,
        &error,
        Duration::from_millis(3),
    );

    let snapshot = metrics.snapshot();
    let recorded = snapshot
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.node_id == node_id.to_string())
        .expect("segmented failure strategy should be recorded");
    assert_eq!(recorded.facts.get("execution_stage").map(String::as_str), Some("partial"));
    assert_eq!(recorded.facts.get("selected_strategy").map(String::as_str), Some("filtered_global_segmented_state"));
    assert_eq!(recorded.facts.get("filtered_global_segmented_failure").map(String::as_str), Some("true"));
    assert_eq!(
        recorded.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(recorded.facts.get("filtered_global_actual_chunk_count").map(String::as_str), Some("2"));
    assert_eq!(recorded.facts.get("filtered_global_actual_segment_count").map(String::as_str), Some("5"));
    assert_eq!(recorded.facts.get("filtered_global_state_rows_merged").map(String::as_str), Some("2"));
    assert_eq!(
        recorded.facts.get("filtered_global_segmented_failure_operation").map(String::as_str),
        Some("query_engine.filtered_global_aggregate.filter_inputs")
    );
    assert_eq!(
        recorded.facts.get("filtered_global_segmented_failure_error_code").map(String::as_str),
        Some("query_memory_limit_exceeded")
    );
    assert_eq!(recorded.facts.get("budget_status").map(String::as_str), Some("exceeded_or_lower_layer_rejected"));
    assert_eq!(recorded.facts.get("budget_operation").map(String::as_str), Some("dataframe::with_columns"));
    assert_eq!(recorded.facts.get("budget_peak_bytes").map(String::as_str), Some("8192"));
    assert_eq!(recorded.facts.get("budget_bytes").map(String::as_str), Some("33554432"));
    assert_eq!(recorded.facts.get("filtered_global_observed_temporary_peak_bytes").map(String::as_str), Some("8192"));
    assert_eq!(recorded.facts.get("filtered_global_observed_retained_peak_bytes").map(String::as_str), Some("2048"));
    assert_eq!(recorded.facts.get("source_adjacent_partial").map(String::as_str), Some("true"));
    assert_eq!(recorded.facts.get("source_chunk_row_group_count").map(String::as_str), Some("4"));
}

#[test]
fn filtered_grouped_strategy_rejects_large_nunique_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let spec = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "unique_values")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let error = select_filtered_grouped_aggregate_strategy(
        &spec,
        &schema,
        300_000_000,
        8 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .expect_err("large unsupported filtered grouped state should be rejected");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("filtered_aggregate_scope"), Some("grouped"));
    assert_eq!(error.fact_value("filtered_input_unsupported_stateful_merge"), Some("true"));
    assert_eq!(error.fact_value("filtered_input_materialization_strategy"), Some("rejected"));
}

#[test]
fn filtered_grouped_strategy_rejects_large_median_state() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Float64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let spec = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let error = select_filtered_grouped_aggregate_strategy(
        &spec,
        &schema,
        300_000_000,
        8 * 1024 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .expect_err("large filtered grouped median state should be rejected instead of segmented");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("filtered_aggregate_scope"), Some("grouped"));
    assert_eq!(error.fact_value("filtered_input_unsupported_stateful_merge"), Some("true"));
    assert_eq!(error.fact_value("filtered_input_materialization_strategy"), Some("rejected"));
    assert_eq!(
        error.fact_value("rejected_candidate_strategies"),
        Some(
            "filtered_grouped_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold;filtered_grouped_segmented_state:unsupported_stateful_filtered_merge"
        )
    );
}

#[test]
fn filtered_grouped_strategy_caps_row_chunks_by_budget() {
    let (schema, spec) = filtered_grouped_q09_like_spec(15);
    let decision = select_filtered_grouped_aggregate_strategy(
        &spec,
        &schema,
        287_997_024,
        32 * 1024 * 1024,
        AggregateExecutionStage::Single,
    )
    .unwrap();

    assert!(decision.use_segmented);
    assert_eq!(
        decision.decision.facts.get("selected_strategy").map(String::as_str),
        Some("filtered_grouped_segmented_state")
    );
    assert_eq!(
        decision.decision.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_grouped_segmented_state")
    );
    assert_eq!(
        decision.decision.facts.get("rejected_candidate_strategies").map(String::as_str),
        Some("filtered_grouped_single_pass_materialized:filtered_input_materialization_estimate_exceeds_threshold")
    );
    let expected_target = (32 * 1024 * 1024 / 8).to_string();
    assert_eq!(
        decision.decision.facts.get("filtered_grouped_segment_target_temporary_bytes").map(String::as_str),
        Some(expected_target.as_str())
    );
    let rows_per_chunk = decision
        .decision
        .facts
        .get("filtered_grouped_rows_per_chunk")
        .and_then(|rows| rows.parse::<u64>().ok())
        .expect("rows per chunk fact should be present");
    assert!(rows_per_chunk < 1_000_000, "row chunk sizing should follow the small budget");
}
