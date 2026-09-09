#[test]
fn large_single_key_does_not_admit_exact_cardinality_probe_candidate() {
    let (schema, spec, decision) = partial_strategy_for_simple_string_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000), None);
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
}

#[test]
fn medium_single_string_avg_does_not_admit_exact_cardinality_probe_candidate() {
    let (schema, spec) = simple_string_avg_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
}

#[test]
fn medium_single_string_sum_keeps_libcudf_without_hot_path_probe() {
    let (schema, spec) = simple_string_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn aggregate_strategy_options_no_longer_enable_hot_path_exact_probe() {
    let (schema, spec) = simple_string_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision_with_options(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        AggregateStrategyExecutionOptions::default(),
        &ExecutionMetrics::detached("aggregate-test"),
    );
    let options =
        AggregateStrategyExecutionOptions::default().with_exact_single_key_cardinality_probe_min_input_rows(100_000);

    assert_eq!(
        exact_single_key_cardinality_probe_candidate_with_options(&decision, &spec, &schema, 204_000, options),
        None
    );
    assert_eq!(
        decision.facts.get("aggregate_exact_single_key_cardinality_probe_min_input_rows").map(String::as_str),
        Some("500000")
    );
}

#[test]
fn medium_single_string_mixed_avg_sum_keeps_libcudf_without_hot_path_probe() {
    let (schema, spec) = simple_string_avg_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn medium_partial_single_string_avg_keeps_libcudf_without_hot_path_probe() {
    let (schema, spec) = simple_string_avg_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn exact_cardinality_probe_candidate_rejects_small_inputs() {
    let (schema, spec, small_decision) = partial_strategy_for_simple_string_sum(10_000);
    assert_eq!(exact_single_key_cardinality_probe_candidate(&small_decision, &spec, &schema, 10_000), None);
}

#[test]
fn large_two_key_does_not_admit_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = fixed_width_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        5_000_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000), None);
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
}

#[test]
fn medium_two_key_keeps_libcudf_without_hot_path_probe() {
    let (schema, spec) = fixed_width_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn large_three_key_does_not_admit_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = q31_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        5_000_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000), None);
}

#[test]
fn three_key_runtime_probe_helper_stays_disabled_for_hot_path() {
    let (schema, spec) = q31_key_sum_spec();
    let wide_input_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial),
        None,
        &spec,
        &schema,
        AggregateStrategyExecutionOptions::DEFAULT_EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(
        exact_grouped_key_cardinality_probe_candidate(
            &wide_input_decision,
            &spec,
            &schema,
            AggregateStrategyExecutionOptions::DEFAULT_EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1
        ),
        None
    );

    let schema = Schema::new(vec![
        Field::new("store", DataType::Int64, false),
        Field::new("quarter", DataType::Int64, false),
        Field::new("year", DataType::Int64, false),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("store"), Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    let all_fixed_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial),
        None,
        &spec,
        &schema,
        5_000_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&all_fixed_decision, &spec, &schema, 5_000_000), None);
}

#[test]
fn five_key_does_not_admit_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = q76_key_sum_spec();
    let decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        6_900_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );

    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 6_900_000), None);
}

#[test]
fn five_key_runtime_probe_helper_stays_disabled_for_hot_path() {
    let (schema, spec) = q76_key_sum_spec();
    let wide_input_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        AggregateStrategyExecutionOptions::DEFAULT_EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(
        exact_grouped_key_cardinality_probe_candidate(
            &wide_input_decision,
            &spec,
            &schema,
            AggregateStrategyExecutionOptions::DEFAULT_EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1
        ),
        None
    );

    let schema = Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("col_name", DataType::Utf8, false),
        Field::new("year", DataType::Int64, false),
        Field::new("quarter", DataType::Int64, false),
        Field::new("category_id", DataType::Int64, false),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![
            Expr::column("channel"),
            Expr::column("col_name"),
            Expr::column("year"),
            Expr::column("quarter"),
            Expr::column("category_id"),
        ],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    let wrong_mix_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        6_900_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&wrong_mix_decision, &spec, &schema, 6_900_000), None);

    let schema = Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("col_name", DataType::Utf8, false),
        Field::new("year_label", DataType::Utf8, false),
        Field::new("quarter", DataType::Int64, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![
            Expr::column("channel"),
            Expr::column("col_name"),
            Expr::column("year_label"),
            Expr::column("quarter"),
            Expr::column("category"),
        ],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    let string_heavy_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        6_900_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&string_heavy_decision, &spec, &schema, 6_900_000), None);
}

#[test]
fn exact_grouped_key_cardinality_probe_candidate_rejects_single_small_and_wide_keys() {
    let (schema, spec, single_decision) = partial_strategy_for_simple_string_sum(5_000_000);
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&single_decision, &spec, &schema, 5_000_000), None);

    let (schema, spec) = multi_key_sum_spec();
    let small_decision = partial_strategy_for_multi_key_sum(10_000);
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&small_decision, &spec, &schema, 10_000), None);

    let mut wide_group_exprs = Vec::new();
    let mut wide_fields = Vec::new();
    for index in 0..=AggregateStrategyExecutionOptions::DEFAULT_EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS {
        let name = format!("k{index}");
        wide_group_exprs.push(Expr::column(name.as_str()));
        wide_fields.push(Field::new(name, DataType::Int32, false));
    }
    wide_fields.push(Field::new("sales", DataType::Int64, true));
    let wide_schema = Schema::new(wide_fields);
    let wide_spec = AggregateSpec::new(
        wide_group_exprs,
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    let wide_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&wide_spec, &wide_schema, AggregateExecutionStage::Partial),
        None,
        &wide_spec,
        &wide_schema,
        5_000_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    assert_eq!(
        exact_grouped_key_cardinality_probe_candidate(&wide_decision, &wide_spec, &wide_schema, 5_000_000),
        None
    );
}

#[test]
fn exact_cardinality_probe_selects_prepared_for_tiny_string_keysets_only() {
    let (_, _, decision) = partial_strategy_for_simple_string_sum(5_000_000);
    let low_cardinality = ExactSingleKeyCardinalityProbe {
        key_column: "county".to_string(),
        input_rows: 5_000_000,
        non_null_distinct_count: 128,
        null_count: 0,
        num_groups: 128,
        reduction_ratio_ppm: reduction_ratio_ppm(5_000_000, 128),
        elapsed: Duration::ZERO,
    };
    assert!(exact_cardinality_probe_should_select_prepared_sort_groupby(&decision, &low_cardinality));

    let broad_string_keyset = ExactSingleKeyCardinalityProbe {
        num_groups: 6_925,
        reduction_ratio_ppm: reduction_ratio_ppm(5_000_000, 6_925),
        ..low_cardinality
    };
    assert!(!exact_cardinality_probe_should_select_prepared_sort_groupby(&decision, &broad_string_keyset));
}

#[test]
fn exact_grouped_key_cardinality_probe_selects_prepared_by_reduction_ratio() {
    let (schema, spec) = fixed_width_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        5_000_000,
        None,
        &ExecutionMetrics::detached("aggregate-test"),
    );
    let low_cardinality = ExactGroupedKeyCardinalityProbe {
        key_columns: vec!["quarter".to_string(), "year".to_string()],
        input_rows: 5_000_000,
        num_groups: 1_000,
        reduction_ratio_ppm: reduction_ratio_ppm(5_000_000, 1_000),
        elapsed: Duration::ZERO,
    };
    assert!(exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby(&decision, &low_cardinality));

    let high_cardinality = ExactGroupedKeyCardinalityProbe {
        num_groups: 4_000_000,
        reduction_ratio_ppm: reduction_ratio_ppm(5_000_000, 4_000_000),
        ..low_cardinality
    };
    assert!(!exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby(&decision, &high_cardinality));
}

#[test]
fn multi_key_single_sum_without_prior_keeps_libcudf_groupby_when_small() {
    let decision = partial_strategy_for_multi_key_sum(10_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_cost_model_small_input")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(
        decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_cost_model_small_input")
    );
}
