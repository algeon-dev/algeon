#[test]
fn large_single_key_keeps_libcudf_with_exact_probe_disabled() {
    let (_, _, decision) = partial_strategy_for_simple_string_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
}

#[test]
fn medium_single_string_avg_keeps_libcudf_with_exact_probe_disabled() {
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
}

#[test]
fn large_two_key_reports_exact_probe_disabled() {
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
