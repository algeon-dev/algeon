use super::*;

#[test]
fn fixed_width_low_cardinality_prior_selects_prepared_sort_groupby() {
    let (schema, spec) = fixed_width_sum_spec();
    let node_id = NodeId::from_index_for_test(7);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 1_000, "libcudf_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(decision, Some(node_id), &spec, &schema, 5_000_000, None, &metrics);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str), Some("none"));
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_prior_low_cardinality")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_prior_grouped_key_output_bytes").map(String::as_str),
        Some("8000")
    );
}

#[test]
fn prior_partial_evidence_selects_the_same_strategy_when_no_decision_row_is_retained() {
    let (schema, spec) = fixed_width_sum_spec();
    let node_id = NodeId::from_index_for_test(7);
    let session = RuntimeObservationSession::new("zero-retention", None, RuntimeObservationConfig::bounded(0, 0));
    let metrics =
        ExecutionMetrics::bind(&session.handle(), PlanExecutionId::new("zero-retention-plan"), "zero-retention", None);
    let facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(&spec, &schema, None)),
        ("input_rows", "5000000".to_string()),
        ("output_group_count", "1000".to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(5_000_000, 1_000).to_string()),
        ("selected_strategy", "libcudf_groupby".to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(node_id, &facts);
    assert!(metrics.snapshot().aggregate_strategy_decisions.is_empty(), "retention must decline the decision row");

    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(decision, Some(node_id), &spec, &schema, 5_000_000, None, &metrics);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(decision.facts.get("prior_cardinality_evidence_scope").map(String::as_str), Some("exact_node"));
}

#[test]
fn fixed_width_low_cardinality_structural_prior_selects_prepared_sort_groupby_for_different_node() {
    let (schema, spec) = fixed_width_sum_spec();
    let prior_node_id = NodeId::from_index_for_test(7);
    let current_node_id = NodeId::from_index_for_test(11);
    let metrics = metrics_with_prior_partial_structural_decision(
        prior_node_id,
        &spec,
        &schema,
        5_000_000,
        1_000,
        "libcudf_groupby",
    );
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        Some(current_node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        &metrics,
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(decision.facts.get("prior_cardinality_evidence_scope").map(String::as_str), Some("structural_reused"));
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_prior_grouped_key_output_bytes").map(String::as_str),
        Some("8000")
    );
}

#[test]
fn exact_node_partial_cardinality_prior_takes_precedence_over_structural_prior() {
    let (schema, spec) = fixed_width_sum_spec();
    let current_node_id = NodeId::from_index_for_test(12);
    let structural_node_id = NodeId::from_index_for_test(13);
    let metrics = metrics_with_prior_partial_structural_decision(
        structural_node_id,
        &spec,
        &schema,
        5_000_000,
        1_000,
        "libcudf_groupby",
    );
    let exact_facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(&spec, &schema, None)),
        ("input_rows", "5000000".to_string()),
        ("output_group_count", "4000000".to_string()),
        ("reduction_ratio_ppm", "800000".to_string()),
        ("selected_strategy", "libcudf_groupby".to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(current_node_id, &exact_facts);
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        Some(current_node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        &metrics,
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(decision.facts.get("prior_cardinality_evidence_scope").map(String::as_str), Some("exact_node"));
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
}

#[test]
fn exact_node_partial_cardinality_prior_with_mismatched_structural_key_is_ignored() {
    let (schema, spec) = fixed_width_sum_spec();
    let current_node_id = NodeId::from_index_for_test(14);
    let metrics = ExecutionMetrics::detached("aggregate-test");
    let mismatched_facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", "agg-v1-fnv64-mismatched".to_string()),
        ("input_rows", "5000000".to_string()),
        ("output_group_count", "1000".to_string()),
        ("reduction_ratio_ppm", "200".to_string()),
        ("selected_strategy", "libcudf_groupby".to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(current_node_id, &mismatched_facts);
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        Some(current_node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        &metrics,
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(decision.facts.get("prior_cardinality_evidence_scope").map(String::as_str), Some("absent"));
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
fn multi_key_string_fixed_low_cardinality_prior_supersedes_encoded_key() {
    let (schema, spec) = multi_key_sum_spec();
    let node_id = NodeId::from_index_for_test(8);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 1_000, "libcudf_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(decision, Some(node_id), &spec, &schema, 5_000_000, None, &metrics);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_superseded_by_prepared_sort_groupby")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_prior_grouped_key_output_bytes").map(String::as_str),
        Some("unknown")
    );
}

#[test]
fn simple_string_low_cardinality_prior_keeps_encoded_key_partial() {
    let (schema, spec) = simple_string_sum_spec();
    let node_id = NodeId::from_index_for_test(9);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 1_000, "libcudf_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(decision, Some(node_id), &spec, &schema, 5_000_000, None, &metrics);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::EncodedKeyPartial);
    assert_eq!(
        decision.facts.get("encoded_key_partial_cost_model_status").map(String::as_str),
        Some("selected_low_cardinality")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("encoded_key_preferred_for_simple_string_key")
    );
}

#[test]
fn aggregate_strategy_options_change_encoded_key_partial_reduction_threshold() {
    let (schema, spec) = simple_string_sum_spec();
    let node_id = NodeId::from_index_for_test(20);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 750_000, "libcudf_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let default_decision = select_partial_aggregate_strategy_decision(
        decision.clone(),
        Some(node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        &metrics,
    );
    let options =
        AggregateStrategyExecutionOptions::default().with_encoded_key_partial_low_cardinality_threshold_ppm(200_000);
    let custom_decision = select_partial_aggregate_strategy_decision_with_options(
        decision,
        Some(node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        options,
        &metrics,
    );

    assert_eq!(default_decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(custom_decision.selected_strategy, AggregateExecutionStrategy::EncodedKeyPartial);
    assert_eq!(
        custom_decision.facts.get("encoded_key_partial_reduction_ratio_threshold_ppm").map(String::as_str),
        Some("200000")
    );
    assert_eq!(
        custom_decision.facts.get("aggregate_encoded_key_partial_low_cardinality_threshold_ppm").map(String::as_str),
        Some("200000")
    );
}

#[test]
fn aggregate_strategy_options_change_prepared_sort_groupby_reduction_threshold() {
    let (schema, spec) = fixed_width_sum_spec();
    let node_id = NodeId::from_index_for_test(21);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 750_000, "libcudf_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let default_decision = select_partial_aggregate_strategy_decision(
        decision.clone(),
        Some(node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        &metrics,
    );
    let options =
        AggregateStrategyExecutionOptions::default().with_prepared_sort_groupby_low_cardinality_threshold_ppm(200_000);
    let custom_decision = select_partial_aggregate_strategy_decision_with_options(
        decision,
        Some(node_id),
        &spec,
        &schema,
        5_000_000,
        None,
        options,
        &metrics,
    );

    assert_eq!(default_decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(custom_decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        custom_decision.facts.get("aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm").map(String::as_str),
        Some("200000")
    );
}

#[test]
fn simple_string_prepared_sort_groupby_prior_keeps_prepared_sort_groupby() {
    let (schema, spec) = simple_string_sum_spec();
    let node_id = NodeId::from_index_for_test(19);
    let metrics =
        metrics_with_prior_partial_decision(node_id, &spec, &schema, None, 5_000_000, 1_000, "prepared_sort_groupby");
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision =
        select_partial_aggregate_strategy_decision(decision, Some(node_id), &spec, &schema, 5_000_000, None, &metrics);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_prior_low_cardinality")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("prior_reduction_ratio_at_or_below_threshold")
    );
}
