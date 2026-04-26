#[test]
fn multi_key_single_sum_without_prior_uses_libcudf_when_large() {
    let decision = partial_strategy_for_multi_key_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("completed_cardinality_evidence_required_for_large_variable_width")
    );
    assert_eq!(
        decision.facts.get("encoded_key_partial_selection_reason").map(String::as_str),
        Some("completed_cardinality_evidence_required_for_large_variable_width")
    );
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
    assert_eq!(
        decision.facts.get("partial_strategy_unseen_path").map(String::as_str),
        Some("streaming_group_key_partition_or_libcudf")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_sort_for_large_group_key_chunks() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let evidence = source_adjacent_group_key_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1,
        1,
    );
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("completed_cardinality_evidence_required_for_large_variable_width")
    );
    assert_eq!(
        decision.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_sort_for_bounded_group_key_chunks() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let evidence = source_adjacent_group_key_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS,
        1,
    );
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        64_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_ne!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("completed_cardinality_evidence_required_for_large_variable_width")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_sort_for_bounded_unpartitioned_chunks() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let evidence = source_adjacent_unpartitioned_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS,
    );
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        64_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_ne!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_sort_for_byte_bounded_unpartitioned_chunks() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let mut evidence = source_adjacent_unpartitioned_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1,
    );
    evidence.source_chunk_estimated_source_read_bytes =
        Some(AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_ne!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_envelope_still_rejects_prepared_for_oversized_unpartitioned_bytes() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let mut evidence = source_adjacent_unpartitioned_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1,
    );
    evidence.source_chunk_estimated_source_read_bytes = Some(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES + 1,
    );
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_for_unknown_read_all_row_groups() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let mut evidence = source_adjacent_unpartitioned_evidence(0);
    evidence.source_chunk_reads_all_row_groups = true;
    evidence.source_chunk_estimated_source_read_bytes = None;
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
}

#[test]
fn source_adjacent_envelope_does_not_select_prepared_even_with_byte_bound_unknown_row_groups() {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let mut evidence = source_adjacent_unpartitioned_evidence(0);
    evidence.source_chunk_reads_all_row_groups = true;
    evidence.source_chunk_estimated_source_read_bytes =
        Some(AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_ne!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_row_group_option_is_reported_but_does_not_select_prepared() {
    let (schema, spec) = multi_key_sum_spec();
    let options =
        AggregateStrategyExecutionOptions::default().with_source_adjacent_prepared_sort_groupby_max_row_groups(0);
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let mut evidence = source_adjacent_group_key_evidence(0, 1);
    evidence.source_chunk_reads_all_row_groups = true;
    evidence.source_chunk_estimated_source_read_bytes = Some(1);
    let decision = select_partial_aggregate_strategy_decision_with_options(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        options,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("aggregate_source_adjacent_prepared_sort_groupby_max_row_groups").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
}

#[test]
fn aggregate_strategy_options_report_source_adjacent_thresholds_without_selecting_prepared() {
    let (schema, spec) = multi_key_sum_spec();
    let options = AggregateStrategyExecutionOptions::default()
        .with_source_adjacent_prepared_sort_groupby_max_row_groups(
            AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1,
        );
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let evidence = source_adjacent_group_key_evidence(
        AggregateStrategyExecutionOptions::DEFAULT_SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1,
        1,
    );
    let decision = select_partial_aggregate_strategy_decision_with_options(
        decision,
        None,
        &spec,
        &schema,
        65_000_000,
        Some(evidence),
        options,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("aggregate_source_adjacent_prepared_sort_groupby_max_row_groups").map(String::as_str),
        Some("65")
    );
    assert_eq!(decision.facts.get("aggregate_strategy_options_version").map(String::as_str), Some("1"));
    assert_eq!(
        decision.facts.get("aggregate_encoded_key_partial_low_cardinality_threshold_ppm").map(String::as_str),
        Some("100000")
    );
}

#[test]
fn source_adjacent_row_balanced_prior_stays_off_prepared_sort_groupby() {
    let (schema, spec) = fixed_width_sum_spec();
    let node_id = NodeId::from_index_for_internal_use(10);
    let evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint: Some(0xfeed_cafe),
        source_adjacent_partial: true,
        source_chunk_row_group_count: 16,
        source_chunk_reads_all_row_groups: false,
        source_chunk_known_row_count: Some(5_000_000),
        source_chunk_estimated_source_read_bytes: Some(512 * 1024 * 1024),
        source_chunk_size_estimate_based: false,
        source_adjacent_partition_strategy: "row_balanced",
        source_adjacent_partition_count: Some(16),
        source_adjacent_non_empty_partition_count: Some(16),
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: Some(1),
    };
    let metrics = metrics_with_prior_partial_decision(
        node_id,
        &spec,
        &schema,
        Some(evidence),
        5_000_000,
        1_000,
        "libcudf_groupby",
    );
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        Some(node_id),
        &spec,
        &schema,
        5_000_000,
        Some(evidence),
        &metrics,
    );

    // Fixed-width priors can still select prepared when evidence matches; row-balanced
    // source-adjacent identity is part of the structural key, so the prior with the
    // matching envelope is reused when present.
    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_prior_low_cardinality")
    );
}
