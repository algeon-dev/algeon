#[test]
fn multi_key_single_sum_without_prior_uses_libcudf_as_cardinality_probe_when_large() {
    let decision = partial_strategy_for_multi_key_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("runtime_cardinality_probe_required")
    );
    assert_eq!(
        decision.facts.get("encoded_key_partial_selection_reason").map(String::as_str),
        Some("runtime_cardinality_probe_required")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_rejects_single_large_source_chunk() {
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
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_group_key_source_partition")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("source_adjacent_row_balanced_partition")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_keeps_bounded_group_key_source_chunks() {
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_keeps_bounded_unpartitioned_source_chunks() {
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_keeps_large_byte_bounded_unpartitioned_source_chunks() {
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_rejects_large_unpartitioned_source_chunks_without_byte_bound() {
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
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_group_key_source_partition")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_rejects_unknown_read_all_row_groups_without_byte_bound() {
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
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_group_key_source_partition")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_uses_byte_bound_for_unknown_unpartitioned_read_all_row_groups() {
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
}

#[test]
fn source_adjacent_prepared_sort_groupby_zero_row_group_threshold_does_not_admit_unknown_group_key_chunks() {
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
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_group_key_source_partition")
    );
}

#[test]
fn aggregate_strategy_options_change_source_adjacent_row_group_gate_and_report_thresholds() {
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PreparedSortGroupby);
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

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_group_key_source_partition")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),
        Some("source_adjacent_row_balanced_partition")
    );
}
