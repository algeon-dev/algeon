use std::collections::BTreeMap;

use arrow::datatypes::{DataType, Field, Schema};

use super::{
    partial_strategy::{
        EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS, ExactGroupedKeyCardinalityProbe,
        ExactSingleKeyCardinalityProbe, SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS,
        SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES, aggregate_structural_key, reduction_ratio_ppm,
    },
    *,
};
use crate::{capability::aggregate_strategy_capability_decision, metrics::QueryMetrics};

fn multi_key_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("county", DataType::Utf8View, true),
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

fn fixed_width_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("sales", DataType::Int64, true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

fn simple_string_sum_spec() -> (Schema, AggregateSpec) {
    let schema =
        Schema::new(vec![Field::new("county", DataType::Utf8View, true), Field::new("sales", DataType::Int64, true)]);
    let spec = AggregateSpec::new(
        vec![Expr::column("county")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

fn partial_strategy_for_multi_key_sum(input_rows: u64) -> AggregateStrategyCapabilityDecision {
    let (schema, spec) = multi_key_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        input_rows,
        None,
        &QueryMetrics::default(),
    )
}

fn partial_strategy_for_simple_string_sum(
    input_rows: u64,
) -> (Schema, AggregateSpec, AggregateStrategyCapabilityDecision) {
    let (schema, spec) = simple_string_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        input_rows,
        None,
        &QueryMetrics::default(),
    );
    (schema, spec, decision)
}

fn metrics_with_prior_partial_decision(
    node_id: NodeId,
    spec: &AggregateSpec,
    schema: &Schema,
    input_evidence: Option<PartialAggregateInputEvidence>,
    input_rows: u64,
    output_group_count: u64,
    selected_strategy: &'static str,
) -> QueryMetrics {
    let mut metrics = QueryMetrics::default();
    let facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(spec, schema, input_evidence)),
        ("input_rows", input_rows.to_string()),
        ("output_group_count", output_group_count.to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_group_count).to_string()),
        ("selected_strategy", selected_strategy.to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(node_id, &facts);
    metrics
}

fn metrics_with_prior_partial_structural_decision(
    node_id: NodeId,
    spec: &AggregateSpec,
    schema: &Schema,
    input_rows: u64,
    output_group_count: u64,
    selected_strategy: &'static str,
) -> QueryMetrics {
    let mut metrics = QueryMetrics::default();
    let facts = BTreeMap::from([
        ("execution_stage", "partial".to_string()),
        ("aggregate_structural_key", aggregate_structural_key(spec, schema, None)),
        ("input_rows", input_rows.to_string()),
        ("output_group_count", output_group_count.to_string()),
        ("reduction_ratio_ppm", reduction_ratio_ppm(input_rows, output_group_count).to_string()),
        ("selected_strategy", selected_strategy.to_string()),
    ]);
    metrics.record_aggregate_strategy_decision(node_id, &facts);
    metrics
}

fn source_adjacent_group_key_evidence(
    row_group_count: u64,
    max_partition_fan_in: u64,
) -> PartialAggregateInputEvidence {
    source_adjacent_evidence("group_key", row_group_count, Some(max_partition_fan_in))
}

fn source_adjacent_unpartitioned_evidence(row_group_count: u64) -> PartialAggregateInputEvidence {
    source_adjacent_evidence("none", row_group_count, None)
}

fn source_adjacent_evidence(
    partition_strategy: &'static str,
    row_group_count: u64,
    max_partition_fan_in: Option<u64>,
) -> PartialAggregateInputEvidence {
    PartialAggregateInputEvidence {
        source_chunk_fingerprint: Some(0xabc),
        source_adjacent_partial: true,
        source_chunk_row_group_count: row_group_count,
        source_chunk_reads_all_row_groups: false,
        source_chunk_known_row_count: Some(row_group_count.saturating_mul(1_000_000)),
        source_chunk_estimated_source_read_bytes: Some(row_group_count.saturating_mul(16 * 1024 * 1024)),
        source_chunk_size_estimate_based: false,
        source_adjacent_partition_strategy: partition_strategy,
        source_adjacent_partition_count: (partition_strategy != "none").then_some(32),
        source_adjacent_non_empty_partition_count: (partition_strategy != "none").then_some(4),
        source_adjacent_partition_key_count: (partition_strategy == "group_key").then_some(2),
        source_adjacent_max_partition_fan_in: max_partition_fan_in,
    }
}

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
    let evidence = source_adjacent_group_key_evidence(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1, 1);
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
    let evidence = source_adjacent_group_key_evidence(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS, 1);
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
    let evidence = source_adjacent_unpartitioned_evidence(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS);
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
    let mut evidence = source_adjacent_unpartitioned_evidence(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1);
    evidence.source_chunk_estimated_source_read_bytes =
        Some(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES);
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
    let mut evidence = source_adjacent_unpartitioned_evidence(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_ROW_GROUPS + 1);
    evidence.source_chunk_estimated_source_read_bytes =
        Some(SOURCE_ADJACENT_PREPARED_SORT_GROUPBY_MAX_SOURCE_READ_BYTES + 1);
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
fn large_single_key_runtime_probe_is_exact_cardinality_probe_candidate() {
    let (schema, spec, decision) = partial_strategy_for_simple_string_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000), Some("county"));
}

#[test]
fn exact_cardinality_probe_candidate_rejects_small_inputs() {
    let (schema, spec, small_decision) = partial_strategy_for_simple_string_sum(10_000);
    assert_eq!(exact_single_key_cardinality_probe_candidate(&small_decision, &spec, &schema, 10_000), None);
}

#[test]
fn large_two_key_runtime_probe_is_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = fixed_width_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        5_000_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(
        exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000),
        Some(vec!["quarter".to_string(), "year".to_string()])
    );
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
    for index in 0..=EXACT_GROUPED_KEY_CARDINALITY_PROBE_MAX_KEY_COLUMNS {
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
        &QueryMetrics::default(),
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
        &QueryMetrics::default(),
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

#[test]
fn fixed_width_low_cardinality_prior_selects_prepared_sort_groupby() {
    let (schema, spec) = fixed_width_sum_spec();
    let node_id = NodeId::from_index_for_internal_use(7);
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
fn fixed_width_low_cardinality_structural_prior_selects_prepared_sort_groupby_for_different_node() {
    let (schema, spec) = fixed_width_sum_spec();
    let prior_node_id = NodeId::from_index_for_internal_use(7);
    let current_node_id = NodeId::from_index_for_internal_use(11);
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
    let current_node_id = NodeId::from_index_for_internal_use(12);
    let structural_node_id = NodeId::from_index_for_internal_use(13);
    let mut metrics = metrics_with_prior_partial_structural_decision(
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
    let current_node_id = NodeId::from_index_for_internal_use(14);
    let mut metrics = QueryMetrics::default();
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
        Some("selected_libcudf_runtime_probe")
    );
}

#[test]
fn multi_key_string_fixed_low_cardinality_prior_supersedes_encoded_key() {
    let (schema, spec) = multi_key_sum_spec();
    let node_id = NodeId::from_index_for_internal_use(8);
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
    let node_id = NodeId::from_index_for_internal_use(9);
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
fn simple_string_prepared_sort_groupby_prior_keeps_prepared_sort_groupby() {
    let (schema, spec) = simple_string_sum_spec();
    let node_id = NodeId::from_index_for_internal_use(19);
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
