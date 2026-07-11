use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::{
    filtered_global::{
        FilteredGlobalSegmentRuntimeStats, record_filtered_global_segmented_failure,
        select_filtered_global_aggregate_strategy,
    },
    filtered_grouped::select_filtered_grouped_aggregate_strategy,
    partial_strategy::{
        ExactGroupedKeyCardinalityProbe, ExactSingleKeyCardinalityProbe, aggregate_structural_key,
        exact_cardinality_probe_should_select_prepared_sort_groupby, exact_grouped_key_cardinality_probe_candidate,
        exact_grouped_key_cardinality_probe_should_select_prepared_sort_groupby,
        exact_single_key_cardinality_probe_candidate, exact_single_key_cardinality_probe_candidate_with_options,
        reduction_ratio_ppm, select_partial_aggregate_strategy_decision,
        select_partial_aggregate_strategy_decision_with_options,
    },
    *,
};
use crate::{
    capability::{AggregateStrategyExecutionOptions, aggregate_strategy_capability_decision},
    error::{QueryError, QueryErrorCode},
    metrics::QueryMetrics,
    runtime::MemoryControlPolicy,
};

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

fn q31_key_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("county", DataType::Utf8, true),
        Field::new("quarter", DataType::Int64, false),
        Field::new("year", DataType::Int64, false),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
    );
    (schema, spec)
}

fn q76_key_sum_spec() -> (Schema, AggregateSpec) {
    let schema = Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("col_name", DataType::Utf8, false),
        Field::new("year", DataType::Int64, false),
        Field::new("quarter", DataType::Int64, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("sales", DataType::Decimal128(17, 2), true),
    ]);
    let spec = AggregateSpec::new(
        vec![
            Expr::column("channel"),
            Expr::column("col_name"),
            Expr::column("year"),
            Expr::column("quarter"),
            Expr::column("category"),
        ],
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

fn simple_string_avg_spec() -> (Schema, AggregateSpec) {
    let schema =
        Schema::new(vec![Field::new("category", DataType::Utf8, true), Field::new("price", DataType::Float64, true)]);
    let spec = AggregateSpec::new(
        vec![Expr::column("category")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("price"), "avg_price")],
    );
    (schema, spec)
}

fn simple_string_avg_sum_spec() -> (Schema, AggregateSpec) {
    let schema =
        Schema::new(vec![Field::new("category", DataType::Utf8, true), Field::new("price", DataType::Float64, true)]);
    let spec = AggregateSpec::new(
        vec![Expr::column("category")],
        vec![
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("price"), "avg_price"),
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("price"), "sum_price"),
        ],
    );
    (schema, spec)
}

fn filtered_global_q09_like_spec(filter_count: usize) -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("quantity", DataType::Int64, true),
        Field::new("discount", DataType::Decimal128(7, 2), true),
        Field::new("net_paid", DataType::Decimal128(7, 2), true),
        Field::new("bucket", DataType::Boolean, true),
    ]));
    let mut aggr_exprs = Vec::new();
    for index in 0..filter_count {
        let value = if index % 3 == 0 {
            Expr::column("quantity")
        } else if index % 3 == 1 {
            Expr::column("discount")
        } else {
            Expr::column("net_paid")
        };
        let function = if index % 3 == 0 {
            AggregateFunction::Count
        } else if index % 3 == 1 {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Avg
        };
        aggr_exprs
            .push(AggregateExpr::new(function, value, format!("a{index}")).with_filter(Some(Expr::column("bucket"))));
    }
    (schema, AggregateSpec::new(Vec::new(), aggr_exprs))
}

fn filtered_global_tpcds_q09_sf100_spec() -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_ext_discount_amt", DataType::Decimal128(7, 2), true),
        Field::new("ss_net_paid", DataType::Decimal128(7, 2), true),
    ]));
    let mut aggr_exprs = Vec::new();
    for (bucket, lower) in [1, 21, 41, 61, 81].into_iter().enumerate() {
        let upper = lower + 19;
        let filter = quantity_between(lower, upper);
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("b{bucket}_cnt"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Avg,
                Expr::column("ss_ext_discount_amt"),
                format!("b{bucket}_avg_discount"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("ss_net_paid"), format!("b{bucket}_avg_net_paid"))
                .with_filter(Some(filter)),
        );
    }
    (schema, AggregateSpec::new(Vec::new(), aggr_exprs))
}

fn quantity_between(lower: i32, upper: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(lower))),
        BinaryOp::And,
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(upper))),
    )
}

fn filtered_grouped_q09_like_spec(filter_count: usize) -> (Arc<Schema>, AggregateSpec) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("quantity", DataType::Int64, true),
        Field::new("discount", DataType::Decimal128(7, 2), true),
        Field::new("net_paid", DataType::Decimal128(7, 2), true),
        Field::new("bucket", DataType::Boolean, true),
    ]));
    let mut aggr_exprs = Vec::new();
    for index in 0..filter_count {
        let value = if index % 3 == 0 {
            Expr::literal(LiteralValue::Int64(1))
        } else if index % 3 == 1 {
            Expr::column("discount")
        } else {
            Expr::column("net_paid")
        };
        let function = if index % 3 == 0 {
            AggregateFunction::Count
        } else if index % 3 == 1 {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Avg
        };
        aggr_exprs
            .push(AggregateExpr::new(function, value, format!("a{index}")).with_filter(Some(Expr::column("bucket"))));
    }
    (schema, AggregateSpec::new(vec![Expr::column("key")], aggr_exprs))
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
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

    let context = ExecutionContext::builder()
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 32 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .expect("execution context builds");
    let mut metrics = QueryMetrics::default();
    let node_id = NodeId::from_index_for_internal_use(7);
    let stats = FilteredGlobalSegmentRuntimeStats {
        chunk_count: 2,
        segment_count: 5,
        max_segment_rows: 10_000,
        state_rows_merged: 2,
        observed_temporary_peak_bytes: 4096,
        observed_retained_peak_bytes: 512,
        unresolved_contracts: BTreeSet::from(["cudf_lower_level_contract_missing".to_string()]),
    };
    let error = QueryError::new(
        QueryErrorCode::CudfResourceExhausted,
        "nexus_query_engine.filtered_global_aggregate.filter_inputs",
        "failed while testing segmented filtered global failure reporting",
    )
    .with_fact("budget_boundary", "test_boundary")
    .with_fact("budget_bytes", 32 * 1024 * 1024)
    .with_fact("budget_enforcement", "stream_bound_reservation")
    .with_fact("budget_status", "exceeded_or_lower_layer_rejected")
    .with_fact("budget_operation", "dataframe::with_columns_with_budget")
    .with_fact("budget_peak_bytes", 8192)
    .with_fact("budget_current_bytes", 2048);
    let input_evidence = Some(source_adjacent_group_key_evidence(4, 3));

    record_filtered_global_segmented_failure(
        &mut metrics,
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

    let recorded = metrics
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
        Some("nexus_query_engine.filtered_global_aggregate.filter_inputs")
    );
    assert_eq!(
        recorded.facts.get("filtered_global_segmented_failure_error_code").map(String::as_str),
        Some("cudf_resource_exhausted")
    );
    assert_eq!(recorded.facts.get("budget_status").map(String::as_str), Some("exceeded_or_lower_layer_rejected"));
    assert_eq!(recorded.facts.get("budget_operation").map(String::as_str), Some("dataframe::with_columns_with_budget"));
    assert_eq!(recorded.facts.get("budget_peak_bytes").map(String::as_str), Some("8192"));
    assert_eq!(recorded.facts.get("filtered_global_observed_temporary_peak_bytes").map(String::as_str), Some("8192"));
    assert_eq!(recorded.facts.get("filtered_global_observed_retained_peak_bytes").map(String::as_str), Some("2048"));
    assert_eq!(
        recorded.facts.get("filtered_input_lower_level_unresolved_contracts").map(String::as_str),
        Some("cudf_lower_level_contract_missing")
    );
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
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
fn large_single_key_runtime_probe_is_exact_cardinality_probe_candidate() {
    let (schema, spec, decision) = partial_strategy_for_simple_string_sum(5_000_000);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 5_000_000), Some("county"));
}

#[test]
fn medium_single_string_avg_runtime_probe_is_exact_cardinality_probe_candidate() {
    let (schema, spec) = simple_string_avg_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), Some("category"));
}

#[test]
fn medium_single_string_sum_runtime_probe_waits_for_libcudf_groupby_cardinality() {
    let (schema, spec) = simple_string_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn aggregate_strategy_options_change_exact_single_key_probe_threshold() {
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
        &QueryMetrics::default(),
    );
    let options =
        AggregateStrategyExecutionOptions::default().with_exact_single_key_cardinality_probe_min_input_rows(100_000);

    assert_eq!(
        exact_single_key_cardinality_probe_candidate_with_options(&decision, &spec, &schema, 204_000, options),
        Some("county")
    );
}

#[test]
fn medium_single_string_mixed_avg_sum_runtime_probe_waits_for_libcudf_groupby_cardinality() {
    let (schema, spec) = simple_string_avg_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn medium_partial_single_string_avg_runtime_probe_waits_for_libcudf_groupby_cardinality() {
    let (schema, spec) = simple_string_avg_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(exact_single_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
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
fn medium_two_key_runtime_probe_waits_for_libcudf_groupby_cardinality() {
    let (schema, spec) = fixed_width_sum_spec();
    let decision = aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial);
    let decision = select_partial_aggregate_strategy_decision(
        decision,
        None,
        &spec,
        &schema,
        204_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_libcudf_runtime_probe")
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 204_000), None);
}

#[test]
fn large_three_key_string_fixed_runtime_probe_is_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = q31_key_sum_spec();
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
        Some(vec!["county".to_string(), "quarter".to_string(), "year".to_string()])
    );
}

#[test]
fn three_key_runtime_probe_rejects_wide_input_and_all_fixed_width_keys() {
    let (schema, spec) = q31_key_sum_spec();
    let wide_input_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Partial),
        None,
        &spec,
        &schema,
        AggregateStrategyExecutionOptions::DEFAULT_EXACT_THREE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1,
        None,
        &QueryMetrics::default(),
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
        &QueryMetrics::default(),
    );
    assert_eq!(exact_grouped_key_cardinality_probe_candidate(&all_fixed_decision, &spec, &schema, 5_000_000), None);
}

#[test]
fn five_key_string_fixed_runtime_probe_is_exact_grouped_key_cardinality_probe_candidate() {
    let (schema, spec) = q76_key_sum_spec();
    let decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        6_900_000,
        None,
        &QueryMetrics::default(),
    );

    assert_eq!(
        exact_grouped_key_cardinality_probe_candidate(&decision, &spec, &schema, 6_900_000),
        Some(vec![
            "channel".to_string(),
            "col_name".to_string(),
            "year".to_string(),
            "quarter".to_string(),
            "category".to_string(),
        ])
    );
}

#[test]
fn five_key_runtime_probe_rejects_wide_input_and_wrong_string_mix() {
    let (schema, spec) = q76_key_sum_spec();
    let wide_input_decision = select_partial_aggregate_strategy_decision(
        aggregate_strategy_capability_decision(&spec, &schema, AggregateExecutionStage::Single),
        None,
        &spec,
        &schema,
        AggregateStrategyExecutionOptions::DEFAULT_EXACT_FIVE_KEY_CARDINALITY_PROBE_MAX_INPUT_ROWS + 1,
        None,
        &QueryMetrics::default(),
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
        &QueryMetrics::default(),
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
        &QueryMetrics::default(),
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
fn aggregate_strategy_options_change_encoded_key_partial_reduction_threshold() {
    let (schema, spec) = simple_string_sum_spec();
    let node_id = NodeId::from_index_for_internal_use(20);
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
    let node_id = NodeId::from_index_for_internal_use(21);
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
