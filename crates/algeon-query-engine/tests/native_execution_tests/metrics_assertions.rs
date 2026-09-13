use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use query_engine::{
    expr::Expr,
    metrics::{BudgetedExecutionEvent, QueryMetrics},
};

pub(super) fn budgeted_events<'a>(metrics: &'a QueryMetrics, operation: &str) -> Vec<&'a BudgetedExecutionEvent> {
    metrics.budgeted_execution_events.iter().filter(|event| event.operation == operation).collect()
}

pub(super) fn assert_budgeted_success_event(metrics: &QueryMetrics, operation: &str) {
    let events = budgeted_events(metrics, operation);
    assert!(!events.is_empty(), "missing budgeted execution event for {operation}");
    for event in events {
        assert!(!event.boundary.is_empty(), "budgeted event should record boundary: {event:?}");
        assert!(event.budget_bytes > 0, "budgeted event should record budget bytes: {event:?}");
        assert_eq!(event.enforcement, "query_domain_member");
        assert_eq!(event.status, "reservation_enforced");
        assert!(event.retained_bytes.is_some(), "budgeted event should record retained bytes: {event:?}");
        assert!(event.output_bytes.is_some(), "budgeted event should record output bytes: {event:?}");
        assert!(event.output_rows.is_some(), "budgeted event should record output rows: {event:?}");
        assert_representative_budgeted_contract(event);
    }
}

pub(super) fn assert_budget_event_high_water(event: &BudgetedExecutionEvent, lower_bound_bytes: Option<u64>) {
    let high_water =
        event.allocator_high_water_bytes.expect("budgeted execution event should record allocator high-water bytes");
    assert!(high_water > 0, "allocator high-water should be positive: {event:?}");
    assert!(
        high_water <= event.budget_bytes,
        "allocator high-water should stay within the hard budget: high_water={high_water}, event={event:?}",
    );
    if let Some(lower_bound_bytes) = lower_bound_bytes {
        assert!(
            high_water >= lower_bound_bytes,
            "allocator high-water should cover retained/output bytes: high_water={high_water}, lower_bound={lower_bound_bytes}, event={event:?}",
        );
    }
}

pub(super) fn assert_source_budget_event_context(
    event: &BudgetedExecutionEvent,
    read_columns: &str,
    projected_columns: &str,
) {
    assert_eq!(event.context_facts.get("source_kind").map(String::as_str), Some("data"));
    assert_eq!(event.context_facts.get("source_read_mode").map(String::as_str), Some("execution_frame"));
    assert_eq!(event.context_facts.get("read_columns").map(String::as_str), Some(read_columns));
    assert_eq!(event.context_facts.get("projected_columns").map(String::as_str), Some(projected_columns));
    assert!(event.context_facts.contains_key("path"), "source event should record source path: {event:?}");
    assert!(event.output_rows.is_some_and(|rows| rows > 0), "source event should record decoded rows: {event:?}");
    assert!(
        event.retained_bytes.is_some_and(|bytes| bytes > 0),
        "source event should record retained bytes: {event:?}"
    );
    assert_eq!(event.retained_bytes, event.output_bytes);
    assert_budget_event_high_water(event, event.output_bytes);
}

pub(super) fn assert_budgeted_success_event_with_rows(metrics: &QueryMetrics, operation: &str, output_rows: u64) {
    let events = budgeted_events(metrics, operation);
    assert!(
        events.iter().any(|event| event.output_rows == Some(output_rows)),
        "missing budgeted execution event for {operation} with {output_rows} rows: {events:?}"
    );
    assert_budgeted_success_event(metrics, operation);
}

pub(super) fn assert_budgeted_success_event_with_boundary(
    metrics: &QueryMetrics,
    operation: &str,
    output_rows: u64,
    boundary: &str,
) {
    let events = budgeted_events(metrics, operation);
    assert!(
        events.iter().any(|event| event.output_rows == Some(output_rows) && event.boundary == boundary),
        "missing budgeted execution event for {operation} with rows={output_rows}, boundary={boundary}: {events:?}"
    );
    assert_budgeted_success_event(metrics, operation);
}

pub(super) fn assert_final_sort_prepare_budget_event(
    metrics: &QueryMetrics,
    operation: &str,
    boundary: &str,
    retained_bytes: Option<u64>,
) {
    let events = budgeted_events(metrics, operation);
    assert!(
        events.iter().any(|event| {
            final_sort_budget_event_has_common_contract(event, boundary)
                && retained_bytes.map_or(event.retained_bytes.is_some_and(|bytes| bytes > 0), |bytes| {
                    event.retained_bytes == Some(bytes)
                })
                && event.allocator_high_water_bytes.is_some_and(|bytes| {
                    bytes > 0 && bytes <= event.budget_bytes && bytes >= event.retained_bytes.unwrap_or(0)
                })
                && event.output_bytes.is_none()
                && event.output_rows.is_none()
        }),
        "missing final-sort prepare event for operation={operation}, boundary={boundary}, \
         retained_bytes={retained_bytes:?}: {events:?}"
    );
}

pub(super) fn assert_final_sort_materialize_budget_event(
    metrics: &QueryMetrics,
    operation: &str,
    boundary: &str,
    output_bytes: Option<u64>,
) {
    let events = budgeted_events(metrics, operation);
    assert!(
        events.iter().any(|event| {
            final_sort_budget_event_has_common_contract(event, boundary)
                && event.retained_bytes.is_some_and(|bytes| bytes > 0)
                && output_bytes.map_or(event.output_bytes.is_some_and(|bytes| bytes > 0), |bytes| {
                    event.output_bytes == Some(bytes)
                })
                && event.allocator_high_water_bytes.is_some_and(|bytes| {
                    bytes > 0 && bytes <= event.budget_bytes && bytes >= event.output_bytes.unwrap_or(0)
                })
                && event.output_rows.is_none()
        }),
        "missing final-sort materialize event for operation={operation}, boundary={boundary}, \
         output_bytes={output_bytes:?}: {events:?}"
    );
}

fn final_sort_budget_event_has_common_contract(event: &BudgetedExecutionEvent, boundary: &str) -> bool {
    event.boundary == boundary
        && event.budget_bytes > 0
        && event.enforcement == "query_domain_member"
        && event.status == "reservation_enforced"
}

fn assert_representative_budgeted_contract(event: &BudgetedExecutionEvent) {
    match event.operation.as_str() {
        "query_engine.aggregate.cast_count"
        | "query_engine.aggregate.cast_reduction"
        | "query_engine.aggregate.finalize_avg"
        | "query_engine.projection.execute" => {
            assert!(
                matches!(
                    event.boundary.as_str(),
                    "chunk_operator"
                        | "final_aggregate_merge"
                        | "final_aggregate_projection"
                        | "post_aggregate_projection"
                        | "source_adjacent_partitioned_partial_aggregate"
                ),
                "DataFrame expression event should record its execution boundary: {event:?}"
            );
        },
        "query_engine.aggregate.execute_partial" => {
            assert!(matches!(
                event.boundary.as_str(),
                "chunk_operator"
                    | "source_adjacent_partitioned_partial_aggregate"
                    | "compiled_aggregate_partial_state_work"
            ));
            assert_budget_event_high_water(event, event.output_bytes.or(event.retained_bytes));
        },
        "query_engine.global_aggregate.count_output" => {
            assert_eq!(event.boundary, "chunk_operator");
        },
        "query_engine.global_aggregate.null_output" => {
            assert!(matches!(event.boundary.as_str(), "chunk_operator" | "final_aggregate_merge"), "{event:?}");
        },
        "query_engine.final_sort.prepare" => {
            assert_eq!(event.boundary, "final_sort");
        },
        "query_engine.final_sort.materialize" => {
            assert_eq!(event.boundary, "final_sort");
        },
        "query_engine.final_sort.topk_prepare" | "query_engine.final_sort.topk_materialize" => {
            assert!(matches!(
                event.boundary.as_str(),
                "final_sort_topk_chunk" | "final_sort_topk_stream_chunk" | "final_sort_topk_merge"
            ));
        },
        "query_engine.local_parquet_chunk.zero_column_count" => {
            assert!(matches!(event.boundary.as_str(), "source_chunk_read" | "source_chunk_prefetch"));
        },
        "query_engine.local_parquet_chunk.read" => {
            assert!(matches!(event.boundary.as_str(), "source_chunk_read" | "join_source_read"));
            assert_source_budget_event_context(
                event,
                event.context_facts["read_columns"].as_str(),
                event.context_facts["projected_columns"].as_str(),
            );
        },
        _ => {},
    }
}

pub(super) fn wide_final_sort_schema(column_count: usize) -> Arc<Schema> {
    assert!(column_count >= 2);
    let mut fields = Vec::with_capacity(column_count);
    fields.push(Field::new("key", DataType::Int32, false));
    fields.extend((0..(column_count - 1)).map(|index| Field::new(format!("sum_m{index:02}"), DataType::Int64, true)));
    Arc::new(Schema::new(fields))
}

pub(super) fn wide_final_sort_projection(column_count: usize) -> Vec<Expr> {
    assert!(column_count >= 2);
    let mut projection = Vec::with_capacity(column_count);
    projection.push(Expr::column("key"));
    projection.extend((0..(column_count - 1)).map(|index| Expr::column("sum_m00").alias(format!("sum_m{index:02}"))));
    projection
}

pub(super) fn assert_partitioned_final_merge_strategy(metrics: &QueryMetrics) {
    let decision = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge")
                && decision.facts.get("selected_strategy").map(String::as_str) == Some("partitioned_merge")
        })
        .expect("final merge should record partitioned merge strategy");
    assert_ne!(
        decision.facts.get("partitioned_merge_partition_count").map(String::as_str),
        Some("0"),
        "partitioned final merge should record a positive partition count"
    );
}

pub(super) fn assert_bounded_mixed_left_only_join_metrics(metrics: &QueryMetrics, join_type: &str, output_rows: u64) {
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, join_type);
    assert_eq!(join_metrics.execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert_eq!(join_metrics.key_count, 1);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert_eq!(join_metrics.mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert_eq!(join_metrics.output_rows, output_rows);
    assert!(join_metrics.gather_map_bytes > 0);
}
