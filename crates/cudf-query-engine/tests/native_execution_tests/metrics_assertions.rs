use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{
    BudgetedExecutionEvent, Expr, MemoryReservationEvent, MemoryReservationEventAction, MemorySpaceId, QueryMetrics,
    ReservationKind,
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
        assert_eq!(event.enforcement, "hard_memory_resource_limit");
        assert_eq!(event.status, "hard_enforced");
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
    assert_eq!(event.unresolved_contract.as_deref(), Some("libcudf_parquet_decoded_output_preflight_unavailable"));
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

pub(super) fn assert_budgeted_success_event_with_contract(
    metrics: &QueryMetrics,
    operation: &str,
    output_rows: u64,
    boundary: &str,
    unresolved_contract: Option<&str>,
) {
    let events = budgeted_events(metrics, operation);
    assert!(
        events.iter().any(|event| {
            event.output_rows == Some(output_rows)
                && event.boundary == boundary
                && event.unresolved_contract.as_deref() == unresolved_contract
        }),
        "missing budgeted execution event for {operation} with rows={output_rows}, boundary={boundary}, \
         unresolved_contract={unresolved_contract:?}: {events:?}"
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
        && event.enforcement == "hard_memory_resource_limit"
        && event.status == "hard_enforced"
        && event.unresolved_contract.as_deref() == Some("libcudf_sort_temporary_workspace_preflight_unavailable")
}

fn assert_representative_budgeted_contract(event: &BudgetedExecutionEvent) {
    match event.operation.as_str() {
        "cudf_query_engine.aggregate.cast_count"
        | "cudf_query_engine.aggregate.cast_reduction"
        | "cudf_query_engine.aggregate.finalize_avg"
        | "cudf_query_engine.projection.execute" => {
            assert!(
                matches!(
                    event.boundary.as_str(),
                    "chunk_operator"
                        | "final_aggregate_merge"
                        | "final_aggregate_projection"
                        | "post_aggregate_projection"
                ),
                "DataFrame expression event should record its execution boundary: {event:?}"
            );
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_dataframe_expression_temporary_workspace_and_transient_relocation_preflight_unavailable")
            );
        },
        "cudf_query_engine.aggregate.execute_partial" => {
            assert_eq!(event.boundary, "chunk_operator");
            assert_budget_event_high_water(event, event.output_bytes.or(event.retained_bytes));
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.global_aggregate.count_output" => {
            assert_eq!(event.boundary, "chunk_operator");
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_scalar_to_series_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.global_aggregate.null_output" => {
            assert!(matches!(event.boundary.as_str(), "chunk_operator" | "final_aggregate_merge"));
            assert_eq!(
                event.unresolved_contract.as_deref(),
                None,
                "fixed-width null output is directly budgeted; only scaled decimal nulls carry a lower-level gap"
            );
        },
        "cudf_query_engine.final_sort.prepare" => {
            assert_eq!(event.boundary, "final_sort");
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_sort_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.final_sort.materialize" => {
            assert_eq!(event.boundary, "final_sort");
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_sort_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.final_sort.topk_prepare" | "cudf_query_engine.final_sort.topk_materialize" => {
            assert!(matches!(event.boundary.as_str(), "final_sort_topk_chunk" | "final_sort_topk_merge"));
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_sort_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.local_parquet_chunk.zero_column_count" => {
            assert_eq!(event.boundary, "source_chunk_read");
            assert_eq!(
                event.unresolved_contract.as_deref(),
                Some("libcudf_scalar_to_series_temporary_workspace_preflight_unavailable")
            );
        },
        "cudf_query_engine.local_parquet_chunk.read" => {
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

pub(super) fn assert_reservation_kind_subsequence(metrics: &QueryMetrics, expected: &[ReservationKind]) {
    assert_reservation_events_memory_space(metrics, MemorySpaceId::Device { device_ordinal: 0 });
    let mut events = metrics.reservation_events.iter();
    for expected_kind in expected {
        events
            .find(|event| event.reservation_kind == *expected_kind)
            .unwrap_or_else(|| panic!("missing ordered reservation event {expected_kind:?}"));
    }
}

pub(super) fn assert_granted_reservation_trace(metrics: &QueryMetrics, expected: &[(ReservationKind, &'static str)]) {
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_reservation_events_memory_space(metrics, MemorySpaceId::Device { device_ordinal: 0 });
    let actual = metrics
        .reservation_events
        .iter()
        .filter(|event| is_ordered_reservation_trace_event(event))
        .map(|event| (event.reservation_kind, event.memory_boundary.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

pub(super) fn assert_reservation_events_memory_space(metrics: &QueryMetrics, expected: MemorySpaceId) {
    assert!(!metrics.reservation_events.is_empty(), "expected reservation events to be recorded");
    for event in &metrics.reservation_events {
        assert_eq!(event.memory_space, expected);
    }
}

pub(super) fn granted_reservation_requested_bytes(
    metrics: &QueryMetrics,
    kind: ReservationKind,
    memory_boundary: &str,
) -> u64 {
    metrics
        .reservation_events
        .iter()
        .find(|event| {
            is_granted_admission_event(event)
                && event.reservation_kind == kind
                && event.memory_boundary == memory_boundary
        })
        .unwrap_or_else(|| panic!("missing granted reservation event {kind:?}/{memory_boundary}"))
        .requested_bytes
}

pub(super) fn retained_bytes_before_grouped_aggregate_source_read(metrics: &QueryMetrics) -> u64 {
    let (source_index, retained_boundary) = grouped_aggregate_source_read_index(metrics);
    metrics.reservation_events[..source_index]
        .iter()
        .rev()
        .find(|event| is_granted_admission_event(event) && event.memory_boundary == retained_boundary)
        .expect("join subtree output should be retained before grouped source read")
        .requested_bytes
}

pub(super) fn grouped_aggregate_chunk_budget_window(metrics: &QueryMetrics) -> (u64, u64, u64) {
    let (source_index, _) = grouped_aggregate_source_read_index(metrics);
    let retained_join_subtree_bytes = retained_bytes_before_grouped_aggregate_source_read(metrics);
    let source_read_bytes = metrics.reservation_events[source_index].requested_bytes;
    let operator_bytes = metrics.reservation_events[source_index + 1..]
        .iter()
        .find(|event| {
            is_granted_admission_event(event)
                && matches!(event.memory_boundary.as_str(), "chunk_operator" | "compiled_aggregate")
        })
        .expect("mixed aggregate input metrics should include grouped aggregate operator")
        .requested_bytes;
    (retained_join_subtree_bytes, source_read_bytes, operator_bytes)
}

fn grouped_aggregate_source_read_index(metrics: &QueryMetrics) -> (usize, &'static str) {
    if let Some(source_index) = metrics
        .reservation_events
        .iter()
        .position(|event| is_granted_admission_event(event) && event.memory_boundary == "source_chunk_read")
    {
        return (source_index, "join_pipeline_projection_output");
    }

    let aggregate_index = metrics
        .reservation_events
        .iter()
        .position(|event| is_granted_admission_event(event) && event.memory_boundary == "compiled_aggregate")
        .expect("compiled mixed aggregate input metrics should include grouped aggregate operator");
    let source_index = metrics.reservation_events[..aggregate_index]
        .iter()
        .rposition(|event| is_granted_admission_event(event) && event.memory_boundary == "join_source_read")
        .expect("compiled mixed aggregate input metrics should include grouped source read");
    metrics.reservation_events[..source_index]
        .iter()
        .rposition(|event| is_granted_admission_event(event) && event.memory_boundary == "compiled_projection_output")
        .expect("compiled mixed aggregate input metrics should retain a join-subtree projection output");
    (source_index, "compiled_projection_output")
}

pub(super) fn granted_reservation_requested_bytes_for_boundary(
    metrics: &QueryMetrics,
    kind: ReservationKind,
    memory_boundary: &str,
) -> Vec<u64> {
    metrics
        .reservation_events
        .iter()
        .filter(|event| {
            is_granted_admission_event(event)
                && event.reservation_kind == kind
                && event.memory_boundary == memory_boundary
        })
        .map(|event| event.requested_bytes)
        .collect()
}

fn is_granted_admission_event(event: &MemoryReservationEvent) -> bool {
    event.outcome.as_str() == "granted"
        && matches!(event.action, MemoryReservationEventAction::Reserve | MemoryReservationEventAction::Grow)
}

fn is_ordered_reservation_trace_event(event: &MemoryReservationEvent) -> bool {
    is_granted_admission_event(event)
        || (event.outcome.as_str() == "applied" && event.action == MemoryReservationEventAction::Transfer)
}

pub(super) fn assert_bounded_mixed_left_only_join_metrics(metrics: &QueryMetrics, join_type: &str, output_rows: u64) {
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::MixedSizeData, "join_mixed_size_data"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
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

pub(super) fn assert_zero_byte_gather_map_reservation(metrics: &QueryMetrics) {
    let gather_map = metrics
        .reservation_events
        .iter()
        .find(|event| event.reservation_kind == ReservationKind::GatherMap)
        .expect("filtered zero-output join should still record a gather-map reservation boundary");
    assert_eq!(gather_map.requested_bytes, 0);
    assert_eq!(gather_map.granted_bytes, 0);
}
