use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_nexus::NATIVE_CUDF_DATAFUSION_METRIC_NAMES;
use nexus_query_engine::{
    ExecutionContext, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey, LocalParquetFile, LocalParquetSource,
    MemoryControlPolicy, MemoryReservationEventAction, MemoryReservationRequest, MemorySpaceId, QueryMetrics,
    QueryPlanBuilder, ReservationKind, SourceChunk, SourceChunkPlanningMode, SourceNode,
};

#[test]
fn query_metrics_records_peak_device_memory_observation() {
    let mut metrics = QueryMetrics::default();

    metrics.record_device_memory_observation(128);
    metrics.record_device_memory_observation(64);
    metrics.record_device_memory_observation(256);

    assert_eq!(metrics.peak_device_memory_bytes, Some(256));
    assert_eq!(metrics.resident_dataframe_peak_bytes, Some(256));
    assert_eq!(metrics.allocator_high_water_bytes, None);
}

#[test]
fn query_metrics_memory_control_defaults_are_unbounded() {
    let metrics = QueryMetrics::default();

    assert_eq!(metrics.memory_control_policy.as_str(), "unbounded");
    assert_eq!(metrics.reservation_requested_bytes, 0);
    assert_eq!(metrics.reservation_granted_bytes, 0);
    assert_eq!(metrics.reservation_live_peak_bytes, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.source_chunks_planned, 0);
    assert_eq!(metrics.source_chunks_attempted, 0);
    assert_eq!(metrics.source_chunks_committed, 0);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.source_cache_hits, 0);
    assert_eq!(metrics.source_cache_misses, 0);
    assert_eq!(metrics.source_cache_retained_bytes, 0);
    assert_eq!(metrics.source_metadata_load_time, std::time::Duration::ZERO);
    assert_eq!(metrics.source_fetch_bytes, 0);
    assert_eq!(metrics.source_fetch_count, 0);
    assert_eq!(metrics.host_fetch_budget_wait_time, std::time::Duration::ZERO);
    assert_eq!(metrics.source_fetch_concurrency_wait_time, std::time::Duration::ZERO);
    assert_eq!(metrics.source_fetch_time, std::time::Duration::ZERO);
    assert_eq!(metrics.source_decode_time, std::time::Duration::ZERO);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.partial_aggregate_input_rows, 0);
    assert_eq!(metrics.partial_aggregate_retained_bytes, 0);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.final_merge_levels, 0);
    assert_eq!(metrics.final_merge_max_fan_in, 0);
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 0);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 0);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 0);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_before, 0);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_after, 0);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_filter_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 0);
}

#[test]
fn query_metrics_count_retries_attempts_and_commits_separately() {
    let mut metrics = QueryMetrics::default();

    metrics.record_source_chunk_planned(4);
    metrics.record_source_chunk_attempt();
    metrics.record_source_chunk_retry();
    metrics.record_source_chunk_attempt();
    metrics.record_source_chunk_commit(2);
    metrics.record_retry_reservation_failure();
    metrics.record_retry_resource_exhausted();

    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.row_groups_planned, 4);
    assert_eq!(metrics.source_chunks_attempted, 2);
    assert_eq!(metrics.source_chunks_retried, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_committed, 2);
    assert_eq!(metrics.retry_count, 2);
    assert_eq!(metrics.retry_reservation_failure_count, 1);
    assert_eq!(metrics.retry_resource_exhausted_count, 1);
}

#[test]
fn query_metrics_reservation_events_record_memory_space() {
    let mut metrics = QueryMetrics::default();
    let memory_space = MemorySpaceId::Device { device_ordinal: 1 };
    let granted = MemoryReservationRequest::new(ReservationKind::SourceRead, 1024, 4096)
        .with_memory_space(memory_space)
        .with_memory_boundary("source_chunk_read");
    let denied = MemoryReservationRequest::new(ReservationKind::JoinOutput, 8192, 4096)
        .with_memory_space(memory_space)
        .with_memory_boundary("join_output");

    metrics.record_reservation_granted_event(&granted, 1024);
    metrics.record_reservation_denied_event(&denied, 4096);

    assert_eq!(metrics.reservation_events.len(), 2);
    assert_eq!(metrics.reservation_events[0].memory_space, memory_space);
    assert_eq!(metrics.reservation_events[0].memory_space.to_string(), "GPU:1");
    assert_eq!(metrics.reservation_events[0].action, MemoryReservationEventAction::Reserve);
    assert_eq!(metrics.reservation_events[0].charged_delta_bytes, 1024);
    assert_eq!(metrics.reservation_events[0].current_reserved_bytes, 1024);
    assert_eq!(metrics.reservation_events[0].budget_bytes, 4096);
    assert_eq!(metrics.reservation_events[1].memory_space, memory_space);
    assert_eq!(metrics.reservation_events[1].charged_delta_bytes, 0);
    assert_eq!(metrics.reservation_events[1].current_reserved_bytes, 0);
    assert_eq!(metrics.reservation_denied_count, 1);
}

#[test]
fn query_metrics_reservation_timeline_records_live_ledger_deltas() {
    let mut metrics = QueryMetrics::default();
    metrics.record_compiled_pipeline_task_execution("t2", "source_read");
    metrics.record_compiled_pipeline_task_execution("t3", "join_probe");
    let request = MemoryReservationRequest::new(ReservationKind::SourceRead, 60, 100)
        .with_memory_boundary("source_read")
        .with_compiled_task_id("t2")
        .with_resident_resource_id("r7");
    let grow = MemoryReservationRequest::new(ReservationKind::SourceRead, 20, 100)
        .with_memory_boundary("source_read")
        .with_compiled_task_id("t2")
        .with_resident_resource_id("r7");
    let denied = MemoryReservationRequest::new(ReservationKind::ProbeSide, 40, 100)
        .with_memory_boundary("join_probe_side")
        .with_compiled_task_id("t3")
        .with_resident_resource_id("r7");

    metrics.record_reservation_granted_timeline_event(&request, MemoryReservationEventAction::Reserve, 60, 60, 60);
    metrics.record_reservation_granted_timeline_event(&grow, MemoryReservationEventAction::Grow, 20, 20, 80);
    metrics.record_reservation_timeline_event(nexus_query_engine::MemoryReservationEvent::from_request(
        &grow,
        nexus_query_engine::MemoryReservationEventOutcome::Applied,
        MemoryReservationEventAction::Shrink,
        0,
        -30,
        50,
    ));
    metrics.record_reservation_denied_timeline_event(&denied, MemoryReservationEventAction::Reserve, 50, 50);

    let actions = metrics.reservation_events.iter().map(|event| event.action.as_str()).collect::<Vec<_>>();
    assert_eq!(actions, ["reserve", "grow", "shrink", "reserve"]);
    let current = metrics.reservation_events.iter().map(|event| event.current_reserved_bytes).collect::<Vec<_>>();
    assert_eq!(current, [60, 80, 50, 50]);
    let deltas = metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
    assert_eq!(deltas, [60, 20, -30, 0]);
    assert_eq!(metrics.reservation_requested_bytes, 120);
    assert_eq!(metrics.reservation_granted_bytes, 80);
    assert_eq!(metrics.reservation_live_peak_bytes, 80);
    assert_eq!(metrics.reservation_denied_count, 1);
    assert_eq!(metrics.reservation_events[0].compiled_task_id.as_deref(), Some("t2"));
    assert_eq!(metrics.reservation_events[0].resident_resource_id.as_deref(), Some("r7"));
    assert_eq!(metrics.reservation_events[3].compiled_task_id.as_deref(), Some("t3"));
    assert_eq!(metrics.max_task_reservation_bytes, 60);
    assert_eq!(metrics.task_operator_metrics["source_read"].task_count, 1);
    assert_eq!(metrics.task_operator_metrics["source_read"].reservation_requested_bytes, 80);
    assert_eq!(metrics.task_operator_metrics["source_read"].reservation_granted_bytes, 80);
    assert_eq!(metrics.task_operator_metrics["source_read"].reservation_released_bytes, 30);
    assert_eq!(metrics.task_operator_metrics["source_read"].reservation_live_peak_bytes, 80);
    assert_eq!(metrics.task_operator_metrics["join_probe"].task_count, 1);
    assert_eq!(metrics.task_operator_metrics["join_probe"].reservation_requested_bytes, 40);
    assert_eq!(metrics.task_operator_metrics["join_probe"].reservation_granted_bytes, 0);
    assert_eq!(metrics.task_operator_metrics["join_probe"].reservation_live_peak_bytes, 0);
    assert_eq!(metrics.task_runtime_metrics["t2"].operator_kind, "source_read");
    assert_eq!(metrics.task_runtime_metrics["t2"].reservation_requested_bytes, 80);
    assert_eq!(metrics.task_runtime_metrics["t2"].reservation_granted_bytes, 80);
    assert_eq!(metrics.task_runtime_metrics["t2"].reservation_released_bytes, 30);
    assert_eq!(metrics.task_runtime_metrics["t2"].reservation_live_peak_bytes, 80);
}

#[test]
fn query_metrics_records_task_elapsed_by_runtime_and_operator_kind() {
    let mut metrics = QueryMetrics::default();

    metrics.record_compiled_pipeline_task_execution("t1", "join_build");
    metrics.record_task_elapsed("t1", "join_build", std::time::Duration::from_millis(12));
    metrics.record_task_elapsed("t1", "join_build", std::time::Duration::from_millis(3));
    metrics.record_compiled_pipeline_task_execution("t2", "join_build");
    metrics.record_task_elapsed("t2", "join_build", std::time::Duration::from_millis(5));
    metrics.record_compiled_pipeline_task_execution("t3", "source_read");
    metrics.record_task_elapsed("t3", "source_read", std::time::Duration::from_millis(7));

    assert_eq!(metrics.task_runtime_metrics["t1"].elapsed_time, std::time::Duration::from_millis(15));
    assert_eq!(metrics.task_runtime_metrics["t2"].elapsed_time, std::time::Duration::from_millis(5));
    assert_eq!(metrics.task_runtime_metrics["t3"].elapsed_time, std::time::Duration::from_millis(7));
    assert_eq!(metrics.task_operator_metrics["join_build"].task_count, 2);
    assert_eq!(metrics.task_operator_metrics["join_build"].elapsed_time, std::time::Duration::from_millis(20));
    assert_eq!(metrics.task_operator_metrics["source_read"].elapsed_time, std::time::Duration::from_millis(7));
}

#[test]
fn query_metrics_reservation_events_retain_legacy_bounded_task_context() {
    let mut metrics = QueryMetrics::default();
    let request = MemoryReservationRequest::new(ReservationKind::SourceRead, 80, 100)
        .with_memory_boundary("source_chunk_read")
        .with_node_id("7")
        .with_stage("source_chunk_read")
        .with_chunk_index(3)
        .with_file_index(2)
        .with_path("/tmp/lineitem.parquet")
        .with_row_groups(vec![10, 11])
        .with_attempt(2, 4)
        .with_compiled_task_id("legacy_bounded.source_read.node7.file2.chunk3.attempt2")
        .with_source_read_contract("conservative_estimate", Some(64), Some(80));

    metrics.record_reservation_denied_timeline_event(&request, MemoryReservationEventAction::Reserve, 20, 60);

    let event = &metrics.reservation_events[0];
    assert_eq!(event.node_id.as_deref(), Some("7"));
    assert_eq!(event.stage.as_deref(), Some("source_chunk_read"));
    assert_eq!(event.chunk_index, Some(3));
    assert_eq!(event.file_index, Some(2));
    assert_eq!(event.path.as_deref(), Some("/tmp/lineitem.parquet"));
    assert_eq!(event.row_groups.as_deref(), Some([10, 11].as_slice()));
    assert_eq!(event.attempt, Some(2));
    assert_eq!(event.max_attempts, Some(4));
    assert_eq!(event.source_read_contract_quality.as_deref(), Some("conservative_estimate"));
    assert_eq!(event.source_read_planning_bytes, Some(64));
    assert_eq!(event.source_read_estimated_bytes, Some(80));
    assert_eq!(metrics.task_operator_metrics["source_read"].task_count, 1);
    assert_eq!(metrics.task_operator_metrics["source_read"].reservation_requested_bytes, 80);
    assert!(event.to_stable_debug_row().contains("row_groups=10,11"));
    assert!(event.to_stable_debug_row().contains("remaining_bytes=40"));
}

#[test]
fn local_parquet_source_cache_key_tracks_scan_contract() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int32, true)]));
    let file = LocalParquetFile::new("/tmp/source-cache-key.parquet")
        .with_stable_id("file-id")
        .with_file_size_bytes(128)
        .with_row_groups(vec![0, 1])
        .with_row_group_rows(vec![10, 20]);
    let source = LocalParquetSource::from_files(vec![file], Arc::clone(&schema), vec!["payload".to_string()]);
    let limited = source.clone().with_scan_limit(Some(10));

    assert_ne!(
        GpuLocalParquetSourceCacheKey::from_source(&source),
        GpuLocalParquetSourceCacheKey::from_source(&limited),
        "source cache key must distinguish source-level scan limits"
    );
}

#[test]
fn local_parquet_source_cache_key_tracks_bounded_chunk_contract() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int32, true)]));
    let file = LocalParquetFile::new("/tmp/source-cache-chunk-key.parquet")
        .with_stable_id("file-id")
        .with_file_size_bytes(512)
        .with_row_groups(vec![0, 1])
        .with_row_group_rows(vec![10, 20]);
    let source = LocalParquetSource::from_files(vec![file], Arc::clone(&schema), vec!["payload".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_node_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let first = SourceChunk {
        source_node_id,
        chunk_index: 0,
        file_index: 0,
        location: source.files[0].location.clone(),
        resolved_source: source.files[0].resolved_source.clone(),
        path: "/tmp/source-cache-chunk-key.parquet".into(),
        stable_id: "file-id".into(),
        row_group_indexes: vec![0],
        reads_all_row_groups: false,
        known_file_bytes: Some(512),
        known_row_group_bytes: Some(128),
        known_projected_compressed_bytes: Some(64),
        known_projected_uncompressed_bytes: Some(128),
        known_pure_filter_compressed_bytes: Some(0),
        known_pure_filter_uncompressed_bytes: Some(0),
        estimated_chunk_bytes: Some(128),
        estimated_source_read_bytes: Some(128),
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: Some(vec![0, 1]),
        source_file_row_group_bytes: Some(vec![128, 384]),
        source_file_projected_compressed_bytes: Some(vec![64, 192]),
        source_file_projected_uncompressed_bytes: Some(vec![128, 384]),
        source_file_pure_filter_compressed_bytes: Some(vec![0, 0]),
        source_file_pure_filter_uncompressed_bytes: Some(vec![0, 0]),
        source_file_row_group_rows: Some(vec![10, 20]),
        read_columns: vec!["payload".to_string()],
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(10),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: None,
        source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test_policy".to_string(),
    };
    let mut second = first.clone();
    second.chunk_index = 1;
    second.row_group_indexes = vec![1];
    second.known_row_count = Some(20);

    assert_ne!(
        GpuLocalParquetSourceCacheKey::from_source_chunk(&source, &first),
        GpuLocalParquetSourceCacheKey::from_source_chunk(&source, &second),
        "source cache key must distinguish bounded source chunks"
    );
}

#[test]
fn local_parquet_source_cache_key_ignores_bounded_planner_identity() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int32, true)]));
    let file = LocalParquetFile::new("/tmp/source-cache-planner-identity.parquet")
        .with_stable_id("file-id")
        .with_file_size_bytes(512)
        .with_row_groups(vec![0, 1])
        .with_row_group_rows(vec![10, 20]);
    let source = LocalParquetSource::from_files(vec![file], Arc::clone(&schema), vec!["payload".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let first_source_node_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let second_source_node_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let first = SourceChunk {
        source_node_id: first_source_node_id,
        chunk_index: 0,
        file_index: 0,
        location: source.files[0].location.clone(),
        resolved_source: source.files[0].resolved_source.clone(),
        path: "/tmp/source-cache-planner-identity.parquet".into(),
        stable_id: "file-id".into(),
        row_group_indexes: vec![0],
        reads_all_row_groups: false,
        known_file_bytes: Some(512),
        known_row_group_bytes: Some(128),
        known_projected_compressed_bytes: Some(64),
        known_projected_uncompressed_bytes: Some(128),
        known_pure_filter_compressed_bytes: Some(0),
        known_pure_filter_uncompressed_bytes: Some(0),
        estimated_chunk_bytes: Some(128),
        estimated_source_read_bytes: Some(128),
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: Some(vec![0, 1]),
        source_file_row_group_bytes: Some(vec![128, 384]),
        source_file_projected_compressed_bytes: Some(vec![64, 192]),
        source_file_projected_uncompressed_bytes: Some(vec![128, 384]),
        source_file_pure_filter_compressed_bytes: Some(vec![0, 0]),
        source_file_pure_filter_uncompressed_bytes: Some(vec![0, 0]),
        source_file_row_group_rows: Some(vec![10, 20]),
        read_columns: vec!["payload".to_string()],
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(10),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: None,
        source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test_policy".to_string(),
    };
    let mut same_contract_different_plan_identity = first.clone();
    same_contract_different_plan_identity.source_node_id = second_source_node_id;
    same_contract_different_plan_identity.chunk_index = 7;

    assert_eq!(
        GpuLocalParquetSourceCacheKey::from_source_chunk(&source, &first),
        GpuLocalParquetSourceCacheKey::from_source_chunk(&source, &same_contract_different_plan_identity),
        "bounded source cache keys must be based on the reusable scan contract, not local planner ids"
    );
}

#[test]
fn bounded_context_accepts_source_cache_with_reservation_accounting() {
    let cache = Arc::new(GpuLocalParquetSourceCache::new(1024));
    let context = ExecutionContext::builder()
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(cache)
        .build()
        .expect("bounded source cache retained bytes are represented in the reservation ledger");

    assert!(context.local_parquet_source_cache().is_some());
}

#[test]
fn native_datafusion_metric_names_cover_bounded_runtime_contract() {
    let names = NATIVE_CUDF_DATAFUSION_METRIC_NAMES;
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "native DataFusion metric names must be unique");

    for expected in [
        "host_to_device_uploads",
        "gpu_resident_input_handoffs",
        "host_to_device_uploaded_bytes",
        "device_to_host_materializations",
        "device_to_host_materialized_bytes",
        "source_read_boundary_direct_device",
        "build_materialization_boundary_query_scoped",
        "build_materialization_boundary_durable",
        "probe_execution_boundary_reusable",
        "result_materialization_boundary_explicit",
        "source_chunks_planned",
        "source_chunks_attempted",
        "source_chunks_committed",
        "source_chunks_retried",
        "source_read_prefetches_started",
        "source_read_prefetches_completed",
        "source_read_prefetches_skipped",
        "source_read_prefetches_abandoned",
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "row_groups_planned",
        "row_groups_committed",
        "partial_aggregate_input_rows",
        "partial_aggregate_retained_bytes",
        "final_merge_max_fan_in",
        "resident_dataframe_peak_bytes_observed",
        "allocator_high_water_bytes_observed",
        "reservation_requested_bytes",
        "reservation_granted_bytes",
        "reservation_live_peak_bytes",
        "reservation_denied_count",
        "max_task_reservation_bytes",
        "max_task_observed_peak_bytes",
        "max_task_observed_peak_bytes_observed",
        "max_task_retained_bytes",
        "max_task_retained_bytes_observed",
        "native_graph_runtime_rows",
        "native_task_operator_rows",
        "native_task_runtime_rows",
        "native_dynamic_filter_runtime_rows",
        "native_source_read_runtime_rows",
        "compiled_pipeline_attempts",
        "compiled_pipeline_tasks_executed",
        "compiled_pipeline_host_arrow_sinks",
        "retry_resource_exhausted_count",
        "retry_reservation_failure_count",
        "adaptive_retry_count",
        "adaptive_spill_count",
        "adaptive_downgrade_count",
        "adaptive_memory_history_lookup_count",
        "adaptive_memory_history_hit_count",
        "adaptive_memory_history_miss_count",
        "dynamic_filter_produced_count",
        "dynamic_filter_skipped_count",
        "dynamic_filter_applied_count",
        "dynamic_filter_row_groups_pruned",
        "dynamic_filter_source_bytes_pruned",
        "dynamic_filter_reader_predicate_rows_before",
        "dynamic_filter_reader_predicate_rows_after",
        "dynamic_filter_reader_predicate_rows_filtered",
        "dynamic_filter_post_decode_rows_before",
        "dynamic_filter_post_decode_rows_after",
        "dynamic_filter_post_decode_rows_filtered",
        "dynamic_filter_source_post_decode_rows_filtered",
        "dynamic_filter_filter_post_decode_rows_filtered",
        "dynamic_filter_join_probe_post_decode_rows_filtered",
        "hash_join_build_count",
        "filtered_join_build_count",
        "one_shot_join_count",
        "hash_join_cache_hits",
        "hash_join_cache_misses",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
        "hash_join_hash_table_bytes",
        "filtered_join_hash_table_bytes",
        "filtered_join_probe_state_bytes",
        "mixed_join_size_data_bytes",
        "one_shot_join_index_bytes",
        "one_shot_join_materialization_bytes",
        "one_shot_join_output_bytes",
        "join_gather_map_bytes",
        "native_aggregate_strategy_rows",
        "native_join_strategy_rows",
        "source_metadata_load_time",
        "source_read_queue_wait_time",
        "source_cache_lookup_time",
        "source_fetch_bytes",
        "source_fetch_count",
        "host_fetch_budget_wait_time",
        "source_fetch_concurrency_wait_time",
        "source_fetch_time",
        "source_decode_time",
        "parquet_read_time",
        "operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ] {
        assert!(names.contains(&expected), "missing native DataFusion metric: {expected}");
    }
}
