use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{
    ExecutionContext, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey, LocalParquetFile, LocalParquetSource,
    MemoryControlPolicy, MemoryReservationRequest, MemorySpaceId, QueryMetrics, ReservationKind,
};
use datafusion_cudf::NATIVE_CUDF_DATAFUSION_METRIC_NAMES;

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
fn query_metrics_phase_c_memory_control_defaults_are_zero() {
    let metrics = QueryMetrics::default();

    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "unbounded");
    assert_eq!(metrics.reservation_requested_bytes, 0);
    assert_eq!(metrics.reservation_granted_bytes, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.source_chunks_planned, 0);
    assert_eq!(metrics.source_chunks_attempted, 0);
    assert_eq!(metrics.source_chunks_committed, 0);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.source_cache_hits, 0);
    assert_eq!(metrics.source_cache_misses, 0);
    assert_eq!(metrics.source_cache_retained_bytes, 0);
    assert_eq!(metrics.source_metadata_load_time, std::time::Duration::ZERO);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.final_merge_levels, 0);
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
    assert_eq!(metrics.reservation_events[1].memory_space, memory_space);
    assert_eq!(metrics.reservation_denied_count, 1);
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
fn bounded_context_rejects_source_cache_until_reservations_are_modeled() {
    let cache = Arc::new(GpuLocalParquetSourceCache::new(1024));
    let error = ExecutionContext::builder()
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(cache)
        .build()
        .expect_err("bounded source cache must stay rejected until retained bytes are reservation-accounted");

    assert_eq!(error.fact_value("reason"), Some("source_cache_requires_reservation_accounting"));
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
        "resident_dataframe_peak_bytes_observed",
        "allocator_high_water_bytes_observed",
        "reservation_requested_bytes",
        "reservation_granted_bytes",
        "reservation_denied_count",
        "retry_resource_exhausted_count",
        "retry_reservation_failure_count",
        "hash_join_build_count",
        "filtered_join_build_count",
        "hash_join_cache_hits",
        "hash_join_cache_misses",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
        "hash_join_hash_table_bytes",
        "filtered_join_hash_table_bytes",
        "filtered_join_probe_state_bytes",
        "mixed_join_size_data_bytes",
        "join_gather_map_bytes",
        "source_metadata_load_time",
        "source_read_queue_wait_time",
        "source_cache_lookup_time",
        "parquet_read_time",
        "operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ] {
        assert!(names.contains(&expected), "missing native DataFusion metric: {expected}");
    }
}
