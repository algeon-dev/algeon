use cudf_query_engine::QueryMetrics;
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
        "row_groups_planned",
        "row_groups_committed",
        "reservation_requested_bytes",
        "reservation_granted_bytes",
        "reservation_denied_count",
        "retry_resource_exhausted_count",
        "retry_reservation_failure_count",
        "hash_join_build_count",
        "hash_join_cache_hits",
        "hash_join_cache_misses",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
        "hash_join_hash_table_bytes",
        "join_gather_map_bytes",
        "parquet_read_time",
        "operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ] {
        assert!(names.contains(&expected), "missing native DataFusion metric: {expected}");
    }
}
