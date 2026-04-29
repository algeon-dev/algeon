use cudf_query_engine::QueryMetrics;

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
