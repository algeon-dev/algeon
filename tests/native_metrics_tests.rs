#[cfg(feature = "bench-support")]
use datafusion_nexus::native_report::bench_support::NATIVE_TPCDS_PARITY_TSV_HEADER;
use datafusion_nexus::{
    native_exec::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES,
    native_report::{
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER, native_report_metric_descriptor, native_report_metric_descriptors,
    },
};
use nexus_query_engine::metrics::{MetricAggregation, MetricUnit, QueryId, QueryMetrics, TraceId};

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
    assert_eq!(metrics.source_chunks_planned, 0);
    assert_eq!(metrics.source_chunks_attempted, 0);
    assert_eq!(metrics.source_chunks_committed, 0);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.source_metadata_load_time, std::time::Duration::ZERO);
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
    metrics.record_retry_resource_exhausted();

    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.row_groups_planned, 4);
    assert_eq!(metrics.source_chunks_attempted, 2);
    assert_eq!(metrics.source_chunks_retried, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_committed, 2);
    assert_eq!(metrics.retry_count, 1);
    assert_eq!(metrics.retry_resource_exhausted_count, 1);
}

#[test]
fn query_metrics_counts_multi_chunk_batch_commits() {
    let mut metrics = QueryMetrics::default();

    metrics.record_source_chunk_commit_count(3, 9);

    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_committed, 9);
}

#[test]
fn query_metrics_restamps_existing_runtime_rows_when_query_id_changes() {
    let mut metrics = QueryMetrics::new("old-query");
    metrics.record_compiled_pipeline_task_execution("task-1", "source_read");
    metrics.record_compiled_pipeline_task_mapping("task-1", "source_read", Some("n7"));
    metrics.set_query_id("capture-query");

    assert_eq!(metrics.query_id().as_str(), "capture-query");
    assert_eq!(metrics.task_runtime_metrics["task-1"].query_id.as_ref().map(QueryId::as_str), Some("capture-query"));
    assert_eq!(
        metrics.task_operator_metrics["source_read"].query_id.as_ref().map(QueryId::as_str),
        Some("capture-query")
    );
    assert!(metrics.plan_task_mappings().iter().all(|row| row.query_id.as_str() == "capture-query"));
}

#[test]
fn query_metrics_json_projection_is_catalog_driven() {
    let mut metrics = QueryMetrics::new("q-json");
    metrics.trace_id = Some(TraceId::new("request-1"));
    metrics = metrics.with_plan_task_mapping_datafusion_shape("AggregateExec(LocalParquetScan)");
    metrics.record_input_batch(42);
    metrics.record_compiled_pipeline_task_execution("task-1", "source_read");
    metrics.record_compiled_pipeline_task_mapping("task-1", "source_read", Some("n7"));
    let json = metrics.to_json();

    assert_eq!(json["schema"], "nexus_query_engine.query_metrics.v1");
    assert_eq!(json["query_id"], "q-json");
    assert_eq!(json["trace_id"], "request-1");
    assert_eq!(json["triage"]["verdict"], "healthy");
    assert_eq!(json["metrics"]["input_rows"]["value"], 42);
    assert!(json["rows"].get("memory_reservations").is_none());
    assert_eq!(json["rows"]["plan_task_mappings"][0]["query_id"], "q-json");
    assert_eq!(json["rows"]["plan_task_mappings"][0]["datafusion_shape"], "AggregateExec(LocalParquetScan)");

    for descriptor in QueryMetrics::descriptors() {
        assert!(
            json["metrics"].get(descriptor.name).is_some(),
            "missing JSON metric value for descriptor {}",
            descriptor.name
        );
    }
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
fn native_datafusion_metric_names_cover_bounded_runtime_contract() {
    let names = NATIVE_NEXUS_DATAFUSION_METRIC_NAMES;
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
        "row_groups_planned",
        "row_groups_committed",
        "partial_aggregate_input_rows",
        "partial_aggregate_retained_bytes",
        "final_merge_max_fan_in",
        "resident_dataframe_peak_bytes_observed",
        "allocator_high_water_bytes_observed",
        "max_task_observed_peak_bytes",
        "max_task_observed_peak_bytes_observed",
        "max_task_retained_bytes",
        "max_task_retained_bytes_observed",
        "native_graph_runtime_rows",
        "native_task_operator_rows",
        "native_task_runtime_rows",
        "native_dynamic_filter_runtime_rows",
        "native_source_read_runtime_rows",
        "native_lakehouse_cache_fill_event_snapshot_rows",
        "compiled_pipeline_attempts",
        "compiled_pipeline_tasks_executed",
        "compiled_pipeline_host_arrow_sinks",
        "retry_resource_exhausted_count",
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
        "parquet_read_time",
        "operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ] {
        assert!(names.contains(&expected), "missing native DataFusion metric: {expected}");
    }

    for removed in [
        "build_materialization_boundary_durable",
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "source_cache_lookup_time",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
    ] {
        assert!(!names.contains(&removed), "removed cache metric must not remain public: {removed}");
    }
}

#[test]
fn native_datafusion_metric_name_order_matches_public_contract_golden() {
    let actual = format!("{}\n", NATIVE_NEXUS_DATAFUSION_METRIC_NAMES.join("\n"));
    assert_eq!(actual, include_str!("fixtures/native_nexus_datafusion_metric_names.txt"));
}

#[test]
fn descriptor_registry_covers_report_tsv_columns_and_datafusion_metrics() {
    for name in NATIVE_NEXUS_DATAFUSION_METRIC_NAMES {
        assert!(native_report_metric_descriptor(name).is_some(), "missing descriptor for metric {name}");
    }
    for header in [
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
    ] {
        for column in header.split('\t') {
            assert!(native_report_metric_descriptor(column).is_some(), "missing descriptor for TSV column {column}");
        }
    }
    #[cfg(feature = "bench-support")]
    for column in NATIVE_TPCDS_PARITY_TSV_HEADER.split('\t') {
        assert!(native_report_metric_descriptor(column).is_some(), "missing descriptor for TSV column {column}");
    }

    let descriptors = QueryMetrics::descriptors();
    let mut names = descriptors.iter().map(|descriptor| descriptor.name).collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), descriptors.len(), "query metric descriptors must be unique");
    assert!(descriptors.iter().all(|descriptor| !descriptor.semantics.is_empty()));

    let report_descriptors = native_report_metric_descriptors();
    let mut report_names = report_descriptors.iter().map(|descriptor| descriptor.name).collect::<Vec<_>>();
    report_names.sort_unstable();
    report_names.dedup();
    assert_eq!(report_names.len(), report_descriptors.len(), "native report descriptors must be unique");

    for (name, unit, aggregation) in [
        ("query_id", MetricUnit::Label, MetricAggregation::Last),
        ("outcome", MetricUnit::Label, MetricAggregation::Last),
        ("reason_code", MetricUnit::Label, MetricAggregation::Last),
        ("edge_normalize_inserted", MetricUnit::Boolean, MetricAggregation::Last),
        ("native_task_runtime_rows", MetricUnit::Count, MetricAggregation::RowCount),
        ("native_plan_task_mapping_rows", MetricUnit::Count, MetricAggregation::RowCount),
    ] {
        let descriptor = native_report_metric_descriptor(name).expect("descriptor must exist");
        assert_eq!(descriptor.unit, unit, "descriptor {name} has wrong unit");
        assert_eq!(descriptor.aggregation, aggregation, "descriptor {name} has wrong aggregation");
    }
}
