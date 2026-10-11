#[cfg(feature = "bench-support")]
use algeon_datafusion::native_report::bench_support::NATIVE_TPCDS_PARITY_TSV_HEADER;
use algeon_datafusion::{
    native_exec::NATIVE_ALGEON_DATAFUSION_METRIC_NAMES,
    native_report::{
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER, native_report_metric_descriptor, native_report_metric_descriptors,
    },
};
use query_planning::correlation::{QueryId, TraceId};
use query_runtime::metrics::{MetricAggregation, MetricUnit, QUERY_METRICS_JSON_SCHEMA, QueryMetrics};

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

    assert_eq!(json["schema"], QUERY_METRICS_JSON_SCHEMA);
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
fn native_datafusion_metric_name_order_matches_public_contract_golden() {
    let mut unique = NATIVE_ALGEON_DATAFUSION_METRIC_NAMES.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), NATIVE_ALGEON_DATAFUSION_METRIC_NAMES.len());
    let actual = format!("{}\n", NATIVE_ALGEON_DATAFUSION_METRIC_NAMES.join("\n"));
    assert_eq!(actual, include_str!("../fixtures/native_algeon_datafusion_metric_names.txt"));
}

#[test]
fn descriptor_registry_covers_report_tsv_columns_and_datafusion_metrics() {
    for name in NATIVE_ALGEON_DATAFUSION_METRIC_NAMES {
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
