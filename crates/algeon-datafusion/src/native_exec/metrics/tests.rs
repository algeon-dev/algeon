use std::{collections::BTreeMap, time::Duration};

use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricCategory};
use query_engine::{
    metrics::{
        JoinOperatorMetrics, OperatorMetrics, QueryMetrics, SourceReadParquetEvidenceMetrics, SourceReadRuntimeMetrics,
    },
    observability::{LakehouseCacheFillEventSnapshot, RuntimeObservationConfig},
};

use super::{
    AlgeonNativeMetrics, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES, NativeMetricsCapture,
    contract::{NATIVE_ALGEON_METRIC_DESCRIPTORS, NativeMetricKind},
    record::{NATIVE_ALGEON_RECORDING_DESCRIPTORS, NativeMetricRecordingKind},
};
use crate::{native_exec::admission::QueryAdmissionFacts, native_report::NativeDataFusionExecutionMetrics};

#[test]
fn native_summary_metrics_set_datafusion_v54_categories() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let _native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let metrics = metrics_set.clone_inner();

    let category_for = |name: &str| {
        metrics.iter().find(|metric| metric.value().name() == name).and_then(|metric| metric.metric_category())
    };

    assert_eq!(category_for("query_grant_bytes"), Some(MetricCategory::Bytes));
    assert_eq!(category_for("resident_dataframe_peak_bytes_observed"), Some(MetricCategory::Rows));
    assert_eq!(category_for("source_chunks_committed"), Some(MetricCategory::Rows));
    assert_eq!(category_for("admission_wait_time"), Some(MetricCategory::Timing));
    assert_eq!(category_for("source_read_time"), Some(MetricCategory::Timing));
    assert_eq!(category_for("output_rows"), Some(MetricCategory::Rows));
    assert_eq!(category_for("output_bytes"), Some(MetricCategory::Bytes));
    assert_eq!(category_for("elapsed_compute"), Some(MetricCategory::Timing));
}

#[test]
fn ordered_descriptors_drive_names_registration_and_legacy_categories() {
    const TIMER_NAMES: &[&str] = &[
        "admission_wait_time",
        "source_metadata_load_time",
        "source_read_queue_wait_time",
        "parquet_read_time",
        "source_read_time",
        "operator_time",
        "chunk_operator_time",
        "final_merge_time",
        "final_sort_time",
        "materialization_time",
    ];

    let descriptor_names =
        NATIVE_ALGEON_METRIC_DESCRIPTORS.iter().map(|descriptor| descriptor.name).collect::<Vec<_>>();
    assert_eq!(descriptor_names, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES);

    let metrics_set = ExecutionPlanMetricsSet::new();
    let _native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let registered = metrics_set.clone_inner();

    for descriptor in NATIVE_ALGEON_METRIC_DESCRIPTORS {
        let legacy_kind = if TIMER_NAMES.contains(&descriptor.name) {
            NativeMetricKind::Timer
        } else if descriptor.name.ends_with("_bytes_observed") {
            NativeMetricKind::Rows
        } else if descriptor.name.contains("bytes") {
            NativeMetricKind::Bytes
        } else {
            NativeMetricKind::Rows
        };
        assert_eq!(descriptor.kind, legacy_kind, "category drift for {}", descriptor.name);

        let matching = registered.iter().filter(|metric| metric.value().name() == descriptor.name).collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "{} must be registered exactly once", descriptor.name);
        let expected_category = match descriptor.kind {
            NativeMetricKind::Rows => MetricCategory::Rows,
            NativeMetricKind::Bytes => MetricCategory::Bytes,
            NativeMetricKind::Timer => MetricCategory::Timing,
        };
        assert_eq!(matching[0].metric_category(), Some(expected_category), "{} category", descriptor.name);
    }
}

#[test]
fn every_ordered_descriptor_has_a_compile_time_recording_mapping() {
    let recording_names = NATIVE_ALGEON_RECORDING_DESCRIPTORS.iter().map(|(name, _kind)| *name).collect::<Vec<_>>();
    assert_eq!(recording_names, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES);
    assert_eq!(
        recording_names.len(),
        99,
        "the ordered public metric count must include the sealed-attempt and graph-claim calibration rows"
    );

    for expected_kind in
        [NativeMetricRecordingKind::Query, NativeMetricRecordingKind::Admission, NativeMetricRecordingKind::Attempt]
    {
        assert!(
            NATIVE_ALGEON_RECORDING_DESCRIPTORS.iter().any(|(_name, kind)| *kind == expected_kind),
            "recording contract must exercise {expected_kind:?} mappings"
        );
    }
}

#[test]
fn labelled_row_recording_targets_do_not_repeat_metric_name_literals() {
    let source = include_str!("record.rs");
    assert!(
        !source.contains(".global_counter(\""),
        "labelled row targets must use descriptor-generated metric-name accessors"
    );
}

#[test]
fn native_metrics_record_query_service_admission_facts() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);

    native_metrics.record_admission_facts(QueryAdmissionFacts {
        admission_sequence: 7,
        admission_wait: Duration::from_millis(3),
        admission_wait_fact: None,
        device_ordinal: 2,
        query_grant_bytes: 4096,
    });

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.admission_sequence, Some(7));
    assert_eq!(metrics.admission_wait_time_ns, Some(3_000_000));
    assert_eq!(metrics.device_ordinal, Some(2));
    assert_eq!(metrics.query_grant_bytes, Some(4096));
}

#[test]
fn aggregate_strategy_rows_bridge_statistical_contract_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "stddev_samp_gpu_native_tolerant".to_string());
    facts.insert("execution_stage", "single".to_string());
    facts.insert("group_key_count", "4".to_string());
    facts.insert("group_key_exprs", "w_warehouse_name,w_warehouse_sk,i_item_sk,d_moy".to_string());
    facts.insert("group_key_dtypes", "Utf8View,Int64,Int64,Int64".to_string());
    facts.insert("aggregate_functions", "stddev_samp,avg".to_string());
    facts.insert("floating_statistical_aggregate", "true".to_string());
    facts.insert("statistical_aggregate_execution_mode", "gpu_native_tolerant".to_string());
    facts.insert("statistical_aggregate_tolerance_contract", "float_tolerant".to_string());
    facts.insert("statistical_aggregate_functions", "stddev_samp".to_string());
    facts.insert("statistical_aggregate_ddof", "stddev_samp:1".to_string());

    query_metrics.record_aggregate_strategy_decision("n12", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.node_id, "n12");
    assert_eq!(row.selected_strategy, "stddev_samp_gpu_native_tolerant");
    assert_eq!(row.floating_statistical_aggregate, "true");
    assert_eq!(row.statistical_aggregate_execution_mode, "gpu_native_tolerant");
    assert_eq!(row.statistical_aggregate_tolerance_contract, "float_tolerant");
    assert_eq!(row.statistical_aggregate_functions, "stddev_samp");
    assert_eq!(row.statistical_aggregate_ddof, "stddev_samp:1");
}

#[test]
fn aggregate_strategy_rows_bridge_strategy_execution_option_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let facts = BTreeMap::from([
        ("selected_strategy", "libcudf_groupby".to_string()),
        ("execution_stage", "partial".to_string()),
        ("aggregate_encoded_key_partial_low_cardinality_threshold_ppm", "175000".to_string()),
        ("aggregate_encoded_key_partial_min_input_rows", "123".to_string()),
        ("aggregate_exact_grouped_key_cardinality_probe_max_key_columns", "4".to_string()),
        ("aggregate_exact_single_string_avg_cardinality_probe_min_input_rows", "101".to_string()),
        ("aggregate_exact_single_key_cardinality_probe_min_input_rows", "202".to_string()),
        ("aggregate_exact_grouped_key_cardinality_probe_min_input_rows", "303".to_string()),
        ("aggregate_exact_single_string_key_prepared_max_groups", "404".to_string()),
        ("aggregate_exact_three_key_cardinality_probe_max_input_rows", "505".to_string()),
        ("aggregate_exact_five_key_cardinality_probe_max_input_rows", "606".to_string()),
        ("aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm", "200000".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_row_groups", "65".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes", "4096".to_string()),
        ("aggregate_source_adjacent_prepared_sort_groupby_max_fan_in", "7".to_string()),
    ]);

    query_metrics.record_aggregate_strategy_decision("n13", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.node_id, "n13");
    assert_eq!(row.aggregate_encoded_key_partial_low_cardinality_threshold_ppm, 175_000);
    assert_eq!(row.aggregate_encoded_key_partial_min_input_rows, 123);
    assert_eq!(row.aggregate_exact_grouped_key_cardinality_probe_max_key_columns, 4);
    assert_eq!(row.aggregate_exact_single_string_avg_cardinality_probe_min_input_rows, 101);
    assert_eq!(row.aggregate_exact_single_key_cardinality_probe_min_input_rows, 202);
    assert_eq!(row.aggregate_exact_grouped_key_cardinality_probe_min_input_rows, 303);
    assert_eq!(row.aggregate_exact_single_string_key_prepared_max_groups, 404);
    assert_eq!(row.aggregate_exact_three_key_cardinality_probe_max_input_rows, 505);
    assert_eq!(row.aggregate_exact_five_key_cardinality_probe_max_input_rows, 606);
    assert_eq!(row.aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm, 200_000);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_row_groups, 65);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes, 4096);
    assert_eq!(row.aggregate_source_adjacent_prepared_sort_groupby_max_fan_in, 7);
}

#[test]
fn aggregate_strategy_rows_bridge_filtered_global_materialization_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "filtered_global_segmented_state".to_string());
    facts.insert("execution_stage", "single".to_string());
    facts.insert("group_key_count", "0".to_string());
    facts.insert("group_key_exprs", "none".to_string());
    facts.insert("group_key_dtypes", "none".to_string());
    facts.insert("aggregate_functions", "count,sum,avg".to_string());
    facts.insert("aggregate_filter_count", "15".to_string());
    facts.insert("filtered_input_materialization_strategy", "row_chunked_segmented_state".to_string());
    facts.insert("filtered_input_materialization_estimated_bytes", "66000000000".to_string());
    facts.insert("filtered_input_materialization_budget_bytes", "85899345920".to_string());
    facts.insert("filtered_input_materialization_threshold_bytes", "42949672960".to_string());
    facts.insert("filtered_global_rows_per_chunk", "1024".to_string());
    facts.insert("filtered_global_inputs_per_segment", "1".to_string());
    facts.insert("filtered_global_segment_target_temporary_bytes", "536870912".to_string());
    facts.insert("filtered_global_actual_chunk_count", "98".to_string());
    facts.insert("filtered_global_actual_segment_count", "196".to_string());
    facts.insert("filtered_global_max_segment_rows", "1024".to_string());
    facts.insert("filtered_global_observed_temporary_peak_bytes", "16777216".to_string());
    facts.insert("filtered_global_observed_retained_peak_bytes", "4096".to_string());
    facts.insert("filtered_global_state_rows_merged", "98".to_string());

    query_metrics.record_aggregate_strategy_decision("n5", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.selected_strategy, "filtered_global_segmented_state");
    assert_eq!(row.aggregate_filter_count, 15);
    assert_eq!(row.filtered_input_materialization_strategy, "row_chunked_segmented_state");
    assert_eq!(row.filtered_input_materialization_estimated_bytes, 66_000_000_000);
    assert_eq!(row.filtered_input_materialization_budget_bytes, 85_899_345_920);
    assert_eq!(row.filtered_input_materialization_threshold_bytes, 42_949_672_960);
    assert_eq!(row.filtered_global_rows_per_chunk, 1024);
    assert_eq!(row.filtered_global_inputs_per_segment, 1);
    assert_eq!(row.filtered_global_segment_target_temporary_bytes, 536_870_912);
    assert_eq!(row.filtered_global_actual_chunk_count, 98);
    assert_eq!(row.filtered_global_actual_segment_count, 196);
    assert_eq!(row.filtered_global_max_segment_rows, 1024);
    assert_eq!(row.filtered_global_observed_temporary_peak_bytes, 16_777_216);
    assert_eq!(row.filtered_global_observed_retained_peak_bytes, 4096);
    assert_eq!(row.filtered_global_state_rows_merged, 98);
}

#[test]
fn aggregate_strategy_rows_bridge_filtered_grouped_materialization_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("selected_strategy", "filtered_grouped_segmented_state".to_string());
    facts.insert("execution_stage", "partial".to_string());
    facts.insert("group_key_count", "1".to_string());
    facts.insert("group_key_exprs", "key".to_string());
    facts.insert("group_key_dtypes", "Int64".to_string());
    facts.insert("aggregate_functions", "count,sum,max".to_string());
    facts.insert("aggregate_filter_count", "90".to_string());
    facts.insert("filtered_input_materialization_strategy", "row_chunked_grouped_segmented_state".to_string());
    facts.insert("filtered_input_materialization_estimated_bytes", "33000000".to_string());
    facts.insert("filtered_input_materialization_budget_bytes", "33554432".to_string());
    facts.insert("filtered_input_materialization_threshold_bytes", "16777216".to_string());
    facts.insert("filtered_grouped_rows_per_chunk", "512".to_string());
    facts.insert("filtered_grouped_inputs_per_segment", "2".to_string());
    facts.insert("filtered_grouped_segment_target_temporary_bytes", "4194304".to_string());
    facts.insert("filtered_grouped_actual_chunk_count", "40".to_string());
    facts.insert("filtered_grouped_actual_segment_count", "1200".to_string());
    facts.insert("filtered_grouped_max_segment_rows", "512".to_string());
    facts.insert("filtered_grouped_observed_temporary_peak_bytes", "1048576".to_string());
    facts.insert("filtered_grouped_observed_retained_peak_bytes", "524288".to_string());
    facts.insert("filtered_grouped_state_rows_merged", "20000".to_string());
    facts.insert("filtered_grouped_high_cardinality_admission", "chunk_local_partial_final_merge".to_string());

    query_metrics.record_aggregate_strategy_decision("n9", &facts);
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.aggregate_strategy_rows.len(), 1);
    let row = &metrics.aggregate_strategy_rows[0];
    assert_eq!(row.selected_strategy, "filtered_grouped_segmented_state");
    assert_eq!(row.aggregate_filter_count, 90);
    assert_eq!(row.filtered_input_materialization_strategy, "row_chunked_grouped_segmented_state");
    assert_eq!(row.filtered_grouped_rows_per_chunk, 512);
    assert_eq!(row.filtered_grouped_inputs_per_segment, 2);
    assert_eq!(row.filtered_grouped_segment_target_temporary_bytes, 4_194_304);
    assert_eq!(row.filtered_grouped_actual_chunk_count, 40);
    assert_eq!(row.filtered_grouped_actual_segment_count, 1200);
    assert_eq!(row.filtered_grouped_max_segment_rows, 512);
    assert_eq!(row.filtered_grouped_observed_temporary_peak_bytes, 1_048_576);
    assert_eq!(row.filtered_grouped_observed_retained_peak_bytes, 524_288);
    assert_eq!(row.filtered_grouped_state_rows_merged, 20_000);
    assert_eq!(row.filtered_grouped_high_cardinality_admission, "chunk_local_partial_final_merge");
}

#[test]
fn join_strategy_rows_preserve_planned_strategy_while_counting_runtime_execution_strategy() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();
    let mut facts = BTreeMap::new();
    facts.insert("join_strategy", "partitioned_hash_equi".to_string());
    facts.insert("estimated_build_rows", "10000".to_string());
    facts.insert("partition_count", "8".to_string());

    query_metrics.record_join_strategy_decision("n6", &facts);
    query_metrics.record_operator_metrics(OperatorMetrics::join(
        "n6",
        JoinOperatorMetrics {
            join_type: "inner".to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: 1,
            build_side: "right".to_string(),
            probe_side: "left".to_string(),
            build_rows: 10_000,
            build_bytes: 64 * 1024 * 1024,
            probe_rows: 1_000_000,
            probe_bytes: 512 * 1024 * 1024,
            gather_map_bytes: 2048,
            output_reservation_bytes: 4096,
            output_rows: 100,
            output_bytes: 8192,
            ..Default::default()
        },
    ));

    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.join_strategy_rows.len(), 1);
    let row = &metrics.join_strategy_rows[0];
    assert_eq!(row.join_strategy, "partitioned_hash_equi");
    assert_eq!(row.execution_strategy, "bounded_one_shot_equi_join");
    assert_eq!(row.partitioned_hash_join_count, 0);
    assert_eq!(row.one_shot_join_count, 1);
}

#[test]
fn source_read_runtime_rows_bridge_runtime_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::default();

    query_metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
        query_id: None,
        source_node_id: "n7".to_string(),
        read_kind: "chunk_batch".to_string(),
        read_strategy: "remote_direct".to_string(),
        parquet_read_evidence: SourceReadParquetEvidenceMetrics {
            input_kind: "remote_path".to_string(),
            remote_read_mode: "require_kvikio".to_string(),
            backend: "cudf_sys_kvikio_remote_datasource".to_string(),
            backend_build_status: "kvikio_remote_io_enabled".to_string(),
            remote_bytes: Some(4096),
            remote_range_count: Some(3),
            device_read_support: "not_reported_by_libcudf".to_string(),
            pruning_input_row_groups: Some(9),
            row_groups_after_stats_filter: Some(6),
            row_groups_after_bloom_filter: Some(2),
            bloom_pruning_status: "bloom_pruned".to_string(),
        },
        batch_index: 2,
        chunk_count: 3,
        first_chunk_index: 4,
        last_chunk_index: 6,
        first_file_index: 10,
        last_file_index: 12,
        file_count: 3,
        row_group_count: 9,
        read_columns_per_file: 2,
        read_columns_total: 6,
        known_file_bytes: 900,
        estimated_source_read_bytes: 700,
        known_row_group_bytes: 600,
        known_projected_compressed_bytes: 500,
        known_projected_uncompressed_bytes: 800,
        known_pure_filter_compressed_bytes: 100,
        known_pure_filter_uncompressed_bytes: 200,
        output_rows: 42,
        reader_predicate: true,
        reader_dynamic_filter_count: 1,
        source_batching_reason: "batched".to_string(),
        source_chunk_planning_mode: "local_parquet_throughput".to_string(),
        source_chunk_planning_reason: "local_parquet_throughput_admitted".to_string(),
        elapsed: std::time::Duration::from_nanos(1234),
    });
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let row = &metrics.source_read_runtime_rows[0];
    assert_eq!(row.source_node_id, "n7");
    assert_eq!(row.read_kind, "chunk_batch");
    assert_eq!(row.read_strategy, "remote_direct");
    assert_eq!(row.batch_index, 2);
    assert_eq!(row.chunk_count, 3);
    assert_eq!(row.first_file_index, 10);
    assert_eq!(row.file_count, 3);
    assert_eq!(row.row_group_count, 9);
    assert_eq!(row.read_columns_total, 6);
    assert_eq!(row.known_projected_uncompressed_bytes, 800);
    assert_eq!(row.known_pure_filter_uncompressed_bytes, 200);
    assert_eq!(row.parquet_read_input_kind, "remote_path");
    assert_eq!(row.remote_read_mode, "require_kvikio");
    assert_eq!(row.parquet_read_backend, "cudf_sys_kvikio_remote_datasource");
    assert_eq!(row.remote_backend_build_status, "kvikio_remote_io_enabled");
    assert_eq!(row.remote_read_bytes, Some(4096));
    assert_eq!(row.remote_read_range_count, Some(3));
    assert_eq!(row.device_read_support, "not_reported_by_libcudf");
    assert_eq!(row.parquet_pruning_input_row_groups, Some(9));
    assert_eq!(row.parquet_row_groups_after_stats_filter, Some(6));
    assert_eq!(row.parquet_row_groups_after_bloom_filter, Some(2));
    assert_eq!(row.parquet_bloom_pruning_status, "bloom_pruned");
    assert_eq!(row.output_rows, 42);
    assert!(row.reader_predicate);
    assert_eq!(row.reader_dynamic_filter_count, 1);
    assert_eq!(row.source_batching_reason, "batched");
    assert_eq!(row.source_chunk_planning_mode, "local_parquet_throughput");
    assert_eq!(row.source_chunk_planning_reason, "local_parquet_throughput_admitted");
    assert_eq!(row.elapsed_ns, 1234);
}

#[test]
fn query_metric_row_bridge_enforces_shared_record_and_byte_caps() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let capture = NativeMetricsCapture::standalone()
        .with_runtime_observation_config(RuntimeObservationConfig::bounded(1, 4096))
        .new_execution();
    let native_metrics = AlgeonNativeMetrics::new_with_capture(&metrics_set, 0, Some(capture));
    let mut query_metrics = QueryMetrics::default();
    for index in 0..128 {
        query_metrics.record_source_read_runtime(SourceReadRuntimeMetrics {
            source_node_id: format!("parquet-source-{index}"),
            read_kind: "chunk_batch".to_string(),
            read_strategy: "local_parquet".to_string(),
            ..SourceReadRuntimeMetrics::default()
        });
    }

    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let (remaining_records, remaining_bytes, truncated, overflow) = native_metrics.row_capture_state();
    assert_eq!(remaining_records, 0);
    assert!(remaining_bytes < 4096);
    assert!(truncated);
    assert!(!overflow);
    assert_eq!(
        metrics_set.clone_inner().sum_by_name(super::SERVER_CAPTURE_LOSS_METRIC_NAME).unwrap().as_usize(),
        1,
        "one bounded boolean marker reports row loss without growing per dropped row"
    );
}

#[test]
fn lakehouse_cache_fill_event_snapshot_rows_bridge_runtime_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics = AlgeonNativeMetrics::new(&metrics_set, 0);
    let mut query_metrics = QueryMetrics::new("q-cache-fill");
    query_metrics.lakehouse_cache_fill_events.push(LakehouseCacheFillEventSnapshot {
        outcome: "failed",
        bytes: None,
        elapsed: std::time::Duration::from_nanos(4321),
        cache_path_hash: "cacheabcd".to_string(),
        stable_id_hash: "stableabcd".to_string(),
        file_index: Some(3),
        chunk_index: Some(5),
        error_code: Some("cudf_dependency_failure".to_string()),
        error_kind: Some("dependency".to_string()),
        error_status: Some("retryable".to_string()),
        error_operation: Some("query_engine.lakehouse_cache.background_fill.fetch".to_string()),
    });
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.lakehouse_cache_fill_event_snapshot_rows.len(), 1);
    let row = &metrics.lakehouse_cache_fill_event_snapshot_rows[0];
    assert_eq!(row.outcome, "failed");
    assert_eq!(row.bytes, None);
    assert_eq!(row.elapsed_ns, 4321);
    assert_eq!(row.cache_path_hash, "cacheabcd");
    assert_eq!(row.stable_id_hash, "stableabcd");
    assert_eq!(row.file_index, Some(3));
    assert_eq!(row.chunk_index, Some(5));
    assert_eq!(row.error_code, "cudf_dependency_failure");
    assert_eq!(row.error_operation, "query_engine.lakehouse_cache.background_fill.fetch");
}

#[test]
fn plan_task_mapping_rows_bridge_correlation_labels() {
    let metrics_set = ExecutionPlanMetricsSet::new();
    let native_metrics =
        AlgeonNativeMetrics::new_with_capture_and_shape(&metrics_set, 0, None, Some("AggregateExec(LocalParquetScan)"));
    let mut query_metrics = QueryMetrics::new("q-mapping");

    query_metrics.record_compiled_pipeline_task_mapping("t3", "source_read", Some("n4"));
    native_metrics.record_query_metrics(&query_metrics);

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics_set.clone_inner());
    assert_eq!(metrics.plan_task_mapping_rows.len(), 1);
    let row = &metrics.plan_task_mapping_rows[0];
    assert_eq!(row.query_id, "q-mapping");
    assert_eq!(row.plan_node_id.as_deref(), Some("n4"));
    assert_eq!(row.compiled_task_id.as_deref(), Some("t3"));
    assert_eq!(row.operator_kind.as_deref(), Some("source_read"));
    assert_eq!(row.datafusion_shape.as_deref(), Some("AggregateExec(LocalParquetScan)"));
}
