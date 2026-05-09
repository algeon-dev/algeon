use super::*;

pub(super) fn test_config() -> LocalParquetBenchmarkConfig {
    LocalParquetBenchmarkConfig {
        input_rows: 16,
        file_count: 4,
        measure_columns: 2,
        batch_size: 4,
        row_group_size: 4,
        group_count: 4,
        limit: 2,
        warmup_iterations: 0,
        iterations: 1,
        minimum_file_bytes: 1,
        minimum_native_gpu_speedup: 1.0,
        native_memory_control_policy: NativeMemoryControlPolicy::Bounded {
            device_budget_bytes: DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
        },
        benchmark_paths: BenchmarkPathSet::NativeOnly,
        compression: BenchmarkCompression::Uncompressed,
        native_plan_shape: BenchmarkPlanShape::DefaultSql,
        expected_outcome: BenchmarkExpectedOutcome::Success,
        output_dir: None,
    }
}

pub(super) fn test_collect_measurement(wall_time: Duration) -> CollectMeasurement {
    CollectMeasurement { wall_time, output_batches: 1, output_rows: 2, output_bytes: 64, batches: Vec::new() }
}

pub(super) fn test_gpu_measurement(
    wall_time: Duration,
    metrics: LargerIslandBenchmarkMetrics,
    planning: LargerIslandBenchmarkPlanning,
    native_datafusion_planning_report: Option<NativeDataFusionPlanningReport>,
    plan_one_line: &str,
    source_physical_plan_one_line: Option<&str>,
    read_columns: Vec<String>,
) -> GpuCollectMeasurement {
    GpuCollectMeasurement {
        wall_time,
        output_batches: 1,
        output_rows: 2,
        output_bytes: 64,
        metrics,
        planning,
        native_datafusion_planning_report,
        plan_one_line: plan_one_line.to_string(),
        source_physical_plan_one_line: source_physical_plan_one_line.map(str::to_string),
        native_candidate_shapes: vec![
            "accepted_replaced:SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))):none"
                .to_string(),
        ],
        read_columns,
        batches: Vec::new(),
    }
}

pub(super) fn test_native_datafusion_planning_report() -> NativeDataFusionPlanningReport {
    NativeDataFusionPlanningReport {
        rows: vec![datafusion_cudf::NativeDataFusionPlanningRow {
            query_id: "local_parquet_benchmark".to_string(),
            physical_root: "SortExec".to_string(),
            physical_shape: "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
            native_candidate_shape: Some(
                "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
            ),
            native_candidate_shape_family: Some("sort_fetch_local_parquet_aggregate".to_string()),
            outcome: datafusion_cudf::NativeDataFusionPlanningOutcome::AcceptedReplaced,
            reason_code: None,
            unsupported_category: None,
            bounded_join_reservation_facts: Default::default(),
            join_diagnostics: Vec::new(),
            source_diagnostics: vec![datafusion_cudf::NativeSourceDiagnosticSummary {
                format: "Parquet".to_string(),
                direct_device_read: true,
                file_count: 1,
                ranged_file_count: 0,
                known_bytes: Some(4096),
                selected_row_groups: Some(2),
                predicate_disposition: "pruning_only".to_string(),
                byte_range_disposition: "full_files".to_string(),
                partition_column_disposition: "absent".to_string(),
                zero_column_count_disposition: "not_required".to_string(),
                schema_mapping_disposition: "direct_by_name".to_string(),
                read_column_count: 4,
                projected_output_column_count: 3,
                pure_filter_column_count: 1,
                rejected_features: Vec::new(),
            }],
            post_native_plan_root: "NativeCudfExec".to_string(),
            post_native_execution_path: datafusion_cudf::NativeDataFusionPostNativeExecutionPath::Native,
            rejected_before_execution: false,
        }],
        tpch_dataset: None,
    }
}

pub(super) fn test_native_metrics() -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy: "bounded",
        phase_c_runtime_gate: "pass",
        input_batches: 1,
        input_rows: 16,
        resident_dataframe_peak_bytes: Some(1024),
        allocator_high_water_bytes: Some(2048),
        reservation_requested_bytes: 1024,
        reservation_granted_bytes: 1024,
        reservation_denied_count: 0,
        source_chunks_planned: 2,
        source_chunks_attempted: 2,
        source_chunks_committed: 2,
        source_chunks_retried: 0,
        row_groups_planned: 2,
        row_groups_committed: 2,
        partial_aggregate_count: 2,
        partial_aggregate_rows: 4,
        final_merge_attempts: 1,
        final_merge_levels: 1,
        retry_count: 0,
        retry_resource_exhausted_count: 0,
        retry_reservation_failure_count: 0,
        peak_device_memory_bytes: Some(1024),
        host_to_device_uploads: 0,
        host_to_device_uploaded_rows: 0,
        host_to_device_uploaded_bytes: 0,
        device_to_host_materializations: 1,
        device_to_host_materialized_rows: 2,
        device_to_host_materialized_bytes: 64,
        parquet_files: 4,
        parquet_bytes_known: 4096,
        row_groups_selected: 2,
        read_columns: 4,
        parquet_read_time: Duration::from_millis(3),
        record_batch_upload_time: Duration::ZERO,
        cudf_operator_time: Duration::from_millis(4),
        record_batch_materialization_time: Duration::from_millis(1),
        elapsed_compute: Some(Duration::from_millis(8)),
        output_rows: Some(2),
    }
}

pub(super) fn assert_key_order(text: &str, before: &str, after: &str) {
    let before_index = text.find(before).unwrap_or_else(|| panic!("missing key {before}"));
    let after_index = text.find(after).unwrap_or_else(|| panic!("missing key {after}"));
    assert!(before_index < after_index, "expected {before} to appear before {after}");
}
