use super::*;

#[test]
fn expected_resource_failure_report_includes_resource_error_facts() {
    let error = AdapterError::new(
        datafusion_cudf::ErrorCode::MemoryReservationFailed,
        "datafusion_cudf.native_exec.execute",
        "native cudf query execution failed",
    )
    .with_fact("native_code", cudf_query_engine::QueryErrorCode::MemoryReservationFailed)
    .with_fact("native_kind", "resource_exhausted")
    .with_fact("native_status", "temporary")
    .with_fact("native_operation", "cudf_query_engine.memory.reserve")
    .with_fact("native_datafusion_phase", "runtime_execution")
    .with_fact("cpu_fallback_after_native_start", false)
    .with_fact("memory_boundary", "final_sort")
    .with_fact("stage", "final_sort")
    .with_fact("attempt", 1)
    .with_fact("row_groups", "0,1")
    .with_fact("resume_boundary", "final_sort")
    .with_fact("committed_chunks", 2);
    let report = ExpectedResourceFailureReport {
        parquet_file_bytes: 4096,
        parquet_row_groups: 32,
        input_rows: 1,
        file_count: 1,
        compression: BenchmarkCompression::Uncompressed,
        native_plan_shape: BenchmarkPlanShape::DefaultSql,
        expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
        final_plan_root: "NativeCudfExec".to_string(),
        planning_report: test_native_datafusion_planning_report(),
        native_candidate_shapes: vec!["accepted_replaced:LocalLimitExec(SortExec(...)):none".to_string()],
        source_physical_plan_one_line: "SortExec(fetch=100)".to_string(),
        source_diagnostic: "files=1".to_string(),
        error: AdapterRuntimeErrorReport::from_adapter_error(&error),
    };

    let text = report.to_text();

    assert!(text.contains("case=local_parquet_expected_resource_failure"));
    assert!(text.contains("bounded_runtime_outcome=expected_resource_failure"));
    assert!(text.contains("no_silent_cpu_fallback=true"));
    assert!(text.contains("native_runtime_executed=true"));
    assert!(text.contains("datafusion_native_exec_planning_outcome=accepted_replaced"));
    assert!(text.contains("native_error_code=memory_reservation_failed"));
    assert!(text.contains("native_error_kind=resource_exhausted"));
    assert!(text.contains("native_error_status=temporary"));
    assert!(text.contains("native_error_facts="));
    assert!(text.contains("memory_boundary:final_sort"));
    assert!(text.contains("stage:final_sort"));
    assert!(text.contains("attempt:1"));
    assert!(text.contains("row_groups:0,1"));
    assert!(text.contains("resume_boundary:final_sort"));
    assert!(text.contains("committed_chunks:2"));
}

#[test]
fn direct_parquet_benchmark_output_names_primary_datafusion_native_path() {
    let config = test_config();
    let metrics = test_native_metrics();
    let planning = LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::from_outcomes([
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))",
                2,
                4,
            ),
        },
    ]));
    let datafusion_planning_report = NativeDataFusionPlanningReport {
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
                file_count: config.file_count,
                ranged_file_count: 0,
                known_bytes: Some(4096),
                selected_row_groups: Some(2),
                predicate_disposition: "pruning_only".to_string(),
                byte_range_disposition: "full_files".to_string(),
                partition_column_disposition: "absent".to_string(),
                zero_column_count_disposition: "not_required".to_string(),
                scan_limit_disposition: "absent".to_string(),
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
    };
    let report = DirectParquetBenchmarkReport {
        parquet_file_bytes: 4096,
        parquet_row_groups: 4,
        input_rows: config.input_rows,
        file_count: config.file_count,
        measure_columns: config.measure_columns,
        batch_size: config.batch_size,
        row_group_size: config.row_group_size,
        group_count: config.group_count,
        warmup_iterations: 0,
        iterations: 1,
        limit: config.limit,
        benchmark_paths: config.benchmark_paths,
        compression: config.compression,
        native_plan_shape: config.native_plan_shape,
        expected_outcome: config.expected_outcome,
        measured_run_order: measured_run_order_summary(1, config.benchmark_paths),
        cpu_comparable: TimedBenchmarkRun::from_samples([Duration::from_millis(100)]),
        cpu_default: TimedBenchmarkRun::from_samples([Duration::from_millis(110)]),
        gpu: None,
        datafusion_native_exec: TimedBenchmarkRun::from_samples([Duration::from_millis(50)]),
        manual_native_lowering: TimedBenchmarkRun::from_samples([Duration::from_millis(55)]),
        cpu_comparable_samples: vec![Duration::from_millis(100)],
        cpu_default_samples: vec![Duration::from_millis(110)],
        gpu_samples: Vec::new(),
        datafusion_native_exec_samples: vec![Duration::from_millis(50)],
        manual_native_lowering_samples: vec![Duration::from_millis(55)],
        gpu_metric_samples: Vec::new(),
        datafusion_native_exec_metric_samples: vec![metrics.clone()],
        manual_native_lowering_metric_samples: vec![metrics.clone()],
        final_cpu_comparable: test_collect_measurement(Duration::from_millis(100)),
        final_cpu_default: test_collect_measurement(Duration::from_millis(110)),
        final_gpu: None,
        final_datafusion_native_exec: test_gpu_measurement(
            Duration::from_millis(50),
            metrics.clone(),
            planning.clone(),
            Some(datafusion_planning_report),
            "NativeCudfExec: native_nodes=5",
            Some("SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))"),
            config.expected_read_columns(),
        ),
        final_manual_native_lowering: test_gpu_measurement(
            Duration::from_millis(55),
            metrics,
            planning,
            None,
            "LocalParquetSource: format=Parquet, direct_device_read=true, sink=host_arrow",
            Some("SortExec(fetch)"),
            config.expected_read_columns(),
        ),
        source_diagnostic: "files=4".to_string(),
    };

    let text = report.to_text();
    assert!(text.contains("cpu_datafusion_wall_ms=110.000"));
    assert!(text.contains("datafusion_native_exec_wall_ms=50.000"));
    assert!(text.contains("manual_native_lowering_wall_ms=55.000"));
    assert!(text.contains("datafusion_native_exec_planning_outcome=accepted_replaced"));
    assert!(text.contains("datafusion_native_exec_planning_post_native_execution_path=native"));
    assert!(text.contains("datafusion_native_exec_planning_source_diagnostics=format=Parquet"));
    assert!(text.contains("read_columns=4,projected_output_columns=3,pure_filter_columns=1"));
    assert!(text.contains("byte_range=full_files,partition_columns=absent,zero_column_count=not_required"));
    let zero_column_count_artifact_text =
        native_source_diagnostics_text(&[datafusion_cudf::NativeSourceDiagnosticSummary {
            format: "Parquet".to_string(),
            direct_device_read: true,
            file_count: 1,
            ranged_file_count: 0,
            known_bytes: Some(4096),
            selected_row_groups: Some(2),
            predicate_disposition: "absent".to_string(),
            byte_range_disposition: "full_files".to_string(),
            partition_column_disposition: "absent".to_string(),
            zero_column_count_disposition: "represented".to_string(),
            scan_limit_disposition: "absent".to_string(),
            schema_mapping_disposition: "direct_by_name".to_string(),
            read_column_count: 0,
            projected_output_column_count: 0,
            pure_filter_column_count: 0,
            rejected_features: Vec::new(),
        }]);
    assert!(zero_column_count_artifact_text.contains("read_columns=0"));
    assert!(zero_column_count_artifact_text.contains("zero_column_count=represented"));
    assert_key_order(&text, "cpu_datafusion_wall_ms", "datafusion_native_exec_wall_ms");
    assert_key_order(&text, "datafusion_native_exec_wall_ms", "manual_native_lowering_wall_ms");
    assert_key_order(&text, "datafusion_native_exec_planning_accepted_count", "manual_native_lowering_plan");
}
