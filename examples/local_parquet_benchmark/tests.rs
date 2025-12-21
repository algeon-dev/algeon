use super::*;

#[test]
fn phase_c_decision_matrix_covers_required_dimensions() {
    let cases = phase_c_decision_matrix();

    assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::ThreeGiB));
    assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::SixGiB));
    assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::TenGiB));
    assert!(cases.iter().any(|case| case.compression == BenchmarkCompression::Uncompressed));
    assert!(cases.iter().any(|case| case.compression == BenchmarkCompression::Snappy));
    assert!(cases.iter().any(|case| case.source_layout == BenchmarkSourceLayout::SingleFileMultiRowGroup));
    assert!(cases.iter().any(|case| case.source_layout == BenchmarkSourceLayout::MultiFile));
    assert!(cases.iter().any(|case| case.native_plan_shape == BenchmarkPlanShape::DefaultSql));
    assert!(cases.iter().any(|case| case.native_plan_shape == BenchmarkPlanShape::DirectFriendly));
    assert!(cases.iter().any(|case| case.expected_outcome == BenchmarkExpectedOutcome::Success));
    assert!(cases.iter().any(|case| case.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure));

    let success_cases =
        cases.iter().filter(|case| case.expected_outcome == BenchmarkExpectedOutcome::Success).collect::<Vec<_>>();
    assert_eq!(success_cases.len(), 16);
    assert!(success_cases.iter().all(|case| case.scale != BenchmarkScalePoint::TenGiB));

    let expected_failure_cases = cases
        .iter()
        .filter(|case| case.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure)
        .collect::<Vec<_>>();
    assert_eq!(expected_failure_cases.len(), 8);
    assert!(expected_failure_cases.iter().all(|case| case.scale == BenchmarkScalePoint::TenGiB));
}

#[test]
fn phase_c_matrix_case_names_and_env_overrides_are_stable() {
    let case = PhaseCMatrixCase {
        scale: BenchmarkScalePoint::TenGiB,
        compression: BenchmarkCompression::Snappy,
        source_layout: BenchmarkSourceLayout::MultiFile,
        native_plan_shape: BenchmarkPlanShape::DirectFriendly,
        expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
    };

    assert_eq!(case.name(), "10gib-snappy-multi_file-direct_friendly-resource_failure");
    assert_eq!(
        case.env_overrides(),
        vec![
            ("CUDF_PARQUET_BENCH_ROWS", "52000000".to_string()),
            ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", TEN_GIB.to_string()),
            ("CUDF_PARQUET_BENCH_COMPRESSION", "snappy".to_string()),
            ("CUDF_PARQUET_BENCH_FILES", "4".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", "direct_friendly".to_string()),
            ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", "resource_failure".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES", "1".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "1".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", "1".to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", "1".to_string()),
        ]
    );

    let mut base = test_config();
    base.benchmark_paths = BenchmarkPathSet::All;
    let configured = case.configured_case(&base, PathBuf::from("target/phase-c-matrix-data"));
    assert_eq!(configured.input_rows, 52_000_000);
    assert_eq!(configured.minimum_file_bytes, TEN_GIB);
    assert_eq!(configured.file_count, 4);
    assert_eq!(configured.benchmark_paths, BenchmarkPathSet::NativeOnly);
    assert_eq!(configured.native_memory_control_policy, forced_resource_failure_policy());
}

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
        parquet_file_bytes: TEN_GIB,
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

    assert!(text.contains("case=phase_c_expected_resource_failure"));
    assert!(text.contains("phase_c_runtime_gate=expected_resource_failure"));
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

#[tokio::test]
async fn supported_sql_benchmark_plan_reports_datafusion_native_accepted() {
    let dir = unique_benchmark_dir(&std::env::temp_dir());
    let config = test_config();
    let path = benchmark_input_path(&dir, &config);
    write_benchmark_parquet_dataset(&path, &config).unwrap();

    let planned = plan_datafusion_native_sql(&path, &benchmark_sql(&config), &config).await.unwrap();

    assert_eq!(planned.plan.name(), "NativeCudfExec");
    assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    assert_eq!(planned.planning_report.accepted_count(), 1);
    assert_eq!(planned.planning_report.rejected_count(), 0);
    let row = &planned.planning_report.rows[0];
    assert_eq!(row.outcome.as_str(), "accepted_replaced");
    assert_eq!(row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(row.post_native_execution_path.as_str(), "native");
    assert!(!row.rejected_before_execution);
    assert!(row.source_diagnostics.iter().any(|diagnostic| diagnostic.direct_device_read));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn rejected_sql_benchmark_plan_reports_reason_without_native_runtime() {
    let dir = unique_benchmark_dir(&std::env::temp_dir());
    let config = test_config();
    let path = benchmark_input_path(&dir, &config);
    write_benchmark_parquet_dataset(&path, &config).unwrap();

    let planned = plan_datafusion_native_sql(&path, rejected_benchmark_sql(), &config).await.unwrap();

    assert_ne!(planned.plan.name(), "NativeCudfExec");
    assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_none());
    assert_eq!(planned.planning_report.accepted_count(), 0);
    assert_eq!(planned.planning_report.rejected_count(), 1);
    let row = &planned.planning_report.rows[0];
    assert_eq!(row.outcome.as_str(), "rejected_before_execution");
    assert!(row.reason_code.is_some());
    assert!(row.rejected_before_execution);
    assert_ne!(row.post_native_execution_path.as_str(), "native");
    assert!(planned.plan.metrics().is_none_or(|metrics| metrics.sum_by_name("host_to_device_uploads").is_none()));

    let mut rejected_config = config.clone();
    rejected_config.expected_outcome = BenchmarkExpectedOutcome::PlanningRejected;
    let report = run_expected_planning_rejection_probe(
        &path,
        parquet_dataset_bytes(&path).unwrap(),
        parquet_dataset_layout(&path).unwrap(),
        "files=4".to_string(),
        &rejected_config,
    )
    .await
    .unwrap();
    let text = report.to_text();
    assert!(text.contains("case=datafusion_native_exec_planning_rejected"));
    assert!(text.contains("native_runtime_executed=false"));
    assert!(text.contains("datafusion_native_exec_planning_outcome=rejected_before_execution"));
    assert!(text.contains("datafusion_native_exec_planning_reason_code="));
    assert!(text.contains("datafusion_native_exec_planning_rejected_before_execution=true"));

    fs::remove_dir_all(dir).unwrap();
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
            outcome: datafusion_cudf::NativeDataFusionPlanningOutcome::AcceptedReplaced,
            reason_code: None,
            unsupported_category: None,
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
        minimum_native_gpu_speedup: config.minimum_native_gpu_speedup,
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

    let matrix_case = PhaseCMatrixCase {
        scale: BenchmarkScalePoint::ThreeGiB,
        compression: BenchmarkCompression::Uncompressed,
        source_layout: BenchmarkSourceLayout::MultiFile,
        native_plan_shape: BenchmarkPlanShape::DefaultSql,
        expected_outcome: BenchmarkExpectedOutcome::Success,
    };
    let matrix_row =
        PhaseCMatrixReportRow::from_success(matrix_case, &report, Path::new("target/phase-c-matrix/logs/case.log"));
    assert!(matrix_row.matches_expected_outcome());
    assert_eq!(matrix_row.outcome.as_str(), "pass");
    assert_eq!(matrix_row.planning_outcome, "accepted_replaced");
    assert!(matrix_row.planning_source_diagnostics.contains("read_columns=4"));
    assert!(matrix_row.planning_source_diagnostics.contains("schema_mapping=direct_by_name"));
    assert_eq!(matrix_row.native_cudf_exec_accepted_count, "1");
    assert_eq!(matrix_row.zero_host_to_device_uploads, "true");
    assert_eq!(matrix_row.one_device_to_host_materialization, "true");

    let matrix_report = PhaseCMatrixReport {
        artifact_dir: PathBuf::from("target/phase-c-matrix"),
        tsv_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.tsv"),
        json_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.json"),
        markdown_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.md"),
        rows: vec![matrix_row],
    };
    let tsv = matrix_report.to_tsv();
    assert!(tsv.starts_with(PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER));
    assert!(tsv.contains("\tpass\taccepted_replaced\tnone\tformat=Parquet"));
    assert!(tsv.contains("read_columns=4,projected_output_columns=3,pure_filter_columns=1"));
    assert!(tsv.contains("\t1\ttrue\ttrue\ttrue\ttrue\t"));
    let json = matrix_report.to_json();
    assert!(json.contains("\"schema_version\": 2"));
    assert!(json.contains("\"case_id\": \"3gib-uncompressed-multi_file-default_sql-success\""));
    assert!(json.contains("\"planning_source_diagnostics\": \"format=Parquet"));
    assert!(json.contains("\"zero_host_to_device_uploads\": \"true\""));
    let markdown = matrix_report.to_markdown();
    assert!(markdown.contains("Decision gate: **pass**"));
    assert!(markdown.contains("3gib-uncompressed-multi_file-default_sql-success"));
    assert!(markdown.contains("schema_mapping=direct_by_name"));
}

#[test]
fn benchmark_payload_pruning_pattern_is_file_local() {
    let config = test_config();
    let file_rows = config.row_group_size * 4;

    assert!(benchmark_payload_value(0, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size - 1, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size, file_rows, &config) < 10);
    assert!(benchmark_payload_value((config.row_group_size * 2) - 1, file_rows, &config) < 10);
    assert!(benchmark_payload_value(config.row_group_size * 2, file_rows, &config) >= 10);
    assert!(benchmark_payload_value((config.row_group_size * 4) - 1, file_rows, &config) >= 10);
}

fn test_config() -> LocalParquetBenchmarkConfig {
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

fn test_collect_measurement(wall_time: Duration) -> CollectMeasurement {
    CollectMeasurement { wall_time, output_batches: 1, output_rows: 2, output_bytes: 64, batches: Vec::new() }
}

fn test_gpu_measurement(
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

fn test_native_datafusion_planning_report() -> NativeDataFusionPlanningReport {
    NativeDataFusionPlanningReport {
        rows: vec![datafusion_cudf::NativeDataFusionPlanningRow {
            query_id: "local_parquet_benchmark".to_string(),
            physical_root: "SortExec".to_string(),
            physical_shape: "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
            native_candidate_shape: Some(
                "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
            ),
            outcome: datafusion_cudf::NativeDataFusionPlanningOutcome::AcceptedReplaced,
            reason_code: None,
            unsupported_category: None,
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

fn test_native_metrics() -> LargerIslandBenchmarkMetrics {
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

fn assert_key_order(text: &str, before: &str, after: &str) {
    let before_index = text.find(before).unwrap_or_else(|| panic!("missing key {before}"));
    let after_index = text.find(after).unwrap_or_else(|| panic!("missing key {after}"));
    assert!(before_index < after_index, "expected {before} to appear before {after}");
}
