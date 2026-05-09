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
