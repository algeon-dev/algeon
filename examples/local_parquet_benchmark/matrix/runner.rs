use super::*;

pub(crate) async fn run_phase_c_matrix(
    base_config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<PhaseCMatrixReport> {
    let artifact_dir = phase_c_matrix_artifact_dir();
    let logs_dir = artifact_dir.join("logs");
    let data_dir = phase_c_matrix_data_dir();
    fs::create_dir_all(&logs_dir)?;
    fs::create_dir_all(&data_dir)?;

    let mut rows = Vec::new();
    for case in phase_c_decision_matrix() {
        let case_id = case.name();
        let log_path = logs_dir.join(format!("{case_id}.log"));
        let case_config = case.configured_case(base_config, data_dir.join(&case_id));
        match run_configured_benchmark_case(&case_config).await {
            Ok(case_report) => {
                let mut log = benchmark_case_prologue(&case_report, &case_config);
                log.push('\n');
                log.push_str(&case_report.report.to_text());
                fs::write(&log_path, log)?;
                rows.push(PhaseCMatrixReportRow::from_case_report(case, &case_report, &log_path));
            },
            Err(error) => {
                fs::write(&log_path, format!("case={case_id}\noutcome=runtime_failure\nerror={error}\n"))?;
                rows.push(PhaseCMatrixReportRow::runtime_failure(case, &error, &log_path));
            },
        }
    }

    let report = PhaseCMatrixReport {
        tsv_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.tsv"),
        json_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.json"),
        markdown_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.md"),
        artifact_dir,
        rows,
    };
    fs::write(&report.tsv_path, report.to_tsv())?;
    fs::write(&report.json_path, report.to_json())?;
    fs::write(&report.markdown_path, report.to_markdown())?;
    Ok(report)
}

pub(crate) fn phase_c_matrix_artifact_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/phase-c-matrix"))
}

pub(crate) fn phase_c_matrix_data_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PHASE_C_MATRIX_DATA_DIR))
}
