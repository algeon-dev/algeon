use super::*;

pub(super) struct ConfiguredBenchmarkCaseReport {
    pub(super) path: PathBuf,
    pub(super) parquet_file_bytes: u64,
    pub(super) dataset_layout: ParquetDatasetLayout,
    pub(super) source_diagnostic: String,
    pub(super) report: BenchmarkCaseReport,
}

#[derive(Debug)]
pub(super) enum BenchmarkCaseReport {
    Success(Box<DirectParquetBenchmarkReport>),
    PlanningRejected(ExpectedPlanningRejectedReport),
    ResourceFailure(ExpectedResourceFailureReport),
}

impl BenchmarkCaseReport {
    pub(super) fn to_text(&self) -> String {
        match self {
            Self::Success(report) => report.to_text(),
            Self::PlanningRejected(report) => report.to_text(),
            Self::ResourceFailure(report) => report.to_text(),
        }
    }
}

pub(super) async fn run_configured_benchmark_case(
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ConfiguredBenchmarkCaseReport> {
    let dir = config.output_dir.clone().unwrap_or_else(default_benchmark_dir);
    fs::create_dir_all(&dir)?;
    let path = benchmark_input_path(&dir, config);

    write_benchmark_parquet_dataset(&path, config)?;
    let parquet_file_bytes = parquet_dataset_bytes(&path)?;
    validate_file_size(parquet_file_bytes, config)?;
    let dataset_layout = parquet_dataset_layout(&path)?;
    validate_dataset_layout(&dataset_layout, config)?;

    let source_diagnostic = local_parquet_scan_diagnostic(&path).await?;
    let report = match config.expected_outcome {
        BenchmarkExpectedOutcome::Success => BenchmarkCaseReport::Success(Box::new(
            run_direct_parquet_benchmark(&path, parquet_file_bytes, dataset_layout, source_diagnostic.clone(), config)
                .await?,
        )),
        BenchmarkExpectedOutcome::PlanningRejected => BenchmarkCaseReport::PlanningRejected(
            run_expected_planning_rejection_probe(
                &path,
                parquet_file_bytes,
                dataset_layout,
                source_diagnostic.clone(),
                config,
            )
            .await?,
        ),
        BenchmarkExpectedOutcome::ResourceFailure => BenchmarkCaseReport::ResourceFailure(
            run_expected_resource_failure_probe(
                &path,
                parquet_file_bytes,
                dataset_layout,
                source_diagnostic.clone(),
                config,
            )
            .await?,
        ),
    };

    Ok(ConfiguredBenchmarkCaseReport { path, parquet_file_bytes, dataset_layout, source_diagnostic, report })
}

pub(super) fn benchmark_case_prologue(
    report: &ConfiguredBenchmarkCaseReport,
    config: &LocalParquetBenchmarkConfig,
) -> String {
    let mut text = String::new();
    push_line(&mut text, format_args!("local_parquet_benchmark"));
    push_line(&mut text, format_args!("parquet_path={}", report.path.display()));
    push_line(&mut text, format_args!("source_diagnostic={}", report.source_diagnostic));
    push_line(&mut text, format_args!("parquet_file_bytes={}", report.parquet_file_bytes));
    push_line(&mut text, format_args!("input_rows={}", config.input_rows));
    push_line(&mut text, format_args!("parquet_files={}", config.file_count));
    push_line(&mut text, format_args!("measure_columns={}", config.measure_columns));
    push_line(&mut text, format_args!("projected_measure_pairs={}", config.projected_measure_pairs()));
    push_line(&mut text, format_args!("group_count={}", config.group_count));
    push_line(&mut text, format_args!("batch_size={}", config.batch_size));
    push_line(&mut text, format_args!("row_group_size={}", config.row_group_size));
    push_line(&mut text, format_args!("parquet_row_groups={}", report.dataset_layout.row_group_count));
    push_line(&mut text, format_args!("minimum_native_gpu_speedup={:.3}", config.minimum_native_gpu_speedup));
    push_line(&mut text, format_args!("native_memory_control_policy={}", config.native_memory_control_policy.as_str()));
    push_line(&mut text, format_args!("benchmark_paths={}", config.benchmark_paths.name()));
    push_line(&mut text, format_args!("compression={}", config.compression.name()));
    push_line(&mut text, format_args!("native_plan_shape={}", config.native_plan_shape.name()));
    push_line(&mut text, format_args!("expected_outcome={}", config.expected_outcome.name()));
    push_line(&mut text, format_args!("phase_c_decision_matrix_cases={}", phase_c_decision_matrix().len()));
    push_line(&mut text, format_args!("limit={}", config.limit));
    push_line(&mut text, format_args!("warmups={}", config.warmup_iterations));
    push_line(&mut text, format_args!("iterations={}", config.iterations));
    text
}
