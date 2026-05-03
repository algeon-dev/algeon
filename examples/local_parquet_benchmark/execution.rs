use super::*;

#[derive(Debug)]
pub(super) struct CollectMeasurement {
    pub(super) wall_time: Duration,
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) output_bytes: usize,
    pub(super) batches: Vec<RecordBatch>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuCollectMeasurement {
    pub(super) wall_time: Duration,
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) output_bytes: usize,
    pub(super) metrics: LargerIslandBenchmarkMetrics,
    pub(super) planning: LargerIslandBenchmarkPlanning,
    pub(super) native_datafusion_planning_report: Option<NativeDataFusionPlanningReport>,
    pub(super) plan_one_line: String,
    pub(super) source_physical_plan_one_line: Option<String>,
    pub(super) native_candidate_shapes: Vec<String>,
    pub(super) read_columns: Vec<String>,
    pub(super) batches: Vec<RecordBatch>,
}

#[derive(Debug)]
pub(super) struct DataFusionNativePlannedExec {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) planning: LargerIslandBenchmarkPlanning,
    pub(super) planning_report: NativeDataFusionPlanningReport,
    pub(super) native_candidate_shapes: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum BenchmarkPath {
    CpuComparable,
    CpuDefault,
    GpuDirect,
    DataFusionNativeExec,
    ManualNativeLowering,
}

impl BenchmarkPath {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::CpuComparable => "cpu_comparable",
            Self::CpuDefault => "cpu_default",
            Self::GpuDirect => "gpu_direct",
            Self::DataFusionNativeExec => "datafusion_native_exec",
            Self::ManualNativeLowering => "manual_native_lowering",
        }
    }
}

pub(super) async fn run_direct_parquet_benchmark(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<DirectParquetBenchmarkReport> {
    for _ in 0..config.warmup_iterations {
        let _ = collect_cpu_comparable_once(path, config).await?;
        let _ = collect_cpu_default_once(path, config).await?;
        if config.benchmark_paths.includes_gpu_direct() {
            let _ = collect_gpu_direct_once(path, config).await?;
        }
        let _ = collect_datafusion_native_exec_once(path, config).await?;
        let _ = collect_manual_native_lowering_once(path, config).await?;
    }

    let mut cpu_comparable_samples = Vec::with_capacity(config.iterations);
    let mut cpu_default_samples = Vec::with_capacity(config.iterations);
    let mut gpu_samples = Vec::with_capacity(config.iterations);
    let mut datafusion_native_exec_samples = Vec::with_capacity(config.iterations);
    let mut manual_native_lowering_samples = Vec::with_capacity(config.iterations);
    let mut gpu_metric_samples = Vec::with_capacity(config.iterations);
    let mut datafusion_native_exec_metric_samples = Vec::with_capacity(config.iterations);
    let mut manual_native_lowering_metric_samples = Vec::with_capacity(config.iterations);
    let mut final_cpu_comparable = None;
    let mut final_cpu_default = None;
    let mut final_gpu = None;
    let mut final_datafusion_native_exec = None;
    let mut final_manual_native_lowering = None;

    for iteration in 0..config.iterations {
        for path_kind in measured_iteration_order(iteration, config.benchmark_paths) {
            match path_kind {
                BenchmarkPath::CpuComparable => {
                    let measurement = collect_cpu_comparable_once(path, config).await?;
                    cpu_comparable_samples.push(measurement.wall_time);
                    final_cpu_comparable = Some(measurement);
                },
                BenchmarkPath::CpuDefault => {
                    let measurement = collect_cpu_default_once(path, config).await?;
                    cpu_default_samples.push(measurement.wall_time);
                    final_cpu_default = Some(measurement);
                },
                BenchmarkPath::GpuDirect => {
                    let measurement = collect_gpu_direct_once(path, config).await?;
                    gpu_samples.push(measurement.wall_time);
                    gpu_metric_samples.push(measurement.metrics.clone());
                    final_gpu = Some(measurement);
                },
                BenchmarkPath::DataFusionNativeExec => {
                    let measurement = collect_datafusion_native_exec_once(path, config).await?;
                    datafusion_native_exec_samples.push(measurement.wall_time);
                    datafusion_native_exec_metric_samples.push(measurement.metrics.clone());
                    final_datafusion_native_exec = Some(measurement);
                },
                BenchmarkPath::ManualNativeLowering => {
                    let measurement = collect_manual_native_lowering_once(path, config).await?;
                    manual_native_lowering_samples.push(measurement.wall_time);
                    manual_native_lowering_metric_samples.push(measurement.metrics.clone());
                    final_manual_native_lowering = Some(measurement);
                },
            }
        }
    }

    let final_cpu_comparable =
        final_cpu_comparable.expect("validated benchmark config requires measured comparable CPU iterations");
    let final_cpu_default =
        final_cpu_default.expect("validated benchmark config requires measured default CPU iterations");
    let mut final_datafusion_native_exec = final_datafusion_native_exec
        .expect("validated benchmark config requires measured DataFusion native iterations");
    let final_manual_native_lowering =
        final_manual_native_lowering.expect("validated benchmark config requires measured manual native iterations");
    if let Some(final_gpu) = &final_gpu {
        compare_outputs(&final_cpu_comparable.batches, &final_gpu.batches)?;
        compare_outputs(&final_cpu_default.batches, &final_gpu.batches)?;
        compare_outputs(&final_gpu.batches, &final_datafusion_native_exec.batches)?;
        compare_outputs(&final_gpu.batches, &final_manual_native_lowering.batches)?;
        validate_gpu_direct_measurement(final_gpu, parquet_file_bytes, config)?;
    }
    compare_outputs(&final_cpu_comparable.batches, &final_datafusion_native_exec.batches)?;
    compare_outputs(&final_cpu_default.batches, &final_datafusion_native_exec.batches)?;
    compare_outputs(&final_datafusion_native_exec.batches, &final_manual_native_lowering.batches)?;
    validate_datafusion_native_exec_measurement(&final_datafusion_native_exec, parquet_file_bytes, config)?;
    validate_manual_native_lowering_measurement(&final_manual_native_lowering, parquet_file_bytes, config)?;
    final_datafusion_native_exec.metrics.phase_c_runtime_gate =
        final_datafusion_native_exec.metrics.assess_phase_c_runtime_gate(true).code();
    let cpu_comparable = TimedBenchmarkRun::from_samples(cpu_comparable_samples.iter().copied());
    let cpu_default = TimedBenchmarkRun::from_samples(cpu_default_samples.iter().copied());
    let gpu = config
        .benchmark_paths
        .includes_gpu_direct()
        .then(|| TimedBenchmarkRun::from_samples(gpu_samples.iter().copied()));
    let datafusion_native_exec = TimedBenchmarkRun::from_samples(datafusion_native_exec_samples.iter().copied());
    let manual_native_lowering = TimedBenchmarkRun::from_samples(manual_native_lowering_samples.iter().copied());
    validate_phase_b_runtime_gate(
        &cpu_comparable,
        &datafusion_native_exec,
        &final_datafusion_native_exec,
        parquet_file_bytes,
        dataset_layout,
        config,
    )?;
    validate_phase_c_runtime_gate(&final_datafusion_native_exec, config)?;

    Ok(DirectParquetBenchmarkReport {
        parquet_file_bytes,
        parquet_row_groups: dataset_layout.row_group_count,
        input_rows: config.input_rows,
        file_count: config.file_count,
        measure_columns: config.measure_columns,
        batch_size: config.batch_size,
        row_group_size: config.row_group_size,
        group_count: config.group_count,
        warmup_iterations: config.warmup_iterations,
        iterations: config.iterations,
        limit: config.limit,
        minimum_native_gpu_speedup: config.minimum_native_gpu_speedup,
        benchmark_paths: config.benchmark_paths,
        compression: config.compression,
        native_plan_shape: config.native_plan_shape,
        expected_outcome: config.expected_outcome,
        measured_run_order: measured_run_order_summary(config.iterations, config.benchmark_paths),
        cpu_comparable,
        cpu_default,
        gpu,
        datafusion_native_exec,
        manual_native_lowering,
        cpu_comparable_samples,
        cpu_default_samples,
        gpu_samples,
        datafusion_native_exec_samples,
        manual_native_lowering_samples,
        gpu_metric_samples,
        datafusion_native_exec_metric_samples,
        manual_native_lowering_metric_samples,
        final_cpu_comparable,
        final_cpu_default,
        final_gpu,
        final_datafusion_native_exec,
        final_manual_native_lowering,
        source_diagnostic,
    })
}

pub(super) async fn collect_cpu_comparable_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<CollectMeasurement> {
    let plan = build_cpu_comparable_plan(path, config).await?;
    collect_cpu_plan(plan).await
}

pub(super) async fn collect_cpu_default_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<CollectMeasurement> {
    let plan = build_cpu_default_plan(path, config).await?;
    collect_cpu_plan(plan).await
}

pub(super) async fn collect_cpu_plan(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<CollectMeasurement> {
    let started = Instant::now();
    let batches = collect(plan, Arc::new(TaskContext::default())).await?;
    Ok(measure_cpu_output(started.elapsed(), batches))
}

pub(super) async fn collect_gpu_direct_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let optimizer = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_min_operators_per_island(2);
    let source = parquet_scan_plan(path).await?;
    let plan = build_gpu_friendly_plan(source, config)?;
    let optimized = optimize_plan(plan, &optimizer)?;
    let planning = LargerIslandBenchmarkPlanning::from_report(&optimized.report);
    if planning.rewritten_count != 1 {
        return Err(plan_error(format!(
            "direct Parquet benchmark expected exactly one rewritten GPU island, got {}",
            planning.rewritten_count
        )));
    }

    let gpu_island = optimized.plan.as_any().downcast_ref::<CudfGpuIslandExec>().ok_or_else(|| {
        plan_error(format!("optimized plan root was {}, expected CudfGpuIslandExec", optimized.plan.name()))
    })?;
    if gpu_island.input_display_name() != "local_parquet_scan"
        || gpu_island.input_boundary_name() != "cudf_parquet_reader"
    {
        return Err(plan_error(format!(
            "optimized GPU plan did not use direct local Parquet input: input={}, input_boundary={}",
            gpu_island.input_display_name(),
            gpu_island.input_boundary_name()
        )));
    }
    let read_columns = gpu_island.local_parquet_read_columns().unwrap_or_default();
    let plan_one_line = displayable(optimized.plan.as_ref()).one_line().to_string();

    let started = Instant::now();
    let batches = collect(Arc::clone(&optimized.plan), Arc::new(TaskContext::default())).await?;
    let wall_time = started.elapsed();
    let metrics = optimized
        .plan
        .metrics()
        .map(|metrics| LargerIslandBenchmarkMetrics::from_metrics(&metrics))
        .unwrap_or_else(|| LargerIslandBenchmarkMetrics::from_metrics(&MetricsSet::new()));
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics,
        planning,
        native_datafusion_planning_report: None,
        plan_one_line,
        source_physical_plan_one_line: None,
        native_candidate_shapes: Vec::new(),
        read_columns,
        batches,
    })
}

pub(super) async fn collect_datafusion_native_exec_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let planned = plan_datafusion_native_sql(path, &benchmark_sql(config), config).await?;
    let DataFusionNativePlannedExec { plan, planning, planning_report, native_candidate_shapes } = planned;
    let Some(native_exec) = plan.as_any().downcast_ref::<NativeCudfExec>() else {
        return Err(plan_error(format!(
            "datafusion_native_exec planning rejected NativeCudfExec path: root={}, accepted_count={}, rejected_count={}, candidates={}",
            plan.name(),
            planning_report.accepted_count(),
            planning_report.rejected_count(),
            native_candidate_shapes.join(";")
        )));
    };
    let read_columns = native_local_parquet_read_columns(native_exec.native_plan())?;
    let source_physical_plan_one_line = planning_report
        .rows
        .first()
        .map(|row| row.physical_shape.clone())
        .unwrap_or_else(|| "not_available".to_string());
    let plan_one_line = displayable(plan.as_ref()).one_line().to_string();

    let started = Instant::now();
    let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
    let wall_time = started.elapsed();
    let metrics = plan.metrics().unwrap_or_else(MetricsSet::new);
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics: benchmark_metrics_from_datafusion_native(&metrics, config.native_memory_control_policy),
        planning,
        native_datafusion_planning_report: Some(planning_report),
        plan_one_line,
        source_physical_plan_one_line: Some(source_physical_plan_one_line),
        native_candidate_shapes,
        read_columns,
        batches,
    })
}

pub(super) async fn plan_datafusion_native_sql(
    path: &Path,
    sql: &str,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<DataFusionNativePlannedExec> {
    let native_execution_config = native_datafusion_execution_config(config)?;
    let optimizer_config =
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_execution_config);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_datafusion_session_config())
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    let path_text =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    ctx.register_parquet("bench", path_text, ParquetReadOptions::default()).await?;
    let df = ctx.sql(sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report = NativeDataFusionPlanningReport::from_retained_rule(
        "local_parquet_benchmark",
        Arc::clone(&plan),
        native_rule.as_ref(),
    );
    let planning = native_rule
        .last_report()
        .map(|report| LargerIslandBenchmarkPlanning::from_report(&report))
        .unwrap_or_else(|| LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::new()));
    let native_candidate_shapes = native_datafusion_candidate_shapes(&planning_report);
    Ok(DataFusionNativePlannedExec { plan, planning, planning_report, native_candidate_shapes })
}

pub(super) async fn collect_manual_native_lowering_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let plan = build_native_physical_plan(path, config).await?;
    let source_physical_plan_one_line = displayable(plan.as_ref()).one_line().to_string();
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan));
    let planning = LargerIslandBenchmarkPlanning::from_report(&lowering.report);
    let native_candidate_shapes = native_candidate_shapes(&lowering.report);
    if planning.rewritten_count != 1 || lowering.plans.len() != 1 {
        return Err(plan_error(format!(
            "native default SQL Parquet benchmark expected exactly one lowered GPU query plan, got rewritten_count={}, native_plan_count={}, candidates={}",
            planning.rewritten_count,
            lowering.plans.len(),
            native_candidate_shapes.join(";")
        )));
    }

    let native_plan =
        lowering.plans.into_iter().next().expect("validated native lowering result contains exactly one plan");
    let read_columns = native_local_parquet_read_columns(&native_plan)?;
    let plan_one_line = native_plan_one_line(&native_plan);

    let started = Instant::now();
    let context = NativeExecutionContext::builder()
        .with_memory_control_policy(config.native_memory_control_policy)
        .build()
        .map_err(native_query_error)?;
    let result = execute_native_query_plan(&context, &native_plan).map_err(native_query_error)?;
    let wall_time = started.elapsed();
    let (output, native_metrics) = result.into_parts();
    let NativeQueryOutput::HostArrow(batch) = output else {
        return Err(plan_error("native default SQL Parquet benchmark expected a HostArrow result sink"));
    };
    let batches = vec![batch];
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics: benchmark_metrics_from_native(&native_metrics),
        planning,
        native_datafusion_planning_report: None,
        plan_one_line,
        source_physical_plan_one_line: Some(source_physical_plan_one_line),
        native_candidate_shapes,
        read_columns,
        batches,
    })
}

pub(super) async fn build_native_physical_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    match config.native_plan_shape {
        BenchmarkPlanShape::DefaultSql => build_cpu_default_plan(path, config).await,
        BenchmarkPlanShape::DirectFriendly => build_cpu_comparable_plan(path, config).await,
    }
}

pub(super) fn native_datafusion_execution_config(
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<NativeCudfExecutionConfig> {
    match config.native_memory_control_policy {
        NativeMemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } => Ok(NativeCudfExecutionConfig::bounded(
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        )),
        NativeMemoryControlPolicy::Unbounded => {
            Err(plan_error("datafusion_native_exec requires CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY=bounded"))
        },
    }
}

pub(super) fn native_datafusion_candidate_shapes(report: &NativeDataFusionPlanningReport) -> Vec<String> {
    report
        .rows
        .iter()
        .map(|row| {
            format!(
                "{}:{}:{}",
                row.outcome.as_str(),
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.reason_code.as_deref().unwrap_or("none")
            )
        })
        .collect()
}

pub(super) fn native_candidate_shapes(report: &datafusion_cudf::PlanningReport) -> Vec<String> {
    report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => format!("rewritten:{}", candidate.shape),
            CandidateOutcome::Rejected { candidate, reason } => {
                format!("rejected:{}:{}", candidate.shape, reason.code())
            },
        })
        .collect()
}

pub(super) fn native_local_parquet_read_columns(plan: &NativeQueryPlan) -> DataFusionResult<Vec<String>> {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            NativePlanNode::Source(NativeSourceNode::LocalParquet(source)) => Some(source.read_columns.clone()),
            _ => None,
        })
        .ok_or_else(|| plan_error("native query plan did not contain a local Parquet source"))
}

pub(super) fn native_plan_one_line(plan: &NativeQueryPlan) -> String {
    plan.to_string().lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" | ")
}

pub(super) fn measured_iteration_order(iteration: usize, path_set: BenchmarkPathSet) -> Vec<BenchmarkPath> {
    match path_set {
        BenchmarkPathSet::NativeOnly => match iteration % 2 {
            0 => vec![
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
            ],
            _ => vec![
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
            ],
        },
        BenchmarkPathSet::All => match iteration % 4 {
            0 => vec![
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
            ],
            1 => vec![
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
            ],
            2 => vec![
                BenchmarkPath::GpuDirect,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
            ],
            _ => vec![
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::GpuDirect,
            ],
        },
    }
}

pub(super) fn measured_run_order_summary(iterations: usize, path_set: BenchmarkPathSet) -> String {
    (0..iterations)
        .map(|iteration| {
            measured_iteration_order(iteration, path_set)
                .into_iter()
                .map(BenchmarkPath::label)
                .collect::<Vec<_>>()
                .join(">")
        })
        .collect::<Vec<_>>()
        .join(";")
}
