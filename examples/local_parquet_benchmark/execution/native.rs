use super::*;

#[derive(Debug)]
pub(crate) struct DataFusionNativePlannedExec {
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    pub(crate) planning: LargerIslandBenchmarkPlanning,
    pub(crate) planning_report: NativeDataFusionPlanningReport,
    pub(crate) native_candidate_shapes: Vec<String>,
}

pub(crate) async fn collect_datafusion_native_exec_once(
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

pub(crate) async fn plan_datafusion_native_sql(
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

pub(crate) async fn collect_manual_native_lowering_once(
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

pub(crate) async fn build_native_physical_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    match config.native_plan_shape {
        BenchmarkPlanShape::DefaultSql => build_cpu_default_plan(path, config).await,
        BenchmarkPlanShape::DirectFriendly => build_cpu_comparable_plan(path, config).await,
    }
}

pub(crate) fn native_datafusion_execution_config(
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

pub(crate) fn native_datafusion_candidate_shapes(report: &NativeDataFusionPlanningReport) -> Vec<String> {
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

pub(crate) fn native_candidate_shapes(report: &datafusion_cudf::PlanningReport) -> Vec<String> {
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

pub(crate) fn native_local_parquet_read_columns(plan: &NativeQueryPlan) -> DataFusionResult<Vec<String>> {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            NativePlanNode::Source(source_node) => match source_node.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source.read_columns.clone()),
                NativeSourceNode::GpuResident(_) => None,
            },
            _ => None,
        })
        .ok_or_else(|| plan_error("native query plan did not contain a local Parquet source"))
}

pub(crate) fn native_plan_one_line(plan: &NativeQueryPlan) -> String {
    plan.to_string().lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" | ")
}
