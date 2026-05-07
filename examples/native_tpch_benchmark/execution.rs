use super::*;

#[derive(Debug, Clone)]
pub(super) struct QueryMeasurement {
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) output_bytes: usize,
    pub(super) output_fingerprint: String,
}

#[derive(Debug, Clone)]
pub(super) struct TimedQueryRun {
    pub(super) samples: Vec<Duration>,
    pub(super) final_measurement: QueryMeasurement,
}

#[derive(Debug, Clone)]
pub(super) struct TimedNativeQueryRun {
    pub(super) samples: Vec<Duration>,
    pub(super) final_measurement: QueryMeasurement,
    pub(super) metrics: NativeDataFusionExecutionMetrics,
    pub(super) raw_metrics: MetricsSet,
}

#[derive(Debug, Clone)]
pub(super) struct NativeRunFailure {
    pub(super) runtime_started: bool,
    pub(super) message: String,
}

#[derive(Debug, Clone)]
pub(super) struct PlannedNativeQuery {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) planning_row: NativeDataFusionPlanningRow,
}

pub(super) async fn plan_native_query(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> DataFusionResult<PlannedNativeQuery> {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(config.native_execution_config());
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    dataset::register_tpch_tables(&ctx, metadata).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report =
        NativeDataFusionPlanningReport::from_retained_rule(query.id, Arc::clone(&plan), native_rule.as_ref());
    let planning_row = planning_report
        .rows
        .into_iter()
        .next()
        .ok_or_else(|| plan_error(format!("native planning report was empty for {}", query.id)))?;
    Ok(PlannedNativeQuery { plan, planning_row })
}

pub(super) async fn run_cpu_query_iterations(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> DataFusionResult<TimedQueryRun> {
    for _ in 0..config.warmup_iterations {
        let _ = collect_cpu_query_once(metadata, query).await?;
    }

    let mut samples = Vec::with_capacity(config.iterations);
    let mut final_measurement = None;
    for _ in 0..config.iterations {
        let started = Instant::now();
        let batches = collect_cpu_query_once(metadata, query).await?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&batches)?);
    }
    Ok(TimedQueryRun {
        samples,
        final_measurement: final_measurement.expect("validated benchmark config requires iterations"),
    })
}

pub(super) async fn run_native_query_iterations(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> Result<TimedNativeQueryRun, NativeRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned = plan_native_query(metadata, query, config).await.map_err(native_not_started)?;
        if planned.planning_row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced {
            let _ = collect_native_plan_once(planned.plan).await.map_err(native_started)?;
        }
    }

    let mut samples = Vec::with_capacity(config.iterations);
    let mut final_measurement = None;
    let mut final_metrics = None;
    for _ in 0..config.iterations {
        let planned = plan_native_query(metadata, query, config).await.map_err(native_not_started)?;
        if planned.planning_row.outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
            return Err(NativeRunFailure {
                runtime_started: false,
                message: format!("{} stopped being a native accepted query during execution", query.id),
            });
        }
        let started = Instant::now();
        let (batches, metrics) = collect_native_plan_once(Arc::clone(&planned.plan)).await.map_err(native_started)?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&batches).map_err(native_started)?);
        final_metrics = Some(metrics);
    }
    let raw_metrics = final_metrics.expect("validated benchmark config requires iterations");
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&raw_metrics);
    Ok(TimedNativeQueryRun {
        samples,
        final_measurement: final_measurement.expect("validated benchmark config requires iterations"),
        metrics,
        raw_metrics,
    })
}

fn native_started(error: DataFusionError) -> NativeRunFailure {
    NativeRunFailure { runtime_started: true, message: error.to_string() }
}

fn native_not_started(error: DataFusionError) -> NativeRunFailure {
    NativeRunFailure { runtime_started: false, message: error.to_string() }
}

async fn collect_cpu_query_once(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
) -> DataFusionResult<Vec<RecordBatch>> {
    let ctx = SessionContext::new_with_config(cpu_session_config());
    dataset::register_tpch_tables(&ctx, metadata).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    collect(plan, Arc::new(TaskContext::default())).await
}

async fn collect_native_plan_once(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<(Vec<RecordBatch>, MetricsSet)> {
    let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
    let metrics = plan.metrics().unwrap_or_default();
    Ok((batches, metrics))
}

fn cpu_session_config() -> SessionConfig {
    tpch_report_session_config()
}

fn measure_output(batches: &[RecordBatch]) -> DataFusionResult<QueryMeasurement> {
    Ok(QueryMeasurement {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
        output_fingerprint: validation::fingerprint_batches(batches)?,
    })
}
