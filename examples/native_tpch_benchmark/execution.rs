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

#[derive(Clone)]
pub(super) struct PlannedCpuQuery {
    _ctx: SessionContext,
    plan: Arc<dyn ExecutionPlan>,
}

#[derive(Debug, Clone)]
pub(super) struct CpuRunFailure {
    pub(super) runtime_started: bool,
    pub(super) message: String,
    pub(super) details: FailureDetails,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct FailureDetails {
    pub(super) code: Option<String>,
    pub(super) kind: Option<String>,
    pub(super) status: Option<String>,
    pub(super) operation: Option<String>,
    pub(super) facts: BTreeMap<String, String>,
}

impl FailureDetails {
    pub(super) fn from_datafusion_error(error: &DataFusionError) -> Self {
        adapter_error_from_datafusion(error).map(Self::from_adapter_error).unwrap_or_default()
    }

    fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
        }
    }
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
    pub(super) details: FailureDetails,
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
    let planning_row = select_benchmark_planning_row(query.id, planning_report.rows)?;
    Ok(PlannedNativeQuery { plan, planning_row })
}

pub(super) fn select_benchmark_planning_row(
    query_id: &str,
    rows: Vec<NativeDataFusionPlanningRow>,
) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        0 => Err(plan_error(format!("native planning report was empty for {query_id}"))),
        1 => Ok(rows.into_iter().next().expect("single planning row must exist")),
        count => Err(plan_error(format!(
            "native TPC-H benchmark requires exactly one native planning row per query; {query_id} produced {count}"
        ))),
    }
}

pub(super) fn validate_decision_grade_native_planning_row(
    query_id: &str,
    row: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    if row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        && row.post_native_execution_path != NativeDataFusionPostNativeExecutionPath::Native
    {
        return Err(plan_error(format!(
            "native TPC-H benchmark requires post_native_execution_path=native for accepted query {query_id}; observed {}",
            row.post_native_execution_path.as_str()
        )));
    }
    Ok(())
}

fn validate_replanned_native_row(
    query_id: &str,
    expected: &NativeDataFusionPlanningRow,
    actual: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    validate_decision_grade_native_planning_row(query_id, actual)?;
    if actual != expected {
        return Err(plan_error(format!(
            "native TPC-H benchmark planning row changed during execution for {query_id}; refusing to merge metrics with a different planning contract"
        )));
    }
    Ok(())
}

pub(super) async fn run_cpu_query_iterations(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> Result<TimedQueryRun, CpuRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned = plan_cpu_query(metadata, query).await?;
        let _ = collect_cpu_plan_once(planned).await?;
    }

    let mut samples = Vec::with_capacity(config.iterations);
    let mut final_measurement = None;
    for _ in 0..config.iterations {
        let planned = plan_cpu_query(metadata, query).await?;
        let started = Instant::now();
        let batches = collect_cpu_plan_once(planned).await?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&batches).map_err(cpu_started)?);
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
    expected_planning_row: &NativeDataFusionPlanningRow,
) -> Result<TimedNativeQueryRun, NativeRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned = plan_native_query(metadata, query, config).await.map_err(native_not_started)?;
        if planned.planning_row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced {
            validate_replanned_native_row(query.id, expected_planning_row, &planned.planning_row)
                .map_err(native_not_started)?;
            let _ = collect_native_plan_once(planned.plan).await.map_err(native_started)?;
        } else {
            return Err(NativeRunFailure {
                runtime_started: false,
                message: format!("{} stopped being a native accepted query during warmup", query.id),
                details: FailureDetails::default(),
            });
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
                details: FailureDetails::default(),
            });
        }
        validate_replanned_native_row(query.id, expected_planning_row, &planned.planning_row)
            .map_err(native_not_started)?;
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
    let details = FailureDetails::from_datafusion_error(&error);
    NativeRunFailure { runtime_started: true, message: error.to_string(), details }
}

fn native_not_started(error: DataFusionError) -> NativeRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    NativeRunFailure { runtime_started: false, message: error.to_string(), details }
}

async fn plan_cpu_query(metadata: &TpchDatasetMetadata, query: &TpchQuery) -> Result<PlannedCpuQuery, CpuRunFailure> {
    let ctx = SessionContext::new_with_config(cpu_session_config());
    dataset::register_tpch_tables(&ctx, metadata).await.map_err(cpu_not_started)?;
    let df = ctx.sql(query.sql).await.map_err(cpu_not_started)?;
    let plan = df.create_physical_plan().await.map_err(cpu_not_started)?;
    Ok(PlannedCpuQuery { _ctx: ctx, plan })
}

async fn collect_cpu_plan_once(planned: PlannedCpuQuery) -> Result<Vec<RecordBatch>, CpuRunFailure> {
    let PlannedCpuQuery { _ctx, plan } = planned;
    collect(plan, Arc::new(TaskContext::default())).await.map_err(cpu_started)
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

fn cpu_started(error: DataFusionError) -> CpuRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    CpuRunFailure { runtime_started: true, message: error.to_string(), details }
}

fn cpu_not_started(error: DataFusionError) -> CpuRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    CpuRunFailure { runtime_started: false, message: error.to_string(), details }
}

fn adapter_error_from_datafusion(error: &DataFusionError) -> Option<&AdapterError> {
    match error {
        DataFusionError::External(source) => source.downcast_ref::<AdapterError>(),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            adapter_error_from_datafusion(source)
        },
        DataFusionError::Collection(errors) => errors.iter().find_map(adapter_error_from_datafusion),
        DataFusionError::Shared(source) => adapter_error_from_datafusion(source),
        _ => None,
    }
}
