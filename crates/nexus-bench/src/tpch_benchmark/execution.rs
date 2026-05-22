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
        error.find_nexus_error().map(Self::from_adapter_error).unwrap_or_default()
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
    pub(super) timing_metrics: NativeTpchTimingMetrics,
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
    plan_native_query_with_resources(metadata, query, config, &NativeTpchRuntimeResources::unavailable()).await
}

async fn plan_native_query_with_resources(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
    resources: &NativeTpchRuntimeResources,
) -> DataFusionResult<PlannedNativeQuery> {
    let optimizer_config = native_optimizer_config(config, resources);
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let builder = SessionStateBuilder::new_with_default_features().with_config(native_session_config(config));
    let builder = resources.install_on(builder)?;
    let state = builder.with_nexus_native_rule(Arc::clone(&native_rule)).build();
    let ctx = SessionContext::new_with_state(state);
    dataset::register_tpch_tables(&ctx, metadata).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report =
        NativeDataFusionPlanningReport::from_retained_rule(query.id, Arc::clone(&plan), native_rule.as_ref());
    let planning_row = select_benchmark_planning_row(query.id, planning_report.rows)?;
    Ok(PlannedNativeQuery { plan, planning_row })
}

pub(super) fn native_optimizer_config(
    config: &NativeTpchBenchmarkConfig,
    resources: &NativeTpchRuntimeResources,
) -> NexusNativeOptimizerConfig {
    let base = resources.optimizer_config().unwrap_or_else(|| {
        NexusNativeOptimizerConfig::default()
            .with_native_execution_config(config.native_execution_config_with_resources(resources))
    });
    config.native_execution.configure_native_optimizer(base).expect("native TPC-H benchmark installs the native rule")
}

fn native_session_config(config: &NativeTpchBenchmarkConfig) -> SessionConfig {
    let mut session_config = tpch_report_session_config();
    if let Some(pushdown_filters) = config.parquet_pushdown_filters {
        session_config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    }
    session_config
}

pub(super) fn select_benchmark_planning_row(
    query_id: &str,
    rows: Vec<NativeDataFusionPlanningRow>,
) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        0 => Err(plan_error(format!("native planning report was empty for {query_id}"))),
        1 => {
            let row = rows.into_iter().next().expect("single planning row must exist");
            if row.query_id != query_id {
                return Err(plan_error(format!(
                    "native TPC-H benchmark planning row query_id mismatch; requested {query_id}, observed {}",
                    row.query_id
                )));
            }
            Ok(row)
        },
        count => {
            let mut whole_native_rows = Vec::new();
            for row in rows {
                if row.query_id != query_id {
                    return Err(plan_error(format!(
                        "native TPC-H benchmark planning row query_id mismatch; requested {query_id}, observed {}",
                        row.query_id
                    )));
                }
                if row.outcome == Some(NativeDataFusionPlanningOutcome::Selected)
                    && row.final_disposition == FinalPlanDisposition::Native
                    && row.native_plan_fragment_count == 1
                    && row.host_datafusion_boundary_count == 0
                    && !row.rejected_before_execution
                {
                    whole_native_rows.push(row);
                }
            }
            match whole_native_rows.len() {
                1 => Ok(whole_native_rows.pop().expect("one whole-native planning row should be present")),
                accepted_count => Err(plan_error(format!(
                    "native TPC-H benchmark requires exactly one whole-native accepted planning row when a query report has multiple rows; {query_id} produced {count} rows and {accepted_count} whole-native accepted rows"
                ))),
            }
        },
    }
}

pub(super) fn validate_decision_grade_native_planning_row(
    query_id: &str,
    row: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    if row.outcome == Some(NativeDataFusionPlanningOutcome::Selected)
        && row.final_disposition != FinalPlanDisposition::Native
    {
        return Err(plan_error(format!(
            "native TPC-H benchmark requires final_disposition=native for accepted query {query_id}; observed {}",
            row.final_disposition.as_str()
        )));
    }
    if row.outcome == Some(NativeDataFusionPlanningOutcome::Selected) {
        if row.rejected_before_execution {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not set rejected_before_execution=true"
            )));
        }
        if row.reason_code.as_deref().is_some_and(|reason_code| !reason_code.is_empty()) {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not include a rejection reason_code"
            )));
        }
        if row.unsupported_category.as_deref().is_some_and(|unsupported_category| !unsupported_category.is_empty()) {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not include an unsupported_category"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_decision_grade_native_not_selected_row(
    query_id: &str,
    row: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    let Some(outcome) = row.outcome else {
        return Err(plan_error(format!(
            "native TPC-H benchmark terminal planning result for {query_id} must be reported as a planning failure"
        )));
    };
    if outcome == NativeDataFusionPlanningOutcome::Selected {
        return Err(plan_error(format!(
            "native TPC-H benchmark not-selected planning row for {query_id} must not report outcome=selected"
        )));
    }
    if row.rejected_before_execution {
        return Err(plan_error(format!(
            "native TPC-H benchmark not-selected planning row for {query_id} must not set rejected_before_execution=true"
        )));
    }
    if row.reason_code.as_deref().is_none_or(str::is_empty) {
        return Err(plan_error(format!(
            "native TPC-H benchmark not-selected planning row for {query_id} must include a stable reason_code"
        )));
    }
    if outcome == NativeDataFusionPlanningOutcome::NotSupported
        && row.unsupported_category.as_deref().is_none_or(str::is_empty)
    {
        return Err(plan_error(format!(
            "native TPC-H benchmark not-supported planning row for {query_id} must include a stable unsupported_category"
        )));
    }
    if outcome == NativeDataFusionPlanningOutcome::NotSelectedByCost && row.unsupported_category.is_some() {
        return Err(plan_error(format!(
            "native TPC-H benchmark cost-not-selected planning row for {query_id} must not include an unsupported_category"
        )));
    }
    if matches!(row.final_disposition, FinalPlanDisposition::Native | FinalPlanDisposition::Rejected) {
        return Err(plan_error(format!(
            "native TPC-H benchmark not-selected planning row for {query_id} must preserve an executable DataFusion or mixed final disposition"
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
        let schema = planned.plan.schema();
        let started = Instant::now();
        let batches = collect_cpu_plan_once(planned).await?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&schema, &batches).map_err(cpu_started)?);
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
    resources: &NativeTpchRuntimeResources,
) -> Result<TimedNativeQueryRun, NativeRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned =
            plan_native_query_with_resources(metadata, query, config, resources).await.map_err(native_not_started)?;
        if planned.planning_row.outcome == Some(NativeDataFusionPlanningOutcome::Selected) {
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
    let mut measured_metrics = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let planned =
            plan_native_query_with_resources(metadata, query, config, resources).await.map_err(native_not_started)?;
        if planned.planning_row.outcome != Some(NativeDataFusionPlanningOutcome::Selected) {
            return Err(NativeRunFailure {
                runtime_started: false,
                message: format!("{} stopped being a native accepted query during execution", query.id),
                details: FailureDetails::default(),
            });
        }
        validate_replanned_native_row(query.id, expected_planning_row, &planned.planning_row)
            .map_err(native_not_started)?;
        let schema = planned.plan.schema();
        let started = Instant::now();
        let (batches, metrics) = collect_native_plan_once(Arc::clone(&planned.plan)).await.map_err(native_started)?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&schema, &batches).map_err(native_started)?);
        measured_metrics.push(metrics);
    }
    let metrics = aggregate_native_execution_metrics(&measured_metrics);
    let timing_metrics = NativeTpchTimingMetrics::mean_from_metrics(&measured_metrics);
    Ok(TimedNativeQueryRun {
        samples,
        final_measurement: final_measurement.expect("validated benchmark config requires iterations"),
        metrics,
        timing_metrics,
    })
}

fn aggregate_native_execution_metrics(metrics: &[MetricsSet]) -> NativeDataFusionExecutionMetrics {
    metrics
        .iter()
        .map(NativeDataFusionExecutionMetrics::from_metrics)
        .reduce(|mut aggregate, current| {
            aggregate.merge_from(current);
            aggregate
        })
        .unwrap_or_default()
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
    match std::env::var("NEXUS_TPCH_CPU_PROFILE").as_deref().map(str::to_ascii_lowercase).as_deref() {
        Ok("untuned") | Ok("report") => tpch_report_session_config(),
        _ => crate::report_workload::tpch::tpch_tuned_cpu_session_config(),
    }
}

fn measure_output(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<QueryMeasurement> {
    Ok(QueryMeasurement {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
        output_fingerprint: validation::fingerprint_batches(schema, batches)?,
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
