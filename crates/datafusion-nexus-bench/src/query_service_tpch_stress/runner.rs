use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{datatypes::SchemaRef as ArrowSchemaRef, record_batch::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::SessionContext,
};
use datafusion_nexus::{
    error::{DataFusionErrorExt, Error as AdapterError},
    native_exec::NexusNativeExecutionConfig,
    native_report::{
        NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
        NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath,
    },
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};
use datafusion_nexus_workloads::tpch::{
    NEXUS_PARQUET_DIR_ENV, NEXUS_TPCH_GENERATION_COMMAND_ENV, NEXUS_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES, TPCH_TABLES,
    TpchDatasetMetadata, TpchDefaultScale, TpchQuery, default_tpch_parquet_root, describe_tpch_dataset,
    discover_tpch_table_layout, register_tpch_table_layout,
};
use nexus_query_engine::{
    error::QueryError,
    harness::StableFnv1a64,
    mode::NativePipelineExecutionMode,
    runtime::{
        QueryMemoryDomainPolicy, QueryService, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId,
    },
};
use tokio::sync::Barrier;

use crate::{
    report_workload::tpch::{tpch_report_session_config, tpch_tuned_cpu_session_config},
    stress_harness::validate_synchronized_admission_queue,
};

mod config;
mod dataset;
mod profile;
mod report;

use dataset::{measure_output, register_tpch_tables, select_planning_row, tpch_query};
use profile::run_profile_stress;
pub async fn run() -> DataFusionResult<()> {
    let config = StressConfig::from_env()?;
    let dataset = describe_tpch_dataset(
        &config.dataset_root,
        std::env::var(NEXUS_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "sf1".to_string()),
        "query_service_tpch_stress",
        std::env::var(NEXUS_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unspecified".to_string()),
    )
    .map_err(|error| plan_error(error.to_string()))?;
    let cpu_baselines = Arc::new(compute_cpu_baselines(&config).await?);
    let mut rows = Vec::new();
    if config.run_overload_probe {
        rows.push(run_overload_probe(&config)?);
    }
    let service = Arc::new(build_query_service(&config)?);
    let mut profile_measured_execution_ms = None;
    if let Some(profile) = &config.profile {
        let result = run_profile_stress(&config, profile, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?;
        profile_measured_execution_ms = Some(result.measured_execution_ms);
        rows.extend(result.rows);
        result.evidence.write_to(&profile.evidence_path)?;
    } else {
        run_warmup(&config, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?;
        let profiler_capture = CudaProfilerCapture::start(config.cuda_profiler_capture)?;
        rows.extend(run_stress(&config, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?);
        wait_for_service_stream_pools_idle(Arc::clone(&service), config.stream_idle_timeout).await?;
        profiler_capture.finish()?;
    }
    let effective_active_max = service.gpu_execution_query_report().peak_active_query_jobs;
    let report = StressReport { config, dataset, rows, profile_measured_execution_ms, effective_active_max };
    print!("{}", report.to_tsv());
    Ok(())
}

#[derive(Debug, Clone)]
struct StressConfig {
    dataset_root: PathBuf,
    bulk_queries: Vec<String>,
    interactive_queries: Vec<String>,
    bulk_iterations: usize,
    interactive_iterations: usize,
    device_capacity_bytes: u64,
    tenant_min_grant_bytes: u64,
    tenant_want_grant_bytes: u64,
    slot_guarantee: SlotGuarantee,
    max_concurrent_queries: usize,
    max_admission_queue_len: usize,
    admission_wait_timeout: Duration,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    source_cache_max_retained_bytes: Option<u64>,
    memory_manager_pool_initial_bytes: Option<u64>,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    query_memory_domain_ceiling_multiplier: Option<u64>,
    warmup_rounds: usize,
    stream_idle_timeout: Duration,
    cuda_profiler_capture: bool,
    profile: Option<StressProfileConfig>,
    run_overload_probe: bool,
    device_ordinals: Vec<usize>,
}

#[derive(Debug, Clone)]
struct StressProfileConfig {
    query_id: String,
    lanes: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    evidence_path: PathBuf,
}

#[derive(Debug, Clone)]
struct CpuBaseline {
    output_rows: usize,
    output_fingerprint: String,
}

#[derive(Debug, Clone)]
struct StressRequest {
    request_index: usize,
    tenant_id: TenantId,
    workload_class: &'static str,
    query_id: String,
    iteration: usize,
}

#[derive(Debug, Clone)]
struct StressRow {
    row_type: &'static str,
    request_index: usize,
    tenant_id: String,
    workload_class: &'static str,
    query_id: String,
    iteration: usize,
    admission_sequence: Option<u64>,
    admission_wait_ms: Option<f64>,
    admission_facts: AdmissionFacts,
    total_wall_ms: f64,
    native_wall_ms: Option<f64>,
    planning_outcome: String,
    post_native_execution_path: String,
    native_runtime_outcome: &'static str,
    parity_status: &'static str,
    cpu_output_rows: Option<usize>,
    native_output_rows: Option<usize>,
    cpu_output_fingerprint: Option<String>,
    native_output_fingerprint: Option<String>,
    error: StructuredFailure,
}

#[derive(Debug, Clone, Copy, Default)]
struct AdmissionFacts {
    device_ordinal: Option<usize>,
    query_grant_bytes: Option<u64>,
    service_capacity_bytes: Option<u64>,
    service_outstanding_bytes_snapshot: Option<u64>,
    service_available_bytes_snapshot: Option<u64>,
}

#[derive(Debug, Clone, Default)]
struct StructuredFailure {
    code: Option<String>,
    kind: Option<String>,
    status: Option<String>,
    operation: Option<String>,
    facts: BTreeMap<String, String>,
    message: Option<String>,
}

enum NativeCollectOutcome {
    Succeeded {
        planning_row: NativeDataFusionPlanningRow,
        measurement: CpuBaseline,
        metrics: NativeDataFusionExecutionMetrics,
    },
    RuntimeFailed {
        planning_row: NativeDataFusionPlanningRow,
        error: DataFusionError,
        metrics: NativeDataFusionExecutionMetrics,
    },
}

struct PreparedNativeQuery {
    planning_row: NativeDataFusionPlanningRow,
    plan: Arc<dyn ExecutionPlan>,
    schema: ArrowSchemaRef,
}

struct ProfileStressResult {
    rows: Vec<StressRow>,
    evidence: ProfileEvidenceReport,
    measured_execution_ms: f64,
}

#[derive(Debug, Clone)]
struct ProfileEvidenceRow {
    phase: &'static str,
    lane: usize,
    iteration: usize,
    request_index: usize,
    admission_sequence: u64,
    device_ordinal: usize,
    query_id: String,
    native_runtime_outcome: &'static str,
    parity_status: &'static str,
    source_cache_hits: usize,
    source_cache_misses: usize,
    source_cache_inserts: usize,
    source_cache_insert_skipped: usize,
    source_cache_retained_bytes: usize,
    native_wall_ms: Option<f64>,
}

#[derive(Debug, Default)]
struct ProfileEvidenceReport {
    rows: Vec<ProfileEvidenceRow>,
}

#[derive(Clone)]
struct ProfileLane {
    lane: usize,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
}

struct PreparedProfileRequest {
    phase: &'static str,
    lane: usize,
    request: StressRequest,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
    prepared: PreparedNativeQuery,
}

struct ProfileRequestResult {
    row: StressRow,
    evidence: ProfileEvidenceRow,
}

struct CudaProfilerCapture {
    active: bool,
}

impl CudaProfilerCapture {
    fn start(enabled: bool) -> DataFusionResult<Self> {
        if !enabled {
            return Ok(Self { active: false });
        }
        cudf::runtime::cuda_profiler_start().map_err(|error| {
            AdapterError::from_cudf(
                "datafusion_nexus_bench.query_service_tpch_stress.profiler_start",
                "failed to start CUDA profiler capture",
                error,
            )
        })?;
        Ok(Self { active: true })
    }

    fn finish(mut self) -> DataFusionResult<()> {
        self.stop()
    }

    fn stop(&mut self) -> DataFusionResult<()> {
        if !self.active {
            return Ok(());
        }
        let result = cudf::runtime::cuda_profiler_stop().map_err(|error| {
            AdapterError::from_cudf(
                "datafusion_nexus_bench.query_service_tpch_stress.profiler_stop",
                "failed to stop CUDA profiler capture",
                error,
            )
        });
        self.active = false;
        result.map_err(DataFusionError::from)
    }
}

impl Drop for CudaProfilerCapture {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("[query_service_tpch_stress] failed to stop CUDA profiler capture: {error}");
        }
    }
}

#[derive(Debug)]
struct StressReport {
    config: StressConfig,
    dataset: TpchDatasetMetadata,
    rows: Vec<StressRow>,
    profile_measured_execution_ms: Option<f64>,
    effective_active_max: usize,
}

async fn compute_cpu_baselines(config: &StressConfig) -> DataFusionResult<HashMap<String, CpuBaseline>> {
    let mut query_ids = BTreeSet::new();
    query_ids.extend(config.bulk_queries.iter().cloned());
    query_ids.extend(config.interactive_queries.iter().cloned());
    if let Some(profile) = &config.profile {
        query_ids.insert(profile.query_id.clone());
    }
    let ctx = SessionContext::new_with_config(tpch_tuned_cpu_session_config());
    register_tpch_tables(&ctx, &config.dataset_root).await?;
    let mut baselines = HashMap::with_capacity(query_ids.len());
    for query_id in query_ids {
        let query = tpch_query(&query_id)?;
        let df = ctx.sql(query.sql).await?;
        let plan = df.create_physical_plan().await?;
        let schema = plan.schema();
        let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
        baselines.insert(query_id, measure_output(&schema, &batches)?);
    }
    Ok(baselines)
}

fn build_query_service(config: &StressConfig) -> DataFusionResult<QueryService> {
    QueryService::new_with_device_ordinals(
        config.service_config(),
        config.device_ordinals.clone(),
        vec![
            tenant_config("bulk", config),
            tenant_config("interactive_a", config),
            tenant_config("interactive_b", config),
        ],
    )
    .map_err(native_error)
}

async fn run_warmup(
    config: &StressConfig,
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<()> {
    for round in 1..=config.warmup_rounds {
        let rows = run_stress(config, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?;
        if let Some(row) =
            rows.iter().find(|row| row.native_runtime_outcome != "succeeded" || row.parity_status != "matched")
        {
            return Err(plan_error(format!(
                "warmup round {round} failed for request_index={} query_id={}: runtime_outcome={} parity_status={} error={}",
                row.request_index,
                row.query_id,
                row.native_runtime_outcome,
                row.parity_status,
                row.error.message.as_deref().unwrap_or("none"),
            )));
        }
        wait_for_service_stream_pools_idle(Arc::clone(&service), config.stream_idle_timeout).await?;
        eprintln!("[query_service_tpch_stress] warmup round {round} complete");
    }
    Ok(())
}

async fn wait_for_service_stream_pools_idle(service: Arc<QueryService>, timeout: Duration) -> DataFusionResult<()> {
    tokio::task::spawn_blocking(move || service.wait_for_stream_pools_idle(timeout))
        .await
        .map_err(|error| plan_error(format!("query service stream-idle wait task join failed: {error}")))?
        .map_err(native_error)
}

async fn run_stress(
    config: &StressConfig,
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<Vec<StressRow>> {
    let requests = build_requests(config);
    let barrier = Arc::new(Barrier::new(requests.len().max(1)));
    let mut tasks = Vec::with_capacity(requests.len());
    for request in requests {
        let service = Arc::clone(&service);
        let config = config.clone();
        let cpu_baselines = Arc::clone(&cpu_baselines);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            run_native_request(&config, service, cpu_baselines, request).await
        }));
    }

    let mut rows = Vec::with_capacity(tasks.len());
    for task in tasks {
        rows.push(task.await.map_err(|error| plan_error(format!("query service stress task join failed: {error}")))??);
    }
    rows.sort_by_key(|row| row.request_index);
    Ok(rows)
}

async fn run_native_request(
    config: &StressConfig,
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
    request: StressRequest,
) -> DataFusionResult<StressRow> {
    let total_started = Instant::now();
    let tenant_for_admission = request.tenant_id.clone();
    let service_for_admission = Arc::clone(&service);
    let handle = match tokio::task::spawn_blocking(move || service_for_admission.admit_query(tenant_for_admission))
        .await
        .map_err(|error| plan_error(format!("query service admission task join failed: {error}")))?
    {
        Ok(handle) => Arc::new(handle),
        Err(error) => {
            return Ok(failure_row(
                &request,
                total_started.elapsed(),
                None,
                None,
                AdmissionFacts::default(),
                "not_planned",
                "not_planned",
                "not_started",
                "not_run",
                StructuredFailure::from_query_error(&error),
                cpu_baselines.get(&request.query_id),
                None,
            ));
        },
    };

    let admission_sequence = Some(handle.admission_sequence());
    let admission_wait_ms = Some(duration_ms(handle.admission_wait()));
    let admission_facts = admission_facts(&service, &handle)?;
    let query = tpch_query(&request.query_id)?;
    let cpu = cpu_baselines.get(&request.query_id);
    let native_started = Instant::now();
    match collect_native_with_handle(config, query, Arc::clone(&handle)).await {
        Ok(NativeCollectOutcome::Succeeded { planning_row, measurement: native, .. }) => {
            let parity = match (cpu, Some(&native)) {
                (Some(cpu), Some(native)) if cpu.output_fingerprint == native.output_fingerprint => "matched",
                (Some(_), Some(_)) => "mismatched",
                (None, Some(_)) => "cpu_missing",
                _ => "not_run",
            };
            Ok(StressRow {
                row_type: "query",
                request_index: request.request_index,
                tenant_id: request.tenant_id.to_string(),
                workload_class: request.workload_class,
                query_id: request.query_id,
                iteration: request.iteration,
                admission_sequence,
                admission_wait_ms,
                admission_facts,
                total_wall_ms: duration_ms(total_started.elapsed()),
                native_wall_ms: Some(duration_ms(native_started.elapsed())),
                planning_outcome: planning_row.outcome.as_str().to_string(),
                post_native_execution_path: planning_row.post_native_execution_path.as_str().to_string(),
                native_runtime_outcome: "succeeded",
                parity_status: parity,
                cpu_output_rows: cpu.map(|cpu| cpu.output_rows),
                native_output_rows: Some(native.output_rows),
                cpu_output_fingerprint: cpu.map(|cpu| cpu.output_fingerprint.clone()),
                native_output_fingerprint: Some(native.output_fingerprint),
                error: StructuredFailure::default(),
            })
        },
        Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error, .. }) => Ok(failure_row(
            &request,
            total_started.elapsed(),
            admission_sequence,
            admission_wait_ms,
            admission_facts,
            planning_row.outcome.as_str(),
            planning_row.post_native_execution_path.as_str(),
            "failed",
            "native_failed",
            StructuredFailure::from_datafusion_error(&error),
            cpu,
            None,
        )),
        Err(error) => Ok(failure_row(
            &request,
            total_started.elapsed(),
            admission_sequence,
            admission_wait_ms,
            admission_facts,
            "unknown",
            "unknown",
            "failed",
            "native_failed",
            StructuredFailure::from_datafusion_error(&error),
            cpu,
            None,
        )),
    }
}

async fn collect_native_with_handle(
    config: &StressConfig,
    query: &TpchQuery,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
) -> DataFusionResult<NativeCollectOutcome> {
    let prepared = prepare_native_with_handle(config, query, handle).await?;
    collect_prepared_native(prepared).await
}

async fn prepare_native_with_handle(
    config: &StressConfig,
    query: &TpchQuery,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
) -> DataFusionResult<PreparedNativeQuery> {
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, &config.dataset_root).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report = NativeDataFusionPlanningReport::from_retained_rule(query.id, Arc::clone(&plan), &native_rule);
    let planning_row = select_planning_row(query.id, planning_report.rows)?;
    if planning_row.outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
        return Err(plan_error(format!(
            "{} was not accepted by native planning; outcome={} reason_code={}",
            query.id,
            planning_row.outcome.as_str(),
            planning_row.reason_code.as_deref().unwrap_or("none"),
        )));
    }
    if planning_row.post_native_execution_path != NativeDataFusionPostNativeExecutionPath::Native {
        return Err(plan_error(format!(
            "{} accepted planning row did not produce post_native_execution_path=native; observed {}",
            query.id,
            planning_row.post_native_execution_path.as_str(),
        )));
    }
    let schema = plan.schema();
    Ok(PreparedNativeQuery { planning_row, plan, schema })
}

async fn collect_prepared_native(prepared: PreparedNativeQuery) -> DataFusionResult<NativeCollectOutcome> {
    let PreparedNativeQuery { planning_row, plan, schema } = prepared;
    let batches = match collect_plan_once(Arc::clone(&plan)).await {
        Ok(batches) => batches,
        Err(error) => {
            let metrics = native_execution_metrics(&plan);
            return Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error, metrics });
        },
    };
    let measurement = measure_output(&schema, &batches)?;
    let metrics = native_execution_metrics(&plan);
    Ok(NativeCollectOutcome::Succeeded { planning_row, measurement, metrics })
}

fn native_execution_metrics(plan: &Arc<dyn ExecutionPlan>) -> NativeDataFusionExecutionMetrics {
    plan.metrics().as_ref().map(NativeDataFusionExecutionMetrics::from_metrics).unwrap_or_default()
}

fn admission_facts(
    service: &QueryService,
    handle: &nexus_query_engine::runtime::QueryHandle,
) -> DataFusionResult<AdmissionFacts> {
    let reports = service.memory_governor_reports().map_err(native_error)?;
    let report = reports
        .into_iter()
        .find(|report| report.device_ordinal == handle.device_ordinal())
        .expect("query service should expose a memory governor report for the handle's device");
    Ok(AdmissionFacts {
        device_ordinal: Some(handle.device_ordinal()),
        query_grant_bytes: Some(handle.device_budget_bytes()),
        service_capacity_bytes: Some(report.capacity_bytes),
        service_outstanding_bytes_snapshot: Some(report.outstanding_bytes),
        service_available_bytes_snapshot: Some(report.available_bytes),
    })
}

async fn collect_plan_once(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<Vec<RecordBatch>> {
    collect(plan, Arc::new(TaskContext::default())).await
}

fn run_overload_probe(config: &StressConfig) -> DataFusionResult<StressRow> {
    let probe_config = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_admission_queue_len: 1,
        admission_wait_timeout: Duration::from_millis(50),
        ..config.service_config()
    };
    let service =
        QueryService::new(probe_config, vec![tenant_config("bulk", config), tenant_config("interactive_a", config)])
            .map_err(native_error)?;
    let active = service.admit_query("bulk").map_err(native_error)?;
    let queued_service = service.clone();
    let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        ready_sender.send(()).expect("overload probe ready signal should send");
        let _queued = queued_service.admit_query("bulk");
    });
    ready_receiver.recv().map_err(|error| plan_error(format!("overload probe setup failed: {error}")))?;
    wait_for_queued_len(&service, 1)?;
    let started = Instant::now();
    let error = service.admit_query("interactive_a").expect_err("filled admission queue must reject next query");
    drop(active);
    waiter.join().map_err(|_| plan_error("overload probe waiter thread panicked"))?;
    Ok(StressRow {
        row_type: "overload_probe",
        request_index: 0,
        tenant_id: "interactive_a".to_string(),
        workload_class: "synthetic",
        query_id: "none".to_string(),
        iteration: 0,
        admission_sequence: None,
        admission_wait_ms: None,
        admission_facts: AdmissionFacts::default(),
        total_wall_ms: duration_ms(started.elapsed()),
        native_wall_ms: None,
        planning_outcome: "not_planned".to_string(),
        post_native_execution_path: "not_planned".to_string(),
        native_runtime_outcome: "not_started",
        parity_status: "not_run",
        cpu_output_rows: None,
        native_output_rows: None,
        cpu_output_fingerprint: None,
        native_output_fingerprint: None,
        error: StructuredFailure::from_query_error(&error),
    })
}

fn wait_for_queued_len(service: &QueryService, expected: usize) -> DataFusionResult<()> {
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        let snapshot = service.admission_snapshot().map_err(native_error)?;
        if snapshot.queued_queries >= expected {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Err(plan_error("timed out waiting for overload probe admission queue"))
}

fn tenant_config(tenant_id: impl Into<TenantId>, config: &StressConfig) -> QueryServiceTenantConfig {
    QueryServiceTenantConfig::with_grant_range(tenant_id, config.tenant_min_grant_bytes, config.tenant_want_grant_bytes)
}

fn build_requests(config: &StressConfig) -> Vec<StressRequest> {
    let mut requests = Vec::new();
    let mut next_index = 1;
    for iteration in 1..=config.bulk_iterations {
        for query_id in &config.bulk_queries {
            requests.push(StressRequest {
                request_index: next_index,
                tenant_id: TenantId::new("bulk"),
                workload_class: "bulk",
                query_id: query_id.clone(),
                iteration,
            });
            next_index += 1;
        }
    }
    for tenant_id in ["interactive_a", "interactive_b"] {
        for iteration in 1..=config.interactive_iterations {
            for query_id in &config.interactive_queries {
                requests.push(StressRequest {
                    request_index: next_index,
                    tenant_id: TenantId::new(tenant_id),
                    workload_class: "interactive",
                    query_id: query_id.clone(),
                    iteration,
                });
                next_index += 1;
            }
        }
    }
    requests
}

fn failure_row(
    request: &StressRequest,
    total_wall: Duration,
    admission_sequence: Option<u64>,
    admission_wait_ms: Option<f64>,
    admission_facts: AdmissionFacts,
    planning_outcome: &str,
    post_native_execution_path: &str,
    native_runtime_outcome: &'static str,
    parity_status: &'static str,
    error: StructuredFailure,
    cpu: Option<&CpuBaseline>,
    native: Option<&CpuBaseline>,
) -> StressRow {
    StressRow {
        row_type: "query",
        request_index: request.request_index,
        tenant_id: request.tenant_id.to_string(),
        workload_class: request.workload_class,
        query_id: request.query_id.clone(),
        iteration: request.iteration,
        admission_sequence,
        admission_wait_ms,
        admission_facts,
        total_wall_ms: duration_ms(total_wall),
        native_wall_ms: None,
        planning_outcome: planning_outcome.to_string(),
        post_native_execution_path: post_native_execution_path.to_string(),
        native_runtime_outcome,
        parity_status,
        cpu_output_rows: cpu.map(|cpu| cpu.output_rows),
        native_output_rows: native.map(|native| native.output_rows),
        cpu_output_fingerprint: cpu.map(|cpu| cpu.output_fingerprint.clone()),
        native_output_fingerprint: native.map(|native| native.output_fingerprint.clone()),
        error,
    }
}

fn format_facts(facts: &BTreeMap<String, String>) -> String {
    if facts.is_empty() {
        return "none".to_string();
    }
    facts.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join(";")
}

fn tsv_escape(value: &str) -> String {
    value.replace('\t', "\\t").replace('\n', "\\n").replace('\r', "\\r")
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn format_ms(value: f64) -> String {
    format!("{value:.3}")
}

fn native_error(error: QueryError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}
