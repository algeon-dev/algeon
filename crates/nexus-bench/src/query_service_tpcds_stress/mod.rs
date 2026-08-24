use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{record_batch::RecordBatch, util::display::array_value_to_string};
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
        CandidateOutcomeKind, FinalPlanDisposition, NativeDataFusionPlanningReport, NativeDataFusionPlanningRow,
    },
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};
use nexus_query_engine::{
    error::QueryError,
    harness::StableFnv1a64,
    runtime::{GpuBackendSession, QueryExecutionConfig, QueueBounds},
};
use nexus_workloads::tpcds::{
    NEXUS_PARQUET_DIR_ENV, NEXUS_TPCDS_GENERATION_COMMAND_ENV, NEXUS_TPCDS_SCALE_FACTOR_ENV, TPCDS_QUERIES,
    TpcdsDatasetMetadata, TpcdsQuery, default_tpcds_sf1_parquet_root, describe_tpcds_dataset,
};
use tokio::{sync::Barrier, time::timeout};

use crate::{
    gpu_backend::GpuBackendOwner,
    report_workload::tpcds::{
        register_tpcds_tables_with_native_metadata, tpcds_native_report_session_config, tpcds_report_session_config,
    },
    stress_harness::validate_synchronized_admission_queue,
};
const TSV_COLUMNS: &[&str] = &[
    "row_type",
    "request_index",
    "client_id",
    "workload_class",
    "query_id",
    "iteration",
    "admission_sequence",
    "admission_wait_ms",
    "total_wall_ms",
    "native_wall_ms",
    "planning_outcome",
    "final_disposition",
    "native_runtime_outcome",
    "parity_status",
    "cpu_output_rows",
    "native_output_rows",
    "cpu_output_fingerprint",
    "native_output_fingerprint",
    "error_code",
    "error_kind",
    "error_status",
    "error_operation",
    "error_facts",
    "error_message",
];
const DATASET_ROOT_ENV: &str = NEXUS_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_INTERACTIVE_ITERATIONS";
const DEVICE_CAPACITY_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_DEVICE_CAPACITY_BYTES";
const MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_MAX_ACTIVE_ATTEMPTS_PER_DEVICE";
const MAX_QUEUED_ATTEMPTS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_MAX_QUEUED_ATTEMPTS";
const MAX_QUEUE_WAIT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_MAX_QUEUE_WAIT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_MAX_ROW_GROUPS_PER_CHUNK";
const QUERY_TIMEOUT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_QUERY_TIMEOUT_MS";
const OVERLOAD_PROBE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCDS_OVERLOAD_PROBE";
pub async fn run() -> DataFusionResult<()> {
    let config = StressConfig::from_env()?;
    let dataset = describe_tpcds_dataset(
        &config.dataset_root,
        std::env::var(NEXUS_TPCDS_SCALE_FACTOR_ENV).unwrap_or_else(|_| "sf1".to_string()),
        "query_service_tpcds_stress",
        std::env::var(NEXUS_TPCDS_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unspecified".to_string()),
    )
    .map_err(|error| plan_error(error.to_string()))?;
    let cpu_baselines = Arc::new(compute_cpu_baselines(&config).await?);
    let mut rows = Vec::new();
    if config.run_overload_probe {
        rows.push(run_overload_probe(&config)?);
    }
    let stress = run_stress(&config, Arc::clone(&cpu_baselines)).await?;
    rows.extend(stress.rows);
    let report = StressReport { config, dataset, rows, effective_active_max: stress.effective_active_max };
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
    max_active_attempts_per_device: usize,
    max_queued_attempts: usize,
    max_queue_wait: Duration,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    query_timeout: Duration,
    run_overload_probe: bool,
}

impl StressConfig {
    fn from_env() -> DataFusionResult<Self> {
        let dataset_root = match std::env::var_os(DATASET_ROOT_ENV) {
            Some(value) => PathBuf::from(value),
            None => default_tpcds_sf1_parquet_root().into(),
        };
        let max_active_attempts_per_device = env_usize(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, 1)?;
        let device_capacity_bytes = env_u64(DEVICE_CAPACITY_BYTES_ENV, default_device_capacity_bytes())?;
        let config = Self {
            dataset_root,
            bulk_queries: query_ids_from_env(
                BULK_QUERIES_ENV,
                &["q01", "q05", "q23", "q28", "q49", "q54", "q70", "q72", "q88"],
            )?,
            interactive_queries: query_ids_from_env(
                INTERACTIVE_QUERIES_ENV,
                &["q10", "q12", "q36", "q61", "q78", "q90"],
            )?,
            bulk_iterations: env_usize(BULK_ITERATIONS_ENV, 1)?,
            interactive_iterations: env_usize(INTERACTIVE_ITERATIONS_ENV, 2)?,
            device_capacity_bytes,
            max_active_attempts_per_device,
            max_queued_attempts: env_usize(MAX_QUEUED_ATTEMPTS_ENV, 96)?,
            max_queue_wait: Duration::from_millis(env_u64(MAX_QUEUE_WAIT_MS_ENV, 60_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
            query_timeout: Duration::from_millis(env_u64(QUERY_TIMEOUT_MS_ENV, 300_000)?),
            run_overload_probe: env_bool(OVERLOAD_PROBE_ENV, true)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero(BULK_ITERATIONS_ENV, self.bulk_iterations)?;
        validate_nonzero(INTERACTIVE_ITERATIONS_ENV, self.interactive_iterations)?;
        validate_nonzero(DEVICE_CAPACITY_BYTES_ENV, self.device_capacity_bytes)?;
        validate_nonzero(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, self.max_active_attempts_per_device)?;
        validate_nonzero(MAX_QUEUED_ATTEMPTS_ENV, self.max_queued_attempts)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if self.max_queue_wait.is_zero() {
            return Err(plan_error(format!("{MAX_QUEUE_WAIT_MS_ENV} must be greater than zero")));
        }
        if self.query_timeout.is_zero() {
            return Err(plan_error(format!("{QUERY_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        validate_synchronized_admission_queue(
            MAX_QUEUED_ATTEMPTS_ENV,
            self.max_queued_attempts,
            self.synchronized_request_count(),
        )
        .map_err(plan_error)?;
        Ok(())
    }

    fn synchronized_request_count(&self) -> usize {
        self.bulk_queries.len().saturating_mul(self.bulk_iterations).saturating_add(
            self.interactive_queries.len().saturating_mul(self.interactive_iterations).saturating_mul(2),
        )
    }

    fn service_config(&self) -> QueryExecutionConfig {
        QueryExecutionConfig {
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            ..QueryExecutionConfig::default()
        }
    }

    fn queue_bounds(&self) -> QueueBounds {
        QueueBounds { max_waiters: self.max_queued_attempts, max_wait: self.max_queue_wait }
    }
}

#[derive(Debug, Clone)]
struct CpuBaseline {
    output_rows: usize,
    output_fingerprint: String,
}

#[derive(Debug, Clone)]
struct StressRequest {
    request_index: usize,
    // Stable TSV workload label; this is not an admission identity.
    client_id: String,
    workload_class: &'static str,
    query_id: String,
    iteration: usize,
}

#[derive(Debug, Clone)]
struct StressRow {
    row_type: &'static str,
    request_index: usize,
    client_id: String,
    workload_class: &'static str,
    query_id: String,
    iteration: usize,
    admission_sequence: Option<u64>,
    admission_wait_ms: Option<f64>,
    total_wall_ms: f64,
    native_wall_ms: Option<f64>,
    planning_outcome: String,
    final_disposition: String,
    native_runtime_outcome: &'static str,
    parity_status: &'static str,
    cpu_output_rows: Option<usize>,
    native_output_rows: Option<usize>,
    cpu_output_fingerprint: Option<String>,
    native_output_fingerprint: Option<String>,
    error: StructuredFailure,
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
    Succeeded { planning_row: NativeDataFusionPlanningRow, measurement: CpuBaseline },
    RuntimeFailed { planning_row: NativeDataFusionPlanningRow, error: DataFusionError },
}

impl StructuredFailure {
    fn from_query_error(error: &QueryError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            message: Some(error.to_string()),
        }
    }

    fn from_datafusion_error(error: &DataFusionError) -> Self {
        error.find_nexus_error().map(Self::from_adapter_error).unwrap_or_else(|| Self {
            code: Some("datafusion_error".to_string()),
            message: Some(error.to_string()),
            ..Self::default()
        })
    }

    fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            message: Some(error.to_string()),
        }
    }

    fn timed_out(timeout: Duration) -> Self {
        let mut facts = BTreeMap::new();
        facts.insert("timeout_ms".to_string(), timeout.as_millis().to_string());
        Self {
            code: Some("query_timeout".to_string()),
            kind: Some("resource_exhausted".to_string()),
            status: Some("temporary".to_string()),
            operation: Some("datafusion_nexus.query_service_tpcds_stress.native_collect".to_string()),
            facts,
            message: Some(format!("native query exceeded timeout_ms={}", timeout.as_millis())),
        }
    }
}

#[derive(Debug)]
struct StressReport {
    config: StressConfig,
    dataset: TpcdsDatasetMetadata,
    rows: Vec<StressRow>,
    effective_active_max: usize,
}

struct StressRun {
    rows: Vec<StressRow>,
    effective_active_max: usize,
}

impl StressReport {
    fn to_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# dataset_root={}\n", self.config.dataset_root.display()));
        output.push_str(&format!("# dataset_metadata_fingerprint={}\n", self.dataset.metadata_fingerprint));
        output.push_str(&format!("# device_capacity_bytes={}\n", self.config.device_capacity_bytes));
        output.push_str(&format!("# max_active_attempts_per_device={}\n", self.config.max_active_attempts_per_device));
        output.push_str(&format!("# effective_active_max={}\n", self.effective_active_max));
        output.push_str(&format!("# max_queued_attempts={}\n", self.config.max_queued_attempts));
        output.push_str(&format!("# max_queue_wait_ms={}\n", self.config.max_queue_wait.as_millis()));
        output.push_str(&format!("# query_timeout_ms={}\n", self.config.query_timeout.as_millis()));
        for summary in client_summaries(&self.rows) {
            output.push_str(&format!(
                "# client_summary client_id={} succeeded={} overloaded={} timed_out={} failed={} max_total_wall_ms={} p95_total_wall_ms={}\n",
                summary.client_id,
                summary.succeeded,
                summary.overloaded,
                summary.timed_out,
                summary.failed,
                format_ms(summary.max_total_wall_ms),
                format_ms(summary.p95_total_wall_ms),
            ));
        }
        output.push_str(&TSV_COLUMNS.join("\t"));
        output.push('\n');
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl StressRow {
    fn to_tsv_line(&self) -> String {
        let fields = [
            self.row_type.to_string(),
            self.request_index.to_string(),
            self.client_id.clone(),
            self.workload_class.to_string(),
            self.query_id.clone(),
            self.iteration.to_string(),
            self.admission_sequence.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_wait_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            format_ms(self.total_wall_ms),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            self.planning_outcome.clone(),
            self.final_disposition.clone(),
            self.native_runtime_outcome.to_string(),
            self.parity_status.to_string(),
            self.cpu_output_rows.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.native_output_rows.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.cpu_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.native_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.error.code.clone().unwrap_or_else(|| "none".to_string()),
            self.error.kind.clone().unwrap_or_else(|| "none".to_string()),
            self.error.status.clone().unwrap_or_else(|| "none".to_string()),
            self.error.operation.clone().unwrap_or_else(|| "none".to_string()),
            format_facts(&self.error.facts),
            self.error.message.clone().unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), TSV_COLUMNS.len());
        fields.into_iter().map(|field| tsv_escape(&field)).collect::<Vec<_>>().join("\t")
    }
}

#[derive(Debug, Clone)]
struct ClientSummary {
    client_id: String,
    succeeded: usize,
    overloaded: usize,
    timed_out: usize,
    failed: usize,
    max_total_wall_ms: f64,
    p95_total_wall_ms: f64,
}

async fn compute_cpu_baselines(config: &StressConfig) -> DataFusionResult<HashMap<String, CpuBaseline>> {
    let mut query_ids = BTreeSet::new();
    query_ids.extend(config.bulk_queries.iter().cloned());
    query_ids.extend(config.interactive_queries.iter().cloned());
    let ctx = SessionContext::new_with_config(tpcds_report_session_config());
    register_tpcds_tables(&ctx, &config.dataset_root).await?;
    let mut baselines = HashMap::with_capacity(query_ids.len());
    for query_id in query_ids {
        eprintln!("[query-service-tpcds] cpu-baseline start {query_id}");
        let started = Instant::now();
        let query = tpcds_query(&query_id)?;
        let df = ctx.sql(query.sql).await?;
        let plan = df.create_physical_plan().await?;
        let schema = plan.schema();
        let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
        baselines.insert(query_id.clone(), measure_output(&schema, &batches)?);
        eprintln!("[query-service-tpcds] cpu-baseline finish {query_id} elapsed_ms={}", started.elapsed().as_millis());
    }
    Ok(baselines)
}

async fn run_stress(
    config: &StressConfig,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<StressRun> {
    let backend = GpuBackendOwner::build(
        config.service_config(),
        config.queue_bounds(),
        vec![GpuBackendOwner::device_config(0, config.device_capacity_bytes, config.max_active_attempts_per_device, 1)],
    )
    .map_err(native_error)?;
    let service = backend.session();
    let requests = build_requests(config);
    let barrier = Arc::new(Barrier::new(requests.len().max(1)));
    let mut tasks = Vec::with_capacity(requests.len());
    for request in requests {
        let service = service.clone();
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
        rows.push(
            task.await
                .map_err(|error| plan_error(format!("query service TPC-DS stress task join failed: {error}")))??,
        );
    }
    rows.sort_by_key(|row| row.request_index);
    let effective_active_max = service.gpu_execution_query_report().map_err(native_error)?.peak_active_query_jobs;
    backend.close_and_wait(Duration::from_secs(30)).map_err(native_error)?;
    Ok(StressRun { rows, effective_active_max })
}

async fn run_native_request(
    config: &StressConfig,
    service: GpuBackendSession,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
    request: StressRequest,
) -> DataFusionResult<StressRow> {
    let total_started = Instant::now();
    let service_for_admission = service.clone();
    let handle = match tokio::task::spawn_blocking(move || service_for_admission.admit_query())
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
    let query = tpcds_query(&request.query_id)?;
    let cpu = cpu_baselines.get(&request.query_id);
    let native_started = Instant::now();
    eprintln!(
        "[query-service-tpcds] native start client={} query={} iteration={} admission_sequence={}",
        request.client_id,
        request.query_id,
        request.iteration,
        handle.admission_sequence()
    );
    let outcome = timeout(config.query_timeout, collect_native_with_handle(config, query, Arc::clone(&handle))).await;
    match outcome {
        Ok(Ok(NativeCollectOutcome::Succeeded { planning_row, measurement: native })) => {
            let parity = match (cpu, Some(&native)) {
                (Some(cpu), Some(native)) if cpu.output_fingerprint == native.output_fingerprint => "matched",
                (Some(_), Some(_)) => "mismatched",
                (None, Some(_)) => "cpu_missing",
                _ => "not_run",
            };
            eprintln!(
                "[query-service-tpcds] native finish client={} query={} iteration={} elapsed_ms={} outcome=succeeded parity={parity}",
                request.client_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(StressRow {
                row_type: "query",
                request_index: request.request_index,
                client_id: request.client_id.to_string(),
                workload_class: request.workload_class,
                query_id: request.query_id,
                iteration: request.iteration,
                admission_sequence,
                admission_wait_ms,
                total_wall_ms: duration_ms(total_started.elapsed()),
                native_wall_ms: Some(duration_ms(native_started.elapsed())),
                planning_outcome: planning_row.outcome.map_or("none", CandidateOutcomeKind::as_str).to_string(),
                final_disposition: planning_row
                    .final_disposition
                    .map_or("none", FinalPlanDisposition::as_str)
                    .to_string(),
                native_runtime_outcome: "succeeded",
                parity_status: parity,
                cpu_output_rows: cpu.map(|cpu| cpu.output_rows),
                native_output_rows: Some(native.output_rows),
                cpu_output_fingerprint: cpu.map(|cpu| cpu.output_fingerprint.clone()),
                native_output_fingerprint: Some(native.output_fingerprint),
                error: StructuredFailure::default(),
            })
        },
        Ok(Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error })) => {
            eprintln!(
                "[query-service-tpcds] native finish client={} query={} iteration={} elapsed_ms={} outcome=failed",
                request.client_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(failure_row(
                &request,
                total_started.elapsed(),
                admission_sequence,
                admission_wait_ms,
                planning_row.outcome.map_or("none", CandidateOutcomeKind::as_str),
                planning_row.final_disposition.map_or("none", FinalPlanDisposition::as_str),
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            ))
        },
        Ok(Err(error)) => {
            eprintln!(
                "[query-service-tpcds] native finish client={} query={} iteration={} elapsed_ms={} outcome=failed",
                request.client_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(failure_row(
                &request,
                total_started.elapsed(),
                admission_sequence,
                admission_wait_ms,
                "unknown",
                "unknown",
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            ))
        },
        Err(_) => {
            eprintln!(
                "[query-service-tpcds] native finish client={} query={} iteration={} elapsed_ms={} outcome=timed_out",
                request.client_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(failure_row(
                &request,
                total_started.elapsed(),
                admission_sequence,
                admission_wait_ms,
                "unknown",
                "unknown",
                "timed_out",
                "timed_out",
                StructuredFailure::timed_out(config.query_timeout),
                cpu,
                None,
            ))
        },
    }
}

async fn collect_native_with_handle(
    config: &StressConfig,
    query: &TpcdsQuery,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
) -> DataFusionResult<NativeCollectOutcome> {
    let optimizer_config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded_default().with_query_service_handle(handle));
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpcds_native_report_session_config())
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, &config.dataset_root).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report = NativeDataFusionPlanningReport::from_retained_rule(query.id, Arc::clone(&plan), &native_rule);
    let planning_row = select_planning_row(query.id, planning_report.rows)?;
    if planning_row.outcome != Some(CandidateOutcomeKind::Selected) {
        return Err(plan_error(format!(
            "{} was not accepted by native planning; outcome={} reason_code={}",
            query.id,
            planning_row.outcome.map_or("none", CandidateOutcomeKind::as_str),
            planning_row.reason_code.as_deref().unwrap_or("none"),
        )));
    }
    if planning_row.final_disposition != Some(FinalPlanDisposition::Native) {
        return Err(plan_error(format!(
            "{} accepted planning row did not produce final_disposition=native; observed {}",
            query.id,
            planning_row.final_disposition.map_or("none", FinalPlanDisposition::as_str),
        )));
    }
    let schema = plan.schema();
    let batches = match collect_plan_once(plan).await {
        Ok(batches) => batches,
        Err(error) => return Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error }),
    };
    let measurement = measure_output(&schema, &batches)?;
    Ok(NativeCollectOutcome::Succeeded { planning_row, measurement })
}

async fn collect_plan_once(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<Vec<RecordBatch>> {
    collect(plan, Arc::new(TaskContext::default())).await
}

fn run_overload_probe(config: &StressConfig) -> DataFusionResult<StressRow> {
    let probe_config = config.service_config();
    let probe_queue_bounds = QueueBounds { max_waiters: 1, max_wait: Duration::from_millis(50) };
    let backend = GpuBackendOwner::build(
        probe_config,
        probe_queue_bounds,
        vec![GpuBackendOwner::device_config(0, config.device_capacity_bytes, 1, 1)],
    )
    .map_err(native_error)?;
    let service = backend.session();
    let active = service.admit_query().map_err(native_error)?;
    let queued_service = service.clone();
    let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        ready_sender.send(()).expect("overload probe ready signal should send");
        let _queued = queued_service.admit_query();
    });
    ready_receiver.recv().map_err(|error| plan_error(format!("overload probe setup failed: {error}")))?;
    wait_for_queued_len(&service, 1)?;
    let started = Instant::now();
    let error = service.admit_query().expect_err("filled admission queue must reject next query");
    drop(active);
    waiter.join().map_err(|_| plan_error("overload probe waiter thread panicked"))?;
    let row = StressRow {
        row_type: "overload_probe",
        request_index: 0,
        client_id: "interactive_a".to_string(),
        workload_class: "synthetic",
        query_id: "none".to_string(),
        iteration: 0,
        admission_sequence: None,
        admission_wait_ms: None,
        total_wall_ms: duration_ms(started.elapsed()),
        native_wall_ms: None,
        planning_outcome: "not_planned".to_string(),
        final_disposition: "not_planned".to_string(),
        native_runtime_outcome: "not_started",
        parity_status: "not_run",
        cpu_output_rows: None,
        native_output_rows: None,
        cpu_output_fingerprint: None,
        native_output_fingerprint: None,
        error: StructuredFailure::from_query_error(&error),
    };
    backend.close_and_wait(Duration::from_secs(30)).map_err(native_error)?;
    Ok(row)
}

fn wait_for_queued_len(service: &GpuBackendSession, expected: usize) -> DataFusionResult<()> {
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

fn build_requests(config: &StressConfig) -> Vec<StressRequest> {
    let mut requests = Vec::new();
    let mut next_index = 1;
    for iteration in 1..=config.bulk_iterations {
        for query_id in &config.bulk_queries {
            requests.push(StressRequest {
                request_index: next_index,
                client_id: "bulk".to_string(),
                workload_class: "bulk",
                query_id: query_id.clone(),
                iteration,
            });
            next_index += 1;
        }
    }
    for client_id in ["interactive_a", "interactive_b"] {
        for iteration in 1..=config.interactive_iterations {
            for query_id in &config.interactive_queries {
                requests.push(StressRequest {
                    request_index: next_index,
                    client_id: client_id.to_string(),
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

#[expect(
    clippy::too_many_arguments,
    reason = "service construction inputs are distinct admission and ownership values"
)]
fn failure_row(
    request: &StressRequest,
    total_wall: Duration,
    admission_sequence: Option<u64>,
    admission_wait_ms: Option<f64>,
    planning_outcome: &str,
    final_disposition: &str,
    native_runtime_outcome: &'static str,
    parity_status: &'static str,
    error: StructuredFailure,
    cpu: Option<&CpuBaseline>,
    native: Option<&CpuBaseline>,
) -> StressRow {
    StressRow {
        row_type: "query",
        request_index: request.request_index,
        client_id: request.client_id.to_string(),
        workload_class: request.workload_class,
        query_id: request.query_id.clone(),
        iteration: request.iteration,
        admission_sequence,
        admission_wait_ms,
        total_wall_ms: duration_ms(total_wall),
        native_wall_ms: None,
        planning_outcome: planning_outcome.to_string(),
        final_disposition: final_disposition.to_string(),
        native_runtime_outcome,
        parity_status,
        cpu_output_rows: cpu.map(|cpu| cpu.output_rows),
        native_output_rows: native.map(|native| native.output_rows),
        cpu_output_fingerprint: cpu.map(|cpu| cpu.output_fingerprint.clone()),
        native_output_fingerprint: native.map(|native| native.output_fingerprint.clone()),
        error,
    }
}

fn client_summaries(rows: &[StressRow]) -> Vec<ClientSummary> {
    let mut by_client: BTreeMap<String, Vec<&StressRow>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.row_type == "query") {
        by_client.entry(row.client_id.clone()).or_default().push(row);
    }
    by_client
        .into_iter()
        .map(|(client_id, rows)| {
            let mut latencies = rows.iter().map(|row| row.total_wall_ms).collect::<Vec<_>>();
            latencies.sort_by(f64::total_cmp);
            let succeeded = rows.iter().filter(|row| row.native_runtime_outcome == "succeeded").count();
            let overloaded = rows.iter().filter(|row| row.error.code.as_deref() == Some("service_overloaded")).count();
            let timed_out = rows.iter().filter(|row| row.native_runtime_outcome == "timed_out").count();
            let failed = rows.len().saturating_sub(succeeded + overloaded + timed_out);
            ClientSummary {
                client_id,
                succeeded,
                overloaded,
                timed_out,
                failed,
                max_total_wall_ms: latencies.last().copied().unwrap_or(0.0),
                p95_total_wall_ms: percentile(&latencies, 0.95),
            }
        })
        .collect()
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index = ((sorted_values.len() - 1) as f64 * quantile).ceil() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

async fn register_tpcds_tables(ctx: &SessionContext, root: &Path) -> DataFusionResult<()> {
    register_tpcds_tables_with_native_metadata(ctx, root).await.map_err(|error| plan_error(error.to_string()))
}

fn select_planning_row(
    query_id: &str,
    rows: Vec<NativeDataFusionPlanningRow>,
) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        1 => {
            let row = rows.into_iter().next().expect("single planning row exists");
            if row.query_id != query_id {
                return Err(plan_error(format!(
                    "planning row query_id mismatch; requested {query_id}, observed {}",
                    row.query_id
                )));
            }
            Ok(row)
        },
        count => Err(plan_error(format!("expected one planning row for {query_id}, observed {count}"))),
    }
}

fn tpcds_query(query_id: &str) -> DataFusionResult<&'static TpcdsQuery> {
    TPCDS_QUERIES
        .iter()
        .find(|query| query.id == query_id)
        .ok_or_else(|| plan_error(format!("unknown TPC-DS query id: {query_id}")))
}

fn measure_output(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<CpuBaseline> {
    Ok(CpuBaseline {
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_fingerprint: fingerprint_batches(schema, batches)?,
    })
}

fn fingerprint_batches(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-query-service-tpcds-output-v1");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(plan_error("TPC-DS output fingerprint requires collected batches to match the planned schema"));
        }
    }
    hasher.update_field("schema_metadata");
    hasher.update_metadata(schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&field.data_type().to_string());
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        hasher.update_metadata(field.metadata());
    }
    hasher.update_field(&batches.iter().map(RecordBatch::num_rows).sum::<usize>().to_string());
    hasher.update_field(&schema.fields().len().to_string());
    for batch in batches {
        for row in 0..batch.num_rows() {
            for column in 0..batch.num_columns() {
                let array = batch.column(column);
                if array.is_null(row) {
                    hasher.update_field("null");
                } else {
                    hasher.update_field("value");
                    hasher.update_field(&array_value_to_string(array.as_ref(), row)?);
                }
            }
        }
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}

fn query_ids_from_env(name: &'static str, default: &[&'static str]) -> DataFusionResult<Vec<String>> {
    match std::env::var_os(name) {
        Some(value) => {
            let parsed = value
                .to_string_lossy()
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if parsed.is_empty() {
                return Err(plan_error(format!("{name} must contain at least one query id")));
            }
            Ok(parsed)
        },
        None => Ok(default.iter().map(|value| (*value).to_string()).collect()),
    }
}

fn validate_query_ids(query_ids: &[String]) -> DataFusionResult<()> {
    let known = TPCDS_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(plan_error(format!("unknown TPC-DS query id in stress config: {query_id}")));
        }
    }
    Ok(())
}

fn env_usize(name: &'static str, default: usize) -> DataFusionResult<usize> {
    match std::env::var_os(name) {
        Some(value) => value
            .to_string_lossy()
            .parse()
            .map_err(|error| plan_error(format!("failed to parse {name} as usize: {error}"))),
        None => Ok(default),
    }
}

fn env_u64(name: &'static str, default: u64) -> DataFusionResult<u64> {
    match std::env::var_os(name) {
        Some(value) => value
            .to_string_lossy()
            .parse()
            .map_err(|error| plan_error(format!("failed to parse {name} as u64: {error}"))),
        None => Ok(default),
    }
}

fn env_bool(name: &'static str, default: bool) -> DataFusionResult<bool> {
    match std::env::var_os(name) {
        Some(value) => match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(plan_error(format!("failed to parse {name} as bool: {other}"))),
        },
        None => Ok(default),
    }
}

fn validate_nonzero<T>(name: &'static str, value: T) -> DataFusionResult<()>
where T: PartialEq + From<u8> {
    if value == T::from(0) {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}

fn default_device_capacity_bytes() -> u64 {
    // This standalone benchmark profile must not probe GPU 0 and reuse that
    // capacity for every selected device. Server backends resolve each device.
    NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES
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

#[cfg(test)]
mod tests {
    use super::*;

    fn queue_test_config() -> StressConfig {
        StressConfig {
            dataset_root: PathBuf::new(),
            bulk_queries: vec!["q01".to_string()],
            interactive_queries: vec!["q10".to_string()],
            bulk_iterations: 1,
            interactive_iterations: 1,
            device_capacity_bytes: 100,
            max_active_attempts_per_device: 2,
            max_queued_attempts: 2,
            max_queue_wait: Duration::from_secs(1),
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            query_timeout: Duration::from_secs(1),
            run_overload_probe: false,
        }
    }

    #[test]
    fn query_ids_reject_unknown_tpcds_id() {
        let error = validate_query_ids(&["q100".to_string()]).unwrap_err();
        assert!(error.to_string().contains("unknown TPC-DS query id"));
    }

    #[test]
    fn tsv_escape_replaces_control_delimiters() {
        assert_eq!(tsv_escape("a\tb\nc\rd"), "a\\tb\\nc\\rd");
    }

    #[test]
    fn stress_tsv_row_contract_matches_columns() {
        assert_eq!(TSV_COLUMNS.first(), Some(&"row_type"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"error_message"));
        let row = StressRow {
            row_type: "query",
            request_index: 1,
            client_id: "interactive".to_string(),
            workload_class: "interactive",
            query_id: "q10".to_string(),
            iteration: 1,
            admission_sequence: Some(1),
            admission_wait_ms: Some(0.0),
            total_wall_ms: 10.0,
            native_wall_ms: Some(5.0),
            planning_outcome: "selected".to_string(),
            final_disposition: "native".to_string(),
            native_runtime_outcome: "succeeded",
            parity_status: "matched",
            cpu_output_rows: Some(1),
            native_output_rows: Some(1),
            cpu_output_fingerprint: Some("cpu".to_string()),
            native_output_fingerprint: Some("native".to_string()),
            error: StructuredFailure::default(),
        };
        assert_eq!(row.to_tsv_line().split('\t').count(), TSV_COLUMNS.len());
    }

    #[test]
    fn stress_config_rejects_queue_smaller_than_synchronized_wave() {
        let config = queue_test_config();

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains(MAX_QUEUED_ATTEMPTS_ENV));
        assert!(error.to_string().contains("synchronized request count (3)"));
    }
}
