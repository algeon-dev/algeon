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
    error::Error as AdapterError,
    native_exec::NativeCudfExecutionConfig,
    native_report::{
        NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPlanningRow,
        NativeDataFusionPostNativeExecutionPath,
    },
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    policy::GpuFallbackPolicy,
    session::CudfSessionStateBuilderExt,
};
use datafusion_nexus_bench::{
    report_workload::tpcds::{
        register_tpcds_tables_with_native_metadata, tpcds_native_report_session_config, tpcds_report_session_config,
    },
    stress_harness::validate_synchronized_admission_queue,
};
use datafusion_nexus_workloads::tpcds::{
    CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_PARQUET_DIR_ENV, CUDF_TPCDS_SCALE_FACTOR_ENV, TPCDS_QUERIES,
    TpcdsDatasetMetadata, TpcdsQuery, default_tpcds_sf1_parquet_root, describe_tpcds_dataset,
};
use nexus_query_engine::{
    error::QueryError,
    harness::StableFnv1a64,
    runtime::{QueryService, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId},
};
use tokio::{sync::Barrier, time::timeout};
const TSV_COLUMNS: &[&str] = &[
    "row_type",
    "request_index",
    "tenant_id",
    "workload_class",
    "query_id",
    "iteration",
    "admission_sequence",
    "admission_wait_ms",
    "total_wall_ms",
    "native_wall_ms",
    "planning_outcome",
    "post_native_execution_path",
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
const DATASET_ROOT_ENV: &str = CUDF_TPCDS_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_INTERACTIVE_ITERATIONS";
const TENANT_BUDGET_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_TENANT_BUDGET_BYTES";
const DEVICE_CAPACITY_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_DEVICE_CAPACITY_BYTES";
const TENANT_MIN_GRANT_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_TENANT_MIN_GRANT_BYTES";
const TENANT_WANT_GRANT_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_TENANT_WANT_GRANT_BYTES";
const SLOT_GUARANTEE_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_SLOT_GUARANTEE";
const MAX_CONCURRENT_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_MAX_CONCURRENT_QUERIES";
const MAX_ADMISSION_QUEUE_LEN_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_MAX_ADMISSION_QUEUE_LEN";
const ADMISSION_WAIT_TIMEOUT_MS_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_ADMISSION_WAIT_TIMEOUT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_MAX_ROW_GROUPS_PER_CHUNK";
const QUERY_TIMEOUT_MS_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_QUERY_TIMEOUT_MS";
const OVERLOAD_PROBE_ENV: &str = "CUDF_QUERY_SERVICE_TPCDS_OVERLOAD_PROBE";
#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = StressConfig::from_env()?;
    let dataset = describe_tpcds_dataset(
        &config.dataset_root,
        std::env::var(CUDF_TPCDS_SCALE_FACTOR_ENV).unwrap_or_else(|_| "sf1".to_string()),
        "query_service_tpcds_stress",
        std::env::var(CUDF_TPCDS_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unspecified".to_string()),
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
    tenant_min_grant_bytes: u64,
    tenant_want_grant_bytes: u64,
    slot_guarantee: SlotGuarantee,
    max_concurrent_queries: usize,
    max_admission_queue_len: usize,
    admission_wait_timeout: Duration,
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
        let max_concurrent_queries = env_usize(MAX_CONCURRENT_QUERIES_ENV, 1)?;
        let legacy_tenant_budget_bytes = optional_env_u64(TENANT_BUDGET_BYTES_ENV)?;
        let tenant_min_grant_bytes = optional_env_u64(TENANT_MIN_GRANT_BYTES_ENV)?;
        let tenant_want_grant_bytes = optional_env_u64(TENANT_WANT_GRANT_BYTES_ENV)?;
        if legacy_tenant_budget_bytes.is_some()
            && (tenant_min_grant_bytes.is_some() || tenant_want_grant_bytes.is_some())
        {
            return Err(plan_error(format!(
                "{TENANT_BUDGET_BYTES_ENV} cannot be combined with {TENANT_MIN_GRANT_BYTES_ENV} or {TENANT_WANT_GRANT_BYTES_ENV}"
            )));
        }
        let device_capacity_bytes = env_u64(
            DEVICE_CAPACITY_BYTES_ENV,
            default_device_capacity_bytes(legacy_tenant_budget_bytes, max_concurrent_queries),
        )?;
        let default_min_grant_bytes = grant_fraction_bytes(device_capacity_bytes, 1, 8);
        let default_want_grant_bytes = grant_fraction_bytes(device_capacity_bytes, 1, 2);
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
            tenant_min_grant_bytes: tenant_min_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_min_grant_bytes),
            tenant_want_grant_bytes: tenant_want_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_want_grant_bytes),
            slot_guarantee: env_slot_guarantee(SLOT_GUARANTEE_ENV, SlotGuarantee::GuaranteedMinimum)?,
            max_concurrent_queries,
            max_admission_queue_len: env_usize(MAX_ADMISSION_QUEUE_LEN_ENV, 96)?,
            admission_wait_timeout: Duration::from_millis(env_u64(ADMISSION_WAIT_TIMEOUT_MS_ENV, 60_000)?),
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
        validate_nonzero(TENANT_MIN_GRANT_BYTES_ENV, self.tenant_min_grant_bytes)?;
        validate_nonzero(TENANT_WANT_GRANT_BYTES_ENV, self.tenant_want_grant_bytes)?;
        if self.tenant_want_grant_bytes < self.tenant_min_grant_bytes {
            return Err(plan_error(format!(
                "{TENANT_WANT_GRANT_BYTES_ENV} must be at least {TENANT_MIN_GRANT_BYTES_ENV}"
            )));
        }
        if self.tenant_min_grant_bytes > self.device_capacity_bytes {
            return Err(plan_error(format!(
                "{TENANT_MIN_GRANT_BYTES_ENV} must not exceed {DEVICE_CAPACITY_BYTES_ENV}"
            )));
        }
        if self.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
            let required_min_capacity = self
                .tenant_min_grant_bytes
                .saturating_mul(u64::try_from(self.max_concurrent_queries).unwrap_or(u64::MAX));
            if required_min_capacity > self.device_capacity_bytes {
                return Err(plan_error(format!(
                    "{TENANT_MIN_GRANT_BYTES_ENV} must fit {MAX_CONCURRENT_QUERIES_ENV} within {DEVICE_CAPACITY_BYTES_ENV} when {SLOT_GUARANTEE_ENV}=guaranteed_minimum"
                )));
            }
        }
        validate_nonzero(MAX_CONCURRENT_QUERIES_ENV, self.max_concurrent_queries)?;
        validate_nonzero(MAX_ADMISSION_QUEUE_LEN_ENV, self.max_admission_queue_len)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if self.admission_wait_timeout.is_zero() {
            return Err(plan_error(format!("{ADMISSION_WAIT_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        if self.query_timeout.is_zero() {
            return Err(plan_error(format!("{QUERY_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        validate_synchronized_admission_queue(
            MAX_ADMISSION_QUEUE_LEN_ENV,
            self.max_admission_queue_len,
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

    fn service_config(&self) -> QueryServiceConfig {
        QueryServiceConfig {
            device_capacity_bytes: Some(self.device_capacity_bytes),
            max_concurrent_queries: self.max_concurrent_queries,
            slot_guarantee: self.slot_guarantee,
            max_admission_queue_len: self.max_admission_queue_len,
            admission_wait_timeout: self.admission_wait_timeout,
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            ..QueryServiceConfig::default()
        }
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
        adapter_error_from_datafusion(error).map(Self::from_adapter_error).unwrap_or_else(|| Self {
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
        output.push_str(&format!("# tenant_min_grant_bytes={}\n", self.config.tenant_min_grant_bytes));
        output.push_str(&format!("# tenant_want_grant_bytes={}\n", self.config.tenant_want_grant_bytes));
        output.push_str(&format!("# slot_guarantee={}\n", slot_guarantee_name(self.config.slot_guarantee)));
        output.push_str(&format!("# max_concurrent_queries={}\n", self.config.max_concurrent_queries));
        output.push_str(&format!("# effective_active_max={}\n", self.effective_active_max));
        output.push_str(&format!("# max_admission_queue_len={}\n", self.config.max_admission_queue_len));
        output.push_str(&format!("# query_timeout_ms={}\n", self.config.query_timeout.as_millis()));
        for summary in tenant_summaries(&self.rows) {
            output.push_str(&format!(
                "# tenant_summary tenant_id={} succeeded={} overloaded={} timed_out={} failed={} max_total_wall_ms={} p95_total_wall_ms={}\n",
                summary.tenant_id,
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
            self.tenant_id.clone(),
            self.workload_class.to_string(),
            self.query_id.clone(),
            self.iteration.to_string(),
            self.admission_sequence.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_wait_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            format_ms(self.total_wall_ms),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            self.planning_outcome.clone(),
            self.post_native_execution_path.clone(),
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
struct TenantSummary {
    tenant_id: String,
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
    let service = Arc::new(
        QueryService::new(
            config.service_config(),
            vec![
                tenant_config("bulk", config),
                tenant_config("interactive_a", config),
                tenant_config("interactive_b", config),
            ],
        )
        .map_err(native_error)?,
    );
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
        rows.push(
            task.await
                .map_err(|error| plan_error(format!("query service TPC-DS stress task join failed: {error}")))??,
        );
    }
    rows.sort_by_key(|row| row.request_index);
    let effective_active_max = service.gpu_execution_query_report().peak_active_query_jobs;
    Ok(StressRun { rows, effective_active_max })
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
        "[query-service-tpcds] native start tenant={} query={} iteration={} admission_sequence={}",
        request.tenant_id,
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
                "[query-service-tpcds] native finish tenant={} query={} iteration={} elapsed_ms={} outcome=succeeded parity={parity}",
                request.tenant_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(StressRow {
                row_type: "query",
                request_index: request.request_index,
                tenant_id: request.tenant_id.to_string(),
                workload_class: request.workload_class,
                query_id: request.query_id,
                iteration: request.iteration,
                admission_sequence,
                admission_wait_ms,
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
        Ok(Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error })) => {
            eprintln!(
                "[query-service-tpcds] native finish tenant={} query={} iteration={} elapsed_ms={} outcome=failed",
                request.tenant_id,
                request.query_id,
                request.iteration,
                native_started.elapsed().as_millis()
            );
            Ok(failure_row(
                &request,
                total_started.elapsed(),
                admission_sequence,
                admission_wait_ms,
                planning_row.outcome.as_str(),
                planning_row.post_native_execution_path.as_str(),
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            ))
        },
        Ok(Err(error)) => {
            eprintln!(
                "[query-service-tpcds] native finish tenant={} query={} iteration={} elapsed_ms={} outcome=failed",
                request.tenant_id,
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
                "[query-service-tpcds] native finish tenant={} query={} iteration={} elapsed_ms={} outcome=timed_out",
                request.tenant_id,
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
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpcds_native_report_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, &config.dataset_root).await?;
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

fn tenant_summaries(rows: &[StressRow]) -> Vec<TenantSummary> {
    let mut by_tenant: BTreeMap<String, Vec<&StressRow>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.row_type == "query") {
        by_tenant.entry(row.tenant_id.clone()).or_default().push(row);
    }
    by_tenant
        .into_iter()
        .map(|(tenant_id, rows)| {
            let mut latencies = rows.iter().map(|row| row.total_wall_ms).collect::<Vec<_>>();
            latencies.sort_by(f64::total_cmp);
            let succeeded = rows.iter().filter(|row| row.native_runtime_outcome == "succeeded").count();
            let overloaded = rows.iter().filter(|row| row.error.code.as_deref() == Some("service_overloaded")).count();
            let timed_out = rows.iter().filter(|row| row.native_runtime_outcome == "timed_out").count();
            let failed = rows.len().saturating_sub(succeeded + overloaded + timed_out);
            TenantSummary {
                tenant_id,
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

fn optional_env_u64(name: &'static str) -> DataFusionResult<Option<u64>> {
    std::env::var_os(name)
        .map(|value| {
            value
                .to_string_lossy()
                .parse()
                .map_err(|error| plan_error(format!("failed to parse {name} as u64: {error}")))
        })
        .transpose()
}

fn env_slot_guarantee(name: &'static str, default: SlotGuarantee) -> DataFusionResult<SlotGuarantee> {
    match std::env::var_os(name) {
        Some(value) => parse_slot_guarantee(name, &value.to_string_lossy()),
        None => Ok(default),
    }
}

fn parse_slot_guarantee(name: &'static str, value: &str) -> DataFusionResult<SlotGuarantee> {
    match value.trim().to_ascii_lowercase().as_str() {
        "guaranteed_minimum" => Ok(SlotGuarantee::GuaranteedMinimum),
        "optimistic" => Ok(SlotGuarantee::Optimistic),
        other => {
            Err(plan_error(format!("failed to parse {name}: expected guaranteed_minimum or optimistic, got {other}")))
        },
    }
}

fn slot_guarantee_name(slot_guarantee: SlotGuarantee) -> &'static str {
    match slot_guarantee {
        SlotGuarantee::GuaranteedMinimum => "guaranteed_minimum",
        SlotGuarantee::Optimistic => "optimistic",
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

fn default_device_capacity_bytes(legacy_tenant_budget_bytes: Option<u64>, max_concurrent_queries: usize) -> u64 {
    if let Some(tenant_budget_bytes) = legacy_tenant_budget_bytes {
        return tenant_budget_bytes.saturating_mul(u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX));
    }
    NativeCudfExecutionConfig::auto_sized_device_budget_bytes()
}

fn grant_fraction_bytes(device_capacity_bytes: u64, numerator: u64, denominator: u64) -> u64 {
    device_capacity_bytes.saturating_mul(numerator) / denominator
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
            tenant_min_grant_bytes: 25,
            tenant_want_grant_bytes: 50,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_concurrent_queries: 2,
            max_admission_queue_len: 2,
            admission_wait_timeout: Duration::from_secs(1),
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
            tenant_id: "interactive".to_string(),
            workload_class: "interactive",
            query_id: "q10".to_string(),
            iteration: 1,
            admission_sequence: Some(1),
            admission_wait_ms: Some(0.0),
            total_wall_ms: 10.0,
            native_wall_ms: Some(5.0),
            planning_outcome: "accepted_replaced".to_string(),
            post_native_execution_path: "native".to_string(),
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

        assert!(error.to_string().contains(MAX_ADMISSION_QUEUE_LEN_ENV));
        assert!(error.to_string().contains("synchronized request count (3)"));
    }
}
