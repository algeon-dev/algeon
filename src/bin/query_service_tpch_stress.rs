use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{record_batch::RecordBatch, util::display::array_value_to_string};
use cudf_query_engine::{QueryError, QueryService, QueryServiceConfig, QueryServiceTenantConfig, TenantId};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::SessionContext,
};
use datafusion_cudf::{
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV,
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error as AdapterError,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
    NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath, TPCH_QUERIES, TPCH_TABLES, TpchQuery,
    default_tpch_parquet_root, describe_tpch_dataset, discover_tpch_table_layout, register_tpch_table_layout,
    tpch_report_session_config, tpch_tuned_cpu_session_config,
};
use tokio::sync::Barrier;

const TSV_SCHEMA_VERSION: u32 = 1;

const DATASET_ROOT_ENV: &str = CUDF_TPCH_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS";
const TENANT_BUDGET_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_TENANT_BUDGET_BYTES";
const MAX_CONCURRENT_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_CONCURRENT_QUERIES";
const MAX_ADMISSION_QUEUE_LEN_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_ADMISSION_QUEUE_LEN";
const ADMISSION_WAIT_TIMEOUT_MS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_ADMISSION_WAIT_TIMEOUT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK";
const OVERLOAD_PROBE_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_OVERLOAD_PROBE";

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = StressConfig::from_env()?;
    let dataset = describe_tpch_dataset(
        &config.dataset_root,
        std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "sf1".to_string()),
        "query_service_tpch_stress",
        std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unspecified".to_string()),
    )
    .map_err(|error| plan_error(error.to_string()))?;
    let cpu_baselines = Arc::new(compute_cpu_baselines(&config).await?);
    let mut rows = Vec::new();
    if config.run_overload_probe {
        rows.push(run_overload_probe(&config)?);
    }
    rows.extend(run_stress(&config, Arc::clone(&cpu_baselines)).await?);
    let report = StressReport { config, dataset, rows };
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
    tenant_budget_bytes: u64,
    max_concurrent_queries: usize,
    max_admission_queue_len: usize,
    admission_wait_timeout: Duration,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    run_overload_probe: bool,
}

impl StressConfig {
    fn from_env() -> DataFusionResult<Self> {
        let dataset_root = match std::env::var_os(DATASET_ROOT_ENV) {
            Some(value) => PathBuf::from(value),
            None => default_tpch_parquet_root(datafusion_cudf::TpchDefaultScale::Sf1).into(),
        };
        let config = Self {
            dataset_root,
            bulk_queries: query_ids_from_env(BULK_QUERIES_ENV, &["q01", "q09", "q18", "q21"])?,
            interactive_queries: query_ids_from_env(INTERACTIVE_QUERIES_ENV, &["q06", "q14", "q22"])?,
            bulk_iterations: env_usize(BULK_ITERATIONS_ENV, 1)?,
            interactive_iterations: env_usize(INTERACTIVE_ITERATIONS_ENV, 4)?,
            tenant_budget_bytes: env_u64(TENANT_BUDGET_BYTES_ENV, default_tenant_budget_bytes())?,
            max_concurrent_queries: env_usize(MAX_CONCURRENT_QUERIES_ENV, 1)?,
            max_admission_queue_len: env_usize(MAX_ADMISSION_QUEUE_LEN_ENV, 64)?,
            admission_wait_timeout: Duration::from_millis(env_u64(ADMISSION_WAIT_TIMEOUT_MS_ENV, 30_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
            run_overload_probe: env_bool(OVERLOAD_PROBE_ENV, true)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero(BULK_ITERATIONS_ENV, self.bulk_iterations)?;
        validate_nonzero(INTERACTIVE_ITERATIONS_ENV, self.interactive_iterations)?;
        validate_nonzero(TENANT_BUDGET_BYTES_ENV, self.tenant_budget_bytes)?;
        validate_nonzero(MAX_CONCURRENT_QUERIES_ENV, self.max_concurrent_queries)?;
        validate_nonzero(MAX_ADMISSION_QUEUE_LEN_ENV, self.max_admission_queue_len)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if self.admission_wait_timeout.is_zero() {
            return Err(plan_error(format!("{ADMISSION_WAIT_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        Ok(())
    }

    fn service_config(&self) -> QueryServiceConfig {
        QueryServiceConfig {
            max_concurrent_queries: self.max_concurrent_queries,
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
}

#[derive(Debug)]
struct StressReport {
    config: StressConfig,
    dataset: datafusion_cudf::TpchDatasetMetadata,
    rows: Vec<StressRow>,
}

impl StressReport {
    fn to_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# query_service_tpch_stress_tsv_schema_version={TSV_SCHEMA_VERSION}\n"));
        output.push_str(&format!("# dataset_root={}\n", self.config.dataset_root.display()));
        output.push_str(&format!("# dataset_metadata_fingerprint={}\n", self.dataset.metadata_fingerprint));
        output.push_str(&format!("# tenant_budget_bytes={}\n", self.config.tenant_budget_bytes));
        output.push_str(&format!("# max_concurrent_queries={}\n", self.config.max_concurrent_queries));
        output.push_str(&format!("# max_admission_queue_len={}\n", self.config.max_admission_queue_len));
        for summary in tenant_summaries(&self.rows) {
            output.push_str(&format!(
                "# tenant_summary tenant_id={} succeeded={} overloaded={} failed={} max_total_wall_ms={} p95_total_wall_ms={}\n",
                summary.tenant_id,
                summary.succeeded,
                summary.overloaded,
                summary.failed,
                format_ms(summary.max_total_wall_ms),
                format_ms(summary.p95_total_wall_ms),
            ));
        }
        output.push_str("row_type\trequest_index\ttenant_id\tworkload_class\tquery_id\titeration\tadmission_sequence\tadmission_wait_ms\ttotal_wall_ms\tnative_wall_ms\tplanning_outcome\tpost_native_execution_path\tnative_runtime_outcome\tparity_status\tcpu_output_rows\tnative_output_rows\tcpu_output_fingerprint\tnative_output_fingerprint\terror_code\terror_kind\terror_status\terror_operation\terror_facts\terror_message\n");
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl StressRow {
    fn to_tsv_line(&self) -> String {
        [
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
        ]
        .into_iter()
        .map(|field| tsv_escape(&field))
        .collect::<Vec<_>>()
        .join("\t")
    }
}

#[derive(Debug, Clone)]
struct TenantSummary {
    tenant_id: String,
    succeeded: usize,
    overloaded: usize,
    failed: usize,
    max_total_wall_ms: f64,
    p95_total_wall_ms: f64,
}

async fn compute_cpu_baselines(config: &StressConfig) -> DataFusionResult<HashMap<String, CpuBaseline>> {
    let mut query_ids = BTreeSet::new();
    query_ids.extend(config.bulk_queries.iter().cloned());
    query_ids.extend(config.interactive_queries.iter().cloned());
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

async fn run_stress(
    config: &StressConfig,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<Vec<StressRow>> {
    let service = Arc::new(
        QueryService::new(
            config.service_config(),
            vec![
                QueryServiceTenantConfig::new("bulk", config.tenant_budget_bytes),
                QueryServiceTenantConfig::new("interactive_a", config.tenant_budget_bytes),
                QueryServiceTenantConfig::new("interactive_b", config.tenant_budget_bytes),
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
    let query = tpch_query(&request.query_id)?;
    let cpu = cpu_baselines.get(&request.query_id);
    let native_started = Instant::now();
    match collect_native_with_handle(config, query, Arc::clone(&handle)).await {
        Ok(NativeCollectOutcome::Succeeded { planning_row, measurement: native }) => {
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
        Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error }) => Ok(failure_row(
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
        )),
        Err(error) => Ok(failure_row(
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
        )),
    }
}

async fn collect_native_with_handle(
    config: &StressConfig,
    query: &TpchQuery,
    handle: Arc<cudf_query_engine::QueryHandle>,
) -> DataFusionResult<NativeCollectOutcome> {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
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
    let service = QueryService::new(
        probe_config,
        vec![
            QueryServiceTenantConfig::new("bulk", config.tenant_budget_bytes),
            QueryServiceTenantConfig::new("interactive_a", config.tenant_budget_bytes),
        ],
    )
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
            let failed = rows.len().saturating_sub(succeeded + overloaded);
            TenantSummary {
                tenant_id,
                succeeded,
                overloaded,
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

async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> DataFusionResult<()> {
    for table in TPCH_TABLES {
        let layout = discover_tpch_table_layout(root, table).map_err(|error| plan_error(error.to_string()))?;
        register_tpch_table_layout(ctx, table, &layout).await.map_err(|error| plan_error(error.to_string()))?;
    }
    Ok(())
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

fn tpch_query(query_id: &str) -> DataFusionResult<&'static TpchQuery> {
    TPCH_QUERIES
        .iter()
        .find(|query| query.id == query_id)
        .ok_or_else(|| plan_error(format!("unknown TPC-H query id: {query_id}")))
}

fn measure_output(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<CpuBaseline> {
    Ok(CpuBaseline {
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_fingerprint: fingerprint_batches(schema, batches)?,
    })
}

fn fingerprint_batches(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-cudf-query-service-tpch-output-v1");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(plan_error("TPC-H output fingerprint requires collected batches to match the planned schema"));
        }
    }
    hasher.update_field("schema_metadata");
    update_metadata(&mut hasher, schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&field.data_type().to_string());
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        update_metadata(&mut hasher, field.metadata());
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

fn update_metadata(hasher: &mut StableFnv1a64, metadata: &HashMap<String, String>) {
    hasher.update_field(&metadata.len().to_string());
    let mut keys = metadata.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        hasher.update_field(key);
        hasher.update_field(metadata.get(key).expect("metadata key collected from map must exist"));
    }
}

#[derive(Debug, Clone)]
struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    fn new() -> Self {
        Self { state: 0xcbf29ce484222325 }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0xff]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
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
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(plan_error(format!("unknown TPC-H query id in stress config: {query_id}")));
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

fn default_tenant_budget_bytes() -> u64 {
    let total = NativeCudfExecutionConfig::auto_sized_device_budget_bytes();
    (total / 3).max(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
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

    #[test]
    fn percentile_uses_ceil_index_for_tail_latency() {
        assert_eq!(percentile(&[], 0.95), 0.0);
        assert_eq!(percentile(&[10.0], 0.95), 10.0);
        assert_eq!(percentile(&[10.0, 20.0, 30.0], 0.95), 30.0);
    }

    #[test]
    fn tsv_escape_removes_row_and_field_breaks() {
        assert_eq!(tsv_escape("a\tb\nc\rd"), "a\\tb\\nc\\rd");
    }

    #[test]
    fn tenant_summary_counts_success_overload_and_failures() {
        let rows = vec![
            StressRow {
                row_type: "query",
                request_index: 1,
                tenant_id: "interactive".to_string(),
                workload_class: "interactive",
                query_id: "q06".to_string(),
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
                cpu_output_fingerprint: Some("a".to_string()),
                native_output_fingerprint: Some("a".to_string()),
                error: StructuredFailure::default(),
            },
            StressRow {
                row_type: "query",
                request_index: 2,
                tenant_id: "interactive".to_string(),
                workload_class: "interactive",
                query_id: "q06".to_string(),
                iteration: 2,
                admission_sequence: None,
                admission_wait_ms: None,
                total_wall_ms: 20.0,
                native_wall_ms: None,
                planning_outcome: "not_planned".to_string(),
                post_native_execution_path: "not_planned".to_string(),
                native_runtime_outcome: "not_started",
                parity_status: "not_run",
                cpu_output_rows: None,
                native_output_rows: None,
                cpu_output_fingerprint: None,
                native_output_fingerprint: None,
                error: StructuredFailure { code: Some("service_overloaded".to_string()), ..Default::default() },
            },
        ];

        let summaries = tenant_summaries(&rows);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].succeeded, 1);
        assert_eq!(summaries[0].overloaded, 1);
        assert_eq!(summaries[0].failed, 0);
        assert_eq!(summaries[0].max_total_wall_ms, 20.0);
    }
}
