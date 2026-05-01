use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{array::Array, record_batch::RecordBatch};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::collect,
    prelude::SessionContext,
};
use datafusion_nexus::{
    error::{DataFusionErrorExt, Error as AdapterError},
    native_exec::NexusNativeExecutionConfig,
    native_report::{
        FinalPlanDisposition, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
        NativeDataFusionPlanningRow,
    },
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
    table_format::{
        TableFormatScanRequest,
        iceberg::{
            GlueIcebergTableConfig, IcebergTableFormatPlanner, NexusIcebergTableProvider, load_glue_table,
            source_access_policy_to_native_source_storage_profile,
        },
    },
};
use nexus_query_engine::{
    error::QueryError,
    runtime::{GpuBackendSession, QueryExecutionConfig, QueueBounds},
};
use tokio::sync::Barrier;

use crate::{
    gpu_backend::GpuBackendOwner, report_workload::tpch::tpch_report_session_config,
    stress_harness::validate_synchronized_admission_queue,
};
const TSV_COLUMNS: &[&str] = &[
    "row_type",
    "request_index",
    "client_id",
    "iteration",
    "admission_sequence",
    "admission_wait_ms",
    "total_wall_ms",
    "native_wall_ms",
    "planning_outcome",
    "final_disposition",
    "native_runtime_outcome",
    "count_status",
    "expected_count",
    "native_count",
    "native_output_rows",
    "error_code",
    "error_kind",
    "error_status",
    "error_operation",
    "error_facts",
    "error_message",
];
const AWS_REGION_ENV: &str = "AWS_REGION";
const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";
const NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_NAME";
const NEXUS_ICEBERG_GLUE_DATABASE_ENV: &str = "NEXUS_ICEBERG_GLUE_DATABASE";
const NEXUS_ICEBERG_GLUE_TABLE_ENV: &str = "NEXUS_ICEBERG_GLUE_TABLE";
const NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "NEXUS_ICEBERG_GLUE_WAREHOUSE";
const NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_ID";
const DEFAULT_CATALOG_NAME: &str = "glue";
const DEFAULT_DATABASE: &str = "tpch_sf1";
const DEFAULT_TABLE: &str = "lineitem";
const DEFAULT_WAREHOUSE: &str = "s3://lakehouse/warehouse";
const DEFAULT_CATALOG_ID: &str = "";
const DEFAULT_REGION: &str = "us-east-1";
const QUERY_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_SQL";
const CLIENT_IDS_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_CLIENT_IDS";
const ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_ITERATIONS";
const ATTEMPT_FLOOR_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_ATTEMPT_FLOOR_BYTES";
const MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_MAX_ACTIVE_ATTEMPTS_PER_DEVICE";
const MAX_QUEUED_ATTEMPTS_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_MAX_QUEUED_ATTEMPTS";
const MAX_QUEUE_WAIT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_MAX_QUEUE_WAIT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_MAX_ROW_GROUPS_PER_CHUNK";
const OVERLOAD_PROBE_ENV: &str = "NEXUS_QUERY_SERVICE_ICEBERG_OVERLOAD_PROBE";

pub async fn run() -> DataFusionResult<()> {
    let config = StressConfig::from_env()?;
    let expected_count = planned_table_row_count(&config.glue).await?;
    let mut rows = Vec::new();
    if config.run_overload_probe {
        rows.push(run_overload_probe(&config)?);
    }
    let stress = run_stress(&config, expected_count).await?;
    rows.extend(stress.rows);
    let report = StressReport { config, expected_count, rows, effective_active_max: stress.effective_active_max };
    print!("{}", report.to_tsv());
    Ok(())
}

#[derive(Debug, Clone)]
struct StressConfig {
    glue: GlueIcebergTableConfig,
    table_alias: String,
    sql: String,
    client_ids: Vec<String>,
    iterations: usize,
    attempt_floor_bytes: u64,
    max_active_attempts_per_device: usize,
    max_queued_attempts: usize,
    max_queue_wait: Duration,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    run_overload_probe: bool,
}

impl StressConfig {
    fn from_env() -> DataFusionResult<Self> {
        let catalog_name = env_string(NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, DEFAULT_CATALOG_NAME);
        let database = env_string(NEXUS_ICEBERG_GLUE_DATABASE_ENV, DEFAULT_DATABASE);
        let table = env_string(NEXUS_ICEBERG_GLUE_TABLE_ENV, DEFAULT_TABLE);
        let warehouse = env_string(NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, DEFAULT_WAREHOUSE);
        let region = std::env::var(AWS_REGION_ENV)
            .or_else(|_| std::env::var(AWS_DEFAULT_REGION_ENV))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());
        let mut glue = GlueIcebergTableConfig::new(catalog_name, database, table, warehouse, region);
        glue.catalog_id = match std::env::var(NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV) {
            Ok(value) if value.trim().is_empty() => None,
            Ok(value) => Some(value),
            Err(_) if DEFAULT_CATALOG_ID.trim().is_empty() => None,
            Err(_) => Some(DEFAULT_CATALOG_ID.to_string()),
        };
        let table_alias = "iceberg_t".to_string();
        let sql =
            std::env::var(QUERY_ENV).unwrap_or_else(|_| format!("select count(*) as row_count from {table_alias}"));
        let config = Self {
            glue,
            table_alias,
            sql,
            client_ids: client_ids_from_env(CLIENT_IDS_ENV, &["bulk", "interactive_a", "interactive_b"])?,
            iterations: env_usize(ITERATIONS_ENV, 2)?,
            attempt_floor_bytes: env_u64(ATTEMPT_FLOOR_BYTES_ENV, default_attempt_floor_bytes())?,
            max_active_attempts_per_device: env_usize(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, 1)?,
            max_queued_attempts: env_usize(MAX_QUEUED_ATTEMPTS_ENV, 64)?,
            max_queue_wait: Duration::from_millis(env_u64(MAX_QUEUE_WAIT_MS_ENV, 30_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
            run_overload_probe: env_bool(OVERLOAD_PROBE_ENV, true)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> DataFusionResult<()> {
        if self.client_ids.len() < 2 {
            return Err(plan_error(format!("{CLIENT_IDS_ENV} must include at least two client IDs")));
        }
        validate_nonzero(ITERATIONS_ENV, self.iterations)?;
        validate_nonzero(ATTEMPT_FLOOR_BYTES_ENV, self.attempt_floor_bytes)?;
        validate_nonzero(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, self.max_active_attempts_per_device)?;
        validate_nonzero(MAX_QUEUED_ATTEMPTS_ENV, self.max_queued_attempts)?;
        validate_synchronized_admission_queue(
            MAX_QUEUED_ATTEMPTS_ENV,
            self.max_queued_attempts,
            self.synchronized_request_count(),
        )
        .map_err(plan_error)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if self.max_queue_wait.is_zero() {
            return Err(plan_error(format!("{MAX_QUEUE_WAIT_MS_ENV} must be greater than zero")));
        }
        Ok(())
    }

    fn synchronized_request_count(&self) -> usize {
        self.client_ids.len().saturating_mul(self.iterations)
    }

    fn service_config(&self) -> QueryExecutionConfig {
        QueryExecutionConfig {
            source_storage_profiles: vec![source_access_policy_to_native_source_storage_profile(
                &self.glue.source_access_policy_from_env(),
            )],
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            ..QueryExecutionConfig::default()
        }
    }

    fn queue_bounds(&self) -> QueueBounds {
        QueueBounds {
            max_waiters: self.max_queued_attempts,
            max_wait: self.max_queue_wait,
            max_overtakes_per_waiter: 4,
        }
    }
}

#[derive(Debug, Clone)]
struct StressRequest {
    request_index: usize,
    client_id: String,
    iteration: usize,
}

#[derive(Debug, Clone)]
struct StressRow {
    row_type: &'static str,
    request_index: usize,
    client_id: String,
    iteration: usize,
    admission_sequence: Option<u64>,
    admission_wait_ms: Option<f64>,
    total_wall_ms: f64,
    native_wall_ms: Option<f64>,
    planning_outcome: String,
    final_disposition: String,
    native_runtime_outcome: &'static str,
    count_status: &'static str,
    expected_count: Option<u64>,
    native_count: Option<u64>,
    native_output_rows: Option<usize>,
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
    Succeeded { planning_row: NativeDataFusionPlanningRow, measurement: NativeMeasurement },
    RuntimeFailed { planning_row: NativeDataFusionPlanningRow, error: DataFusionError },
}

#[derive(Debug, Clone)]
struct NativeMeasurement {
    output_rows: usize,
    count_value: Option<u64>,
}

#[derive(Debug)]
struct StressReport {
    config: StressConfig,
    expected_count: Option<u64>,
    rows: Vec<StressRow>,
    effective_active_max: usize,
}

struct StressRun {
    rows: Vec<StressRow>,
    effective_active_max: usize,
}

impl StressReport {
    fn to_tsv(&self) -> String {
        let diagnostics = self.config.glue.diagnostics_from_env();
        let mut output = String::new();
        output.push_str(&format!("# table_ident={}.{}\n", diagnostics.database, diagnostics.table));
        output.push_str(&format!("# warehouse={}\n", diagnostics.warehouse));
        output.push_str(&format!("# region={}\n", diagnostics.region));
        output.push_str(&format!("# credential_source={}\n", diagnostics.credential_source.code()));
        output.push_str(&format!(
            "# expected_count={}\n",
            self.expected_count.map(|count| count.to_string()).unwrap_or_else(|| "unknown".to_string())
        ));
        output.push_str(&format!("# attempt_floor_bytes={}\n", self.config.attempt_floor_bytes));
        output.push_str(&format!("# max_active_attempts_per_device={}\n", self.config.max_active_attempts_per_device));
        output.push_str(&format!("# max_queued_attempts={}\n", self.config.max_queued_attempts));
        output.push_str(&format!("# max_queue_wait_ms={}\n", self.config.max_queue_wait.as_millis()));
        output.push_str(&format!("# effective_active_max={}\n", self.effective_active_max));
        for summary in client_summaries(&self.rows) {
            output.push_str(&format!(
                "# client_summary client_id={} succeeded={} overloaded={} failed={} max_total_wall_ms={} p95_total_wall_ms={}\n",
                summary.client_id,
                summary.succeeded,
                summary.overloaded,
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
            self.iteration.to_string(),
            self.admission_sequence.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_wait_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            format_ms(self.total_wall_ms),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            self.planning_outcome.clone(),
            self.final_disposition.clone(),
            self.native_runtime_outcome.to_string(),
            self.count_status.to_string(),
            self.expected_count.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.native_count.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.native_output_rows.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
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
}

async fn planned_table_row_count(config: &GlueIcebergTableConfig) -> DataFusionResult<Option<u64>> {
    let loaded = load_glue_table(config.clone()).await.map_err(DataFusionError::from)?;
    let scan_plan = IcebergTableFormatPlanner::new(loaded.resolved)
        .resolve_table_format_scan(TableFormatScanRequest::default())
        .await
        .map_err(DataFusionError::from)?;
    Ok(scan_plan
        .data_files
        .iter()
        .map(|file| file.row_count)
        .try_fold(0u64, |total, count| count.map(|count| total.saturating_add(count))))
}

async fn run_stress(config: &StressConfig, expected_count: Option<u64>) -> DataFusionResult<StressRun> {
    let backend = GpuBackendOwner::build(
        config.service_config(),
        config.queue_bounds(),
        vec![GpuBackendOwner::device_config(
            0,
            config.attempt_floor_bytes.max(1),
            config.max_active_attempts_per_device,
            1,
        )],
    )
    .map_err(native_error)?;
    let service = backend.session();
    let requests = build_requests(config);
    let barrier = Arc::new(Barrier::new(requests.len().max(1)));
    let mut tasks = Vec::with_capacity(requests.len());
    for request in requests {
        let config = config.clone();
        let service = service.clone();
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            run_native_request(&config, service, request, expected_count).await
        }));
    }
    let mut rows = Vec::with_capacity(tasks.len());
    for task in tasks {
        rows.push(task.await.map_err(|error| plan_error(format!("query service Iceberg task join failed: {error}")))??);
    }
    rows.sort_by_key(|row| row.request_index);
    let effective_active_max = service.gpu_execution_query_report().map_err(native_error)?.peak_active_query_jobs;
    backend.close_and_wait(Duration::from_secs(30)).map_err(native_error)?;
    Ok(StressRun { rows, effective_active_max })
}

async fn run_native_request(
    config: &StressConfig,
    service: GpuBackendSession,
    request: StressRequest,
    expected_count: Option<u64>,
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
                expected_count,
                None,
                StructuredFailure::from_query_error(&error),
            ));
        },
    };
    let admission_sequence = Some(handle.admission_sequence());
    let admission_wait_ms = Some(duration_ms(handle.admission_wait()));
    let native_started = Instant::now();
    match collect_native_with_handle(config, Arc::clone(&handle)).await {
        Ok(NativeCollectOutcome::Succeeded { planning_row, measurement }) => {
            let count_status = match (expected_count, measurement.count_value) {
                (Some(expected), Some(actual)) if expected == actual => "matched",
                (Some(_), Some(_)) => "mismatched",
                (None, Some(_)) => "expected_unknown",
                _ => "not_count_query",
            };
            Ok(StressRow {
                row_type: "query",
                request_index: request.request_index,
                client_id: request.client_id.to_string(),
                iteration: request.iteration,
                admission_sequence,
                admission_wait_ms,
                total_wall_ms: duration_ms(total_started.elapsed()),
                native_wall_ms: Some(duration_ms(native_started.elapsed())),
                planning_outcome: planning_row
                    .outcome
                    .map_or("none", NativeDataFusionPlanningOutcome::as_str)
                    .to_string(),
                final_disposition: planning_row.final_disposition.as_str().to_string(),
                native_runtime_outcome: "succeeded",
                count_status,
                expected_count,
                native_count: measurement.count_value,
                native_output_rows: Some(measurement.output_rows),
                error: StructuredFailure::default(),
            })
        },
        Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error }) => Ok(failure_row(
            &request,
            total_started.elapsed(),
            admission_sequence,
            admission_wait_ms,
            planning_row.outcome.map_or("none", NativeDataFusionPlanningOutcome::as_str),
            planning_row.final_disposition.as_str(),
            "failed",
            "native_failed",
            expected_count,
            None,
            StructuredFailure::from_datafusion_error(&error),
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
            expected_count,
            None,
            StructuredFailure::from_datafusion_error(&error),
        )),
    }
}

async fn collect_native_with_handle(
    config: &StressConfig,
    handle: Arc<nexus_query_engine::runtime::QueryHandle>,
) -> DataFusionResult<NativeCollectOutcome> {
    let optimizer_config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded_default().with_query_service_handle(handle));
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let loaded = load_glue_table(config.glue.clone()).await.map_err(DataFusionError::from)?;
    let provider = NexusIcebergTableProvider::try_new_resolved(loaded.resolved)
        .map_err(DataFusionError::from)?
        .with_catalog_type("glue");
    ctx.register_table(config.table_alias.as_str(), Arc::new(provider))?;
    let df = ctx.sql(&config.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report =
        NativeDataFusionPlanningReport::from_retained_rule("iceberg_count", Arc::clone(&plan), &native_rule);
    let planning_row = select_planning_row(planning_report.rows)?;
    if planning_row.outcome != Some(NativeDataFusionPlanningOutcome::Selected) {
        return Err(plan_error(format!(
            "remote Iceberg query was not accepted by native planning; outcome={} reason_code={}",
            planning_row.outcome.map_or("none", NativeDataFusionPlanningOutcome::as_str),
            planning_row.reason_code.as_deref().unwrap_or("none"),
        )));
    }
    if planning_row.final_disposition != FinalPlanDisposition::Native {
        return Err(plan_error(format!(
            "remote Iceberg query did not produce final_disposition=native; observed {}",
            planning_row.final_disposition.as_str(),
        )));
    }
    let batches = match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => batches,
        Err(error) => return Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error }),
    };
    let measurement = measure_native_output(&batches)?;
    Ok(NativeCollectOutcome::Succeeded { planning_row, measurement })
}

fn run_overload_probe(config: &StressConfig) -> DataFusionResult<StressRow> {
    let probe_config = config.service_config();
    let probe_queue_bounds =
        QueueBounds { max_waiters: 1, max_wait: Duration::from_millis(50), max_overtakes_per_waiter: 4 };
    let backend = GpuBackendOwner::build(
        probe_config,
        probe_queue_bounds,
        vec![GpuBackendOwner::device_config(0, config.attempt_floor_bytes.max(1), 1, 1)],
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
        client_id: "interactive".to_string(),
        iteration: 0,
        admission_sequence: None,
        admission_wait_ms: None,
        total_wall_ms: duration_ms(started.elapsed()),
        native_wall_ms: None,
        planning_outcome: "not_planned".to_string(),
        final_disposition: "not_planned".to_string(),
        native_runtime_outcome: "not_started",
        count_status: "not_run",
        expected_count: None,
        native_count: None,
        native_output_rows: None,
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
    let mut request_index = 1;
    for iteration in 1..=config.iterations {
        for client_id in &config.client_ids {
            requests.push(StressRequest { request_index, client_id: client_id.clone(), iteration });
            request_index += 1;
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
    final_disposition: &str,
    native_runtime_outcome: &'static str,
    count_status: &'static str,
    expected_count: Option<u64>,
    native_count: Option<u64>,
    error: StructuredFailure,
) -> StressRow {
    StressRow {
        row_type: "query",
        request_index: request.request_index,
        client_id: request.client_id.to_string(),
        iteration: request.iteration,
        admission_sequence,
        admission_wait_ms,
        total_wall_ms: duration_ms(total_wall),
        native_wall_ms: None,
        planning_outcome: planning_outcome.to_string(),
        final_disposition: final_disposition.to_string(),
        native_runtime_outcome,
        count_status,
        expected_count,
        native_count,
        native_output_rows: None,
        error,
    }
}

fn measure_native_output(batches: &[RecordBatch]) -> DataFusionResult<NativeMeasurement> {
    let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
    let count_value = if batches.len() == 1 && batches[0].num_columns() == 1 && batches[0].num_rows() == 1 {
        scalar_u64(batches[0].column(0).as_ref(), 0)
    } else {
        None
    };
    Ok(NativeMeasurement { output_rows, count_value })
}

fn scalar_u64(array: &dyn Array, row: usize) -> Option<u64> {
    if array.is_null(row) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        return values.value(row).into();
    }
    if let Some(values) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
        return u64::try_from(values.value(row)).ok();
    }
    if let Some(values) = array.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        return Some(u64::from(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<arrow::array::Int32Array>() {
        return u64::try_from(values.value(row)).ok();
    }
    None
}

fn select_planning_row(rows: Vec<NativeDataFusionPlanningRow>) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        1 => Ok(rows.into_iter().next().expect("single planning row exists")),
        count => Err(plan_error(format!("expected one planning row for remote Iceberg query, observed {count}"))),
    }
}

#[derive(Debug, Clone)]
struct ClientSummary {
    client_id: String,
    succeeded: usize,
    overloaded: usize,
    failed: usize,
    max_total_wall_ms: f64,
    p95_total_wall_ms: f64,
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
            let failed = rows.len().saturating_sub(succeeded + overloaded);
            ClientSummary {
                client_id,
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

fn client_ids_from_env(name: &'static str, default: &[&'static str]) -> DataFusionResult<Vec<String>> {
    let values = match std::env::var_os(name) {
        Some(value) => value
            .to_string_lossy()
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>(),
        None => default.iter().map(|value| (*value).to_string()).collect(),
    };
    if values.is_empty() {
        return Err(plan_error(format!("{name} must contain at least one client id")));
    }
    let mut seen = BTreeSet::new();
    for value in &values {
        if !seen.insert(value.clone()) {
            return Err(plan_error(format!("{name} contains duplicate client id {value}")));
        }
    }
    Ok(values)
}

fn env_string(name: &'static str, default: &'static str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
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

fn default_attempt_floor_bytes() -> u64 {
    // Keep the standalone profile bounded without deriving a shared floor
    // from GPU 0. Server backends resolve their defaults per device.
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
            glue: GlueIcebergTableConfig::new("glue", "db", "table", "s3://warehouse", "us-east-1"),
            table_alias: "iceberg_t".to_string(),
            sql: "select count(*) from iceberg_t".to_string(),
            client_ids: vec!["a".to_string(), "b".to_string()],
            iterations: 2,
            attempt_floor_bytes: 1,
            max_active_attempts_per_device: 1,
            max_queued_attempts: 3,
            max_queue_wait: Duration::from_secs(1),
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            run_overload_probe: false,
        }
    }

    #[test]
    fn scalar_u64_reads_common_count_types() {
        let values = arrow::array::Int64Array::from(vec![42]);
        assert_eq!(scalar_u64(&values, 0), Some(42));
    }

    #[test]
    fn client_ids_reject_duplicates() {
        unsafe {
            std::env::set_var(CLIENT_IDS_ENV, "a,a");
        }
        let error = client_ids_from_env(CLIENT_IDS_ENV, &["x"]).unwrap_err();
        unsafe {
            std::env::remove_var(CLIENT_IDS_ENV);
        }
        assert!(error.to_string().contains("duplicate client id"));
    }

    #[test]
    fn stress_tsv_row_contract_matches_columns() {
        assert_eq!(TSV_COLUMNS.first(), Some(&"row_type"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"error_message"));
        let row = StressRow {
            row_type: "query",
            request_index: 1,
            client_id: "interactive".to_string(),
            iteration: 1,
            admission_sequence: Some(1),
            admission_wait_ms: Some(0.0),
            total_wall_ms: 10.0,
            native_wall_ms: Some(5.0),
            planning_outcome: "selected".to_string(),
            final_disposition: "native".to_string(),
            native_runtime_outcome: "succeeded",
            count_status: "matched",
            expected_count: Some(1),
            native_count: Some(1),
            native_output_rows: Some(1),
            error: StructuredFailure::default(),
        };
        assert_eq!(row.to_tsv_line().split('\t').count(), TSV_COLUMNS.len());
    }

    #[test]
    fn stress_config_rejects_queue_smaller_than_synchronized_wave() {
        let config = queue_test_config();

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains(MAX_QUEUED_ATTEMPTS_ENV));
        assert!(error.to_string().contains("synchronized request count (4)"));
    }
}
