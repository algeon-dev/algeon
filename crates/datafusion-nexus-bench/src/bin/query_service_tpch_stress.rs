use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
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
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error as AdapterError,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome,
    NativeDataFusionPlanningReport, NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath,
};
use datafusion_nexus_bench::{
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES, TPCH_TABLES,
    TpchQuery, default_tpch_parquet_root, describe_tpch_dataset, discover_tpch_table_layout,
    register_tpch_table_layout, tpch_report_session_config, tpch_tuned_cpu_session_config,
};
use nexus_query_engine::{
    QueryError, QueryService, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId,
    harness::StableFnv1a64,
};
use tokio::sync::Barrier;

const TSV_COLUMNS: &[&str] = &[
    "row_type",
    "request_index",
    "tenant_id",
    "workload_class",
    "query_id",
    "iteration",
    "admission_sequence",
    "admission_wait_ms",
    "device_ordinal",
    "query_grant_bytes",
    "service_capacity_bytes",
    "service_outstanding_bytes_snapshot",
    "service_available_bytes_snapshot",
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

const DATASET_ROOT_ENV: &str = CUDF_TPCH_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS";
const TENANT_BUDGET_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_TENANT_BUDGET_BYTES";
const DEVICE_CAPACITY_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_DEVICE_CAPACITY_BYTES";
const TENANT_MIN_GRANT_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_TENANT_MIN_GRANT_BYTES";
const TENANT_WANT_GRANT_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_TENANT_WANT_GRANT_BYTES";
const SLOT_GUARANTEE_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_SLOT_GUARANTEE";
const MAX_CONCURRENT_QUERIES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_CONCURRENT_QUERIES";
const DEVICE_ORDINALS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_DEVICE_ORDINALS";
const MAX_ADMISSION_QUEUE_LEN_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_ADMISSION_QUEUE_LEN";
const ADMISSION_WAIT_TIMEOUT_MS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_ADMISSION_WAIT_TIMEOUT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK";
const SOURCE_CACHE_MAX_RETAINED_BYTES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_SOURCE_CACHE_MAX_RETAINED_BYTES";
const WARMUP_ROUNDS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_WARMUP_ROUNDS";
const STREAM_IDLE_TIMEOUT_MS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_STREAM_IDLE_TIMEOUT_MS";
const CUDA_PROFILER_CAPTURE_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_CUDA_PROFILER_CAPTURE";
const PROFILE_LANES_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_PROFILE_LANES";
const PROFILE_QUERY_ID_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_PROFILE_QUERY_ID";
const PROFILE_WARMUP_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_PROFILE_WARMUP_ITERATIONS";
const PROFILE_MEASURED_ITERATIONS_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_PROFILE_MEASURED_ITERATIONS";
const PROFILE_EVIDENCE_PATH_ENV: &str = "CUDF_QUERY_SERVICE_TPCH_PROFILE_EVIDENCE_PATH";
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
    let service = Arc::new(build_query_service(&config)?);
    if let Some(profile) = &config.profile {
        let result = run_profile_stress(&config, profile, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?;
        rows.extend(result.rows);
        result.evidence.write_to(&profile.evidence_path)?;
    } else {
        run_warmup(&config, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?;
        let profiler_capture = CudaProfilerCapture::start(config.cuda_profiler_capture)?;
        rows.extend(run_stress(&config, Arc::clone(&service), Arc::clone(&cpu_baselines)).await?);
        wait_for_service_stream_pools_idle(Arc::clone(&service), config.stream_idle_timeout).await?;
        profiler_capture.finish()?;
    }
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

impl StressConfig {
    fn from_env() -> DataFusionResult<Self> {
        let dataset_root = match std::env::var_os(DATASET_ROOT_ENV) {
            Some(value) => PathBuf::from(value),
            None => default_tpch_parquet_root(datafusion_nexus_bench::TpchDefaultScale::Sf1).into(),
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
        let profile = StressProfileConfig::from_env()?;
        let config = Self {
            dataset_root,
            bulk_queries: query_ids_from_env(BULK_QUERIES_ENV, &["q01", "q09", "q18", "q21"])?,
            interactive_queries: query_ids_from_env(INTERACTIVE_QUERIES_ENV, &["q06", "q14", "q22"])?,
            bulk_iterations: env_usize(BULK_ITERATIONS_ENV, 1)?,
            interactive_iterations: env_usize(INTERACTIVE_ITERATIONS_ENV, 4)?,
            device_capacity_bytes,
            tenant_min_grant_bytes: tenant_min_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_min_grant_bytes),
            tenant_want_grant_bytes: tenant_want_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_want_grant_bytes),
            slot_guarantee: env_slot_guarantee(SLOT_GUARANTEE_ENV, SlotGuarantee::GuaranteedMinimum)?,
            max_concurrent_queries,
            max_admission_queue_len: env_usize(MAX_ADMISSION_QUEUE_LEN_ENV, 64)?,
            admission_wait_timeout: Duration::from_millis(env_u64(ADMISSION_WAIT_TIMEOUT_MS_ENV, 30_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
            source_cache_max_retained_bytes: optional_env_u64(SOURCE_CACHE_MAX_RETAINED_BYTES_ENV)?,
            warmup_rounds: env_usize(WARMUP_ROUNDS_ENV, 0)?,
            stream_idle_timeout: Duration::from_millis(env_u64(STREAM_IDLE_TIMEOUT_MS_ENV, 30_000)?),
            cuda_profiler_capture: env_bool(CUDA_PROFILER_CAPTURE_ENV, false)?,
            profile,
            run_overload_probe: env_bool(OVERLOAD_PROBE_ENV, true)?,
            device_ordinals: env_device_ordinals(DEVICE_ORDINALS_ENV)?,
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
        if let Some(max_retained_bytes) = self.source_cache_max_retained_bytes {
            validate_nonzero(SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, max_retained_bytes)?;
        }
        if self.stream_idle_timeout.is_zero() {
            return Err(plan_error(format!("{STREAM_IDLE_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        if self.admission_wait_timeout.is_zero() {
            return Err(plan_error(format!("{ADMISSION_WAIT_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        if let Some(profile) = &self.profile {
            validate_nonzero(PROFILE_LANES_ENV, profile.lanes)?;
            validate_nonzero(PROFILE_WARMUP_ITERATIONS_ENV, profile.warmup_iterations)?;
            validate_nonzero(PROFILE_MEASURED_ITERATIONS_ENV, profile.measured_iterations)?;
            if profile.warmup_iterations < 2 {
                return Err(plan_error(format!(
                    "{PROFILE_WARMUP_ITERATIONS_ENV} must be at least 2 so the final warmup iteration can verify a cache-only path"
                )));
            }
            if profile.lanes > self.max_concurrent_queries {
                return Err(plan_error(format!("{PROFILE_LANES_ENV} must not exceed {MAX_CONCURRENT_QUERIES_ENV}")));
            }
            if self.source_cache_max_retained_bytes.is_none() {
                return Err(plan_error(format!(
                    "{SOURCE_CACHE_MAX_RETAINED_BYTES_ENV} must be set when {PROFILE_LANES_ENV} is enabled"
                )));
            }
            if self.warmup_rounds != 0 {
                return Err(plan_error(format!(
                    "{WARMUP_ROUNDS_ENV} cannot be combined with {PROFILE_LANES_ENV}; profile warmup is controlled by {PROFILE_WARMUP_ITERATIONS_ENV}"
                )));
            }
            validate_query_ids(std::slice::from_ref(&profile.query_id))?;
        }
        Ok(())
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
            local_parquet_source_cache_max_retained_bytes: self.source_cache_max_retained_bytes,
            ..QueryServiceConfig::default()
        }
    }
}

impl StressProfileConfig {
    fn from_env() -> DataFusionResult<Option<Self>> {
        let lanes = env_usize(PROFILE_LANES_ENV, 0)?;
        if lanes == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            query_id: std::env::var(PROFILE_QUERY_ID_ENV).unwrap_or_else(|_| "q06".to_string()),
            lanes,
            warmup_iterations: env_usize(PROFILE_WARMUP_ITERATIONS_ENV, 2)?,
            measured_iterations: env_usize(PROFILE_MEASURED_ITERATIONS_ENV, 8)?,
            evidence_path: std::env::var_os(PROFILE_EVIDENCE_PATH_ENV)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("query_service_tpch_profile_evidence.tsv")),
        }))
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
    handle: Arc<nexus_query_engine::QueryHandle>,
}

struct PreparedProfileRequest {
    phase: &'static str,
    lane: usize,
    request: StressRequest,
    handle: Arc<nexus_query_engine::QueryHandle>,
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
    dataset: datafusion_nexus_bench::TpchDatasetMetadata,
    rows: Vec<StressRow>,
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
        output.push_str(&format!("# max_admission_queue_len={}\n", self.config.max_admission_queue_len));
        output.push_str(&format!("# max_source_chunk_bytes={}\n", self.config.max_source_chunk_bytes));
        output.push_str(&format!("# max_row_groups_per_chunk={}\n", self.config.max_row_groups_per_chunk));
        output.push_str(&format!(
            "# source_cache_max_retained_bytes={}\n",
            self.config.source_cache_max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string())
        ));
        output.push_str(&format!("# warmup_rounds={}\n", self.config.warmup_rounds));
        output.push_str(&format!("# stream_idle_timeout_ms={}\n", self.config.stream_idle_timeout.as_millis()));
        output.push_str(&format!("# cuda_profiler_capture={}\n", self.config.cuda_profiler_capture));
        if let Some(profile) = &self.config.profile {
            output.push_str(&format!("# profile_query_id={}\n", profile.query_id));
            output.push_str(&format!("# profile_lanes={}\n", profile.lanes));
            output.push_str(&format!("# profile_warmup_iterations={}\n", profile.warmup_iterations));
            output.push_str(&format!("# profile_measured_iterations={}\n", profile.measured_iterations));
            output.push_str(&format!("# profile_evidence_path={}\n", profile.evidence_path.display()));
        }
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
        output.push_str(&TSV_COLUMNS.join("\t"));
        output.push('\n');
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl ProfileEvidenceReport {
    const COLUMNS: [&'static str; 15] = [
        "phase",
        "lane",
        "iteration",
        "request_index",
        "admission_sequence",
        "device_ordinal",
        "query_id",
        "native_runtime_outcome",
        "parity_status",
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "native_wall_ms",
    ];

    fn write_to(&self, path: &Path) -> DataFusionResult<()> {
        fs::write(path, self.to_tsv()).map_err(|error| {
            plan_error(format!("failed to write profile evidence sidecar {}: {error}", path.display()))
        })
    }

    fn to_tsv(&self) -> String {
        let mut output = Self::COLUMNS.join("\t");
        output.push('\n');
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl ProfileEvidenceRow {
    fn to_tsv_line(&self) -> String {
        let fields = [
            self.phase.to_string(),
            self.lane.to_string(),
            self.iteration.to_string(),
            self.request_index.to_string(),
            self.admission_sequence.to_string(),
            self.device_ordinal.to_string(),
            self.query_id.clone(),
            self.native_runtime_outcome.to_string(),
            self.parity_status.to_string(),
            self.source_cache_hits.to_string(),
            self.source_cache_misses.to_string(),
            self.source_cache_inserts.to_string(),
            self.source_cache_insert_skipped.to_string(),
            self.source_cache_retained_bytes.to_string(),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), ProfileEvidenceReport::COLUMNS.len());
        fields.into_iter().map(|field| tsv_escape(&field)).collect::<Vec<_>>().join("\t")
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
            self.admission_facts.device_ordinal.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_facts.query_grant_bytes.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_capacity_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_outstanding_bytes_snapshot
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_available_bytes_snapshot
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
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
    failed: usize,
    max_total_wall_ms: f64,
    p95_total_wall_ms: f64,
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

async fn run_profile_stress(
    config: &StressConfig,
    profile: &StressProfileConfig,
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<ProfileStressResult> {
    let lanes = admit_profile_lanes(Arc::clone(&service), profile.lanes).await?;
    let expected_leases = profile_expected_root_leases(&lanes);
    let mut evidence = ProfileEvidenceReport::default();

    for iteration in 1..=profile.warmup_iterations {
        let batch = prepare_profile_iteration(config, profile, &lanes, "warmup", iteration).await?;
        let batches = batch.into_iter().map(|request| vec![request]).collect();
        let results = execute_profile_batches(Arc::clone(&service), Arc::clone(&cpu_baselines), batches).await?;
        wait_for_service_stream_pools_quiescent(
            Arc::clone(&service),
            expected_leases.clone(),
            config.stream_idle_timeout,
        )
        .await?;
        log_profile_cache_evidence(&results);
        validate_profile_results(&results, iteration >= 2)?;
        evidence.rows.extend(results.into_iter().map(|result| result.evidence));
        eprintln!(
            "[query_service_tpch_stress] profile warmup iteration {iteration}/{} complete",
            profile.warmup_iterations
        );
    }

    let measured_batches = prepare_profile_measurements(config, profile, &lanes).await?;
    wait_for_service_stream_pools_quiescent(Arc::clone(&service), expected_leases.clone(), config.stream_idle_timeout)
        .await?;

    let profiler_capture = CudaProfilerCapture::start(config.cuda_profiler_capture)?;
    let execution_result =
        execute_profile_batches(Arc::clone(&service), Arc::clone(&cpu_baselines), measured_batches).await;
    let quiescence_result =
        wait_for_service_stream_pools_quiescent(Arc::clone(&service), expected_leases, config.stream_idle_timeout)
            .await;
    let profiler_result = profiler_capture.finish();
    let results = execution_result?;
    quiescence_result?;
    profiler_result?;

    log_profile_cache_evidence(&results);
    validate_profile_results(&results, true)?;
    let rows = results.iter().map(|result| result.row.clone()).collect::<Vec<_>>();
    evidence.rows.extend(results.into_iter().map(|result| result.evidence));

    drop(lanes);
    wait_for_service_stream_pools_idle(service, config.stream_idle_timeout).await?;
    Ok(ProfileStressResult { rows, evidence })
}

async fn admit_profile_lanes(service: Arc<QueryService>, lanes: usize) -> DataFusionResult<Vec<ProfileLane>> {
    let mut admitted = Vec::with_capacity(lanes);
    for lane in 1..=lanes {
        let service_for_admission = Arc::clone(&service);
        let handle = tokio::task::spawn_blocking(move || service_for_admission.admit_query("bulk"))
            .await
            .map_err(|error| plan_error(format!("profile lane admission task join failed: {error}")))?
            .map_err(native_error)?;
        admitted.push(ProfileLane { lane, handle: Arc::new(handle) });
    }
    Ok(admitted)
}

fn profile_expected_root_leases(lanes: &[ProfileLane]) -> Vec<(usize, usize)> {
    let mut by_device = BTreeMap::new();
    for lane in lanes {
        *by_device.entry(lane.handle.device_ordinal()).or_insert(0) += 1;
    }
    by_device.into_iter().collect()
}

async fn wait_for_service_stream_pools_quiescent(
    service: Arc<QueryService>,
    expected_leases: Vec<(usize, usize)>,
    timeout: Duration,
) -> DataFusionResult<()> {
    tokio::task::spawn_blocking(move || service.wait_for_stream_pools_quiescent(&expected_leases, timeout))
        .await
        .map_err(|error| plan_error(format!("query service stream-quiescence wait task join failed: {error}")))?
        .map_err(native_error)
}

async fn prepare_profile_iteration(
    config: &StressConfig,
    profile: &StressProfileConfig,
    lanes: &[ProfileLane],
    phase: &'static str,
    iteration: usize,
) -> DataFusionResult<Vec<PreparedProfileRequest>> {
    let query = tpch_query(&profile.query_id)?;
    let mut prepared = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let request = profile_request(profile, lane.lane, iteration);
        let plan = prepare_native_with_handle(config, query, Arc::clone(&lane.handle)).await?;
        prepared.push(PreparedProfileRequest {
            phase,
            lane: lane.lane,
            request,
            handle: Arc::clone(&lane.handle),
            prepared: plan,
        });
    }
    Ok(prepared)
}

async fn prepare_profile_measurements(
    config: &StressConfig,
    profile: &StressProfileConfig,
    lanes: &[ProfileLane],
) -> DataFusionResult<Vec<Vec<PreparedProfileRequest>>> {
    let query = tpch_query(&profile.query_id)?;
    let mut batches = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let mut requests = Vec::with_capacity(profile.measured_iterations);
        for iteration in 1..=profile.measured_iterations {
            let request = profile_request(profile, lane.lane, iteration);
            let plan = prepare_native_with_handle(config, query, Arc::clone(&lane.handle)).await?;
            requests.push(PreparedProfileRequest {
                phase: "measured",
                lane: lane.lane,
                request,
                handle: Arc::clone(&lane.handle),
                prepared: plan,
            });
        }
        batches.push(requests);
    }
    Ok(batches)
}

fn profile_request(profile: &StressProfileConfig, lane: usize, iteration: usize) -> StressRequest {
    StressRequest {
        request_index: (iteration - 1) * profile.lanes + lane,
        tenant_id: TenantId::new("bulk"),
        workload_class: "profile",
        query_id: profile.query_id.clone(),
        iteration,
    }
}

async fn execute_profile_batches(
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
    batches: Vec<Vec<PreparedProfileRequest>>,
) -> DataFusionResult<Vec<ProfileRequestResult>> {
    let barrier = Arc::new(Barrier::new(batches.len().max(1)));
    let mut tasks = Vec::with_capacity(batches.len());
    for batch in batches {
        let service = Arc::clone(&service);
        let cpu_baselines = Arc::clone(&cpu_baselines);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let mut results = Vec::with_capacity(batch.len());
            for request in batch {
                results.push(execute_prepared_profile_request(&service, &cpu_baselines, request).await?);
            }
            Ok::<Vec<ProfileRequestResult>, DataFusionError>(results)
        }));
    }

    let mut results = Vec::new();
    for task in tasks {
        results.extend(task.await.map_err(|error| plan_error(format!("profile lane task join failed: {error}")))??);
    }
    results.sort_by_key(|result| (result.row.iteration, result.evidence.lane));
    Ok(results)
}

async fn execute_prepared_profile_request(
    service: &QueryService,
    cpu_baselines: &HashMap<String, CpuBaseline>,
    prepared: PreparedProfileRequest,
) -> DataFusionResult<ProfileRequestResult> {
    let PreparedProfileRequest { phase, lane, request, handle, prepared } = prepared;
    let total_started = Instant::now();
    let native_started = Instant::now();
    let admission_sequence = handle.admission_sequence();
    let device_ordinal = handle.device_ordinal();
    let admission_wait_ms = Some(duration_ms(handle.admission_wait()));
    let admission_facts = admission_facts(service, &handle)?;
    let cpu = cpu_baselines.get(&request.query_id);

    let (row, metrics) = match collect_prepared_native(prepared).await {
        Ok(NativeCollectOutcome::Succeeded { planning_row, measurement: native, metrics }) => {
            let parity = match cpu {
                Some(cpu) if cpu.output_fingerprint == native.output_fingerprint => "matched",
                Some(_) => "mismatched",
                None => "cpu_missing",
            };
            let row = StressRow {
                row_type: "query",
                request_index: request.request_index,
                tenant_id: request.tenant_id.to_string(),
                workload_class: request.workload_class,
                query_id: request.query_id.clone(),
                iteration: request.iteration,
                admission_sequence: Some(admission_sequence),
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
            };
            (row, metrics)
        },
        Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error, metrics }) => {
            let row = failure_row(
                &request,
                total_started.elapsed(),
                Some(admission_sequence),
                admission_wait_ms,
                admission_facts,
                planning_row.outcome.as_str(),
                planning_row.post_native_execution_path.as_str(),
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            );
            (row, metrics)
        },
        Err(error) => {
            let row = failure_row(
                &request,
                total_started.elapsed(),
                Some(admission_sequence),
                admission_wait_ms,
                admission_facts,
                "unknown",
                "unknown",
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            );
            (row, NativeDataFusionExecutionMetrics::default())
        },
    };

    let evidence = ProfileEvidenceRow {
        phase,
        lane,
        iteration: row.iteration,
        request_index: row.request_index,
        admission_sequence,
        device_ordinal,
        query_id: row.query_id.clone(),
        native_runtime_outcome: row.native_runtime_outcome,
        parity_status: row.parity_status,
        source_cache_hits: metrics.source_cache_hits,
        source_cache_misses: metrics.source_cache_misses,
        source_cache_inserts: metrics.source_cache_inserts,
        source_cache_insert_skipped: metrics.source_cache_insert_skipped,
        source_cache_retained_bytes: metrics.source_cache_retained_bytes,
        native_wall_ms: row.native_wall_ms,
    };
    Ok(ProfileRequestResult { row, evidence })
}

fn validate_profile_results(results: &[ProfileRequestResult], require_cache_only: bool) -> DataFusionResult<()> {
    for result in results {
        let row = &result.row;
        let evidence = &result.evidence;
        if row.native_runtime_outcome != "succeeded" || row.parity_status != "matched" {
            return Err(plan_error(format!(
                "profile {} lane {} iteration {} failed: runtime_outcome={} parity_status={} error={}",
                evidence.phase,
                evidence.lane,
                evidence.iteration,
                row.native_runtime_outcome,
                row.parity_status,
                row.error.message.as_deref().unwrap_or("none"),
            )));
        }
        if require_cache_only
            && (evidence.source_cache_hits == 0
                || evidence.source_cache_misses != 0
                || evidence.source_cache_inserts != 0)
        {
            return Err(plan_error(format!(
                "profile {} lane {} iteration {} was not cache-only: hits={} misses={} inserts={} skipped={}",
                evidence.phase,
                evidence.lane,
                evidence.iteration,
                evidence.source_cache_hits,
                evidence.source_cache_misses,
                evidence.source_cache_inserts,
                evidence.source_cache_insert_skipped,
            )));
        }
    }
    Ok(())
}

fn log_profile_cache_evidence(results: &[ProfileRequestResult]) {
    for result in results {
        let evidence = &result.evidence;
        eprintln!(
            "[query_service_tpch_stress] profile phase={} lane={} iteration={} device={} hits={} misses={} inserts={} skipped={} retained_bytes={} native_wall_ms={}",
            evidence.phase,
            evidence.lane,
            evidence.iteration,
            evidence.device_ordinal,
            evidence.source_cache_hits,
            evidence.source_cache_misses,
            evidence.source_cache_inserts,
            evidence.source_cache_insert_skipped,
            evidence.source_cache_retained_bytes,
            evidence.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
        );
    }
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
    handle: Arc<nexus_query_engine::QueryHandle>,
) -> DataFusionResult<NativeCollectOutcome> {
    let prepared = prepare_native_with_handle(config, query, handle).await?;
    collect_prepared_native(prepared).await
}

async fn prepare_native_with_handle(
    config: &StressConfig,
    query: &TpchQuery,
    handle: Arc<nexus_query_engine::QueryHandle>,
) -> DataFusionResult<PreparedNativeQuery> {
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
    handle: &nexus_query_engine::QueryHandle,
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

fn measure_output(schema: &ArrowSchemaRef, batches: &[RecordBatch]) -> DataFusionResult<CpuBaseline> {
    Ok(CpuBaseline {
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_fingerprint: fingerprint_batches(schema, batches)?,
    })
}

fn fingerprint_batches(schema: &ArrowSchemaRef, batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-query-service-tpch-output-v1");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(plan_error("TPC-H output fingerprint requires collected batches to match the planned schema"));
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
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(plan_error(format!("unknown TPC-H query id in stress config: {query_id}")));
        }
    }
    Ok(())
}

fn env_device_ordinals(name: &'static str) -> DataFusionResult<Vec<usize>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(vec![0]);
    };
    let mut ordinals = Vec::new();
    for part in value.to_string_lossy().split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let ordinal = part
            .parse::<usize>()
            .map_err(|error| plan_error(format!("failed to parse {name} as device ordinals: {error}")))?;
        if ordinals.contains(&ordinal) {
            return Err(plan_error(format!("{name} must not contain duplicate ordinals")));
        }
        ordinals.push(ordinal);
    }
    if ordinals.is_empty() {
        return Err(plan_error(format!("{name} must contain at least one device ordinal")));
    }
    Ok(ordinals)
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
    fn stress_tsv_row_contract_matches_columns() {
        assert_eq!(TSV_COLUMNS.first(), Some(&"row_type"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"error_message"));
        let row = StressRow {
            row_type: "query",
            request_index: 1,
            tenant_id: "interactive".to_string(),
            workload_class: "interactive",
            query_id: "q06".to_string(),
            iteration: 1,
            admission_sequence: Some(1),
            admission_wait_ms: Some(0.0),
            admission_facts: AdmissionFacts::default(),
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
    fn stress_config_rejects_overcommitted_guaranteed_minimum_band() {
        let config = StressConfig {
            dataset_root: PathBuf::new(),
            bulk_queries: vec!["q01".to_string()],
            interactive_queries: vec!["q06".to_string()],
            bulk_iterations: 1,
            interactive_iterations: 1,
            device_capacity_bytes: 100,
            tenant_min_grant_bytes: 60,
            tenant_want_grant_bytes: 60,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_concurrent_queries: 2,
            max_admission_queue_len: 1,
            admission_wait_timeout: Duration::from_secs(1),
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            source_cache_max_retained_bytes: None,
            warmup_rounds: 0,
            stream_idle_timeout: Duration::from_secs(1),
            cuda_profiler_capture: false,
            profile: None,
            run_overload_probe: false,
            device_ordinals: vec![0],
        };

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains(SLOT_GUARANTEE_ENV));
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
                admission_facts: AdmissionFacts::default(),
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
                admission_facts: AdmissionFacts::default(),
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
