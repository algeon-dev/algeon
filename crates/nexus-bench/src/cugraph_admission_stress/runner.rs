//! cuGraph-heavy admission and memory-governance stress harness.
//!
//! The library-mode stress binary builds one process-scoped
//! `NexusGpuBackend`, installs cuGraph SQL on per-worker `SessionContext`s,
//! fires a configurable concurrent workload, and reports
//!
//! - one TSV row per executed query (admission facts, reservation facts,
//!   graph memory-model facts, wall times, structured failure), and
//! - a periodic diagnostics TSV sampled from the admission snapshot, the
//!   memory-governor report, the device failure report, and the physical
//!   CUDA async-pool counters (`DeviceMemoryManager::pool_usage`).
//!
//! Everything is env-var driven like the sibling `query_service_*_stress`
//! binaries. GPU runs must be wrapped in `flock /tmp/cudf-gpu.lock`.

use std::{
    error::Error,
    fs,
    io::Write as _,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{array::Array, datatypes::SchemaRef, record_batch::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect, displayable, metrics::MetricsSet},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    backend::{
        GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceConfig, NexusGpuDeviceProfile,
        NexusGpuQueuePolicy,
    },
    cugraph_sql::CugraphSqlConfig,
    native_exec::NexusNativeExecutionConfig,
    native_report::NativeDataFusionExecutionMetrics,
    planner::NexusNativePhysicalOptimizerRule,
    report::CandidateOutcome,
    session::NexusSessionStateBuilderExt,
};
use nexus_query_engine::harness::StableFnv1a64;
use tokio::sync::Barrier;

mod config;
mod report;
mod sampling;
mod workload;

#[cfg(test)]
use config::parse_workload;
use config::{ENV_FOS, ENV_PAPERS, ENV_PRINT_PLANNING};
#[cfg(test)]
use report::{DEVICE_SAMPLE_COLUMNS, QUERY_COLUMNS, SAMPLE_COLUMNS, device_peaks};
use report::{query_rows_makespan_ms, write_outputs};
#[cfg(test)]
use sampling::device_sample_rows;
use sampling::spawn_sampler;
use workload::{catalog, expand_workload};

/// Retained optimizer rules (diagnostic planning-report mode only).
static PLANNING_RULES: Mutex<Vec<Arc<NexusNativePhysicalOptimizerRule>>> = Mutex::new(Vec::new());

#[derive(Debug, Clone)]
struct HarnessConfig {
    edges_parquet: String,
    papers_parquet: Option<String>,
    fos_parquet: Option<String>,
    workload: Vec<(String, usize)>,
    iterations: usize,
    stagger_ms: u64,
    max_active_attempts_per_device: usize,
    default_query_limit_bytes: Option<u64>,
    max_queue_wait_ms: u64,
    budget_fraction: Option<f64>,
    device_budget_bytes: Option<u64>,
    sample_interval_ms: u64,
    query_timeout_ms: u64,
    out_dir: PathBuf,
    run_id: String,
    bfs_seed: Option<i64>,
    batch_size: Option<usize>,
    device_ordinals: Vec<usize>,
    warmup_iterations: usize,
    cuda_profiler_capture: bool,
}

#[derive(Debug, Clone)]
struct QuerySpec {
    label: String,
    sql: String,
    needs_papers: bool,
    needs_fos: bool,
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let config = HarnessConfig::from_env()?;
    fs::create_dir_all(&config.out_dir)?;

    let backend = build_backend(&config)?;

    // Seed discovery for traversal algorithms: cheap via parquet stats.
    let seed = match config.bfs_seed {
        Some(seed) => seed,
        None => discover_seed(&backend, &config).await?,
    };
    let catalog_specs = catalog(seed);

    let requests = expand_workload(&config, &catalog_specs)?;
    eprintln!(
        "# cugraph_admission_stress run_id={} requests={} workload={:?}",
        config.run_id,
        requests.len(),
        config.workload
    );

    // Diagnostics sampler on a dedicated OS thread.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let samples = Arc::new(Mutex::new(SampleCollection::default()));
    let sampler_handle = spawn_sampler(&backend, &config, Arc::clone(&stop_flag), Arc::clone(&samples));

    // Pre-build one session per worker task so registration cost is off the
    // measured path, then fire everything through one barrier.
    let mut workers = Vec::with_capacity(requests.len());
    for request in &requests {
        let ctx = build_session(&backend, &config, request).await?;
        workers.push((request.clone(), ctx));
    }

    let barrier = Arc::new(Barrier::new(workers.len()));
    let profiler_active = Arc::new(AtomicBool::new(false));
    let profiler_error = Arc::new(Mutex::new(None::<String>));
    let run_started = Instant::now();
    let mut tasks = Vec::with_capacity(workers.len());
    for (request, ctx) in workers {
        let barrier = Arc::clone(&barrier);
        let profiler_active = Arc::clone(&profiler_active);
        let profiler_error = Arc::clone(&profiler_error);
        let config = config.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            if config.stagger_ms > 0 && request.task_index > 0 {
                tokio::time::sleep(Duration::from_millis(config.stagger_ms * request.task_index as u64)).await;
            }

            if !config.uses_phased_execution() {
                let mut rows = Vec::with_capacity(config.iterations);
                for iteration in 0..config.iterations {
                    rows.push(run_one(&config, &ctx, &request, iteration).await);
                }
                return WorkerRows { warmup: Vec::new(), measured: rows };
            }

            let mut warmup = Vec::with_capacity(config.warmup_iterations);
            for iteration in 0..config.warmup_iterations {
                if iteration > 0 {
                    barrier.wait().await;
                }
                warmup.push(run_one(&config, &ctx, &request, iteration).await);
            }

            if barrier.wait().await.is_leader() && config.cuda_profiler_capture {
                match cudf::runtime::cuda_profiler_start() {
                    Ok(()) => profiler_active.store(true, Ordering::SeqCst),
                    Err(error) => {
                        *profiler_error.lock().unwrap() =
                            Some(format!("failed to start CUDA profiler capture: {error}"));
                    },
                }
            }
            barrier.wait().await;

            let mut measured = Vec::with_capacity(config.iterations);
            for iteration in 0..config.iterations {
                if iteration > 0 {
                    barrier.wait().await;
                }
                measured.push(run_one(&config, &ctx, &request, iteration).await);
            }

            if barrier.wait().await.is_leader()
                && profiler_active.swap(false, Ordering::SeqCst)
                && let Err(error) = cudf::runtime::cuda_profiler_stop()
            {
                *profiler_error.lock().unwrap() = Some(format!("failed to stop CUDA profiler capture: {error}"));
            }
            barrier.wait().await;
            WorkerRows { warmup, measured }
        }));
    }

    let mut query_rows = Vec::new();
    let mut warmup_rows = Vec::new();
    for task in tasks {
        match task.await {
            Ok(rows) => {
                warmup_rows.extend(rows.warmup);
                query_rows.extend(rows.measured);
            },
            Err(join_error) => eprintln!("worker task panicked: {join_error}"),
        }
    }
    let makespan_ms = if config.uses_phased_execution() {
        query_rows_makespan_ms(&query_rows).unwrap_or_else(|| duration_ms(run_started.elapsed()))
    } else {
        duration_ms(run_started.elapsed())
    };

    stop_flag.store(true, Ordering::SeqCst);
    let _ = sampler_handle.join();

    let effective_active_max = backend.gpu_execution_query_report().unwrap().peak_active_query_jobs;
    let bfs_singleflight = backend.cugraph_bfs_singleflight_snapshot();
    let samples = samples.lock().unwrap();
    write_outputs(
        &config,
        &warmup_rows,
        &query_rows,
        &samples.aggregate,
        &samples.by_device,
        makespan_ms,
        effective_active_max,
        bfs_singleflight,
    )?;
    if let Some(error) = profiler_error.lock().unwrap().take() {
        return Err(error.into());
    }
    eprintln!("# done run_id={} makespan_ms={makespan_ms:.1} rows={}", config.run_id, query_rows.len());
    Ok(())
}

struct WorkerRows {
    warmup: Vec<QueryRow>,
    measured: Vec<QueryRow>,
}

fn build_backend(config: &HarnessConfig) -> Result<NexusGpuBackend, Box<dyn Error>> {
    let execution_config = NexusNativeExecutionConfig::bounded(
        config.device_budget_bytes.unwrap_or(NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES),
        256 * 1024 * 1024,
        1024,
        3,
        1,
    );
    let memory_limit = match config.device_budget_bytes {
        Some(limit_bytes) => GpuDeviceMemoryLimit::Explicit { limit_bytes },
        None => GpuDeviceMemoryLimit::Automatic {
            headroom_bytes: NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
            fallback_bytes: NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            fraction: config.budget_fraction.map(nexus_query_engine::runtime::GpuMemoryFraction::new).transpose()?,
        },
    };
    let admission = NexusGpuQueuePolicy::default().with_max_queue_wait(Duration::from_millis(config.max_queue_wait_ms));
    let devices = config.device_ordinals.iter().copied().map(|ordinal| {
        let mut profile = NexusGpuDeviceProfile::new(ordinal)
            .with_memory_limit(memory_limit)
            .with_max_active_attempts(config.max_active_attempts_per_device);
        if let Some(default_query_limit_bytes) = config.default_query_limit_bytes {
            profile = profile.with_default_query_limit_bytes(default_query_limit_bytes);
        }
        NexusGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)
    });
    let builder = NexusGpuBackend::builder()
        .devices(devices)
        .with_native_execution_config(execution_config)
        .queue_policy(admission);
    Ok(builder.build()?)
}

async fn build_session(
    backend: &NexusGpuBackend,
    config: &HarnessConfig,
    request: &Request,
) -> Result<SessionContext, Box<dyn Error>> {
    let mut session_config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    if let Some(batch_size) = config.batch_size {
        session_config = session_config.with_batch_size(batch_size);
    }
    let builder = SessionStateBuilder::new().with_config(session_config).with_default_features();
    let builder = backend.install_on(builder)?.try_with_cugraph_sql(CugraphSqlConfig::default())?;
    // Retain the rule only when its PlanningReports must outlive optimizer execution.
    let builder = if std::env::var(ENV_PRINT_PLANNING).is_ok() {
        let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(backend.optimizer_config().clone()));
        PLANNING_RULES.lock().unwrap().push(Arc::clone(&rule));
        builder.with_nexus_native_rule(rule)
    } else {
        builder
    };
    let ctx = SessionContext::new_with_state(builder.build());

    ctx.register_parquet("citation_edges", &config.edges_parquet, ParquetReadOptions::default()).await?;
    if request.spec.needs_papers {
        let path =
            config.papers_parquet.as_deref().ok_or_else(|| format!("{} requires {ENV_PAPERS}", request.spec.label))?;
        ctx.register_parquet("papers", path, ParquetReadOptions::default()).await?;
    }
    if request.spec.needs_fos {
        let path = config.fos_parquet.as_deref().ok_or_else(|| format!("{} requires {ENV_FOS}", request.spec.label))?;
        ctx.register_parquet("paper_fos", path, ParquetReadOptions::default()).await?;
    }
    Ok(ctx)
}

async fn discover_seed(backend: &NexusGpuBackend, config: &HarnessConfig) -> Result<i64, Box<dyn Error>> {
    let request = Request {
        task_index: 0,
        spec: QuerySpec { label: "seed".into(), sql: String::new(), needs_papers: false, needs_fos: false },
    };
    let ctx = build_session(backend, config, &request).await?;
    let batches = ctx.sql("SELECT MIN(src) AS s FROM citation_edges").await?.collect().await?;
    let scalar = batches
        .first()
        .and_then(|batch| batch.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().map(|a| a.value(0)))
        .ok_or("seed discovery failed")?;
    Ok(scalar)
}

#[derive(Debug, Clone)]
struct Request {
    task_index: usize,
    spec: QuerySpec,
}

async fn run_one(config: &HarnessConfig, ctx: &SessionContext, request: &Request, iteration: usize) -> QueryRow {
    let start_epoch_ms = epoch_ms();
    let started = Instant::now();
    let mut row = QueryRow::new(config, request, iteration, start_epoch_ms);

    let plan: Arc<dyn ExecutionPlan> = match ctx.sql(&request.spec.sql).await {
        Ok(df) => match df.create_physical_plan().await {
            Ok(plan) => plan,
            Err(error) => {
                return row.finish_failed(started, format!("physical_planning_failed: {error}"));
            },
        },
        Err(error) => {
            return row.finish_failed(started, format!("sql_planning_failed: {error}"));
        },
    };

    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    row.native_lowered = plan_display.contains("CugraphAlgoExec") || plan_display.contains("NexusNative");
    if std::env::var(ENV_PRINT_PLANNING).is_ok() {
        for rule in PLANNING_RULES.lock().unwrap().iter() {
            if let Some(report) = rule.last_report() {
                for outcome in report.outcomes() {
                    match outcome {
                        CandidateOutcome::Selected { candidate } => {
                            eprintln!("[planning] {} SELECTED shape={}", request.spec.label, candidate.shape);
                        },
                        CandidateOutcome::Deferred { candidate, planning_stage } => {
                            eprintln!(
                                "[planning] {} DEFERRED shape={} stage={} reason={}",
                                request.spec.label,
                                candidate.shape,
                                planning_stage,
                                datafusion_nexus::report::DEFERRED_PRODUCER_BINDING_REASON_CODE
                            );
                        },
                        CandidateOutcome::NotSupported { candidate, capability_reason } => {
                            eprintln!(
                                "[planning] {} NOT_SUPPORTED shape={} reason={}",
                                request.spec.label,
                                candidate.shape,
                                capability_reason.code()
                            );
                        },
                        CandidateOutcome::NotSelectedByCost { candidate, cost_reason, .. } => {
                            eprintln!(
                                "[planning] {} NOT_SELECTED_BY_COST shape={} reason={}",
                                request.spec.label,
                                candidate.shape,
                                cost_reason.code()
                            );
                        },
                    }
                }
            }
        }
    }

    let task_ctx = Arc::new(TaskContext::from(&ctx.state()));
    let outcome =
        tokio::time::timeout(Duration::from_millis(config.query_timeout_ms), collect(Arc::clone(&plan), task_ctx))
            .await;
    match outcome {
        Ok(Ok(batches)) => {
            row.output_rows = Some(batches.iter().map(|batch| batch.num_rows()).sum());
            row.absorb_metrics(&NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref())));
            match fingerprint_batches(&plan.schema(), &batches) {
                Ok(fingerprint) => {
                    row.output_fingerprint = Some(fingerprint);
                    row.finish_ok(started)
                },
                Err(error) => row.finish_failed(started, format!("output_fingerprint_failed: {error}")),
            }
        },
        Ok(Err(error)) => {
            row.absorb_metrics(&NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref())));
            row.finish_failed(started, error.to_string())
        },
        Err(_) => row.finish_failed(started, format!("query_timeout after {} ms", config.query_timeout_ms)),
    }
}

fn fingerprint_batches(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<String, arrow::error::ArrowError> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-cugraph-admission-output-multiset-v2");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(arrow::error::ArrowError::SchemaError(
                "cuGraph stress output fingerprint requires collected batches to match the planned schema".to_string(),
            ));
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
    let mut row_fingerprints = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut row_hasher = StableFnv1a64::new();
            row_hasher.update_field("datafusion-nexus-cugraph-admission-row-v1");
            for column in 0..batch.num_columns() {
                let array = batch.column(column);
                if array.is_null(row) {
                    row_hasher.update_field("null");
                } else {
                    row_hasher.update_field("value");
                    row_hasher.update_field(&array_value_to_string(array.as_ref(), row)?);
                }
            }
            row_fingerprints.push(row_hasher.finish());
        }
    }
    row_fingerprints.sort_unstable();
    for row_fingerprint in row_fingerprints {
        hasher.update_bytes(&row_fingerprint.to_le_bytes());
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}

fn plan_metrics_recursive(plan: &dyn ExecutionPlan) -> MetricsSet {
    let mut metrics = MetricsSet::new();
    append_plan_metrics(&mut metrics, plan);
    metrics
}

fn append_plan_metrics(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
    if let Some(plan_metrics) = plan.metrics() {
        for metric in plan_metrics.iter() {
            metrics.push(Arc::clone(metric));
        }
    }
    for child in plan.children() {
        append_plan_metrics(metrics, child.as_ref());
    }
}

#[derive(Debug, Clone)]
struct QueryRow {
    run_id: String,
    label: String,
    task_index: usize,
    iteration: usize,
    start_epoch_ms: u128,
    end_epoch_ms: u128,
    wall_ms: f64,
    outcome: &'static str,
    admission_sequence: Option<usize>,
    admission_wait_ms: Option<f64>,
    device_ordinal: Option<usize>,
    query_grant_bytes: Option<usize>,
    allocator_high_water_bytes: Option<usize>,
    max_task_observed_peak_bytes: Option<usize>,
    graph_algorithm: Option<String>,
    graph_directed: Option<bool>,
    graph_renumber: Option<bool>,
    graph_construction_policy: Option<String>,
    graph_peak_bytes: Option<usize>,
    graph_elapsed_ms: Option<f64>,
    graph_output_device_bytes: Option<usize>,
    output_rows: Option<usize>,
    output_fingerprint: Option<String>,
    native_lowered: bool,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct SampleRow {
    epoch_ms: u128,
    governor_capacity_bytes: u64,
    governor_outstanding_bytes: u64,
    governor_available_bytes: u64,
    active_grants: usize,
    active_queries: usize,
    queued_queries: usize,
    pool_reserved_current_bytes: u64,
    pool_reserved_high_bytes: u64,
    pool_used_current_bytes: u64,
    pool_used_high_bytes: u64,
    allocation_failure_count: u64,
    trim_retry_count: u64,
}

#[derive(Debug, Default)]
struct SampleCollection {
    aggregate: Vec<SampleRow>,
    by_device: Vec<DeviceSampleRow>,
}

#[derive(Debug, Clone, Copy, Default)]
struct DeviceSampleFacts {
    governor_capacity_bytes: u64,
    governor_outstanding_bytes: u64,
    governor_available_bytes: u64,
    active_grants: usize,
    gpu_execution_worker_count: usize,
    gpu_execution_active_jobs: usize,
    gpu_execution_queued_jobs: usize,
    gpu_execution_active_query_jobs: usize,
    pool_reserved_current_bytes: u64,
    pool_reserved_high_bytes: u64,
    pool_used_current_bytes: u64,
    pool_used_high_bytes: u64,
    allocation_failure_count: u64,
    trim_retry_count: u64,
}

#[derive(Debug, Clone)]
struct DeviceSampleRow {
    epoch_ms: u128,
    device_ordinal: usize,
    active_queries: usize,
    queued_queries: usize,
    facts: DeviceSampleFacts,
}

fn fmt_opt<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn fmt_opt_str(value: &Option<String>) -> String {
    value.clone().unwrap_or_else(|| "none".to_string())
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn epoch_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_nanos() as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests;
