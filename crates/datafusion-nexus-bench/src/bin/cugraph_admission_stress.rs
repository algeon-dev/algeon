//! cuGraph-heavy admission and memory-governance stress harness.
//!
//! Library-mode experiment binary: builds one process-scoped
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
    collections::BTreeMap,
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

use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect, displayable, metrics::MetricsSet},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfSessionStateBuilderExt, CugraphSqlConfig, MemoryShare,
    NativeDataFusionExecutionMetrics, NexusAdmissionPolicy, NexusGpuBackend, NexusMemoryPolicy, SlotGuarantee,
};
use tokio::sync::Barrier;

const SCHEMA_VERSION: u32 = 7;

const ENV_EDGES: &str = "CUDF_CUGRAPH_ADMISSION_EDGES_PARQUET";
const ENV_PAPERS: &str = "CUDF_CUGRAPH_ADMISSION_PAPERS_PARQUET";
const ENV_FOS: &str = "CUDF_CUGRAPH_ADMISSION_FOS_PARQUET";
const ENV_WORKLOAD: &str = "CUDF_CUGRAPH_ADMISSION_WORKLOAD";
const ENV_ITERATIONS: &str = "CUDF_CUGRAPH_ADMISSION_ITERATIONS";
const ENV_STAGGER_MS: &str = "CUDF_CUGRAPH_ADMISSION_STAGGER_MS";
const ENV_MAX_CONCURRENT: &str = "CUDF_CUGRAPH_ADMISSION_MAX_CONCURRENT_QUERIES";
const ENV_SLOT_GUARANTEE: &str = "CUDF_CUGRAPH_ADMISSION_SLOT_GUARANTEE";
const ENV_GRANT_MIN: &str = "CUDF_CUGRAPH_ADMISSION_GRANT_MIN";
const ENV_GRANT_WANT: &str = "CUDF_CUGRAPH_ADMISSION_GRANT_WANT";
const ENV_WAIT_TIMEOUT_MS: &str = "CUDF_CUGRAPH_ADMISSION_WAIT_TIMEOUT_MS";
const ENV_BUDGET_FRACTION: &str = "CUDF_CUGRAPH_ADMISSION_BUDGET_FRACTION";
const ENV_DEVICE_BUDGET_BYTES: &str = "CUDF_CUGRAPH_ADMISSION_DEVICE_BUDGET_BYTES";
const ENV_REQUESTED_GRANT_MIN_BYTES: &str = "CUDF_CUGRAPH_ADMISSION_REQUESTED_GRANT_MIN_BYTES";
const ENV_REQUESTED_GRANT_BYTES: &str = "CUDF_CUGRAPH_ADMISSION_REQUESTED_GRANT_BYTES";
const ENV_RAW_INPUT_EDGES: &str = "CUDF_CUGRAPH_ADMISSION_RAW_INPUT_EDGES";
const ENV_SAMPLE_INTERVAL_MS: &str = "CUDF_CUGRAPH_ADMISSION_SAMPLE_INTERVAL_MS";
const ENV_QUERY_TIMEOUT_MS: &str = "CUDF_CUGRAPH_ADMISSION_QUERY_TIMEOUT_MS";
const ENV_OUT_DIR: &str = "CUDF_CUGRAPH_ADMISSION_OUT_DIR";
const ENV_RUN_ID: &str = "CUDF_CUGRAPH_ADMISSION_RUN_ID";
const ENV_BFS_SEED: &str = "CUDF_CUGRAPH_ADMISSION_BFS_SEED";
const ENV_BATCH_SIZE: &str = "CUDF_CUGRAPH_ADMISSION_BATCH_SIZE";
const ENV_PRINT_PLANNING: &str = "CUDF_CUGRAPH_ADMISSION_PRINT_PLANNING";

/// Retained optimizer rules (diagnostic planning-report mode only).
static PLANNING_RULES: Mutex<Vec<Arc<CudfNativePhysicalOptimizerRule>>> = Mutex::new(Vec::new());

const QUERY_COLUMNS: &[&str] = &[
    "run_id",
    "label",
    "task_index",
    "iteration",
    "start_epoch_ms",
    "end_epoch_ms",
    "wall_ms",
    "outcome",
    "admission_sequence",
    "admission_wait_ms",
    "device_ordinal",
    "query_grant_bytes",
    "reservation_requested_bytes",
    "reservation_granted_bytes",
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "allocator_high_water_bytes",
    "max_task_observed_peak_bytes",
    "max_task_reservation_bytes",
    "graph_algorithm",
    "graph_class",
    "graph_outcome",
    "graph_raw_input_edges",
    "graph_input_edges",
    "graph_vertex_estimate",
    "graph_vertex_estimate_source",
    "graph_vertex_width_bytes",
    "graph_weight_width_bytes",
    "graph_directed",
    "graph_renumber",
    "graph_construction_policy",
    "graph_edge_normalize_inserted",
    "graph_calibration_source",
    "graph_seed_count",
    "graph_seed_mode",
    "graph_model_requested_bytes",
    "graph_requested_bytes",
    "graph_budget_bytes",
    "graph_model_error_ratio",
    "graph_statistics_exact_vertices",
    "graph_statistics_degree_max",
    "graph_statistics_degree_p99",
    "graph_statistics_sum_min_degree_oriented",
    "graph_statistics_oriented_edge_count",
    "graph_statistics_peak_chunk_min_degree_sum",
    "graph_statistics_sum_out_degree",
    "graph_peak_bytes",
    "graph_reason",
    "graph_elapsed_ms",
    "graph_output_device_bytes",
    "output_rows",
    "native_lowered",
    "error",
];

const SAMPLE_COLUMNS: &[&str] = &[
    "epoch_ms",
    "governor_capacity_bytes",
    "governor_outstanding_bytes",
    "governor_available_bytes",
    "active_grants",
    "active_queries",
    "queued_queries",
    "pool_reserved_current_bytes",
    "pool_reserved_high_bytes",
    "pool_used_current_bytes",
    "pool_used_high_bytes",
    "allocation_failure_count",
    "trim_retry_count",
];

#[derive(Debug, Clone)]
struct HarnessConfig {
    edges_parquet: String,
    papers_parquet: Option<String>,
    fos_parquet: Option<String>,
    workload: Vec<(String, usize)>,
    iterations: usize,
    stagger_ms: u64,
    max_concurrent_queries: usize,
    slot_guarantee: SlotGuarantee,
    grant_min: Option<MemoryShare>,
    grant_want: Option<MemoryShare>,
    wait_timeout_ms: u64,
    budget_fraction: Option<f64>,
    device_budget_bytes: Option<u64>,
    requested_grant_min_bytes: Option<u64>,
    requested_grant_bytes: Option<u64>,
    raw_input_edges: Option<u64>,
    sample_interval_ms: u64,
    query_timeout_ms: u64,
    out_dir: PathBuf,
    run_id: String,
    bfs_seed: Option<i64>,
    batch_size: Option<usize>,
}

impl HarnessConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let workload_raw = std::env::var(ENV_WORKLOAD).unwrap_or_else(|_| "catalog_smoke".to_string());
        let workload = parse_workload(&workload_raw)?;
        let out_dir = PathBuf::from(std::env::var(ENV_OUT_DIR).unwrap_or_else(|_| ".".to_string()));
        let run_id = std::env::var(ENV_RUN_ID)
            .unwrap_or_else(|_| format!("run-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()));
        Ok(Self {
            edges_parquet: std::env::var(ENV_EDGES).map_err(|_| format!("{ENV_EDGES} is required"))?,
            papers_parquet: std::env::var(ENV_PAPERS).ok(),
            fos_parquet: std::env::var(ENV_FOS).ok(),
            workload,
            iterations: env_parse(ENV_ITERATIONS, 1)?,
            stagger_ms: env_parse(ENV_STAGGER_MS, 0)?,
            max_concurrent_queries: env_parse(ENV_MAX_CONCURRENT, 2)?,
            slot_guarantee: parse_slot_guarantee(
                &std::env::var(ENV_SLOT_GUARANTEE).unwrap_or_else(|_| "guaranteed_minimum".to_string()),
            )?,
            grant_min: parse_memory_share_env(ENV_GRANT_MIN)?,
            grant_want: parse_memory_share_env(ENV_GRANT_WANT)?,
            wait_timeout_ms: env_parse(ENV_WAIT_TIMEOUT_MS, 60_000)?,
            budget_fraction: parse_optional_f64(ENV_BUDGET_FRACTION)?,
            device_budget_bytes: parse_optional_u64(ENV_DEVICE_BUDGET_BYTES)?,
            requested_grant_min_bytes: parse_optional_u64(ENV_REQUESTED_GRANT_MIN_BYTES)?,
            requested_grant_bytes: parse_optional_u64(ENV_REQUESTED_GRANT_BYTES)?,
            raw_input_edges: parse_optional_u64(ENV_RAW_INPUT_EDGES)?,
            sample_interval_ms: env_parse(ENV_SAMPLE_INTERVAL_MS, 100)?,
            query_timeout_ms: env_parse(ENV_QUERY_TIMEOUT_MS, 600_000)?,
            out_dir,
            run_id,
            bfs_seed: parse_optional_i64(ENV_BFS_SEED)?,
            batch_size: parse_optional_usize(ENV_BATCH_SIZE)?,
        })
    }
}

#[derive(Debug, Clone)]
struct QuerySpec {
    label: String,
    sql: String,
    needs_papers: bool,
    needs_fos: bool,
}

fn catalog(seed: i64) -> Vec<QuerySpec> {
    let spec = |label: &str, sql: String, needs_papers: bool, needs_fos: bool| QuerySpec {
        label: label.to_string(),
        sql,
        needs_papers,
        needs_fos,
    };
    vec![
        spec(
            "pagerank",
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank('citation_edges', 'src', 'dst', 'weight')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "pagerank_nw",
            "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank('citation_edges', 'src', 'dst', NULL)"
                .to_string(),
            false,
            false,
        ),
        spec(
            "bfs",
            format!(
                "SELECT COUNT(*) AS reached FROM cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"depth_limit\": 3, \"output_mode\": \"normalized\"}}')"
            ),
            false,
            false,
        ),
        spec("edge_scan", "SELECT src, dst FROM citation_edges".to_string(), false, false),
        spec(
            "bfs_deep",
            format!(
                "SELECT COUNT(*) AS reached FROM cugraph_bfs('citation_edges', {seed}, 'src', 'dst', NULL, \
                 '{{\"output_mode\": \"normalized\"}}')"
            ),
            false,
            false,
        ),
        spec(
            "sssp",
            format!("SELECT COUNT(*) AS n FROM cugraph_sssp('citation_edges', {seed}, 'src', 'dst', 'weight')"),
            false,
            false,
        ),
        spec(
            "wcc",
            "SELECT COUNT(*) AS n FROM cugraph_weakly_connected_components('citation_edges', 'src', 'dst')".to_string(),
            false,
            false,
        ),
        spec(
            "louvain",
            "SELECT COUNT(DISTINCT partition) AS communities FROM cugraph_louvain('citation_edges', 'src', 'dst', \
             NULL, '{\"directed\": false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "core_number",
            "SELECT COUNT(*) AS n FROM cugraph_core_number('citation_edges', 'src', 'dst', NULL, '{\"directed\": \
             false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "triangles",
            "SELECT COUNT(*) AS n FROM cugraph_triangle_count_all('citation_edges', 'src', 'dst', NULL, \
             '{\"directed\": false}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "degrees",
            "SELECT COUNT(*) AS n FROM cugraph_degrees_all('citation_edges', 'src', 'dst')".to_string(),
            false,
            false,
        ),
        spec(
            "betweenness",
            "SELECT COUNT(*) AS n FROM cugraph_betweenness_centrality('citation_edges', 'src', 'dst', NULL, \
             '{\"k\": 64}')"
                .to_string(),
            false,
            false,
        ),
        spec(
            "rel_join",
            "SELECT p.year, COUNT(*) AS papers, AVG(p.n_citation) AS avg_citation FROM paper_fos f JOIN papers p ON \
             p.paper_id = f.paper_id WHERE f.fos_name = 'Artificial intelligence' GROUP BY p.year ORDER BY p.year"
                .to_string(),
            true,
            true,
        ),
        spec(
            "rel_scan_agg",
            "SELECT year, COUNT(*) AS papers, SUM(n_citation) AS citations FROM papers GROUP BY year ORDER BY year"
                .to_string(),
            true,
            false,
        ),
        spec(
            "graph_enrich",
            "WITH pr AS (SELECT vertex, value FROM cugraph_pagerank('citation_edges', 'src', 'dst', 'weight')) \
             SELECT p.year, COUNT(*) AS papers, MAX(pr.value) AS top_pagerank FROM pr JOIN papers p ON p.paper_id = \
             pr.vertex GROUP BY p.year ORDER BY p.year"
                .to_string(),
            true,
            false,
        ),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
    let samples = Arc::new(Mutex::new(Vec::<SampleRow>::new()));
    let sampler_handle = spawn_sampler(&backend, &config, Arc::clone(&stop_flag), Arc::clone(&samples));

    // Pre-build one session per worker task so registration cost is off the
    // measured path, then fire everything through one barrier.
    let mut workers = Vec::with_capacity(requests.len());
    for request in &requests {
        let ctx = build_session(&backend, &config, request).await?;
        workers.push((request.clone(), ctx));
    }

    let barrier = Arc::new(Barrier::new(workers.len()));
    let run_started = Instant::now();
    let mut tasks = Vec::with_capacity(workers.len());
    for (request, ctx) in workers {
        let barrier = Arc::clone(&barrier);
        let config = config.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            if config.stagger_ms > 0 && request.task_index > 0 {
                tokio::time::sleep(Duration::from_millis(config.stagger_ms * request.task_index as u64)).await;
            }
            let mut rows = Vec::with_capacity(config.iterations);
            for iteration in 0..config.iterations {
                rows.push(run_one(&config, &ctx, &request, iteration).await);
            }
            rows
        }));
    }

    let mut query_rows = Vec::new();
    for task in tasks {
        match task.await {
            Ok(rows) => query_rows.extend(rows),
            Err(join_error) => eprintln!("worker task panicked: {join_error}"),
        }
    }
    let makespan_ms = duration_ms(run_started.elapsed());

    stop_flag.store(true, Ordering::SeqCst);
    let _ = sampler_handle.join();

    write_outputs(&config, &query_rows, &samples.lock().unwrap(), makespan_ms)?;
    eprintln!("# done run_id={} makespan_ms={makespan_ms:.1} rows={}", config.run_id, query_rows.len());
    Ok(())
}

fn build_backend(config: &HarnessConfig) -> Result<NexusGpuBackend, Box<dyn Error>> {
    let memory_policy = match config.device_budget_bytes {
        Some(bytes) => NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: bytes,
            max_source_chunk_bytes: 256 * 1024 * 1024,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        },
        None => NexusMemoryPolicy::BoundedAuto { device_ordinals: vec![0], budget_fraction: config.budget_fraction },
    };
    let mut admission = NexusAdmissionPolicy::default()
        .with_max_concurrent_queries(config.max_concurrent_queries)
        .with_slot_guarantee(config.slot_guarantee)
        .with_wait_timeout(Duration::from_millis(config.wait_timeout_ms));
    if let Some(min) = config.grant_min {
        admission = admission.with_grant_min(min);
    }
    if let Some(want) = config.grant_want {
        admission = admission.with_grant_want(want);
    }
    Ok(NexusGpuBackend::builder().memory_policy(memory_policy).admission(admission).build()?)
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
    let builder = backend.install_on_with_cugraph_sql(builder, CugraphSqlConfig::default())?;
    // Diagnostic mode: swap in a retained copy of the optimizer rule (same
    // config, so the rule-name dedup key replaces the backend's instance) so
    // per-query PlanningReports can be read back.
    let builder = if std::env::var(ENV_PRINT_PLANNING).is_ok() {
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(backend.optimizer_config().clone()));
        PLANNING_RULES.lock().unwrap().push(Arc::clone(&rule));
        builder.with_cudf_native_rule(rule)
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
    if let Some(bytes) = config.requested_grant_min_bytes {
        ctx.sql(&format!("SET datafusion_nexus.request.requested_grant_min_bytes = {bytes}")).await?.collect().await?;
    }
    if let Some(bytes) = config.requested_grant_bytes {
        ctx.sql(&format!("SET datafusion_nexus.request.requested_grant_bytes = {bytes}")).await?.collect().await?;
    }
    if let Some(raw_input_edges) = config.raw_input_edges {
        ctx.sql(&format!("SET datafusion_nexus.request.graph_raw_input_edges = {raw_input_edges}"))
            .await?
            .collect()
            .await?;
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

fn expand_workload(config: &HarnessConfig, specs: &[QuerySpec]) -> Result<Vec<Request>, Box<dyn Error>> {
    let mut requests = Vec::new();
    let mut task_index = 0;
    for (label, count) in &config.workload {
        if label == "catalog_smoke" {
            for spec in specs {
                requests.push(Request { task_index, spec: spec.clone() });
                task_index += 1;
            }
            continue;
        }
        let spec = specs
            .iter()
            .find(|spec| &spec.label == label)
            .ok_or_else(|| format!("unknown workload label `{label}`"))?;
        for _ in 0..*count {
            requests.push(Request { task_index, spec: spec.clone() });
            task_index += 1;
        }
    }
    Ok(requests)
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
    row.native_lowered = plan_display.contains("CugraphAlgoExec") || plan_display.contains("NativeCudf");
    if std::env::var(ENV_PRINT_PLANNING).is_ok() {
        for rule in PLANNING_RULES.lock().unwrap().iter() {
            if let Some(report) = rule.last_report() {
                for outcome in report.outcomes() {
                    match outcome {
                        CandidateOutcome::Rewritten { candidate } => {
                            eprintln!("[planning] {} REWRITTEN shape={}", request.spec.label, candidate.shape);
                        },
                        CandidateOutcome::Rejected { candidate, reason } => {
                            eprintln!(
                                "[planning] {} REJECTED shape={} reason={}",
                                request.spec.label,
                                candidate.shape,
                                reason.code()
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
            row.finish_ok(started)
        },
        Ok(Err(error)) => {
            row.absorb_metrics(&NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref())));
            row.finish_failed(started, error.to_string())
        },
        Err(_) => row.finish_failed(started, format!("query_timeout after {} ms", config.query_timeout_ms)),
    }
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
    reservation_requested_bytes: usize,
    reservation_granted_bytes: usize,
    reservation_live_peak_bytes: usize,
    reservation_denied_count: usize,
    allocator_high_water_bytes: Option<usize>,
    max_task_observed_peak_bytes: Option<usize>,
    max_task_reservation_bytes: usize,
    graph_algorithm: Option<String>,
    graph_class: Option<String>,
    graph_outcome: Option<String>,
    graph_raw_input_edges: Option<usize>,
    graph_input_edges: Option<usize>,
    graph_vertex_estimate: Option<usize>,
    graph_vertex_estimate_source: Option<String>,
    graph_vertex_width_bytes: Option<usize>,
    graph_weight_width_bytes: Option<usize>,
    graph_directed: Option<bool>,
    graph_renumber: Option<bool>,
    graph_construction_policy: Option<String>,
    graph_edge_normalize_inserted: Option<bool>,
    graph_calibration_source: Option<String>,
    graph_seed_count: Option<usize>,
    graph_seed_mode: Option<String>,
    graph_model_requested_bytes: Option<usize>,
    graph_requested_bytes: Option<usize>,
    graph_budget_bytes: Option<usize>,
    graph_model_error_ratio: Option<String>,
    graph_statistics_exact_vertices: Option<usize>,
    graph_statistics_degree_max: Option<usize>,
    graph_statistics_degree_p99: Option<usize>,
    graph_statistics_sum_min_degree_oriented: Option<usize>,
    graph_statistics_oriented_edge_count: Option<usize>,
    graph_statistics_peak_chunk_min_degree_sum: Option<usize>,
    graph_statistics_sum_out_degree: Option<usize>,
    graph_peak_bytes: Option<usize>,
    graph_reason: Option<String>,
    graph_elapsed_ms: Option<f64>,
    graph_output_device_bytes: Option<usize>,
    output_rows: Option<usize>,
    native_lowered: bool,
    error: Option<String>,
}

impl QueryRow {
    fn new(config: &HarnessConfig, request: &Request, iteration: usize, start_epoch_ms: u128) -> Self {
        Self {
            run_id: config.run_id.clone(),
            label: request.spec.label.clone(),
            task_index: request.task_index,
            iteration,
            start_epoch_ms,
            end_epoch_ms: 0,
            wall_ms: 0.0,
            outcome: "unknown",
            admission_sequence: None,
            admission_wait_ms: None,
            device_ordinal: None,
            query_grant_bytes: None,
            reservation_requested_bytes: 0,
            reservation_granted_bytes: 0,
            reservation_live_peak_bytes: 0,
            reservation_denied_count: 0,
            allocator_high_water_bytes: None,
            max_task_observed_peak_bytes: None,
            max_task_reservation_bytes: 0,
            graph_algorithm: None,
            graph_class: None,
            graph_outcome: None,
            graph_raw_input_edges: None,
            graph_input_edges: None,
            graph_vertex_estimate: None,
            graph_vertex_estimate_source: None,
            graph_vertex_width_bytes: None,
            graph_weight_width_bytes: None,
            graph_directed: None,
            graph_renumber: None,
            graph_construction_policy: None,
            graph_edge_normalize_inserted: None,
            graph_calibration_source: None,
            graph_seed_count: None,
            graph_seed_mode: None,
            graph_model_requested_bytes: None,
            graph_requested_bytes: None,
            graph_budget_bytes: None,
            graph_model_error_ratio: None,
            graph_statistics_exact_vertices: None,
            graph_statistics_degree_max: None,
            graph_statistics_degree_p99: None,
            graph_statistics_sum_min_degree_oriented: None,
            graph_statistics_oriented_edge_count: None,
            graph_statistics_peak_chunk_min_degree_sum: None,
            graph_statistics_sum_out_degree: None,
            graph_peak_bytes: None,
            graph_reason: None,
            graph_elapsed_ms: None,
            graph_output_device_bytes: None,
            output_rows: None,
            native_lowered: false,
            error: None,
        }
    }

    fn absorb_metrics(&mut self, metrics: &NativeDataFusionExecutionMetrics) {
        self.admission_sequence = metrics.admission_sequence;
        self.admission_wait_ms = metrics.admission_wait_time_ns.map(|ns| ns as f64 / 1_000_000.0);
        self.device_ordinal = metrics.device_ordinal;
        self.query_grant_bytes = metrics.query_grant_bytes;
        self.reservation_requested_bytes = metrics.reservation_requested_bytes;
        self.reservation_granted_bytes = metrics.reservation_granted_bytes;
        self.reservation_live_peak_bytes = metrics.reservation_live_peak_bytes;
        self.reservation_denied_count = metrics.reservation_denied_count;
        self.allocator_high_water_bytes = metrics.allocator_high_water_bytes;
        self.max_task_observed_peak_bytes = metrics.max_task_observed_peak_bytes;
        self.max_task_reservation_bytes = metrics.max_task_reservation_bytes;
        if let Some(reservation) = metrics.graph_reservation_rows.first() {
            self.graph_algorithm = Some(reservation.algorithm.clone());
            self.graph_class = Some(reservation.class.clone());
            self.graph_outcome = Some(reservation.outcome.clone());
            self.graph_raw_input_edges = reservation.raw_input_edges;
            self.graph_input_edges = Some(reservation.input_edges);
            self.graph_vertex_estimate = Some(reservation.vertex_estimate);
            self.graph_vertex_estimate_source = Some(reservation.vertex_estimate_source.clone());
            self.graph_vertex_width_bytes = Some(reservation.vertex_width_bytes);
            self.graph_weight_width_bytes = reservation.weight_width_bytes;
            self.graph_directed = Some(reservation.directed);
            self.graph_renumber = Some(reservation.renumber);
            self.graph_construction_policy = Some(reservation.construction_policy.clone());
            self.graph_edge_normalize_inserted = Some(reservation.edge_normalize_inserted);
            self.graph_calibration_source = Some(reservation.calibration_source.clone());
            self.graph_seed_count = reservation.seed_count;
            self.graph_seed_mode = Some(reservation.seed_mode.clone());
            self.graph_model_requested_bytes = Some(reservation.model_requested_bytes);
            self.graph_requested_bytes = Some(reservation.requested_bytes);
            self.graph_budget_bytes = reservation.budget_bytes;
            self.graph_model_error_ratio = reservation.graph_model_error_ratio.clone();
            self.graph_statistics_exact_vertices = reservation.graph_statistics_exact_vertices;
            self.graph_statistics_degree_max = reservation.graph_statistics_degree_max;
            self.graph_statistics_degree_p99 = reservation.graph_statistics_degree_p99;
            self.graph_statistics_sum_min_degree_oriented = reservation.graph_statistics_sum_min_degree_oriented;
            self.graph_statistics_oriented_edge_count = reservation.graph_statistics_oriented_edge_count;
            self.graph_statistics_peak_chunk_min_degree_sum = reservation.graph_statistics_peak_chunk_min_degree_sum;
            self.graph_statistics_sum_out_degree = reservation.graph_statistics_sum_out_degree;
            self.graph_reason = Some(reservation.reason.clone());
        }
        if let Some(runtime) = metrics.graph_runtime_rows.first() {
            self.graph_elapsed_ms = Some(runtime.elapsed_ns as f64 / 1_000_000.0);
            self.graph_output_device_bytes = runtime.output_device_bytes;
            self.graph_peak_bytes = runtime.graph_peak_bytes;
            if self.graph_algorithm.is_none() {
                self.graph_algorithm = Some(runtime.algorithm.clone());
            }
        }
    }

    fn finish_ok(mut self, started: Instant) -> Self {
        self.outcome = "succeeded";
        self.end_epoch_ms = epoch_ms();
        self.wall_ms = duration_ms(started.elapsed());
        self
    }

    fn finish_failed(mut self, started: Instant, error: String) -> Self {
        self.outcome = "failed";
        self.end_epoch_ms = epoch_ms();
        self.wall_ms = duration_ms(started.elapsed());
        self.error = Some(error);
        self
    }

    fn to_tsv(&self) -> String {
        let fields = [
            self.run_id.clone(),
            self.label.clone(),
            self.task_index.to_string(),
            self.iteration.to_string(),
            self.start_epoch_ms.to_string(),
            self.end_epoch_ms.to_string(),
            format!("{:.3}", self.wall_ms),
            self.outcome.to_string(),
            fmt_opt(self.admission_sequence),
            self.admission_wait_ms.map_or_else(|| "none".to_string(), |value| format!("{value:.3}")),
            fmt_opt(self.device_ordinal),
            fmt_opt(self.query_grant_bytes),
            self.reservation_requested_bytes.to_string(),
            self.reservation_granted_bytes.to_string(),
            self.reservation_live_peak_bytes.to_string(),
            self.reservation_denied_count.to_string(),
            fmt_opt(self.allocator_high_water_bytes),
            fmt_opt(self.max_task_observed_peak_bytes),
            self.max_task_reservation_bytes.to_string(),
            fmt_opt_str(&self.graph_algorithm),
            fmt_opt_str(&self.graph_class),
            fmt_opt_str(&self.graph_outcome),
            fmt_opt(self.graph_raw_input_edges),
            fmt_opt(self.graph_input_edges),
            fmt_opt(self.graph_vertex_estimate),
            fmt_opt_str(&self.graph_vertex_estimate_source),
            fmt_opt(self.graph_vertex_width_bytes),
            fmt_opt(self.graph_weight_width_bytes),
            self.graph_directed.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.graph_renumber.map_or_else(|| "none".to_string(), |value| value.to_string()),
            fmt_opt_str(&self.graph_construction_policy),
            self.graph_edge_normalize_inserted.map_or_else(|| "none".to_string(), |value| value.to_string()),
            fmt_opt_str(&self.graph_calibration_source),
            fmt_opt(self.graph_seed_count),
            fmt_opt_str(&self.graph_seed_mode),
            fmt_opt(self.graph_model_requested_bytes),
            fmt_opt(self.graph_requested_bytes),
            fmt_opt(self.graph_budget_bytes),
            fmt_opt_str(&self.graph_model_error_ratio),
            fmt_opt(self.graph_statistics_exact_vertices),
            fmt_opt(self.graph_statistics_degree_max),
            fmt_opt(self.graph_statistics_degree_p99),
            fmt_opt(self.graph_statistics_sum_min_degree_oriented),
            fmt_opt(self.graph_statistics_oriented_edge_count),
            fmt_opt(self.graph_statistics_peak_chunk_min_degree_sum),
            fmt_opt(self.graph_statistics_sum_out_degree),
            fmt_opt(self.graph_peak_bytes),
            fmt_opt_str(&self.graph_reason),
            self.graph_elapsed_ms.map_or_else(|| "none".to_string(), |value| format!("{value:.3}")),
            fmt_opt(self.graph_output_device_bytes),
            fmt_opt(self.output_rows),
            self.native_lowered.to_string(),
            self.error.as_deref().map(tsv_field).unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), QUERY_COLUMNS.len());
        fields.join("\t")
    }
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

impl SampleRow {
    fn to_tsv(&self) -> String {
        [
            self.epoch_ms.to_string(),
            self.governor_capacity_bytes.to_string(),
            self.governor_outstanding_bytes.to_string(),
            self.governor_available_bytes.to_string(),
            self.active_grants.to_string(),
            self.active_queries.to_string(),
            self.queued_queries.to_string(),
            self.pool_reserved_current_bytes.to_string(),
            self.pool_reserved_high_bytes.to_string(),
            self.pool_used_current_bytes.to_string(),
            self.pool_used_high_bytes.to_string(),
            self.allocation_failure_count.to_string(),
            self.trim_retry_count.to_string(),
        ]
        .join("\t")
    }
}

fn spawn_sampler(
    backend: &NexusGpuBackend,
    config: &HarnessConfig,
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<SampleRow>>>,
) -> std::thread::JoinHandle<()> {
    let backend = backend.clone();
    let interval = Duration::from_millis(config.sample_interval_ms);
    std::thread::spawn(move || {
        let manager = cudf::DeviceMemoryManager::install_for_device(
            cudf::DeviceId::new(0).expect("device 0 must exist for sampling"),
        )
        .ok();
        while !stop_flag.load(Ordering::SeqCst) {
            let mut row = SampleRow {
                epoch_ms: epoch_ms(),
                governor_capacity_bytes: 0,
                governor_outstanding_bytes: 0,
                governor_available_bytes: 0,
                active_grants: 0,
                active_queries: 0,
                queued_queries: 0,
                pool_reserved_current_bytes: 0,
                pool_reserved_high_bytes: 0,
                pool_used_current_bytes: 0,
                pool_used_high_bytes: 0,
                allocation_failure_count: 0,
                trim_retry_count: 0,
            };
            if let Ok(reports) = backend.query_service().memory_governor_reports()
                && let Some(report) = reports.first()
            {
                row.governor_capacity_bytes = report.capacity_bytes;
                row.governor_outstanding_bytes = report.outstanding_bytes;
                row.governor_available_bytes = report.available_bytes;
                row.active_grants = report.grants.len();
            }
            if let Ok(snapshot) = backend.query_service().admission_snapshot() {
                row.active_queries = snapshot.active_queries;
                row.queued_queries = snapshot.queued_queries;
            }
            for failure in backend.query_service().device_memory_failure_reports() {
                row.allocation_failure_count += failure.allocation_failure_count;
                row.trim_retry_count += failure.trim_retry_count;
            }
            if let Some(manager) = manager.as_ref()
                && let Ok(usage) = manager.pool_usage()
            {
                row.pool_reserved_current_bytes = usage.reserved_current_bytes;
                row.pool_reserved_high_bytes = usage.reserved_high_bytes;
                row.pool_used_current_bytes = usage.used_current_bytes;
                row.pool_used_high_bytes = usage.used_high_bytes;
            }
            samples.lock().unwrap().push(row);
            std::thread::sleep(interval);
        }
    })
}

fn write_outputs(
    config: &HarnessConfig,
    query_rows: &[QueryRow],
    samples: &[SampleRow],
    makespan_ms: f64,
) -> Result<(), Box<dyn Error>> {
    let queries_path = config.out_dir.join("queries.tsv");
    let mut queries_file = fs::File::create(&queries_path)?;
    writeln!(queries_file, "# cugraph_admission_stress_queries_tsv_schema_version={SCHEMA_VERSION}")?;
    writeln!(queries_file, "{}", QUERY_COLUMNS.join("\t"))?;
    for row in query_rows {
        writeln!(queries_file, "{}", row.to_tsv())?;
    }

    let samples_path = config.out_dir.join("samples.tsv");
    let mut samples_file = fs::File::create(&samples_path)?;
    writeln!(samples_file, "# cugraph_admission_stress_samples_tsv_schema_version={SCHEMA_VERSION}")?;
    writeln!(samples_file, "{}", SAMPLE_COLUMNS.join("\t"))?;
    for row in samples {
        writeln!(samples_file, "{}", row.to_tsv())?;
    }

    type LabelStats = (usize, usize, Vec<f64>, Vec<f64>);
    let mut label_stats: BTreeMap<&str, LabelStats> = BTreeMap::new();
    for row in query_rows {
        let entry = label_stats.entry(row.label.as_str()).or_insert((0, 0, Vec::new(), Vec::new()));
        if row.outcome == "succeeded" {
            entry.0 += 1;
            entry.2.push(row.wall_ms);
            if let Some(wait) = row.admission_wait_ms {
                entry.3.push(wait);
            }
        } else {
            entry.1 += 1;
        }
    }
    let summary = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "run_id": config.run_id,
        "makespan_ms": makespan_ms,
        "config": {
            "workload": config.workload.iter().map(|(label, count)| format!("{label}:{count}")).collect::<Vec<_>>(),
            "iterations": config.iterations,
            "stagger_ms": config.stagger_ms,
            "max_concurrent_queries": config.max_concurrent_queries,
            "slot_guarantee": format!("{:?}", config.slot_guarantee),
            "grant_min": config.grant_min.map(|share| format!("{share:?}")),
            "grant_want": config.grant_want.map(|share| format!("{share:?}")),
            "wait_timeout_ms": config.wait_timeout_ms,
            "budget_fraction": config.budget_fraction,
            "device_budget_bytes": config.device_budget_bytes,
            "requested_grant_min_bytes": config.requested_grant_min_bytes,
            "requested_grant_bytes": config.requested_grant_bytes,
            "raw_input_edges": config.raw_input_edges,
            "query_timeout_ms": config.query_timeout_ms,
            "edges_parquet": config.edges_parquet,
        },
        "labels": label_stats.iter().map(|(label, (ok, failed, walls, waits))| {
            serde_json::json!({
                "label": label,
                "succeeded": ok,
                "failed": failed,
                "wall_ms_mean": mean(walls),
                "wall_ms_p50": percentile(walls, 0.50),
                "wall_ms_p95": percentile(walls, 0.95),
                "admission_wait_ms_mean": mean(waits),
                "admission_wait_ms_max": waits.iter().cloned().fold(f64::NAN, f64::max),
            })
        }).collect::<Vec<_>>(),
        "peaks": {
            "pool_used_high_bytes": samples.iter().map(|s| s.pool_used_high_bytes).max(),
            "pool_reserved_high_bytes": samples.iter().map(|s| s.pool_reserved_high_bytes).max(),
            "governor_outstanding_bytes_max": samples.iter().map(|s| s.governor_outstanding_bytes).max(),
            "active_queries_max": samples.iter().map(|s| s.active_queries).max(),
            "queued_queries_max": samples.iter().map(|s| s.queued_queries).max(),
            "allocation_failure_count_last": samples.last().map(|s| s.allocation_failure_count),
        },
    });
    fs::write(config.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { f64::NAN } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn parse_workload(raw: &str) -> Result<Vec<(String, usize)>, Box<dyn Error>> {
    let mut workload = Vec::new();
    for item in raw.split(',').map(str::trim).filter(|item| !item.is_empty()) {
        match item.split_once(':') {
            Some((label, count)) => workload.push((label.to_string(), count.parse()?)),
            None => workload.push((item.to_string(), 1)),
        }
    }
    if workload.is_empty() {
        return Err("empty workload".into());
    }
    Ok(workload)
}

fn parse_slot_guarantee(raw: &str) -> Result<SlotGuarantee, Box<dyn Error>> {
    match raw {
        "guaranteed_minimum" => Ok(SlotGuarantee::GuaranteedMinimum),
        "optimistic" => Ok(SlotGuarantee::Optimistic),
        other => Err(format!("unsupported slot guarantee `{other}`").into()),
    }
}

fn parse_memory_share_env(name: &str) -> Result<Option<MemoryShare>, Box<dyn Error>> {
    let Ok(raw) = std::env::var(name) else {
        return Ok(None);
    };
    let raw = raw.trim();
    if raw == "policy_default" {
        return Ok(Some(MemoryShare::PolicyDefault));
    }
    if raw.contains('.') {
        return Ok(Some(MemoryShare::Fraction(raw.parse()?)));
    }
    Ok(Some(MemoryShare::Bytes(raw.parse()?)))
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where T::Err: Error + 'static {
    match std::env::var(name) {
        Ok(raw) => Ok(raw.trim().parse::<T>()?),
        Err(_) => Ok(default),
    }
}

fn parse_optional_f64(name: &str) -> Result<Option<f64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<f64>().map_err(Into::into)).transpose()
}

fn parse_optional_u64(name: &str) -> Result<Option<u64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<u64>().map_err(Into::into)).transpose()
}

fn parse_optional_i64(name: &str) -> Result<Option<i64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<i64>().map_err(Into::into)).transpose()
}

fn parse_optional_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<usize>().map_err(Into::into)).transpose()
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
mod tests {
    use datafusion_nexus::{NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow};

    use super::*;

    #[test]
    fn query_tsv_row_contract_matches_columns() {
        let config = HarnessConfig {
            edges_parquet: "edges.parquet".into(),
            papers_parquet: None,
            fos_parquet: None,
            workload: vec![("pagerank".into(), 1)],
            iterations: 1,
            stagger_ms: 0,
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            grant_min: None,
            grant_want: None,
            wait_timeout_ms: 1000,
            budget_fraction: None,
            device_budget_bytes: None,
            requested_grant_min_bytes: None,
            requested_grant_bytes: None,
            raw_input_edges: None,
            sample_interval_ms: 100,
            query_timeout_ms: 1000,
            out_dir: PathBuf::from("."),
            run_id: "test".into(),
            bfs_seed: None,
            batch_size: None,
        };
        let request = Request {
            task_index: 0,
            spec: QuerySpec { label: "pagerank".into(), sql: "SELECT 1".into(), needs_papers: false, needs_fos: false },
        };
        let row = QueryRow::new(&config, &request, 0, 0).finish_failed(Instant::now(), "x\ty".into());
        assert_eq!(row.to_tsv().split('\t').count(), QUERY_COLUMNS.len());
        assert_eq!(QUERY_COLUMNS.first(), Some(&"run_id"));
        assert_eq!(QUERY_COLUMNS.last(), Some(&"error"));
    }

    #[test]
    fn query_row_absorbs_graph_calibration_fields() {
        let config = HarnessConfig {
            edges_parquet: "edges.parquet".into(),
            papers_parquet: None,
            fos_parquet: None,
            workload: vec![("pagerank".into(), 1)],
            iterations: 1,
            stagger_ms: 0,
            max_concurrent_queries: 2,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            grant_min: None,
            grant_want: None,
            wait_timeout_ms: 1000,
            budget_fraction: None,
            device_budget_bytes: None,
            requested_grant_min_bytes: None,
            requested_grant_bytes: None,
            raw_input_edges: None,
            sample_interval_ms: 100,
            query_timeout_ms: 1000,
            out_dir: PathBuf::from("."),
            run_id: "test".into(),
            bfs_seed: None,
            batch_size: None,
        };
        let request = Request {
            task_index: 0,
            spec: QuerySpec { label: "pagerank".into(), sql: "SELECT 1".into(), needs_papers: false, needs_fos: false },
        };
        let mut row = QueryRow::new(&config, &request, 0, 0);
        row.absorb_metrics(&NativeDataFusionExecutionMetrics {
            graph_reservation_rows: vec![NativeGraphReservationMetricRow {
                node_id: "g0".into(),
                algorithm: "triangle_count_all".into(),
                class: "workspace_heavy_data_dependent".into(),
                outcome: "admitted".into(),
                raw_input_edges: Some(45),
                input_edges: 90,
                vertex_upper_bound: 180,
                vertex_estimate: 12,
                vertex_estimate_source: "runtime_exact_distinct".into(),
                vertex_width_bytes: 8,
                weight_width_bytes: Some(8),
                has_edge_id: false,
                directed: false,
                renumber: true,
                construction_policy: "raw_libcugraph".into(),
                edge_normalize_inserted: true,
                calibration_source: "raw_input_edges".into(),
                seed_count: None,
                seed_mode: "none".into(),
                data_dependent: true,
                model_requested_bytes: 2048,
                requested_bytes: 4096,
                budget_bytes: Some(8192),
                graph_model_error_ratio: Some("2.000".into()),
                graph_statistics_exact_vertices: Some(12),
                graph_statistics_degree_max: Some(8),
                graph_statistics_degree_p99: Some(8),
                graph_statistics_sum_min_degree_oriented: Some(123),
                graph_statistics_oriented_edge_count: Some(45),
                graph_statistics_peak_chunk_min_degree_sum: Some(45),
                graph_statistics_sum_out_degree: Some(90),
                reason: "none".into(),
            }],
            graph_runtime_rows: vec![NativeGraphRuntimeMetricRow {
                algorithm: "triangle_count_all".into(),
                elapsed_ns: 1_000_000,
                graph_peak_bytes: Some(1024),
                ..Default::default()
            }],
            ..Default::default()
        });

        let tsv = row.to_tsv();
        let fields = tsv.split('\t').collect::<Vec<_>>();
        let field = |name: &str| fields[QUERY_COLUMNS.iter().position(|column| *column == name).unwrap()];
        assert_eq!(field("graph_raw_input_edges"), "45");
        assert_eq!(field("graph_vertex_width_bytes"), "8");
        assert_eq!(field("graph_weight_width_bytes"), "8");
        assert_eq!(field("graph_directed"), "false");
        assert_eq!(field("graph_edge_normalize_inserted"), "true");
        assert_eq!(field("graph_calibration_source"), "raw_input_edges");
        assert_eq!(field("graph_seed_mode"), "none");
        assert_eq!(field("graph_model_requested_bytes"), "2048");
        assert_eq!(field("graph_model_error_ratio"), "2.000");
        assert_eq!(field("graph_statistics_exact_vertices"), "12");
        assert_eq!(field("graph_statistics_degree_max"), "8");
        assert_eq!(field("graph_statistics_degree_p99"), "8");
        assert_eq!(field("graph_statistics_sum_min_degree_oriented"), "123");
        assert_eq!(field("graph_statistics_oriented_edge_count"), "45");
        assert_eq!(field("graph_statistics_peak_chunk_min_degree_sum"), "45");
        assert_eq!(field("graph_statistics_sum_out_degree"), "90");
        assert_eq!(field("graph_peak_bytes"), "1024");
    }

    #[test]
    fn workload_parsing_accepts_counts_and_bare_labels() {
        let parsed = parse_workload("pagerank:4, louvain:2 ,bfs").unwrap();
        assert_eq!(parsed, vec![("pagerank".into(), 4), ("louvain".into(), 2), ("bfs".into(), 1)]);
    }
}
