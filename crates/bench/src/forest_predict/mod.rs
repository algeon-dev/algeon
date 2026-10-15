//! Benchmarks `cuml_forest_predict` execution over EMBER2024 and TPC-H
//! `lineitem` forest workloads discovered by `workloads::forest`.
//!
//! Each test case executes a SQL statement with a designated model and Float32
//! feature inputs. Parse and bind timings measure `SessionContext::sql` and
//! physical planning with model metadata inspection. Operator stages
//! (import, pack, predict, assemble) derive from runtime observation snapshots;
//! total wall-clock time additionally covers input scans and result downloads.
//! Native scans read Parquet directly on device, whereas unlowered plans require
//! host decode, staging, and upload. Device memory metrics reflect peak
//! `nvidia-smi memory.used` during execution minus the pre-run baseline.
//! Because query domain memory pools retain allocations post-execution, running
//! individual test cases in dedicated processes ensures isolated memory metrics.
//!
//! The process holds `/tmp/cudf-gpu.lock` exclusively for its whole run. Each
//! case appends one row to `<output-dir>/forest_predict.tsv` and writes
//! `cases/<case>.json`, `samples/<case>.csv` (`id,prediction` for
//! `scripts/dev/check_forest_predict_samples.py`), and
//! `observations/<case>.jsonl` (the last run's sealed runtime observation).
//!
//! ```text
//! forest_predict [--cases name,...] [--repeat N] [--sample-rows N]
//!                [--output-dir DIR] [--model-root DIR]
//! ```
//!
//! `ALGEON_EMBER2024_DIR`, `ALGEON_FOREST_TPCH_PARQUET_DIR`, and
//! `ALGEON_FOREST_LINEITEM_MODEL` override the fixture locations.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write as _},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    cuml_sql::{CumlModelStoreConfig, CumlSqlConfig},
    native_exec::AlgeonNativeExecutionConfig,
    native_report::NativeDataFusionExecutionMetrics,
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Array, Float32Array, Int64Array, UInt64Array},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, metrics::MetricsSet},
    prelude::{SessionConfig, SessionContext},
};
use futures::TryStreamExt;
use query_runtime::observability::{
    RuntimeObservationPayload, RuntimeOperatorStageState, SealedRuntimeObservationSnapshot,
    write_runtime_observation_jsonl,
};
use serde::Serialize;
use workloads::{
    forest::{
        ALGEON_EMBER2024_DIR_ENV, ALGEON_FOREST_LINEITEM_MODEL_ENV, ALGEON_FOREST_TPCH_PARQUET_DIR_ENV,
        DEFAULT_LOCAL_EMBER2024_DIR, DEFAULT_LOCAL_FOREST_LINEITEM_MODEL, Ember2024DatasetMetadata,
        ForestModelMetadata, LINEITEM_FOREST_FEATURES, LineitemForestWorkload, describe_ember2024_dataset,
        describe_lineitem_forest_workload, ember2024_table_name, register_ember2024_tables,
    },
    tpch::{DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR, TpchTableLayout, register_tpch_table_layout},
};

type BenchResult<T> = Result<T, Box<dyn Error>>;

const GPU_LOCK_PATH: &str = "/tmp/cudf-gpu.lock";
/// Matches the stage 0 CPU baselines (`scripts/dev/bench_forest_stage0.py`,
/// LightGBM/XGBoost/GTIL at 16 threads); here it sizes the Tokio runtime and
/// DataFusion's CPU partitions.
pub const CPU_THREADS: usize = 16;
const EMBER_MODEL: &str = "EMBER2024_all";
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
/// Bound on the wait for the attempt owner to deliver the sealed observation
/// after the result stream ends.
const SEALED_OBSERVATION_TIMEOUT: Duration = Duration::from_secs(30);

const COLUMNS: &[&str] = &[
    "case",
    "model",
    "model_format",
    "precision",
    "features",
    "rows",
    "repeat",
    "parse_bind_ms",
    "execute_ms",
    "import_ms",
    "pack_ms",
    "predict_ms",
    "assemble_ms",
    "unattributed_ms",
    "predict_batches",
    "host_to_device_bytes",
    "query_grant_bytes",
    "device_high_water_mib",
    "rows_per_sec_execute",
    "failure",
];

#[derive(Debug, Clone, Copy)]
enum CaseInput {
    EmberSplit(&'static str),
    Lineitem,
}

#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    input: CaseInput,
    /// Row cap: `id < limit` on EMBER (whose `id` is the split row ordinal),
    /// `LIMIT` on lineitem. `None` scores the whole relation.
    limit: Option<u64>,
}

const CASES: &[Case] = &[
    Case { name: "ember_test_50k", input: CaseInput::EmberSplit("test"), limit: Some(50_000) },
    Case { name: "ember_test_100k", input: CaseInput::EmberSplit("test"), limit: Some(100_000) },
    Case { name: "ember_test_606k", input: CaseInput::EmberSplit("test"), limit: Some(606_000) },
    Case { name: "ember_test_1m", input: CaseInput::EmberSplit("test"), limit: Some(1_000_000) },
    Case { name: "ember_test_full", input: CaseInput::EmberSplit("test"), limit: None },
    Case { name: "ember_train_full", input: CaseInput::EmberSplit("train"), limit: None },
    Case { name: "lineitem_100k", input: CaseInput::Lineitem, limit: Some(100_000) },
    Case { name: "lineitem_1m", input: CaseInput::Lineitem, limit: Some(1_000_000) },
    Case { name: "lineitem_10m", input: CaseInput::Lineitem, limit: Some(10_000_000) },
    Case { name: "lineitem_full", input: CaseInput::Lineitem, limit: None },
];

struct Options {
    cases: Vec<Case>,
    repeat: usize,
    sample_rows: usize,
    output_dir: PathBuf,
    model_root: PathBuf,
}

fn parse_options() -> BenchResult<Options> {
    let mut cases = CASES.to_vec();
    let mut repeat = 3;
    let mut sample_rows = 20_000;
    let mut output_dir = PathBuf::from("target/forest_predict");
    let mut model_root = PathBuf::from(".");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} requires a value"));
        match flag.as_str() {
            "--cases" => {
                cases = value()?
                    .split(',')
                    .map(|name| {
                        CASES.iter().copied().find(|case| case.name == name).ok_or_else(|| {
                            let known = CASES.iter().map(|case| case.name).collect::<Vec<_>>().join(", ");
                            format!("unknown case {name}; known cases: {known}")
                        })
                    })
                    .collect::<Result<_, _>>()?;
            },
            "--repeat" => repeat = value()?.parse()?,
            "--sample-rows" => sample_rows = value()?.parse()?,
            "--output-dir" => output_dir = PathBuf::from(value()?),
            "--model-root" => model_root = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if repeat == 0 {
        return Err("--repeat must be positive".into());
    }
    Ok(Options { cases, repeat, sample_rows, output_dir, model_root })
}

pub async fn run() -> BenchResult<()> {
    let options = parse_options()?;
    let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(GPU_LOCK_PATH)?;
    lock.lock()?;
    for directory in ["cases", "samples", "observations"] {
        fs::create_dir_all(options.output_dir.join(directory))?;
    }
    let device_idle_mib = device_memory_used_mib();

    let needs_ember = options.cases.iter().any(|case| matches!(case.input, CaseInput::EmberSplit(_)));
    let needs_lineitem = options.cases.iter().any(|case| matches!(case.input, CaseInput::Lineitem));
    let ember = needs_ember
        .then(|| describe_ember2024_dataset(&env_path(ALGEON_EMBER2024_DIR_ENV, DEFAULT_LOCAL_EMBER2024_DIR)))
        .transpose()?;
    let lineitem = needs_lineitem
        .then(|| {
            describe_lineitem_forest_workload(
                &env_path(ALGEON_FOREST_TPCH_PARQUET_DIR_ENV, DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR),
                &env_path(ALGEON_FOREST_LINEITEM_MODEL_ENV, DEFAULT_LOCAL_FOREST_LINEITEM_MODEL),
            )
        })
        .transpose()?;

    let backend = AlgeonGpuBackend::builder()
        .with_execution_mode(AlgeonGpuExecutionMode::NativePreferred)
        .with_native_execution_config(AlgeonNativeExecutionConfig::default())
        .devices([AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)])
        .build()?;
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(session_config()).with_default_features())?
        .try_with_cuml_sql_config(CumlSqlConfig { model_store: Some(CumlModelStoreConfig::new(&options.model_root)) })?
        .build();
    let ctx = SessionContext::new_with_state(state);
    if let Some(ember) = &ember {
        register_ember2024_tables(&ctx, ember).await?;
    }
    if let Some(lineitem) = &lineitem {
        register_tpch_table_layout(&ctx, "lineitem", &TpchTableLayout::from_metadata(&lineitem.lineitem)).await?;
    }

    let environment = Environment::probe(options.repeat, device_idle_mib);
    println!("# {}", serde_json::to_string(&environment)?);
    println!("{}", COLUMNS.join("\t"));
    for case in &options.cases {
        let spec = CaseSpec::new(*case, ember.as_ref(), lineitem.as_ref(), &options.model_root)?;
        let report = run_case(&ctx, &backend, &spec, &options, device_idle_mib).await?;
        println!("{}", report.to_tsv());
        write_case_report(&options.output_dir, &environment, ember.as_ref(), lineitem.as_ref(), &report)?;
    }

    // The session holds a backend handle; release it before waiting for shutdown.
    drop(ctx);
    backend.close_and_wait_async(Duration::from_secs(60)).await?;
    Ok(())
}

/// The final native optimizer rule lowers a lowerable `input` subquery into a
/// GPU-resident producer that reads Parquet on the device. `CPU_THREADS`
/// partitions apply to an input that stays a DataFusion plan; DataFusion
/// decodes that input on the host and stages it before the upload.
fn session_config() -> SessionConfig {
    SessionConfig::new().with_target_partitions(CPU_THREADS)
}

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| default.to_string()))
}

struct CaseSpec {
    case: Case,
    model: ForestModelMetadata,
    features: usize,
    sql: String,
}

impl CaseSpec {
    fn new(
        case: Case,
        ember: Option<&Ember2024DatasetMetadata>,
        lineitem: Option<&LineitemForestWorkload>,
        model_root: &Path,
    ) -> BenchResult<Self> {
        let (model, features, input) = match case.input {
            CaseInput::EmberSplit(split) => {
                let ember = ember.ok_or("EMBER2024 fixture was not described")?;
                if ember.split(split).is_none() {
                    return Err(format!("EMBER2024 split {split} is not prepared").into());
                }
                let model = ember.model(EMBER_MODEL).ok_or_else(|| format!("{EMBER_MODEL} model is missing"))?;
                let filter = case.limit.map(|limit| format!(" WHERE id < {limit}")).unwrap_or_default();
                let input = format!(
                    "SELECT id, {} FROM {}{filter}",
                    ember.feature_columns().join(", "),
                    ember2024_table_name(split)
                );
                (model.clone(), ember.feature_count, input)
            },
            CaseInput::Lineitem => {
                let lineitem = lineitem.ok_or("lineitem forest workload was not described")?;
                let limit = case.limit.map(|limit| format!(" LIMIT {limit}")).unwrap_or_default();
                (
                    lineitem.model.clone(),
                    LINEITEM_FOREST_FEATURES.len(),
                    format!("{}{limit}", lineitem.input_sql("lineitem")),
                )
            },
        };
        let relative = relative_model_path(model_root, &model.path)?;
        let sql = format!(
            "SELECT row_ordinal, id, prediction FROM cuml_forest_predict(input => ({input}), model => '{relative}', model_format => '{}')",
            model.model_format
        );
        Ok(Self { case, model, features, sql })
    }
}

fn relative_model_path(root: &Path, model: &Path) -> BenchResult<String> {
    let root = fs::canonicalize(root)?;
    let model = fs::canonicalize(model)?;
    let relative = model
        .strip_prefix(&root)
        .map_err(|_| format!("model {} is outside --model-root {}", model.display(), root.display()))?;
    Ok(relative.to_str().ok_or("model path is not UTF-8")?.to_string())
}

#[derive(Debug, Clone, Default, Serialize)]
struct RunMeasurement {
    rows: u64,
    parse_bind_ms: f64,
    execute_ms: f64,
    import_ms: f64,
    pack_ms: f64,
    predict_ms: f64,
    assemble_ms: f64,
    predict_batches: usize,
    precision: Option<String>,
    /// Finished operator-stage time by stage kind, including non-ML stages.
    stage_ms: BTreeMap<String, f64>,
    host_to_device_bytes: usize,
    query_grant_bytes: Option<u64>,
    /// `nvidia-smi memory.used` high water minus the pre-backend idle reading.
    device_high_water_mib: Option<u64>,
}

impl RunMeasurement {
    fn unattributed_ms(&self) -> f64 {
        self.execute_ms - self.import_ms - self.pack_ms - self.predict_ms - self.assemble_ms
    }
}

#[derive(Debug, Serialize)]
struct CaseReport {
    case: &'static str,
    /// Registered relation the input subquery reads.
    input: String,
    model: String,
    model_path: PathBuf,
    model_format: &'static str,
    model_sha256: String,
    features: usize,
    row_limit: Option<u64>,
    sql_bytes: usize,
    sample_file: Option<String>,
    sample_rows: usize,
    /// Median of each field across the timed runs; the warm-up run is excluded.
    median: RunMeasurement,
    runs: Vec<RunMeasurement>,
    /// First error; the case's remaining runs are skipped.
    failure: Option<String>,
}

impl CaseReport {
    fn to_tsv(&self) -> String {
        let median = &self.median;
        let optional = |value: Option<u64>| value.map_or_else(|| "none".to_string(), |value| value.to_string());
        let rows_per_sec =
            if median.execute_ms > 0.0 { median.rows as f64 / (median.execute_ms / 1000.0) } else { 0.0 };
        [
            self.case.to_string(),
            self.model.clone(),
            self.model_format.to_string(),
            median.precision.clone().unwrap_or_else(|| "unknown".to_string()),
            self.features.to_string(),
            median.rows.to_string(),
            self.runs.len().to_string(),
            format!("{:.3}", median.parse_bind_ms),
            format!("{:.3}", median.execute_ms),
            format!("{:.3}", median.import_ms),
            format!("{:.3}", median.pack_ms),
            format!("{:.3}", median.predict_ms),
            format!("{:.3}", median.assemble_ms),
            format!("{:.3}", median.unattributed_ms()),
            median.predict_batches.to_string(),
            median.host_to_device_bytes.to_string(),
            optional(median.query_grant_bytes),
            optional(median.device_high_water_mib),
            format!("{rows_per_sec:.0}"),
            self.failure.clone().unwrap_or_else(|| "none".to_string()),
        ]
        .join("\t")
    }
}

async fn run_case(
    ctx: &SessionContext,
    backend: &AlgeonGpuBackend,
    spec: &CaseSpec,
    options: &Options,
    device_idle_mib: Option<u64>,
) -> BenchResult<CaseReport> {
    let mut runs = Vec::with_capacity(options.repeat);
    let mut last_batches = Vec::new();
    let mut last_snapshot = None;
    let mut failure = None;
    for iteration in 0..=options.repeat {
        let (measurement, batches, snapshot) = match run_once(ctx, backend, spec, device_idle_mib).await {
            Ok(run) => run,
            Err(error) => {
                eprintln!("[forest_predict] {} run {iteration} failed: {error}", spec.case.name);
                failure = Some(error.to_string().replace(['\t', '\n', '\r'], " "));
                break;
            },
        };
        eprintln!(
            "[forest_predict] {} run {iteration}{}: rows={} execute_ms={:.1}",
            spec.case.name,
            if iteration == 0 { " (warm-up)" } else { "" },
            measurement.rows,
            measurement.execute_ms
        );
        if iteration > 0 {
            runs.push(measurement);
        }
        last_batches = batches;
        last_snapshot = Some(snapshot);
    }
    if let Some(snapshot) = &last_snapshot {
        let mut out = BufWriter::new(File::create(
            options.output_dir.join("observations").join(format!("{}.jsonl", spec.case.name)),
        )?);
        write_runtime_observation_jsonl(snapshot, &mut out)?;
        out.flush()?;
    }
    let (sample_file, sample_rows) = if failure.is_none() {
        let sample_file = format!("samples/{}.csv", spec.case.name);
        let sample_rows = write_sample(&options.output_dir.join(&sample_file), &last_batches, options.sample_rows)?;
        (Some(sample_file), sample_rows)
    } else {
        (None, 0)
    };
    let input = match spec.case.input {
        CaseInput::EmberSplit(split) => ember2024_table_name(split),
        CaseInput::Lineitem => "lineitem".to_string(),
    };
    Ok(CaseReport {
        case: spec.case.name,
        input,
        model: spec.model.name.clone(),
        model_path: spec.model.path.clone(),
        model_format: spec.model.model_format,
        model_sha256: spec.model.sha256.clone(),
        features: spec.features,
        row_limit: spec.case.limit,
        sql_bytes: spec.sql.len(),
        sample_file,
        sample_rows,
        median: if failure.is_none() { median_measurement(&runs) } else { RunMeasurement::default() },
        runs,
        failure,
    })
}

/// Plans and executes the statement as one attempt and returns the attempt's
/// sealed observation, which its owner delivers after the report publishes.
async fn run_once(
    ctx: &SessionContext,
    backend: &AlgeonGpuBackend,
    spec: &CaseSpec,
    device_idle_mib: Option<u64>,
) -> BenchResult<(RunMeasurement, Vec<RecordBatch>, SealedRuntimeObservationSnapshot)> {
    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    let (sealed_sender, sealed_receiver) = mpsc::sync_channel(1);
    attempt.register_sealed_report_observer(move |_report, snapshot| {
        let _ = sealed_sender.try_send(snapshot);
    });
    let started = Instant::now();
    let plan = attempt.plan_sql(ctx, &spec.sql).await?;
    let parse_bind_ms = started.elapsed().as_secs_f64() * 1000.0;

    let sampler = DeviceMemorySampler::start();
    let started = Instant::now();
    let batches: Vec<RecordBatch> = attempt.execute_physical_plan(Arc::clone(&plan), ctx)?.try_collect().await?;
    let execute_ms = started.elapsed().as_secs_f64() * 1000.0;
    let device_peak_mib = sampler.finish();

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    let snapshot = sealed_receiver
        .recv_timeout(SEALED_OBSERVATION_TIMEOUT)
        .map_err(|_| "no sealed runtime observation was captured for the forest statement")?;
    let mut measurement = stage_measurement(&snapshot);
    measurement.rows = batches.iter().map(|batch| batch.num_rows() as u64).sum();
    measurement.parse_bind_ms = parse_bind_ms;
    measurement.execute_ms = execute_ms;
    measurement.host_to_device_bytes = metrics.host_to_device_uploaded_bytes;
    measurement.device_high_water_mib =
        device_peak_mib.zip(device_idle_mib).map(|(peak, idle)| peak.saturating_sub(idle));
    Ok((measurement, batches, snapshot))
}

/// Sums finished operator stages and takes the admitted grant. Import and
/// predict are `ml_native` stages told apart by their `forest_import:` /
/// `forest_predict:` strategy labels.
fn stage_measurement(snapshot: &SealedRuntimeObservationSnapshot) -> RunMeasurement {
    let mut measurement = RunMeasurement::default();
    let mut started = HashMap::new();
    for event in snapshot.events() {
        if let RuntimeObservationPayload::Admission(admission) = &event.payload {
            measurement.query_grant_bytes = measurement.query_grant_bytes.max(admission.grant_bytes);
        }
        let RuntimeObservationPayload::OperatorStage(stage) = &event.payload else {
            continue;
        };
        let Some(span) = event.span_id else {
            continue;
        };
        match stage.lifecycle {
            RuntimeOperatorStageState::Started => {
                started.insert(span, event.monotonic_ns);
            },
            RuntimeOperatorStageState::Finished => {
                let Some(start) = started.remove(&span) else {
                    continue;
                };
                let ms = event.monotonic_ns.saturating_sub(start) as f64 / 1e6;
                let kind = stage.stage_kind.as_str();
                *measurement.stage_ms.entry(kind.to_string()).or_default() += ms;
                let label = stage.strategy_label.as_deref().unwrap_or_default();
                match kind {
                    "ml_pack" => measurement.pack_ms += ms,
                    "ml_assemble" => measurement.assemble_ms += ms,
                    "ml_native" if label.starts_with("forest_import:") => {
                        measurement.import_ms += ms;
                        measurement.precision = label.rsplit(':').next().map(str::to_string);
                    },
                    "ml_native" if label.starts_with("forest_predict:") => {
                        measurement.predict_ms += ms;
                        measurement.predict_batches += 1;
                    },
                    _ => {},
                }
            },
            RuntimeOperatorStageState::Failed { .. } | RuntimeOperatorStageState::Cancelled => {
                started.remove(&span);
            },
        }
    }
    measurement
}

fn median_measurement(runs: &[RunMeasurement]) -> RunMeasurement {
    let median = |value: fn(&RunMeasurement) -> f64| {
        let mut values = runs.iter().map(value).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let first = &runs[0];
    RunMeasurement {
        rows: first.rows,
        parse_bind_ms: median(|run| run.parse_bind_ms),
        execute_ms: median(|run| run.execute_ms),
        import_ms: median(|run| run.import_ms),
        pack_ms: median(|run| run.pack_ms),
        predict_ms: median(|run| run.predict_ms),
        assemble_ms: median(|run| run.assemble_ms),
        predict_batches: first.predict_batches,
        precision: first.precision.clone(),
        stage_ms: first
            .stage_ms
            .keys()
            .map(|kind| {
                let mut values = runs.iter().filter_map(|run| run.stage_ms.get(kind).copied()).collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                (kind.clone(), values[values.len() / 2])
            })
            .collect(),
        host_to_device_bytes: first.host_to_device_bytes,
        query_grant_bytes: first.query_grant_bytes,
        device_high_water_mib: runs.iter().filter_map(|run| run.device_high_water_mib).max(),
    }
}

/// Writes `id,prediction` for every `stride`-th `row_ordinal` to sample across
/// all prediction batches.
fn write_sample(path: &Path, batches: &[RecordBatch], sample_rows: usize) -> BenchResult<usize> {
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let stride = (total / sample_rows.max(1)).max(1) as u64;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "id,prediction")?;
    let mut written = 0;
    for batch in batches {
        let ordinals = batch.column(0).as_any().downcast_ref::<UInt64Array>().ok_or("row_ordinal is not UInt64")?;
        let ids = batch.column(1).as_any().downcast_ref::<Int64Array>().ok_or("id is not Int64")?;
        let predictions = batch.column(2).as_any().downcast_ref::<Float32Array>().ok_or("prediction is not Float32")?;
        for row in 0..batch.num_rows() {
            if ordinals.value(row) % stride == 0 {
                writeln!(out, "{},{:e}", ids.value(row), predictions.value(row))?;
                written += 1;
            }
        }
    }
    out.flush()?;
    Ok(written)
}

fn plan_metrics_recursive(plan: &dyn ExecutionPlan) -> MetricsSet {
    fn append(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
        if let Some(plan_metrics) = plan.metrics() {
            for metric in plan_metrics.iter() {
                metrics.push(Arc::clone(metric));
            }
        }
        for child in plan.children() {
            append(metrics, child.as_ref());
        }
    }
    let mut metrics = MetricsSet::new();
    append(&mut metrics, plan);
    metrics
}

/// Polls `nvidia-smi` for device 0's `memory.used`. It sees every process on
/// the device, which the exclusive GPU lock limits to this one.
struct DeviceMemorySampler {
    stop: Arc<AtomicBool>,
    worker: thread::JoinHandle<Option<u64>>,
}

impl DeviceMemorySampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut peak = None;
            while !worker_stop.load(Ordering::Relaxed) {
                peak = peak.max(device_memory_used_mib());
                thread::sleep(MEMORY_SAMPLE_INTERVAL);
            }
            peak
        });
        Self { stop, worker }
    }

    fn finish(self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.join().ok().flatten().max(device_memory_used_mib())
    }
}

fn device_memory_used_mib() -> Option<u64> {
    nvidia_smi_query("memory.used")?.parse().ok()
}

fn nvidia_smi_query(field: &str) -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["-i", "0", &format!("--query-gpu={field}"), "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[derive(Debug, Serialize)]
struct Environment {
    unix_time_secs: u64,
    gpu_name: Option<String>,
    gpu_driver: Option<String>,
    gpu_memory_total_mib: Option<String>,
    device_idle_mib: Option<u64>,
    cpu_threads: usize,
    repeat: usize,
    warmup_runs: usize,
    ember_model: &'static str,
}

impl Environment {
    fn probe(repeat: usize, device_idle_mib: Option<u64>) -> Self {
        Self {
            unix_time_secs: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs()),
            gpu_name: nvidia_smi_query("name"),
            gpu_driver: nvidia_smi_query("driver_version"),
            gpu_memory_total_mib: nvidia_smi_query("memory.total"),
            device_idle_mib,
            cpu_threads: CPU_THREADS,
            repeat,
            warmup_runs: 1,
            ember_model: EMBER_MODEL,
        }
    }
}

fn write_case_report(
    output_dir: &Path,
    environment: &Environment,
    ember: Option<&Ember2024DatasetMetadata>,
    lineitem: Option<&LineitemForestWorkload>,
    report: &CaseReport,
) -> BenchResult<()> {
    let tsv_path = output_dir.join("forest_predict.tsv");
    let write_header = !tsv_path.exists();
    let mut tsv = BufWriter::new(OpenOptions::new().create(true).append(true).open(&tsv_path)?);
    if write_header {
        writeln!(tsv, "{}", COLUMNS.join("\t"))?;
    }
    writeln!(tsv, "{}", report.to_tsv())?;
    tsv.flush()?;
    let dataset = match report.input.as_str() {
        "lineitem" => lineitem.map(|lineitem| {
            serde_json::json!({
                "tpch_root": lineitem.tpch_root,
                "rows": lineitem.lineitem.row_count,
                "bytes": lineitem.lineitem.byte_size,
            })
        }),
        _ => ember.map(|ember| serde_json::json!(ember.to_neutral_metadata())),
    };
    let document = serde_json::json!({ "environment": environment, "dataset": dataset, "case": report });
    fs::write(output_dir.join("cases").join(format!("{}.json", report.case)), serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}
