//! Explicit ClickBench operator-stage probe (not a Cargo test target).
//!
//! Plans selected ClickBench queries through the normal native optimizer,
//! executes the lowered native IR under a full runtime observation session, and
//! periodically exports live snapshots from a watchdog while `collect`/execute
//! is in flight. A deadline requests cooperative cancellation and the machine-
//! readable verdict records the last active operator stage.

use std::{
    collections::BTreeSet,
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use datafusion::{
    physical_plan::{ExecutionPlan, collect},
    prelude::SessionContext,
};
use datafusion_nexus::{
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig},
    planner::NexusNativeOptimizerConfig,
    policy::GpuFallbackPolicy,
    session::NexusSessionStateBuilderExt,
};
use datafusion_nexus_bench::report_workload::clickbench::{
    clickbench_native_profile_settings_from_env, clickbench_report_session_config,
};
use datafusion_nexus_workloads::clickbench::{
    CLICKBENCH_QUERIES, ClickbenchWorkloadError, discover_clickbench_layout, prepare_clickbench_report_dataset,
    register_clickbench_table,
};
use nexus_query_engine::{
    exec::{execute_query_plan_with_runtime_observation, output::QueryOutput},
    mode::NativePipelineExecutionMode,
    observability::{
        RuntimeObservationConfig, RuntimeObservationPayload, RuntimeObservationSession, RuntimeOperatorStageState,
        RuntimeQueryId, write_runtime_observation_jsonl,
    },
    plan::QueryPlan,
    runtime::{ExecutionCancellation, ExecutionContextBuilder, MemoryControlPolicy},
};
use serde::Serialize;

const DEFAULT_QUERY_IDS: &[&str] = &["q28", "q29", "q34", "q35"];
const DEFAULT_DEADLINE_SECS: u64 = 120;
const DEFAULT_SNAPSHOT_INTERVAL_MS: u64 = 500;
const LIVE_RECORDS: u64 = 65_536;
const LIVE_BYTES: u64 = 8 * 1024 * 1024;

type ToolResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
struct ProbeConfig {
    query_ids: Vec<String>,
    artifact_dir: PathBuf,
    deadline: Duration,
    snapshot_interval: Duration,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    fallback_policy: GpuFallbackPolicy,
    mode: ProbeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    /// Full DataFusion collect path (uses adapter observation via native exec).
    DatafusionCollect,
    /// Direct native IR execution with an owned observation session + cancel.
    NativeObserved,
}

#[derive(Debug, Serialize)]
struct ProbeVerdict {
    schema: &'static str,
    version: u32,
    query_id: String,
    profile: String,
    mode: String,
    completed: bool,
    cancelled: bool,
    cancellation_drained: bool,
    elapsed_ms: u64,
    output_rows: Option<u64>,
    output_fingerprint: Option<String>,
    error: Option<String>,
    last_active_stage: Option<String>,
    last_active_stage_state: Option<String>,
    stages_started: Vec<String>,
    stages_finished: Vec<String>,
    live_snapshot_count: u64,
    sealed_jsonl: Option<String>,
    live_dir: String,
    peak_vram_bytes_sample: Option<u64>,
    reservation_live_peak_bytes: Option<u64>,
}

pub async fn run(args: Vec<String>) -> ToolResult<()> {
    if args.iter().any(|arg| matches!(arg.as_str(), "-h" | "--help")) {
        print_usage();
        return Ok(());
    }
    let config = parse_args(args)?;
    fs::create_dir_all(&config.artifact_dir)?;
    let profile = clickbench_native_profile_settings_from_env();
    let (root, _metadata) = prepare_clickbench_report_dataset()?.ok_or_else(|| {
        ClickbenchWorkloadError::InvalidDataset(
            "no ClickBench fixture found; set NEXUS_PARQUET_DIR or stage fixture/clickbench/data".to_string(),
        )
    })?;

    let mut verdicts = Vec::new();
    for query_id in &config.query_ids {
        let verdict = match config.mode {
            ProbeMode::NativeObserved => probe_native_observed(query_id, &root, &profile, &config).await?,
            ProbeMode::DatafusionCollect => probe_datafusion_collect(query_id, &root, &profile, &config).await?,
        };
        let verdict_path = config.artifact_dir.join(format!("{query_id}.verdict.json"));
        fs::write(&verdict_path, serde_json::to_vec_pretty(&verdict)?)?;
        eprintln!(
            "[clickbench-stage-probe] query={} completed={} cancelled={} last_stage={} elapsed_ms={}",
            verdict.query_id,
            verdict.completed,
            verdict.cancelled,
            verdict.last_active_stage.as_deref().unwrap_or("none"),
            verdict.elapsed_ms
        );
        verdicts.push(verdict);
    }

    let table_path = config.artifact_dir.join("verdict_table.json");
    fs::write(&table_path, serde_json::to_vec_pretty(&verdicts)?)?;
    print_tsv_summary(&verdicts)?;
    Ok(())
}

async fn probe_native_observed(
    query_id: &str,
    root: &Path,
    profile: &datafusion_nexus_bench::report_workload::clickbench::ClickbenchNativeProfileSettings,
    config: &ProbeConfig,
) -> ToolResult<ProbeVerdict> {
    let sql = clickbench_sql(query_id)?;
    let physical = plan_native_physical(query_id, root, sql, profile, config).await?;
    let native_plan = peel_native_query_plan(physical.as_ref())
        .ok_or_else(|| format!("query {query_id} did not lower to NexusNativeExec/NexusNativeComposedExec"))?;

    let query_dir = config.artifact_dir.join(query_id);
    let live_dir = query_dir.join("live");
    fs::create_dir_all(&live_dir)?;

    let session = RuntimeObservationSession::new(
        RuntimeQueryId::new(query_id),
        None,
        RuntimeObservationConfig::bounded(LIVE_RECORDS, LIVE_BYTES),
    );
    let handle = session.handle();
    let cancellation = ExecutionCancellation::new();
    let stop = Arc::new(AtomicBool::new(false));
    let live_count = Arc::new(Mutex::new(0u64));
    let watchdog = spawn_watchdog(
        handle.clone(),
        live_dir.clone(),
        config.snapshot_interval,
        Arc::clone(&stop),
        Arc::clone(&live_count),
        Some(cancellation.clone()),
        config.deadline,
    );

    let started = Instant::now();
    let runtime = cudf::Runtime::new().map_err(|error| format!("cudf runtime: {error}"))?;
    let observed = ExecutionContextBuilder::new()
        .with_runtime(runtime)
        .with_query_id(query_id)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: profile.device_budget_bytes,
            max_source_chunk_bytes: profile.max_source_chunk_bytes,
            max_row_groups_per_chunk: profile.max_row_groups_per_chunk,
            max_retry_attempts: profile.max_retry_attempts,
            min_retry_chunk_row_groups: profile.min_retry_chunk_row_groups,
        })
        .with_source_chunk_planning_mode(profile.source_chunk_planning_mode)
        .with_native_pipeline_execution_mode(config.native_pipeline_execution_mode)
        .with_cancellation(cancellation.clone())
        .with_runtime_observation_session(session)
        .build_with_runtime_observation()
        .map_err(|error| format!("build observed context: {}", error.query_error()))?;
    let (context, session) = observed.into_parts();

    let outcome = execute_query_plan_with_runtime_observation(&context, &native_plan, session);
    stop.store(true, Ordering::Release);
    let _ = watchdog.join();
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let (result, metrics, sealed) = outcome.into_parts();
    let cancelled = cancellation.is_cancelled();
    let mut completed = false;
    let mut output_rows = None;
    let mut output_fingerprint = None;
    let mut error = None;
    let cancellation_drained = match &result {
        Ok(QueryOutput::HostArrow(batch)) => {
            completed = true;
            output_rows = Some(batch.num_rows() as u64);
            output_fingerprint = Some(fingerprint_batch_rows(batch.num_rows(), batch.num_columns()));
            cancelled
        },
        Ok(QueryOutput::GpuResident(handle)) => {
            completed = true;
            output_rows = Some(handle.num_rows() as u64);
            output_fingerprint = Some(format!("gpu_rows:{}", handle.num_rows()));
            cancelled
        },
        Err(err) => {
            let code = err.code().as_str().to_string();
            error = Some(format!("{code}: {err}"));
            cancelled && code == "cancelled"
        },
    };

    let sealed_jsonl = if let Some(snapshot) = sealed.as_ref() {
        let path = query_dir.join("sealed.jsonl");
        let mut file = fs::File::create(&path)?;
        write_runtime_observation_jsonl(snapshot, &mut file)?;
        Some(path.display().to_string())
    } else {
        None
    };

    let stage_summary = summarize_stages(sealed.as_ref(), &live_dir);
    Ok(ProbeVerdict {
        schema: "datafusion_nexus.clickbench_stage_probe.v1",
        version: 1,
        query_id: query_id.to_string(),
        profile: profile.profile_name.to_string(),
        mode: "native_observed".to_string(),
        completed,
        cancelled,
        cancellation_drained,
        elapsed_ms,
        output_rows,
        output_fingerprint,
        error,
        last_active_stage: stage_summary.last_active_stage,
        last_active_stage_state: stage_summary.last_active_stage_state,
        stages_started: stage_summary.stages_started,
        stages_finished: stage_summary.stages_finished,
        live_snapshot_count: *live_count.lock().unwrap_or_else(|error| error.into_inner()),
        sealed_jsonl,
        live_dir: live_dir.display().to_string(),
        peak_vram_bytes_sample: None,
        reservation_live_peak_bytes: Some(metrics.reservation_live_peak_bytes),
    })
}

async fn probe_datafusion_collect(
    query_id: &str,
    root: &Path,
    profile: &datafusion_nexus_bench::report_workload::clickbench::ClickbenchNativeProfileSettings,
    config: &ProbeConfig,
) -> ToolResult<ProbeVerdict> {
    let sql = clickbench_sql(query_id)?;
    let physical = plan_native_physical(query_id, root, sql, profile, config).await?;
    let query_dir = config.artifact_dir.join(query_id);
    let live_dir = query_dir.join("live");
    fs::create_dir_all(&live_dir)?;

    // Adapter path seals only after completion; watchdog still writes "progress"
    // markers so wall time is captured even without live Started samples.
    let stop = Arc::new(AtomicBool::new(false));
    let live_count = Arc::new(Mutex::new(0u64));
    let progress_dir = live_dir.clone();
    let stop_w = Arc::clone(&stop);
    let count_w = Arc::clone(&live_count);
    let interval = config.snapshot_interval;
    let deadline = config.deadline;
    let watchdog = thread::spawn(move || {
        let origin = Instant::now();
        let mut index = 0u64;
        while !stop_w.load(Ordering::Acquire) {
            if origin.elapsed() >= deadline {
                break;
            }
            let marker = progress_dir.join(format!("progress_{index:04}.txt"));
            let _ = fs::write(marker, format!("elapsed_ms={}\n", origin.elapsed().as_millis()));
            index += 1;
            *count_w.lock().unwrap_or_else(|error| error.into_inner()) = index;
            thread::sleep(interval);
        }
    });

    let started = Instant::now();
    let ctx = SessionContext::new();
    let result = collect(physical, ctx.task_ctx()).await;
    stop.store(true, Ordering::Release);
    let _ = watchdog.join();
    let elapsed_ms = started.elapsed().as_millis() as u64;

    let mut completed = false;
    let mut output_rows = None;
    let mut output_fingerprint = None;
    let mut error = None;
    match result {
        Ok(batches) => {
            completed = true;
            let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
            output_rows = Some(rows as u64);
            output_fingerprint =
                Some(fingerprint_batch_rows(rows, batches.first().map(|b| b.num_columns()).unwrap_or(0)));
        },
        Err(err) => error = Some(err.to_string()),
    }

    Ok(ProbeVerdict {
        schema: "datafusion_nexus.clickbench_stage_probe.v1",
        version: 1,
        query_id: query_id.to_string(),
        profile: profile.profile_name.to_string(),
        mode: "datafusion_collect".to_string(),
        completed,
        cancelled: false,
        cancellation_drained: false,
        elapsed_ms,
        output_rows,
        output_fingerprint,
        error,
        last_active_stage: None,
        last_active_stage_state: None,
        stages_started: Vec::new(),
        stages_finished: Vec::new(),
        live_snapshot_count: *live_count.lock().unwrap_or_else(|error| error.into_inner()),
        sealed_jsonl: None,
        live_dir: live_dir.display().to_string(),
        peak_vram_bytes_sample: None,
        reservation_live_peak_bytes: None,
    })
}

async fn plan_native_physical(
    query_id: &str,
    root: &Path,
    sql: &str,
    profile: &datafusion_nexus_bench::report_workload::clickbench::ClickbenchNativeProfileSettings,
    config: &ProbeConfig,
) -> ToolResult<Arc<dyn ExecutionPlan>> {
    let layout = discover_clickbench_layout(root)?;
    let optimizer_config = NexusNativeOptimizerConfig::new(config.fallback_policy)
        .with_native_pipeline_execution_mode(config.native_pipeline_execution_mode)
        .with_native_execution_config(
            NexusNativeExecutionConfig::bounded(
                profile.device_budget_bytes,
                profile.max_source_chunk_bytes,
                profile.max_row_groups_per_chunk,
                profile.max_retry_attempts,
                profile.min_retry_chunk_row_groups,
            )
            .with_source_chunk_planning_mode(profile.source_chunk_planning_mode)
            .with_query_id(query_id)
            .with_native_pipeline_execution_mode(config.native_pipeline_execution_mode),
        );

    let state = datafusion::execution::SessionStateBuilder::new()
        .with_config(clickbench_report_session_config())
        .with_default_features()
        .with_nexus_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_clickbench_table(&ctx, &layout).await?;
    let framed = format!("-- query_id={query_id}\n{sql}");
    let df = ctx.sql(&framed).await?;
    Ok(df.create_physical_plan().await?)
}

fn peel_native_query_plan(plan: &dyn ExecutionPlan) -> Option<QueryPlan> {
    if let Some(native) = plan.downcast_ref::<NexusNativeExec>() {
        return Some(native.native_plan().clone());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(native.native_plan().clone());
    }
    for child in plan.children() {
        if let Some(found) = peel_native_query_plan(child.as_ref()) {
            return Some(found);
        }
    }
    None
}

fn spawn_watchdog(
    handle: nexus_query_engine::observability::RuntimeObservationHandle,
    live_dir: PathBuf,
    interval: Duration,
    stop: Arc<AtomicBool>,
    live_count: Arc<Mutex<u64>>,
    cancellation: Option<ExecutionCancellation>,
    deadline: Duration,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let origin = Instant::now();
        let mut index = 0u64;
        let mut cancel_requested = false;
        while !stop.load(Ordering::Acquire) {
            if let Some(snapshot) = handle.live_snapshot() {
                let path = live_dir.join(format!("live_{index:04}.jsonl"));
                if let Ok(mut file) = fs::File::create(&path) {
                    let _ = write_runtime_observation_jsonl_live(&snapshot, &mut file);
                }
                index = index.saturating_add(1);
                *live_count.lock().unwrap_or_else(|error| error.into_inner()) = index;
            }
            if !cancel_requested && origin.elapsed() >= deadline {
                if let Some(cancellation) = &cancellation {
                    cancellation.cancel();
                    cancel_requested = true;
                    let _ = fs::write(live_dir.join("cancellation_requested.txt"), "1\n");
                } else {
                    break;
                }
            }
            thread::sleep(interval);
        }
    })
}

fn write_runtime_observation_jsonl_live(
    snapshot: &nexus_query_engine::observability::LiveRuntimeObservationSnapshot,
    out: &mut dyn Write,
) -> io::Result<()> {
    // Live snapshots share the sealed exporter shape except footer is incomplete.
    // Serialize manually as JSON lines via the public sealed writer when possible;
    // fall back to event-only export.
    writeln!(
        out,
        "{}",
        serde_json::json!({
            "record_type": "live_marker",
            "schema": nexus_query_engine::observability::RUNTIME_OBSERVATION_JSONL_SCHEMA,
            "event_count": snapshot.events().len(),
        })
    )?;
    for event in snapshot.events() {
        writeln!(out, "{}", serde_json::to_string(event).map_err(io::Error::other)?)?;
    }
    Ok(())
}

struct StageSummary {
    last_active_stage: Option<String>,
    last_active_stage_state: Option<String>,
    stages_started: Vec<String>,
    stages_finished: Vec<String>,
}

fn absorb_stage(
    started: &mut BTreeSet<String>,
    finished: &mut BTreeSet<String>,
    last_active: &mut Option<String>,
    last_state: &mut Option<String>,
    stage_kind: String,
    state: String,
) {
    match state.as_str() {
        "started" => {
            started.insert(stage_kind.clone());
            *last_active = Some(stage_kind);
            *last_state = Some(state);
        },
        "finished" => {
            finished.insert(stage_kind.clone());
            if last_active.as_deref() == Some(stage_kind.as_str()) {
                *last_state = Some(state);
            }
        },
        _ => {
            *last_active = Some(stage_kind);
            *last_state = Some(state);
        },
    }
}

fn summarize_stages(
    sealed: Option<&nexus_query_engine::observability::SealedRuntimeObservationSnapshot>,
    live_dir: &Path,
) -> StageSummary {
    let mut started = BTreeSet::new();
    let mut finished = BTreeSet::new();
    let mut last_active = None;
    let mut last_state = None;

    if let Some(snapshot) = sealed {
        for event in snapshot.events() {
            if let RuntimeObservationPayload::OperatorStage(stage) = &event.payload {
                let kind = stage.stage_kind.as_str().to_string();
                let state = match &stage.lifecycle {
                    RuntimeOperatorStageState::Started => "started",
                    RuntimeOperatorStageState::Finished => "finished",
                    RuntimeOperatorStageState::Failed { .. } => "failed",
                    RuntimeOperatorStageState::Cancelled => "cancelled",
                }
                .to_string();
                absorb_stage(&mut started, &mut finished, &mut last_active, &mut last_state, kind, state);
            }
        }
    }

    // Prefer the latest live fragment if sealed is empty/truncated.
    if last_active.is_none()
        && let Ok(entries) = fs::read_dir(live_dir)
    {
        let mut paths: Vec<_> = entries.filter_map(|entry| entry.ok().map(|e| e.path())).collect();
        paths.sort();
        if let Some(last) = paths.last()
            && let Ok(text) = fs::read_to_string(last)
        {
            for line in text.lines().rev() {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line)
                    && let Some(payload) = value.get("payload")
                    && payload.get("kind").and_then(|v| v.as_str()) == Some("operator_stage")
                {
                    let kind = payload.get("stage_kind").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    let state = payload
                        .get("lifecycle")
                        .and_then(|v| v.get("state"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    absorb_stage(&mut started, &mut finished, &mut last_active, &mut last_state, kind, state);
                    break;
                }
            }
        }
    }

    // If Finished is the latest sealed view but some Started stages never
    // finished, prefer an open stage as the residual long pole.
    if last_state.as_deref() == Some("finished")
        && let Some(open) = started.difference(&finished).next().cloned()
    {
        last_active = Some(open);
        last_state = Some("started".to_string());
    }

    StageSummary {
        last_active_stage: last_active,
        last_active_stage_state: last_state,
        stages_started: started.into_iter().collect(),
        stages_finished: finished.into_iter().collect(),
    }
}

fn clickbench_sql(query_id: &str) -> ToolResult<&'static str> {
    CLICKBENCH_QUERIES
        .iter()
        .find(|query| query.id == query_id)
        .map(|query| query.sql)
        .ok_or_else(|| format!("unknown ClickBench query id: {query_id}").into())
}

fn fingerprint_batch_rows(rows: usize, cols: usize) -> String {
    format!("rows={rows};cols={cols}")
}

fn print_tsv_summary(verdicts: &[ProbeVerdict]) -> ToolResult<()> {
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "query_id\tcompleted\tcancelled\telapsed_ms\tlast_active_stage\tlast_active_stage_state\toutput_rows\terror"
    )?;
    for verdict in verdicts {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            verdict.query_id,
            verdict.completed,
            verdict.cancelled,
            verdict.elapsed_ms,
            verdict.last_active_stage.as_deref().unwrap_or(""),
            verdict.last_active_stage_state.as_deref().unwrap_or(""),
            verdict.output_rows.map(|v| v.to_string()).unwrap_or_default(),
            verdict.error.as_deref().unwrap_or("").replace('\t', " "),
        )?;
    }
    Ok(())
}

fn parse_args(args: Vec<String>) -> ToolResult<ProbeConfig> {
    let mut query_ids = Vec::new();
    let mut artifact_dir = None;
    let mut deadline_secs = DEFAULT_DEADLINE_SECS;
    let mut snapshot_interval_ms = DEFAULT_SNAPSHOT_INTERVAL_MS;
    let mut mode = ProbeMode::NativeObserved;
    let mut fallback_policy = GpuFallbackPolicy::GpuOnly;
    let mut pipeline_mode = native_pipeline_execution_mode_from_env();

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--query" | "--queries" => {
                let value = iter.next().ok_or("--query requires a value")?;
                for part in value.split(',') {
                    let part = part.trim();
                    if !part.is_empty() {
                        query_ids.push(part.to_string());
                    }
                }
            },
            "--artifact-dir" => {
                artifact_dir = Some(PathBuf::from(iter.next().ok_or("--artifact-dir requires a value")?));
            },
            "--deadline-secs" => {
                deadline_secs = iter
                    .next()
                    .ok_or("--deadline-secs requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --deadline-secs: {error}"))?;
            },
            "--snapshot-interval-ms" => {
                snapshot_interval_ms = iter
                    .next()
                    .ok_or("--snapshot-interval-ms requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --snapshot-interval-ms: {error}"))?;
            },
            "--mode" => {
                let value = iter.next().ok_or("--mode requires a value")?;
                mode = match value.as_str() {
                    "native_observed" | "native" => ProbeMode::NativeObserved,
                    "datafusion_collect" | "df" => ProbeMode::DatafusionCollect,
                    other => return Err(format!("unsupported --mode {other}").into()),
                };
            },
            "--fallback-policy" => {
                let value = iter.next().ok_or("--fallback-policy requires a value")?;
                fallback_policy = match value.as_str() {
                    "gpu_only" => GpuFallbackPolicy::GpuOnly,
                    "gpu_preferred" => GpuFallbackPolicy::GpuPreferred,
                    other => return Err(format!("unsupported fallback policy {other}").into()),
                };
            },
            "--pipeline-mode" => {
                let value = iter.next().ok_or("--pipeline-mode requires a value")?;
                pipeline_mode = parse_pipeline_mode(&value)?;
            },
            other => return Err(format!("unknown clickbench-stage-probe argument: {other}").into()),
        }
    }

    if query_ids.is_empty() {
        query_ids = DEFAULT_QUERY_IDS.iter().map(|id| (*id).to_string()).collect();
    }
    let artifact_dir = artifact_dir.unwrap_or_else(default_artifact_dir);
    Ok(ProbeConfig {
        query_ids,
        artifact_dir,
        deadline: Duration::from_secs(deadline_secs),
        snapshot_interval: Duration::from_millis(snapshot_interval_ms),
        native_pipeline_execution_mode: pipeline_mode,
        fallback_policy,
        mode,
    })
}

fn default_artifact_dir() -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    PathBuf::from(format!("target/clickbench_hang_remediation/phase2/probe_{stamp}"))
}

fn native_pipeline_execution_mode_from_env() -> NativePipelineExecutionMode {
    env::var("NEXUS_NATIVE_PIPELINE_EXECUTION_MODE")
        .ok()
        .and_then(|value| parse_pipeline_mode(value.trim()).ok())
        .unwrap_or(NativePipelineExecutionMode::StrictBounded)
}

fn parse_pipeline_mode(value: &str) -> ToolResult<NativePipelineExecutionMode> {
    match value {
        "strict_bounded" | "strict" => Ok(NativePipelineExecutionMode::StrictBounded),
        "adaptive_bounded" | "adaptive" => Ok(NativePipelineExecutionMode::Adaptive),
        other => Err(format!("unsupported pipeline mode {other}").into()),
    }
}

fn print_usage() {
    eprintln!(
        "clickbench-stage-probe\n\
         \n\
         USAGE:\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features clickbench-stage-probe -- \\\n\
             clickbench-stage-probe [--query q28,q29,q34,q35] [--artifact-dir DIR] \\\n\
             [--deadline-secs 120] [--snapshot-interval-ms 500] [--mode native_observed|datafusion_collect]\n\
         \n\
         Environment:\n\
           NEXUS_PARQUET_DIR, NEXUS_CLICKBENCH_NATIVE_PROFILE, NEXUS_NATIVE_PIPELINE_EXECUTION_MODE\n\
           and other NEXUS_CLICKBENCH_NATIVE_* knobs are honored via the shared ClickBench profile."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let config = parse_args(vec![]).expect("defaults");
        assert_eq!(config.query_ids, ["q28", "q29", "q34", "q35"]);
        assert_eq!(config.mode, ProbeMode::NativeObserved);
    }

    #[test]
    fn parse_query_list() {
        let config = parse_args(vec!["--query".into(), "q29,q34".into()]).expect("parse");
        assert_eq!(config.query_ids, ["q29", "q34"]);
    }
}
