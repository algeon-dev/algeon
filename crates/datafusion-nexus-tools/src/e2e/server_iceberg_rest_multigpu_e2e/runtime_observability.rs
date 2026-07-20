use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::record_batch::RecordBatch;
use arrow_flight::{
    FlightData, FlightInfo, Ticket, decode::FlightRecordBatchStream, error::FlightError,
    flight_service_client::FlightServiceClient, sql::client::FlightSqlServiceClient,
};
use futures::TryStreamExt;
use prost::Message as _;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    sync::{Barrier, mpsc},
    task::JoinSet,
};
use tonic::{Code, Request, transport::Channel};

use super::{
    E2eResult,
    flight::{
        CANCEL_RESULT_CANCELLING, action_frames, cancel_query, get_admission_stats, get_query_stats,
        query_id_from_statement_flight_info,
    },
    server::{ProcessExit, ServerChild, ServerSpawnConfig},
    validation::{parse_u64_field, tsv_section_single_row, wait_for_idle},
};

const CONTROL_SCHEMA: &str = "datafusion_nexus.query_runtime_artifact.control.v1";
const JSONL_SCHEMA: &str = "nexus_query_engine.runtime_observation.v1";
const QUERY_ID_METADATA: &str = "x-datafusion-nexus-query-id";
const CORRELATION_HEADER: &str = "x-datafusion-nexus-correlation-id";
const DEFAULT_FIXTURE_ROOT: &str = "/opt/dlami/nvme/fixture/data";
const DEFAULT_SERVER_BINARY: &str = "target/release/datafusion_nexus_server";
const DEFAULT_EVIDENCE_PATH: &str = "target/runtime_observability_e2e/evidence.json";
const DEFAULT_SERVER_LOG_PATH: &str = "target/runtime_observability_e2e/server";
const EVIDENCE_SCHEMA: &str = "datafusion_nexus.server_runtime_observability_e2e.v2";
const EVIDENCE_VERSION: u32 = 2;
const TABLE_NAME: &str = "runtime_observability_lineitem";
const ORDERS_TABLE_NAME: &str = "runtime_observability_orders";
const FLIGHT_MAX_ENCODED_FRAME_BYTES: usize = 4 * 1024 * 1024;
const BROAD_PROJECTION_ROWS: usize = 262_144;
// Each dedicated probe may own at most 2 * 64 KiB, so the 192 KiB global cap
// admits every artifact individually but requires eviction across the set.
const GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES: u64 = 64 * 1024;
const GLOBAL_BYTE_PRESSURE_RETAINED_BYTES: u64 = 192 * 1024;
const GLOBAL_BYTE_PRESSURE_RETAINED_COUNT: usize = 64;
const GLOBAL_BYTE_PRESSURE_TTL_SECS: u64 = 60 * 60;
const GLOBAL_BYTE_PRESSURE_MAX_QUERIES: usize = 16;
const GLOBAL_BYTE_PRESSURE_SETUP_QUERIES: usize = 2;

const OBS_RECORDS_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS";
const OBS_QUERY_BYTES_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES";
const OBS_RETAINED_COUNT_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT";
const OBS_RETAINED_BYTES_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES";
const OBS_TTL_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS";
const OBS_GRACE_ENV: &str = "NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS";

const FATAL_PATTERNS: &[&str] = &[
    "panicked at",
    "sigsegv",
    "sigabrt",
    "segmentation fault",
    "core dumped",
    "cuda_error_illegal_address",
    "illegal memory access",
    "an illegal memory access",
    "unspecified launch failure",
    "launch failed",
    "device lost",
    " xid ",
    "out of memory",
    "outofmemory",
    "allocation failed",
    "fatal runtime",
    "poisoned",
];

pub(super) async fn run(args: Vec<String>) -> E2eResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }
    let params = RuntimeParams::parse(args)?;
    params.validate_filesystem()?;
    ensure_parent(&params.evidence_path)?;
    ensure_parent(&params.server_log_path)?;

    let started = Instant::now();
    let mut evidence = RunEvidence::new(&params);
    let run_result = run_matrix(&params, &mut evidence).await;
    let fatal_result = scan_fatal_patterns(&evidence.server_logs);
    evidence.elapsed_ms = millis(started.elapsed());
    evidence.fatal_patterns = fatal_result.evidence;
    if let Err(error) = &run_result {
        evidence.status = "failed".to_string();
        evidence.failure = Some(error.to_string());
    } else if !evidence.fatal_patterns.matches.is_empty() {
        evidence.status = "failed".to_string();
        evidence.failure = Some("fatal server log pattern detected".to_string());
    } else {
        evidence.status = "passed".to_string();
    }
    fs::write(&params.evidence_path, serde_json::to_vec_pretty(&evidence)?)?;

    run_result?;
    if !evidence.fatal_patterns.matches.is_empty() {
        return Err(format!("fatal server log patterns: {:?}", evidence.fatal_patterns.matches).into());
    }
    println!("server_runtime_observability_e2e ok: {}", params.evidence_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!(
        "server-runtime-observability-e2e\n\
         \n\
         OPTIONS:\n\
           --fixture-root PATH          Fixture root (default: {DEFAULT_FIXTURE_ROOT})\n\
           --catalog-path PATH          Directory containing lineitem.parquet and orders.parquet\n\
           --server-binary PATH         Built server executable\n\
           --device ORDINAL             CUDA device ordinal (default: 0)\n\
           --concurrency N              Overlapping queries and per-device limit (default: 4)\n\
           --waves N                    Native success waves (default: 2)\n\
           --restart-iterations N       Hard termination/restart iterations (default: 1)\n\
           --profile matrix|native|cpu_fallback\n\
           --bounded-mode strict_bounded|adaptive\n\
           --evidence-json PATH         Machine-readable evidence output\n\
           --server-log PATH            Prefix for per-process stdout/stderr logs\n\
           --connect-timeout-secs N\n\
           --query-timeout-secs N\n\
           --poll-timeout-secs N\n\
           --shutdown-timeout-secs N\n\
           --per-query-records N\n\
           --per-query-bytes N\n\
           --retained-count N\n\
           --retained-bytes N\n\
           --retention-ttl-secs N\n\
           --shutdown-grace-secs N\n\
           --profile-once               One short overlapping native wave, then exit"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileSelection {
    Matrix,
    Native,
    CpuFallback,
}

impl ProfileSelection {
    fn parse(raw: &str) -> E2eResult<Self> {
        match raw {
            "matrix" => Ok(Self::Matrix),
            "native" => Ok(Self::Native),
            "cpu_fallback" | "cpu-fallback" => Ok(Self::CpuFallback),
            _ => Err(format!("--profile must be matrix, native, or cpu_fallback; got {raw:?}").into()),
        }
    }

    fn runs_native(self) -> bool {
        matches!(self, Self::Matrix | Self::Native)
    }

    fn runs_cpu(self) -> bool {
        matches!(self, Self::Matrix | Self::CpuFallback)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::Native => "native",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

#[derive(Debug, Clone)]
struct ObservabilityLimits {
    per_query_records: u64,
    per_query_bytes: u64,
    retained_count: usize,
    retained_bytes: u64,
    ttl_secs: u64,
    shutdown_grace_secs: u64,
}

#[derive(Debug, Clone)]
struct RuntimeParams {
    fixture_root: PathBuf,
    catalog_path: PathBuf,
    server_binary: PathBuf,
    device: usize,
    concurrency: usize,
    waves: usize,
    restart_iterations: usize,
    profile: ProfileSelection,
    bounded_mode: String,
    evidence_path: PathBuf,
    server_log_path: PathBuf,
    connect_timeout: Duration,
    query_timeout: Duration,
    poll_timeout: Duration,
    shutdown_timeout: Duration,
    profile_once: bool,
    observability: ObservabilityLimits,
}

impl RuntimeParams {
    fn parse(args: Vec<String>) -> E2eResult<Self> {
        let mut fixture_root = PathBuf::from(DEFAULT_FIXTURE_ROOT);
        let mut catalog_path = None;
        let mut server_binary = PathBuf::from(DEFAULT_SERVER_BINARY);
        let mut device = 0usize;
        let mut concurrency = 4usize;
        let mut waves = 2usize;
        let mut restart_iterations = 1usize;
        let mut profile = ProfileSelection::Matrix;
        let mut bounded_mode = "strict_bounded".to_string();
        let mut evidence_path = PathBuf::from(DEFAULT_EVIDENCE_PATH);
        let mut server_log_path = PathBuf::from(DEFAULT_SERVER_LOG_PATH);
        let mut connect_timeout = Duration::from_secs(60);
        let mut query_timeout = Duration::from_secs(180);
        let mut poll_timeout = Duration::from_secs(30);
        let mut shutdown_timeout = Duration::from_secs(30);
        let mut profile_once = false;
        let mut per_query_records = None;
        let mut per_query_bytes = None;
        let mut retained_count = None;
        let mut retained_bytes = None;
        let mut ttl_secs = None;
        let mut shutdown_grace_secs = None;
        let mut seen = BTreeSet::new();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            if !seen.insert(flag.to_string()) {
                return Err(format!("duplicate option {flag}").into());
            }
            if flag == "--profile-once" {
                profile_once = true;
                index += 1;
                continue;
            }
            let value = args.get(index + 1).ok_or_else(|| format!("missing value for {flag}"))?;
            match flag {
                "--fixture-root" => fixture_root = PathBuf::from(value),
                "--catalog-path" => catalog_path = Some(PathBuf::from(value)),
                "--server-binary" => server_binary = PathBuf::from(value),
                "--device" => device = parse_usize(flag, value)?,
                "--concurrency" => concurrency = parse_usize(flag, value)?,
                "--waves" => waves = parse_usize(flag, value)?,
                "--restart-iterations" => restart_iterations = parse_usize(flag, value)?,
                "--profile" => profile = ProfileSelection::parse(value)?,
                "--bounded-mode" if matches!(value.as_str(), "strict_bounded" | "adaptive") => {
                    bounded_mode.clone_from(value)
                },
                "--bounded-mode" => {
                    return Err(format!("--bounded-mode must be strict_bounded or adaptive; got {value:?}").into());
                },
                "--evidence-json" => evidence_path = PathBuf::from(value),
                "--server-log" => server_log_path = PathBuf::from(value),
                "--connect-timeout-secs" => connect_timeout = positive_duration(flag, value)?,
                "--query-timeout-secs" => query_timeout = positive_duration(flag, value)?,
                "--poll-timeout-secs" => poll_timeout = positive_duration(flag, value)?,
                "--shutdown-timeout-secs" => shutdown_timeout = positive_duration(flag, value)?,
                "--per-query-records" => per_query_records = Some(positive_u64(flag, value)?),
                "--per-query-bytes" => per_query_bytes = Some(positive_u64(flag, value)?),
                "--retained-count" => retained_count = Some(positive_usize(flag, value)?),
                "--retained-bytes" => retained_bytes = Some(positive_u64(flag, value)?),
                "--retention-ttl-secs" => ttl_secs = Some(positive_u64(flag, value)?),
                "--shutdown-grace-secs" => shutdown_grace_secs = Some(positive_u64(flag, value)?),
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        let catalog_path = catalog_path.unwrap_or_else(|| fixture_root.join("tpch/sf1/parquet"));
        let observability = ObservabilityLimits {
            per_query_records: cli_or_env_u64(per_query_records, OBS_RECORDS_ENV, 256)?,
            per_query_bytes: cli_or_env_u64(per_query_bytes, OBS_QUERY_BYTES_ENV, 65_536)?,
            retained_count: cli_or_env_usize(retained_count, OBS_RETAINED_COUNT_ENV, 4)?,
            retained_bytes: cli_or_env_u64(retained_bytes, OBS_RETAINED_BYTES_ENV, 1_048_576)?,
            ttl_secs: cli_or_env_u64(ttl_secs, OBS_TTL_ENV, 2)?,
            shutdown_grace_secs: cli_or_env_u64(shutdown_grace_secs, OBS_GRACE_ENV, 2)?,
        };
        if !(2..=64).contains(&concurrency) {
            return Err("--concurrency must be between 2 and 64 so overlap can be proven".into());
        }
        if !(1..=100).contains(&waves) {
            return Err("--waves must be between 1 and 100".into());
        }
        if restart_iterations > 20 {
            return Err("--restart-iterations must be between 0 and 20".into());
        }
        if observability.retained_count < concurrency {
            return Err(format!(
                "--retained-count ({}) must be at least --concurrency ({concurrency})",
                observability.retained_count
            )
            .into());
        }
        if observability.retained_bytes < observability.per_query_bytes {
            return Err("--retained-bytes must be at least --per-query-bytes".into());
        }
        if profile.runs_native() && !profile_once && observability.retained_bytes > 2 * 1024 * 1024 {
            return Err(
                "--retained-bytes must not exceed 2 MiB for the explicit serialized-byte pressure scenario".into()
            );
        }
        if profile_once && !profile.runs_native() {
            return Err("--profile-once requires --profile native or matrix".into());
        }
        if evidence_path == server_log_path {
            return Err("--evidence-json and --server-log must be different paths".into());
        }
        Ok(Self {
            fixture_root,
            catalog_path,
            server_binary,
            device,
            concurrency,
            waves,
            restart_iterations,
            profile,
            bounded_mode,
            evidence_path,
            server_log_path,
            connect_timeout,
            query_timeout,
            poll_timeout,
            shutdown_timeout,
            profile_once,
            observability,
        })
    }

    fn validate_filesystem(&self) -> E2eResult<()> {
        if !self.fixture_root.is_dir() {
            return Err(format!("fixture root does not exist: {}", self.fixture_root.display()).into());
        }
        if !self.server_binary.is_file() {
            return Err(format!("server binary does not exist: {}", self.server_binary.display()).into());
        }
        let lineitem = self.lineitem_path();
        if !lineitem.is_file() {
            return Err(format!("catalog path must contain lineitem.parquet: {}", lineitem.display()).into());
        }
        let orders = self.orders_path();
        if !orders.is_file() {
            return Err(format!("catalog path must contain orders.parquet: {}", orders.display()).into());
        }
        Ok(())
    }

    fn lineitem_path(&self) -> PathBuf {
        if self.catalog_path.extension().is_some_and(|extension| extension == "parquet") {
            self.catalog_path.clone()
        } else {
            self.catalog_path.join("lineitem.parquet")
        }
    }

    fn orders_path(&self) -> PathBuf {
        if self.catalog_path.extension().is_some_and(|extension| extension == "parquet") {
            self.catalog_path.parent().unwrap_or_else(|| Path::new("")).join("orders.parquet")
        } else {
            self.catalog_path.join("orders.parquet")
        }
    }
}

fn parse_usize(flag: &str, raw: &str) -> E2eResult<usize> {
    raw.parse::<usize>().map_err(|error| format!("invalid {flag}={raw:?}: {error}").into())
}

fn positive_usize(flag: &str, raw: &str) -> E2eResult<usize> {
    let value = parse_usize(flag, raw)?;
    if value == 0 { Err(format!("{flag} must be greater than zero").into()) } else { Ok(value) }
}

fn positive_u64(flag: &str, raw: &str) -> E2eResult<u64> {
    let value = raw.parse::<u64>().map_err(|error| format!("invalid {flag}={raw:?}: {error}"))?;
    if value == 0 { Err(format!("{flag} must be greater than zero").into()) } else { Ok(value) }
}

fn positive_duration(flag: &str, raw: &str) -> E2eResult<Duration> {
    Ok(Duration::from_secs(positive_u64(flag, raw)?))
}

fn env_positive_u64(name: &str) -> E2eResult<Option<u64>> {
    std::env::var(name).ok().map(|raw| positive_u64(name, &raw)).transpose()
}

fn env_positive_usize(name: &str) -> E2eResult<Option<usize>> {
    std::env::var(name).ok().map(|raw| positive_usize(name, &raw)).transpose()
}

fn cli_or_env_u64(cli: Option<u64>, name: &str, default: u64) -> E2eResult<u64> {
    match cli {
        Some(value) => Ok(value),
        None => Ok(env_positive_u64(name)?.unwrap_or(default)),
    }
}

fn cli_or_env_usize(cli: Option<usize>, name: &str, default: usize) -> E2eResult<usize> {
    match cli {
        Some(value) => Ok(value),
        None => Ok(env_positive_usize(name)?.unwrap_or(default)),
    }
}

#[derive(Debug, Serialize)]
struct RunEvidence {
    schema: &'static str,
    version: u32,
    status: String,
    failure: Option<String>,
    started_unix_ms: u128,
    elapsed_ms: u64,
    hardware: Value,
    config: Value,
    scenarios: Vec<ScenarioEvidence>,
    waves: Vec<WaveEvidence>,
    retention: Option<RetentionEvidence>,
    restarts: Vec<RestartEvidence>,
    processes: Vec<ProcessEvidence>,
    graph: Value,
    server_logs: Vec<PathBuf>,
    fatal_patterns: FatalPatternEvidence,
}

impl RunEvidence {
    fn new(params: &RuntimeParams) -> Self {
        Self {
            schema: EVIDENCE_SCHEMA,
            version: EVIDENCE_VERSION,
            status: "running".to_string(),
            failure: None,
            started_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |value| value.as_millis()),
            elapsed_ms: 0,
            hardware: hardware_evidence(),
            config: json!({
                "fixture_root": params.fixture_root,
                "catalog_path": params.catalog_path,
                "lineitem_path": params.lineitem_path(),
                "orders_path": params.orders_path(),
                "server_binary": params.server_binary,
                "device": params.device,
                "concurrency": params.concurrency,
                "waves": params.waves,
                "restart_iterations": params.restart_iterations,
                "profile": params.profile.label(),
                "bounded_mode": params.bounded_mode,
                "profile_once": params.profile_once,
                "timeouts_secs": {
                    "connect": params.connect_timeout.as_secs(),
                    "query": params.query_timeout.as_secs(),
                    "poll": params.poll_timeout.as_secs(),
                    "shutdown": params.shutdown_timeout.as_secs(),
                },
                "observability": {
                    "per_query_records": params.observability.per_query_records,
                    "per_query_bytes": params.observability.per_query_bytes,
                    "retained_count": params.observability.retained_count,
                    "retained_bytes": params.observability.retained_bytes,
                    "ttl_secs": params.observability.ttl_secs,
                    "shutdown_grace_secs": params.observability.shutdown_grace_secs,
                },
            }),
            scenarios: Vec::new(),
            waves: Vec::new(),
            retention: None,
            restarts: Vec::new(),
            processes: Vec::new(),
            graph: json!({
                "covered": false,
                "reason": "not requested by this local Parquet matrix; use the explicit Flight graph smoke against a graph-enabled server"
            }),
            server_logs: Vec::new(),
            fatal_patterns: FatalPatternEvidence::default(),
        }
    }
}

#[derive(Debug, Serialize)]
struct WaveEvidence {
    wave: usize,
    started: usize,
    completed: usize,
    passed: usize,
    failed: usize,
    elapsed_ms: u64,
    throughput_queries_per_second: f64,
    peak_active_queries: u64,
    peak_active_query_jobs: u64,
    overlap_proven: bool,
}

#[derive(Debug, Serialize)]
struct ScenarioEvidence {
    scenario: String,
    profile: String,
    query_id: String,
    correlation_id: String,
    outcome: String,
    output_batches: usize,
    output_rows: usize,
    encoded_frame_count: Option<usize>,
    record_batch_chunks: Option<usize>,
    max_encoded_frame_bytes: Option<usize>,
    stats_assertions: Vec<String>,
    artifact: ArtifactEvidence,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactEvidence {
    state: String,
    reason: Option<String>,
    frame_count: usize,
    wire_bytes: usize,
    jsonl_records: usize,
    topology_records: usize,
    timeline_events: usize,
    plan_execution_ids: Vec<String>,
    terminal_kind: Option<String>,
    retained_records: Option<u64>,
    retained_bytes: Option<u64>,
    truncated: Option<bool>,
    degraded: Option<bool>,
    poison_recoveries: Option<u64>,
    dropped_by_class: Option<Value>,
    dropped_by_reason: Option<Value>,
}

#[derive(Debug, Serialize)]
struct RetentionEvidence {
    generated_query_ids: Vec<String>,
    live_query_id: String,
    live_state_during_pressure: String,
    sealed_count: usize,
    unavailable: BTreeMap<String, String>,
    not_found: Vec<String>,
    generated_wire_bytes: usize,
    capture_budget: CaptureBudgetEvidence,
    global_byte_pressure: Option<GlobalBytePressureEvidence>,
    ttl_query_id: String,
    ttl_terminal_state: String,
    ttl_unavailable_elapsed_ms: u64,
    ttl_not_found_elapsed_ms: u64,
    no_silent_loss: bool,
}

#[derive(Debug, Serialize)]
struct CaptureBudgetEvidence {
    query_id: String,
    input_bytes: usize,
    per_query_capture_limit: u64,
    state: String,
    reason: String,
    frame_count: usize,
    stats_incomplete: bool,
    retained_byte_loss_recorded: bool,
}

#[derive(Debug, Serialize)]
struct GlobalBytePressureEvidence {
    server_label: String,
    per_query_bytes: u64,
    retained_count_limit: usize,
    retained_byte_limit: u64,
    ttl_secs: u64,
    generated_query_ids: Vec<String>,
    artifact_charge_bytes: BTreeMap<String, u64>,
    sealed_query_ids_after_pressure: Vec<String>,
    evicted: BTreeMap<String, String>,
    generated_charge_bytes: u64,
    max_observed_generated_retained_bytes: u64,
    retained_bytes_after_pressure: u64,
    setup_artifacts_known_evicted: bool,
    exact_retained_set_proven: bool,
    count_limit_not_reached: bool,
    byte_limit_respected: bool,
    no_silent_loss: bool,
}

#[derive(Debug, Serialize)]
struct RestartEvidence {
    iteration: usize,
    termination: String,
    in_flight_query_id: Option<String>,
    in_flight_query_ids: Vec<String>,
    client_terminal_results: Vec<String>,
    exit: ProcessEvidence,
    recovery_query_id: String,
    recovery_artifact_state: String,
    recovery_idle: bool,
}

#[derive(Debug, Serialize)]
struct ProcessEvidence {
    label: String,
    pid: Option<u32>,
    profile: String,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
    exit_success: bool,
    exit_code: Option<i32>,
    exit_signal: Option<i32>,
    exit_display: String,
    elapsed_ms: u64,
}

#[derive(Debug, Default, Serialize)]
struct FatalPatternEvidence {
    scanned_files: usize,
    matches: Vec<Value>,
}

struct FatalScanResult {
    evidence: FatalPatternEvidence,
}

#[derive(Debug, Clone, Copy)]
enum ChildProfile {
    Native,
    CpuFallback,
}

impl ChildProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

struct RunningServer {
    label: String,
    profile: ChildProfile,
    pid: Option<u32>,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
    started: Instant,
    child: ServerChild,
    channel: Channel,
}

async fn run_matrix(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    if params.profile.runs_native() {
        run_native_matrix(params, evidence).await?;
    }
    if params.profile.runs_cpu() && !params.profile_once {
        run_cpu_phase(params, evidence).await?;
    }
    Ok(())
}

async fn run_native_matrix(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    let mut server = spawn_server(params, evidence, "native-initial", ChildProfile::Native).await?;
    setup_local_table(params, &server.channel, "native-initial").await?;

    let effective_waves = if params.profile_once { 1 } else { params.waves };
    for wave in 0..effective_waves {
        let (mut rows, wave_evidence) = run_success_wave(params, &server.channel, wave).await?;
        evidence.scenarios.append(&mut rows);
        evidence.waves.push(wave_evidence);
    }
    if !params.profile_once {
        evidence.scenarios.push(run_native_join(params, &server.channel).await?);
        evidence.scenarios.push(run_variable_width_projection(params, &server.channel).await?);
    }
    wait_for_idle_with_timeout(params, &server.channel).await?;

    if params.profile_once {
        let process = terminate_server(params, &mut server, "TERM", true).await?;
        evidence.processes.push(process);
        return Ok(());
    }

    evidence.scenarios.append(&mut run_structured_failures(params, &server.channel).await?);
    evidence.scenarios.push(run_cancel_query(params, &server.channel).await?);
    evidence.scenarios.push(run_drop_before_first_poll(params, &server.channel).await?);
    evidence.scenarios.push(run_midstream_drop(params, &server.channel).await?);
    evidence.retention = Some(run_retention_pressure(params, &server.channel, evidence).await?);
    wait_for_idle_with_timeout(params, &server.channel).await?;

    let (graceful_exit, graceful_query_ids, client_terminal_results) =
        terminate_gracefully_with_in_flight(params, &mut server).await?;
    evidence.processes.push(graceful_exit);

    let mut recovery = spawn_server(params, evidence, "native-graceful-restart", ChildProfile::Native).await?;
    setup_local_table(params, &recovery.channel, "native-graceful-restart").await?;
    let recovery_row = run_success_query(params, &recovery.channel, "graceful_restart_recovery", 0).await?;
    wait_for_idle_with_timeout(params, &recovery.channel).await?;
    evidence.restarts.push(RestartEvidence {
        iteration: 0,
        termination: "graceful_sigterm".to_string(),
        in_flight_query_id: None,
        in_flight_query_ids: graceful_query_ids,
        client_terminal_results,
        exit: evidence.processes.last().expect("graceful process evidence").clone_for_restart(),
        recovery_query_id: recovery_row.query_id.clone(),
        recovery_artifact_state: recovery_row.artifact.state.clone(),
        recovery_idle: true,
    });
    evidence.scenarios.push(recovery_row);

    for iteration in 1..=params.restart_iterations {
        let (query_id, stream) = start_in_flight_projection(params, &recovery.channel, iteration).await?;
        let active_frames =
            require_action_frames(&recovery.channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let active = parse_artifact(&query_id, None, &active_frames, None, None)?;
        if active.state != "active" {
            return Err(format!(
                "hard restart iteration {iteration} query {query_id} must be active before SIGKILL, got {}",
                active.state
            )
            .into());
        }
        let hard_exit = terminate_server(params, &mut recovery, "KILL", false).await?;
        drop(stream);
        if hard_exit.exit_signal != Some(9) {
            return Err(format!(
                "hard restart iteration {iteration} expected signal 9, got code={:?} signal={:?}",
                hard_exit.exit_code, hard_exit.exit_signal
            )
            .into());
        }
        evidence.processes.push(hard_exit);

        let label = format!("native-hard-restart-{iteration}");
        recovery = spawn_server(params, evidence, &label, ChildProfile::Native).await?;
        setup_local_table(params, &recovery.channel, &label).await?;
        let recovery_row = run_success_query(params, &recovery.channel, "hard_restart_recovery", iteration).await?;
        wait_for_idle_with_timeout(params, &recovery.channel).await?;
        evidence.restarts.push(RestartEvidence {
            iteration,
            termination: "hard_sigkill".to_string(),
            in_flight_query_id: Some(query_id.clone()),
            in_flight_query_ids: vec![query_id],
            client_terminal_results: vec!["stream_dropped_after_sigkill".to_string()],
            exit: evidence.processes.last().expect("hard process evidence").clone_for_restart(),
            recovery_query_id: recovery_row.query_id.clone(),
            recovery_artifact_state: recovery_row.artifact.state.clone(),
            recovery_idle: true,
        });
        evidence.scenarios.push(recovery_row);
    }

    let final_exit = terminate_server(params, &mut recovery, "TERM", true).await?;
    evidence.processes.push(final_exit);

    let global_byte_pressure = run_global_retained_byte_pressure_phase(params, evidence).await?;
    evidence
        .retention
        .as_mut()
        .expect("native retention evidence is installed before the dedicated byte-pressure phase")
        .global_byte_pressure = Some(global_byte_pressure);
    Ok(())
}

async fn run_cpu_phase(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    let mut server = spawn_server(params, evidence, "cpu-fallback", ChildProfile::CpuFallback).await?;
    setup_local_table(params, &server.channel, "cpu-fallback").await?;
    evidence.scenarios.push(run_cpu_fallback(params, &server.channel).await?);
    wait_for_idle_with_timeout(params, &server.channel).await?;
    let process = terminate_server(params, &mut server, "TERM", true).await?;
    evidence.processes.push(process);
    Ok(())
}

async fn spawn_server(
    params: &RuntimeParams,
    evidence: &mut RunEvidence,
    label: &str,
    profile: ChildProfile,
) -> E2eResult<RunningServer> {
    let addr = reserve_loopback_addr()?;
    let stdout_log = suffixed_log_path(&params.server_log_path, label, "stdout");
    let stderr_log = suffixed_log_path(&params.server_log_path, label, "stderr");
    let mut env = BTreeMap::from([
        ("TOKIO_WORKER_THREADS".to_string(), "1".to_string()),
        ("NEXUS_SERVER_BIND".to_string(), addr.to_string()),
        ("NEXUS_SERVER_LOG".to_string(), "info".to_string()),
        ("NEXUS_SERVER_GPU_DEVICES".to_string(), params.device.to_string()),
        ("NEXUS_SERVER_MAX_CONCURRENT_QUERIES".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY".to_string(), "least_loaded".to_string()),
        ("NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS".to_string(), params.query_timeout.as_millis().to_string()),
        ("NEXUS_SERVER_SESSION_PROFILE".to_string(), profile.label().to_string()),
        ("NEXUS_SERVER_NATIVE_MEMORY_POLICY".to_string(), "bounded".to_string()),
        ("NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE".to_string(), params.bounded_mode.clone()),
        ("NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS".to_string(), params.query_timeout.as_millis().to_string()),
        ("NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS".to_string(), params.poll_timeout.as_secs().to_string()),
        (OBS_RECORDS_ENV.to_string(), params.observability.per_query_records.to_string()),
        (OBS_QUERY_BYTES_ENV.to_string(), params.observability.per_query_bytes.to_string()),
        (OBS_RETAINED_COUNT_ENV.to_string(), params.observability.retained_count.to_string()),
        (OBS_RETAINED_BYTES_ENV.to_string(), params.observability.retained_bytes.to_string()),
        (OBS_TTL_ENV.to_string(), params.observability.ttl_secs.to_string()),
        (OBS_GRACE_ENV.to_string(), params.observability.shutdown_grace_secs.to_string()),
    ]);
    if matches!(profile, ChildProfile::CpuFallback) {
        env.insert("DATAFUSION_EXECUTION_TARGET_PARTITIONS".to_string(), "1".to_string());
    }
    let started = Instant::now();
    let mut child = ServerChild::spawn_config(ServerSpawnConfig {
        server_binary: params.server_binary.clone(),
        stdout_path: stdout_log.clone(),
        stderr_path: stderr_log.clone(),
        env,
    })?;
    let pid = child.process_id();
    evidence.server_logs.push(stdout_log.clone());
    evidence.server_logs.push(stderr_log.clone());
    let url = format!("http://{addr}");
    let channel = child.connect_with_timeout(&url, params.connect_timeout).await?;
    child.require_running(label)?;
    Ok(RunningServer { label: label.to_string(), profile, pid, stdout_log, stderr_log, started, child, channel })
}

async fn terminate_server(
    params: &RuntimeParams,
    server: &mut RunningServer,
    signal: &str,
    require_success: bool,
) -> E2eResult<ProcessEvidence> {
    let exit = server.child.terminate(signal, params.shutdown_timeout).await?;
    if require_success && !exit.success {
        return Err(format!("{} did not exit successfully after SIG{signal}: {}", server.label, exit.display).into());
    }
    Ok(process_evidence(server, exit))
}

fn process_evidence(server: &RunningServer, exit: ProcessExit) -> ProcessEvidence {
    ProcessEvidence {
        label: server.label.clone(),
        pid: server.pid,
        profile: server.profile.label().to_string(),
        stdout_log: server.stdout_log.clone(),
        stderr_log: server.stderr_log.clone(),
        exit_success: exit.success,
        exit_code: exit.code,
        exit_signal: exit.signal,
        exit_display: exit.display,
        elapsed_ms: millis(server.started.elapsed()),
    }
}

impl ProcessEvidence {
    fn clone_for_restart(&self) -> Self {
        Self {
            label: self.label.clone(),
            pid: self.pid,
            profile: self.profile.clone(),
            stdout_log: self.stdout_log.clone(),
            stderr_log: self.stderr_log.clone(),
            exit_success: self.exit_success,
            exit_code: self.exit_code,
            exit_signal: self.exit_signal,
            exit_display: self.exit_display.clone(),
            elapsed_ms: self.elapsed_ms,
        }
    }
}

fn reserve_loopback_addr() -> E2eResult<std::net::SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn suffixed_log_path(base: &Path, label: &str, stream: &str) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let name = base.file_name().and_then(|name| name.to_str()).unwrap_or("server");
    parent.join(format!("{name}.{label}.{stream}.log"))
}

async fn setup_local_table(params: &RuntimeParams, channel: &Channel, label: &str) -> E2eResult<()> {
    let mut client = correlated_client(channel, &format!("setup-{label}"));
    for (table, path) in [(TABLE_NAME, params.lineitem_path()), (ORDERS_TABLE_NAME, params.orders_path())] {
        let sql = format!("CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{}'", sql_path(&path)?);
        tokio::time::timeout(params.query_timeout, client.execute_update(sql, None))
            .await
            .map_err(|_| format!("{label} table setup exceeded {} seconds", params.query_timeout.as_secs()))??;
    }
    Ok(())
}

fn sql_path(path: &Path) -> E2eResult<String> {
    let raw = path.to_str().ok_or_else(|| format!("fixture path is not UTF-8: {}", path.display()))?;
    Ok(raw.replace('\'', "''"))
}

fn native_sql() -> String {
    format!(
        "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM {TABLE_NAME} \
         WHERE l_shipdate >= DATE '1994-01-01' \
         AND l_shipdate < DATE '1995-01-01' \
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
    )
}

fn cancellation_projection_sql() -> String {
    format!("SELECT l_orderkey, l_partkey FROM {TABLE_NAME}")
}

fn broad_projection_sql() -> String {
    format!(
        "SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, \
         l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, \
         l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment \
         FROM {TABLE_NAME} LIMIT {BROAD_PROJECTION_ROWS}"
    )
}

fn native_join_sql() -> String {
    format!(
        "SELECT COUNT(*) AS joined_rows FROM {TABLE_NAME} l \
         JOIN {ORDERS_TABLE_NAME} o ON l.l_orderkey = o.o_orderkey \
         WHERE l.l_orderkey = 1"
    )
}

fn correlated_client(channel: &Channel, correlation_id: &str) -> FlightSqlServiceClient<Channel> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    client.set_header(CORRELATION_HEADER, correlation_id);
    client
}

async fn run_success_wave(
    params: &RuntimeParams,
    channel: &Channel,
    wave: usize,
) -> E2eResult<(Vec<ScenarioEvidence>, WaveEvidence)> {
    let started = Instant::now();
    let barrier = Arc::new(Barrier::new(params.concurrency + 1));
    let mut tasks = JoinSet::new();
    for index in 0..params.concurrency {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let params = params.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            run_success_query(&params, &channel, &format!("native_wave_{wave}"), index).await
        });
    }
    barrier.wait().await;
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut peak_active_queries = 0;
    let mut peak_active_query_jobs = 0;
    while !tasks.is_empty() {
        if let Ok(Ok(payload)) = tokio::time::timeout(Duration::from_millis(250), get_admission_stats(channel)).await {
            let (active_queries, active_query_jobs, reported_peak) = admission_activity(&payload)?;
            peak_active_queries = peak_active_queries.max(active_queries);
            peak_active_query_jobs = peak_active_query_jobs.max(active_query_jobs).max(reported_peak);
        }
        while let Some(result) = tasks.try_join_next() {
            match result {
                Ok(Ok(row)) => rows.push(row),
                Ok(Err(error)) => failures.push(error.to_string()),
                Err(error) => failures.push(format!("task join failure: {error}")),
            }
        }
        if !tasks.is_empty() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
    rows.sort_by(|left, right| left.correlation_id.cmp(&right.correlation_id));
    let elapsed = started.elapsed();
    let wave_evidence = WaveEvidence {
        wave,
        started: params.concurrency,
        completed: rows.len() + failures.len(),
        passed: rows.len(),
        failed: failures.len(),
        elapsed_ms: millis(elapsed),
        throughput_queries_per_second: rows.len() as f64 / elapsed.as_secs_f64().max(0.001),
        peak_active_queries,
        peak_active_query_jobs,
        overlap_proven: peak_active_queries >= 2 || peak_active_query_jobs >= 2,
    };
    if !failures.is_empty() {
        return Err(format!("native wave {wave} failures: {failures:?}").into());
    }
    if !wave_evidence.overlap_proven {
        return Err(format!(
            "native wave {wave} did not prove overlap: active_queries={} active_query_jobs={}",
            wave_evidence.peak_active_queries, wave_evidence.peak_active_query_jobs
        )
        .into());
    }
    Ok((rows, wave_evidence))
}

async fn run_success_query(
    params: &RuntimeParams,
    channel: &Channel,
    scenario: &str,
    index: usize,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = format!("runtime-e2e-{scenario}-{index}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(native_sql(), None))
        .await
        .map_err(|_| format!("{scenario} GetFlightInfo timed out"))??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| format!("{scenario} DoGet timed out"))??;
    let batches = &collected.batches;
    if batches.iter().map(RecordBatch::num_rows).sum::<usize>() != 1 {
        return Err(format!("{scenario} native aggregate did not return exactly one row").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if artifact.plan_execution_ids.is_empty() {
        return Err(format!("native query {query_id} artifact has no plan_execution_id").into());
    }
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        profile: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: batches.len(),
        output_rows: 1,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "query_id".to_string(),
            "correlation_id".to_string(),
            "native_path".to_string(),
            "runtime_succeeded".to_string(),
            "no_multiple_sessions".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

async fn run_native_join(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-native-join".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(native_join_sql(), None))
        .await
        .map_err(|_| "native join GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "native join DoGet timed out")??;
    let batches = &collected.batches;
    if batches.len() != 1 || batches[0].num_rows() != 1 || batches[0].num_columns() != 1 {
        return Err(format!("native join expected one scalar result, got {batches:?}").into());
    }
    let value = arrow::util::display::array_value_to_string(batches[0].column(0), 0)?;
    if value != "6" {
        return Err(format!("native join expected TPCH SF1 order 1 to have 6 lineitems, got {value:?}").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if artifact.plan_execution_ids.is_empty() {
        return Err(format!("native join query {query_id} artifact has no plan_execution_id").into());
    }
    Ok(ScenarioEvidence {
        scenario: "native_join".to_string(),
        profile: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: 1,
        output_rows: 1,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "exact_join_count=6".to_string(),
            "native_path".to_string(),
            "runtime_succeeded".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

async fn run_variable_width_projection(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-variable-width-projection".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "variable-width projection GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "variable-width projection DoGet timed out")??;
    let output_rows = collected.batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if output_rows != BROAD_PROJECTION_ROWS || collected.batches.iter().any(|batch| batch.num_columns() != 16) {
        return Err(format!(
            "variable-width projection returned {output_rows} rows across {} columns, expected {BROAD_PROJECTION_ROWS} rows across 16 columns",
            collected.batches.first().map_or(0, RecordBatch::num_columns)
        )
        .into());
    }
    let frame_evidence = collected.frame_evidence();
    validate_variable_frame_evidence(frame_evidence)?;

    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: "variable_width_projection".to_string(),
        profile: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: collected.batches.len(),
        output_rows,
        encoded_frame_count: Some(frame_evidence.encoded_frame_count),
        record_batch_chunks: Some(frame_evidence.record_batch_chunks),
        max_encoded_frame_bytes: Some(frame_evidence.max_encoded_frame_bytes),
        stats_assertions: vec![
            "broad_projection_columns=16".to_string(),
            format!("record_batch_chunks={}", frame_evidence.record_batch_chunks),
            format!("max_encoded_frame_bytes={}", frame_evidence.max_encoded_frame_bytes),
            format!("frame_limit_bytes={FLIGHT_MAX_ENCODED_FRAME_BYTES}"),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

struct CollectedFlight {
    batches: Vec<RecordBatch>,
    encoded_frame_count: usize,
    max_encoded_frame_bytes: usize,
}

impl CollectedFlight {
    fn record_batch_chunks(&self) -> usize {
        self.batches.len()
    }

    fn frame_evidence(&self) -> VariableFrameEvidence {
        VariableFrameEvidence {
            encoded_frame_count: self.encoded_frame_count,
            record_batch_chunks: self.record_batch_chunks(),
            max_encoded_frame_bytes: self.max_encoded_frame_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VariableFrameEvidence {
    encoded_frame_count: usize,
    record_batch_chunks: usize,
    max_encoded_frame_bytes: usize,
}

fn validate_variable_frame_evidence(evidence: VariableFrameEvidence) -> E2eResult<()> {
    if evidence.record_batch_chunks < 2 {
        return Err("variable-width projection did not exercise Flight record-batch chunking".into());
    }
    if evidence.max_encoded_frame_bytes > FLIGHT_MAX_ENCODED_FRAME_BYTES {
        return Err(format!(
            "variable-width projection emitted a {} byte Flight frame above the {} byte limit",
            evidence.max_encoded_frame_bytes, FLIGHT_MAX_ENCODED_FRAME_BYTES
        )
        .into());
    }
    if evidence.encoded_frame_count < evidence.record_batch_chunks.saturating_add(1) {
        return Err("variable-width projection did not observe schema plus record-batch Flight frames".into());
    }
    Ok(())
}

fn observe_encoded_frame(frame: &FlightData, frame_count: &AtomicUsize, max_frame_bytes: &AtomicUsize) {
    frame_count.fetch_add(1, Ordering::Relaxed);
    max_frame_bytes.fetch_max(frame.encoded_len(), Ordering::Relaxed);
}

async fn collect_flight_info(channel: &Channel, flight_info: FlightInfo) -> E2eResult<CollectedFlight> {
    let mut batches = Vec::new();
    let encoded_frame_count = Arc::new(AtomicUsize::new(0));
    let max_encoded_frame_bytes = Arc::new(AtomicUsize::new(0));
    for endpoint in flight_info.endpoint {
        let ticket = endpoint.ticket.ok_or("Flight SQL endpoint missing ticket")?;
        let response = FlightServiceClient::new(channel.clone()).do_get(Request::new(ticket)).await?.into_inner();
        let frame_count = Arc::clone(&encoded_frame_count);
        let max_frame_bytes = Arc::clone(&max_encoded_frame_bytes);
        let observed = response.map_err(FlightError::from).map_ok(move |frame| {
            observe_encoded_frame(&frame, &frame_count, &max_frame_bytes);
            frame
        });
        let mut stream = FlightRecordBatchStream::new_from_flight_data(observed);
        while let Some(batch) = stream.try_next().await? {
            batches.push(batch);
        }
    }
    Ok(CollectedFlight {
        batches,
        encoded_frame_count: encoded_frame_count.load(Ordering::Relaxed),
        max_encoded_frame_bytes: max_encoded_frame_bytes.load(Ordering::Relaxed),
    })
}

async fn run_structured_failures(params: &RuntimeParams, channel: &Channel) -> E2eResult<Vec<ScenarioEvidence>> {
    let select = run_structured_failure_case(
        params,
        channel,
        "invalid_select",
        "SELECT missing_runtime_observability_column FROM runtime_observability_lineitem",
        false,
    )
    .await?;
    let ddl = run_structured_failure_case(
        params,
        channel,
        "invalid_ddl_update",
        "DROP TABLE runtime_observability_table_that_does_not_exist",
        true,
    )
    .await?;
    Ok(vec![select, ddl])
}

async fn run_structured_failure_case(
    params: &RuntimeParams,
    channel: &Channel,
    scenario: &str,
    sql: &str,
    update: bool,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let error = if update {
        tokio::time::timeout(params.query_timeout, client.execute_update(sql.to_string(), None))
            .await
            .map_err(|_| format!("{scenario} timed out"))?
            .expect_err("invalid DDL update must fail")
    } else {
        tokio::time::timeout(params.query_timeout, client.execute(sql.to_string(), None))
            .await
            .map_err(|_| format!("{scenario} timed out"))?
            .expect_err("invalid SELECT must fail")
    };
    let status = flight_status(error)?;
    if !matches!(status.code(), Code::InvalidArgument | Code::NotFound | Code::Unimplemented) {
        return Err(format!("{scenario} returned unexpected status {status:?}").into());
    }
    let query_id = status_metadata(&status, QUERY_ID_METADATA, scenario)?;
    let response_correlation = status_metadata(&status, CORRELATION_HEADER, scenario)?;
    if response_correlation != correlation_id {
        return Err(format!(
            "{scenario} response correlation id {response_correlation:?} did not match {correlation_id:?}"
        )
        .into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_contains(&stats, "# outcome=failed\n", "failed outcome")?;
    require_stats_contains(&stats, "[structured_error]\n", "structured error section")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("failed"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        profile: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "failed".to_string(),
        output_batches: 0,
        output_rows: 0,
        encoded_frame_count: None,
        record_batch_chunks: None,
        max_encoded_frame_bytes: None,
        stats_assertions: vec![
            "structured_error".to_string(),
            "query_id".to_string(),
            "correlation_id".to_string(),
            "runtime_failed".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

async fn run_cancel_query(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "cancel_query";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(cancellation_projection_sql(), None))
        .await
        .map_err(|_| "cancel query GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let result = cancel_query(channel, &flight_info).await?;
    if result.result != CANCEL_RESULT_CANCELLING {
        return Err(format!(
            "CancelQuery for {query_id} returned {}, expected {CANCEL_RESULT_CANCELLING}",
            result.result
        )
        .into());
    }
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 0, 0).await
}

async fn run_drop_before_first_poll(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "drop_before_first_poll";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "drop-before-poll GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| "drop-before-poll DoGet timed out")??;
    drop(stream);
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 0, 0).await
}

async fn run_midstream_drop(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "midstream_drop";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "midstream-drop GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let mut stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| "midstream-drop DoGet timed out")??;
    let first = tokio::time::timeout(params.query_timeout, stream.try_next())
        .await
        .map_err(|_| "midstream-drop first batch timed out")??
        .ok_or("midstream-drop stream ended before its first batch")?;
    let rows = first.num_rows();
    drop(stream);
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 1, rows).await
}

async fn terminal_scenario(
    params: &RuntimeParams,
    channel: &Channel,
    started: Instant,
    scenario: &str,
    query_id: String,
    correlation_id: String,
    output_batches: usize,
    output_rows: usize,
) -> E2eResult<ScenarioEvidence> {
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_contains(&stats, "# outcome=failed\n", "cancelled query outcome")?;
    require_stats_contains(&stats, "[structured_error]\n", "cancelled structured error")?;
    require_stats_contains(&stats, "code=cancelled\n", "cancelled structured error code")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("cancelled"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        profile: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "cancelled".to_string(),
        output_batches,
        output_rows,
        encoded_frame_count: None,
        record_batch_chunks: None,
        max_encoded_frame_bytes: None,
        stats_assertions: vec![
            "sealed".to_string(),
            "structured_cancelled".to_string(),
            "terminal_cancelled".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

async fn run_cpu_fallback(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-cpu-fallback".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(
        params.query_timeout,
        client.execute(
            format!(
                "SELECT l_orderkey FROM {TABLE_NAME} \
                 WHERE l_orderkey % l_partkey >= 0 ORDER BY l_orderkey LIMIT 2"
            ),
            None,
        ),
    )
    .await
    .map_err(|_| "CPU fallback GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "CPU fallback DoGet timed out")??;
    let batches = &collected.batches;
    let output_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if output_rows != 2 || batches.iter().any(|batch| batch.num_columns() != 1) {
        return Err(format!("CPU fallback query returned {output_rows} rows, expected two one-column rows").into());
    }
    let values = batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(|row| arrow::util::display::array_value_to_string(batch.column(0), row))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values != ["1", "1"] {
        return Err(format!("CPU fallback query returned {values:?}, expected [\"1\", \"1\"]").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "outcome", "rejected_before_execution")?;
    require_stats_field(&stats, "planning", "reason_code", "unsupported_expression")?;
    require_stats_field(&stats, "planning", "unsupported_category", "expression")?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "cpu")?;
    require_stats_field(&stats, "planning", "native_plan_fragment_count", "0")?;
    require_stats_field(&stats, "planning", "rejected_before_execution", "true")?;
    require_stats_field(&stats, "execution", "planning_outcome", "rejected_before_execution")?;
    require_stats_field(&stats, "execution", "post_native_execution_path", "cpu")?;
    require_stats_field(&stats, "execution", "native_runtime_started", "false")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "not_started")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if !artifact.plan_execution_ids.is_empty() {
        return Err(format!(
            "CPU fallback query {query_id} unexpectedly reported native plan_execution_ids {:?}",
            artifact.plan_execution_ids
        )
        .into());
    }
    Ok(ScenarioEvidence {
        scenario: "cpu_fallback".to_string(),
        profile: "cpu_fallback".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: batches.len(),
        output_rows,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "exact_result=[1,1]".to_string(),
            "unsupported_expression_rejected_before_execution".to_string(),
            "cpu_path".to_string(),
            "native_not_started".to_string(),
            "no_native_plan_execution_id".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

fn first_ticket(flight_info: &FlightInfo) -> E2eResult<Ticket> {
    flight_info
        .endpoint
        .first()
        .and_then(|endpoint| endpoint.ticket.clone())
        .ok_or_else(|| "FlightInfo missing statement ticket".into())
}

async fn start_in_flight_projection(
    params: &RuntimeParams,
    channel: &Channel,
    index: usize,
) -> E2eResult<(String, FlightRecordBatchStream)> {
    let correlation_id = format!("runtime-e2e-in-flight-{index}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| format!("in-flight projection {index} GetFlightInfo timed out"))??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| format!("in-flight projection {index} DoGet timed out"))??;
    Ok((query_id, stream))
}

async fn terminate_gracefully_with_in_flight(
    params: &RuntimeParams,
    server: &mut RunningServer,
) -> E2eResult<(ProcessEvidence, Vec<String>, Vec<String>)> {
    let mut statements = Vec::with_capacity(params.concurrency);
    for index in 0..params.concurrency {
        let correlation_id = format!("runtime-e2e-graceful-in-flight-{index}");
        let mut client = correlated_client(&server.channel, &correlation_id);
        let info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
            .await
            .map_err(|_| format!("graceful in-flight {index} GetFlightInfo timed out"))??;
        let query_id = query_id_from_statement_flight_info(&info)?;
        statements.push((query_id, first_ticket(&info)?));
    }

    let start = Arc::new(Barrier::new(statements.len() + 1));
    let release = Arc::new(Barrier::new(statements.len() + 1));
    let (ready_tx, mut ready_rx) = mpsc::channel(statements.len());
    let mut tasks = JoinSet::new();
    let query_ids = statements.iter().map(|(query_id, _)| query_id.clone()).collect::<Vec<_>>();
    for (query_id, ticket) in statements {
        let channel = server.channel.clone();
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);
        let ready_tx = ready_tx.clone();
        tasks.spawn(async move {
            start.wait().await;
            let mut client = FlightSqlServiceClient::new(channel);
            let mut stream = client.do_get(ticket).await?;
            ready_tx.send(query_id.clone()).await?;
            release.wait().await;
            let mut batches = 0usize;
            let mut rows = 0usize;
            loop {
                match stream.try_next().await {
                    Ok(Some(batch)) => {
                        batches += 1;
                        rows += batch.num_rows();
                    },
                    Ok(None) => {
                        return Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                            query_id,
                            format!("eof:batches={batches},rows={rows}"),
                        ));
                    },
                    Err(error) => {
                        return Ok((query_id, format!("stream_error:{error}")));
                    },
                }
            }
        });
    }
    drop(ready_tx);
    start.wait().await;
    let ready_deadline = Instant::now() + params.poll_timeout;
    let mut ready = BTreeSet::new();
    while ready.len() < query_ids.len() {
        let remaining = ready_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(
                format!("only {}/{} graceful restart streams became ready", ready.len(), query_ids.len()).into()
            );
        }
        let query_id = tokio::time::timeout(remaining, ready_rx.recv())
            .await
            .map_err(|_| "graceful restart stream readiness timed out")?
            .ok_or("graceful restart readiness channel closed early")?;
        ready.insert(query_id);
    }
    let mut active = 0usize;
    for query_id in &query_ids {
        let frames = require_action_frames(&server.channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let artifact = parse_artifact(query_id, None, &frames, None, None)?;
        if artifact.state == "active" {
            active += 1;
        }
    }
    if active < 2 || active != query_ids.len() {
        return Err(format!(
            "graceful restart requires all {} concurrent streams active, observed {active}",
            query_ids.len()
        )
        .into());
    }

    let termination = terminate_server(params, server, "TERM", true);
    tokio::pin!(termination);
    let exit = tokio::select! {
        result = &mut termination => {
            release.wait().await;
            result?
        },
        () = tokio::time::sleep(Duration::from_millis(50)) => {
            release.wait().await;
            termination.await?
        },
    };
    let mut terminal_results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        let (query_id, terminal) = result.map_err(|error| format!("graceful client task failed: {error}"))??;
        terminal_results.push(format!("{query_id}:{terminal}"));
    }
    terminal_results.sort();
    if terminal_results.len() != query_ids.len() {
        return Err(format!(
            "graceful restart captured {} client terminal results for {} streams",
            terminal_results.len(),
            query_ids.len()
        )
        .into());
    }
    Ok((exit, query_ids, terminal_results))
}

async fn run_retention_pressure(
    params: &RuntimeParams,
    channel: &Channel,
    evidence: &mut RunEvidence,
) -> E2eResult<RetentionEvidence> {
    let (live_query_id, live_stream) = start_in_flight_projection(params, channel, 10_000).await?;
    require_artifact_state(channel, &live_query_id, "active").await?;

    let generation_count = params.observability.retained_count.saturating_add(2);
    let mut generated = Vec::with_capacity(generation_count);
    let mut generated_wire_bytes = 0usize;
    for index in 0..generation_count {
        let row = run_success_query(params, channel, "retention_count_pressure", index).await?;
        generated_wire_bytes = generated_wire_bytes.saturating_add(row.artifact.wire_bytes);
        generated.push((row.query_id.clone(), Instant::now()));
        evidence.scenarios.push(row);
        require_artifact_state(channel, &live_query_id, "active").await?;
    }

    let mut sealed_count = 0usize;
    let mut unavailable = BTreeMap::new();
    let not_found = Vec::new();
    let no_silent_loss = true;
    for (query_id, sealed_at) in &generated {
        match optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await? {
            Some(frames) => {
                let artifact = parse_artifact(query_id, None, &frames, None, None)?;
                match artifact.state.as_str() {
                    "sealed" => sealed_count += 1,
                    "unavailable" => {
                        let reason = artifact.reason.unwrap_or_else(|| "missing_reason".to_string());
                        if reason != "evicted" {
                            return Err(format!(
                                "count retention query {query_id} became unavailable for {reason}, expected evicted"
                            )
                            .into());
                        }
                        unavailable.insert(query_id.clone(), reason);
                    },
                    other => return Err(format!("retention query {query_id} has unexpected state {other}").into()),
                }
            },
            None => {
                let elapsed = sealed_at.elapsed();
                return Err(format!(
                    "count retention query {query_id} became NotFound after {}ms without an unavailable tombstone",
                    millis(elapsed)
                )
                .into());
            },
        }
    }
    if sealed_count > params.observability.retained_count {
        return Err(format!(
            "retention kept {sealed_count} artifacts above count limit {}",
            params.observability.retained_count
        )
        .into());
    }
    if unavailable.is_empty() && not_found.is_empty() {
        return Err("count retention pressure did not evict any generated artifact".into());
    }

    let (ttl_query_id, ttl_sealed_at) = generated.last().expect("retention generation is nonempty").clone();
    let (ttl_terminal_state, ttl_unavailable_elapsed_ms, ttl_not_found_elapsed_ms) =
        wait_through_artifact_ttl(params, channel, &ttl_query_id, ttl_sealed_at).await?;
    require_artifact_state(channel, &live_query_id, "active").await?;

    let padding_len = usize::try_from(params.observability.per_query_bytes).unwrap_or(usize::MAX).saturating_add(4096);
    if padding_len > 2 * 1024 * 1024 + 4096 {
        return Err(
            "--per-query-bytes above 2 MiB makes the capture-budget statement unsafe; use a smaller value".into()
        );
    }
    let padded_sql = format!("/*{}*/ {}", "x".repeat(padding_len), native_sql());
    let capture_budget_input_bytes = padded_sql.len();
    let mut client = correlated_client(channel, "runtime-e2e-capture-budget-pressure");
    let info = tokio::time::timeout(params.query_timeout, client.execute(padded_sql, None))
        .await
        .map_err(|_| "capture-budget GetFlightInfo timed out")??;
    let capture_budget_query_id = query_id_from_statement_flight_info(&info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, info))
        .await
        .map_err(|_| "capture-budget DoGet timed out")??;
    if collected.batches.iter().map(RecordBatch::num_rows).sum::<usize>() != 1 {
        return Err("capture-budget Parquet query did not return one row".into());
    }
    let capture_stats = wait_for_sealed_stats(params, channel, &capture_budget_query_id).await?;
    let stats_incomplete = capture_stats.contains("# incomplete=true\n");
    let retained_byte_loss_recorded = capture_stats.contains("# capture_loss.retained_byte_limit=");
    if !stats_incomplete || !retained_byte_loss_recorded {
        return Err(format!(
            "capture-budget query {capture_budget_query_id} did not report bounded retained-byte loss:\n{capture_stats}"
        )
        .into());
    }
    let capture_frames = wait_for_terminal_artifact(params, channel, &capture_budget_query_id).await?;
    let capture_artifact = parse_artifact(&capture_budget_query_id, None, &capture_frames, None, None)?;
    let capture_budget_reason = capture_artifact.reason.clone().unwrap_or_else(|| "missing_reason".to_string());
    if capture_artifact.state != "unavailable" || capture_budget_reason != "capture_budget" {
        return Err(format!(
            "capture-budget query expected unavailable/capture_budget, got {}/{}",
            capture_artifact.state, capture_budget_reason
        )
        .into());
    }
    require_artifact_state(channel, &live_query_id, "active").await?;
    drop(live_stream);

    Ok(RetentionEvidence {
        generated_query_ids: generated.into_iter().map(|(query_id, _)| query_id).collect(),
        live_query_id,
        live_state_during_pressure: "active".to_string(),
        sealed_count,
        unavailable,
        not_found,
        generated_wire_bytes,
        capture_budget: CaptureBudgetEvidence {
            query_id: capture_budget_query_id,
            input_bytes: capture_budget_input_bytes,
            per_query_capture_limit: params.observability.per_query_bytes,
            state: capture_artifact.state,
            reason: capture_budget_reason,
            frame_count: capture_artifact.frame_count,
            stats_incomplete,
            retained_byte_loss_recorded,
        },
        global_byte_pressure: None,
        ttl_query_id,
        ttl_terminal_state,
        ttl_unavailable_elapsed_ms,
        ttl_not_found_elapsed_ms,
        no_silent_loss,
    })
}

async fn run_global_retained_byte_pressure_phase(
    params: &RuntimeParams,
    evidence: &mut RunEvidence,
) -> E2eResult<GlobalBytePressureEvidence> {
    let mut pressure_params = params.clone();
    pressure_params.observability = ObservabilityLimits {
        per_query_records: 256,
        per_query_bytes: GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES,
        retained_count: GLOBAL_BYTE_PRESSURE_RETAINED_COUNT,
        retained_bytes: GLOBAL_BYTE_PRESSURE_RETAINED_BYTES,
        ttl_secs: GLOBAL_BYTE_PRESSURE_TTL_SECS,
        shutdown_grace_secs: params.observability.shutdown_grace_secs,
    };
    let label = "native-global-byte-pressure";
    let mut server = spawn_server(&pressure_params, evidence, label, ChildProfile::Native).await?;
    let phase_result = async {
        setup_local_table(&pressure_params, &server.channel, label).await?;
        run_global_retained_byte_pressure(&pressure_params, &server.channel, label, &mut evidence.scenarios).await
    }
    .await;
    let termination = terminate_server(&pressure_params, &mut server, "TERM", true).await;
    match (phase_result, termination) {
        (Ok(phase), Ok(process)) => {
            evidence.processes.push(process);
            Ok(phase)
        },
        (Err(error), Ok(process)) => {
            evidence.processes.push(process);
            Err(error)
        },
        (Ok(_), Err(error)) => Err(error),
        (Err(phase_error), Err(termination_error)) => Err(format!(
            "global byte-pressure phase failed: {phase_error:#}; dedicated server termination also failed: {termination_error:#}"
        )
        .into()),
    }
}

#[derive(Debug)]
struct GlobalBytePressureSnapshot {
    sealed_query_ids: Vec<String>,
    sealed_charge_bytes: BTreeMap<String, u64>,
    evicted: BTreeMap<String, String>,
    retained_bytes: u64,
}

async fn run_global_retained_byte_pressure(
    params: &RuntimeParams,
    channel: &Channel,
    server_label: &str,
    scenarios: &mut Vec<ScenarioEvidence>,
) -> E2eResult<GlobalBytePressureEvidence> {
    if GLOBAL_BYTE_PRESSURE_MAX_QUERIES.saturating_add(GLOBAL_BYTE_PRESSURE_SETUP_QUERIES)
        >= params.observability.retained_count
    {
        return Err("global byte-pressure count limit does not leave room for every bounded probe query".into());
    }

    let mut generated_query_ids = Vec::new();
    let mut artifact_charge_bytes = BTreeMap::new();
    let mut generated_charge_bytes = 0u64;
    let mut max_observed_generated_retained_bytes = 0u64;
    let mut previously_evicted = BTreeSet::new();
    let mut terminal_snapshot = None;

    for index in 0..GLOBAL_BYTE_PRESSURE_MAX_QUERIES {
        let row = run_success_query(params, channel, "global_retained_byte_pressure", index).await?;
        let stats = get_query_stats(channel, &row.query_id).await?;
        let charge = artifact_charge(row.artifact.wire_bytes, stats.len())?;
        if charge > params.observability.retained_bytes {
            return Err(format!(
                "global byte-pressure query {} individually charged {charge} bytes above cap {}",
                row.query_id, params.observability.retained_bytes
            )
            .into());
        }
        generated_charge_bytes = generated_charge_bytes
            .checked_add(charge)
            .ok_or_else(|| "global byte-pressure generated charge accounting overflowed".to_string())?;
        artifact_charge_bytes.insert(row.query_id.clone(), charge);
        generated_query_ids.push(row.query_id.clone());
        scenarios.push(row);

        let snapshot = observe_global_byte_pressure_snapshot(params, channel, &generated_query_ids).await?;
        if snapshot.retained_bytes > params.observability.retained_bytes {
            return Err(format!(
                "global byte-pressure retained {} generated bytes above cap {}",
                snapshot.retained_bytes, params.observability.retained_bytes
            )
            .into());
        }
        if previously_evicted.iter().any(|query_id| !snapshot.evicted.contains_key(query_id)) {
            return Err("global byte-pressure eviction state regressed to a retained artifact".into());
        }
        for (query_id, observed_charge) in &snapshot.sealed_charge_bytes {
            let expected_charge = artifact_charge_bytes
                .get(query_id)
                .ok_or_else(|| format!("global byte-pressure snapshot returned unknown query {query_id}"))?;
            if observed_charge != expected_charge {
                return Err(format!(
                    "global byte-pressure query {query_id} charge changed from {expected_charge} to {observed_charge}"
                )
                .into());
            }
        }
        previously_evicted.extend(snapshot.evicted.keys().cloned());
        max_observed_generated_retained_bytes = max_observed_generated_retained_bytes.max(snapshot.retained_bytes);

        if !snapshot.evicted.is_empty() && generated_charge_bytes > params.observability.retained_bytes {
            terminal_snapshot = Some(snapshot);
            break;
        }
    }

    let snapshot = terminal_snapshot.ok_or_else(|| {
        format!(
            "global retained-byte pressure did not evict a generated artifact after {} bounded Parquet queries",
            GLOBAL_BYTE_PRESSURE_MAX_QUERIES
        )
    })?;
    let count_limit_not_reached = generated_query_ids.len().saturating_add(GLOBAL_BYTE_PRESSURE_SETUP_QUERIES)
        < params.observability.retained_count;
    if !count_limit_not_reached {
        return Err(format!(
            "global byte-pressure generated {} queries plus {} setup queries against count limit {}",
            generated_query_ids.len(),
            GLOBAL_BYTE_PRESSURE_SETUP_QUERIES,
            params.observability.retained_count
        )
        .into());
    }
    if snapshot.sealed_query_ids.is_empty() {
        return Err("global byte-pressure evicted every generated artifact instead of retaining a bounded set".into());
    }

    Ok(GlobalBytePressureEvidence {
        server_label: server_label.to_string(),
        per_query_bytes: params.observability.per_query_bytes,
        retained_count_limit: params.observability.retained_count,
        retained_byte_limit: params.observability.retained_bytes,
        ttl_secs: params.observability.ttl_secs,
        generated_query_ids,
        artifact_charge_bytes,
        sealed_query_ids_after_pressure: snapshot.sealed_query_ids,
        evicted: snapshot.evicted,
        generated_charge_bytes,
        max_observed_generated_retained_bytes,
        retained_bytes_after_pressure: snapshot.retained_bytes,
        setup_artifacts_known_evicted: true,
        exact_retained_set_proven: true,
        count_limit_not_reached,
        byte_limit_respected: true,
        no_silent_loss: true,
    })
}

async fn observe_global_byte_pressure_snapshot(
    params: &RuntimeParams,
    channel: &Channel,
    query_ids: &[String],
) -> E2eResult<GlobalBytePressureSnapshot> {
    let mut sealed_query_ids = Vec::new();
    let mut sealed_charge_bytes = BTreeMap::new();
    let mut evicted = BTreeMap::new();
    let mut retained_bytes = 0u64;
    for query_id in query_ids {
        let frames = optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let Some(frames) = frames else {
            return Err(format!(
                "global byte-pressure query {query_id} became NotFound before the configured {}s TTL",
                params.observability.ttl_secs
            )
            .into());
        };
        let artifact = parse_artifact(query_id, None, &frames, None, Some(&params.observability))?;
        match artifact.state.as_str() {
            "sealed" => {
                let stats = get_query_stats(channel, query_id).await?;
                let charge = artifact_charge(artifact.wire_bytes, stats.len())?;
                retained_bytes = retained_bytes
                    .checked_add(charge)
                    .ok_or_else(|| "global byte-pressure retained charge accounting overflowed".to_string())?;
                sealed_query_ids.push(query_id.clone());
                sealed_charge_bytes.insert(query_id.clone(), charge);
            },
            "unavailable" if artifact.reason.as_deref() == Some("evicted") => {
                evicted.insert(query_id.clone(), "evicted".to_string());
            },
            state => {
                return Err(format!(
                    "global byte-pressure query {query_id} entered {state}/{:?}, expected sealed or unavailable/evicted",
                    artifact.reason
                )
                .into());
            },
        }
    }
    if sealed_query_ids.len().saturating_add(evicted.len()) != query_ids.len() {
        return Err("global byte-pressure snapshot silently lost a generated query".into());
    }
    Ok(GlobalBytePressureSnapshot { sealed_query_ids, sealed_charge_bytes, evicted, retained_bytes })
}

fn artifact_charge(runtime_wire_bytes: usize, stats_bytes: usize) -> E2eResult<u64> {
    u64::try_from(runtime_wire_bytes)
        .ok()
        .and_then(|runtime| u64::try_from(stats_bytes).ok().and_then(|stats| runtime.checked_add(stats)))
        .ok_or_else(|| "runtime artifact exact byte charge overflowed".into())
}

async fn wait_for_sealed_stats(params: &RuntimeParams, channel: &Channel, query_id: &str) -> E2eResult<String> {
    let deadline = Instant::now() + params.poll_timeout;
    loop {
        let stats = get_query_stats(channel, query_id).await?;
        if stats.contains("# sealed=true\n") {
            return Ok(stats);
        }
        if Instant::now() >= deadline {
            return Err(
                format!("query {query_id} stats did not seal within {:?}:\n{stats}", params.poll_timeout).into()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn wait_for_sealed_artifact(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
) -> E2eResult<Vec<Vec<u8>>> {
    let frames = wait_for_terminal_artifact(params, channel, query_id).await?;
    let state = control_state(query_id, &frames)?;
    if state != "sealed" {
        return Err(format!("query {query_id} artifact became {state}, expected sealed").into());
    }
    let repeated = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
    if repeated != frames {
        return Err(format!("query {query_id} sealed artifact bytes changed across repeated retrieval").into());
    }
    Ok(frames)
}

async fn wait_for_terminal_artifact(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
) -> E2eResult<Vec<Vec<u8>>> {
    let deadline = Instant::now() + params.poll_timeout;
    loop {
        let frames = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        match control_state(query_id, &frames)?.as_str() {
            "sealed" | "unavailable" => return Ok(frames),
            "pending" | "active" if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            },
            state => {
                return Err(
                    format!("query {query_id} artifact remained {state} beyond {:?}", params.poll_timeout).into()
                );
            },
        }
    }
}

async fn require_action_frames(channel: &Channel, action_type: &str, body: &[u8]) -> E2eResult<Vec<Vec<u8>>> {
    action_frames(channel, action_type, body.to_vec()).await
}

async fn optional_action_frames(channel: &Channel, action_type: &str, body: &[u8]) -> E2eResult<Option<Vec<Vec<u8>>>> {
    match action_frames(channel, action_type, body.to_vec()).await {
        Ok(frames) => Ok(Some(frames)),
        Err(error) if boxed_status_code(error.as_ref()) == Some(Code::NotFound) => Ok(None),
        Err(error) => Err(error),
    }
}

fn boxed_status_code(error: &(dyn std::error::Error + Send + Sync + 'static)) -> Option<Code> {
    let flight = error.downcast_ref::<FlightError>()?;
    match flight {
        FlightError::Tonic(status) => Some(status.code()),
        _ => None,
    }
}

fn control_state(query_id: &str, frames: &[Vec<u8>]) -> E2eResult<String> {
    let first = frames.first().ok_or("runtime artifact action returned no frames")?;
    let control: Value = serde_json::from_slice(first)?;
    if control.get("schema").and_then(Value::as_str) != Some(CONTROL_SCHEMA)
        || control.get("version").and_then(Value::as_u64) != Some(1)
        || control.get("query_id").and_then(Value::as_str) != Some(query_id)
    {
        return Err(format!("invalid runtime artifact control for {query_id}: {control}").into());
    }
    Ok(control
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("runtime artifact control lacks state: {control}"))?
        .to_string())
}

fn require_stats_identity(stats: &str, query_id: &str, correlation_id: &str) -> E2eResult<()> {
    require_stats_contains(stats, &format!("# query_id={query_id}\n"), "query id")?;
    require_stats_contains(stats, &format!("# correlation_id={correlation_id}\n"), "correlation id")?;
    require_stats_contains(stats, "# sealed=true\n", "sealed stats")?;
    if stats.to_ascii_lowercase().contains("multiple_sessions") {
        return Err(format!("query {query_id} stats reported multiple_sessions").into());
    }
    Ok(())
}

fn require_stats_contains(stats: &str, needle: &str, label: &str) -> E2eResult<()> {
    if !stats.contains(needle) {
        return Err(format!("query stats missing {label} {needle:?}:\n{stats}").into());
    }
    Ok(())
}

fn require_stats_field(stats: &str, section: &str, field: &str, expected: &str) -> E2eResult<()> {
    let row = tsv_section_single_row(stats, section)?;
    let actual = row.get(field).ok_or_else(|| format!("query stats [{section}] missing field {field}: {row:?}"))?;
    if actual != expected {
        return Err(format!("query stats [{section}] expected {field}={expected:?}, got {actual:?}").into());
    }
    Ok(())
}

fn parse_artifact(
    query_id: &str,
    expected_correlation_id: Option<&str>,
    frames: &[Vec<u8>],
    expected_terminal: Option<&str>,
    expected_limits: Option<&ObservabilityLimits>,
) -> E2eResult<ArtifactEvidence> {
    let wire_bytes = frames.iter().map(Vec::len).sum();
    let control_frame = frames.first().ok_or("runtime artifact action returned no frames")?;
    let control: Value = serde_json::from_slice(control_frame)?;
    if control.get("schema").and_then(Value::as_str) != Some(CONTROL_SCHEMA)
        || control.get("version").and_then(Value::as_u64) != Some(1)
        || control.get("query_id").and_then(Value::as_str) != Some(query_id)
    {
        return Err(format!("invalid runtime artifact control frame for {query_id}: {control}").into());
    }
    let state = control
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("runtime artifact control lacks state: {control}"))?;
    let retryable = control
        .get("retryable")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("runtime artifact control lacks retryable: {control}"))?;
    let reason = control.get("reason").and_then(Value::as_str).map(str::to_string);
    let expected_frame_count = if state == "sealed" { 2 } else { 1 };
    if frames.len() != expected_frame_count {
        return Err(format!(
            "runtime artifact {query_id} state {state} returned {} frames, expected {expected_frame_count}",
            frames.len()
        )
        .into());
    }
    match state {
        "pending" | "active" if !retryable => {
            return Err(format!("runtime artifact {query_id} state {state} must be retryable").into());
        },
        "sealed" | "unavailable" if retryable => {
            return Err(format!("runtime artifact {query_id} state {state} must not be retryable").into());
        },
        "pending" | "active" | "sealed" | "unavailable" => {},
        other => return Err(format!("runtime artifact {query_id} has unknown state {other:?}").into()),
    }
    if state == "unavailable" && reason.is_none() {
        return Err(format!("runtime artifact {query_id} unavailable control lacks reason").into());
    }
    if state != "sealed" {
        if expected_terminal.is_some() {
            return Err(format!("runtime artifact {query_id} is {state}, expected sealed terminal").into());
        }
        return Ok(ArtifactEvidence {
            state: state.to_string(),
            reason,
            frame_count: frames.len(),
            wire_bytes,
            jsonl_records: 0,
            topology_records: 0,
            timeline_events: 0,
            plan_execution_ids: Vec::new(),
            terminal_kind: None,
            retained_records: None,
            retained_bytes: None,
            truncated: None,
            degraded: None,
            poison_recoveries: None,
            dropped_by_class: None,
            dropped_by_reason: None,
        });
    }

    let jsonl = std::str::from_utf8(&frames[1])?;
    if !jsonl.ends_with('\n') {
        return Err(format!("runtime artifact {query_id} JSONL lacks a final newline").into());
    }
    if jsonl.to_ascii_lowercase().contains("multiple_sessions") {
        return Err(format!("runtime artifact {query_id} reported multiple_sessions").into());
    }
    let records = jsonl
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str::<Value>(line)
                .map_err(|error| format!("runtime artifact {query_id} JSONL line {index} is invalid: {error}").into())
        })
        .collect::<E2eResult<Vec<_>>>()?;
    if records.len() < 2 {
        return Err(format!("runtime artifact {query_id} needs manifest and footer records").into());
    }
    let manifest_indices = record_indices(&records, "manifest");
    let footer_indices = record_indices(&records, "footer");
    if manifest_indices != [0] || footer_indices != [records.len() - 1] {
        return Err(format!(
            "runtime artifact {query_id} requires one first manifest and one last footer; manifest={manifest_indices:?} footer={footer_indices:?}"
        )
        .into());
    }
    let manifest = records[0].get("data").ok_or("manifest lacks data")?;
    if manifest.get("schema").and_then(Value::as_str) != Some(JSONL_SCHEMA)
        || manifest.get("query_id").and_then(Value::as_str) != Some(query_id)
    {
        return Err(format!("runtime artifact {query_id} has invalid manifest: {manifest}").into());
    }
    if let Some(expected) = expected_correlation_id
        && manifest.get("trace_id").and_then(Value::as_str) != Some(expected)
    {
        return Err(format!(
            "runtime artifact {query_id} trace_id {:?} did not match correlation id {expected:?}",
            manifest.get("trace_id")
        )
        .into());
    }
    let manifest_record_limit = manifest
        .pointer("/limits/max_retained_records")
        .and_then(Value::as_u64)
        .ok_or("manifest lacks max_retained_records")?;
    let manifest_byte_limit = manifest
        .pointer("/limits/max_retained_bytes")
        .and_then(Value::as_u64)
        .ok_or("manifest lacks max_retained_bytes")?;
    if let Some(limits) = expected_limits
        && (manifest_record_limit != limits.per_query_records || manifest_byte_limit != limits.per_query_bytes)
    {
        return Err(format!(
            "runtime artifact {query_id} manifest limits records={manifest_record_limit} bytes={manifest_byte_limit}, expected records={} bytes={}",
            limits.per_query_records, limits.per_query_bytes
        )
        .into());
    }

    const TOPOLOGY_TYPES: &[&str] = &[
        "plan_node",
        "plan_edge",
        "task_definition",
        "compiled_edge",
        "barrier_definition",
        "sink_definition",
        "reservation_gate",
        "job_definition",
        "worker_definition",
        "resource",
    ];
    let mut topology_records = 0usize;
    let mut timeline_events = 0usize;
    let mut timeline_started = false;
    let mut last_topology_rank = None;
    let mut last_timeline_sequence = None;
    let mut retained_sequences = BTreeSet::new();
    let mut plan_execution_ids = BTreeSet::new();
    for (index, record) in records.iter().enumerate().skip(1).take(records.len() - 2) {
        let record_type = record
            .get("record_type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("runtime artifact {query_id} record {index} lacks record_type"))?;
        if TOPOLOGY_TYPES.contains(&record_type) {
            if timeline_started {
                return Err(format!("runtime artifact {query_id} topology record follows timeline at {index}").into());
            }
            let rank = TOPOLOGY_TYPES
                .iter()
                .position(|candidate| *candidate == record_type)
                .expect("known topology record type");
            if last_topology_rank.is_some_and(|previous| rank < previous) {
                return Err(format!(
                    "runtime artifact {query_id} topology record type {record_type} is outside canonical enum order"
                )
                .into());
            }
            last_topology_rank = Some(rank);
            topology_records += 1;
        } else if record_type == "timeline_event" {
            timeline_started = true;
            timeline_events += 1;
        } else {
            return Err(format!("runtime artifact {query_id} has unexpected record type {record_type:?}").into());
        }
        let sequence = record
            .pointer("/data/sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("runtime artifact {query_id} record {index} lacks numeric sequence"))?;
        if record_type == "timeline_event" {
            if last_timeline_sequence.is_some_and(|previous| sequence <= previous) {
                return Err(format!(
                    "runtime artifact {query_id} timeline sequence is not strictly increasing: {sequence} after {last_timeline_sequence:?}"
                )
                .into());
            }
            last_timeline_sequence = Some(sequence);
        }
        if !retained_sequences.insert(sequence) {
            return Err(format!("runtime artifact {query_id} repeats sequence {sequence}").into());
        }
        collect_plan_execution_ids(record, &mut plan_execution_ids)?;
    }

    let footer = records.last().and_then(|record| record.get("data")).ok_or("footer lacks data")?;
    if footer.get("complete").and_then(Value::as_bool) != Some(true) {
        return Err(format!("runtime artifact {query_id} footer is incomplete: {footer}").into());
    }
    let terminal_kind = footer.pointer("/terminal/kind").and_then(Value::as_str).map(str::to_string);
    if let Some(expected) = expected_terminal
        && terminal_kind.as_deref() != Some(expected)
    {
        return Err(format!(
            "runtime artifact {query_id} terminal {:?} did not match stats outcome {expected:?}",
            terminal_kind
        )
        .into());
    }
    let retained_records =
        footer.get("retained_records").and_then(Value::as_u64).ok_or("footer lacks retained_records")?;
    let retained_bytes = footer.get("retained_bytes").and_then(Value::as_u64).ok_or("footer lacks retained_bytes")?;
    let observed_retained = u64::try_from(topology_records + timeline_events)?;
    if retained_records != observed_retained
        || retained_records > manifest_record_limit
        || retained_bytes > manifest_byte_limit
        || (retained_records == 0) != (retained_bytes == 0)
    {
        return Err(format!(
            "runtime artifact {query_id} footer retention inconsistent: records={retained_records}/{observed_retained}/{manifest_record_limit}, bytes={retained_bytes}/{manifest_byte_limit}"
        )
        .into());
    }
    let truncated = footer.get("truncated").and_then(Value::as_bool).ok_or("footer lacks truncated")?;
    let degraded = footer.get("degraded").and_then(Value::as_bool).ok_or("footer lacks degraded")?;
    let poison_recoveries =
        footer.get("poison_recoveries").and_then(Value::as_u64).ok_or("footer lacks poison_recoveries")?;
    validate_footer_sequences(query_id, footer, retained_records, &retained_sequences, truncated)?;
    let plan_execution_ids = plan_execution_ids.into_iter().collect::<Vec<_>>();
    Ok(ArtifactEvidence {
        state: state.to_string(),
        reason,
        frame_count: frames.len(),
        wire_bytes,
        jsonl_records: records.len(),
        topology_records,
        timeline_events,
        plan_execution_ids,
        terminal_kind,
        retained_records: Some(retained_records),
        retained_bytes: Some(retained_bytes),
        truncated: Some(truncated),
        degraded: Some(degraded),
        poison_recoveries: Some(poison_recoveries),
        dropped_by_class: footer.get("dropped_by_class").cloned(),
        dropped_by_reason: footer.get("dropped_by_reason").cloned(),
    })
}

fn record_indices(records: &[Value], record_type: &str) -> Vec<usize> {
    records
        .iter()
        .enumerate()
        .filter_map(|(index, record)| {
            (record.get("record_type").and_then(Value::as_str) == Some(record_type)).then_some(index)
        })
        .collect()
}

fn validate_footer_sequences(
    query_id: &str,
    footer: &Value,
    retained_records: u64,
    retained_sequences: &BTreeSet<u64>,
    truncated: bool,
) -> E2eResult<()> {
    let first_dropped = optional_u64_field(footer, "first_dropped_sequence")?;
    let last_attempted = optional_u64_field(footer, "last_attempted_sequence")?;
    let dropped_by_class = loss_count(footer, "dropped_by_class")?;
    let dropped_by_reason = loss_count(footer, "dropped_by_reason")?;
    if truncated != first_dropped.is_some() || first_dropped.is_some() != (dropped_by_class > 0) {
        return Err(format!(
            "runtime artifact {query_id} footer loss flags disagree: truncated={truncated} first_dropped={first_dropped:?} dropped_by_class={dropped_by_class}"
        )
        .into());
    }
    if dropped_by_reason < dropped_by_class {
        return Err(format!(
            "runtime artifact {query_id} reason loss {dropped_by_reason} is below class loss {dropped_by_class}"
        )
        .into());
    }
    match last_attempted {
        Some(last) => {
            if retained_records.saturating_add(dropped_by_class) != last {
                return Err(format!(
                    "runtime artifact {query_id} attempted sequence {last} does not reconcile retained={retained_records} + class_dropped={dropped_by_class}"
                )
                .into());
            }
            if retained_sequences.last().is_some_and(|sequence| *sequence > last) {
                return Err(
                    format!("runtime artifact {query_id} retained sequence exceeds last attempted {last}").into()
                );
            }
            if let Some(first) = first_dropped
                && (first == 0 || first > last || retained_sequences.contains(&first))
            {
                return Err(format!(
                    "runtime artifact {query_id} invalid first dropped sequence {first} for last attempted {last}"
                )
                .into());
            }
        },
        None if retained_records == 0 && dropped_by_class == 0 => {},
        None => {
            return Err(format!(
                "runtime artifact {query_id} lacks last_attempted_sequence with retained={retained_records} dropped={dropped_by_class}"
            )
            .into());
        },
    }
    Ok(())
}

fn optional_u64_field(value: &Value, field: &str) -> E2eResult<Option<u64>> {
    match value.get(field) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("footer field {field} is not an unsigned integer: {value}").into()),
    }
}

fn loss_count(footer: &Value, field: &str) -> E2eResult<u64> {
    let rows =
        footer.get(field).and_then(Value::as_array).ok_or_else(|| format!("footer field {field} is not an array"))?;
    rows.iter().try_fold(0u64, |total, row| {
        let count =
            row.get("count").and_then(Value::as_u64).ok_or_else(|| format!("footer {field} row lacks count: {row}"))?;
        Ok(total.saturating_add(count))
    })
}

fn collect_plan_execution_ids(value: &Value, output: &mut BTreeSet<String>) -> E2eResult<()> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "plan_execution_id"
                    && let Some(id) = value.as_str()
                {
                    if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
                        return Err(format!("illegal plan_execution_id {id:?}").into());
                    }
                    output.insert(id.to_string());
                }
                collect_plan_execution_ids(value, output)?;
            }
        },
        Value::Array(values) => {
            for value in values {
                collect_plan_execution_ids(value, output)?;
            }
        },
        _ => {},
    }
    Ok(())
}

async fn require_artifact_state(channel: &Channel, query_id: &str, expected: &str) -> E2eResult<()> {
    let frames = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
    let artifact = parse_artifact(query_id, None, &frames, None, None)?;
    if artifact.state != expected {
        return Err(format!("query {query_id} artifact state {}, expected {expected}", artifact.state).into());
    }
    Ok(())
}

async fn wait_through_artifact_ttl(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
    sealed_at: Instant,
) -> E2eResult<(String, u64, u64)> {
    let retained_ttl = Duration::from_secs(params.observability.ttl_secs);
    let full_lifetime = retained_ttl.saturating_mul(2);
    let tolerance = Duration::from_millis(100).min(retained_ttl);
    let unavailable_deadline = sealed_at + retained_ttl + params.poll_timeout;
    let not_found_deadline = sealed_at + full_lifetime + params.poll_timeout;
    let mut unavailable_elapsed = None;
    loop {
        match optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await? {
            Some(frames) => {
                let artifact = parse_artifact(query_id, None, &frames, None, None)?;
                match artifact.state.as_str() {
                    "sealed" if unavailable_elapsed.is_none() => {},
                    "unavailable" if artifact.reason.as_deref() == Some("evicted") => {
                        let elapsed = sealed_at.elapsed();
                        if elapsed.saturating_add(tolerance) < retained_ttl {
                            return Err(format!(
                                "TTL query {query_id} became unavailable after {}ms, before retained TTL {}ms",
                                millis(elapsed),
                                millis(retained_ttl)
                            )
                            .into());
                        }
                        unavailable_elapsed.get_or_insert(elapsed);
                    },
                    other => {
                        return Err(format!("TTL query {query_id} entered unexpected state {other}").into());
                    },
                }
            },
            None => {
                let elapsed = sealed_at.elapsed();
                let Some(unavailable_elapsed) = unavailable_elapsed else {
                    return Err(format!("TTL query {query_id} became NotFound without an unavailable tombstone").into());
                };
                if elapsed.saturating_add(tolerance) < full_lifetime {
                    return Err(format!(
                        "TTL query {query_id} became NotFound after {}ms, before retained+tombstone TTL {}ms",
                        millis(elapsed),
                        millis(full_lifetime)
                    )
                    .into());
                }
                return Ok((
                    "unavailable:evicted_then_not_found".to_string(),
                    millis(unavailable_elapsed),
                    millis(elapsed),
                ));
            },
        }
        if unavailable_elapsed.is_none() && Instant::now() >= unavailable_deadline {
            return Err(format!("TTL query {query_id} did not become unavailable within the bounded TTL poll").into());
        }
        if Instant::now() >= not_found_deadline {
            return Err(format!("TTL query {query_id} tombstone did not expire within the bounded poll").into());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn admission_activity(payload: &str) -> E2eResult<(u64, u64, u64)> {
    let admission = tsv_section_single_row(payload, "admission")?;
    let gpu = tsv_section_single_row(payload, "gpu_execution")?;
    Ok((
        parse_u64_field(&admission, "active_queries", "runtime observability wave")?,
        parse_u64_field(&gpu, "active_query_jobs", "runtime observability wave")?,
        parse_u64_field(&gpu, "peak_active_query_jobs", "runtime observability wave")?,
    ))
}

fn flight_status(error: FlightError) -> E2eResult<tonic::Status> {
    match error {
        FlightError::Tonic(status) => Ok(*status),
        other => Err(format!("expected Flight tonic status, got {other:?}").into()),
    }
}

fn status_metadata(status: &tonic::Status, name: &str, context: &str) -> E2eResult<String> {
    status
        .metadata()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| format!("{context} status lacks {name} metadata: {status:?}").into())
}

async fn wait_for_idle_with_timeout(params: &RuntimeParams, channel: &Channel) -> E2eResult<String> {
    tokio::time::timeout(params.poll_timeout, wait_for_idle(channel))
        .await
        .map_err(|_| format!("server did not become idle within {:?}", params.poll_timeout))?
}

fn scan_fatal_patterns(paths: &[PathBuf]) -> FatalScanResult {
    let mut evidence = FatalPatternEvidence::default();
    for path in paths {
        let Ok(contents) = fs::read_to_string(path) else { continue };
        evidence.scanned_files += 1;
        for (line_index, line) in contents.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            let mut matched =
                FATAL_PATTERNS.iter().copied().filter(|pattern| lower.contains(pattern)).collect::<BTreeSet<_>>();
            if lower.contains("thread") && lower.contains("panicked") {
                matched.insert("thread .* panicked");
            }
            for pattern in matched {
                evidence.matches.push(json!({
                    "path": path,
                    "line": line_index + 1,
                    "pattern": pattern,
                    "excerpt": line.chars().take(500).collect::<String>(),
                }));
            }
        }
    }
    FatalScanResult { evidence }
}

fn hardware_evidence() -> Value {
    let query = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output();
    let (nvidia_smi_status, gpus, nvidia_smi_stderr) = match query {
        Ok(output) => (
            output.status.to_string(),
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>(),
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ),
        Err(error) => ("spawn_failed".to_string(), Vec::new(), error.to_string()),
    };
    let kernel = Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    json!({
        "nvidia_smi_status": nvidia_smi_status,
        "nvidia_smi_stderr": nvidia_smi_stderr,
        "gpus": gpus,
        "kernel": kernel,
        "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
    })
}

fn ensure_parent(path: &Path) -> E2eResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUERY_ID: &str = "parser-fixture-query";
    const TRACE_ID: &str = "parser-fixture-correlation";

    fn control_frame(state: &str, retryable: bool, reason: Option<&str>) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": CONTROL_SCHEMA,
            "version": 1,
            "query_id": QUERY_ID,
            "state": state,
            "retryable": retryable,
            "reason": reason,
        }))
        .unwrap()
    }

    fn manifest() -> Value {
        json!({
            "record_type": "manifest",
            "data": {
                "schema": JSONL_SCHEMA,
                "query_id": QUERY_ID,
                "trace_id": TRACE_ID,
                "limits": {
                    "max_retained_records": 16,
                    "max_retained_bytes": 8192,
                }
            }
        })
    }

    fn footer(
        retained_records: u64,
        retained_bytes: u64,
        truncated: bool,
        first_dropped_sequence: Option<u64>,
        last_attempted_sequence: Option<u64>,
        dropped_by_class: Vec<Value>,
        dropped_by_reason: Vec<Value>,
    ) -> Value {
        json!({
            "record_type": "footer",
            "data": {
                "complete": true,
                "terminal": {"kind": "succeeded"},
                "retained_records": retained_records,
                "retained_bytes": retained_bytes,
                "truncated": truncated,
                "degraded": false,
                "poison_recoveries": 0,
                "first_dropped_sequence": first_dropped_sequence,
                "last_attempted_sequence": last_attempted_sequence,
                "dropped_by_class": dropped_by_class,
                "dropped_by_reason": dropped_by_reason,
            }
        })
    }

    fn sealed_frames(middle: Vec<Value>, footer: Value) -> Vec<Vec<u8>> {
        let mut records = vec![manifest()];
        records.extend(middle);
        records.push(footer);
        let mut jsonl = records
            .into_iter()
            .map(|record| serde_json::to_string(&record).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
        jsonl.push(b'\n');
        vec![control_frame("sealed", false, None), jsonl]
    }

    fn canonical_middle() -> Vec<Value> {
        vec![
            json!({
                "record_type": "plan_node",
                "data": {"sequence": 2, "plan_execution_id": "exec-1"}
            }),
            json!({"record_type": "task_definition", "data": {"sequence": 1}}),
            json!({"record_type": "timeline_event", "data": {"sequence": 3}}),
            json!({"record_type": "timeline_event", "data": {"sequence": 4}}),
        ]
    }

    #[test]
    fn sealed_two_frame_fixture_accepts_canonical_topology_independent_of_topology_sequence() {
        let frames = sealed_frames(canonical_middle(), footer(4, 512, false, None, Some(4), vec![], vec![]));

        let artifact = parse_artifact(QUERY_ID, Some(TRACE_ID), &frames, Some("succeeded"), None).unwrap();

        assert_eq!(artifact.state, "sealed");
        assert_eq!(artifact.frame_count, 2);
        assert_eq!(artifact.jsonl_records, 6);
        assert_eq!(artifact.topology_records, 2);
        assert_eq!(artifact.timeline_events, 2);
        assert_eq!(artifact.plan_execution_ids, ["exec-1"]);
        assert_eq!(artifact.retained_records, Some(4));
        assert_eq!(artifact.truncated, Some(false));
    }

    #[test]
    fn control_fixture_accepts_pending_active_and_unavailable_states() {
        for (state, retryable, reason) in [
            ("pending", true, None),
            ("active", true, None),
            ("unavailable", false, Some("capture_budget")),
            ("unavailable", false, Some("evicted")),
            ("unavailable", false, Some("global_budget")),
        ] {
            let frames = vec![control_frame(state, retryable, reason)];
            let artifact = parse_artifact(QUERY_ID, None, &frames, None, None).unwrap();
            assert_eq!(artifact.state, state);
            assert_eq!(artifact.reason.as_deref(), reason);
            assert_eq!(artifact.frame_count, 1);
            assert_eq!(artifact.jsonl_records, 0);
        }
    }

    #[test]
    fn evidence_v2_separates_capture_and_global_retained_byte_pressure() {
        assert_eq!(EVIDENCE_SCHEMA, "datafusion_nexus.server_runtime_observability_e2e.v2");
        assert_eq!(EVIDENCE_VERSION, 2);
        assert!(GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES < GLOBAL_BYTE_PRESSURE_RETAINED_BYTES);
        assert!(
            GLOBAL_BYTE_PRESSURE_MAX_QUERIES + GLOBAL_BYTE_PRESSURE_SETUP_QUERIES < GLOBAL_BYTE_PRESSURE_RETAINED_COUNT
        );
        assert_eq!(artifact_charge(38_000, 12_000).unwrap(), 50_000);
    }

    #[test]
    fn parser_rejects_noncanonical_topology_and_nonincreasing_timeline_sequence() {
        let noncanonical = vec![
            json!({"record_type": "task_definition", "data": {"sequence": 1}}),
            json!({"record_type": "plan_node", "data": {"sequence": 2}}),
        ];
        let error = parse_artifact(
            QUERY_ID,
            None,
            &sealed_frames(noncanonical, footer(2, 128, false, None, Some(2), vec![], vec![])),
            Some("succeeded"),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("outside canonical enum order"), "{error}");

        let timeline = vec![
            json!({"record_type": "timeline_event", "data": {"sequence": 3}}),
            json!({"record_type": "timeline_event", "data": {"sequence": 2}}),
        ];
        let error = parse_artifact(
            QUERY_ID,
            None,
            &sealed_frames(timeline, footer(2, 128, false, None, Some(2), vec![], vec![])),
            Some("succeeded"),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("timeline sequence is not strictly increasing"), "{error}");
    }

    #[test]
    fn footer_loss_fixture_reconciles_and_rejects_inconsistent_reason_loss() {
        let middle = vec![
            json!({"record_type": "plan_node", "data": {"sequence": 1}}),
            json!({"record_type": "timeline_event", "data": {"sequence": 2}}),
        ];
        let valid_footer = footer(
            2,
            128,
            true,
            Some(3),
            Some(3),
            vec![json!({"class": "timeline", "count": 1})],
            vec![json!({"reason": "record_limit", "count": 1})],
        );
        let artifact =
            parse_artifact(QUERY_ID, None, &sealed_frames(middle.clone(), valid_footer), Some("succeeded"), None)
                .unwrap();
        assert_eq!(artifact.truncated, Some(true));
        assert_eq!(artifact.dropped_by_class, Some(json!([{"class": "timeline", "count": 1}])));

        let invalid_footer =
            footer(2, 128, true, Some(3), Some(3), vec![json!({"class": "timeline", "count": 1})], vec![]);
        let error = parse_artifact(QUERY_ID, None, &sealed_frames(middle, invalid_footer), Some("succeeded"), None)
            .unwrap_err();
        assert!(error.to_string().contains("reason loss 0 is below class loss 1"), "{error}");
    }

    #[test]
    fn parser_rejects_malformed_multiple_sessions_and_oversized_plan_execution_id() {
        let malformed = vec![control_frame("sealed", false, None), b"{not-json}\n".to_vec()];
        let error = parse_artifact(QUERY_ID, None, &malformed, Some("succeeded"), None).unwrap_err();
        assert!(error.to_string().contains("JSONL line 0 is invalid"), "{error}");

        let multiple_sessions = vec![json!({
            "record_type": "timeline_event",
            "data": {"sequence": 1, "diagnostic": "multiple_sessions"}
        })];
        let error = parse_artifact(
            QUERY_ID,
            None,
            &sealed_frames(multiple_sessions, footer(1, 64, false, None, Some(1), vec![], vec![])),
            Some("succeeded"),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("reported multiple_sessions"), "{error}");

        let oversized = vec![json!({
            "record_type": "plan_node",
            "data": {"sequence": 1, "plan_execution_id": "x".repeat(257)}
        })];
        let error = parse_artifact(
            QUERY_ID,
            None,
            &sealed_frames(oversized, footer(1, 64, false, None, Some(1), vec![], vec![])),
            Some("succeeded"),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("illegal plan_execution_id"), "{error}");
    }

    #[test]
    fn variable_frame_evidence_aggregates_and_rejects_transport_oversize() {
        let count = AtomicUsize::new(0);
        let max = AtomicUsize::new(0);
        let frames =
            [8usize, 32, 16].map(|bytes| FlightData { data_body: vec![b'x'; bytes].into(), ..FlightData::default() });
        for frame in &frames {
            observe_encoded_frame(frame, &count, &max);
        }
        let evidence = VariableFrameEvidence {
            encoded_frame_count: count.load(Ordering::Relaxed),
            record_batch_chunks: 2,
            max_encoded_frame_bytes: max.load(Ordering::Relaxed),
        };
        assert_eq!(evidence.encoded_frame_count, 3);
        assert_eq!(evidence.max_encoded_frame_bytes, frames[1].encoded_len());
        validate_variable_frame_evidence(evidence).unwrap();

        let oversized =
            VariableFrameEvidence { max_encoded_frame_bytes: FLIGHT_MAX_ENCODED_FRAME_BYTES + 1, ..evidence };
        let error = validate_variable_frame_evidence(oversized).unwrap_err();
        assert!(error.to_string().contains("above the 4194304 byte limit"), "{error}");
    }
}
