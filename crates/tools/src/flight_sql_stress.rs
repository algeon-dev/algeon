//! Concurrent Flight SQL stress driver.
//!
//! `flight-sql-stress` drives a Flight SQL server from a recorded profile and
//! records one bounded row per query attempt in `attempts.tsv` plus a
//! `summary.json` reduction.  A profile names the exact in-process
//! `run-artifact.json` cell it matches; the summary fails closed on a shape
//! mismatch and reports the server-minus-in-process throughput/latency delta.
//!
//! Load semantics:
//! - Closed-loop: `clients` worker tasks each run attempts
//!   sequentially until the shared deadline. The deadline stops new
//!   submissions; an in-flight attempt always runs to its terminal state, so
//!   the measured window can overrun the configured duration slightly.
//! - Open-loop: a deterministic schedule of uniformly
//!   spaced submit instants (period 1/R from run start; no RNG, so the
//!   schedule is fixed entirely by the configuration). Each submit spawns a
//!   task bounded by a semaphore of `clients` permits, so at most `clients`
//!   attempts are in flight; under saturation submit instants slip and
//!   missed schedule slots are not replayed in bursts. The achievable rate is
//!   additionally bounded by the tokio timer resolution.
//!
//! Connection modes: pooled (default) connects one client per worker slot
//! once with the long `connect_with_retry` budget used by
//! `flight-sql-query`; churn connects a fresh client per attempt with a short
//! retry budget because the server is already expected to accept connections.
//!
//! Ticket cancellation: every Nth attempt per worker
//! (closed-loop) or every Nth scheduled attempt (open-loop) starts the DoGet
//! stream of the first endpoint ticket, waits `--cancel-after-ms` (default
//! 50), then cancels: first via the standard `CancelFlightInfo` action on
//! the attempt's FlightInfo; when the server does not support that action
//! (any action error, including `Unimplemented`/`InvalidArgument`), by
//! dropping the DoGet stream, which the server treats as statement
//! cancellation. Outcome classification after a cancel attempt:
//! terminal tonic `Cancelled` stream error or a client-initiated stream drop
//! records `cancelled`; a cleanly drained stream means the query won the
//! race and records `succeeded`; any other terminal error records `failed`.
//! The `cancel_path` column records `none`, `flight_info`, or `drop`.
//!
//! Output discipline: `--out-dir` must not already exist (create-new
//! semantics). `attempts.tsv` carries the `ATTEMPT_COLUMNS` header plus one
//! row per attempt; `summary.json` retains only a redacted profile projection,
//! outcome counts, nearest-rank latency percentiles, a matched in-process
//! reduction, and their delta. SQL text, raw profile contents, URL userinfo,
//! and raw error detail are never persisted.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use arrow_flight::error::FlightError;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use crate::SINGLE_NODE_RUN_ARTIFACT_SCHEMA;

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const NO_ERROR_CLASS: &str = "none";
const PROFILE_SCHEMA: &str = "algeon_datafusion.flight_sql_stress.run_profile.v1";
const PROFILE_VERSION: u32 = 1;
const ARTIFACT_SCHEMA: &str = "algeon_datafusion.flight_sql_stress.run_artifact.v1";
const ARTIFACT_VERSION: u32 = 1;
const USAGE: &str = "flight-sql-stress --config flight-stress-profile.json --out-dir DIR";

const ATTEMPT_COLUMNS: &[&str] = &[
    "sequence",
    "worker",
    "label",
    "mode",
    "phase",
    "start_offset_ns",
    "latency_ns",
    "outcome",
    "error_class",
    "rows",
    "cancel_path",
];

/// How an attempt obtained its Flight SQL connection.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ConnectionMode {
    Pooled,
    Churn,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArrivalMode {
    Concurrency,
    Rate,
}

impl ArrivalMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Concurrency => "concurrency",
            Self::Rate => "rate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptPhase {
    Warmup,
    Measurement,
}

impl AttemptPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measurement => "measurement",
        }
    }
}

impl ConnectionMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pooled => "pooled",
            Self::Churn => "churn",
        }
    }
}

/// Terminal state of one attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptOutcome {
    Succeeded,
    Failed,
    Cancelled,
}

impl AttemptOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Which cancellation path effected (or was attempted for) a cancel attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CancelPath {
    None,
    FlightInfo,
    Drop,
}

impl CancelPath {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FlightInfo => "flight_info",
            Self::Drop => "drop",
        }
    }
}

/// Ticket-cancellation schedule: every `every_n`th attempt waits `after_ms`
/// after starting the DoGet stream before cancelling.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CancelConfig {
    every_n: u64,
    after_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightStressProfile {
    schema: String,
    version: u32,
    name: String,
    server: FlightServerProfile,
    statement: FlightStatementProfile,
    load: FlightLoadProfile,
    matched_in_process: MatchedInProcessProfile,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightServerProfile {
    url: String,
    config_file: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightStatementProfile {
    label: String,
    sql: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightLoadProfile {
    arrival_mode: ArrivalMode,
    clients: usize,
    warmup_seconds: u64,
    measurement_seconds: u64,
    #[serde(default)]
    rate_per_second: Option<u64>,
    connection_mode: ConnectionMode,
    #[serde(default)]
    cancel: Option<CancelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchedInProcessProfile {
    run_artifact: PathBuf,
    workload: MatchedWorkloadProfile,
    #[serde(default)]
    step_index: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchedWorkloadProfile {
    kind: String,
    query_label: String,
}

impl FlightStressProfile {
    fn parse_bytes(bytes: &[u8]) -> Result<Self, String> {
        let raw: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| "--config must contain valid JSON".to_string())?;
        ensure_redacted_json(&raw, "")?;
        let profile: Self =
            serde_json::from_value(raw).map_err(|error| format!("invalid Flight stress profile: {error}"))?;
        profile.validate()?;
        Ok(profile)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != PROFILE_SCHEMA {
            return Err(format!("Flight stress profile schema must be {PROFILE_SCHEMA}"));
        }
        if self.version != PROFILE_VERSION {
            return Err(format!("Flight stress profile version must be {PROFILE_VERSION}"));
        }
        if !is_identifier(&self.name) {
            return Err("profile.name must be a 1-64 character ASCII identifier".to_string());
        }
        validate_url(&self.server.url)?;
        if self.server.config_file.as_os_str().is_empty() {
            return Err("server.config_file must not be empty".to_string());
        }
        if !is_identifier(&self.statement.label) {
            return Err("statement.label must be a 1-64 character ASCII identifier".to_string());
        }
        if self.statement.sql.trim().is_empty() || self.statement.sql.len() > 16 * 1024 {
            return Err("statement.sql must be 1-16384 non-whitespace bytes".to_string());
        }
        if self.load.clients == 0 {
            return Err("load.clients must be greater than zero".to_string());
        }
        if self.load.measurement_seconds == 0 {
            return Err("load.measurement_seconds must be greater than zero".to_string());
        }
        self.load
            .warmup_seconds
            .checked_add(self.load.measurement_seconds)
            .ok_or_else(|| "load warmup_seconds + measurement_seconds overflows".to_string())?;
        match (self.load.arrival_mode, self.load.rate_per_second) {
            (ArrivalMode::Concurrency, None) => {},
            (ArrivalMode::Concurrency, Some(_)) => {
                return Err("load.rate_per_second is only valid for load.arrival_mode=rate".to_string());
            },
            (ArrivalMode::Rate, Some(rate)) if rate > 0 => {},
            (ArrivalMode::Rate, _) => {
                return Err("rate load requires a non-zero load.rate_per_second".to_string());
            },
        }
        if let Some(cancel) = self.load.cancel
            && cancel.every_n == 0
        {
            return Err("load.cancel.every_n must be greater than zero".to_string());
        }
        if self.matched_in_process.run_artifact.as_os_str().is_empty() {
            return Err("matched_in_process.run_artifact must not be empty".to_string());
        }
        if !is_identifier(&self.matched_in_process.workload.kind) {
            return Err("matched_in_process.workload.kind must be a 1-64 character ASCII identifier".to_string());
        }
        if !is_identifier(&self.matched_in_process.workload.query_label) {
            return Err("matched_in_process.workload.query_label must be a 1-64 character ASCII identifier".to_string());
        }
        if self.statement.label != self.matched_in_process.workload.query_label {
            return Err("statement.label must equal matched_in_process.workload.query_label".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StressConfig {
    /// Full connection URL. Never persisted; only `url_host_port` is echoed.
    url: String,
    /// URL redacted to `host:port` (no userinfo, path, or query).
    url_host_port: String,
    /// SQL executed per attempt. Never persisted; only `sql_label` is.
    sql: String,
    sql_label: String,
    profile_name: String,
    profile_sha256: String,
    server_config_sha256: String,
    clients: usize,
    warmup_seconds: u64,
    measurement_seconds: u64,
    arrival_mode: ArrivalMode,
    rate_per_second: Option<u64>,
    connection_mode: ConnectionMode,
    cancel: Option<CancelConfig>,
    matched_in_process: MatchedInProcess,
    out_dir: PathBuf,
}

impl StressConfig {
    fn mode(&self) -> ConnectionMode {
        self.connection_mode
    }

    fn duration(&self) -> Duration {
        Duration::from_secs(self.warmup_seconds + self.measurement_seconds)
    }

    fn phase_for_offset(&self, offset: Duration) -> AttemptPhase {
        if offset < Duration::from_secs(self.warmup_seconds) { AttemptPhase::Warmup } else { AttemptPhase::Measurement }
    }
}

fn parse_error(reason: impl Into<String>) -> String {
    format!("{}\nusage: {USAGE}", reason.into())
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    let mut config: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut seen = BTreeSet::new();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let flag = arg.as_str();
        if !matches!(flag, "--config" | "--out-dir") {
            return Err(parse_error(format!("unknown flag `{flag}`")));
        }
        if !seen.insert(flag) {
            return Err(parse_error(format!("duplicate flag `{flag}`")));
        }
        let value = match iter.next() {
            Some(value) if !value.starts_with("--") => PathBuf::from(value),
            _ => return Err(parse_error(format!("flag `{flag}` requires a value"))),
        };
        match flag {
            "--config" => config = Some(value),
            "--out-dir" => out_dir = Some(value),
            _ => unreachable!("flag validated above"),
        }
    }
    Ok((
        config.ok_or_else(|| parse_error("missing required flag `--config`"))?,
        out_dir.ok_or_else(|| parse_error("missing required flag `--out-dir`"))?,
    ))
}

/// Redact a connection URL to `host:port` so artifacts never carry userinfo,
/// path, or query components.
fn validate_url(url: &str) -> Result<String, String> {
    let parsed = url::Url::parse(url).map_err(|_| "server.url must be an absolute http(s) URL".to_string())?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("server.url must use http or https".to_string());
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("server.url must not include credentials".to_string());
    }
    if parsed.query().is_some() || parsed.fragment().is_some() || parsed.path() != "/" {
        return Err("server.url must not include a path, query, or fragment".to_string());
    }
    let host = parsed.host_str().ok_or_else(|| "server.url must include a host".to_string())?;
    let port = parsed.port_or_known_default().ok_or_else(|| "server.url must use http or https".to_string())?;
    Ok(format!("{host}:{port}"))
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// The selected in-process cell. These are the only baseline facts copied
/// into a Flight artifact; the source artifact itself remains the authority.
#[derive(Debug, Clone)]
struct MatchedInProcess {
    artifact_sha256: String,
    profile_name: String,
    profile_sha256: String,
    workload_kind: String,
    query_label: String,
    step_index: Option<usize>,
    measurement_seconds: f64,
    application_qps: f64,
    gpu_reaching_qps: f64,
    latency_p50_ns: u64,
    latency_p95_ns: u64,
    latency_p99_ns: u64,
}

pub(crate) fn read_stress_config(config_path: &Path, out_dir: PathBuf) -> Result<StressConfig, String> {
    let profile_bytes = fs::read(config_path).map_err(|_| format!("cannot read --config {}", config_path.display()))?;
    let profile = FlightStressProfile::parse_bytes(&profile_bytes)?;
    let profile_sha256 = sha256_bytes(&profile_bytes);
    let profile_dir =
        config_path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let server_config_path = if profile.server.config_file.is_absolute() {
        profile.server.config_file.clone()
    } else {
        profile_dir.join(&profile.server.config_file)
    };
    let server_config = fs::read(&server_config_path)
        .map_err(|_| format!("cannot read server.config_file {}", server_config_path.display()))?;
    ensure_redacted_server_config(&server_config)?;
    let baseline_path = if profile.matched_in_process.run_artifact.is_absolute() {
        profile.matched_in_process.run_artifact.clone()
    } else {
        profile_dir.join(&profile.matched_in_process.run_artifact)
    };
    let baseline_bytes = fs::read(&baseline_path)
        .map_err(|_| format!("cannot read matched_in_process.run_artifact {}", baseline_path.display()))?;
    let baseline_json: serde_json::Value = serde_json::from_slice(&baseline_bytes)
        .map_err(|_| "matched_in_process.run_artifact must contain valid JSON".to_string())?;
    ensure_redacted_json(&baseline_json, "")?;
    let matched_in_process = matched_in_process_from_value(
        &baseline_json,
        sha256_bytes(&baseline_bytes),
        &profile.matched_in_process,
        &profile.load,
    )?;
    let url_host_port = validate_url(&profile.server.url)?;
    Ok(StressConfig {
        url: profile.server.url,
        url_host_port,
        sql: profile.statement.sql,
        sql_label: profile.statement.label,
        profile_name: profile.name,
        profile_sha256,
        server_config_sha256: sha256_bytes(&server_config),
        clients: profile.load.clients,
        warmup_seconds: profile.load.warmup_seconds,
        measurement_seconds: profile.load.measurement_seconds,
        arrival_mode: profile.load.arrival_mode,
        rate_per_second: profile.load.rate_per_second,
        connection_mode: profile.load.connection_mode,
        cancel: profile.load.cancel,
        matched_in_process,
        out_dir,
    })
}

fn ensure_redacted_server_config(bytes: &[u8]) -> Result<(), String> {
    let source = std::str::from_utf8(bytes).map_err(|_| "server.config_file must be UTF-8".to_string())?;
    if has_inline_secret(source) {
        return Err("server.config_file contains credential-like text".to_string());
    }
    for line in source.lines() {
        let key = line.split_once('=').map_or(line, |(key, _)| key).trim();
        if !key.is_empty() && !key.starts_with('#') && is_sensitive_key(key) {
            return Err("server.config_file contains a credential-like field".to_string());
        }
    }
    Ok(())
}

fn matched_in_process_from_value(
    artifact: &serde_json::Value,
    artifact_sha256: String,
    expected: &MatchedInProcessProfile,
    load: &FlightLoadProfile,
) -> Result<MatchedInProcess, String> {
    let schema = string_at(artifact, "/schema", "matched in-process artifact")?;
    if schema != SINGLE_NODE_RUN_ARTIFACT_SCHEMA {
        return Err(format!("matched in-process artifact schema must be {SINGLE_NODE_RUN_ARTIFACT_SCHEMA}"));
    }
    if string_at(artifact, "/status", "matched in-process artifact")? != "measured_run" {
        return Err("matched in-process artifact status must be measured_run".to_string());
    }

    let profile_name = string_at(artifact, "/profile/name", "matched in-process artifact")?.to_string();
    let profile_sha256 = string_at(artifact, "/profile/sha256", "matched in-process artifact")?.to_string();
    if !is_sha256(&profile_sha256) {
        return Err("matched in-process artifact profile.sha256 must be a SHA-256 hex digest".to_string());
    }
    let scheduling = value_at(artifact, "/profile/scheduling", "matched in-process artifact")?;
    let arrival_mode = string_at(scheduling, "/arrival_mode", "matched in-process profile scheduling")?;
    if arrival_mode != load.arrival_mode.as_str() {
        return Err("Flight load.arrival_mode does not match the selected in-process cell".to_string());
    }
    if u64_at(scheduling, "/warmup_seconds", "matched in-process profile scheduling")? != load.warmup_seconds {
        return Err("Flight load.warmup_seconds does not match the in-process profile".to_string());
    }

    let workloads = array_at(artifact, "/profile/workloads", "matched in-process artifact")?;
    let selected_workload = workloads.iter().find(|workload| {
        workload.get("kind").and_then(serde_json::Value::as_str) == Some(expected.workload.kind.as_str())
    });
    let Some(selected_workload) = selected_workload else {
        return Err("matched in-process artifact does not select the requested workload kind".to_string());
    };
    let selected_query =
        selected_workload.get("queries").and_then(serde_json::Value::as_array).is_some_and(|queries| {
            queries.iter().any(|query| query.as_str() == Some(expected.workload.query_label.as_str()))
        });
    if !selected_query {
        return Err("matched in-process artifact does not select the requested query label".to_string());
    }
    let selected_query_count = workloads
        .iter()
        .filter_map(|workload| workload.get("queries").and_then(serde_json::Value::as_array))
        .map(Vec::len)
        .sum::<usize>();
    if selected_query_count != 1 {
        return Err("matched in-process artifact must select exactly one query for a one-statement Flight comparison"
            .to_string());
    }

    let steps = scheduling.get("steps").and_then(serde_json::Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let (cell, aggregate, step_index) = if steps.is_empty() {
        if expected.step_index.is_some() {
            return Err("matched_in_process.step_index is only valid for an in-process steps profile".to_string());
        }
        let aggregate = value_at(artifact, "/execution/measurement/aggregate", "matched in-process artifact")?;
        (scheduling, aggregate, None)
    } else {
        let index = expected
            .step_index
            .ok_or_else(|| "matched_in_process.step_index is required for an in-process steps profile".to_string())?;
        let cell = steps
            .get(index)
            .ok_or_else(|| "matched_in_process.step_index is outside the in-process profile steps".to_string())?;
        let aggregate = value_at(
            artifact,
            &format!("/execution/measurement/aggregate/steps/{index}"),
            "matched in-process artifact",
        )?;
        (cell, aggregate, Some(index))
    };
    validate_load_cell(cell, load)?;

    let throughput = value_at(aggregate, "/throughput", "matched in-process aggregate")?;
    let latency = value_at(aggregate, "/latency", "matched in-process aggregate")?;
    let measurement_seconds = finite_f64_at(throughput, "/measurement_seconds", "matched in-process throughput")?;
    if (measurement_seconds - load.measurement_seconds as f64).abs() > f64::EPSILON {
        return Err("matched in-process aggregate measurement_seconds does not match the selected cell".to_string());
    }
    Ok(MatchedInProcess {
        artifact_sha256,
        profile_name,
        profile_sha256,
        workload_kind: expected.workload.kind.clone(),
        query_label: expected.workload.query_label.clone(),
        step_index,
        measurement_seconds,
        application_qps: finite_f64_at(throughput, "/application_qps", "matched in-process throughput")?,
        gpu_reaching_qps: finite_f64_at(throughput, "/gpu_reaching_qps", "matched in-process throughput")?,
        latency_p50_ns: u64_at(latency, "/p50_ns", "matched in-process latency")?,
        latency_p95_ns: u64_at(latency, "/p95_ns", "matched in-process latency")?,
        latency_p99_ns: u64_at(latency, "/p99_ns", "matched in-process latency")?,
    })
}

fn validate_load_cell(cell: &serde_json::Value, load: &FlightLoadProfile) -> Result<(), String> {
    if u64_at(cell, "/measurement_seconds", "selected in-process cell")? != load.measurement_seconds {
        return Err("Flight load.measurement_seconds does not match the selected in-process cell".to_string());
    }
    match load.arrival_mode {
        ArrivalMode::Concurrency => {
            let concurrency = u64_at(cell, "/concurrency", "selected in-process cell")?;
            if concurrency != load.clients as u64 {
                return Err("Flight load.clients does not match selected in-process concurrency".to_string());
            }
        },
        ArrivalMode::Rate => {
            let rate = u64_at(cell, "/arrival_rate_per_second", "selected in-process cell")?;
            if Some(rate) != load.rate_per_second {
                return Err("Flight load.rate_per_second does not match the selected in-process rate".to_string());
            }
        },
    }
    Ok(())
}

fn value_at<'a>(value: &'a serde_json::Value, pointer: &str, context: &str) -> Result<&'a serde_json::Value, String> {
    value.pointer(pointer).ok_or_else(|| format!("{context} is missing {pointer}"))
}

fn string_at<'a>(value: &'a serde_json::Value, pointer: &str, context: &str) -> Result<&'a str, String> {
    value_at(value, pointer, context)?.as_str().ok_or_else(|| format!("{context} field {pointer} must be a string"))
}

fn u64_at(value: &serde_json::Value, pointer: &str, context: &str) -> Result<u64, String> {
    value_at(value, pointer, context)?
        .as_u64()
        .ok_or_else(|| format!("{context} field {pointer} must be a non-negative integer"))
}

fn finite_f64_at(value: &serde_json::Value, pointer: &str, context: &str) -> Result<f64, String> {
    match value_at(value, pointer, context)?.as_f64() {
        Some(number) if number.is_finite() && number >= 0.0 => Ok(number),
        _ => Err(format!("{context} field {pointer} must be a finite non-negative number")),
    }
}

fn array_at<'a>(value: &'a serde_json::Value, pointer: &str, context: &str) -> Result<&'a [serde_json::Value], String> {
    value_at(value, pointer, context)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{context} field {pointer} must be an array"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// The profile and the imported in-process artifact are both untrusted input.
/// Reject common secret-bearing fields and credential-shaped strings before
/// parsing, and re-apply this guard to the output before it is written.
fn ensure_redacted_json(value: &serde_json::Value, pointer: &str) -> Result<(), String> {
    match value {
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                let field = format!("{pointer}/{}", key.replace('~', "~0").replace('/', "~1"));
                if is_sensitive_key(key) {
                    return Err(format!("credential-like field is forbidden at {field}"));
                }
                ensure_redacted_json(value, &field)?;
            }
        },
        serde_json::Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                ensure_redacted_json(value, &format!("{pointer}/{index}"))?;
            }
        },
        serde_json::Value::String(value) if has_inline_secret(value) => {
            return Err(format!("credential-like text is forbidden at {pointer}"));
        },
        _ => {},
    }
    Ok(())
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '_'], "");
    [
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "credential",
        "apikey",
        "privatekey",
        "accesskey",
        "sessionkey",
        "cookie",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn has_inline_secret(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("://")
        && normalized
            .split("://")
            .nth(1)
            .is_some_and(|rest| rest.split('/').next().is_some_and(|authority| authority.contains('@')))
        || normalized.contains("-----begin") && normalized.contains("private key-----")
        || normalized.contains("bearer ")
        || normalized.contains("x-amz-signature=")
}

/// Bounded per-attempt observation. Error detail is a class label only.
#[derive(Debug, Clone)]
struct AttemptRecord {
    sequence: u64,
    worker: usize,
    label: String,
    mode: ConnectionMode,
    phase: AttemptPhase,
    start_offset_ns: u64,
    latency_ns: u64,
    outcome: AttemptOutcome,
    error_class: String,
    rows: u64,
    cancel_path: CancelPath,
}

impl AttemptRecord {
    fn to_tsv(&self) -> String {
        let fields = [
            self.sequence.to_string(),
            self.worker.to_string(),
            tsv_field(&self.label),
            self.mode.as_str().to_string(),
            self.phase.as_str().to_string(),
            self.start_offset_ns.to_string(),
            self.latency_ns.to_string(),
            self.outcome.as_str().to_string(),
            self.error_class.clone(),
            self.rows.to_string(),
            self.cancel_path.as_str().to_string(),
        ];
        debug_assert_eq!(fields.len(), ATTEMPT_COLUMNS.len());
        fields.join("\t")
    }
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

/// Bounded error classification: tonic code names for RPC failures, fixed
/// labels for non-tonic flight and transport failures. Never raw text.
fn flight_error_class(error: &FlightError) -> String {
    match error {
        FlightError::Tonic(status) => format!("{:?}", status.code()),
        FlightError::Arrow(_) => "flight_arrow".to_string(),
        FlightError::NotYetImplemented(_) => "flight_not_yet_implemented".to_string(),
        FlightError::ProtocolError(_) => "flight_protocol".to_string(),
        FlightError::DecodeError(_) => "flight_decode".to_string(),
        FlightError::ExternalError(_) => "flight_external".to_string(),
    }
}

#[derive(Debug, Default)]
struct ModeCounts {
    attempts: u64,
    succeeded: u64,
    failed: u64,
    cancelled: u64,
}

#[derive(Debug, Default)]
struct Reduction {
    attempts: u64,
    succeeded: u64,
    failed: u64,
    cancelled: u64,
    error_classes: BTreeMap<String, u64>,
    cancel_paths: BTreeMap<&'static str, u64>,
    pooled: ModeCounts,
    churn: ModeCounts,
    succeeded_latencies_ns: Vec<u64>,
}

fn reduce(records: &[AttemptRecord]) -> Reduction {
    let mut reduction = Reduction::default();
    for record in records {
        reduction.attempts += 1;
        let mode_counts = match record.mode {
            ConnectionMode::Pooled => &mut reduction.pooled,
            ConnectionMode::Churn => &mut reduction.churn,
        };
        mode_counts.attempts += 1;
        match record.outcome {
            AttemptOutcome::Succeeded => {
                reduction.succeeded += 1;
                mode_counts.succeeded += 1;
                reduction.succeeded_latencies_ns.push(record.latency_ns);
            },
            AttemptOutcome::Failed => {
                reduction.failed += 1;
                mode_counts.failed += 1;
            },
            AttemptOutcome::Cancelled => {
                reduction.cancelled += 1;
                mode_counts.cancelled += 1;
            },
        }
        if record.error_class != NO_ERROR_CLASS {
            *reduction.error_classes.entry(record.error_class.clone()).or_default() += 1;
        }
        *reduction.cancel_paths.entry(record.cancel_path.as_str()).or_default() += 1;
    }
    reduction
}

/// Nearest-rank percentile: the value at rank `ceil(q * n)` of sorted values.
fn percentile_nearest_rank(values: &[u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((quantile * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    Some(sorted[rank - 1])
}

fn mode_summary(counts: &ModeCounts) -> serde_json::Value {
    serde_json::json!({
        "attempts": counts.attempts,
        "succeeded": counts.succeeded,
        "failed": counts.failed,
        "cancelled": counts.cancelled,
    })
}

fn build_summary(config: &StressConfig, records: &[AttemptRecord], wall: Duration) -> serde_json::Value {
    let warmup_attempts = records.iter().filter(|record| record.phase == AttemptPhase::Warmup).count() as u64;
    let measured =
        records.iter().filter(|record| record.phase == AttemptPhase::Measurement).cloned().collect::<Vec<_>>();
    let reduction = reduce(&measured);
    let measurement_seconds = config.measurement_seconds as f64;
    let application_qps = reduction.succeeded as f64 / measurement_seconds;
    let offered_qps = reduction.attempts as f64 / measurement_seconds;
    let latencies = &reduction.succeeded_latencies_ns;
    serde_json::json!({
        "schema": ARTIFACT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "profile": {
            "name": config.profile_name,
            "sha256": config.profile_sha256,
        },
        "config": {
            "url": config.url_host_port,
            "server_config_sha256": config.server_config_sha256,
            "sql_label": config.sql_label,
            "clients": config.clients,
            "arrival_mode": config.arrival_mode.as_str(),
            "warmup_seconds": config.warmup_seconds,
            "measurement_seconds": config.measurement_seconds,
            "rate_per_second": config.rate_per_second,
            "mode": config.mode().as_str(),
            "cancel": config
                .cancel
                .map(|cancel| serde_json::json!({ "every_n": cancel.every_n, "after_ms": cancel.after_ms })),
        },
        "window": {
            "elapsed_until_all_attempts_seconds": wall.as_secs_f64(),
            "warmup_attempts": warmup_attempts,
            "measurement_attempts": reduction.attempts,
        },
        "attempts": reduction.attempts,
        "succeeded": reduction.succeeded,
        "failed": reduction.failed,
        "cancelled": reduction.cancelled,
        "throughput": {
            "measurement_seconds": measurement_seconds,
            "offered_qps": offered_qps,
            "successful_completions": reduction.succeeded,
            "application_qps": application_qps,
        },
        "latency_ns": {
            "samples": latencies.len(),
            "p50": percentile_nearest_rank(latencies, 0.50),
            "p95": percentile_nearest_rank(latencies, 0.95),
            "p99": percentile_nearest_rank(latencies, 0.99),
        },
        "error_classes": reduction.error_classes,
        "cancel_paths": reduction.cancel_paths,
        "modes": {
            "pooled": mode_summary(&reduction.pooled),
            "churn": mode_summary(&reduction.churn),
        },
        "matched_in_process": {
            "artifact_sha256": config.matched_in_process.artifact_sha256,
            "profile": {
                "name": config.matched_in_process.profile_name,
                "sha256": config.matched_in_process.profile_sha256,
            },
            "selection": {
                "workload_kind": config.matched_in_process.workload_kind,
                "query_label": config.matched_in_process.query_label,
                "step_index": config.matched_in_process.step_index,
            },
            "aggregate": {
                "measurement_seconds": config.matched_in_process.measurement_seconds,
                "application_qps": config.matched_in_process.application_qps,
                "gpu_reaching_qps": config.matched_in_process.gpu_reaching_qps,
                "latency_ns": {
                    "p50": config.matched_in_process.latency_p50_ns,
                    "p95": config.matched_in_process.latency_p95_ns,
                    "p99": config.matched_in_process.latency_p99_ns,
                },
            },
        },
        "delta_from_in_process": {
            "application_qps": floating_delta(application_qps, config.matched_in_process.application_qps),
            "latency_ns": {
                "p50": latency_delta(percentile_nearest_rank(latencies, 0.50), config.matched_in_process.latency_p50_ns),
                "p95": latency_delta(percentile_nearest_rank(latencies, 0.95), config.matched_in_process.latency_p95_ns),
                "p99": latency_delta(percentile_nearest_rank(latencies, 0.99), config.matched_in_process.latency_p99_ns),
            },
            "gpu_reaching_qps": {
                "in_process": config.matched_in_process.gpu_reaching_qps,
                "server": serde_json::Value::Null,
                "status": "not_observable_over_flight",
            },
        },
    })
}

fn floating_delta(server: f64, in_process: f64) -> serde_json::Value {
    serde_json::json!({
        "server": server,
        "in_process": in_process,
        "server_minus_in_process": server - in_process,
        "percent_of_in_process": (in_process != 0.0).then(|| (server - in_process) * 100.0 / in_process),
    })
}

fn latency_delta(server: Option<u64>, in_process: u64) -> serde_json::Value {
    match server {
        Some(server) => serde_json::json!({
            "server": server,
            "in_process": in_process,
            "server_minus_in_process": i128::from(server) - i128::from(in_process),
            "percent_of_in_process": (in_process != 0)
                .then(|| (server as f64 - in_process as f64) * 100.0 / in_process as f64),
        }),
        None => serde_json::json!({
            "server": serde_json::Value::Null,
            "in_process": in_process,
            "server_minus_in_process": serde_json::Value::Null,
            "percent_of_in_process": serde_json::Value::Null,
        }),
    }
}

#[cfg(not(test))]
use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

#[cfg(not(test))]
use arrow_flight::{Action, CancelFlightInfoRequest, FlightInfo, Ticket, sql::client::FlightSqlServiceClient};
#[cfg(not(test))]
use futures::TryStreamExt;
#[cfg(not(test))]
use prost::Message;
#[cfg(not(test))]
use tokio::{
    sync::{Mutex, Semaphore, mpsc},
    task::JoinSet,
    time::{MissedTickBehavior, interval, sleep},
};
#[cfg(not(test))]
use tonic::transport::Channel;

#[cfg(not(test))]
use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};

#[cfg(not(test))]
type FlightClient = FlightSqlServiceClient<Channel>;

/// Pooled worker-slot connects mirror flight_sql_query's long budget: the
/// server may still be starting when the run begins.
#[cfg(not(test))]
const POOLED_CONNECT_ATTEMPTS: usize = 600;
/// Churn connects expect the server to accept connections already.
#[cfg(not(test))]
const CHURN_CONNECT_ATTEMPTS: usize = 3;
#[cfg(not(test))]
const CONNECT_BACKOFF: Duration = Duration::from_millis(100);
#[cfg(not(test))]
const TRANSPORT_ERROR_CLASS: &str = "transport";

#[cfg(not(test))]
pub async fn run(args: Vec<String>) -> ToolResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        eprintln!("{USAGE}");
        return Ok(());
    }
    let (profile_path, out_dir) = parse_args(&args)?;
    let config = read_stress_config(&profile_path, out_dir)?;
    std::fs::create_dir(&config.out_dir).map_err(|error| {
        format!(
            "flight-sql-stress out-dir `{}` must not already exist and have an existing parent: {error}",
            config.out_dir.display()
        )
    })?;
    let started = Instant::now();
    let deadline =
        started.checked_add(config.duration()).ok_or("flight-sql-stress duration exceeds the monotonic clock range")?;
    let (record_tx, record_rx) = mpsc::unbounded_channel::<AttemptRecord>();
    let sequence = Arc::new(AtomicU64::new(0));
    let config = Arc::new(config);
    match config.arrival_mode {
        ArrivalMode::Rate => {
            let rate = config.rate_per_second.ok_or("validated rate profile lost its rate")? as f64;
            run_open_loop(Arc::clone(&config), started, deadline, rate, record_tx, sequence).await?;
        },
        ArrivalMode::Concurrency => {
            run_closed_loop(Arc::clone(&config), started, deadline, record_tx, sequence).await?
        },
    }
    // Both load loops consume their sender once every worker/task clone is
    // dropped, so the receiver observes a closed channel here.
    let mut records = Vec::new();
    let mut record_rx = record_rx;
    while let Some(record) = record_rx.recv().await {
        records.push(record);
    }
    records.sort_by_key(|record| record.sequence);
    let wall = started.elapsed();
    write_outputs(&config, &records, wall)?;
    let measured = records.iter().filter(|record| record.phase == AttemptPhase::Measurement).collect::<Vec<_>>();
    eprintln!(
        "flight-sql-stress: {} measured attempts ({} succeeded, {} failed, {} cancelled) in {:.3}s against {} -> {}",
        measured.len(),
        measured.iter().filter(|record| record.outcome == AttemptOutcome::Succeeded).count(),
        measured.iter().filter(|record| record.outcome == AttemptOutcome::Failed).count(),
        measured.iter().filter(|record| record.outcome == AttemptOutcome::Cancelled).count(),
        wall.as_secs_f64(),
        config.url_host_port,
        config.out_dir.display(),
    );
    Ok(())
}

#[cfg(not(test))]
async fn connect(url: &str, attempts: usize, url_host_port: &str) -> ToolResult<FlightClient> {
    connect_with_retry(
        url,
        ConnectionRetryConfig::fixed_attempts(attempts, CONNECT_BACKOFF),
        FlightSqlServiceClient::new,
        |error| format!("Flight SQL server did not accept connections at {url_host_port}: {:?}", Some(error)),
    )
    .await
}

#[cfg(not(test))]
async fn run_closed_loop(
    config: Arc<StressConfig>,
    started: Instant,
    deadline: Instant,
    record_tx: mpsc::UnboundedSender<AttemptRecord>,
    sequence: Arc<AtomicU64>,
) -> ToolResult<()> {
    let mut workers = Vec::new();
    for worker in 0..config.clients {
        let config = Arc::clone(&config);
        let record_tx = record_tx.clone();
        let sequence = Arc::clone(&sequence);
        workers.push(tokio::spawn(async move {
            closed_loop_worker(worker, config, started, deadline, record_tx, sequence).await
        }));
    }
    for worker in workers {
        worker.await.map_err(|error| format!("flight-sql-stress worker task failed to join: {error}"))??;
    }
    Ok(())
}

#[cfg(not(test))]
async fn closed_loop_worker(
    worker: usize,
    config: Arc<StressConfig>,
    started: Instant,
    deadline: Instant,
    record_tx: mpsc::UnboundedSender<AttemptRecord>,
    sequence: Arc<AtomicU64>,
) -> ToolResult<()> {
    let mut pooled = match config.mode() {
        ConnectionMode::Churn => None,
        ConnectionMode::Pooled => Some(connect(&config.url, POOLED_CONNECT_ATTEMPTS, &config.url_host_port).await?),
    };
    let mut attempt_index = 0u64;
    while Instant::now() < deadline {
        attempt_index += 1;
        let cancel = config.cancel.filter(|cancel| attempt_index.is_multiple_of(cancel.every_n));
        let sequence_number = sequence.fetch_add(1, Ordering::Relaxed);
        let record = match pooled.as_mut() {
            Some(client) => attempt_with_client(client, &config, worker, cancel, sequence_number, started).await,
            None => attempt_churn(&config, worker, cancel, sequence_number, started).await,
        };
        let _ = record_tx.send(record);
    }
    Ok(())
}

#[cfg(not(test))]
async fn run_open_loop(
    config: Arc<StressConfig>,
    started: Instant,
    deadline: Instant,
    rate: f64,
    record_tx: mpsc::UnboundedSender<AttemptRecord>,
    sequence: Arc<AtomicU64>,
) -> ToolResult<()> {
    // Pooled mode connects one client per slot up front; churn mode leaves
    // the pool empty and connects per attempt.
    let mut pool: Vec<Arc<Mutex<FlightClient>>> = Vec::new();
    if config.mode() == ConnectionMode::Pooled {
        for _ in 0..config.clients {
            pool.push(Arc::new(Mutex::new(
                connect(&config.url, POOLED_CONNECT_ATTEMPTS, &config.url_host_port).await?,
            )));
        }
    }
    // Slot handoff doubles as worker identity for the `worker` column; the
    // semaphore bounds in-flight attempts to `clients` as documented.
    let (slot_tx, mut slot_rx) = mpsc::channel::<usize>(config.clients);
    for slot in 0..config.clients {
        slot_tx.send(slot).await.map_err(|_| "flight-sql-stress slot channel closed unexpectedly")?;
    }
    let semaphore = Arc::new(Semaphore::new(config.clients));
    let mut interval = interval(Duration::from_secs_f64(1.0 / rate));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut scheduled = 0u64;
    let mut in_flight = JoinSet::new();
    loop {
        interval.tick().await;
        if Instant::now() >= deadline {
            break;
        }
        scheduled += 1;
        let cancel = config.cancel.filter(|cancel| scheduled.is_multiple_of(cancel.every_n));
        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .map_err(|_| "flight-sql-stress scheduler semaphore closed unexpectedly")?;
        let slot = slot_rx.recv().await.ok_or("flight-sql-stress slot channel closed unexpectedly")?;
        let sequence_number = sequence.fetch_add(1, Ordering::Relaxed);
        let config = Arc::clone(&config);
        let record_tx = record_tx.clone();
        let slot_tx = slot_tx.clone();
        let pooled_client = pool.get(slot).cloned();
        in_flight.spawn(async move {
            // Held until the task ends so at most `clients` attempts run at once.
            let _permit = permit;
            let record = match pooled_client {
                Some(client) => {
                    let mut guard = client.lock().await;
                    attempt_with_client(&mut guard, &config, slot, cancel, sequence_number, started).await
                },
                None => attempt_churn(&config, slot, cancel, sequence_number, started).await,
            };
            let _ = record_tx.send(record);
            let _ = slot_tx.send(slot).await;
        });
    }
    let mut join_failures = 0usize;
    while let Some(result) = in_flight.join_next().await {
        if result.is_err() {
            join_failures += 1;
        }
    }
    if join_failures > 0 {
        return Err(format!("flight-sql-stress: {join_failures} scheduled attempt task(s) failed to join").into());
    }
    Ok(())
}

/// One attempt against a churn connection: connect fresh, run, drop the
/// connection. A connect failure is a failed attempt, never a run abort.
#[cfg(not(test))]
async fn attempt_churn(
    config: &StressConfig,
    worker: usize,
    cancel: Option<CancelConfig>,
    sequence: u64,
    started: Instant,
) -> AttemptRecord {
    // The churn attempt starts before connecting so the offset marks
    // submission and the latency includes connection establishment.
    let attempt_start = Instant::now();
    match connect(&config.url, CHURN_CONNECT_ATTEMPTS, &config.url_host_port).await {
        Ok(mut client) => {
            attempt_with_client_at(&mut client, config, worker, cancel, sequence, started, attempt_start).await
        },
        Err(_) => base_record(config, worker, sequence, started, attempt_start).finish(
            attempt_start,
            AttemptOutcome::Failed,
            TRANSPORT_ERROR_CLASS,
        ),
    }
}

#[cfg(not(test))]
fn base_record(
    config: &StressConfig,
    worker: usize,
    sequence: u64,
    started: Instant,
    attempt_start: Instant,
) -> AttemptRecord {
    AttemptRecord {
        sequence,
        worker,
        label: config.sql_label.clone(),
        mode: config.mode(),
        phase: config.phase_for_offset(attempt_start.duration_since(started)),
        start_offset_ns: attempt_start.duration_since(started).as_nanos() as u64,
        latency_ns: 0,
        outcome: AttemptOutcome::Failed,
        error_class: NO_ERROR_CLASS.to_string(),
        rows: 0,
        cancel_path: CancelPath::None,
    }
}

#[cfg(not(test))]
impl AttemptRecord {
    fn finish(mut self, attempt_start: Instant, outcome: AttemptOutcome, error_class: impl Into<String>) -> Self {
        self.latency_ns = attempt_start.elapsed().as_nanos() as u64;
        self.outcome = outcome;
        self.error_class = error_class.into();
        self
    }
}

/// Execute the configured SQL once on an established pooled client,
/// optionally cancelling mid-execution per the cancel schedule.
#[cfg(not(test))]
async fn attempt_with_client(
    client: &mut FlightClient,
    config: &StressConfig,
    worker: usize,
    cancel: Option<CancelConfig>,
    sequence: u64,
    started: Instant,
) -> AttemptRecord {
    attempt_with_client_at(client, config, worker, cancel, sequence, started, Instant::now()).await
}

#[cfg(not(test))]
async fn attempt_with_client_at(
    client: &mut FlightClient,
    config: &StressConfig,
    worker: usize,
    cancel: Option<CancelConfig>,
    sequence: u64,
    started: Instant,
    attempt_start: Instant,
) -> AttemptRecord {
    let record = base_record(config, worker, sequence, started, attempt_start);
    let flight_info = match client.execute(config.sql.clone(), None).await {
        Ok(flight_info) => flight_info,
        Err(error) => return record.finish(attempt_start, AttemptOutcome::Failed, flight_error_class(&error)),
    };
    let tickets = flight_info.endpoint.iter().filter_map(|endpoint| endpoint.ticket.clone()).collect::<Vec<Ticket>>();
    match cancel {
        Some(cancel) => cancel_attempt(client, flight_info, &tickets, cancel, record, attempt_start).await,
        None => {
            let mut record = record;
            for ticket in tickets {
                if let Err(error_class) = drain_ticket(client, ticket, &mut record).await {
                    return record.finish(attempt_start, AttemptOutcome::Failed, error_class);
                }
            }
            record.finish(attempt_start, AttemptOutcome::Succeeded, NO_ERROR_CLASS)
        },
    }
}

/// Drain one endpoint ticket to its terminal state, accumulating row counts.
#[cfg(not(test))]
async fn drain_ticket(client: &mut FlightClient, ticket: Ticket, record: &mut AttemptRecord) -> Result<(), String> {
    let mut stream = client.do_get(ticket).await.map_err(|error| flight_error_class(&error))?;
    while let Some(batch) = stream.try_next().await.map_err(|error| flight_error_class(&error))? {
        record.rows += batch.num_rows() as u64;
    }
    Ok(())
}

/// Cancel-scheduled attempt: start the first endpoint's DoGet stream, wait
/// the configured delay, then cancel via `CancelFlightInfo`, falling back to
/// dropping the stream when the server does not support the action.
#[cfg(not(test))]
async fn cancel_attempt(
    client: &mut FlightClient,
    flight_info: FlightInfo,
    tickets: &[Ticket],
    cancel: CancelConfig,
    mut record: AttemptRecord,
    attempt_start: Instant,
) -> AttemptRecord {
    let Some(first_ticket) = tickets.first() else {
        return record.finish(attempt_start, AttemptOutcome::Succeeded, NO_ERROR_CLASS);
    };
    let mut stream = match client.do_get(first_ticket.clone()).await {
        Ok(stream) => stream,
        Err(error) => return record.finish(attempt_start, AttemptOutcome::Failed, flight_error_class(&error)),
    };
    sleep(Duration::from_millis(cancel.after_ms)).await;
    match cancel_via_flight_info(client, flight_info).await {
        Ok(()) => {
            record.cancel_path = CancelPath::FlightInfo;
            loop {
                match stream.try_next().await {
                    Ok(Some(batch)) => record.rows += batch.num_rows() as u64,
                    // The query drained cleanly despite the accepted cancel
                    // request; the query won the race.
                    Ok(None) => return record.finish(attempt_start, AttemptOutcome::Succeeded, NO_ERROR_CLASS),
                    Err(error) => {
                        let outcome = match &error {
                            FlightError::Tonic(status) if status.code() == tonic::Code::Cancelled => {
                                AttemptOutcome::Cancelled
                            },
                            _ => AttemptOutcome::Failed,
                        };
                        return record.finish(attempt_start, outcome, flight_error_class(&error));
                    },
                }
            }
        },
        Err(()) => {
            // Dropping the DoGet stream is the server's statement-cancellation
            // signal; it cannot wait for end of scope because the drop itself
            // is the cancel mechanism being measured.
            drop(stream);
            record.cancel_path = CancelPath::Drop;
            record.finish(attempt_start, AttemptOutcome::Cancelled, NO_ERROR_CLASS)
        },
    }
}

/// Issue the standard `CancelFlightInfo` action, mirroring
/// `arrow_flight::FlightClient::cancel_flight_info`, which the SQL client
/// does not expose. Any error or empty response means the server does not
/// support the action and the caller falls back to the drop path.
#[cfg(not(test))]
async fn cancel_via_flight_info(client: &mut FlightClient, flight_info: FlightInfo) -> Result<(), ()> {
    let request = CancelFlightInfoRequest::new(flight_info);
    let action = Action { r#type: "CancelFlightInfo".to_string(), body: request.encode_to_vec().into() };
    let mut stream = client.do_action(action).await.map_err(|_| ())?;
    stream.try_next().await.map_err(|_| ())?.ok_or(())?;
    Ok(())
}

#[cfg(not(test))]
fn write_outputs(config: &StressConfig, records: &[AttemptRecord], wall: Duration) -> ToolResult<()> {
    let attempts_path = config.out_dir.join("attempts.tsv");
    let mut attempts_file = std::fs::File::create(&attempts_path)?;
    writeln!(attempts_file, "{}", ATTEMPT_COLUMNS.join("\t"))?;
    for record in records {
        writeln!(attempts_file, "{}", record.to_tsv())?;
    }
    let summary = build_summary(config, records, wall);
    ensure_redacted_json(&summary, "")?;
    std::fs::write(config.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASELINE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn profile_json() -> serde_json::Value {
        serde_json::json!({
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "name": "stress-s8-flight-sf1",
            "server": {
                "url": "http://flight.example.com:5050",
                "config_file": "server-s8-s2-scan-sf1.toml"
            },
            "statement": { "label": "q06", "sql": "SELECT 1" },
            "load": {
                "arrival_mode": "concurrency",
                "clients": 8,
                "warmup_seconds": 10,
                "measurement_seconds": 120,
                "connection_mode": "pooled"
            },
            "matched_in_process": {
                "run_artifact": "baseline/run-artifact.json",
                "workload": { "kind": "tpch", "query_label": "q06" }
            }
        })
    }

    fn profile() -> FlightStressProfile {
        FlightStressProfile::parse_bytes(&serde_json::to_vec(&profile_json()).expect("profile JSON")).expect("profile")
    }

    fn baseline_json() -> serde_json::Value {
        serde_json::json!({
            "schema": SINGLE_NODE_RUN_ARTIFACT_SCHEMA,
            "status": "measured_run",
            "profile": {
                "name": "stress-s1-embedded-sf1",
                "sha256": BASELINE_SHA,
                "scheduling": {
                    "arrival_mode": "concurrency",
                    "concurrency": 8,
                    "warmup_seconds": 10,
                    "measurement_seconds": 120,
                    "steps": []
                },
                "workloads": [{ "kind": "tpch", "queries": ["q06"] }]
            },
            "execution": {
                "measurement": {
                    "aggregate": {
                        "throughput": {
                            "measurement_seconds": 120.0,
                            "application_qps": 3.0,
                            "gpu_reaching_qps": 3.0
                        },
                        "latency": { "p50_ns": 10, "p95_ns": 20, "p99_ns": 30 }
                    }
                }
            }
        })
    }

    fn config() -> StressConfig {
        let profile = profile();
        let matched_in_process =
            matched_in_process_from_value(&baseline_json(), "a".repeat(64), &profile.matched_in_process, &profile.load)
                .expect("matched baseline");
        StressConfig {
            url: profile.server.url,
            url_host_port: "flight.example.com:5050".to_string(),
            sql: profile.statement.sql,
            sql_label: profile.statement.label,
            profile_name: profile.name,
            profile_sha256: "b".repeat(64),
            server_config_sha256: "c".repeat(64),
            clients: profile.load.clients,
            warmup_seconds: profile.load.warmup_seconds,
            measurement_seconds: profile.load.measurement_seconds,
            arrival_mode: profile.load.arrival_mode,
            rate_per_second: profile.load.rate_per_second,
            connection_mode: profile.load.connection_mode,
            cancel: profile.load.cancel,
            matched_in_process,
            out_dir: PathBuf::from("out"),
        }
    }

    fn record(
        sequence: u64,
        phase: AttemptPhase,
        mode: ConnectionMode,
        outcome: AttemptOutcome,
        error_class: &str,
        cancel_path: CancelPath,
        latency_ns: u64,
    ) -> AttemptRecord {
        AttemptRecord {
            sequence,
            worker: 0,
            label: "q06".to_string(),
            mode,
            phase,
            start_offset_ns: 0,
            latency_ns,
            outcome,
            error_class: error_class.to_string(),
            rows: 0,
            cancel_path,
        }
    }

    #[test]
    fn cli_requires_only_recorded_config_and_output_location() {
        let parsed = parse_args(&[
            "--config".to_string(),
            "profile.json".to_string(),
            "--out-dir".to_string(),
            "out".to_string(),
        ])
        .expect("profile CLI");
        assert_eq!(parsed, (PathBuf::from("profile.json"), PathBuf::from("out")));
        let error = parse_args(&["--sql".to_string(), "SELECT 1".to_string()]).expect_err("ad-hoc flags rejected");
        assert!(error.contains("unknown flag `--sql`"));
    }

    #[test]
    fn profile_is_validated_and_keeps_sql_out_of_serialized_projection() {
        let profile = profile();
        assert_eq!(profile.load.connection_mode, ConnectionMode::Pooled);
        assert_eq!(validate_url(&profile.server.url).expect("URL"), "flight.example.com:5050");

        let mut invalid = profile_json();
        invalid["server"]["url"] = serde_json::json!("http://user:secret@flight.example.com:5050");
        let error = FlightStressProfile::parse_bytes(&serde_json::to_vec(&invalid).expect("JSON"))
            .expect_err("credentials rejected");
        assert!(error.contains("credential-like"));

        let mut secret_key = profile_json();
        secret_key["server"]["api_token"] = serde_json::json!("not persisted");
        let error = FlightStressProfile::parse_bytes(&serde_json::to_vec(&secret_key).expect("JSON"))
            .expect_err("key rejected");
        assert!(error.contains("credential-like field"));
    }

    #[test]
    fn shipped_s8_profiles_and_server_companions_are_validated() {
        for profile in [
            include_bytes!("../fixtures/stress/stress-s8-flight-pooled-sf1.flight-profile").as_slice(),
            include_bytes!("../fixtures/stress/stress-s8-flight-churn-sf1.flight-profile").as_slice(),
            include_bytes!("../fixtures/stress/stress-s8-flight-pooled-knee-sf10.flight-profile").as_slice(),
            include_bytes!("../fixtures/stress/stress-s8-flight-churn-knee-sf10.flight-profile").as_slice(),
        ] {
            FlightStressProfile::parse_bytes(profile).expect("shipped S8 profile");
        }
        ensure_redacted_server_config(include_bytes!("../fixtures/stress/server-s8-s2-scan-sf1.toml"))
            .expect("S2 SF1 server companion");
        ensure_redacted_server_config(include_bytes!("../fixtures/stress/server-s8-s2-knee-sf10.toml"))
            .expect("S2 server companion");
    }

    #[test]
    fn matched_artifact_requires_same_workload_and_load_cell() {
        let profile = profile();
        let baseline = baseline_json();
        let matched =
            matched_in_process_from_value(&baseline, "a".repeat(64), &profile.matched_in_process, &profile.load)
                .expect("matched baseline");
        assert_eq!(matched.profile_name, "stress-s1-embedded-sf1");
        assert_eq!(matched.application_qps, 3.0);

        let mut wrong_concurrency = baseline_json();
        wrong_concurrency["profile"]["scheduling"]["concurrency"] = serde_json::json!(4);
        let error = matched_in_process_from_value(
            &wrong_concurrency,
            "a".repeat(64),
            &profile.matched_in_process,
            &profile.load,
        )
        .expect_err("mismatch rejected");
        assert!(error.contains("load.clients"));

        let mut wrong_query = baseline_json();
        wrong_query["profile"]["workloads"][0]["queries"] = serde_json::json!(["q01"]);
        let error =
            matched_in_process_from_value(&wrong_query, "a".repeat(64), &profile.matched_in_process, &profile.load)
                .expect_err("query mismatch rejected");
        assert!(error.contains("query label"));

        let mut mixed_queries = baseline_json();
        mixed_queries["profile"]["workloads"][0]["queries"] = serde_json::json!(["q01", "q06"]);
        let error =
            matched_in_process_from_value(&mixed_queries, "a".repeat(64), &profile.matched_in_process, &profile.load)
                .expect_err("mixed baseline rejected");
        assert!(error.contains("exactly one query"));
    }

    #[test]
    fn matched_step_uses_the_selected_step_aggregate() {
        let mut profile = profile_json();
        profile["matched_in_process"]["step_index"] = serde_json::json!(1);
        let profile = FlightStressProfile::parse_bytes(&serde_json::to_vec(&profile).expect("JSON")).expect("profile");
        let mut artifact = baseline_json();
        artifact["profile"]["scheduling"]["measurement_seconds"] = serde_json::json!(0);
        artifact["profile"]["scheduling"]["steps"] = serde_json::json!([
            { "concurrency": 4, "measurement_seconds": 60 },
            { "concurrency": 8, "measurement_seconds": 120 }
        ]);
        artifact["execution"]["measurement"]["aggregate"]["steps"] = serde_json::json!([
            {
                "throughput": { "measurement_seconds": 60.0, "application_qps": 1.0, "gpu_reaching_qps": 1.0 },
                "latency": { "p50_ns": 1, "p95_ns": 2, "p99_ns": 3 }
            },
            {
                "throughput": { "measurement_seconds": 120.0, "application_qps": 4.0, "gpu_reaching_qps": 4.0 },
                "latency": { "p50_ns": 40, "p95_ns": 50, "p99_ns": 60 }
            }
        ]);
        let matched =
            matched_in_process_from_value(&artifact, "a".repeat(64), &profile.matched_in_process, &profile.load)
                .expect("second step matches");
        assert_eq!(matched.step_index, Some(1));
        assert_eq!(matched.application_qps, 4.0);
        assert_eq!(matched.latency_p99_ns, 60);
    }

    #[test]
    fn summary_reduces_only_measurement_attempts_and_reports_delta() {
        let mut config = config();
        config.measurement_seconds = 2;
        config.matched_in_process.measurement_seconds = 2.0;
        config.matched_in_process.application_qps = 1.0;
        config.matched_in_process.latency_p50_ns = 100;
        config.matched_in_process.latency_p95_ns = 100;
        config.matched_in_process.latency_p99_ns = 100;
        let records = vec![
            record(
                0,
                AttemptPhase::Warmup,
                ConnectionMode::Pooled,
                AttemptOutcome::Succeeded,
                NO_ERROR_CLASS,
                CancelPath::None,
                1,
            ),
            record(
                1,
                AttemptPhase::Measurement,
                ConnectionMode::Pooled,
                AttemptOutcome::Succeeded,
                NO_ERROR_CLASS,
                CancelPath::None,
                100,
            ),
            record(
                2,
                AttemptPhase::Measurement,
                ConnectionMode::Pooled,
                AttemptOutcome::Failed,
                "Internal",
                CancelPath::None,
                50,
            ),
        ];
        let summary = build_summary(&config, &records, Duration::from_secs(13));
        assert_eq!(summary["window"]["warmup_attempts"], 1);
        assert_eq!(summary["attempts"], 2);
        assert_eq!(summary["throughput"]["application_qps"], 0.5);
        assert_eq!(summary["delta_from_in_process"]["application_qps"]["server_minus_in_process"], -0.5);
        assert_eq!(summary["delta_from_in_process"]["latency_ns"]["p95"]["server_minus_in_process"], 0);
        assert_eq!(summary["delta_from_in_process"]["gpu_reaching_qps"]["status"], "not_observable_over_flight");
        let serialized = serde_json::to_string(&summary).expect("summary serializes");
        assert!(!serialized.contains("SELECT 1"));
        ensure_redacted_json(&summary, "").expect("redacted summary");
    }

    #[test]
    fn tsv_keeps_a_bounded_phase_column() {
        let mut row = record(
            9,
            AttemptPhase::Measurement,
            ConnectionMode::Churn,
            AttemptOutcome::Failed,
            "Unavailable",
            CancelPath::Drop,
            1234,
        );
        row.worker = 2;
        row.label = "bad\tlabel\n".to_string();
        row.rows = 41;
        row.start_offset_ns = 777;
        let tsv = row.to_tsv();
        assert_eq!(tsv, "9\t2\tbad label \tchurn\tmeasurement\t777\t1234\tfailed\tUnavailable\t41\tdrop");
        assert_eq!(tsv.split('\t').count(), ATTEMPT_COLUMNS.len());
    }

    #[test]
    fn percentile_and_flight_error_class_stay_bounded() {
        assert_eq!(percentile_nearest_rank(&[50, 10, 40, 20, 30], 0.95), Some(50));
        let tonic_error = FlightError::Tonic(Box::new(tonic::Status::cancelled("secret detail")));
        assert_eq!(flight_error_class(&tonic_error), "Cancelled");
        assert_eq!(flight_error_class(&FlightError::ProtocolError("secret".to_string())), "flight_protocol");
    }
}
