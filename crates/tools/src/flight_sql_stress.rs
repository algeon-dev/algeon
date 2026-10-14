//! Concurrent Flight SQL stress driver.
//!
//! `flight-sql-stress` drives a Flight SQL server from a recorded profile and
//! records one bounded row per query attempt in `attempts.tsv` plus a
//! `summary.json` reduction. A profile names one or more statements. It may
//! also name the in-process `run-artifact.json` cell it matches; the summary
//! then fails closed on a shape mismatch and reports the
//! server-minus-in-process throughput/latency delta. A profile without that
//! cell is a server-only run, and its summary has no delta section.
//!
//! # Load semantics
//!
//! A closed-loop stream runs the statement list in its own order, a
//! Fisher-Yates permutation driven by SplitMix64 seeded with
//! `load.seed + stream`. The k-th open-loop attempt runs the statement chosen
//! by the k-th SplitMix64 draw from `load.seed`. A one-statement profile needs
//! no seed.
//!
//! - Streams: `clients` streams each run `passes` passes over their order.
//!   Attempts in the first `warmup_passes` passes are warm-up. Every stream
//!   then waits at a barrier, so measurement starts once all warm-up attempts
//!   have ended. The run ends when every stream finishes its passes.
//! - Concurrency: `clients` streams walk their order repeatedly until the
//!   shared deadline. The deadline stops new submissions; an in-flight
//!   attempt always runs to its terminal state, so the run can overrun the
//!   configured duration slightly.
//! - Rate (open-loop): uniformly spaced submit instants (period 1/R from run
//!   start). Each submit spawns a task bounded by a semaphore of `clients`
//!   permits, so at most `clients` attempts are in flight; under saturation
//!   submit instants slip and missed schedule slots are not replayed in
//!   bursts. The summary reports scheduled slots, submissions, permit waits
//!   and the largest lag behind slot `k / R`; any permit wait marks the run
//!   `client_bound_reached`. A slot whose permit frees after the deadline
//!   counts its wait and submits nothing. The achievable rate is additionally
//!   bounded by the tokio timer resolution.
//!
//! Pooled clients connect before the run clock starts. Time-based modes
//! classify an attempt submitted before `warmup_seconds` as warm-up. Latency
//! and outcome counts reduce measurement attempts. `completed_qps` counts
//! successes that end inside the throughput window and divides by its length.
//! For time-based modes the window is the configured measurement span
//! `[warmup_seconds, warmup_seconds + measurement_seconds]`, so attempts that
//! finish after the deadline are recorded but excluded from the rate, as in
//! the in-process runner's `application_qps`. For streams the window runs from
//! the warm-up barrier to the last measurement completion. `offered_qps`
//! divides measurement submissions by the same length.
//!
//! # Connection modes
//!
//! The pooled mode (default) connects one client per worker slot once, with
//! a long `connect_with_retry` budget that waits for a server still starting.
//! The churn mode connects a fresh client per attempt with a short retry
//! budget, because the server is already expected to accept connections.
//!
//! # Ticket cancellation
//!
//! Ticket cancellation applies to every Nth attempt per stream (closed-loop)
//! or every Nth scheduled attempt (open-loop). A cancel attempt starts the
//! DoGet stream of the first endpoint ticket, waits `after_ms`, then cancels.
//! It first uses the standard `CancelFlightInfo` action on the attempt's
//! FlightInfo. When the server does not support that action (any action
//! error, including `Unimplemented`/`InvalidArgument`), it cancels by
//! dropping the DoGet stream, which the server treats as statement
//! cancellation. After a cancel attempt, a terminal tonic `Cancelled` stream
//! error or a client-initiated stream drop records `cancelled`. A cleanly
//! drained stream means the query completed before the cancel took effect and
//! records `succeeded`. Any other terminal error records `failed`.
//! The `cancel_path` column records `none`, `flight_info`, or `drop`.
//!
//! # Admission sampling and result checking
//!
//! `admission_sampling` polls `GetAdmissionStats` on its own connection and
//! writes `admission.tsv`. After the last attempt it keeps polling until every
//! device has no active attempts and no live grant bytes, for at most 60 s;
//! see [`admission_sampling`]. `results` either
//! records each statement's first successful result as a reference or checks
//! every successful result against recorded references; see [`reference`](mod@reference).
//! Collecting, writing and comparing a result happen after its latency is
//! taken and before the stream submits its next attempt. Cancel-scheduled
//! attempts are not checked. A result that cannot be decoded, recorded or
//! compared records `check = error`.
//!
//! # Output discipline
//!
//! The `--out-dir` path must not already exist (create-new semantics).
//! `attempts.tsv` carries the `ATTEMPT_COLUMNS` header plus one row per
//! attempt. Its error columns hold the tonic code name and the server's
//! `x-algeon-error-code` and `x-algeon-reason` identifiers. `summary.json`
//! retains only a redacted profile projection, statement and reference
//! hashes, outcome counts, nearest-rank latency percentiles, and the optional
//! admission, result and in-process sections. SQL text, raw profile contents,
//! URL userinfo, and raw error detail are never persisted.

mod admission_sampling;
mod reference;

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
#[cfg(not(test))]
use std::{
    io::Write,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(not(test))]
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use arrow_flight::error::FlightError;
#[cfg(not(test))]
use arrow_flight::{Action, CancelFlightInfoRequest, FlightInfo, Ticket, sql::client::FlightSqlServiceClient};
#[cfg(not(test))]
use futures::TryStreamExt;
#[cfg(not(test))]
use prost::Message;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
#[cfg(not(test))]
use tokio::{
    sync::{Barrier, Mutex, mpsc, oneshot},
    task::{JoinSet, spawn_blocking},
    time::sleep,
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError},
    time::{Instant, MissedTickBehavior, interval},
};
#[cfg(not(test))]
use tonic::transport::Channel;

use self::admission_sampling::AdmissionLog;
#[cfg(not(test))]
use self::admission_sampling::sample_admission;
#[cfg(feature = "flight-sql-stress-check")]
use self::reference::{FloatTolerance, Reference};
#[cfg(not(test))]
use self::reference::{ResultCapture, reference_path, write_reference};
#[cfg(not(test))]
use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};
use crate::{SINGLE_NODE_RUN_ARTIFACT_SCHEMA, split_mix64::SplitMix64};

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Label for an absent error class, code, reason or pass.
const NONE_LABEL: &str = "none";
/// Label for a server metadata value that is not a bounded identifier.
const INVALID_LABEL: &str = "invalid";
const PROFILE_SCHEMA: &str = "algeon_datafusion.flight_sql_stress.run_profile";
const ARTIFACT_SCHEMA: &str = "algeon_datafusion.flight_sql_stress.run_artifact";
const USAGE: &str = "flight-sql-stress --config flight-stress-profile.json --out-dir DIR";
/// SQL is held in memory and sent once per attempt; the bound keeps profile
/// mistakes (a data file named as `sql_file`) from reaching the server while
/// admitting every vendored TPC-H and TPC-DS query.
const MAX_SQL_BYTES: usize = 16 * 1024;
/// The open-loop period `1/R` must stay at least one nanosecond, because a
/// zero tokio interval period panics.
const MAX_RATE_PER_SECOND: f64 = 1e9;
const DEFAULT_ADMISSION_SAMPLING_INTERVAL_MS: u64 = 1000;
const ERROR_CODE_METADATA_KEY: &str = "x-algeon-error-code";
const REASON_METADATA_KEY: &str = "x-algeon-reason";
const SERVICE_OVERLOADED_CODE: &str = "service_overloaded";

const ATTEMPT_COLUMNS: &[&str] = &[
    "sequence",
    "stream",
    "pass",
    "statement",
    "mode",
    "phase",
    "submit_offset_ns",
    "end_offset_ns",
    "latency_ns",
    "outcome",
    "error_class",
    "error_code",
    "error_reason",
    "rows",
    "cancel_path",
    "check",
];

/// How an attempt obtained its Flight SQL connection.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ConnectionMode {
    Pooled,
    Churn,
}

impl ConnectionMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pooled => "pooled",
            Self::Churn => "churn",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArrivalMode {
    Concurrency,
    Rate,
    Streams,
}

impl ArrivalMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Concurrency => "concurrency",
            Self::Rate => "rate",
            Self::Streams => "streams",
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

/// Result-check outcome of one attempt. Only successful, non-cancel attempts
/// of a run with `results` are recorded or compared. `Error` marks a result
/// that could not be decoded, recorded or compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckOutcome {
    Match,
    Mismatch,
    Error,
    NotChecked,
}

impl CheckOutcome {
    const ALL: [Self; 4] = [Self::Match, Self::Mismatch, Self::Error, Self::NotChecked];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::Error => "error",
            Self::NotChecked => "not_checked",
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
    name: String,
    server: FlightServerProfile,
    statements: Vec<FlightStatementProfile>,
    load: FlightLoadProfile,
    #[serde(default)]
    admission_sampling: Option<AdmissionSamplingProfile>,
    #[serde(default)]
    results: Option<ResultsProfile>,
    #[serde(default)]
    matched_in_process: Option<MatchedInProcessProfile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightServerProfile {
    url: String,
    config_file: PathBuf,
}

/// A statement carries inline `sql` or a `sql_file` path relative to the
/// profile, never both.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightStatementProfile {
    name: String,
    #[serde(default)]
    sql: Option<String>,
    #[serde(default)]
    sql_file: Option<PathBuf>,
}

/// Mode-specific fields are optional here and checked by
/// [`FlightLoadProfile::shape`]: time-based modes take `warmup_seconds` and
/// `measurement_seconds`, rate adds `rate_per_second`, and streams take
/// `passes` and `warmup_passes`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlightLoadProfile {
    arrival_mode: ArrivalMode,
    clients: usize,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    warmup_seconds: Option<u64>,
    #[serde(default)]
    measurement_seconds: Option<u64>,
    #[serde(default)]
    rate_per_second: Option<f64>,
    #[serde(default)]
    passes: Option<u64>,
    #[serde(default)]
    warmup_passes: Option<u64>,
    connection_mode: ConnectionMode,
    #[serde(default)]
    cancel: Option<CancelConfig>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissionSamplingProfile {
    #[serde(default = "default_admission_sampling_interval_ms")]
    interval_ms: u64,
}

const fn default_admission_sampling_interval_ms() -> u64 {
    DEFAULT_ADMISSION_SAMPLING_INTERVAL_MS
}

/// Exactly one of `record_reference` or `check_reference`; the float
/// tolerances apply to checking and default to exact comparison.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultsProfile {
    #[serde(default)]
    record_reference: Option<PathBuf>,
    #[serde(default)]
    check_reference: Option<PathBuf>,
    #[serde(default)]
    float_abs_tolerance: Option<f64>,
    #[serde(default)]
    float_rel_tolerance: Option<f64>,
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

/// The validated load of one run.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LoadShape {
    Concurrency { warmup_seconds: u64, measurement_seconds: u64 },
    Rate { warmup_seconds: u64, measurement_seconds: u64, rate_per_second: f64 },
    Streams { passes: u64, warmup_passes: u64 },
}

impl LoadShape {
    const fn arrival_mode(self) -> ArrivalMode {
        match self {
            Self::Concurrency { .. } => ArrivalMode::Concurrency,
            Self::Rate { .. } => ArrivalMode::Rate,
            Self::Streams { .. } => ArrivalMode::Streams,
        }
    }

    /// Streams classify by pass; time-based modes by submit offset.
    fn phase(self, pass: Option<u64>, submit_offset: Duration) -> AttemptPhase {
        let warmup = match self {
            Self::Streams { warmup_passes, .. } => pass.is_some_and(|pass| pass < warmup_passes),
            Self::Concurrency { warmup_seconds, .. } | Self::Rate { warmup_seconds, .. } => {
                submit_offset < Duration::from_secs(warmup_seconds)
            },
        };
        if warmup { AttemptPhase::Warmup } else { AttemptPhase::Measurement }
    }

    fn json(self) -> serde_json::Value {
        match self {
            Self::Concurrency { warmup_seconds, measurement_seconds } => serde_json::json!({
                "warmup_seconds": warmup_seconds,
                "measurement_seconds": measurement_seconds,
            }),
            Self::Rate { warmup_seconds, measurement_seconds, rate_per_second } => serde_json::json!({
                "warmup_seconds": warmup_seconds,
                "measurement_seconds": measurement_seconds,
                "rate_per_second": rate_per_second,
            }),
            Self::Streams { passes, warmup_passes } => serde_json::json!({
                "passes": passes,
                "warmup_passes": warmup_passes,
            }),
        }
    }
}

impl FlightLoadProfile {
    fn shape(&self, statement_count: usize) -> Result<LoadShape, String> {
        if self.clients == 0 {
            return Err("load.clients must be greater than zero".to_string());
        }
        if statement_count > 1 && self.seed.is_none() {
            return Err("load.seed is required when the profile names more than one statement".to_string());
        }
        if let Some(cancel) = self.cancel
            && cancel.every_n == 0
        {
            return Err("load.cancel.every_n must be greater than zero".to_string());
        }
        if self.arrival_mode == ArrivalMode::Streams {
            if self.warmup_seconds.is_some() || self.measurement_seconds.is_some() || self.rate_per_second.is_some() {
                return Err("load.warmup_seconds, load.measurement_seconds and load.rate_per_second are not valid \
                            for load.arrival_mode=streams"
                    .to_string());
            }
            let passes = self
                .passes
                .filter(|passes| *passes > 0)
                .ok_or_else(|| "streams load requires a non-zero load.passes".to_string())?;
            let warmup_passes =
                self.warmup_passes.ok_or_else(|| "streams load requires load.warmup_passes".to_string())?;
            if warmup_passes >= passes {
                return Err("load.warmup_passes must be less than load.passes".to_string());
            }
            return Ok(LoadShape::Streams { passes, warmup_passes });
        }
        if self.passes.is_some() || self.warmup_passes.is_some() {
            return Err("load.passes and load.warmup_passes are only valid for load.arrival_mode=streams".to_string());
        }
        let warmup_seconds = self
            .warmup_seconds
            .ok_or_else(|| format!("{} load requires load.warmup_seconds", self.arrival_mode.as_str()))?;
        let measurement_seconds = self
            .measurement_seconds
            .filter(|seconds| *seconds > 0)
            .ok_or_else(|| "load.measurement_seconds must be greater than zero".to_string())?;
        warmup_seconds
            .checked_add(measurement_seconds)
            .ok_or_else(|| "load warmup_seconds + measurement_seconds overflows".to_string())?;
        match (self.arrival_mode, self.rate_per_second) {
            (ArrivalMode::Concurrency, None) => Ok(LoadShape::Concurrency { warmup_seconds, measurement_seconds }),
            (ArrivalMode::Concurrency, Some(_)) => {
                Err("load.rate_per_second is only valid for load.arrival_mode=rate".to_string())
            },
            (ArrivalMode::Rate, Some(rate)) if rate.is_finite() && rate > 0.0 && rate <= MAX_RATE_PER_SECOND => {
                Ok(LoadShape::Rate { warmup_seconds, measurement_seconds, rate_per_second: rate })
            },
            (ArrivalMode::Rate, _) => {
                Err(format!("rate load requires load.rate_per_second in (0, {MAX_RATE_PER_SECOND}]"))
            },
            (ArrivalMode::Streams, _) => unreachable!("streams returned above"),
        }
    }
}

impl ResultsProfile {
    fn validate(&self) -> Result<(), String> {
        let tolerances =
            [("float_abs_tolerance", self.float_abs_tolerance), ("float_rel_tolerance", self.float_rel_tolerance)];
        match (&self.record_reference, &self.check_reference) {
            (Some(dir), None) if !dir.as_os_str().is_empty() => {
                if tolerances.iter().any(|(_, tolerance)| tolerance.is_some()) {
                    return Err("results float tolerances are only valid with results.check_reference".to_string());
                }
            },
            (None, Some(dir)) if !dir.as_os_str().is_empty() => {
                for (field, tolerance) in tolerances {
                    if tolerance.is_some_and(|tolerance| !tolerance.is_finite() || tolerance < 0.0) {
                        return Err(format!("results.{field} must be a finite non-negative number"));
                    }
                }
            },
            _ => return Err("results must set exactly one non-empty record_reference or check_reference".to_string()),
        }
        Ok(())
    }
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
        if !is_identifier(&self.name) {
            return Err("profile.name must be a 1-64 character ASCII identifier".to_string());
        }
        validate_url(&self.server.url)?;
        if self.server.config_file.as_os_str().is_empty() {
            return Err("server.config_file must not be empty".to_string());
        }
        if self.statements.is_empty() {
            return Err("statements must name at least one statement".to_string());
        }
        let mut names = BTreeSet::new();
        for (index, statement) in self.statements.iter().enumerate() {
            if !is_identifier(&statement.name) {
                return Err(format!("statements[{index}].name must be a 1-64 character ASCII identifier"));
            }
            if !names.insert(statement.name.as_str()) {
                return Err(format!("statements[{index}].name repeats an earlier statement name"));
            }
            match (&statement.sql, &statement.sql_file) {
                (Some(sql), None) => {
                    validate_sql_text(sql).map_err(|reason| format!("statements[{index}].sql {reason}"))?
                },
                (None, Some(path)) if !path.as_os_str().is_empty() => {},
                _ => return Err(format!("statements[{index}] must set exactly one of sql or a non-empty sql_file")),
            }
        }
        self.load.shape(self.statements.len())?;
        if self.admission_sampling.is_some_and(|sampling| sampling.interval_ms == 0) {
            return Err("admission_sampling.interval_ms must be greater than zero".to_string());
        }
        if let Some(results) = &self.results {
            results.validate()?;
        }
        if let Some(matched) = &self.matched_in_process {
            if matched.run_artifact.as_os_str().is_empty() {
                return Err("matched_in_process.run_artifact must not be empty".to_string());
            }
            if !is_identifier(&matched.workload.kind) {
                return Err("matched_in_process.workload.kind must be a 1-64 character ASCII identifier".to_string());
            }
            if !is_identifier(&matched.workload.query_label) {
                return Err(
                    "matched_in_process.workload.query_label must be a 1-64 character ASCII identifier".to_string()
                );
            }
            if self.statements.len() != 1 || self.statements[0].name != matched.workload.query_label {
                return Err(
                    "matched_in_process requires exactly one statement named matched_in_process.workload.query_label"
                        .to_string(),
                );
            }
            if self.load.arrival_mode == ArrivalMode::Streams {
                return Err("matched_in_process requires load.arrival_mode concurrency or rate".to_string());
            }
        }
        Ok(())
    }
}

fn validate_sql_text(sql: &str) -> Result<(), String> {
    if sql.trim().is_empty() || sql.len() > MAX_SQL_BYTES {
        return Err(format!("must be 1-{MAX_SQL_BYTES} bytes with non-whitespace text"));
    }
    Ok(())
}

/// One statement of the run. The SQL is never persisted; only its name and
/// SHA-256 are.
#[derive(Debug)]
struct StressStatement {
    name: String,
    sql: String,
    sql_sha256: String,
}

/// How a run treats successful results.
#[derive(Debug)]
enum ResultsConfig {
    /// Write each statement's first successful result under `dir`.
    Record { dir: PathBuf },
    /// Compare every successful result with its statement's reference.
    #[cfg(feature = "flight-sql-stress-check")]
    Check { tolerance: FloatTolerance, references: BTreeMap<String, Reference> },
}

impl ResultsConfig {
    const fn mode(&self) -> &'static str {
        match self {
            Self::Record { .. } => "record",
            #[cfg(feature = "flight-sql-stress-check")]
            Self::Check { .. } => "check",
        }
    }
}

#[derive(Debug)]
pub(crate) struct StressConfig {
    /// Full connection URL. Never persisted; only `url_host_port` is echoed.
    url: String,
    /// URL redacted to `host:port` (no userinfo, path, or query).
    url_host_port: String,
    statements: Vec<StressStatement>,
    profile_name: String,
    profile_sha256: String,
    server_config_sha256: String,
    clients: usize,
    seed: Option<u64>,
    shape: LoadShape,
    connection_mode: ConnectionMode,
    cancel: Option<CancelConfig>,
    admission_sampling_interval: Option<Duration>,
    results: Option<ResultsConfig>,
    matched_in_process: Option<MatchedInProcess>,
    out_dir: PathBuf,
}

impl StressConfig {
    fn statement_seed(&self) -> u64 {
        // A missing seed is only valid for one statement, whose order and
        // draws do not depend on it.
        self.seed.unwrap_or_default()
    }
}

/// The statement order of one closed-loop stream.
fn stream_order(seed: u64, stream: usize, count: usize) -> Vec<usize> {
    let mut order = (0..count).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(seed.wrapping_add(stream as u64));
    for index in (1..count).rev() {
        order.swap(index, rng.next_below(index + 1));
    }
    order
}

/// The statements of the first `count` open-loop attempts, in schedule order.
fn open_loop_statements(seed: u64, statement_count: usize) -> impl Iterator<Item = usize> {
    let mut rng = SplitMix64::new(seed);
    std::iter::repeat_with(move || rng.next_below(statement_count))
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

fn resolve_profile_path(profile_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { profile_dir.join(path) }
}

pub(crate) fn read_stress_config(config_path: &Path, out_dir: PathBuf) -> Result<StressConfig, String> {
    let profile_bytes = fs::read(config_path).map_err(|_| format!("cannot read --config {}", config_path.display()))?;
    let profile = FlightStressProfile::parse_bytes(&profile_bytes)?;
    let profile_sha256 = sha256_bytes(&profile_bytes);
    let profile_dir =
        config_path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let server_config_path = resolve_profile_path(profile_dir, &profile.server.config_file);
    let server_config = fs::read(&server_config_path)
        .map_err(|_| format!("cannot read server.config_file {}", server_config_path.display()))?;
    ensure_redacted_server_config(&server_config)?;
    let statements = profile
        .statements
        .iter()
        .enumerate()
        .map(|(index, statement)| read_statement(profile_dir, index, statement))
        .collect::<Result<Vec<_>, _>>()?;
    let shape = profile.load.shape(statements.len())?;
    let matched_in_process = profile
        .matched_in_process
        .as_ref()
        .map(|expected| read_matched_in_process(profile_dir, expected, shape, profile.load.clients))
        .transpose()?;
    let results =
        profile.results.as_ref().map(|results| read_results_config(profile_dir, results, &statements)).transpose()?;
    let url_host_port = validate_url(&profile.server.url)?;
    Ok(StressConfig {
        url: profile.server.url,
        url_host_port,
        statements,
        profile_name: profile.name,
        profile_sha256,
        server_config_sha256: sha256_bytes(&server_config),
        clients: profile.load.clients,
        seed: profile.load.seed,
        shape,
        connection_mode: profile.load.connection_mode,
        cancel: profile.load.cancel,
        admission_sampling_interval: profile
            .admission_sampling
            .map(|sampling| Duration::from_millis(sampling.interval_ms)),
        results,
        matched_in_process,
        out_dir,
    })
}

fn read_statement(
    profile_dir: &Path,
    index: usize,
    statement: &FlightStatementProfile,
) -> Result<StressStatement, String> {
    let sql = match (&statement.sql, &statement.sql_file) {
        (Some(sql), None) => sql.clone(),
        (None, Some(path)) => {
            let path = resolve_profile_path(profile_dir, path);
            let bytes =
                fs::read(&path).map_err(|_| format!("cannot read statements[{index}].sql_file {}", path.display()))?;
            let sql = String::from_utf8(bytes).map_err(|_| format!("statements[{index}].sql_file must be UTF-8"))?;
            validate_sql_text(&sql).map_err(|reason| format!("statements[{index}].sql_file {reason}"))?;
            if has_inline_secret(&sql) {
                return Err(format!("statements[{index}].sql_file contains credential-like text"));
            }
            sql
        },
        _ => return Err(format!("statements[{index}] must set exactly one of sql or sql_file")),
    };
    Ok(StressStatement { name: statement.name.clone(), sql_sha256: sha256_bytes(sql.as_bytes()), sql })
}

fn read_results_config(
    profile_dir: &Path,
    results: &ResultsProfile,
    statements: &[StressStatement],
) -> Result<ResultsConfig, String> {
    match (&results.record_reference, &results.check_reference) {
        (Some(dir), None) => Ok(ResultsConfig::Record { dir: resolve_profile_path(profile_dir, dir) }),
        (None, Some(dir)) => read_check_config(&resolve_profile_path(profile_dir, dir), results, statements),
        _ => Err("results must set exactly one of record_reference or check_reference".to_string()),
    }
}

/// Loads every statement's reference before the run, so a missing or
/// unreadable reference fails the run before any load is offered.
#[cfg(feature = "flight-sql-stress-check")]
fn read_check_config(
    dir: &Path,
    results: &ResultsProfile,
    statements: &[StressStatement],
) -> Result<ResultsConfig, String> {
    let tolerance = FloatTolerance {
        absolute: results.float_abs_tolerance.unwrap_or(0.0),
        relative: results.float_rel_tolerance.unwrap_or(0.0),
    };
    let references = statements
        .iter()
        .map(|statement| {
            let path = self::reference::reference_path(dir, &statement.name);
            let bytes = fs::read(&path).map_err(|_| format!("cannot read reference {}", path.display()))?;
            let reference = Reference::from_bytes(&bytes, &statement.sql, tolerance)
                .map_err(|reason| format!("{}: {reason}", path.display()))?;
            Ok((statement.name.clone(), reference))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    Ok(ResultsConfig::Check { tolerance, references })
}

#[cfg(not(feature = "flight-sql-stress-check"))]
fn read_check_config(
    _dir: &Path,
    _results: &ResultsProfile,
    _statements: &[StressStatement],
) -> Result<ResultsConfig, String> {
    Err("results.check_reference requires algeon_tools built with the flight-sql-stress-check feature".to_string())
}

fn read_matched_in_process(
    profile_dir: &Path,
    expected: &MatchedInProcessProfile,
    shape: LoadShape,
    clients: usize,
) -> Result<MatchedInProcess, String> {
    let baseline_path = resolve_profile_path(profile_dir, &expected.run_artifact);
    let baseline_bytes = fs::read(&baseline_path)
        .map_err(|_| format!("cannot read matched_in_process.run_artifact {}", baseline_path.display()))?;
    let baseline_json: serde_json::Value = serde_json::from_slice(&baseline_bytes)
        .map_err(|_| "matched_in_process.run_artifact must contain valid JSON".to_string())?;
    ensure_redacted_json(&baseline_json, "")?;
    matched_in_process_from_value(&baseline_json, sha256_bytes(&baseline_bytes), expected, shape, clients)
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
    shape: LoadShape,
    clients: usize,
) -> Result<MatchedInProcess, String> {
    let (warmup_seconds, measurement_seconds) = match shape {
        LoadShape::Concurrency { warmup_seconds, measurement_seconds }
        | LoadShape::Rate { warmup_seconds, measurement_seconds, .. } => (warmup_seconds, measurement_seconds),
        LoadShape::Streams { .. } => {
            return Err("matched_in_process requires load.arrival_mode concurrency or rate".to_string());
        },
    };
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
    if arrival_mode != shape.arrival_mode().as_str() {
        return Err("Flight load.arrival_mode does not match the selected in-process cell".to_string());
    }
    if u64_at(scheduling, "/warmup_seconds", "matched in-process profile scheduling")? != warmup_seconds {
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
    validate_load_cell(cell, shape, clients, measurement_seconds)?;

    let throughput = value_at(aggregate, "/throughput", "matched in-process aggregate")?;
    let latency = value_at(aggregate, "/latency", "matched in-process aggregate")?;
    let aggregate_seconds = finite_f64_at(throughput, "/measurement_seconds", "matched in-process throughput")?;
    if (aggregate_seconds - measurement_seconds as f64).abs() > f64::EPSILON {
        return Err("matched in-process aggregate measurement_seconds does not match the selected cell".to_string());
    }
    Ok(MatchedInProcess {
        artifact_sha256,
        profile_name,
        profile_sha256,
        workload_kind: expected.workload.kind.clone(),
        query_label: expected.workload.query_label.clone(),
        step_index,
        measurement_seconds: aggregate_seconds,
        application_qps: finite_f64_at(throughput, "/application_qps", "matched in-process throughput")?,
        gpu_reaching_qps: finite_f64_at(throughput, "/gpu_reaching_qps", "matched in-process throughput")?,
        latency_p50_ns: u64_at(latency, "/p50_ns", "matched in-process latency")?,
        latency_p95_ns: u64_at(latency, "/p95_ns", "matched in-process latency")?,
        latency_p99_ns: u64_at(latency, "/p99_ns", "matched in-process latency")?,
    })
}

fn validate_load_cell(
    cell: &serde_json::Value,
    shape: LoadShape,
    clients: usize,
    measurement_seconds: u64,
) -> Result<(), String> {
    if u64_at(cell, "/measurement_seconds", "selected in-process cell")? != measurement_seconds {
        return Err("Flight load.measurement_seconds does not match the selected in-process cell".to_string());
    }
    match shape {
        LoadShape::Concurrency { .. } => {
            let concurrency = u64_at(cell, "/concurrency", "selected in-process cell")?;
            if concurrency != clients as u64 {
                return Err("Flight load.clients does not match selected in-process concurrency".to_string());
            }
        },
        LoadShape::Rate { rate_per_second, .. } => {
            let rate = u64_at(cell, "/arrival_rate_per_second", "selected in-process cell")?;
            if (rate as f64 - rate_per_second).abs() > f64::EPSILON {
                return Err("Flight load.rate_per_second does not match the selected in-process rate".to_string());
            }
        },
        LoadShape::Streams { .. } => {
            return Err("matched_in_process requires load.arrival_mode concurrency or rate".to_string());
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

/// Bounded error labels of one attempt: the transport class and the server's
/// public code and reason identifiers. Raw error text is never kept.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ErrorFacts {
    class: String,
    code: String,
    reason: String,
}

impl ErrorFacts {
    fn none() -> Self {
        Self::with_class(NONE_LABEL)
    }

    fn with_class(class: &str) -> Self {
        Self { class: class.to_string(), code: NONE_LABEL.to_string(), reason: NONE_LABEL.to_string() }
    }

    fn from_flight(error: &FlightError) -> Self {
        let class = flight_error_class(error);
        match error {
            FlightError::Tonic(status) => Self {
                class,
                code: status_metadata_label(status, ERROR_CODE_METADATA_KEY),
                reason: status_metadata_label(status, REASON_METADATA_KEY),
            },
            _ => Self { class, ..Self::none() },
        }
    }

    fn is_none(&self) -> bool {
        self.class == NONE_LABEL
    }
}

/// A metadata value that is not a bounded identifier is recorded as
/// `invalid` rather than copied.
fn status_metadata_label(status: &tonic::Status, key: &str) -> String {
    match status.metadata().get(key).map(|value| value.to_str()) {
        None => NONE_LABEL.to_string(),
        Some(Ok(value)) if is_identifier(value) => value.to_string(),
        Some(_) => INVALID_LABEL.to_string(),
    }
}

/// Bounded per-attempt observation. Error detail is label-only.
#[derive(Debug, Clone)]
struct AttemptRecord {
    sequence: u64,
    stream: usize,
    /// Closed-loop pass over the stream's order; open-loop attempts have none.
    pass: Option<u64>,
    statement: String,
    mode: ConnectionMode,
    phase: AttemptPhase,
    submit_offset_ns: u64,
    end_offset_ns: u64,
    latency_ns: u64,
    outcome: AttemptOutcome,
    error: ErrorFacts,
    rows: u64,
    cancel_path: CancelPath,
    check: CheckOutcome,
}

impl AttemptRecord {
    fn to_tsv(&self) -> String {
        let fields = [
            self.sequence.to_string(),
            self.stream.to_string(),
            self.pass.map_or_else(|| NONE_LABEL.to_string(), |pass| pass.to_string()),
            tsv_field(&self.statement),
            self.mode.as_str().to_string(),
            self.phase.as_str().to_string(),
            self.submit_offset_ns.to_string(),
            self.end_offset_ns.to_string(),
            self.latency_ns.to_string(),
            self.outcome.as_str().to_string(),
            self.error.class.clone(),
            self.error.code.clone(),
            self.error.reason.clone(),
            self.rows.to_string(),
            self.cancel_path.as_str().to_string(),
            self.check.as_str().to_string(),
        ];
        debug_assert_eq!(fields.len(), ATTEMPT_COLUMNS.len());
        fields.join("\t")
    }
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

/// Maps a flight error to a bounded label: the tonic code name for RPC failures,
/// and fixed labels for non-tonic flight and transport failures. Raw error text
/// is never returned.
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
    /// Error count by public code, then reason.
    errors: BTreeMap<String, BTreeMap<String, u64>>,
    checks: BTreeMap<&'static str, u64>,
    cancel_paths: BTreeMap<&'static str, u64>,
    pooled: ModeCounts,
    churn: ModeCounts,
    succeeded_latencies_ns: Vec<u64>,
    /// Latencies of `service_overloaded` rejections by reason.
    rejected_latencies_ns: BTreeMap<String, Vec<u64>>,
}

fn reduce<'a>(records: impl IntoIterator<Item = &'a AttemptRecord>) -> Reduction {
    let mut reduction = Reduction {
        checks: CheckOutcome::ALL.iter().map(|check| (check.as_str(), 0)).collect(),
        ..Reduction::default()
    };
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
                if record.error.code == SERVICE_OVERLOADED_CODE {
                    reduction
                        .rejected_latencies_ns
                        .entry(record.error.reason.clone())
                        .or_default()
                        .push(record.latency_ns);
                }
            },
            AttemptOutcome::Cancelled => {
                reduction.cancelled += 1;
                mode_counts.cancelled += 1;
            },
        }
        if !record.error.is_none() {
            *reduction.error_classes.entry(record.error.class.clone()).or_default() += 1;
            *reduction
                .errors
                .entry(record.error.code.clone())
                .or_default()
                .entry(record.error.reason.clone())
                .or_default() += 1;
        }
        *reduction.checks.entry(record.check.as_str()).or_default() += 1;
        *reduction.cancel_paths.entry(record.cancel_path.as_str()).or_default() += 1;
    }
    reduction
}

/// Returns the nearest-rank percentile, the value at rank `ceil(q * n)` of the sorted values.
fn percentile_nearest_rank(values: &[u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((quantile * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    Some(sorted[rank - 1])
}

/// The span whose successful completions count toward `completed_qps`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ThroughputWindow {
    start_ns: u64,
    end_ns: u64,
    seconds: f64,
}

impl ThroughputWindow {
    /// Time-based modes use the configured measurement span, so an attempt
    /// that completes after the deadline is excluded, as the in-process
    /// runner's `application_qps` excludes it. Streams start at the barrier
    /// every stream reaches after its warm-up passes (`measurement_start_ns`)
    /// and end at the last measurement completion.
    fn select(shape: LoadShape, measured: &[&AttemptRecord], measurement_start_ns: Option<u64>) -> Option<Self> {
        match shape {
            LoadShape::Concurrency { warmup_seconds, measurement_seconds }
            | LoadShape::Rate { warmup_seconds, measurement_seconds, .. } => Some(Self {
                start_ns: warmup_seconds.saturating_mul(1_000_000_000),
                end_ns: (warmup_seconds + measurement_seconds).saturating_mul(1_000_000_000),
                seconds: measurement_seconds as f64,
            }),
            LoadShape::Streams { .. } => {
                let start_ns = measurement_start_ns?;
                let end_ns = measured.iter().map(|record| record.end_offset_ns).max()?;
                (end_ns > start_ns).then(|| Self { start_ns, end_ns, seconds: (end_ns - start_ns) as f64 / 1e9 })
            },
        }
    }

    fn completed<'a>(self, records: impl IntoIterator<Item = &'a AttemptRecord>) -> u64 {
        records
            .into_iter()
            .filter(|record| {
                record.outcome == AttemptOutcome::Succeeded
                    && (self.start_ns..=self.end_ns).contains(&record.end_offset_ns)
            })
            .count() as u64
    }
}

fn mode_summary(counts: &ModeCounts) -> serde_json::Value {
    serde_json::json!({
        "attempts": counts.attempts,
        "succeeded": counts.succeeded,
        "failed": counts.failed,
        "cancelled": counts.cancelled,
    })
}

/// `completed` counts successes inside `window`; `offered_qps` divides the
/// measurement attempts (submitted inside the window) by its seconds.
fn reduction_summary(
    reduction: &Reduction,
    completed: u64,
    window: Option<ThroughputWindow>,
    open_loop: bool,
) -> serde_json::Value {
    let latencies = &reduction.succeeded_latencies_ns;
    let per_second = |count: u64| window.map(|window| count as f64 / window.seconds);
    let mut summary = serde_json::json!({
        "attempts": reduction.attempts,
        "succeeded": reduction.succeeded,
        "failed": reduction.failed,
        "cancelled": reduction.cancelled,
        "throughput": {
            "successful_completions": completed,
            "completed_qps": per_second(completed),
            "offered_qps": per_second(reduction.attempts),
        },
        "latency_ns": {
            "samples": latencies.len(),
            "p50": percentile_nearest_rank(latencies, 0.50),
            "p95": percentile_nearest_rank(latencies, 0.95),
            "p99": percentile_nearest_rank(latencies, 0.99),
        },
        "error_classes": reduction.error_classes,
        "errors": reduction.errors,
        "checks": reduction.checks,
    });
    if open_loop {
        summary["rejected_latency_p50_ns"] = reduction
            .rejected_latencies_ns
            .iter()
            .map(|(reason, latencies)| (reason.clone(), serde_json::json!(percentile_nearest_rank(latencies, 0.50))))
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    summary
}

/// A reference file this run wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecordedReference {
    sha256: String,
    rows: u64,
}

/// Open-loop schedule accounting. A permit wait means `clients` bounded the
/// submissions, so the offered rate fell below the configured rate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct OpenLoopStats {
    /// Schedule slots that fired before the deadline.
    scheduled: u64,
    /// Attempts started; a slot whose permit arrived after the deadline is not.
    submitted: u64,
    /// Slots that found every permit held, including a slot whose permit
    /// freed only after the deadline.
    permit_waits: u64,
    /// Largest delay of a submission behind its slot `k / rate` from run start.
    max_schedule_lag_ns: u64,
}

impl OpenLoopStats {
    fn slot_offset(slot: u64, rate_per_second: f64) -> Duration {
        Duration::from_secs_f64(slot as f64 / rate_per_second)
    }

    fn record_submission(&mut self, slot_offset: Duration, submit_offset: Duration) {
        self.submitted += 1;
        let lag = submit_offset.saturating_sub(slot_offset).as_nanos() as u64;
        self.max_schedule_lag_ns = self.max_schedule_lag_ns.max(lag);
    }

    fn summary(self, rate_per_second: f64) -> serde_json::Value {
        serde_json::json!({
            "configured_rate_per_second": rate_per_second,
            "scheduled": self.scheduled,
            "submitted": self.submitted,
            "permit_waits": self.permit_waits,
            "client_bound_reached": self.permit_waits > 0,
            "max_schedule_lag_ns": self.max_schedule_lag_ns,
        })
    }
}

/// Facts produced while the run executed, beside the attempt records.
#[derive(Debug, Default)]
struct RunExtras {
    /// References this run recorded, by statement name.
    recorded_references: BTreeMap<String, RecordedReference>,
    admission: Option<AdmissionLog>,
    /// Offset of the streams warm-up barrier release.
    measurement_start_ns: Option<u64>,
    open_loop: Option<OpenLoopStats>,
}

fn config_summary(config: &StressConfig) -> serde_json::Value {
    let mut summary = serde_json::json!({
        "url": config.url_host_port,
        "server_config_sha256": config.server_config_sha256,
        "statements": config
            .statements
            .iter()
            .map(|statement| serde_json::json!({ "name": statement.name, "sql_sha256": statement.sql_sha256 }))
            .collect::<Vec<_>>(),
        "clients": config.clients,
        "arrival_mode": config.shape.arrival_mode().as_str(),
        "mode": config.connection_mode.as_str(),
        "cancel": config
            .cancel
            .map(|cancel| serde_json::json!({ "every_n": cancel.every_n, "after_ms": cancel.after_ms })),
    });
    if let Some(seed) = config.seed {
        summary["seed"] = seed.into();
    }
    if let serde_json::Value::Object(fields) = config.shape.json() {
        summary.as_object_mut().expect("config summary is an object").extend(fields);
    }
    summary
}

/// Reference hashes and row counts, plus check counts over both phases. A
/// zero-row reference is listed in `empty_references`, because every result of
/// that statement matches any other empty result.
fn results_summary(results: &ResultsConfig, records: &[AttemptRecord], extras: &RunExtras) -> serde_json::Value {
    let reference_json = |sha256: &str, rows: u64| serde_json::json!({ "sha256": sha256, "rows": rows });
    let (references, empty, tolerance) = match results {
        ResultsConfig::Record { .. } => {
            let references = extras
                .recorded_references
                .iter()
                .map(|(name, reference)| (name.clone(), reference_json(&reference.sha256, reference.rows)))
                .collect::<serde_json::Map<_, _>>();
            let empty = extras.recorded_references.iter().filter(|(_, reference)| reference.rows == 0);
            (references, empty.map(|(name, _)| name.clone()).collect::<Vec<_>>(), None::<(f64, f64)>)
        },
        #[cfg(feature = "flight-sql-stress-check")]
        ResultsConfig::Check { references, tolerance } => {
            let summaries = references
                .iter()
                .map(|(name, reference)| {
                    let mut summary = reference_json(&reference.sha256, reference.rows);
                    summary["comparison"] = if reference.ordered { "ordered" } else { "unordered" }.into();
                    (name.clone(), summary)
                })
                .collect::<serde_json::Map<_, _>>();
            let empty = references.iter().filter(|(_, reference)| reference.rows == 0);
            (
                summaries,
                empty.map(|(name, _)| name.clone()).collect::<Vec<_>>(),
                Some((tolerance.absolute, tolerance.relative)),
            )
        },
    };
    let mut summary = serde_json::Map::new();
    summary.insert("mode".to_string(), results.mode().into());
    summary.insert("references".to_string(), references.into());
    summary.insert("empty_references".to_string(), empty.into());
    summary.insert("checks_all_phases".to_string(), serde_json::json!(reduce(records).checks));
    if let Some((absolute, relative)) = tolerance {
        // A successful attempt of a checking run that was not compared means
        // the check path skipped it; the gate requires zero.
        let unchecked = records
            .iter()
            .filter(|record| {
                record.outcome == AttemptOutcome::Succeeded
                    && record.cancel_path == CancelPath::None
                    && record.check == CheckOutcome::NotChecked
            })
            .count();
        summary.insert("unchecked_successes".to_string(), unchecked.into());
        summary.insert("float_abs_tolerance".to_string(), absolute.into());
        summary.insert("float_rel_tolerance".to_string(), relative.into());
    }
    summary.into()
}

fn build_summary(
    config: &StressConfig,
    records: &[AttemptRecord],
    wall: Duration,
    extras: &RunExtras,
) -> serde_json::Value {
    let warmup_attempts = records.iter().filter(|record| record.phase == AttemptPhase::Warmup).count() as u64;
    let measured = records.iter().filter(|record| record.phase == AttemptPhase::Measurement).collect::<Vec<_>>();
    let window = ThroughputWindow::select(config.shape, &measured, extras.measurement_start_ns);
    let completed = |statement: Option<&str>| {
        window.map_or(0, |window| {
            window.completed(records.iter().filter(|record| statement.is_none_or(|name| record.statement == name)))
        })
    };
    let open_loop = config.shape.arrival_mode() == ArrivalMode::Rate;
    let overall = reduce(measured.iter().copied());
    let overall_completed = completed(None);
    let statements = config
        .statements
        .iter()
        .map(|statement| {
            let reduction = reduce(measured.iter().copied().filter(|record| record.statement == statement.name));
            let summary = reduction_summary(&reduction, completed(Some(&statement.name)), window, open_loop);
            (statement.name.clone(), summary)
        })
        .collect::<serde_json::Map<_, _>>();
    let mut summary = serde_json::json!({
        "schema": ARTIFACT_SCHEMA,
        "profile": {
            "name": config.profile_name,
            "sha256": config.profile_sha256,
        },
        "config": config_summary(config),
        "window": {
            "elapsed_until_all_attempts_seconds": wall.as_secs_f64(),
            "warmup_attempts": warmup_attempts,
            "measurement_attempts": overall.attempts,
            "measurement_start_offset_ns": window.map(|window| window.start_ns),
            "measurement_end_offset_ns": window.map(|window| window.end_ns),
            "measured_window_seconds": window.map(|window| window.seconds),
            "measurement_attempts_ending_after_window": window.map(|window| {
                measured.iter().filter(|record| record.end_offset_ns > window.end_ns).count()
            }),
        },
        "overall": reduction_summary(&overall, overall_completed, window, open_loop),
        "statements": statements,
        "cancel_paths": overall.cancel_paths,
        "modes": {
            "pooled": mode_summary(&overall.pooled),
            "churn": mode_summary(&overall.churn),
        },
    });
    if let Some(admission) = &extras.admission {
        summary["admission"] = admission.summary();
    }
    if let (LoadShape::Rate { rate_per_second, .. }, Some(stats)) = (config.shape, extras.open_loop) {
        summary["open_loop"] = stats.summary(rate_per_second);
    }
    if let Some(results) = &config.results {
        summary["results"] = results_summary(results, records, extras);
    }
    if let Some(matched) = &config.matched_in_process {
        let latencies = &overall.succeeded_latencies_ns;
        summary["matched_in_process"] = serde_json::json!({
            "artifact_sha256": matched.artifact_sha256,
            "profile": {
                "name": matched.profile_name,
                "sha256": matched.profile_sha256,
            },
            "selection": {
                "workload_kind": matched.workload_kind,
                "query_label": matched.query_label,
                "step_index": matched.step_index,
            },
            "aggregate": {
                "measurement_seconds": matched.measurement_seconds,
                "application_qps": matched.application_qps,
                "gpu_reaching_qps": matched.gpu_reaching_qps,
                "latency_ns": {
                    "p50": matched.latency_p50_ns,
                    "p95": matched.latency_p95_ns,
                    "p99": matched.latency_p99_ns,
                },
            },
        });
        summary["delta_from_in_process"] = serde_json::json!({
            "completed_qps": floating_delta(
                window.map(|window| overall_completed as f64 / window.seconds),
                matched.application_qps,
            ),
            "latency_ns": {
                "p50": latency_delta(percentile_nearest_rank(latencies, 0.50), matched.latency_p50_ns),
                "p95": latency_delta(percentile_nearest_rank(latencies, 0.95), matched.latency_p95_ns),
                "p99": latency_delta(percentile_nearest_rank(latencies, 0.99), matched.latency_p99_ns),
            },
            "gpu_reaching_qps": {
                "in_process": matched.gpu_reaching_qps,
                "server": serde_json::Value::Null,
                "status": "not_observable_over_flight",
            },
        });
    }
    summary
}

fn floating_delta(server: Option<f64>, in_process: f64) -> serde_json::Value {
    match server {
        Some(server) => serde_json::json!({
            "server": server,
            "in_process": in_process,
            "server_minus_in_process": server - in_process,
            "percent_of_in_process": (in_process != 0.0).then(|| (server - in_process) * 100.0 / in_process),
        }),
        None => serde_json::json!({
            "server": serde_json::Value::Null,
            "in_process": in_process,
            "server_minus_in_process": serde_json::Value::Null,
            "percent_of_in_process": serde_json::Value::Null,
        }),
    }
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
type FlightClient = FlightSqlServiceClient<Channel>;

/// Pooled worker-slot connects mirror flight_sql_query's long budget, because the
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

/// State shared by every attempt of one run.
#[cfg(not(test))]
struct RunContext {
    config: StressConfig,
    started: Instant,
    sequence: AtomicU64,
    records: Mutex<Vec<AttemptRecord>>,
    /// Statements whose reference an attempt claimed, with the written file
    /// once the write succeeded.
    recorded_references: Mutex<BTreeMap<String, Option<RecordedReference>>>,
    /// Offset at which the streams warm-up barrier released.
    measurement_start_ns: OnceLock<u64>,
}

#[cfg(not(test))]
impl RunContext {
    async fn push(&self, record: AttemptRecord) {
        self.records.lock().await.push(record);
    }

    fn offset_ns(&self) -> u64 {
        self.started.elapsed().as_nanos() as u64
    }
}

/// Which attempt to run: the submitting stream (open-loop: in-flight slot),
/// its pass, the statement index and the cancel schedule.
#[cfg(not(test))]
#[derive(Debug, Clone, Copy)]
struct AttemptPlan {
    stream: usize,
    pass: Option<u64>,
    statement: usize,
    cancel: Option<CancelConfig>,
}

/// After the last attempt, the sampler polls until every device reports no
/// active attempts and no live grant bytes, or this long has passed.
#[cfg(not(test))]
const AFTER_RUN_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

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
    if let Some(ResultsConfig::Record { dir }) = &config.results {
        std::fs::create_dir(dir).map_err(|error| {
            format!(
                "flight-sql-stress results.record_reference `{}` must not already exist and have an existing parent: \
                 {error}",
                dir.display()
            )
        })?;
    }
    let sampler_client = match config.admission_sampling_interval {
        Some(_) => Some(connect(&config.url, POOLED_CONNECT_ATTEMPTS, &config.url_host_port).await?),
        None => None,
    };
    // Pooled clients connect before the run starts, so connection setup is
    // outside warm-up and the open-loop schedule, and a refused connection
    // fails the run before any stream waits at the warm-up barrier.
    let clients = connect_clients(&config).await?;
    let started = Instant::now();
    let ctx = Arc::new(RunContext {
        config,
        started,
        sequence: AtomicU64::new(0),
        records: Mutex::new(Vec::new()),
        recorded_references: Mutex::new(BTreeMap::new()),
        measurement_start_ns: OnceLock::new(),
    });
    let (stop_sampler, sampler) = match (sampler_client, ctx.config.admission_sampling_interval) {
        (Some(client), Some(period)) => {
            let (stop_tx, stop_rx) = oneshot::channel();
            let sampler = sample_admission(client, started, period, AFTER_RUN_IDLE_TIMEOUT, stop_rx);
            (Some(stop_tx), Some(tokio::spawn(sampler)))
        },
        _ => (None, None),
    };
    let open_loop = match ctx.config.shape {
        LoadShape::Streams { .. } => {
            run_closed_loop(Arc::clone(&ctx), clients, None).await?;
            None
        },
        LoadShape::Concurrency { warmup_seconds, measurement_seconds } => {
            let deadline = run_deadline(started, warmup_seconds, measurement_seconds)?;
            run_closed_loop(Arc::clone(&ctx), clients, Some(deadline)).await?;
            None
        },
        LoadShape::Rate { warmup_seconds, measurement_seconds, rate_per_second } => {
            let deadline = run_deadline(started, warmup_seconds, measurement_seconds)?;
            Some(run_open_loop(Arc::clone(&ctx), clients, deadline, rate_per_second).await?)
        },
    };
    let wall = started.elapsed();
    if let Some(stop) = stop_sampler {
        // The sampler may already have ended; its after-run polls still run.
        let _ = stop.send(());
    }
    let admission = match sampler {
        Some(sampler) => {
            Some(sampler.await.map_err(|error| format!("flight-sql-stress admission sampler failed: {error}"))?)
        },
        None => None,
    };
    let mut records = std::mem::take(&mut *ctx.records.lock().await);
    records.sort_by_key(|record| record.sequence);
    let claimed = std::mem::take(&mut *ctx.recorded_references.lock().await);
    let unwritten = claimed.iter().filter(|(_, reference)| reference.is_none()).count();
    let extras = RunExtras {
        recorded_references: claimed.into_iter().filter_map(|(name, reference)| Some((name, reference?))).collect(),
        admission,
        measurement_start_ns: ctx.measurement_start_ns.get().copied(),
        open_loop,
    };
    let config = &ctx.config;
    write_outputs(config, &records, wall, &extras)?;
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
    if unwritten > 0 {
        return Err(format!("flight-sql-stress: {unwritten} reference(s) could not be written").into());
    }
    Ok(())
}

#[cfg(not(test))]
fn run_deadline(started: Instant, warmup_seconds: u64, measurement_seconds: u64) -> ToolResult<Instant> {
    started
        .checked_add(Duration::from_secs(warmup_seconds + measurement_seconds))
        .ok_or_else(|| "flight-sql-stress duration exceeds the monotonic clock range".into())
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

/// One pooled client per stream or open-loop slot; churn mode connects per
/// attempt and gets none.
#[cfg(not(test))]
async fn connect_clients(config: &StressConfig) -> ToolResult<Vec<Option<FlightClient>>> {
    let mut clients = Vec::with_capacity(config.clients);
    for _ in 0..config.clients {
        clients.push(match config.connection_mode {
            ConnectionMode::Churn => None,
            ConnectionMode::Pooled => Some(connect(&config.url, POOLED_CONNECT_ATTEMPTS, &config.url_host_port).await?),
        });
    }
    Ok(clients)
}

#[cfg(not(test))]
async fn run_closed_loop(
    ctx: Arc<RunContext>,
    clients: Vec<Option<FlightClient>>,
    deadline: Option<Instant>,
) -> ToolResult<()> {
    let barrier = Arc::new(Barrier::new(clients.len()));
    let mut streams = Vec::new();
    for (stream, client) in clients.into_iter().enumerate() {
        let ctx = Arc::clone(&ctx);
        let barrier = Arc::clone(&barrier);
        streams.push(tokio::spawn(async move { closed_loop_stream(stream, client, ctx, deadline, barrier).await }));
    }
    for stream in streams {
        stream.await.map_err(|error| format!("flight-sql-stress stream task failed to join: {error}"))?;
    }
    Ok(())
}

/// One closed-loop stream: walks its statement order pass by pass until it
/// finishes its passes (streams) or reaches the deadline (concurrency). A
/// streams run holds every stream at a barrier before its first measured pass,
/// so the measurement window opens once all warm-up attempts have ended.
#[cfg(not(test))]
async fn closed_loop_stream(
    stream: usize,
    mut pooled: Option<FlightClient>,
    ctx: Arc<RunContext>,
    deadline: Option<Instant>,
    barrier: Arc<Barrier>,
) {
    let config = &ctx.config;
    let order = stream_order(config.statement_seed(), stream, config.statements.len());
    let mut attempt_index = 0u64;
    for pass in 0u64.. {
        if let LoadShape::Streams { passes, warmup_passes } = config.shape {
            if pass >= passes {
                break;
            }
            if pass == warmup_passes && barrier.wait().await.is_leader() {
                let _ = ctx.measurement_start_ns.set(ctx.offset_ns());
            }
        }
        for &statement in &order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return;
            }
            attempt_index += 1;
            let plan = AttemptPlan {
                stream,
                pass: Some(pass),
                statement,
                cancel: config.cancel.filter(|cancel| attempt_index.is_multiple_of(cancel.every_n)),
            };
            let record = match pooled.as_mut() {
                Some(client) => attempt_with_client(client, &ctx, plan).await,
                None => attempt_churn(&ctx, plan).await,
            };
            ctx.push(record).await;
        }
    }
}

#[cfg(not(test))]
async fn run_open_loop(
    ctx: Arc<RunContext>,
    clients: Vec<Option<FlightClient>>,
    deadline: Instant,
    rate: f64,
) -> ToolResult<OpenLoopStats> {
    let config = &ctx.config;
    let pool = clients.into_iter().map(|client| client.map(|client| Arc::new(Mutex::new(client)))).collect::<Vec<_>>();
    // Slot handoff doubles as worker identity for the `stream` column; the
    // semaphore bounds in-flight attempts to `clients` as documented.
    let (slot_tx, mut slot_rx) = mpsc::channel::<usize>(config.clients);
    for slot in 0..config.clients {
        slot_tx.send(slot).await.map_err(|_| "flight-sql-stress slot channel closed unexpectedly")?;
    }
    let mut statements = open_loop_statements(config.statement_seed(), config.statements.len());
    let mut in_flight = JoinSet::new();
    let stats = schedule_open_loop(
        config.clients,
        ctx.started,
        deadline,
        rate,
        async |ordinal: u64, permit: OwnedSemaphorePermit| -> ToolResult<()> {
            let statement = statements.next().ok_or("flight-sql-stress statement schedule ended unexpectedly")?;
            let cancel = config.cancel.filter(|cancel| ordinal.is_multiple_of(cancel.every_n));
            let slot = slot_rx.recv().await.ok_or("flight-sql-stress slot channel closed unexpectedly")?;
            let plan = AttemptPlan { stream: slot, pass: None, statement, cancel };
            let ctx = Arc::clone(&ctx);
            let slot_tx = slot_tx.clone();
            let pooled_client = pool.get(slot).cloned().flatten();
            in_flight.spawn(async move {
                // Held until the task ends so at most `clients` attempts run at once.
                let _permit = permit;
                let record = match pooled_client {
                    Some(client) => {
                        let mut guard = client.lock().await;
                        attempt_with_client(&mut guard, &ctx, plan).await
                    },
                    None => attempt_churn(&ctx, plan).await,
                };
                ctx.push(record).await;
                let _ = slot_tx.send(slot).await;
            });
            Ok(())
        },
    )
    .await?;
    let mut join_failures = 0usize;
    while let Some(result) = in_flight.join_next().await {
        if result.is_err() {
            join_failures += 1;
        }
    }
    if join_failures > 0 {
        return Err(format!("flight-sql-stress: {join_failures} scheduled attempt task(s) failed to join").into());
    }
    Ok(stats)
}

/// Fires schedule slot `k` at `k / rate` after the first tick until
/// `deadline`. Each slot takes one of `clients` permits and counts a permit
/// wait when none is free. `submit` then starts the slot's attempt with
/// its 1-based schedule ordinal and the permit, unless the deadline passed
/// during the wait.
async fn schedule_open_loop(
    clients: usize,
    started: Instant,
    deadline: Instant,
    rate: f64,
    mut submit: impl AsyncFnMut(u64, OwnedSemaphorePermit) -> ToolResult<()>,
) -> ToolResult<OpenLoopStats> {
    let mut interval = interval(Duration::from_secs_f64(1.0 / rate));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let semaphore = Arc::new(Semaphore::new(clients));
    let mut stats = OpenLoopStats::default();
    loop {
        interval.tick().await;
        if Instant::now() >= deadline {
            break;
        }
        let slot_offset = OpenLoopStats::slot_offset(stats.scheduled, rate);
        stats.scheduled += 1;
        let permit = match Arc::clone(&semaphore).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                // The client bound was reached when the slot fired, even if
                // the permit only frees after the deadline.
                stats.permit_waits += 1;
                Arc::clone(&semaphore)
                    .acquire_owned()
                    .await
                    .map_err(|_| "flight-sql-stress scheduler semaphore closed unexpectedly")?
            },
            Err(TryAcquireError::Closed) => {
                return Err("flight-sql-stress scheduler semaphore closed unexpectedly".into());
            },
        };
        // A permit that frees after the deadline would submit outside the
        // measurement span.
        if Instant::now() >= deadline {
            break;
        }
        stats.record_submission(slot_offset, started.elapsed());
        submit(stats.scheduled, permit).await?;
    }
    Ok(stats)
}

/// An attempt against a churn connection connects fresh, runs, and drops the
/// connection. A connect failure is a failed attempt, never a run abort.
#[cfg(not(test))]
async fn attempt_churn(ctx: &Arc<RunContext>, plan: AttemptPlan) -> AttemptRecord {
    // The churn attempt starts before connecting so the offset marks
    // submission and the latency includes connection establishment.
    let attempt_start = Instant::now();
    let config = &ctx.config;
    match connect(&config.url, CHURN_CONNECT_ATTEMPTS, &config.url_host_port).await {
        Ok(mut client) => attempt_with_client_at(&mut client, ctx, plan, attempt_start).await,
        Err(_) => base_record(ctx, plan, attempt_start).finish(
            ctx.started,
            attempt_start,
            AttemptOutcome::Failed,
            ErrorFacts::with_class(TRANSPORT_ERROR_CLASS),
        ),
    }
}

#[cfg(not(test))]
fn base_record(ctx: &RunContext, plan: AttemptPlan, attempt_start: Instant) -> AttemptRecord {
    let submit_offset = attempt_start.duration_since(ctx.started);
    AttemptRecord {
        sequence: ctx.sequence.fetch_add(1, Ordering::Relaxed),
        stream: plan.stream,
        pass: plan.pass,
        statement: ctx.config.statements[plan.statement].name.clone(),
        mode: ctx.config.connection_mode,
        phase: ctx.config.shape.phase(plan.pass, submit_offset),
        submit_offset_ns: submit_offset.as_nanos() as u64,
        end_offset_ns: submit_offset.as_nanos() as u64,
        latency_ns: 0,
        outcome: AttemptOutcome::Failed,
        error: ErrorFacts::none(),
        rows: 0,
        cancel_path: CancelPath::None,
        check: CheckOutcome::NotChecked,
    }
}

#[cfg(not(test))]
impl AttemptRecord {
    fn finish(mut self, started: Instant, attempt_start: Instant, outcome: AttemptOutcome, error: ErrorFacts) -> Self {
        let end = Instant::now();
        self.latency_ns = end.duration_since(attempt_start).as_nanos() as u64;
        self.end_offset_ns = end.duration_since(started).as_nanos() as u64;
        self.outcome = outcome;
        self.error = error;
        self
    }
}

/// Execute the planned statement once on an established pooled client,
/// optionally cancelling mid-execution per the cancel schedule.
#[cfg(not(test))]
async fn attempt_with_client(client: &mut FlightClient, ctx: &Arc<RunContext>, plan: AttemptPlan) -> AttemptRecord {
    attempt_with_client_at(client, ctx, plan, Instant::now()).await
}

#[cfg(not(test))]
async fn attempt_with_client_at(
    client: &mut FlightClient,
    ctx: &Arc<RunContext>,
    plan: AttemptPlan,
    attempt_start: Instant,
) -> AttemptRecord {
    let mut record = base_record(ctx, plan, attempt_start);
    let sql = ctx.config.statements[plan.statement].sql.clone();
    let flight_info = match client.execute(sql, None).await {
        Ok(flight_info) => flight_info,
        Err(error) => {
            return record.finish(ctx.started, attempt_start, AttemptOutcome::Failed, ErrorFacts::from_flight(&error));
        },
    };
    let tickets = flight_info.endpoint.iter().filter_map(|endpoint| endpoint.ticket.clone()).collect::<Vec<Ticket>>();
    if let Some(cancel) = plan.cancel {
        return cancel_attempt(client, flight_info, &tickets, cancel, record, ctx.started, attempt_start).await;
    }
    let mut capture = ctx.config.results.as_ref().map(|_| (None, Vec::new()));
    for ticket in tickets {
        if let Err(error) = drain_ticket(client, ticket, &mut record, capture.as_mut()).await {
            return record.finish(ctx.started, attempt_start, AttemptOutcome::Failed, error);
        }
    }
    let record = record.finish(ctx.started, attempt_start, AttemptOutcome::Succeeded, ErrorFacts::none());
    let Some((schema, batches)) = capture else {
        return record;
    };
    // A result without endpoints has only the FlightInfo schema.
    let schema = match schema {
        Some(schema) => schema,
        None => match flight_info.try_decode_schema() {
            Ok(schema) => Arc::new(schema),
            Err(_) => return AttemptRecord { check: CheckOutcome::Error, ..record },
        },
    };
    settle_result(ctx, plan.statement, record, ResultCapture { schema, batches }).await
}

/// Records or checks one successful result after its latency was taken.
#[cfg(not(test))]
async fn settle_result(
    ctx: &Arc<RunContext>,
    statement: usize,
    record: AttemptRecord,
    capture: ResultCapture,
) -> AttemptRecord {
    match &ctx.config.results {
        None => record,
        Some(ResultsConfig::Record { dir }) => {
            let name = ctx.config.statements[statement].name.clone();
            {
                let mut claimed = ctx.recorded_references.lock().await;
                if claimed.contains_key(&name) {
                    return record;
                }
                claimed.insert(name.clone(), None);
            }
            let path = reference_path(dir, &name);
            let rows = capture.rows();
            match spawn_blocking(move || write_reference(&path, &capture)).await {
                Ok(Ok(sha256)) => {
                    ctx.recorded_references.lock().await.insert(name, Some(RecordedReference { sha256, rows }));
                    record
                },
                Ok(Err(reason)) => {
                    eprintln!("flight-sql-stress: {reason}");
                    AttemptRecord { check: CheckOutcome::Error, ..record }
                },
                Err(_) => {
                    eprintln!("flight-sql-stress: the reference writer for {name} did not complete");
                    AttemptRecord { check: CheckOutcome::Error, ..record }
                },
            }
        },
        #[cfg(feature = "flight-sql-stress-check")]
        Some(ResultsConfig::Check { .. }) => {
            let context = Arc::clone(ctx);
            let check = spawn_blocking(move || match &context.config.results {
                Some(ResultsConfig::Check { tolerance, references }) => references
                    .get(&context.config.statements[statement].name)
                    .map_or(CheckOutcome::Error, |reference| reference.check(&capture, *tolerance)),
                _ => CheckOutcome::Error,
            })
            .await
            // A comparator panic is a check error, never an unchecked pass.
            .unwrap_or(CheckOutcome::Error);
            AttemptRecord { check, ..record }
        },
    }
}

/// Drain one endpoint ticket to its terminal state, accumulating row counts
/// and, for a run that records or checks results, the batches and schema.
#[cfg(not(test))]
async fn drain_ticket(
    client: &mut FlightClient,
    ticket: Ticket,
    record: &mut AttemptRecord,
    mut capture: Option<&mut (Option<SchemaRef>, Vec<RecordBatch>)>,
) -> Result<(), ErrorFacts> {
    let mut stream = client.do_get(ticket).await.map_err(|error| ErrorFacts::from_flight(&error))?;
    while let Some(batch) = stream.try_next().await.map_err(|error| ErrorFacts::from_flight(&error))? {
        record.rows += batch.num_rows() as u64;
        if let Some((_, batches)) = capture.as_deref_mut() {
            batches.push(batch);
        }
    }
    if let Some((schema, _)) = capture
        && schema.is_none()
    {
        *schema = stream.schema().cloned();
    }
    Ok(())
}

/// A cancel-scheduled attempt starts the first endpoint's DoGet stream, waits
/// the configured delay, then cancels via `CancelFlightInfo`. It falls back to
/// dropping the stream when the server does not support the action.
#[cfg(not(test))]
async fn cancel_attempt(
    client: &mut FlightClient,
    flight_info: FlightInfo,
    tickets: &[Ticket],
    cancel: CancelConfig,
    mut record: AttemptRecord,
    started: Instant,
    attempt_start: Instant,
) -> AttemptRecord {
    let Some(first_ticket) = tickets.first() else {
        return record.finish(started, attempt_start, AttemptOutcome::Succeeded, ErrorFacts::none());
    };
    let mut stream = match client.do_get(first_ticket.clone()).await {
        Ok(stream) => stream,
        Err(error) => {
            return record.finish(started, attempt_start, AttemptOutcome::Failed, ErrorFacts::from_flight(&error));
        },
    };
    sleep(Duration::from_millis(cancel.after_ms)).await;
    match cancel_via_flight_info(client, flight_info).await {
        Ok(()) => {
            record.cancel_path = CancelPath::FlightInfo;
            loop {
                match stream.try_next().await {
                    Ok(Some(batch)) => record.rows += batch.num_rows() as u64,
                    // The stream drained cleanly despite the accepted cancel
                    // request, so the attempt completed before the cancel took effect.
                    Ok(None) => {
                        return record.finish(started, attempt_start, AttemptOutcome::Succeeded, ErrorFacts::none());
                    },
                    Err(error) => {
                        let outcome = match &error {
                            FlightError::Tonic(status) if status.code() == tonic::Code::Cancelled => {
                                AttemptOutcome::Cancelled
                            },
                            _ => AttemptOutcome::Failed,
                        };
                        return record.finish(started, attempt_start, outcome, ErrorFacts::from_flight(&error));
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
            record.finish(started, attempt_start, AttemptOutcome::Cancelled, ErrorFacts::none())
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
fn write_outputs(
    config: &StressConfig,
    records: &[AttemptRecord],
    wall: Duration,
    extras: &RunExtras,
) -> ToolResult<()> {
    let attempts_path = config.out_dir.join("attempts.tsv");
    let mut attempts_file = std::fs::File::create(&attempts_path)?;
    writeln!(attempts_file, "{}", ATTEMPT_COLUMNS.join("\t"))?;
    for record in records {
        writeln!(attempts_file, "{}", record.to_tsv())?;
    }
    if let Some(admission) = &extras.admission {
        std::fs::write(config.out_dir.join("admission.tsv"), admission.to_tsv())?;
    }
    let summary = build_summary(config, records, wall, extras);
    ensure_redacted_json(&summary, "")?;
    std::fs::write(config.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::future::pending;
    #[cfg(feature = "flight-sql-stress-check")]
    use std::sync::Arc;

    #[cfg(feature = "flight-sql-stress-check")]
    use arrow::{
        array::{Float64Array, Int64Array, RecordBatch},
        datatypes::{DataType, Field, Schema},
    };
    use tokio::{
        sync::oneshot,
        time::{sleep, sleep_until, timeout},
    };

    #[cfg(feature = "flight-sql-stress-check")]
    use super::reference::{FloatTolerance, Reference, ResultCapture, decode_reference, encode_reference};
    use super::{
        admission_sampling::{
            AdmissionLimits, DeviceAdmission, DeviceLimits, parse_admission_payload, sample_admission_from,
        },
        *,
    };

    /// The workloads TPC-H catalog includes these files, so a statement whose
    /// SQL equals its file equals the catalog query of the same id.
    fn tpch_catalog_sql(query_id: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../workloads/src/tpch/queries/{query_id}.sql"));
        fs::read_to_string(path).expect("TPC-H catalog query file")
    }

    const BASELINE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn profile_json() -> serde_json::Value {
        serde_json::json!({
            "schema": PROFILE_SCHEMA,
            "name": "flight-pooled",
            "server": {
                "url": "http://flight.example.com:5050",
                "config_file": "server.toml"
            },
            "statements": [{ "name": "q06", "sql": "SELECT 1" }],
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

    fn streams_profile_json() -> serde_json::Value {
        let mut profile = profile_json();
        profile.as_object_mut().expect("profile object").remove("matched_in_process");
        profile["statements"] = serde_json::json!([
            { "name": "q01", "sql": "SELECT 1" },
            { "name": "q06", "sql": "SELECT 6" }
        ]);
        profile["load"] = serde_json::json!({
            "arrival_mode": "streams",
            "clients": 2,
            "seed": 7,
            "passes": 2,
            "warmup_passes": 1,
            "connection_mode": "pooled"
        });
        profile
    }

    fn parse(profile: &serde_json::Value) -> Result<FlightStressProfile, String> {
        FlightStressProfile::parse_bytes(&serde_json::to_vec(profile).expect("profile JSON"))
    }

    fn profile() -> FlightStressProfile {
        parse(&profile_json()).expect("profile")
    }

    fn baseline_json() -> serde_json::Value {
        serde_json::json!({
            "schema": SINGLE_NODE_RUN_ARTIFACT_SCHEMA,
            "status": "measured_run",
            "profile": {
                "name": "flight-baseline",
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

    fn matched(profile: &FlightStressProfile, artifact: &serde_json::Value) -> Result<MatchedInProcess, String> {
        let expected = profile.matched_in_process.as_ref().expect("matched cell");
        let shape = profile.load.shape(profile.statements.len()).expect("load shape");
        matched_in_process_from_value(artifact, "a".repeat(64), expected, shape, profile.load.clients)
    }

    /// Builds the run configuration of an inline-SQL profile without reading
    /// the server companion or the baseline file.
    fn config_from(profile: &FlightStressProfile) -> StressConfig {
        let statements = profile
            .statements
            .iter()
            .enumerate()
            .map(|(index, statement)| read_statement(Path::new("."), index, statement))
            .collect::<Result<Vec<_>, _>>()
            .expect("inline statements");
        StressConfig {
            url: profile.server.url.clone(),
            url_host_port: "flight.example.com:5050".to_string(),
            shape: profile.load.shape(statements.len()).expect("load shape"),
            statements,
            profile_name: profile.name.clone(),
            profile_sha256: "b".repeat(64),
            server_config_sha256: "c".repeat(64),
            clients: profile.load.clients,
            seed: profile.load.seed,
            connection_mode: profile.load.connection_mode,
            cancel: profile.load.cancel,
            admission_sampling_interval: None,
            results: None,
            matched_in_process: profile
                .matched_in_process
                .as_ref()
                .map(|_| matched(profile, &baseline_json()).expect("matched baseline")),
            out_dir: PathBuf::from("out"),
        }
    }

    fn config() -> StressConfig {
        config_from(&profile())
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
            stream: 0,
            pass: Some(0),
            statement: "q06".to_string(),
            mode,
            phase,
            submit_offset_ns: 0,
            end_offset_ns: latency_ns,
            latency_ns,
            outcome,
            error: ErrorFacts::with_class(error_class),
            rows: 0,
            cancel_path,
            check: CheckOutcome::NotChecked,
        }
    }

    /// A measured pooled attempt of `statement` submitted at `submit_ns`.
    fn measured(
        sequence: u64,
        statement: &str,
        submit_ns: u64,
        latency_ns: u64,
        outcome: AttemptOutcome,
        error: ErrorFacts,
        check: CheckOutcome,
    ) -> AttemptRecord {
        AttemptRecord {
            sequence,
            stream: 0,
            pass: None,
            statement: statement.to_string(),
            mode: ConnectionMode::Pooled,
            phase: AttemptPhase::Measurement,
            submit_offset_ns: submit_ns,
            end_offset_ns: submit_ns + latency_ns,
            latency_ns,
            outcome,
            error,
            rows: 0,
            cancel_path: CancelPath::None,
            check,
        }
    }

    fn overloaded(reason: &str) -> ErrorFacts {
        ErrorFacts {
            class: "ResourceExhausted".to_string(),
            code: SERVICE_OVERLOADED_CODE.to_string(),
            reason: reason.to_string(),
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
        let error = parse(&invalid).expect_err("credentials rejected");
        assert!(error.contains("credential-like"));

        let mut secret_key = profile_json();
        secret_key["server"]["api_token"] = serde_json::json!("not persisted");
        let error = parse(&secret_key).expect_err("key rejected");
        assert!(error.contains("credential-like field"));
    }

    #[test]
    fn profile_validation_names_the_conflicting_statement_load_and_results_fields() {
        let streams = parse(&streams_profile_json()).expect("streams profile");
        assert_eq!(
            streams.load.shape(streams.statements.len()).expect("shape"),
            LoadShape::Streams { passes: 2, warmup_passes: 1 }
        );
        let mut rate = streams_profile_json();
        rate["load"] = serde_json::json!({
            "arrival_mode": "rate", "clients": 64, "seed": 7, "warmup_seconds": 0,
            "measurement_seconds": 300, "rate_per_second": 0.75, "connection_mode": "pooled"
        });
        let rate = parse(&rate).expect("fractional rate");
        assert_eq!(
            rate.load.shape(2).expect("shape"),
            LoadShape::Rate { warmup_seconds: 0, measurement_seconds: 300, rate_per_second: 0.75 }
        );

        type ProfileMutation = Box<dyn Fn(&mut serde_json::Value)>;
        let cases: Vec<(&str, ProfileMutation)> = vec![
            ("at least one statement", Box::new(|p| p["statements"] = serde_json::json!([]))),
            ("repeats an earlier statement name", Box::new(|p| p["statements"][1]["name"] = "q01".into())),
            (
                "exactly one of sql or a non-empty sql_file",
                Box::new(|p| p["statements"][0]["sql_file"] = "q01.sql".into()),
            ),
            (
                "exactly one of sql or a non-empty sql_file",
                Box::new(|p| {
                    p["statements"][0].as_object_mut().expect("statement").remove("sql");
                }),
            ),
            ("statements[1].sql must be 1-16384", Box::new(|p| p["statements"][1]["sql"] = " ".into())),
            (
                "load.seed is required",
                Box::new(|p| {
                    p["load"].as_object_mut().expect("load").remove("seed");
                }),
            ),
            ("not valid for load.arrival_mode=streams", Box::new(|p| p["load"]["measurement_seconds"] = 60.into())),
            ("warmup_passes must be less than load.passes", Box::new(|p| p["load"]["warmup_passes"] = 2.into())),
            ("non-zero load.passes", Box::new(|p| p["load"]["passes"] = 0.into())),
            (
                "only valid for load.arrival_mode=streams",
                Box::new(|p| {
                    p["load"] = serde_json::json!({
                        "arrival_mode": "concurrency", "clients": 2, "seed": 7, "warmup_seconds": 0,
                        "measurement_seconds": 60, "passes": 1, "connection_mode": "pooled"
                    });
                }),
            ),
            (
                "rate load requires load.rate_per_second",
                Box::new(|p| {
                    p["load"] = serde_json::json!({
                        "arrival_mode": "rate", "clients": 2, "seed": 7, "warmup_seconds": 0,
                        "measurement_seconds": 60, "rate_per_second": 0.0, "connection_mode": "pooled"
                    });
                }),
            ),
            (
                "admission_sampling.interval_ms",
                Box::new(|p| p["admission_sampling"] = serde_json::json!({ "interval_ms": 0 })),
            ),
            (
                "exactly one non-empty record_reference or check_reference",
                Box::new(|p| {
                    p["results"] = serde_json::json!({ "record_reference": "ref", "check_reference": "ref" });
                }),
            ),
            (
                "only valid with results.check_reference",
                Box::new(|p| {
                    p["results"] = serde_json::json!({ "record_reference": "ref", "float_abs_tolerance": 0.1 });
                }),
            ),
            (
                "float_rel_tolerance must be a finite non-negative",
                Box::new(|p| {
                    p["results"] = serde_json::json!({ "check_reference": "ref", "float_rel_tolerance": -1.0 });
                }),
            ),
            (
                "matched_in_process requires exactly one statement",
                Box::new(|p| {
                    p["matched_in_process"] = profile_json()["matched_in_process"].clone();
                }),
            ),
        ];
        for (expected, mutate) in cases {
            let mut profile = streams_profile_json();
            mutate(&mut profile);
            let error = parse(&profile).expect_err(expected);
            assert!(error.contains(expected), "{expected}: {error}");
        }

        let mut streams_matched = profile_json();
        streams_matched["load"] = serde_json::json!({
            "arrival_mode": "streams", "clients": 2, "passes": 1, "warmup_passes": 0, "connection_mode": "pooled"
        });
        let error = parse(&streams_matched).expect_err("streams cannot match an in-process cell");
        assert!(error.contains("requires load.arrival_mode concurrency or rate"), "{error}");
    }

    #[test]
    fn stream_orders_are_seeded_permutations() {
        let order = stream_order(20_261_013, 0, 22);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..22).collect::<Vec<_>>());
        assert_eq!(order, stream_order(20_261_013, 0, 22));
        // Pinned so a recorded seed keeps naming the same orders.
        assert_eq!(stream_order(7, 0, 5), STREAM_7_0_ORDER);
        assert_eq!(stream_order(7, 1, 5), STREAM_7_1_ORDER);
        assert_ne!(stream_order(20_261_013, 1, 22), order);
        assert_ne!(stream_order(20_261_014, 0, 22), order);
        assert_eq!(stream_order(0, 3, 1), vec![0]);

        let draws = open_loop_statements(7, 22).take(64).collect::<Vec<_>>();
        assert_eq!(draws, open_loop_statements(7, 22).take(64).collect::<Vec<_>>());
        assert!(draws.iter().all(|statement| *statement < 22));
        assert_eq!(open_loop_statements(7, 5).take(6).collect::<Vec<_>>(), OPEN_LOOP_7_DRAWS);
        assert!(open_loop_statements(7, 1).take(8).all(|statement| statement == 0));
    }

    const STREAM_7_0_ORDER: [usize; 5] = [3, 4, 2, 0, 1];
    const STREAM_7_1_ORDER: [usize; 5] = [0, 1, 4, 2, 3];
    const OPEN_LOOP_7_DRAWS: [usize; 6] = [1, 0, 4, 2, 2, 1];

    #[test]
    fn warmup_is_the_leading_passes_for_streams_and_the_leading_seconds_otherwise() {
        let streams = LoadShape::Streams { passes: 3, warmup_passes: 1 };
        let late = Duration::from_secs(3600);
        assert_eq!(streams.phase(Some(0), late), AttemptPhase::Warmup);
        assert_eq!(streams.phase(Some(1), Duration::ZERO), AttemptPhase::Measurement);
        assert_eq!(
            LoadShape::Streams { passes: 1, warmup_passes: 0 }.phase(Some(0), Duration::ZERO),
            AttemptPhase::Measurement
        );
        let timed = LoadShape::Concurrency { warmup_seconds: 5, measurement_seconds: 30 };
        assert_eq!(timed.phase(Some(9), Duration::from_millis(4_999)), AttemptPhase::Warmup);
        assert_eq!(timed.phase(Some(0), Duration::from_secs(5)), AttemptPhase::Measurement);
    }

    #[test]
    fn shipped_flight_profiles_match_the_baseline_and_readable_server_companion() {
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/stress");
        let baseline_bytes = fs::read(fixture_dir.join("flight-baseline.json")).expect("baseline profile");
        let baseline_profile: serde_json::Value = serde_json::from_slice(&baseline_bytes).expect("baseline JSON");
        let mut artifact = baseline_json();
        artifact["profile"] = baseline_profile.clone();
        artifact["profile"]["sha256"] = serde_json::json!(sha256_bytes(&baseline_bytes));
        artifact["execution"]["measurement"]["aggregate"]["throughput"]["measurement_seconds"] =
            baseline_profile["scheduling"]["measurement_seconds"].clone();

        let mut paired_profiles = Vec::new();
        for (file, mode) in [
            ("flight-pooled.flight-profile", ConnectionMode::Pooled),
            ("flight-churn.flight-profile", ConnectionMode::Churn),
        ] {
            let profile_bytes = fs::read(fixture_dir.join(file)).expect("Flight profile");
            let profile = FlightStressProfile::parse_bytes(&profile_bytes).expect("shipped Flight profile");
            assert_eq!(profile.load.connection_mode, mode);
            let companion = fixture_dir.join(&profile.server.config_file);
            let server_bytes = fs::read(&companion).expect("profile server companion");
            ensure_redacted_server_config(&server_bytes).expect("redacted server companion");
            matched(&profile, &artifact).expect("shipped baseline matches Flight workload and load");

            let statement = &profile.statements[0];
            assert_eq!(
                statement.sql.as_deref().expect("inline SQL").split_whitespace().collect::<Vec<_>>(),
                tpch_catalog_sql(&statement.name).split_whitespace().collect::<Vec<_>>()
            );

            let mut paired: serde_json::Value = serde_json::from_slice(&profile_bytes).expect("Flight JSON");
            paired.as_object_mut().expect("profile object").remove("name");
            paired["load"].as_object_mut().expect("load object").remove("connection_mode");
            paired_profiles.push(paired);
        }
        assert_eq!(paired_profiles[0], paired_profiles[1]);
    }

    #[test]
    fn shipped_tpch_streams_profile_names_every_catalog_query_with_its_companions() {
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/stress");
        let config = read_stress_config(&fixture_dir.join("tpch-sf100-streams.flight-profile"), PathBuf::from("out"))
            .expect("shipped TPC-H streams profile");
        assert!(config.matched_in_process.is_none());
        assert_eq!(config.shape, LoadShape::Streams { passes: 2, warmup_passes: 1 });
        assert_eq!(config.admission_sampling_interval, Some(Duration::from_secs(1)));
        let names = config.statements.iter().map(|statement| statement.name.clone()).collect::<Vec<_>>();
        assert_eq!(names, (1..=22).map(|query| format!("q{query:02}")).collect::<Vec<_>>());
        for statement in &config.statements {
            assert_eq!(statement.sql, tpch_catalog_sql(&statement.name), "{}", statement.name);
        }

        let server = fs::read_to_string(fixture_dir.join("tpch-sf100-server.toml")).expect("server companion");
        let init_path = server
            .lines()
            .find_map(|line| line.strip_prefix("init_sql = \""))
            .and_then(|rest| rest.strip_suffix('"'))
            .expect("init_sql");
        let init = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(init_path))
            .expect("init_sql resolves from the repository root");
        for table in ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"] {
            assert!(
                init.contains(&format!(
                    "CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf100/parquet/{table}.parquet';"
                )),
                "{table}"
            );
        }
    }

    #[test]
    fn matched_artifact_requires_same_workload_and_load_cell() {
        let profile = profile();
        let matched_baseline = matched(&profile, &baseline_json()).expect("matched baseline");
        assert_eq!(matched_baseline.profile_name, "flight-baseline");
        assert_eq!(matched_baseline.application_qps, 3.0);

        let mut wrong_concurrency = baseline_json();
        wrong_concurrency["profile"]["scheduling"]["concurrency"] = serde_json::json!(4);
        let error = matched(&profile, &wrong_concurrency).expect_err("mismatch rejected");
        assert!(error.contains("load.clients"));

        let mut wrong_query = baseline_json();
        wrong_query["profile"]["workloads"][0]["queries"] = serde_json::json!(["q01"]);
        let error = matched(&profile, &wrong_query).expect_err("query mismatch rejected");
        assert!(error.contains("query label"));

        let mut mixed_queries = baseline_json();
        mixed_queries["profile"]["workloads"][0]["queries"] = serde_json::json!(["q01", "q06"]);
        let error = matched(&profile, &mixed_queries).expect_err("mixed baseline rejected");
        assert!(error.contains("exactly one query"));
    }

    #[test]
    fn matched_step_uses_the_selected_step_aggregate() {
        let mut profile = profile_json();
        profile["matched_in_process"]["step_index"] = serde_json::json!(1);
        let profile = parse(&profile).expect("profile");
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
        let matched = matched(&profile, &artifact).expect("second step matches");
        assert_eq!(matched.step_index, Some(1));
        assert_eq!(matched.application_qps, 4.0);
        assert_eq!(matched.latency_p99_ns, 60);
    }

    #[test]
    fn summary_reduces_only_measurement_attempts_and_reports_delta() {
        let mut config = config();
        let baseline = config.matched_in_process.as_mut().expect("matched baseline");
        baseline.application_qps = 0.5;
        baseline.latency_p50_ns = 100;
        baseline.latency_p95_ns = 100;
        baseline.latency_p99_ns = 100;
        let second = 1_000_000_000_u64;
        let attempt = |sequence, phase, outcome, error_class, submit_ns: u64, latency_ns: u64| {
            let mut attempt =
                record(sequence, phase, ConnectionMode::Pooled, outcome, error_class, CancelPath::None, latency_ns);
            attempt.submit_offset_ns = submit_ns;
            attempt.end_offset_ns = submit_ns + latency_ns;
            attempt
        };
        let mut warmup = attempt(0, AttemptPhase::Warmup, AttemptOutcome::Succeeded, NONE_LABEL, 0, 1);
        warmup.check = CheckOutcome::Mismatch;
        // The profile measures [10 s, 130 s]. The success submitted at 129 s
        // ends after the deadline: it is a measurement attempt with a latency,
        // and it is excluded from completed_qps.
        let records = vec![
            warmup,
            attempt(1, AttemptPhase::Measurement, AttemptOutcome::Succeeded, NONE_LABEL, 10 * second, 100),
            attempt(2, AttemptPhase::Measurement, AttemptOutcome::Failed, "Internal", 20 * second, 50),
            attempt(3, AttemptPhase::Measurement, AttemptOutcome::Succeeded, NONE_LABEL, 129 * second, 2 * second),
        ];
        let summary = build_summary(&config, &records, Duration::from_secs(133), &RunExtras::default());
        assert_eq!(summary["window"]["warmup_attempts"], 1);
        assert_eq!(summary["window"]["measurement_start_offset_ns"], 10 * second);
        assert_eq!(summary["window"]["measurement_end_offset_ns"], 130 * second);
        assert_eq!(summary["window"]["measured_window_seconds"], 120.0);
        assert_eq!(summary["window"]["measurement_attempts_ending_after_window"], 1);
        assert_eq!(summary["overall"]["attempts"], 3);
        assert_eq!(summary["overall"]["throughput"]["successful_completions"], 1);
        assert_eq!(summary["overall"]["throughput"]["completed_qps"], 1.0 / 120.0);
        assert_eq!(summary["overall"]["throughput"]["offered_qps"], 3.0 / 120.0);
        assert_eq!(summary["overall"]["latency_ns"]["samples"], 2);
        assert_eq!(summary["overall"]["checks"]["mismatch"], 0);
        assert_eq!(summary["statements"]["q06"]["attempts"], 3);
        assert_eq!(summary["delta_from_in_process"]["completed_qps"]["server_minus_in_process"], 1.0 / 120.0 - 0.5);
        assert_eq!(summary["delta_from_in_process"]["latency_ns"]["p95"]["server_minus_in_process"], 2 * second - 100);
        assert_eq!(summary["delta_from_in_process"]["gpu_reaching_qps"]["status"], "not_observable_over_flight");
        assert!(summary["overall"].get("rejected_latency_p50_ns").is_none());
        assert!(summary.get("admission").is_none() && summary.get("results").is_none());
        assert!(summary.get("open_loop").is_none());
        let serialized = serde_json::to_string(&summary).expect("summary serializes");
        assert!(!serialized.contains("SELECT 1"));
        ensure_redacted_json(&summary, "").expect("redacted summary");
    }

    #[test]
    fn streams_window_opens_at_the_warmup_barrier_and_closes_at_the_last_completion() {
        let config = config_from(&parse(&streams_profile_json()).expect("streams profile"));
        let second = 1_000_000_000_u64;
        let mut warmup =
            measured(0, "q01", 0, 4 * second, AttemptOutcome::Succeeded, ErrorFacts::none(), CheckOutcome::NotChecked);
        warmup.phase = AttemptPhase::Warmup;
        // The barrier released at 5 s; the first measured submit is later.
        let records = vec![
            warmup,
            measured(
                1,
                "q01",
                5 * second + 100,
                second,
                AttemptOutcome::Succeeded,
                ErrorFacts::none(),
                CheckOutcome::NotChecked,
            ),
            measured(
                2,
                "q06",
                6 * second,
                3 * second,
                AttemptOutcome::Succeeded,
                ErrorFacts::none(),
                CheckOutcome::NotChecked,
            ),
        ];
        let extras = RunExtras { measurement_start_ns: Some(5 * second), ..RunExtras::default() };
        let summary = build_summary(&config, &records, Duration::from_secs(9), &extras);
        assert_eq!(summary["window"]["measurement_start_offset_ns"], 5 * second);
        assert_eq!(summary["window"]["measured_window_seconds"], 4.0);
        assert_eq!(summary["overall"]["throughput"]["completed_qps"], 0.5);
        assert_eq!(summary["statements"]["q06"]["throughput"]["completed_qps"], 0.25);
        // Without the barrier offset a streams run has no window to divide by.
        let summary = build_summary(&config, &records, Duration::from_secs(9), &RunExtras::default());
        assert!(summary["overall"]["throughput"]["completed_qps"].is_null());
    }

    #[test]
    fn server_only_summary_reduces_each_statement_and_rejections_by_reason() {
        let mut profile = streams_profile_json();
        profile["load"] = serde_json::json!({
            "arrival_mode": "rate", "clients": 64, "seed": 7, "warmup_seconds": 0,
            "measurement_seconds": 4, "rate_per_second": 2.5, "connection_mode": "pooled"
        });
        let mut config = config_from(&parse(&profile).expect("rate profile"));
        config.results = Some(ResultsConfig::Record { dir: PathBuf::from("reference") });
        let records = vec![
            measured(0, "q01", 0, 1_000_000_000, AttemptOutcome::Succeeded, ErrorFacts::none(), CheckOutcome::Match),
            measured(
                1,
                "q06",
                500_000_000,
                300,
                AttemptOutcome::Failed,
                overloaded("admission_queue_full"),
                CheckOutcome::NotChecked,
            ),
            measured(
                2,
                "q06",
                600_000_000,
                100,
                AttemptOutcome::Failed,
                overloaded("admission_queue_full"),
                CheckOutcome::NotChecked,
            ),
            measured(
                3,
                "q06",
                700_000_000,
                200,
                AttemptOutcome::Failed,
                overloaded("admission_queue_full"),
                CheckOutcome::NotChecked,
            ),
            measured(
                4,
                "q01",
                900_000_000,
                3_100_000_000,
                AttemptOutcome::Failed,
                overloaded("queue_wait_elapsed"),
                CheckOutcome::NotChecked,
            ),
            measured(
                5,
                "q01",
                1_000_000_000,
                1_000_000_000,
                AttemptOutcome::Succeeded,
                ErrorFacts::none(),
                CheckOutcome::Mismatch,
            ),
            // Ends after the 4 s deadline, so it is excluded from completed_qps.
            measured(
                6,
                "q06",
                3_900_000_000,
                600_000_000,
                AttemptOutcome::Succeeded,
                ErrorFacts::none(),
                CheckOutcome::NotChecked,
            ),
        ];
        let extras = RunExtras {
            recorded_references: BTreeMap::from([
                ("q01".to_string(), RecordedReference { sha256: "d".repeat(64), rows: 4 }),
                ("q06".to_string(), RecordedReference { sha256: "e".repeat(64), rows: 0 }),
            ]),
            open_loop: Some(OpenLoopStats { scheduled: 11, submitted: 10, permit_waits: 2, max_schedule_lag_ns: 7 }),
            ..RunExtras::default()
        };
        let summary = build_summary(&config, &records, Duration::from_secs(5), &extras);
        assert!(summary.get("matched_in_process").is_none());
        assert!(summary.get("delta_from_in_process").is_none());
        assert_eq!(summary["config"]["statements"][1]["name"], "q06");
        assert_eq!(summary["config"]["seed"], 7);
        assert_eq!(summary["config"]["rate_per_second"], 2.5);
        assert_eq!(summary["window"]["measured_window_seconds"], 4.0);
        let overall = &summary["overall"];
        assert_eq!(overall["throughput"]["completed_qps"], 0.5);
        assert_eq!(overall["throughput"]["offered_qps"], 1.75);
        assert_eq!(summary["window"]["measurement_attempts_ending_after_window"], 1);
        assert_eq!(summary["open_loop"]["configured_rate_per_second"], 2.5);
        assert_eq!(summary["open_loop"]["scheduled"], 11);
        assert_eq!(summary["open_loop"]["submitted"], 10);
        assert_eq!(summary["open_loop"]["client_bound_reached"], true);
        assert_eq!(overall["errors"]["service_overloaded"]["admission_queue_full"], 3);
        assert_eq!(overall["errors"]["service_overloaded"]["queue_wait_elapsed"], 1);
        assert_eq!(overall["rejected_latency_p50_ns"]["admission_queue_full"], 200);
        assert_eq!(overall["rejected_latency_p50_ns"]["queue_wait_elapsed"], 3_100_000_000_u64);
        assert_eq!(overall["checks"]["match"], 1);
        assert_eq!(overall["checks"]["mismatch"], 1);
        let q01 = &summary["statements"]["q01"];
        assert_eq!(q01["attempts"], 3);
        assert_eq!(q01["throughput"]["completed_qps"], 0.5);
        assert_eq!(q01["latency_ns"]["p50"], 1_000_000_000_u64);
        assert_eq!(summary["statements"]["q06"]["throughput"]["completed_qps"], 0.0);
        assert_eq!(summary["statements"]["q06"]["latency_ns"]["p50"], 600_000_000);
        assert_eq!(summary["results"]["mode"], "record");
        assert_eq!(summary["results"]["references"]["q01"]["sha256"], "d".repeat(64));
        assert_eq!(summary["results"]["references"]["q01"]["rows"], 4);
        assert_eq!(summary["results"]["empty_references"], serde_json::json!(["q06"]));
        assert_eq!(summary["results"]["checks_all_phases"]["not_checked"], 5);
        assert!(summary["results"].get("unchecked_successes").is_none());
        ensure_redacted_json(&summary, "").expect("redacted summary");
    }

    #[test]
    fn open_loop_stats_track_the_largest_schedule_lag() {
        let mut stats = OpenLoopStats { scheduled: 3, permit_waits: 1, ..OpenLoopStats::default() };
        assert_eq!(OpenLoopStats::slot_offset(2, 4.0), Duration::from_millis(500));
        stats.record_submission(OpenLoopStats::slot_offset(0, 4.0), Duration::from_millis(1));
        stats.record_submission(OpenLoopStats::slot_offset(1, 4.0), Duration::from_millis(900));
        // A submission ahead of its slot has no lag.
        stats.record_submission(OpenLoopStats::slot_offset(2, 4.0), Duration::from_millis(400));
        assert_eq!(
            stats,
            OpenLoopStats { scheduled: 3, submitted: 3, permit_waits: 1, max_schedule_lag_ns: 650_000_000 }
        );
        let summary = stats.summary(4.0);
        assert_eq!(summary["client_bound_reached"], true);
        assert_eq!(summary["configured_rate_per_second"], 4.0);
        assert_eq!(OpenLoopStats::default().summary(4.0)["client_bound_reached"], false);
    }

    #[tokio::test(start_paused = true)]
    async fn open_loop_counts_a_permit_wait_that_ends_after_the_deadline() {
        let started = Instant::now();
        // One client at 10 slots/s for 1 s. The first attempt holds the only
        // permit for 2 s, so slot 1 waits past the deadline and submits nothing.
        let stats = schedule_open_loop(1, started, started + Duration::from_secs(1), 10.0, async |_, permit| {
            tokio::spawn(async move {
                let _permit = permit;
                sleep(Duration::from_secs(2)).await;
            });
            Ok(())
        })
        .await
        .expect("open-loop schedule");
        assert_eq!(stats, OpenLoopStats { scheduled: 2, submitted: 1, permit_waits: 1, max_schedule_lag_ns: 0 });
        assert_eq!(stats.summary(10.0)["client_bound_reached"], true);
    }

    #[test]
    fn tsv_keeps_bounded_stream_statement_and_error_columns() {
        let mut row = record(
            9,
            AttemptPhase::Measurement,
            ConnectionMode::Churn,
            AttemptOutcome::Failed,
            "Unavailable",
            CancelPath::Drop,
            1234,
        );
        row.stream = 2;
        row.pass = Some(1);
        row.statement = "bad\tlabel\n".to_string();
        row.rows = 41;
        row.submit_offset_ns = 777;
        row.end_offset_ns = 2011;
        row.error.code = SERVICE_OVERLOADED_CODE.to_string();
        row.error.reason = "queue_wait_elapsed".to_string();
        let tsv = row.to_tsv();
        assert_eq!(
            tsv,
            "9\t2\t1\tbad label \tchurn\tmeasurement\t777\t2011\t1234\tfailed\tUnavailable\tservice_overloaded\t\
             queue_wait_elapsed\t41\tdrop\tnot_checked"
        );
        assert_eq!(tsv.split('\t').count(), ATTEMPT_COLUMNS.len());
        row.pass = None;
        assert_eq!(row.to_tsv().split('\t').nth(2), Some(NONE_LABEL));
    }

    #[test]
    fn percentile_and_flight_error_labels_stay_bounded() {
        assert_eq!(percentile_nearest_rank(&[50, 10, 40, 20, 30], 0.95), Some(50));
        let tonic_error = FlightError::Tonic(Box::new(tonic::Status::cancelled("secret detail")));
        assert_eq!(flight_error_class(&tonic_error), "Cancelled");
        assert_eq!(ErrorFacts::from_flight(&tonic_error), ErrorFacts::with_class("Cancelled"));
        assert_eq!(flight_error_class(&FlightError::ProtocolError("secret".to_string())), "flight_protocol");

        let mut status = tonic::Status::resource_exhausted("admission queue is full: secret detail");
        status.metadata_mut().insert(ERROR_CODE_METADATA_KEY, "service_overloaded".parse().expect("metadata"));
        status.metadata_mut().insert(REASON_METADATA_KEY, "admission_queue_full".parse().expect("metadata"));
        assert_eq!(ErrorFacts::from_flight(&FlightError::Tonic(Box::new(status))), overloaded("admission_queue_full"));
        let mut status = tonic::Status::internal("detail");
        status.metadata_mut().insert(REASON_METADATA_KEY, "free text with spaces".parse().expect("metadata"));
        let facts = ErrorFacts::from_flight(&FlightError::Tonic(Box::new(status)));
        assert_eq!((facts.code.as_str(), facts.reason.as_str()), (NONE_LABEL, INVALID_LABEL));
    }

    const ADMISSION_PAYLOAD: &str = "# backend=query_service\n# device_ordinals=0\n# device_profile_count=1\n\
         # device_0_max_active_attempts=2\n# device_0_default_query_limit_bytes=50729201664\n\
         # max_queued_attempts=64\n# max_queue_wait_ms=30000\n# backend_metrics_backend_generation=1\n\
         # backend_metrics_generation=7\n\n[backend_metrics]\nbackend_generation\tgeneration\tconfigured_devices\n\
         1\t7\t1\n\n[backend_device_admission]\ndevice_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\t\
         resident_owner_bytes\tmax_resident_owner_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\t\
         health\n0\t12\t2\t101458403328\t0\t0\t0\t0\tOpen\n";

    #[test]
    fn admission_samples_keep_device_occupancy_and_the_configured_limits_once() {
        let payload = parse_admission_payload(ADMISSION_PAYLOAD).expect("admission payload");
        let open = DeviceAdmission {
            device_ordinal: 0,
            active_attempts: 2,
            live_grant_bytes: 101_458_403_328,
            health: "Open".to_string(),
        };
        assert_eq!(payload.devices, vec![open.clone()]);
        assert_eq!(
            payload.limits,
            AdmissionLimits {
                max_queued_attempts: 64,
                max_queue_wait_ms: 30_000,
                devices: vec![DeviceLimits {
                    device_ordinal: 0,
                    max_active_attempts: 2,
                    default_query_limit_bytes: 50_729_201_664,
                }],
            }
        );
        let missing_limit = ADMISSION_PAYLOAD.replace("# max_queue_wait_ms=30000\n", "");
        assert!(parse_admission_payload(&missing_limit).expect_err("missing limit").contains("max_queue_wait_ms"));

        let mut log = AdmissionLog::new(1000);
        assert_eq!(log.observe(5, Ok(ADMISSION_PAYLOAD.to_string())), Some(vec![open]));
        assert_eq!(log.observe(6, Err(())), None);
        // After the run, a busy poll keeps the sampler polling; an idle one ends it.
        assert!(!log.observe_after_run(7, Duration::from_millis(1), Ok(ADMISSION_PAYLOAD.to_string())));
        assert_eq!(log.summary()["after_run_idle"], false);
        assert!(log.summary().get("time_to_idle_ns").is_none());
        let idle = ADMISSION_PAYLOAD.replace("0\t12\t2\t101458403328", "0\t13\t0\t0");
        assert!(log.observe_after_run(9, Duration::from_millis(3), Ok(idle)));
        assert_eq!(
            log.to_tsv(),
            "sample\toffset_ns\tdevice_ordinal\tactive_attempts\tlive_grant_bytes\thealth\n\
             0\t5\t0\t2\t101458403328\tOpen\n1\t7\t0\t2\t101458403328\tOpen\n2\t9\t0\t0\t0\tOpen\n"
        );
        let summary = log.summary();
        assert_eq!(summary["samples"], 3);
        assert_eq!(summary["failed_samples"], 1);
        assert_eq!(summary["limits"]["devices"][0]["max_active_attempts"], 2);
        assert_eq!(summary["after_run"][0]["live_grant_bytes"], 0);
        assert_eq!(summary["after_run_idle"], true);
        assert_eq!(summary["time_to_idle_ns"], 3_000_000);
    }

    #[tokio::test(start_paused = true)]
    async fn admission_sampler_returns_within_the_idle_timeout_when_the_server_never_answers() {
        let idle_timeout = Duration::from_secs(60);
        let (stop_tx, stop_rx) = oneshot::channel();
        let sampler = tokio::spawn(sample_admission_from(
            pending::<Result<String, ()>>,
            Instant::now(),
            Duration::from_secs(1),
            idle_timeout,
            stop_rx,
        ));
        sleep(Duration::from_secs(5)).await;
        stop_tx.send(()).expect("sampler waits for stop");
        let stopped = Instant::now();
        let log = timeout(2 * idle_timeout, sampler).await.expect("sampler returns").expect("sampler task");
        assert_eq!(stopped.elapsed(), idle_timeout);
        // The in-run call is abandoned at stop; only the after-run call fails.
        assert_eq!((log.samples, log.failed_samples, log.after_run_idle), (0, 1, Some(false)));
    }

    #[tokio::test(start_paused = true)]
    async fn admission_sampler_fails_a_hung_in_run_call_and_keeps_sampling() {
        let started = Instant::now();
        let idle = ADMISSION_PAYLOAD.replace("0\t12\t2\t101458403328", "0\t13\t0\t0");
        let mut calls = 0;
        let fetch = move || {
            calls += 1;
            let answer = match calls {
                1 => None,
                2 | 3 => Some(ADMISSION_PAYLOAD.to_string()),
                _ => Some(idle.clone()),
            };
            async move {
                match answer {
                    Some(payload) => Ok(payload),
                    None => pending().await,
                }
            }
        };
        let (stop_tx, stop_rx) = oneshot::channel();
        let sampler = tokio::spawn(sample_admission_from(
            fetch,
            started,
            Duration::from_secs(1),
            Duration::from_secs(3),
            stop_rx,
        ));
        sleep_until(started + Duration::from_millis(4500)).await;
        stop_tx.send(()).expect("sampler waits for stop");
        let log = timeout(Duration::from_secs(10), sampler).await.expect("sampler returns").expect("sampler task");
        // The first call times out at 3 s; the delayed ticker then samples at
        // 3 s and 4 s, and the after-run poll at 4.5 s finds the device idle.
        let offsets = log.rows.iter().map(|(_, offset_ns, _)| *offset_ns).collect::<Vec<_>>();
        assert_eq!(offsets, vec![3_000_000_000, 4_000_000_000, 4_500_000_000]);
        assert_eq!((log.samples, log.failed_samples, log.after_run_idle), (3, 1, Some(true)));
    }

    #[cfg(feature = "flight-sql-stress-check")]
    fn capture(keys: &[i64], values: &[f64]) -> ResultCapture {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys.to_vec())), Arc::new(Float64Array::from(values.to_vec()))],
        )
        .expect("batch");
        ResultCapture { schema, batches: vec![batch] }
    }

    #[cfg(feature = "flight-sql-stress-check")]
    #[test]
    fn reference_checks_follow_the_parity_order_and_float_rules() {
        let exact = FloatTolerance { absolute: 0.0, relative: 0.0 };
        let bytes = encode_reference(&capture(&[1, 2], &[1.0, 2.0])).expect("encode");
        assert_eq!(decode_reference(&bytes).expect("decode").batches[0].num_rows(), 2);
        let swapped = capture(&[2, 1], &[2.0, 1.0]);

        // ORDER BY names every output column, so row order is part of the result.
        let ordered = Reference::from_bytes(&bytes, "SELECT k, v FROM t ORDER BY k, v", exact).expect("ordered");
        assert!(ordered.ordered);
        assert_eq!(ordered.sha256, sha256_bytes(&bytes));
        assert_eq!(ordered.check(&capture(&[1, 2], &[1.0, 2.0]), exact), CheckOutcome::Match);
        assert_eq!(ordered.check(&swapped, exact), CheckOutcome::Mismatch);

        // A partial ORDER BY compares the rows as a multiset.
        let unordered = Reference::from_bytes(&bytes, "SELECT k, v FROM t ORDER BY k", exact).expect("unordered");
        assert!(!unordered.ordered);
        assert_eq!(unordered.check(&swapped, exact), CheckOutcome::Match);
        assert_eq!(unordered.check(&capture(&[1, 2], &[1.0, 3.0]), exact), CheckOutcome::Mismatch);
        assert_eq!(unordered.check(&capture(&[1], &[1.0]), exact), CheckOutcome::Mismatch);

        let near = capture(&[1, 2], &[1.0 + 1e-9, 2.0]);
        assert_eq!(unordered.check(&near, exact), CheckOutcome::Mismatch);
        let tolerant = FloatTolerance { absolute: 1e-6, relative: 0.0 };
        assert_eq!(unordered.check(&near, tolerant), CheckOutcome::Match);

        assert!(Reference::from_bytes(b"not arrow", "SELECT 1", exact).is_err());
    }

    #[cfg(feature = "flight-sql-stress-check")]
    #[test]
    fn check_summary_counts_check_errors_unchecked_successes_and_empty_references() {
        let exact = FloatTolerance { absolute: 0.0, relative: 0.0 };
        let rows = encode_reference(&capture(&[1], &[1.0])).expect("encode");
        let empty = encode_reference(&capture(&[], &[])).expect("encode");
        let mut config = config_from(&parse(&streams_profile_json()).expect("streams profile"));
        config.results = Some(ResultsConfig::Check {
            tolerance: exact,
            references: BTreeMap::from([
                ("q01".to_string(), Reference::from_bytes(&rows, "SELECT 1", exact).expect("reference")),
                ("q06".to_string(), Reference::from_bytes(&empty, "SELECT 6", exact).expect("reference")),
            ]),
        });
        let success =
            |sequence, check| measured(sequence, "q01", 0, 1, AttemptOutcome::Succeeded, ErrorFacts::none(), check);
        let records = vec![
            success(0, CheckOutcome::Match),
            success(1, CheckOutcome::Error),
            success(2, CheckOutcome::NotChecked),
        ];
        let summary = build_summary(&config, &records, Duration::from_secs(1), &RunExtras::default());
        let results = &summary["results"];
        assert_eq!(results["mode"], "check");
        assert_eq!(results["checks_all_phases"]["error"], 1);
        assert_eq!(results["unchecked_successes"], 1);
        assert_eq!(results["references"]["q01"]["rows"], 1);
        assert_eq!(results["references"]["q06"]["comparison"], "unordered");
        assert_eq!(results["empty_references"], serde_json::json!(["q06"]));
        assert_eq!(summary["overall"]["checks"]["error"], 1);
    }
}
