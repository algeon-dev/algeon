use std::{error::Error, path::PathBuf, str::FromStr, time::Duration};

use super::{
    cap::CapAssignment,
    workload::{MixEntry, parse_mix},
};

pub(super) const ENV_MODE: &str = "NEXUS_GOODPUT_MODE";
pub(super) const ENV_MIX: &str = "NEXUS_GOODPUT_MIX";
pub(super) const ENV_TPCH_DIR: &str = "NEXUS_GOODPUT_TPCH_DIR";
pub(super) const ENV_TPCDS_DIR: &str = "NEXUS_GOODPUT_TPCDS_DIR";
pub(super) const ENV_GRAPH_EDGES: &str = "NEXUS_GOODPUT_GRAPH_EDGES";
pub(super) const ENV_TPCH_SCALE: &str = "NEXUS_GOODPUT_TPCH_SCALE";
pub(super) const ENV_TPCDS_SCALE: &str = "NEXUS_GOODPUT_TPCDS_SCALE";
pub(super) const ENV_GRAPH_DATASET: &str = "NEXUS_GOODPUT_GRAPH_DATASET";
const ENV_ARRIVAL: &str = "NEXUS_GOODPUT_ARRIVAL";
const ENV_RATE_QPS: &str = "NEXUS_GOODPUT_RATE_QPS";
const ENV_RATE_MULTIPLIER: &str = "NEXUS_GOODPUT_RATE_MULTIPLIER";
const ENV_WAVE_SIZE: &str = "NEXUS_GOODPUT_WAVE_SIZE";
const ENV_WARMUP_SECONDS: &str = "NEXUS_GOODPUT_WARMUP_SECONDS";
const ENV_WINDOW_SECONDS: &str = "NEXUS_GOODPUT_WINDOW_SECONDS";
const ENV_SEED: &str = "NEXUS_GOODPUT_SEED";
const ENV_DEVICE_ORDINALS: &str = "NEXUS_GOODPUT_DEVICE_ORDINALS";
const ENV_DEVICE_CAPACITY_BYTES: &str = "NEXUS_GOODPUT_DEVICE_CAPACITY_BYTES";
const ENV_QUERY_CAP_BYTES: &str = "NEXUS_GOODPUT_QUERY_CAP_BYTES";
const ENV_REQUEST_CAP_MANIFEST: &str = "NEXUS_GOODPUT_REQUEST_CAP_MANIFEST";
const ENV_MAX_ACTIVE_ATTEMPTS: &str = "NEXUS_GOODPUT_MAX_ACTIVE_ATTEMPTS";
const ENV_MAX_QUEUED_ATTEMPTS: &str = "NEXUS_GOODPUT_MAX_QUEUED_ATTEMPTS";
const ENV_MAX_QUEUE_WAIT_MS: &str = "NEXUS_GOODPUT_MAX_QUEUE_WAIT_MS";
const ENV_MAX_SOURCE_CHUNK_BYTES: &str = "NEXUS_GOODPUT_MAX_SOURCE_CHUNK_BYTES";
const ENV_MAX_ROW_GROUPS_PER_CHUNK: &str = "NEXUS_GOODPUT_MAX_ROW_GROUPS_PER_CHUNK";
const ENV_SLO_PATH: &str = "NEXUS_GOODPUT_SLO_PATH";
const ENV_SLO_MULTIPLIER: &str = "NEXUS_GOODPUT_SLO_MULTIPLIER";
const ENV_SLO_SENSITIVITY: &str = "NEXUS_GOODPUT_SLO_SENSITIVITY_MULTIPLIERS";
const ENV_POOL_STATE: &str = "NEXUS_GOODPUT_POOL_STATE";
const ENV_CALIBRATION_REPS: &str = "NEXUS_GOODPUT_CALIBRATION_REPS";
const ENV_CALIBRATION_WARMUP_REPS: &str = "NEXUS_GOODPUT_CALIBRATION_WARMUP_REPS";
const ENV_QUERY_TIMEOUT_MS: &str = "NEXUS_GOODPUT_QUERY_TIMEOUT_MS";
const ENV_FINGERPRINT_MAX_ROWS: &str = "NEXUS_GOODPUT_FINGERPRINT_MAX_ROWS";
const ENV_SAMPLE_INTERVAL_MS: &str = "NEXUS_GOODPUT_SAMPLE_INTERVAL_MS";
const ENV_OUT_DIR: &str = "NEXUS_GOODPUT_OUT_DIR";
const ENV_RUN_ID: &str = "NEXUS_GOODPUT_RUN_ID";
const ENV_GROUP: &str = "NEXUS_GOODPUT_GROUP";
const ENV_SOURCE_IDENTITY: &str = "NEXUS_GOODPUT_SOURCE_IDENTITY";

/// Default calibration store. Relative paths resolve against the process
/// working directory, which the matrix driver sets to the repository root.
pub(super) const DEFAULT_SLO_PATH: &str = "crates/nexus-bench/calibration/goodput_slo.tsv";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// Single-flight measurement that writes the frozen SLO/fingerprint store.
    Calibrate,
    /// Open-loop comparison measured against an already frozen store.
    Measure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArrivalProcess {
    /// Seeded exponential inter-arrival times at a fixed mean rate.
    Poisson,
    /// One synchronized release of `wave_size` requests, repeated per window.
    Wave,
}

/// Pool state of the device allocator that the observation window sees.
///
/// A `Cold` run reports the first arrivals against a freshly installed pool;
/// a `Hot` run discards a warmup segment first. The two are never averaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolState {
    Cold,
    Hot,
}

impl ArrivalProcess {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Poisson => "poisson",
            Self::Wave => "wave",
        }
    }
}

impl PoolState {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Hot => "hot",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct GoodputConfig {
    pub(super) mode: Mode,
    pub(super) mix: Vec<MixEntry>,
    pub(super) tpch_root: Option<PathBuf>,
    pub(super) tpcds_root: Option<PathBuf>,
    pub(super) graph_edges: Option<PathBuf>,
    pub(super) tpch_scale: String,
    pub(super) tpcds_scale: String,
    pub(super) graph_dataset: String,
    pub(super) arrival: ArrivalProcess,
    pub(super) rate_qps: Option<f64>,
    pub(super) rate_multiplier: f64,
    pub(super) wave_size: usize,
    pub(super) warmup_seconds: f64,
    pub(super) window_seconds: f64,
    pub(super) seed: u64,
    pub(super) device_ordinals: Vec<usize>,
    pub(super) device_capacity_bytes: u64,
    pub(super) query_cap_bytes: u64,
    pub(super) request_cap_manifest: Option<PathBuf>,
    pub(super) cap_assignment: CapAssignment,
    pub(super) max_active_attempts: usize,
    pub(super) max_queued_attempts: usize,
    pub(super) max_queue_wait: Duration,
    pub(super) max_source_chunk_bytes: u64,
    pub(super) max_row_groups_per_chunk: usize,
    pub(super) slo_path: PathBuf,
    pub(super) slo_multiplier: f64,
    pub(super) slo_sensitivity_multipliers: Vec<f64>,
    pub(super) pool_state: PoolState,
    pub(super) calibration_reps: usize,
    pub(super) calibration_warmup_reps: usize,
    pub(super) query_timeout: Duration,
    pub(super) fingerprint_max_rows: usize,
    pub(super) sample_interval: Duration,
    pub(super) out_dir: PathBuf,
    pub(super) run_id: String,
    pub(super) group: String,
    pub(super) source_identity: String,
}

impl GoodputConfig {
    pub(super) fn from_env() -> Result<Self, Box<dyn Error>> {
        let mode = match env_string(ENV_MODE, "measure").as_str() {
            "calibrate" => Mode::Calibrate,
            "measure" => Mode::Measure,
            other => return Err(format!("{ENV_MODE} must be calibrate or measure, got {other}").into()),
        };
        let arrival = match env_string(ENV_ARRIVAL, "poisson").as_str() {
            "poisson" => ArrivalProcess::Poisson,
            "wave" => ArrivalProcess::Wave,
            other => return Err(format!("{ENV_ARRIVAL} must be poisson or wave, got {other}").into()),
        };
        let pool_state = match env_string(ENV_POOL_STATE, "hot").as_str() {
            "cold" => PoolState::Cold,
            "hot" => PoolState::Hot,
            other => return Err(format!("{ENV_POOL_STATE} must be cold or hot, got {other}").into()),
        };
        let device_capacity_bytes = env_parse(ENV_DEVICE_CAPACITY_BYTES, 0_u64)?;
        let request_cap_manifest = env_path(ENV_REQUEST_CAP_MANIFEST);
        let cap_assignment =
            request_cap_manifest.as_deref().map(CapAssignment::from_path).transpose()?.unwrap_or_default();
        let config = Self {
            mode,
            mix: parse_mix(&env_required(ENV_MIX)?)?,
            tpch_root: env_path(ENV_TPCH_DIR),
            tpcds_root: env_path(ENV_TPCDS_DIR),
            graph_edges: env_path(ENV_GRAPH_EDGES),
            tpch_scale: env_string(ENV_TPCH_SCALE, "unspecified"),
            tpcds_scale: env_string(ENV_TPCDS_SCALE, "unspecified"),
            graph_dataset: env_string(ENV_GRAPH_DATASET, "unspecified"),
            arrival,
            rate_qps: env_parse_optional::<f64>(ENV_RATE_QPS)?,
            rate_multiplier: env_parse(ENV_RATE_MULTIPLIER, 1.0_f64)?,
            wave_size: env_parse(ENV_WAVE_SIZE, 0_usize)?,
            warmup_seconds: env_parse(ENV_WARMUP_SECONDS, 60.0_f64)?,
            window_seconds: env_parse(ENV_WINDOW_SECONDS, 120.0_f64)?,
            seed: env_parse(ENV_SEED, 20_260_910_u64)?,
            device_ordinals: parse_device_ordinals(&env_string(ENV_DEVICE_ORDINALS, "0"))?,
            device_capacity_bytes,
            query_cap_bytes: env_parse(ENV_QUERY_CAP_BYTES, 0_u64)?,
            request_cap_manifest,
            cap_assignment,
            max_active_attempts: env_parse(ENV_MAX_ACTIVE_ATTEMPTS, 1_usize)?,
            max_queued_attempts: env_parse(ENV_MAX_QUEUED_ATTEMPTS, 32_usize)?,
            max_queue_wait: Duration::from_millis(env_parse(ENV_MAX_QUEUE_WAIT_MS, 300_000_u64)?),
            max_source_chunk_bytes: env_parse(ENV_MAX_SOURCE_CHUNK_BYTES, 4_294_967_296_u64)?,
            max_row_groups_per_chunk: env_parse(ENV_MAX_ROW_GROUPS_PER_CHUNK, 1_024_usize)?,
            slo_path: env_path(ENV_SLO_PATH).unwrap_or_else(|| PathBuf::from(DEFAULT_SLO_PATH)),
            slo_multiplier: env_parse(ENV_SLO_MULTIPLIER, 3.0_f64)?,
            slo_sensitivity_multipliers: parse_multipliers(&env_string(ENV_SLO_SENSITIVITY, "1.5,6"))?,
            pool_state,
            calibration_reps: env_parse(ENV_CALIBRATION_REPS, 5_usize)?,
            calibration_warmup_reps: env_parse(ENV_CALIBRATION_WARMUP_REPS, 1_usize)?,
            query_timeout: Duration::from_millis(env_parse(ENV_QUERY_TIMEOUT_MS, 900_000_u64)?),
            fingerprint_max_rows: env_parse(ENV_FINGERPRINT_MAX_ROWS, 1_000_000_usize)?,
            sample_interval: Duration::from_millis(env_parse(ENV_SAMPLE_INTERVAL_MS, 250_u64)?),
            out_dir: env_path(ENV_OUT_DIR).unwrap_or_else(|| PathBuf::from("target/goodput")),
            run_id: env_string(ENV_RUN_ID, "run"),
            group: env_string(ENV_GROUP, "ungrouped"),
            source_identity: env_string(ENV_SOURCE_IDENTITY, "unrecorded"),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.device_capacity_bytes == 0 {
            return Err(format!("{ENV_DEVICE_CAPACITY_BYTES} must be set and greater than zero").into());
        }
        if self.query_cap_bytes == 0 {
            return Err(format!("{ENV_QUERY_CAP_BYTES} must be set and greater than zero").into());
        }
        if self.query_cap_bytes > self.device_capacity_bytes {
            return Err(format!("{ENV_QUERY_CAP_BYTES} must not exceed {ENV_DEVICE_CAPACITY_BYTES}").into());
        }
        self.cap_assignment.validate(self.device_capacity_bytes)?;
        if self.max_active_attempts == 0 {
            return Err(format!("{ENV_MAX_ACTIVE_ATTEMPTS} must be greater than zero").into());
        }
        if self.max_queued_attempts == 0 {
            return Err(format!("{ENV_MAX_QUEUED_ATTEMPTS} must be greater than zero").into());
        }
        if self.max_source_chunk_bytes == 0 || self.max_row_groups_per_chunk == 0 {
            return Err(format!(
                "{ENV_MAX_SOURCE_CHUNK_BYTES} and {ENV_MAX_ROW_GROUPS_PER_CHUNK} must be greater than zero"
            )
            .into());
        }
        if self.slo_multiplier <= 0.0 {
            return Err(format!("{ENV_SLO_MULTIPLIER} must be greater than zero").into());
        }
        if self.calibration_reps == 0 {
            return Err(format!("{ENV_CALIBRATION_REPS} must be greater than zero").into());
        }
        if self.sample_interval.is_zero() {
            return Err(format!("{ENV_SAMPLE_INTERVAL_MS} must be greater than zero").into());
        }
        match self.mode {
            Mode::Calibrate => self.validate_calibration(),
            Mode::Measure => self.validate_measurement(),
        }
    }

    fn validate_calibration(&self) -> Result<(), Box<dyn Error>> {
        if self.max_active_attempts != 1 {
            return Err(format!(
                "calibration is single-flight by definition; {ENV_MAX_ACTIVE_ATTEMPTS} must be 1, got {}",
                self.max_active_attempts
            )
            .into());
        }
        if self.device_ordinals.len() != 1 {
            return Err(format!("calibration must pin one device; {ENV_DEVICE_ORDINALS} selected more than one").into());
        }
        Ok(())
    }

    fn validate_measurement(&self) -> Result<(), Box<dyn Error>> {
        if self.window_seconds <= 0.0 {
            return Err(format!("{ENV_WINDOW_SECONDS} must be greater than zero").into());
        }
        if self.warmup_seconds < 0.0 {
            return Err(format!("{ENV_WARMUP_SECONDS} must not be negative").into());
        }
        if self.pool_state == PoolState::Cold && self.warmup_seconds != 0.0 {
            return Err(format!("{ENV_POOL_STATE}=cold requires {ENV_WARMUP_SECONDS}=0").into());
        }
        if self.pool_state == PoolState::Hot && self.warmup_seconds <= 0.0 {
            return Err(format!("{ENV_POOL_STATE}=hot requires a positive {ENV_WARMUP_SECONDS}").into());
        }
        match self.arrival {
            ArrivalProcess::Wave => {
                if self.wave_size == 0 {
                    return Err(format!("{ENV_ARRIVAL}=wave requires {ENV_WAVE_SIZE} greater than zero").into());
                }
                if self.wave_size > self.max_queued_attempts.saturating_add(self.max_active_attempts) {
                    return Err(format!(
                        "{ENV_WAVE_SIZE} ({}) exceeds {ENV_MAX_QUEUED_ATTEMPTS} + {ENV_MAX_ACTIVE_ATTEMPTS} ({}); the \
                         wave would be shed by queue bounds rather than measured",
                        self.wave_size,
                        self.max_queued_attempts + self.max_active_attempts
                    )
                    .into());
                }
            },
            ArrivalProcess::Poisson => {
                if self.rate_qps.is_some_and(|rate| rate <= 0.0) {
                    return Err(format!("{ENV_RATE_QPS} must be greater than zero").into());
                }
                if self.rate_qps.is_none() && self.rate_multiplier <= 0.0 {
                    return Err(format!("{ENV_RATE_MULTIPLIER} must be greater than zero").into());
                }
            },
        }
        Ok(())
    }

    /// Stream lane count required so `max_active_attempts` can actually run
    /// concurrently. Backend construction rejects a smaller lane count.
    pub(super) fn lane_count(&self) -> usize {
        self.max_active_attempts
    }

    pub(super) fn stream_pool_size(&self) -> usize {
        self.max_active_attempts.saturating_mul(4).max(self.max_active_attempts)
    }

    pub(super) fn run_dir(&self) -> PathBuf {
        self.out_dir.join(&self.run_id)
    }
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_required(name: &str) -> Result<String, Box<dyn Error>> {
    let value = env_string(name, "");
    if value.is_empty() {
        return Err(format!("{name} is required").into());
    }
    Ok(value)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).filter(|path| !path.as_os_str().is_empty())
}

fn env_parse<T: FromStr>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where T::Err: std::fmt::Display {
    match env_parse_optional(name)? {
        Some(value) => Ok(value),
        None => Ok(default),
    }
}

fn env_parse_optional<T: FromStr>(name: &str) -> Result<Option<T>, Box<dyn Error>>
where T::Err: std::fmt::Display {
    let raw = env_string(name, "");
    if raw.is_empty() {
        return Ok(None);
    }
    raw.parse::<T>().map(Some).map_err(|error| format!("failed to parse {name}: {error}").into())
}

pub(super) fn parse_device_ordinals(raw: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut ordinals = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let ordinal = part.parse::<usize>().map_err(|error| format!("failed to parse device ordinal: {error}"))?;
        if ordinals.contains(&ordinal) {
            return Err(format!("duplicate device ordinal {ordinal}").into());
        }
        ordinals.push(ordinal);
    }
    if ordinals.is_empty() {
        return Err("at least one device ordinal is required".into());
    }
    Ok(ordinals)
}

pub(super) fn parse_multipliers(raw: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut multipliers = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let multiplier = part.parse::<f64>().map_err(|error| format!("failed to parse SLO multiplier: {error}"))?;
        if multiplier <= 0.0 {
            return Err(format!("SLO multiplier must be greater than zero, got {multiplier}").into());
        }
        multipliers.push(multiplier);
    }
    Ok(multipliers)
}
