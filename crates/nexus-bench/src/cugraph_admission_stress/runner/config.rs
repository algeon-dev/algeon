use std::{
    error::Error,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use super::*;

const ENV_EDGES: &str = "NEXUS_CUGRAPH_ADMISSION_EDGES_PARQUET";
pub(super) const ENV_PAPERS: &str = "NEXUS_CUGRAPH_ADMISSION_PAPERS_PARQUET";
pub(super) const ENV_FOS: &str = "NEXUS_CUGRAPH_ADMISSION_FOS_PARQUET";
const ENV_WORKLOAD: &str = "NEXUS_CUGRAPH_ADMISSION_WORKLOAD";
const ENV_ITERATIONS: &str = "NEXUS_CUGRAPH_ADMISSION_ITERATIONS";
const ENV_STAGGER_MS: &str = "NEXUS_CUGRAPH_ADMISSION_STAGGER_MS";
const ENV_MAX_ACTIVE_ATTEMPTS_PER_DEVICE: &str = "NEXUS_CUGRAPH_ADMISSION_MAX_ACTIVE_ATTEMPTS_PER_DEVICE";
const ENV_ATTEMPT_FLOOR_BYTES: &str = "NEXUS_CUGRAPH_ADMISSION_ATTEMPT_FLOOR_BYTES";
const ENV_MAX_QUEUE_WAIT_MS: &str = "NEXUS_CUGRAPH_ADMISSION_MAX_QUEUE_WAIT_MS";
const ENV_BUDGET_FRACTION: &str = "NEXUS_CUGRAPH_ADMISSION_BUDGET_FRACTION";
const ENV_DEVICE_BUDGET_BYTES: &str = "NEXUS_CUGRAPH_ADMISSION_DEVICE_BUDGET_BYTES";
const ENV_RAW_INPUT_EDGES: &str = "NEXUS_CUGRAPH_ADMISSION_RAW_INPUT_EDGES";
const ENV_SAMPLE_INTERVAL_MS: &str = "NEXUS_CUGRAPH_ADMISSION_SAMPLE_INTERVAL_MS";
const ENV_QUERY_TIMEOUT_MS: &str = "NEXUS_CUGRAPH_ADMISSION_QUERY_TIMEOUT_MS";
const ENV_OUT_DIR: &str = "NEXUS_CUGRAPH_ADMISSION_OUT_DIR";
const ENV_RUN_ID: &str = "NEXUS_CUGRAPH_ADMISSION_RUN_ID";
const ENV_BFS_SEED: &str = "NEXUS_CUGRAPH_ADMISSION_BFS_SEED";
const ENV_BATCH_SIZE: &str = "NEXUS_CUGRAPH_ADMISSION_BATCH_SIZE";
pub(super) const ENV_PRINT_PLANNING: &str = "NEXUS_CUGRAPH_ADMISSION_PRINT_PLANNING";
const ENV_DEVICE_ORDINALS: &str = "NEXUS_CUGRAPH_ADMISSION_DEVICE_ORDINALS";
const REMOVED_ENV_SOURCE_CACHE_MAX_RETAINED_BYTES: &str = "NEXUS_CUGRAPH_ADMISSION_SOURCE_CACHE_MAX_RETAINED_BYTES";
const REMOVED_ENV_GRAPH_CACHE_MAX_RETAINED_BYTES: &str = "NEXUS_CUGRAPH_ADMISSION_GRAPH_CACHE_MAX_RETAINED_BYTES";
const ENV_WARMUP_ITERATIONS: &str = "NEXUS_CUGRAPH_ADMISSION_WARMUP_ITERATIONS";
const ENV_CUDA_PROFILER_CAPTURE: &str = "NEXUS_CUGRAPH_ADMISSION_CUDA_PROFILER_CAPTURE";

impl HarnessConfig {
    pub(super) fn from_env() -> Result<Self, Box<dyn Error>> {
        reject_retired_cache_profile_env()?;
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
            max_active_attempts_per_device: env_parse(ENV_MAX_ACTIVE_ATTEMPTS_PER_DEVICE, 2)?,
            attempt_floor_bytes: parse_optional_u64(ENV_ATTEMPT_FLOOR_BYTES)?,
            max_queue_wait_ms: env_parse(ENV_MAX_QUEUE_WAIT_MS, 60_000)?,
            budget_fraction: parse_optional_f64(ENV_BUDGET_FRACTION)?,
            device_budget_bytes: parse_optional_u64(ENV_DEVICE_BUDGET_BYTES)?,
            raw_input_edges: parse_optional_u64(ENV_RAW_INPUT_EDGES)?,
            sample_interval_ms: env_parse(ENV_SAMPLE_INTERVAL_MS, 100)?,
            query_timeout_ms: env_parse(ENV_QUERY_TIMEOUT_MS, 600_000)?,
            out_dir,
            run_id,
            bfs_seed: parse_optional_i64(ENV_BFS_SEED)?,
            batch_size: parse_optional_usize(ENV_BATCH_SIZE)?,
            device_ordinals: parse_device_ordinals(ENV_DEVICE_ORDINALS)?,
            warmup_iterations: env_parse(ENV_WARMUP_ITERATIONS, 0)?,
            cuda_profiler_capture: env_bool(ENV_CUDA_PROFILER_CAPTURE, false)?,
        })
    }

    pub(super) fn uses_phased_execution(&self) -> bool {
        self.warmup_iterations > 0 || self.cuda_profiler_capture
    }
}

fn reject_retired_cache_profile_env() -> Result<(), Box<dyn Error>> {
    for name in [REMOVED_ENV_SOURCE_CACHE_MAX_RETAINED_BYTES, REMOVED_ENV_GRAPH_CACHE_MAX_RETAINED_BYTES] {
        if std::env::var_os(name).is_some() {
            return Err(format!("{name} is no longer supported; remove this setting").into());
        }
    }
    Ok(())
}

pub(super) fn parse_workload(raw: &str) -> Result<Vec<(String, usize)>, Box<dyn Error>> {
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

pub(super) fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where T::Err: Error + 'static {
    match std::env::var(name) {
        Ok(raw) => Ok(raw.trim().parse::<T>()?),
        Err(_) => Ok(default),
    }
}

pub(super) fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(format!("failed to parse {name} as bool: {other}").into()),
        },
        Err(_) => Ok(default),
    }
}

pub(super) fn parse_optional_f64(name: &str) -> Result<Option<f64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<f64>().map_err(Into::into)).transpose()
}

pub(super) fn parse_device_ordinals(name: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let Ok(raw) = std::env::var(name) else {
        return Ok(vec![0]);
    };
    let mut ordinals = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let ordinal = part.parse::<usize>()?;
        if ordinals.contains(&ordinal) {
            return Err(format!("{name} must not contain duplicate ordinals").into());
        }
        ordinals.push(ordinal);
    }
    if ordinals.is_empty() {
        return Err(format!("{name} must contain at least one device ordinal").into());
    }
    Ok(ordinals)
}

pub(super) fn parse_optional_u64(name: &str) -> Result<Option<u64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<u64>().map_err(Into::into)).transpose()
}

pub(super) fn parse_optional_i64(name: &str) -> Result<Option<i64>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<i64>().map_err(Into::into)).transpose()
}

pub(super) fn parse_optional_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    std::env::var(name).ok().map(|raw| raw.trim().parse::<usize>().map_err(Into::into)).transpose()
}
