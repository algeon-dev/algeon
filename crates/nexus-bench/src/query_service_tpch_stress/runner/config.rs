use super::*;

const DATASET_ROOT_ENV: &str = NEXUS_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS";
const DEVICE_CAPACITY_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_DEVICE_CAPACITY_BYTES";
pub(super) const ATTEMPT_FLOOR_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_ATTEMPT_FLOOR_BYTES";
const SOFT_WORKING_SET_HINT_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_SOFT_WORKING_SET_HINT_BYTES";
const MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_ACTIVE_ATTEMPTS_PER_DEVICE";
const DEVICE_ORDINALS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_DEVICE_ORDINALS";
pub(super) const MAX_QUEUED_ATTEMPTS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_QUEUED_ATTEMPTS";
const MAX_QUEUE_WAIT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_QUEUE_WAIT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK";
const REMOVED_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_SOURCE_CACHE_MAX_RETAINED_BYTES";
const WARMUP_ROUNDS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_WARMUP_ROUNDS";
const STREAM_IDLE_TIMEOUT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_STREAM_IDLE_TIMEOUT_MS";
const CUDA_PROFILER_CAPTURE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_CUDA_PROFILER_CAPTURE";
const PROFILE_LANES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_PROFILE_LANES";
const PROFILE_QUERY_ID_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_PROFILE_QUERY_ID";
const PROFILE_WARMUP_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_PROFILE_WARMUP_ITERATIONS";
const PROFILE_MEASURED_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_PROFILE_MEASURED_ITERATIONS";
const PROFILE_EVIDENCE_PATH_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_PROFILE_EVIDENCE_PATH";
const OVERLOAD_PROBE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_OVERLOAD_PROBE";
impl StressConfig {
    pub(super) fn from_env() -> DataFusionResult<Self> {
        reject_retired_source_cache_env()?;
        let dataset_root = match std::env::var_os(DATASET_ROOT_ENV) {
            Some(value) => PathBuf::from(value),
            None => default_tpch_parquet_root(TpchDefaultScale::Sf1).into(),
        };
        let max_active_attempts_per_device = env_usize(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, 1)?;
        let attempt_floor_bytes = optional_env_u64(ATTEMPT_FLOOR_BYTES_ENV)?;
        let soft_working_set_hint_bytes = optional_env_u64(SOFT_WORKING_SET_HINT_BYTES_ENV)?;
        let device_capacity_bytes = env_u64(DEVICE_CAPACITY_BYTES_ENV, default_device_capacity_bytes())?;
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
            attempt_floor_bytes: attempt_floor_bytes.unwrap_or(default_min_grant_bytes),
            soft_working_set_hint_bytes: soft_working_set_hint_bytes.unwrap_or(default_want_grant_bytes),
            max_active_attempts_per_device,
            max_queued_attempts: env_usize(MAX_QUEUED_ATTEMPTS_ENV, 64)?,
            max_queue_wait: Duration::from_millis(env_u64(MAX_QUEUE_WAIT_MS_ENV, 30_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
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

    pub(super) fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero(BULK_ITERATIONS_ENV, self.bulk_iterations)?;
        validate_nonzero(INTERACTIVE_ITERATIONS_ENV, self.interactive_iterations)?;
        validate_nonzero(DEVICE_CAPACITY_BYTES_ENV, self.device_capacity_bytes)?;
        validate_nonzero(ATTEMPT_FLOOR_BYTES_ENV, self.attempt_floor_bytes)?;
        validate_nonzero(SOFT_WORKING_SET_HINT_BYTES_ENV, self.soft_working_set_hint_bytes)?;
        if self.soft_working_set_hint_bytes < self.attempt_floor_bytes {
            return Err(plan_error(format!(
                "{SOFT_WORKING_SET_HINT_BYTES_ENV} must be at least {ATTEMPT_FLOOR_BYTES_ENV}"
            )));
        }
        if self.attempt_floor_bytes > self.device_capacity_bytes {
            return Err(plan_error(format!("{ATTEMPT_FLOOR_BYTES_ENV} must not exceed {DEVICE_CAPACITY_BYTES_ENV}")));
        }
        let floor_total = self.attempt_floor_bytes.saturating_mul(self.max_active_attempts_per_device as u64);
        if floor_total > self.device_capacity_bytes {
            return Err(plan_error(format!(
                "{ATTEMPT_FLOOR_BYTES_ENV} times {MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV} must not exceed {DEVICE_CAPACITY_BYTES_ENV}"
            )));
        }
        validate_nonzero(MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV, self.max_active_attempts_per_device)?;
        validate_nonzero(MAX_QUEUED_ATTEMPTS_ENV, self.max_queued_attempts)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if self.stream_idle_timeout.is_zero() {
            return Err(plan_error(format!("{STREAM_IDLE_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        if self.max_queue_wait.is_zero() {
            return Err(plan_error(format!("{MAX_QUEUE_WAIT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        if self.profile.is_none() {
            validate_synchronized_admission_queue(
                MAX_QUEUED_ATTEMPTS_ENV,
                self.max_queued_attempts,
                self.synchronized_request_count(),
            )
            .map_err(plan_error)?;
        }
        if let Some(profile) = &self.profile {
            validate_nonzero(PROFILE_LANES_ENV, profile.lanes)?;
            validate_nonzero(PROFILE_WARMUP_ITERATIONS_ENV, profile.warmup_iterations)?;
            validate_nonzero(PROFILE_MEASURED_ITERATIONS_ENV, profile.measured_iterations)?;
            if profile.lanes > self.max_active_attempts_per_device {
                return Err(plan_error(format!(
                    "{PROFILE_LANES_ENV} must not exceed {MAX_ACTIVE_ATTEMPTS_PER_DEVICE_ENV}"
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

    pub(super) fn synchronized_request_count(&self) -> usize {
        self.bulk_queries.len().saturating_mul(self.bulk_iterations).saturating_add(
            self.interactive_queries.len().saturating_mul(self.interactive_iterations).saturating_mul(2),
        )
    }

    pub(super) fn service_config(&self) -> QueryExecutionConfig {
        QueryExecutionConfig {
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            ..QueryExecutionConfig::default()
        }
    }

    pub(super) fn queue_bounds(&self) -> QueueBounds {
        QueueBounds {
            max_waiters: self.max_queued_attempts,
            max_wait: self.max_queue_wait,
            max_overtakes_per_waiter: 4,
        }
    }
}

fn reject_retired_source_cache_env() -> DataFusionResult<()> {
    if std::env::var_os(REMOVED_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV).is_some() {
        return Err(plan_error(format!(
            "{REMOVED_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV} is no longer supported; remove this setting"
        )));
    }
    Ok(())
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

fn default_device_capacity_bytes() -> u64 {
    // This standalone benchmark profile must not probe GPU 0 and reuse that
    // capacity for every selected device. Server backends resolve each device.
    NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES
}

fn grant_fraction_bytes(device_capacity_bytes: u64, numerator: u64, denominator: u64) -> u64 {
    device_capacity_bytes.saturating_mul(numerator) / denominator
}
