use super::*;

const DATASET_ROOT_ENV: &str = NEXUS_PARQUET_DIR_ENV;
const BULK_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_BULK_QUERY_IDS";
const INTERACTIVE_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_QUERY_IDS";
const BULK_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_BULK_ITERATIONS";
const INTERACTIVE_ITERATIONS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_INTERACTIVE_ITERATIONS";
const TENANT_BUDGET_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_TENANT_BUDGET_BYTES";
const DEVICE_CAPACITY_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_DEVICE_CAPACITY_BYTES";
const TENANT_MIN_GRANT_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_TENANT_MIN_GRANT_BYTES";
const TENANT_WANT_GRANT_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_TENANT_WANT_GRANT_BYTES";
pub(super) const SLOT_GUARANTEE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_SLOT_GUARANTEE";
const MAX_CONCURRENT_QUERIES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_CONCURRENT_QUERIES";
const DEVICE_ORDINALS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_DEVICE_ORDINALS";
pub(super) const MAX_ADMISSION_QUEUE_LEN_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_ADMISSION_QUEUE_LEN";
const ADMISSION_WAIT_TIMEOUT_MS_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_ADMISSION_WAIT_TIMEOUT_MS";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_SOURCE_CHUNK_BYTES";
const MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MAX_ROW_GROUPS_PER_CHUNK";
const SOURCE_CACHE_MAX_RETAINED_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_SOURCE_CACHE_MAX_RETAINED_BYTES";
const MEMORY_MANAGER_POOL_INITIAL_BYTES_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_MEMORY_MANAGER_POOL_INITIAL_BYTES";
const NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_QUERY_SERVICE_TPCH_NATIVE_PIPELINE_EXECUTION_MODE";
const QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV: &str =
    "NEXUS_QUERY_SERVICE_TPCH_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER";
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
        let dataset_root = match std::env::var_os(DATASET_ROOT_ENV) {
            Some(value) => PathBuf::from(value),
            None => default_tpch_parquet_root(TpchDefaultScale::Sf1).into(),
        };
        let max_concurrent_queries = env_usize(MAX_CONCURRENT_QUERIES_ENV, 1)?;
        let legacy_tenant_budget_bytes = optional_env_u64(TENANT_BUDGET_BYTES_ENV)?;
        let tenant_min_grant_bytes = optional_env_u64(TENANT_MIN_GRANT_BYTES_ENV)?;
        let tenant_want_grant_bytes = optional_env_u64(TENANT_WANT_GRANT_BYTES_ENV)?;
        if legacy_tenant_budget_bytes.is_some()
            && (tenant_min_grant_bytes.is_some() || tenant_want_grant_bytes.is_some())
        {
            return Err(plan_error(format!(
                "{TENANT_BUDGET_BYTES_ENV} cannot be combined with {TENANT_MIN_GRANT_BYTES_ENV} or {TENANT_WANT_GRANT_BYTES_ENV}"
            )));
        }
        let device_capacity_bytes = env_u64(
            DEVICE_CAPACITY_BYTES_ENV,
            default_device_capacity_bytes(legacy_tenant_budget_bytes, max_concurrent_queries),
        )?;
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
            tenant_min_grant_bytes: tenant_min_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_min_grant_bytes),
            tenant_want_grant_bytes: tenant_want_grant_bytes
                .or(legacy_tenant_budget_bytes)
                .unwrap_or(default_want_grant_bytes),
            slot_guarantee: env_slot_guarantee(SLOT_GUARANTEE_ENV, SlotGuarantee::GuaranteedMinimum)?,
            max_concurrent_queries,
            max_admission_queue_len: env_usize(MAX_ADMISSION_QUEUE_LEN_ENV, 64)?,
            admission_wait_timeout: Duration::from_millis(env_u64(ADMISSION_WAIT_TIMEOUT_MS_ENV, 30_000)?),
            max_source_chunk_bytes: env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, 64 * 1024 * 1024)?,
            max_row_groups_per_chunk: env_usize(MAX_ROW_GROUPS_PER_CHUNK_ENV, 1)?,
            source_cache_max_retained_bytes: optional_env_u64(SOURCE_CACHE_MAX_RETAINED_BYTES_ENV)?,
            memory_manager_pool_initial_bytes: optional_env_u64(MEMORY_MANAGER_POOL_INITIAL_BYTES_ENV)?,
            native_pipeline_execution_mode: env_native_pipeline_execution_mode(
                NATIVE_PIPELINE_EXECUTION_MODE_ENV,
                NativePipelineExecutionMode::StrictBounded,
            )?,
            query_memory_domain_ceiling_multiplier: optional_env_u64(QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV)?,
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
        validate_nonzero(TENANT_MIN_GRANT_BYTES_ENV, self.tenant_min_grant_bytes)?;
        validate_nonzero(TENANT_WANT_GRANT_BYTES_ENV, self.tenant_want_grant_bytes)?;
        if self.tenant_want_grant_bytes < self.tenant_min_grant_bytes {
            return Err(plan_error(format!(
                "{TENANT_WANT_GRANT_BYTES_ENV} must be at least {TENANT_MIN_GRANT_BYTES_ENV}"
            )));
        }
        if self.tenant_min_grant_bytes > self.device_capacity_bytes {
            return Err(plan_error(format!(
                "{TENANT_MIN_GRANT_BYTES_ENV} must not exceed {DEVICE_CAPACITY_BYTES_ENV}"
            )));
        }
        if self.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
            let required_min_capacity = self
                .tenant_min_grant_bytes
                .saturating_mul(u64::try_from(self.max_concurrent_queries).unwrap_or(u64::MAX));
            if required_min_capacity > self.device_capacity_bytes {
                return Err(plan_error(format!(
                    "{TENANT_MIN_GRANT_BYTES_ENV} must fit {MAX_CONCURRENT_QUERIES_ENV} within {DEVICE_CAPACITY_BYTES_ENV} when {SLOT_GUARANTEE_ENV}=guaranteed_minimum"
                )));
            }
        }
        validate_nonzero(MAX_CONCURRENT_QUERIES_ENV, self.max_concurrent_queries)?;
        validate_nonzero(MAX_ADMISSION_QUEUE_LEN_ENV, self.max_admission_queue_len)?;
        validate_nonzero(MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        if let Some(max_retained_bytes) = self.source_cache_max_retained_bytes {
            validate_nonzero(SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, max_retained_bytes)?;
        }
        if let Some(initial_pool_bytes) = self.memory_manager_pool_initial_bytes {
            validate_nonzero(MEMORY_MANAGER_POOL_INITIAL_BYTES_ENV, initial_pool_bytes)?;
            if initial_pool_bytes > self.device_capacity_bytes {
                return Err(plan_error(format!(
                    "{MEMORY_MANAGER_POOL_INITIAL_BYTES_ENV} must not exceed {DEVICE_CAPACITY_BYTES_ENV}"
                )));
            }
            usize::try_from(initial_pool_bytes).map_err(|_| {
                plan_error(format!("{MEMORY_MANAGER_POOL_INITIAL_BYTES_ENV} exceeds the platform usize range"))
            })?;
            usize::try_from(self.device_capacity_bytes)
                .map_err(|_| plan_error(format!("{DEVICE_CAPACITY_BYTES_ENV} exceeds the platform usize range")))?;
        }
        if let Some(multiplier) = self.query_memory_domain_ceiling_multiplier {
            validate_nonzero(QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV, multiplier)?;
        }
        if self.stream_idle_timeout.is_zero() {
            return Err(plan_error(format!("{STREAM_IDLE_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        if self.admission_wait_timeout.is_zero() {
            return Err(plan_error(format!("{ADMISSION_WAIT_TIMEOUT_MS_ENV} must be greater than zero")));
        }
        validate_query_ids(&self.bulk_queries)?;
        validate_query_ids(&self.interactive_queries)?;
        if self.profile.is_none() {
            validate_synchronized_admission_queue(
                MAX_ADMISSION_QUEUE_LEN_ENV,
                self.max_admission_queue_len,
                self.synchronized_request_count(),
            )
            .map_err(plan_error)?;
        }
        if let Some(profile) = &self.profile {
            validate_nonzero(PROFILE_LANES_ENV, profile.lanes)?;
            validate_nonzero(PROFILE_WARMUP_ITERATIONS_ENV, profile.warmup_iterations)?;
            validate_nonzero(PROFILE_MEASURED_ITERATIONS_ENV, profile.measured_iterations)?;
            if self.source_cache_max_retained_bytes.is_some() && profile.warmup_iterations < 2 {
                return Err(plan_error(format!(
                    "{PROFILE_WARMUP_ITERATIONS_ENV} must be at least 2 so the final warmup iteration can verify a cache-only path"
                )));
            }
            if profile.lanes > self.max_concurrent_queries {
                return Err(plan_error(format!("{PROFILE_LANES_ENV} must not exceed {MAX_CONCURRENT_QUERIES_ENV}")));
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

    pub(super) fn service_config(&self) -> QueryServiceConfig {
        QueryServiceConfig {
            device_capacity_bytes: Some(self.device_capacity_bytes),
            max_concurrent_queries: self.max_concurrent_queries,
            slot_guarantee: self.slot_guarantee,
            max_admission_queue_len: self.max_admission_queue_len,
            admission_wait_timeout: self.admission_wait_timeout,
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            local_parquet_source_cache_max_retained_bytes: self.source_cache_max_retained_bytes,
            native_pipeline_execution_mode: self.native_pipeline_execution_mode,
            query_memory_domain: self
                .query_memory_domain_ceiling_multiplier
                .map(|multiplier| QueryMemoryDomainPolicy::new().with_ceiling_multiplier(multiplier)),
            device_memory_manager_base: self.memory_manager_pool_initial_bytes.map_or_else(
                cudf::DeviceMemoryManagerBase::default,
                |initial_pool_bytes| cudf::DeviceMemoryManagerBase::Pool {
                    initial_pool_size: usize::try_from(initial_pool_bytes)
                        .expect("validated pool initial bytes fit usize"),
                    maximum_pool_size: Some(
                        usize::try_from(self.device_capacity_bytes).expect("validated device capacity fits usize"),
                    ),
                },
            ),
            ..QueryServiceConfig::default()
        }
    }
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

fn env_slot_guarantee(name: &'static str, default: SlotGuarantee) -> DataFusionResult<SlotGuarantee> {
    match std::env::var_os(name) {
        Some(value) => parse_slot_guarantee(name, &value.to_string_lossy()),
        None => Ok(default),
    }
}

fn env_native_pipeline_execution_mode(
    name: &'static str,
    default: NativePipelineExecutionMode,
) -> DataFusionResult<NativePipelineExecutionMode> {
    match std::env::var_os(name) {
        Some(value) => match value.to_string_lossy().trim().to_ascii_lowercase().as_str() {
            "strict_bounded" => Ok(NativePipelineExecutionMode::StrictBounded),
            "adaptive" => Ok(NativePipelineExecutionMode::Adaptive),
            other => {
                Err(plan_error(format!("failed to parse {name}: expected strict_bounded or adaptive, got {other}")))
            },
        },
        None => Ok(default),
    }
}

fn parse_slot_guarantee(name: &'static str, value: &str) -> DataFusionResult<SlotGuarantee> {
    match value.trim().to_ascii_lowercase().as_str() {
        "guaranteed_minimum" => Ok(SlotGuarantee::GuaranteedMinimum),
        "optimistic" => Ok(SlotGuarantee::Optimistic),
        other => {
            Err(plan_error(format!("failed to parse {name}: expected guaranteed_minimum or optimistic, got {other}")))
        },
    }
}

pub(super) fn slot_guarantee_name(slot_guarantee: SlotGuarantee) -> &'static str {
    match slot_guarantee {
        SlotGuarantee::GuaranteedMinimum => "guaranteed_minimum",
        SlotGuarantee::Optimistic => "optimistic",
    }
}

pub(super) fn native_pipeline_execution_mode_name(mode: NativePipelineExecutionMode) -> &'static str {
    match mode {
        NativePipelineExecutionMode::StrictBounded => "strict_bounded",
        NativePipelineExecutionMode::Adaptive => "adaptive",
    }
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

fn default_device_capacity_bytes(legacy_tenant_budget_bytes: Option<u64>, max_concurrent_queries: usize) -> u64 {
    if let Some(tenant_budget_bytes) = legacy_tenant_budget_bytes {
        return tenant_budget_bytes.saturating_mul(u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX));
    }
    NexusNativeExecutionConfig::auto_sized_device_budget_bytes()
}

fn grant_fraction_bytes(device_capacity_bytes: u64, numerator: u64, denominator: u64) -> u64 {
    device_capacity_bytes.saturating_mul(numerator) / denominator
}
