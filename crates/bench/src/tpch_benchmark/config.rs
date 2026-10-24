use std::{collections::BTreeSet, time::Duration};

use algeon_datafusion::{
    backend::AlgeonGpuBackend,
    native_exec::SourceChunking,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
};
use query_planning::execution_contracts::SourceChunkPlanningMode;
use query_runtime::runtime::{GpuDeviceMemoryLimit, SourceSchedulerPolicy};

use super::{
    AlgeonNativeExecutionConfig, DataFusionError, DataFusionResult, DeviceProvenance, SessionStateBuilder,
    TPCH_QUERIES, TpchQuery, plan_error,
};
use crate::{gpu_backend::build_installed_gpu_backend, report_workload::RelationalExecutionSelection};

pub(super) const ALGEON_TPCH_BENCH_NATIVE_PROFILE_ENV: &str = "ALGEON_TPCH_BENCH_NATIVE_PROFILE";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
pub(super) const ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE";

pub(super) const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
pub(super) const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT: usize = 2;
const REMOVED_NATIVE_SOURCE_CACHE_POLICY_ENV: &str = "ALGEON_TPCH_BENCH_NATIVE_SOURCE_CACHE_POLICY";
const REMOVED_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV: &str =
    "ALGEON_TPCH_BENCH_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES";
const REMOVED_NATIVE_MEMORY_POLICY_ENV: &str = "ALGEON_TPCH_BENCH_NATIVE_MEMORY_POLICY";

pub(super) const TPCH_NATIVE_PROFILE_CONSERVATIVE: &str = "conservative";
pub(super) const TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT: &str = "sf100_gpu_throughput";
pub(super) const TPCH_NATIVE_PROFILE_CUSTOM: &str = "custom";
pub(super) const TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES: u64 = 34_359_738_368;
pub(super) const TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeTpchBenchmarkConfig {
    pub(super) query_ids: Vec<String>,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
    pub(super) dataset_mode: NativeTpchDatasetMode,
    pub(super) parquet_pushdown_filters: Option<bool>,
    pub(super) native_execution: RelationalExecutionSelection,
    pub(super) native_memory_policy: NativeTpchMemoryPolicy,
    pub(super) max_source_reads_in_flight: usize,
}

#[derive(Debug)]
pub(super) struct NativeTpchRuntimeResources {
    backend: Option<AlgeonGpuBackend>,
}

impl NativeTpchRuntimeResources {
    pub(super) fn unavailable() -> Self {
        Self { backend: None }
    }

    pub(super) fn shutdown(mut self) -> DataFusionResult<()> {
        if let Some(backend) = self.backend.take() {
            backend
                .close_and_wait(Duration::from_secs(30))
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
        }
        Ok(())
    }

    pub(super) fn backend(&self) -> Option<&AlgeonGpuBackend> {
        self.backend.as_ref()
    }

    pub(super) fn optimizer_config(&self) -> Option<AlgeonNativeOptimizerConfig> {
        self.backend.as_ref().map(|backend| backend.optimizer_config().clone())
    }

    pub(super) fn install_on(&self, builder: SessionStateBuilder) -> DataFusionResult<SessionStateBuilder> {
        match self.backend.as_ref() {
            Some(backend) => backend.install_on(builder).map_err(|error| DataFusionError::External(Box::new(error))),
            None => Ok(builder),
        }
    }

    pub(super) fn device_provenance(&self) -> Vec<DeviceProvenance> {
        self.backend
            .as_ref()
            .map(|backend| {
                backend
                    .device_startup_records()
                    .iter()
                    .map(|record| DeviceProvenance {
                        ordinal: record.device_ordinal,
                        model_name: record.model_name.clone(),
                        uuid: record.uuid.clone(),
                        physical_total_bytes: record.physical_total_bytes,
                        cuda_total_bytes: record.facts.total_bytes,
                        managed_capacity_bytes: record.capacity_bytes,
                        default_query_limit_bytes: record.resource_profile.default_query_limit_bytes,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl NativeTpchBenchmarkConfig {
    pub(super) fn from_env() -> DataFusionResult<Self> {
        reject_retired_source_cache_env()?;
        reject_retired_memory_policy_env()?;
        let config = Self {
            query_ids: query_ids_from_env("ALGEON_TPCH_BENCH_QUERY_IDS")?,
            warmup_iterations: env_usize("ALGEON_TPCH_BENCH_WARMUPS", 0)?,
            iterations: env_usize("ALGEON_TPCH_BENCH_ITERATIONS", 1)?,
            dataset_mode: NativeTpchDatasetMode::from_env("ALGEON_TPCH_BENCH_DATASET_MODE")?,
            parquet_pushdown_filters: optional_env_bool("ALGEON_TPCH_BENCH_PARQUET_PUSHDOWN_FILTERS")?,
            native_execution: RelationalExecutionSelection::strict_native(),
            native_memory_policy: NativeTpchMemoryPolicy::bounded_from_profile_settings(
                native_tpch_profile_settings_from_env()?,
            ),
            max_source_reads_in_flight: env_usize(
                "ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT",
                DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT,
            )?,
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("ALGEON_TPCH_BENCH_ITERATIONS", self.iterations)?;
        validate_query_ids(&self.query_ids)?;
        validate_nonzero("ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT", self.max_source_reads_in_flight)?;
        self.native_memory_policy.validate()
    }

    pub(super) fn selected_queries(&self) -> Vec<&'static TpchQuery> {
        self.query_ids
            .iter()
            .map(|query_id| {
                TPCH_QUERIES
                    .iter()
                    .find(|query| query.id == query_id)
                    .expect("validated benchmark config references only known TPC-H query ids")
            })
            .collect()
    }

    pub(super) fn native_execution_config_with_resources(
        &self,
        _resources: &NativeTpchRuntimeResources,
    ) -> AlgeonNativeExecutionConfig {
        let policy = self.native_memory_policy;
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(
                policy.max_source_chunk_bytes,
                policy.max_row_groups_per_chunk,
                policy.max_retry_attempts,
                policy.min_retry_chunk_row_groups,
            )
            .expect("validated native memory policy has nonzero source chunking"),
        )
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(self.max_source_reads_in_flight))
        .with_source_chunk_planning_mode(policy.source_chunk_planning_mode)
    }

    pub(super) fn new_runtime_resources(&self) -> DataFusionResult<NativeTpchRuntimeResources> {
        let execution_config = self.native_execution_config_with_resources(&NativeTpchRuntimeResources::unavailable());
        let optimizer_config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(execution_config);
        let backend = build_installed_gpu_backend(
            optimizer_config,
            self.native_memory_policy.device_budget.memory_limit(),
            self.max_source_reads_in_flight,
            1,
            AlgeonGpuExecutionMode::NativeRequired,
        )
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
        Ok(NativeTpchRuntimeResources { backend: Some(backend) })
    }
}

impl Default for NativeTpchBenchmarkConfig {
    fn default() -> Self {
        Self {
            query_ids: TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect(),
            warmup_iterations: 0,
            iterations: 1,
            dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf1),
            parquet_pushdown_filters: None,
            native_execution: RelationalExecutionSelection::strict_native(),
            native_memory_policy: NativeTpchMemoryPolicy::default(),
            max_source_reads_in_flight: DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchDatasetMode {
    External(NativeTpchScaleFactor),
    AllowTiny,
}

impl NativeTpchDatasetMode {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let mode = match std::env::var_os(name) {
            None => "sf1_external".to_string(),
            Some(value) => value.to_string_lossy().to_ascii_lowercase(),
        };
        match mode.as_str() {
            "sf1_external" | "external" | "sf1" => Ok(Self::External(NativeTpchScaleFactor::Sf1)),
            "sf10_external" | "sf10" => Ok(Self::External(NativeTpchScaleFactor::Sf10)),
            "sf100_external" | "sf100" => Ok(Self::External(NativeTpchScaleFactor::Sf100)),
            "allow_tiny" | "dev_tiny" | "tiny" => Ok(Self::AllowTiny),
            other => Err(plan_error(format!(
                "invalid {name}: {other}; expected sf1_external, sf10_external, sf100_external, or allow_tiny"
            ))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::External(scale_factor) => scale_factor.dataset_mode_str(),
            Self::AllowTiny => "allow_tiny",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchScaleFactor {
    Sf1,
    Sf10,
    Sf100,
}

impl NativeTpchScaleFactor {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Sf1 => "sf1",
            Self::Sf10 => "sf10",
            Self::Sf100 => "sf100",
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Sf1 => "SF1",
            Self::Sf10 => "SF10",
            Self::Sf100 => "SF100",
        }
    }

    pub(super) const fn dataset_mode_str(self) -> &'static str {
        match self {
            Self::Sf1 => "sf1_external",
            Self::Sf10 => "sf10_external",
            Self::Sf100 => "sf100_external",
        }
    }
}

/// Managed device capacity requested by the benchmark runtime backend.
/// The per-query cap matches the resolved managed capacity, without an
/// independent default query limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchDeviceBudget {
    /// Resolves capacity from the installed device via the runtime automatic
    /// policy (physical memory minus device reserve threshold).
    Automatic,
    /// Explicit managed capacity; the runtime rejects values above owned
    /// device memory with a structured error and does not clamp them.
    Explicit { bytes: u64 },
}

impl NativeTpchDeviceBudget {
    pub(super) fn memory_limit(self) -> GpuDeviceMemoryLimit {
        match self {
            Self::Automatic => GpuDeviceMemoryLimit::default(),
            Self::Explicit { bytes } => GpuDeviceMemoryLimit::Explicit { limit_bytes: bytes },
        }
    }

    pub(super) const fn as_policy_str(self) -> &'static str {
        match self {
            Self::Automatic => "automatic",
            Self::Explicit { .. } => "explicit",
        }
    }
}

impl std::fmt::Display for NativeTpchDeviceBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => f.write_str("automatic"),
            Self::Explicit { bytes } => write!(f, "{bytes}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeTpchMemoryPolicy {
    pub(super) profile_name: &'static str,
    pub(super) profile_source: &'static str,
    pub(super) device_budget: NativeTpchDeviceBudget,
    pub(super) max_source_chunk_bytes: u64,
    pub(super) max_row_groups_per_chunk: usize,
    pub(super) max_retry_attempts: usize,
    pub(super) min_retry_chunk_row_groups: usize,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
}

impl NativeTpchMemoryPolicy {
    #[cfg(test)]
    pub(super) const fn bounded_custom(
        device_budget: NativeTpchDeviceBudget,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_chunk_planning_mode: SourceChunkPlanningMode,
    ) -> Self {
        Self {
            profile_name: TPCH_NATIVE_PROFILE_CUSTOM,
            profile_source: "direct_config",
            device_budget,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            source_chunk_planning_mode,
        }
    }

    pub(super) fn bounded_from_profile_settings(profile: NativeTpchProfileSettings) -> Self {
        Self {
            profile_name: profile.profile_name,
            profile_source: profile.profile_source,
            device_budget: profile.device_budget,
            max_source_chunk_bytes: profile.max_source_chunk_bytes,
            max_row_groups_per_chunk: profile.max_row_groups_per_chunk,
            max_retry_attempts: profile.max_retry_attempts,
            min_retry_chunk_row_groups: profile.min_retry_chunk_row_groups,
            source_chunk_planning_mode: profile.source_chunk_planning_mode,
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        "bounded"
    }

    fn validate(self) -> DataFusionResult<()> {
        validate_device_budget(self.device_budget)?;
        validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, self.max_source_chunk_bytes)?;
        validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, self.max_row_groups_per_chunk)?;
        validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, self.max_retry_attempts)?;
        validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, self.min_retry_chunk_row_groups)?;
        Ok(())
    }
}

impl Default for NativeTpchMemoryPolicy {
    fn default() -> Self {
        Self {
            profile_name: TPCH_NATIVE_PROFILE_CONSERVATIVE,
            profile_source: "default",
            device_budget: NativeTpchDeviceBudget::Automatic,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeTpchProfileSettings {
    pub(super) profile_name: &'static str,
    pub(super) profile_source: &'static str,
    pub(super) device_budget: NativeTpchDeviceBudget,
    pub(super) max_source_chunk_bytes: u64,
    pub(super) max_row_groups_per_chunk: usize,
    pub(super) max_retry_attempts: usize,
    pub(super) min_retry_chunk_row_groups: usize,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
}

impl NativeTpchProfileSettings {
    fn from_base(base: NativeTpchProfileBase) -> Self {
        Self {
            profile_name: base.name,
            profile_source: base.source,
            device_budget: base.device_budget,
            max_source_chunk_bytes: base.max_source_chunk_bytes,
            max_row_groups_per_chunk: base.max_row_groups_per_chunk,
            max_retry_attempts: base.max_retry_attempts,
            min_retry_chunk_row_groups: base.min_retry_chunk_row_groups,
            source_chunk_planning_mode: base.source_chunk_planning_mode,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeTpchProfileBase {
    name: &'static str,
    source: &'static str,
    device_budget: NativeTpchDeviceBudget,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_chunk_planning_mode: SourceChunkPlanningMode,
}

fn native_tpch_profile_settings_from_env() -> DataFusionResult<NativeTpchProfileSettings> {
    native_tpch_profile_settings_with_env(|name| {
        std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
    })
}

pub(super) fn native_tpch_profile_settings_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> DataFusionResult<NativeTpchProfileSettings> {
    let requested_profile = get(ALGEON_TPCH_BENCH_NATIVE_PROFILE_ENV).map(|value| value.trim().to_ascii_lowercase());
    let base = native_tpch_profile_base(requested_profile.as_deref())?;
    let mut resolved = NativeTpchProfileSettings::from_base(base);
    let device_budget_bytes = get(ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV);
    let max_source_chunk_bytes = get(ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV);
    let max_row_groups_per_chunk = get(ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV);
    let max_retry_attempts = get(ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV);
    let min_retry_chunk_row_groups = get(ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV);
    let source_chunk_planning_mode = get(ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV);
    let had_override = device_budget_bytes.is_some()
        || max_source_chunk_bytes.is_some()
        || max_row_groups_per_chunk.is_some()
        || max_retry_attempts.is_some()
        || min_retry_chunk_row_groups.is_some()
        || source_chunk_planning_mode.is_some();

    if let Some(value) = device_budget_bytes {
        resolved.device_budget = NativeTpchDeviceBudget::Explicit {
            bytes: parse_env_u64_value(ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, value)?,
        };
    }
    if let Some(value) = max_source_chunk_bytes {
        resolved.max_source_chunk_bytes =
            parse_env_u64_value(ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, value)?;
    }
    if let Some(value) = max_row_groups_per_chunk {
        resolved.max_row_groups_per_chunk =
            parse_env_usize_value(ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, value)?;
    }
    if let Some(value) = max_retry_attempts {
        resolved.max_retry_attempts = parse_env_usize_value(ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, value)?;
    }
    if let Some(value) = min_retry_chunk_row_groups {
        resolved.min_retry_chunk_row_groups =
            parse_env_usize_value(ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, value)?;
    }
    if let Some(value) = source_chunk_planning_mode {
        resolved.source_chunk_planning_mode = parse_env_source_chunk_planning_mode(value)?;
    }
    validate_tpch_profile_settings(&resolved)?;

    let conservative = native_tpch_profile_base(Some(TPCH_NATIVE_PROFILE_CONSERVATIVE))?;
    let sf100 = native_tpch_profile_base(Some(TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT))?;
    let requested_without_override = requested_profile.is_some() && !had_override;
    if same_tpch_profile_values(&resolved, conservative) {
        resolved.profile_name = TPCH_NATIVE_PROFILE_CONSERVATIVE;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if same_tpch_profile_values(&resolved, sf100) {
        resolved.profile_name = TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if had_override {
        resolved.profile_name = TPCH_NATIVE_PROFILE_CUSTOM;
        resolved.profile_source = "custom_overrides";
    }

    Ok(resolved)
}

fn validate_tpch_profile_settings(profile: &NativeTpchProfileSettings) -> DataFusionResult<()> {
    validate_device_budget(profile.device_budget)?;
    validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, profile.max_source_chunk_bytes)?;
    validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, profile.max_row_groups_per_chunk)?;
    validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, profile.max_retry_attempts)?;
    validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, profile.min_retry_chunk_row_groups)?;
    Ok(())
}

fn native_tpch_profile_base(profile: Option<&str>) -> DataFusionResult<NativeTpchProfileBase> {
    let profile = profile.unwrap_or(TPCH_NATIVE_PROFILE_CONSERVATIVE);
    match profile {
        TPCH_NATIVE_PROFILE_CONSERVATIVE => Ok(NativeTpchProfileBase {
            name: TPCH_NATIVE_PROFILE_CONSERVATIVE,
            source: "default",
            device_budget: NativeTpchDeviceBudget::Automatic,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        }),
        TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT => Ok(NativeTpchProfileBase {
            name: TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
            source: "env_profile",
            device_budget: NativeTpchDeviceBudget::Automatic,
            max_source_chunk_bytes: TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: SourceChunkPlanningMode::ParquetThroughput,
        }),
        other => Err(plan_error(format!(
            "invalid {ALGEON_TPCH_BENCH_NATIVE_PROFILE_ENV}: {other}; expected {TPCH_NATIVE_PROFILE_CONSERVATIVE} or {TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT}"
        ))),
    }
}

fn same_tpch_profile_values(profile: &NativeTpchProfileSettings, base: NativeTpchProfileBase) -> bool {
    profile.device_budget == base.device_budget
        && profile.max_source_chunk_bytes == base.max_source_chunk_bytes
        && profile.max_row_groups_per_chunk == base.max_row_groups_per_chunk
        && profile.max_retry_attempts == base.max_retry_attempts
        && profile.min_retry_chunk_row_groups == base.min_retry_chunk_row_groups
        && profile.source_chunk_planning_mode == base.source_chunk_planning_mode
}

fn reject_retired_source_cache_env() -> DataFusionResult<()> {
    for name in [REMOVED_NATIVE_SOURCE_CACHE_POLICY_ENV, REMOVED_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV] {
        if std::env::var_os(name).is_some() {
            return Err(plan_error(format!("{name} is no longer supported; remove this setting")));
        }
    }
    Ok(())
}

fn reject_retired_memory_policy_env() -> DataFusionResult<()> {
    if std::env::var_os(REMOVED_NATIVE_MEMORY_POLICY_ENV).is_some() {
        return Err(plan_error(format!(
            "{REMOVED_NATIVE_MEMORY_POLICY_ENV} is no longer supported; the TPC-H benchmark always uses bounded backend admission"
        )));
    }
    Ok(())
}

fn query_ids_from_env(name: &'static str) -> DataFusionResult<Vec<String>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect());
    };
    let query_ids = value
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|query_id| !query_id.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if query_ids.is_empty() {
        return Err(plan_error(format!("{name} must name at least one query id")));
    }
    Ok(query_ids)
}

fn validate_query_ids(query_ids: &[String]) -> DataFusionResult<()> {
    if query_ids.is_empty() {
        return Err(plan_error("ALGEON_TPCH_BENCH_QUERY_IDS must name at least one query id"));
    }

    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for query_id in query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(plan_error(format!("unknown TPC-H query id {query_id}")));
        }
        if !seen.insert(query_id.as_str()) {
            return Err(plan_error(format!("duplicate TPC-H query id {query_id}")));
        }
    }
    Ok(())
}

fn env_usize(name: &'static str, default: usize) -> DataFusionResult<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<usize>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_usize_value(name: &'static str, value: String) -> DataFusionResult<usize> {
    let value = value.trim();
    value.parse::<usize>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_u64_value(name: &'static str, value: String) -> DataFusionResult<u64> {
    let value = value.trim();
    value.parse::<u64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_source_chunk_planning_mode(value: String) -> DataFusionResult<SourceChunkPlanningMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(SourceChunkPlanningMode::Conservative),
        "parquet_throughput" => Ok(SourceChunkPlanningMode::ParquetThroughput),
        other => Err(plan_error(format!(
            "invalid {ALGEON_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV}: {other}; expected conservative or parquet_throughput"
        ))),
    }
}

fn optional_env_bool(name: &'static str) -> DataFusionResult<Option<bool>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    match value.to_string_lossy().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(Some(true)),
        "false" | "0" | "no" | "off" => Ok(Some(false)),
        other => Err(plan_error(format!("invalid {name}: {other}; expected true or false"))),
    }
}

fn validate_device_budget(device_budget: NativeTpchDeviceBudget) -> DataFusionResult<()> {
    match device_budget {
        NativeTpchDeviceBudget::Automatic => Ok(()),
        NativeTpchDeviceBudget::Explicit { bytes } => {
            validate_nonzero(ALGEON_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, bytes)
        },
    }
}

fn validate_nonzero<T>(name: &'static str, value: T) -> DataFusionResult<()>
where T: PartialEq + From<u8> + std::fmt::Display {
    if value == T::from(0) {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}
