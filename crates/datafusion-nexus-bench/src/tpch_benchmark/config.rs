use super::*;

pub(super) const CUDF_TPCH_BENCH_NATIVE_PROFILE_ENV: &str = "CUDF_TPCH_BENCH_NATIVE_PROFILE";
pub(super) const CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES";
pub(super) const CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str =
    "CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES";
pub(super) const CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str =
    "CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub(super) const CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS";
pub(super) const CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str =
    "CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
pub(super) const CUDF_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str =
    "CUDF_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE";

pub(super) const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 14 * 1024 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
pub(super) const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT: usize = 2;
pub(super) const DEFAULT_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub(super) const TPCH_NATIVE_PROFILE_CONSERVATIVE: &str = "conservative";
pub(super) const TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT: &str = "sf100_gpu_throughput";
pub(super) const TPCH_NATIVE_PROFILE_CUSTOM: &str = "custom";
pub(super) const TPCH_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES: u64 = 102_005_473_280;
pub(super) const TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES: u64 = 34_359_738_368;
pub(super) const TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeTpchBenchmarkConfig {
    pub(super) query_ids: Vec<String>,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
    pub(super) dataset_mode: NativeTpchDatasetMode,
    pub(super) parquet_pushdown_filters: Option<bool>,
    pub(super) native_memory_policy: NativeTpchMemoryPolicy,
    pub(super) native_pipeline_execution_mode: NativePipelineExecutionMode,
    pub(super) max_source_reads_in_flight: usize,
    pub(super) native_source_cache_policy: NativeTpchSourceCachePolicy,
}

#[derive(Debug, Clone)]
pub(super) struct NativeTpchRuntimeResources {
    pub(super) runtime: Option<Arc<cudf::Runtime>>,
    pub(super) source_cache: Option<Arc<nexus_query_engine::runtime::cache::local_parquet::GpuLocalParquetSourceCache>>,
}

impl NativeTpchBenchmarkConfig {
    pub(super) fn from_env() -> DataFusionResult<Self> {
        let config = Self {
            query_ids: query_ids_from_env("CUDF_TPCH_BENCH_QUERY_IDS")?,
            warmup_iterations: env_usize("CUDF_TPCH_BENCH_WARMUPS", 0)?,
            iterations: env_usize("CUDF_TPCH_BENCH_ITERATIONS", 1)?,
            dataset_mode: NativeTpchDatasetMode::from_env("CUDF_TPCH_BENCH_DATASET_MODE")?,
            parquet_pushdown_filters: optional_env_bool("CUDF_TPCH_BENCH_PARQUET_PUSHDOWN_FILTERS")?,
            native_memory_policy: NativeTpchMemoryPolicy::from_env("CUDF_TPCH_BENCH_NATIVE_MEMORY_POLICY")?,
            native_pipeline_execution_mode: native_pipeline_execution_mode_from_env(
                "CUDF_TPCH_BENCH_NATIVE_PIPELINE_EXECUTION_MODE",
            )?,
            max_source_reads_in_flight: env_usize(
                "CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT",
                DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT,
            )?,
            native_source_cache_policy: NativeTpchSourceCachePolicy::from_env(
                "CUDF_TPCH_BENCH_NATIVE_SOURCE_CACHE_POLICY",
            )?,
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("CUDF_TPCH_BENCH_ITERATIONS", self.iterations)?;
        validate_query_ids(&self.query_ids)?;
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT", self.max_source_reads_in_flight)?;
        if let Some(max_retained_bytes) = self.native_source_cache_policy.max_retained_bytes() {
            validate_nonzero("CUDF_TPCH_BENCH_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", max_retained_bytes)?;
        }
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
        resources: &NativeTpchRuntimeResources,
    ) -> NativeCudfExecutionConfig {
        let config = match self.native_memory_policy {
            // Explicit benchmark override used to compare legacy whole-plan
            // execution against the bounded production profile.
            NativeTpchMemoryPolicy::Unbounded => NativeCudfExecutionConfig::unbounded(),
            NativeTpchMemoryPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
                source_chunk_planning_mode,
                ..
            } => NativeCudfExecutionConfig::bounded(
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            )
            .with_source_scheduler_policy(nexus_query_engine::runtime::SourceSchedulerPolicy::new(
                self.max_source_reads_in_flight,
            ))
            .with_source_chunk_planning_mode(source_chunk_planning_mode),
        };
        config
            .with_optional_shared_runtime(resources.runtime.clone())
            .with_optional_local_parquet_source_cache(resources.source_cache.clone())
            .with_native_pipeline_execution_mode(self.native_pipeline_execution_mode)
    }

    pub(super) fn new_runtime_resources(&self) -> DataFusionResult<NativeTpchRuntimeResources> {
        let runtime = Arc::new(
            cudf::Runtime::new()
                .map_err(|error| plan_error(format!("failed to initialize native benchmark runtime: {error}")))?,
        );
        let source_cache = match self.native_source_cache_policy {
            NativeTpchSourceCachePolicy::Disabled => None,
            NativeTpchSourceCachePolicy::GpuResident { max_retained_bytes } => Some(Arc::new(
                nexus_query_engine::runtime::cache::local_parquet::GpuLocalParquetSourceCache::new(max_retained_bytes),
            )),
        };
        Ok(NativeTpchRuntimeResources { runtime: Some(runtime), source_cache })
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
            native_memory_policy: NativeTpchMemoryPolicy::default(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            max_source_reads_in_flight: DEFAULT_NATIVE_MAX_SOURCE_READS_IN_FLIGHT,
            native_source_cache_policy: NativeTpchSourceCachePolicy::Disabled,
        }
    }
}

trait NativeExecutionConfigSourceCacheExt {
    fn with_optional_shared_runtime(self, runtime: Option<Arc<cudf::Runtime>>) -> Self;

    fn with_optional_local_parquet_source_cache(
        self,
        cache: Option<Arc<nexus_query_engine::runtime::cache::local_parquet::GpuLocalParquetSourceCache>>,
    ) -> Self;
}

impl NativeExecutionConfigSourceCacheExt for NativeCudfExecutionConfig {
    fn with_optional_shared_runtime(self, runtime: Option<Arc<cudf::Runtime>>) -> Self {
        match runtime {
            Some(runtime) => self.with_shared_runtime(runtime),
            None => self,
        }
    }

    fn with_optional_local_parquet_source_cache(
        self,
        cache: Option<Arc<nexus_query_engine::runtime::cache::local_parquet::GpuLocalParquetSourceCache>>,
    ) -> Self {
        match cache {
            Some(cache) => self.with_local_parquet_source_cache(cache),
            None => self,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchMemoryPolicy {
    Unbounded,
    Bounded {
        profile_name: &'static str,
        profile_source: &'static str,
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
    },
}

impl NativeTpchMemoryPolicy {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let policy = match std::env::var_os(name) {
            None => "bounded".to_string(),
            Some(value) => value.to_string_lossy().to_ascii_lowercase(),
        };
        match policy.as_str() {
            "unbounded" => Ok(Self::Unbounded),
            "bounded" => Ok(Self::bounded_from_profile_settings(native_tpch_profile_settings_from_env()?)),
            other => Err(plan_error(format!("invalid {name}: {other}; expected bounded or unbounded"))),
        }
    }

    #[cfg(test)]
    pub(super) const fn bounded_custom(
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
    ) -> Self {
        Self::Bounded {
            profile_name: TPCH_NATIVE_PROFILE_CUSTOM,
            profile_source: "direct_config",
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            source_chunk_planning_mode,
        }
    }

    pub(super) fn bounded_from_profile_settings(profile: NativeTpchProfileSettings) -> Self {
        Self::Bounded {
            profile_name: profile.profile_name,
            profile_source: profile.profile_source,
            device_budget_bytes: profile.device_budget_bytes,
            max_source_chunk_bytes: profile.max_source_chunk_bytes,
            max_row_groups_per_chunk: profile.max_row_groups_per_chunk,
            max_retry_attempts: profile.max_retry_attempts,
            min_retry_chunk_row_groups: profile.min_retry_chunk_row_groups,
            source_chunk_planning_mode: profile.source_chunk_planning_mode,
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded { .. } => "bounded",
        }
    }

    fn validate(self) -> DataFusionResult<()> {
        let Self::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            ..
        } = self
        else {
            return Ok(());
        };
        validate_nonzero(CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, device_budget_bytes)?;
        validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, max_source_chunk_bytes)?;
        validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, max_row_groups_per_chunk)?;
        validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, max_retry_attempts)?;
        validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, min_retry_chunk_row_groups)?;
        Ok(())
    }
}

impl Default for NativeTpchMemoryPolicy {
    fn default() -> Self {
        Self::Bounded {
            profile_name: TPCH_NATIVE_PROFILE_CONSERVATIVE,
            profile_source: "default",
            device_budget_bytes: DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeTpchProfileSettings {
    pub(super) profile_name: &'static str,
    pub(super) profile_source: &'static str,
    pub(super) device_budget_bytes: u64,
    pub(super) max_source_chunk_bytes: u64,
    pub(super) max_row_groups_per_chunk: usize,
    pub(super) max_retry_attempts: usize,
    pub(super) min_retry_chunk_row_groups: usize,
    pub(super) source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
}

impl NativeTpchProfileSettings {
    fn from_base(base: NativeTpchProfileBase) -> Self {
        Self {
            profile_name: base.name,
            profile_source: base.source,
            device_budget_bytes: base.device_budget_bytes,
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
    device_budget_bytes: u64,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
}

fn native_tpch_profile_settings_from_env() -> DataFusionResult<NativeTpchProfileSettings> {
    native_tpch_profile_settings_with_env(|name| {
        std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
    })
}

pub(super) fn native_tpch_profile_settings_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> DataFusionResult<NativeTpchProfileSettings> {
    let requested_profile = get(CUDF_TPCH_BENCH_NATIVE_PROFILE_ENV).map(|value| value.trim().to_ascii_lowercase());
    let base = native_tpch_profile_base(requested_profile.as_deref())?;
    let mut resolved = NativeTpchProfileSettings::from_base(base);
    let device_budget_bytes = get(CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV);
    let max_source_chunk_bytes = get(CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV);
    let max_row_groups_per_chunk = get(CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV);
    let max_retry_attempts = get(CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV);
    let min_retry_chunk_row_groups = get(CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV);
    let source_chunk_planning_mode = get(CUDF_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV);
    let had_override = device_budget_bytes.is_some()
        || max_source_chunk_bytes.is_some()
        || max_row_groups_per_chunk.is_some()
        || max_retry_attempts.is_some()
        || min_retry_chunk_row_groups.is_some()
        || source_chunk_planning_mode.is_some();

    if let Some(value) = device_budget_bytes {
        resolved.device_budget_bytes = parse_env_u64_value(CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, value)?;
    }
    if let Some(value) = max_source_chunk_bytes {
        resolved.max_source_chunk_bytes =
            parse_env_u64_value(CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, value)?;
    }
    if let Some(value) = max_row_groups_per_chunk {
        resolved.max_row_groups_per_chunk =
            parse_env_usize_value(CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, value)?;
    }
    if let Some(value) = max_retry_attempts {
        resolved.max_retry_attempts = parse_env_usize_value(CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, value)?;
    }
    if let Some(value) = min_retry_chunk_row_groups {
        resolved.min_retry_chunk_row_groups =
            parse_env_usize_value(CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, value)?;
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
    validate_nonzero(CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, profile.device_budget_bytes)?;
    validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, profile.max_source_chunk_bytes)?;
    validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, profile.max_row_groups_per_chunk)?;
    validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, profile.max_retry_attempts)?;
    validate_nonzero(CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, profile.min_retry_chunk_row_groups)?;
    Ok(())
}

fn native_tpch_profile_base(profile: Option<&str>) -> DataFusionResult<NativeTpchProfileBase> {
    let profile = profile.unwrap_or(TPCH_NATIVE_PROFILE_CONSERVATIVE);
    match profile {
        TPCH_NATIVE_PROFILE_CONSERVATIVE => Ok(NativeTpchProfileBase {
            name: TPCH_NATIVE_PROFILE_CONSERVATIVE,
            source: "default",
            device_budget_bytes: DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative,
        }),
        TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT => Ok(NativeTpchProfileBase {
            name: TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
            source: "env_profile",
            device_budget_bytes: TPCH_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput,
        }),
        other => Err(plan_error(format!(
            "invalid {CUDF_TPCH_BENCH_NATIVE_PROFILE_ENV}: {other}; expected {TPCH_NATIVE_PROFILE_CONSERVATIVE} or {TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT}"
        ))),
    }
}

fn same_tpch_profile_values(profile: &NativeTpchProfileSettings, base: NativeTpchProfileBase) -> bool {
    profile.device_budget_bytes == base.device_budget_bytes
        && profile.max_source_chunk_bytes == base.max_source_chunk_bytes
        && profile.max_row_groups_per_chunk == base.max_row_groups_per_chunk
        && profile.max_retry_attempts == base.max_retry_attempts
        && profile.min_retry_chunk_row_groups == base.min_retry_chunk_row_groups
        && profile.source_chunk_planning_mode == base.source_chunk_planning_mode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchSourceCachePolicy {
    Disabled,
    GpuResident { max_retained_bytes: u64 },
}

impl NativeTpchSourceCachePolicy {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let policy = match std::env::var_os(name) {
            None => "disabled".to_string(),
            Some(value) => value.to_string_lossy().to_ascii_lowercase(),
        };
        match policy.as_str() {
            "disabled" | "off" | "none" => Ok(Self::Disabled),
            "gpu_resident" | "gpu-resident" | "enabled" => Ok(Self::GpuResident {
                max_retained_bytes: env_u64(
                    "CUDF_TPCH_BENCH_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES",
                    DEFAULT_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES,
                )?,
            }),
            other => Err(plan_error(format!("invalid {name}: {other}; expected disabled or gpu_resident"))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::GpuResident { .. } => "gpu_resident",
        }
    }

    pub(super) const fn max_retained_bytes(self) -> Option<u64> {
        match self {
            Self::Disabled => None,
            Self::GpuResident { max_retained_bytes } => Some(max_retained_bytes),
        }
    }
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
        return Err(plan_error("CUDF_TPCH_BENCH_QUERY_IDS must name at least one query id"));
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

fn env_u64(name: &'static str, default: u64) -> DataFusionResult<u64> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<u64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_usize_value(name: &'static str, value: String) -> DataFusionResult<usize> {
    let value = value.trim();
    value.parse::<usize>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_u64_value(name: &'static str, value: String) -> DataFusionResult<u64> {
    let value = value.trim();
    value.parse::<u64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn parse_env_source_chunk_planning_mode(
    value: String,
) -> DataFusionResult<nexus_query_engine::runtime::SourceChunkPlanningMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative),
        "local_parquet_throughput" | "local_throughput" => {
            Ok(nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput)
        },
        other => Err(plan_error(format!(
            "invalid {CUDF_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV}: {other}; expected conservative or local_parquet_throughput"
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

fn native_pipeline_execution_mode_from_env(name: &'static str) -> DataFusionResult<NativePipelineExecutionMode> {
    let mode = match std::env::var_os(name) {
        None => "strict_bounded".to_string(),
        Some(value) => value.to_string_lossy().to_ascii_lowercase(),
    };
    mode.parse::<NativePipelineExecutionMode>()
        .map_err(|error| plan_error(format!("invalid {name}: {}; expected strict_bounded or adaptive", error.value())))
}

fn validate_nonzero<T>(name: &'static str, value: T) -> DataFusionResult<()>
where T: PartialEq + From<u8> + std::fmt::Display {
    if value == T::from(0) {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}
