use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{datatypes::Schema, error::ArrowError, record_batch::RecordBatch};
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    error::DataFusionError,
    execution::{SessionStateBuilder, context::TaskContext},
    parquet::errors::ParquetError,
    physical_plan::{ExecutionPlan, collect},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy,
    NativeCudfExecutionConfig, NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
    NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
    register_source_unique_keys_for_path,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

mod dataset;
mod parity;
mod tiny;

pub use dataset::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_PARQUET_DIR_ENV,
    CUDF_TPCDS_SCALE_FACTOR_ENV, DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR, DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR, TPCDS_TABLES,
    TpcdsDatasetMetadata, TpcdsTableFileMetadata, TpcdsTableLayout, TpcdsTableMetadata, default_tpcds_sf1_parquet_root,
    describe_tpcds_dataset, describe_tpcds_dataset_with_content_fingerprint, discover_tpcds_table_layout,
    external_tpcds_dataset_metadata, prepare_tpcds_report_dataset, tiny_tpcds_dataset_metadata,
};
pub use parity::{
    TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, TpcdsParityProgressEvent,
    tpcds_parity_report_rows_with_external_oracles, tpcds_parity_workload_items,
    tpcds_parity_workload_items_with_external_oracles, tpcds_parity_workload_items_with_session_configs,
};
pub use tiny::write_tiny_tpcds_parquet_dataset;

pub use super::tpcds_queries::{TPCDS_QUERIES, TPCDS_QUERY_COUNT, TpcdsQuery, tpcds_query_ids};

/// Device-byte budget for bounded native TPC-DS execution.
///
/// When unset, the workload uses `nvidia-smi` memory.total minus 500 MiB, with
/// a 256 MiB fallback if GPU memory cannot be detected. Invalid values panic
/// during config construction.
pub const CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES";
/// Named native memory/source profile used by TPC-DS benchmark/report
/// surfaces. Accepts `conservative` (default) or `sf100_gpu_throughput`;
/// individual `CUDF_TPCDS_NATIVE_*` knobs may still override the profile.
pub const CUDF_TPCDS_NATIVE_PROFILE_ENV: &str = "CUDF_TPCDS_NATIVE_PROFILE";
/// Planning target, in bytes, for native Parquet source chunks.
pub const CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Planning target for row groups per native Parquet source chunk.
pub const CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum bounded retry attempts after resource or reservation pressure.
pub const CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS";
/// Minimum row-group granularity used when retrying with smaller chunks.
pub const CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Source chunk planning mode used by bounded native TPC-DS execution. Accepts
/// `conservative` or `local_parquet_throughput`; defaults to the selected
/// profile.
pub const CUDF_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str = "CUDF_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE";
/// Execution contract for native floating statistical aggregates. Accepts
/// `datafusion_exact_host_welford` or `gpu_native_tolerant`; defaults depend on
/// the caller's workload surface.
pub const CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Migration alias for
/// [`CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV`], with the
/// same accepted values.
pub const CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV: &str = "CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE";
/// Enables or disables CPU preference for TPC-DS hybrid analytics planning.
///
/// Native all-GPU report/parity surfaces intentionally keep CPU preference
/// disabled while SQL may include GPU-only `cugraph_*` table functions.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_ENV: &str = "CUDF_TPCDS_NATIVE_CPU_PREFERENCE";
/// Row-group threshold for small/selective CPU preference on the TPC-DS hybrid
/// analytics surface.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS";
/// Source-read byte threshold for small/selective CPU preference on the TPC-DS
/// hybrid analytics surface.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES";
/// Row-group threshold for residual-join CPU preference on the TPC-DS hybrid
/// analytics surface.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS";
/// Source-read byte threshold for residual-join CPU preference on the TPC-DS
/// hybrid analytics surface.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES";
/// Top-k fetch threshold for small/selective CPU preference on the TPC-DS
/// hybrid analytics surface.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH_ENV: &str = "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH";
/// Hybrid analytics residual-join row-group threshold.
///
/// This stays intentionally close to the engine default while admitting
/// low-byte residual joins whose pruning evidence is spread across a modest
/// number of Parquet row groups.
pub const TPCDS_HYBRID_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS: usize = 32;

/// Conservative bounded profile: current production-safe defaults.
pub const TPCDS_NATIVE_PROFILE_CONSERVATIVE: &str = "conservative";
/// SF100 throughput profile: TPC-H-like large local Parquet batches for GPU benchmarking.
pub const TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT: &str = "sf100_gpu_throughput";
/// Profile name used when explicit knobs do not match a named profile.
pub const TPCDS_NATIVE_PROFILE_CUSTOM: &str = "custom";
/// SF100 throughput profile device budget, matching the 95 GiB probe.
pub const TPCDS_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES: u64 = 102_005_473_280;
/// SF100 throughput profile source byte target: 32 GiB.
pub const TPCDS_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES: u64 = 34_359_738_368;
/// SF100 throughput profile row-group batch target.
pub const TPCDS_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;

/// Resolved bounded native TPC-DS memory/source profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsNativeProfileSettings {
    pub profile_name: &'static str,
    pub profile_source: &'static str,
    pub device_budget_bytes: u64,
    pub max_source_chunk_bytes: u64,
    pub max_row_groups_per_chunk: usize,
    pub max_retry_attempts: usize,
    pub min_retry_chunk_row_groups: usize,
    pub source_chunk_planning_mode: nexus_query_engine::SourceChunkPlanningMode,
}

impl TpcdsNativeProfileSettings {
    fn from_base(base: NativeProfileBase) -> Self {
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
struct NativeProfileBase {
    name: &'static str,
    source: &'static str,
    device_budget_bytes: u64,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_chunk_planning_mode: nexus_query_engine::SourceChunkPlanningMode,
}

/// Result type used by TPC-DS fixture and workload helpers.
pub type TpcdsWorkloadResult<T> = std::result::Result<T, TpcdsWorkloadError>;

/// Error categories surfaced by TPC-DS fixture discovery and workload helpers.
#[derive(Debug)]
pub enum TpcdsWorkloadError {
    /// Filesystem or process I/O failed.
    Io(std::io::Error),
    /// Arrow schema or record-batch handling failed.
    Arrow(ArrowError),
    /// Parquet reading or writing failed.
    Parquet(ParquetError),
    /// DataFusion planning or execution failed.
    DataFusion(DataFusionError),
    /// A fixture path could not be represented as UTF-8.
    NonUtf8Path(PathBuf),
    /// Fixture metadata was incomplete or inconsistent.
    InvalidDataset(String),
}

impl Display for TpcdsWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid TPC-DS dataset: {message}"),
        }
    }
}

impl StdError for TpcdsWorkloadError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::DataFusion(error) => Some(error),
            Self::NonUtf8Path(_) => None,
            Self::InvalidDataset(_) => None,
        }
    }
}

impl From<std::io::Error> for TpcdsWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for TpcdsWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for TpcdsWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for TpcdsWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

/// Session config used by TPC-DS native report surfaces.
///
/// Uses one target partition, disables DataFusion repartitioning, dynamic
/// filter pushdown, and leaves Parquet pushdown filters enabled for scan
/// pruning parity with report binaries. This base config remains compatible
/// with the CPU DataFusion oracle; native-only planner guards are applied by
/// [`tpcds_native_report_session_config`].
pub fn tpcds_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config.options_mut().execution.parquet.pushdown_filters = true;
    config
}

/// Session config used for native TPC-DS planning and execution contexts.
///
/// This starts from [`tpcds_report_session_config`] and disables DataFusion
/// table-level statistics collection. DataFusion 54.0.0's crates.io physical
/// join statistics can overflow on large multi-join TPC-DS plans; native source
/// planning reads row-group metadata directly from Parquet footers instead of
/// relying on those table estimates.
/// DataFusion 54 defaults uncorrelated scalar subqueries to `ScalarSubqueryExec`;
/// native TPC-DS report/admission contracts require the relationalized join form
/// the native IR can lower.
pub fn tpcds_native_report_session_config() -> SessionConfig {
    native_tpcds_session_config(tpcds_report_session_config())
}

fn native_tpcds_session_config(mut config: SessionConfig) -> SessionConfig {
    config.options_mut().execution.collect_statistics = false;
    config.options_mut().optimizer.enable_physical_uncorrelated_scalar_subquery = false;
    config
}

/// Native optimizer config used by TPC-DS report surfaces.
///
/// Numeric env vars parse unsigned integers. Source chunk bytes, row-group
/// limits are planning targets; positive-value validation happens when the
/// native execution context is built. CPU-preference knobs are intentionally
/// dormant until the native SQL surface can distinguish ordinary TPC-DS shapes
/// from GPU-only `cugraph_*` calls.
pub fn tpcds_native_optimizer_config() -> CudfOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default(
        nexus_query_engine::StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    )
}

/// Native optimizer config using an already-resolved TPC-DS native profile.
pub fn tpcds_native_optimizer_config_with_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> CudfOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        nexus_query_engine::StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
        profile,
    )
}

/// Native optimizer config for TPC-DS analytics/performance report surfaces.
///
/// The analytics default uses GPU-native tolerant floating statistical
/// aggregate kernels. Set
/// [`CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV`] or the
/// migration [`CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV`] to
/// `datafusion_exact_host_welford` when a run intentionally needs bit-exact
/// DataFusion compatibility.
pub fn tpcds_native_analytics_optimizer_config() -> CudfOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default(
        nexus_query_engine::StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
}

/// Analytics/performance optimizer config using an already-resolved TPC-DS native profile.
pub fn tpcds_native_analytics_optimizer_config_with_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> CudfOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        nexus_query_engine::StatisticalAggregateExecutionMode::GpuNativeTolerant,
        profile,
    )
}

/// Analytics/performance optimizer config for the `hybrid_cpu_preferred`
/// benchmark surface. Ordinary native TPC-DS report/parity surfaces keep CPU
/// preference disabled so all-native validation remains exact; the hybrid
/// benchmark uses this config to quantify the conservative CPU-preference model.
pub fn tpcds_native_analytics_optimizer_config_with_cpu_preference_and_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> CudfOptimizerConfig {
    tpcds_native_analytics_optimizer_config_with_profile_settings(profile)
        .with_native_plan_preference_options(tpcds_native_cpu_preference_options_from_env())
}

/// Resolves TPC-DS hybrid analytics CPU-preference options from env.
pub fn tpcds_native_cpu_preference_options_from_env() -> nexus_query_engine::NativePlanPreferenceOptions {
    tpcds_native_cpu_preference_options_with_env(|name| std::env::var(name).ok())
}

/// Resolves TPC-DS hybrid analytics CPU-preference options using an injected
/// env reader for tests and benchmark harnesses.
pub fn tpcds_native_cpu_preference_options_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> nexus_query_engine::NativePlanPreferenceOptions {
    let mut options = nexus_query_engine::NativePlanPreferenceOptions {
        max_small_selective_residual_join_row_groups: TPCDS_HYBRID_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS,
        ..nexus_query_engine::NativePlanPreferenceOptions::default()
    };

    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_ENV) {
        options.enabled = parse_env_bool(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_ENV, value);
    }
    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV) {
        options.max_small_selective_row_groups =
            parse_env_usize(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV) {
        options.max_small_selective_source_bytes =
            parse_env_u64(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV, value);
    }
    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV) {
        options.max_small_selective_residual_join_row_groups =
            parse_env_usize(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV) {
        options.max_small_selective_residual_join_source_bytes =
            parse_env_u64(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV, value);
    }
    if let Some(value) = get(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH_ENV) {
        options.max_topk_fetch = parse_env_usize(CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH_ENV, value);
    }

    options
}

fn tpcds_native_optimizer_config_with_statistical_aggregate_default(
    statistical_aggregate_execution_mode_default: nexus_query_engine::StatisticalAggregateExecutionMode,
) -> CudfOptimizerConfig {
    let profile = tpcds_native_profile_settings_from_env();
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        statistical_aggregate_execution_mode_default,
        &profile,
    )
}

fn tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
    statistical_aggregate_execution_mode_default: nexus_query_engine::StatisticalAggregateExecutionMode,
    profile: &TpcdsNativeProfileSettings,
) -> CudfOptimizerConfig {
    let statistical_aggregate_execution_mode =
        optional_env_statistical_aggregate_execution_mode().unwrap_or(statistical_aggregate_execution_mode_default);
    // Keep CPU preference disabled for now: users can submit SQL containing
    // `cugraph_*` table functions, and those plans must stay on the GPU rather
    // than being routed through DataFusion CPU fallback as a small/selective
    // preference. Leave the env constants above in place so the old benchmark
    // gate can be restored once graph SQL has an explicit GPU-only boundary.
    let preference_options = nexus_query_engine::NativePlanPreferenceOptions::disabled();
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(
            NativeCudfExecutionConfig::bounded(
                profile.device_budget_bytes,
                profile.max_source_chunk_bytes,
                profile.max_row_groups_per_chunk,
                profile.max_retry_attempts,
                profile.min_retry_chunk_row_groups,
            )
            .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode)
            .with_source_chunk_planning_mode(profile.source_chunk_planning_mode),
        )
        .with_native_plan_preference_options(preference_options)
}

/// Resolves the TPC-DS native bounded memory/source profile from env.
///
/// `CUDF_TPCDS_NATIVE_PROFILE` chooses a named base profile:
/// `conservative` keeps the existing bounded defaults, while
/// `sf100_gpu_throughput` applies the TPC-H-like SF100 local Parquet profile.
/// The individual `CUDF_TPCDS_NATIVE_*` knobs override either base. If
/// those final values exactly match a named profile, the resolved
/// `profile_name` is that profile; otherwise it is `custom`.
pub fn tpcds_native_profile_settings_from_env() -> TpcdsNativeProfileSettings {
    tpcds_native_profile_settings_with_env(|name| std::env::var(name).ok())
}

pub fn tpcds_native_profile_settings_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> TpcdsNativeProfileSettings {
    let requested_profile = get(CUDF_TPCDS_NATIVE_PROFILE_ENV).map(|value| value.trim().to_ascii_lowercase());
    let base = native_profile_base(requested_profile.as_deref());
    let mut resolved = TpcdsNativeProfileSettings::from_base(base);
    let device_budget_bytes = get(CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV);
    let max_source_chunk_bytes = get(CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV);
    let max_row_groups_per_chunk = get(CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV);
    let max_retry_attempts = get(CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV);
    let min_retry_chunk_row_groups = get(CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV);
    let source_chunk_planning_mode = get(CUDF_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV);
    let had_override = device_budget_bytes.is_some()
        || max_source_chunk_bytes.is_some()
        || max_row_groups_per_chunk.is_some()
        || max_retry_attempts.is_some()
        || min_retry_chunk_row_groups.is_some()
        || source_chunk_planning_mode.is_some();

    if let Some(value) = device_budget_bytes {
        resolved.device_budget_bytes = parse_env_u64(CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV, value);
    }
    if let Some(value) = max_source_chunk_bytes {
        resolved.max_source_chunk_bytes = parse_env_u64(CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, value);
    }
    if let Some(value) = max_row_groups_per_chunk {
        resolved.max_row_groups_per_chunk = parse_env_usize(CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, value);
    }
    if let Some(value) = max_retry_attempts {
        resolved.max_retry_attempts = parse_env_usize(CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, value);
    }
    if let Some(value) = min_retry_chunk_row_groups {
        resolved.min_retry_chunk_row_groups = parse_env_usize(CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = source_chunk_planning_mode {
        resolved.source_chunk_planning_mode = parse_env_source_chunk_planning_mode(value);
    }
    validate_tpcds_native_profile_settings(&resolved);

    let conservative = native_profile_base(Some(TPCDS_NATIVE_PROFILE_CONSERVATIVE));
    let sf100 = native_profile_base(Some(TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT));
    let requested_without_override = requested_profile.is_some() && !had_override;
    if same_profile_values(&resolved, conservative) {
        resolved.profile_name = TPCDS_NATIVE_PROFILE_CONSERVATIVE;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if same_profile_values(&resolved, sf100) {
        resolved.profile_name = TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if had_override {
        resolved.profile_name = TPCDS_NATIVE_PROFILE_CUSTOM;
        resolved.profile_source = "custom_overrides";
    }
    resolved
}

fn validate_tpcds_native_profile_settings(profile: &TpcdsNativeProfileSettings) {
    validate_positive(CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV, profile.device_budget_bytes);
    validate_positive(CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, profile.max_source_chunk_bytes);
    validate_positive(CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, profile.max_row_groups_per_chunk);
    validate_positive(CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, profile.max_retry_attempts);
    validate_positive(CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, profile.min_retry_chunk_row_groups);
}

fn validate_positive<T>(name: &'static str, value: T)
where T: PartialEq + From<u8> {
    assert!(value != T::from(0), "unsupported {name}=0; expected positive unsigned integer");
}

fn native_profile_base(profile: Option<&str>) -> NativeProfileBase {
    let profile = profile.unwrap_or(TPCDS_NATIVE_PROFILE_CONSERVATIVE);
    match profile {
        TPCDS_NATIVE_PROFILE_CONSERVATIVE => NativeProfileBase {
            name: TPCDS_NATIVE_PROFILE_CONSERVATIVE,
            source: "default",
            device_budget_bytes: NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
            max_source_chunk_bytes: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::SourceChunkPlanningMode::Conservative,
        },
        TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT => NativeProfileBase {
            name: TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
            source: "env_profile",
            device_budget_bytes: TPCDS_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TPCDS_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TPCDS_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::SourceChunkPlanningMode::LocalParquetThroughput,
        },
        other => panic!(
            "unsupported {CUDF_TPCDS_NATIVE_PROFILE_ENV}={other}; expected {TPCDS_NATIVE_PROFILE_CONSERVATIVE} or {TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT}"
        ),
    }
}

fn same_profile_values(profile: &TpcdsNativeProfileSettings, base: NativeProfileBase) -> bool {
    profile.device_budget_bytes == base.device_budget_bytes
        && profile.max_source_chunk_bytes == base.max_source_chunk_bytes
        && profile.max_row_groups_per_chunk == base.max_row_groups_per_chunk
        && profile.max_retry_attempts == base.max_retry_attempts
        && profile.min_retry_chunk_row_groups == base.min_retry_chunk_row_groups
        && profile.source_chunk_planning_mode == base.source_chunk_planning_mode
}

fn parse_env_u64(name: &'static str, value: String) -> u64 {
    value
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
}

fn parse_env_usize(name: &'static str, value: String) -> usize {
    value
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
}

fn parse_env_bool(name: &'static str, value: String) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enabled" => true,
        "0" | "false" | "no" | "off" | "disabled" => false,
        other => panic!("unsupported {name}={other}; expected true/false, yes/no, on/off, enabled/disabled, or 1/0"),
    }
}

fn parse_env_source_chunk_planning_mode(value: String) -> nexus_query_engine::SourceChunkPlanningMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => nexus_query_engine::SourceChunkPlanningMode::Conservative,
        "local_parquet_throughput" | "local_throughput" => {
            nexus_query_engine::SourceChunkPlanningMode::LocalParquetThroughput
        },
        other => panic!(
            "unsupported {CUDF_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV}={other}; expected conservative or local_parquet_throughput"
        ),
    }
}

fn optional_env_statistical_aggregate_execution_mode() -> Option<nexus_query_engine::StatisticalAggregateExecutionMode>
{
    optional_env_statistical_aggregate_execution_mode_for(CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV)
        .or_else(|| {
            optional_env_statistical_aggregate_execution_mode_for(CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV)
        })
}

fn optional_env_statistical_aggregate_execution_mode_for(
    name: &'static str,
) -> Option<nexus_query_engine::StatisticalAggregateExecutionMode> {
    std::env::var(name).ok().map(|value| {
        value.parse::<nexus_query_engine::StatisticalAggregateExecutionMode>().unwrap_or_else(|error| {
            panic!("unsupported {name}={value}; expected datafusion_exact_host_welford or gpu_native_tolerant: {error}")
        })
    })
}

/// Builds fallback-pressure planning items for all TPC-DS queries.
///
/// Planning failures are returned as per-query workload items so reports keep
/// a row for each query instead of aborting the whole workload.
pub async fn tpcds_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> TpcdsWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpcds_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => NativeFallbackPressureWorkloadItem::new(query.id, plan),
                Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
}

/// Builds native-planning report items using the default TPC-DS native config.
pub async fn tpcds_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    tpcds_datafusion_planning_workload_items_with_optimizer_config(
        root,
        session_config,
        tpcds_native_optimizer_config(),
    )
    .await
}

/// Builds native-planning report items using an explicit optimizer config.
///
/// The native rule retains diagnostics for the most recently planned query;
/// missing diagnostics are captured as per-query planning failures.
pub async fn tpcds_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_tpcds_session_config(session_config))
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => NativeDataFusionPlanningWorkloadItem::new(query.id, plan, diagnostics),
                    None => NativeDataFusionPlanningWorkloadItem::planning_failure(
                        query.id,
                        "native_diagnostics",
                        "missing_native_diagnostics",
                        format!("missing retained native planning diagnostics for {}", query.id),
                    ),
                },
                Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
}

/// Builds native execution report items using the default TPC-DS native config.
///
/// Native runtime starts only for queries where the optimizer produced at
/// least one native rewrite; other rows keep `NotStarted` runtime status.
pub async fn tpcds_datafusion_execution_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    tpcds_datafusion_execution_workload_items_with_optimizer_config(
        root,
        session_config,
        tpcds_native_optimizer_config(),
    )
    .await
}

/// Builds native execution report items using an explicit optimizer config.
///
/// This may touch CUDA because accepted native plans are executed.
pub async fn tpcds_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    tpcds_datafusion_execution_workload_items_filtered(root, session_config, optimizer_config, None, None).await
}

/// Per-query lifecycle events emitted by the filtered TPC-DS execution loop.
/// `query_index` is 1-based and counts only the queries selected for this run.
pub enum TpcdsExecutionProgressEvent<'a> {
    /// A selected query is about to be planned and possibly executed.
    QueryStarted {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable TPC-DS query id.
        query_id: &'a str,
    },
    /// A selected query finished. `elapsed_ms` is wall-clock milliseconds.
    QueryFinished {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable TPC-DS query id.
        query_id: &'a str,
        /// Elapsed wall-clock milliseconds for planning and execution.
        elapsed_ms: u128,
        /// Stable runtime outcome label.
        runtime_outcome: &'a str,
    },
}

/// Builds execution report items for an optional exact query-id subset.
///
/// Unknown query ids are silently ignored. Progress callbacks are emitted in
/// selected-query order: one start event, then one finish event per query.
/// Runtime outcome strings match the execution report vocabulary.
pub async fn tpcds_datafusion_execution_workload_items_filtered(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_ids: Option<&[&str]>,
    mut progress: Option<&mut dyn FnMut(TpcdsExecutionProgressEvent<'_>)>,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_tpcds_session_config(session_config))
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;

    let selected: Vec<&TpcdsQuery> = match query_ids {
        Some(ids) => TPCDS_QUERIES.iter().filter(|q| ids.iter().any(|id| id == &q.id)).collect(),
        None => TPCDS_QUERIES.iter().collect(),
    };
    let total = selected.len();
    let mut items = Vec::with_capacity(total);
    for (index, query) in selected.into_iter().enumerate() {
        let query_index = index + 1;
        if let Some(cb) = progress.as_deref_mut() {
            cb(TpcdsExecutionProgressEvent::QueryStarted { query_index, total, query_id: query.id });
        }
        let started = std::time::Instant::now();
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => {
                        let runtime = if diagnostics.report.rewritten_count() > 0 {
                            execute_tpcds_native_plan(Arc::clone(&plan)).await
                        } else {
                            NativeDataFusionRuntime::NotStarted
                        };
                        NativeDataFusionExecutionWorkloadItem::new(query.id, plan, diagnostics, runtime)
                    },
                    None => NativeDataFusionExecutionWorkloadItem::planning_failure(
                        query.id,
                        "native_diagnostics",
                        "missing_native_diagnostics",
                        format!("missing retained native planning diagnostics for {}", query.id),
                    ),
                },
                Err(error) => NativeDataFusionExecutionWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeDataFusionExecutionWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        if let Some(cb) = progress.as_deref_mut() {
            let runtime_outcome = match &item.input {
                NativeDataFusionExecutionWorkloadInput::PlanningFailure(_) => "planning_failure",
                NativeDataFusionExecutionWorkloadInput::Planned { runtime, .. } => match runtime {
                    NativeDataFusionRuntime::NotStarted => "not_started",
                    NativeDataFusionRuntime::Succeeded { .. } => "succeeded",
                    NativeDataFusionRuntime::Failed { .. } => "failed",
                },
            };
            cb(TpcdsExecutionProgressEvent::QueryFinished {
                query_index,
                total,
                query_id: query.id,
                elapsed_ms: started.elapsed().as_millis(),
                runtime_outcome,
            });
        }
        items.push(item);
    }
    Ok(items)
}

async fn execute_tpcds_native_plan(plan: Arc<dyn ExecutionPlan>) -> NativeDataFusionRuntime {
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let output_batches = batches.len();
            let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
            let output_bytes = batches.iter().map(get_record_batch_memory_size).sum();
            let metrics = plan.metrics().unwrap_or_default();
            NativeDataFusionRuntime::Succeeded { output_batches, output_rows, output_bytes, metrics }
        },
        Err(error) => NativeDataFusionRuntime::Failed { message: error.to_string() },
    }
}

pub async fn register_tpcds_tables(ctx: &SessionContext, root: &Path) -> TpcdsWorkloadResult<()> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let path = layout.anchor.to_str().ok_or_else(|| TpcdsWorkloadError::NonUtf8Path(layout.anchor.clone()))?;
        let schema = tpcds_table_schema(&layout)?;
        if let Some(key) = tpcds_source_unique_key(table) {
            register_source_unique_keys_for_path(&layout.anchor, &[key]);
            for file in &layout.files {
                register_source_unique_keys_for_path(file, &[key]);
            }
        }
        ctx.register_parquet(table, path, ParquetReadOptions::default().schema(&schema)).await?;
    }
    Ok(())
}

fn tpcds_table_schema(layout: &TpcdsTableLayout) -> TpcdsWorkloadResult<Schema> {
    let first_file = layout.files.first().ok_or_else(|| {
        TpcdsWorkloadError::InvalidDataset("expected at least one Parquet file for TPC-DS table".to_string())
    })?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(File::open(first_file)?)?.schema().as_ref().clone())
}

pub(crate) fn tpcds_source_unique_key(table: &str) -> Option<&'static [&'static str]> {
    match table {
        "call_center" => Some(&["cc_call_center_sk"]),
        "catalog_page" => Some(&["cp_catalog_page_sk"]),
        "customer" => Some(&["c_customer_sk"]),
        "customer_address" => Some(&["ca_address_sk"]),
        "customer_demographics" => Some(&["cd_demo_sk"]),
        "date_dim" => Some(&["d_date_sk"]),
        "household_demographics" => Some(&["hd_demo_sk"]),
        "income_band" => Some(&["ib_income_band_sk"]),
        "item" => Some(&["i_item_sk"]),
        "promotion" => Some(&["p_promo_sk"]),
        "reason" => Some(&["r_reason_sk"]),
        "ship_mode" => Some(&["sm_ship_mode_sk"]),
        "store" => Some(&["s_store_sk"]),
        "time_dim" => Some(&["t_time_sk"]),
        "warehouse" => Some(&["w_warehouse_sk"]),
        "web_page" => Some(&["wp_web_page_sk"]),
        "web_site" => Some(&["web_site_sk"]),
        _ => None,
    }
}
