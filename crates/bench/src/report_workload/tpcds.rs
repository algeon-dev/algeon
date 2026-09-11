use std::{path::Path, sync::Arc, time::Duration};

use algeon_datafusion::{
    backend::AlgeonGpuBackend,
    native_exec::AlgeonNativeExecutionConfig,
    native_report::{
        NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
        NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
    },
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule},
    session::AlgeonSessionStateBuilderExt,
    source_cardinality::schema_with_source_unique_keys,
};
use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use workloads::tpcds::{
    TPCDS_QUERIES, TPCDS_TABLES, TpcdsQuery, WorkloadError, WorkloadResult, discover_tpcds_table_layout,
    registration::tpcds_table_schema,
};

mod parity;
pub use parity::{
    TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, TpcdsParityProgressEvent,
    tpcds_parity_report_rows_with_external_oracles,
    tpcds_parity_report_rows_with_external_oracles_and_device_memory_limit, tpcds_parity_workload_items,
    tpcds_parity_workload_items_with_external_oracles, tpcds_parity_workload_items_with_session_configs,
};

/// Device-byte budget for bounded native TPC-DS execution.
///
/// When unset, the workload uses CUDA/cudf inventory total memory minus
/// 500 MiB, with a 256 MiB fallback if GPU memory cannot be detected. Invalid
/// values panic during config construction.
pub const ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES";
/// Named native memory/source profile used by TPC-DS benchmark/report
/// surfaces. Accepts `conservative` (default) or `sf100_gpu_throughput`;
/// individual `ALGEON_TPCDS_NATIVE_*` knobs may still override the profile.
pub const ALGEON_TPCDS_NATIVE_PROFILE_ENV: &str = "ALGEON_TPCDS_NATIVE_PROFILE";
/// Planning target, in bytes, for native Parquet source chunks.
pub const ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Planning target for row groups per native Parquet source chunk.
pub const ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum bounded retry attempts after resource or reservation pressure.
pub const ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS";
/// Minimum row-group granularity used when retrying with smaller chunks.
pub const ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Source chunk planning mode used by bounded native TPC-DS execution. Accepts
/// `conservative` or `local_parquet_throughput`; defaults to the selected
/// profile.
pub const ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str = "ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE";
/// Execution contract for native floating statistical aggregates. Accepts
/// `datafusion_exact_host_welford` or `gpu_native_tolerant`; defaults depend on
/// the caller's workload surface.
pub const ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Enables or disables native cost selection for TPC-DS analytics planning.
///
/// Native all-GPU report/parity surfaces intentionally keep cost selection
/// disabled while SQL may include GPU-only `cugraph_*` table functions.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_ENV: &str = "ALGEON_TPCDS_NATIVE_COST_SELECTION";
/// Row-group threshold for small/selective cost selection on the TPC-DS
/// analytics surface.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV: &str =
    "ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS";
/// Source-read byte threshold for small/selective cost selection on the TPC-DS
/// analytics surface.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV: &str =
    "ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES";
/// Row-group threshold for residual-join cost selection on the TPC-DS
/// analytics surface.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV: &str =
    "ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS";
/// Source-read byte threshold for residual-join cost selection on the TPC-DS
/// analytics surface.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV: &str =
    "ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES";
/// Top-k fetch threshold for small/selective cost selection on the TPC-DS
/// analytics surface.
pub const ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH_ENV: &str =
    "ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH";
/// TPC-DS cost-selection residual-join row-group threshold.
///
/// This stays intentionally close to the engine default while admitting
/// low-byte residual joins whose pruning evidence is spread across a modest
/// number of Parquet row groups.
pub const TPCDS_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS: usize = 32;

/// Conservative bounded profile: current production-safe defaults.
pub const TPCDS_NATIVE_PROFILE_CONSERVATIVE: &str = "conservative";
/// SF100 throughput profile: TPC-H-like large local Parquet batches for GPU benchmarking.
pub const TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT: &str = "sf100_gpu_throughput";
/// Profile name used when explicit knobs do not match a named profile.
pub const TPCDS_NATIVE_PROFILE_CUSTOM: &str = "custom";
/// SF100 throughput profile device budget, matching the 95 GiB probe.
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
    pub source_chunk_planning_mode: query_engine::runtime::SourceChunkPlanningMode,
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
    source_chunk_planning_mode: query_engine::runtime::SourceChunkPlanningMode,
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
/// table-level statistics collection. DataFusion physical join statistics
/// can overflow on large multi-join TPC-DS plans; native source planning
/// reads row-group metadata directly from Parquet footers instead of
/// relying on those table estimates.
/// DataFusion defaults uncorrelated scalar subqueries to `ScalarSubqueryExec`;
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

pub(crate) fn build_tpcds_gpu_backend(
    optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: query_engine::runtime::GpuDeviceMemoryLimit,
) -> WorkloadResult<AlgeonGpuBackend> {
    let source_work_per_attempt =
        optimizer_config.native_execution_config().source_scheduler_policy().max_source_reads_in_flight();
    crate::gpu_backend::build_installed_gpu_backend(
        optimizer_config,
        device_memory_limit,
        source_work_per_attempt,
        1,
        algeon_datafusion::planner::DataFusionCpuExecutionPolicy::Allow,
    )
    .map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)).into())
}

pub(crate) fn close_tpcds_gpu_backend(backend: AlgeonGpuBackend) -> WorkloadResult<()> {
    backend
        .close_and_wait(Duration::from_secs(30))
        .map(|_| ())
        .map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)).into())
}

/// Native optimizer config used by TPC-DS report surfaces.
///
/// Numeric env vars parse unsigned integers. Source chunk bytes, row-group
/// limits are planning targets; positive-value validation happens when the
/// native execution context is built. Cost-selection knobs are intentionally
/// dormant until the native SQL surface can distinguish ordinary TPC-DS shapes
/// from GPU-only `cugraph_*` calls.
pub fn tpcds_native_optimizer_config() -> AlgeonNativeOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default(
        query_engine::mode::StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    )
}

/// Native optimizer config using an already-resolved TPC-DS native profile.
pub fn tpcds_native_optimizer_config_with_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> AlgeonNativeOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        query_engine::mode::StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
        profile,
    )
}

/// Native optimizer config for TPC-DS analytics/performance report surfaces.
///
/// The analytics default uses GPU-native tolerant floating statistical
/// aggregate kernels. Set
/// [`ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV`] to
/// `datafusion_exact_host_welford` when a run intentionally needs bit-exact
/// DataFusion compatibility.
pub fn tpcds_native_analytics_optimizer_config() -> AlgeonNativeOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default(
        query_engine::mode::StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
}

/// Analytics/performance optimizer config using an already-resolved TPC-DS native profile.
pub fn tpcds_native_analytics_optimizer_config_with_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> AlgeonNativeOptimizerConfig {
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        query_engine::mode::StatisticalAggregateExecutionMode::GpuNativeTolerant,
        profile,
    )
}

/// Analytics/performance optimizer config for the `cost_declined_datafusion`
/// benchmark surface. Ordinary native TPC-DS report/parity surfaces keep cost
/// selection disabled so all-native validation remains exact; the comparison
/// benchmark uses this config to quantify the conservative cost model.
pub fn tpcds_native_analytics_optimizer_config_with_cost_selection_and_profile_settings(
    profile: &TpcdsNativeProfileSettings,
) -> AlgeonNativeOptimizerConfig {
    tpcds_native_analytics_optimizer_config_with_profile_settings(profile)
        .with_native_plan_cost_options(tpcds_native_cost_selection_options_from_env())
}

/// Resolves TPC-DS analytics cost-selection options from env.
pub fn tpcds_native_cost_selection_options_from_env() -> query_engine::plan::NativePlanCostOptions {
    tpcds_native_cost_selection_options_with_env(|name| std::env::var(name).ok())
}

/// Resolves TPC-DS analytics cost-selection options using an injected
/// env reader for tests and benchmark harnesses.
pub fn tpcds_native_cost_selection_options_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> query_engine::plan::NativePlanCostOptions {
    let mut options = query_engine::plan::NativePlanCostOptions {
        max_small_selective_residual_join_row_groups: TPCDS_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS,
        ..query_engine::plan::NativePlanCostOptions::default()
    };

    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_ENV) {
        options.enabled = parse_env_bool(ALGEON_TPCDS_NATIVE_COST_SELECTION_ENV, value);
    }
    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV) {
        options.max_small_selective_row_groups =
            parse_env_usize(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV) {
        options.max_small_selective_source_bytes =
            parse_env_u64(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV, value);
    }
    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV) {
        options.max_small_selective_residual_join_row_groups =
            parse_env_usize(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV) {
        options.max_small_selective_residual_join_source_bytes =
            parse_env_u64(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV, value);
    }
    if let Some(value) = get(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH_ENV) {
        options.max_topk_fetch = parse_env_usize(ALGEON_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH_ENV, value);
    }

    options
}

fn tpcds_native_optimizer_config_with_statistical_aggregate_default(
    statistical_aggregate_execution_mode_default: query_engine::mode::StatisticalAggregateExecutionMode,
) -> AlgeonNativeOptimizerConfig {
    let profile = tpcds_native_profile_settings_from_env();
    tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
        statistical_aggregate_execution_mode_default,
        &profile,
    )
}

fn tpcds_native_optimizer_config_with_statistical_aggregate_default_and_profile(
    statistical_aggregate_execution_mode_default: query_engine::mode::StatisticalAggregateExecutionMode,
    profile: &TpcdsNativeProfileSettings,
) -> AlgeonNativeOptimizerConfig {
    let statistical_aggregate_execution_mode =
        optional_env_statistical_aggregate_execution_mode().unwrap_or(statistical_aggregate_execution_mode_default);
    // Keep cost selection disabled for ordinary report surfaces: users can submit SQL containing
    // `cugraph_*` table functions, and those plans must stay on the GPU rather
    // than being declined by cost. The dedicated comparison benchmark installs
    // its own cost-selection configuration after the GPU-island boundary is explicit.
    let cost_options = query_engine::plan::NativePlanCostOptions::disabled();
    AlgeonNativeOptimizerConfig::default()
        .with_native_execution_config(
            AlgeonNativeExecutionConfig::bounded(
                profile.device_budget_bytes,
                profile.max_source_chunk_bytes,
                profile.max_row_groups_per_chunk,
                profile.max_retry_attempts,
                profile.min_retry_chunk_row_groups,
            )
            .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode)
            .with_source_chunk_planning_mode(profile.source_chunk_planning_mode),
        )
        .with_native_plan_cost_options(cost_options)
}

/// Resolves the TPC-DS native bounded memory/source profile from env.
///
/// `ALGEON_TPCDS_NATIVE_PROFILE` chooses a named base profile:
/// `conservative` keeps the existing bounded defaults, while
/// `sf100_gpu_throughput` applies the TPC-H-like SF100 local Parquet profile.
/// The individual `ALGEON_TPCDS_NATIVE_*` knobs override either base. If
/// those final values exactly match a named profile, the resolved
/// `profile_name` is that profile; otherwise it is `custom`.
pub fn tpcds_native_profile_settings_from_env() -> TpcdsNativeProfileSettings {
    tpcds_native_profile_settings_and_device_memory_limit_from_env().0
}

pub fn tpcds_native_profile_settings_with_env(
    get: impl FnMut(&'static str) -> Option<String>,
) -> TpcdsNativeProfileSettings {
    tpcds_native_profile_settings_and_device_memory_limit_with_env(get).0
}

pub fn tpcds_native_profile_settings_and_device_memory_limit_from_env()
-> (TpcdsNativeProfileSettings, query_engine::runtime::GpuDeviceMemoryLimit) {
    tpcds_native_profile_settings_and_device_memory_limit_with_env(|name| std::env::var(name).ok())
}

fn tpcds_native_profile_settings_and_device_memory_limit_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> (TpcdsNativeProfileSettings, query_engine::runtime::GpuDeviceMemoryLimit) {
    let requested_profile = get(ALGEON_TPCDS_NATIVE_PROFILE_ENV).map(|value| value.trim().to_ascii_lowercase());
    let base = native_profile_base(requested_profile.as_deref());
    let mut resolved = TpcdsNativeProfileSettings::from_base(base);
    let device_budget_bytes = get(ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV);
    let max_source_chunk_bytes = get(ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV);
    let max_row_groups_per_chunk = get(ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV);
    let max_retry_attempts = get(ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV);
    let min_retry_chunk_row_groups = get(ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV);
    let source_chunk_planning_mode = get(ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV);
    let device_budget_overridden = device_budget_bytes.is_some();
    let had_override = device_budget_overridden
        || max_source_chunk_bytes.is_some()
        || max_row_groups_per_chunk.is_some()
        || max_retry_attempts.is_some()
        || min_retry_chunk_row_groups.is_some()
        || source_chunk_planning_mode.is_some();

    if let Some(value) = device_budget_bytes {
        resolved.device_budget_bytes = parse_env_u64(ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV, value);
    }
    if let Some(value) = max_source_chunk_bytes {
        resolved.max_source_chunk_bytes = parse_env_u64(ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, value);
    }
    if let Some(value) = max_row_groups_per_chunk {
        resolved.max_row_groups_per_chunk = parse_env_usize(ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, value);
    }
    if let Some(value) = max_retry_attempts {
        resolved.max_retry_attempts = parse_env_usize(ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, value);
    }
    if let Some(value) = min_retry_chunk_row_groups {
        resolved.min_retry_chunk_row_groups =
            parse_env_usize(ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, value);
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
    let device_memory_limit = if device_budget_overridden {
        query_engine::runtime::GpuDeviceMemoryLimit::Explicit { limit_bytes: resolved.device_budget_bytes }
    } else {
        query_engine::runtime::GpuDeviceMemoryLimit::default()
    };
    (resolved, device_memory_limit)
}

fn validate_tpcds_native_profile_settings(profile: &TpcdsNativeProfileSettings) {
    validate_positive(ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV, profile.device_budget_bytes);
    validate_positive(ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, profile.max_source_chunk_bytes);
    validate_positive(ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, profile.max_row_groups_per_chunk);
    validate_positive(ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, profile.max_retry_attempts);
    validate_positive(ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, profile.min_retry_chunk_row_groups);
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
            device_budget_bytes: AlgeonNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            max_source_chunk_bytes: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: query_engine::runtime::SourceChunkPlanningMode::Conservative,
        },
        TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT => NativeProfileBase {
            name: TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
            source: "env_profile",
            // The throughput profile tunes source chunking only; capacity
            // still resolves from the installed device so a same-series card
            // with slightly less CUDA-visible memory can initialize.
            device_budget_bytes: AlgeonNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            max_source_chunk_bytes: TPCDS_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TPCDS_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput,
        },
        other => panic!(
            "unsupported {ALGEON_TPCDS_NATIVE_PROFILE_ENV}={other}; expected {TPCDS_NATIVE_PROFILE_CONSERVATIVE} or {TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT}"
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

fn parse_env_source_chunk_planning_mode(value: String) -> query_engine::runtime::SourceChunkPlanningMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => query_engine::runtime::SourceChunkPlanningMode::Conservative,
        "local_parquet_throughput" | "local_throughput" => {
            query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
        },
        other => panic!(
            "unsupported {ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV}={other}; expected conservative or local_parquet_throughput"
        ),
    }
}

fn optional_env_statistical_aggregate_execution_mode() -> Option<query_engine::mode::StatisticalAggregateExecutionMode>
{
    optional_env_statistical_aggregate_execution_mode_for(ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV)
}

fn optional_env_statistical_aggregate_execution_mode_for(
    name: &'static str,
) -> Option<query_engine::mode::StatisticalAggregateExecutionMode> {
    std::env::var(name).ok().map(|value| {
        value.parse::<query_engine::mode::StatisticalAggregateExecutionMode>().unwrap_or_else(|error| {
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
) -> WorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpcds_tables_with_native_metadata(&ctx, root).await?;

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
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
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
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(AlgeonNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_tpcds_session_config(session_config))
        .with_algeon_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables_with_native_metadata(&ctx, root).await?;

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
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let (profile, device_memory_limit) = tpcds_native_profile_settings_and_device_memory_limit_from_env();
    tpcds_datafusion_execution_workload_items_filtered_with_device_memory_limit(
        root,
        session_config,
        tpcds_native_optimizer_config_with_profile_settings(&profile),
        device_memory_limit,
        None,
        None,
    )
    .await
}

/// Builds native execution report items using an explicit optimizer config.
///
/// This may touch CUDA because accepted native plans are executed.
pub async fn tpcds_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    tpcds_datafusion_execution_workload_items_filtered_with_device_memory_limit(
        root,
        session_config,
        optimizer_config,
        query_engine::runtime::GpuDeviceMemoryLimit::default(),
        None,
        None,
    )
    .await
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
    optimizer_config: AlgeonNativeOptimizerConfig,
    query_ids: Option<&[&str]>,
    progress: Option<&mut dyn FnMut(TpcdsExecutionProgressEvent<'_>)>,
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    tpcds_datafusion_execution_workload_items_filtered_with_device_memory_limit(
        root,
        session_config,
        optimizer_config,
        query_engine::runtime::GpuDeviceMemoryLimit::default(),
        query_ids,
        progress,
    )
    .await
}

pub async fn tpcds_datafusion_execution_workload_items_filtered_with_device_memory_limit(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: query_engine::runtime::GpuDeviceMemoryLimit,
    query_ids: Option<&[&str]>,
    mut progress: Option<&mut dyn FnMut(TpcdsExecutionProgressEvent<'_>)>,
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let backend = build_tpcds_gpu_backend(optimizer_config, device_memory_limit)?;
    let native_rule = Arc::new(AlgeonNativePhysicalOptimizerRule::new(backend.optimizer_config().clone()));
    let items = async {
        let builder =
            SessionStateBuilder::new_with_default_features().with_config(native_tpcds_session_config(session_config));
        let builder = backend
            .install_on(builder)
            .map_err(|error| datafusion::error::DataFusionError::External(Box::new(error)))?;
        let state = builder.with_algeon_native_rule(Arc::clone(&native_rule)).build();
        let ctx = SessionContext::new_with_state(state);
        register_tpcds_tables_with_native_metadata(&ctx, root).await?;

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
                            let runtime = if diagnostics.report.selected_count() > 0 {
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
                        NativeDataFusionRuntime::Interrupted { outcome, .. } => outcome.as_str(),
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
    .await;
    match items {
        Ok(items) => {
            close_tpcds_gpu_backend(backend)?;
            Ok(items)
        },
        Err(error) => {
            // The rule retains backend configuration until its contexts are released.
            drop(native_rule);
            let _ = close_tpcds_gpu_backend(backend);
            Err(error)
        },
    }
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
        Err(error) => NativeDataFusionRuntime::Failed { message: error.to_string(), metrics: plan.metrics() },
    }
}

/// Registers TPC-DS tables with unique-key evidence carried in each provider
/// schema's metadata, so native lowering proves single-row scans from the
/// scan's own schema instead of external process state.
pub async fn register_tpcds_tables_with_native_metadata(ctx: &SessionContext, root: &Path) -> WorkloadResult<()> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let path = layout.anchor.to_str().ok_or_else(|| WorkloadError::NonUtf8Path(layout.anchor.clone()))?;
        let mut schema = tpcds_table_schema(&layout)?;
        if let Some(key) = tpcds_source_unique_key(table) {
            schema = schema_with_source_unique_keys(&schema, &[key]);
        }
        ctx.register_parquet(table, path, ParquetReadOptions::default().schema(&schema)).await?;
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use algeon_datafusion::source_cardinality::SOURCE_UNIQUE_KEYS_METADATA_KEY;

    use super::*;

    #[test]
    fn tpcds_profile_memory_limit_tracks_device_budget_provenance() {
        let (_, default_limit) = tpcds_native_profile_settings_and_device_memory_limit_with_env(|_| None);
        assert_eq!(default_limit, query_engine::runtime::GpuDeviceMemoryLimit::default());

        let (_, source_only_limit) = tpcds_native_profile_settings_and_device_memory_limit_with_env(|name| {
            (name == ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV).then(|| (1024_u64 * 1024 * 1024).to_string())
        });
        assert_eq!(source_only_limit, query_engine::runtime::GpuDeviceMemoryLimit::default());

        let (_, throughput_limit) = tpcds_native_profile_settings_and_device_memory_limit_with_env(|name| {
            (name == ALGEON_TPCDS_NATIVE_PROFILE_ENV).then(|| TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT.to_string())
        });
        assert_eq!(throughput_limit, query_engine::runtime::GpuDeviceMemoryLimit::default());

        let explicit_budget = 17 * 1024 * 1024 * 1024;
        let (_, overridden_limit) = tpcds_native_profile_settings_and_device_memory_limit_with_env(|name| {
            (name == ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV).then(|| explicit_budget.to_string())
        });
        assert_eq!(
            overridden_limit,
            query_engine::runtime::GpuDeviceMemoryLimit::Explicit { limit_bytes: explicit_budget }
        );
    }

    #[tokio::test]
    async fn tpcds_registration_carries_unique_key_metadata_in_provider_schema() {
        let root =
            std::env::temp_dir().join(format!("algeon-datafusion-tpcds-native-metadata-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        workloads::tpcds::write_tiny_tpcds_parquet_dataset(&root).unwrap();
        let ctx = SessionContext::new();
        register_tpcds_tables_with_native_metadata(&ctx, &root).await.unwrap();

        let provider = ctx.table_provider("reason").await.unwrap();
        let schema = provider.schema();
        assert_eq!(
            schema.field_with_name("r_reason_sk").unwrap().metadata().get(SOURCE_UNIQUE_KEYS_METADATA_KEY),
            Some(&r#"[["r_reason_sk"]]"#.to_string())
        );
        // Source-scoped evidence must not reach the shared schema namespace,
        // where a join would drop one keyed source's entry for another's.
        assert_eq!(schema.metadata().get(SOURCE_UNIQUE_KEYS_METADATA_KEY), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
