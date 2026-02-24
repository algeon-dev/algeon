use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{error::ArrowError, record_batch::RecordBatch};
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    error::DataFusionError,
    execution::{SessionStateBuilder, context::TaskContext},
    parquet::errors::ParquetError,
    physical_plan::{ExecutionPlan, collect},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy,
    NativeCudfExecutionConfig, NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
    NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
};

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
    TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, tpcds_parity_workload_items,
    tpcds_parity_workload_items_with_external_oracles,
};
pub use tiny::write_tiny_tpcds_parquet_dataset;

pub use super::tpcds_queries::{TPCDS_QUERIES, TPCDS_QUERY_COUNT, TpcdsQuery, tpcds_query_ids};

/// Device-byte budget for bounded native TPC-DS execution.
///
/// When unset, the workload uses `nvidia-smi` memory.total minus 500 MiB, with
/// a 256 MiB fallback if GPU memory cannot be detected. Invalid values panic
/// during config construction.
pub const CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES";
/// Planning target, in bytes, for native Parquet source chunks.
pub const CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Planning target for row groups per native Parquet source chunk.
pub const CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum bounded retry attempts after resource or reservation pressure.
pub const CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS";
/// Minimum row-group granularity used when retrying with smaller chunks.
pub const CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Execution contract for native floating statistical aggregates.
pub const CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Migration alias for [`CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV`].
pub const CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV: &str = "CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE";
/// Temporarily ignored. Native planning must not prefer CPU for conservative
/// small/selective shapes while SQL may include GPU-only `cugraph_*` table
/// functions.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_ENV: &str = "CUDF_TPCDS_NATIVE_CPU_PREFERENCE";
/// Row-group threshold for small/selective CPU preference.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_ROW_GROUPS";
/// Source-read byte threshold for small/selective CPU preference.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_SMALL_SELECTIVE_SOURCE_BYTES";
/// Row-group threshold for residual-join CPU preference.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_ROW_GROUPS";
/// Source-read byte threshold for residual-join CPU preference.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV: &str =
    "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_RESIDUAL_JOIN_SOURCE_BYTES";
/// Top-k fetch threshold for small/selective CPU preference.
pub const CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH_ENV: &str = "CUDF_TPCDS_NATIVE_CPU_PREFERENCE_MAX_TOPK_FETCH";

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
/// Uses one target partition, disables DataFusion repartitioning and dynamic
/// filter pushdown, and leaves Parquet pushdown filters enabled for scan
/// pruning parity with report binaries.
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
/// This starts from [`tpcds_report_session_config`] but disables DataFusion
/// table-statistics collection. SF100 multi-join shapes can overflow
/// DataFusion's table-level cardinality estimates during physical
/// optimization; native source planning reads row-group metadata directly from
/// Parquet footers instead of relying on those estimates.
pub fn tpcds_native_report_session_config() -> SessionConfig {
    native_tpcds_session_config(tpcds_report_session_config())
}

fn native_tpcds_session_config(mut config: SessionConfig) -> SessionConfig {
    config.options_mut().execution.collect_statistics = false;
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
        cudf_query_engine::StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
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
        cudf_query_engine::StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
}

fn tpcds_native_optimizer_config_with_statistical_aggregate_default(
    statistical_aggregate_execution_mode_default: cudf_query_engine::StatisticalAggregateExecutionMode,
) -> CudfOptimizerConfig {
    let device_budget_bytes = optional_env_u64(CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV)
        .unwrap_or_else(NativeCudfExecutionConfig::auto_sized_device_budget_bytes);
    let max_source_chunk_bytes = optional_env_u64(CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    let max_row_groups_per_chunk = optional_env_usize(CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    let max_retry_attempts = optional_env_usize(CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS);
    let min_retry_chunk_row_groups = optional_env_usize(CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS);
    let statistical_aggregate_execution_mode =
        optional_env_statistical_aggregate_execution_mode().unwrap_or(statistical_aggregate_execution_mode_default);
    // Keep CPU preference disabled for now: users can submit SQL containing
    // `cugraph_*` table functions, and those plans must stay on the GPU rather
    // than being routed through DataFusion CPU fallback as a small/selective
    // preference. Leave the env constants above in place so the old benchmark
    // gate can be restored once graph SQL has an explicit GPU-only boundary.
    let preference_options = cudf_query_engine::NativePlanPreferenceOptions::disabled();
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(
            NativeCudfExecutionConfig::bounded(
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            )
            .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode),
        )
        .with_native_plan_preference_options(preference_options)
}

fn optional_env_u64(name: &'static str) -> Option<u64> {
    std::env::var(name).ok().map(|value| {
        value
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
    })
}

fn optional_env_usize(name: &'static str) -> Option<usize> {
    std::env::var(name).ok().map(|value| {
        value
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
    })
}

fn optional_env_statistical_aggregate_execution_mode() -> Option<cudf_query_engine::StatisticalAggregateExecutionMode> {
    optional_env_statistical_aggregate_execution_mode_for(CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV)
        .or_else(|| {
            optional_env_statistical_aggregate_execution_mode_for(CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV)
        })
}

fn optional_env_statistical_aggregate_execution_mode_for(
    name: &'static str,
) -> Option<cudf_query_engine::StatisticalAggregateExecutionMode> {
    std::env::var(name).ok().map(|value| {
        value.parse::<cudf_query_engine::StatisticalAggregateExecutionMode>().unwrap_or_else(|error| {
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

pub(super) async fn register_tpcds_tables(ctx: &SessionContext, root: &Path) -> TpcdsWorkloadResult<()> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let path = layout.anchor.to_str().ok_or_else(|| TpcdsWorkloadError::NonUtf8Path(layout.anchor.clone()))?;
        ctx.register_parquet(table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}
