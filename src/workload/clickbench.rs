use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{error::ArrowError, record_batch::RecordBatch};
use datafusion::{
    catalog::TableProvider,
    common::utils::memory::get_record_batch_memory_size,
    datasource::{
        file_format::options::ReadOptions,
        listing::{ListingTable, ListingTableConfig, ListingTableUrl},
    },
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

pub use dataset::{
    CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE, CUDF_CLICKBENCH_CONTENT_FINGERPRINT_ENV,
    CUDF_CLICKBENCH_GENERATION_COMMAND_ENV, CUDF_CLICKBENCH_PARQUET_DIR_ENV, ClickbenchDatasetMetadata,
    ClickbenchLayout, DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR, DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET,
    DEFAULT_NVME_CLICKBENCH_PARTITIONED_DIR, describe_clickbench_dataset,
    describe_clickbench_dataset_with_content_fingerprint, discover_clickbench_layout,
    external_clickbench_dataset_metadata, prepare_clickbench_report_dataset,
};

pub use super::clickbench_queries::{
    CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, ClickbenchQuery, clickbench_query_ids,
};

pub const CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES";
pub const CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES";
pub const CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub const CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS";
pub const CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str =
    "CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";

pub type ClickbenchWorkloadResult<T> = std::result::Result<T, ClickbenchWorkloadError>;

#[derive(Debug)]
pub enum ClickbenchWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for ClickbenchWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid ClickBench dataset: {message}"),
        }
    }
}

impl StdError for ClickbenchWorkloadError {
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

impl From<std::io::Error> for ClickbenchWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for ClickbenchWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for ClickbenchWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for ClickbenchWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

/// Decision-grade session config for ClickBench reports. Mirrors the TPC-H/DS
/// report configs (single partition, no repartitioning, dynamic filter
/// pushdown disabled) and toggles `binary_as_string=true` because the
/// upstream Parquet encoding stores every string column as `BYTE_ARRAY`.
pub fn clickbench_report_session_config() -> SessionConfig {
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
    config.options_mut().execution.parquet.binary_as_string = true;
    config
}

/// Tuned CPU session config for ClickBench parity baselines. Mirrors
/// `tpch_tuned_cpu_session_config` but keeps the ClickBench-specific
/// `binary_as_string` toggle.
pub fn clickbench_tuned_cpu_session_config() -> SessionConfig {
    let target_partitions = std::thread::available_parallelism().map(usize::from).unwrap_or(1);
    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(8192)
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true)
        .with_repartition_joins(true)
        .with_repartition_sorts(true);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.parquet.binary_as_string = true;
    config
}

pub fn clickbench_native_optimizer_config() -> CudfOptimizerConfig {
    let device_budget_bytes = optional_env_u64(CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV)
        .unwrap_or_else(default_clickbench_device_budget_bytes);
    let max_source_chunk_bytes =
        optional_env_u64(CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV).unwrap_or(64 * 1024 * 1024);
    let max_row_groups_per_chunk = optional_env_usize(CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV).unwrap_or(1);
    let max_retry_attempts = optional_env_usize(CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV).unwrap_or(3);
    let min_retry_chunk_row_groups =
        optional_env_usize(CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV).unwrap_or(1);
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        ),
    )
}

/// Default device budget when `CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES` is
/// unset. Queries `nvidia-smi` for the primary GPU's total memory and
/// reserves 500 MiB for libcudf/driver overhead. Falls back to 256 MiB on
/// failure. Matches the TPC-DS heuristic so ClickBench reports can run on the
/// same GPU sizes without an explicit override.
fn default_clickbench_device_budget_bytes() -> u64 {
    const FALLBACK_BYTES: u64 = 256 * 1024 * 1024;
    const HEADROOM_BYTES: u64 = 500 * 1024 * 1024;
    let output = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"])
        .output()
    {
        Ok(output) if output.status.success() => output.stdout,
        _ => return FALLBACK_BYTES,
    };
    let total_mib: u64 = match std::str::from_utf8(&output).ok().and_then(|s| s.trim().lines().next()) {
        Some(line) => match line.trim().parse() {
            Ok(value) => value,
            Err(_) => return FALLBACK_BYTES,
        },
        None => return FALLBACK_BYTES,
    };
    let total_bytes = total_mib.saturating_mul(1024 * 1024);
    total_bytes.checked_sub(HEADROOM_BYTES).filter(|&v| v > 0).unwrap_or(FALLBACK_BYTES)
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

/// Register the ClickBench fixture as `hits_raw` + the canonical `hits` view.
///
/// The view applies the `EventDate` cast from `ClickBench/datafusion/create.sql`
/// so upstream queries reference dates rather than the underlying epoch-day
/// integer. Callers must use a [`SessionContext`] whose config has
/// `execution.parquet.binary_as_string = true` (use
/// [`clickbench_report_session_config`] or
/// [`clickbench_tuned_cpu_session_config`]); otherwise the upstream blob/
/// string columns will be loaded as `Binary` and the queries that filter on
/// string literals will fail to plan.
pub async fn register_clickbench_table(
    ctx: &SessionContext,
    layout: &ClickbenchLayout,
) -> ClickbenchWorkloadResult<()> {
    match layout {
        ClickbenchLayout::Single { file } => {
            let path_str = file.to_str().ok_or_else(|| ClickbenchWorkloadError::NonUtf8Path(file.clone()))?;
            ctx.register_parquet(CLICKBENCH_RAW_TABLE, path_str, ParquetReadOptions::default()).await?;
        },
        ClickbenchLayout::Partitioned { files, .. } => {
            register_clickbench_partitioned_shards(ctx, files).await?;
        },
    }
    let view_sql = format!(
        r#"CREATE VIEW {CLICKBENCH_TABLE} AS SELECT * EXCEPT ("EventDate"), CAST(CAST("EventDate" AS INTEGER) AS DATE) AS "EventDate" FROM {CLICKBENCH_RAW_TABLE}"#
    );
    ctx.sql(&view_sql).await?;
    Ok(())
}

/// Register the discovered `hits_*.parquet` shards as an explicit multi-path
/// listing so the registration never silently picks up stray Parquet files
/// that live alongside the canonical shards (e.g. user exports or stale
/// snapshots). Mirrors the multi-path branch of `register_tpch_table_layout`
/// for sibling-split tables, but threads the session's `binary_as_string`
/// option through via `ParquetReadOptions::to_listing_options` so the
/// upstream blob/string columns surface as UTF-8 strings.
async fn register_clickbench_partitioned_shards(
    ctx: &SessionContext,
    files: &[PathBuf],
) -> ClickbenchWorkloadResult<()> {
    let urls = files
        .iter()
        .map(|path| -> ClickbenchWorkloadResult<ListingTableUrl> {
            let absolute = path.canonicalize().map_err(|error| {
                ClickbenchWorkloadError::InvalidDataset(format!(
                    "failed to canonicalize ClickBench shard {}: {error}",
                    path.display()
                ))
            })?;
            let absolute_str =
                absolute.to_str().ok_or_else(|| ClickbenchWorkloadError::NonUtf8Path(absolute.clone()))?;
            Ok(ListingTableUrl::parse(format!("file://{absolute_str}"))?)
        })
        .collect::<ClickbenchWorkloadResult<Vec<_>>>()?;
    if urls.is_empty() {
        return Err(ClickbenchWorkloadError::InvalidDataset(
            "register_clickbench_partitioned_shards called with zero shards".to_string(),
        ));
    }
    let session_state = ctx.state();
    let listing_options =
        ParquetReadOptions::default().to_listing_options(session_state.config(), session_state.default_table_options());
    let inferred_schema = listing_options.infer_schema(&session_state, &urls[0]).await?;
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(listing_options)
        .with_schema(inferred_schema);
    let listing_table: Arc<dyn TableProvider> = Arc::new(ListingTable::try_new(config)?);
    ctx.register_table(CLICKBENCH_RAW_TABLE, listing_table)?;
    Ok(())
}

pub async fn clickbench_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let mut items = Vec::with_capacity(CLICKBENCH_QUERIES.len());
    for query in CLICKBENCH_QUERIES {
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

pub async fn clickbench_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    clickbench_datafusion_planning_workload_items_with_optimizer_config(
        root,
        session_config,
        clickbench_native_optimizer_config(),
    )
    .await
}

pub async fn clickbench_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let mut items = Vec::with_capacity(CLICKBENCH_QUERIES.len());
    for query in CLICKBENCH_QUERIES {
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

pub async fn clickbench_datafusion_execution_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    clickbench_datafusion_execution_workload_items_with_optimizer_config(
        root,
        session_config,
        clickbench_native_optimizer_config(),
    )
    .await
}

pub async fn clickbench_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    clickbench_datafusion_execution_workload_items_filtered(root, session_config, optimizer_config, None, None).await
}

/// Per-query lifecycle events emitted by the filtered ClickBench execution
/// loop. `query_index` is 1-based and counts only the queries selected for
/// this run.
pub enum ClickbenchExecutionProgressEvent<'a> {
    QueryStarted { query_index: usize, total: usize, query_id: &'a str },
    QueryFinished { query_index: usize, total: usize, query_id: &'a str, elapsed_ms: u128, runtime_outcome: &'a str },
}

pub async fn clickbench_datafusion_execution_workload_items_filtered(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_ids: Option<&[&str]>,
    mut progress: Option<&mut dyn FnMut(ClickbenchExecutionProgressEvent<'_>)>,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let selected: Vec<&ClickbenchQuery> = match query_ids {
        Some(ids) => CLICKBENCH_QUERIES.iter().filter(|q| ids.iter().any(|id| id == &q.id)).collect(),
        None => CLICKBENCH_QUERIES.iter().collect(),
    };
    let total = selected.len();
    let mut items = Vec::with_capacity(total);
    for (index, query) in selected.into_iter().enumerate() {
        let query_index = index + 1;
        if let Some(cb) = progress.as_deref_mut() {
            cb(ClickbenchExecutionProgressEvent::QueryStarted { query_index, total, query_id: query.id });
        }
        let started = std::time::Instant::now();
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => {
                        let runtime = if diagnostics.report.rewritten_count() > 0 {
                            execute_clickbench_native_plan(Arc::clone(&plan)).await
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
            cb(ClickbenchExecutionProgressEvent::QueryFinished {
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

async fn execute_clickbench_native_plan(plan: Arc<dyn ExecutionPlan>) -> NativeDataFusionRuntime {
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
