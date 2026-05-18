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
    NativeCudfExecutionConfig, NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
};

mod dataset;
mod parity;
mod tiny;

pub use dataset::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_PARQUET_DIR_ENV,
    CUDF_TPCDS_SCALE_FACTOR_ENV, DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR, TPCDS_TABLES, TpcdsDatasetMetadata,
    TpcdsTableFileMetadata, TpcdsTableLayout, TpcdsTableMetadata, describe_tpcds_dataset,
    describe_tpcds_dataset_with_content_fingerprint, discover_tpcds_table_layout, external_tpcds_dataset_metadata,
    prepare_tpcds_report_dataset, tiny_tpcds_dataset_metadata,
};
pub use parity::{
    TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, tpcds_parity_workload_items,
    tpcds_parity_workload_items_with_external_oracles,
};
pub use tiny::write_tiny_tpcds_parquet_dataset;

pub use super::tpcds_queries::{TPCDS_QUERIES, TPCDS_QUERY_COUNT, TpcdsQuery, tpcds_query_ids};

pub const CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES";
pub const CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES";
pub const CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub const CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS";
pub const CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";

pub type TpcdsWorkloadResult<T> = std::result::Result<T, TpcdsWorkloadError>;

#[derive(Debug)]
pub enum TpcdsWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
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

pub fn tpcds_native_optimizer_config() -> CudfOptimizerConfig {
    let device_budget_bytes = optional_env_u64(CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV).unwrap_or(256 * 1024 * 1024);
    let max_source_chunk_bytes =
        optional_env_u64(CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV).unwrap_or(64 * 1024 * 1024);
    let max_row_groups_per_chunk = optional_env_usize(CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV).unwrap_or(1);
    let max_retry_attempts = optional_env_usize(CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV).unwrap_or(3);
    let min_retry_chunk_row_groups = optional_env_usize(CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV).unwrap_or(1);
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

pub async fn tpcds_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
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

pub async fn tpcds_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
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
