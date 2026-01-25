use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{error::ArrowError, record_batch::RecordBatch};
use datafusion::{
    common::{ScalarValue, utils::memory::get_record_batch_memory_size},
    error::DataFusionError,
    execution::{SessionStateBuilder, context::TaskContext},
    functions_aggregate::count::count_udaf,
    logical_expr::Operator,
    parquet::errors::ParquetError,
    physical_expr::{
        PhysicalExpr,
        aggregate::AggregateExprBuilder,
        expressions::{BinaryExpr, IsNotNullExpr, col, in_list, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect,
        filter::FilterExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use dataset::resolve_tpch_table_path;

pub use crate::tpch_queries::{TPCH_QUERIES, TPCH_QUERY_COUNT, TpchQuery, tpch_query_ids};
use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy,
    NativeCudfExecutionConfig, NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
};

mod dataset;
mod tiny;

pub use dataset::{
    CUDF_TPCH_CONTENT_FINGERPRINT_ENV, CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV,
    CUDF_TPCH_SCALE_FACTOR_ENV, TPCH_TABLES, TpchDatasetMetadata, TpchTableFileMetadata, TpchTableMetadata,
    describe_tpch_dataset, describe_tpch_dataset_with_content_fingerprint, external_tpch_dataset_metadata,
    prepare_tpch_report_dataset, tiny_tpch_dataset_metadata,
};
pub use tiny::write_tiny_tpch_parquet_dataset;

pub type TpchWorkloadResult<T> = std::result::Result<T, TpchWorkloadError>;

#[derive(Debug)]
pub enum TpchWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for TpchWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid TPC-H dataset: {message}"),
        }
    }
}

impl StdError for TpchWorkloadError {
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

impl From<std::io::Error> for TpchWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for TpchWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for TpchWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for TpchWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

pub fn tpch_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

pub fn tpch_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1))
}

pub async fn tpch_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
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

pub async fn tpch_scalar_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_SCALAR_FALLBACK_PRESSURE_QUERIES.len());
    let lineitem_scan = ctx.table("lineitem").await?.create_physical_plan().await?;
    let lineitem_schema = lineitem_scan.schema();
    let in_list_with_null = in_list(
        col("l_orderkey", &lineitem_schema)?,
        vec![lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(None))],
        &false,
        lineitem_schema.as_ref(),
    )?;
    items.push(NativeFallbackPressureWorkloadItem::new(
        "q12_scalar_in_list_null",
        count_filter_plan(Arc::clone(&lineitem_scan), in_list_with_null)?,
    ));

    let interval_minus_date: Arc<dyn PhysicalExpr> = Arc::new(IsNotNullExpr::new(Arc::new(BinaryExpr::new(
        lit(ScalarValue::new_interval_mdn(0, 1, 0)),
        Operator::Minus,
        col("l_shipdate", &lineitem_schema)?,
    ))));
    items.push(NativeFallbackPressureWorkloadItem::new(
        "q01_scalar_interval_minus_date",
        count_filter_plan(lineitem_scan, interval_minus_date)?,
    ));

    for query in TPCH_SCALAR_FALLBACK_PRESSURE_QUERIES {
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

fn count_filter_plan(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan)?);
    let filter_schema = filter.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&filter_schema))
        .alias("count_rows")
        .build()?;
    Ok(Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(Vec::new()),
        vec![Arc::new(count)],
        vec![None],
        filter,
        filter_schema,
    )?))
}

pub async fn tpch_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let optimizer_config = tpch_native_optimizer_config();
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
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

pub async fn tpch_datafusion_execution_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> TpchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let optimizer_config = tpch_native_optimizer_config();
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => {
                        let runtime = if diagnostics.report.rewritten_count() > 0 {
                            execute_tpch_native_plan(Arc::clone(&plan)).await
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

async fn execute_tpch_native_plan(plan: Arc<dyn ExecutionPlan>) -> NativeDataFusionRuntime {
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

struct TpchScalarFallbackPressureQuery {
    id: &'static str,
    sql: &'static str,
}

const TPCH_SCALAR_FALLBACK_PRESSURE_QUERIES: &[TpchScalarFallbackPressureQuery] = &[
    TpchScalarFallbackPressureQuery {
        id: "q22_scalar_substring_dynamic_start",
        sql: "
SELECT COUNT(*) AS count_rows
FROM customer
WHERE SUBSTRING(c_phone FROM c_custkey FOR 2) = '13'
",
    },
    TpchScalarFallbackPressureQuery {
        id: "q07_scalar_extract_hour",
        sql: "
SELECT COUNT(*) AS count_rows
FROM orders
WHERE EXTRACT(HOUR FROM o_orderdate) = 1
",
    },
];

async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> TpchWorkloadResult<()> {
    for table in TPCH_TABLES {
        let path = resolve_tpch_table_path(root, table)?;
        let path = path.to_str().ok_or_else(|| TpchWorkloadError::NonUtf8Path(path.clone()))?;
        ctx.register_parquet(*table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}
