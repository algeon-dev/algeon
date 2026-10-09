use std::{path::Path, sync::Arc};

use algeon_datafusion::{
    native_exec::{AlgeonNativeExecutionConfig, below_native_admission},
    native_report::{
        NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime,
        NativeFallbackPressureWorkloadItem,
    },
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, plan_with_native_diagnostics},
    session::{AlgeonSessionStateBuilderExt, apply_algeon_option_overrides},
};
use arrow::record_batch::RecordBatch;
use datafusion::{
    common::{ScalarValue, utils::memory::get_record_batch_memory_size},
    error::DataFusionError,
    execution::{SessionStateBuilder, context::TaskContext},
    functions_aggregate::count::count_udaf,
    logical_expr::Operator,
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
    prelude::{SessionConfig, SessionContext},
};
use query_runtime::runtime::GpuDeviceMemoryLimit;
use workloads::tpch::{TPCH_QUERIES, WorkloadResult, register_tpch_tables};

use crate::gpu_backend::{build_report_gpu_backend, close_report_gpu_backend};

mod analytical_olap;

pub use analytical_olap::{
    ANALYTICAL_OLAP_QUERIES, ANALYTICAL_OLAP_QUERY_COUNT, AnalyticalOlapCapabilityFirstAdmission, AnalyticalOlapQuery,
    AnalyticalOlapWorkloadClass, analytical_olap_datafusion_planning_workload_items,
    analytical_olap_datafusion_planning_workload_items_with_optimizer_config,
    analytical_olap_fallback_pressure_workload_items, analytical_olap_query_ids,
};
/// Session config used by TPC-H native report surfaces.
///
/// Uses one target partition and disables DataFusion repartitioning and
/// dynamic filter pushdown so planning/execution reports are stable and easy
/// to compare across native lowering changes. It also applies the DataFusion
/// option overrides of a relational Algeon execution mode, because plain
/// DataFusion contexts built from this config (fallback-pressure items and the
/// `report` CPU profile) must plan the physical shapes an Algeon session plans.
pub fn tpch_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    apply_algeon_option_overrides(config.options_mut(), AlgeonGpuExecutionMode::NativePreferred);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

/// Tuned CPU DataFusion baseline for TPC-H benchmark comparisons.
///
/// Uses available CPU parallelism, batch size 8192, repartitioning for scans,
/// joins, aggregates, and sorts, dynamic filter pushdown enabled, and Parquet
/// pushdown filters disabled to match the current decision-grade CPU profile.
pub fn tpch_tuned_cpu_session_config() -> SessionConfig {
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
    config
}

/// Device budget that native TPC-H report callers give the backend executing
/// [`tpch_native_optimizer_config`]. Callers that must size a device to the
/// same bound read this constant. The literal is not restated at each call
/// site.
pub const TPCH_NATIVE_DEVICE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Conservative native optimizer config used by default TPC-H report surfaces.
///
/// The bounded profile uses the default source chunking: no source-byte
/// ceiling, one row group per chunk, three retry attempts, and one row group as
/// the retry floor. Decision-grade benchmark binaries override this through
/// their own TPC-H env-var surface.
pub fn tpch_native_optimizer_config() -> AlgeonNativeOptimizerConfig {
    AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::default())
}

/// Builds fallback-pressure planning items for all TPC-H queries.
///
/// SQL and physical-planning failures are encoded as per-query rows so report
/// consumers can distinguish planning failures from native shape rejection.
pub async fn tpch_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> WorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
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
) -> WorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
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
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    tpch_datafusion_planning_workload_items_with_optimizer_config(root, session_config, tpch_native_optimizer_config())
        .await
}

pub async fn tpch_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCH_QUERIES.len());
    for query in TPCH_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => {
                let (state, logical) = df.into_parts();
                match plan_with_native_diagnostics(&state, &logical).await {
                    Ok((plan, diagnostics)) => NativeDataFusionPlanningWorkloadItem::new(query.id, plan, diagnostics),
                    Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                        query.id,
                        "physical_planning",
                        "physical_planning_failed",
                        error.to_string(),
                    ),
                }
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
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    tpch_datafusion_execution_workload_items_with_optimizer_config(root, session_config, tpch_native_optimizer_config())
        .await
}

pub async fn tpch_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let backend = build_report_gpu_backend(optimizer_config, GpuDeviceMemoryLimit::default())?;
    let items: WorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> = async {
        let builder = SessionStateBuilder::new_with_default_features().with_config(session_config);
        let builder = backend.install_on(builder).map_err(|error| DataFusionError::External(Box::new(error)))?;
        let ctx = SessionContext::new_with_state(builder.build());
        register_tpch_tables(&ctx, root).await?;

        let mut items = Vec::with_capacity(TPCH_QUERIES.len());
        for query in TPCH_QUERIES {
            let item = match ctx.sql(query.sql).await {
                Ok(df) => {
                    let (state, logical) = df.into_parts();
                    match plan_with_native_diagnostics(&state, &logical).await {
                        Ok((plan, diagnostics)) => {
                            let runtime = if diagnostics.report.selected_count() > 0 {
                                execute_tpch_native_plan(Arc::clone(&plan)).await
                            } else {
                                NativeDataFusionRuntime::NotStarted
                            };
                            NativeDataFusionExecutionWorkloadItem::new(query.id, plan, diagnostics, runtime)
                        },
                        Err(error) => NativeDataFusionExecutionWorkloadItem::planning_failure(
                            query.id,
                            "physical_planning",
                            "physical_planning_failed",
                            error.to_string(),
                        ),
                    }
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
    .await;
    match items {
        Ok(items) => {
            close_report_gpu_backend(backend)?;
            Ok(items)
        },
        Err(error) => {
            let _ = close_report_gpu_backend(backend);
            Err(error)
        },
    }
}

async fn execute_tpch_native_plan(plan: Arc<dyn ExecutionPlan>) -> NativeDataFusionRuntime {
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let output_batches = batches.len();
            let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
            let output_bytes = batches.iter().map(get_record_batch_memory_size).sum();
            let metrics = below_native_admission(&plan).metrics().unwrap_or_default();
            NativeDataFusionRuntime::Succeeded { output_batches, output_rows, output_bytes, metrics }
        },
        Err(error) => NativeDataFusionRuntime::Failed {
            message: error.to_string(),
            metrics: below_native_admission(&plan).metrics(),
        },
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
