use std::{path::Path, sync::Arc};

use algeon_datafusion::{
    native_exec::AlgeonNativeExecutionConfig,
    native_report::{
        NativeDataFusionPlanningWorkloadItem, NativeFallbackPressureWorkloadItem, WorkloadDatasetMetadata,
    },
    planner::{AlgeonNativeOptimizerConfig, plan_with_native_diagnostics},
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{
    common::ScalarValue,
    error::DataFusionError,
    execution::SessionStateBuilder,
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
        filter::FilterExec,
    },
    prelude::{SessionConfig, SessionContext},
};
use workloads::tpch::{TPCH_QUERIES, TpchDatasetMetadata, WorkloadResult, register_tpch_tables};

pub(crate) trait TestReportDatasetMetadataExt {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata;
}

impl TestReportDatasetMetadataExt for TpchDatasetMetadata {
    fn to_workload_dataset_metadata(&self) -> WorkloadDatasetMetadata {
        let metadata = self.to_neutral_metadata();
        WorkloadDatasetMetadata {
            workload: metadata.workload,
            root: metadata.root,
            scale_factor: metadata.scale_factor,
            provenance: metadata.provenance,
            generation_command: metadata.generation_command,
            table_count: metadata.table_count,
            file_count: metadata.file_count,
            total_rows: metadata.total_rows,
            total_bytes: metadata.total_bytes,
            metadata_fingerprint: metadata.metadata_fingerprint,
            content_fingerprint: metadata.content_fingerprint,
            table_file_counts: metadata.table_file_counts,
            table_row_counts: metadata.table_row_counts,
            table_byte_sizes: metadata.table_byte_sizes,
            table_row_group_counts: metadata.table_row_group_counts,
        }
    }
}

pub(crate) fn tpch_native_optimizer_config() -> AlgeonNativeOptimizerConfig {
    AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::default())
}

pub(crate) async fn tpch_scalar_fallback_pressure_workload_items(
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

pub(crate) async fn tpch_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    tpch_datafusion_planning_workload_items_with_optimizer_config(root, session_config, tpch_native_optimizer_config())
        .await
}

pub(crate) async fn tpch_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    datafusion_planning_workload_items(
        root,
        session_config,
        optimizer_config,
        TPCH_QUERIES.iter().map(|query| (query.id, query.sql)),
    )
    .await
}

pub(crate) async fn datafusion_planning_workload_items<'a>(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
    queries: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpch_tables(&ctx, root).await?;

    let mut items = Vec::new();
    for (query_id, sql) in queries {
        let item = match ctx.sql(sql).await {
            Ok(df) => {
                let (state, logical) = df.into_parts();
                match plan_with_native_diagnostics(&state, &logical).await {
                    Ok((plan, diagnostics)) => NativeDataFusionPlanningWorkloadItem::new(query_id, plan, diagnostics),
                    Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                        query_id,
                        "physical_planning",
                        "physical_planning_failed",
                        error.to_string(),
                    ),
                }
            },
            Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                query_id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
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
