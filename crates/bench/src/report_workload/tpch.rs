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

use super::{datafusion_planning_workload_item, fallback_pressure_workload_item};
use crate::gpu_backend::{build_report_gpu_backend, close_report_gpu_backend};

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
        items.push(fallback_pressure_workload_item(&ctx, query.id, query.sql).await);
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
        items.push(fallback_pressure_workload_item(&ctx, query.id, query.sql).await);
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
        items.push(datafusion_planning_workload_item(&ctx, query.id, query.sql).await);
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
        id: "q07_scalar_extract_quarter",
        sql: "
SELECT COUNT(*) AS count_rows
FROM orders
WHERE EXTRACT(QUARTER FROM o_orderdate) = 1
",
    },
];

/// Number of curated analytical OLAP TPC-H-shaped queries.
pub const ANALYTICAL_OLAP_QUERY_COUNT: usize = ANALYTICAL_OLAP_QUERIES.len();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyticalOlapQuery {
    pub id: &'static str,
    pub sql: &'static str,
    pub workload_class: AnalyticalOlapWorkloadClass,
    pub capability_first_admission: AnalyticalOlapCapabilityFirstAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticalOlapWorkloadClass {
    StarSchemaJoin,
    SelfJoin,
    DecorrelatableSubquery,
    MultiJoinResidentPressure,
    AggregateSortTopK,
}

impl AnalyticalOlapWorkloadClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StarSchemaJoin => "star_schema_join",
            Self::SelfJoin => "self_join",
            Self::DecorrelatableSubquery => "decorrelatable_subquery",
            Self::MultiJoinResidentPressure => "multi_join_resident_pressure",
            Self::AggregateSortTopK => "aggregate_sort_topk",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticalOlapCapabilityFirstAdmission {
    Accepted,
    ExpectedSupersetWin,
}

impl AnalyticalOlapCapabilityFirstAdmission {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::ExpectedSupersetWin => "expected_superset_win",
        }
    }
}

/// Curated analytical OLAP query table in report order.
pub const ANALYTICAL_OLAP_QUERIES: &[AnalyticalOlapQuery] = &[
    AnalyticalOlapQuery {
        id: "adhoc_olap_01_customer_order_lineitem_star",
        workload_class: AnalyticalOlapWorkloadClass::StarSchemaJoin,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT c.c_mktsegment, o.o_orderstatus, SUM(l.l_extendedprice) AS revenue
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
GROUP BY c.c_mktsegment, o.o_orderstatus
ORDER BY revenue DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_02_part_supplier_supply_star",
        workload_class: AnalyticalOlapWorkloadClass::StarSchemaJoin,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT p.p_brand, s.s_nationkey, SUM(ps.ps_supplycost) AS supply_cost
FROM part p
JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
GROUP BY p.p_brand, s.s_nationkey
ORDER BY supply_cost ASC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_03_region_nation_customer_orders",
        workload_class: AnalyticalOlapWorkloadClass::StarSchemaJoin,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT r.r_name, n.n_name, COUNT(o.o_orderkey) AS order_count
FROM region r
JOIN nation n ON r.r_regionkey = n.n_regionkey
JOIN customer c ON n.n_nationkey = c.c_nationkey
JOIN orders o ON c.c_custkey = o.o_custkey
GROUP BY r.r_name, n.n_name
ORDER BY order_count DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_04_supplier_same_nation_self_join",
        workload_class: AnalyticalOlapWorkloadClass::SelfJoin,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT s1.s_nationkey, COUNT(*) AS supplier_pairs
FROM supplier s1
JOIN supplier s2 ON s1.s_nationkey = s2.s_nationkey
WHERE s1.s_suppkey < s2.s_suppkey
GROUP BY s1.s_nationkey
ORDER BY supplier_pairs DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_05_customer_order_self_join",
        workload_class: AnalyticalOlapWorkloadClass::SelfJoin,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT o1.o_custkey, COUNT(*) AS order_pairs
FROM orders o1
JOIN orders o2 ON o1.o_custkey = o2.o_custkey
WHERE o1.o_orderkey <> o2.o_orderkey
GROUP BY o1.o_custkey
ORDER BY order_pairs DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_06_customer_in_join_aggregate_subquery",
        workload_class: AnalyticalOlapWorkloadClass::DecorrelatableSubquery,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::ExpectedSupersetWin,
        sql: "
SELECT c.c_custkey, SUM(o.o_totalprice) AS customer_total
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
WHERE c.c_custkey IN (
    SELECT o2.o_custkey
    FROM orders o2
    JOIN lineitem l2 ON o2.o_orderkey = l2.l_orderkey
    GROUP BY o2.o_custkey
)
GROUP BY c.c_custkey
ORDER BY customer_total DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_07_nested_exists_customer_orders_lineitem",
        workload_class: AnalyticalOlapWorkloadClass::DecorrelatableSubquery,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT c.c_custkey, COUNT(o.o_orderkey) AS matched_orders
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
WHERE EXISTS (
    SELECT 1
    FROM lineitem l
    WHERE l.l_orderkey = o.o_orderkey
)
GROUP BY c.c_custkey
ORDER BY matched_orders DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_08_order_revenue_reused_cte",
        workload_class: AnalyticalOlapWorkloadClass::MultiJoinResidentPressure,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::ExpectedSupersetWin,
        sql: "
WITH order_revenue AS (
    SELECT o.o_orderkey, o.o_custkey, SUM(l.l_extendedprice) AS revenue
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY o.o_orderkey, o.o_custkey
)
SELECT high.o_custkey, SUM(high.revenue) AS high_revenue
FROM order_revenue high
JOIN order_revenue low ON high.o_custkey = low.o_custkey
WHERE high.o_orderkey >= low.o_orderkey
GROUP BY high.o_custkey
ORDER BY high_revenue DESC
LIMIT 5
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_09_nation_supplier_lineitem_topk",
        workload_class: AnalyticalOlapWorkloadClass::AggregateSortTopK,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT n.n_name, SUM(l.l_extendedprice) AS revenue
FROM nation n
JOIN supplier s ON n.n_nationkey = s.s_nationkey
JOIN lineitem l ON s.s_suppkey = l.l_suppkey
GROUP BY n.n_name
ORDER BY revenue DESC
LIMIT 3
",
    },
    AnalyticalOlapQuery {
        id: "adhoc_olap_10_part_lineitem_supplier_topk",
        workload_class: AnalyticalOlapWorkloadClass::AggregateSortTopK,
        capability_first_admission: AnalyticalOlapCapabilityFirstAdmission::Accepted,
        sql: "
SELECT p.p_type, s.s_nationkey, SUM(l.l_quantity) AS quantity
FROM part p
JOIN lineitem l ON p.p_partkey = l.l_partkey
JOIN supplier s ON l.l_suppkey = s.s_suppkey
GROUP BY p.p_type, s.s_nationkey
ORDER BY quantity DESC
LIMIT 5
",
    },
];

pub fn analytical_olap_query_ids() -> impl Iterator<Item = &'static str> {
    ANALYTICAL_OLAP_QUERIES.iter().map(|query| query.id)
}

pub async fn analytical_olap_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> WorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    register_tpch_tables(&ctx, root).await?;
    let mut items = Vec::with_capacity(ANALYTICAL_OLAP_QUERIES.len());
    for query in ANALYTICAL_OLAP_QUERIES {
        items.push(fallback_pressure_workload_item(&ctx, query.id, query.sql).await);
    }
    Ok(items)
}

pub async fn analytical_olap_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    analytical_olap_datafusion_planning_workload_items_with_optimizer_config(
        root,
        session_config,
        tpch_native_optimizer_config(),
    )
    .await
}

pub async fn analytical_olap_datafusion_planning_workload_items_with_optimizer_config(
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

    let mut items = Vec::with_capacity(ANALYTICAL_OLAP_QUERIES.len());
    for query in ANALYTICAL_OLAP_QUERIES {
        items.push(datafusion_planning_workload_item(&ctx, query.id, query.sql).await);
    }
    Ok(items)
}
