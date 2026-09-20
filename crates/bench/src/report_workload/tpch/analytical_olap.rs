use std::path::Path;

use algeon_datafusion::{
    native_report::{NativeDataFusionPlanningWorkloadItem, NativeFallbackPressureWorkloadItem},
    planner::{AlgeonNativeOptimizerConfig, plan_with_native_diagnostics},
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use super::{WorkloadResult, register_tpch_tables, tpch_native_optimizer_config};

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
