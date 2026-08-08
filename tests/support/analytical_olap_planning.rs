use std::path::Path;

use datafusion::prelude::SessionConfig;
use datafusion_nexus::{native_report::NativeDataFusionPlanningWorkloadItem, planner::NexusNativeOptimizerConfig};
use nexus_workloads::tpch::WorkloadResult;

use super::tpch_report::{datafusion_planning_workload_items, tpch_native_optimizer_config};
use crate::analytical_olap_workload::ANALYTICAL_OLAP_QUERIES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnalyticalOlapCurrentStrictAdmission {
    Accepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnalyticalOlapStrictAdmissionExpectation {
    pub(crate) query_id: &'static str,
    pub(crate) admission: AnalyticalOlapCurrentStrictAdmission,
}

pub(crate) const ANALYTICAL_OLAP_STRICT_ADMISSION_EXPECTATIONS: &[AnalyticalOlapStrictAdmissionExpectation] = &[
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_01_customer_order_lineitem_star",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_02_part_supplier_supply_star",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_03_region_nation_customer_orders",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_04_supplier_same_nation_self_join",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_05_customer_order_self_join",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_06_customer_in_join_aggregate_subquery",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_07_nested_exists_customer_orders_lineitem",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_08_order_revenue_reused_cte",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_09_nation_supplier_lineitem_topk",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
    AnalyticalOlapStrictAdmissionExpectation {
        query_id: "adhoc_olap_10_part_lineitem_supplier_topk",
        admission: AnalyticalOlapCurrentStrictAdmission::Accepted,
    },
];

pub(crate) fn analytical_olap_strict_admission(query_id: &str) -> Option<AnalyticalOlapCurrentStrictAdmission> {
    ANALYTICAL_OLAP_STRICT_ADMISSION_EXPECTATIONS
        .iter()
        .find(|expectation| expectation.query_id == query_id)
        .map(|expectation| expectation.admission)
}

pub(crate) async fn analytical_olap_datafusion_planning_workload_items(
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

pub(crate) async fn analytical_olap_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: NexusNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    datafusion_planning_workload_items(
        root,
        session_config,
        optimizer_config,
        ANALYTICAL_OLAP_QUERIES.iter().map(|query| (query.id, query.sql)),
    )
    .await
}
