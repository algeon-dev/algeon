use std::path::Path;

use algeon_datafusion::{native_report::NativeDataFusionPlanningWorkloadItem, planner::AlgeonNativeOptimizerConfig};
use datafusion::prelude::SessionConfig;
use workloads::tpch::WorkloadResult;

use super::tpch_report::{datafusion_planning_workload_items, tpch_native_optimizer_config};
use crate::analytical_olap_workload::ANALYTICAL_OLAP_QUERIES;

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
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> WorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    datafusion_planning_workload_items(
        root,
        session_config,
        optimizer_config,
        ANALYTICAL_OLAP_QUERIES.iter().map(|query| (query.id, query.sql)),
    )
    .await
}
