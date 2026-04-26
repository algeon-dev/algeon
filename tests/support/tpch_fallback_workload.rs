use std::path::Path;

use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_nexus::native_report::NativeFallbackPressureWorkloadItem;
use nexus_workloads::tpch::{TPCH_QUERIES, TpchWorkloadResult, register_tpch_tables};

pub(crate) async fn tpch_fallback_pressure_workload_items(
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
