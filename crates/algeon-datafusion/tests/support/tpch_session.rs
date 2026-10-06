use algeon_datafusion::{planner::AlgeonGpuExecutionMode, session::apply_algeon_option_overrides};
use datafusion::prelude::SessionConfig;

/// Stable DataFusion shape used by TPC-H adapter integration tests.
pub(crate) fn tpch_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    // Plain DataFusion contexts built from this config model the plans an
    // Algeon session lowers, so they take the session's option overrides.
    apply_algeon_option_overrides(config.options_mut(), AlgeonGpuExecutionMode::NativePreferred);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}
