/// ClickBench adapter-specific report item builders.
pub mod clickbench;
/// TPC-DS adapter-specific report and parity item builders.
pub mod tpcds;
/// TPC-H adapter-specific report item builders.
pub mod tpch;

use algeon_datafusion::{
    native_report::{NativeDataFusionPlanningWorkloadItem, NativeFallbackPressureWorkloadItem},
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, plan_with_native_diagnostics},
};
use datafusion::prelude::SessionContext;

async fn fallback_pressure_workload_item(
    ctx: &SessionContext,
    query_id: &str,
    sql: &str,
) -> NativeFallbackPressureWorkloadItem {
    match ctx.sql(sql).await {
        Ok(df) => match df.create_physical_plan().await {
            Ok(plan) => NativeFallbackPressureWorkloadItem::new(query_id, plan),
            Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                query_id,
                "physical_planning",
                "physical_planning_failed",
                error.to_string(),
            ),
        },
        Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
            query_id,
            "sql_to_logical_plan",
            "sql_planning_failed",
            error.to_string(),
        ),
    }
}

async fn datafusion_planning_workload_item(
    ctx: &SessionContext,
    query_id: &str,
    sql: &str,
) -> NativeDataFusionPlanningWorkloadItem {
    match ctx.sql(sql).await {
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
    }
}

/// Session-level selection for relational benchmark execution.
///
/// A CPU baseline deliberately does not install the relational native rule.
/// Native selections install the rule under an execution mode; the strict one
/// rejects any DataFusion CPU operator in the completed physical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalExecutionSelection {
    CpuBaseline,
    Native { execution_mode: AlgeonGpuExecutionMode },
}

impl RelationalExecutionSelection {
    pub const fn native() -> Self {
        Self::Native { execution_mode: AlgeonGpuExecutionMode::NativePreferred }
    }

    pub const fn strict_native() -> Self {
        Self::Native { execution_mode: AlgeonGpuExecutionMode::NativeRequired }
    }

    pub const fn installs_native_rule(self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub const fn execution_mode(self) -> Option<AlgeonGpuExecutionMode> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { execution_mode } => Some(execution_mode),
        }
    }

    /// Applies this selection to a native optimizer configuration.
    ///
    /// `None` is intentional for CPU baselines. Callers must construct a
    /// DataFusion session without a relational native rule in that case.
    pub fn configure_native_optimizer(
        self,
        config: AlgeonNativeOptimizerConfig,
    ) -> Option<AlgeonNativeOptimizerConfig> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { execution_mode } => Some(config.with_execution_mode(execution_mode)),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuBaseline => "cpu_baseline",
            Self::Native { execution_mode } if execution_mode.rejects_datafusion_cpu() => "native_no_datafusion_cpu",
            Self::Native { .. } => "native",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_native_selection_installs_native_required_mode() {
        let config = RelationalExecutionSelection::strict_native()
            .configure_native_optimizer(AlgeonNativeOptimizerConfig::default())
            .expect("strict native selection installs a native rule");

        assert_eq!(config.execution_mode(), AlgeonGpuExecutionMode::NativeRequired);
    }

    #[test]
    fn cpu_baseline_selection_does_not_produce_a_native_optimizer_config() {
        assert!(
            RelationalExecutionSelection::CpuBaseline
                .configure_native_optimizer(AlgeonNativeOptimizerConfig::default())
                .is_none()
        );
    }
}
