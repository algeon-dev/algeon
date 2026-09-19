/// ClickBench adapter-specific report item builders.
pub mod clickbench;
/// TPC-DS adapter-specific report and parity item builders.
pub mod tpcds;
/// TPC-H adapter-specific report item builders.
pub mod tpch;

use algeon_datafusion::planner::{AlgeonNativeOptimizerConfig, DataFusionCpuExecutionPolicy};

/// Session-level selection for relational benchmark execution.
///
/// A CPU baseline deliberately does not install the relational native rule.
/// Native selections install the rule with the canonical completed-plan
/// DataFusion CPU policy checked against the completed physical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalExecutionSelection {
    CpuBaseline,
    Native { datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy },
}

impl RelationalExecutionSelection {
    pub const fn native() -> Self {
        Self::Native { datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy::Allow }
    }

    pub const fn strict_native() -> Self {
        Self::Native { datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy::Reject }
    }

    pub const fn installs_native_rule(self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub const fn datafusion_cpu_execution_policy(self) -> Option<DataFusionCpuExecutionPolicy> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { datafusion_cpu_execution_policy } => Some(datafusion_cpu_execution_policy),
        }
    }

    /// Applies this selection to a native optimizer configuration.
    ///
    /// `None` is intentional for CPU baselines: callers must construct a
    /// DataFusion session without a relational native rule in that case.
    pub fn configure_native_optimizer(
        self,
        config: AlgeonNativeOptimizerConfig,
    ) -> Option<AlgeonNativeOptimizerConfig> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { datafusion_cpu_execution_policy } => {
                Some(config.with_datafusion_cpu_execution_policy(datafusion_cpu_execution_policy))
            },
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuBaseline => "cpu_baseline",
            Self::Native { datafusion_cpu_execution_policy } => match datafusion_cpu_execution_policy {
                DataFusionCpuExecutionPolicy::Allow => "native",
                _ => "native_no_datafusion_cpu",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_native_selection_installs_the_reject_cpu_policy() {
        let config = RelationalExecutionSelection::strict_native()
            .configure_native_optimizer(AlgeonNativeOptimizerConfig::default())
            .expect("strict native selection installs a native rule");

        assert_eq!(config.datafusion_cpu_execution_policy(), DataFusionCpuExecutionPolicy::Reject);
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
