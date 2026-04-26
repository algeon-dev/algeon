/// ClickBench adapter-specific report item builders.
pub mod clickbench;
/// TPC-DS adapter-specific report and parity item builders.
pub mod tpcds;
/// TPC-H adapter-specific report item builders.
pub mod tpch;

use datafusion_nexus::planner::{FinalPlanRequirement, NexusNativeOptimizerConfig};

/// Session-level selection for relational benchmark execution.
///
/// A CPU baseline deliberately does not install the relational native rule.
/// Native selections install the rule, optionally with a requirement checked
/// against the completed physical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalExecutionSelection {
    CpuBaseline,
    Native { final_plan_requirement: Option<FinalPlanRequirement> },
}

impl RelationalExecutionSelection {
    pub const fn native() -> Self {
        Self::Native { final_plan_requirement: None }
    }

    pub const fn strict_native() -> Self {
        Self::Native { final_plan_requirement: Some(FinalPlanRequirement::NoDataFusionCpu) }
    }

    pub const fn installs_native_rule(self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub const fn final_plan_requirement(self) -> Option<FinalPlanRequirement> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { final_plan_requirement } => final_plan_requirement,
        }
    }

    /// Applies this selection to a native optimizer configuration.
    ///
    /// `None` is intentional for CPU baselines: callers must construct a
    /// DataFusion session without a relational native rule in that case.
    pub fn configure_native_optimizer(self, config: NexusNativeOptimizerConfig) -> Option<NexusNativeOptimizerConfig> {
        match self {
            Self::CpuBaseline => None,
            Self::Native { final_plan_requirement } => Some(match final_plan_requirement {
                Some(requirement) => config.with_final_plan_requirement(requirement),
                None => config,
            }),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuBaseline => "cpu_baseline",
            Self::Native { final_plan_requirement: None } => "native",
            Self::Native { final_plan_requirement: Some(FinalPlanRequirement::NoDataFusionCpu) } => {
                "native_no_datafusion_cpu"
            },
            Self::Native { final_plan_requirement: Some(_) } => "native_with_final_requirement",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_native_selection_installs_the_no_datafusion_cpu_requirement() {
        let config = RelationalExecutionSelection::strict_native()
            .configure_native_optimizer(NexusNativeOptimizerConfig::default())
            .expect("strict native selection installs a native rule");

        assert_eq!(config.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
    }

    #[test]
    fn cpu_baseline_selection_does_not_produce_a_native_optimizer_config() {
        assert!(
            RelationalExecutionSelection::CpuBaseline
                .configure_native_optimizer(NexusNativeOptimizerConfig::default())
                .is_none()
        );
    }
}
