//! DataFusion physical optimizer integration for native cuDF execution.
//!
//! The planner exposes one public optimizer rule, but session integration
//! inserts it in two optimizer stages: a count-only probe before DataFusion aggregate
//! statistics and the final native rewrite immediately before final plan
//! validation. Native candidate misses preserve the DataFusion baseline plan.
//! Final-plan requirements, when configured, inspect that exact resulting plan
//! after native rewrite; they never trigger a second plan.
//!
//! [`FinalPlanDisposition`] describes one completed executable plan only
//! (`DataFusion`, `Native`, or `Mixed`); planning failures have no disposition.
//! [`FinalPlanCoverage`] carries the single physical-plan walk behind that
//! verdict, and [`classify_final_plan`] reads the same coverage instead of
//! reclassifying. Candidate decisions stay per-candidate under
//! [`crate::report::CandidateOutcome`].

use datafusion::physical_plan::ExecutionPlan;
pub use nexus_query_engine::{
    contract_report::{CapabilityContractGateRow, CapabilityContractReport},
    mode::StatisticalAggregateExecutionMode,
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions},
    plan::NativePlanCostOptions,
    source::SourceDiagnostics,
};

use crate::native_exec::NexusNativeExecutionConfig;

pub(crate) mod native;

pub use native::{
    CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON, NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME,
    NativeCapabilityAcceptedCandidateDiagnostics, NativeCapabilityCompiledGraphDiagnostics,
    NativeCapabilityContractDiagnostics, NativeCapabilityDeferredCandidateDiagnostics,
    NativeCapabilityRejectedCandidateDiagnostics, NativePlanningDiagnostics, NexusFinalPlanRequirementRule,
    NexusNativePhysicalOptimizerRule,
};

pub use crate::final_plan_observation::FinalPlanCoverage;

/// The execution ownership observed in a completed physical plan.
///
/// [`Self::DataFusion`], [`Self::Native`], and [`Self::Mixed`] are returned by
/// [`classify_final_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FinalPlanDisposition {
    /// The final plan contains only DataFusion execution.
    DataFusion,
    /// The final plan executes entirely through a recognized native runtime node.
    Native,
    /// The final plan combines native runtime nodes with DataFusion execution.
    Mixed,
}

impl FinalPlanDisposition {
    /// Stable label for diagnostics and typed requirement errors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DataFusion => "datafusion",
            Self::Native => "native",
            Self::Mixed => "mixed",
        }
    }
}

impl std::fmt::Display for FinalPlanDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A non-default constraint applied to the completed physical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FinalPlanRequirement {
    /// Reject a final plan that executes any DataFusion CPU operator.
    NoDataFusionCpu,
}

impl FinalPlanRequirement {
    /// Stable label used in structured requirement errors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoDataFusionCpu => "no_datafusion_cpu",
        }
    }
}

impl std::fmt::Display for FinalPlanRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classify the exact completed physical plan without invoking planning.
///
/// The classifier intentionally recognizes only runtime nodes whose execution
/// ownership is known at this adapter boundary. Unknown wrappers or operators
/// around native fragments are conservative [`FinalPlanDisposition::Mixed`]
/// plans rather than being treated as native-only.
pub fn classify_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    crate::final_plan_observation::observe_final_plan(plan).disposition()
}

/// DataFusion-facing native optimizer configuration.
///
/// Defaults preserve the DataFusion baseline when native candidates are not
/// accepted. Native execution uses a bounded planning template, query-engine
/// optimizer options use their own defaults, and plan-cost selection is
/// disabled unless a workload opts in. The native
/// [`ExecutionContext`](nexus_query_engine::runtime::ExecutionContext) is
/// acquired lazily from backend admission through [`NexusNativeExecutionConfig`].
#[derive(Debug, Clone)]
pub struct NexusNativeOptimizerConfig {
    native_execution_config: NexusNativeExecutionConfig,
    native_optimizer_options: NativeOptimizerOptions,
    native_plan_cost_options: NativePlanCostOptions,
    final_plan_requirements: Vec<FinalPlanRequirement>,
}

impl Default for NexusNativeOptimizerConfig {
    fn default() -> Self {
        Self {
            native_execution_config: NexusNativeExecutionConfig::bounded_default(),
            native_optimizer_options: NativeOptimizerOptions::default(),
            native_plan_cost_options: NativePlanCostOptions::disabled(),
            final_plan_requirements: Vec::new(),
        }
    }
}

impl NexusNativeOptimizerConfig {
    /// Native execution settings copied into rewritten execution nodes.
    pub fn native_execution_config(&self) -> &NexusNativeExecutionConfig {
        &self.native_execution_config
    }

    /// Query-engine optimizer options applied after lowering.
    pub fn native_optimizer_options(&self) -> &NativeOptimizerOptions {
        &self.native_optimizer_options
    }

    /// Native plan cost-selection rules applied after native optimizer rewrites.
    ///
    /// The adapter default is disabled to preserve generic optimizer rewrite
    /// behavior; workload configs can opt in when their benchmark evidence
    /// shows native fixed overhead dominates small selective shapes.
    pub fn native_plan_cost_options(&self) -> &NativePlanCostOptions {
        &self.native_plan_cost_options
    }

    /// Completed-plan constraints enforced after native rewriting, without
    /// planning a second time.
    pub fn final_plan_requirements(&self) -> &[FinalPlanRequirement] {
        &self.final_plan_requirements
    }

    /// Add a completed-plan requirement. Repeating a requirement is idempotent.
    pub fn with_final_plan_requirement(mut self, requirement: FinalPlanRequirement) -> Self {
        if !self.final_plan_requirements.contains(&requirement) {
            self.final_plan_requirements.push(requirement);
        }
        self
    }

    /// Require the exact completed physical plan to contain no DataFusion CPU execution.
    pub fn with_no_datafusion_cpu_requirement(self) -> Self {
        self.with_final_plan_requirement(FinalPlanRequirement::NoDataFusionCpu)
    }

    /// Floating statistical aggregate execution contract used by native execution.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.native_execution_config.statistical_aggregate_execution_mode()
    }

    /// Replace the native execution config copied into accepted native plans.
    pub fn with_native_execution_config(mut self, execution_config: NexusNativeExecutionConfig) -> Self {
        self.native_execution_config = execution_config;
        self
    }

    /// Attach a query id to the native execution config only when the caller has not supplied one.
    pub(crate) fn with_query_id_if_absent(mut self, query_id: impl Into<nexus_query_engine::metrics::QueryId>) -> Self {
        self.native_execution_config = self.native_execution_config.with_query_id_if_absent(query_id);
        self
    }

    /// Replace DataFusion-free native optimizer options.
    pub fn with_native_optimizer_options(mut self, options: NativeOptimizerOptions) -> Self {
        self.native_optimizer_options = options;
        self
    }

    /// Replace native plan cost-selection options.
    pub fn with_native_plan_cost_options(mut self, options: NativePlanCostOptions) -> Self {
        self.native_plan_cost_options = options;
        self
    }

    /// Set the floating statistical aggregate contract without rebuilding the
    /// rest of the config.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.native_execution_config =
            self.native_execution_config.clone().with_statistical_aggregate_execution_mode(mode);
        self
    }
}
