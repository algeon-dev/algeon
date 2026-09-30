//! DataFusion physical optimizer integration for native cuDF execution.
//!
//! The planner exposes one public optimizer rule, but session integration
//! inserts it in two optimizer stages: a count-only probe before DataFusion aggregate
//! statistics and the final native rewrite immediately before final plan
//! validation. Native candidate misses preserve the DataFusion baseline plan.
//! A reject DataFusion CPU policy, when configured, is the last step of the
//! final rewrite and inspects the plan that rewrite produced; it never
//! triggers a second plan.
//!
//! [`FinalPlanDisposition`] describes one completed executable plan only
//! (`DataFusion`, `Native`, or `Mixed`); planning failures have no disposition.
//! [`FinalPlanCoverage`] carries the single physical-plan walk behind that
//! verdict, and [`classify_final_plan`] reads the same coverage instead of
//! reclassifying. Candidate decisions stay per-candidate under
//! [`crate::report::CandidateOutcome`].

use datafusion::physical_plan::ExecutionPlan;
pub use query_engine::{
    contract_report::{CapabilityContractReport, CapabilityDecisionRow},
    mode::StatisticalAggregateExecutionMode,
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions},
    plan::NativePlanCostOptions,
    source::SourceDiagnostics,
};

use crate::{
    error::{Error, ErrorCode, Result},
    final_plan_observation::observe_final_plan,
    native_exec::AlgeonNativeExecutionConfig,
};

pub(crate) mod native;

pub use native::{
    AlgeonNativePhysicalOptimizerRule, CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
    NativeCapabilityAcceptedCandidateDiagnostics, NativeCapabilityCompiledGraphDiagnostics,
    NativeCapabilityContractDiagnostics, NativeCapabilityDeferredCandidateDiagnostics,
    NativeCapabilityRejectedCandidateDiagnostics, NativePlanningDiagnostics, plan_with_native_diagnostics,
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

#[cfg(test)]
mod execution_mode_tests {
    use super::*;

    #[test]
    fn execution_modes_map_to_one_canonical_policy_pair() {
        for mode in [
            AlgeonGpuExecutionMode::NativePreferred,
            AlgeonGpuExecutionMode::FunctionsOnly,
            AlgeonGpuExecutionMode::NativeRequired,
        ] {
            let config = AlgeonNativeOptimizerConfig::default().with_execution_mode(mode);
            assert_eq!(config.execution_mode().unwrap(), mode);
            config.validate().unwrap();
        }
    }

    #[test]
    fn disabled_relational_rewrite_cannot_reject_datafusion_cpu_execution() {
        let config = AlgeonNativeOptimizerConfig::default()
            .with_relational_native_rewrite_policy(RelationalNativeRewritePolicy::Disabled)
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);

        let error = config.validate().expect_err("contradictory policies must reject before planning");
        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("reason"), Some("relational_native_rewrite_disabled_with_cpu_reject"));
    }
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

/// Policy for DataFusion CPU execution in the completed physical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum DataFusionCpuExecutionPolicy {
    /// Preserve the DataFusion baseline when native candidates are not accepted.
    #[default]
    Allow,
    /// Reject a completed plan that executes any DataFusion CPU operator.
    Reject,
}

/// Whether ordinary relational DataFusion plans are candidates for native
/// lowering. Explicit GPU UDTF nodes remain eligible in both policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum RelationalNativeRewritePolicy {
    /// Run the existing native relational selection rules.
    #[default]
    Enabled,
    /// Preserve ordinary relational operators for DataFusion CPU execution.
    Disabled,
}

/// Product-level mapping of relational selection and final CPU policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum AlgeonGpuExecutionMode {
    /// Prefer native relational execution and allow a DataFusion CPU result.
    #[default]
    NativePreferred,
    /// Keep ordinary SQL in DataFusion and execute only explicit GPU UDTFs
    /// on the GPU.
    FunctionsOnly,
    /// Require the completed plan to contain no DataFusion CPU execution.
    NativeRequired,
}

/// Classify the exact completed physical plan without invoking planning.
///
/// The classifier intentionally recognizes only runtime nodes whose execution
/// ownership is known at this adapter boundary. Unknown wrappers or operators
/// around native fragments are conservative [`FinalPlanDisposition::Mixed`]
/// plans rather than being treated as native-only.
pub fn classify_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    observe_final_plan(plan).disposition()
}

/// DataFusion-facing native optimizer configuration.
///
/// Defaults preserve the DataFusion baseline when native candidates are not
/// accepted. Native execution uses a bounded planning template, query-engine
/// optimizer options use their own defaults, and plan-cost selection is
/// disabled unless a workload opts in. The native
/// [`ExecutionContext`](query_engine::runtime::ExecutionContext) is
/// acquired lazily from backend admission through [`AlgeonNativeExecutionConfig`].
#[derive(Debug, Clone)]
pub struct AlgeonNativeOptimizerConfig {
    native_execution_config: AlgeonNativeExecutionConfig,
    native_optimizer_options: NativeOptimizerOptions,
    native_plan_cost_options: NativePlanCostOptions,
    datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy,
    relational_native_rewrite_policy: RelationalNativeRewritePolicy,
}

impl Default for AlgeonNativeOptimizerConfig {
    fn default() -> Self {
        Self {
            native_execution_config: AlgeonNativeExecutionConfig::bounded_default(),
            native_optimizer_options: NativeOptimizerOptions::default(),
            native_plan_cost_options: NativePlanCostOptions::disabled(),
            datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy::Allow,
            relational_native_rewrite_policy: RelationalNativeRewritePolicy::Enabled,
        }
    }
}

impl AlgeonNativeOptimizerConfig {
    /// Native execution settings copied into rewritten execution nodes.
    pub fn native_execution_config(&self) -> &AlgeonNativeExecutionConfig {
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

    /// Completed-plan DataFusion CPU policy enforced after native rewriting,
    /// without planning a second time.
    pub fn datafusion_cpu_execution_policy(&self) -> DataFusionCpuExecutionPolicy {
        self.datafusion_cpu_execution_policy
    }

    /// Set the completed-plan DataFusion CPU policy.
    pub fn with_datafusion_cpu_execution_policy(mut self, policy: DataFusionCpuExecutionPolicy) -> Self {
        self.datafusion_cpu_execution_policy = policy;
        self
    }

    /// Return whether ordinary relational SQL participates in native lowering.
    pub fn relational_native_rewrite_policy(&self) -> RelationalNativeRewritePolicy {
        self.relational_native_rewrite_policy
    }
    /// Set whether ordinary relational SQL participates in native lowering.
    pub fn with_relational_native_rewrite_policy(mut self, policy: RelationalNativeRewritePolicy) -> Self {
        self.relational_native_rewrite_policy = policy;
        self
    }
    /// Return the canonical product mode represented by this configuration.
    ///
    /// A disabled relational rewrite combined with CPU rejection is invalid.
    pub fn execution_mode(&self) -> Result<AlgeonGpuExecutionMode> {
        match (self.relational_native_rewrite_policy, self.datafusion_cpu_execution_policy) {
            (RelationalNativeRewritePolicy::Enabled, DataFusionCpuExecutionPolicy::Allow) => {
                Ok(AlgeonGpuExecutionMode::NativePreferred)
            },
            (RelationalNativeRewritePolicy::Disabled, DataFusionCpuExecutionPolicy::Allow) => {
                Ok(AlgeonGpuExecutionMode::FunctionsOnly)
            },
            (RelationalNativeRewritePolicy::Enabled, DataFusionCpuExecutionPolicy::Reject) => {
                Ok(AlgeonGpuExecutionMode::NativeRequired)
            },
            (RelationalNativeRewritePolicy::Disabled, DataFusionCpuExecutionPolicy::Reject) => Err(Error::new(
                ErrorCode::InvalidRequest,
                "algeon_datafusion.planner.config",
                "relational native rewrite cannot be disabled when DataFusion CPU execution is rejected",
            )
            .with_fact("reason", "relational_native_rewrite_disabled_with_cpu_reject")),
        }
    }
    /// Apply one canonical product mode without storing parallel mode state.
    pub fn with_execution_mode(self, mode: AlgeonGpuExecutionMode) -> Self {
        match mode {
            AlgeonGpuExecutionMode::NativePreferred => self
                .with_relational_native_rewrite_policy(RelationalNativeRewritePolicy::Enabled)
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Allow),
            AlgeonGpuExecutionMode::FunctionsOnly => self
                .with_relational_native_rewrite_policy(RelationalNativeRewritePolicy::Disabled)
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Allow),
            AlgeonGpuExecutionMode::NativeRequired => self
                .with_relational_native_rewrite_policy(RelationalNativeRewritePolicy::Enabled)
                .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
        }
    }
    /// Reject contradictory optimizer policies before planning begins.
    pub fn validate(&self) -> Result<()> {
        self.execution_mode().map(|_| ())
    }

    /// Floating statistical aggregate execution contract used by native execution.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.native_execution_config.statistical_aggregate_execution_mode()
    }

    /// Replace the native execution config copied into accepted native plans.
    pub fn with_native_execution_config(mut self, execution_config: AlgeonNativeExecutionConfig) -> Self {
        self.native_execution_config = execution_config;
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
    ///
    /// Defaults to [`StatisticalAggregateExecutionMode::GpuNativeTolerant`],
    /// whose results can differ from DataFusion in floating-point bits. Select
    /// [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`] when
    /// results must match DataFusion exactly.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.native_execution_config =
            self.native_execution_config.clone().with_statistical_aggregate_execution_mode(mode);
        self
    }
}
