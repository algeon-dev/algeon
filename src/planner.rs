//! DataFusion physical optimizer integration for native cuDF execution.
//!
//! The planner exposes one public optimizer rule, but session integration
//! inserts it in two optimizer stages: a count-only probe before DataFusion aggregate
//! statistics and the final native rewrite immediately before final plan
//! validation. Native candidate misses preserve the DataFusion baseline plan.
//! Final-plan requirements, when configured, inspect that exact resulting plan
//! after native rewrite; they never trigger a second plan.

use std::{num::NonZeroUsize, time::Duration};

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
    physical_plan::ExecutionPlan,
};

use crate::{
    native_exec::{HostExecutionQosLimiter, NexusNativeExecutionConfig},
    session_config::optimizer_config_with_datafusion_options,
};

mod native;

pub use native::{
    NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NEXUS_HOST_EXECUTION_QOS_RULE_NAME,
    NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics, NativePlanningDiagnostics,
    NativeStrictBoundedAcceptedCandidateDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, NativeStrictBoundedRejectedCandidateDiagnostics,
    NexusFinalPlanRequirementRule, NexusHostExecutionQosRule, NexusNativePhysicalOptimizerRule,
    STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};

/// The execution ownership observed in a completed physical plan.
///
/// [`Self::DataFusion`], [`Self::Native`], and [`Self::Mixed`] are returned by
/// [`classify_final_plan`]. [`Self::Rejected`] records the terminal outcome
/// when a final-plan requirement rejects an otherwise completed plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FinalPlanDisposition {
    /// The final plan contains only DataFusion execution.
    DataFusion,
    /// The final plan executes entirely through a recognized native runtime node.
    Native,
    /// The final plan combines native runtime nodes with DataFusion execution.
    Mixed,
    /// A final-plan requirement rejected the completed planning result.
    Rejected,
}

impl FinalPlanDisposition {
    /// Stable label for diagnostics and typed requirement errors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DataFusion => "datafusion",
            Self::Native => "native",
            Self::Mixed => "mixed",
            Self::Rejected => "rejected",
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

/// Host-side execution QoS applied once to a completed DataFusion or mixed
/// physical plan.
///
/// This configuration is intentionally independent of
/// [`NexusNativeOptimizerConfig`]. CPU-baseline and cuGraph-only sessions use
/// the same outer statement bound without installing relational native
/// selection. Clones share the limiter instance and therefore one bound.
#[derive(Debug, Clone)]
pub struct HostExecutionQosConfig {
    limiter: HostExecutionQosLimiter,
}

impl HostExecutionQosConfig {
    /// Create a host statement bound with indefinite wait time.
    pub fn new(max_concurrent_statements: NonZeroUsize) -> Self {
        Self { limiter: HostExecutionQosLimiter::new_with_wait_timeout(max_concurrent_statements, None) }
    }

    /// Set the maximum time a statement waits for a host execution permit.
    ///
    /// A zero duration restores indefinite waiting. This is independent of
    /// GPU query-service admission timeouts.
    pub fn with_wait_timeout(self, wait_timeout: Duration) -> Self {
        Self {
            limiter: HostExecutionQosLimiter::new_with_wait_timeout(
                self.max_concurrent_statements(),
                (!wait_timeout.is_zero()).then_some(wait_timeout),
            ),
        }
    }

    /// Maximum number of host-executing statements allowed concurrently.
    pub fn max_concurrent_statements(&self) -> NonZeroUsize {
        self.limiter.max_concurrent_statements()
    }

    /// Maximum host-permit wait time, or `None` for an indefinite wait.
    pub fn wait_timeout(&self) -> Option<Duration> {
        self.limiter.wait_timeout()
    }

    pub(crate) fn limiter(&self) -> HostExecutionQosLimiter {
        self.limiter.clone()
    }
}

/// Classify the exact completed physical plan without invoking planning.
///
/// The classifier intentionally recognizes only runtime nodes whose execution
/// ownership is known at this adapter boundary. Unknown wrappers or operators
/// around native fragments are conservative [`FinalPlanDisposition::Mixed`]
/// plans rather than being treated as native-only.
pub fn classify_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    crate::final_plan_observation::observe_final_plan(plan).disposition
}

/// DataFusion-facing native optimizer configuration.
///
/// Defaults preserve the DataFusion baseline when native candidates are not
/// accepted. Native execution uses an auto-sized bounded profile, query-engine
/// optimizer options use their own defaults, and plan-cost selection is
/// disabled unless a workload opts in. The native
/// [`ExecutionContext`](nexus_query_engine::runtime::ExecutionContext) is still built
/// lazily at execution time through [`NexusNativeExecutionConfig`], so allocator
/// and memory-control validation fail at the DataFusion/native boundary rather
/// than during rule construction.
#[derive(Debug, Clone)]
pub struct NexusNativeOptimizerConfig {
    native_execution_config: NexusNativeExecutionConfig,
    native_optimizer_options: nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_plan_cost_options: nexus_query_engine::plan::NativePlanCostOptions,
    final_plan_requirements: Vec<FinalPlanRequirement>,
}

impl Default for NexusNativeOptimizerConfig {
    fn default() -> Self {
        Self {
            native_execution_config: NexusNativeExecutionConfig::bounded_auto_sized(),
            native_optimizer_options: nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
            native_plan_cost_options: nexus_query_engine::plan::NativePlanCostOptions::disabled(),
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
    pub fn native_optimizer_options(&self) -> &nexus_query_engine::optimizer::NativeOptimizerOptions {
        &self.native_optimizer_options
    }

    /// Native plan cost-selection rules applied after native optimizer rewrites.
    ///
    /// The adapter default is disabled to preserve generic optimizer rewrite
    /// behavior; workload configs can opt in when their benchmark evidence
    /// shows native fixed overhead dominates small selective shapes.
    pub fn native_plan_cost_options(&self) -> &nexus_query_engine::plan::NativePlanCostOptions {
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

    /// Strict bounded or adaptive pipeline mode used by native execution.
    pub fn native_pipeline_execution_mode(&self) -> nexus_query_engine::mode::NativePipelineExecutionMode {
        self.native_execution_config.native_pipeline_execution_mode()
    }

    /// Floating statistical aggregate execution contract used by native execution.
    pub fn statistical_aggregate_execution_mode(&self) -> nexus_query_engine::mode::StatisticalAggregateExecutionMode {
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
    pub fn with_native_optimizer_options(
        mut self,
        options: nexus_query_engine::optimizer::NativeOptimizerOptions,
    ) -> Self {
        self.native_optimizer_options = options;
        self
    }

    /// Replace native plan cost-selection options.
    pub fn with_native_plan_cost_options(mut self, options: nexus_query_engine::plan::NativePlanCostOptions) -> Self {
        self.native_plan_cost_options = options;
        self
    }

    #[cfg(feature = "server")]
    pub(crate) fn with_unbounded_native_execution_limiter(
        mut self,
        limiter: crate::native_exec::UnboundedNativeExecutionLimiter,
    ) -> Self {
        self.native_execution_config =
            self.native_execution_config.clone().with_unbounded_native_execution_limiter(limiter);
        self
    }

    /// Set strict bounded or adaptive execution without rebuilding the rest of the config.
    pub fn with_native_pipeline_execution_mode(
        mut self,
        mode: nexus_query_engine::mode::NativePipelineExecutionMode,
    ) -> Self {
        self.native_execution_config = self.native_execution_config.clone().with_native_pipeline_execution_mode(mode);
        self
    }

    /// Set the floating statistical aggregate contract without rebuilding the
    /// rest of the config.
    pub fn with_statistical_aggregate_execution_mode(
        mut self,
        mode: nexus_query_engine::mode::StatisticalAggregateExecutionMode,
    ) -> Self {
        self.native_execution_config =
            self.native_execution_config.clone().with_statistical_aggregate_execution_mode(mode);
        self
    }

    /// Use one caller-provided cudf runtime for native execution produced by this optimizer.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.native_execution_config = self.native_execution_config.clone().with_runtime(runtime);
        self
    }

    /// Apply registered `datafusion_nexus.native.*` DataFusion session options
    /// to this optimizer config.
    pub fn with_datafusion_config_options(&self, options: &ConfigOptions) -> DataFusionResult<Self> {
        optimizer_config_with_datafusion_options(self, options)
    }
}
