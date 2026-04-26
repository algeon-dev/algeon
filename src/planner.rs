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
    native_exec::{CpuFallbackStatementLimiter, NexusNativeExecutionConfig},
    session_config::optimizer_config_with_datafusion_options,
};

mod native;

pub use native::{
    NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NativeAdaptivePipelineCandidateDiagnostics,
    NativeAdaptivePipelineDiagnostics, NativePlanningDiagnostics, NativeStrictBoundedAcceptedCandidateDiagnostics,
    NativeStrictBoundedCompiledGraphDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeStrictBoundedRejectedCandidateDiagnostics, NexusFinalPlanRequirementRule, NexusNativePhysicalOptimizerRule,
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

/// Classify the exact completed physical plan without invoking planning.
///
/// The classifier intentionally recognizes only runtime nodes whose execution
/// ownership is known at this adapter boundary. Unknown wrappers or operators
/// around native fragments are conservative [`FinalPlanDisposition::Mixed`]
/// plans rather than being treated as native-only.
pub fn classify_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    if plan.downcast_ref::<crate::native_exec::NexusNativeExec>().is_some()
        || plan.downcast_ref::<crate::native_exec::NexusNativeGpuProducerExec>().is_some()
    {
        return FinalPlanDisposition::Native;
    }
    if plan.downcast_ref::<crate::native_exec::NexusNativeComposedExec>().is_some() {
        return classify_native_composed_final_plan(plan);
    }
    if crate::native_exec::graph::graph_execution_plan(plan).is_some() {
        return classify_graph_final_plan(plan);
    }
    if is_transparent_native_runtime_wrapper(plan)
        && let [child] = plan.children().as_slice()
    {
        return classify_final_plan(child.as_ref());
    }

    let child_dispositions = plan.children();
    if child_dispositions.is_empty() {
        return FinalPlanDisposition::DataFusion;
    }
    if child_dispositions.iter().any(|child| classify_final_plan(child.as_ref()) != FinalPlanDisposition::DataFusion) {
        FinalPlanDisposition::Mixed
    } else {
        FinalPlanDisposition::DataFusion
    }
}

fn classify_native_composed_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    // A composed wrapper owns its native consumer, but its producer children
    // may contain DataFusion work before they yield GPU-resident results.
    // Treat the complete composed plan as native only when every producer is
    // independently native.
    if plan.children().iter().all(|child| matches!(classify_final_plan(child.as_ref()), FinalPlanDisposition::Native)) {
        FinalPlanDisposition::Native
    } else {
        FinalPlanDisposition::Mixed
    }
}

fn classify_graph_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    if plan.children().iter().all(|child| graph_child_is_gpu_resident_at_runtime(child.as_ref())) {
        FinalPlanDisposition::Native
    } else {
        // A graph runtime node is native, but any non-native edge or predicate
        // child still executes through DataFusion before it runs.
        FinalPlanDisposition::Mixed
    }
}

fn graph_child_is_gpu_resident_at_runtime(plan: &dyn ExecutionPlan) -> bool {
    if is_native_graph_runtime_child(plan) {
        return classify_final_plan(plan) == FinalPlanDisposition::Native;
    }
    if plan.name() != "CooperativeExec" {
        return false;
    }
    let children = plan.children();
    let [child] = children.as_slice() else {
        return false;
    };
    is_native_graph_runtime_child(child.as_ref()) && classify_final_plan(child.as_ref()) == FinalPlanDisposition::Native
}

fn is_native_graph_runtime_child(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<crate::native_exec::NexusNativeExec>().is_some()
        || plan.downcast_ref::<crate::native_exec::NexusNativeComposedExec>().is_some()
}

fn is_transparent_native_runtime_wrapper(plan: &dyn ExecutionPlan) -> bool {
    crate::native_exec::plan_is_query_admission_wrapped(plan)
        || crate::native_exec::plan_is_unbounded_native_execution_limited(plan)
}

/// DataFusion-facing native optimizer configuration.
///
/// Defaults preserve the DataFusion baseline when native candidates are not
/// accepted. Native execution uses an auto-sized bounded profile, query-engine
/// optimizer options use their own defaults, and plan-preference fallback is
/// disabled unless a workload opts in. The native
/// [`ExecutionContext`](nexus_query_engine::runtime::ExecutionContext) is still built
/// lazily at execution time through [`NexusNativeExecutionConfig`], so allocator
/// and memory-control validation fail at the DataFusion/native boundary rather
/// than during rule construction.
#[derive(Debug, Clone)]
pub struct NexusNativeOptimizerConfig {
    native_execution_config: NexusNativeExecutionConfig,
    native_optimizer_options: nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_plan_preference_options: nexus_query_engine::plan::NativePlanPreferenceOptions,
    final_plan_requirements: Vec<FinalPlanRequirement>,
    cpu_fallback_statement_limiter: Option<CpuFallbackStatementLimiter>,
    cpu_fallback_wait_timeout: Option<Duration>,
}

impl Default for NexusNativeOptimizerConfig {
    fn default() -> Self {
        Self {
            native_execution_config: NexusNativeExecutionConfig::bounded_auto_sized(),
            native_optimizer_options: nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
            native_plan_preference_options: nexus_query_engine::plan::NativePlanPreferenceOptions::disabled(),
            final_plan_requirements: Vec::new(),
            cpu_fallback_statement_limiter: None,
            cpu_fallback_wait_timeout: None,
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

    /// Native-vs-CPU preference rules applied after native optimizer rewrites.
    ///
    /// The adapter default is disabled to preserve generic optimizer rewrite
    /// behavior; workload configs can opt in when their benchmark evidence
    /// shows native fixed overhead dominates small selective shapes.
    pub fn native_plan_preference_options(&self) -> &nexus_query_engine::plan::NativePlanPreferenceOptions {
        &self.native_plan_preference_options
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

    /// Host-side concurrency limiter for physical plans that remain on DataFusion CPU fallback.
    pub fn cpu_fallback_max_concurrent_statements(&self) -> Option<NonZeroUsize> {
        self.cpu_fallback_statement_limiter.as_ref().map(CpuFallbackStatementLimiter::max_concurrent_statements)
    }

    /// Maximum time a CPU fallback statement waits for the fallback limiter.
    ///
    /// `None` means wait indefinitely. Library configs default to `None`;
    /// server configs install their own default unless disabled.
    pub fn cpu_fallback_wait_timeout(&self) -> Option<Duration> {
        self.cpu_fallback_wait_timeout
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

    /// Replace native-vs-CPU preference options.
    pub fn with_native_plan_preference_options(
        mut self,
        options: nexus_query_engine::plan::NativePlanPreferenceOptions,
    ) -> Self {
        self.native_plan_preference_options = options;
        self
    }

    /// Bound concurrent CPU fallback statements that were not rewritten to native execution.
    ///
    /// This limiter is independent of GPU query-service admission. It exists
    /// to bound DataFusion CPU fallback work without consuming native GPU
    /// admission slots. Library users opt in explicitly; the Flight SQL server
    /// installs its own default unless disabled by server configuration.
    pub fn with_cpu_fallback_max_concurrent_statements(mut self, max_concurrent_statements: NonZeroUsize) -> Self {
        self.cpu_fallback_statement_limiter = Some(CpuFallbackStatementLimiter::new_with_wait_timeout(
            max_concurrent_statements,
            self.cpu_fallback_wait_timeout,
        ));
        self
    }

    /// Set the maximum time a CPU fallback statement waits for the fallback limiter.
    ///
    /// Passing a zero duration restores indefinite waiting. The setting is
    /// independent of GPU query-service admission timeouts.
    pub fn with_cpu_fallback_wait_timeout(mut self, wait_timeout: Duration) -> Self {
        self.cpu_fallback_wait_timeout = (!wait_timeout.is_zero()).then_some(wait_timeout);
        if let Some(limit) = self.cpu_fallback_max_concurrent_statements() {
            self.cpu_fallback_statement_limiter =
                Some(CpuFallbackStatementLimiter::new_with_wait_timeout(limit, self.cpu_fallback_wait_timeout));
        }
        self
    }

    /// Disable the host-side CPU fallback statement limiter.
    pub fn without_cpu_fallback_max_concurrent_statements(mut self) -> Self {
        self.cpu_fallback_statement_limiter = None;
        self
    }

    pub(crate) fn cpu_fallback_statement_limiter(&self) -> Option<CpuFallbackStatementLimiter> {
        self.cpu_fallback_statement_limiter.clone()
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
