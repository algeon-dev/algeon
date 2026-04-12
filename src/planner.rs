//! DataFusion physical optimizer integration for native cuDF execution.
//!
//! The planner exposes one public optimizer rule, but session integration
//! inserts it in two optimizer stages: a count-only probe before DataFusion aggregate
//! statistics and the final native rewrite immediately before final plan
//! validation. `GpuPreferred` keeps DataFusion CPU fallback available during
//! planning; `GpuOnly` turns native candidate rejection into an optimizer
//! error. Once a native execution node starts running, CPU fallback is no
//! longer attempted.

use std::{fmt, num::NonZeroUsize, time::Duration};

use datafusion::common::{Result as DataFusionResult, config::ConfigOptions};

use crate::{
    error::{Error, ErrorCode},
    native_exec::{CpuFallbackStatementLimiter, NativeCudfExecutionConfig},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::GpuCandidate,
    session_config::optimizer_config_with_datafusion_options,
};

mod leaf_guard;
mod native;

pub use leaf_guard::{CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfTableFormatGpuOnlyLeafGuardRule};
pub use native::{
    CudfNativePhysicalOptimizerRule, NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics,
    NativePlanningDiagnostics, NativeStrictBoundedAcceptedCandidateDiagnostics,
    NativeStrictBoundedCompiledGraphDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeStrictBoundedRejectedCandidateDiagnostics, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};

/// DataFusion-facing native optimizer configuration.
///
/// Defaults are production-oriented: GPU is preferred but not mandatory,
/// native execution uses an auto-sized bounded profile, and query-engine
/// optimizer options use their own defaults, and plan-preference fallback is
/// disabled unless a workload opts in. The native
/// [`ExecutionContext`](nexus_query_engine::ExecutionContext) is still built
/// lazily at execution time through [`NativeCudfExecutionConfig`], so allocator
/// and memory-control validation fail at the DataFusion/native boundary rather
/// than during rule construction.
#[derive(Debug, Clone)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    native_execution_config: NativeCudfExecutionConfig,
    native_optimizer_options: nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: nexus_query_engine::NativePlanPreferenceOptions,
    cpu_fallback_statement_limiter: Option<CpuFallbackStatementLimiter>,
    cpu_fallback_wait_timeout: Option<Duration>,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            native_execution_config: NativeCudfExecutionConfig::bounded_auto_sized(),
            native_optimizer_options: nexus_query_engine::NativeOptimizerOptions::default(),
            native_plan_preference_options: nexus_query_engine::NativePlanPreferenceOptions::disabled(),
            cpu_fallback_statement_limiter: None,
            cpu_fallback_wait_timeout: None,
        }
    }
}

impl CudfOptimizerConfig {
    /// Build from an explicit fallback policy; all other fields use conservative defaults.
    pub fn new(fallback_policy: GpuFallbackPolicy) -> Self {
        Self { fallback_policy, ..Self::default() }
    }

    /// Planning fallback policy used after native candidate analysis.
    pub fn fallback_policy(&self) -> GpuFallbackPolicy {
        self.fallback_policy
    }

    /// Native execution settings copied into rewritten execution nodes.
    pub fn native_execution_config(&self) -> &NativeCudfExecutionConfig {
        &self.native_execution_config
    }

    /// Query-engine optimizer options applied after lowering.
    pub fn native_optimizer_options(&self) -> &nexus_query_engine::NativeOptimizerOptions {
        &self.native_optimizer_options
    }

    /// Native-vs-CPU preference rules applied after native optimizer rewrites.
    ///
    /// The adapter default is disabled to preserve generic optimizer rewrite
    /// behavior; workload configs can opt in when their benchmark evidence
    /// shows native fixed overhead dominates small selective shapes.
    pub fn native_plan_preference_options(&self) -> &nexus_query_engine::NativePlanPreferenceOptions {
        &self.native_plan_preference_options
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
    pub fn native_pipeline_execution_mode(&self) -> nexus_query_engine::NativePipelineExecutionMode {
        self.native_execution_config.native_pipeline_execution_mode()
    }

    /// Floating statistical aggregate execution contract used by native execution.
    pub fn statistical_aggregate_execution_mode(&self) -> nexus_query_engine::StatisticalAggregateExecutionMode {
        self.native_execution_config.statistical_aggregate_execution_mode()
    }

    /// Replace the native execution config copied into accepted native plans.
    pub fn with_native_execution_config(mut self, execution_config: NativeCudfExecutionConfig) -> Self {
        self.native_execution_config = execution_config;
        self
    }

    /// Attach a query id to the native execution config only when the caller has not supplied one.
    pub(crate) fn with_query_id_if_absent(mut self, query_id: impl Into<nexus_query_engine::QueryId>) -> Self {
        self.native_execution_config = self.native_execution_config.with_query_id_if_absent(query_id);
        self
    }

    /// Replace DataFusion-free native optimizer options.
    pub fn with_native_optimizer_options(mut self, options: nexus_query_engine::NativeOptimizerOptions) -> Self {
        self.native_optimizer_options = options;
        self
    }

    /// Replace native-vs-CPU preference options.
    pub fn with_native_plan_preference_options(
        mut self,
        options: nexus_query_engine::NativePlanPreferenceOptions,
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
        mode: nexus_query_engine::NativePipelineExecutionMode,
    ) -> Self {
        self.native_execution_config = self.native_execution_config.clone().with_native_pipeline_execution_mode(mode);
        self
    }

    /// Set the floating statistical aggregate contract without rebuilding the
    /// rest of the config.
    pub fn with_statistical_aggregate_execution_mode(
        mut self,
        mode: nexus_query_engine::StatisticalAggregateExecutionMode,
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

fn candidate_rejected_error(candidate: &GpuCandidate, reason: FallbackReason, config: &CudfOptimizerConfig) -> Error {
    Error::new(
        ErrorCode::NativeCandidateRejected,
        "datafusion_nexus.native.optimize_plan",
        "cuDF native candidate rejected by adapter policy",
    )
    .with_fact("candidate_shape", candidate.shape.as_str())
    .with_fact("fallback_reason", reason.code())
    .with_fact("fallback_policy", config.fallback_policy())
    .with_fact("operator_count", candidate.operator_count)
}

impl fmt::Display for GpuFallbackPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuOnly => f.write_str("GpuOnly"),
            Self::GpuPreferred => f.write_str("GpuPreferred"),
            Self::CpuOnly => f.write_str("CpuOnly"),
        }
    }
}
