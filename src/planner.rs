use std::{
    fmt,
    sync::{Arc, Mutex},
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, limit::LocalLimitExec,
        projection::ProjectionExec, sorts::sort::SortExec,
    },
};

use crate::{
    error::{Error, ErrorCode},
    native_exec::{NativeCudfExec, NativeCudfExecutionConfig},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{CandidateOutcome, GpuCandidate, PlanningReport},
    session::{CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME},
};

mod native;

pub use native::{
    CudfNativePhysicalOptimizerRule, NativePlanningDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};

#[derive(Debug, Clone)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    native_execution_config: NativeCudfExecutionConfig,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            native_execution_config: NativeCudfExecutionConfig::unbounded(),
        }
    }
}

impl CudfOptimizerConfig {
    pub fn new(fallback_policy: GpuFallbackPolicy) -> Self {
        Self { fallback_policy, ..Self::default() }
    }

    pub fn fallback_policy(&self) -> GpuFallbackPolicy {
        self.fallback_policy
    }

    pub fn native_execution_config(&self) -> &NativeCudfExecutionConfig {
        &self.native_execution_config
    }

    pub fn with_native_execution_config(mut self, execution_config: NativeCudfExecutionConfig) -> Self {
        self.native_execution_config = execution_config;
        self
    }

    /// Use one caller-provided cudf runtime for native execution produced by this optimizer.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.native_execution_config = self.native_execution_config.clone().with_runtime(runtime);
        self
    }
}

fn candidate_rejected_error(candidate: &GpuCandidate, reason: FallbackReason, config: &CudfOptimizerConfig) -> Error {
    Error::new(
        ErrorCode::NativeCandidateRejected,
        "datafusion_cudf.native.optimize_plan",
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
