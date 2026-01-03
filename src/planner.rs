use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, config::ConfigOptions},
    datasource::{physical_plan::ParquetSource, source::DataSourceExec},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, limit::LocalLimitExec,
        projection::ProjectionExec, sorts::sort::SortExec,
    },
};

use crate::{
    capability::{aggregate_rejection, expr_rejection, projection_rejection, sort_rejection},
    error::{Error, ErrorCode},
    exec::CudfExecutionConfig,
    native_exec::{NativeCudfExec, NativeCudfExecutionConfig},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{CandidateOutcome, GpuCandidate, PlanningReport},
    session::{
        CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        CUDF_PHYSICAL_OPTIMIZER_RULE_NAME,
    },
};

mod gpu_island;
mod native;

pub use gpu_island::{CudfPhysicalOptimizerRule, optimize_plan};
pub use native::{CudfNativePhysicalOptimizerRule, NativePlanningDiagnostics};

#[derive(Debug, Clone)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    min_operators_per_island: usize,
    execution_config: CudfExecutionConfig,
    native_execution_config: Option<NativeCudfExecutionConfig>,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            min_operators_per_island: 2,
            execution_config: CudfExecutionConfig::default(),
            native_execution_config: None,
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

    pub fn min_operators_per_island(&self) -> usize {
        self.min_operators_per_island
    }

    pub fn execution_config(&self) -> &CudfExecutionConfig {
        &self.execution_config
    }

    pub fn native_execution_config(&self) -> Option<&NativeCudfExecutionConfig> {
        self.native_execution_config.as_ref()
    }

    pub fn native_execution_enabled(&self) -> bool {
        self.native_execution_config.is_some()
    }

    pub fn with_min_operators_per_island(mut self, min_operators_per_island: usize) -> Self {
        self.min_operators_per_island = min_operators_per_island.max(1);
        self
    }

    pub fn with_execution_config(mut self, execution_config: CudfExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    pub fn with_native_execution_config(mut self, native_execution_config: NativeCudfExecutionConfig) -> Self {
        self.native_execution_config = Some(native_execution_config);
        self
    }

    /// Use one caller-provided cudf runtime for GPU islands produced by this optimizer.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.execution_config = self.execution_config.clone().with_runtime(runtime);
        self
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub plan: Arc<dyn ExecutionPlan>,
    pub report: PlanningReport,
}

fn candidate_rejected_error(candidate: &GpuCandidate, reason: FallbackReason, config: &CudfOptimizerConfig) -> Error {
    Error::new(
        ErrorCode::GpuCandidateRejected,
        "datafusion_cudf.optimize_plan",
        "GPU candidate rejected by adapter policy",
    )
    .with_fact("candidate_shape", candidate.shape.as_str())
    .with_fact("fallback_reason", reason.code())
    .with_fact("fallback_policy", config.fallback_policy())
    .with_fact("operator_count", candidate.operator_count)
    .with_fact("min_operators_per_island", config.min_operators_per_island())
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
