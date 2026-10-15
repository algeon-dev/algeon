//! DataFusion physical optimizer integration for native cuDF execution.
//!
//! The planner exposes a public optimizer rule inserted across two optimizer stages.
//! An initial count-only probe precedes DataFusion aggregate statistics, followed
//! by the final native rewrite immediately prior to final plan validation.
//! When native candidates are unavailable, the DataFusion baseline plan is preserved.
//! Under `AlgeonGpuExecutionMode::NativeRequired`, the final rewrite stage rejects
//! any completed plan with a disposition other than `Native` (excepting feature-disabled
//! cuGraph plans executing no operators). This validation verifies the rewritten plan
//! directly without additional planning passes.
//!
//! [`FinalPlanDisposition`] describes one completed executable plan
//! (`DataFusion`, `Native`, or `Mixed`); planning failures have no disposition.
//! [`FinalPlanCoverage`] carries the single physical-plan walk behind that
//! verdict, and [`classify_final_plan`] reads the same coverage without
//! reclassifying. Candidate decisions stay per-candidate under
//! [`crate::report::CandidateOutcome`].

use datafusion::physical_plan::ExecutionPlan;
pub use query_planning::{
    contract_report::{CapabilityContractReport, CapabilityDecisionRow},
    mode::StatisticalAggregateExecutionMode,
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions},
    source::SourceDiagnostics,
};

use crate::{final_plan_observation::observe_final_plan, native_exec::AlgeonNativeExecutionConfig};

pub(crate) mod native;

pub use native::{
    AlgeonNativePhysicalOptimizerRule, NativeCapabilityAcceptedCandidateDiagnostics,
    NativeCapabilityCompiledGraphDiagnostics, NativeCapabilityContractDiagnostics,
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

/// Which plans native lowering rewrites and whether the completed plan may
/// execute DataFusion CPU operators.
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

impl AlgeonGpuExecutionMode {
    /// Whether ordinary relational SQL is a candidate for native lowering.
    /// Explicit GPU UDTF nodes are candidates in every mode.
    pub const fn rewrites_relational_plans(self) -> bool {
        !matches!(self, Self::FunctionsOnly)
    }

    /// Whether a completed plan that executes any DataFusion CPU operator is
    /// rejected.
    pub const fn rejects_datafusion_cpu(self) -> bool {
        matches!(self, Self::NativeRequired)
    }

    /// Returns the configuration string accepted by `native.execution_mode`
    /// and reported by `algeon_version()`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FunctionsOnly => "functions_only",
            Self::NativePreferred => "native_preferred",
            Self::NativeRequired => "native_required",
        }
    }
}

impl std::fmt::Display for AlgeonGpuExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classify the exact completed physical plan without invoking planning.
///
/// Evaluates execution ownership across physical plan nodes. Operators or
/// wrappers without recognized native execution ownership are classified
/// conservatively as [`FinalPlanDisposition::Mixed`].
pub fn classify_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanDisposition {
    observe_final_plan(plan).disposition()
}

/// DataFusion-facing native optimizer configuration.
///
/// Defaults preserve the DataFusion baseline when native candidates are not
/// accepted. Native execution uses a bounded planning template and
/// query-engine optimizer options use their own defaults. The native
/// [`ExecutionContext`](query_runtime::runtime::ExecutionContext) is
/// acquired lazily from backend admission through [`AlgeonNativeExecutionConfig`].
#[derive(Debug, Clone)]
pub struct AlgeonNativeOptimizerConfig {
    native_execution_config: AlgeonNativeExecutionConfig,
    native_optimizer_options: NativeOptimizerOptions,
    execution_mode: AlgeonGpuExecutionMode,
}

impl Default for AlgeonNativeOptimizerConfig {
    fn default() -> Self {
        Self {
            native_execution_config: AlgeonNativeExecutionConfig::default(),
            native_optimizer_options: NativeOptimizerOptions::default(),
            execution_mode: AlgeonGpuExecutionMode::NativePreferred,
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

    /// The execution mode native rewriting and the completed-plan check
    /// follow.
    pub fn execution_mode(&self) -> AlgeonGpuExecutionMode {
        self.execution_mode
    }

    /// Set the execution mode.
    pub fn with_execution_mode(mut self, mode: AlgeonGpuExecutionMode) -> Self {
        self.execution_mode = mode;
        self
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

    /// Set the floating statistical aggregate contract without rebuilding the
    /// rest of the config.
    ///
    /// Defaults to [`StatisticalAggregateExecutionMode::GpuNativeTolerant`],
    /// whose results can differ from DataFusion in floating-point bits. Select
    /// [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`] when
    /// results must be identical to DataFusion's. In that mode a candidate whose
    /// statistical aggregate input DataFusion reads through more than one
    /// partition stays on DataFusion, whose own result then depends on the
    /// order partitions deliver rows.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.native_execution_config =
            self.native_execution_config.clone().with_statistical_aggregate_execution_mode(mode);
        self
    }
}
