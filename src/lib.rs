//! DataFusion adapter for cudf-rs GPU islands.

pub mod benchmark;
mod capability;
pub mod error;
pub mod exec;
pub mod native;
pub mod native_exec;
pub mod native_report;
pub mod planner;
pub mod policy;
pub mod report;
pub mod session;

pub use benchmark::{
    BenchmarkGateReason, BenchmarkGateStatus, LargerIslandBenchmarkAssessment, LargerIslandBenchmarkConfig,
    LargerIslandBenchmarkDecisionCriteria, LargerIslandBenchmarkMetrics, LargerIslandBenchmarkPlanning,
    LargerIslandBenchmarkReport, PhaseCRuntimeGateStatus, TimedBenchmarkRun, build_larger_island_benchmark_plan,
    run_larger_island_benchmark, run_larger_island_benchmark_with_optimizer,
};
pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
pub use exec::{CudfExecutionConfig, CudfGpuIslandExec};
pub use native::{NativeLoweringResult, lower_physical_plan_to_native};
pub use native_exec::{NATIVE_CUDF_DATAFUSION_METRIC_NAMES, NativeCudfExec, NativeCudfExecutionConfig};
pub use native_report::{
    NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeDataFusionPlanningOutcome,
    NativeDataFusionPlanningReport, NativeDataFusionPlanningRow, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionPostNativeExecutionPath, NativeFallbackPressureOutcome, NativeFallbackPressureReport,
    NativeFallbackPressureRow, NativeFallbackPressureWorkloadItem, NativeSourceDiagnosticSummary,
};
pub use planner::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule, NativePlanningDiagnostics,
    OptimizationResult, optimize_plan,
};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
pub use session::{
    CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_PHYSICAL_OPTIMIZER_RULE_NAME, CudfSessionStateBuilderExt,
    PhysicalOptimizerRules, cudf_physical_optimizer_rules, cudf_physical_optimizer_rules_with_retained_rules,
    cudf_physical_optimizer_rules_with_rule,
};
