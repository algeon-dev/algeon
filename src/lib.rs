//! DataFusion adapter for cudf-rs GPU islands.

pub mod benchmark;
mod capability;
pub mod error;
pub mod exec;
pub mod native;
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
pub use native_report::{
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeFallbackPressureOutcome,
    NativeFallbackPressureReport, NativeFallbackPressureRow, NativeFallbackPressureWorkloadItem,
    NativeSourceDiagnosticSummary,
};
pub use planner::{CudfOptimizerConfig, CudfPhysicalOptimizerRule, OptimizationResult, optimize_plan};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
pub use session::{
    CUDF_PHYSICAL_OPTIMIZER_RULE_NAME, CudfSessionStateBuilderExt, PhysicalOptimizerRules,
    cudf_physical_optimizer_rules, cudf_physical_optimizer_rules_with_rule,
};
