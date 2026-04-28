//! DataFusion adapter for cudf-rs GPU islands.

pub mod error;
pub mod exec;
pub mod planner;
pub mod policy;
pub mod report;
pub mod session;

pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
pub use exec::{CudfExecutionConfig, CudfGpuIslandExec};
pub use planner::{CudfOptimizerConfig, CudfPhysicalOptimizerRule, OptimizationResult, optimize_plan};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
pub use session::{
    CUDF_PHYSICAL_OPTIMIZER_RULE_NAME, CudfSessionStateBuilderExt, PhysicalOptimizerRules,
    cudf_physical_optimizer_rules, cudf_physical_optimizer_rules_with_rule,
};
