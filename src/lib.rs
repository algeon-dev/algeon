//! DataFusion adapter for cudf-rs GPU islands.

pub mod error;
pub mod exec;
pub mod planner;
pub mod policy;
pub mod report;

pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
pub use exec::CudfGpuIslandExec;
pub use planner::{CudfOptimizerConfig, CudfPhysicalOptimizerRule, OptimizationResult, optimize_plan};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
