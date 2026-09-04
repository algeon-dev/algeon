//! DataFusion adapter for native GPU execution.
//!
//! Embedded applications normally enter through:
//!
//! 1. Build one process-scoped [`backend::NexusGpuBackend`] and install it on
//!    each caller-owned DataFusion session with
//!    [`backend::NexusGpuBackend::install_on`].
//! 2. Run each query through [`backend::NexusGpuBackend::begin_attempt`] so
//!    planning, admission, execution, and the terminal report share one
//!    invocation lifecycle.
//!
//! Strict production policy can enforce that no DataFusion CPU execution
//! remains in the final physical plan by configuring
//! [`planner::NexusNativeOptimizerConfig::with_no_datafusion_cpu_requirement`].
//!
//! Optional query diagnostic coverage can be inspected ahead of execution via
//! [`gpu_coverage::validate_query`], which clones planning state and performs
//! an isolated planning-only pass without beginning an execution attempt or
//! acquiring admission.
//!
//! Public diagnostics separate three decision layers with different owners
//! and cardinalities:
//!
//! - Candidate decisions ([`report::CandidateOutcome`], one row per optimizer
//!   candidate) explain why a candidate was selected, deferred, or rejected;
//!   [`report::CandidateOutcomeKind`] owns the stable projection labels.
//! - Final-plan coverage ([`gpu_coverage::QueryGpuPlanningOutcome`], at most
//!   one executable plan per validation) describes whether the completed plan
//!   runs on DataFusion, native, or both, or why planning produced no
//!   executable plan at all.
//! - Attempt disposition ([`attempt`], one row per sealed execution attempt)
//!   records how a started attempt terminated. It is owned by the runtime and
//!   never re-derived from plan-time types.
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
//! use datafusion_nexus::{
//!     attempt::AttemptOptions,
//!     backend::{
//!         GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceConfig, NexusGpuDeviceProfile,
//!     },
//! };
//! use futures::TryStreamExt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let device = NexusGpuDeviceConfig::new(
//!     NexusGpuDeviceProfile::new(0),
//!     GpuMemoryOwnership::WholeDeviceExclusive,
//! );
//! let backend = NexusGpuBackend::builder().devices([device]).build()?;
//! let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! let attempt = backend.begin_attempt(AttemptOptions::default())?;
//! let completion = attempt.completion();
//! let stream = attempt.sql(&ctx, "SELECT 1 AS value").await?;
//! let _batches = stream.try_collect::<Vec<_>>().await?;
//! let _report = completion.await;
//!
//! backend.close_and_wait(Duration::from_secs(10))?;
//! # Ok(())
//! # }
//! ```
//!
//! Deployment features are `cugraph`, `cuvs`, `iceberg`, and `nvml`. The
//! `bench-support` and `server-internal` features expose workspace-internal
//! support surfaces and are not ordinary application choices.
//!
//! DataFusion-specific planning belongs here. DataFusion-free native IR,
//! runtime policy, capability admission, and native
//! [`nexus_query_engine::plan::QueryPlan`] execution live in the engine crate.
//! Public diagnostics and configuration are grouped under [`attempt`],
//! [`capability`], [`gpu_coverage`], [`native_report`], [`observability`],
//! and [`report`].

#[cfg(test)]
extern crate self as datafusion_nexus;

pub mod attempt;
#[cfg(feature = "server-internal")]
#[doc(hidden)]
pub use attempt::server_internal;
pub mod backend;
pub mod capability;
pub mod cugraph_sql;
pub mod cuvs_sql;
pub mod error;
mod final_plan_observation;
pub mod gpu_coverage;
pub mod gpu_coverage_sql;
pub mod gpu_functions;
mod native;
pub mod native_exec;
pub mod native_report;
pub mod observability;
mod physical_plan;
pub mod planner;
pub mod report;
pub mod session;
pub mod source_cardinality;
pub mod table_format;
