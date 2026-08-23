//! DataFusion adapter for native GPU execution.
//!
//! Embedded applications normally enter through three APIs:
//!
//! 1. Build one process-scoped [`backend::NexusGpuBackend`] and install it on
//!    each caller-owned DataFusion session with
//!    [`backend::NexusGpuBackend::install_on`].
//! 2. Run each query through [`backend::NexusGpuBackend::begin_attempt`] so
//!    planning, admission, execution, and the terminal report share one
//!    invocation lifecycle.
//! 3. Call [`gpu_coverage::validate_query`] before execution when the
//!    application must inspect whether supported physical-plan candidates
//!    produce a GPU-only, mixed, or DataFusion plan.
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
//! use datafusion_nexus::{
//!     attempt::AttemptOptions,
//!     backend::{GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile},
//!     gpu_coverage::validate_query,
//! };
//! use futures::TryStreamExt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let backend = NexusGpuBackend::builder()
//!     .device_profiles([NexusGpuDeviceProfile::new(0)])
//!     .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
//!     .build()?;
//! let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! let _coverage = validate_query(&ctx, "SELECT 1 AS value").await?;
//! let attempt = backend.begin_attempt(AttemptOptions::default())?;
//! let stream = attempt.sql(&ctx, "SELECT 1 AS value").await?;
//! let completion = stream.completion();
//! let _batches = stream.try_collect::<Vec<_>>().await?;
//! let _report = completion.await;
//!
//! drop(ctx);
//! backend.close_and_wait(Duration::from_secs(10))?;
//! # Ok(())
//! # }
//! ```
//!
//! Deployment features are `cugraph`, `cuvs`, `iceberg`, and `nvml`. The
//! `test-support`, `bench-support`, and `server-internal` features expose
//! workspace-internal support surfaces and are not ordinary application
//! choices.
//!
//! DataFusion-specific planning belongs here. DataFusion-free native IR,
//! runtime policy, capability admission, and native
//! [`nexus_query_engine::plan::QueryPlan`] execution live in the engine crate.
//! Public diagnostics and configuration are grouped under [`attempt`],
//! [`capability`], [`gpu_coverage`], [`native_report`], [`observability`],
//! [`report`], and [`session_config`].

pub mod attempt;
pub mod backend;
pub mod capability;
pub mod cugraph_sql;
pub mod cuvs_sql;
pub mod error;
mod final_plan_observation;
pub mod gpu_coverage;
pub mod gpu_coverage_sql;
pub mod gpu_functions;
#[doc(hidden)]
pub mod native;
pub mod native_exec;
pub mod native_report;
pub mod observability;
mod physical_plan;
pub mod planner;
pub mod report;
pub mod session;
pub mod session_config;
pub mod source_cardinality;
pub mod table_format;
