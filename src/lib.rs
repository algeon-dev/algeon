//! Embedded adapter from DataFusion to native GPU execution.
//!
//! The canonical lifecycle is [`backend::NexusGpuBackend`], then session
//! installation, then [`attempt::NexusAttempt`]: the builder assembles the
//! immutable device envelope, [`session`] installs the native optimizer on a
//! caller-owned session builder, and each attempt plans and executes one
//! invocation before publishing one terminal report. Strict production policy
//! can enforce that no DataFusion CPU execution remains in the final physical
//! plan by configuring
//! [`planner::NexusNativeOptimizerConfig::with_datafusion_cpu_execution_policy`].
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
//! // Ordinal `0` is the first process-visible CUDA device.
//! // `WholeDeviceExclusive` promises process-wide allocator ownership; it
//! // does not allocate device memory by itself.
//! let device = NexusGpuDeviceConfig::new(
//!     NexusGpuDeviceProfile::new(0),
//!     GpuMemoryOwnership::WholeDeviceExclusive,
//! );
//! let backend = NexusGpuBackend::builder().devices([device]).build()?;
//! let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! let attempt = backend.begin_attempt(AttemptOptions::default())?;
//! // Take the completion receiver before planning so the terminal report is
//! // observable even when planning itself fails.
//! let completion = attempt.completion();
//! let stream = attempt.sql(&ctx, "SELECT 1 AS value").await?;
//! let _batches = stream.try_collect::<Vec<_>>().await?;
//! let terminal = completion.await;
//! println!("terminal outcome: {:?}", terminal.outcome);
//!
//! // Async shutdown waits for GPU-resource and observation quiescence without
//! // blocking the executor worker.
//! backend.close_and_wait_async(Duration::from_secs(10)).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Applications start from [`backend`], install sessions through [`session`],
//! and execute through [`attempt`]. Planning-only diagnosis lives in
//! [`gpu_coverage`]; runtime exporter projection lives in [`observability`].
//!
//! Optional engines change plan classification only while enabled: with the
//! `cugraph` feature off, cuGraph table functions fail closed with a typed
//! capability error; with `cuvs` off, cuVS planner calls fail closed; with
//! `iceberg` off, Iceberg providers are unavailable; with `nvml` off, NVML
//! telemetry reads report unavailable. The `bench-support`,
//! `server-internal`, and `qualification` features are internal harnesses, not
//! application choices.

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
