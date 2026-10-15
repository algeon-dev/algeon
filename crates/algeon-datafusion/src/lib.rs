//! Embedded adapter from DataFusion to native GPU execution.
//!
//! The execution lifecycle progresses from [`backend::AlgeonGpuBackend`] through
//! session installation to [`attempt::AlgeonAttempt`]. The backend builder
//! assembles the immutable device envelope, [`session`] installs the native optimizer
//! on a caller-owned session builder, and each attempt plans and executes one
//! invocation before publishing a terminal report. Strict production policy
//! can enforce that no DataFusion CPU execution remains in the final physical
//! plan via [`planner::AlgeonGpuExecutionMode::NativeRequired`].
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use algeon_datafusion::{
//!     attempt::AttemptOptions,
//!     backend::{
//!         AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership,
//!     },
//! };
//! use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
//! use futures::TryStreamExt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Ordinal `0` is the first process-visible CUDA device.
//! // `WholeDeviceExclusive` promises process-wide allocator ownership; it
//! // does not allocate device memory by itself.
//! let device = AlgeonGpuDeviceConfig::new(
//!     AlgeonGpuDeviceProfile::new(0),
//!     GpuMemoryOwnership::WholeDeviceExclusive,
//! );
//! let backend = AlgeonGpuBackend::builder().devices([device]).build()?;
//! let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! let attempt = backend.begin_attempt(AttemptOptions::default())?;
//! // Taking the completion receiver prior to planning ensures the terminal report
//! // remains observable even if planning fails.
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
//! and execute through [`attempt`]. [`build_info`] reports the version,
//! commit, and enabled engines of this build. Planning-only diagnosis lives in
//! [`gpu_coverage`]; runtime exporter projection lives in [`observability`].
//!
//! Optional engines change plan classification when enabled. Without the
//! `cugraph` feature, cuGraph UDTFs fail closed with a typed
//! capability error. Without `cuvs` or `cuml`, the corresponding SQL family
//! remains discoverable but execution fails before input staging. Without
//! `nvml`, device telemetry reports `FeatureDisabled` for NVML and leaves the
//! NVML values `None`; CUDA telemetry, admission and memory accounting are
//! unchanged. The `bench-support`
//! and `qualification` features serve as internal harnesses; `bench-support`
//! provides application and test support facilities outside the embedded and
//! server report contract, whereas `qualification` closes admission on one live
//! device ledger for the multi-GPU validation tool. Enable `server` to embed
//! the Algeon Flight SQL server alongside the adapter.

#![deny(missing_docs)]

#[cfg(test)]
extern crate self as algeon_datafusion;

pub mod attempt;
pub mod backend;
pub mod build_info;
pub mod capability;
#[cfg(all(test, any(feature = "cugraph", feature = "server")))]
#[path = "../tests/support/closing_backend.rs"]
mod closing_backend;
pub mod cugraph_sql;
pub mod cuml_sql;
pub mod cuvs_sql;
pub mod error;
mod final_plan_observation;
pub mod gpu_coverage;
pub mod gpu_functions;
mod gpu_query_planner;
mod name_suggestion;
mod native;
pub mod native_exec;
pub mod native_report;
pub mod observability;
mod physical_plan;
pub mod planner;
pub mod report;
mod s3_storage;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod source_cardinality;
