#![doc = include_str!("../README.md")]
//!
//! ## Entry points
//!
//! | Step | Type | Module |
//! | --- | --- | --- |
//! | Build one backend per process | [`AlgeonGpuBackend`] | [`backend`] |
//! | Install it on a session builder | [`AlgeonGpuBackend::install_on`], [`AlgeonSessionStateBuilderExt`](session::AlgeonSessionStateBuilderExt) | [`session`] |
//! | Run one query | [`AlgeonAttempt`] | [`attempt`] |
//!
//! The backend builder fixes the device envelope at construction. Each
//! attempt plans and executes one query, then publishes a terminal report
//! through [`AlgeonAttempt::completion`]. Under
//! [`AlgeonGpuExecutionMode::NativeRequired`](planner::AlgeonGpuExecutionMode::NativeRequired)
//! the optimizer rejects a completed plan that still executes DataFusion CPU
//! operators.
//!
//! ## Other modules
//!
//! | Purpose | Modules |
//! | --- | --- |
//! | SQL table functions | [`cugraph_sql`], [`cuvs_sql`], [`cuml_sql`], [`gpu_functions`] |
//! | Reports and diagnostics | [`report`], [`native_report`], [`gpu_coverage`], [`observability`], [`build_info`] |
//! | Errors and reason codes | [`error`], [`capability`] |
//! | Plan structures exposed for tooling | [`planner`], [`native_exec`], [`source_cardinality`] |
//! | Flight SQL server, feature `server` | `server` |
//!
//! ## Features
//!
//! The table under Dependencies lists what each feature adds and the native
//! libraries it links. With a feature disabled:
//!
//! - `cugraph`: cuGraph table functions fail closed with a typed capability
//!   error.
//! - `cuvs`, `cuml`: the SQL family stays discoverable and execution fails
//!   before input staging.
//! - `nvml`: device telemetry reports `FeatureDisabled` for NVML and leaves
//!   the NVML values `None`. CUDA telemetry, admission and memory accounting
//!   do not change.
//! - `server`: the `server` module is absent.
//!
//! `bench-support` and `qualification` serve tooling in the Algeon repository.
//! The first adds application and test support outside the embedded and
//! server report contract; the second closes admission on one live device
//! ledger for the multi-GPU validation tool.

#![cfg_attr(docsrs, feature(doc_cfg))]
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

pub use attempt::{AlgeonAttempt, AttemptOptions};
pub use backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership};
