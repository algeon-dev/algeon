//! DataFusion adapter for native GPU execution.
//!
//! The crate exposes the DataFusion-facing boundary around the native
//! [`nexus_query_engine`] IR:
//!
//! - [`session`] wires the native physical optimizer into a
//!   `SessionStateBuilder` while preserving DataFusion's final validation rule.
//! - [`planner`] owns native rewrite policy, fallback behavior, and the
//!   strict/adaptive planning diagnostics consumed by reports.
//! - [`native_exec`] contains the `ExecutionPlan` wrappers that either
//!   materialize native GPU results back to Arrow or pass GPU-resident
//!   fragments between native subplans.
//! - [`native_report`] defines the TSV/report contracts emitted by benchmark
//!   and diagnostic binaries. Column order is part of the public surface.
//!
//! DataFusion-specific planning belongs here. DataFusion-free native IR,
//! runtime policy, capability admission, and cudf execution live in
//! [`nexus_query_engine`].

pub mod backend;
pub mod capability;
pub mod cugraph_sql;
pub mod cuvs_sql;
pub mod error;
mod final_plan_observation;
pub mod gpu_coverage;
pub mod gpu_coverage_sql;
pub mod gpu_functions;
pub mod native;
pub mod native_exec;
pub mod native_report;
mod physical_plan;
pub mod planner;
pub mod report;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod session_config;
pub mod source_cardinality;
pub mod table_format;
