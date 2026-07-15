//! Benchmark, report, triage, and stress applications for `datafusion-nexus`.
//!
//! Adapter-neutral workload catalogs and fixtures live in the sibling
//! `datafusion-nexus-workloads` crate. This crate exposes only adapter-specific
//! orchestration and application support; it does not re-export workloads.

pub mod report_metadata;
pub mod report_workload;
pub mod tpch_benchmark;

#[doc(hidden)]
pub mod stress_harness;

#[cfg(feature = "iceberg")]
pub mod iceberg_smoke;
