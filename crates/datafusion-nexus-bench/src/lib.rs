//! Benchmark, report, triage, and stress applications for `datafusion-nexus`.
//!
//! Adapter-neutral workload catalogs and fixtures live in the sibling
//! `datafusion-nexus-workloads` crate. This crate exposes only adapter-specific
//! orchestration and application support; it does not re-export workloads.

pub mod query_service_tpch_stress;
pub mod report_metadata;
pub mod report_workload;
pub mod tpcds_three_way_bench;
pub mod tpch_benchmark;

#[doc(hidden)]
pub mod stress_harness;

#[cfg(feature = "cugraph")]
pub mod cugraph_admission_stress;

#[cfg(feature = "iceberg")]
pub mod iceberg_smoke;
