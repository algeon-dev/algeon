//! Benchmark, report, triage, and stress applications for `algeon-datafusion`.
//!
//! Adapter-neutral workload catalogs and fixtures live in the sibling
//! `workloads` crate. This crate exposes only adapter-specific
//! orchestration and application support; it does not re-export workloads.

pub mod clickbench_benchmark;
pub mod query_service_goodput;
pub mod report_metadata;
pub mod report_workload;
pub mod tpcds_three_way_bench;
pub mod tpch_benchmark;

mod gpu_backend;
mod loaded_libraries;
mod plan_artifacts;

#[doc(hidden)]
pub mod stress_harness;

#[cfg(feature = "cugraph")]
pub mod cugraph_admission_stress;
#[cfg(feature = "cugraph")]
pub mod cugraph_sql_construction_report;
