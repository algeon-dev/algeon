//! Benchmark, report, triage, and stress applications for `algeon-datafusion`.
//!
//! Workload catalogs and fixtures reside in the sibling `workloads` crate, while
//! adapter-specific orchestration and application harnesses live in this crate
//! without re-exporting workload definitions.

pub mod clickbench_benchmark;
pub mod query_service_goodput;
pub mod report_metadata;
pub mod report_workload;
pub mod sort_lane_sweep;
pub mod tpcds_gpu_cpu_bench;
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
#[cfg(feature = "cuml")]
pub mod forest_predict;
