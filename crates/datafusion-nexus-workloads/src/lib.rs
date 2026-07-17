//! Adapter-neutral TPC-H, TPC-DS, and ClickBench workload contracts.
//!
//! This crate owns canonical query ordering, Parquet dataset discovery and
//! metadata, tiny fixture writers, and DataFusion table registration. It must
//! not depend on the `datafusion-nexus` adapter or its report types.

/// Explicit Parquet fixture root for the workload being run by this process.
///
/// A process runs one workload at a time, so TPC-H, TPC-DS, and ClickBench do
/// not need workload-specific environment-variable names. The supplied root
/// is strict: the selected workload validates that it has the layout it
/// requires.
pub const NEXUS_PARQUET_DIR_ENV: &str = "NEXUS_PARQUET_DIR";

pub mod clickbench;
pub mod metadata;
pub mod tpcds;
pub mod tpch;
