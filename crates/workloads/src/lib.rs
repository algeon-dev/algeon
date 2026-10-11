//! Adapter-neutral TPC-H, TPC-DS, and ClickBench workload contracts.
//!
//! This crate owns canonical query ordering, Parquet dataset discovery and
//! metadata, tiny fixture writers, and DataFusion table registration. It must
//! not depend on the `algeon-datafusion` adapter or its report types.

/// Explicit Parquet fixture root for the workload being run by this process.
///
/// A process runs one workload at a time, so TPC-H, TPC-DS, and ClickBench do
/// not need workload-specific environment-variable names. The supplied root
/// is strict: the selected workload validates that it has the layout it
/// requires.
pub const ALGEON_PARQUET_DIR_ENV: &str = "ALGEON_PARQUET_DIR";

pub mod clickbench;
pub mod error;
pub mod forest;
pub mod graph;
pub mod metadata;
mod parquet_metadata;
pub mod tpcds;
pub mod tpch;

pub use error::{WorkloadError, WorkloadResult};
