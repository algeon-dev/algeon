//! Adapter-neutral TPC-H, TPC-DS, and ClickBench workload contracts.
//!
//! Scope includes canonical query ordering, Parquet dataset discovery and
//! metadata, test fixture generation, and DataFusion table registration,
//! with no dependency on the `algeon-datafusion` adapter or its report types.

/// Explicit Parquet fixture root for the active workload.
///
/// Because a process executes a single workload at a time, TPC-H, TPC-DS,
/// and ClickBench share this environment variable. The selected workload
/// verifies the required directory layout and tables on startup.
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
