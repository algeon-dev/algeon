//! Adapter-neutral TPC-H, TPC-DS, and ClickBench workload contracts.
//!
//! This crate owns canonical query ordering, Parquet dataset discovery and
//! metadata, tiny fixture writers, and DataFusion table registration. It must
//! not depend on the `datafusion-nexus` adapter or its report types.

pub mod clickbench;
pub mod metadata;
pub mod tpcds;
pub mod tpch;
