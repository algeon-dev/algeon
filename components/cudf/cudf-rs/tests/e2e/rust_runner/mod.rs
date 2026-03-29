//! Rust runner for the E2E conformance test framework.
//!
//! This module provides reusable, library-shaped code for:
//! - parsing JSON fixture manifests into typed Rust structures
//! - representing normalized results in a shared schema
//! - constructing DataFrames from manifest datasets
//! - executing manifest operations through cudf-rs
//! - normalizing results for cross-language comparison
//! - structured comparison with diff reporting

pub mod compare;
pub mod dataset;
pub mod execute;
pub mod manifest;
pub mod normalize;

pub use compare::{Mismatch, compare_results};
pub use dataset::build_dataframe;
pub use execute::{ExecutionError, execute_operation, normalize_manifest_error};
pub use manifest::Fixture;
pub use normalize::{NormalizedError, NormalizedResult};
