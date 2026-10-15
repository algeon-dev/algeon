//! High-level user-facing table and column APIs.
//!
//! # Module Organization
//!
//! - [`DataFrame`] and [`Series`] are the primary entry points.
//! - [`dataframe`] contains DataFrame-specific configuration and result types.
//! - [`factorization`] contains categorical encoding result and option types.
//! - [`series`] contains Series-specific traits and helpers.

pub mod dataframe;
pub mod factorization;
pub mod series;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;

pub use dataframe::DataFrame;
pub use series::{HostColumnData, IntoSeries, Series};
