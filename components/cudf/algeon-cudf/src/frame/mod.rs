//! High-level user-facing table and column APIs.
//!
//! `frame` is the main high-level namespace below the crate root:
//!
//! - [`DataFrame`] and [`Series`] are the primary entry points.
//! - [`dataframe`] contains DataFrame-specific configuration and result types.
//! - [`factorization`] contains categorical encoding result and option types.
//! - [`series`] contains Series-specific traits and helpers.

pub mod dataframe;
pub mod factorization;
pub mod series;

pub use dataframe::DataFrame;
pub use series::{HostColumnData, IntoSeries, Series};
