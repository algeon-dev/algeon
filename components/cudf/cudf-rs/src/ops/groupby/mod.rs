//! GPU-accelerated groupby aggregation.
//!
//! Most users should use [`DataFrame::groupby`](crate::DataFrame::groupby)
//! and the high-level [`GroupBy`] facade. Low-level aggregation execution
//! remains available through [`groupby_aggregate`].

mod agg;
mod dataframe;

pub use agg::{Aggregation, AggregationRequest, GroupByResult, NullPolicy, groupby_aggregate};
pub use dataframe::{Agg, AggSpec, GroupBy, GroupByOptions};
