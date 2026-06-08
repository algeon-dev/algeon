//! GPU-accelerated groupby aggregation.
//!
//! Most users should use [`DataFrame::groupby`](crate::DataFrame::groupby)
//! and the high-level [`GroupBy`] API. Low-level aggregation execution
//! remains available through [`groupby_aggregate`].

mod agg;
mod dataframe;

pub use agg::{AggregationRequest, GroupByAgg, GroupByResult, NullPolicy, groupby_aggregate};
pub use dataframe::{AggSpec, GroupBy, GroupByExecutionScope, GroupByOptions};
