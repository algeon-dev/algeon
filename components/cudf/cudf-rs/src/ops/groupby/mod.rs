//! GPU-accelerated groupby aggregation.
//!
//! Most users should use [`DataFrame::groupby`](crate::DataFrame::groupby)
//! and the high-level [`GroupBy`] API. Low-level aggregation execution
//! remains available through [`groupby_aggregate`].

mod agg;
mod dataframe;

pub(crate) use agg::groupby_aggregate_impl;
pub use agg::{AggregationRequest, GroupByAgg, GroupByAggregationOutput, NullPolicy, groupby_aggregate};
pub use dataframe::{AggSpec, GroupBy, GroupByExecutionScope, GroupByOptions};
