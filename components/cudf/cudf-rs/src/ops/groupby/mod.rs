//! GPU-accelerated groupby aggregation.
//!
//! Most users should use [`DataFrame::groupby`](crate::DataFrame::groupby)
//! and the high-level [`GroupBy`] API. Low-level aggregation execution
//! remains available through [`groupby_aggregate`].

mod agg;
mod dataframe;
mod errors;
mod output;
mod shift;
pub(crate) mod testing_hooks;

pub(crate) use agg::groupby_aggregate_impl;
pub use agg::{
    AggregationRequest, GroupByAgg, GroupByAggregationOutput, GroupByScanOutput, GroupByScanRequest, NullPolicy,
    groupby_aggregate, groupby_scan_sorted,
};
pub use dataframe::{AggSpec, GroupBy, GroupByExecutionScope, GroupByOptions, ShiftSpec};
pub(crate) use output::value_counts_result_columns;
pub use shift::{GroupByShiftOutput, GroupByShiftRequest, groupby_shift};
