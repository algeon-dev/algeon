//! GPU-accelerated groupby aggregation.
//!
//! Run groupby work through [`DataFrame::groupby`](crate::DataFrame::groupby)
//! and [`GroupBy::on`]; this module holds the aggregation and option types
//! those calls take.

mod agg;
mod dataframe;
mod errors;
mod output;
mod shift;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;

pub(crate) use agg::{
    AggregationRequest, GroupByAggregationOutput, GroupByScanRequest, groupby_aggregate_impl, groupby_scan_sorted,
};
pub use agg::{GroupByAgg, NullPolicy, PreparedGroupByFacts};
pub use dataframe::{AggSpec, GroupBy, GroupByOps, GroupByOptions, PreparedDataFrameGroupBy, ShiftSpec};
pub(crate) use output::value_counts_result_columns;
