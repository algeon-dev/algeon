//! Reference-counted GPU column vector.
//!
//! `Series` wraps an `Arc<Column>` without schema metadata, providing the
//! high-level interface over low-level [`Column`](crate::core::Column) handles.
//! Cloning increments the reference count without copying GPU buffer memory.

mod constructors;
mod context;
mod datetime;
mod display;
mod host_column_data;
mod owned;
pub(crate) mod reduction;
mod strings;

pub use context::SeriesOps;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use context::{
    trigger_series_row_range_end_limit_error_for_test, trigger_series_row_range_overflow_error_for_test,
    trigger_series_row_range_start_limit_error_for_test,
};
pub use datetime::SeriesDatetimeOps;
pub use host_column_data::HostColumnData;
pub use owned::{IntoSeries, Series};
pub use reduction::{ScalarReductionOutput, ScalarReductionOutputShape, SeriesPreparedScalarReduction};
pub use strings::SeriesStringOps;
