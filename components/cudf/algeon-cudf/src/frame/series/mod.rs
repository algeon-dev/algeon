//! An `Arc`-shared GPU column vector; the primary user-facing 1D type.
//!
//! `Series` wraps an `Arc<Column>` without schema metadata. It is the
//! high-level counterpart to the low-level owned [`Column`](crate::core::Column)
//! type.
//!
//! Clone is cheap: it bumps the `Arc` reference count; no GPU data is copied.

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
