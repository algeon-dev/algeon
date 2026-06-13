use crate::{Series, core::Result};

/// Trigger the Series row-range start limit path used by integration tests.
pub fn trigger_series_row_range_start_limit_error_for_test() -> Result<Series> {
    crate::frame::series::trigger_series_row_range_start_limit_error_for_test()
}

/// Trigger the Series row-range end limit path used by integration tests.
pub fn trigger_series_row_range_end_limit_error_for_test() -> Result<Series> {
    crate::frame::series::trigger_series_row_range_end_limit_error_for_test()
}

/// Trigger the Series row-range overflow path used by integration tests.
pub fn trigger_series_row_range_overflow_error_for_test() -> Result<Series> {
    crate::frame::series::trigger_series_row_range_overflow_error_for_test()
}

/// Trigger the `Series::nunique` negative-distinct-count foreign-contract path
/// used by integration tests.
pub fn trigger_nunique_negative_distinct_count_for_test() -> Result<()> {
    crate::frame::series::reduction::trigger_nunique_negative_distinct_count_for_test()
}
