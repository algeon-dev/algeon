use crate::{core::Result, frame::dataframe::DataFrame};

/// Trigger the `Series::value_counts` missing-key-columns result-shape path
/// used by integration tests.
pub fn trigger_value_counts_missing_key_columns_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_missing_key_columns_for_test()
}

/// Trigger the `Series::value_counts` missing-count-results result-shape path
/// used by integration tests.
pub fn trigger_value_counts_missing_count_results_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_missing_count_results_for_test()
}

/// Trigger the `Series::value_counts` empty-count-result result-shape path
/// used by integration tests.
pub fn trigger_value_counts_empty_count_result_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_empty_count_result_for_test()
}

/// Trigger the high-level groupby result-group-index invariant path used by
/// integration tests.
pub fn trigger_groupby_result_group_index_out_of_bounds_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_result_group_index_out_of_bounds_for_test()
}

/// Trigger the high-level groupby result-column-index invariant path used by
/// integration tests.
pub fn trigger_groupby_result_column_index_out_of_bounds_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_result_column_index_out_of_bounds_for_test()
}

/// Trigger the high-level groupby duplicate result-column consumption
/// invariant path used by integration tests.
pub fn trigger_groupby_duplicate_result_column_consumption_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_duplicate_result_column_consumption_for_test()
}

/// Trigger the low-level groupby request-count limit validation path used by
/// integration tests.
pub fn trigger_groupby_request_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_request_count_limit_error_for_test()
}

/// Trigger the low-level groupby per-request aggregation-count limit
/// validation path used by integration tests.
pub fn trigger_groupby_request_agg_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_request_agg_count_limit_error_for_test()
}

/// Trigger the low-level groupby total aggregation-handle count limit
/// validation path used by integration tests.
pub fn trigger_groupby_total_agg_handle_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_total_agg_handle_count_limit_error_for_test()
}
