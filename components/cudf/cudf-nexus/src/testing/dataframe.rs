use crate::{
    Error,
    core::{Result, Table},
    frame::dataframe::DataFrame,
};

/// Testing-only: build the canonical `DataFrame::new` column-count mismatch
/// error shape used by integration-test helpers.
pub fn new_column_length_mismatch_error_for_test(column_name_count: usize, table_column_count: usize) -> Error {
    crate::core::schema::column_name_count_mismatch_error(
        "dataframe::new",
        "number of names must equal number of columns",
        column_name_count,
        table_column_count,
    )
}

/// Trigger the internal DataFrame reassembly helper directly for contract
/// tests.
pub fn reassemble_dataframe_from_table_and_names_for_test(table: Table, names: Vec<String>) -> Result<DataFrame> {
    crate::internal::testing_hooks::reassemble_dataframe_from_table_and_names_for_test(table, names)
}

/// Trigger the high-level DataFrame dropna subset-count limit validation
/// contract.
pub fn trigger_dataframe_dropna_subset_count_limit_error_for_test() -> Result<()> {
    crate::frame::dataframe::trigger_dataframe_dropna_subset_count_limit_error_for_test()
}

/// Trigger the DataFrame row-range start limit path used by integration tests.
pub fn trigger_dataframe_row_range_start_limit_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_start_limit_error_for_test()
}

/// Trigger the DataFrame row-range end limit path used by integration tests.
pub fn trigger_dataframe_row_range_end_limit_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_end_limit_error_for_test()
}

/// Trigger the DataFrame row-range overflow path used by integration tests.
pub fn trigger_dataframe_row_range_overflow_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_overflow_error_for_test()
}

/// Trigger the concat-by-name missing planned dtype invariant path used by
/// integration tests.
pub fn trigger_concat_by_name_missing_planned_dtype_for_test() -> Result<()> {
    crate::frame::dataframe::trigger_concat_by_name_missing_planned_dtype_for_test()
}
