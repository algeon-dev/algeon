use super::table_column_count_limit_error;
use crate::core::error::Result;

/// Trigger the stable owned-table column-count limit validation contract.
pub fn trigger_table_column_count_limit_error_for_test() -> Result<()> {
    Err(table_column_count_limit_error("table::from_columns", "number of columns exceeds i32::MAX", usize::MAX))
}

/// Trigger the stable borrowed-table-view column-count limit validation
/// contract.
pub fn trigger_table_view_column_count_limit_error_for_test() -> Result<()> {
    Err(table_column_count_limit_error(
        "table_view::from_column_views",
        "number of column views exceeds i32::MAX",
        usize::MAX,
    ))
}
