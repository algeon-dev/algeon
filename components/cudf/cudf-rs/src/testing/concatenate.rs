use crate::core::Result;

/// Trigger the concatenate-column input-count limit validation contract.
pub fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    crate::ops::concatenate::trigger_concatenate_column_count_limit_error_for_test()
}

/// Trigger the concatenate-table input-count limit validation contract.
pub fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    crate::ops::concatenate::trigger_concatenate_table_count_limit_error_for_test()
}
