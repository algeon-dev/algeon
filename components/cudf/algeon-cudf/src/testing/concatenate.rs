use crate::{
    core::Result,
    ops::concatenate::{
        trigger_concatenate_column_count_limit_error_for_test as concatenate_trigger_column_count_limit_error_for_test,
        trigger_concatenate_table_count_limit_error_for_test as concatenate_trigger_table_count_limit_error_for_test,
    },
};

/// Trigger the concatenate-column input-count limit validation contract.
pub fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    concatenate_trigger_column_count_limit_error_for_test()
}

/// Trigger the concatenate-table input-count limit validation contract.
pub fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    concatenate_trigger_table_count_limit_error_for_test()
}
