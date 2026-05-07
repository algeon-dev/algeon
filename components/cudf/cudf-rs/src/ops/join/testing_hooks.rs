use super::dataframe::{
    trigger_merge_reassembly_key_count_error_for_test_impl, trigger_projected_cross_join_input_row_limit_for_test_impl,
    trigger_projected_cross_join_output_row_overflow_for_test_impl,
};
use crate::{DataFrame, core::error::Result};

pub(crate) fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
    trigger_merge_reassembly_key_count_error_for_test_impl()
}

pub(crate) fn trigger_projected_cross_join_output_row_overflow_for_test() -> Result<usize> {
    trigger_projected_cross_join_output_row_overflow_for_test_impl()
}

pub(crate) fn trigger_projected_cross_join_input_row_limit_for_test() -> Result<usize> {
    trigger_projected_cross_join_input_row_limit_for_test_impl()
}
