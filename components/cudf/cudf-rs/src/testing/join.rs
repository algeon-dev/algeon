use crate::{core::Result, frame::dataframe::DataFrame};

/// Trigger the merge-result reassembly invariant branch used by integration
/// tests to lock the downstream error contract.
pub fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
    crate::ops::join::testing_hooks::trigger_merge_reassembly_key_count_error_for_test()
}

/// Trigger projected cross-join output row-count overflow before reservation
/// or materialization.
pub fn trigger_projected_cross_join_output_row_overflow_for_test() -> Result<usize> {
    crate::ops::join::testing_hooks::trigger_projected_cross_join_output_row_overflow_for_test()
}

/// Trigger projected cross-join input row-count rejection when output rows
/// would otherwise be zero.
pub fn trigger_projected_cross_join_input_row_limit_for_test() -> Result<usize> {
    crate::ops::join::testing_hooks::trigger_projected_cross_join_input_row_limit_for_test()
}
