use crate::{core::Result, frame::dataframe::DataFrame};

/// Trigger the merge-result reassembly invariant branch used by integration
/// tests to lock the downstream error contract.
pub fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
    crate::ops::join::testing_hooks::trigger_merge_reassembly_key_count_error_for_test()
}
