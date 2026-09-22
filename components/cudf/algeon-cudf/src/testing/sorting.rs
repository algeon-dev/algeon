use crate::{
    core::Result,
    ops::sorting::testing_hooks::trigger_sort_key_count_limit_error_for_test as sorting_trigger_sort_key_count_limit_error_for_test,
};

/// Trigger the low-level sorting key-count limit validation path used by
/// integration tests.
pub fn trigger_sort_key_count_limit_error_for_test() -> Result<()> {
    sorting_trigger_sort_key_count_limit_error_for_test()
}
