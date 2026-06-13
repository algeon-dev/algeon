use crate::core::Result;

/// Trigger the low-level sorting key-count limit validation path used by
/// integration tests.
pub fn trigger_sort_key_count_limit_error_for_test() -> Result<()> {
    crate::ops::sorting::testing_hooks::trigger_sort_key_count_limit_error_for_test()
}
