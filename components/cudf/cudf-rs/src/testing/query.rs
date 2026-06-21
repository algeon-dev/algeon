use crate::core::Result;

/// Trigger the query count scalar range-limit path used by integration tests.
pub fn trigger_count_scalar_overflow_for_test() -> Result<()> {
    crate::query::testing_hooks::trigger_count_scalar_overflow_for_test()
}
