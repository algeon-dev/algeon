use super::checked_num_keys;
use crate::core::error::Result;

/// Trigger the low-level sorting key-count limit validation path used by
/// integration tests.
pub fn trigger_sort_key_count_limit_error_for_test() -> Result<()> {
    checked_num_keys("sorting::sort", usize::MAX).map(|_| ())
}
