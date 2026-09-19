use super::{InteropError, checked_column_index};

pub(crate) fn trigger_column_index_limit_for_test() -> Result<(), InteropError> {
    checked_column_index("testing::interop::column_index_limit", usize::MAX).map(|_| ())
}
