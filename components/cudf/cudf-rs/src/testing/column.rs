use crate::{core::Result, testing::ColumnView};

/// Construct a malformed non-empty STRING column view without an offsets
/// child so integration tests can lock down foreign-contract validation.
pub fn synthetic_string_view_without_children(size: usize) -> Result<ColumnView<'static>> {
    crate::core::column::synthetic_string_view_without_children(size)
}

/// Trigger the internal raw-byte materialization helper on a borrowed column
/// view so integration tests can lock down its downstream error contract.
pub fn column_view_to_raw_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_raw_bytes_for_test(view)
}

/// Trigger the internal fixed-width host-byte materialization helper on a
/// borrowed column view so integration tests can lock down its downstream
/// error contract.
pub fn column_view_to_fixed_width_host_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_fixed_width_host_bytes_for_test(view)
}

/// Trigger the fixed-width host-byte materialization overflow contract used by
/// integration tests.
pub fn trigger_column_view_fixed_width_host_byte_range_overflow_for_test() -> Result<()> {
    crate::core::column::trigger_fixed_width_host_byte_range_overflow_for_test()
}
