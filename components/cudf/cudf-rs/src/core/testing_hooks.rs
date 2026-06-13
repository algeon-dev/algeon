use std::ffi::c_void;

use super::{
    error::{self, Result},
    memory::DeviceMemoryResource,
    types,
};

pub(crate) fn stats_memory_resource() -> Result<DeviceMemoryResource> {
    DeviceMemoryResource::__test_stats()
}

pub(crate) fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    types::validity_to_ffi_bitmask(validity)
}

pub(crate) fn trigger_invariant_violation_for_test() -> Result<()> {
    Err(error::invariant_violation(
        "testing::trigger_invariant_violation_for_test",
        "injected invariant violation for tests",
    ))
}

pub(crate) fn trigger_null_handle_contract_error_for_test() -> Result<()> {
    error::require_non_null::<c_void>(
        std::ptr::null_mut(),
        "testing::trigger_null_handle_contract_error_for_test",
        "test_handle",
    )
    .map(|_| ())
}

pub(crate) fn trigger_range_limit_error_for_test() -> Result<()> {
    error::usize_to_i32(usize::MAX, "testing::trigger_range_limit_error_for_test", "row_count").map(|_| ())
}
