use std::ffi::c_void;

/// Return the Arrow C Device ABI device_type code for CPU roots.
pub fn arrow_device_cpu_type_for_test() -> i32 {
    crate::interop::arrow_device::testing_hooks::arrow_device_cpu_type_for_test()
}

/// Return the Arrow C Device ABI device_type code for CUDA_HOST roots.
pub fn arrow_device_cuda_host_type_for_test() -> i32 {
    crate::interop::arrow_device::testing_hooks::arrow_device_cuda_host_type_for_test()
}

/// Return the Arrow C Device ABI device_type code for CUDA_MANAGED roots.
pub fn arrow_device_cuda_managed_type_for_test() -> i32 {
    crate::interop::arrow_device::testing_hooks::arrow_device_cuda_managed_type_for_test()
}

/// Trigger the shared interop projection active-range overflow contract used
/// by integration tests.
pub fn trigger_interop_projection_active_range_overflow_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_active_range_overflow_for_test()
}

/// Trigger the shared interop projection negative-offset validation contract.
pub fn trigger_interop_projection_negative_offset_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_negative_offset_for_test()
}

/// Trigger the shared interop projection missing-string-offsets-child
/// contract.
pub fn trigger_interop_projection_missing_string_offsets_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_missing_string_offsets_child_for_test()
}

/// Trigger the shared interop projection list-offsets-width contract.
pub fn trigger_interop_projection_list_offsets_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_list_offsets_width_mismatch_for_test()
}

/// Trigger the shared interop projection dictionary-indices-width contract.
pub fn trigger_interop_projection_dictionary_indices_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_dictionary_indices_width_mismatch_for_test()
}

/// Trigger the shared interop projection host-offsets-range contract.
pub fn trigger_interop_projection_host_offsets_too_small_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_host_offsets_too_small_for_test()
}

/// Trigger the shared interop projection missing-null-mask contract.
pub fn trigger_interop_projection_missing_null_mask_for_test() -> std::result::Result<(), rapids_interop::InteropError>
{
    crate::interop::projection::testing_hooks::trigger_missing_null_mask_for_test()
}

/// Trigger the shared interop projection struct-child-count contract.
pub fn trigger_interop_projection_struct_missing_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::projection::testing_hooks::trigger_struct_missing_child_for_test()
}

/// Trigger the shared interop invalid-view column-index i32-limit contract.
pub fn trigger_interop_column_index_limit_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    crate::interop::testing_hooks::trigger_column_index_limit_for_test()
}

/// Testing-only hook: overwrite one owned ArrowDeviceArray handle's
/// `device_type` field in place.
///
/// # Safety
///
/// `raw_device_array` must be a valid ArrowDeviceArray handle produced by
/// this crate's Arrow device bridge and still owned by a live Rust wrapper.
pub unsafe fn force_device_arrow_array_device_type_for_test(raw_device_array: *mut c_void, device_type: i32) {
    unsafe {
        crate::interop::arrow_device::testing_hooks::force_device_arrow_array_device_type_for_test(
            raw_device_array,
            device_type,
        )
    }
}
