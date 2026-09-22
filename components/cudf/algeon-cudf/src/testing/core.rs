pub use crate::core::{
    error::testing_hooks::{
        ffi_cuda_status_error_for_test, ffi_io_status_error_for_test, trigger_canonical_key_in_detail_path_for_test,
        trigger_if_absent_error_annotation_contract_for_test, trigger_invariant_violation_for_test,
        trigger_null_handle_contract_error_for_test, trigger_range_limit_error_for_test,
        trigger_reanchor_if_needed_same_operation_contract_for_test, trigger_same_operation_reanchor_contract_for_test,
        trigger_truthful_error_contract_for_test, trigger_with_column_error_annotation_contract_for_test,
    },
    table::testing_hooks::{
        trigger_table_column_count_limit_error_for_test, trigger_table_view_column_count_limit_error_for_test,
    },
    testing_hooks::validity_to_host_null_mask_bytes,
};
use crate::{
    DeviceId,
    core::{DeviceMemoryResource, Result, scalar::trigger_negative_string_length_error_for_test},
};

/// Adopt an owned test-created memory-resource handle.
///
/// This adapter does not construct a native test resource. Test code must use
/// `cudf_sys::test_support` to create the handle, then transfer its unique
/// ownership here so the normal safe owner performs device-aware teardown.
///
/// # Safety
///
/// `raw` must be a non-null, uniquely owned `cudf_memory_resource_handle`
/// created for `device`. It must not have been adopted elsewhere, and no
/// borrowed view of it may outlive the returned owner.
pub unsafe fn adopt_owned_memory_resource_for_test(
    raw: cudf_sys::cudf_memory_resource_handle,
    device: DeviceId,
) -> Result<DeviceMemoryResource> {
    unsafe { DeviceMemoryResource::adopt_owned_for_testing(raw, device) }
}

/// Trigger the string-scalar negative-length foreign metadata path used by
/// integration tests.
pub fn trigger_negative_string_scalar_length_for_test() -> Result<()> {
    trigger_negative_string_length_error_for_test()
}
