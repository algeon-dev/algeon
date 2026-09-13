pub use crate::interop::arrow_device::testing_hooks::{
    arrow_device_cpu_type_for_test, arrow_device_cuda_host_type_for_test, arrow_device_cuda_managed_type_for_test,
    quarantine_owner_unknown_device_array_for_test,
};
use crate::interop::{
    projection::testing_hooks::{
        trigger_active_range_overflow_for_test as projection_trigger_active_range_overflow_for_test,
        trigger_dictionary_indices_width_mismatch_for_test as projection_trigger_dictionary_indices_width_mismatch_for_test,
        trigger_host_offsets_too_small_for_test as projection_trigger_host_offsets_too_small_for_test,
        trigger_list_offsets_width_mismatch_for_test as projection_trigger_list_offsets_width_mismatch_for_test,
        trigger_missing_null_mask_for_test as projection_trigger_missing_null_mask_for_test,
        trigger_missing_string_offsets_child_for_test as projection_trigger_missing_string_offsets_child_for_test,
        trigger_negative_offset_for_test as projection_trigger_negative_offset_for_test,
        trigger_struct_missing_child_for_test as projection_trigger_struct_missing_child_for_test,
    },
    testing_hooks::trigger_column_index_limit_for_test as interop_trigger_column_index_limit_for_test,
};

/// Trigger the shared interop projection active-range overflow contract used
/// by integration tests.
pub fn trigger_interop_projection_active_range_overflow_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_active_range_overflow_for_test()
}

/// Trigger the shared interop projection negative-offset validation contract.
pub fn trigger_interop_projection_negative_offset_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_negative_offset_for_test()
}

/// Trigger the shared interop projection missing-string-offsets-child
/// contract.
pub fn trigger_interop_projection_missing_string_offsets_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_missing_string_offsets_child_for_test()
}

/// Trigger the shared interop projection list-offsets-width contract.
pub fn trigger_interop_projection_list_offsets_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_list_offsets_width_mismatch_for_test()
}

/// Trigger the shared interop projection dictionary-indices-width contract.
pub fn trigger_interop_projection_dictionary_indices_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_dictionary_indices_width_mismatch_for_test()
}

/// Trigger the shared interop projection host-offsets-range contract.
pub fn trigger_interop_projection_host_offsets_too_small_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_host_offsets_too_small_for_test()
}

/// Trigger the shared interop projection missing-null-mask contract.
pub fn trigger_interop_projection_missing_null_mask_for_test() -> std::result::Result<(), rapids_interop::InteropError>
{
    projection_trigger_missing_null_mask_for_test()
}

/// Trigger the shared interop projection struct-child-count contract.
pub fn trigger_interop_projection_struct_missing_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_trigger_struct_missing_child_for_test()
}

/// Trigger the shared interop invalid-view column-index i32-limit contract.
pub fn trigger_interop_column_index_limit_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    interop_trigger_column_index_limit_for_test()
}
