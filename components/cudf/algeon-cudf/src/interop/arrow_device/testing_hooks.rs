/// Return the Arrow C Device ABI device_type code for CPU roots.
pub fn arrow_device_cpu_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CPU as i32
}

/// Return the Arrow C Device ABI device_type code for CUDA_HOST roots.
pub fn arrow_device_cuda_host_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_HOST as i32
}

/// Return the Arrow C Device ABI device_type code for CUDA_MANAGED roots.
pub fn arrow_device_cuda_managed_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED as i32
}

/// Exercise owner-unknown Arrow device-array teardown with a pointer that must
/// never cross FFI.
pub fn quarantine_owner_unknown_device_array_for_test() {
    let dangling = std::ptr::NonNull::<cudf_sys::cudf_arrow_device_array_handle_state>::dangling();
    super::drop_device_arrow_array(dangling, None);
}
