pub(crate) fn arrow_device_cpu_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CPU as i32
}

pub(crate) fn arrow_device_cuda_host_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_HOST as i32
}

pub(crate) fn arrow_device_cuda_managed_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED as i32
}

pub(crate) fn quarantine_owner_unknown_device_array_for_test() {
    let dangling = std::ptr::NonNull::<cudf_sys::cudf_arrow_device_array_handle_state>::dangling();
    super::drop_device_arrow_array(dangling, None);
}
