use std::ffi::c_void;

pub(crate) fn arrow_device_cpu_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CPU
}

pub(crate) fn arrow_device_cuda_host_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_HOST
}

pub(crate) fn arrow_device_cuda_managed_type_for_test() -> i32 {
    cudf_sys::CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED
}

pub(crate) unsafe fn force_device_arrow_array_device_type_for_test(raw_device_array: *mut c_void, device_type: i32) {
    unsafe { cudf_sys::cudf_test_arrow_device_array_set_device_type(raw_device_array, device_type) }
}
