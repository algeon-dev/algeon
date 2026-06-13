use std::os::raw::c_void;

use crate::cudf_status_t;

extern "C" {
    // -- Device ---------------------------------------------------------------

    /// Query the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_get_device(device: *mut i32) -> cudf_status_t;

    /// Set the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_set_device(device: i32) -> cudf_status_t;

    /// Query the number of CUDA-capable devices visible to the process.
    pub fn cudf_cuda_get_device_count(count: *mut i32) -> cudf_status_t;

    /// Resolve the CUDA device ordinal that owns a device pointer.
    pub fn cudf_cuda_pointer_get_device(ptr: *const c_void, device: *mut i32) -> cudf_status_t;
}
