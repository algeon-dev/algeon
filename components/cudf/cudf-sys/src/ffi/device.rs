use std::os::raw::{c_char, c_void};

use crate::cudf_status_t;

pub const CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 1;
pub const CUDF_DEVICE_ATTRIBUTE_MANAGED_MEMORY: i32 = 2;
pub const CUDF_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: i32 = 3;

extern "C" {
    // -- Device ---------------------------------------------------------------

    /// Query the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_get_device(device: *mut i32) -> cudf_status_t;

    /// Set the CUDA device ordinal for the calling thread.
    pub fn cudf_cuda_set_device(device: i32) -> cudf_status_t;

    /// Start profiler collection for the current process.
    pub fn cudf_cuda_profiler_start() -> cudf_status_t;

    /// Stop profiler collection for the current process.
    pub fn cudf_cuda_profiler_stop() -> cudf_status_t;

    /// Query the number of CUDA-capable devices visible to the process.
    pub fn cudf_cuda_get_device_count(count: *mut i32) -> cudf_status_t;

    /// Resolve the CUDA device ordinal that owns a device pointer.
    pub fn cudf_cuda_pointer_get_device(ptr: *const c_void, device: *mut i32) -> cudf_status_t;

    /// Query the CUDA PCI bus id for a CUDA device.
    pub fn cudf_cuda_device_pci_bus_id(device: i32, out: *mut c_char, out_len: usize) -> cudf_status_t;

    /// Query free and total memory for a CUDA device.
    pub fn cudf_device_memory_info(device: i32, free_bytes: *mut usize, total_bytes: *mut usize) -> cudf_status_t;

    /// Query a CUDA device attribute selected by CUDF_DEVICE_ATTRIBUTE_*.
    pub fn cudf_device_attribute(device: i32, attr: i32, out: *mut i32) -> cudf_status_t;
}
