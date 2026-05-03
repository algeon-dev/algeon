use std::os::raw::c_void;

use crate::{cudf_memory_resource_handle, cudf_status_t};

extern "C" {
    // -- Memory Resource -----------------------------------------------------

    /// Create an owned CUDA-backed RMM memory resource handle.
    ///
    /// Attach this handle to [`crate::ExecutionContext`] or
    /// [`crate::cudf_exec_options_t`] when calling `_ex` APIs to direct
    /// allocations through it.
    pub fn cudf_memory_resource_create_cuda(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    /// Create an owned pool memory resource handle.
    ///
    /// Like [`cudf_memory_resource_create_cuda`], this creates an owned
    /// resource object without changing the process-global allocator.
    pub fn cudf_memory_resource_create_pool(
        initial_pool_size: usize,
        maximum_pool_size: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Destroy a memory resource created by this shim on its owning device.
    pub fn cudf_memory_resource_destroy(handle: cudf_memory_resource_handle, device: i32) -> cudf_status_t;

    /// Borrow the underlying raw `device_memory_resource*` from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_memory_resource_raw(handle: cudf_memory_resource_handle, out: *mut *mut c_void) -> cudf_status_t;
}
