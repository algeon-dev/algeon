//! Device memory resource management (RMM).

use crate::error::{self, Result};
use std::ptr::{self, NonNull};

/// An owned device memory resource.
///
/// Wraps an `rmm::mr::device_memory_resource` via the C shim.
pub struct DeviceMemoryResource {
    handle: NonNull<std::ffi::c_void>,
}

unsafe impl Send for DeviceMemoryResource {}
unsafe impl Sync for DeviceMemoryResource {}

impl DeviceMemoryResource {
    /// Create a CUDA (default) memory resource.
    pub fn cuda() -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status(unsafe {
            cudf_sys::cudf_memory_resource_create_cuda(&mut handle)
        })?;
        Ok(Self {
            handle: NonNull::new(handle)
                .expect("cudf_memory_resource_create_cuda returned null on success"),
        })
    }

    /// Create a pool memory resource.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or 0 for unbounded.
    pub fn pool(initial_pool_size: usize, maximum_pool_size: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status(unsafe {
            cudf_sys::cudf_memory_resource_create_pool(
                initial_pool_size,
                maximum_pool_size,
                &mut handle,
            )
        })?;
        Ok(Self {
            handle: NonNull::new(handle)
                .expect("cudf_memory_resource_create_pool returned null on success"),
        })
    }
}

impl Drop for DeviceMemoryResource {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_memory_resource_destroy(self.handle.as_ptr());
        }
    }
}
