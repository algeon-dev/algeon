//! Device memory resource management (RMM).
//!
//! # Example
//!
//! ```no_run
//! use cudf::memory::DeviceMemoryResource;
//!
//! // Use the default CUDA allocator.
//! let _mr = DeviceMemoryResource::cuda().unwrap();
//!
//! // Or use a pool allocator for faster repeated allocations.
//! let _pool = DeviceMemoryResource::pool(
//!     256 * 1024 * 1024, // 256 MB initial pool
//!     0,                 // 0 = unbounded maximum
//! )
//! .unwrap();
//! ```

use std::ptr::{self, NonNull};

use crate::error::{self, Result};

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
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut handle) },
            "memory::device_memory_resource::cuda",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("cudf_memory_resource_create_cuda returned null on success") })
    }

    /// Create a pool memory resource.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or 0 for unbounded.
    pub fn pool(initial_pool_size: usize, maximum_pool_size: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in_with(
            unsafe { cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, maximum_pool_size, &mut handle) },
            "memory::device_memory_resource::pool",
            [
                ("initial_pool_size", initial_pool_size.to_string()),
                ("maximum_pool_size", maximum_pool_size.to_string()),
            ],
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("cudf_memory_resource_create_pool returned null on success") })
    }
}

impl Drop for DeviceMemoryResource {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_memory_resource_destroy(self.handle.as_ptr());
        }
    }
}
