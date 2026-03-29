//! Device memory resource management (RMM).
//!
//! These handles back explicit retained execution domains via [`crate::Runtime`]
//! and borrowed overrides via [`crate::ExecutionContext`].
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit memory-resource
//! handles, and the current high-level public slice can route selected
//! operations and constructors through them.
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

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::error::{self, Result};

/// An owned device memory resource.
///
/// Wraps an `rmm::mr::device_memory_resource` via the C shim.
///
/// Use this with [`crate::Runtime::builder`] to seed a retained explicit
/// execution domain, or with [`crate::ExecutionContext::with_memory_resource`]
/// to override allocator selection for one supported operation.
#[derive(Clone)]
pub struct DeviceMemoryResource {
    handle: Arc<DeviceMemoryResourceHandle>,
}

struct DeviceMemoryResourceHandle {
    raw: NonNull<std::ffi::c_void>,
}

// SAFETY: RMM device_memory_resource implementations are thread-safe — allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

impl DeviceMemoryResource {
    /// Create a CUDA (default) memory resource handle.
    pub fn cuda() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut handle) },
            "memory::device_memory_resource::cuda",
        )?;
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: NonNull::new(handle).expect("cudf_memory_resource_create_cuda returned null on success"),
            }),
        })
    }

    /// Create a pool memory resource handle.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or 0 for unbounded.
    pub fn pool(initial_pool_size: usize, maximum_pool_size: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with(
            unsafe { cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, maximum_pool_size, &mut handle) },
            "memory::device_memory_resource::pool",
            [
                ("initial_pool_size", initial_pool_size.to_string()),
                ("maximum_pool_size", maximum_pool_size.to_string()),
            ],
        )?;
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: NonNull::new(handle).expect("cudf_memory_resource_create_pool returned null on success"),
            }),
        })
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    #[doc(hidden)]
    pub fn __debug_raw_handle_value(&self) -> usize {
        self.handle.raw.as_ptr() as usize
    }
}

impl Drop for DeviceMemoryResourceHandle {
    fn drop(&mut self) {
        // SAFETY: self.raw is valid (NonNull from construction). The shared
        // handle is destroyed exactly once when the last clone drops.
        unsafe {
            cudf_sys::cudf_memory_resource_destroy(self.raw.as_ptr());
        }
    }
}
