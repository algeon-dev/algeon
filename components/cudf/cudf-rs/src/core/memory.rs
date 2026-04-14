//! Device memory resource management (RMM).
//!
//! These handles back explicit retained provenance via [`crate::Runtime`] and
//! borrowed overrides via [`crate::ExecutionContext`].
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit memory-resource
//! handles, and the current high-level public slice can route selected
//! operations and constructors through them.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DeviceId, Runtime, core::memory::DeviceMemoryResource};
//!
//! // Canonical runtime setup: let the builder create the pool after it has
//! // resolved the runtime's target device.
//! let _runtime = Runtime::builder()
//!     .with_device(DeviceId::new(0).unwrap())
//!     .with_pool_memory_resource(256 * 1024 * 1024, None)
//!     .build()
//!     .unwrap();
//!
//! // Advanced explicit-handle setup: create an MR immediately on a chosen device.
//! let _pool = DeviceMemoryResource::pool_on(
//!     DeviceId::new(0).unwrap(),
//!     256 * 1024 * 1024, // 256 MB initial pool
//!     None,              // None = unbounded maximum
//! )
//! .unwrap();
//! ```

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::device_guard::DeviceGuard,
};

/// An owned device memory resource.
///
/// Wraps an `rmm::mr::device_memory_resource` via the C shim.
///
/// Each resource records the CUDA device it was created on. This provenance is
/// used by [`crate::Runtime`] builder validation to reject cross-device
/// handle combinations.
///
/// Use this with [`crate::Runtime::builder`] to seed explicit retained
/// provenance, or with [`crate::ExecutionContext::with_memory_resource`]
/// to override allocator selection for one supported operation.
#[derive(Clone)]
pub struct DeviceMemoryResource {
    handle: Arc<DeviceMemoryResourceHandle>,
}

struct DeviceMemoryResourceHandle {
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

// SAFETY: RMM device_memory_resource implementations are thread-safe — allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

impl DeviceMemoryResource {
    fn raw_maximum_pool_size(maximum_pool_size: Option<usize>) -> usize {
        maximum_pool_size.unwrap_or(0)
    }

    fn from_created_handle(handle: *mut std::ffi::c_void, device: DeviceId, constructor: &'static str) -> Self {
        Self { handle: Arc::new(DeviceMemoryResourceHandle { raw: NonNull::new(handle).expect(constructor), device }) }
    }

    /// Create a CUDA (default) memory resource handle on the calling thread's
    /// current device.
    pub fn cuda() -> Result<Self> {
        Self::cuda_on(DeviceId::current()?)
    }

    /// Create a CUDA (default) memory resource handle on an explicit device.
    ///
    /// The resource is created immediately on `device`, and the calling
    /// thread's previous ambient CUDA device is restored before this function
    /// returns.
    pub fn cuda_on(device: DeviceId) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut handle) },
            "memory::device_memory_resource::cuda_on",
        )?;
        Ok(Self::from_created_handle(handle, device, "cudf_memory_resource_create_cuda returned null on success"))
    }

    /// Create a pool memory resource handle on the calling thread's current
    /// device.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn pool(initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Result<Self> {
        Self::pool_on(DeviceId::current()?, initial_pool_size, maximum_pool_size)
    }

    /// Create a pool memory resource handle on an explicit device.
    ///
    /// The resource is created immediately on `device`, and the calling
    /// thread's previous ambient CUDA device is restored before this function
    /// returns.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn pool_on(device: DeviceId, initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let raw_maximum_pool_size = Self::raw_maximum_pool_size(maximum_pool_size);
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with(
            unsafe {
                cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, raw_maximum_pool_size, &mut handle)
            },
            "memory::device_memory_resource::pool_on",
            [
                ("initial_pool_size", initial_pool_size.to_string()),
                ("maximum_pool_size", maximum_pool_size.map_or_else(|| "None".to_string(), |size| size.to_string())),
            ],
        )?;
        Ok(Self::from_created_handle(handle, device, "cudf_memory_resource_create_pool returned null on success"))
    }

    /// The CUDA device this memory resource was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Borrow this owner as a low-level `cudf-sys` memory-resource handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `DeviceMemoryResource` remains responsible for
    /// destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.handle.raw.as_ptr()) }
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_raw(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::interop_raw_handle",
        )?;
        Ok(out)
    }

    #[doc(hidden)]
    pub fn __debug_raw_handle_value(&self) -> usize {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_raw(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::__debug_raw_handle_value",
        )
        .expect("failed to read raw device memory resource pointer");
        out as usize
    }

    #[doc(hidden)]
    pub fn __test_stats() -> Result<Self> {
        let device = DeviceId::current()?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_create_stats(&mut handle) },
            "memory::device_memory_resource::__test_stats",
        )?;
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: NonNull::new(handle).expect("cudf_test_memory_resource_create_stats returned null on success"),
                device,
            }),
        })
    }

    #[doc(hidden)]
    pub fn __test_allocation_count(&self) -> Result<i64> {
        let mut out = 0_i64;
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_allocation_count(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::__test_allocation_count",
        )?;
        Ok(out)
    }
}

impl Drop for DeviceMemoryResourceHandle {
    fn drop(&mut self) {
        let status = unsafe { cudf_sys::cudf_memory_resource_destroy(self.raw.as_ptr(), self.device.ordinal_raw()) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!(
                "[cudf::DeviceMemoryResourceHandle] warning: failed to destroy memory resource on device {}",
                self.device.ordinal_raw()
            );
        }
    }
}

impl std::fmt::Debug for DeviceMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryResource")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
