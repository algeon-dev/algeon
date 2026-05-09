//! Device memory resource management (RMM).
//!
//! A device memory resource is an RMM allocator handle.
//!
//! In `cudf-rs`, these handles participate in execution in two distinct ways:
//!
//! - [`crate::Runtime`] owns one default memory resource as part of its
//!   retained-provenance bundle for a workflow.
//! - [`crate::ExecutionOverrides`] can borrow `&DeviceMemoryResource` to
//!   override allocator selection for one supported operation without mutating
//!   the runtime's default.
//!
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit
//! memory-resource handles, and `cudf-rs` routes supported operations and
//! constructors through the resolved allocator for the active runtime/override
//! context.
//!
//! Prefer [`crate::RuntimeBuilder::with_pool_memory_resource`] when you want
//! the runtime builder to create and own a pool after it has resolved the
//! target CUDA device. Use [`DeviceMemoryResource::pool_on`] or
//! [`DeviceMemoryResource::cuda_on`] when you need an explicit handle first so
//! you can install it into one or more runtimes, or borrow it for a one-off
//! override. In all cases, the runtime, stream, and memory resource must be on
//! the same CUDA device.
//!
//! # Examples
//!
//! ```no_run
//! use cudf::{
//!     DeviceId, ExecutionOverrides, Runtime, SortOptions,
//!     core::memory::DeviceMemoryResource,
//! };
//!
//! let device = DeviceId::new(0).unwrap();
//!
//! // Recommended: let the builder create and own the default allocator for
//! // this workflow after it has resolved the runtime's target device.
//! let runtime = Runtime::builder()
//!     .with_device(device)
//!     .with_pool_memory_resource(256 * 1024 * 1024, None)
//!     .build()
//!     .unwrap();
//!
//! // Advanced: create an allocator handle immediately on an explicit device.
//! let shared_pool = DeviceMemoryResource::pool_on(
//!     device,
//!     512 * 1024 * 1024, // 512 MiB initial pool
//!     None,              // None = unbounded maximum
//! )
//! .unwrap();
//!
//! // Install that handle as another runtime's owned default allocator.
//! let runtime_with_explicit_mr = Runtime::builder()
//!     .with_device(device)
//!     .with_memory_resource(shared_pool.clone())
//!     .build()
//!     .unwrap();
//!
//! // Or borrow it for one supported operation without changing the runtime's
//! // default allocator.
//! let series = runtime.series(&[3i32, 1, 2]).unwrap();
//! let ctx = ExecutionOverrides::new().with_memory_resource(&shared_pool);
//! let sorted = series.sort_values_in(SortOptions::new(), &ctx).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(runtime_with_explicit_mr.memory_resource().device(), device);
//! assert_eq!(sorted.len(), 3);
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

/// Usage counters reported by a production budget-tracking memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryResourceUsage {
    pub allocation_limit_bytes: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_bytes: u64,
    pub current_aligned_bytes: u64,
    pub peak_aligned_bytes: u64,
    pub total_allocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub failed_allocation_count: u64,
    pub largest_failed_allocation_bytes: u64,
}

impl From<cudf_sys::cudf_memory_resource_usage_t> for DeviceMemoryResourceUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            allocation_limit_bytes: value.allocation_limit_bytes,
            current_bytes: value.current_bytes,
            peak_bytes: value.peak_bytes,
            total_bytes: value.total_bytes,
            current_aligned_bytes: value.current_aligned_bytes,
            peak_aligned_bytes: value.peak_aligned_bytes,
            total_allocations: value.total_allocations,
            current_allocations: value.current_allocations,
            peak_allocations: value.peak_allocations,
            failed_allocation_count: value.failed_allocation_count,
            largest_failed_allocation_bytes: value.largest_failed_allocation_bytes,
        }
    }
}

/// An owned device memory resource.
///
/// Wraps an `rmm::mr::device_memory_resource` via the C shim.
///
/// Each resource records the CUDA device it was created on. This provenance is
/// used by [`crate::Runtime`] builder validation to reject cross-device
/// handle combinations.
///
/// `DeviceMemoryResource` does not execute work by itself. Install it into
/// [`crate::Runtime::builder`] when you want it to become a runtime's owned
/// default allocator, or borrow it with
/// [`crate::ExecutionOverrides::with_memory_resource`] when you want a
/// same-device override for one supported operation.
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

    fn from_created_handle(handle: *mut std::ffi::c_void, device: DeviceId, constructor: &'static str) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: error::require_non_null(handle, constructor, "device memory resource")?,
                device,
            }),
        })
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
        Self::from_created_handle(handle, device, "memory::device_memory_resource::cuda_on")
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
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, raw_maximum_pool_size, &mut handle)
            },
            "memory::device_memory_resource::pool_on",
            [
                ("initial_pool_size", initial_pool_size.to_string()),
                ("maximum_pool_size", maximum_pool_size.map_or_else(|| "None".to_string(), |size| size.to_string())),
            ],
        )?;
        Self::from_created_handle(handle, device, "memory::device_memory_resource::pool_on")
    }

    /// Create a hard-limiting memory resource around this resource.
    ///
    /// The returned resource uses `self` as its upstream allocator and rejects
    /// allocations once the adaptor's live allocated bytes would exceed
    /// `allocation_limit_bytes`. Existing upstream allocations are not counted.
    ///
    /// This is an execution boundary, not a preflight sizing contract: callers
    /// still need operation-specific sizing APIs for retained outputs they
    /// intend to reserve before execution.
    pub fn limiting(&self, allocation_limit_bytes: usize) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_limiting(
                    self.handle.raw.as_ptr(),
                    allocation_limit_bytes,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::limiting",
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle(handle, self.device(), "memory::device_memory_resource::limiting")
    }

    /// Create a hard-limiting memory resource that also reports current and
    /// peak allocation counters.
    ///
    /// The returned resource uses `self` as its upstream allocator, rejects
    /// allocations once aligned live bytes through the adaptor would exceed
    /// `allocation_limit_bytes`, and exposes requested/aligned live-byte
    /// counters through [`DeviceMemoryResource::budget_tracking_usage`].
    pub fn budget_tracking_limit(&self, allocation_limit_bytes: usize) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_budget_tracking(
                    self.handle.raw.as_ptr(),
                    allocation_limit_bytes,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::budget_tracking_limit",
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle(handle, self.device(), "memory::device_memory_resource::budget_tracking_limit")
    }

    /// Read usage counters from a budget-tracking memory resource.
    ///
    /// The returned counters are diagnostic snapshots loaded independently;
    /// they are not a multi-field atomic live-polling snapshot. Operation
    /// reports should query usage after the budgeted call returns.
    ///
    /// Returns an error if this resource was not created by
    /// [`DeviceMemoryResource::budget_tracking_limit`].
    pub fn budget_tracking_usage(&self) -> Result<DeviceMemoryResourceUsage> {
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::device_memory_resource::budget_tracking_usage",
        )?;
        Ok(usage.into())
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
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_raw(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::__debug_raw_handle_value",
        )
        .map(|_| out as usize)
        .unwrap_or(0)
    }
    pub(crate) fn __test_stats() -> Result<Self> {
        let device = DeviceId::current()?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_create_stats(&mut handle) },
            "memory::device_memory_resource::__test_stats",
        )?;
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: error::require_non_null(
                    handle,
                    "memory::device_memory_resource::__test_stats",
                    "stats memory resource",
                )?,
                device,
            }),
        })
    }
    pub(crate) fn __test_allocation_count(&self) -> Result<i64> {
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
