//! Device memory resource management (RMM).
//!
//! A device memory resource is an RMM allocator handle.
//!
//! In `cudf-rs`, these handles participate in execution in two distinct ways:
//!
//! - [`crate::Runtime`] carries one default memory-resource handle as part of
//!   its retained-provenance bundle. By default this is the process-local
//!   [`DeviceMemoryManager`] resource for the runtime's CUDA device.
//! - [`crate::ExecutionOverrides`] can borrow `&DeviceMemoryResource` to
//!   override allocator selection for one supported operation without mutating
//!   the runtime's default.
//!
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit
//! memory-resource handles, and `cudf-rs` routes supported operations and
//! constructors through the resolved allocator for the active runtime/override
//! context.
//!
//! Query-style bounded execution should use the device-level manager plus
//! stream-bound reservations: allocate a [`crate::Runtime`], create an
//! [`crate::ExecutionBudgetToken`] or [`StreamReservationLease`], and execute
//! with the lease's stream and manager resource. This keeps one long-lived
//! allocator adaptor per device while charging allocations to the active
//! stream-bound reservation.
//!
//! Explicit CUDA and pool resources remain available for advanced library use,
//! diagnostics, tests, and non-query workflows. Use
//! [`crate::RuntimeBuilder::with_pool_memory_resource`] when you specifically
//! want one fresh pool owned by a runtime, or [`DeviceMemoryResource::pool_on`]
//! / [`DeviceMemoryResource::cuda_on`] when you need an explicit handle first
//! so you can install it into one or more runtimes, or borrow it for a one-off
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
//! // Default query-backend path: use the device-level manager resource.
//! let runtime = Runtime::builder().with_device(device).build().unwrap();
//! let manager = runtime.device_memory_manager().unwrap();
//! let lease = runtime.stream_reservation(7, 64 * 1024 * 1024).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(manager.resource().device(), device);
//! assert_eq!(lease.memory_resource().device(), device);
//!
//! // Advanced: create a runtime-owned pool when you specifically need
//! // allocator isolation instead of stream-bound query reservations.
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
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::{self, NonNull},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::{device_guard::DeviceGuard, stream::CudaStream},
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

/// Usage counters for a stream-bound reservation.
///
/// These counters are recorded by the device-level reservation-aware memory
/// manager. `current_*` counters include retained allocations that were
/// created while the reservation was attached, even after the reservation has
/// been detached from its stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservationUsage {
    pub reservation_id: u64,
    pub attached: bool,
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

impl From<cudf_sys::cudf_memory_resource_usage_t> for ReservationUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            reservation_id: value.reserved[0],
            attached: value.reserved[1] != 0,
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

impl From<ReservationUsage> for DeviceMemoryResourceUsage {
    fn from(value: ReservationUsage) -> Self {
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
    // Keeps callback state alive until after `raw` is destroyed in Drop.
    _failure_callback: Option<Arc<FailureCallbackState>>,
}

// SAFETY: RMM device_memory_resource implementations are thread-safe — allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

type FailureCallback = dyn Fn(usize) -> bool + Send + Sync + 'static;

struct FailureCallbackState {
    callback: Arc<FailureCallback>,
    panicked: AtomicBool,
}

unsafe extern "C" fn failure_callback_trampoline(bytes: usize, callback_arg: *mut c_void) -> i32 {
    if callback_arg.is_null() {
        return 0;
    }
    let state = unsafe { &*(callback_arg as *const FailureCallbackState) };
    if state.panicked.load(Ordering::Acquire) {
        return 0;
    }
    match catch_unwind(AssertUnwindSafe(|| (state.callback)(bytes))) {
        Ok(retry) => i32::from(retry),
        Err(_) => {
            state.panicked.store(true, Ordering::Release);
            0
        },
    }
}

/// Process-local reservation-aware memory manager for one CUDA device.
///
/// The native layer creates at most one manager per device. The manager owns a
/// shared physical pool and one reservation-aware adaptor installed as that
/// device's current RMM resource. Query reservations attach to streams; normal
/// execution options keep using the same device-level adaptor as their memory
/// resource.
#[derive(Clone, Debug)]
pub struct DeviceMemoryManager {
    handle: Arc<DeviceMemoryManagerHandle>,
    resource: DeviceMemoryResource,
}

#[derive(Debug)]
struct DeviceMemoryManagerHandle {
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

// SAFETY: the native manager is process-local, internally synchronized, and
// intentionally shared across threads for one CUDA device.
unsafe impl Send for DeviceMemoryManagerHandle {}
unsafe impl Sync for DeviceMemoryManagerHandle {}

/// A live reservation attached to a CUDA stream.
///
/// Dropping this value detaches the reservation from the stream. Allocations
/// made while it was attached remain accounted to the reservation identity
/// until those retained objects are destroyed.
#[derive(Debug)]
pub struct StreamReservationLease {
    // Field order is load-bearing: Rust drops fields in declaration order,
    // and the reservation must detach while the stream (whose raw value keys
    // the native attachment map) and the manager are still alive. If the
    // stream dropped first, a recycled raw stream value could inherit this
    // dying reservation's accounting.
    reservation: ReservationHandle,
    stream: CudaStream,
    manager: DeviceMemoryManager,
}

#[derive(Debug)]
struct ReservationHandle {
    raw: NonNull<std::ffi::c_void>,
}

// SAFETY: native detach/usage paths are synchronized by the manager.
unsafe impl Send for ReservationHandle {}
unsafe impl Sync for ReservationHandle {}

impl DeviceMemoryManager {
    /// Explicitly install the memory domain for `device` (idempotent).
    ///
    /// cudf-rs owns the memory management of every CUDA device it uses: the
    /// first time a device's manager is created, its reservation-aware
    /// adaptor is installed as RMM's current resource for that device — and
    /// it stays installed for the process lifetime. By default that happens
    /// implicitly on the first `Runtime` build for the device; services
    /// should instead call this once per device at startup so the
    /// installation point is explicit, failures surface before the first
    /// query, and other in-process rmm consumers see the final resource from
    /// the start.
    ///
    /// ```no_run
    /// use cudf::{DeviceId, DeviceMemoryManager, Runtime};
    ///
    /// // Service startup: claim every device the service will use.
    /// for ordinal in 0..DeviceId::count().unwrap() {
    ///     DeviceMemoryManager::install_for_device(
    ///         DeviceId::new(ordinal).unwrap(),
    ///     )
    ///     .unwrap();
    /// }
    ///
    /// // Each pipeline then gets its own Runtime (own stream) plus a
    /// // stream reservation for budgeting; they share the device domain.
    /// let rt = Runtime::builder().build().unwrap();
    /// let _lease = rt.stream_reservation(1, 1 << 30).unwrap();
    /// ```
    pub fn install_for_device(device: DeviceId) -> Result<Self> {
        Self::for_device(device)
    }

    /// Resolve the process-local manager for `device`, creating it if needed.
    ///
    /// On first creation the manager installs its reservation-aware adaptor as
    /// the current RMM resource for `device`. Later calls return aliases to
    /// the same manager and resource.
    pub fn for_device(device: DeviceId) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut manager = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_for_device(device.ordinal_raw(), &mut manager) },
            "memory::device_memory_manager::for_device",
            [("device", device.ordinal().to_string())],
        )?;
        let manager = error::require_non_null(manager, "memory::device_memory_manager::for_device", "memory manager")?;

        let mut resource = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_manager_resource(manager.as_ptr(), &mut resource) },
            "memory::device_memory_manager::resource",
        )?;
        let resource =
            DeviceMemoryResource::from_created_handle(resource, device, "memory::device_memory_manager::resource")?;

        Ok(Self { handle: Arc::new(DeviceMemoryManagerHandle { raw: manager, device }), resource })
    }

    /// CUDA device owned by this manager.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Device-level reservation-aware adaptor for explicit execution options.
    pub fn resource(&self) -> &DeviceMemoryResource {
        &self.resource
    }

    /// Attach a hard live-byte reservation to `stream`.
    pub fn reserve_stream(
        &self,
        stream: &CudaStream,
        reservation_id: u64,
        limit_bytes: usize,
    ) -> Result<StreamReservationLease> {
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::reserve_stream",
                "stream reservation requires stream and memory manager to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_memory_manager")
            .with_detail("reason", "cross_device_stream_reservation"));
        }

        let mut reservation = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_attach_reservation(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    reservation_id,
                    limit_bytes,
                    &mut reservation,
                )
            },
            "memory::device_memory_manager::reserve_stream",
            [
                ("device", self.device().ordinal().to_string()),
                ("reservation_id", reservation_id.to_string()),
                ("limit_bytes", limit_bytes.to_string()),
            ],
        )?;
        let reservation = ReservationHandle {
            raw: error::require_non_null(
                reservation,
                "memory::device_memory_manager::reserve_stream",
                "stream reservation",
            )?,
        };
        Ok(StreamReservationLease { manager: self.clone(), stream: stream.clone(), reservation })
    }
}

impl StreamReservationLease {
    /// Manager that owns the reservation-aware device resource.
    pub fn manager(&self) -> &DeviceMemoryManager {
        &self.manager
    }

    /// CUDA stream this reservation is attached to.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Device-level memory resource to place in explicit execution options.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.manager.resource()
    }

    /// Snapshot usage counters for this reservation.
    pub fn usage(&self) -> Result<ReservationUsage> {
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_reservation_usage(self.reservation.raw.as_ptr(), &mut usage) },
            "memory::stream_reservation_lease::usage",
        )?;
        Ok(usage.into())
    }

    /// Borrow this lease as per-operation execution overrides.
    pub fn execution_overrides(&self) -> crate::ExecutionOverrides<'_> {
        crate::ExecutionOverrides::new().with_stream(&self.stream).with_memory_resource(self.memory_resource())
    }
}

impl Drop for ReservationHandle {
    fn drop(&mut self) {
        let status = unsafe { cudf_sys::cudf_reservation_detach(self.raw.as_ptr()) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!("[cudf::StreamReservationLease] warning: failed to detach stream reservation");
        }
    }
}

impl DeviceMemoryResource {
    fn raw_maximum_pool_size(maximum_pool_size: Option<usize>) -> usize {
        maximum_pool_size.unwrap_or(0)
    }

    fn from_created_handle(handle: *mut std::ffi::c_void, device: DeviceId, constructor: &'static str) -> Result<Self> {
        Self::from_created_handle_with_callback(handle, device, constructor, None)
    }

    fn from_created_handle_with_callback(
        handle: *mut std::ffi::c_void,
        device: DeviceId,
        constructor: &'static str,
        failure_callback: Option<Arc<FailureCallbackState>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: error::require_non_null(handle, constructor, "device memory resource")?,
                device,
                _failure_callback: failure_callback,
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
    /// thread's previous CUDA current device is restored before this function
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
    /// thread's previous CUDA current device is restored before this function
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

    /// Create a failure-callback memory resource around this resource.
    ///
    /// The returned resource calls `callback(bytes)` when the upstream resource
    /// throws an RMM out-of-memory error. Returning `true` retries the failed
    /// allocation; returning `false` preserves the original allocation failure.
    ///
    /// The callback runs inside RMM's allocation path on the allocating thread,
    /// so it must be re-entrancy-safe and must not allocate through the same
    /// memory resource. Panics are caught at the FFI boundary and treated as
    /// `false`, so Rust unwinding never crosses into C++.
    pub fn with_failure_callback(&self, callback: Arc<dyn Fn(usize) -> bool + Send + Sync + 'static>) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let callback_state = Arc::new(FailureCallbackState { callback, panicked: AtomicBool::new(false) });
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        // The callback state is stored in the returned handle so `callback_arg`
        // remains valid until the native resource is destroyed.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_failure_callback(
                    self.handle.raw.as_ptr(),
                    Some(failure_callback_trampoline),
                    Arc::as_ptr(&callback_state) as *mut c_void,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::with_failure_callback",
            [("upstream_device", self.device().ordinal().to_string())],
        )?;
        Self::from_created_handle_with_callback(
            handle,
            self.device(),
            "memory::device_memory_resource::with_failure_callback",
            Some(callback_state),
        )
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
                _failure_callback: None,
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
            if let Some(callback) = self._failure_callback.take() {
                // Native destroy can be rejected when the resource is installed
                // as current. In that case the native adaptor may still call
                // `callback_arg`, so leak only the callback state instead of
                // leaving a dangling Rust pointer behind.
                std::mem::forget(callback);
            }
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
