//! Owned device memory resource handles and their teardown.

use std::{
    ffi::c_void,
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};

use super::{
    domain::{QueryMemoryDomainHandle, ReservationArenaHandle, ReservationHandle},
    failure::{
        DeviceMemoryFailureInfo, FailureCallbackState, failure_callback_trampoline, wrap_simple_failure_callback,
    },
    usage::{DeviceMemoryPoolUsage, DeviceMemoryResourceUsage},
};
use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::{device_guard::DeviceGuard, stream::CudaStream},
};

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
    pub(super) handle: Arc<DeviceMemoryResourceHandle>,
}

pub(super) struct DeviceMemoryResourceHandle {
    pub(super) raw: Option<NonNull<cudf_sys::cudf_memory_resource_handle_state>>,
    pub(super) device: DeviceId,
    // Pool-backed resources release physical allocations during native
    // destruction. Keep every runtime stream that selected such a resource so
    // its outstanding work can retire before that release begins.
    pub(super) retirement_streams: Option<Arc<RetirementStreams>>,
    // Keeps callback states alive until after `raw` and any native upstream
    // lifetimes retained by derived resources are destroyed in Drop.
    pub(super) _failure_callbacks: Vec<Arc<FailureCallbackState>>,
}

/// Streams that have submitted work through one destructible memory-resource
/// lineage.
///
/// The set lives with the resource handle rather than an execution domain:
/// domains can borrow another stream or share a pool across runtimes, while
/// only the final native resource destruction needs a retirement fence. Each
/// entry retains its stream owner so field-drop order cannot destroy the stream
/// before the resource has been fenced.
#[derive(Default)]
pub(super) struct RetirementStreams {
    streams: Mutex<Vec<CudaStream>>,
}

impl RetirementStreams {
    /// Create a dedicated registry for one short-lived resource lineage. It
    /// owns `stream` until that resource has fenced and released it during
    /// teardown, and is intentionally not shared with the upstream resource.
    fn for_scoped_resource(stream: &CudaStream) -> Arc<Self> {
        let streams = Arc::new(Self::default());
        streams.register(stream);
        streams
    }

    fn register(&self, stream: &CudaStream) {
        let mut streams = match self.streams.lock() {
            Ok(streams) => streams,
            Err(poisoned) => poisoned.into_inner(),
        };
        if streams.iter().any(|registered| registered.identity() == stream.identity()) {
            return;
        }
        streams.push(stream.clone());
    }

    fn synchronize_for_teardown(&self) -> Result<()> {
        let streams = match self.streams.lock() {
            Ok(streams) => streams.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        for stream in streams {
            stream.synchronize_for_teardown("DeviceMemoryResourceHandle::drop")?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn registered_stream_count(&self) -> usize {
        match self.streams.lock() {
            Ok(streams) => streams.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

// SAFETY: RMM device_memory_resource implementations are thread-safe; allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

impl DeviceMemoryResource {
    fn raw_maximum_pool_size(maximum_pool_size: Option<usize>) -> usize {
        maximum_pool_size.unwrap_or(0)
    }

    pub(super) fn from_created_handle(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
        constructor: &'static str,
    ) -> Result<Self> {
        Self::from_created_handle_with_callbacks(handle, device, constructor, None, Vec::new())
    }

    /// Adopt an owned native memory-resource handle for unstable test support.
    ///
    /// # Safety
    /// `handle` must be a non-null, uniquely owned memory-resource handle
    /// created for `device`. It must not be borrowed elsewhere and must be
    /// destroyed with `cudf_memory_resource_destroy` exactly once.
    pub(crate) unsafe fn adopt_owned_for_testing(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
    ) -> Result<Self> {
        Self::from_created_handle(handle, device, "testing::core::adopt_owned_memory_resource_for_test")
    }

    pub(super) fn from_created_handle_with_callbacks(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
        constructor: &'static str,
        retirement_streams: Option<Arc<RetirementStreams>>,
        failure_callbacks: Vec<Arc<FailureCallbackState>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: Some(error::require_non_null(handle, constructor, "device memory resource")?),
                device,
                retirement_streams,
                _failure_callbacks: failure_callbacks,
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
        Self::from_created_handle_with_callbacks(
            handle,
            device,
            "memory::device_memory_resource::pool_on",
            Some(Arc::new(RetirementStreams::default())),
            Vec::new(),
        )
    }

    /// Create a cudaMallocAsync-backed memory resource handle on the calling
    /// thread's current device.
    ///
    /// `release_threshold = None` keeps the CUDA default used by this crate:
    /// retain idle memory across synchronizations until an explicit
    /// [`Self::trim_to`] call.
    pub fn cuda_async(initial_pool_size: Option<usize>, release_threshold: Option<usize>) -> Result<Self> {
        Self::cuda_async_on(DeviceId::current()?, initial_pool_size, release_threshold)
    }

    /// Create a cudaMallocAsync-backed memory resource handle on an explicit
    /// device.
    pub fn cuda_async_on(
        device: DeviceId,
        initial_pool_size: Option<usize>,
        release_threshold: Option<usize>,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_cuda_async(
                    initial_pool_size.unwrap_or(0),
                    release_threshold.unwrap_or(usize::MAX) as u64,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::cuda_async_on",
            [
                ("initial_pool_size", initial_pool_size.map_or_else(|| "None".to_string(), |size| size.to_string())),
                ("release_threshold", release_threshold.map_or_else(|| "None".to_string(), |size| size.to_string())),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            device,
            "memory::device_memory_resource::cuda_async_on",
            Some(Arc::new(RetirementStreams::default())),
            Vec::new(),
        )
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
                cudf_sys::cudf_memory_resource_create_limiting(self.handle.raw(), allocation_limit_bytes, &mut handle)
            },
            "memory::device_memory_resource::limiting",
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::limiting",
            self.handle.retirement_streams.clone(),
            self.handle._failure_callbacks.clone(),
        )
    }

    /// Create a failure-callback memory resource around this resource.
    ///
    /// The callback observes each failed allocation before the resource
    /// preserves the original allocation failure.
    ///
    /// The callback runs inside RMM's allocation path on the allocating thread,
    /// so it must be re-entrancy-safe and must not allocate through the same
    /// memory resource. Panics are caught at the FFI boundary, so Rust
    /// unwinding never crosses into C++ or changes the allocation failure.
    pub fn with_failure_callback(&self, callback: Arc<dyn Fn(usize) + Send + Sync + 'static>) -> Result<Self> {
        self.with_failure_callback_details(wrap_simple_failure_callback(callback))
    }

    /// Create an observation-only failure-callback memory resource around this
    /// resource.
    ///
    /// The callback receives device, request, retry-attempt, and async-pool
    /// usage facts without changing allocation control flow.
    pub fn with_failure_callback_details(
        &self,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let callback_state = Arc::new(FailureCallbackState::new(callback));
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        // The callback state is stored in the returned handle so `callback_arg`
        // remains valid until the native resource is destroyed.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_failure_callback_v2(
                    self.handle.raw(),
                    Some(failure_callback_trampoline),
                    Arc::as_ptr(&callback_state) as *mut c_void,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::with_failure_callback",
            [("upstream_device", self.device().ordinal().to_string())],
        )?;
        let mut failure_callbacks = self.handle._failure_callbacks.clone();
        failure_callbacks.push(callback_state);
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::with_failure_callback",
            self.handle.retirement_streams.clone(),
            failure_callbacks,
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
        self.create_budget_tracking_limit(
            allocation_limit_bytes,
            self.handle.retirement_streams.clone(),
            "memory::device_memory_resource::budget_tracking_limit",
        )
    }

    /// Create one child budget tracker with a private retirement registry for
    /// `stream`.
    ///
    /// Unlike [`Self::budget_tracking_limit`], this child does not inherit the
    /// upstream allocator's retirement-stream registry. Its final handle drop
    /// fences and releases only `stream` before the child adaptor is destroyed.
    /// This is for retained per-operation allocation scopes whose runtime owns
    /// exactly one stream; callers must keep the returned resource alive until
    /// that runtime and every native owner using it have been destroyed.
    pub(crate) fn budget_tracking_limit_for_scope(
        &self,
        allocation_limit_bytes: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_resource::budget_tracking_limit_for_scope",
                "scoped budget tracker requires its stream and upstream resource to share one CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_upstream_memory_resource")
            .with_detail("reason", "cross_device_scoped_budget_tracker"));
        }
        self.create_budget_tracking_limit(
            allocation_limit_bytes,
            Some(RetirementStreams::for_scoped_resource(stream)),
            "memory::device_memory_resource::budget_tracking_limit_for_scope",
        )
    }

    fn create_budget_tracking_limit(
        &self,
        allocation_limit_bytes: usize,
        retirement_streams: Option<Arc<RetirementStreams>>,
        operation: &'static str,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_budget_tracking(
                    self.handle.raw(),
                    allocation_limit_bytes,
                    &mut handle,
                )
            },
            operation,
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            operation,
            retirement_streams,
            self.handle._failure_callbacks.clone(),
        )
    }

    #[cfg(test)]
    pub(crate) fn __debug_retirement_stream_count(&self) -> usize {
        self.handle.retirement_streams.as_ref().map_or(0, |streams| streams.registered_stream_count())
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
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_usage(self.handle.raw(), &mut usage) },
            "memory::device_memory_resource::budget_tracking_usage",
        )?;
        Ok(usage.into())
    }

    /// Read physical CUDA async-pool counters for this resource.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_pool_usage(self.handle.raw(), &mut usage) },
            "memory::device_memory_resource::pool_usage",
            [("device", self.device().ordinal().to_string())],
        )?;
        Ok(usage.into())
    }

    /// Explicitly trim idle bytes from this resource's CUDA async pool.
    pub fn trim_to(&self, keep_bytes: u64) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_trim_to(self.handle.raw(), keep_bytes) },
            "memory::device_memory_resource::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
        )
    }

    /// The CUDA device this memory resource was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Register a same-device stream whose work must retire before this
    /// pool-backed resource can release its physical backing allocation.
    pub(crate) fn register_stream_for_retirement(&self, stream: &CudaStream) {
        debug_assert_eq!(stream.device(), self.device());
        if let Some(retirement_streams) = &self.handle.retirement_streams {
            retirement_streams.register(stream);
        }
    }

    /// Borrow this owner as a low-level `cudf-sys` memory-resource handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `DeviceMemoryResource` remains responsible for
    /// destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.handle.raw()) }
    }

    pub(crate) fn as_raw(&self) -> cudf_sys::cudf_memory_resource_handle {
        self.handle.raw()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_raw(self.handle.raw(), &mut out) },
            "memory::device_memory_resource::interop_raw_handle",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        self.interop_raw_handle().map(|raw| raw as usize).unwrap_or(0)
    }
}

impl Drop for DeviceMemoryResourceHandle {
    fn drop(&mut self) {
        let retirement_error = self
            .raw
            .is_some()
            .then_some(self.retirement_streams.as_ref())
            .flatten()
            .and_then(|retirement_streams| retirement_streams.synchronize_for_teardown().err());
        if let Some(error) = retirement_error {
            // A pool may release backing allocations in native destruction.
            // Without a successful fence, leaking is safer than racing queued
            // device work with that release. Leak the registered stream owners
            // with it: letting their native stream handles drop could otherwise
            // retire queued work after the allocator has been quarantined.
            let _ = self.raw.take();
            std::mem::forget(std::mem::take(&mut self.retirement_streams));
            if !self._failure_callbacks.is_empty() {
                std::mem::forget(std::mem::take(&mut self._failure_callbacks));
            }
            crate::runtime::quarantine_device_resource(self.device, "DeviceMemoryResourceHandle::drop", &error);
            return;
        }
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        let mut status = None;
        crate::runtime::drop_device_resource(&mut raw, self.device, "DeviceMemoryResourceHandle::drop", |owned| {
            status = Some(unsafe { cudf_sys::cudf_memory_resource_destroy(owned, self.device.ordinal_raw()) });
        });
        if status != Some(cudf_sys::CUDF_SUCCESS) {
            if !self._failure_callbacks.is_empty() {
                // Native destroy can be rejected when the resource is installed
                // as current. In that case the native adaptor may still call
                // `callback_arg`, so leak only the callback states instead of
                // leaving dangling Rust pointers behind.
                std::mem::forget(std::mem::take(&mut self._failure_callbacks));
            }
            if let Some(status) = status {
                record_native_teardown_failure(self.device, "DeviceMemoryResourceHandle::drop", status);
            }
        }
    }
}

impl DeviceMemoryResourceHandle {
    fn raw(&self) -> cudf_sys::cudf_memory_resource_handle {
        self.raw.expect("live memory resource retains its native handle").as_ptr()
    }
}

impl ReservationHandle {
    pub(super) fn raw(&self) -> cudf_sys::cudf_reservation_handle {
        self.raw.expect("live reservation retains its native handle").as_ptr()
    }
}

impl ReservationArenaHandle {
    pub(super) fn raw(&self) -> cudf_sys::cudf_reservation_arena_handle {
        self.raw.expect("live reservation arena retains its native handle").as_ptr()
    }
}

impl QueryMemoryDomainHandle {
    pub(super) fn raw(&self) -> cudf_sys::cudf_query_memory_domain_handle {
        self.raw.expect("live query memory domain retains its native handle").as_ptr()
    }
}

pub(super) fn record_native_teardown_failure(
    device: DeviceId,
    operation: &'static str,
    status: cudf_sys::cudf_status_t,
) {
    if status == cudf_sys::CUDF_ERROR_FATAL_CUDA {
        crate::runtime::report_device_fault(
            device,
            crate::runtime::DeviceFaultKind::FatalDeviceOperation,
            operation,
            Some(crate::runtime::DeviceFaultNativeStatus::Cudf(status)),
        );
    }
    crate::runtime::device_health::record_teardown_failure(
        device,
        operation,
        &format_args!("native teardown failed with status {status}"),
    );
}

impl std::fmt::Debug for DeviceMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryResource")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
