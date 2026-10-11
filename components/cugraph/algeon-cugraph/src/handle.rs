//! GPU resource handle for cuGraph operations.
//!
//! [`ResourceHandle`] owns cuGraph execution provenance while
//! [`EnteredContext`] is the short-lived execution capability used for GPU
//! work on the current thread. Both types are **not** `Send` or `Sync`
//! because GPU state is thread-local.

use std::{fmt, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull, sync::Arc};

use cudf::{
    CurrentResourceRouteLease, DeviceId, DeviceMemoryManager,
    interop::contract::{ExecutionBinding, InteropError, InteropErrorKind},
    runtime::{BindingExtensionGuard, BindingExtensionInitError, BindingExtensionLease},
};
use cugraph_sys::{cugraph_resource_handle_get_comm_size, cugraph_resource_handle_get_rank, cugraph_resource_handle_t};

use crate::{
    cuda::{DeviceGuard, DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result},
};

enum ResourceHandleState {
    #[cfg(test)]
    LocalOwned {
        raw: *mut cugraph_resource_handle_t,
        device: i32,
    },
    RuntimeBound {
        lease: ResourceHandleLease,
    },
    ExternalBorrowed {
        raw: *mut cugraph_resource_handle_t,
        device: i32,
    },
    ExternalOwned {
        raw: *mut cugraph_resource_handle_t,
        device: i32,
    },
}

impl ResourceHandleState {
    fn standalone_raw(&self) -> Option<*mut cugraph_resource_handle_t> {
        match self {
            #[cfg(test)]
            Self::LocalOwned { raw, .. } => Some(*raw),
            Self::ExternalBorrowed { raw, .. } | Self::ExternalOwned { raw, .. } => Some(*raw),
            Self::RuntimeBound { .. } => None,
        }
    }

    fn device(&self) -> i32 {
        match self {
            #[cfg(test)]
            Self::LocalOwned { device, .. } => *device,
            Self::ExternalBorrowed { device, .. } | Self::ExternalOwned { device, .. } => *device,
            Self::RuntimeBound { lease } => lease.device(),
        }
    }

    fn runtime(&self) -> Option<&cudf::Runtime> {
        match self {
            Self::RuntimeBound { lease } => Some(lease.runtime()),
            #[cfg(test)]
            Self::LocalOwned { .. } => None,
            Self::ExternalBorrowed { .. } | Self::ExternalOwned { .. } => None,
        }
    }

    fn runtime_bound_lease(&self) -> Option<ResourceHandleLease> {
        match self {
            Self::RuntimeBound { lease } => Some(lease.clone()),
            #[cfg(test)]
            Self::LocalOwned { .. } => None,
            Self::ExternalBorrowed { .. } | Self::ExternalOwned { .. } => None,
        }
    }

    fn is_runtime_bound(&self) -> bool {
        matches!(self, Self::RuntimeBound { .. })
    }

    fn uses_external_comm_metadata(&self) -> bool {
        matches!(self, Self::ExternalBorrowed { .. } | Self::ExternalOwned { .. })
    }
}

#[repr(C)]
struct cugraph_rs_runtime_resource_handle_t {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn cugraph_rs_runtime_resource_handle_create(
        cuda_stream: *mut std::ffi::c_void,
        device_memory_resource: *mut std::ffi::c_void,
        memory_resource_owner: *mut std::ffi::c_void,
    ) -> *mut cugraph_rs_runtime_resource_handle_t;
    fn cugraph_rs_runtime_resource_handle_get(
        handle: *mut cugraph_rs_runtime_resource_handle_t,
    ) -> *mut cugraph_resource_handle_t;
    #[cfg(test)]
    fn cugraph_rs_runtime_resource_handle_effective_device_memory(
        handle: *mut cugraph_rs_runtime_resource_handle_t,
        out: *mut usize,
    ) -> i32;
}

struct RuntimeResourceHandleOwner {
    raw: NonNull<cugraph_rs_runtime_resource_handle_t>,
    device: i32,
    // cuGraph's graph construction keeps some device allocations behind RMM's
    // current-resource API. Keep this lease until after RAFT/cuGraph teardown
    // so those allocations always reach the runtime's explicit resource
    // without replacing cuDF's permanent current resource.
    _current_resource_route: CurrentResourceRouteLease,
}

// SAFETY: the shim owns a RAFT handle constructed from an explicit cudf stream
// and memory resource; it carries no creator-thread identity or secondary
// stream pool. Every native use is scoped by the exact binding extension's
// stable execution gate and a DeviceGuard for the recorded CUDA device. The
// gate serializes the same native owner across threads, while ResourceHandle
// and EnteredContext remain !Send/!Sync thread-local execution capabilities.
unsafe impl Send for RuntimeResourceHandleOwner {}
// SAFETY: see the `Send` impl above. Shared access cannot bypass the stable
// gate, and Drop consumes the pointer exactly once. If device entry cannot be
// proven, Drop quarantines the native pointer and does not invoke FFI.
unsafe impl Sync for RuntimeResourceHandleOwner {}

impl RuntimeResourceHandleOwner {
    fn create(runtime: &cudf::Runtime, operation: &'static str) -> Result<Self> {
        let binding =
            runtime.execution_binding().map_err(|error| map_runtime_execution_binding_error(error, operation))?;
        let runtime_device = runtime.device().ordinal() as i32;
        validate_runtime_binding_device(runtime_device, binding, operation)?;

        let _guard = DeviceGuard::enter_owner_device(runtime_device, operation)?;
        let manager = DeviceMemoryManager::for_device(runtime.device())
            .map_err(|error| map_current_resource_route_error(error, operation))?;
        let current_resource_route = manager
            .route_current_resource(runtime.stream(), runtime.memory_resource())
            .map_err(|error| map_current_resource_route_error(error, operation))?;
        // SAFETY: `binding` is borrowed from a live explicit cudf runtime. The
        // shim borrows the stream and memory-resource handles for the lifetime
        // guarded by the binding-extension lease's retained domain.
        let shim = unsafe {
            cugraph_rs_runtime_resource_handle_create(
                binding.cuda_stream().as_raw(),
                binding.memory_resource().as_raw(),
                runtime.memory_resource().as_borrowed_sys().as_raw().cast(),
            )
        };
        let raw = NonNull::new(shim).ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, operation)
                .with_message("failed to create runtime-bound cuGraph resource handle shim")
                .with_context("device", runtime_device.to_string())
        })?;

        // SAFETY: `raw` is a newly created live shim handle.
        let resource_handle = unsafe { cugraph_rs_runtime_resource_handle_get(raw.as_ptr()) };
        if resource_handle.is_null() {
            let mut owned = raw.as_ptr();
            drop_device_resource(&mut owned, runtime_device, operation, DeviceResourceKind::RuntimeResourceHandle);
            return Err(Error::new(ErrorKind::Unexpected, operation)
                .with_message("runtime-bound handle shim returned a null cuGraph resource handle")
                .with_context("device", runtime_device.to_string()));
        }

        Ok(Self { raw, device: runtime_device, _current_resource_route: current_resource_route })
    }

    fn resource_handle(&self) -> *mut cugraph_resource_handle_t {
        // SAFETY: the entered binding-extension lease retains this live shim.
        unsafe { cugraph_rs_runtime_resource_handle_get(self.raw.as_ptr()) }
    }

    #[cfg(test)]
    fn effective_device_memory(&self) -> Result<usize> {
        let mut out = 0;
        // SAFETY: the binding-extension lease retains the shim and its RAFT
        // handle for this call.
        let status = unsafe { cugraph_rs_runtime_resource_handle_effective_device_memory(self.raw.as_ptr(), &mut out) };
        if status == 0 {
            Ok(out)
        } else {
            Err(Error::new(ErrorKind::InvalidState, "RuntimeResourceHandleOwner::effective_device_memory")
                .with_message("cuGraph effective device memory provider rejected the runtime-bound handle")
                .with_context("provider_status", status.to_string()))
        }
    }
}

impl Drop for RuntimeResourceHandleOwner {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        drop_device_resource(
            &mut raw,
            self.device,
            "RuntimeResourceHandleOwner::drop",
            DeviceResourceKind::RuntimeResourceHandle,
        );
    }
}

/// Cloneable owner lease for runtime-bound cuGraph handles.
///
/// Native graph objects can retain allocations or RAFT-side state associated
/// with a runtime-bound cuGraph resource handle used for graph construction or
/// an algorithm call. The Rust `Graph` wrapper keeps these leases until graph
/// teardown so the native shim and cudf runtime outlive any graph state that
/// may reference them.
#[derive(Clone)]
pub(crate) struct ResourceHandleLease {
    owner: BindingExtensionLease<RuntimeResourceHandleOwner>,
    runtime: Arc<cudf::Runtime>,
}

impl ResourceHandleLease {
    fn new(owner: BindingExtensionLease<RuntimeResourceHandleOwner>, runtime: Arc<cudf::Runtime>) -> Self {
        Self { owner, runtime }
    }

    fn enter(&self, operation: &'static str) -> Result<BindingExtensionGuard<'_, RuntimeResourceHandleOwner>> {
        self.owner.enter(operation).map_err(|error| map_binding_extension_lifecycle_error(error, operation))
    }

    pub(crate) fn with_teardown_gate<R>(&self, drop_native: impl FnOnce() -> R) -> R {
        self.owner.with_teardown_gate(drop_native)
    }

    fn device(&self) -> i32 {
        self.runtime.device().ordinal() as i32
    }

    pub(crate) fn runtime(&self) -> &cudf::Runtime {
        self.runtime.as_ref()
    }

    pub(crate) fn same_binding(&self, other: &Self) -> bool {
        self.runtime.binding_id() == other.runtime.binding_id()
    }

    pub(crate) fn same_native_owner(&self, other: &Self) -> bool {
        self.owner.same_generation(&other.owner)
    }

    #[cfg(test)]
    fn generation_matches(&self, other: &Self) -> bool {
        self.owner.same_generation(&other.owner)
    }
}

fn map_binding_extension_lifecycle_error(error: cudf::Error, operation: &'static str) -> Error {
    Error::new(ErrorKind::InvalidState, operation)
        .with_message("cudf runtime binding-extension lifecycle rejected cuGraph handle access")
        .with_context("cudf_code", error.code().to_string())
        .with_context("cudf_operation", error.operation().to_string())
        .with_context("cudf_message", error.message().to_string())
        .set_source(error)
}

fn map_current_resource_route_error(error: cudf::Error, operation: &'static str) -> Error {
    Error::new(ErrorKind::InvalidState, operation)
        .with_message("failed to register the runtime memory resource for legacy current-RMM allocations")
        .with_context("cudf_code", error.code().to_string())
        .with_context("cudf_operation", error.operation().to_string())
        .with_context("cudf_message", error.message().to_string())
        .set_source(error)
}

fn map_binding_extension_init_error(error: BindingExtensionInitError<Error>, operation: &'static str) -> Error {
    match error {
        BindingExtensionInitError::Lifecycle(error) => map_binding_extension_lifecycle_error(error, operation),
        BindingExtensionInitError::Factory(error) => error,
    }
}

fn map_runtime_execution_binding_error(error: InteropError, operation: &'static str) -> Error {
    let mut mapped = match error.kind() {
        InteropErrorKind::ExecutionBindingUnavailable => Error::new(ErrorKind::Unexpected, operation),
        _ => Error::new(ErrorKind::InvalidState, operation),
    }
    .with_message("failed to extract cudf runtime execution binding")
    .with_context("interop_kind", format!("{:?}", error.kind()))
    .set_source(error.clone());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped
}

fn validate_runtime_binding_device(
    runtime_device: i32,
    binding: ExecutionBinding<'_>,
    operation: &'static str,
) -> Result<()> {
    if binding.device() == runtime_device {
        return Ok(());
    }

    Err(Error::new(ErrorKind::InvalidState, operation)
        .with_message("runtime execution binding device did not match the runtime owner device")
        .with_context("runtime_device", runtime_device.to_string())
        .with_context("binding_device", binding.device().to_string()))
}

/// RAII wrapper around `cugraph_resource_handle_t`.
///
/// Manages GPU resources (CUDA stream and the underlying cuGraph resource
/// handle). Automatically freed on drop (unless created via
/// [`from_raw_resource_handle`](Self::from_raw_resource_handle), which borrows
/// an external cuGraph resource handle). Not thread-safe; must be used from
/// the thread that created it.
pub struct ResourceHandle {
    state: ResourceHandleState,
    // !Send + !Sync marker: cuGraph handle owns thread-affine GPU resources.
    _nosend_nosync: PhantomData<*mut ()>,
}

/// Thread-scoped execution capability for GPU work on a [`ResourceHandle`].
///
/// Values of this type are created only through [`ResourceHandle::enter`].
/// Holding an `EnteredContext` guarantees that the current thread is entered
/// on the owning CUDA device for the lifetime of the borrow.
pub struct EnteredContext<'a> {
    owner: &'a ResourceHandle,
    _guard: DeviceGuard,
    // Field order is intentional: restore the thread's CUDA device before
    // releasing the binding extension's stable execution gate.
    runtime_extension: Option<BindingExtensionGuard<'a, RuntimeResourceHandleOwner>>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl fmt::Debug for EnteredContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnteredContext").field("device", &self.device()).finish_non_exhaustive()
    }
}

impl EnteredContext<'_> {
    /// Borrow the entered raw cuGraph resource-handle pointer.
    pub fn handle_ptr(&self) -> *mut cugraph_resource_handle_t {
        match &self.runtime_extension {
            Some(extension) => extension.value().resource_handle(),
            None => self
                .owner
                .state
                .standalone_raw()
                .expect("an entered non-runtime handle always has a standalone pointer"),
        }
    }

    /// The CUDA device this execution session is entered on.
    pub fn device(&self) -> i32 {
        self.owner.device()
    }

    /// The retained cudf runtime, if this owner is runtime-bound.
    pub fn runtime(&self) -> Option<&cudf::Runtime> {
        self.owner.runtime()
    }

    pub(crate) fn runtime_bound_lease(&self) -> Option<ResourceHandleLease> {
        self.owner.runtime_bound_lease()
    }
}

impl fmt::Debug for ResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceHandle").finish_non_exhaustive()
    }
}

impl ResourceHandle {
    fn current_device_ordinal(operation: &'static str) -> Result<i32> {
        let device = DeviceId::current().map_err(|error| {
            Error::new(ErrorKind::Unexpected, operation)
                .with_message(format!("failed to query current CUDA device: {}", error))
                .set_source(error)
        })?;
        Ok(device.ordinal() as i32)
    }

    /// Create a new resource handle with default single-GPU cuGraph resources.
    ///
    /// Internal helper for source-local tests. Delegates to `for_current_device`.
    #[cfg(test)]
    pub(crate) fn new() -> Result<Self> {
        Self::for_current_device()
    }

    /// Create a new resource handle on the calling thread's current CUDA device.
    #[cfg(test)]
    pub(crate) fn for_current_device() -> Result<Self> {
        Self::new_on_device(Self::current_device_ordinal("ResourceHandle::for_current_device")?)
    }

    /// Create a new resource handle bound to a cudf runtime owner.
    ///
    /// This constructor binds cuGraph to the runtime's CUDA stream and RAFT
    /// workspace/resource-aware allocation context through a local native shim
    /// that owns both the RAFT handle and the cuGraph resource-handle wrapper.
    /// All handles for the same exact runtime stream/MR binding share one shim.
    pub fn for_runtime(runtime: Arc<cudf::Runtime>) -> Result<Self> {
        let operation = "ResourceHandle::for_runtime";
        let owner = runtime
            .get_or_try_init_binding_extension(operation, || {
                RuntimeResourceHandleOwner::create(runtime.as_ref(), operation)
            })
            .map_err(|error| map_binding_extension_init_error(error, operation))?;
        let lease = ResourceHandleLease::new(owner, runtime);
        Ok(Self { state: ResourceHandleState::RuntimeBound { lease }, _nosend_nosync: PhantomData })
    }

    #[cfg(test)]
    fn new_on_device(device: i32) -> Result<Self> {
        if device < 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::new_on_device")
                .with_message(format!("device ordinal must be >= 0, got {}", device)));
        }

        DeviceId::new(device as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, "ResourceHandle::new_on_device")
                .with_message(format!("invalid CUDA device ordinal {}: {}", device, error))
                .set_source(error)
        })?;

        let _guard = DeviceGuard::enter_owner_device(device, "ResourceHandle::new_on_device")?;
        // SAFETY: cuGraph creates a fresh handle and does not retain the null
        // communicator pointer beyond the call.
        let raw = unsafe { cugraph_sys::cugraph_create_resource_handle(std::ptr::null_mut()) };
        if raw.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "ResourceHandle::new_on_device")
                .with_message("cugraph_create_resource_handle returned NULL"));
        }
        Ok(Self { state: ResourceHandleState::LocalOwned { raw, device }, _nosend_nosync: PhantomData })
    }

    /// Wrap an existing cuGraph resource handle created externally.
    ///
    /// This constructor is intended for multi-GPU workflows where a
    /// per-thread `cugraph_resource_handle_t*` has already been created
    /// externally (for example by an MTMG or MPI-aware native layer). The
    /// returned wrapper does **not** own the handle; it will not be freed on
    /// drop. The wrapped handle is assumed to belong to the calling thread's
    /// current CUDA device at the moment this function is called.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `raw` is a valid, non-null pointer to a `cugraph_resource_handle_t`
    ///   created by cuGraph-compatible native code
    /// - the calling thread's current CUDA device is the device that `raw` is
    ///   bound to
    /// - the handle remains valid for the lifetime of this `ResourceHandle`
    /// - the handle is not freed externally while this wrapper exists
    /// - the calling thread obeys the external runtime's thread-affinity rules
    /// - graph operations run through this handle are not ordered by
    ///   [`Graph::park`](crate::graph::Graph::park), so the graph becomes
    ///   ineligible for parked transfer
    pub unsafe fn from_raw_resource_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }
        let device = Self::current_device_ordinal("ResourceHandle::from_raw_resource_handle")?;
        let _guard = DeviceGuard::enter_owner_device(device, "ResourceHandle::from_raw_resource_handle")?;
        Ok(Self { state: ResourceHandleState::ExternalBorrowed { raw, device }, _nosend_nosync: PhantomData })
    }

    /// Adopt ownership of an existing cuGraph resource handle created externally.
    ///
    /// This constructor is intended for native MTMG or MPI-aware shims that
    /// allocate a fresh `cugraph_resource_handle_t*` for Rust and transfer
    /// destruction responsibility to Rust. The returned wrapper owns the
    /// handle and will free it on drop using `cugraph_free_resource_handle`.
    /// The adopted handle is assumed to belong to the calling thread's
    /// current CUDA device at the moment this function is called.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `raw` is a valid, non-null pointer to a `cugraph_resource_handle_t`
    /// - ownership of `raw` is transferred exactly once to this wrapper
    /// - no other owner will free `raw` after this call succeeds
    /// - the calling thread's current CUDA device is the device that `raw` is
    ///   bound to
    /// - the calling thread obeys the external runtime's thread-affinity rules
    /// - graph operations run through this handle are not ordered by
    ///   [`Graph::park`](crate::graph::Graph::park), so the graph becomes
    ///   ineligible for parked transfer
    ///
    /// For a borrowed/non-owning wrapper, use
    /// [`from_raw_resource_handle`](Self::from_raw_resource_handle) instead.
    pub unsafe fn adopt_raw_resource_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::adopt_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }
        let device = Self::current_device_ordinal("ResourceHandle::adopt_raw_resource_handle")?;
        let _guard = DeviceGuard::enter_owner_device(device, "ResourceHandle::adopt_raw_resource_handle")?;
        Ok(Self { state: ResourceHandleState::ExternalOwned { raw, device }, _nosend_nosync: PhantomData })
    }

    /// Get the raw C pointer. For internal use by other modules.
    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *mut cugraph_resource_handle_t {
        self.state.standalone_raw().expect("runtime-bound handle pointers require an EnteredContext")
    }

    /// Consume this wrapper and return the underlying raw handle pointer.
    ///
    /// This suppresses Rust-side `Drop`, so no automatic
    /// `cugraph_free_resource_handle` call will occur after this returns.
    ///
    /// Ownership semantics are preserved from the source wrapper:
    /// - if `self` owned the handle, the caller becomes responsible for freeing
    ///   it according to the cuGraph resource-handle ownership contract
    /// - if `self` only borrowed the handle, the caller still does not own it
    ///   and must follow the original external destruction contract
    ///
    /// Runtime-bound handles cannot be converted into standalone raw resource
    /// handles because the native shim must continue to own the paired RAFT
    /// handle and destroy both objects together.
    pub fn into_raw_resource_handle(self) -> Result<*mut cugraph_resource_handle_t> {
        if self.state.is_runtime_bound() {
            return Err(Error::new(ErrorKind::InvalidState, "ResourceHandle::into_raw_resource_handle")
                .with_message("runtime-bound handles cannot be converted into standalone raw resource handles"));
        }

        let this = ManuallyDrop::new(self);
        Ok(this.state.standalone_raw().expect("runtime-bound handles were rejected before raw extraction"))
    }

    /// The CUDA device this handle was created on.
    pub fn device(&self) -> i32 {
        self.state.device()
    }

    pub fn runtime(&self) -> Option<&cudf::Runtime> {
        self.state.runtime()
    }

    pub(crate) fn runtime_bound_lease(&self) -> Option<ResourceHandleLease> {
        self.state.runtime_bound_lease()
    }

    /// The communicator rank carried by this handle.
    ///
    /// Handles created through [`for_runtime`](Self::for_runtime)
    /// always return `0`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn rank(&self) -> i32 {
        if self.state.uses_external_comm_metadata() {
            let Ok(_guard) = DeviceGuard::enter_owner_device(self.device(), "ResourceHandle::rank") else {
                return 0;
            };
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle, and owner-device entry was proved above.
            unsafe {
                cugraph_resource_handle_get_rank(
                    self.state.standalone_raw().expect("external handles always have a standalone pointer"),
                )
            }
        } else {
            0
        }
    }

    /// The communicator size carried by this handle.
    ///
    /// Handles created through [`for_runtime`](Self::for_runtime)
    /// always return `1`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn comm_size(&self) -> i32 {
        if self.state.uses_external_comm_metadata() {
            let Ok(_guard) = DeviceGuard::enter_owner_device(self.device(), "ResourceHandle::comm_size") else {
                return 1;
            };
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle, and owner-device entry was proved above.
            unsafe {
                cugraph_resource_handle_get_comm_size(
                    self.state.standalone_raw().expect("external handles always have a standalone pointer"),
                )
            }
        } else {
            1
        }
    }

    /// Enter the owner device on the current thread and create an execution
    /// session for GPU-capable work. Runtime-bound handles serialize entry for
    /// the shared native owner; nested same-owner entry on one thread is
    /// rejected without waiting, so it cannot deadlock.
    pub fn enter(&self, operation: &'static str) -> Result<EnteredContext<'_>> {
        let runtime_extension = match &self.state {
            ResourceHandleState::RuntimeBound { lease } => Some(lease.enter(operation)?),
            #[cfg(test)]
            ResourceHandleState::LocalOwned { .. } => None,
            ResourceHandleState::ExternalBorrowed { .. } | ResourceHandleState::ExternalOwned { .. } => None,
        };
        let guard = DeviceGuard::enter_owner_device(self.device(), operation)?;
        Ok(EnteredContext { owner: self, _guard: guard, runtime_extension, _nosend_nosync: PhantomData })
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        match &mut self.state {
            #[cfg(test)]
            ResourceHandleState::LocalOwned { raw, device } => {
                drop_device_resource(raw, *device, "ResourceHandle::drop", DeviceResourceKind::ResourceHandle);
            },
            ResourceHandleState::ExternalOwned { raw, device } => {
                drop_device_resource(raw, *device, "ResourceHandle::drop", DeviceResourceKind::ResourceHandle);
            },
            ResourceHandleState::RuntimeBound { .. } | ResourceHandleState::ExternalBorrowed { .. } => {},
        }
    }
}

// ResourceHandle is NOT Send/Sync; GPU state is thread-local.

#[cfg(test)]
mod tests {
    use std::{
        ptr,
        sync::atomic::{AtomicBool, Ordering},
    };

    use cudf::{
        DeviceMemoryResource, QueryMemoryDomainOptions,
        testing::{
            device_safety::{DeviceEntryTestCall as TestCudaCall, DeviceSafetyTestScope as TestCudaHooks},
            run_named_cases,
        },
    };
    use cugraph_sys::cugraph_create_resource_handle;

    use super::*;
    use crate::DeviceArray;

    struct ReleaseProbe(Arc<AtomicBool>);

    impl Drop for ReleaseProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// A runtime whose execution domains retain a probe that records when the
    /// last of them is released.
    fn probed_runtime() -> (Arc<cudf::Runtime>, Arc<AtomicBool>) {
        let released = Arc::new(AtomicBool::new(false));
        let runtime = cudf::Runtime::builder()
            .build()
            .expect("runtime")
            .with_retained_lifetime(Arc::new(ReleaseProbe(Arc::clone(&released))))
            .expect("probed runtime");
        (Arc::new(runtime), released)
    }

    fn budgeted_runtime(capacity_bytes: usize) -> (Arc<cudf::Runtime>, DeviceMemoryResource) {
        let device = DeviceId::current().expect("current device");
        let upstream = DeviceMemoryResource::cuda_async_on(device, None, Some(0)).expect("CUDA async resource");
        let budget = upstream.budget_tracking_limit(capacity_bytes).expect("budget tracking resource");
        let runtime =
            Arc::new(cudf::Runtime::builder().with_memory_resource(budget.clone()).build().expect("budgeted runtime"));
        (runtime, budget)
    }

    fn effective_device_memory(entered: &EnteredContext<'_>) -> usize {
        entered
            .runtime_extension
            .as_ref()
            .expect("runtime-bound context")
            .value()
            .effective_device_memory()
            .expect("effective device memory")
    }

    fn state_variant_name(state: &ResourceHandleState) -> &'static str {
        match state {
            ResourceHandleState::LocalOwned { .. } => "LocalOwned",
            ResourceHandleState::RuntimeBound { .. } => "RuntimeBound",
            ResourceHandleState::ExternalBorrowed { .. } => "ExternalBorrowed",
            ResourceHandleState::ExternalOwned { .. } => "ExternalOwned",
        }
    }

    #[test]
    fn native_owner_cases_1() {
        run_named_cases(&[
            ("handle::tests::test_for_runtime_retains_execution_owner", test_for_runtime_retains_execution_owner),
            (
                "handle::tests::runtime_bound_handles_reuse_one_native_owner_for_exact_binding_hot_hits",
                runtime_bound_handles_reuse_one_native_owner_for_exact_binding_hot_hits,
            ),
            (
                "handle::tests::runtime_bound_handles_do_not_share_native_owner_across_bindings",
                runtime_bound_handles_do_not_share_native_owner_across_bindings,
            ),
            (
                "handle::tests::runtime_binding_store_and_native_owner_do_not_form_a_cycle",
                runtime_binding_store_and_native_owner_do_not_form_a_cycle,
            ),
            (
                "handle::tests::test_for_runtime_uses_runtime_shim_instead_of_direct_cugraph_ownership",
                test_for_runtime_uses_runtime_shim_instead_of_direct_cugraph_ownership,
            ),
            (
                "handle::tests::test_from_raw_resource_handle_uses_current_device",
                test_from_raw_resource_handle_uses_current_device,
            ),
            (
                "handle::tests::test_adopt_raw_resource_handle_uses_current_device",
                test_adopt_raw_resource_handle_uses_current_device,
            ),
            (
                "handle::tests::test_from_raw_resource_handle_does_not_free_on_drop",
                test_from_raw_resource_handle_does_not_free_on_drop,
            ),
            ("handle::tests::test_adopt_raw_resource_handle_marks_owned", test_adopt_raw_resource_handle_marks_owned),
            (
                "handle::tests::test_entered_context_exposes_runtime_provenance",
                test_entered_context_exposes_runtime_provenance,
            ),
            (
                "handle::tests::runtime_handle_effective_memory_tracks_aligned_domain_charge",
                runtime_handle_effective_memory_tracks_aligned_domain_charge,
            ),
            (
                "handle::tests::runtime_handles_keep_distinct_effective_memory_providers",
                runtime_handles_keep_distinct_effective_memory_providers,
            ),
            (
                "handle::tests::nested_entry_for_same_native_owner_is_rejected_without_deadlock",
                nested_entry_for_same_native_owner_is_rejected_without_deadlock,
            ),
            (
                "handle::tests::test_into_raw_resource_handle_releases_rust_ownership",
                test_into_raw_resource_handle_releases_rust_ownership,
            ),
            (
                "handle::tests::test_runtime_bound_into_raw_resource_handle_is_rejected",
                test_runtime_bound_into_raw_resource_handle_is_rejected,
            ),
        ]);
    }

    fn test_for_runtime_retains_execution_owner() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        assert_eq!(Arc::strong_count(&runtime), 1, "test setup should hold one runtime owner");

        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");
        assert_eq!(handle.device(), runtime.device().ordinal() as i32);
        assert_eq!(Arc::strong_count(&runtime), 2, "runtime-bound handle should retain the runtime owner");

        drop(handle);
        assert_eq!(Arc::strong_count(&runtime), 1, "dropping the handle should release the retained runtime owner");
    }

    fn runtime_bound_handles_reuse_one_native_owner_for_exact_binding_hot_hits() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let derived = Arc::new(runtime.with_retained_lifetime(Arc::new(())).expect("derived runtime"));
        assert!(!Arc::ptr_eq(&runtime, &derived));

        let first = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("first resource handle");
        let first_lease = first.runtime_bound_lease().expect("first runtime lease");
        let first_raw = first.enter("hot_hit_first").expect("enter first").handle_ptr();

        for index in 0..100 {
            let owner = if index % 2 == 0 { Arc::clone(&runtime) } else { Arc::clone(&derived) };
            let handle = ResourceHandle::for_runtime(owner).expect("hot-hit resource handle");
            let lease = handle.runtime_bound_lease().expect("hot-hit runtime lease");
            assert!(first_lease.same_binding(&lease));
            assert!(first_lease.same_native_owner(&lease));
            let entered = handle.enter("hot_hit").expect("enter hot hit");
            assert_eq!(entered.handle_ptr(), first_raw);
        }
    }

    fn runtime_bound_handles_do_not_share_native_owner_across_bindings() {
        let first_runtime = Arc::new(cudf::Runtime::builder().build().expect("first runtime"));
        let second_runtime = Arc::new(cudf::Runtime::builder().build().expect("independent runtime"));
        assert_ne!(first_runtime.binding_id(), second_runtime.binding_id());

        let first = ResourceHandle::for_runtime(first_runtime).expect("first resource handle");
        let second = ResourceHandle::for_runtime(second_runtime).expect("second resource handle");
        let first_lease = first.runtime_bound_lease().expect("first runtime lease");
        let second_lease = second.runtime_bound_lease().expect("second runtime lease");

        assert!(!first_lease.same_binding(&second_lease));
        assert!(!first_lease.same_native_owner(&second_lease));
        let first_raw = first.enter("different_binding_first").expect("enter first").handle_ptr();
        let second_raw = second.enter("different_binding_second").expect("enter second").handle_ptr();
        assert_ne!(first_raw, second_raw);
    }

    fn runtime_binding_store_and_native_owner_do_not_form_a_cycle() {
        let (runtime, released) = probed_runtime();
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");
        drop(runtime);
        assert!(!released.load(Ordering::SeqCst), "the resource-handle lease retains its execution domain");
        drop(handle);
        assert!(
            released.load(Ordering::SeqCst),
            "dropping the last native-owner lease must release the execution domain"
        );
    }

    #[test]
    fn runtime_bound_owner_quarantines_before_releasing_runtime_when_device_entry_fails() {
        let (runtime, released) = probed_runtime();
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");
        drop(runtime);
        assert!(!released.load(Ordering::SeqCst), "the runtime-bound lease retains the execution domain");

        let reset = TestCudaHooks::install(vec![TestCudaCall::GetDevice {
            status: cudf_sys::CUDF_ERROR_FATAL_CUDA,
            device: 0,
        }]);
        let before = reset.health_snapshot(DeviceId::DEFAULT).quarantined_teardown_count();

        drop(handle);

        assert!(reset.health_snapshot(DeviceId::DEFAULT).quarantined_teardown_count() > before);
        assert!(
            released.load(Ordering::SeqCst),
            "quarantining the native owner must not retain or cycle the cudf runtime"
        );
    }

    fn test_for_runtime_uses_runtime_shim_instead_of_direct_cugraph_ownership() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));

        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");

        match &handle.state {
            ResourceHandleState::RuntimeBound { lease } => {
                let entered = handle.enter("runtime_state_pointer").expect("enter runtime-bound state");
                assert!(!entered.handle_ptr().is_null(), "entered runtime-bound state must expose a live pointer");
                assert_eq!(lease.device(), handle.device(), "runtime-bound state must carry the handle device");
            },
            state => panic!("expected runtime-bound state variant, got {}", state_variant_name(state)),
        }
    }

    #[test]
    fn test_from_raw_resource_handle_rejects_null() {
        // SAFETY: this exercises input validation on a null pointer only.
        let err = unsafe { ResourceHandle::from_raw_resource_handle(ptr::null_mut()) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raw_resource_handle");
    }

    fn test_from_raw_resource_handle_uses_current_device() {
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();
        let current = DeviceId::current().expect("current device");

        // SAFETY: `raw` comes from a live handle; the test only validates the
        // new current-device based constructor path.
        let wrapped = unsafe { ResourceHandle::from_raw_resource_handle(raw) }.expect("from_raw_resource_handle");
        assert_eq!(wrapped.device(), current.ordinal() as i32);
        match &wrapped.state {
            ResourceHandleState::ExternalBorrowed { raw: state_raw, device, .. } => {
                assert_eq!(*state_raw, raw);
                assert_eq!(*device, current.ordinal() as i32);
            },
            state => panic!("expected external borrowed state variant, got {}", state_variant_name(state)),
        }
    }

    #[test]
    fn test_adopt_raw_resource_handle_rejects_null() {
        // SAFETY: this exercises input validation on a null pointer only.
        let err =
            unsafe { ResourceHandle::adopt_raw_resource_handle(ptr::null_mut()) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::adopt_raw_resource_handle");
    }

    fn test_adopt_raw_resource_handle_uses_current_device() {
        // SAFETY: cuGraph returns a newly allocated raw handle or null.
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");
        let current = DeviceId::current().expect("current device");

        // SAFETY: ownership of `raw` is intentionally transferred to the wrapper.
        let adopted = unsafe { ResourceHandle::adopt_raw_resource_handle(raw) }.expect("adopt_raw_resource_handle");
        assert_eq!(adopted.device(), current.ordinal() as i32);
    }

    fn test_from_raw_resource_handle_does_not_free_on_drop() {
        // Create an owned handle, take its raw pointer, wrap it in a
        // non-owning handle, and drop the non-owning one first. The owned
        // handle should still be valid afterwards (no double-free).
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();

        {
            // SAFETY: `raw` is valid for the lifetime of `owned`.
            let borrowed = unsafe { ResourceHandle::from_raw_resource_handle(raw) }.expect("from_raw_resource_handle");
            assert!(!borrowed.as_ptr().is_null());
            // `borrowed` drops here without freeing `raw`.
        }

        // `owned` is still valid; dropping it cleans up the actual handle.
        assert!(!owned.as_ptr().is_null());
    }

    fn test_adopt_raw_resource_handle_marks_owned() {
        // SAFETY: cuGraph returns a newly allocated raw handle or null.
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");

        // SAFETY: ownership of `raw` is intentionally transferred to the wrapper.
        let adopted = unsafe { ResourceHandle::adopt_raw_resource_handle(raw) }.expect("adopt_raw_resource_handle");
        assert_eq!(adopted.as_ptr(), raw);
        match &adopted.state {
            ResourceHandleState::ExternalOwned { raw: state_raw, .. } => {
                assert_eq!(*state_raw, raw, "external owned state must retain the adopted raw handle");
            },
            state => panic!("expected external owned state variant, got {}", state_variant_name(state)),
        }
    }

    #[test]
    fn test_enter_temporarily_enters_owner_device_and_restores_ambient_device() {
        let raw = std::ptr::NonNull::<cugraph_resource_handle_t>::dangling().as_ptr();
        let handle = ResourceHandle {
            state: ResourceHandleState::ExternalBorrowed { raw, device: 1 },
            _nosend_nosync: PhantomData,
        };
        let _reset = TestCudaHooks::install(vec![
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
            TestCudaCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_SUCCESS },
        ]);

        {
            let entered = handle.enter("ResourceHandle::enter").expect("entered context");
            assert_eq!(entered.device(), 1);
            assert_eq!(entered.handle_ptr(), raw);
            assert!(entered.runtime().is_none());
        }
    }

    #[test]
    fn test_restore_failure_rejects_future_work_on_failed_ambient_device() {
        let raw = std::ptr::NonNull::<cugraph_resource_handle_t>::dangling().as_ptr();
        let handle = ResourceHandle {
            state: ResourceHandleState::ExternalBorrowed { raw, device: 1 },
            _nosend_nosync: PhantomData,
        };
        let _reset = TestCudaHooks::install(vec![
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
            TestCudaCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_ERROR_FATAL_CUDA },
        ]);

        {
            let _entered = handle.enter("first_enter").expect("entered context");
        }

        let failed_ambient_handle = ResourceHandle {
            state: ResourceHandleState::ExternalBorrowed { raw, device: 0 },
            _nosend_nosync: PhantomData,
        };
        let err = failed_ambient_handle
            .enter("unhealthy_ambient_enter")
            .expect_err("failed ambient device should reject later work");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "unhealthy_ambient_enter");
        assert!(err.is_persistent());
    }

    fn test_entered_context_exposes_runtime_provenance() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");

        let entered = handle.enter("ResourceHandle::enter").expect("entered context");
        let session_runtime = entered.runtime().expect("runtime-bound handle should expose runtime");
        assert_eq!(session_runtime.device().ordinal(), runtime.device().ordinal());
    }

    fn runtime_handle_effective_memory_tracks_aligned_domain_charge() {
        const CAPACITY_BYTES: usize = 1 << 20;
        let (runtime, budget) = budgeted_runtime(CAPACITY_BYTES);
        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");
        let entered = handle.enter("effective_memory_charge_test").expect("entered context");

        assert_eq!(effective_device_memory(&entered), CAPACITY_BYTES);
        let allocation = DeviceArray::from_slice(&entered, &[7_i32; 257]).expect("device allocation");
        let usage = budget.budget_tracking_usage().expect("budget usage");
        assert!(usage.current_aligned_bytes > 0);
        assert_eq!(
            effective_device_memory(&entered),
            CAPACITY_BYTES - usize::try_from(usage.current_aligned_bytes).expect("aligned charge fits usize")
        );

        drop(allocation);
        assert_eq!(budget.budget_tracking_usage().expect("released usage").current_aligned_bytes, 0);
        assert_eq!(effective_device_memory(&entered), CAPACITY_BYTES);
    }

    #[test]
    fn runtime_handle_effective_memory_accepts_query_domain_member_resource() {
        const CAPACITY_BYTES: usize = 1 << 20;
        let base = cudf::Runtime::builder().with_device_memory_manager().build().expect("base runtime");
        let manager = base.device_memory_manager().expect("device memory manager");
        let domain = manager
            .query_memory_domain(
                81_001,
                QueryMemoryDomainOptions::new(CAPACITY_BYTES).with_ceiling_bytes(CAPACITY_BYTES),
            )
            .expect("query memory domain");
        let lease = domain.reserve_stream(base.stream(), 81_101, CAPACITY_BYTES).expect("domain member");
        let runtime = Arc::new(
            cudf::Runtime::builder()
                .with_device(base.device())
                .with_stream(base.stream().clone())
                .with_memory_resource(lease.memory_resource().clone())
                .build()
                .expect("member runtime"),
        );
        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");
        let entered = handle.enter("query_domain_member_effective_memory_test").expect("entered context");

        assert_eq!(effective_device_memory(&entered), CAPACITY_BYTES);
        let allocation = DeviceArray::from_slice(&entered, &[9_i32; 257]).expect("device allocation");
        let usage = domain.usage().expect("domain usage");
        assert!(usage.current_aligned_bytes > 0);
        assert_eq!(
            effective_device_memory(&entered),
            CAPACITY_BYTES - usize::try_from(usage.current_aligned_bytes).expect("aligned charge fits usize")
        );

        drop(allocation);
        assert_eq!(domain.usage().expect("released usage").current_aligned_bytes, 0);
        assert_eq!(effective_device_memory(&entered), CAPACITY_BYTES);
    }

    fn runtime_handles_keep_distinct_effective_memory_providers() {
        const SMALL_CAPACITY_BYTES: usize = 1 << 20;
        const LARGE_CAPACITY_BYTES: usize = 2 << 20;
        let (small_runtime, _small_budget) = budgeted_runtime(SMALL_CAPACITY_BYTES);
        let (large_runtime, _large_budget) = budgeted_runtime(LARGE_CAPACITY_BYTES);
        let small = ResourceHandle::for_runtime(small_runtime).expect("small resource handle");
        let large = ResourceHandle::for_runtime(large_runtime).expect("large resource handle");

        let small_entered = small.enter("small_effective_memory_test").expect("small entered context");
        assert_eq!(effective_device_memory(&small_entered), SMALL_CAPACITY_BYTES);
        drop(small_entered);
        let large_entered = large.enter("large_effective_memory_test").expect("large entered context");
        assert_eq!(effective_device_memory(&large_entered), LARGE_CAPACITY_BYTES);
    }

    fn nested_entry_for_same_native_owner_is_rejected_without_deadlock() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let first = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("first resource handle");
        let second = ResourceHandle::for_runtime(runtime).expect("second resource handle");
        let first_lease = first.runtime_bound_lease().expect("first lease");
        let second_lease = second.runtime_bound_lease().expect("second lease");
        assert!(first_lease.generation_matches(&second_lease));

        let first_entered = first.enter("nested_same_owner_first").expect("first entered context");
        let error = second.enter("nested_same_owner_second").expect_err("nested same-owner entry must be rejected");
        assert_eq!(error.kind(), ErrorKind::InvalidState);
        assert_eq!(error.operation(), "nested_same_owner_second");
        assert!(error.to_string().contains("nested_binding_extension_execution"));

        drop(first_entered);
        second.enter("same_owner_after_release").expect("entry succeeds after prior context drops");
    }

    #[test]
    fn concurrent_entry_for_same_native_owner_is_serialized() {
        use std::{sync::mpsc, time::Duration};

        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let first = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("first resource handle");
        let first_entered = first.enter("concurrent_same_owner_first").expect("first entered context");
        let first_raw = first_entered.handle_ptr() as usize;
        let (ready_tx, ready_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();

        let worker = std::thread::spawn(move || {
            let second = ResourceHandle::for_runtime(runtime).expect("second resource handle");
            ready_tx.send(()).expect("send ready");
            let _entered = second.enter("concurrent_same_owner_second").expect("serialized second entry");
            acquired_tx.send(_entered.handle_ptr() as usize).expect("send acquired");
        });

        ready_rx.recv_timeout(Duration::from_secs(5)).expect("worker ready");
        assert!(
            acquired_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "second entry must remain blocked while the first context owns the native handle"
        );
        drop(first_entered);
        assert_eq!(acquired_rx.recv_timeout(Duration::from_secs(5)).expect("worker acquired after release"), first_raw);
        worker.join().expect("concurrent entry worker");
    }

    fn test_into_raw_resource_handle_releases_rust_ownership() {
        let handle = ResourceHandle::new().expect("resource handle");
        let device = handle.device();
        let mut raw = handle.into_raw_resource_handle().expect("into_raw_resource_handle");
        assert!(!raw.is_null(), "into_raw_resource_handle should return the underlying pointer");

        drop_device_resource(
            &mut raw,
            device,
            "test_into_raw_resource_handle_releases_rust_ownership",
            DeviceResourceKind::ResourceHandle,
        );
    }

    fn test_runtime_bound_into_raw_resource_handle_is_rejected() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");

        let err = handle.into_raw_resource_handle().expect_err("runtime-bound handle should reject raw extraction");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ResourceHandle::into_raw_resource_handle");
    }
}
