//! GPU resource handle for cuGraph operations.
//!
//! [`ResourceHandle`] owns cuGraph execution provenance while
//! [`EnteredContext`] is the short-lived execution capability used for GPU
//! work on the current thread. Both types are **not** `Send` or `Sync`
//! because GPU state is thread-local.

use std::{fmt, marker::PhantomData, mem::ManuallyDrop, ptr, ptr::NonNull, rc::Rc, sync::Arc};

use cudf::DeviceId;
use cugraph_sys::{
    cugraph_create_resource_handle, cugraph_free_resource_handle, cugraph_resource_handle_get_comm_size,
    cugraph_resource_handle_get_rank, cugraph_resource_handle_t,
};
use rapids_interop::{ExecutionBinding, InteropError, InteropErrorKind};

use crate::{
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result},
};

enum ResourceHandleState {
    LocalOwned { raw: *mut cugraph_resource_handle_t, device: i32 },
    RuntimeBound { lease: ResourceHandleLease },
    ExternalBorrowed { raw: *mut cugraph_resource_handle_t, device: i32 },
    ExternalOwned { raw: *mut cugraph_resource_handle_t, device: i32 },
}

impl ResourceHandleState {
    fn raw(&self) -> *mut cugraph_resource_handle_t {
        match self {
            Self::LocalOwned { raw, .. } | Self::ExternalBorrowed { raw, .. } | Self::ExternalOwned { raw, .. } => *raw,
            Self::RuntimeBound { lease } => lease.raw(),
        }
    }

    fn device(&self) -> i32 {
        match self {
            Self::LocalOwned { device, .. }
            | Self::ExternalBorrowed { device, .. }
            | Self::ExternalOwned { device, .. } => *device,
            Self::RuntimeBound { lease } => lease.device(),
        }
    }

    fn runtime(&self) -> Option<&cudf::Runtime> {
        match self {
            Self::RuntimeBound { lease } => Some(lease.runtime()),
            Self::LocalOwned { .. } | Self::ExternalBorrowed { .. } | Self::ExternalOwned { .. } => None,
        }
    }

    fn runtime_bound_lease(&self) -> Option<ResourceHandleLease> {
        match self {
            Self::RuntimeBound { lease } => Some(lease.clone()),
            Self::LocalOwned { .. } | Self::ExternalBorrowed { .. } | Self::ExternalOwned { .. } => None,
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
#[allow(non_camel_case_types)]
struct cugraph_rs_runtime_resource_handle_t {
    _private: [u8; 0],
}

extern "C" {
    fn cugraph_rs_runtime_resource_handle_create(
        cuda_stream: *mut std::ffi::c_void,
        device_memory_resource: *mut std::ffi::c_void,
    ) -> *mut cugraph_rs_runtime_resource_handle_t;
    fn cugraph_rs_runtime_resource_handle_get(
        handle: *mut cugraph_rs_runtime_resource_handle_t,
    ) -> *mut cugraph_resource_handle_t;
    fn cugraph_rs_runtime_resource_handle_destroy(handle: *mut cugraph_rs_runtime_resource_handle_t);
}

struct RuntimeResourceHandleOwner {
    raw: NonNull<cugraph_rs_runtime_resource_handle_t>,
    device: i32,
}

impl RuntimeResourceHandleOwner {
    fn create(runtime: &Arc<cudf::Runtime>, operation: &'static str) -> Result<(Self, *mut cugraph_resource_handle_t)> {
        let binding =
            runtime.execution_binding().map_err(|error| map_runtime_execution_binding_error(error, operation))?;
        let runtime_device = runtime.device().ordinal() as i32;
        validate_runtime_binding_device(runtime_device, binding, operation)?;

        let _guard = DeviceGuard::switch_to(runtime_device, operation)?;
        // SAFETY: `binding` is borrowed from a live explicit cudf runtime. The
        // shim borrows the stream and memory-resource handles for exactly the
        // lifetime guarded by the retained `Arc<cudf::Runtime>`.
        let shim = unsafe {
            cugraph_rs_runtime_resource_handle_create(
                binding.cuda_stream().as_raw(),
                binding.memory_resource().as_raw(),
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
            // SAFETY: `raw` owns a live shim handle created just above.
            unsafe { cugraph_rs_runtime_resource_handle_destroy(raw.as_ptr()) };
            return Err(Error::new(ErrorKind::Unexpected, operation)
                .with_message("runtime-bound handle shim returned a null cuGraph resource handle")
                .with_context("device", runtime_device.to_string()));
        }

        Ok((Self { raw, device: runtime_device }, resource_handle))
    }
}

impl Drop for RuntimeResourceHandleOwner {
    fn drop(&mut self) {
        let _guard = DeviceGuard::switch_to_for_drop(self.device, "RuntimeResourceHandleOwner::drop").ok();
        // SAFETY: this owner is the unique runtime-bound owner for the shim.
        unsafe { cugraph_rs_runtime_resource_handle_destroy(self.raw.as_ptr()) };
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
    // Keep `shim` before `runtime` so the native runtime-bound shim is torn
    // down before releasing the retained cudf runtime owner.
    shim: Rc<RuntimeResourceHandleOwner>,
    runtime: Arc<cudf::Runtime>,
    raw: *mut cugraph_resource_handle_t,
    device: i32,
}

impl ResourceHandleLease {
    fn new(
        shim: RuntimeResourceHandleOwner,
        runtime: Arc<cudf::Runtime>,
        raw: *mut cugraph_resource_handle_t,
        device: i32,
    ) -> Self {
        Self { shim: Rc::new(shim), runtime, raw, device }
    }

    fn raw(&self) -> *mut cugraph_resource_handle_t {
        self.raw
    }

    fn device(&self) -> i32 {
        self.device
    }

    fn runtime(&self) -> &cudf::Runtime {
        self.runtime.as_ref()
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.shim, &other.shim)
    }

    #[cfg(test)]
    fn shim_device(&self) -> i32 {
        self.shim.device
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
/// an external cuGraph resource handle). Not thread-safe — must be used from
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
    #[allow(dead_code)]
    guard: DeviceGuard,
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
        self.owner.state.raw()
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
    /// Internal helper for source-local tests and implementation plumbing.
    #[allow(dead_code)]
    pub(crate) fn new() -> Result<Self> {
        Self::for_current_device()
    }

    /// Create a new resource handle on the calling thread's current CUDA device.
    #[allow(dead_code)]
    pub(crate) fn for_current_device() -> Result<Self> {
        Self::new_on_device(Self::current_device_ordinal("ResourceHandle::for_current_device")?)
    }

    /// Create a new resource handle bound to a cudf runtime owner.
    ///
    /// This constructor binds cuGraph to the runtime's CUDA stream and RAFT
    /// workspace/resource-aware allocation context through a local native shim
    /// that owns both the RAFT handle and the cuGraph resource-handle wrapper.
    pub fn for_runtime(runtime: Arc<cudf::Runtime>) -> Result<Self> {
        let operation = "ResourceHandle::for_runtime";
        let device = runtime.device().ordinal() as i32;
        let (shim, raw) = RuntimeResourceHandleOwner::create(&runtime, operation)?;
        let lease = ResourceHandleLease::new(shim, runtime, raw, device);
        Ok(Self { state: ResourceHandleState::RuntimeBound { lease }, _nosend_nosync: PhantomData })
    }

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

        let _guard = DeviceGuard::switch_to(device, "ResourceHandle::new_on_device")?;
        // SAFETY: cuGraph creates a fresh handle and does not retain the null
        // communicator pointer beyond the call.
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
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
    /// returned wrapper does **not** own the handle — it will not be freed on
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
    pub unsafe fn from_raw_resource_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }
        let device = Self::current_device_ordinal("ResourceHandle::from_raw_resource_handle")?;
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
    ///
    /// For a borrowed/non-owning wrapper, use
    /// [`from_raw_resource_handle`](Self::from_raw_resource_handle) instead.
    pub unsafe fn adopt_raw_resource_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::adopt_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }
        let device = Self::current_device_ordinal("ResourceHandle::adopt_raw_resource_handle")?;
        Ok(Self { state: ResourceHandleState::ExternalOwned { raw, device }, _nosend_nosync: PhantomData })
    }

    /// Wrap an existing external cuGraph resource handle.
    ///
    /// Deprecated alias for
    /// [`from_raw_resource_handle`](Self::from_raw_resource_handle).
    ///
    /// # Safety
    ///
    /// Same safety requirements as
    /// [`from_raw_resource_handle`](Self::from_raw_resource_handle).
    #[deprecated(
        note = "use ResourceHandle::from_raw_resource_handle(raw); this API wraps a cugraph_resource_handle_t*, not a raft::handle_t*"
    )]
    pub unsafe fn from_raft_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        // SAFETY: this deprecated alias forwards the exact same contract to
        // `from_raw_resource_handle`.
        unsafe { Self::from_raw_resource_handle(raw) }
    }

    /// Get the raw C pointer. For internal use by other modules.
    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *mut cugraph_resource_handle_t {
        self.state.raw()
    }

    /// Borrow the underlying raw cuGraph resource-handle pointer.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `self` remains alive, and callers must not free it unless
    /// ownership has been transferred separately.
    #[allow(dead_code)]
    pub(crate) fn as_raw_resource_handle(&self) -> *mut cugraph_resource_handle_t {
        self.state.raw()
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
        Ok(this.state.raw())
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
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle for the lifetime of `self`.
            unsafe { cugraph_resource_handle_get_rank(self.state.raw()) }
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
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle for the lifetime of `self`.
            unsafe { cugraph_resource_handle_get_comm_size(self.state.raw()) }
        } else {
            1
        }
    }

    /// Enter the owner device on the current thread and create an execution
    /// session for GPU-capable work.
    pub fn enter(&self, operation: &'static str) -> Result<EnteredContext<'_>> {
        let guard = DeviceGuard::switch_to(self.device(), operation)?;
        Ok(EnteredContext { owner: self, guard, _nosend_nosync: PhantomData })
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        match &mut self.state {
            ResourceHandleState::LocalOwned { raw, device } | ResourceHandleState::ExternalOwned { raw, device }
                if !raw.is_null() =>
            {
                let _guard = DeviceGuard::switch_to_for_drop(*device, "ResourceHandle::drop").ok();
                // SAFETY: owned handles are freed exactly once here.
                unsafe { cugraph_free_resource_handle(*raw) };
                *raw = ptr::null_mut();
            },
            ResourceHandleState::LocalOwned { .. }
            | ResourceHandleState::RuntimeBound { .. }
            | ResourceHandleState::ExternalBorrowed { .. }
            | ResourceHandleState::ExternalOwned { .. } => {},
        }
    }
}

// ResourceHandle is NOT Send/Sync — GPU state is thread-local.

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::*;
    use crate::cuda::{TestCudaCall, TestCudaHooks};

    fn state_variant_name(state: &ResourceHandleState) -> &'static str {
        match state {
            ResourceHandleState::LocalOwned { .. } => "LocalOwned",
            ResourceHandleState::RuntimeBound { .. } => "RuntimeBound",
            ResourceHandleState::ExternalBorrowed { .. } => "ExternalBorrowed",
            ResourceHandleState::ExternalOwned { .. } => "ExternalOwned",
        }
    }

    #[test]
    fn test_resource_handle_creation() {
        let handle = ResourceHandle::new().expect("Failed to create resource handle");
        // Handle should be non-null and droppable
        assert!(!handle.as_ptr().is_null());
        match &handle.state {
            ResourceHandleState::LocalOwned { raw, device } => {
                assert_eq!(*raw, handle.as_ptr(), "local owned state must carry the created raw handle");
                assert_eq!(*device, handle.device(), "local owned state must carry the handle device");
            },
            state => panic!("expected local owned state variant, got {}", state_variant_name(state)),
        }
    }

    #[test]
    fn test_resource_handle_not_send_sync() {
        assert_not_impl_any!(ResourceHandle: Send, Sync);
    }

    #[test]
    fn test_entered_context_not_send_sync() {
        assert_not_impl_any!(EnteredContext<'static>: Send, Sync);
    }

    #[test]
    fn test_for_runtime_matches_runtime_device() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");

        assert_eq!(handle.device(), runtime.device().ordinal() as i32);
    }

    #[test]
    fn test_for_runtime_retains_execution_owner() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        assert_eq!(Arc::strong_count(&runtime), 1, "test setup should hold one runtime owner");

        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");
        assert_eq!(handle.device(), runtime.device().ordinal() as i32);
        assert_eq!(Arc::strong_count(&runtime), 2, "runtime-bound handle should retain the runtime owner");

        drop(handle);
        assert_eq!(Arc::strong_count(&runtime), 1, "dropping the handle should release the retained runtime owner");
    }

    #[test]
    fn test_for_runtime_uses_runtime_shim_instead_of_direct_cugraph_ownership() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));

        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");

        match &handle.state {
            ResourceHandleState::RuntimeBound { lease } => {
                assert!(!lease.raw().is_null(), "runtime-bound state must retain the live cuGraph handle pointer");
                assert_eq!(lease.device(), handle.device(), "runtime-bound state must carry the handle device");
                assert!(
                    lease.shim_device() >= 0,
                    "runtime-bound state must retain the native shim that owns the coupled RAFT + cuGraph lifetime"
                );
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

    #[test]
    fn test_from_raw_resource_handle_uses_current_device() {
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();
        let current = DeviceId::current().expect("current device");

        // SAFETY: `raw` comes from a live handle; the test only validates the
        // new current-device based constructor path.
        let wrapped = unsafe { ResourceHandle::from_raw_resource_handle(raw) }.expect("from_raw_resource_handle");
        assert_eq!(wrapped.device(), current.ordinal() as i32);
        match &wrapped.state {
            ResourceHandleState::ExternalBorrowed { raw: state_raw, device } => {
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

    #[test]
    fn test_adopt_raw_resource_handle_uses_current_device() {
        // SAFETY: cuGraph returns a newly allocated raw handle or null.
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");
        let current = DeviceId::current().expect("current device");

        // SAFETY: ownership of `raw` is intentionally transferred to the wrapper.
        let adopted = unsafe { ResourceHandle::adopt_raw_resource_handle(raw) }.expect("adopt_raw_resource_handle");
        assert_eq!(adopted.device(), current.ordinal() as i32);
    }

    #[test]
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

        // `owned` is still valid — dropping it cleans up the actual handle.
        assert!(!owned.as_ptr().is_null());
    }

    #[test]
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
    fn test_enter_switches_to_owner_device_and_restores_previous_device() {
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
    fn test_enter_propagates_device_guard_poison() {
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

        let err = handle.enter("poisoned_enter").expect_err("poisoned guard should fail later enter");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "poisoned_enter");
        assert!(err.is_persistent());
    }

    #[test]
    fn test_entered_context_exposes_runtime_provenance() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("resource handle");

        let entered = handle.enter("ResourceHandle::enter").expect("entered context");
        let session_runtime = entered.runtime().expect("runtime-bound handle should expose runtime");
        assert_eq!(session_runtime.device().ordinal(), runtime.device().ordinal());
    }

    #[test]
    fn test_into_raw_resource_handle_releases_rust_ownership() {
        let handle = ResourceHandle::new().expect("resource handle");
        let raw = handle.into_raw_resource_handle().expect("into_raw_resource_handle");
        assert!(!raw.is_null(), "into_raw_resource_handle should return the underlying pointer");

        // SAFETY: ownership was intentionally released from `handle` above.
        unsafe { cugraph_free_resource_handle(raw) };
    }

    #[test]
    fn test_runtime_bound_into_raw_resource_handle_is_rejected() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(runtime).expect("resource handle");

        let err = handle.into_raw_resource_handle().expect_err("runtime-bound handle should reject raw extraction");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ResourceHandle::into_raw_resource_handle");
    }

    #[test]
    fn test_rank_and_comm_size_on_single_gpu_handle() {
        let handle = ResourceHandle::new().expect("resource handle");
        assert_eq!(handle.rank(), 0);
        assert_eq!(handle.comm_size(), 1);
    }
}
