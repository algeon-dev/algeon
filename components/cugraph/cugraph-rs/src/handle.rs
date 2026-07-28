//! GPU resource handle for cuGraph operations.
//!
//! Every cuGraph operation requires a [`ResourceHandle`] which manages the
//! underlying CUDA stream and cuGraph resource handle. The handle is **not**
//! `Send` or `Sync` because GPU state is thread-local.

use std::{fmt, marker::PhantomData, mem::ManuallyDrop, ptr};

use cudf::DeviceId;
use cugraph_sys::{
    cugraph_create_resource_handle, cugraph_free_resource_handle, cugraph_resource_handle_get_comm_size,
    cugraph_resource_handle_get_rank, cugraph_resource_handle_t,
};

use crate::{
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommMetadata {
    DefaultSingleGpu,
    ExternalHandle,
}

/// RAII wrapper around `cugraph_resource_handle_t`.
///
/// Manages GPU resources (CUDA stream and the underlying cuGraph resource
/// handle). Automatically freed on drop (unless created via
/// [`from_raw_resource_handle`](Self::from_raw_resource_handle), which borrows
/// an external cuGraph resource handle). Not thread-safe — must be used from
/// the thread that created it.
pub struct ResourceHandle {
    raw: *mut cugraph_resource_handle_t,
    device: i32,
    /// Whether this wrapper owns the handle and should free it on drop.
    owned: bool,
    comm_metadata: CommMetadata,
    // !Send + !Sync marker: cuGraph handle owns thread-affine GPU resources.
    _nosend_nosync: PhantomData<*mut ()>,
}

impl fmt::Debug for ResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceHandle").finish_non_exhaustive()
    }
}

impl ResourceHandle {
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
        let device = DeviceId::current().map_err(|error| {
            Error::new(ErrorKind::Unexpected, "ResourceHandle::for_current_device")
                .with_message(format!("failed to query current CUDA device: {}", error))
                .set_source(error)
        })?;
        Self::new_on_device(device.ordinal() as i32)
    }

    /// Create a new resource handle on the same CUDA device as a cudf runtime.
    ///
    /// This aligns cuGraph device placement with the supplied
    /// [`cudf::Runtime`], but it does not reuse the runtime's CUDA stream or
    /// memory resource.
    pub fn for_runtime_device(runtime: &cudf::Runtime) -> Result<Self> {
        Self::new_on_device(runtime.device().ordinal() as i32)
    }

    /// Create a new resource handle on the specified CUDA device.
    pub(crate) fn new_on_device(device: i32) -> Result<Self> {
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
        Ok(Self {
            raw,
            device,
            owned: true,
            comm_metadata: CommMetadata::DefaultSingleGpu,
            _nosend_nosync: PhantomData,
        })
    }

    /// Wrap an existing cuGraph resource handle created externally.
    ///
    /// This constructor is intended for multi-GPU workflows where a
    /// per-thread `cugraph_resource_handle_t*` has already been created
    /// externally (for example by an MTMG or MPI-aware native layer). The
    /// returned wrapper does **not** own the handle — it will not be freed on
    /// drop.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `raw` is a valid, non-null pointer to a `cugraph_resource_handle_t`
    ///   created by cuGraph-compatible native code
    /// - `device` is the CUDA device ordinal that this handle is bound to
    /// - the handle remains valid for the lifetime of this `ResourceHandle`
    /// - the handle is not freed externally while this wrapper exists
    /// - the calling thread obeys the external runtime's thread-affinity rules
    pub unsafe fn from_raw_resource_handle(raw: *mut cugraph_resource_handle_t, device: i32) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }

        if device < 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raw_resource_handle")
                .with_message(format!("device ordinal must be >= 0, got {}", device)));
        }

        DeviceId::new(device as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raw_resource_handle")
                .with_message(format!("invalid CUDA device ordinal {}: {}", device, error))
                .set_source(error)
        })?;
        Ok(Self { raw, device, owned: false, comm_metadata: CommMetadata::ExternalHandle, _nosend_nosync: PhantomData })
    }

    /// Adopt ownership of an existing cuGraph resource handle created externally.
    ///
    /// This constructor is intended for native MTMG or MPI-aware shims that
    /// allocate a fresh `cugraph_resource_handle_t*` for Rust and transfer
    /// destruction responsibility to Rust. The returned wrapper owns the
    /// handle and will free it on drop using `cugraph_free_resource_handle`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `raw` is a valid, non-null pointer to a `cugraph_resource_handle_t`
    /// - ownership of `raw` is transferred exactly once to this wrapper
    /// - no other owner will free `raw` after this call succeeds
    /// - `device` is the CUDA device ordinal that this handle is bound to
    /// - the calling thread obeys the external runtime's thread-affinity rules
    ///
    /// For a borrowed/non-owning wrapper, use
    /// [`from_raw_resource_handle`](Self::from_raw_resource_handle) instead.
    pub unsafe fn adopt_raw_resource_handle(raw: *mut cugraph_resource_handle_t, device: i32) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::adopt_raw_resource_handle")
                .with_message("resource handle pointer must not be null"));
        }

        if device < 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::adopt_raw_resource_handle")
                .with_message(format!("device ordinal must be >= 0, got {}", device)));
        }

        DeviceId::new(device as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, "ResourceHandle::adopt_raw_resource_handle")
                .with_message(format!("invalid CUDA device ordinal {}: {}", device, error))
                .set_source(error)
        })?;
        Ok(Self { raw, device, owned: true, comm_metadata: CommMetadata::ExternalHandle, _nosend_nosync: PhantomData })
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
        note = "use ResourceHandle::from_raw_resource_handle(raw, device); this API wraps a cugraph_resource_handle_t*, not a raft::handle_t*"
    )]
    pub unsafe fn from_raft_handle(raw: *mut cugraph_resource_handle_t, device: i32) -> Result<Self> {
        // SAFETY: this deprecated alias forwards the exact same contract to
        // `from_raw_resource_handle`.
        unsafe { Self::from_raw_resource_handle(raw, device) }
    }

    /// Get the raw C pointer. For internal use by other modules.
    pub(crate) fn as_ptr(&self) -> *mut cugraph_resource_handle_t {
        self.raw
    }

    /// Borrow the underlying raw cuGraph resource-handle pointer.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `self` remains alive, and callers must not free it unless
    /// ownership has been transferred separately.
    pub fn as_raw_resource_handle(&self) -> *mut cugraph_resource_handle_t {
        self.raw
    }

    /// Consume this wrapper and return the underlying raw handle pointer.
    ///
    /// This suppresses Rust-side `Drop`, so no automatic
    /// `cugraph_free_resource_handle` call will occur after this returns.
    ///
    /// Ownership semantics are preserved from the source wrapper:
    /// - if `self` owned the handle, the caller becomes responsible for
    ///   eventually freeing it
    /// - if `self` only borrowed the handle, the caller still does not own it
    ///   and must follow the original external destruction contract
    pub fn into_raw_resource_handle(self) -> *mut cugraph_resource_handle_t {
        let this = ManuallyDrop::new(self);
        this.raw
    }

    /// The CUDA device this handle was created on.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// The communicator rank carried by this handle.
    ///
    /// Handles created through [`for_runtime_device`](Self::for_runtime_device)
    /// always return `0`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn rank(&self) -> i32 {
        match self.comm_metadata {
            CommMetadata::DefaultSingleGpu => 0,
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle for the lifetime of `self`.
            CommMetadata::ExternalHandle => unsafe { cugraph_resource_handle_get_rank(self.raw) },
        }
    }

    /// The communicator size carried by this handle.
    ///
    /// Handles created through [`for_runtime_device`](Self::for_runtime_device)
    /// always return `1`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn comm_size(&self) -> i32 {
        match self.comm_metadata {
            CommMetadata::DefaultSingleGpu => 1,
            // SAFETY: externally wrapped handles promise a live cuGraph
            // resource handle for the lifetime of `self`.
            CommMetadata::ExternalHandle => unsafe { cugraph_resource_handle_get_comm_size(self.raw) },
        }
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to(self.device, "ResourceHandle::drop").ok();
            // SAFETY: owned handles are freed exactly once here.
            unsafe { cugraph_free_resource_handle(self.raw) };
        }
    }
}

// ResourceHandle is NOT Send/Sync — GPU state is thread-local.

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::*;

    #[test]
    fn test_resource_handle_creation() {
        let handle = ResourceHandle::new().expect("Failed to create resource handle");
        // Handle should be non-null and droppable
        assert!(!handle.as_ptr().is_null());
    }

    #[test]
    fn test_resource_handle_not_send_sync() {
        assert_not_impl_any!(ResourceHandle: Send, Sync);
    }

    #[test]
    fn test_for_runtime_device_matches_runtime_device() {
        let runtime = cudf::Runtime::builder().build().expect("runtime");
        let handle = ResourceHandle::for_runtime_device(&runtime).expect("resource handle");

        assert_eq!(handle.device(), runtime.device().ordinal() as i32);
    }

    #[test]
    fn test_from_raw_resource_handle_rejects_null() {
        // SAFETY: this exercises input validation on a null pointer only.
        let err =
            unsafe { ResourceHandle::from_raw_resource_handle(ptr::null_mut(), 0) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raw_resource_handle");
    }

    #[test]
    fn test_from_raw_resource_handle_rejects_invalid_device() {
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();

        // SAFETY: `raw` comes from a live handle; the test only validates the
        // device argument path.
        let err =
            unsafe { ResourceHandle::from_raw_resource_handle(raw, -1) }.expect_err("should reject invalid device");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raw_resource_handle");
    }

    #[test]
    fn test_adopt_raw_resource_handle_rejects_null() {
        // SAFETY: this exercises input validation on a null pointer only.
        let err =
            unsafe { ResourceHandle::adopt_raw_resource_handle(ptr::null_mut(), 0) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::adopt_raw_resource_handle");
    }

    #[test]
    fn test_adopt_raw_resource_handle_rejects_invalid_device() {
        // SAFETY: cuGraph returns a newly allocated raw handle or null.
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");

        // SAFETY: `raw` is live; the test only validates the device argument path.
        let err =
            unsafe { ResourceHandle::adopt_raw_resource_handle(raw, -1) }.expect_err("should reject invalid device");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::adopt_raw_resource_handle");

        // SAFETY: `raw` was allocated above and has not been freed yet.
        unsafe { cugraph_free_resource_handle(raw) };
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
            let borrowed = unsafe { ResourceHandle::from_raw_resource_handle(raw, owned.device()) }
                .expect("from_raw_resource_handle");
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
        let adopted = unsafe { ResourceHandle::adopt_raw_resource_handle(raw, 0) }.expect("adopt_raw_resource_handle");
        assert_eq!(adopted.as_ptr(), raw);
        assert!(adopted.owned, "adopted handle should transfer destruction to Rust");
    }

    #[test]
    fn test_as_raw_resource_handle_exposes_pointer() {
        let handle = ResourceHandle::new().expect("resource handle");
        assert_eq!(handle.as_raw_resource_handle(), handle.as_ptr());
    }

    #[test]
    fn test_into_raw_resource_handle_releases_rust_ownership() {
        let handle = ResourceHandle::new().expect("resource handle");
        let raw = handle.into_raw_resource_handle();
        assert!(!raw.is_null(), "into_raw_resource_handle should return the underlying pointer");

        // SAFETY: ownership was intentionally released from `handle` above.
        unsafe { cugraph_free_resource_handle(raw) };
    }

    #[test]
    fn test_rank_and_comm_size_on_single_gpu_handle() {
        let handle = ResourceHandle::new().expect("resource handle");
        assert_eq!(handle.rank(), 0);
        assert_eq!(handle.comm_size(), 1);
    }
}
