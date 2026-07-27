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

use crate::error::{Error, ErrorKind, Result};

fn check_cudf_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let error = match status {
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => Error::new(ErrorKind::OutOfMemory, operation)
            .with_message(format!("cudf CUDA helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf CUDA helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_FATAL_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf CUDA helper failed with status {}", status))
            .set_persistent(),
        _ => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf CUDA helper failed with status {}", status)),
    };
    Err(error)
}

struct DeviceGuard {
    previous: i32,
    restore: bool,
}

impl DeviceGuard {
    fn switch_to(target: i32, operation: &'static str) -> Result<Self> {
        let mut current = 0;
        check_cudf_status(unsafe { cudf_sys::cudf_cuda_get_device(&mut current) }, operation)?;

        if current != target {
            check_cudf_status(unsafe { cudf_sys::cudf_cuda_set_device(target) }, operation)?;
        }

        Ok(Self { previous: current, restore: current != target })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.restore {
            let _ = unsafe { cudf_sys::cudf_cuda_set_device(self.previous) };
        }
    }
}

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
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::ResourceHandle;
    /// let handle =
    ///     ResourceHandle::for_current_device().expect("GPU must be available");
    /// ```
    pub fn new() -> Result<Self> {
        Self::for_current_device()
    }

    /// Create a new resource handle on the calling thread's current CUDA device.
    pub fn for_current_device() -> Result<Self> {
        let device = DeviceId::current().map_err(|error| {
            Error::new(ErrorKind::Unexpected, "ResourceHandle::for_current_device")
                .with_message(format!("failed to query current CUDA device: {}", error))
                .set_source(error)
        })?;
        Self::new_on_device(device.ordinal() as i32)
    }

    /// Create a new resource handle on the specified CUDA device.
    pub fn new_on_device(device: i32) -> Result<Self> {
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
    #[deprecated(
        note = "use ResourceHandle::from_raw_resource_handle(raw, device); this API wraps a cugraph_resource_handle_t*, not a raft::handle_t*"
    )]
    pub unsafe fn from_raft_handle(raw: *mut cugraph_resource_handle_t, device: i32) -> Result<Self> {
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
    /// Handles created through [`new`](Self::new),
    /// [`for_current_device`](Self::for_current_device), or
    /// [`new_on_device`](Self::new_on_device) always return `0`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn rank(&self) -> i32 {
        match self.comm_metadata {
            CommMetadata::DefaultSingleGpu => 0,
            CommMetadata::ExternalHandle => unsafe { cugraph_resource_handle_get_rank(self.raw) },
        }
    }

    /// The communicator size carried by this handle.
    ///
    /// Handles created through [`new`](Self::new),
    /// [`for_current_device`](Self::for_current_device), or
    /// [`new_on_device`](Self::new_on_device) always return `1`.
    ///
    /// Externally wrapped handles query cuGraph's communicator metadata and
    /// are intended for MTMG/MG-native handles that already carry a valid
    /// communicator.
    pub fn comm_size(&self) -> i32 {
        match self.comm_metadata {
            CommMetadata::DefaultSingleGpu => 1,
            CommMetadata::ExternalHandle => unsafe { cugraph_resource_handle_get_comm_size(self.raw) },
        }
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
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
    fn test_from_raw_resource_handle_rejects_null() {
        let err =
            unsafe { ResourceHandle::from_raw_resource_handle(ptr::null_mut(), 0) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raw_resource_handle");
    }

    #[test]
    fn test_from_raw_resource_handle_rejects_invalid_device() {
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();

        let err =
            unsafe { ResourceHandle::from_raw_resource_handle(raw, -1) }.expect_err("should reject invalid device");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raw_resource_handle");
    }

    #[test]
    fn test_adopt_raw_resource_handle_rejects_null() {
        let err =
            unsafe { ResourceHandle::adopt_raw_resource_handle(ptr::null_mut(), 0) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::adopt_raw_resource_handle");
    }

    #[test]
    fn test_adopt_raw_resource_handle_rejects_invalid_device() {
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");

        let err =
            unsafe { ResourceHandle::adopt_raw_resource_handle(raw, -1) }.expect_err("should reject invalid device");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::adopt_raw_resource_handle");

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
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        assert!(!raw.is_null(), "native handle allocation should succeed");

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

        unsafe { cugraph_free_resource_handle(raw) };
    }

    #[test]
    fn test_rank_and_comm_size_on_single_gpu_handle() {
        let handle = ResourceHandle::new().expect("resource handle");
        assert_eq!(handle.rank(), 0);
        assert_eq!(handle.comm_size(), 1);
    }
}
