//! GPU resource handle for cuGraph operations.
//!
//! Every cuGraph operation requires a [`ResourceHandle`] which manages the
//! underlying CUDA stream and RAFT handle. The handle is **not** `Send` or
//! `Sync` because GPU state is thread-local.

use std::{fmt, marker::PhantomData, ptr};

use cudf::DeviceId;
use cugraph_sys::{cugraph_create_resource_handle, cugraph_free_resource_handle, cugraph_resource_handle_t};

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

/// RAII wrapper around `cugraph_resource_handle_t`.
///
/// Manages GPU resources (CUDA stream, RAFT handle). Automatically freed on
/// drop (unless created via [`from_raft_handle`](Self::from_raft_handle),
/// which borrows an external RAFT handle). Not thread-safe — must be used
/// from the thread that created it.
pub struct ResourceHandle {
    raw: *mut cugraph_resource_handle_t,
    device: i32,
    /// Whether this wrapper owns the handle and should free it on drop.
    owned: bool,
    // !Send + !Sync marker: cuGraph handle owns thread-affine GPU resources.
    _nosend_nosync: PhantomData<*mut ()>,
}

impl fmt::Debug for ResourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceHandle").finish_non_exhaustive()
    }
}

impl ResourceHandle {
    /// Create a new resource handle with a default internal RAFT handle.
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
        Ok(Self { raw, device, owned: true, _nosend_nosync: PhantomData })
    }

    /// Wrap an existing RAFT resource handle for use with cuGraph.
    ///
    /// This constructor is intended for multi-GPU / MPI workflows where a RAFT
    /// communicator has already been initialized externally (e.g. via
    /// `raft::handle_t` in C++). The returned wrapper does **not** own the
    /// handle — it will not be freed on drop.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `raw` is a valid, non-null pointer to a `cugraph_resource_handle_t`
    ///   obtained from a RAFT runtime
    /// - the handle remains valid for the lifetime of this `ResourceHandle`
    /// - the handle is not freed externally while this wrapper exists
    pub unsafe fn from_raft_handle(raw: *mut cugraph_resource_handle_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(ErrorKind::InvalidInput, "ResourceHandle::from_raft_handle")
                .with_message("RAFT handle pointer must not be null"));
        }
        let device = DeviceId::current().map_err(|error| {
            Error::new(ErrorKind::Unexpected, "ResourceHandle::from_raft_handle")
                .with_message(format!("failed to query current CUDA device: {}", error))
                .set_source(error)
        })?;
        Ok(Self { raw, device: device.ordinal() as i32, owned: false, _nosend_nosync: PhantomData })
    }

    // NOTE: `rank()` and `comm_size()` accessors are not yet available because
    // the cuGraph C API does not expose `cugraph_resource_handle_get_rank` or
    // `cugraph_resource_handle_get_comm_size`. When these functions are added
    // upstream, safe wrappers should be added here.

    /// Get the raw C pointer. For internal use by other modules.
    pub(crate) fn as_ptr(&self) -> *mut cugraph_resource_handle_t {
        self.raw
    }

    /// The CUDA device this handle was created on.
    pub fn device(&self) -> i32 {
        self.device
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
    fn test_from_raft_handle_rejects_null() {
        let err = unsafe { ResourceHandle::from_raft_handle(ptr::null_mut()) }.expect_err("should reject null");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "ResourceHandle::from_raft_handle");
    }

    #[test]
    fn test_from_raft_handle_does_not_free_on_drop() {
        // Create an owned handle, take its raw pointer, wrap it in a
        // non-owning handle, and drop the non-owning one first. The owned
        // handle should still be valid afterwards (no double-free).
        let owned = ResourceHandle::new().expect("owned handle");
        let raw = owned.as_ptr();

        {
            // SAFETY: `raw` is valid for the lifetime of `owned`.
            let borrowed = unsafe { ResourceHandle::from_raft_handle(raw) }.expect("from_raft_handle");
            assert!(!borrowed.as_ptr().is_null());
            // `borrowed` drops here without freeing `raw`.
        }

        // `owned` is still valid — dropping it cleans up the actual handle.
        assert!(!owned.as_ptr().is_null());
    }
}
