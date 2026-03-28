//! GPU resource handle for cuGraph operations.
//!
//! Every cuGraph operation requires a [`ResourceHandle`] which manages the
//! underlying CUDA stream and RAFT handle. The handle is **not** `Send` or
//! `Sync` because GPU state is thread-local.

use std::{fmt, marker::PhantomData, ptr};

use cugraph_sys::{cugraph_create_resource_handle, cugraph_free_resource_handle, cugraph_resource_handle_t};

use crate::error::{Error, ErrorKind, Result};

/// RAII wrapper around `cugraph_resource_handle_t`.
///
/// Manages GPU resources (CUDA stream, RAFT handle). Automatically freed on
/// drop. Not thread-safe — must be used from the thread that created it.
pub struct ResourceHandle {
    raw: *mut cugraph_resource_handle_t,
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
    pub fn new() -> Result<Self> {
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        if raw.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "ResourceHandle::new")
                .with_message("cugraph_create_resource_handle returned NULL"));
        }
        Ok(Self { raw, _nosend_nosync: PhantomData })
    }

    /// Get the raw C pointer. For internal use by other modules.
    pub(crate) fn as_ptr(&self) -> *mut cugraph_resource_handle_t {
        self.raw
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
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
}
