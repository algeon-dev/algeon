//! CUDA stream wrapper.
//!
//! This is an owned `rmm::cuda_stream` wrapper used by [`crate::Runtime`] and
//! borrowed [`crate::ExecutionContext`] overrides.
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit stream handles,
//! and the high-level API can now route selected operations through retained
//! explicit domains or per-operation borrowed overrides.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::error::{self, Result};

/// An owned CUDA stream.
///
/// Wraps an `rmm::cuda_stream` via the C shim. The stream is destroyed when
/// this value is dropped.
///
/// Use this with [`crate::Runtime::builder`] to seed an explicit retained
/// execution domain, or with [`crate::ExecutionContext::with_stream`] to
/// override one operation.
#[derive(Clone)]
pub struct CudaStream {
    handle: Arc<CudaStreamHandle>,
}

struct CudaStreamHandle {
    raw: NonNull<std::ffi::c_void>,
}

// SAFETY: CUDA streams are thread-safe by design — operations can be enqueued
// from any thread. The handle is an opaque pointer to a CUDA runtime object
// that supports concurrent access. Sync is safe because synchronize() and
// is_ready() are read-only queries on stream state.
unsafe impl Send for CudaStreamHandle {}
unsafe impl Sync for CudaStreamHandle {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(unsafe { cudf_sys::cudf_stream_create(&mut handle) }, "stream::new")?;
        Ok(Self {
            handle: Arc::new(CudaStreamHandle {
                raw: NonNull::new(handle).expect("cudf_stream_create returned null on success"),
            }),
        })
    }

    /// Block the calling thread until all operations already enqueued on this
    /// stream complete.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_synchronize(self.handle.raw.as_ptr()) },
            "stream::synchronize",
        )
    }

    /// Check whether all operations already enqueued on this stream have
    /// completed (non-blocking).
    pub fn is_ready(&self) -> Result<bool> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_is_ready(self.handle.raw.as_ptr(), &mut out) },
            "stream::is_ready",
        )?;
        Ok(out != 0)
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    pub(crate) fn debug_handle_value(&self) -> Result<usize> {
        let mut out: usize = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_stream_handle_value(self.handle.raw.as_ptr(), &mut out) },
            "stream::debug_handle_value",
        )?;
        Ok(out)
    }

    #[doc(hidden)]
    pub fn __debug_raw_handle_value(&self) -> usize {
        self.debug_handle_value().expect("failed to read CUDA stream debug handle value")
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        // SAFETY: self.raw is valid (NonNull from construction). The shared
        // handle is destroyed exactly once when the last clone drops.
        unsafe {
            cudf_sys::cudf_stream_destroy(self.raw.as_ptr());
        }
    }
}
