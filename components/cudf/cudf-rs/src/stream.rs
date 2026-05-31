//! CUDA stream wrapper.
//!
//! This is currently a standalone wrapper around an owned `rmm::cuda_stream`.
//! The rest of the `cudf` API does not yet accept custom streams, so
//! high-level dataframe and column operations still run on cuDF's default
//! stream.

use std::ptr::{self, NonNull};

use crate::error::{self, Result};

/// An owned CUDA stream.
///
/// Wraps an `rmm::cuda_stream` via the C shim. The stream is destroyed when
/// this value is dropped.
///
/// Today this type lets callers manage or inspect standalone CUDA stream
/// state, but it does not steer the stream used by other `cudf` operations.
pub struct CudaStream {
    handle: NonNull<std::ffi::c_void>,
}

// CUDA streams are safe to send between threads.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(unsafe { cudf_sys::cudf_stream_create(&mut handle) }, "stream::new")?;
        Ok(Self { handle: NonNull::new(handle).expect("cudf_stream_create returned null on success") })
    }

    /// Block the calling thread until all operations already enqueued on this
    /// stream complete.
    pub fn synchronize(&self) -> Result<()> {
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_synchronize(self.handle.as_ptr()) },
            "stream::synchronize",
        )
    }

    /// Check whether all operations already enqueued on this stream have
    /// completed (non-blocking).
    pub fn is_ready(&self) -> Result<bool> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_is_ready(self.handle.as_ptr(), &mut out) },
            "stream::is_ready",
        )?;
        Ok(out != 0)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_stream_destroy(self.handle.as_ptr());
        }
    }
}
