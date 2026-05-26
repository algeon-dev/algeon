//! CUDA stream wrapper.
//!
//! Per CLAUDE.md: no async runtime in bindings. Users control their own
//! runtime and can call `stream.synchronize()` from within
//! `tokio::task::spawn_blocking` or similar if desired.

use crate::error::{self, Result};
use std::ptr::{self, NonNull};

/// An owned CUDA stream.
///
/// Wraps an `rmm::cuda_stream` via the C shim. The stream is destroyed
/// when this value is dropped.
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
        error::check_status(unsafe { cudf_sys::cudf_stream_create(&mut handle) })?;
        Ok(Self {
            handle: NonNull::new(handle).expect("cudf_stream_create returned null on success"),
        })
    }

    /// Block the calling thread until all operations on this stream complete.
    pub fn synchronize(&self) -> Result<()> {
        error::check_status(unsafe { cudf_sys::cudf_stream_synchronize(self.handle.as_ptr()) })
    }

    /// Check whether all operations on this stream have completed (non-blocking).
    pub fn is_ready(&self) -> Result<bool> {
        let mut out: i32 = 0;
        error::check_status(unsafe {
            cudf_sys::cudf_stream_is_ready(self.handle.as_ptr(), &mut out)
        })?;
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
