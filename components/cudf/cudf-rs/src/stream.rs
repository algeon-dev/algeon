//! CUDA stream wrapper.
//!
//! This is currently a standalone wrapper around an owned `rmm::cuda_stream`.
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit stream handles,
//! but the high-level API (`DataFrame`, `Series`) does not yet expose
//! stream control — operations still run on cuDF's default stream.

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

// SAFETY: CUDA streams are thread-safe by design — operations can be enqueued
// from any thread. The handle is an opaque pointer to a CUDA runtime object
// that supports concurrent access. Sync is safe because synchronize() and
// is_ready() are read-only queries on stream state.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(unsafe { cudf_sys::cudf_stream_create(&mut handle) }, "stream::new")?;
        Ok(Self { handle: NonNull::new(handle).expect("cudf_stream_create returned null on success") })
    }

    /// Block the calling thread until all operations already enqueued on this
    /// stream complete.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_synchronize(self.handle.as_ptr()) },
            "stream::synchronize",
        )
    }

    /// Check whether all operations already enqueued on this stream have
    /// completed (non-blocking).
    pub fn is_ready(&self) -> Result<bool> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_is_ready(self.handle.as_ptr(), &mut out) },
            "stream::is_ready",
        )?;
        Ok(out != 0)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once.
        unsafe {
            cudf_sys::cudf_stream_destroy(self.handle.as_ptr());
        }
    }
}
