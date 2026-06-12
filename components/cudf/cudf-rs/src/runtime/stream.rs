//! CUDA stream wrapper.
//!
//! This is an owned `rmm::cuda_stream` wrapper used by [`crate::Runtime`] and
//! borrowed [`crate::ExecutionContext`] overrides.
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit stream handles,
//! and the high-level API can now route selected operations through retained
//! provenance defaults or per-operation borrowed overrides.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{device::DeviceId, event::CudaEvent};
use crate::core::error::{self, Result};

/// An owned CUDA stream.
///
/// Wraps an `rmm::cuda_stream` via the C shim. The stream is destroyed when
/// this value is dropped.
///
/// Each stream records the CUDA device it was created on. This provenance is
/// used by [`crate::Runtime`] builder validation to reject cross-device
/// handle combinations.
///
/// Use this with [`crate::Runtime::builder`] to seed explicit retained
/// provenance, or with [`crate::ExecutionContext::with_stream`] to
/// override one operation.
#[derive(Clone)]
pub struct CudaStream {
    handle: Arc<CudaStreamHandle>,
}

struct CudaStreamHandle {
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

// SAFETY: CUDA streams are thread-safe by design — operations can be enqueued
// from any thread. The handle is an opaque pointer to a CUDA runtime object
// that supports concurrent access. Sync is safe because synchronize() and
// is_ready() are read-only queries on stream state.
unsafe impl Send for CudaStreamHandle {}
unsafe impl Sync for CudaStreamHandle {}

impl CudaStream {
    /// Create a new CUDA stream on the calling thread's current device.
    pub fn new() -> Result<Self> {
        let device = DeviceId::current()?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(unsafe { cudf_sys::cudf_stream_create(&mut handle) }, "stream::new")?;
        Ok(Self {
            handle: Arc::new(CudaStreamHandle {
                raw: NonNull::new(handle).expect("cudf_stream_create returned null on success"),
                device,
            }),
        })
    }

    /// The CUDA device this stream was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
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

    /// Make this stream wait for `event` before subsequently enqueued work
    /// proceeds.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_wait_event(self.handle.raw.as_ptr(), event.as_raw()) },
            "stream::wait_event",
        )
    }

    /// Borrow this owner as a low-level `cudf-sys` stream handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `CudaStream` remains responsible for destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedStream<'_> {
        unsafe { cudf_sys::BorrowedStream::from_raw(self.handle.raw.as_ptr()) }
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_stream_raw(self.handle.raw.as_ptr(), &mut out) },
            "stream::interop_raw_handle",
        )?;
        Ok(out)
    }

    pub(crate) fn debug_handle_value(&self) -> Result<usize> {
        let mut out: usize = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_stream_handle_value(self.handle.raw.as_ptr(), &mut out) },
            "stream::debug_handle_value",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        self.debug_handle_value().expect("failed to read CUDA stream debug handle value")
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        // Switch to the device this stream was created on before destroying it.
        // In multi-GPU scenarios, destroying a CUDA stream while on the wrong
        // device is undefined behavior.
        let mut current: i32 = 0;
        let get_status = unsafe { cudf_sys::cudf_cuda_get_device(&mut current) };
        if get_status == cudf_sys::CUDF_SUCCESS && current != self.device.ordinal_raw() {
            let set_status = unsafe { cudf_sys::cudf_cuda_set_device(self.device.ordinal_raw()) };
            if set_status != cudf_sys::CUDF_SUCCESS {
                eprintln!(
                    "[cudf::CudaStreamHandle] warning: failed to switch to device {} for stream destruction",
                    self.device.ordinal_raw()
                );
            }
        }
        // SAFETY: self.raw is valid (NonNull from construction). The shared
        // handle is destroyed exactly once when the last clone drops.
        unsafe {
            cudf_sys::cudf_stream_destroy(self.raw.as_ptr());
        }
        // Restore the previous device (best-effort).
        if get_status == cudf_sys::CUDF_SUCCESS && current != self.device.ordinal_raw() {
            unsafe { cudf_sys::cudf_cuda_set_device(current) };
        }
    }
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
