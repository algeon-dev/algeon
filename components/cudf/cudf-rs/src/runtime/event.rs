//! CUDA event wrapper.
//!
//! This is an owned CUDA event used for explicit device-side ordering and
//! future producer-owned completion witnesses.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{device::DeviceId, stream::CudaStream};
use crate::core::error::{self, Result};

/// An owned CUDA event.
///
/// The event is created on the calling thread's current device and destroyed
/// when the last clone drops. Use events to express completion, readiness,
/// and cross-stream ordering without forcing immediate host synchronization.
#[derive(Clone)]
pub struct CudaEvent {
    handle: Arc<CudaEventHandle>,
}

struct CudaEventHandle {
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

// SAFETY: CUDA events are thread-safe synchronization objects. The handle is
// an opaque pointer to a CUDA runtime event that may be recorded, queried, and
// synchronized from any thread. Shared access does not permit host-side
// mutation of Rust memory, and the handle is destroyed exactly once when the
// last Arc clone drops.
unsafe impl Send for CudaEventHandle {}
unsafe impl Sync for CudaEventHandle {}

impl CudaEvent {
    /// Create a new CUDA event on the calling thread's current device.
    pub fn new() -> Result<Self> {
        let device = DeviceId::current()?;
        let mut handle = ptr::null_mut();
        error::check_status_in(unsafe { cudf_sys::cudf_event_create(&mut handle) }, "event::new")?;
        Ok(Self {
            handle: Arc::new(CudaEventHandle {
                raw: NonNull::new(handle).expect("cudf_event_create returned null on success"),
                device,
            }),
        })
    }

    /// The CUDA device this event was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Record this event on `stream`.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        error::check_status_in(
            unsafe { cudf_sys::cudf_event_record(self.handle.raw.as_ptr(), stream.as_raw()) },
            "event::record",
        )
    }

    /// Block the calling thread until this event completes.
    pub fn synchronize(&self) -> Result<()> {
        error::check_status_in(
            unsafe { cudf_sys::cudf_event_synchronize(self.handle.raw.as_ptr()) },
            "event::synchronize",
        )
    }

    /// Check whether this event has completed.
    pub fn is_ready(&self) -> Result<bool> {
        let mut out = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_event_is_ready(self.handle.raw.as_ptr(), &mut out) },
            "event::is_ready",
        )?;
        Ok(out != 0)
    }

    /// Borrow this owner as a low-level `cudf-sys` event handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `CudaEvent` remains responsible for destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedEvent<'_> {
        unsafe { cudf_sys::BorrowedEvent::from_raw(self.handle.raw.as_ptr()) }
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_event_raw(self.handle.raw.as_ptr(), &mut out) },
            "event::interop_raw_handle",
        )?;
        Ok(out)
    }

    #[doc(hidden)]
    pub fn __debug_raw_handle_value(&self) -> usize {
        self.handle.raw.as_ptr() as usize
    }

    #[doc(hidden)]
    pub fn __debug_interop_raw_handle_value(&self) -> usize {
        self.interop_raw_handle()
            .expect("event::__debug_interop_raw_handle_value should resolve the borrowed CUDA event pointer")
            as usize
    }
}

impl Drop for CudaEventHandle {
    fn drop(&mut self) {
        // Switch to the device this event was created on before destroying it.
        // In multi-GPU scenarios, destroying a CUDA event while on the wrong
        // device is undefined behavior.
        let mut current: i32 = 0;
        let get_status = unsafe { cudf_sys::cudf_cuda_get_device(&mut current) };
        if get_status == cudf_sys::CUDF_SUCCESS && current != self.device.ordinal_raw() {
            let set_status = unsafe { cudf_sys::cudf_cuda_set_device(self.device.ordinal_raw()) };
            if set_status != cudf_sys::CUDF_SUCCESS {
                eprintln!(
                    "[cudf::CudaEventHandle] warning: failed to switch to device {} for event destruction",
                    self.device.ordinal_raw()
                );
            }
        }
        unsafe {
            cudf_sys::cudf_event_destroy(self.raw.as_ptr());
        }
        // Restore the previous device (best-effort).
        if get_status == cudf_sys::CUDF_SUCCESS && current != self.device.ordinal_raw() {
            unsafe { cudf_sys::cudf_cuda_set_device(current) };
        }
    }
}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEvent")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
