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
/// when the last clone drops.
#[derive(Clone)]
pub struct CudaEvent {
    handle: Arc<CudaEventHandle>,
}

struct CudaEventHandle {
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

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
        unsafe {
            cudf_sys::cudf_event_destroy(self.raw.as_ptr());
        }
    }
}
