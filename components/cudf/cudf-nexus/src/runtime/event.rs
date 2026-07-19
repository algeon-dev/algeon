//! CUDA event wrapper.
//!
//! This is an owned CUDA event used for explicit device-side ordering,
//! readiness queries, and interop completion handoff.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    device::DeviceId, device_guard::DeviceEntryGuard, device_health::check_device_status_in, stream::CudaStream,
};
use crate::core::error::{self, Error, ErrorCode, Result};

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
    raw: Option<NonNull<cudf_sys::cudf_event_handle_state>>,
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
        let _entry = DeviceEntryGuard::enter(device, "event::new")?;
        let mut handle = ptr::null_mut();
        check_device_status_in(unsafe { cudf_sys::cudf_event_create(&mut handle) }, device, "event::new")?;
        Ok(Self {
            handle: Arc::new(CudaEventHandle {
                raw: Some(error::require_non_null(handle, "event::new", "CUDA event")?),
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
        if self.device() != stream.device() {
            return Err(Error::new_with(
                ErrorCode::RuntimeBindingMismatch,
                "event::record",
                "CUDA event and stream must have the same immutable owner device",
            )
            .with_detail("event_device", self.device().ordinal().to_string())
            .with_detail("stream_device", stream.device().ordinal().to_string())
            .with_detail("reason", "cross_device_event_stream"));
        }
        let _entry = DeviceEntryGuard::enter(self.device(), "event::record")?;
        check_device_status_in(
            unsafe { cudf_sys::cudf_event_record(self.handle.raw(), stream.as_raw()) },
            self.device(),
            "event::record",
        )
    }

    /// Block the calling thread until this event completes.
    pub fn synchronize(&self) -> Result<()> {
        let _entry = DeviceEntryGuard::enter(self.device(), "event::synchronize")?;
        check_device_status_in(
            unsafe { cudf_sys::cudf_event_synchronize(self.handle.raw()) },
            self.device(),
            "event::synchronize",
        )
    }

    /// Check whether this event has completed.
    pub fn is_ready(&self) -> Result<bool> {
        let _entry = DeviceEntryGuard::enter(self.device(), "event::is_ready")?;
        let mut out = 0;
        check_device_status_in(
            unsafe { cudf_sys::cudf_event_is_ready(self.handle.raw(), &mut out) },
            self.device(),
            "event::is_ready",
        )?;
        Ok(out != 0)
    }

    /// Borrow this owner as a low-level `cudf-sys` event handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `CudaEvent` remains responsible for destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedEvent<'_> {
        unsafe { cudf_sys::BorrowedEvent::from_raw(self.handle.raw()) }
    }

    pub(crate) fn as_raw(&self) -> cudf_sys::cudf_event_handle {
        self.handle.raw()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let _entry = DeviceEntryGuard::enter(self.device(), "event::interop_raw_handle")?;
        let mut out = std::ptr::null_mut();
        check_device_status_in(
            unsafe { cudf_sys::cudf_event_raw(self.handle.raw(), &mut out) },
            self.device(),
            "event::interop_raw_handle",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        self.handle.raw() as usize
    }
    pub(crate) fn __debug_interop_raw_handle_value(&self) -> usize {
        self.interop_raw_handle().map_or(0, |raw| raw as usize)
    }
}

impl Drop for CudaEventHandle {
    fn drop(&mut self) {
        let raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        crate::core::native_teardown::drop_device_owned_handle(
            raw,
            self.device,
            "CudaEventHandle::drop",
            |owned| unsafe { cudf_sys::cudf_event_destroy(owned) },
        );
    }
}

impl CudaEventHandle {
    fn raw(&self) -> cudf_sys::cudf_event_handle {
        self.raw.expect("live CudaEvent always retains its native event").as_ptr()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope};

    #[test]
    fn drop_get_device_failure_quarantines_event_without_destructor_ffi() {
        let event = CudaEvent::new().expect("event");
        let device = event.device();
        let scope = DeviceSafetyTestScope::install([DeviceEntryTestCall::GetDevice {
            status: cudf_sys::CUDF_ERROR_FATAL_CUDA,
            device: device.ordinal() as i32,
        }]);

        drop(event);

        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), crate::runtime::DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.quarantined_teardown_count(), 1);
        assert_eq!(scope.remaining_call_count(), 0);
    }
}
