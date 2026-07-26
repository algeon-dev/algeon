//! CUDA stream wrapper.
//!
//! This is an owned `rmm::cuda_stream` wrapper used either as a
//! [`crate::Runtime`] default or as a borrowed
//! [`crate::ExecutionOverrides`] override for one operation.
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit stream handles,
//! and `cudf` routes supported operations through the resolved stream for
//! the active runtime/override context.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    device::DeviceId, device_guard::DeviceEntryGuard, device_health::check_device_status_in, event::CudaEvent,
};
use crate::core::error::{self, Error, ErrorCode, Result};

/// An owned CUDA stream.
///
/// Wraps an `rmm::cuda_stream` via the C shim. The stream is destroyed when
/// this value is dropped.
///
/// Each stream records the CUDA device it was created on. This provenance is
/// used by [`crate::Runtime`] builder validation to reject cross-device
/// handle combinations.
///
/// Use this with [`crate::Runtime::builder`] to supply an explicit default
/// stream for retained provenance, or with [`crate::ExecutionOverrides::with_stream`] to
/// override one operation.
#[derive(Clone)]
pub struct CudaStream {
    handle: Arc<CudaStreamHandle>,
}

struct CudaStreamHandle {
    raw: Option<NonNull<std::ffi::c_void>>,
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
        let _entry = DeviceEntryGuard::enter(device, "stream::new")?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        check_device_status_in(unsafe { cudf_sys::cudf_stream_create(&mut handle) }, device, "stream::new")?;
        Ok(Self {
            handle: Arc::new(CudaStreamHandle {
                raw: Some(error::require_non_null(handle, "stream::new", "CUDA stream")?),
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
        let _entry = DeviceEntryGuard::enter(self.device(), "stream::synchronize")?;
        // SAFETY: self.handle is valid (NonNull from construction).
        check_device_status_in(
            unsafe { cudf_sys::cudf_stream_synchronize(self.handle.raw()) },
            self.device(),
            "stream::synchronize",
        )
    }

    /// Check whether all operations already enqueued on this stream have
    /// completed (non-blocking).
    pub fn is_ready(&self) -> Result<bool> {
        let _entry = DeviceEntryGuard::enter(self.device(), "stream::is_ready")?;
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
        check_device_status_in(
            unsafe { cudf_sys::cudf_stream_is_ready(self.handle.raw(), &mut out) },
            self.device(),
            "stream::is_ready",
        )?;
        Ok(out != 0)
    }

    /// Make this stream wait for `event` before subsequently enqueued work
    /// proceeds.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        if self.device() != event.device() {
            return Err(Error::new_with(
                ErrorCode::RuntimeBindingMismatch,
                "stream::wait_event",
                "CUDA stream and event must have the same immutable owner device",
            )
            .with_detail("stream_device", self.device().ordinal().to_string())
            .with_detail("event_device", event.device().ordinal().to_string())
            .with_detail("reason", "cross_device_stream_event"));
        }
        let _entry = DeviceEntryGuard::enter(self.device(), "stream::wait_event")?;
        check_device_status_in(
            unsafe { cudf_sys::cudf_stream_wait_event(self.handle.raw(), event.as_raw()) },
            self.device(),
            "stream::wait_event",
        )
    }

    /// Borrow this owner as a low-level `cudf-sys` stream handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `CudaStream` remains responsible for destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedStream<'_> {
        unsafe { cudf_sys::BorrowedStream::from_raw(self.handle.raw()) }
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let _entry = DeviceEntryGuard::enter(self.device(), "stream::interop_raw_handle")?;
        let mut out = std::ptr::null_mut();
        check_device_status_in(
            unsafe { cudf_sys::cudf_stream_raw(self.handle.raw(), &mut out) },
            self.device(),
            "stream::interop_raw_handle",
        )?;
        Ok(out)
    }

    pub(crate) fn debug_handle_value(&self) -> Result<usize> {
        let _entry = DeviceEntryGuard::enter(self.device(), "stream::debug_handle_value")?;
        let mut out: usize = 0;
        check_device_status_in(
            unsafe { cudf_sys::cudf_test_stream_handle_value(self.handle.raw(), &mut out) },
            self.device(),
            "stream::debug_handle_value",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        self.debug_handle_value().unwrap_or(0)
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        let raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        crate::core::native_teardown::drop_device_owned_handle(
            raw,
            self.device,
            "CudaStreamHandle::drop",
            |owned| unsafe { cudf_sys::cudf_stream_destroy(owned) },
        );
    }
}

impl CudaStreamHandle {
    fn raw(&self) -> *mut std::ffi::c_void {
        self.raw.expect("live CudaStream always retains its native stream").as_ptr()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope};

    #[test]
    fn drop_get_device_failure_quarantines_without_destroying_on_an_unproven_device() {
        let stream = CudaStream::new().expect("stream");
        let device = stream.device();
        let scope = DeviceSafetyTestScope::install([DeviceEntryTestCall::GetDevice {
            status: cudf_sys::CUDF_ERROR_FATAL_CUDA,
            device: device.ordinal() as i32,
        }]);

        drop(stream);

        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), crate::runtime::DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.quarantined_teardown_count(), 1);
        assert_eq!(scope.remaining_call_count(), 0);
    }

    #[test]
    fn drop_set_device_failure_quarantines_without_wrong_device_destruction() {
        let stream = CudaStream::new().expect("stream");
        let device = stream.device();
        let ambient = if device.ordinal() == 0 { 1 } else { 0 };
        let scope = DeviceSafetyTestScope::install([
            DeviceEntryTestCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: ambient },
            DeviceEntryTestCall::SetDevice {
                expected_target: device.ordinal() as i32,
                status: cudf_sys::CUDF_ERROR_FATAL_CUDA,
            },
        ]);

        drop(stream);

        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), crate::runtime::DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.quarantined_teardown_count(), 1);
        assert_eq!(scope.remaining_call_count(), 0);
    }
}
