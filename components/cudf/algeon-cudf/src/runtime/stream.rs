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

/// Keeps one attempt's fork-stream route registered while its native work
/// drains. The route owns clones of every stream handle so the native provider
/// cannot return a raw stream after its Rust owner has been released.
///
/// The last stream of the registered set is the parent's reserved fork: the
/// parent is served only from it and no alias can ever claim it. The
/// preceding streams are the auxiliary forks, one per live alias.
#[must_use = "dropping the registration removes the attempt stream route"]
pub struct AttemptStreamProviderRegistration {
    parent: CudaStream,
    fork_streams: Vec<CudaStream>,
    parent_raw: usize,
}

/// Keeps one additional attempt-member stream mapped to the provider until
/// that stream lease has completed and can safely return to its pool.
///
/// The alias holds one auxiliary fork of its route alone for its whole
/// lifetime, so work it launches can never share a stream with another live
/// alias or with the route's parent.
#[must_use = "dropping the alias removes the attempt stream route"]
pub struct AttemptStreamProviderAlias {
    stream: CudaStream,
    raw: usize,
    claimed_fork_index: usize,
}

impl AttemptStreamProviderAlias {
    /// The auxiliary fork index this alias holds alone.
    pub fn claimed_fork_index(&self) -> usize {
        self.claimed_fork_index
    }
}

impl std::fmt::Debug for AttemptStreamProviderAlias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptStreamProviderAlias")
            .field("stream", &self.stream)
            .field("claimed_fork_index", &self.claimed_fork_index)
            .finish()
    }
}

impl std::fmt::Debug for AttemptStreamProviderRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttemptStreamProviderRegistration")
            .field("parent", &self.parent)
            .field("auxiliary_count", &self.auxiliary_count())
            .finish()
    }
}

impl AttemptStreamProviderRegistration {
    /// Register a bounded, attempt-owned fork stream set for `parent`.
    ///
    /// `fork_streams` holds the auxiliary forks followed by the parent's
    /// reserved fork, so it needs at least two entries. Every stream must be a
    /// live stream on the same CUDA device as the parent. The caller must also
    /// attach each stream to the attempt's memory-domain member before native
    /// work can allocate through it.
    pub fn register(parent: &CudaStream, fork_streams: &[CudaStream]) -> Result<Self> {
        let operation = "runtime::AttemptStreamProviderRegistration::register";
        if fork_streams.len() < 2 {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                operation,
                "attempt fork-stream provider requires one reserved parent fork and at least one auxiliary fork",
            ));
        }
        if let Some((index, stream)) =
            fork_streams.iter().enumerate().find(|(_, stream)| stream.device() != parent.device())
        {
            return Err(Error::new_with(
                ErrorCode::RuntimeBindingMismatch,
                operation,
                "attempt fork-stream provider streams must share the parent device",
            )
            .with_detail("parent_device", parent.device().ordinal().to_string())
            .with_detail("auxiliary_index", index.to_string())
            .with_detail("auxiliary_device", stream.device().ordinal().to_string()));
        }

        let retained_parent = parent.clone();
        let retained_fork_streams = fork_streams.to_vec();
        let _entry = DeviceEntryGuard::enter(parent.device(), operation)?;
        let parent_raw = parent.interop_raw_handle()?;
        let fork_streams_raw = fork_streams.iter().map(CudaStream::interop_raw_handle).collect::<Result<Vec<_>>>()?;
        error::check_status_in(
            // SAFETY: all pointers refer to live CUDA streams retained by the
            // returned registration and by the caller's attempt owners.
            unsafe {
                cudf_sys::cudf_stream_provider_register_attempt(
                    parent_raw,
                    fork_streams_raw.as_ptr(),
                    fork_streams_raw.len(),
                )
            },
            operation,
        )?;
        Ok(Self { parent: retained_parent, fork_streams: retained_fork_streams, parent_raw: parent_raw as usize })
    }

    /// Return the parent stream retained by this route.
    pub fn parent(&self) -> &CudaStream {
        &self.parent
    }

    /// Return the number of attempt-owned auxiliary forks in the route: every
    /// registered fork but the parent's reserved one.
    pub fn auxiliary_count(&self) -> usize {
        self.fork_streams.len() - 1
    }

    /// Drain every provider-owned auxiliary stream before the route and its
    /// query-domain members are detached during attempt teardown.
    pub fn synchronize_auxiliary_for_teardown(&self) -> Result<()> {
        for stream in &self.fork_streams {
            stream.synchronize_for_teardown("runtime::AttemptStreamProviderRegistration::teardown")?;
        }
        Ok(())
    }

    /// Attach one additional live attempt-member stream as a valid fork parent
    /// holding one auxiliary fork of this route alone.
    ///
    /// The route claims an auxiliary fork no other live alias holds and
    /// confines every fork issued from `stream`, and from the fork it is
    /// served, to it; the route picks the index, since its claim table is the
    /// only record of which forks are taken. The parent's reserved fork is
    /// never claimable. It fails with a typed invalid-input error when every
    /// auxiliary fork is already claimed. The claim is a property of the
    /// alias, so nested forks issued from it on any thread stay inside it, and
    /// detaching the returned alias releases it.
    pub fn attach(&self, stream: &CudaStream) -> Result<AttemptStreamProviderAlias> {
        let operation = "runtime::AttemptStreamProviderRegistration::attach";
        if stream.device() != self.parent.device() {
            return Err(Error::new_with(
                ErrorCode::RuntimeBindingMismatch,
                operation,
                "attempt fork-stream aliases must share the parent device",
            )
            .with_detail("parent_device", self.parent.device().ordinal().to_string())
            .with_detail("stream_device", stream.device().ordinal().to_string()));
        }
        let _entry = DeviceEntryGuard::enter(stream.device(), operation)?;
        let raw = stream.interop_raw_handle()?;
        let mut claimed_fork_index: usize = 0;
        error::check_status_in(
            // SAFETY: `self.parent_raw` and `raw` refer to live CUDA streams
            // retained by the registration and returned alias respectively,
            // and the out pointer refers to a live local.
            unsafe {
                cudf_sys::cudf_stream_provider_attach_attempt_stream(
                    self.parent_raw as *mut std::ffi::c_void,
                    raw,
                    &raw mut claimed_fork_index,
                )
            },
            operation,
        )
        .map_err(|error| error.with_detail("auxiliary_fork_count", self.auxiliary_count().to_string()))?;
        Ok(AttemptStreamProviderAlias { stream: stream.clone(), raw: raw as usize, claimed_fork_index })
    }
}

impl Drop for AttemptStreamProviderRegistration {
    fn drop(&mut self) {
        let operation = "runtime::AttemptStreamProviderRegistration::drop";
        // This native operation only removes a process-local stream map entry;
        // it must still run when the owner device is unhealthy or unavailable.
        // The registration owns the parent and fork stream clones, so the
        // map is removed before any raw stream can die.
        let status =
            unsafe { cudf_sys::cudf_stream_provider_unregister_attempt(self.parent_raw as *mut std::ffi::c_void) };
        if let Err(error) = error::check_status_in_without_device_report(status, operation) {
            eprintln!("[cudf] failed to remove attempt stream provider route: {error}");
        }
    }
}

impl Drop for AttemptStreamProviderAlias {
    fn drop(&mut self) {
        let operation = "runtime::AttemptStreamProviderAlias::drop";
        // Alias removal only updates the process-local route map, so it must
        // run even if the owner device is unhealthy or unavailable.
        let status = unsafe { cudf_sys::cudf_stream_provider_detach_attempt_stream(self.raw as *mut _) };
        if let Err(error) = error::check_status_in_without_device_report(status, operation) {
            eprintln!("[cudf] failed to remove attempt stream provider alias: {error}");
        }
    }
}

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
    raw: Option<NonNull<cudf_sys::cudf_stream_handle_state>>,
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

    /// Process-unique identity of this stream's native handle.
    ///
    /// This is an opaque key for maps keyed by stream identity (for example,
    /// one query-domain member per stream); it is not stable across process
    /// runs and must not be persisted.
    pub fn identity(&self) -> usize {
        self.handle.raw() as usize
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

    /// Synchronize before destroying a device-owned resource that retains this
    /// stream. Unlike normal execution, teardown must still attempt owner-device
    /// entry after the device has been marked unhealthy so it can either retire
    /// the resource safely or quarantine it.
    #[doc(hidden)]
    pub fn synchronize_for_teardown(&self, operation: &'static str) -> Result<()> {
        let _entry = DeviceEntryGuard::enter_for_teardown(self.device(), operation)?;
        // SAFETY: the resource-retirement registry retains a clone of this
        // stream until after the fence has completed.
        check_device_status_in(
            unsafe { cudf_sys::cudf_stream_synchronize(self.handle.raw()) },
            self.device(),
            operation,
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

    pub(crate) fn as_raw(&self) -> cudf_sys::cudf_stream_handle {
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
        self.interop_raw_handle().map(|raw| raw as usize)
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
    fn raw(&self) -> cudf_sys::cudf_stream_handle {
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
