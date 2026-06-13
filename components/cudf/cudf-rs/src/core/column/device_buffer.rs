//! Owned GPU device buffer backed by `rmm::device_buffer`.
//!
//! **Advanced/Core.** This is a low-level building block for zero-copy
//! ownership transfer into [`Column`](super::Column). Most users should
//! construct columns via [`Runtime::column_from_slice`](crate::Runtime::column_from_slice)
//! or [`Runtime::series`](crate::Runtime::series) instead.

use std::{
    mem,
    os::raw::c_void,
    ptr::{self, NonNull},
};

use crate::{
    core::{
        error::{self, Result},
        types::CudfType,
    },
    runtime::{Runtime, device_guard::DeviceGuard},
};

/// An owned block of GPU device memory.
///
/// `DeviceBuffer` wraps an `rmm::device_buffer` via the cudf-sys C shim.
/// The underlying GPU memory is freed when this value is dropped.
/// It snapshots the explicit runtime device that created it so later zero-copy
/// adoption into a [`Column`](super::Column) can reject cross-device misuse
/// before handing ownership to libcudf.
///
/// The primary use case is to prepare device memory externally (e.g., from
/// another CUDA library) and then transfer ownership into a
/// [`Column`](super::Column) via
/// [`Runtime::column_from_device_buffer`](crate::Runtime::column_from_device_buffer)
/// **without** any device-to-device copy.
///
/// # Example
///
/// ```no_run
/// use cudf::{Runtime, TypeId, core::DeviceBuffer};
///
/// // Upload host data to a device buffer (this copies from host to GPU).
/// let rt = Runtime::new().unwrap();
/// let buf = rt.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();
///
/// // Adopt the existing device allocation into a Column without another copy.
/// let col = rt.column_from_device_buffer(buf, 3, TypeId::Int32).unwrap();
/// ```
pub struct DeviceBuffer {
    handle: NonNull<c_void>,
    device_snapshot: i32,
}

// SAFETY: DeviceBuffer owns a GPU device allocation via an opaque C++ handle.
// The handle is not thread-local. Send is safe because ownership transfer
// between threads is supported by the CUDA runtime. Sync is safe because all
// methods that observe state create independent FFI queries.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    fn create_on_current_device(size_bytes: usize, device_snapshot: i32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_create(size_bytes, &mut handle) },
            "device_buffer::create_on_current_device",
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, "device_buffer::create_on_current_device", "device buffer")?,
            device_snapshot,
        })
    }

    pub(crate) fn new_in_runtime(runtime: &Runtime, size_bytes: usize) -> Result<Self> {
        let _guard = DeviceGuard::new(runtime.device())?;
        Self::create_on_current_device(size_bytes, runtime.device().ordinal_raw())
    }

    pub(crate) fn from_host_data_in_runtime<T: CudfType>(runtime: &Runtime, data: &[T]) -> Result<Self> {
        let _guard = DeviceGuard::new(runtime.device())?;
        let byte_len = mem::size_of_val(data);
        let buf = Self::create_on_current_device(byte_len, runtime.device().ordinal_raw())?;
        if byte_len > 0 {
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_device_buffer_copy_from_host(
                        buf.handle.as_ptr(),
                        data.as_ptr() as *const c_void,
                        byte_len,
                    )
                },
                "device_buffer::from_host_data_in_runtime",
            )?;
        }
        Ok(buf)
    }

    /// Returns the size of this buffer in bytes.
    pub fn size_bytes(&self) -> Result<usize> {
        let mut out: usize = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_size(self.handle.as_ptr(), &mut out) },
            "device_buffer::size_bytes",
        )?;
        Ok(out)
    }

    /// Returns `true` if the buffer has zero bytes.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.size_bytes()? == 0)
    }

    /// Returns the raw device pointer.
    ///
    /// The pointer refers to GPU memory and cannot be dereferenced from host
    /// code. It is intended for diagnostics and for passing to GPU libraries.
    ///
    /// For zero-size buffers this call still succeeds, but the returned
    /// pointer may be null because there is no addressable device range.
    pub fn data_ptr(&self) -> Result<*mut c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_data(self.handle.as_ptr(), &mut out) },
            "device_buffer::data_ptr",
        )?;
        Ok(out)
    }

    /// Consume this buffer and return the raw FFI handle **without** running
    /// `Drop`.
    ///
    /// The caller is responsible for either passing the handle to an FFI
    /// function that consumes it (e.g., `cudf_column_adopt_device_buffer`) or
    /// reconstructing a `DeviceBuffer` via [`from_raw_parts`](Self::from_raw_parts).
    pub(crate) fn into_raw_parts(self) -> (*mut c_void, i32) {
        let this = mem::ManuallyDrop::new(self);
        (this.handle.as_ptr(), this.device_snapshot)
    }

    pub(crate) fn device_snapshot(&self) -> i32 {
        self.device_snapshot
    }

    /// Reconstruct a `DeviceBuffer` from raw FFI parts, taking ownership.
    ///
    /// # Safety
    ///
    /// The handle must be a valid `cudf_device_buffer_handle` that has not
    /// been consumed by another FFI call.
    pub(crate) unsafe fn from_raw_parts(handle: *mut c_void, device_snapshot: i32) -> Result<Self> {
        Ok(Self {
            handle: error::require_non_null(handle, "device_buffer::from_raw_parts", "device buffer")?,
            device_snapshot,
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once, so the handle is destroyed exactly once.
        unsafe {
            cudf_sys::cudf_device_buffer_destroy(self.handle.as_ptr());
        }
    }
}
