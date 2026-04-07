//! Owned GPU device buffer backed by `rmm::device_buffer`.
//!
//! **Advanced/Core.** This is a low-level building block for zero-copy
//! ownership transfer into [`Column`](super::Column). Most users should
//! construct columns via [`Column::from_slice`](super::Column::from_slice)
//! or [`Series`](crate::Series) instead.

use std::{
    mem,
    os::raw::c_void,
    ptr::{self, NonNull},
};

use crate::core::{
    error::{self, Result},
    types::CudfType,
};

fn current_device_snapshot() -> i32 {
    let mut device: i32 = 0;
    let _ = unsafe { cudf_sys::cudf_cuda_get_device(&mut device) };
    device
}

/// An owned block of GPU device memory.
///
/// `DeviceBuffer` wraps an `rmm::device_buffer` via the cudf-sys C shim.
/// The underlying GPU memory is freed when this value is dropped.
/// It also snapshots the ambient CUDA device at creation time so later
/// zero-copy adoption into a [`Column`](super::Column) preserves the buffer's
/// original device provenance even if the calling thread switches devices.
///
/// The primary use case is to prepare device memory externally (e.g., from
/// another CUDA library) and then transfer ownership into a
/// [`Column`](super::Column) via
/// [`Column::from_device_buffer`](super::Column::from_device_buffer)
/// **without** any device-to-device copy.
///
/// # Example
///
/// ```no_run
/// use cudf::{
///     TypeId,
///     core::{Column, DeviceBuffer},
/// };
///
/// // Upload host data to a device buffer (this copies from host to GPU).
/// let buf = DeviceBuffer::from_host_data(&[1i32, 2, 3]).unwrap();
///
/// // Adopt the existing device allocation into a Column without another copy.
/// let col = Column::from_device_buffer(buf, 3, TypeId::Int32).unwrap();
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
    /// Allocate a new device buffer of `size_bytes` bytes.
    ///
    /// The contents are uninitialized. For `size_bytes == 0`, the buffer is
    /// valid but empty.
    pub fn new(size_bytes: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_create(size_bytes, &mut handle) },
            "device_buffer::new",
        )?;
        Ok(Self {
            handle: NonNull::new(handle).expect("null device_buffer handle on success"),
            device_snapshot: current_device_snapshot(),
        })
    }

    /// Create a device buffer by copying a host slice to GPU memory.
    ///
    /// The buffer size is `data.len() * size_of::<T>()` bytes. For an empty
    /// slice the buffer is valid but empty.
    pub fn from_host_data<T: CudfType>(data: &[T]) -> Result<Self> {
        let byte_len = mem::size_of_val(data);
        let buf = Self::new(byte_len)?;
        if byte_len > 0 {
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_device_buffer_copy_from_host(
                        buf.handle.as_ptr(),
                        data.as_ptr() as *const c_void,
                        byte_len,
                    )
                },
                "device_buffer::from_host_data",
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

    /// Reconstruct a `DeviceBuffer` from raw FFI parts, taking ownership.
    ///
    /// # Safety
    ///
    /// The handle must be a valid `cudf_device_buffer_handle` that has not
    /// been consumed by another FFI call.
    pub(crate) unsafe fn from_raw_parts(handle: *mut c_void, device_snapshot: i32) -> Self {
        Self { handle: NonNull::new(handle).expect("null device_buffer handle in from_raw_parts"), device_snapshot }
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
