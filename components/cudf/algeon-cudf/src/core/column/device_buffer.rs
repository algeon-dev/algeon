//! Owned GPU device buffer backed by `rmm::device_buffer`.
//!
//! **Advanced/Core.** This is a low-level building block for zero-copy
//! ownership transfer into [`Column`](super::Column). Most users should
//! construct columns via [`Runtime::column_from_slice`](crate::Runtime::column_from_slice)
//! or [`Runtime::series`](crate::Runtime::series) instead.

use std::{
    fmt, mem,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    core::{
        error::{self, Result},
        native_teardown::drop_device_owned_handle,
        types::CudfType,
    },
    ops::validation::{self, FiniteValidationProof, Int64RangeValidationProof},
    runtime::{DeviceId, ExecutionBindingId, ExecutionPlacement, Runtime, execution_domain::ExecutionDomain},
};

/// An owned block of GPU device memory.
///
/// `DeviceBuffer` wraps an `rmm::device_buffer` via the cudf-sys C shim.
/// The underlying GPU memory is freed when this value is dropped.
/// It retains the explicit runtime domain that created it so the stream and
/// memory-resource owners outlive the allocation and so later zero-copy
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
    handle: NonNull<cudf_sys::cudf_device_buffer_handle_state>,
    domain: Arc<ExecutionDomain>,
}

/// Borrowing proof that one entire [`DeviceBuffer`] is an aligned Float32
/// allocation containing only finite values.
///
/// The shared borrow prevents safe mutable pointer access while the proof is
/// live. The proof is bound to the allocation's exact retained execution
/// binding and covers every byte in the buffer.
pub struct ValidatedFloat32DeviceBuffer<'a> {
    buffer: &'a DeviceBuffer,
    proof: FiniteValidationProof,
}

impl ValidatedFloat32DeviceBuffer<'_> {
    /// Number of Float32 values covered by this whole-buffer proof.
    pub fn len(&self) -> usize {
        self.proof.validated_values()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read-only pointer to the finite Float32 allocation.
    pub fn as_ptr(&self) -> Result<*const c_void> {
        self.buffer.as_ptr()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.proof.binding_id()
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.proof.placement()
    }
}

impl fmt::Debug for ValidatedFloat32DeviceBuffer<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFloat32DeviceBuffer")
            .field("len", &self.len())
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}

/// Borrowing proof that one entire [`DeviceBuffer`] is an aligned Int64
/// allocation whose values all lie in one half-open range.
pub struct ValidatedInt64DeviceBufferRange<'a> {
    buffer: &'a DeviceBuffer,
    proof: Int64RangeValidationProof,
}

impl ValidatedInt64DeviceBufferRange<'_> {
    pub fn len(&self) -> usize {
        self.proof.validated_values()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn min_inclusive(&self) -> i64 {
        self.proof.min_inclusive()
    }

    pub fn max_exclusive(&self) -> i64 {
        self.proof.max_exclusive()
    }

    /// Read-only pointer to the range-validated Int64 allocation.
    pub fn as_ptr(&self) -> Result<*const c_void> {
        self.buffer.as_ptr()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.proof.binding_id()
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.proof.placement()
    }
}

impl fmt::Debug for ValidatedInt64DeviceBufferRange<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedInt64DeviceBufferRange")
            .field("len", &self.len())
            .field("min_inclusive", &self.min_inclusive())
            .field("max_exclusive", &self.max_exclusive())
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}

// SAFETY: DeviceBuffer owns a GPU device allocation via an opaque C++ handle.
// The handle is not thread-local. Send is safe because ownership transfer
// between threads is supported by the CUDA runtime. Sync is safe because all
// shared methods do not mutate the allocation; host copies and mutable pointer
// access require exclusive ownership.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    fn create_on_current_device(size_bytes: usize, domain: Arc<ExecutionDomain>) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_create_ex(size_bytes, exec_options.options(), &mut handle) },
            "device_buffer::create_on_current_device",
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, "device_buffer::create_on_current_device", "device buffer")?,
            domain,
        })
    }

    pub(crate) fn new_in_runtime(runtime: &Runtime, size_bytes: usize) -> Result<Self> {
        let domain = Arc::clone(runtime.execution_domain());
        Self::create_on_current_device(size_bytes, domain)
    }

    pub(crate) fn from_host_data_in_runtime<T: CudfType>(runtime: &Runtime, data: &[T]) -> Result<Self> {
        let domain = Arc::clone(runtime.execution_domain());
        let byte_len = mem::size_of_val(data);
        let buf = Self::create_on_current_device(byte_len, domain)?;
        if byte_len > 0 {
            let exec_options = buf.domain.enter_execution()?;
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_device_buffer_copy_from_host_ex(
                        buf.handle.as_ptr(),
                        data.as_ptr() as *const c_void,
                        byte_len,
                        exec_options.options(),
                    )
                },
                "device_buffer::from_host_data_in_runtime",
            )?;
        }
        Ok(buf)
    }

    /// Issue an asynchronous host-to-device copy of `data` into this buffer
    /// at byte offset `dst_offset` on the runtime's stream, without
    /// synchronizing.
    ///
    /// The caller must keep the source region valid and unmodified until the
    /// copy completes in stream order (record a [`CudaEvent`] on the same
    /// stream after this call and wait on it before reusing the region). For
    /// true asynchronous overlap `data` should live in pinned host memory
    /// (see `crate::runtime::host_staging`); pageable sources fall back to
    /// the driver's staged copy path.
    ///
    /// [`CudaEvent`]: crate::runtime::CudaEvent
    pub(crate) fn copy_from_host_range_async(&mut self, dst_offset: usize, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let exec_options = self.domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_device_buffer_copy_from_host_range_async_ex(
                    self.handle.as_ptr(),
                    dst_offset,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    exec_options.options(),
                )
            },
            "device_buffer::copy_from_host_range_async",
        )
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

    /// Returns the raw device pointer for read-only access.
    ///
    /// The pointer refers to GPU memory and cannot be dereferenced from host
    /// code. It is intended for diagnostics and for passing to GPU libraries.
    ///
    /// For zero-size buffers this call still succeeds, but the returned
    /// pointer may be null because there is no addressable device range.
    pub fn as_ptr(&self) -> Result<*const c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_data(self.handle.as_ptr(), &mut out) },
            "device_buffer::as_ptr",
        )?;
        Ok(out.cast_const())
    }

    /// Returns the raw device pointer for mutable access.
    ///
    /// The pointer refers to GPU memory and cannot be dereferenced from host
    /// code. The exclusive `&mut self` receiver prevents safe callers from
    /// obtaining a mutation-capable pointer while aliases to this owner are
    /// active.
    ///
    /// For zero-size buffers this call still succeeds, but the returned
    /// pointer may be null because there is no addressable device range.
    pub fn as_mut_ptr(&mut self) -> Result<*mut c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_device_buffer_data(self.handle.as_ptr(), &mut out) },
            "device_buffer::as_mut_ptr",
        )?;
        Ok(out)
    }

    /// CUDA device on which this allocation resides.
    pub fn device(&self) -> DeviceId {
        self.domain.device_id()
    }

    /// Diagnostic summary of this allocation's retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.domain.placement()
    }

    /// Stable identity of the exact stream and memory-resource binding that
    /// owns this allocation.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.domain.binding_id()
    }

    /// Validate this entire allocation as aligned Float32 values and return a
    /// borrowing finite-value proof on the exact retained execution binding.
    pub fn validate_finite_float32(&self) -> Result<ValidatedFloat32DeviceBuffer<'_>> {
        let proof = validation::validate_finite_float32_buffer(self)
            .map_err(|error| error.reanchor("device_buffer::validate_finite_float32"))?;
        Ok(ValidatedFloat32DeviceBuffer { buffer: self, proof })
    }

    /// Require finite Float64 values across this entire allocation on its retained binding.
    pub fn validate_finite_float64(&self) -> Result<()> {
        validation::validate_finite_float64_buffer(self)
            .map(|_| ())
            .map_err(|error| error.reanchor("device_buffer::validate_finite_float64"))
    }

    /// Validate this entire allocation as aligned Int64 values in
    /// `[min_inclusive, max_exclusive)` and return a borrowing proof on the
    /// exact retained execution binding.
    pub fn validate_int64_range(
        &self,
        min_inclusive: i64,
        max_exclusive: i64,
    ) -> Result<ValidatedInt64DeviceBufferRange<'_>> {
        let proof = validation::validate_int64_buffer_range(self, min_inclusive, max_exclusive)
            .map_err(|error| error.reanchor("device_buffer::validate_int64_range"))?;
        Ok(ValidatedInt64DeviceBufferRange { buffer: self, proof })
    }

    /// Validate that each Int64 value is in the range or equals `sentinel`.
    pub fn validate_int64_range_or_sentinel(
        &self,
        min_inclusive: i64,
        max_exclusive: i64,
        sentinel: i64,
    ) -> Result<()> {
        validation::validate_int64_buffer_range_or_sentinel(self, min_inclusive, max_exclusive, sentinel)
            .map_err(|error| error.reanchor("device_buffer::validate_int64_range_or_sentinel"))
    }

    /// Add a finite scalar to each Float32 value on the retained execution stream.
    /// Non-finite values retain their IEEE non-finite status.
    pub fn add_scalar_to_float32(&mut self, scalar: f32) -> Result<()> {
        validation::add_scalar_to_float32_buffer(self, scalar)
    }

    /// Fill Int64 values where paired Float32 values are infinite on the retained execution stream.
    pub fn fill_int64_where_float32_infinite(&mut self, paired_values: &Self, fill_value: i64) -> Result<()> {
        validation::fill_int64_where_float32_infinite(self, paired_values, fill_value)
            .map_err(|error| error.reanchor("device_buffer::fill_int64_where_float32_infinite"))
    }

    /// Replace each Int64 equal to `from` with `to` and set its paired Float32 to positive infinity.
    pub fn replace_int64_value_and_set_paired_float32_infinite(
        &mut self,
        paired_values: &mut Self,
        from: i64,
        to: i64,
    ) -> Result<()> {
        validation::replace_int64_value_and_set_paired_float32_infinite(self, paired_values, from, to)
    }

    /// Require finite Float32 values where paired Int64 values differ from `excluded_value`.
    pub fn validate_finite_float32_unless_int64_equals(&self, paired_values: &Self, excluded_value: i64) -> Result<()> {
        validation::validate_finite_float32_unless_int64_equals(self, paired_values, excluded_value)
            .map_err(|error| error.reanchor("device_buffer::validate_finite_float32_unless_int64_equals"))
    }

    /// Consume this buffer and return the raw FFI handle **without** running
    /// `Drop`.
    ///
    /// The caller is responsible for either passing the handle to an FFI
    /// function that consumes it (e.g., `cudf_column_adopt_device_buffer`) or
    /// reconstructing a `DeviceBuffer` via [`from_raw_parts`](Self::from_raw_parts).
    pub(crate) fn into_raw_parts(self) -> (cudf_sys::cudf_device_buffer_handle, Arc<ExecutionDomain>) {
        let this = mem::ManuallyDrop::new(self);
        let Self { handle: _, domain: _ } = &*this;
        let domain = unsafe { ptr::read(&this.domain) };
        (this.handle.as_ptr(), domain)
    }

    pub(crate) fn device_snapshot(&self) -> i32 {
        self.domain.device_snapshot()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    /// Reconstruct a `DeviceBuffer` from raw FFI parts, taking ownership.
    ///
    /// # Safety
    ///
    /// The handle must be a valid `cudf_device_buffer_handle` that has not
    /// been consumed by another FFI call.
    pub(crate) unsafe fn from_raw_parts(
        handle: cudf_sys::cudf_device_buffer_handle,
        domain: Arc<ExecutionDomain>,
    ) -> Result<Self> {
        Ok(Self { handle: error::require_non_null(handle, "device_buffer::from_raw_parts", "device buffer")?, domain })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        drop_device_owned_handle(self.handle.as_ptr(), self.domain.device_id(), "DeviceBuffer::drop", |owned| unsafe {
            cudf_sys::cudf_device_buffer_destroy(owned)
        });
    }
}
