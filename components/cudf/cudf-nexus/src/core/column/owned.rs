//! GPU column types.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).

use std::{
    any::Any,
    marker::PhantomData,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{device_slice::DeviceSlice, view::ColumnView};
use crate::{
    core::{
        error::{
            self, DeviceIdFact, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, FACT_LEFT_DEVICE,
            FACT_LEFT_TYPE_ID, FACT_RIGHT_DEVICE, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result,
        },
        scalar::Scalar,
        types::{
            CudfType, DataType, TypeId, count_nulls, cudf_null_mask_allocation_size_bytes, validity_to_bitmask,
            validity_to_ffi_bitmask,
        },
    },
    ops::{binary, unary},
    runtime::{
        Runtime,
        execution_domain::{ConsumerCompletionFence, ExecutionDomain},
    },
};

fn usize_to_i32_len(len: usize, what: &'static str, operation: &'static str) -> Result<i32> {
    i32::try_from(len).map_err(|_| error::range_limit_error(operation, what, "i32::MAX", len))
}

fn type_id_label(type_id: TypeId) -> String {
    type_id.to_string()
}

fn fixed_width_type_required_error(
    operation: &'static str,
    message: &'static str,
    type_id: TypeId,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedDtype, operation, message)
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("reason", "requires_fixed_width_type")
        .with_detail("remedy", remedy)
}

fn null_device_ptr_error(operation: &'static str, row_count: usize, type_id: TypeId) -> Error {
    Error::new_with(ErrorCode::NullUnexpected, operation, "device_ptr is null but row_count > 0")
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("resource", "device_ptr")
        .with_detail("reason", "null_pointer_with_positive_row_count")
        .with_detail("remedy", "ensure device_ptr is non-null whenever row_count is positive")
        .with_detail("row_count", row_count.to_string())
}

fn checked_required_buffer_bytes(
    operation: &'static str,
    resource: &'static str,
    row_count: usize,
    elem_size_bytes: i32,
    type_id: TypeId,
) -> Result<usize> {
    row_count.checked_mul(elem_size_bytes as usize).ok_or_else(|| {
        Error::new_with(ErrorCode::UnsupportedOperation, operation, "required buffer size overflowed usize")
            .with_fact(FACT_TYPE_ID, type_id)
            .with_detail("resource", resource)
            .with_detail("reason", "size_overflowed_usize")
            .with_detail(
                "remedy",
                "reduce row_count or choose a smaller fixed-width type so the required buffer size fits within usize",
            )
            .with_detail("row_count", row_count.to_string())
            .with_detail("elem_size_bytes", elem_size_bytes.to_string())
    })
}

fn buffer_size_mismatch_error(
    operation: &'static str,
    resource: &'static str,
    message: &'static str,
    actual_bytes: usize,
    expected_bytes: usize,
    row_count: usize,
    type_id: TypeId,
) -> Error {
    Error::new_with(ErrorCode::ColumnLengthMismatch, operation, message)
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("resource", resource)
        .with_detail("reason", "size_mismatch_with_row_count")
        .with_detail("remedy", format!("ensure {resource} has exactly row_count * element_size bytes"))
        .with_detail("actual_bytes", actual_bytes.to_string())
        .with_detail("expected_bytes", expected_bytes.to_string())
        .with_detail("row_count", row_count.to_string())
}

fn null_mask_buffer_too_small_error(
    operation: &'static str,
    actual_bytes: usize,
    expected_bytes: usize,
    row_count: usize,
) -> Error {
    Error::new_with(ErrorCode::ColumnLengthMismatch, operation, "null mask buffer too small")
        .with_detail("resource", "null_mask_buffer")
        .with_detail("reason", "insufficient_allocation_size")
        .with_detail(
            "remedy",
            "provide at least cudf_null_mask_allocation_size_bytes(row_count) bytes in null_mask_buffer",
        )
        .with_detail("actual_bytes", actual_bytes.to_string())
        .with_detail("expected_bytes", expected_bytes.to_string())
        .with_detail("row_count", row_count.to_string())
}

fn null_count_exceeds_row_count_error(operation: &'static str, null_count: usize, row_count: usize) -> Error {
    Error::new_with(ErrorCode::ColumnLengthMismatch, operation, "null_count exceeds row_count")
        .with_fact(FACT_EXPECTED_ROWS, row_count)
        .with_fact(FACT_ACTUAL_ROWS, null_count)
        .with_detail("resource", "null_count")
        .with_detail("reason", "exceeds_row_count")
        .with_detail("remedy", "ensure null_count does not exceed row_count")
}

/// An owned GPU column backed by a `cudf::column`.
///
/// The underlying device memory is freed when this value is dropped.
pub struct Column {
    handle: NonNull<cudf_sys::cudf_column_handle_state>,
    domain: Arc<ExecutionDomain>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
}

// SAFETY: Column owns a GPU device buffer via an opaque C++ handle. The handle
// is not thread-local. Send is safe because ownership transfer between threads is
// supported by the CUDA runtime. Sync is safe because all methods that observe state
// create independent FFI view handles internally.
unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    fn from_handle_with_domain(handle: cudf_sys::cudf_column_handle, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_handle_with_domain_and_owner(handle, domain, None)
    }

    fn create_empty_in_domain(type_id: TypeId, domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_create_empty_ex(type_id.to_raw(), exec_options.options(), &mut handle) },
            operation,
        )?;
        Self::from_handle_with_domain(handle, domain)
    }

    fn create_empty_with_dtype_in_domain(
        dtype: DataType,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let empty = Self::create_empty_in_domain(dtype.id(), Arc::clone(&domain), operation)?;
        if !dtype.id().is_decimal() || dtype.scale() == 0 {
            return Ok(empty);
        }

        let empty_view = empty.view()?;
        unary::cast(&empty_view, &dtype).map_err(|error| error.reanchor(operation))
    }

    fn from_handle_with_domain_and_owner(
        handle: cudf_sys::cudf_column_handle,
        domain: Arc<ExecutionDomain>,
        retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: error::require_non_null(handle, "column::from_handle_with_domain_and_owner", "column")?,
            domain,
            retained_owner,
        })
    }

    pub(crate) fn copy_into_domain(&self, domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        // Two-way ordering with the source provenance: the copy enqueued below
        // reads this column on the target stream, and the fence (dropped after
        // the FFI call) keeps the source stream from reusing its buffers under
        // the in-flight copy.
        let _consumer_fence = ExecutionDomain::insert_stream_dependency(&self.domain, &domain, operation)?;
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_copy_ex(self.handle.as_ptr(), exec_options.options(), &mut handle) },
            operation,
        )?;
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn empty_in_domain(
        type_id: TypeId,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        Self::create_empty_in_domain(type_id, domain, operation)
    }

    pub(crate) fn empty_with_dtype_in_domain(
        dtype: DataType,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        Self::create_empty_with_dtype_in_domain(dtype, domain, operation)
    }

    /// Create an empty column of the given type inside the given explicit
    /// runtime.
    ///
    /// The resulting column carries the runtime's retained provenance so it
    /// can participate in explicit-provenance pipelines without triggering
    /// mixed-provenance rejection. No device memory is allocated for the
    /// column data itself.
    pub(crate) fn empty_with_runtime(runtime: &Runtime, dtype: DataType) -> Result<Self> {
        Self::create_empty_with_dtype_in_domain(
            dtype,
            Arc::clone(runtime.execution_domain()),
            "column::empty_with_runtime",
        )
    }

    /// Create a column from a host slice of fixed-width elements.
    ///
    /// # Example
    /// ```no_run
    /// use cudf::Runtime;
    /// let rt = Runtime::new().unwrap();
    /// let col = rt.column_from_slice(&[1i32, 2, 3]).unwrap();
    /// ```
    pub(crate) fn from_slice_with_runtime<T: CudfType>(runtime: &Runtime, data: &[T]) -> Result<Self> {
        Self::from_slice_impl(data, None, Arc::clone(runtime.execution_domain()), "column::from_slice_with_runtime")
    }

    pub(crate) fn from_slice_in_domain<T: CudfType>(data: &[T], domain: &Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_slice_impl(data, None, Arc::clone(domain), "column::from_slice_in_domain")
    }

    fn from_slice_impl<T: CudfType>(
        data: &[T],
        validity: Option<&[bool]>,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if let Some(validity) = validity
            && data.len() != validity.len()
        {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                operation,
                "data and validity slices must have the same length",
            )
            .with_fact(FACT_EXPECTED_ROWS, data.len())
            .with_fact(FACT_ACTUAL_ROWS, validity.len()));
        }

        let mut handle = ptr::null_mut();
        let size = usize_to_i32_len(data.len(), "row_count", operation)?;
        let null_mask = validity.map(validity_to_ffi_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.enter_execution()?;
        // SAFETY: data and optional null_mask are valid for the reported sizes.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    data.as_ptr() as *const c_void,
                    size,
                    null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                    null_count,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
        )?;
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn from_fixed_width_host_bytes_with_runtime(
        runtime: &Runtime,
        dtype: DataType,
        data: &[u8],
        validity: Option<&[bool]>,
    ) -> Result<Self> {
        let operation = "column::from_fixed_width_host_bytes_with_runtime";
        let type_id = dtype.id();
        let elem_size = type_id.size_in_bytes()?.ok_or_else(|| {
            fixed_width_type_required_error(
                operation,
                "fixed-width host byte import requires a fixed-width type",
                type_id,
                "use a fixed-width cudf dtype when importing raw host bytes",
            )
        })? as usize;

        if type_id == TypeId::Dictionary32 {
            return Err(fixed_width_type_required_error(
                operation,
                "dictionary columns require keys and indices, not a raw fixed-width value buffer",
                type_id,
                "use a primitive, temporal, or decimal fixed-width dtype for raw host byte import",
            ));
        }

        if !data.len().is_multiple_of(elem_size) {
            return Err(buffer_size_mismatch_error(
                operation,
                "data_buffer",
                "data buffer length must be a multiple of the fixed-width element size",
                data.len(),
                data.len().div_ceil(elem_size) * elem_size,
                data.len() / elem_size,
                type_id,
            ));
        }

        let row_count = data.len() / elem_size;
        let size = usize_to_i32_len(row_count, "row_count", operation)?;
        if let Some(validity) = validity
            && validity.len() != row_count
        {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                operation,
                "data and validity slices must have the same length",
            )
            .with_fact(FACT_EXPECTED_ROWS, row_count)
            .with_fact(FACT_ACTUAL_ROWS, validity.len()));
        }

        let null_mask = validity.map(validity_to_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = runtime.execution_domain().enter_execution()?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_column_create_fixed_width_from_host_ex(
                    type_id.to_raw(),
                    dtype.scale(),
                    data.as_ptr() as *const c_void,
                    size,
                    null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                    null_count,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
            [
                ("type_id_label", type_id_label(type_id)),
                ("scale", dtype.scale().to_string()),
                ("row_count", row_count.to_string()),
            ],
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(runtime.execution_domain()))
    }

    /// Create a nullable column from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub(crate) fn from_slice_nullable_with_runtime<T: CudfType>(
        runtime: &Runtime,
        data: &[T],
        validity: &[bool],
    ) -> Result<Self> {
        Self::from_slice_impl(
            data,
            Some(validity),
            Arc::clone(runtime.execution_domain()),
            "column::from_slice_nullable_with_runtime",
        )
    }

    /// Deep-copy this column.
    pub fn deep_copy(&self) -> Result<Self> {
        self.copy_into_domain(Arc::clone(&self.domain), "column::deep_copy")
    }

    /// Returns the data type of this column.
    pub fn data_type(&self) -> Result<DataType> {
        self.view()?.data_type()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> Result<usize> {
        self.view()?.len()
    }

    /// Returns `true` if the column has zero elements.
    pub fn is_empty(&self) -> Result<bool> {
        self.view()?.is_empty()
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> Result<usize> {
        self.view()?.null_count()
    }

    /// Returns `true` if this column has a null mask allocated.
    pub fn nullable(&self) -> Result<bool> {
        self.view()?.nullable()
    }

    /// Returns the raw device (GPU) data pointer of this column.
    ///
    /// This is a zero-copy observation of the underlying device allocation.
    pub fn device_data_ptr(&self) -> Result<*const c_void> {
        self.view()?.device_data_ptr()
    }

    /// Returns a typed descriptor of the active device memory region.
    pub fn device_slice<T: CudfType>(&self) -> Result<DeviceSlice<'_>> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "column::device_slice",
                "column type does not match requested type",
            )
            .with_fact(FACT_LEFT_TYPE_ID, dt.id())
            .with_fact(FACT_RIGHT_TYPE_ID, T::TYPE_ID));
        }

        let base_ptr = self.device_data_ptr()?;
        let len = self.len()?;
        let elem_size = std::mem::size_of::<T>();

        Ok(DeviceSlice { ptr: base_ptr as *mut c_void, len, type_id: T::TYPE_ID, elem_size, _lifetime: PhantomData })
    }

    /// Copy the column data to a host `Vec<T>`.
    pub fn to_vec<T: CudfType>(&self) -> Result<Vec<T>> {
        self.view()?.to_vec::<T>()
    }

    /// Copy the column data to a host `Vec<Option<T>>`.
    pub fn to_optional_vec<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        self.view()?.to_optional_vec::<T>()
    }

    /// Copy string column data to a host `Vec<String>`, mapping null elements
    /// to empty strings.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        self.view()?.to_strings_lossy()
    }

    /// Copy string data to host as `Vec<Option<String>>`, distinguishing null
    /// from empty string.
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        self.view()?.to_optional_strings()
    }

    /// Create an internal immutable view handle of this column.
    pub(crate) fn view(&self) -> Result<ColumnView<'_>> {
        let mut view_handle = ptr::null_mut();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view(self.handle.as_ptr(), &mut view_handle) },
            "column::view",
        )?;
        unsafe { ColumnView::from_raw_with_domain(view_handle, Arc::clone(&self.domain)) }
    }

    pub(crate) fn payload_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        let exec_options = self.domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_payload_size_ex(self.handle.as_ptr(), exec_options.options(), &mut out) },
            "column::payload_size_bytes",
        )?;
        Ok(out)
    }

    pub(crate) fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_device_allocated_size(self.handle.as_ptr(), &mut out) },
            "column::device_allocated_size_bytes",
        )?;
        Ok(out)
    }

    pub(crate) fn into_raw_parts(self) -> (cudf_sys::cudf_column_handle, Option<Arc<dyn Any + Send + Sync>>) {
        let this = std::mem::ManuallyDrop::new(self);
        let Self { handle: _, domain: _, retained_owner: _ } = &*this;
        let handle = this.handle.as_ptr();
        let retained_owner = unsafe { std::ptr::read(&this.retained_owner) };
        let _domain = unsafe { std::ptr::read(&this.domain) };
        (handle, retained_owner)
    }

    /// Construct a `Column` from a raw FFI handle and an explicit retained domain.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw_with_domain(
        handle: cudf_sys::cudf_column_handle,
        domain: Arc<ExecutionDomain>,
    ) -> Result<Self> {
        Self::from_handle_with_domain(handle, domain)
    }

    /// Construct a `Column` from a raw FFI handle, explicit retained domain,
    /// and retained external owner.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_handle` and the caller must
    /// transfer ownership. `retained_owner` must keep any external resources
    /// required by the raw handle alive for the full lifetime of the returned
    /// column.
    pub(crate) unsafe fn from_raw_with_domain_and_owner(
        handle: cudf_sys::cudf_column_handle,
        domain: Arc<ExecutionDomain>,
        retained_owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        Self::from_handle_with_domain_and_owner(handle, domain, Some(retained_owner))
    }

    /// Create a column by copying data from an existing device (GPU) pointer.
    ///
    /// Performs a device-to-device copy; the new column owns its own memory
    /// independently of the source buffer.
    ///
    /// # Safety
    ///
    /// `device_ptr` must point to valid GPU memory containing at least `len`
    /// elements of the type identified by `type_id`. For `len == 0`,
    /// `device_ptr` may be null.
    ///
    /// Any prior GPU writes to `device_ptr` must be visible to the default
    /// stream used by this API before calling (i.e., establish stream/event
    /// ordering if writes occurred on a different stream).
    ///
    /// Only fixed-width types are supported.
    /// Create a column by copying from a borrowed device pointer inside the
    /// given explicit runtime.
    ///
    /// Uses `cudf_column_create_from_device_data_ex` which routes the copy
    /// through the runtime's stream and memory resource. This variant does
    /// not synchronize before returning. The caller must keep `device_ptr`
    /// valid until the runtime's stream has completed the copy, for example
    /// by calling [`Runtime::synchronize`].
    ///
    /// Only fixed-width types are supported.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to a valid device allocation containing at
    ///   least `len` elements of the type identified by `type_id`.
    /// - Any prior GPU writes to `device_ptr` must be ordered before the
    ///   runtime's stream used by this API. If the data was produced on a
    ///   different stream, establish stream or event ordering before calling.
    /// - The pointer must remain valid until the runtime's stream has
    ///   completed the copy, because this `_with_runtime` variant does not
    ///   synchronize before returning.
    pub(crate) unsafe fn from_device_ptr_with_runtime(
        runtime: &Runtime,
        device_ptr: *const c_void,
        len: usize,
        type_id: TypeId,
    ) -> Result<Self> {
        let operation = "column::from_device_ptr_with_runtime";

        if type_id.size_in_bytes()?.is_none() {
            return Err(fixed_width_type_required_error(
                operation,
                "from_device_ptr only supports fixed-width types",
                type_id,
                "use a fixed-width type when importing a raw device pointer as a cudf column",
            ));
        }

        let size = usize_to_i32_len(len, "row_count", operation)?;

        if len > 0 && device_ptr.is_null() {
            return Err(null_device_ptr_error(operation, len, type_id));
        }

        let output_domain = Arc::clone(runtime.execution_domain());
        let exec_options = output_domain.enter_execution()?;
        let mut handle = ptr::null_mut();
        // SAFETY: caller guarantees device_ptr is valid for `size` elements.
        // The _ex variant does NOT synchronize — caller must keep device_ptr
        // valid until the runtime's stream completes.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_column_create_from_device_data_ex(
                    type_id.to_raw(),
                    device_ptr,
                    size,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
            [("row_count", len.to_string()), ("type_id_label", type_id_label(type_id))],
        )?;
        Self::from_handle_with_domain(handle, output_domain)
    }

    /// Create a STRING type column from a slice of string references.
    ///
    /// Each string is copied to GPU memory. The resulting column has no null mask.
    ///
    /// # Example
    /// ```no_run
    /// use cudf::Runtime;
    /// let rt = Runtime::new().unwrap();
    /// let col = rt.column_from_strs(&["hello", "world"]).unwrap();
    /// ```
    pub(crate) fn from_strs_with_runtime(runtime: &Runtime, data: &[&str]) -> Result<Self> {
        Self::from_strs_impl(data, None, Arc::clone(runtime.execution_domain()), "column::from_strs_with_runtime")
    }

    pub(crate) fn from_strs_in_domain(data: &[&str], domain: &Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_strs_impl(data, None, Arc::clone(domain), "column::from_strs_in_domain")
    }

    fn from_strs_impl(
        data: &[&str],
        validity: Option<&[bool]>,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if let Some(validity) = validity
            && data.len() != validity.len()
        {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                operation,
                "data and validity slices must have the same length",
            )
            .with_fact(FACT_EXPECTED_ROWS, data.len())
            .with_fact(FACT_ACTUAL_ROWS, validity.len()));
        }

        let num_strings = usize_to_i32_len(data.len(), "row_count", operation)?;

        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr() as *const c_char).collect();
        let lengths: Vec<i32> =
            data.iter().map(|s| usize_to_i32_len(s.len(), "string_element", operation)).collect::<Result<Vec<_>>>()?;
        let null_mask = validity.map(validity_to_ffi_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.enter_execution()?;

        let mut handle = ptr::null_mut();
        // SAFETY: ptrs and lengths are derived from valid &str slices and have
        // num_strings elements. No null mask (ptr::null(), 0).
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_strings_ex(
                    ptrs.as_ptr() as *mut *const c_char,
                    lengths.as_ptr(),
                    num_strings,
                    null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                    null_count,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
        )?;
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn nulls_in_domain(
        dtype: DataType,
        len: usize,
        domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let size = usize_to_i32_len(len, "row_count", operation)?;
        if len == 0 {
            return Self::create_empty_with_dtype_in_domain(dtype, Arc::clone(domain), operation);
        }

        let type_id = dtype.id();
        let raw = match type_id {
            TypeId::String => {
                let dummy: Vec<&str> = vec![""; len];
                let validity = vec![false; len];
                Self::from_strs_impl(&dummy, Some(&validity), Arc::clone(domain), operation)
            },
            _ => {
                let elem_size = type_id.size_in_bytes().map_err(|e| e.reanchor(operation))?.ok_or_else(|| {
                    fixed_width_type_required_error(
                        operation,
                        "null-column construction requires a fixed-width type or STRING",
                        type_id,
                        "use a fixed-width type or STRING when constructing an all-null column",
                    )
                })?;

                let data_len_bytes =
                    checked_required_buffer_bytes(operation, "data_buffer_size", len, elem_size, type_id)?;

                let data = vec![0u8; data_len_bytes];
                let validity = vec![false; len];
                let null_mask = validity_to_ffi_bitmask(&validity);
                let null_count = size;
                let exec_options = domain.enter_execution()?;

                let mut handle = ptr::null_mut();
                // SAFETY: data is zeroed bytes of the correct size, null_mask marks
                // all elements null. type_id.to_raw() is valid for fixed-width types.
                error::check_status_in_with_details(
                    unsafe {
                        cudf_sys::cudf_column_create_numeric_ex(
                            type_id.to_raw(),
                            data.as_ptr() as *const c_void,
                            size,
                            null_mask.as_ptr(),
                            null_count,
                            exec_options.options(),
                            &mut handle,
                        )
                    },
                    operation,
                    [("type_id_label", type_id_label(type_id)), ("row_count", len.to_string())],
                )?;
                Self::from_handle_with_domain(handle, Arc::clone(domain))
            },
        }?;

        if !dtype.id().is_decimal() || dtype.scale() == 0 {
            return Ok(raw);
        }

        let raw_view = raw.view()?;
        unary::cast(&raw_view, &dtype).map_err(|error| error.reanchor(operation))
    }

    /// Create an all-null column of the given type and length inside the given
    /// explicit runtime.
    ///
    /// The resulting column carries the runtime's retained provenance so it can
    /// participate in explicit-provenance pipelines without triggering
    /// mixed-provenance rejection.
    pub(crate) fn nulls_with_runtime(runtime: &Runtime, dtype: DataType, len: usize) -> Result<Self> {
        Self::nulls_in_domain(dtype, len, runtime.execution_domain(), "column::nulls_with_runtime")
    }

    /// Create a column by repeating a scalar value `size` times inside the
    /// given explicit runtime.
    ///
    /// If the input scalar belongs to a different retained provenance, it is
    /// recreated in the target runtime first so placement and execution
    /// resource selection remain honest.
    pub(crate) fn from_scalar_with_runtime(runtime: &Runtime, scalar: &Scalar, size: usize) -> Result<Self> {
        if ExecutionDomain::same_provenance(scalar.execution_domain(), runtime.execution_domain()) {
            return Self::from_scalar(scalar, size);
        }

        let runtime_scalar = scalar.copy_into_domain(runtime.execution_domain())?;
        Self::from_scalar(&runtime_scalar, size)
    }

    pub(crate) fn from_sequence_with_runtime<T: CudfType>(
        runtime: &Runtime,
        init: T,
        step: T,
        size: usize,
    ) -> Result<Self> {
        let size_i32 = i32::try_from(size)
            .map_err(|_| error::range_limit_error("column::from_sequence_with_runtime", "size", "i32::MAX", size))?;
        let init_scalar = runtime.scalar(init)?;
        let step_scalar = runtime.scalar(step)?;
        let mut handle = ptr::null_mut();
        let exec_options = runtime.execution_domain().enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_sequence_ex(
                    init_scalar.raw_handle(),
                    step_scalar.raw_handle(),
                    size_i32,
                    exec_options.options(),
                    &mut handle,
                )
            },
            "column::from_sequence_with_runtime",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(runtime.execution_domain()))
    }

    pub(crate) fn from_sequence_in_domain<T: CudfType>(
        init: T,
        step: T,
        size: usize,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let size_i32 =
            i32::try_from(size).map_err(|_| error::range_limit_error(operation, "size", "i32::MAX", size))?;
        let init_scalar = Scalar::from_value_in_domain(init, &domain)?;
        let step_scalar = Scalar::from_value_in_domain(step, &domain)?;
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_sequence_ex(
                    init_scalar.raw_handle(),
                    step_scalar.raw_handle(),
                    size_i32,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
        )?;
        Self::from_handle_with_domain(handle, domain)
    }

    /// Create a nullable STRING type column from host string data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    /// For null rows, the string content is ignored.
    pub(crate) fn from_strs_nullable_with_runtime(runtime: &Runtime, data: &[&str], validity: &[bool]) -> Result<Self> {
        Self::from_strs_impl(
            data,
            Some(validity),
            Arc::clone(runtime.execution_domain()),
            "column::from_strs_nullable_with_runtime",
        )
    }

    /// Create a column by repeating a scalar value `size` times.
    ///
    /// Uses `cudf::make_column_from_scalar` under the hood. For `size == 0`,
    /// an empty column of the scalar's type is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` exceeds `i32::MAX` (FFI limit).
    pub fn from_scalar(scalar: &Scalar, size: usize) -> Result<Self> {
        let size_i32 = i32::try_from(size)
            .map_err(|_| error::range_limit_error("column::from_scalar", "size", "i32::MAX", size))?;
        let mut handle = ptr::null_mut();
        let exec_options = scalar.domain.enter_execution()?;
        // SAFETY: scalar.raw_handle() is valid (Scalar maintains NonNull invariant).
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_from_scalar_ex(scalar.raw_handle(), size_i32, &mut handle, exec_options.options())
            },
            "column::from_scalar",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(&scalar.domain))
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    pub(crate) fn with_execution_domain(mut self, domain: Arc<ExecutionDomain>) -> Self {
        self.domain = domain;
        self
    }
    pub(crate) fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
    pub(crate) fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }
    pub(crate) fn __debug_execution_domain_mr_value(&self) -> usize {
        self.domain.debug_mr_value()
    }
    pub(crate) fn __debug_rebind_execution_domain_explicit(mut self, identity: u64) -> Result<Self> {
        let target = ExecutionDomain::debug_explicit(identity)?;
        let _consumer_fence = ExecutionDomain::insert_stream_dependency(
            &self.domain,
            &target,
            "column::__debug_rebind_execution_domain_explicit",
        )?;
        self.domain = target;
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Zero-copy device buffer adoption
// ---------------------------------------------------------------------------

fn require_runtime_matches_buffer_device(
    runtime: &Runtime,
    buffer_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<(Arc<ExecutionDomain>, ConsumerCompletionFence)> {
    let runtime_device = runtime.device();
    let device_snapshot = buffer_domain.device_snapshot();
    if runtime_device.ordinal() as i32 != device_snapshot {
        return Err(Error::new_with(
            ErrorCode::CrossDeviceInputs,
            operation,
            "device buffer belongs to a different CUDA device than the target runtime",
        )
        .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(runtime_device.ordinal() as i32))
        .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(device_snapshot))
        .with_detail("resource", "device_buffer")
        .with_detail("remedy", "move the buffer into a Runtime bound to the same CUDA device before adopting it"));
    }

    let output_domain = Arc::clone(runtime.execution_domain());
    let consumer_fence = ExecutionDomain::insert_stream_dependency(buffer_domain, &output_domain, operation)?;
    Ok((output_domain, consumer_fence))
}

impl Column {
    /// Create a column by adopting (moving) an existing [`DeviceBuffer`].
    ///
    /// **Zero-copy:** no device-to-device memcpy occurs. The buffer's device
    /// memory becomes the column's data buffer directly. The `DeviceBuffer` is
    /// consumed and must not be used after this call.
    ///
    /// Only fixed-width types are supported. The buffer's byte size must equal
    /// `row_count * type_id.size_in_bytes()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::{Runtime, TypeId, core::DeviceBuffer};
    ///
    /// let rt = Runtime::new().unwrap();
    /// let buf = rt.device_buffer_from_host_data(&[1i32, 2, 3]).unwrap();
    /// let col = rt.column_from_device_buffer(buf, 3, TypeId::Int32).unwrap();
    /// assert_eq!(col.len().unwrap(), 3);
    /// ```
    pub(crate) fn from_device_buffer_with_runtime(
        runtime: &Runtime,
        buffer: super::DeviceBuffer,
        row_count: usize,
        type_id: TypeId,
    ) -> Result<Self> {
        let operation = "column::from_device_buffer_with_runtime";

        let elem_size = type_id.size_in_bytes().map_err(|e| e.reanchor(operation))?.ok_or_else(|| {
            fixed_width_type_required_error(
                operation,
                "from_device_buffer only supports fixed-width types",
                type_id,
                "use a fixed-width type when adopting a device buffer as a cudf column",
            )
        })?;

        let row_count_i32 = usize_to_i32_len(row_count, "row_count", operation)?;

        let expected_bytes = checked_required_buffer_bytes(operation, "data_buffer", row_count, elem_size, type_id)?;
        let actual_bytes = buffer.size_bytes().map_err(|e| e.reanchor(operation))?;
        if actual_bytes != expected_bytes {
            return Err(buffer_size_mismatch_error(
                operation,
                "data_buffer",
                "buffer size does not match row_count * element size",
                actual_bytes,
                expected_bytes,
                row_count,
                type_id,
            ));
        }

        let (raw_buf, buffer_domain) = buffer.into_raw_parts();
        let (output_domain, _consumer_fence) =
            match require_runtime_matches_buffer_device(runtime, &buffer_domain, operation) {
                Ok(resolved) => resolved,
                Err(err) => {
                    let _cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_buf, buffer_domain) };
                    return Err(err);
                },
            };

        let mut handle = ptr::null_mut();
        let status =
            unsafe { cudf_sys::cudf_column_adopt_device_buffer(type_id.to_raw(), raw_buf, row_count_i32, &mut handle) };

        match error::check_status_in(status, operation) {
            Ok(()) => {
                let retained_owner: Arc<dyn Any + Send + Sync> = buffer_domain;
                Self::from_handle_with_domain_and_owner(handle, output_domain, Some(retained_owner))
            },
            Err(err) => {
                let _cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_buf, Arc::clone(&buffer_domain)) };
                Err(err)
            },
        }
    }

    /// Create a nullable column by adopting data and null-mask
    /// [`DeviceBuffer`]s.
    ///
    /// **Zero-copy:** no device-to-device memcpy occurs. Both buffers'
    /// device memory becomes the column's data and null-mask directly.
    ///
    /// The `null_mask_buffer` must contain at least
    /// `cudf::bitmask_allocation_size_bytes(row_count)` bytes of bitmask data.
    /// In practice this is `ceil(row_count / 8)` rounded up to a 64-byte
    /// allocation boundary. `null_count` must not exceed `row_count`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::{Runtime, TypeId, core::DeviceBuffer};
    ///
    /// let rt = Runtime::new().unwrap();
    /// let data_buf = rt.device_buffer_from_host_data(&[10i32, 20, 30]).unwrap();
    /// // Bitmask: bits 0 and 2 set → rows 0 and 2 are valid, row 1 is null.
    /// let mut mask = [0u8; 64];
    /// mask[0] = 0x05;
    /// let mask_buf = rt.device_buffer_from_host_data(&mask).unwrap();
    /// let col = rt
    ///     .column_from_device_buffer_nullable(
    ///         data_buf,
    ///         mask_buf,
    ///         1,
    ///         3,
    ///         TypeId::Int32,
    ///     )
    ///     .unwrap();
    /// ```
    pub(crate) fn from_device_buffer_nullable_with_runtime(
        runtime: &Runtime,
        data_buffer: super::DeviceBuffer,
        null_mask_buffer: super::DeviceBuffer,
        null_count: usize,
        row_count: usize,
        type_id: TypeId,
    ) -> Result<Self> {
        let operation = "column::from_device_buffer_nullable_with_runtime";

        let elem_size = type_id.size_in_bytes().map_err(|e| e.reanchor(operation))?.ok_or_else(|| {
            fixed_width_type_required_error(
                operation,
                "from_device_buffer_nullable only supports fixed-width types",
                type_id,
                "use a fixed-width type when adopting nullable device buffers as a cudf column",
            )
        })?;

        let row_count_i32 = usize_to_i32_len(row_count, "row_count", operation)?;

        let null_count_i32 = i32::try_from(null_count)
            .map_err(|_| error::range_limit_error(operation, "null_count", "i32::MAX", null_count))?;

        if null_count > row_count {
            return Err(null_count_exceeds_row_count_error(operation, null_count, row_count));
        }

        let expected_data_bytes =
            checked_required_buffer_bytes(operation, "data_buffer", row_count, elem_size, type_id)?;
        let actual_data_bytes = data_buffer.size_bytes().map_err(|e| e.reanchor(operation))?;
        if actual_data_bytes != expected_data_bytes {
            return Err(buffer_size_mismatch_error(
                operation,
                "data_buffer",
                "data buffer size does not match row_count * element size",
                actual_data_bytes,
                expected_data_bytes,
                row_count,
                type_id,
            ));
        }

        let min_mask_bytes = cudf_null_mask_allocation_size_bytes(row_count);
        let actual_mask_bytes = null_mask_buffer.size_bytes().map_err(|e| e.reanchor(operation))?;
        if actual_mask_bytes < min_mask_bytes {
            return Err(null_mask_buffer_too_small_error(operation, actual_mask_bytes, min_mask_bytes, row_count));
        }

        let data_device_snapshot = data_buffer.device_snapshot();
        let mask_device_snapshot = null_mask_buffer.device_snapshot();
        if data_device_snapshot != mask_device_snapshot {
            return Err(Error::new_with(
                ErrorCode::CrossDeviceInputs,
                operation,
                "data buffer and null mask buffer must belong to the same CUDA device",
            )
            .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(data_device_snapshot))
            .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(mask_device_snapshot))
            .with_detail("resource", "null_mask_buffer")
            .with_detail("remedy", "ensure both buffers were allocated on the same CUDA device"));
        }

        let (raw_data, data_domain) = data_buffer.into_raw_parts();
        let (raw_mask, mask_domain) = null_mask_buffer.into_raw_parts();
        let (output_domain, _consumer_fence) =
            match require_runtime_matches_buffer_device(runtime, &data_domain, operation) {
                Ok(resolved) => resolved,
                Err(err) => {
                    let _data_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_data, data_domain) };
                    let _mask_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_mask, mask_domain) };
                    return Err(err);
                },
            };
        let _mask_consumer_fence =
            match ExecutionDomain::insert_stream_dependency(&mask_domain, &output_domain, operation) {
                Ok(fence) => fence,
                Err(err) => {
                    let _data_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_data, data_domain) };
                    let _mask_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_mask, mask_domain) };
                    return Err(err);
                },
            };

        let mut handle = ptr::null_mut();
        let status = unsafe {
            cudf_sys::cudf_column_adopt_device_buffer_nullable(
                type_id.to_raw(),
                raw_data,
                raw_mask,
                null_count_i32,
                row_count_i32,
                &mut handle,
            )
        };

        match error::check_status_in(status, operation) {
            Ok(()) => {
                let retained_owner: Arc<dyn Any + Send + Sync> = Arc::new(vec![data_domain, mask_domain]);
                Self::from_handle_with_domain_and_owner(handle, output_domain, Some(retained_owner))
            },
            Err(err) => {
                let _data_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_data, Arc::clone(&data_domain)) };
                let _mask_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_mask, Arc::clone(&mask_domain)) };
                Err(err)
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Binary operations on Column
// ---------------------------------------------------------------------------

impl Column {
    /// Generic binary operation with inferred output type.
    ///
    /// # Cost
    /// GPU-allocating: performs device-side compute and produces a new output
    /// column. No host materialization occurs.
    pub fn binary_op(&self, rhs: &Column, op: binary::BinaryOperator) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            let out_type =
                binary::infer_binary_output_dtype_in(lv.data_type()?, rv.data_type()?, op, "column::binary_op")?;
            binary::binary_operation_vv(&lv, &rv, op, &out_type)
        })()
        .map_err(|e| e.reanchor("column::binary_op").with_detail("operator", op.name()))
    }

    /// Generic binary operation with explicit output type (escape hatch).
    ///
    /// # Cost
    /// GPU-allocating: performs device-side compute and produces a new output
    /// column of the requested type. No host materialization occurs.
    pub fn binary_op_with_type(
        &self,
        rhs: &Column,
        op: binary::BinaryOperator,
        output_type: &DataType,
    ) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            binary::binary_operation_vv(&lv, &rv, op, output_type)
        })()
        .map_err(|e| {
            e.reanchor("column::binary_op_with_type")
                .with_detail("operator", op.name())
                .with_detail("output_type_id", output_type.id().to_string())
                .with_detail("output_type_scale", output_type.scale().to_string())
        })
    }

    // -- Arithmetic methods ---------------------------------------------------

    /// Element-wise addition: `self + rhs`. Output type inferred.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let a = rt.column_from_slice(&[1i32, 2, 3]).unwrap();
    /// let b = rt.column_from_slice(&[10i32, 20, 30]).unwrap();
    /// let sum = a.add(&b).unwrap();
    /// // sum: [11, 22, 33]
    ///
    /// // Also available via std::ops: &a + &b
    /// ```
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn add(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Add).map_err(|e| e.reanchor("column::add"))
    }

    /// Element-wise subtraction: `self - rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn sub(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Sub).map_err(|e| e.reanchor("column::sub"))
    }

    /// Element-wise multiplication: `self * rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn mul(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mul).map_err(|e| e.reanchor("column::mul"))
    }

    /// Element-wise division: `self / rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Div).map_err(|e| e.reanchor("column::div"))
    }

    /// Element-wise true division (always float output).
    ///
    /// # Cost
    /// GPU-allocating: produces a new floating-point output column on the
    /// device.
    pub fn true_div(&self, rhs: &Column) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            let out_type = DataType::new(TypeId::Float64);
            binary::binary_operation_vv(&lv, &rv, binary::BinaryOperator::TrueDiv, &out_type)
        })()
        .map_err(|e| e.reanchor("column::true_div").with_detail("operator", "true_div"))
    }

    /// Element-wise floor division.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn floor_div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::FloorDiv).map_err(|e| e.reanchor("column::floor_div"))
    }

    /// Element-wise remainder: `self % rhs`.
    ///
    /// This is the preferred alias for [`Column::modulo`] to match the
    /// higher-level `Series::rem` naming.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn rem(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mod).map_err(|e| e.reanchor("column::rem"))
    }

    /// Element-wise modulo: `self % rhs`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn modulo(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mod).map_err(|e| e.reanchor("column::modulo"))
    }

    // -- Comparison methods ---------------------------------------------------

    /// Element-wise equal: `self == rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn eq(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Equal).map_err(|e| e.reanchor("column::eq"))
    }

    /// Element-wise not-equal: `self != rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ne(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.reanchor("column::ne"))
    }

    /// Element-wise less-than: `self < rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn lt(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Less).map_err(|e| e.reanchor("column::lt"))
    }

    /// Element-wise greater-than: `self > rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn gt(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Greater).map_err(|e| e.reanchor("column::gt"))
    }

    /// Element-wise less-or-equal: `self <= rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn le(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.reanchor("column::le"))
    }

    /// Element-wise greater-or-equal: `self >= rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ge(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::GreaterEqual).map_err(|e| e.reanchor("column::ge"))
    }

    // -- Scalar operations ----------------------------------------------------

    /// Element-wise: `self op scalar`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: performs device-side compute and produces a new output
    /// column. No host materialization occurs.
    pub fn binary_op_scalar(&self, rhs: &Scalar, op: binary::BinaryOperator) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let out_type = binary::infer_binary_output_dtype_in(
                lv.data_type()?,
                rhs.data_type()?,
                op,
                "column::binary_op_scalar",
            )?;
            binary::binary_operation_vs(&lv, rhs, op, &out_type)
        })()
        .map(|col| col.with_execution_domain(Arc::clone(&self.domain)))
        .map_err(|e| e.reanchor("column::binary_op_scalar").with_detail("operator", op.name()))
    }

    /// Element-wise: `self op scalar` with explicit output type.
    ///
    /// # Cost
    /// GPU-allocating: performs device-side compute and produces a new output
    /// column of the requested type. No host materialization occurs.
    pub fn binary_op_scalar_with_type(
        &self,
        rhs: &Scalar,
        op: binary::BinaryOperator,
        output_type: &DataType,
    ) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            binary::binary_operation_vs(&lv, rhs, op, output_type)
        })()
        .map(|col| col.with_execution_domain(Arc::clone(&self.domain)))
        .map_err(|e| {
            e.reanchor("column::binary_op_scalar_with_type")
                .with_detail("operator", op.name())
                .with_detail("output_type_id", output_type.id().to_string())
                .with_detail("output_type_scale", output_type.scale().to_string())
        })
    }

    /// Element-wise: `self + scalar`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let col = rt.column_from_slice(&[1i32, 2, 3]).unwrap();
    /// let offset = rt.scalar(100i32).unwrap();
    /// let result = col.add_scalar(&offset).unwrap();
    /// // result: [101, 102, 103]
    ///
    /// // Also available via std::ops: &col + 100i32
    /// ```
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn add_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Add).map_err(|e| e.reanchor("column::add_scalar"))
    }

    /// Element-wise: `self - scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn sub_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Sub).map_err(|e| e.reanchor("column::sub_scalar"))
    }

    /// Element-wise: `self * scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn mul_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Mul).map_err(|e| e.reanchor("column::mul_scalar"))
    }

    /// Element-wise: `self / scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn div_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Div).map_err(|e| e.reanchor("column::div_scalar"))
    }

    /// Element-wise: `self == scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn eq_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Equal).map_err(|e| e.reanchor("column::eq_scalar"))
    }

    /// Element-wise: `self != scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ne_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.reanchor("column::ne_scalar"))
    }

    /// Element-wise: `self < scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn lt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Less).map_err(|e| e.reanchor("column::lt_scalar"))
    }

    /// Element-wise: `self > scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn gt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Greater).map_err(|e| e.reanchor("column::gt_scalar"))
    }

    /// Element-wise: `self <= scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn le_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.reanchor("column::le_scalar"))
    }

    /// Element-wise: `self >= scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ge_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::GreaterEqual).map_err(|e| e.reanchor("column::ge_scalar"))
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        crate::core::native_teardown::drop_device_owned_handle(
            self.handle.as_ptr(),
            self.domain.device_id(),
            "Column::drop",
            |owned| unsafe { cudf_sys::cudf_column_destroy(owned) },
        );
    }
}
