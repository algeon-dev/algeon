//! GPU column types.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).

use super::{device_slice::DeviceSlice, view::ColumnView};
use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        scalar::Scalar,
        types::{
            CudfType, DataType, TypeId, count_nulls, cudf_null_mask_allocation_size_bytes, validity_to_ffi_bitmask,
        },
    },
    ops::{binary, unary},
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

fn usize_to_i32_len(len: usize, what: &'static str, operation: &'static str) -> Result<i32> {
    i32::try_from(len).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, format!("{} length exceeds i32::MAX", what))
            .with_operation(operation)
            .with_context("len", len.to_string())
    })
}
use std::{
    any::Any,
    marker::PhantomData,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

/// An owned GPU column backed by a `cudf::column`.
///
/// The underlying device memory is freed when this value is dropped.
pub struct Column {
    handle: NonNull<c_void>,
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
    fn from_handle_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self::from_handle_with_domain_and_owner(handle, domain, None)
    }

    fn create_empty_in_domain(type_id: TypeId, domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_create_empty(type_id.to_raw(), &mut handle) },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
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
        unary::cast(&empty_view, &dtype).map_err(|error| error.with_operation(operation))
    }

    fn from_handle_with_domain_and_owner(
        handle: *mut c_void,
        domain: Arc<ExecutionDomain>,
        retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Self {
        Self { handle: NonNull::new(handle).expect("null column handle"), domain, retained_owner }
    }

    pub(crate) fn copy_into_domain(&self, domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_copy_ex(self.handle.as_ptr(), &exec_options, &mut handle) },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
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
        Self::create_empty_with_dtype_in_domain(dtype, Arc::clone(runtime.execution_domain()), "column::empty_with_runtime")
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
        if let Some(validity) = validity {
            if data.len() != validity.len() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "data and validity slices must have the same length",
                )
                .with_operation(operation)
                .with_context("data_len", data.len().to_string())
                .with_context("validity_len", validity.len().to_string()));
            }
        }

        let mut handle = ptr::null_mut();
        let size = usize_to_i32_len(data.len(), "data", operation)?;
        let null_mask = validity.map(validity_to_ffi_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        // SAFETY: data and optional null_mask are valid for the reported sizes.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    data.as_ptr() as *const c_void,
                    size,
                    null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                    null_count,
                    &exec_options,
                    &mut handle,
                )
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
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
            return Err(Error::new(ErrorKind::TypeError, "column type does not match requested type")
                .with_operation("column::device_slice")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
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
        Ok(ColumnView {
            handle: NonNull::new(view_handle).expect("null handle on success"),
            domain: Arc::clone(&self.domain),
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn payload_size_bytes(&self) -> Result<usize> {
        let mut out = 0usize;
        let exec_options = self.domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_payload_size_ex(self.handle.as_ptr(), &exec_options, &mut out) },
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

    pub(crate) fn into_raw_parts(self) -> (*mut c_void, Option<Arc<dyn Any + Send + Sync>>) {
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
    pub(crate) unsafe fn from_raw_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
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
        handle: *mut c_void,
        domain: Arc<ExecutionDomain>,
        retained_owner: Arc<dyn Any + Send + Sync>,
    ) -> Self {
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
            return Err(Error::new(ErrorKind::InvalidOperation, "from_device_ptr only supports fixed-width types")
                .with_operation(operation)
                .with_context("len", len.to_string())
                .with_context("type_id", format!("{:?}", type_id).to_lowercase()));
        }

        if len > 0 && device_ptr.is_null() {
            return Err(Error::new(ErrorKind::NullError, "device_ptr is null but len > 0")
                .with_operation(operation)
                .with_context("len", len.to_string())
                .with_context("type_id", format!("{:?}", type_id).to_lowercase()));
        }

        let size = usize_to_i32_len(len, "data", operation)?;
        let output_domain = Arc::clone(runtime.execution_domain());
        let exec_options = output_domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        let mut handle = ptr::null_mut();
        // SAFETY: caller guarantees device_ptr is valid for `size` elements.
        // The _ex variant does NOT synchronize — caller must keep device_ptr
        // valid until the runtime's stream completes.
        error::check_status_in_with(
            unsafe {
                cudf_sys::cudf_column_create_from_device_data_ex(
                    type_id.to_raw(),
                    device_ptr,
                    size,
                    &exec_options,
                    &mut handle,
                )
            },
            operation,
            [("len", len.to_string()), ("type_id", format!("{:?}", type_id).to_lowercase())],
        )?;
        Ok(Self::from_handle_with_domain(handle, output_domain))
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
        if let Some(validity) = validity {
            if data.len() != validity.len() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "data and validity slices must have the same length",
                )
                .with_operation(operation)
                .with_context("data_len", data.len().to_string())
                .with_context("validity_len", validity.len().to_string()));
            }
        }

        let num_strings = usize_to_i32_len(data.len(), "data", operation)?;

        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr() as *const c_char).collect();
        let lengths: Vec<i32> =
            data.iter().map(|s| usize_to_i32_len(s.len(), "string_element", operation)).collect::<Result<Vec<_>>>()?;
        let null_mask = validity.map(validity_to_ffi_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");

        let mut handle = ptr::null_mut();
        // SAFETY: ptrs and lengths are derived from valid &str slices and have
        // num_strings elements. No null mask (ptr::null(), 0).
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_strings_ex(
                    ptrs.as_ptr(),
                    lengths.as_ptr(),
                    num_strings,
                    null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                    null_count,
                    &exec_options,
                    &mut handle,
                )
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
    }

    pub(crate) fn nulls_in_domain(
        dtype: DataType,
        len: usize,
        domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let size = usize_to_i32_len(len, "data", operation)?;
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
                let elem_size = type_id.size_in_bytes().map_err(|e| e.with_operation(operation))?.ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        format!("{operation} does not support variable-width type {:?}", type_id),
                    )
                    .with_operation(operation)
                    .with_context("type_id", format!("{:?}", type_id).to_lowercase())
                })?;

                let data_len_bytes = len.checked_mul(elem_size as usize).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "column byte size overflow")
                        .with_operation(operation)
                        .with_context("len", len.to_string())
                        .with_context("elem_size_bytes", elem_size.to_string())
                })?;

                let data = vec![0u8; data_len_bytes];
                let validity = vec![false; len];
                let null_mask = validity_to_ffi_bitmask(&validity);
                let null_count = size;
                let exec_options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");

                let mut handle = ptr::null_mut();
                // SAFETY: data is zeroed bytes of the correct size, null_mask marks
                // all elements null. type_id.to_raw() is valid for fixed-width types.
                error::check_status_in_with(
                    unsafe {
                        cudf_sys::cudf_column_create_numeric_ex(
                            type_id.to_raw(),
                            data.as_ptr() as *const c_void,
                            size,
                            null_mask.as_ptr(),
                            null_count,
                            &exec_options,
                            &mut handle,
                        )
                    },
                    operation,
                    [("type_id", format!("{:?}", type_id).to_lowercase()), ("len", len.to_string())],
                )?;
                Ok(Self::from_handle_with_domain(handle, Arc::clone(domain)))
            },
        }?;

        if !dtype.id().is_decimal() || dtype.scale() == 0 {
            return Ok(raw);
        }

        let raw_view = raw.view()?;
        unary::cast(&raw_view, &dtype).map_err(|error| error.with_operation(operation))
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
        let size_i32 = i32::try_from(size).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "size exceeds i32::MAX")
                .with_operation("column::from_sequence_with_runtime")
        })?;
        let init_scalar = runtime.scalar(init)?;
        let step_scalar = runtime.scalar(step)?;
        let mut handle = ptr::null_mut();
        let exec_options =
            runtime.execution_domain().explicit_exec_options_raw().expect("execution domain is always explicit");
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_sequence_ex(
                    init_scalar.raw_handle(),
                    step_scalar.raw_handle(),
                    size_i32,
                    &exec_options,
                    &mut handle,
                )
            },
            "column::from_sequence_with_runtime",
        )?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(runtime.execution_domain())))
    }

    pub(crate) fn from_sequence_in_domain<T: CudfType>(
        init: T,
        step: T,
        size: usize,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let size_i32 = i32::try_from(size)
            .map_err(|_| Error::new(ErrorKind::InvalidOperation, "size exceeds i32::MAX").with_operation(operation))?;
        let init_scalar = Scalar::from_value_in_domain(init, &domain)?;
        let step_scalar = Scalar::from_value_in_domain(step, &domain)?;
        let mut handle = ptr::null_mut();
        let exec_options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_sequence_ex(
                    init_scalar.raw_handle(),
                    step_scalar.raw_handle(),
                    size_i32,
                    &exec_options,
                    &mut handle,
                )
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
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
        let size_i32 = i32::try_from(size).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "size exceeds i32::MAX").with_operation("column::from_scalar")
        })?;
        let mut handle = ptr::null_mut();
        let exec_options = scalar.domain.explicit_exec_options_raw().expect("execution domain is always explicit");
        // SAFETY: scalar.raw_handle() is valid (Scalar maintains NonNull invariant).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_from_scalar_ex(scalar.raw_handle(), size_i32, &mut handle, &exec_options) },
            "column::from_scalar",
        )?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(&scalar.domain)))
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
        self.domain = ExecutionDomain::debug_explicit(identity)?;
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Zero-copy device buffer adoption
// ---------------------------------------------------------------------------

fn require_runtime_matches_buffer_device(
    runtime: &Runtime,
    device_snapshot: i32,
    operation: &'static str,
) -> Result<Arc<ExecutionDomain>> {
    let runtime_device = runtime.device();
    if runtime_device.ordinal() as i32 != device_snapshot {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "device buffer belongs to a different CUDA device than the target runtime",
        )
        .with_operation(operation)
        .with_context("runtime_device", runtime_device.ordinal().to_string())
        .with_context("buffer_device", device_snapshot.to_string()));
    }

    Ok(Arc::clone(runtime.execution_domain()))
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

        let elem_size = type_id.size_in_bytes().map_err(|e| e.with_operation(operation))?.ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "from_device_buffer only supports fixed-width types")
                .with_operation(operation)
                .with_context("type_id", format!("{:?}", type_id).to_lowercase())
        })?;

        let row_count_i32 = usize_to_i32_len(row_count, "row_count", operation)?;

        let expected_bytes = row_count * (elem_size as usize);
        let actual_bytes = buffer.size_bytes().map_err(|e| e.with_operation(operation))?;
        if actual_bytes != expected_bytes {
            return Err(Error::new(ErrorKind::InvalidOperation, "buffer size does not match row_count * element size")
                .with_operation(operation)
                .with_context("buffer_bytes", actual_bytes.to_string())
                .with_context("expected_bytes", expected_bytes.to_string())
                .with_context("row_count", row_count.to_string())
                .with_context("type_id", format!("{:?}", type_id).to_lowercase()));
        }

        let (raw_buf, device_snapshot) = buffer.into_raw_parts();
        let output_domain = match require_runtime_matches_buffer_device(runtime, device_snapshot, operation) {
            Ok(domain) => domain,
            Err(err) => {
                let _cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_buf, device_snapshot) };
                return Err(err);
            },
        };

        let mut handle = ptr::null_mut();
        let status =
            unsafe { cudf_sys::cudf_column_adopt_device_buffer(type_id.to_raw(), raw_buf, row_count_i32, &mut handle) };

        match error::check_status_in(status, operation) {
            Ok(()) => Ok(Self::from_handle_with_domain(handle, output_domain)),
            Err(err) => {
                let _cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_buf, device_snapshot) };
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

        let elem_size = type_id.size_in_bytes().map_err(|e| e.with_operation(operation))?.ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "from_device_buffer_nullable only supports fixed-width types")
                .with_operation(operation)
                .with_context("type_id", format!("{:?}", type_id).to_lowercase())
        })?;

        let row_count_i32 = usize_to_i32_len(row_count, "row_count", operation)?;

        let null_count_i32 = i32::try_from(null_count).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "null_count exceeds i32::MAX")
                .with_operation(operation)
                .with_context("null_count", null_count.to_string())
        })?;

        if null_count > row_count {
            return Err(Error::new(ErrorKind::InvalidOperation, "null_count exceeds row_count")
                .with_operation(operation)
                .with_context("null_count", null_count.to_string())
                .with_context("row_count", row_count.to_string()));
        }

        let expected_data_bytes = row_count * (elem_size as usize);
        let actual_data_bytes = data_buffer.size_bytes().map_err(|e| e.with_operation(operation))?;
        if actual_data_bytes != expected_data_bytes {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "data buffer size does not match row_count * element size",
            )
            .with_operation(operation)
            .with_context("buffer_bytes", actual_data_bytes.to_string())
            .with_context("expected_bytes", expected_data_bytes.to_string()));
        }

        let min_mask_bytes = cudf_null_mask_allocation_size_bytes(row_count);
        let actual_mask_bytes = null_mask_buffer.size_bytes().map_err(|e| e.with_operation(operation))?;
        if actual_mask_bytes < min_mask_bytes {
            return Err(Error::new(ErrorKind::InvalidOperation, "null mask buffer too small")
                .with_operation(operation)
                .with_context("mask_bytes", actual_mask_bytes.to_string())
                .with_context("min_required_bytes", min_mask_bytes.to_string())
                .with_context("row_count", row_count.to_string()));
        }

        let data_device_snapshot = data_buffer.device_snapshot();
        let mask_device_snapshot = null_mask_buffer.device_snapshot();
        if data_device_snapshot != mask_device_snapshot {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "data buffer and null mask buffer must belong to the same CUDA device",
            )
            .with_operation(operation)
            .with_context("data_device", data_device_snapshot.to_string())
            .with_context("mask_device", mask_device_snapshot.to_string()));
        }

        let (raw_data, data_device_snapshot) = data_buffer.into_raw_parts();
        let (raw_mask, mask_device_snapshot) = null_mask_buffer.into_raw_parts();
        let output_domain = match require_runtime_matches_buffer_device(runtime, data_device_snapshot, operation) {
            Ok(domain) => domain,
            Err(err) => {
                let _data_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_data, data_device_snapshot) };
                let _mask_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_mask, mask_device_snapshot) };
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
            Ok(()) => Ok(Self::from_handle_with_domain(handle, output_domain)),
            Err(err) => {
                let _data_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_data, data_device_snapshot) };
                let _mask_cleanup = unsafe { super::DeviceBuffer::from_raw_parts(raw_mask, mask_device_snapshot) };
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
            let out_type = binary::infer_binary_output_type(lv.data_type()?.id(), rv.data_type()?.id(), op)?;
            binary::binary_operation_vv(&lv, &rv, op, &out_type)
        })()
        .map_err(|e| e.with_operation("column::binary_op").with_context("operator", format!("{:?}", op).to_lowercase()))
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
            e.with_operation("column::binary_op_with_type")
                .with_context("operator", format!("{:?}", op).to_lowercase())
                .with_context("output_type_id", format!("{:?}", output_type.id()).to_lowercase())
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
        self.binary_op(rhs, binary::BinaryOperator::Add).map_err(|e| e.with_operation("column::add"))
    }

    /// Element-wise subtraction: `self - rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn sub(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Sub).map_err(|e| e.with_operation("column::sub"))
    }

    /// Element-wise multiplication: `self * rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn mul(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mul).map_err(|e| e.with_operation("column::mul"))
    }

    /// Element-wise division: `self / rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Div).map_err(|e| e.with_operation("column::div"))
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
        .map_err(|e| e.with_operation("column::true_div").with_context("operator", "true_div"))
    }

    /// Element-wise floor division.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn floor_div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::FloorDiv).map_err(|e| e.with_operation("column::floor_div"))
    }

    /// Element-wise remainder: `self % rhs`.
    ///
    /// This is the preferred alias for [`Column::modulo`] to match the
    /// higher-level `Series::rem` naming.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn rem(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mod).map_err(|e| e.with_operation("column::rem"))
    }

    /// Element-wise modulo: `self % rhs`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn modulo(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mod).map_err(|e| e.with_operation("column::modulo"))
    }

    // -- Comparison methods ---------------------------------------------------

    /// Element-wise equal: `self == rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn eq(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Equal).map_err(|e| e.with_operation("column::eq"))
    }

    /// Element-wise not-equal: `self != rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ne(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.with_operation("column::ne"))
    }

    /// Element-wise less-than: `self < rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn lt(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Less).map_err(|e| e.with_operation("column::lt"))
    }

    /// Element-wise greater-than: `self > rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn gt(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Greater).map_err(|e| e.with_operation("column::gt"))
    }

    /// Element-wise less-or-equal: `self <= rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn le(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.with_operation("column::le"))
    }

    /// Element-wise greater-or-equal: `self >= rhs`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ge(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::GreaterEqual).map_err(|e| e.with_operation("column::ge"))
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
            let out_type = binary::infer_binary_output_type(lv.data_type()?.id(), rhs.data_type()?.id(), op)?;
            binary::binary_operation_vs(&lv, rhs, op, &out_type)
        })()
        .map(|col| col.with_execution_domain(Arc::clone(&self.domain)))
        .map_err(|e| {
            e.with_operation("column::binary_op_scalar").with_context("operator", format!("{:?}", op).to_lowercase())
        })
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
            e.with_operation("column::binary_op_scalar_with_type")
                .with_context("operator", format!("{:?}", op).to_lowercase())
                .with_context("output_type_id", format!("{:?}", output_type.id()).to_lowercase())
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
        self.binary_op_scalar(rhs, binary::BinaryOperator::Add).map_err(|e| e.with_operation("column::add_scalar"))
    }

    /// Element-wise: `self - scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn sub_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Sub).map_err(|e| e.with_operation("column::sub_scalar"))
    }

    /// Element-wise: `self * scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn mul_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Mul).map_err(|e| e.with_operation("column::mul_scalar"))
    }

    /// Element-wise: `self / scalar`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new output column on the device.
    pub fn div_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Div).map_err(|e| e.with_operation("column::div_scalar"))
    }

    /// Element-wise: `self == scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn eq_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Equal).map_err(|e| e.with_operation("column::eq_scalar"))
    }

    /// Element-wise: `self != scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ne_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.with_operation("column::ne_scalar"))
    }

    /// Element-wise: `self < scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn lt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Less).map_err(|e| e.with_operation("column::lt_scalar"))
    }

    /// Element-wise: `self > scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn gt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Greater).map_err(|e| e.with_operation("column::gt_scalar"))
    }

    /// Element-wise: `self <= scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn le_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.with_operation("column::le_scalar"))
    }

    /// Element-wise: `self >= scalar`. Returns Bool8 column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device.
    pub fn ge_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::GreaterEqual)
            .map_err(|e| e.with_operation("column::ge_scalar"))
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once, so the handle is destroyed exactly once.
        unsafe {
            cudf_sys::cudf_column_destroy(self.handle.as_ptr());
        }
    }
}
