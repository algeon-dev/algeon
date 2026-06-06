//! GPU column types: owned `Column` and borrowed `ColumnView`.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).

use super::view::ColumnView;
use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, TypeId, count_nulls, validity_to_bitmask},
    },
    ops::binary,
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
}

// SAFETY: Column owns a GPU device buffer via an opaque C++ handle. The handle
// is not thread-local. Send is safe because ownership transfer between threads is
// supported by the CUDA runtime. Sync is safe because all methods that observe state
// (view(), len(), data_type()) create independent FFI view handles.
unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    fn from_handle_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self { handle: NonNull::new(handle).expect("null column handle"), domain }
    }

    pub(crate) fn copy_into_domain(&self, domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.explicit_exec_options_raw();
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe { cudf_sys::cudf_column_copy_ex(self.handle.as_ptr(), options, &mut handle) },
                None => unsafe { cudf_sys::cudf_column_copy(self.handle.as_ptr(), &mut handle) },
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain))
    }

    /// Create an empty column of the given type.
    pub fn empty(type_id: TypeId) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: type_id.to_raw() produces a valid cudf_type_id_t. handle is
        // initialized to null and set by the C shim on success.
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_create_empty(type_id.to_raw(), &mut handle) },
            "column::empty",
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
    }

    /// Create an empty column of the given type inside the given explicit
    /// runtime.
    ///
    /// The resulting column carries the runtime's retained provenance so it
    /// can participate in explicit-provenance pipelines without triggering
    /// mixed-provenance rejection. No device memory is allocated for the
    /// column data itself.
    pub fn empty_with_runtime(runtime: &Runtime, type_id: TypeId) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_create_empty(type_id.to_raw(), &mut handle) },
            "column::empty_with_runtime",
        )?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(runtime.execution_domain())))
    }

    /// Create a column from a host slice of fixed-width elements.
    ///
    /// # Example
    /// ```no_run
    /// use cudf::Column;
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// ```
    pub fn from_slice<T: CudfType>(data: &[T]) -> Result<Self> {
        Self::from_slice_impl(data, None, None, "column::from_slice")
    }

    /// Create a column from a host slice inside the given explicit runtime.
    pub fn from_slice_with_runtime<T: CudfType>(runtime: &Runtime, data: &[T]) -> Result<Self> {
        Self::from_slice_impl(
            data,
            None,
            Some(Arc::clone(runtime.execution_domain())),
            "column::from_slice_with_runtime",
        )
    }

    fn from_slice_impl<T: CudfType>(
        data: &[T],
        validity: Option<&[bool]>,
        domain: Option<Arc<ExecutionDomain>>,
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
        let null_mask = validity.map(validity_to_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());
        // SAFETY: data and optional null_mask are valid for the reported sizes.
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe {
                    cudf_sys::cudf_column_create_numeric_ex(
                        T::TYPE_ID.to_raw(),
                        data.as_ptr() as *const c_void,
                        size,
                        null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                        null_count,
                        options,
                        &mut handle,
                    )
                },
                None => unsafe {
                    cudf_sys::cudf_column_create_numeric(
                        T::TYPE_ID.to_raw(),
                        data.as_ptr() as *const c_void,
                        size,
                        null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                        null_count,
                        &mut handle,
                    )
                },
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain.unwrap_or_else(ExecutionDomain::ambient_default)))
    }

    /// Create a nullable column from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn from_slice_nullable<T: CudfType>(data: &[T], validity: &[bool]) -> Result<Self> {
        Self::from_slice_impl(data, Some(validity), None, "column::from_slice_nullable")
    }

    /// Create a nullable column from host data inside the given explicit runtime.
    pub fn from_slice_nullable_with_runtime<T: CudfType>(
        runtime: &Runtime,
        data: &[T],
        validity: &[bool],
    ) -> Result<Self> {
        Self::from_slice_impl(
            data,
            Some(validity),
            Some(Arc::clone(runtime.execution_domain())),
            "column::from_slice_nullable_with_runtime",
        )
    }

    /// Deep-copy this column.
    pub fn copy(&self) -> Result<Self> {
        self.copy_into_domain(Arc::clone(&self.domain), "column::copy")
    }

    /// Create an immutable view of this column.
    ///
    /// The returned `ColumnView` borrows from `self` and cannot outlive it.
    pub fn view(&self) -> Result<ColumnView<'_>> {
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
        let exec_options = self.domain.explicit_exec_options_raw();
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe {
                    cudf_sys::cudf_column_payload_size_ex(self.handle.as_ptr(), options, &mut out)
                },
                None => unsafe { cudf_sys::cudf_column_payload_size(self.handle.as_ptr(), &mut out) },
            },
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

    /// Returns the raw FFI handle. For internal use by `Table::from_columns`.
    pub(crate) fn into_raw(self) -> *mut c_void {
        let handle = self.handle.as_ptr();
        std::mem::forget(self); // Don't drop — ownership transferred.
        handle
    }

    /// Construct a `Column` from a raw FFI handle, taking ownership.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw(handle: *mut c_void) -> Self {
        Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default())
    }

    /// Construct a `Column` from a raw FFI handle and an explicit retained domain.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self::from_handle_with_domain(handle, domain)
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
    pub unsafe fn from_device_ptr(device_ptr: *const c_void, len: usize, type_id: TypeId) -> Result<Self> {
        if type_id.size_in_bytes()?.is_none() {
            return Err(Error::new(ErrorKind::InvalidOperation, "from_device_ptr only supports fixed-width types")
                .with_operation("column::from_device_ptr")
                .with_context("len", len.to_string())
                .with_context("type_id", format!("{:?}", type_id).to_lowercase()));
        }

        if len > 0 && device_ptr.is_null() {
            return Err(Error::new(ErrorKind::NullError, "device_ptr is null but len > 0")
                .with_operation("column::from_device_ptr")
                .with_context("len", len.to_string())
                .with_context("type_id", format!("{:?}", type_id).to_lowercase()));
        }

        let size = usize_to_i32_len(len, "data", "column::from_device_ptr")?;
        let mut handle = ptr::null_mut();
        // SAFETY: caller guarantees device_ptr is valid for `size` elements.
        // Null-ptr + zero-len case is checked above.
        error::check_status_in_with(
            unsafe { cudf_sys::cudf_column_create_from_device_data(type_id.to_raw(), device_ptr, size, &mut handle) },
            "column::from_device_ptr",
            [("len", len.to_string()), ("type_id", format!("{:?}", type_id).to_lowercase())],
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
    }

    /// Create a column by copying from a borrowed device pointer inside the
    /// given explicit runtime.
    ///
    /// Uses `cudf_column_create_from_device_data_ex` which routes the copy
    /// through the runtime's stream and memory resource. Unlike
    /// [`from_device_ptr`](Self::from_device_ptr), **this variant does NOT
    /// synchronize before returning**. The caller must keep `device_ptr`
    /// valid until the runtime's stream has completed the copy — e.g. by
    /// calling [`Runtime::synchronize`].
    ///
    /// Only fixed-width types are supported.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to a valid device allocation containing at
    ///   least `len` elements of the type identified by `type_id`.
    /// - The pointer must remain valid until the runtime's stream has
    ///   completed the copy.
    pub unsafe fn from_device_ptr_with_runtime(
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
        let exec_options = output_domain.explicit_exec_options_raw();
        let opts_ptr = exec_options.as_ref().map_or(ptr::null(), |o| o as *const _);
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
                    opts_ptr,
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
    /// use cudf::Column;
    /// let col = Column::from_strings(&["hello", "world"]).unwrap();
    /// ```
    pub fn from_strings(data: &[&str]) -> Result<Self> {
        Self::from_strings_impl(data, None, None, "column::from_strings")
    }

    /// Create a STRING type column from host string data inside the given explicit runtime.
    pub fn from_strings_with_runtime(runtime: &Runtime, data: &[&str]) -> Result<Self> {
        Self::from_strings_impl(
            data,
            None,
            Some(Arc::clone(runtime.execution_domain())),
            "column::from_strings_with_runtime",
        )
    }

    fn from_strings_impl(
        data: &[&str],
        validity: Option<&[bool]>,
        domain: Option<Arc<ExecutionDomain>>,
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
        let null_mask = validity.map(validity_to_bitmask);
        let null_count = validity.map(count_nulls).unwrap_or(0);
        let exec_options = domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());

        let mut handle = ptr::null_mut();
        // SAFETY: ptrs and lengths are derived from valid &str slices and have
        // num_strings elements. No null mask (ptr::null(), 0).
        error::check_status_in(
            match exec_options.as_ref() {
                Some(exec_options) => unsafe {
                    cudf_sys::cudf_column_create_strings_ex(
                        ptrs.as_ptr(),
                        lengths.as_ptr(),
                        num_strings,
                        null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                        null_count,
                        exec_options,
                        &mut handle,
                    )
                },
                None => unsafe {
                    cudf_sys::cudf_column_create_strings(
                        ptrs.as_ptr(),
                        lengths.as_ptr(),
                        num_strings,
                        null_mask.as_ref().map_or(ptr::null(), |mask| mask.as_ptr()),
                        null_count,
                        &mut handle,
                    )
                },
            },
            operation,
        )?;
        Ok(Self::from_handle_with_domain(handle, domain.unwrap_or_else(ExecutionDomain::ambient_default)))
    }

    /// Create an all-null column of the given type and length.
    ///
    /// Every element is null (invalid). For `len == 0` this is equivalent
    /// to [`Column::empty`].
    pub fn nulls(type_id: TypeId, len: usize) -> Result<Self> {
        let size = usize_to_i32_len(len, "data", "column::nulls")?;
        if len == 0 {
            return Self::empty(type_id);
        }

        match type_id {
            TypeId::String => {
                let dummy: Vec<&str> = vec![""; len];
                let validity = vec![false; len];
                Self::from_strings_nullable(&dummy, &validity)
            },
            _ => {
                let elem_size =
                    type_id.size_in_bytes().map_err(|e| e.with_operation("column::nulls"))?.ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidOperation,
                            format!("column::nulls does not support variable-width type {:?}", type_id),
                        )
                        .with_operation("column::nulls")
                        .with_context("type_id", format!("{:?}", type_id).to_lowercase())
                    })?;

                let data_len_bytes = len.checked_mul(elem_size as usize).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "column::nulls byte size overflow")
                        .with_operation("column::nulls")
                        .with_context("len", len.to_string())
                        .with_context("elem_size_bytes", elem_size.to_string())
                })?;

                let data = vec![0u8; data_len_bytes];
                let validity = vec![false; len];
                let null_mask = validity_to_bitmask(&validity);
                let null_count = size;

                let mut handle = ptr::null_mut();
                // SAFETY: data is zeroed bytes of the correct size, null_mask marks
                // all elements null. type_id.to_raw() is valid for fixed-width types.
                error::check_status_in_with(
                    unsafe {
                        cudf_sys::cudf_column_create_numeric(
                            type_id.to_raw(),
                            data.as_ptr() as *const c_void,
                            size,
                            null_mask.as_ptr(),
                            null_count,
                            &mut handle,
                        )
                    },
                    "column::nulls",
                    [("type_id", format!("{:?}", type_id).to_lowercase()), ("len", len.to_string())],
                )?;
                Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
            },
        }
    }

    /// Create an all-null column of the given type and length inside the given
    /// explicit runtime.
    ///
    /// The resulting column carries the runtime's retained provenance so it can
    /// participate in explicit-provenance pipelines without triggering
    /// mixed-provenance rejection.
    pub fn nulls_with_runtime(runtime: &Runtime, type_id: TypeId, len: usize) -> Result<Self> {
        let _size = usize_to_i32_len(len, "data", "column::nulls_with_runtime")?;
        if len == 0 {
            return Self::empty_with_runtime(runtime, type_id);
        }

        match type_id {
            TypeId::String => {
                let dummy: Vec<&str> = vec![""; len];
                let validity = vec![false; len];
                Self::from_strings_nullable_with_runtime(runtime, &dummy, &validity)
            },
            _ => {
                // Build an ambient-domain null column, then stamp the runtime domain.
                let col = Self::nulls(type_id, len)?;
                Ok(col.with_execution_domain(Arc::clone(runtime.execution_domain())))
            },
        }
    }

    /// Create a column by repeating a scalar value `size` times inside the
    /// given explicit runtime.
    ///
    /// The resulting column carries the runtime's retained provenance.
    pub fn from_scalar_with_runtime(runtime: &Runtime, scalar: &Scalar, size: usize) -> Result<Self> {
        let col = Self::from_scalar(scalar, size)?;
        Ok(col.with_execution_domain(Arc::clone(runtime.execution_domain())))
    }

    /// Create a nullable STRING type column from host string data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    /// For null rows, the string content is ignored.
    pub fn from_strings_nullable(data: &[&str], validity: &[bool]) -> Result<Self> {
        Self::from_strings_impl(data, Some(validity), None, "column::from_strings_nullable")
    }

    /// Create a nullable STRING type column inside the given explicit runtime.
    pub fn from_strings_nullable_with_runtime(runtime: &Runtime, data: &[&str], validity: &[bool]) -> Result<Self> {
        Self::from_strings_impl(
            data,
            Some(validity),
            Some(Arc::clone(runtime.execution_domain())),
            "column::from_strings_nullable_with_runtime",
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
        let exec_options = scalar.domain.explicit_exec_options_raw();
        let opts_ptr = exec_options.as_ref().map_or(ptr::null(), |o| o as *const _);
        // SAFETY: scalar.raw_handle() is valid (Scalar maintains NonNull invariant).
        // opts_ptr is either null (ambient default) or a valid exec_options pointer.
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_from_scalar_ex(scalar.raw_handle(), size_i32, &mut handle, opts_ptr) },
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

    #[doc(hidden)]
    pub fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }

    #[doc(hidden)]
    pub fn __debug_retained_provenance_kind(&self) -> &'static str {
        self.__debug_execution_domain_kind()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }

    #[doc(hidden)]
    pub fn __debug_retained_provenance_id(&self) -> u64 {
        self.__debug_execution_domain_id()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }

    #[doc(hidden)]
    pub fn __debug_retained_provenance_stream_value(&self) -> usize {
        self.__debug_execution_domain_stream_value()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_mr_value(&self) -> usize {
        self.domain.debug_mr_value()
    }

    #[doc(hidden)]
    pub fn __debug_retained_provenance_mr_value(&self) -> usize {
        self.__debug_execution_domain_mr_value()
    }

    #[doc(hidden)]
    pub fn __debug_rebind_execution_domain_explicit(mut self, identity: u64) -> Result<Self> {
        self.domain = ExecutionDomain::debug_explicit(identity)?;
        Ok(self)
    }

    #[doc(hidden)]
    pub fn __debug_rebind_retained_provenance_explicit(self, identity: u64) -> Result<Self> {
        self.__debug_rebind_execution_domain_explicit(identity)
    }
}

// ---------------------------------------------------------------------------
// Binary operations on Column
// ---------------------------------------------------------------------------

impl Column {
    /// Generic binary operation with inferred output type.
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
    /// use cudf::Column;
    ///
    /// let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    /// let sum = a.add(&b).unwrap();
    /// // sum: [11, 22, 33]
    ///
    /// // Also available via std::ops: &a + &b
    /// ```
    pub fn add(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Add).map_err(|e| e.with_operation("column::add"))
    }

    /// Element-wise subtraction: `self - rhs`. Output type inferred.
    pub fn sub(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Sub).map_err(|e| e.with_operation("column::sub"))
    }

    /// Element-wise multiplication: `self * rhs`. Output type inferred.
    pub fn mul(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mul).map_err(|e| e.with_operation("column::mul"))
    }

    /// Element-wise division: `self / rhs`. Output type inferred.
    pub fn div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Div).map_err(|e| e.with_operation("column::div"))
    }

    /// Element-wise true division (always float output).
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
    pub fn floor_div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::FloorDiv).map_err(|e| e.with_operation("column::floor_div"))
    }

    /// Element-wise modulo: `self % rhs`.
    pub fn modulo(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Mod).map_err(|e| e.with_operation("column::modulo"))
    }

    // -- Comparison methods ---------------------------------------------------

    /// Element-wise equal: `self == rhs`. Returns Bool8 column.
    pub fn eq_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Equal).map_err(|e| e.with_operation("column::eq_col"))
    }

    /// Element-wise not-equal: `self != rhs`. Returns Bool8 column.
    pub fn ne_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.with_operation("column::ne_col"))
    }

    /// Element-wise less-than: `self < rhs`. Returns Bool8 column.
    pub fn lt_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Less).map_err(|e| e.with_operation("column::lt_col"))
    }

    /// Element-wise greater-than: `self > rhs`. Returns Bool8 column.
    pub fn gt_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::Greater).map_err(|e| e.with_operation("column::gt_col"))
    }

    /// Element-wise less-or-equal: `self <= rhs`. Returns Bool8 column.
    pub fn le_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.with_operation("column::le_col"))
    }

    /// Element-wise greater-or-equal: `self >= rhs`. Returns Bool8 column.
    pub fn ge_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, binary::BinaryOperator::GreaterEqual).map_err(|e| e.with_operation("column::ge_col"))
    }

    // -- Scalar operations ----------------------------------------------------

    /// Element-wise: `self op scalar`. Output type inferred.
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
    /// use cudf::{Column, Scalar};
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let offset = Scalar::from_value(100i32).unwrap();
    /// let result = col.add_scalar(&offset).unwrap();
    /// // result: [101, 102, 103]
    ///
    /// // Also available via std::ops: &col + 100i32
    /// ```
    pub fn add_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Add).map_err(|e| e.with_operation("column::add_scalar"))
    }

    /// Element-wise: `self - scalar`.
    pub fn sub_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Sub).map_err(|e| e.with_operation("column::sub_scalar"))
    }

    /// Element-wise: `self * scalar`.
    pub fn mul_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Mul).map_err(|e| e.with_operation("column::mul_scalar"))
    }

    /// Element-wise: `self / scalar`.
    pub fn div_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Div).map_err(|e| e.with_operation("column::div_scalar"))
    }

    /// Element-wise: `self == scalar`. Returns Bool8 column.
    pub fn eq_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Equal).map_err(|e| e.with_operation("column::eq_scalar"))
    }

    /// Element-wise: `self != scalar`. Returns Bool8 column.
    pub fn ne_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::NotEqual).map_err(|e| e.with_operation("column::ne_scalar"))
    }

    /// Element-wise: `self < scalar`. Returns Bool8 column.
    pub fn lt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Less).map_err(|e| e.with_operation("column::lt_scalar"))
    }

    /// Element-wise: `self > scalar`. Returns Bool8 column.
    pub fn gt_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::Greater).map_err(|e| e.with_operation("column::gt_scalar"))
    }

    /// Element-wise: `self <= scalar`. Returns Bool8 column.
    pub fn le_scalar(&self, rhs: &Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, binary::BinaryOperator::LessEqual).map_err(|e| e.with_operation("column::le_scalar"))
    }

    /// Element-wise: `self >= scalar`. Returns Bool8 column.
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
