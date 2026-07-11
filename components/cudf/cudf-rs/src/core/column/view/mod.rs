use std::{
    marker::PhantomData,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

mod errors;
mod testing_hooks;

use errors::{
    classify_column_view_metadata_error, classify_string_view_contract_error,
    foreign_column_view_metadata_contract_error, foreign_string_view_contract_error,
};
pub(crate) use testing_hooks::{
    column_view_to_fixed_width_host_bytes_for_test, column_view_to_raw_bytes_for_test,
    synthetic_string_view_without_children, trigger_fixed_width_host_byte_range_overflow_for_test,
};

use super::{convert::copy_view_to_owned, device_slice::DeviceSlice, owned::Column};
use crate::{
    core::{
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_EXPECTED_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID,
            FACT_TYPE_ID, Result,
        },
        types::{CudfType, DataType, TypeId},
    },
    ops::search,
    runtime::execution_domain::ExecutionDomain,
};

fn fixed_width_host_byte_range_overflow_error(
    operation: &'static str,
    row_count: usize,
    elem_size_bytes: usize,
    type_id: TypeId,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, "host byte range overflowed")
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("resource", "host_byte_range")
        .with_detail("reason", "size_overflowed_usize")
        .with_detail(
            "remedy",
            "reduce row_count or choose a smaller fixed-width type so the active host byte range fits within usize",
        )
        .with_detail("row_count", row_count.to_string())
        .with_detail("elem_size_bytes", elem_size_bytes.to_string())
}

/// A non-owning, immutable view of a GPU column.
///
/// The lifetime `'a` ties this view to the owning `Column` (or `Table`),
/// preventing use-after-free at compile time.
pub struct ColumnView<'a> {
    pub(crate) handle: NonNull<c_void>,
    pub(crate) domain: Arc<ExecutionDomain>,
    pub(crate) _lifetime: PhantomData<&'a ()>,
}

impl<'a> ColumnView<'a> {
    /// Create a `ColumnView` from a raw FFI handle plus retained provenance.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_view_handle` and must remain
    /// valid for lifetime `'a`.
    pub(crate) unsafe fn from_raw_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Ok(Self {
            handle: error::require_non_null(handle, "column_view::from_raw_with_domain", "column view")?,
            domain,
            _lifetime: PhantomData,
        })
    }

    /// Returns the raw FFI handle pointer. For internal use by operations
    /// that need to pass column views to C shim functions.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    pub(crate) fn slice_with_operation(&self, offset: i32, length: i32, operation: &'static str) -> Result<Self> {
        let exec_options = self.domain.exec_options_raw();
        let mut sliced = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_slice_ex(self.handle.as_ptr(), offset, length, &exec_options, &mut sliced)
            },
            operation,
        )?;

        Ok(Self {
            handle: error::require_non_null(sliced, operation, "sliced column view")?,
            domain: Arc::clone(&self.domain),
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn duplicate_with_operation(&self, operation: &'static str) -> Result<Self> {
        let row_count = self.len()?;
        let len = i32::try_from(row_count)
            .map_err(|_| error::range_limit_error(operation, "row_count", "i32::MAX", row_count))?;

        self.slice_with_operation(0, len, operation)
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

    fn synchronize_host_visibility(&self, operation: &'static str) -> Result<()> {
        self.domain.synchronize_for_host_visibility(operation)
    }

    /// Returns the data type of this column.
    pub fn data_type(&self) -> Result<DataType> {
        let mut raw_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
        let mut scale = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_data_type(self.handle.as_ptr(), &mut raw_id, &mut scale) },
            "column_view::data_type",
        )
        .map_err(classify_column_view_metadata_error)?;
        DataType::from_raw_parts(raw_id, scale)
    }

    /// Returns the number of elements.
    pub fn len(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_size(self.handle.as_ptr(), &mut out) },
            "column_view::len",
        )
        .map_err(classify_column_view_metadata_error)?;
        usize::try_from(out).map_err(|_| {
            foreign_column_view_metadata_contract_error("column_view::len", "FFI returned negative column size")
                .with_detail("raw_size", out.to_string())
        })
    }

    /// Returns `true` if the column has zero elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of null (invalid) elements.
    ///
    /// This is a metadata read when libcudf already carries an exact null
    /// count. For borrowed views whose null count is unknown, libcudf computes
    /// the count on this view's retained execution stream before returning.
    pub fn null_count(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_null_count(self.handle.as_ptr(), &mut out) },
            "column_view::null_count",
        )
        .map_err(classify_column_view_metadata_error)?;
        if out < 0 {
            let exec_options = self.domain.exec_options_raw();
            // SAFETY: self.handle is valid (NonNull from construction/from_raw).
            error::check_status_in(
                unsafe { cudf_sys::cudf_column_view_null_count_ex(self.handle.as_ptr(), &exec_options, &mut out) },
                "column_view::null_count",
            )
            .map_err(classify_column_view_metadata_error)?;
        }
        usize::try_from(out).map_err(|_| {
            foreign_column_view_metadata_contract_error("column_view::null_count", "column null count must be >= 0")
                .with_detail("null_count", out.to_string())
        })
    }

    /// Returns `true` if this column has a null mask allocated.
    pub fn nullable(&self) -> Result<bool> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_nullable(self.handle.as_ptr(), &mut out) },
            "column_view::nullable",
        )
        .map_err(classify_column_view_metadata_error)?;
        Ok(out != 0)
    }

    /// Returns the number of child columns (for nested types).
    pub fn num_children(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_num_children(self.handle.as_ptr(), &mut out) },
            "column_view::num_children",
        )
        .map_err(classify_column_view_metadata_error)?;
        usize::try_from(out).map_err(|_| {
            foreign_column_view_metadata_contract_error("column_view::num_children", "column child count must be >= 0")
                .with_detail("num_children", out.to_string())
        })
    }

    /// Get a child column view by index.
    pub fn child(&self, index: usize) -> Result<ColumnView<'a>> {
        let num_children = self.num_children()?;
        if index >= num_children {
            return Err(Error::new_with(
                ErrorCode::IndexOutOfBounds,
                "column_view::child",
                "column child index out of bounds",
            )
            .with_detail("child_index", index.to_string())
            .with_detail("num_children", num_children.to_string()));
        }

        let index = i32::try_from(index)
            .map_err(|_| error::range_limit_error("column_view::child", "child_index", "i32::MAX", index))?;

        let mut child_handle = ptr::null_mut();
        // SAFETY: self.handle is valid and the child index is validated above.
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_child(self.handle.as_ptr(), index, &mut child_handle) },
            "column_view::child",
        )?;
        Ok(ColumnView {
            handle: error::require_non_null(child_handle, "column_view::child", "column child view")?,
            domain: Arc::clone(&self.domain),
            _lifetime: PhantomData,
        })
    }

    /// Returns the logical child column view for a LIST column after applying
    /// the root list offset to the child range.
    pub(crate) fn list_sliced_child(&self) -> Result<ColumnView<'a>> {
        let mut child_handle = ptr::null_mut();
        let exec_options = self.domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_list_sliced_child_ex(self.handle.as_ptr(), &exec_options, &mut child_handle)
            },
            "column_view::list_sliced_child",
        )?;
        Ok(ColumnView {
            handle: error::require_non_null(child_handle, "column_view::list_sliced_child", "list child view")?,
            domain: Arc::clone(&self.domain),
            _lifetime: PhantomData,
        })
    }

    /// Returns the raw device (GPU) data pointer of this column view.
    ///
    /// This is a zero-copy operation — no data moves between GPU and host.
    /// The returned pointer points to device memory containing `len()` elements
    /// of `data_type()`.
    ///
    /// For `STRING` columns the data pointer is null. String raw-buffer access
    /// uses a separate offsets child plus string-specific chars helpers.
    ///
    /// # Note
    ///
    /// For sliced column views the pointer is to the **base** of the device
    /// buffer. Use [`offset()`](Self::offset) to compute the start of the
    /// active region.
    pub fn device_data_ptr(&self) -> Result<*const c_void> {
        let mut out: *const c_void = std::ptr::null();
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_data_ptr(self.handle.as_ptr(), &mut out) },
            "column_view::device_data_ptr",
        )?;
        Ok(out)
    }

    pub(crate) fn string_chars_ptr_and_size(&self) -> Result<(*const c_void, usize)> {
        let mut out_ptr: *const c_void = std::ptr::null();
        let mut out_size: usize = 0;
        let exec_options = self.domain.exec_options_raw();
        let status = unsafe {
            cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(
                self.handle.as_ptr(),
                &exec_options,
                &mut out_ptr,
                &mut out_size,
            )
        };
        error::check_status_in(status, "column_view::string_chars_ptr_and_size")
            .map_err(classify_string_view_contract_error)?;
        Ok((out_ptr, out_size))
    }

    fn validate_string_view_layout_for_host_materialization(&self, operation: &'static str) -> Result<()> {
        let num_children =
            self.num_children().map_err(|e| classify_string_view_contract_error(e).reanchor(operation))?;
        if num_children >= 1 {
            return Ok(());
        }

        Err(foreign_string_view_contract_error(
            "column_view::string_chars_ptr_and_size",
            "STRING column view must expose offsets and chars children",
        )
        .with_fact(FACT_EXPECTED_COUNT, 1usize)
        .with_fact(FACT_ACTUAL_COUNT, num_children)
        .reanchor(operation))
    }

    pub(crate) fn null_mask_ptr(&self) -> Result<Option<*const c_void>> {
        let mut out: *const c_void = std::ptr::null();
        let status = unsafe { cudf_sys::cudf_column_view_null_mask_ptr(self.handle.as_ptr(), &mut out) };
        if status == cudf_sys::CUDF_ERROR_NULL {
            return Ok(None);
        }
        error::check_status_in(status, "column_view::null_mask_ptr")?;
        Ok(Some(out))
    }

    /// Returns the element offset of this column view into its parent buffer.
    ///
    /// Returns 0 for non-sliced columns. For sliced views, the actual data
    /// starts at `device_data_ptr() + offset * element_size`.
    pub fn offset(&self) -> Result<i32> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_offset(self.handle.as_ptr(), &mut out) },
            "column_view::offset",
        )
        .map_err(classify_column_view_metadata_error)?;
        Ok(out)
    }

    /// Copy the column data to a host `Vec<T>`.
    ///
    /// The column's type must match `T::TYPE_ID`. Only valid for fixed-width types.
    ///
    /// # Synchronization
    /// This materializes device data into host memory and may synchronize the
    /// bound execution domain before returning.
    pub fn to_vec<T: CudfType>(&self) -> Result<Vec<T>> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "column_view::to_vec",
                "column type does not match requested type",
            )
            .with_fact(FACT_LEFT_TYPE_ID, dt.id())
            .with_fact(FACT_RIGHT_TYPE_ID, T::TYPE_ID));
        }

        let size = self.len()?;
        // SAFETY: `CudfType` is sealed to built-in primitives with valid
        // all-zero bit patterns.
        let mut buf = vec![unsafe { std::mem::zeroed::<T>() }; size];
        let buf_size = size * std::mem::size_of::<T>();

        self.synchronize_host_visibility("column_view::to_vec")?;

        // SAFETY: buf is a valid mutable buffer of buf_size bytes. self.handle is
        // valid. Type was checked above to match T::TYPE_ID.
        let exec_options = self.domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_to_host_ex(
                    self.handle.as_ptr(),
                    buf.as_mut_ptr() as *mut c_void,
                    buf_size,
                    &exec_options,
                )
            },
            "column_view::to_vec",
        )?;

        Ok(buf)
    }

    /// Copy fixed-width column data to host as `Vec<Option<T>>`, encoding
    /// null positions as `None`.
    ///
    /// Non-nullable columns return all `Some`. For nullable columns, this
    /// performs two separate device-to-host transfers (data buffer + null-mask
    /// buffer), then merges them on the host.
    ///
    /// For `bool`, libcudf stores values as raw bytes and does not guarantee a
    /// canonical value for null slots. This method materializes Bool8 through a
    /// byte buffer and only converts valid positions, avoiding invalid `bool`
    /// construction.
    ///
    /// # Synchronization
    /// This materializes device data and null-mask state into host memory and
    /// may synchronize the bound execution domain before returning.
    pub fn to_optional_vec<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        if T::TYPE_ID == TypeId::Bool8 {
            return self.to_optional_vec_bool::<T>();
        }

        let vals = self.to_vec::<T>()?;
        match self.to_host_validity()? {
            None => Ok(vals.into_iter().map(Some).collect()),
            Some(mask) => Ok(vals.into_iter().zip(mask).map(|(v, valid)| if valid { Some(v) } else { None }).collect()),
        }
    }

    /// Copy column data to a host `Vec<u8>`, one byte per element.
    ///
    /// Requires the column to be a 1-byte fixed-width type (Bool8, Int8, Uint8).
    /// Unlike `to_vec::<bool>()`, this never constructs a `bool` from raw device
    /// memory, avoiding UB when null slots contain non-canonical byte values.
    pub(crate) fn to_raw_bytes(&self) -> Result<Vec<u8>> {
        let dt = self.data_type()?;
        let is_one_byte = matches!(dt.id(), TypeId::Bool8 | TypeId::Int8 | TypeId::UInt8);
        if !is_one_byte {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                "column_view::to_raw_bytes",
                "to_raw_bytes requires a 1-byte type",
            )
            .with_fact(FACT_TYPE_ID, dt.id()));
        }

        let size = self.len()?;
        let mut buf = vec![0u8; size];

        self.synchronize_host_visibility("column_view::to_raw_bytes")?;

        let exec_options = self.domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_to_host_ex(
                    self.handle.as_ptr(),
                    buf.as_mut_ptr() as *mut c_void,
                    size,
                    &exec_options,
                )
            },
            "column_view::to_raw_bytes",
        )?;

        Ok(buf)
    }

    fn to_optional_vec_bool<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        let raw = self.to_raw_bytes()?;
        let bool_values: Vec<Option<bool>> = match self.to_host_validity()? {
            None => raw.into_iter().map(|b| Some(b != 0)).collect(),
            Some(mask) => raw.into_iter().zip(mask).map(|(b, valid)| if valid { Some(b != 0) } else { None }).collect(),
        };

        let boxed: Box<dyn std::any::Any> = Box::new(bool_values);
        boxed.downcast::<Vec<Option<T>>>().map(|typed| *typed).map_err(|_| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "column_view::to_optional_vec",
                "internal bool materialization helper requires T = bool",
            )
            .with_fact(FACT_TYPE_ID, T::TYPE_ID)
            .with_detail("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase())
        })
    }

    /// Copy any fixed-width column view to raw host bytes.
    ///
    /// The returned buffer covers only the active logical range of this view.
    pub(crate) fn to_fixed_width_host_bytes(&self) -> Result<Vec<u8>> {
        let dt = self.data_type()?;
        if self.offset()? != 0 {
            // Normalize sliced views to a zero-offset owned column first so the
            // resulting host bytes cover only the active logical range.
            let normalized = super::convert::copy_view_to_owned(self)?;
            return normalized.view()?.to_fixed_width_host_bytes();
        }

        let elem_size = usize::try_from(dt.id().size_in_bytes()?.ok_or_else(|| {
            Error::new_with(
                ErrorCode::UnsupportedDtype,
                "column_view::to_fixed_width_host_bytes",
                "to_fixed_width_host_bytes requires a fixed-width type",
            )
            .with_fact(FACT_TYPE_ID, dt.id())
        })?)
        .map_err(|_| {
            Error::new_with(
                ErrorCode::ForeignInvariantViolation,
                "column_view::to_fixed_width_host_bytes",
                "fixed-width element size must be >= 0",
            )
            .with_fact(FACT_TYPE_ID, dt.id())
        })?;

        let size = self.len()?;
        let buf_size = size.checked_mul(elem_size).ok_or_else(|| {
            fixed_width_host_byte_range_overflow_error(
                "column_view::to_fixed_width_host_bytes",
                size,
                elem_size,
                dt.id(),
            )
        })?;
        let mut buf = vec![0u8; buf_size];

        self.synchronize_host_visibility("column_view::to_fixed_width_host_bytes")?;

        let exec_options = self.domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_to_host_ex(
                    self.handle.as_ptr(),
                    buf.as_mut_ptr() as *mut c_void,
                    buf_size,
                    &exec_options,
                )
            },
            "column_view::to_fixed_width_host_bytes",
        )?;

        Ok(buf)
    }

    fn to_host_null_mask_bytes_with_operation(&self, operation: &'static str) -> Result<Option<Vec<u8>>> {
        if !self.nullable().map_err(|e| e.reanchor(operation))? {
            return Ok(None);
        }

        let size = self.len().map_err(|e| e.reanchor(operation))?;
        let mask_bytes = size.div_ceil(8);
        // Ensure we allocate at least the bitmask alignment size that cudf expects.
        // cudf pads to 64-byte boundaries, so allocate enough.
        let alloc_bytes = mask_bytes.div_ceil(64) * 64;
        let mut mask = vec![0u8; alloc_bytes];

        self.synchronize_host_visibility(operation)?;

        // SAFETY: mask buffer is large enough (padded to 64-byte boundaries).
        // self.handle is valid and nullable (checked above).
        let exec_options = self.domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_null_mask_to_host_ex(
                    self.handle.as_ptr(),
                    mask.as_mut_ptr(),
                    alloc_bytes,
                    &exec_options,
                )
            },
            operation,
        )?;

        mask.truncate(mask_bytes);
        Ok(Some(mask))
    }

    /// Copy the packed null mask for the active logical range to host bytes.
    ///
    /// Returns `None` if the column has no null mask (all elements valid).
    ///
    /// The returned bytes are packed least-significant-bit first and trimmed to
    /// the active range (`len().div_ceil(8)` bytes), not cuDF's internal padded
    /// allocation size.
    ///
    /// # Synchronization
    /// This materializes null-mask bytes into host memory and may synchronize
    /// the bound execution domain before returning.
    pub fn to_host_null_mask_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.to_host_null_mask_bytes_with_operation("column_view::to_host_null_mask_bytes")
    }

    /// Copy the null mask to a host `Vec<bool>`.
    ///
    /// Returns `None` if the column has no null mask (all elements valid).
    ///
    /// # Synchronization
    /// This materializes null-mask bits into host memory and may synchronize
    /// the bound execution domain before returning.
    pub fn to_host_validity(&self) -> Result<Option<Vec<bool>>> {
        let Some(mask) = self.to_host_null_mask_bytes_with_operation("column_view::to_host_validity")? else {
            return Ok(None);
        };

        let size = self.len().map_err(|e| e.reanchor("column_view::to_host_validity"))?;
        let validity: Vec<bool> = (0..size).map(|i| (mask[i / 8] >> (i % 8)) & 1 != 0).collect();
        Ok(Some(validity))
    }

    /// Copy string column data to a host `Vec<String>`, mapping null elements
    /// to empty strings (`""`).
    ///
    /// The column's type must be `TypeId::String`.
    ///
    /// This is a lossy conversion: nulls and genuinely empty strings become
    /// indistinguishable. Use [`to_optional_strings`](Self::to_optional_strings)
    /// when null fidelity matters.
    ///
    /// # Synchronization
    /// This materializes string offsets and character data into host memory and
    /// may synchronize the bound execution domain before returning.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        let strings = self.to_strings_host_raw("column_view::to_strings_lossy")?;
        match self.to_host_validity().map_err(|e| e.reanchor("column_view::to_strings_lossy"))? {
            None => Ok(strings),
            Some(valid) => Ok(strings
                .into_iter()
                .zip(valid)
                .map(|(value, is_valid)| if is_valid { value } else { String::new() })
                .collect()),
        }
    }

    fn to_strings_host_raw(&self, operation: &'static str) -> Result<Vec<String>> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::String {
            return Err(Error::new_with(ErrorCode::TypeMismatch, operation, "column type is not STRING")
                .with_fact(FACT_LEFT_TYPE_ID, dt.id())
                .with_fact(FACT_RIGHT_TYPE_ID, TypeId::String));
        }

        if self.offset()? != 0 {
            // Normalize sliced views to a zero-offset owned column first so
            // host string materialization only covers the active logical range.
            let normalized = copy_view_to_owned(self)?;
            return normalized.view()?.to_strings_host_raw(operation);
        }

        let size = self.len()?;
        if size == 0 {
            return Ok(Vec::new());
        }

        self.validate_string_view_layout_for_host_materialization(operation)?;

        let offsets_count = size + 1;
        let mut offsets = vec![0i32; offsets_count];
        let offsets_buf_size = std::mem::size_of_val(offsets.as_slice());
        let (_, chars_size) = self.string_chars_ptr_and_size().map_err(|e| e.reanchor(operation))?;
        let expected_total_chars_bytes = i32::try_from(chars_size).map_err(|_| {
            foreign_string_view_contract_error(operation, "string chars span exceeds i32::MAX")
                .with_detail("chars_size", chars_size.to_string())
        })?;
        let mut chars = vec![0u8; chars_size];
        let exec_options = self.domain.exec_options_raw();
        let chars_buf = if chars.is_empty() { ptr::null_mut() } else { chars.as_mut_ptr() as *mut c_char };
        let mut total_chars_bytes = 0;

        // SAFETY: offsets has space for n+1 entries, chars is sized from the
        // explicit active chars-span query for this logical STRING view, and
        // self.handle is a valid STRING column view.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_strings_to_host_ex(
                    self.handle.as_ptr(),
                    offsets.as_mut_ptr(),
                    offsets_buf_size,
                    chars_buf,
                    chars.len(),
                    &mut total_chars_bytes,
                    &exec_options,
                )
            },
            operation,
        )
        .map_err(classify_string_view_contract_error)?;

        if total_chars_bytes != expected_total_chars_bytes {
            return Err(foreign_string_view_contract_error(
                operation,
                "string chars size changed during host materialization",
            )
            .with_detail("expected_total_chars_bytes", expected_total_chars_bytes.to_string())
            .with_detail("actual_total_chars_bytes", total_chars_bytes.to_string()));
        }

        // Parse strings from offsets + chars
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let start = usize::try_from(offsets[i]).map_err(|_| {
                foreign_string_view_contract_error(operation, "string offset is negative")
                    .with_detail("row", i.to_string())
                    .with_detail("offset_value", offsets[i].to_string())
            })?;
            let end = usize::try_from(offsets[i + 1]).map_err(|_| {
                foreign_string_view_contract_error(operation, "string offset is negative")
                    .with_detail("row", (i + 1).to_string())
                    .with_detail("offset_value", offsets[i + 1].to_string())
            })?;
            let s = std::str::from_utf8(&chars[start..end]).map_err(|e| {
                foreign_string_view_contract_error(
                    operation,
                    format!("invalid UTF-8 in string column at row {}: {}", i, e),
                )
                .with_detail("row", i.to_string())
            })?;
            result.push(s.to_string());
        }

        Ok(result)
    }

    /// Copy string data to host as `Vec<Option<String>>`, distinguishing null
    /// from empty string.
    ///
    /// Null positions return `None`; valid positions return `Some(s)`.
    /// Non-nullable columns return all `Some`.
    ///
    /// # Synchronization
    /// This materializes string offsets, character data, and null-state into
    /// host memory and may synchronize the bound execution domain before
    /// returning.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    /// column type is not `STRING`.
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        let strings = self.to_strings_host_raw("column_view::to_optional_strings")?;
        match self.to_host_validity().map_err(|e| e.reanchor("column_view::to_optional_strings"))? {
            None => Ok(strings.into_iter().map(Some).collect()),
            Some(valid) => Ok(strings.into_iter().zip(valid).map(|(s, v)| if v { Some(s) } else { None }).collect()),
        }
    }

    /// Returns a [`DeviceSlice`] describing the active device memory region
    /// of this column, with type validation.
    ///
    /// The returned slice borrows `self`, so it cannot outlive this view.
    ///
    /// This is the safe, typed alternative to [`device_data_ptr()`](Self::device_data_ptr)
    /// and [`offset()`](Self::offset). The returned `DeviceSlice` already accounts
    /// for the column's offset — no manual pointer arithmetic needed.
    ///
    /// This is an advanced raw-pointer helper for code that already knows how
    /// to consume a device pointer + length pair. For external cross-crate
    /// interop, prefer [`rapids_interop::TryBorrowColumnView`], which preserves
    /// null-mask and execution-binding semantics in addition to the data
    /// pointer.
    ///
    /// # Type checking
    ///
    /// Returns an error if the column's element type does not match `T::TYPE_ID`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let col = rt.column_from_slice(&[1i32, 2, 3, 4]).unwrap();
    /// let slice = col.device_slice::<i32>().unwrap();
    ///
    /// assert_eq!(slice.len(), 4);
    ///
    /// let (left, right) = slice.split_at(2);
    /// assert_eq!(left.len(), 2);
    /// assert_eq!(right.len(), 2);
    /// ```
    pub fn device_slice<T: CudfType>(&self) -> Result<DeviceSlice<'_>> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "column_view::device_slice",
                "column type does not match requested type",
            )
            .with_fact(FACT_LEFT_TYPE_ID, dt.id())
            .with_fact(FACT_RIGHT_TYPE_ID, T::TYPE_ID));
        }

        let base_ptr = self.device_data_ptr()?;
        let offset = self.offset()? as usize;
        let len = self.len()?;
        let elem_size = std::mem::size_of::<T>();

        // Compute the offset-adjusted pointer. This encapsulates the pointer
        // arithmetic that would otherwise leak into application code.
        // SAFETY: base_ptr is a device pointer returned by FFI. The add() is
        // pointer arithmetic within the column's device buffer; the result is
        // never dereferenced on the host.
        let ptr = if base_ptr.is_null() {
            base_ptr as *mut c_void
        } else {
            unsafe { (base_ptr as *const u8).add(offset * elem_size) as *mut c_void }
        };

        Ok(DeviceSlice { ptr, len, type_id: T::TYPE_ID, elem_size, _lifetime: PhantomData })
    }
}
impl ColumnView<'_> {
    /// Return a boolean mask indicating which elements appear in `values`.
    ///
    /// Element `i` of the result is `true` when `self[i]` is found anywhere
    /// in `values`, `false` otherwise. Null positions in `self` produce null
    /// entries in the output.
    ///
    /// Both `self` and `values` must have the same data type.
    pub fn isin(&self, values: &ColumnView<'_>) -> Result<Column> {
        let input_type_id = match self.data_type() {
            Ok(dt) => dt.id(),
            Err(e) => return Err(e.reanchor("column_view::isin")),
        };
        let input_type_id_str = format!("{:?}", input_type_id).to_lowercase();

        let values_type_id = values
            .data_type()
            .map_err(|e| e.reanchor("column_view::isin").with_detail("input_type_id", input_type_id_str.clone()))?;
        let values_type_id_str = format!("{:?}", values_type_id.id()).to_lowercase();

        if input_type_id != values_type_id.id() {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "column_view::isin",
                "isin requires both inputs to have the same dtype",
            )
            .with_fact(FACT_LEFT_TYPE_ID, input_type_id)
            .with_fact(FACT_RIGHT_TYPE_ID, values_type_id.id()));
        }

        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.execution_domain(),
            values.execution_domain(),
            "column_view::isin",
        )
        .map_err(|e| {
            e.with_detail("input_type_id", input_type_id_str.clone())
                .with_detail("values_type_id", values_type_id_str.clone())
        })?;

        let _consumer_fence =
            ExecutionDomain::insert_stream_dependency(values.execution_domain(), &self.domain, "column_view::isin")
                .map_err(|e| {
                    e.with_detail("input_type_id", input_type_id_str.clone())
                        .with_detail("values_type_id", values_type_id_str.clone())
                })?;

        let exec_options = self.domain.exec_options_raw();
        search::contains_impl(values, self, &self.domain, &exec_options, "column_view::isin").map_err(|e| {
            e.reanchor("column_view::isin")
                .with_detail("input_type_id", input_type_id_str)
                .with_detail("values_type_id", values_type_id_str)
        })
    }
}
impl<'a> Drop for ColumnView<'a> {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_column_view_destroy(self.handle.as_ptr());
        }
    }
}
