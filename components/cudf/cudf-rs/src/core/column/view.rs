use std::{
    marker::PhantomData,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{convert::copy_view_to_owned, device_slice::DeviceSlice, owned::Column};
use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        table::Table,
        types::{CudfType, DataType, TypeId},
    },
    ops::{dictionary, join, search, sorting, stream_compaction},
    runtime::execution_domain::ExecutionDomain,
};

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
    pub(crate) unsafe fn from_raw_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self { handle: NonNull::new(handle).expect("null view handle"), domain, _lifetime: PhantomData }
    }

    /// Returns the raw FFI handle pointer. For internal use by operations
    /// that need to pass column views to C shim functions.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
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
        )?;
        DataType::from_raw_parts(raw_id, scale)
    }

    /// Returns the number of elements.
    pub fn len(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_size(self.handle.as_ptr(), &mut out) },
            "column_view::len",
        )?;
        Ok(out as usize)
    }

    /// Returns `true` if the column has zero elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> Result<i32> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_null_count(self.handle.as_ptr(), &mut out) },
            "column_view::null_count",
        )?;
        Ok(out)
    }

    /// Returns `true` if this column has a null mask allocated.
    pub fn nullable(&self) -> Result<bool> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_nullable(self.handle.as_ptr(), &mut out) },
            "column_view::nullable",
        )?;
        Ok(out != 0)
    }

    /// Returns the number of child columns (for nested types).
    pub fn num_children(&self) -> Result<i32> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction/from_raw).
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_num_children(self.handle.as_ptr(), &mut out) },
            "column_view::num_children",
        )?;
        Ok(out)
    }

    /// Get a child column view by index.
    pub fn child(&self, index: i32) -> Result<ColumnView<'a>> {
        if index < 0 {
            return Err(Error::new(ErrorKind::InvalidOperation, "column child index must be >= 0")
                .with_operation("column_view::child")
                .with_context("child_index", index.to_string()));
        }

        let mut child_handle = ptr::null_mut();
        // SAFETY: self.handle is valid. index >= 0 is checked above.
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_child(self.handle.as_ptr(), index, &mut child_handle) },
            "column_view::child",
        )?;
        Ok(ColumnView {
            handle: NonNull::new(child_handle).expect("null handle on success"),
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
    /// For `STRING` columns the data pointer is null — use [`child()`](Self::child)
    /// to access the offsets (child 0) and character data (child 1) instead.
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
        )?;
        Ok(out)
    }

    /// Copy the column data to a host `Vec<T>`.
    ///
    /// The column's type must match `T::TYPE_ID`. Only valid for fixed-width types.
    pub fn to_vec<T: CudfType>(&self) -> Result<Vec<T>> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new(ErrorKind::TypeError, "column type does not match requested type")
                .with_operation("column_view::to_vec")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
        }

        let size = self.len()?;
        // SAFETY: All CudfType implementors have valid all-zero bit patterns.
        let mut buf = vec![unsafe { std::mem::zeroed::<T>() }; size];
        let buf_size = size * std::mem::size_of::<T>();

        self.synchronize_host_visibility("column_view::to_vec")?;

        // SAFETY: buf is a valid mutable buffer of buf_size bytes. self.handle is
        // valid. Type was checked above to match T::TYPE_ID.
        if let Some(exec_options) = self.domain.explicit_exec_options_raw() {
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
        } else {
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_view_to_host(self.handle.as_ptr(), buf.as_mut_ptr() as *mut c_void, buf_size)
                },
                "column_view::to_vec",
            )?;
        }

        Ok(buf)
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
            return Err(Error::new(ErrorKind::TypeError, "to_raw_bytes requires a 1-byte type")
                .with_operation("column_view::to_raw_bytes")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase()));
        }

        let size = self.len()?;
        let mut buf = vec![0u8; size];

        self.synchronize_host_visibility("column_view::to_raw_bytes")?;

        if let Some(exec_options) = self.domain.explicit_exec_options_raw() {
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
        } else {
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_view_to_host(self.handle.as_ptr(), buf.as_mut_ptr() as *mut c_void, size)
                },
                "column_view::to_raw_bytes",
            )?;
        }

        Ok(buf)
    }

    /// Copy the null mask to a host `Vec<bool>`.
    ///
    /// Returns `None` if the column has no null mask (all elements valid).
    pub fn validity(&self) -> Result<Option<Vec<bool>>> {
        if !self.nullable()? {
            return Ok(None);
        }

        let size = self.len()?;
        let mask_bytes = size.div_ceil(8);
        // Ensure we allocate at least the bitmask alignment size that cudf expects.
        // cudf pads to 64-byte boundaries, so allocate enough.
        let alloc_bytes = mask_bytes.div_ceil(64) * 64;
        let mut mask = vec![0u8; alloc_bytes];

        self.synchronize_host_visibility("column_view::validity")?;

        // SAFETY: mask buffer is large enough (padded to 64-byte boundaries).
        // self.handle is valid and nullable (checked above).
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_null_mask_to_host(self.handle.as_ptr(), mask.as_mut_ptr(), alloc_bytes)
            },
            "column_view::validity",
        )?;

        // Unpack bitmask to bool vec.
        let validity: Vec<bool> = (0..size).map(|i| (mask[i / 8] >> (i % 8)) & 1 != 0).collect();

        Ok(Some(validity))
    }

    /// Copy string column data to a host `Vec<String>`.
    ///
    /// The column's type must be `TypeId::String`.
    ///
    /// For null elements, the returned string is empty (`""`). Use
    /// [`validity()`](Self::validity) to distinguish nulls from empty strings.
    pub fn to_strings(&self) -> Result<Vec<String>> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::String {
            return Err(Error::new(ErrorKind::TypeError, "column type is not STRING")
                .with_operation("column_view::to_strings")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase()));
        }

        let size = self.len()?;
        if size == 0 {
            return Ok(Vec::new());
        }

        // Allocate offsets buffer: n+1 elements
        let offsets_count = size + 1;
        let offsets_buf_size = offsets_count * std::mem::size_of::<i32>();
        let mut offsets = vec![0i32; offsets_count];
        let exec_options = self.domain.explicit_exec_options_raw();

        // First call to get total_chars_bytes: pass a generous chars buffer initially.
        // We do a two-phase approach: first get offsets + total size, then get chars.
        let mut total_chars_bytes: i32 = 0;

        // Phase 1: get offsets and total char bytes (with null chars_buf if 0 chars)
        // SAFETY: offsets buffer has n+1 capacity. self.handle is valid STRING column.
        error::check_status_in(
            if let Some(exec_options) = exec_options.as_ref() {
                unsafe {
                    cudf_sys::cudf_column_view_strings_to_host_ex(
                        self.handle.as_ptr(),
                        offsets.as_mut_ptr(),
                        offsets_buf_size,
                        ptr::null_mut(), // chars_buf not needed yet for size query
                        0,               // chars_buf_size = 0
                        &mut total_chars_bytes,
                        exec_options,
                    )
                }
            } else {
                unsafe {
                    cudf_sys::cudf_column_view_strings_to_host(
                        self.handle.as_ptr(),
                        offsets.as_mut_ptr(),
                        offsets_buf_size,
                        ptr::null_mut(), // chars_buf not needed yet for size query
                        0,               // chars_buf_size = 0
                        &mut total_chars_bytes,
                    )
                }
            },
            "column_view::to_strings",
        )?;

        // Phase 2: now allocate chars buffer and re-call
        let mut chars = vec![0u8; total_chars_bytes as usize];
        if total_chars_bytes > 0 {
            // Need to re-fetch since the first call already wrote offsets but
            // may not have written chars. Re-call to get everything.
            // SAFETY: chars buffer is sized to total_chars_bytes from Phase 1.
            error::check_status_in(
                if let Some(exec_options) = exec_options.as_ref() {
                    unsafe {
                        cudf_sys::cudf_column_view_strings_to_host_ex(
                            self.handle.as_ptr(),
                            offsets.as_mut_ptr(),
                            offsets_buf_size,
                            chars.as_mut_ptr() as *mut c_char,
                            total_chars_bytes as usize,
                            &mut total_chars_bytes,
                            exec_options,
                        )
                    }
                } else {
                    unsafe {
                        cudf_sys::cudf_column_view_strings_to_host(
                            self.handle.as_ptr(),
                            offsets.as_mut_ptr(),
                            offsets_buf_size,
                            chars.as_mut_ptr() as *mut c_char,
                            total_chars_bytes as usize,
                            &mut total_chars_bytes,
                        )
                    }
                },
                "column_view::to_strings",
            )?;
        }

        // Parse strings from offsets + chars
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let s = std::str::from_utf8(&chars[start..end]).map_err(|e| {
                Error::new(ErrorKind::InvalidOperation, format!("invalid UTF-8 in string column at row {}: {}", i, e))
                    .with_operation("column_view::to_strings")
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
    /// # Errors
    ///
    /// Returns `TypeError` if the column type is not `STRING`.
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        let strings = self.to_strings()?;
        match self.validity()? {
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
    /// # Type checking
    ///
    /// Returns an error if the column's element type does not match `T::TYPE_ID`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    /// let view = col.view().unwrap();
    /// let slice = view.device_slice::<i32>().unwrap();
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
            return Err(Error::new(ErrorKind::TypeError, "column type does not match requested type")
                .with_operation("column_view::device_slice")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
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
    /// Encode column values as integer codes with corresponding unique values.
    ///
    /// Returns `(codes, uniques)` where:
    /// - `codes`: Int32 column where `codes[i]` is the index of row `i`'s value
    ///   in the `uniques` column
    /// - `uniques`: Column of unique values from this column
    ///
    /// Null positions in the input produce null entries in `codes`.
    pub fn factorize(&self) -> Result<(Column, Column)> {
        let output_domain = Arc::clone(&self.domain);
        let input_type_id = match self.data_type() {
            Ok(dt) => dt.id(),
            Err(e) => return Err(e.with_operation("column_view::factorize")),
        };
        let input_type_id_str = format!("{:?}", input_type_id).to_lowercase();

        let dict = dictionary::encode(self, TypeId::Int32).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let dict_view = dict.view().map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;

        let indices_view = dictionary::indices(&dict_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let keys_view = dictionary::keys(&dict_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;

        let codes = copy_view_to_owned(&indices_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let uniques = copy_view_to_owned(&keys_view)
            .map_err(|e| e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str))?;

        Ok((codes.with_execution_domain(Arc::clone(&output_domain)), uniques.with_execution_domain(output_domain)))
    }

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
            Err(e) => return Err(e.with_operation("column_view::isin")),
        };
        let input_type_id_str = format!("{:?}", input_type_id).to_lowercase();

        let values_type_id = values.data_type().map_err(|e| {
            e.with_operation("column_view::isin").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let values_type_id_str = format!("{:?}", values_type_id.id()).to_lowercase();

        if !ExecutionDomain::same_provenance(values.execution_domain(), &self.domain) {
            values.execution_domain().fence_stream("column_view::isin").map_err(|e| {
                e.with_context("input_type_id", input_type_id_str.clone())
                    .with_context("values_type_id", values_type_id_str.clone())
            })?;
        }

        let exec_options = self.domain.explicit_exec_options_raw();
        search::contains_with_exec_options(values, self, exec_options.as_ref(), "column_view::isin")
            .map(|column| column.with_execution_domain(Arc::clone(&self.domain)))
            .map_err(|e| {
                e.with_operation("column_view::isin")
                    .with_context("input_type_id", input_type_id_str)
                    .with_context("values_type_id", values_type_id_str)
            })
    }

    /// Map values through a lookup table.
    ///
    /// For each element in `self`, find its position in `keys` and return
    /// the corresponding element from `values`. Elements not found in `keys`
    /// become null. Implemented via a left join internally.
    ///
    /// Preconditions:
    /// - `self` and `keys` must have the same type.
    /// - `keys` and `values` must have the same length.
    /// - `keys` must be unique (duplicate mapping keys are rejected).
    ///
    /// # Example
    /// ```no_run
    /// use cudf::Column;
    /// let input = Column::from_strings(&["a", "b", "c", "a"]).unwrap();
    /// let keys = Column::from_strings(&["a", "b", "c"]).unwrap();
    /// let values = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let mapped = input
    ///     .view()
    ///     .unwrap()
    ///     .map(&keys.view().unwrap(), &values.view().unwrap())
    ///     .unwrap();
    /// ```
    pub fn map(&self, keys: &ColumnView<'_>, values: &ColumnView<'_>) -> Result<Column> {
        let input_type_id = self.data_type().map_err(|e| e.with_operation("column_view::map"))?.id();
        let keys_type_id = keys.data_type().map_err(|e| e.with_operation("column_view::map"))?.id();
        let values_type_id = values.data_type().map_err(|e| e.with_operation("column_view::map"))?.id();

        let input_type_id_str = format!("{:?}", input_type_id).to_lowercase();
        let keys_type_id_str = format!("{:?}", keys_type_id).to_lowercase();
        let values_type_id_str = format!("{:?}", values_type_id).to_lowercase();

        if input_type_id != keys_type_id {
            return Err(Error::new(ErrorKind::InvalidOperation, "self and keys must have the same type")
                .with_operation("column_view::map")
                .with_context("input_type_id", input_type_id_str)
                .with_context("keys_type_id", keys_type_id_str)
                .with_context("values_type_id", values_type_id_str));
        }

        let input_count = self.len().map_err(|e| e.with_operation("column_view::map"))?;
        let keys_count = keys.len().map_err(|e| e.with_operation("column_view::map"))?;
        let values_count = values.len().map_err(|e| e.with_operation("column_view::map"))?;

        if keys_count != values_count {
            return Err(Error::new(ErrorKind::InvalidOperation, "keys and values must have the same length")
                .with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string()));
        }

        if input_count == 0 {
            return Column::empty(values_type_id).map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            });
        }

        let keys_owned = copy_view_to_owned(keys).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let mapper_key_table = Table::from_columns(vec![keys_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let distinct_keys = stream_compaction::distinct(
            &mapper_key_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            &[0],
            stream_compaction::DuplicateKeepOption::KeepAny,
            stream_compaction::NullEquality::Equal,
            stream_compaction::NanEquality::AllEqual,
        )
        .map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let distinct_keys_count = distinct_keys
            .view()
            .map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?
            .num_rows()
            .map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?;

        if distinct_keys_count as usize != keys_count {
            return Err(Error::new(ErrorKind::InvalidOperation, "keys must be unique")
                .with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
                .with_context("distinct_keys_count", distinct_keys_count.to_string()));
        }

        // Build a row-index column [0, 1, 2, ..., len-1] as left payload
        // to restore original ordering after the join.
        let row_indices: Vec<i32> = (0..input_count as i32).collect();
        let idx_col = Column::from_slice(&row_indices).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // Build tables
        let self_owned = copy_view_to_owned(self).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let self_key_table = Table::from_columns(vec![self_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let left_payload = Table::from_columns(vec![idx_col]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let values_owned = copy_view_to_owned(values).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let mapper_val_table = Table::from_columns(vec![values_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // Left join: self_keys LEFT JOIN mapper_keys
        let result = join::left_join(
            &self_key_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            &mapper_key_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            &left_payload.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            &mapper_val_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            join::NullEquality::Equal,
        )
        .map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // result.left = [left_key, row_index], result.right = [right_key, mapped_value]
        // Extract row_index from left (column 1) and mapped_value from right (column 1).
        // Sort by row_index to restore original order.
        let left_view = result.left.view().map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let right_view = result.right.view().map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let idx_cv = left_view.column(1).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let val_cv = right_view.column(1).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // Build tables for sort_by_key: values=[mapped_value], keys=[row_index]
        let idx_owned = copy_view_to_owned(&idx_cv).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let val_owned = copy_view_to_owned(&val_cv).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let val_table = Table::from_columns(vec![val_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let key_table = Table::from_columns(vec![idx_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let sorted = sorting::sort_by_key(
            &val_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            &key_table.view().map_err(|e| {
                e.with_operation("column_view::map")
                    .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                    .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                    .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                    .with_context("input_count", input_count.to_string())
                    .with_context("keys_count", keys_count.to_string())
                    .with_context("values_count", values_count.to_string())
            })?,
            Some(&[sorting::Order::Ascending]),
            None,
        )
        .map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // Extract the sorted value column
        let sorted_view = sorted.view().map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let sorted_val = sorted_view.column(0).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        copy_view_to_owned(&sorted_val).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
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
