//! GPU column types: owned `Column` and borrowed `ColumnView`.

use crate::error::{self, Error, ErrorKind, Result};

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
};

use crate::types::{CudfType, DataType, TypeId};

/// An owned GPU column backed by a `cudf::column`.
///
/// The underlying device memory is freed when this value is dropped.
pub struct Column {
    handle: NonNull<c_void>,
}

unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    /// Create an empty column of the given type.
    pub fn empty(type_id: TypeId) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_create_empty(type_id.to_raw(), &mut handle) },
            "column::empty",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a column from a host slice of fixed-width elements.
    ///
    /// # Example
    /// ```no_run
    /// use cudf::column::Column;
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// ```
    pub fn from_slice<T: CudfType>(data: &[T]) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let size = usize_to_i32_len(data.len(), "data", "column::from_slice")?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_numeric(
                    T::TYPE_ID.to_raw(),
                    data.as_ptr() as *const c_void,
                    size,
                    ptr::null(),
                    0,
                    &mut handle,
                )
            },
            "column::from_slice",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a nullable column from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn from_slice_nullable<T: CudfType>(data: &[T], validity: &[bool]) -> Result<Self> {
        if data.len() != validity.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "data and validity slices must have the same length")
                .with_operation("column::from_slice_nullable")
                .with_context("data_len", data.len().to_string())
                .with_context("validity_len", validity.len().to_string()));
        }

        let size = usize_to_i32_len(data.len(), "data", "column::from_slice_nullable")?;
        let null_mask = crate::types::validity_to_bitmask(validity);
        let null_count = crate::types::count_nulls(validity);

        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_numeric(
                    T::TYPE_ID.to_raw(),
                    data.as_ptr() as *const c_void,
                    size,
                    null_mask.as_ptr(),
                    null_count,
                    &mut handle,
                )
            },
            "column::from_slice_nullable",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Deep-copy this column.
    pub fn copy(&self) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_copy(self.handle.as_ptr(), &mut handle) },
            "column::copy",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create an immutable view of this column.
    ///
    /// The returned `ColumnView` borrows from `self` and cannot outlive it.
    pub fn view(&self) -> Result<ColumnView<'_>> {
        let mut view_handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view(self.handle.as_ptr(), &mut view_handle) },
            "column::view",
        )?;
        Ok(ColumnView { handle: NonNull::new(view_handle).expect("null handle on success"), _lifetime: PhantomData })
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
        Self { handle: NonNull::new(handle).expect("null column handle") }
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
        error::check_status_in_with(
            unsafe { cudf_sys::cudf_column_create_from_device_data(type_id.to_raw(), device_ptr, size, &mut handle) },
            "column::from_device_ptr",
            [("len", len.to_string()), ("type_id", format!("{:?}", type_id).to_lowercase())],
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a STRING type column from a slice of string references.
    ///
    /// Each string is copied to GPU memory. The resulting column has no null mask.
    ///
    /// # Example
    /// ```no_run
    /// use cudf::column::Column;
    /// let col = Column::from_strings(&["hello", "world"]).unwrap();
    /// ```
    pub fn from_strings(data: &[&str]) -> Result<Self> {
        let num_strings = usize_to_i32_len(data.len(), "data", "column::from_strings")?;

        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr() as *const c_char).collect();
        let lengths: Vec<i32> = data
            .iter()
            .map(|s| usize_to_i32_len(s.len(), "string_element", "column::from_strings"))
            .collect::<Result<Vec<_>>>()?;

        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_strings(
                    ptrs.as_ptr(),
                    lengths.as_ptr(),
                    num_strings,
                    ptr::null(),
                    0,
                    &mut handle,
                )
            },
            "column::from_strings",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
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
                let null_mask = crate::types::validity_to_bitmask(&validity);
                let null_count = size;

                let mut handle = ptr::null_mut();
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
                Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
            },
        }
    }

    /// Create a nullable STRING type column from host string data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    /// For null rows, the string content is ignored.
    pub fn from_strings_nullable(data: &[&str], validity: &[bool]) -> Result<Self> {
        if data.len() != validity.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "data and validity slices must have the same length")
                .with_operation("column::from_strings_nullable")
                .with_context("data_len", data.len().to_string())
                .with_context("validity_len", validity.len().to_string()));
        }

        let num_strings = usize_to_i32_len(data.len(), "data", "column::from_strings_nullable")?;

        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr() as *const c_char).collect();
        let lengths: Vec<i32> = data
            .iter()
            .map(|s| usize_to_i32_len(s.len(), "string_element", "column::from_strings_nullable"))
            .collect::<Result<Vec<_>>>()?;

        let null_mask = crate::types::validity_to_bitmask(validity);
        let null_count = crate::types::count_nulls(validity);

        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_create_strings(
                    ptrs.as_ptr(),
                    lengths.as_ptr(),
                    num_strings,
                    if null_count > 0 { null_mask.as_ptr() } else { ptr::null() },
                    null_count,
                    &mut handle,
                )
            },
            "column::from_strings_nullable",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }
}

// ---------------------------------------------------------------------------
// Binary operations on Column
// ---------------------------------------------------------------------------

impl Column {
    /// Generic binary operation with inferred output type.
    pub fn binary_op(&self, rhs: &Column, op: crate::binary::BinaryOperator) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            let out_type = crate::binary::infer_binary_output_type(lv.data_type()?.id(), rv.data_type()?.id(), op)?;
            crate::binary::binary_operation_vv(&lv, &rv, op, &out_type)
        })()
        .map_err(|e| e.with_operation("column::binary_op").with_context("operator", format!("{:?}", op).to_lowercase()))
    }

    /// Generic binary operation with explicit output type (escape hatch).
    pub fn binary_op_with_type(
        &self,
        rhs: &Column,
        op: crate::binary::BinaryOperator,
        output_type: &DataType,
    ) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            crate::binary::binary_operation_vv(&lv, &rv, op, output_type)
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
    /// use cudf::column::Column;
    ///
    /// let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    /// let sum = a.add(&b).unwrap();
    /// // sum: [11, 22, 33]
    ///
    /// // Also available via std::ops: &a + &b
    /// ```
    pub fn add(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Add).map_err(|e| e.with_operation("column::add"))
    }

    /// Element-wise subtraction: `self - rhs`. Output type inferred.
    pub fn sub(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Sub).map_err(|e| e.with_operation("column::sub"))
    }

    /// Element-wise multiplication: `self * rhs`. Output type inferred.
    pub fn mul(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Mul).map_err(|e| e.with_operation("column::mul"))
    }

    /// Element-wise division: `self / rhs`. Output type inferred.
    pub fn div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Div).map_err(|e| e.with_operation("column::div"))
    }

    /// Element-wise true division (always float output).
    pub fn true_div(&self, rhs: &Column) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let rv = rhs.view()?;
            let out_type = DataType::new(TypeId::Float64);
            crate::binary::binary_operation_vv(&lv, &rv, crate::binary::BinaryOperator::TrueDiv, &out_type)
        })()
        .map_err(|e| e.with_operation("column::true_div").with_context("operator", "true_div"))
    }

    /// Element-wise floor division.
    pub fn floor_div(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::FloorDiv).map_err(|e| e.with_operation("column::floor_div"))
    }

    /// Element-wise modulo: `self % rhs`.
    pub fn modulo(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Mod).map_err(|e| e.with_operation("column::modulo"))
    }

    // -- Comparison methods ---------------------------------------------------

    /// Element-wise equal: `self == rhs`. Returns Bool8 column.
    pub fn eq_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Equal).map_err(|e| e.with_operation("column::eq_col"))
    }

    /// Element-wise not-equal: `self != rhs`. Returns Bool8 column.
    pub fn ne_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::NotEqual).map_err(|e| e.with_operation("column::ne_col"))
    }

    /// Element-wise less-than: `self < rhs`. Returns Bool8 column.
    pub fn lt_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Less).map_err(|e| e.with_operation("column::lt_col"))
    }

    /// Element-wise greater-than: `self > rhs`. Returns Bool8 column.
    pub fn gt_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::Greater).map_err(|e| e.with_operation("column::gt_col"))
    }

    /// Element-wise less-or-equal: `self <= rhs`. Returns Bool8 column.
    pub fn le_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::LessEqual).map_err(|e| e.with_operation("column::le_col"))
    }

    /// Element-wise greater-or-equal: `self >= rhs`. Returns Bool8 column.
    pub fn ge_col(&self, rhs: &Column) -> Result<Column> {
        self.binary_op(rhs, crate::binary::BinaryOperator::GreaterEqual).map_err(|e| e.with_operation("column::ge_col"))
    }

    // -- Scalar operations ----------------------------------------------------

    /// Element-wise: `self op scalar`. Output type inferred.
    pub fn binary_op_scalar(&self, rhs: &crate::scalar::Scalar, op: crate::binary::BinaryOperator) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            let out_type = crate::binary::infer_binary_output_type(lv.data_type()?.id(), rhs.data_type()?.id(), op)?;
            crate::binary::binary_operation_vs(&lv, rhs, op, &out_type)
        })()
        .map_err(|e| {
            e.with_operation("column::binary_op_scalar").with_context("operator", format!("{:?}", op).to_lowercase())
        })
    }

    /// Element-wise: `self op scalar` with explicit output type.
    pub fn binary_op_scalar_with_type(
        &self,
        rhs: &crate::scalar::Scalar,
        op: crate::binary::BinaryOperator,
        output_type: &DataType,
    ) -> Result<Column> {
        (|| {
            let lv = self.view()?;
            crate::binary::binary_operation_vs(&lv, rhs, op, output_type)
        })()
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
    /// use cudf::{column::Column, scalar::Scalar};
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let offset = Scalar::from_value(100i32).unwrap();
    /// let result = col.add_scalar(&offset).unwrap();
    /// // result: [101, 102, 103]
    ///
    /// // Also available via std::ops: &col + 100i32
    /// ```
    pub fn add_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Add)
            .map_err(|e| e.with_operation("column::add_scalar"))
    }

    /// Element-wise: `self - scalar`.
    pub fn sub_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Sub)
            .map_err(|e| e.with_operation("column::sub_scalar"))
    }

    /// Element-wise: `self * scalar`.
    pub fn mul_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Mul)
            .map_err(|e| e.with_operation("column::mul_scalar"))
    }

    /// Element-wise: `self / scalar`.
    pub fn div_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Div)
            .map_err(|e| e.with_operation("column::div_scalar"))
    }

    /// Element-wise: `self == scalar`. Returns Bool8 column.
    pub fn eq_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Equal)
            .map_err(|e| e.with_operation("column::eq_scalar"))
    }

    /// Element-wise: `self != scalar`. Returns Bool8 column.
    pub fn ne_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::NotEqual)
            .map_err(|e| e.with_operation("column::ne_scalar"))
    }

    /// Element-wise: `self < scalar`. Returns Bool8 column.
    pub fn lt_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Less)
            .map_err(|e| e.with_operation("column::lt_scalar"))
    }

    /// Element-wise: `self > scalar`. Returns Bool8 column.
    pub fn gt_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::Greater)
            .map_err(|e| e.with_operation("column::gt_scalar"))
    }

    /// Element-wise: `self <= scalar`. Returns Bool8 column.
    pub fn le_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::LessEqual)
            .map_err(|e| e.with_operation("column::le_scalar"))
    }

    /// Element-wise: `self >= scalar`. Returns Bool8 column.
    pub fn ge_scalar(&self, rhs: &crate::scalar::Scalar) -> Result<Column> {
        self.binary_op_scalar(rhs, crate::binary::BinaryOperator::GreaterEqual)
            .map_err(|e| e.with_operation("column::ge_scalar"))
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_column_destroy(self.handle.as_ptr());
        }
    }
}

/// A non-owning, immutable view of a GPU column.
///
/// The lifetime `'a` ties this view to the owning `Column` (or `Table`),
/// preventing use-after-free at compile time.
pub struct ColumnView<'a> {
    handle: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> ColumnView<'a> {
    /// Create a `ColumnView` from a raw FFI handle.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_column_view_handle` and must remain
    /// valid for lifetime `'a`.
    pub(crate) unsafe fn from_raw(handle: *mut c_void) -> Self {
        Self { handle: NonNull::new(handle).expect("null view handle"), _lifetime: PhantomData }
    }

    /// Returns the raw FFI handle pointer. For internal use by operations
    /// that need to pass column views to C shim functions.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    /// Returns the data type of this column.
    pub fn data_type(&self) -> Result<DataType> {
        let mut raw_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_type(self.handle.as_ptr(), &mut raw_id) },
            "column_view::data_type",
        )?;
        Ok(DataType::new(TypeId::from_raw(raw_id)))
    }

    /// Returns the number of elements.
    pub fn len(&self) -> Result<i32> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_size(self.handle.as_ptr(), &mut out) },
            "column_view::len",
        )?;
        Ok(out)
    }

    /// Returns `true` if the column has zero elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> Result<i32> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_null_count(self.handle.as_ptr(), &mut out) },
            "column_view::null_count",
        )?;
        Ok(out)
    }

    /// Returns `true` if this column has a null mask allocated.
    pub fn nullable(&self) -> Result<bool> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_nullable(self.handle.as_ptr(), &mut out) },
            "column_view::nullable",
        )?;
        Ok(out != 0)
    }

    /// Returns the number of child columns (for nested types).
    pub fn num_children(&self) -> Result<i32> {
        let mut out: i32 = 0;
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
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_view_child(self.handle.as_ptr(), index, &mut child_handle) },
            "column_view::child",
        )?;
        Ok(ColumnView { handle: NonNull::new(child_handle).expect("null handle on success"), _lifetime: PhantomData })
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

        let size = self.len()? as usize;
        let mut buf = vec![unsafe { std::mem::zeroed::<T>() }; size];
        let buf_size = size * std::mem::size_of::<T>();

        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_to_host(self.handle.as_ptr(), buf.as_mut_ptr() as *mut c_void, buf_size)
            },
            "column_view::to_vec",
        )?;

        Ok(buf)
    }

    /// Copy the null mask to a host `Vec<bool>`.
    ///
    /// Returns `None` if the column has no null mask (all elements valid).
    pub fn validity(&self) -> Result<Option<Vec<bool>>> {
        if !self.nullable()? {
            return Ok(None);
        }

        let size = self.len()? as usize;
        let mask_bytes = size.div_ceil(8);
        // Ensure we allocate at least the bitmask alignment size that cudf expects.
        // cudf pads to 64-byte boundaries, so allocate enough.
        let alloc_bytes = mask_bytes.div_ceil(64) * 64;
        let mut mask = vec![0u8; alloc_bytes];

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
        let offsets_count = (size + 1) as usize;
        let offsets_buf_size = offsets_count * std::mem::size_of::<i32>();
        let mut offsets = vec![0i32; offsets_count];

        // First call to get total_chars_bytes: pass a generous chars buffer initially.
        // We do a two-phase approach: first get offsets + total size, then get chars.
        let mut total_chars_bytes: i32 = 0;

        // Phase 1: get offsets and total char bytes (with null chars_buf if 0 chars)
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_strings_to_host(
                    self.handle.as_ptr(),
                    offsets.as_mut_ptr(),
                    offsets_buf_size,
                    ptr::null_mut(), // chars_buf not needed yet for size query
                    0,               // chars_buf_size = 0
                    &mut total_chars_bytes,
                )
            },
            "column_view::to_strings",
        )?;

        // Phase 2: now allocate chars buffer and re-call
        let mut chars = vec![0u8; total_chars_bytes as usize];
        if total_chars_bytes > 0 {
            // Need to re-fetch since the first call already wrote offsets but
            // may not have written chars. Re-call to get everything.
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_column_view_strings_to_host(
                        self.handle.as_ptr(),
                        offsets.as_mut_ptr(),
                        offsets_buf_size,
                        chars.as_mut_ptr() as *mut c_char,
                        total_chars_bytes as usize,
                        &mut total_chars_bytes,
                    )
                },
                "column_view::to_strings",
            )?;
        }

        // Parse strings from offsets + chars
        let mut result = Vec::with_capacity(size as usize);
        for i in 0..size as usize {
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

    /// Returns a [`DeviceSlice`] describing the active device memory region
    /// of this column, with type validation.
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
    /// use cudf::column::Column;
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
    pub fn device_slice<T: CudfType>(&self) -> Result<DeviceSlice> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new(ErrorKind::TypeError, "column type does not match requested type")
                .with_operation("column_view::device_slice")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
        }

        let base_ptr = self.device_data_ptr()?;
        let offset = self.offset()? as usize;
        let len = self.len()? as usize;
        let elem_size = std::mem::size_of::<T>();

        // Compute the offset-adjusted pointer. This encapsulates the pointer
        // arithmetic that would otherwise leak into application code.
        let ptr = if base_ptr.is_null() {
            base_ptr as *mut c_void
        } else {
            unsafe { (base_ptr as *const u8).add(offset * elem_size) as *mut c_void }
        };

        Ok(DeviceSlice { ptr, len, type_id: T::TYPE_ID, elem_size })
    }
}

/// An opaque descriptor of a contiguous region of GPU device memory.
///
/// `DeviceSlice` is a safe, typed handle that carries a device pointer, element
/// count, and type information. It does **not** copy or dereference the pointer —
/// device pointers are not accessible from host code, so holding one cannot cause
/// undefined behavior.
///
/// Create a `DeviceSlice` from a [`ColumnView`] via
/// [`device_slice()`](ColumnView::device_slice). Use [`split_at()`](Self::split_at)
/// to partition it into sub-regions (e.g., source and destination vertex IDs).
///
/// `DeviceSlice` is `Send + Sync` because the underlying data lives on the GPU
/// and is not host-dereferenceable.
#[derive(Debug, Clone)]
pub struct DeviceSlice {
    ptr: *mut c_void,
    len: usize,
    type_id: TypeId,
    elem_size: usize,
}

// Device pointers are not dereferenceable from host code, so sharing across
// threads is safe — no host-side data race is possible.
unsafe impl Send for DeviceSlice {}
unsafe impl Sync for DeviceSlice {}

impl DeviceSlice {
    /// Number of elements in this device slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this slice contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The cudf type of the elements.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Returns the raw device pointer.
    ///
    /// This pointer refers to GPU memory and cannot be dereferenced from host
    /// code. It is intended for passing to GPU libraries (e.g., cugraph) that
    /// accept device pointers.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns the size in bytes of one element.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Split this slice into two at the given element index.
    ///
    /// Returns `(left, right)` where `left` has `mid` elements and `right`
    /// has `len - mid` elements.
    ///
    /// # Panics
    ///
    /// Panics if `mid > self.len`.
    pub fn split_at(&self, mid: usize) -> (DeviceSlice, DeviceSlice) {
        assert!(mid <= self.len, "split_at: mid ({mid}) > len ({})", self.len);

        let right_ptr = if self.ptr.is_null() || self.elem_size == 0 {
            self.ptr
        } else {
            unsafe { (self.ptr as *const u8).add(mid * self.elem_size) as *mut c_void }
        };

        (
            DeviceSlice { ptr: self.ptr, len: mid, type_id: self.type_id, elem_size: self.elem_size },
            DeviceSlice { ptr: right_ptr, len: self.len - mid, type_id: self.type_id, elem_size: self.elem_size },
        )
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
        let input_type_id = match self.data_type() {
            Ok(dt) => dt.id(),
            Err(e) => return Err(e.with_operation("column_view::factorize")),
        };
        let input_type_id_str = format!("{:?}", input_type_id).to_lowercase();

        let dict = crate::dictionary::encode(self, TypeId::Int32).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let dict_view = dict.view().map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;

        let indices_view = crate::dictionary::indices(&dict_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let keys_view = crate::dictionary::keys(&dict_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;

        let codes = copy_view_to_owned(&indices_view).map_err(|e| {
            e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str.clone())
        })?;
        let uniques = copy_view_to_owned(&keys_view)
            .map_err(|e| e.with_operation("column_view::factorize").with_context("input_type_id", input_type_id_str))?;

        Ok((codes, uniques))
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

        crate::search::contains(values, self).map_err(|e| {
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
    /// use cudf::column::Column;
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
        let mapper_key_table = crate::table::Table::from_columns(vec![keys_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let distinct_keys = crate::stream_compaction::distinct(
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
            crate::stream_compaction::DuplicateKeepOption::KeepAny,
            crate::stream_compaction::NullEquality::Equal,
            crate::stream_compaction::NanEquality::AllEqual,
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

        if distinct_keys_count != keys_count {
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
        let row_indices: Vec<i32> = (0..input_count).collect();
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
        let self_key_table = crate::table::Table::from_columns(vec![self_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let left_payload = crate::table::Table::from_columns(vec![idx_col]).map_err(|e| {
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
        let mapper_val_table = crate::table::Table::from_columns(vec![values_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        // Left join: self_keys LEFT JOIN mapper_keys
        let result = crate::join::left_join(
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
            crate::join::NullEquality::Equal,
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
        let val_table = crate::table::Table::from_columns(vec![val_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;
        let key_table = crate::table::Table::from_columns(vec![idx_owned]).map_err(|e| {
            e.with_operation("column_view::map")
                .with_context("input_type_id", format!("{:?}", input_type_id).to_lowercase())
                .with_context("keys_type_id", format!("{:?}", keys_type_id).to_lowercase())
                .with_context("values_type_id", format!("{:?}", values_type_id).to_lowercase())
                .with_context("input_count", input_count.to_string())
                .with_context("keys_count", keys_count.to_string())
                .with_context("values_count", values_count.to_string())
        })?;

        let sorted = crate::sorting::sort_by_key(
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
            Some(&[crate::sorting::Order::Ascending]),
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

/// Deep-copy a `ColumnView` into an owned `Column`.
///
/// Uses `cudf_column_copy_from_view`, which supports fixed-width, string,
/// and nested types.
pub(crate) fn copy_view_to_owned(cv: &ColumnView<'_>) -> Result<Column> {
    let len = cv.len()?;
    if len == 0 {
        return Column::empty(cv.data_type()?.id());
    }
    let mut handle = ptr::null_mut();
    error::check_status_in(
        unsafe { cudf_sys::cudf_column_copy_from_view(cv.raw_handle(), &mut handle) },
        "column::copy_view_to_owned",
    )?;
    Ok(unsafe { Column::from_raw(handle) })
}

impl<'a> Drop for ColumnView<'a> {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_column_view_destroy(self.handle.as_ptr());
        }
    }
}
