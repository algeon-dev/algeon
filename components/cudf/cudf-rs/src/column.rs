//! GPU column types: owned `Column` and borrowed `ColumnView`.

use crate::error::{self, Error, ErrorKind, Result};

fn usize_to_i32_len(len: usize, what: &'static str, operation: &'static str) -> Result<i32> {
    i32::try_from(len).map_err(|_| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("{} length exceeds i32::MAX", what),
        )
        .with_operation(operation)
        .with_context("len", len.to_string())
    })
}
use crate::types::{CudfType, DataType, TypeId};
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

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
        Ok(Self {
            handle: NonNull::new(handle).expect("null handle on success"),
        })
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
        Ok(Self {
            handle: NonNull::new(handle).expect("null handle on success"),
        })
    }

    /// Create a nullable column from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn from_slice_nullable<T: CudfType>(data: &[T], validity: &[bool]) -> Result<Self> {
        if data.len() != validity.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "data and validity slices must have the same length",
            )
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
        Ok(Self {
            handle: NonNull::new(handle).expect("null handle on success"),
        })
    }

    /// Deep-copy this column.
    pub fn copy(&self) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_column_copy(self.handle.as_ptr(), &mut handle) },
            "column::copy",
        )?;
        Ok(Self {
            handle: NonNull::new(handle).expect("null handle on success"),
        })
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
        Ok(ColumnView {
            handle: NonNull::new(view_handle).expect("null handle on success"),
            _lifetime: PhantomData,
        })
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
        Self {
            handle: NonNull::new(handle).expect("null column handle"),
        }
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
        Self {
            handle: NonNull::new(handle).expect("null view handle"),
            _lifetime: PhantomData,
        }
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
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "column child index must be >= 0",
            )
            .with_operation("column_view::child")
            .with_context("child_index", index.to_string()));
        }

        let mut child_handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_child(self.handle.as_ptr(), index, &mut child_handle)
            },
            "column_view::child",
        )?;
        Ok(ColumnView {
            handle: NonNull::new(child_handle).expect("null handle on success"),
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
            return Err(Error::new(
                ErrorKind::TypeError,
                "column type does not match requested type",
            )
            .with_operation("column_view::to_vec")
            .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
            .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
        }

        let size = self.len()? as usize;
        let mut buf = vec![unsafe { std::mem::zeroed::<T>() }; size];
        let buf_size = size * std::mem::size_of::<T>();

        error::check_status_in(
            unsafe {
                cudf_sys::cudf_column_view_to_host(
                    self.handle.as_ptr(),
                    buf.as_mut_ptr() as *mut c_void,
                    buf_size,
                )
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
                cudf_sys::cudf_column_view_null_mask_to_host(
                    self.handle.as_ptr(),
                    mask.as_mut_ptr(),
                    alloc_bytes,
                )
            },
            "column_view::validity",
        )?;

        // Unpack bitmask to bool vec.
        let validity: Vec<bool> = (0..size)
            .map(|i| (mask[i / 8] >> (i % 8)) & 1 != 0)
            .collect();

        Ok(Some(validity))
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
            return Err(Error::new(
                ErrorKind::TypeError,
                "column type does not match requested type",
            )
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

        Ok(DeviceSlice {
            ptr,
            len,
            type_id: T::TYPE_ID,
            elem_size,
        })
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
        assert!(
            mid <= self.len,
            "split_at: mid ({mid}) > len ({})",
            self.len
        );

        let right_ptr = if self.ptr.is_null() || self.elem_size == 0 {
            self.ptr
        } else {
            unsafe { (self.ptr as *const u8).add(mid * self.elem_size) as *mut c_void }
        };

        (
            DeviceSlice {
                ptr: self.ptr,
                len: mid,
                type_id: self.type_id,
                elem_size: self.elem_size,
            },
            DeviceSlice {
                ptr: right_ptr,
                len: self.len - mid,
                type_id: self.type_id,
                elem_size: self.elem_size,
            },
        )
    }
}

impl<'a> Drop for ColumnView<'a> {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_column_view_destroy(self.handle.as_ptr());
        }
    }
}
