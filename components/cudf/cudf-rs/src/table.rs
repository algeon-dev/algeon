//! GPU table types: owned `Table` and borrowed `TableView`.

use std::{
    marker::PhantomData,
    os::raw::c_void,
    ptr::{self, NonNull},
};

use crate::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
};

/// An owned GPU table backed by a `cudf::table`.
///
/// A table is an ordered collection of columns of equal length. The
/// underlying device memory is freed when this value is dropped.
pub struct Table {
    handle: NonNull<c_void>,
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    /// Create a table from a vector of columns.
    ///
    /// The columns are *moved* into the table — do not use them afterwards.
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        let num_columns = i32::try_from(columns.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "number of columns exceeds i32::MAX")
                .with_operation("table::from_columns")
                .with_context("num_columns", columns.len().to_string())
        })?;
        let mut handles: Vec<*mut c_void> = columns.into_iter().map(|c| c.into_raw()).collect();

        let mut table_handle = ptr::null_mut();
        let status = unsafe { cudf_sys::cudf_table_create(handles.as_mut_ptr(), num_columns, &mut table_handle) };
        // If creation failed, we've already released ownership of the columns
        // into the handles vec. The C shim nulls them out on success, but on
        // failure we need to destroy them ourselves.
        if status != cudf_sys::CUDF_SUCCESS {
            for &h in &handles {
                if !h.is_null() {
                    unsafe { cudf_sys::cudf_column_destroy(h) };
                }
            }
            error::check_status_in(status, "table::from_columns")?;
        }

        Ok(Self { handle: NonNull::new(table_handle).expect("null handle on success") })
    }

    /// Deep-copy this table.
    pub fn copy(&self) -> Result<Self> {
        let mut handle = ptr::null_mut();
        error::check_status_in(unsafe { cudf_sys::cudf_table_copy(self.handle.as_ptr(), &mut handle) }, "table::copy")?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create an immutable view of this table.
    pub fn view(&self) -> Result<TableView<'_>> {
        let mut view_handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view(self.handle.as_ptr(), &mut view_handle) },
            "table::view",
        )?;
        Ok(TableView { handle: NonNull::new(view_handle).expect("null handle on success"), _lifetime: PhantomData })
    }

    /// Construct a `Table` from a raw FFI handle, taking ownership.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_table_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw(handle: *mut c_void) -> Self {
        Self { handle: NonNull::new(handle).expect("null table handle") }
    }

    /// Consume this table and return its columns as owned [`Column`] values.
    ///
    /// The table is destroyed in the process. The returned columns are in
    /// the same order as they were when the table was created.
    ///
    /// This is a zero-copy operation on the GPU — column device memory is
    /// transferred, not copied.
    pub fn into_columns(self) -> Result<Vec<Column>> {
        let raw_handle = self.handle.as_ptr();
        // Prevent the Drop impl from destroying the table — ownership
        // is transferred to the C shim which will consume it.
        std::mem::forget(self);

        let mut out_columns: *mut *mut c_void = ptr::null_mut();
        let mut out_count: i32 = 0;

        let status = unsafe { cudf_sys::cudf_table_release_columns(raw_handle, &mut out_columns, &mut out_count) };

        if status != cudf_sys::CUDF_SUCCESS {
            // On failure, the C shim did NOT consume the table.
            // We must destroy it ourselves to avoid a leak.
            unsafe { cudf_sys::cudf_table_destroy(raw_handle) };
            error::check_status_in(status, "table::into_columns")?;
            unreachable!();
        }

        if out_count == 0 {
            return Ok(Vec::new());
        }

        let count = out_count as usize;
        let col_ptrs = unsafe { std::slice::from_raw_parts(out_columns, count) };

        let columns: Vec<Column> = col_ptrs.iter().map(|&ptr| unsafe { Column::from_raw(ptr) }).collect();

        // Free the array (not the columns — they are now owned by Column).
        unsafe { cudf_sys::cudf_free_column_array(out_columns) };

        Ok(columns)
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_table_destroy(self.handle.as_ptr());
        }
    }
}

/// A non-owning, immutable view of a GPU table.
pub struct TableView<'a> {
    handle: NonNull<c_void>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> TableView<'a> {
    /// Returns the number of columns.
    pub fn num_columns(&self) -> Result<i32> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_num_columns(self.handle.as_ptr(), &mut out) },
            "table_view::num_columns",
        )?;
        Ok(out)
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> Result<i32> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_num_rows(self.handle.as_ptr(), &mut out) },
            "table_view::num_rows",
        )?;
        Ok(out)
    }

    /// Get a column view by index.
    pub fn column(&self, index: i32) -> Result<ColumnView<'a>> {
        if index < 0 {
            return Err(Error::new(ErrorKind::InvalidOperation, "table column index must be >= 0")
                .with_operation("table_view::column")
                .with_context("column_index", index.to_string()));
        }

        let mut col_handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_column(self.handle.as_ptr(), index, &mut col_handle) },
            "table_view::column",
        )?;
        Ok(unsafe { ColumnView::from_raw(col_handle) })
    }

    /// Returns the raw FFI handle. For internal I/O use.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }
}

impl<'a> Drop for TableView<'a> {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_table_view_destroy(self.handle.as_ptr());
        }
    }
}
