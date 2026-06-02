//! GPU table types: owned `Table` and borrowed `TableView`.
//!
//! **Advanced/Core.** Most users should use [`DataFrame`](crate::DataFrame) /
//! [`DataFrameView`](crate::DataFrameView).

use std::{
    marker::PhantomData,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
    execution_domain::ExecutionDomain,
};

/// An owned GPU table backed by a `cudf::table`.
///
/// A table is an ordered collection of columns of equal length. The
/// underlying device memory is freed when this value is dropped.
pub struct Table {
    handle: NonNull<c_void>,
    domain: Arc<ExecutionDomain>,
}

// SAFETY: Table owns a GPU table via an opaque C++ handle. The handle is not
// thread-local. Send is safe because ownership transfer between threads is
// supported by the CUDA runtime. Sync is safe because all methods that observe
// state (view(), num_columns()) create independent FFI view handles.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    fn from_handle_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self { handle: NonNull::new(handle).expect("null table handle"), domain }
    }

    /// Create a table from a vector of columns.
    ///
    /// The columns are *moved* into the table — do not use them afterwards.
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        let domain = if let Some(first) = columns.first() {
            let domain = Arc::clone(first.execution_domain());
            for (index, column) in columns.iter().enumerate().skip(1) {
                if !ExecutionDomain::same_identity(&domain, column.execution_domain()) {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "all columns in a table must share one execution domain",
                    )
                    .with_operation("table::from_columns")
                    .with_context("left_domain", domain.debug_label())
                    .with_context("right_domain", column.execution_domain().debug_label())
                    .with_context("column_index", index.to_string()));
                }
            }
            domain
        } else {
            ExecutionDomain::ambient_default()
        };

        let num_columns = i32::try_from(columns.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "number of columns exceeds i32::MAX")
                .with_operation("table::from_columns")
                .with_context("num_columns", columns.len().to_string())
        })?;
        let mut handles: Vec<*mut c_void> = columns.into_iter().map(|c| c.into_raw()).collect();

        let mut table_handle = ptr::null_mut();
        // SAFETY: handles contains valid column pointers from into_raw(). num_columns matches.
        let status = unsafe { cudf_sys::cudf_table_create(handles.as_mut_ptr(), num_columns, &mut table_handle) };
        // If creation failed, we've already released ownership of the columns
        // into the handles vec. The C shim nulls them out on success, but on
        // failure we need to destroy them ourselves.
        if status != cudf_sys::CUDF_SUCCESS {
            for &h in &handles {
                if !h.is_null() {
                    // SAFETY: Each handle was valid from into_raw(). On failure
                    // the C shim did not consume them, so we must clean up.
                    unsafe { cudf_sys::cudf_column_destroy(h) };
                }
            }
            error::check_status_in(status, "table::from_columns")?;
        }

        Ok(Self::from_handle_with_domain(table_handle, domain))
    }

    /// Deep-copy this table.
    pub fn copy(&self) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(unsafe { cudf_sys::cudf_table_copy(self.handle.as_ptr(), &mut handle) }, "table::copy")?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(&self.domain)))
    }

    /// Create an immutable view of this table.
    pub fn view(&self) -> Result<TableView<'_>> {
        let mut view_handle = ptr::null_mut();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
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
        Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default())
    }

    pub(crate) fn with_execution_domain(mut self, domain: Arc<ExecutionDomain>) -> Self {
        self.domain = domain;
        self
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
        let domain = Arc::clone(&self.domain);
        // Prevent the Drop impl from destroying the table — ownership
        // is transferred to the C shim which will consume it.
        std::mem::forget(self);

        let mut out_columns: *mut *mut c_void = ptr::null_mut();
        let mut out_count: i32 = 0;

        // SAFETY: raw_handle is valid (we just read it from self before forget).
        let status = unsafe { cudf_sys::cudf_table_release_columns(raw_handle, &mut out_columns, &mut out_count) };

        if status != cudf_sys::CUDF_SUCCESS {
            // On failure, the C shim did NOT consume the table.
            // We must destroy it ourselves to avoid a leak.
            // SAFETY: raw_handle is still valid because the C shim did not consume it.
            unsafe { cudf_sys::cudf_table_destroy(raw_handle) };
            error::check_status_in(status, "table::into_columns")?;
            unreachable!();
        }

        if out_count == 0 {
            return Ok(Vec::new());
        }

        let count = out_count as usize;
        // SAFETY: out_columns and count are returned by a successful FFI call.
        // The pointer is valid for count elements.
        let col_ptrs = unsafe { std::slice::from_raw_parts(out_columns, count) };

        // SAFETY: Each pointer was returned by the successful FFI call and
        // represents an owned column handle being transferred to Column RAII.
        let columns: Vec<Column> =
            col_ptrs.iter().map(|&ptr| unsafe { Column::from_raw_with_domain(ptr, Arc::clone(&domain)) }).collect();

        // Free the array (not the columns — they are now owned by Column).
        // SAFETY: out_columns was allocated by the C shim for the column pointer array.
        unsafe { cudf_sys::cudf_free_column_array(out_columns) };

        Ok(columns)
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once, so the handle is destroyed exactly once.
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
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_num_columns(self.handle.as_ptr(), &mut out) },
            "table_view::num_columns",
        )?;
        Ok(out)
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> Result<i32> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
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
        // SAFETY: self.handle is valid. index >= 0 is checked above.
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_column(self.handle.as_ptr(), index, &mut col_handle) },
            "table_view::column",
        )?;
        // SAFETY: col_handle was returned by a successful FFI call.
        Ok(unsafe { ColumnView::from_raw(col_handle) })
    }

    /// Build a projected table view by selecting a subset of columns.
    ///
    /// The returned view borrows the same underlying device columns as `self`.
    ///
    /// # Cost
    ///
    /// Zero-copy with respect to device data: this allocates only a new
    /// table-view handle over the selected columns.
    pub fn select_columns(&self, indices: &[usize]) -> Result<Self> {
        let num_columns = usize::try_from(self.num_columns()?).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "number of table columns must be >= 0")
                .with_operation("table_view::select_columns")
        })?;

        let indices_i32: Vec<i32> = indices
            .iter()
            .map(|&index| {
                if index >= num_columns {
                    Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("table column index {} out of bounds for {} columns", index, num_columns),
                    )
                    .with_operation("table_view::select_columns")
                    .with_context("column_index", index.to_string())
                    .with_context("num_columns", num_columns.to_string()))
                } else {
                    i32::try_from(index).map_err(|_| {
                        Error::new(ErrorKind::InvalidOperation, "table column index exceeds i32::MAX")
                            .with_operation("table_view::select_columns")
                            .with_context("column_index", index.to_string())
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let mut out = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_table_view_select_columns(
                    self.handle.as_ptr(),
                    indices_i32.as_ptr(),
                    indices_i32.len() as i32,
                    &mut out,
                )
            },
            "table_view::select_columns",
        )?;

        Ok(TableView { handle: NonNull::new(out).expect("null handle on success"), _lifetime: PhantomData })
    }

    /// Construct a `TableView` from a slice of `ColumnView`s.
    ///
    /// This is useful when you have individually-owned columns and need to
    /// present them as a table to libcudf operations.
    pub fn from_column_views(views: &[&ColumnView<'a>]) -> Result<Self> {
        let num_columns = i32::try_from(views.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "number of column views exceeds i32::MAX")
                .with_operation("table_view::from_column_views")
                .with_context("num_columns", views.len().to_string())
        })?;

        let mut handles: Vec<*mut c_void> = views.iter().map(|v| v.raw_handle()).collect();

        let mut out = ptr::null_mut();
        // SAFETY: handles are valid column view pointers from raw_handle(). num_columns matches.
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_from_column_views(handles.as_mut_ptr(), num_columns, &mut out) },
            "table_view::from_column_views",
        )?;

        Ok(TableView { handle: NonNull::new(out).expect("null handle on success"), _lifetime: PhantomData })
    }

    /// Returns the raw FFI handle. For internal I/O use.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }
}

impl<'a> Drop for TableView<'a> {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once, so the handle is destroyed exactly once.
        unsafe {
            cudf_sys::cudf_table_view_destroy(self.handle.as_ptr());
        }
    }
}
