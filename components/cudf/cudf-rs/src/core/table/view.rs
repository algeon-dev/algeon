use std::{
    marker::PhantomData,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use smallvec::SmallVec;

use super::TABLE_HANDLE_INLINE_CAPACITY;
use crate::{
    core::{
        column::ColumnView,
        error::{self, Error, ErrorKind, Result},
    },
    runtime::execution_domain::ExecutionDomain,
};

/// A non-owning, immutable view of a GPU table.
pub struct TableView<'a> {
    handle: NonNull<c_void>,
    domain: Arc<ExecutionDomain>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> TableView<'a> {
    pub(crate) fn from_raw_handle(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self { handle: NonNull::new(handle).expect("null handle on success"), domain, _lifetime: PhantomData }
    }

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
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_column(self.handle.as_ptr(), index, &mut col_handle) },
            "table_view::column",
        )?;
        // SAFETY: col_handle was returned by a successful FFI call.
        Ok(unsafe { ColumnView::from_raw_with_domain(col_handle, Arc::clone(&self.domain)) })
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

        Ok(Self::from_raw_handle(out, Arc::clone(&self.domain)))
    }

    /// Construct a `TableView` from a slice of `ColumnView`s.
    ///
    /// This is useful when you have individually-owned columns and need to
    /// present them as a table to libcudf operations.
    pub fn from_column_views(views: &[&ColumnView<'a>]) -> Result<Self> {
        let domain = if let Some(first) = views.first() {
            let domain = Arc::clone(first.execution_domain());
            for (index, view) in views.iter().enumerate().skip(1) {
                ExecutionDomain::require_same_provenance(
                    &domain,
                    view.execution_domain(),
                    "table_view::from_column_views",
                    "all column views in a table view must share one retained provenance",
                    "use copy_into_runtime(...) on higher-level Series/DataFrame values or rebuild the table view from columns already placed in one Runtime before calling this operation",
                )
                .map_err(|e| e.with_context("column_index", index.to_string()))?;
            }
            domain
        } else {
            ExecutionDomain::ambient_default()
        };

        let num_columns = i32::try_from(views.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "number of column views exceeds i32::MAX")
                .with_operation("table_view::from_column_views")
                .with_context("num_columns", views.len().to_string())
        })?;

        let mut handles: SmallVec<[*mut c_void; TABLE_HANDLE_INLINE_CAPACITY]> =
            views.iter().map(|v| v.raw_handle()).collect();

        let mut out = ptr::null_mut();
        // SAFETY: handles are valid column view pointers from raw_handle(). num_columns matches.
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_from_column_views(handles.as_mut_ptr(), num_columns, &mut out) },
            "table_view::from_column_views",
        )?;

        Ok(Self::from_raw_handle(out, domain))
    }

    /// Returns the raw FFI handle. For internal I/O use.
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
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_mr_value(&self) -> usize {
        self.domain.debug_mr_value()
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
