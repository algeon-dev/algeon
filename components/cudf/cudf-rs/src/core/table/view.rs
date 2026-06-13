use std::{
    marker::PhantomData,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use smallvec::SmallVec;

use super::{TABLE_HANDLE_INLINE_CAPACITY, table_column_count_limit_error};
use crate::{
    core::{
        column::ColumnView,
        error::{self, Error, ErrorCode, ErrorKind, Result},
    },
    runtime::execution_domain::ExecutionDomain,
};

#[allow(dead_code)]
enum TableViewBacking<'a> {
    None,
    ColumnViews(Vec<ColumnView<'a>>),
}

/// A non-owning, immutable view of a GPU table.
pub struct TableView<'a> {
    handle: NonNull<c_void>,
    domain: Arc<ExecutionDomain>,
    _backing: TableViewBacking<'a>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> TableView<'a> {
    fn foreign_negative_num_columns_error(observed: i32) -> Error {
        Error::new(ErrorKind::InvariantViolation, "FFI returned negative table column count")
            .with_code(ErrorCode::InvariantViolation)
            .with_operation("table_view::num_columns")
            .with_context("resource", "table_column_count")
            .with_context("reason", "invalid_foreign_table_metadata")
            .with_context("table_column_count", observed.to_string())
            .with_context("remedy", "ensure the foreign table view reports a non-negative column count")
    }

    fn foreign_negative_num_rows_error(observed: i32) -> Error {
        Error::new(ErrorKind::InvariantViolation, "FFI returned negative table row count")
            .with_code(ErrorCode::InvariantViolation)
            .with_operation("table_view::num_rows")
            .with_context("resource", "table_row_count")
            .with_context("reason", "invalid_foreign_table_metadata")
            .with_context("table_row_count", observed.to_string())
            .with_context("remedy", "ensure the foreign table view reports a non-negative row count")
    }

    fn empty_requires_authoritative_source_error(operation: &'static str) -> Error {
        Error::new(
            ErrorKind::InvalidInput,
            "creating an empty table view requires an authoritative source because no input column views exist",
        )
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", "empty_borrowed_table_view_requires_authoritative_source")
        .with_context(
            "remedy",
            "borrow at least one column view or call view() on an authoritative Table created by a Runtime",
        )
    }

    fn row_count_mismatch_error(
        operation: &'static str,
        column_index: usize,
        actual_rows: usize,
        expected_rows: usize,
    ) -> Error {
        Error::new(
            ErrorKind::InvalidInput,
            format!("column {column_index} has {actual_rows} rows but column 0 has {expected_rows} rows"),
        )
        .with_code(ErrorCode::ColumnLengthMismatch)
        .with_operation(operation)
        .with_context("column_index", column_index.to_string())
        .with_context("actual_rows", actual_rows.to_string())
        .with_context("expected_rows", expected_rows.to_string())
    }

    fn validate_uniform_row_counts(views: &[ColumnView<'a>], operation: &'static str) -> Result<()> {
        let Some(first) = views.first() else {
            return Ok(());
        };

        let first_rows = first.len().map_err(|e| e.with_operation(operation).with_context("column_index", "0"))?;

        for (index, view) in views.iter().enumerate().skip(1) {
            let rows =
                view.len().map_err(|e| e.with_operation(operation).with_context("column_index", index.to_string()))?;
            if rows != first_rows {
                return Err(Self::row_count_mismatch_error(operation, index, rows, first_rows));
            }
        }

        Ok(())
    }

    fn from_raw_handle_with_backing(
        handle: *mut c_void,
        domain: Arc<ExecutionDomain>,
        backing: TableViewBacking<'a>,
    ) -> Result<Self> {
        Ok(Self {
            handle: error::require_non_null(handle, "table_view::from_raw_handle_with_backing", "table view")?,
            domain,
            _backing: backing,
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn from_raw_handle(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_raw_handle_with_backing(handle, domain, TableViewBacking::None)
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_num_columns(self.handle.as_ptr(), &mut out) },
            "table_view::num_columns",
        )?;
        usize::try_from(out).map_err(|_| Self::foreign_negative_num_columns_error(out))
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> Result<usize> {
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_num_rows(self.handle.as_ptr(), &mut out) },
            "table_view::num_rows",
        )?;
        usize::try_from(out).map_err(|_| Self::foreign_negative_num_rows_error(out))
    }

    /// Get a column view by index.
    pub fn column(&self, index: usize) -> Result<ColumnView<'a>> {
        let num_columns = self.num_columns()?;
        if index >= num_columns {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("table column index {} out of bounds for {} columns", index, num_columns),
            )
            .with_code(ErrorCode::IndexOutOfBounds)
            .with_operation("table_view::column")
            .with_context("column_index", index.to_string())
            .with_context("table_column_count", num_columns.to_string()));
        }

        let index = i32::try_from(index).map_err(|_| {
            Error::new(ErrorKind::InvalidInput, "table column index exceeds i32::MAX")
                .with_operation("table_view::column")
                .with_context("column_index", index.to_string())
        })?;

        let mut col_handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_column(self.handle.as_ptr(), index, &mut col_handle) },
            "table_view::column",
        )?;
        // SAFETY: col_handle was returned by a successful FFI call.
        unsafe { ColumnView::from_raw_with_domain(col_handle, Arc::clone(&self.domain)) }
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
        let num_columns = self.num_columns()?;

        let indices_i32: Vec<i32> = indices
            .iter()
            .map(|&index| {
                if index >= num_columns {
                    Err(Error::new(
                        ErrorKind::InvalidInput,
                        format!("table column index {} out of bounds for {} columns", index, num_columns),
                    )
                    .with_code(ErrorCode::IndexOutOfBounds)
                    .with_operation("table_view::select_columns")
                    .with_context("column_index", index.to_string())
                    .with_context("table_column_count", num_columns.to_string()))
                } else {
                    i32::try_from(index).map_err(|_| {
                        Error::new(ErrorKind::InvalidInput, "table column index exceeds i32::MAX")
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
                    error::usize_to_i32(indices_i32.len(), "table_view::select_columns", "num_selected_columns")?,
                    &mut out,
                )
            },
            "table_view::select_columns",
        )?;

        Self::from_raw_handle(out, Arc::clone(&self.domain))
    }

    pub(crate) fn from_borrowed_column_views_in_domain(
        views: Vec<ColumnView<'a>>,
        authoritative_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let domain = if let Some(first) = views.first() {
            let domain = Arc::clone(first.execution_domain());
            for (index, view) in views.iter().enumerate().skip(1) {
                ExecutionDomain::require_same_provenance(
                    &domain,
                    view.execution_domain(),
                    operation,
                    "all column views in a table view must share one retained provenance",
                    "use copy_into_runtime(...) on higher-level Series/DataFrame values or rebuild the table view from columns already placed in one Runtime before calling this operation",
                )
                .map_err(|e| e.with_context("column_index", index.to_string()))?;
            }
            if let Some(authoritative_domain) = authoritative_domain.as_ref() {
                ExecutionDomain::require_same_provenance(
                    authoritative_domain,
                    &domain,
                    operation,
                    "table-view construction requires all columns to match the authoritative retained provenance",
                    "use copy_into_runtime(...) to relocate all columns into one Runtime before constructing the table view",
                )?;
                Arc::clone(authoritative_domain)
            } else {
                domain
            }
        } else if let Some(authoritative_domain) = authoritative_domain {
            authoritative_domain
        } else {
            return Err(Self::empty_requires_authoritative_source_error(operation));
        };

        Self::validate_uniform_row_counts(&views, operation)?;

        let num_columns = i32::try_from(views.len()).map_err(|_| {
            table_column_count_limit_error(operation, "number of column views exceeds i32::MAX", views.len())
        })?;

        let mut handles: SmallVec<[*mut c_void; TABLE_HANDLE_INLINE_CAPACITY]> =
            views.iter().map(|v| v.raw_handle()).collect();

        let mut out = ptr::null_mut();
        // SAFETY: handles are valid column view pointers from raw_handle(). num_columns matches.
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view_from_column_views(handles.as_mut_ptr(), num_columns, &mut out) },
            operation,
        )?;

        Self::from_raw_handle_with_backing(out, domain, TableViewBacking::ColumnViews(views))
    }

    pub(crate) fn from_column_views_in_domain(
        views: &[&ColumnView<'a>],
        authoritative_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let retained_views: Vec<ColumnView<'a>> =
            views.iter().map(|view| view.duplicate_with_operation(operation)).collect::<Result<Vec<_>>>()?;
        Self::from_borrowed_column_views_in_domain(retained_views, authoritative_domain, operation)
    }

    /// Construct a `TableView` from a slice of `ColumnView`s.
    ///
    /// This is useful when you have individually-owned columns and need to
    /// present them as a table to libcudf operations.
    pub fn from_column_views(views: &[&ColumnView<'a>]) -> Result<Self> {
        Self::from_column_views_in_domain(views, None, "table_view::from_column_views")
    }

    /// Returns the raw FFI handle. For internal I/O use.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
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
