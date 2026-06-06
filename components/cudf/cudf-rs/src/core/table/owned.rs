use std::{
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use smallvec::SmallVec;

use super::{TABLE_HANDLE_INLINE_CAPACITY, TableView};
use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorKind, Result},
    },
    runtime::execution_domain::ExecutionDomain,
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
                if !ExecutionDomain::same_provenance(&domain, column.execution_domain()) {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "all columns in a table must share one retained provenance",
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
        let mut handles: SmallVec<[*mut c_void; TABLE_HANDLE_INLINE_CAPACITY]> =
            columns.into_iter().map(|c| c.into_raw()).collect();

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
        let exec_options = self.domain.explicit_exec_options_raw();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe { cudf_sys::cudf_table_copy_ex(self.handle.as_ptr(), options, &mut handle) },
                None => unsafe { cudf_sys::cudf_table_copy(self.handle.as_ptr(), &mut handle) },
            },
            "table::copy",
        )?;
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
        Ok(TableView::from_raw_handle(view_handle, Arc::clone(&self.domain)))
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

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
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
