use std::{
    any::Any,
    ptr::{self, NonNull},
    sync::Arc,
};

use smallvec::SmallVec;

use super::{TABLE_HANDLE_INLINE_CAPACITY, table_column_count_limit_error, view::TableView};
use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
        native_teardown::drop_device_owned_handle,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// An owned GPU table backed by a `cudf::table`.
///
/// A table is an ordered collection of columns of equal length. The
/// underlying device memory is freed when this value is dropped.
pub struct Table {
    handle: NonNull<cudf_sys::cudf_table_handle_state>,
    domain: Arc<ExecutionDomain>,
    retained_owners: Option<Vec<Option<Arc<dyn Any + Send + Sync>>>>,
}

// SAFETY: Table owns a GPU table via an opaque C++ handle. The handle is not
// thread-local. Send is safe because ownership transfer between threads is
// supported by the CUDA runtime. Sync is safe because all methods that observe
// state create independent FFI view handles internally.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    fn empty_requires_runtime_error(operation: &'static str) -> Error {
        Error::new_with(
            ErrorCode::EmptyConstructionRequiresRuntime,
            operation,
            "creating an empty table requires an explicit runtime because no authoritative input columns exist",
        )
        .with_detail("remedy", "use Runtime::empty_table()")
    }

    fn validate_uniform_row_counts(columns: &[Column], operation: &'static str) -> Result<()> {
        let Some(first) = columns.first() else {
            return Ok(());
        };

        let first_rows = first
            .view()
            .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, 0))?
            .len()
            .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, 0))?;

        for (index, column) in columns.iter().enumerate().skip(1) {
            let rows = column
                .view()
                .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, index))?
                .len()
                .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, index))?;
            if rows != first_rows {
                return Err(error::row_count_mismatch_error(operation, index, rows, first_rows));
            }
        }

        Ok(())
    }

    fn from_handle_with_domain(handle: cudf_sys::cudf_table_handle, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_handle_with_domain_and_owners(handle, domain, None)
    }

    fn from_handle_with_domain_and_owners(
        handle: cudf_sys::cudf_table_handle,
        domain: Arc<ExecutionDomain>,
        retained_owners: Option<Vec<Option<Arc<dyn Any + Send + Sync>>>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: error::require_non_null(handle, "table::from_handle_with_domain_and_owners", "table")?,
            domain,
            retained_owners,
        })
    }

    fn from_columns_impl(
        columns: Vec<Column>,
        authoritative_domain: Option<Arc<ExecutionDomain>>,
        explicit_num_rows: Option<usize>,
        operation: &'static str,
    ) -> Result<Self> {
        let domain = if let Some(first) = columns.first() {
            let domain = Arc::clone(first.execution_domain());
            for (index, column) in columns.iter().enumerate().skip(1) {
                ExecutionDomain::require_same_provenance(
                    &domain,
                    column.execution_domain(),
                    operation,
                    "all columns in a table must share one retained provenance",
                    "use copy_into_runtime(...) on higher-level Series/DataFrame values or rebuild the table from columns already placed in one Runtime before calling this operation",
                )
                .map_err(|e| e.with_fact(FACT_COLUMN_INDEX, index))?;
            }
            if let Some(authoritative_domain) = authoritative_domain.as_ref() {
                ExecutionDomain::require_same_provenance(
                    authoritative_domain,
                    &domain,
                    operation,
                    "table construction requires all columns to match the authoritative retained provenance",
                    "use copy_into_runtime(...) to relocate all columns into one Runtime before constructing the table",
                )?;
                Arc::clone(authoritative_domain)
            } else {
                domain
            }
        } else if let Some(authoritative_domain) = authoritative_domain {
            authoritative_domain
        } else {
            return Err(Self::empty_requires_runtime_error(operation));
        };

        Self::validate_uniform_row_counts(&columns, operation)?;

        if let (Some(expected_rows), Some(first)) = (explicit_num_rows, columns.first()) {
            let actual_rows = first
                .view()
                .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, 0))?
                .len()
                .map_err(|e| e.reanchor(operation).with_fact(FACT_COLUMN_INDEX, 0))?;
            if actual_rows != expected_rows {
                return Err(error::row_count_mismatch_error(operation, 0, actual_rows, expected_rows));
            }
        }

        let explicit_num_rows_i32 =
            explicit_num_rows.map(|rows| error::usize_to_i32(rows, operation, "row_count")).transpose()?;

        let num_columns = i32::try_from(columns.len()).map_err(|_| {
            table_column_count_limit_error(operation, "number of columns exceeds i32::MAX", columns.len())
        })?;
        let mut retained_owners = Vec::with_capacity(columns.len());
        let mut handles: SmallVec<[cudf_sys::cudf_column_handle; TABLE_HANDLE_INLINE_CAPACITY]> =
            SmallVec::with_capacity(columns.len());
        for column in columns {
            let (handle, retained_owner) = column.into_raw_parts();
            handles.push(handle);
            retained_owners.push(retained_owner);
        }
        let table_retained_owners = retained_owners.iter().any(Option::is_some).then_some(retained_owners);

        let mut table_handle = ptr::null_mut();
        // SAFETY: handles contains valid column pointers from into_raw(). num_columns matches.
        let status = unsafe {
            match explicit_num_rows_i32 {
                Some(num_rows) => cudf_sys::cudf_table_create_with_num_rows(
                    handles.as_mut_ptr(),
                    num_columns,
                    num_rows,
                    &mut table_handle,
                ),
                None => cudf_sys::cudf_table_create(handles.as_mut_ptr(), num_columns, &mut table_handle),
            }
        };
        // If creation failed, we've already released ownership of the columns
        // into the handles vec. The C shim nulls them out on success, but on
        // failure we need to destroy them ourselves.
        if status != cudf_sys::CUDF_SUCCESS {
            for &h in &handles {
                if !h.is_null() {
                    drop_device_owned_handle(
                        h,
                        domain.device_id(),
                        "table::from_columns::cleanup_column",
                        |owned| unsafe { cudf_sys::cudf_column_destroy(owned) },
                    );
                }
            }
            error::check_status_in(status, operation)?;
        }

        Self::from_handle_with_domain_and_owners(table_handle, domain, table_retained_owners)
    }

    /// Create a table from a vector of columns.
    ///
    /// The columns are *moved* into the table; do not use them afterwards.
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        Self::from_columns_impl(columns, None, None, "table::from_columns")
    }

    pub(crate) fn from_columns_in_domain_with_num_rows(
        columns: Vec<Column>,
        num_rows: usize,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        Self::from_columns_impl(columns, Some(domain), Some(num_rows), operation)
    }

    pub(crate) fn empty_in_domain(domain: Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        Self::from_columns_in_domain_with_num_rows(Vec::new(), 0, domain, operation)
    }

    /// Deep-copy this table.
    pub fn deep_copy(&self) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = self.domain.enter_execution()?;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_copy_ex(self.handle.as_ptr(), exec_options.options(), &mut handle) },
            "table::deep_copy",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(&self.domain))
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> Result<usize> {
        self.view()?.num_columns()
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> Result<usize> {
        self.view()?.num_rows()
    }

    /// Create an internal immutable view handle of this table.
    pub(crate) fn view(&self) -> Result<TableView<'_>> {
        let mut view_handle = ptr::null_mut();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_table_view(self.handle.as_ptr(), &mut view_handle) },
            "table::view",
        )?;
        TableView::from_raw_handle(view_handle, Arc::clone(&self.domain))
    }

    pub(crate) unsafe fn from_raw_with_domain(
        handle: cudf_sys::cudf_table_handle,
        domain: Arc<ExecutionDomain>,
    ) -> Result<Self> {
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    /// Consume this table and return its columns as owned [`Column`] values.
    ///
    /// The table is destroyed in the process. The returned columns are in
    /// the same order as they were when the table was created.
    ///
    /// This is a zero-copy operation on the GPU; column device memory is
    /// transferred, not copied.
    pub fn into_columns(self) -> Result<Vec<Column>> {
        let this = std::mem::ManuallyDrop::new(self);
        let Self { handle: _, domain: _, retained_owners: _ } = &*this;
        let raw_handle = this.handle.as_ptr();
        let domain = unsafe { std::ptr::read(&this.domain) };
        let retained_owners = unsafe { std::ptr::read(&this.retained_owners) };

        let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
        let mut out_count: i32 = 0;

        // SAFETY: raw_handle is valid (we just read it from self before forget).
        let status = unsafe { cudf_sys::cudf_table_release_columns(raw_handle, &mut out_columns, &mut out_count) };

        if status != cudf_sys::CUDF_SUCCESS {
            let primary_error = error::check_status_in(status, "table::into_columns").unwrap_err();
            // On failure, the C shim did NOT consume the table.
            // We must destroy it ourselves to avoid a leak.
            drop_device_owned_handle(
                raw_handle,
                domain.device_id(),
                "table::into_columns::cleanup_table",
                |owned| unsafe { cudf_sys::cudf_table_destroy(owned) },
            );
            return Err(primary_error);
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
        let owners = retained_owners.unwrap_or_else(|| vec![None; count]);
        debug_assert_eq!(owners.len(), count, "retained owners count mismatch");
        let columns: Vec<Column> = col_ptrs
            .iter()
            .zip(owners)
            .map(|(&ptr, retained_owner)| match retained_owner {
                Some(retained_owner) => unsafe {
                    Column::from_raw_with_domain_and_owner(ptr, Arc::clone(&domain), retained_owner)
                },
                None => unsafe { Column::from_raw_with_domain(ptr, Arc::clone(&domain)) },
            })
            .collect::<Result<Vec<_>>>()?;

        // Free the array (not the columns; they are now owned by Column).
        // SAFETY: out_columns was allocated by the C shim for the column pointer array.
        unsafe { cudf_sys::cudf_free_column_array(out_columns) };

        Ok(columns)
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

impl Drop for Table {
    fn drop(&mut self) {
        drop_device_owned_handle(self.handle.as_ptr(), self.domain.device_id(), "Table::drop", |owned| unsafe {
            cudf_sys::cudf_table_destroy(owned)
        });
    }
}
