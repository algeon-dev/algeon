//! GPU-accelerated concatenation of columns and tables.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{column::Column, concatenate};
//!
//! let a = Column::from_slice(&[1i32, 2]).unwrap();
//! let b = Column::from_slice(&[3i32, 4, 5]).unwrap();
//!
//! let combined = concatenate::concatenate_columns(&[
//!     &a.view().unwrap(),
//!     &b.view().unwrap(),
//! ])
//! .unwrap();
//! // combined: [1, 2, 3, 4, 5]
//! ```

use std::ptr;

use crate::{
    column::{Column, ColumnView},
    error::{self, Result},
    table::{Table, TableView},
};

/// Concatenate multiple columns into a single column.
///
/// All columns must have the same data type.
pub fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let handles: Vec<*mut std::os::raw::c_void> = columns.iter().map(|c| c.raw_handle()).collect();
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All column view handles are valid (NonNull maintained by ColumnView).
        // handles slice is valid for handles.len() elements.
        unsafe { cudf_sys::cudf_concatenate_columns(handles.as_ptr(), handles.len() as i32, &mut out) },
        "concatenate::concatenate_columns",
        [("num_columns", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Concatenate multiple tables into a single table.
///
/// All tables must have the same number of columns with matching types.
pub fn concatenate_tables(tables: &[&TableView<'_>]) -> Result<Table> {
    let handles: Vec<*mut std::os::raw::c_void> = tables.iter().map(|t| t.raw_handle()).collect();
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All table view handles are valid (NonNull maintained by TableView).
        // handles slice is valid for handles.len() elements.
        unsafe { cudf_sys::cudf_concatenate_tables(handles.as_ptr(), handles.len() as i32, &mut out) },
        "concatenate::concatenate_tables",
        [("num_tables", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw(out) })
}
