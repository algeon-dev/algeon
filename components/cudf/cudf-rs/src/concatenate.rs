//! GPU-accelerated concatenation of columns and tables.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::ptr;

/// Concatenate multiple columns into a single column.
///
/// All columns must have the same data type.
pub fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let handles: Vec<*mut std::os::raw::c_void> =
        columns.iter().map(|c| c.raw_handle()).collect();
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_concatenate_columns(
                handles.as_ptr(),
                handles.len() as i32,
                &mut out,
            )
        },
        "concatenate::concatenate_columns",
        [("num_columns", handles.len().to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Concatenate multiple tables into a single table.
///
/// All tables must have the same number of columns with matching types.
pub fn concatenate_tables(tables: &[&TableView<'_>]) -> Result<Table> {
    let handles: Vec<*mut std::os::raw::c_void> =
        tables.iter().map(|t| t.raw_handle()).collect();
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_concatenate_tables(
                handles.as_ptr(),
                handles.len() as i32,
                &mut out,
            )
        },
        "concatenate::concatenate_tables",
        [("num_tables", handles.len().to_string())],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}
