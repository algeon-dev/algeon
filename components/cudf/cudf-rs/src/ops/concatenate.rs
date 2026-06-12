//! GPU-accelerated concatenation of columns and tables.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{Runtime, Series};
//!
//! let rt = Runtime::new().unwrap();
//! let a = rt.series("value", &[1i32, 2]).unwrap();
//! let b = rt.series("value", &[3i32, 4, 5]).unwrap();
//!
//! let combined = Series::concat(&[&a, &b]).unwrap();
//! assert_eq!(combined.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
//! ```

use std::ptr;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

fn checked_len_to_i32(len: usize, operation: &'static str, field: &'static str) -> Result<i32> {
    i32::try_from(len).map_err(|_| {
        Error::new(ErrorKind::InvalidArgument, format!("{field} exceeds i32::MAX"))
            .with_operation(operation)
            .with_context(field, len.to_string())
    })
}

fn concatenate_requires_input_error(operation: &'static str, input_kind: &'static str) -> Error {
    Error::new(ErrorKind::InvalidOperation, format!("{operation} requires at least one input {input_kind}"))
        .with_operation(operation)
}

fn compatible_column_domain(columns: &[&ColumnView<'_>]) -> Result<std::sync::Arc<ExecutionDomain>> {
    let Some(first) = columns.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_columns", "column"));
    };
    let output_domain = std::sync::Arc::clone(first.execution_domain());
    // Fence any input from a different execution domain so its GPU work
    // completes before the concatenation kernel reads its data.
    for col in &columns[1..] {
        ExecutionDomain::require_same_device_for_mixed_inputs(
            first.execution_domain(),
            col.execution_domain(),
            "concatenate::concatenate_columns",
        )?;
        if !ExecutionDomain::same_provenance(col.execution_domain(), &output_domain) {
            col.execution_domain().fence_stream("concatenate::concatenate_columns")?;
        }
    }
    Ok(output_domain)
}

fn compatible_table_domain(tables: &[&TableView<'_>]) -> Result<std::sync::Arc<ExecutionDomain>> {
    let Some(first) = tables.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_tables", "table"));
    };
    let output_domain = std::sync::Arc::clone(first.execution_domain());
    // Fence any input from a different execution domain so its GPU work
    // completes before the concatenation kernel reads its data.
    for tbl in &tables[1..] {
        ExecutionDomain::require_same_device_for_mixed_inputs(
            first.execution_domain(),
            tbl.execution_domain(),
            "concatenate::concatenate_tables",
        )?;
        if !ExecutionDomain::same_provenance(tbl.execution_domain(), &output_domain) {
            tbl.execution_domain().fence_stream("concatenate::concatenate_tables")?;
        }
    }
    Ok(output_domain)
}

/// Concatenate multiple columns into a single column.
///
/// All columns must have the same data type.
pub fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let output_domain = compatible_column_domain(columns)?;
    let exec_options = output_domain.explicit_exec_options_raw();
    concatenate_columns_with_exec(columns, &output_domain, exec_options.as_ref())
}

pub(crate) fn concatenate_columns_with_exec(
    columns: &[&ColumnView<'_>],
    output_domain: &std::sync::Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<Column> {
    let handles: Vec<*mut std::os::raw::c_void> = columns.iter().map(|c| c.raw_handle()).collect();
    let num_columns = checked_len_to_i32(handles.len(), "concatenate::concatenate_columns", "num_columns")?;
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All column view handles are valid (NonNull maintained by ColumnView).
        // handles slice is valid for handles.len() elements.
        match exec_options {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_concatenate_columns_ex(handles.as_ptr(), num_columns, exec_options, &mut out)
            },
            None => unsafe { cudf_sys::cudf_concatenate_columns(handles.as_ptr(), num_columns, &mut out) },
        },
        "concatenate::concatenate_columns",
        [("num_columns", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) })
}

/// Concatenate multiple tables into a single table.
///
/// All tables must have the same number of columns with matching types.
pub fn concatenate_tables(tables: &[&TableView<'_>]) -> Result<Table> {
    let output_domain = compatible_table_domain(tables)?;
    let exec_options = output_domain.explicit_exec_options_raw();
    concatenate_tables_with_exec(tables, &output_domain, exec_options.as_ref())
}

pub(crate) fn concatenate_tables_with_exec(
    tables: &[&TableView<'_>],
    output_domain: &std::sync::Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<Table> {
    let handles: Vec<*mut std::os::raw::c_void> = tables.iter().map(|t| t.raw_handle()).collect();
    let num_tables = checked_len_to_i32(handles.len(), "concatenate::concatenate_tables", "num_tables")?;
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All table view handles are valid (NonNull maintained by TableView).
        // handles slice is valid for handles.len() elements.
        match exec_options {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_concatenate_tables_ex(handles.as_ptr(), num_tables, exec_options, &mut out)
            },
            None => unsafe { cudf_sys::cudf_concatenate_tables(handles.as_ptr(), num_tables, &mut out) },
        },
        "concatenate::concatenate_tables",
        [("num_tables", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) })
}
