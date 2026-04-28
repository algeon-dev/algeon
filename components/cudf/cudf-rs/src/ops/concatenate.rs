//! GPU-accelerated concatenation of columns and tables.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{Runtime, Series};
//!
//! let rt = Runtime::new().unwrap();
//! let a = rt.series(&[1i32, 2]).unwrap();
//! let b = rt.series(&[3i32, 4, 5]).unwrap();
//!
//! let combined = Series::concat(&[&a, &b]).unwrap();
//! assert_eq!(combined.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
//! ```

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

fn checked_len_to_i32(
    len: usize,
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    value_key: &'static str,
    remedy: &'static str,
) -> Result<i32> {
    i32::try_from(len)
        .map_err(|_| error::limit_validation_error(operation, message, resource, value_key, "i32::MAX", len, remedy))
}

fn input_column_count_limit_error(operation: &'static str, column_count: usize) -> Error {
    error::limit_validation_error(
        operation,
        "number of input columns exceeds i32::MAX",
        "input_columns",
        "column_count",
        "i32::MAX",
        column_count,
        "reduce column_count to fit within i32::MAX",
    )
}

fn input_table_count_limit_error(operation: &'static str, table_count: usize) -> Error {
    error::limit_validation_error(
        operation,
        "number of input tables exceeds i32::MAX",
        "input_tables",
        "table_count",
        "i32::MAX",
        table_count,
        "reduce table_count to fit within i32::MAX",
    )
}

fn concatenate_requires_input_error(operation: &'static str, input_kind: &'static str) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        format!("{operation} requires at least one input {input_kind}"),
    )
    .with_detail("reason", "empty_input")
    .with_detail("remedy", format!("provide at least one input {input_kind}"))
}

fn validate_column_types(columns: &[&ColumnView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = columns.split_first() else {
        return Ok(());
    };
    let left_type = first.data_type().map_err(|e| e.reanchor(operation))?.id();

    for (offset, column) in rest.iter().enumerate() {
        let right_type = column.data_type().map_err(|e| e.reanchor(operation))?.id();
        if left_type != right_type {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                "all input columns must have the same data type for concatenation",
            )
            .with_fact(FACT_COLUMN_INDEX, offset + 1)
            .with_fact(FACT_LEFT_TYPE_ID, left_type)
            .with_fact(FACT_RIGHT_TYPE_ID, right_type));
        }
    }

    Ok(())
}

fn validate_table_column_types(tables: &[&TableView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = tables.split_first() else {
        return Ok(());
    };
    let column_count = first.num_columns().map_err(|e| e.reanchor(operation))?;

    for table in rest {
        let table_column_count = table.num_columns().map_err(|e| e.reanchor(operation))?;
        if table_column_count != column_count {
            return Ok(());
        }

        for index in 0..column_count {
            let left_type =
                first.column(index).and_then(|column| column.data_type()).map_err(|e| e.reanchor(operation))?.id();
            let right_type =
                table.column(index).and_then(|column| column.data_type()).map_err(|e| e.reanchor(operation))?.id();

            if left_type != right_type {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "all input tables must have matching column data types for concatenation",
                )
                .with_fact(FACT_COLUMN_INDEX, index)
                .with_fact(FACT_LEFT_TYPE_ID, left_type)
                .with_fact(FACT_RIGHT_TYPE_ID, right_type));
            }
        }
    }

    Ok(())
}

fn compatible_column_execution(columns: &[&ColumnView<'_>]) -> Result<ResolvedExecution> {
    let Some(first) = columns.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_columns", "column"));
    };
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        first.execution_domain(),
        columns[1..].iter().map(|column| column.execution_domain()),
        &ctx,
        "concatenate::concatenate_columns",
    )
}

fn compatible_table_execution(tables: &[&TableView<'_>]) -> Result<ResolvedExecution> {
    let Some(first) = tables.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_tables", "table"));
    };
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        first.execution_domain(),
        tables[1..].iter().map(|table| table.execution_domain()),
        &ctx,
        "concatenate::concatenate_tables",
    )
}

/// Concatenate multiple columns into a single column.
///
/// All columns must have the same data type.
pub fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let execution = compatible_column_execution(columns)?;
    concatenate_columns_with_exec(columns, execution.output_domain(), Some(execution.exec_options()))
}

pub(crate) fn concatenate_columns_with_exec(
    columns: &[&ColumnView<'_>],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<Column> {
    validate_column_types(columns, "concatenate::concatenate_columns")?;
    let handles: Vec<*mut std::os::raw::c_void> = columns.iter().map(|c| c.raw_handle()).collect();
    let num_columns = checked_len_to_i32(
        handles.len(),
        "concatenate::concatenate_columns",
        "number of input columns exceeds i32::MAX",
        "input_columns",
        "column_count",
        "reduce column_count to fit within i32::MAX",
    )?;
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
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
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Concatenate multiple tables into a single table.
///
/// All tables must have the same number of columns with matching types.
pub fn concatenate_tables(tables: &[&TableView<'_>]) -> Result<Table> {
    let execution = compatible_table_execution(tables)?;
    concatenate_tables_with_exec(tables, execution.output_domain(), Some(execution.exec_options()))
}

pub(crate) fn concatenate_tables_with_exec(
    tables: &[&TableView<'_>],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<Table> {
    validate_table_column_types(tables, "concatenate::concatenate_tables")?;
    let handles: Vec<*mut std::os::raw::c_void> = tables.iter().map(|t| t.raw_handle()).collect();
    let num_tables = checked_len_to_i32(
        handles.len(),
        "concatenate::concatenate_tables",
        "number of input tables exceeds i32::MAX",
        "input_tables",
        "table_count",
        "reduce table_count to fit within i32::MAX",
    )?;
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
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
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    Err(input_column_count_limit_error("concatenate::concatenate_columns", usize::MAX))
}

pub(crate) fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    Err(input_table_count_limit_error("concatenate::concatenate_tables", usize::MAX))
}
