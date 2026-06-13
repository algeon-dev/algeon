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
        error::{self, Error, ErrorCode, ErrorKind, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

fn checked_len_to_i32(
    len: usize,
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    value_key: &'static str,
    remedy: &'static str,
) -> Result<i32> {
    i32::try_from(len).map_err(|_| {
        error::limit_validation_error(
            ErrorKind::InvalidInput,
            operation,
            message,
            resource,
            value_key,
            "i32::MAX",
            len,
            remedy,
        )
    })
}

fn input_column_count_limit_error(operation: &'static str, column_count: usize) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
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
        ErrorKind::InvalidInput,
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
    Error::new(ErrorKind::InvalidInput, format!("{operation} requires at least one input {input_kind}"))
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", "empty_input")
        .with_context("remedy", format!("provide at least one input {input_kind}"))
}

fn type_label<T: std::fmt::Debug>(value: T) -> String {
    format!("{value:?}").to_lowercase()
}

fn validate_column_types(columns: &[&ColumnView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = columns.split_first() else {
        return Ok(());
    };
    let left_type = first.data_type().map_err(|e| e.with_operation(operation))?.id();

    for (offset, column) in rest.iter().enumerate() {
        let right_type = column.data_type().map_err(|e| e.with_operation(operation))?.id();
        if left_type != right_type {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "all input columns must have the same data type for concatenation",
            )
            .with_code(ErrorCode::TypeMismatch)
            .with_operation(operation)
            .with_context("column_index", (offset + 1).to_string())
            .with_context("left_type_id", type_label(left_type))
            .with_context("right_type_id", type_label(right_type)));
        }
    }

    Ok(())
}

fn validate_table_column_types(tables: &[&TableView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = tables.split_first() else {
        return Ok(());
    };
    let column_count = first.num_columns().map_err(|e| e.with_operation(operation))?;

    for table in rest {
        let table_column_count = table.num_columns().map_err(|e| e.with_operation(operation))?;
        if table_column_count != column_count {
            return Ok(());
        }

        for index in 0..column_count {
            let left_type = first
                .column(index)
                .and_then(|column| column.data_type())
                .map_err(|e| e.with_operation(operation))?
                .id();
            let right_type = table
                .column(index)
                .and_then(|column| column.data_type())
                .map_err(|e| e.with_operation(operation))?
                .id();

            if left_type != right_type {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "all input tables must have matching column data types for concatenation",
                )
                .with_code(ErrorCode::TypeMismatch)
                .with_operation(operation)
                .with_context("column_index", index.to_string())
                .with_context("left_type_id", type_label(left_type))
                .with_context("right_type_id", type_label(right_type)));
            }
        }
    }

    Ok(())
}

fn compatible_column_domain(columns: &[&ColumnView<'_>]) -> Result<Arc<ExecutionDomain>> {
    let Some(first) = columns.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_columns", "column"));
    };
    let output_domain = Arc::clone(first.execution_domain());
    // Order any secondary input stream before the authoritative output stream
    // so the concatenation kernel only reads completed device data.
    for col in &columns[1..] {
        ExecutionDomain::require_same_device_for_mixed_inputs(
            first.execution_domain(),
            col.execution_domain(),
            "concatenate::concatenate_columns",
        )?;
        if !ExecutionDomain::same_provenance(col.execution_domain(), &output_domain) {
            ExecutionDomain::insert_stream_dependency(
                col.execution_domain(),
                &output_domain,
                "concatenate::concatenate_columns",
            )?;
        }
    }
    Ok(output_domain)
}

fn compatible_table_domain(tables: &[&TableView<'_>]) -> Result<Arc<ExecutionDomain>> {
    let Some(first) = tables.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_tables", "table"));
    };
    let output_domain = Arc::clone(first.execution_domain());
    // Order any secondary input stream before the authoritative output stream
    // so the concatenation kernel only reads completed device data.
    for tbl in &tables[1..] {
        ExecutionDomain::require_same_device_for_mixed_inputs(
            first.execution_domain(),
            tbl.execution_domain(),
            "concatenate::concatenate_tables",
        )?;
        if !ExecutionDomain::same_provenance(tbl.execution_domain(), &output_domain) {
            ExecutionDomain::insert_stream_dependency(
                tbl.execution_domain(),
                &output_domain,
                "concatenate::concatenate_tables",
            )?;
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
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
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
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    Err(input_column_count_limit_error("concatenate::concatenate_columns", usize::MAX))
}

pub(crate) fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    Err(input_table_count_limit_error("concatenate::concatenate_tables", usize::MAX))
}
