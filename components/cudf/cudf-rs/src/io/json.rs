//! JSON / JSON Lines (NDJSON) file I/O.

use std::{
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use typed_builder::TypedBuilder;

use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    io::IoTableResult,
    runtime::Runtime,
};

/// Options for reading a JSON file.
///
/// Used by both the low-level [`read_json`] function and the high-level
/// [`DataFrame::read_json_with_options`](crate::DataFrame::read_json_with_options).
///
/// # Defaults
///
/// - `lines` = `true` (JSON Lines / NDJSON format, one JSON object per line)
#[derive(Debug, Clone, TypedBuilder)]
pub struct JsonTableReadOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// If `true` (default), read as JSON Lines (one JSON object per line).
    /// If `false`, read as a single JSON array.
    #[builder(default = true)]
    lines: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

/// Read a JSON file into a low-level table result.
pub fn read_json(options: &JsonTableReadOptions) -> Result<IoTableResult> {
    let filepath = super::path_to_cstring(&options.filepath, "io::json::read_json")?;
    let output_domain = options.runtime.as_ref().map(|runtime| Arc::clone(runtime.execution_domain()));
    let exec_options = output_domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut std::os::raw::c_char = ptr::null_mut();
    let mut num_names: i32 = 0;
    error::check_status_in_with(
        match exec_options.as_ref() {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_read_json_with_metadata_ex(
                    filepath.as_ptr(),
                    i32::from(options.lines),
                    &mut table_handle,
                    &mut names_ptr,
                    &mut num_names,
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_read_json_with_metadata(
                    filepath.as_ptr(),
                    i32::from(options.lines),
                    &mut table_handle,
                    &mut names_ptr,
                    &mut num_names,
                )
            },
        },
        "io::json::read_json",
        [("lines", options.lines.to_string())],
    )?;
    let column_names = unsafe {
        let names = super::collect_column_names(names_ptr, num_names, "io::json::read_json");
        if !names_ptr.is_null() && num_names >= 0 {
            cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        }
        names?
    };

    let table = match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    };

    Ok(IoTableResult { table, column_names: if num_names == 0 { None } else { Some(column_names) } })
}

/// Options for writing a low-level [`Table`] to a JSON file.
///
/// **Advanced.** Most users should prefer [`DataFrame::write_json`](crate::DataFrame::write_json).
///
/// # Defaults
///
/// - `lines` = `true` (JSON Lines / NDJSON format)
/// - `include_nulls` = `true`
#[derive(Debug, Clone, TypedBuilder)]
pub struct JsonTableWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// Column names for the output.
    #[builder(default, setter(transform = |names: &[&str]| names.iter().map(|s| s.to_string()).collect()))]
    column_names: Vec<String>,
    /// If `true` (default), write as JSON Lines (one JSON object per line).
    #[builder(default = true)]
    lines: bool,
    /// If `true` (default), include null fields in output.
    #[builder(default = true)]
    include_nulls: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl JsonTableWriteOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a JSON file.
pub fn write_json(table: &TableView<'_>, options: &JsonTableWriteOptions) -> Result<()> {
    let exec_options =
        options.runtime.as_ref().and_then(|runtime| runtime.execution_domain().explicit_exec_options_raw());
    write_json_with_exec(table, options, exec_options.as_ref())
}

pub(crate) fn write_json_with_exec(
    table: &TableView<'_>,
    options: &JsonTableWriteOptions,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<()> {
    let operation = "io::json::write_json";
    let filepath = super::path_to_cstring(&options.filepath, operation)?;

    let c_names: Vec<CString> = options
        .column_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            CString::new(name.as_str()).map_err(|_| {
                Error::new(ErrorKind::InvalidOperation, "column name contains interior null byte")
                    .with_operation(operation)
                    .with_context("column_index", index.to_string())
            })
        })
        .collect::<Result<_>>()?;
    let c_name_ptrs: Vec<*const std::os::raw::c_char> = c_names.iter().map(|s| s.as_ptr()).collect();

    let names_ptr = if c_name_ptrs.is_empty() { ptr::null() } else { c_name_ptrs.as_ptr() };

    error::check_status_in_with(
        match exec_options {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_write_json_ex(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    names_ptr,
                    c_names.len() as i32,
                    i32::from(options.lines),
                    i32::from(options.include_nulls),
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_write_json(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    names_ptr,
                    c_names.len() as i32,
                    i32::from(options.lines),
                    i32::from(options.include_nulls),
                )
            },
        },
        operation,
        [("lines", options.lines.to_string()), ("include_nulls", options.include_nulls.to_string())],
    )
}
