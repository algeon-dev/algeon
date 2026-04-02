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
        error::{self, Result},
        table::{Table, TableView},
    },
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
pub struct JsonReaderOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// If `true` (default), read as JSON Lines (one JSON object per line).
    /// If `false`, read as a single JSON array.
    #[builder(default = true)]
    lines: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

/// Result of reading a JSON file with column name metadata.
pub struct JsonTableWithMetadata {
    /// The GPU table read from the file.
    pub table: Table,
    /// Column names extracted from the JSON keys.
    pub column_names: Vec<String>,
}

/// Read a JSON file into a `Table`.
pub fn read_json(options: &JsonReaderOptions) -> Result<Table> {
    let filepath = super::path_to_cstring(&options.filepath, "io::json::read_json")?;
    let output_domain = options.runtime.as_ref().map(|runtime| Arc::clone(runtime.execution_domain()));
    let exec_options = output_domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());

    let mut table_handle = ptr::null_mut();
    error::check_status_in_with(
        match exec_options.as_ref() {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_read_json_ex(
                    filepath.as_ptr(),
                    i32::from(options.lines),
                    &mut table_handle,
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_read_json(filepath.as_ptr(), i32::from(options.lines), &mut table_handle)
            },
        },
        "io::json::read_json",
        [("lines", options.lines.to_string())],
    )?;
    Ok(match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    })
}

/// Read a JSON file, returning both the `Table` and column names.
pub fn read_json_with_metadata(options: &JsonReaderOptions) -> Result<JsonTableWithMetadata> {
    let filepath = super::path_to_cstring(&options.filepath, "io::json::read_json_with_metadata")?;
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
        "io::json::read_json_with_metadata",
        [("lines", options.lines.to_string())],
    )?;

    let column_names = collect_column_names(names_ptr, num_names);

    let table = match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    };

    Ok(JsonTableWithMetadata { table, column_names })
}

/// Collect column names from a C-allocated string array and free the C memory.
fn collect_column_names(names_ptr: *mut *mut std::os::raw::c_char, num_names: i32) -> Vec<String> {
    if names_ptr.is_null() || num_names <= 0 {
        return Vec::new();
    }
    let names: Vec<String> = (0..num_names as usize)
        .map(|i| {
            let cstr = unsafe { std::ffi::CStr::from_ptr(*names_ptr.add(i)) };
            cstr.to_string_lossy().into_owned()
        })
        .collect();
    unsafe { cudf_sys::cudf_io_free_column_names(names_ptr, num_names) };
    names
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
pub struct JsonWriterOptions {
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

impl JsonWriterOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a JSON file.
pub fn write_json(table: &TableView<'_>, options: &JsonWriterOptions) -> Result<()> {
    let exec_options =
        options.runtime.as_ref().and_then(|runtime| runtime.execution_domain().explicit_exec_options_raw());
    write_json_with_exec(table, options, exec_options.as_ref())
}

pub(crate) fn write_json_with_exec(
    table: &TableView<'_>,
    options: &JsonWriterOptions,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::json::write_json")?;

    let c_names: Vec<CString> = options
        .column_names
        .iter()
        .map(|s| CString::new(s.as_str()).expect("column name should not contain null byte"))
        .collect();
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
        "io::json::write_json",
        [("lines", options.lines.to_string()), ("include_nulls", options.include_nulls.to_string())],
    )
}
