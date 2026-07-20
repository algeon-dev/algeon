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
        error::{self, FACT_COLUMN_INDEX, Result},
        schema,
        table::Table,
    },
    io::TableReadOutput,
    runtime::Runtime,
};

/// Options for reading a JSON file into a low-level [`Table`].
///
/// **Advanced.** Most users should prefer
/// [`DataFrame::read_json`](crate::DataFrame::read_json).
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
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl JsonTableReadOptions {
    pub(crate) fn from_parts(filepath: PathBuf, lines: bool, runtime: Runtime) -> Self {
        Self { filepath, lines, runtime }
    }
}

/// Read a JSON file into a low-level table result.
pub fn read_json_table(options: &JsonTableReadOptions) -> Result<TableReadOutput> {
    const OPERATION: &str = "io::json::read_json_table";
    let filepath = super::path_to_cstring(&options.filepath, OPERATION)?;
    super::validate_input_path(&options.filepath, OPERATION)?;
    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.enter_execution()?;

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut std::os::raw::c_char = ptr::null_mut();
    let mut num_names: i32 = 0;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_io_read_json_with_metadata_ex(
                filepath.as_ptr(),
                i32::from(options.lines),
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                exec_options.options(),
            )
        },
        OPERATION,
        [("lines", options.lines.to_string())],
    )?;

    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    // Wrap early so table is freed via RAII if collect_column_names() fails.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;

    let column_names = unsafe {
        let names = super::collect_column_names(names_ptr, num_names, OPERATION);
        if !names_ptr.is_null() && num_names >= 0 {
            cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        }
        names?
    };

    Ok(TableReadOutput { table, column_names: if num_names == 0 { None } else { Some(column_names) } })
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
pub fn write_json(table: &Table, options: &JsonTableWriteOptions) -> Result<()> {
    let operation = "io::json::write_json";
    let table_view = table.view()?;
    let execution = super::resolve_write_execution(&table_view, options.runtime(), operation)?;
    write_json_with_exec(&table_view, options, execution.exec_options())
}

pub(crate) fn write_json_with_exec(
    table: &crate::core::TableView<'_>,
    options: &JsonTableWriteOptions,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<()> {
    let operation = "io::json::write_json";
    let filepath = super::path_to_cstring(&options.filepath, operation)?;

    // Validate column name count matches the table before the FFI call.
    if !options.column_names.is_empty() {
        let expected = table.num_columns()?;
        if options.column_names.len() != expected {
            return Err(schema::column_name_count_mismatch_error(
                operation,
                format!(
                    "column name count ({}) does not match table column count ({})",
                    options.column_names.len(),
                    expected
                ),
                options.column_names.len(),
                expected,
            ));
        }
    }

    let c_names: Vec<CString> = options
        .column_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            CString::new(name.as_str()).map_err(|_| {
                error::interior_null_byte_error(operation, "column_name", "column name contains interior null byte")
                    .with_fact(FACT_COLUMN_INDEX, index)
                    .with_column(name.to_string())
            })
        })
        .collect::<Result<_>>()?;
    let c_name_ptrs: Vec<*const std::os::raw::c_char> = c_names.iter().map(|s| s.as_ptr()).collect();

    let names_ptr = if c_name_ptrs.is_empty() { ptr::null() } else { c_name_ptrs.as_ptr() };

    error::check_status_in_with_details(
        unsafe {
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
        operation,
        [("lines", options.lines.to_string()), ("include_nulls", options.include_nulls.to_string())],
    )
}
