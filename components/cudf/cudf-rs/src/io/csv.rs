//! CSV file I/O.

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

/// Options for reading a CSV file into a low-level [`Table`].
///
/// **Advanced.** Most users should prefer [`DataFrame::read_csv`](crate::DataFrame::read_csv).
///
/// This type controls CSV parsing only. Truthful schema names, when available
/// from a header row, are returned through [`IoTableResult`].
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvReaderOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default = b',')]
    delimiter: u8,
    #[builder(default = true)]
    has_header: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

/// Read a CSV file into a low-level table result.
pub fn read_csv(options: &CsvReaderOptions) -> Result<IoTableResult> {
    let filepath = super::path_to_cstring(&options.filepath, "io::csv::read_csv")?;
    let output_domain = options.runtime.as_ref().map(|runtime| Arc::clone(runtime.execution_domain()));
    let exec_options = output_domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());

    let mut table_handle = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: filepath is a valid CString (converted above).
        match exec_options.as_ref() {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_read_csv_ex(
                    filepath.as_ptr(),
                    options.delimiter as std::os::raw::c_char,
                    if options.has_header { 1 } else { 0 },
                    &mut table_handle,
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_read_csv(
                    filepath.as_ptr(),
                    options.delimiter as std::os::raw::c_char,
                    if options.has_header { 1 } else { 0 },
                    &mut table_handle,
                )
            },
        },
        "io::csv::read_csv",
        [("delimiter_char", (options.delimiter as char).to_string()), ("has_header", options.has_header.to_string())],
    )?;
    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    let table = match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    };
    let num_columns = {
        let view = table.view()?;
        view.num_columns()? as usize
    };
    let column_names = if options.has_header {
        let names = read_header_names(&options.filepath, options.delimiter)?;
        validate_header_name_count(names.len(), num_columns)?;
        Some(names)
    } else {
        None
    };

    Ok(IoTableResult { table, column_names })
}

/// Options for writing a low-level [`Table`] to a CSV file.
///
/// **Advanced.** Most users should prefer [`DataFrame::write_csv`](crate::DataFrame::write_csv).
///
/// Column names here control only the emitted CSV header row. They do not
/// rename the input table because [`Table`] itself is unnamed.
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvWriterOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    column_names: Option<Vec<String>>,
    #[builder(default = b',')]
    delimiter: u8,
    #[builder(default = true)]
    include_header: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl CsvWriterOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a CSV file.
pub fn write_csv(table: &TableView<'_>, options: &CsvWriterOptions) -> Result<()> {
    let exec_options =
        options.runtime.as_ref().and_then(|runtime| runtime.execution_domain().explicit_exec_options_raw());
    write_csv_with_exec(table, options, exec_options.as_ref())
}

pub(crate) fn write_csv_with_exec(
    table: &TableView<'_>,
    options: &CsvWriterOptions,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::csv::write_csv")?;

    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()? as usize;
        if names.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column name count ({}) does not match table column count ({})", names.len(), expected),
            )
            .with_operation("io::csv::write_csv")
            .with_context("column_name_count", names.len().to_string())
            .with_context("table_column_count", expected.to_string()));
        }
    }

    // Convert column names to CStrings at the call site.
    let cstring_names: Option<Vec<CString>> = options
        .column_names
        .as_ref()
        .map(|names| {
            names
                .iter()
                .map(|s| {
                    CString::new(s.as_str()).map_err(|_| {
                        crate::core::error::Error::new(
                            crate::core::error::ErrorKind::InvalidOperation,
                            "column name contains interior null byte",
                        )
                        .with_operation("io::csv::write_csv")
                        .with_context("column_name", s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    // Build the pointer array outside the match so it outlives the FFI call.
    let name_ptrs_vec: Vec<*const std::os::raw::c_char> =
        cstring_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) =
        if name_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32) };

    error::check_status_in_with(
        // SAFETY: table handle is valid (NonNull). filepath is a valid CString (converted above).
        // name_ptrs is either null or a valid pointer to num_cols CString pointers.
        match exec_options {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_write_csv_ex(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    name_ptrs,
                    num_cols,
                    options.delimiter as std::os::raw::c_char,
                    if options.include_header { 1 } else { 0 },
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_write_csv(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    name_ptrs,
                    num_cols,
                    options.delimiter as std::os::raw::c_char,
                    if options.include_header { 1 } else { 0 },
                )
            },
        },
        "io::csv::write_csv",
        [
            ("num_column_names", num_cols.to_string()),
            ("delimiter_char", (options.delimiter as char).to_string()),
            ("include_header", options.include_header.to_string()),
        ],
    )
}

fn read_header_names(path: &Path, delimiter: u8) -> Result<Vec<String>> {
    let operation = "io::csv::read_csv";
    let mut reader =
        ::csv::ReaderBuilder::new().has_headers(true).delimiter(delimiter).from_path(path).map_err(|source| {
            Error::new(ErrorKind::IoError, "failed to open CSV file for header parsing")
                .with_operation(operation)
                .with_context("path", path.display().to_string())
                .set_source(source)
        })?;

    let headers = reader.headers().map_err(|source| {
        Error::new(ErrorKind::IoError, "failed to read CSV header")
            .with_operation(operation)
            .with_context("path", path.display().to_string())
            .set_source(source)
    })?;

    Ok(headers.iter().map(str::to_owned).collect())
}

fn validate_header_name_count(name_count: usize, num_columns: usize) -> Result<()> {
    if name_count != num_columns {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("header column count ({name_count}) does not match table column count ({num_columns})"),
        )
        .with_operation("io::csv::read_csv")
        .with_context("header_column_count", name_count.to_string())
        .with_context("table_column_count", num_columns.to_string()));
    }

    Ok(())
}
