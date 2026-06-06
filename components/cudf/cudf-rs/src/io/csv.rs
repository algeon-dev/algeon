//! CSV file I/O.

use std::{
    ffi::CString,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    runtime::Runtime,
};

/// Options for reading a CSV file into a low-level [`Table`].
///
/// This type controls parsing only. It does not carry schema names because
/// the low-level CSV reader returns an unnamed [`Table`].
pub struct CsvReaderOptions {
    filepath: PathBuf,
    delimiter: u8,
    has_header: bool,
    runtime: Option<Runtime>,
}

impl CsvReaderOptions {
    /// Create options to read from the given file path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { filepath: path.as_ref().to_path_buf(), delimiter: b',', has_header: true, runtime: None }
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether the file has a header row (default: `true`).
    pub fn has_header(mut self, has: bool) -> Self {
        self.has_header = has;
        self
    }

    /// Seed explicit execution provenance from the given runtime.
    pub fn with_runtime(mut self, runtime: &Runtime) -> Self {
        self.runtime = Some(runtime.clone());
        self
    }
}

/// Read a CSV file into a `Table`.
pub fn read_csv(options: &CsvReaderOptions) -> Result<Table> {
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
    Ok(match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    })
}

/// Options for writing a low-level [`Table`] to a CSV file.
///
/// Column names here control only the emitted CSV header row. They do not
/// rename the input table because [`Table`] itself is unnamed.
pub struct CsvWriterOptions {
    filepath: PathBuf,
    column_names: Option<Vec<String>>,
    delimiter: u8,
    include_header: bool,
    runtime: Option<Runtime>,
}

impl CsvWriterOptions {
    /// Create options to write to the given file path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            filepath: path.as_ref().to_path_buf(),
            column_names: None,
            delimiter: b',',
            include_header: true,
            runtime: None,
        }
    }

    /// Set column names for the output file.
    pub fn column_names(mut self, names: &[&str]) -> Self {
        self.column_names = Some(names.iter().map(|s| (*s).to_string()).collect());
        self
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether to include a header row (default: `true`).
    pub fn include_header(mut self, include: bool) -> Self {
        self.include_header = include;
        self
    }

    /// Route the write through the given runtime's stream.
    pub fn with_runtime(mut self, runtime: &Runtime) -> Self {
        self.runtime = Some(runtime.clone());
        self
    }

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
