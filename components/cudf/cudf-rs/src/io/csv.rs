//! CSV file I/O.

use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// Options for reading a CSV file.
pub struct CsvReaderOptions {
    filepath: CString,
    delimiter: u8,
    has_header: bool,
}

impl CsvReaderOptions {
    /// Create options to read from the given file path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            crate::error::Error::new(
                crate::error::ErrorKind::InvalidOperation,
                "Path is not valid UTF-8",
            )
        })?;
        let filepath = CString::new(path_str).map_err(|_| {
            crate::error::Error::new(
                crate::error::ErrorKind::InvalidOperation,
                "Path contains interior null byte",
            )
        })?;
        Ok(Self {
            filepath,
            delimiter: b',',
            has_header: true,
        })
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
}

/// Read a CSV file into a `Table`.
pub fn read_csv(options: &CsvReaderOptions) -> Result<Table> {
    let mut table_handle = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_io_read_csv(
                options.filepath.as_ptr(),
                options.delimiter as std::os::raw::c_char,
                if options.has_header { 1 } else { 0 },
                &mut table_handle,
            )
        },
        "io::csv::read_csv",
        [
            ("delimiter_char", (options.delimiter as char).to_string()),
            ("has_header", options.has_header.to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(table_handle) })
}

/// Options for writing a CSV file.
pub struct CsvWriterOptions {
    filepath: CString,
    column_names: Option<Vec<CString>>,
    delimiter: u8,
    include_header: bool,
}

impl CsvWriterOptions {
    /// Create options to write to the given file path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            crate::error::Error::new(
                crate::error::ErrorKind::InvalidOperation,
                "Path is not valid UTF-8",
            )
        })?;
        let filepath = CString::new(path_str).map_err(|_| {
            crate::error::Error::new(
                crate::error::ErrorKind::InvalidOperation,
                "Path contains interior null byte",
            )
        })?;
        Ok(Self {
            filepath,
            column_names: None,
            delimiter: b',',
            include_header: true,
        })
    }

    /// Set column names for the output file.
    pub fn column_names(mut self, names: &[&str]) -> Self {
        self.column_names = Some(
            names
                .iter()
                .map(|s| CString::new(*s).expect("column name contains null byte"))
                .collect(),
        );
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
}

/// Write a table to a CSV file.
pub fn write_csv(table: &TableView<'_>, options: &CsvWriterOptions) -> Result<()> {
    // Build the pointer array outside the match so it outlives the FFI call.
    let name_ptrs_vec: Vec<*const std::os::raw::c_char> = options
        .column_names
        .as_ref()
        .map(|names| names.iter().map(|c| c.as_ptr()).collect())
        .unwrap_or_default();
    let (name_ptrs, num_cols) = if name_ptrs_vec.is_empty() {
        (ptr::null(), 0)
    } else {
        (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32)
    };

    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_io_write_csv(
                table.raw_handle(),
                options.filepath.as_ptr(),
                name_ptrs,
                num_cols,
                options.delimiter as std::os::raw::c_char,
                if options.include_header { 1 } else { 0 },
            )
        },
        "io::csv::write_csv",
        [
            ("num_column_names", num_cols.to_string()),
            ("delimiter_char", (options.delimiter as char).to_string()),
            ("include_header", options.include_header.to_string()),
        ],
    )
}
