//! Parquet file I/O.

use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// Options for reading a Parquet file.
pub struct ParquetReaderOptions {
    filepath: CString,
    columns: Option<Vec<CString>>,
    skip_rows: i64,
    num_rows: i64,
}

impl ParquetReaderOptions {
    /// Create options to read from the given file path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let filepath =
            super::path_to_cstring(path.as_ref(), "io::parquet::parquet_reader_options")?;
        Ok(Self {
            filepath,
            columns: None,
            skip_rows: 0,
            num_rows: -1, // -1 means read all
        })
    }

    /// Only read the specified columns (by name).
    pub fn columns(mut self, names: &[&str]) -> Self {
        self.columns = Some(
            names
                .iter()
                .map(|s| CString::new(*s).expect("column name contains null byte"))
                .collect(),
        );
        self
    }

    /// Skip the first `n` rows.
    pub fn skip_rows(mut self, n: i64) -> Self {
        self.skip_rows = n;
        self
    }

    /// Read at most `n` rows.
    pub fn num_rows(mut self, n: i64) -> Self {
        self.num_rows = n;
        self
    }
}

/// Read a Parquet file into a `Table`.
pub fn read_parquet(options: &ParquetReaderOptions) -> Result<Table> {
    let mut table_handle = ptr::null_mut();

    // Build the pointer array outside the match so it outlives the FFI call.
    let col_ptrs_vec: Vec<*const std::os::raw::c_char> = options
        .columns
        .as_ref()
        .map(|cols| cols.iter().map(|c| c.as_ptr()).collect())
        .unwrap_or_default();
    let (col_ptrs, num_cols) = if col_ptrs_vec.is_empty() {
        (ptr::null(), 0)
    } else {
        (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32)
    };

    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_io_read_parquet(
                options.filepath.as_ptr(),
                col_ptrs,
                num_cols,
                options.skip_rows,
                options.num_rows,
                &mut table_handle,
            )
        },
        "io::parquet::read_parquet",
        [
            ("selected_column_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", options.num_rows.to_string()),
        ],
    )?;

    Ok(unsafe { Table::from_raw(table_handle) })
}

/// Options for writing a Parquet file.
pub struct ParquetWriterOptions {
    filepath: CString,
    column_names: Option<Vec<CString>>,
}

impl ParquetWriterOptions {
    /// Create options to write to the given file path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let filepath =
            super::path_to_cstring(path.as_ref(), "io::parquet::parquet_writer_options")?;
        Ok(Self {
            filepath,
            column_names: None,
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
}

/// Write a table to a Parquet file.
pub fn write_parquet(table: &TableView<'_>, options: &ParquetWriterOptions) -> Result<()> {
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
            cudf_sys::cudf_io_write_parquet(
                table.raw_handle(),
                options.filepath.as_ptr(),
                name_ptrs,
                num_cols,
            )
        },
        "io::parquet::write_parquet",
        [("output_column_name_count", num_cols.to_string())],
    )
}
