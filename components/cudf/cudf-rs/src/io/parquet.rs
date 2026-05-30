//! Parquet file I/O.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
    path::Path,
    ptr,
};

use crate::{
    error::{self, Result},
    table::{Table, TableView},
};

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
        let filepath = super::path_to_cstring(path.as_ref(), "io::parquet::parquet_reader_options")?;
        Ok(Self {
            filepath,
            columns: None,
            skip_rows: 0,
            num_rows: -1, // -1 means read all
        })
    }

    /// Only read the specified columns (by name).
    pub fn columns(mut self, names: &[&str]) -> Self {
        self.columns = Some(names.iter().map(|s| CString::new(*s).expect("column name contains null byte")).collect());
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
    let col_ptrs_vec: Vec<*const std::os::raw::c_char> =
        options.columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (col_ptrs, num_cols) =
        if col_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32) };

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

/// A table read from Parquet along with its column name metadata.
pub struct ParquetTableWithMetadata {
    /// The table containing the data.
    pub table: Table,
    /// The column names from the Parquet file schema.
    pub column_names: Vec<String>,
}

/// Read a Parquet file into a `Table` along with column name metadata.
///
/// This is the same as [`read_parquet`] but additionally returns the column
/// names stored in the Parquet file's schema. This avoids having to hardcode
/// column indices when working with Parquet files.
pub fn read_parquet_with_metadata(options: &ParquetReaderOptions) -> Result<ParquetTableWithMetadata> {
    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;

    // Build the pointer array outside the match so it outlives the FFI call.
    let col_ptrs_vec: Vec<*const std::os::raw::c_char> =
        options.columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (col_ptrs, num_cols) =
        if col_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32) };

    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_io_read_parquet_with_metadata(
                options.filepath.as_ptr(),
                col_ptrs,
                num_cols,
                options.skip_rows,
                options.num_rows,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
            )
        },
        "io::parquet::read_parquet_with_metadata",
        [
            ("selected_column_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", options.num_rows.to_string()),
        ],
    )?;

    // Convert the C string array into Vec<String> and free the C memory.
    let column_names = unsafe {
        let names = collect_column_names(names_ptr, num_names);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };

    Ok(ParquetTableWithMetadata { table: unsafe { Table::from_raw(table_handle) }, column_names })
}

unsafe fn collect_column_names(names_ptr: *mut *mut c_char, num_names: i32) -> Result<Vec<String>> {
    let operation = "io::parquet::read_parquet_with_metadata";

    if num_names < 0 {
        return Err(crate::error::Error::new(
            crate::error::ErrorKind::InvalidOperation,
            "ffi returned negative column name count",
        )
        .with_operation(operation)
        .with_context("num_names", num_names.to_string()));
    }

    if num_names > 0 && names_ptr.is_null() {
        return Err(crate::error::Error::new(
            crate::error::ErrorKind::InvalidOperation,
            "ffi returned null column name array with positive count",
        )
        .with_operation(operation)
        .with_context("num_names", num_names.to_string())
        .with_context("names_ptr_null", "true"));
    }

    let mut names = Vec::with_capacity(num_names as usize);
    for i in 0..num_names {
        let name_ptr = *names_ptr.add(i as usize);
        if name_ptr.is_null() {
            return Err(crate::error::Error::new(
                crate::error::ErrorKind::InvalidOperation,
                "ffi returned null column name entry",
            )
            .with_operation(operation)
            .with_context("num_names", num_names.to_string())
            .with_context("name_index", i.to_string()));
        }

        let c_str = CStr::from_ptr(name_ptr);
        names.push(c_str.to_string_lossy().into_owned());
    }

    Ok(names)
}

/// Options for writing a Parquet file.
pub struct ParquetWriterOptions {
    filepath: CString,
    column_names: Option<Vec<CString>>,
}

impl ParquetWriterOptions {
    /// Create options to write to the given file path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let filepath = super::path_to_cstring(path.as_ref(), "io::parquet::parquet_writer_options")?;
        Ok(Self { filepath, column_names: None })
    }

    /// Set column names for the output file.
    pub fn column_names(mut self, names: &[&str]) -> Self {
        self.column_names =
            Some(names.iter().map(|s| CString::new(*s).expect("column name contains null byte")).collect());
        self
    }
}

/// Write a table to a Parquet file.
pub fn write_parquet(table: &TableView<'_>, options: &ParquetWriterOptions) -> Result<()> {
    // Build the pointer array outside the match so it outlives the FFI call.
    let name_ptrs_vec: Vec<*const std::os::raw::c_char> =
        options.column_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) =
        if name_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32) };

    error::check_status_in_with(
        unsafe { cudf_sys::cudf_io_write_parquet(table.raw_handle(), options.filepath.as_ptr(), name_ptrs, num_cols) },
        "io::parquet::write_parquet",
        [("output_column_name_count", num_cols.to_string())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn test_collect_column_names_rejects_negative_count() {
        let err = unsafe { collect_column_names(ptr::null_mut(), -1) }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "io::parquet::read_parquet_with_metadata");
        assert!(err.context().iter().any(|(k, v)| *k == "num_names" && v == "-1"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_names_ptr_with_positive_count() {
        let err = unsafe { collect_column_names(ptr::null_mut(), 1) }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "io::parquet::read_parquet_with_metadata");
        assert!(err.context().iter().any(|(k, v)| *k == "names_ptr_null" && v == "true"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_entry_pointer() {
        let mut ptrs: Vec<*mut c_char> = vec![ptr::null_mut()];
        let err = unsafe { collect_column_names(ptrs.as_mut_ptr(), 1) }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "io::parquet::read_parquet_with_metadata");
        assert!(err.context().iter().any(|(k, v)| *k == "name_index" && v == "0"));
    }

    #[test]
    fn test_collect_column_names_success() {
        let mut raw = [CString::new("alpha").unwrap(), CString::new("beta").unwrap()];
        let mut ptrs: Vec<*mut c_char> = raw.iter_mut().map(|s| s.as_ptr() as *mut c_char).collect();
        let names = unsafe { collect_column_names(ptrs.as_mut_ptr(), ptrs.len() as i32) }.unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
