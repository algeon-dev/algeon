//! Parquet file I/O.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
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
    runtime::Runtime,
};

/// Options for reading a Parquet file.
#[derive(TypedBuilder)]
pub struct ParquetReaderOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    columns: Option<Vec<String>>,
    #[builder(default)]
    skip_rows: i64,
    #[builder(default = -1)]
    num_rows: i64,
    #[builder(default, setter(transform = |groups: &[i32]| if groups.is_empty() { None } else { Some(groups.to_vec()) }))]
    row_groups: Option<Vec<i32>>,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
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
/// Returns the table data and the column names stored in the Parquet file's
/// schema. This avoids having to hardcode column indices when working with
/// Parquet files.
pub fn read_parquet_with_metadata(options: ParquetReaderOptions) -> Result<ParquetTableWithMetadata> {
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::read_parquet_with_metadata")?;

    // Validate that row_groups is not combined with skip_rows or num_rows.
    if options.row_groups.is_some() && (options.skip_rows > 0 || options.num_rows >= 0) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "row_groups cannot be combined with skip_rows or num_rows",
        )
        .with_operation("io::parquet::read_parquet_with_metadata")
        .with_context("skip_rows", options.skip_rows.to_string())
        .with_context("num_rows", options.num_rows.to_string()));
    }

    let output_domain = options.runtime.as_ref().map(|runtime| Arc::clone(runtime.execution_domain()));
    let exec_options = output_domain.as_ref().and_then(|domain| domain.explicit_exec_options_raw());

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;

    // Convert column names to CStrings at the call site, then build the pointer array.
    let cstring_columns: Option<Vec<CString>> = options
        .columns
        .as_ref()
        .map(|cols| {
            cols.iter()
                .map(|s| {
                    CString::new(s.as_str()).map_err(|_| {
                        Error::new(ErrorKind::InvalidOperation, "column name contains interior null byte")
                            .with_operation("io::parquet::read_parquet_with_metadata")
                            .with_context("column_name", s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let col_ptrs_vec: Vec<*const c_char> =
        cstring_columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (col_ptrs, num_cols) =
        if col_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32) };

    // Build row_groups pointer; the Option<Vec> must outlive the FFI call.
    let (rg_ptr, num_rg) = match options.row_groups.as_ref() {
        Some(rg) if !rg.is_empty() => (rg.as_ptr(), rg.len() as i32),
        _ => (ptr::null(), 0),
    };

    error::check_status_in_with(
        // SAFETY: filepath is a valid CString. col_ptrs is either null or a
        // valid pointer to num_cols CString pointers. rg_ptr is either null
        // or a valid pointer to num_rg i32 values.
        match exec_options.as_ref() {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_read_parquet_with_metadata_ex(
                    filepath.as_ptr(),
                    col_ptrs,
                    num_cols,
                    options.skip_rows,
                    options.num_rows,
                    rg_ptr,
                    num_rg,
                    &mut table_handle,
                    &mut names_ptr,
                    &mut num_names,
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_read_parquet_with_metadata(
                    filepath.as_ptr(),
                    col_ptrs,
                    num_cols,
                    options.skip_rows,
                    options.num_rows,
                    rg_ptr,
                    num_rg,
                    &mut table_handle,
                    &mut names_ptr,
                    &mut num_names,
                )
            },
        },
        "io::parquet::read_parquet_with_metadata",
        [
            ("selected_column_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", options.num_rows.to_string()),
        ],
    )?;

    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    // Wrap early so table is freed via RAII if collect_column_names() fails.
    let table = match output_domain {
        Some(domain) => unsafe { Table::from_raw(table_handle).with_execution_domain(domain) },
        None => unsafe { Table::from_raw(table_handle) },
    };

    // Convert the C string array into Vec<String> and free the C memory.
    // SAFETY: names_ptr and num_names were returned by a successful FFI call.
    // cudf_io_free_column_names frees the C-allocated name strings and array.
    let column_names = unsafe {
        let names = collect_column_names(names_ptr, num_names);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };

    Ok(ParquetTableWithMetadata { table, column_names })
}

unsafe fn collect_column_names(names_ptr: *mut *mut c_char, num_names: i32) -> Result<Vec<String>> {
    let operation = "io::parquet::read_parquet_with_metadata";

    if num_names < 0 {
        return Err(crate::core::error::Error::new(
            crate::core::error::ErrorKind::InvalidOperation,
            "ffi returned negative column name count",
        )
        .with_operation(operation)
        .with_context("num_names", num_names.to_string()));
    }

    if num_names > 0 && names_ptr.is_null() {
        return Err(crate::core::error::Error::new(
            crate::core::error::ErrorKind::InvalidOperation,
            "ffi returned null column name array with positive count",
        )
        .with_operation(operation)
        .with_context("num_names", num_names.to_string())
        .with_context("names_ptr_null", "true"));
    }

    let mut names = Vec::with_capacity(num_names as usize);
    for i in 0..num_names {
        // SAFETY: names_ptr is valid for num_names elements (checked above).
        let name_ptr = *names_ptr.add(i as usize);
        if name_ptr.is_null() {
            return Err(crate::core::error::Error::new(
                crate::core::error::ErrorKind::InvalidOperation,
                "ffi returned null column name entry",
            )
            .with_operation(operation)
            .with_context("num_names", num_names.to_string())
            .with_context("name_index", i.to_string()));
        }

        // SAFETY: name_ptr is non-null (checked above) and points to a
        // valid null-terminated C string allocated by the C shim.
        let c_str = CStr::from_ptr(name_ptr);
        names.push(c_str.to_string_lossy().into_owned());
    }

    Ok(names)
}

/// Options for writing a Parquet file.
#[derive(TypedBuilder)]
pub struct ParquetWriterOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    column_names: Option<Vec<String>>,
    #[builder(default)]
    row_group_size_rows: i32,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetWriterOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a Parquet file.
///
/// Returns an error if `column_names` were set on the options and their count
/// does not match the number of columns in the table.
pub fn write_parquet(table: &TableView<'_>, options: &ParquetWriterOptions) -> Result<()> {
    let exec_options =
        options.runtime.as_ref().and_then(|runtime| runtime.execution_domain().explicit_exec_options_raw());
    write_parquet_with_exec(table, options, exec_options.as_ref())
}

pub(crate) fn write_parquet_with_exec(
    table: &TableView<'_>,
    options: &ParquetWriterOptions,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::write_parquet")?;

    // Validate column name count matches the table before the FFI call.
    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()? as usize;
        if names.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column name count ({}) does not match table column count ({})", names.len(), expected),
            )
            .with_operation("io::parquet::write_parquet")
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
                        Error::new(ErrorKind::InvalidOperation, "column name contains interior null byte")
                            .with_operation("io::parquet::write_parquet")
                            .with_context("column_name", s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    // Build the pointer array; the Vec must outlive the FFI call.
    let name_ptrs_vec: Vec<*const c_char> =
        cstring_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) =
        if name_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32) };

    error::check_status_in_with(
        // SAFETY: table handle is valid (NonNull). filepath is a valid CString.
        // name_ptrs is either null or a valid pointer to num_cols CString pointers.
        match exec_options {
            Some(exec_options) => unsafe {
                cudf_sys::cudf_io_write_parquet_ex(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    name_ptrs,
                    num_cols,
                    options.row_group_size_rows,
                    exec_options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_io_write_parquet(
                    table.raw_handle(),
                    filepath.as_ptr(),
                    name_ptrs,
                    num_cols,
                    options.row_group_size_rows,
                )
            },
        },
        "io::parquet::write_parquet",
        [("output_column_name_count", num_cols.to_string())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::ErrorKind;

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
