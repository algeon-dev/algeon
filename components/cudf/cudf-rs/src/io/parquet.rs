//! Parquet file I/O.

use std::{
    ffi::CString,
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
    io::IoTableResult,
    runtime::Runtime,
};

/// Options for reading a Parquet file.
///
/// Used by both the low-level [`read_parquet`] function and the high-level
/// [`DataFrame::read_parquet_with_options`](crate::DataFrame::read_parquet_with_options).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetReadOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    columns: Option<Vec<String>>,
    #[builder(default)]
    skip_rows: usize,
    #[builder(default, setter(transform = |num_rows: usize| Some(num_rows)))]
    num_rows: Option<usize>,
    #[builder(default, setter(transform = |groups: &[i32]| if groups.is_empty() { None } else { Some(groups.to_vec()) }))]
    row_groups: Option<Vec<i32>>,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

fn checked_i64_from_usize(value: usize, field: &'static str, operation: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::new(ErrorKind::InvalidArgument, format!("{field} exceeds i64::MAX"))
            .with_operation(operation)
            .with_context(field, value.to_string())
    })
}

/// Read a Parquet file into a low-level table result.
///
/// Returns the table data and the column names stored in the Parquet file's
/// schema. This avoids having to hardcode column indices when working with
/// Parquet files.
pub fn read_parquet(options: &ParquetReadOptions) -> Result<IoTableResult> {
    const OPERATION: &str = "io::parquet::read_parquet";
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::read_parquet")?;
    let num_rows_context = options.num_rows.map(|num_rows| num_rows.to_string()).unwrap_or_else(|| "all".to_string());

    // Validate that row_groups is not combined with skip_rows or num_rows.
    if options.row_groups.is_some() && (options.skip_rows > 0 || options.num_rows.is_some()) {
        return Err(Error::new(ErrorKind::InvalidArgument, "row_groups cannot be combined with skip_rows or num_rows")
            .with_operation(OPERATION)
            .with_context("skip_rows", options.skip_rows.to_string())
            .with_context("num_rows", num_rows_context));
    }
    let skip_rows = checked_i64_from_usize(options.skip_rows, "skip_rows", OPERATION)?;
    let num_rows = match options.num_rows {
        Some(num_rows) => checked_i64_from_usize(num_rows, "num_rows", OPERATION)?,
        None => -1,
    };

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
                        Error::new(ErrorKind::InvalidArgument, "column name contains interior null byte")
                            .with_operation("io::parquet::read_parquet")
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
                    skip_rows,
                    num_rows,
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
                    skip_rows,
                    num_rows,
                    rg_ptr,
                    num_rg,
                    &mut table_handle,
                    &mut names_ptr,
                    &mut num_names,
                )
            },
        },
        "io::parquet::read_parquet",
        [
            ("selected_column_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", num_rows_context),
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
        let names = super::collect_column_names(names_ptr, num_names, "io::parquet::read_parquet");
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };

    Ok(IoTableResult { table, column_names: Some(column_names) })
}

/// Options for writing a Parquet file.
///
/// **Advanced.** Most users should prefer [`DataFrame::write_parquet`](crate::DataFrame::write_parquet).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetTableWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    column_names: Option<Vec<String>>,
    #[builder(default)]
    row_group_size_rows: i32,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetTableWriteOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a Parquet file.
///
/// Returns an error if `column_names` were set on the options and their count
/// does not match the number of columns in the table.
pub fn write_parquet(table: &TableView<'_>, options: &ParquetTableWriteOptions) -> Result<()> {
    let exec_options =
        options.runtime.as_ref().and_then(|runtime| runtime.execution_domain().explicit_exec_options_raw());
    write_parquet_with_exec(table, options, exec_options.as_ref())
}

pub(crate) fn write_parquet_with_exec(
    table: &TableView<'_>,
    options: &ParquetTableWriteOptions,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::write_parquet")?;

    // Validate column name count matches the table before the FFI call.
    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()? as usize;
        if names.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
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
                        Error::new(ErrorKind::InvalidArgument, "column name contains interior null byte")
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
