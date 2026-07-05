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
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
        schema,
        table::Table,
    },
    io::TableReadOutput,
    ops::ast::AstExpression,
    runtime::Runtime,
};

/// Options for reading a Parquet file into a low-level [`Table`].
///
/// **Advanced.** Most users should prefer
/// [`DataFrame::read_parquet`](crate::DataFrame::read_parquet).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetTableReadOptions {
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
    #[builder(default, setter(strip_option, into))]
    predicate: Option<Arc<AstExpression>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl ParquetTableReadOptions {
    pub(crate) fn from_parts(
        filepath: PathBuf,
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self { filepath, columns, skip_rows, num_rows, row_groups, predicate, runtime }
    }
}

fn checked_i64_from_usize(value: usize, field: &'static str, operation: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| error::range_limit_error(operation, field, "i64::MAX", value))
}

fn row_group_combination_error(
    row_group_count: usize,
    skip_rows: usize,
    num_rows_context: String,
    operation: &'static str,
) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "row_groups cannot be combined with skip_rows or num_rows",
    )
    .with_detail("row_group_count", row_group_count.to_string())
    .with_detail("skip_rows", skip_rows.to_string())
    .with_detail("num_rows", num_rows_context)
    .with_detail("resource", "row_groups")
    .with_detail("reason", "incompatible_option_combination")
    .with_detail("remedy", "use row_groups by itself, or remove row_groups and use skip_rows/num_rows")
}

/// Read a Parquet file into a low-level table result.
///
/// Returns the table data and the column names stored in the Parquet file's
/// schema. This avoids having to hardcode column indices when working with
/// Parquet files.
pub fn read_parquet_table(options: &ParquetTableReadOptions) -> Result<TableReadOutput> {
    const OPERATION: &str = "io::parquet::read_parquet_table";
    let filepath = super::path_to_cstring(&options.filepath, OPERATION)?;
    super::validate_input_path(&options.filepath, OPERATION)?;
    let num_rows_context = options.num_rows.map(|num_rows| num_rows.to_string()).unwrap_or_else(|| "all".to_string());

    // Validate that row_groups is not combined with skip_rows or num_rows.
    if options.row_groups.is_some() && (options.skip_rows > 0 || options.num_rows.is_some()) {
        return Err(row_group_combination_error(
            options.row_groups.as_ref().map_or(0, Vec::len),
            options.skip_rows,
            num_rows_context,
            OPERATION,
        ));
    }
    let skip_rows = checked_i64_from_usize(options.skip_rows, "skip_rows", OPERATION)?;
    let num_rows = match options.num_rows {
        Some(num_rows) => checked_i64_from_usize(num_rows, "num_rows", OPERATION)?,
        None => -1,
    };

    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.exec_options_raw();

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;

    // Convert column names to CStrings at the call site, then build the pointer array.
    let cstring_columns: Option<Vec<CString>> = options
        .columns
        .as_ref()
        .map(|cols| {
            cols.iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            OPERATION,
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
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
    let predicate = options.predicate.as_ref().map_or(ptr::null_mut(), |predicate| predicate.raw_handle());

    error::check_status_in_with_details(
        // SAFETY: filepath is a valid CString. col_ptrs is either null or a
        // valid pointer to num_cols CString pointers. rg_ptr is either null
        // or a valid pointer to num_rg i32 values. predicate is either null
        // or an owned AST handle retained by options for the duration of this call.
        unsafe {
            cudf_sys::cudf_io_read_parquet_with_metadata_filter_ex(
                filepath.as_ptr(),
                col_ptrs,
                num_cols,
                skip_rows,
                num_rows,
                rg_ptr,
                num_rg,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &exec_options,
            )
        },
        OPERATION,
        [
            ("column_name_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", num_rows_context),
        ],
    )?;

    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    // Wrap early so table is freed via RAII if collect_column_names() fails.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;

    // Convert the C string array into Vec<String> and free the C memory.
    // SAFETY: names_ptr and num_names were returned by a successful FFI call.
    // cudf_io_free_column_names frees the C-allocated name strings and array.
    let column_names = unsafe {
        let names = super::collect_column_names(names_ptr, num_names, OPERATION);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };

    Ok(TableReadOutput { table, column_names: Some(column_names) })
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
pub fn write_parquet(table: &Table, options: &ParquetTableWriteOptions) -> Result<()> {
    let operation = "io::parquet::write_parquet";
    let table_view = table.view()?;
    let execution = super::resolve_write_execution(&table_view, options.runtime(), operation)?;
    write_parquet_with_exec(&table_view, options, execution.exec_options())
}

pub(crate) fn write_parquet_with_exec(
    table: &crate::core::TableView<'_>,
    options: &ParquetTableWriteOptions,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::write_parquet")?;

    // Validate column name count matches the table before the FFI call.
    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()?;
        if names.len() != expected {
            return Err(schema::column_name_count_mismatch_error(
                "io::parquet::write_parquet",
                format!("column name count ({}) does not match table column count ({})", names.len(), expected),
                names.len(),
                expected,
            ));
        }
    }

    // Convert column names to CStrings at the call site.
    let cstring_names: Option<Vec<CString>> = options
        .column_names
        .as_ref()
        .map(|names| {
            names
                .iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            "io::parquet::write_parquet",
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
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

    error::check_status_in_with_details(
        // SAFETY: table handle is valid (NonNull). filepath is a valid CString.
        // name_ptrs is either null or a valid pointer to num_cols CString pointers.
        unsafe {
            cudf_sys::cudf_io_write_parquet_ex(
                table.raw_handle(),
                filepath.as_ptr(),
                name_ptrs,
                num_cols,
                options.row_group_size_rows,
                exec_options,
            )
        },
        "io::parquet::write_parquet",
        [("column_name_count", num_cols.to_string())],
    )
}
