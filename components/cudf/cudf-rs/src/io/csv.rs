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
        error::{self, FACT_COLUMN_INDEX, Result},
        schema,
        table::Table,
    },
    io::TableReadOutput,
    runtime::Runtime,
};

/// Options for reading a CSV file into a low-level [`Table`].
///
/// **Advanced.** Most users should prefer [`DataFrame::read_csv`](crate::DataFrame::read_csv).
///
/// This type controls CSV parsing only. Truthful schema names, when available
/// from a header row, are returned through [`TableReadOutput`].
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvTableReadOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default = b',')]
    delimiter: u8,
    #[builder(default = true)]
    has_header: bool,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

/// Read a CSV file into a low-level table result.
pub fn read_csv(options: &CsvTableReadOptions) -> Result<TableReadOutput> {
    let operation = "io::csv::read_csv";
    let filepath = super::path_to_cstring(&options.filepath, operation)?;
    super::validate_input_path(&options.filepath, operation)?;
    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.exec_options_raw();

    let mut table_handle = ptr::null_mut();
    let _io_guard = super::lock_libcudf_io(operation)?;
    error::check_status_in_with_details(
        // SAFETY: filepath is a valid CString (converted above).
        unsafe {
            cudf_sys::cudf_io_read_csv_ex(
                filepath.as_ptr(),
                options.delimiter as std::os::raw::c_char,
                if options.has_header { 1 } else { 0 },
                &mut table_handle,
                &exec_options,
            )
        },
        operation,
        [("delimiter_char", (options.delimiter as char).to_string()), ("has_header", options.has_header.to_string())],
    )?;
    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;
    let num_columns = {
        let view = table.view()?;
        view.num_columns()?
    };
    let column_names = if options.has_header {
        let names = read_header_names(&options.filepath, options.delimiter)?;
        validate_header_name_count(names.len(), num_columns)?;
        Some(names)
    } else {
        None
    };

    Ok(TableReadOutput { table, column_names })
}

/// Options for writing a low-level [`Table`] to a CSV file.
///
/// **Advanced.** Most users should prefer [`DataFrame::write_csv`](crate::DataFrame::write_csv).
///
/// Column names here control only the emitted CSV header row. They do not
/// rename the input table because [`Table`] itself is unnamed.
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvTableWriteOptions {
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

impl CsvTableWriteOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Write a table to a CSV file.
pub fn write_csv(table: &Table, options: &CsvTableWriteOptions) -> Result<()> {
    let operation = "io::csv::write_csv";
    let table_view = table.view()?;
    let execution = super::resolve_write_execution(&table_view, options.runtime(), operation)?;
    write_csv_with_exec(&table_view, options, execution.exec_options())
}

pub(crate) fn write_csv_with_exec(
    table: &crate::core::TableView<'_>,
    options: &CsvTableWriteOptions,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::csv::write_csv")?;

    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()?;
        if names.len() != expected {
            return Err(schema::column_name_count_mismatch_error(
                "io::csv::write_csv",
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
                            "io::csv::write_csv",
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

    // Build the pointer array outside the match so it outlives the FFI call.
    let name_ptrs_vec: Vec<*const std::os::raw::c_char> =
        cstring_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) =
        if name_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32) };

    let _io_guard = super::lock_libcudf_io("io::csv::write_csv")?;
    error::check_status_in_with_details(
        // SAFETY: table handle is valid (NonNull). filepath is a valid CString (converted above).
        // name_ptrs is either null or a valid pointer to num_cols CString pointers.
        unsafe {
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
        "io::csv::write_csv",
        [
            ("column_name_count", num_cols.to_string()),
            ("delimiter_char", (options.delimiter as char).to_string()),
            ("include_header", options.include_header.to_string()),
        ],
    )
}

fn read_header_names(path: &Path, delimiter: u8) -> Result<Vec<String>> {
    let operation = "io::csv::read_csv";
    let mut reader =
        ::csv::ReaderBuilder::new().has_headers(true).delimiter(delimiter).from_path(path).map_err(|source| {
            super::map_path_io_error(source, path, operation, "failed to open CSV file for header parsing")
        })?;

    let headers = reader
        .headers()
        .map_err(|source| super::map_path_io_error(source, path, operation, "failed to read CSV header"))?;

    Ok(headers.iter().map(str::to_owned).collect())
}

fn validate_header_name_count(name_count: usize, num_columns: usize) -> Result<()> {
    if name_count != num_columns {
        return Err(schema::column_name_count_mismatch_error(
            "io::csv::read_csv",
            format!("header column count ({name_count}) does not match table column count ({num_columns})"),
            name_count,
            num_columns,
        ));
    }

    Ok(())
}
