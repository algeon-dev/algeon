//! Parquet table writer.

use std::{
    ffi::CString,
    os::raw::c_char,
    path::{Path, PathBuf},
    ptr,
};

use typed_builder::TypedBuilder;

use super::super::{path_to_cstring, resolve_write_execution};
use crate::{
    core::{
        error::{self, FACT_COLUMN_INDEX, Result},
        schema,
        table::Table,
    },
    runtime::Runtime,
};

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
    #[builder(default)]
    dictionary_policy: ParquetDictionaryPolicy,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetTableWriteOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Parquet dictionary encoding policy for cuDF writes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ParquetDictionaryPolicy {
    /// Use libcudf's default writer policy.
    #[default]
    Default,
    /// Use libcudf's adaptive dictionary policy.
    Adaptive,
    /// Disable dictionary encoding.
    Never,
    /// Always attempt dictionary encoding.
    Always,
}

impl ParquetDictionaryPolicy {
    pub(crate) const fn ffi_value(self) -> i32 {
        match self {
            Self::Default => -1,
            Self::Adaptive => 0,
            Self::Never => 1,
            Self::Always => 2,
        }
    }
}

/// Write a table to a Parquet file.
///
/// Returns an error if `column_names` were set on the options and their count
/// does not match the number of columns in the table.
pub fn write_parquet(table: &Table, options: &ParquetTableWriteOptions) -> Result<()> {
    let operation = "io::parquet::write_parquet";
    let table_view = table.view()?;
    let execution = resolve_write_execution(&table_view, options.runtime(), operation)?;
    write_parquet_with_exec(&table_view, options, execution.exec_options())
}

pub(crate) fn write_parquet_with_exec(
    table: &crate::core::TableView<'_>,
    options: &ParquetTableWriteOptions,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<()> {
    let filepath = path_to_cstring(&options.filepath, "io::parquet::write_parquet")?;

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
    let mut name_ptrs_vec: Vec<*const c_char> =
        cstring_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) = if name_ptrs_vec.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        (name_ptrs_vec.as_mut_ptr(), name_ptrs_vec.len() as i32)
    };

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
                options.dictionary_policy.ffi_value(),
                exec_options,
            )
        },
        "io::parquet::write_parquet",
        [("column_name_count", num_cols.to_string())],
    )
}
