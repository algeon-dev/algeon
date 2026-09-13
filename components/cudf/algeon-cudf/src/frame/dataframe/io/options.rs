//! Parquet and JSON read and write option types.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use typed_builder::TypedBuilder;

use crate::{
    core::error::{Error, ErrorCode, Result},
    io::{
        json::{JsonTableReadOptions, JsonTableWriteOptions},
        parquet::{ParquetDictionaryPolicy, ParquetTableReadOptions, ParquetTableWriteOptions},
    },
    ops::ast::AstExpression,
    runtime::Runtime,
};

/// Options for [`DataFrame::read_parquet_with_options`](crate::DataFrame::read_parquet_with_options).
///
/// This is the schema-aware Parquet reader API for `DataFrame`. It selects one
/// source (`filepath` for path-based reads, or `filebytes` for borrowed
/// host-buffer reads), controls file-format slicing and column selection, then
/// lifts truthful Parquet schema metadata into the resulting `DataFrame`. A
/// supplied projection must contain at least one column name; omit `columns` to
/// read all columns.
///
/// `filebytes` is borrowed only for the duration of the reader call. The
/// returned `DataFrame` owns device-resident decoded columns and does not
/// retain a pointer into the host buffer.
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetReadOptions<'a> {
    #[builder(default, setter(transform = |path: impl AsRef<Path>| Some(path.as_ref().to_path_buf())))]
    filepath: Option<PathBuf>,
    #[builder(default, setter(transform = |bytes: &'a [u8]| Some(bytes)))]
    filebytes: Option<&'a [u8]>,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    pub(super) columns: Option<Vec<String>>,
    #[builder(default)]
    skip_rows: usize,
    #[builder(default, setter(transform = |num_rows: usize| Some(num_rows)))]
    num_rows: Option<usize>,
    #[builder(default, setter(transform = |groups: &[i32]| if groups.is_empty() { None } else { Some(groups.to_vec()) }))]
    row_groups: Option<Vec<i32>>,
    #[builder(default, setter(strip_option, into))]
    predicate: Option<Arc<AstExpression>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    pub(super) runtime: Runtime,
}

impl<'a> ParquetReadOptions<'a> {
    pub(super) fn to_table_options(&self) -> Result<ParquetTableReadOptions<'a>> {
        match (self.filepath.clone(), self.filebytes) {
            (Some(filepath), None) => Ok(ParquetTableReadOptions::from_parts(
                filepath,
                self.columns.clone(),
                self.skip_rows,
                self.num_rows,
                self.row_groups.clone(),
                self.predicate.clone(),
                self.runtime.clone(),
            )),
            (None, Some(filebytes)) => Ok(ParquetTableReadOptions::from_host_buffer_parts(
                filebytes,
                self.columns.clone(),
                self.skip_rows,
                self.num_rows,
                self.row_groups.clone(),
                self.predicate.clone(),
                self.runtime.clone(),
            )),
            (Some(_), Some(_)) => Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                "dataframe::read_parquet_with_options",
                "Parquet read options must select exactly one source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "ambiguous_source")
            .with_detail("remedy", "set either filepath or filebytes, not both")),
            (None, None) => Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                "dataframe::read_parquet_with_options",
                "Parquet read options must select a source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "missing_source")
            .with_detail("remedy", "set filepath for path reads or filebytes for host-buffer reads")),
        }
    }

    pub(super) fn to_host_buffer_table_options<'b>(&self, filebytes: &'b [u8]) -> ParquetTableReadOptions<'b> {
        ParquetTableReadOptions::from_host_buffer_parts(
            filebytes,
            self.columns.clone(),
            self.skip_rows,
            self.num_rows,
            self.row_groups.clone(),
            self.predicate.clone(),
            self.runtime.clone(),
        )
    }
}

/// Options for [`DataFrame::read_parquet_many_with_options`](crate::DataFrame::read_parquet_many_with_options).
///
/// This is a conservative host-filesystem convenience for reading multiple
/// Parquet files in caller-provided order, then concatenating the decoded
/// tables on device. It does not perform glob expansion; each path is passed
/// to the low-level Parquet reader literally. If globbing is desired, expand
/// patterns in the shell or caller before building these options. It also does
/// not perform predicate pushdown, partition-column inference, or GDS. A
/// supplied projection must contain at least one column name; omit `columns` to
/// read all columns.
#[derive(Debug, Clone, TypedBuilder)]
pub struct MultiParquetReadOptions {
    #[builder(setter(transform = |paths: &[impl AsRef<Path>]| paths.iter().map(|path| path.as_ref().to_path_buf()).collect()))]
    pub(super) filepaths: Vec<PathBuf>,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    pub(super) columns: Option<Vec<String>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl MultiParquetReadOptions {
    pub(super) fn from_parts(filepaths: Vec<PathBuf>, columns: Option<Vec<String>>, runtime: Runtime) -> Self {
        Self { filepaths, columns, runtime }
    }

    pub(super) fn to_single_file_options(&self, filepath: PathBuf) -> ParquetTableReadOptions<'static> {
        ParquetTableReadOptions::from_parts(filepath, self.columns.clone(), 0, None, None, None, self.runtime.clone())
    }
}

/// Options for [`DataFrame::read_json_with_options`](crate::DataFrame::read_json_with_options).
///
/// This is the schema-aware JSON reader API for `DataFrame`. It controls the
/// file-format interpretation while keeping JSON object keys as DataFrame
/// schema names when the reader can recover them truthfully.
///
/// # Defaults
///
/// - `lines` = `true` (JSON Lines / NDJSON format, one JSON object per line)
#[derive(Debug, Clone, TypedBuilder)]
pub struct JsonReadOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default = true)]
    lines: bool,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl JsonReadOptions {
    pub(super) fn to_table_options(&self) -> JsonTableReadOptions {
        JsonTableReadOptions::from_parts(self.filepath.clone(), self.lines, self.runtime.clone())
    }
}

pub(super) fn empty_parquet_projection_error(operation: &'static str) -> Error {
    Error::new_with(ErrorCode::EmptyInput, operation, "parquet projection requires at least one column name")
        .with_detail("resource", "columns")
        .with_detail("reason", "empty_projection")
        .with_detail("column_name_count", "0")
        .with_detail("remedy", "omit the projection to read all columns, or provide at least one column name")
}

pub(super) fn validate_parquet_projection(columns: Option<&[String]>, operation: &'static str) -> Result<()> {
    if matches!(columns, Some(names) if names.is_empty()) {
        return Err(empty_parquet_projection_error(operation));
    }

    Ok(())
}

/// Options for [`DataFrame::write_json_with_options`](crate::DataFrame::write_json_with_options).
///
/// Column names always come from the DataFrame schema. This type only
/// controls file-format knobs such as whether to write JSON Lines format
/// and whether to include null fields.
///
/// # Defaults
///
/// - `lines` = `true` (JSON Lines / NDJSON format)
/// - `include_nulls` = `true`
#[derive(Debug, Clone, TypedBuilder)]
pub struct JsonWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// If `true` (default), write as JSON Lines (one JSON object per line).
    #[builder(default = true)]
    lines: bool,
    /// If `true` (default), include null fields in output.
    #[builder(default = true)]
    include_nulls: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl JsonWriteOptions {
    pub(super) fn to_table_options(&self, column_names: &[&str]) -> JsonTableWriteOptions {
        match self.runtime.as_ref() {
            Some(rt) => JsonTableWriteOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .lines(self.lines)
                .include_nulls(self.include_nulls)
                .with_runtime(rt)
                .build(),
            None => JsonTableWriteOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .lines(self.lines)
                .include_nulls(self.include_nulls)
                .build(),
        }
    }
}

/// Options for [`DataFrame::write_parquet_with_options`](crate::DataFrame::write_parquet_with_options).
///
/// Column names always come from the DataFrame schema. This type only
/// controls file-format knobs such as row group size.
///
/// # Defaults
///
/// - `row_group_size_rows` = `0` (use libcudf default)
/// - `dictionary_policy` = libcudf default
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// Target number of rows per row group. `0` (default) uses the libcudf
    /// default.
    #[builder(default)]
    row_group_size_rows: i32,
    /// Dictionary encoding policy. The default keeps libcudf's writer default.
    #[builder(default)]
    dictionary_policy: ParquetDictionaryPolicy,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetWriteOptions {
    pub(super) fn to_table_options(&self, column_names: &[&str]) -> ParquetTableWriteOptions {
        let base = ParquetTableWriteOptions::builder()
            .filepath(&self.filepath)
            .column_names(column_names)
            .row_group_size_rows(self.row_group_size_rows)
            .dictionary_policy(self.dictionary_policy);
        match &self.runtime {
            Some(rt) => base.with_runtime(rt).build(),
            None => base.build(),
        }
    }
}
