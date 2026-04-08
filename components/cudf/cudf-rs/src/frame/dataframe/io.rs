use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use typed_builder::TypedBuilder;

use super::owned::DataFrame;
use crate::{
    core::error::{Error, ErrorKind, Result},
    io::{
        IoTableResult,
        csv::{self, CsvReaderOptions, CsvWriterOptions},
        json::{self, JsonReaderOptions, JsonWriterOptions},
        parquet::{self, ParquetReaderOptions, ParquetWriterOptions},
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// Options for [`DataFrame::read_csv_with_options`].
///
/// This is the schema-aware CSV reader API for `DataFrame`. Unlike the
/// lower-level [`crate::io::csv::CsvReaderOptions`], it can also resolve
/// DataFrame column names from either an explicit override, low-level header
/// metadata, or generated fallback names.
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvReadOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default = b',')]
    delimiter: u8,
    #[builder(default = true)]
    has_header: bool,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    column_names: Option<Vec<String>>,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl CsvReadOptions {
    fn to_table_options(&self) -> CsvReaderOptions {
        match self.runtime.as_ref() {
            Some(rt) => CsvReaderOptions::builder()
                .filepath(&self.filepath)
                .delimiter(self.delimiter)
                .has_header(self.has_header)
                .with_runtime(rt)
                .build(),
            None => CsvReaderOptions::builder()
                .filepath(&self.filepath)
                .delimiter(self.delimiter)
                .has_header(self.has_header)
                .build(),
        }
    }

    fn resolve_column_names(&self, result: &IoTableResult) -> Result<Vec<String>> {
        let operation = "dataframe::read_csv_with_options";
        let num_columns = {
            let view = result.table.view()?;
            view.num_columns()? as usize
        };

        if let Some(names) = self.column_names.as_ref() {
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names.clone());
        }

        if let Some(names) = result.column_names.as_ref() {
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names.clone());
        }

        Ok((0..num_columns).map(|idx| format!("column_{idx}")).collect())
    }
}

/// Options for [`DataFrame::write_csv_with_options`].
///
/// This is the schema-aware CSV writer API for `DataFrame`. Column names
/// always come from the DataFrame schema; this type only controls
/// file-format knobs such as delimiter and whether the header row is
/// included.
#[derive(Debug, Clone, TypedBuilder)]
pub struct CsvWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default = b',')]
    delimiter: u8,
    #[builder(default = true)]
    include_header: bool,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl CsvWriteOptions {
    fn to_table_options(&self, column_names: &[&str]) -> CsvWriterOptions {
        match self.runtime.as_ref() {
            Some(rt) => CsvWriterOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .delimiter(self.delimiter)
                .include_header(self.include_header)
                .with_runtime(rt)
                .build(),
            None => CsvWriterOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .delimiter(self.delimiter)
                .include_header(self.include_header)
                .build(),
        }
    }
}

fn validate_dataframe_csv_column_name_count(
    name_count: usize,
    num_columns: usize,
    operation: &'static str,
) -> Result<()> {
    if name_count != num_columns {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("column name count ({name_count}) does not match table column count ({num_columns})"),
        )
        .with_operation(operation)
        .with_context("column_name_count", name_count.to_string())
        .with_context("table_column_count", num_columns.to_string()));
    }

    Ok(())
}

fn prepare_write_domain(
    frame_domain: &Arc<ExecutionDomain>,
    write_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<()> {
    ExecutionDomain::require_same_device_for_mixed_inputs(frame_domain, write_domain, operation).map_err(|e| {
        e.with_context("frame_domain", frame_domain.debug_label())
            .with_context("write_domain", write_domain.debug_label())
    })?;

    if !ExecutionDomain::same_provenance(frame_domain, write_domain) {
        frame_domain.fence_stream(operation)?;
    }

    Ok(())
}

impl DataFrame {
    /// Read a CSV file into a `DataFrame`.
    ///
    /// If the caller does not provide explicit names through
    /// [`CsvReadOptions::column_names`], names come from the header
    /// row when `has_header == true`, or fall back to generated names such as
    /// `"column_0"` when reading headerless CSV.
    pub fn read_csv(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_csv_with_options(CsvReadOptions::builder().filepath(path).build())
    }

    /// Read a CSV file into a `DataFrame` using caller-provided options.
    ///
    /// When the low-level CSV reader returns truthful header metadata, those
    /// names are used unless the caller provides an explicit override.
    /// Generated fallback names are used only for headerless CSV results.
    pub fn read_csv_with_options(options: CsvReadOptions) -> Result<Self> {
        let result = csv::read_csv(&options.to_table_options())?;
        let names = options.resolve_column_names(&result)?;
        Self::from_table(result.table, names)
    }

    /// Write this `DataFrame` to a CSV file, preserving the DataFrame schema
    /// as the output header row.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_csv_with_options(CsvWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a CSV file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output column
    /// names so the high-level round-trip path remains truthful.
    pub fn write_csv_with_options(&self, options: CsvWriteOptions) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let opts = options.to_table_options(&names);
        let write_domain = if let Some(runtime) = opts.runtime() {
            Arc::clone(runtime.execution_domain())
        } else {
            Arc::clone(&self.domain)
        };
        prepare_write_domain(&self.domain, &write_domain, "dataframe::write_csv_with_options")?;
        let tv = self.build_table_view()?;
        let exec_options = write_domain.explicit_exec_options_raw();
        csv::write_csv_with_exec(tv.table(), &opts, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_csv_with_options")
    }

    /// Read a Parquet file into a `DataFrame`, using column names from the
    /// file's schema metadata.
    ///
    /// This is a convenience wrapper around
    /// [`read_parquet_with_options`](Self::read_parquet_with_options).
    pub fn read_parquet(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_parquet_with_options(ParquetReaderOptions::builder().filepath(path).build())
    }

    /// Read a Parquet file into a `DataFrame` using caller-provided
    /// [`ParquetReaderOptions`].
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    pub fn read_parquet_with_options(options: ParquetReaderOptions) -> Result<Self> {
        let result = parquet::read_parquet(options)?;
        let names = result.column_names.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "parquet reader did not return column names; this is unexpected for the parquet format",
            )
            .with_operation("dataframe::read_parquet_with_options")
        })?;
        Self::from_table(result.table, names)
    }

    /// Write this `DataFrame` to a Parquet file, embedding column names in
    /// the file's schema metadata.
    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_parquet_with_options(ParquetWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a Parquet file using caller-provided options.
    pub fn write_parquet_with_options(&self, options: ParquetWriteOptions) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let table_options = options.to_table_options(&names);
        let write_domain = if let Some(runtime) = table_options.runtime() {
            Arc::clone(runtime.execution_domain())
        } else {
            Arc::clone(&self.domain)
        };
        prepare_write_domain(&self.domain, &write_domain, "dataframe::write_parquet_with_options")?;
        let tv = self.build_table_view()?;
        let exec_options = write_domain.explicit_exec_options_raw();
        parquet::write_parquet_with_exec(tv.table(), &table_options, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_parquet_with_options")
    }

    // -----------------------------------------------------------------------
    // JSON I/O
    // -----------------------------------------------------------------------

    /// Read a JSON Lines (NDJSON) file into a `DataFrame`, using key names
    /// from the JSON objects as column names.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_json_with_options(JsonReaderOptions::builder().filepath(path).build())
    }

    /// Read a JSON file into a `DataFrame` using caller-provided
    /// [`JsonReaderOptions`].
    ///
    /// Column names are extracted from JSON key names when the low-level
    /// reader can recover them truthfully from the file.
    pub fn read_json_with_options(options: JsonReaderOptions) -> Result<Self> {
        let result = json::read_json(&options)?;
        Self::from_table(result.table, result.column_names.unwrap_or_default())
    }

    /// Write this `DataFrame` to a JSON Lines (NDJSON) file, using column
    /// names from the DataFrame schema as JSON keys.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_json_with_options(JsonWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a JSON file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output JSON key
    /// names so the high-level round-trip path remains truthful.
    pub fn write_json_with_options(&self, options: JsonWriteOptions) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let opts = options.to_table_options(&names);
        let write_domain = if let Some(runtime) = opts.runtime() {
            Arc::clone(runtime.execution_domain())
        } else {
            Arc::clone(&self.domain)
        };
        prepare_write_domain(&self.domain, &write_domain, "dataframe::write_json_with_options")?;
        let tv = self.build_table_view()?;
        let exec_options = write_domain.explicit_exec_options_raw();
        json::write_json_with_exec(tv.table(), &opts, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_json_with_options")
    }
}

/// Options for [`DataFrame::write_json_with_options`].
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
    fn to_table_options(&self, column_names: &[&str]) -> JsonWriterOptions {
        match self.runtime.as_ref() {
            Some(rt) => JsonWriterOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .lines(self.lines)
                .include_nulls(self.include_nulls)
                .with_runtime(rt)
                .build(),
            None => JsonWriterOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .lines(self.lines)
                .include_nulls(self.include_nulls)
                .build(),
        }
    }
}

/// Options for [`DataFrame::write_parquet_with_options`].
///
/// Column names always come from the DataFrame schema. This type only
/// controls file-format knobs such as row group size.
///
/// # Defaults
///
/// - `row_group_size_rows` = `0` (use libcudf default)
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    /// Target number of rows per row group. `0` (default) uses the libcudf
    /// default.
    #[builder(default)]
    row_group_size_rows: i32,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetWriteOptions {
    fn to_table_options(&self, column_names: &[&str]) -> ParquetWriterOptions {
        let base = ParquetWriterOptions::builder()
            .filepath(&self.filepath)
            .column_names(column_names)
            .row_group_size_rows(self.row_group_size_rows);
        match &self.runtime {
            Some(rt) => base.with_runtime(rt).build(),
            None => base.build(),
        }
    }
}
