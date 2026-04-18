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
        csv::{self, CsvTableReadOptions, CsvTableWriteOptions},
        json::{self, JsonTableReadOptions, JsonTableWriteOptions},
        parquet::{self, ParquetTableReadOptions, ParquetTableWriteOptions},
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// Options for [`DataFrame::read_csv_with_options`].
///
/// This is the schema-aware CSV reader API for `DataFrame`. Parsing knobs such
/// as delimiter and header handling are forwarded to the low-level CSV reader.
/// Optional `column_names` override the file header only at the DataFrame
/// schema layer; they do not change the low-level reader metadata contract.
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
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl CsvReadOptions {
    fn to_table_options(&self) -> CsvTableReadOptions {
        CsvTableReadOptions::builder()
            .filepath(&self.filepath)
            .delimiter(self.delimiter)
            .has_header(self.has_header)
            .with_runtime(&self.runtime)
            .build()
    }

    fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
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
    fn to_table_options(&self, column_names: &[&str]) -> CsvTableWriteOptions {
        match self.runtime.as_ref() {
            Some(rt) => CsvTableWriteOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .delimiter(self.delimiter)
                .include_header(self.include_header)
                .with_runtime(rt)
                .build(),
            None => CsvTableWriteOptions::builder()
                .filepath(&self.filepath)
                .column_names(column_names)
                .delimiter(self.delimiter)
                .include_header(self.include_header)
                .build(),
        }
    }
}

/// Options for [`DataFrame::read_parquet_with_options`].
///
/// This is the schema-aware Parquet reader API for `DataFrame`. It controls
/// file-format slicing and column selection, then lifts truthful Parquet
/// schema metadata into the resulting `DataFrame`.
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
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl ParquetReadOptions {
    fn to_table_options(&self) -> ParquetTableReadOptions {
        ParquetTableReadOptions::from_parts(
            self.filepath.clone(),
            self.columns.clone(),
            self.skip_rows,
            self.num_rows,
            self.row_groups.clone(),
            self.runtime.clone(),
        )
    }
}

/// Options for [`DataFrame::read_json_with_options`].
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
    fn to_table_options(&self) -> JsonTableReadOptions {
        JsonTableReadOptions::from_parts(self.filepath.clone(), self.lines, self.runtime.clone())
    }
}

fn resolve_csv_column_names(options: &CsvReadOptions, result: &IoTableResult) -> Result<Vec<String>> {
    let operation = "dataframe::read_csv_with_options";
    let num_columns = {
        let view = result.table.view()?;
        view.num_columns()?
    };

    if let Some(names) = options.column_names() {
        validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
        return Ok(names.to_vec());
    }

    if let Some(names) = result.column_names.as_ref() {
        validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
        return Ok(names.clone());
    }

    Ok((0..num_columns).map(|idx| format!("column_{idx}")).collect())
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
    ///
    /// # Cost
    /// Same as [`read_csv_with_options`](Self::read_csv_with_options):
    /// host file I/O, GPU allocation for the decoded columns, and host-to-
    /// device transfer of parsed field data.
    pub fn read_csv(runtime: &Runtime, path: impl AsRef<Path>) -> Result<Self> {
        Self::read_csv_with_options(CsvReadOptions::builder().filepath(path).with_runtime(runtime).build())
    }

    /// Read a CSV file into a `DataFrame` using caller-provided options.
    ///
    /// When the low-level CSV reader returns truthful header metadata, those
    /// names are used unless the caller provides an explicit override.
    /// Generated fallback names are used only for headerless CSV results.
    ///
    /// # Cost
    /// Performs host file I/O, allocates new GPU-backed columns for the
    /// decoded result, and copies parsed field data into device memory. When
    /// `has_header` is `true`, this high-level API also parses the CSV header
    /// on the host to recover truthful column names.
    pub fn read_csv_with_options(options: CsvReadOptions) -> Result<Self> {
        let table_options = options.to_table_options();
        let result = csv::read_csv(&table_options)?;
        let names = resolve_csv_column_names(&options, &result)?;
        Self::from_table(result.table, names)
    }

    /// Write this `DataFrame` to a CSV file, preserving the DataFrame schema
    /// as the output header row.
    ///
    /// # Cost
    /// Same as [`write_csv_with_options`](Self::write_csv_with_options):
    /// serializes device-backed columns to a host file sink and synchronizes
    /// the retained write execution domain before returning.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_csv_with_options(CsvWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a CSV file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output column
    /// names so the high-level round-trip path remains truthful.
    ///
    /// # Cost
    /// Serializes device-backed columns to a host file sink. This may fence
    /// the DataFrame's retained execution domain before launching the write
    /// when the caller supplies a different write runtime on the same device,
    /// and it synchronizes the retained write execution domain before
    /// returning so the output file is host-visible on success.
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
        csv::write_csv_with_exec(&tv, &opts, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_csv_with_options")
    }

    /// Read a Parquet file into a `DataFrame`, using column names from the
    /// file's schema metadata.
    ///
    /// This is a convenience wrapper around
    /// [`read_parquet_with_options`](Self::read_parquet_with_options).
    ///
    /// # Cost
    /// Same as [`read_parquet_with_options`](Self::read_parquet_with_options):
    /// host file I/O, GPU allocation for the decoded columns, and host-to-
    /// device transfer of decoded column data.
    pub fn read_parquet(runtime: &Runtime, path: impl AsRef<Path>) -> Result<Self> {
        Self::read_parquet_with_options(ParquetReadOptions::builder().filepath(path).with_runtime(runtime).build())
    }

    /// Read a Parquet file into a `DataFrame` using caller-provided
    /// [`ParquetReadOptions`].
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    ///
    /// # Cost
    /// Performs host file I/O, allocates new GPU-backed columns for the
    /// decoded result, and copies decoded column data into device memory.
    /// Column names are recovered from Parquet schema metadata rather than
    /// synthesized heuristically.
    pub fn read_parquet_with_options(options: ParquetReadOptions) -> Result<Self> {
        let table_options = options.to_table_options();
        let result = parquet::read_parquet_table(&table_options)?;
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
    ///
    /// # Cost
    /// Same as
    /// [`write_parquet_with_options`](Self::write_parquet_with_options):
    /// serializes device-backed columns to a host file sink and synchronizes
    /// the retained write execution domain before returning.
    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_parquet_with_options(ParquetWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a Parquet file using caller-provided options.
    ///
    /// # Cost
    /// Serializes device-backed columns to a host file sink. This may fence
    /// the DataFrame's retained execution domain before launching the write
    /// when the caller supplies a different write runtime on the same device,
    /// and it synchronizes the retained write execution domain before
    /// returning so the output file is host-visible on success.
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
        parquet::write_parquet_with_exec(&tv, &table_options, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_parquet_with_options")
    }

    // -----------------------------------------------------------------------
    // JSON I/O
    // -----------------------------------------------------------------------

    /// Read a JSON Lines (NDJSON) file into a `DataFrame`, using key names
    /// from the JSON objects as column names.
    ///
    /// # Cost
    /// Same as [`read_json_with_options`](Self::read_json_with_options):
    /// host file I/O, GPU allocation for the decoded columns, and host-to-
    /// device transfer of decoded field data.
    pub fn read_json(runtime: &Runtime, path: impl AsRef<Path>) -> Result<Self> {
        Self::read_json_with_options(JsonReadOptions::builder().filepath(path).with_runtime(runtime).build())
    }

    /// Read a JSON file into a `DataFrame` using caller-provided
    /// [`JsonReadOptions`].
    ///
    /// Column names are extracted from JSON key names when the low-level
    /// reader can recover them truthfully from the file.
    ///
    /// # Cost
    /// Performs host file I/O, allocates new GPU-backed columns for the
    /// decoded result, and copies decoded field data into device memory.
    pub fn read_json_with_options(options: JsonReadOptions) -> Result<Self> {
        let table_options = options.to_table_options();
        let result = json::read_json_table(&table_options)?;
        Self::from_table(result.table, result.column_names.unwrap_or_default())
    }

    /// Write this `DataFrame` to a JSON Lines (NDJSON) file, using column
    /// names from the DataFrame schema as JSON keys.
    ///
    /// # Cost
    /// Same as [`write_json_with_options`](Self::write_json_with_options):
    /// serializes device-backed columns to a host file sink and synchronizes
    /// the retained write execution domain before returning.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_json_with_options(JsonWriteOptions::builder().filepath(path).build())
    }

    /// Write this `DataFrame` to a JSON file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output JSON key
    /// names so the high-level round-trip path remains truthful.
    ///
    /// # Cost
    /// Serializes device-backed columns to a host file sink. This may fence
    /// the DataFrame's retained execution domain before launching the write
    /// when the caller supplies a different write runtime on the same device,
    /// and it synchronizes the retained write execution domain before
    /// returning so the output file is host-visible on success.
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
        json::write_json_with_exec(&tv, &opts, exec_options.as_ref())?;
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
    fn to_table_options(&self, column_names: &[&str]) -> JsonTableWriteOptions {
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
    fn to_table_options(&self, column_names: &[&str]) -> ParquetTableWriteOptions {
        let base = ParquetTableWriteOptions::builder()
            .filepath(&self.filepath)
            .column_names(column_names)
            .row_group_size_rows(self.row_group_size_rows);
        match &self.runtime {
            Some(rt) => base.with_runtime(rt).build(),
            None => base.build(),
        }
    }
}
