use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use super::owned::DataFrame;
use crate::{
    core::error::{Error, ErrorKind, Result},
    io::{
        csv::{self, CsvReaderOptions, CsvWriterOptions},
        parquet::{self, ParquetReaderOptions, ParquetWriterOptions},
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// Options for [`DataFrame::read_csv_with_options`].
///
/// This is the schema-aware CSV reader API for `DataFrame`. Unlike the
/// lower-level [`crate::io::csv::CsvReaderOptions`], it can also resolve
/// DataFrame column names from either an explicit override, the CSV header
/// row, or generated fallback names.
#[derive(Debug, Clone)]
pub struct DataFrameCsvReaderOptions {
    filepath: PathBuf,
    delimiter: u8,
    has_header: bool,
    column_names: Option<Vec<String>>,
    runtime: Option<Runtime>,
}

impl DataFrameCsvReaderOptions {
    /// Create options to read a CSV file into a `DataFrame`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            filepath: path.as_ref().to_path_buf(),
            delimiter: b',',
            has_header: true,
            column_names: None,
            runtime: None,
        }
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether the file has a header row (default: `true`).
    pub fn has_header(mut self, has: bool) -> Self {
        self.has_header = has;
        self
    }

    /// Override the output DataFrame column names.
    ///
    /// When set, these names take precedence over both header-derived names
    /// and generated fallback names.
    pub fn column_names(mut self, names: &[&str]) -> Self {
        self.column_names = Some(names.iter().map(|s| (*s).to_string()).collect());
        self
    }

    /// Seed explicit execution provenance from the given runtime.
    pub fn with_runtime(mut self, runtime: &Runtime) -> Self {
        self.runtime = Some(runtime.clone());
        self
    }

    fn to_table_options(&self) -> CsvReaderOptions {
        let mut options = CsvReaderOptions::new(&self.filepath).delimiter(self.delimiter).has_header(self.has_header);
        if let Some(runtime) = self.runtime.as_ref() {
            options = options.with_runtime(runtime);
        }
        options
    }

    fn resolve_column_names(&self, num_columns: usize) -> Result<Vec<String>> {
        let operation = "dataframe::read_csv_with_options";

        if let Some(names) = self.column_names.as_ref() {
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names.clone());
        }

        if self.has_header {
            let names = self.read_header_names()?;
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names);
        }

        Ok((0..num_columns).map(|idx| format!("column_{idx}")).collect())
    }

    fn read_header_names(&self) -> Result<Vec<String>> {
        let operation = "dataframe::read_csv_with_options";

        let mut reader =
            ::csv::ReaderBuilder::new().has_headers(true).delimiter(self.delimiter).from_path(&self.filepath).map_err(
                |source| {
                    Error::new(ErrorKind::IoError, "failed to open CSV file for header parsing")
                        .with_operation(operation)
                        .with_context("path", self.filepath.display().to_string())
                        .set_source(source)
                },
            )?;

        let headers = reader.headers().map_err(|source| {
            Error::new(ErrorKind::IoError, "failed to read CSV header")
                .with_operation(operation)
                .with_context("path", self.filepath.display().to_string())
                .set_source(source)
        })?;

        Ok(headers.iter().map(str::to_owned).collect())
    }
}

/// Options for [`DataFrame::write_csv_with_options`].
///
/// This is the schema-aware CSV writer API for `DataFrame`. Column names
/// always come from the DataFrame schema; this type only controls
/// file-format knobs such as delimiter and whether the header row is
/// included.
#[derive(Debug, Clone)]
pub struct DataFrameCsvWriterOptions {
    filepath: PathBuf,
    delimiter: u8,
    include_header: bool,
    runtime: Option<Runtime>,
}
impl DataFrameCsvWriterOptions {
    /// Create options to write a `DataFrame` to a CSV file.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { filepath: path.as_ref().to_path_buf(), delimiter: b',', include_header: true, runtime: None }
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether to include a header row (default: `true`).
    pub fn include_header(mut self, include: bool) -> Self {
        self.include_header = include;
        self
    }

    /// Route the write through the given runtime's stream.
    pub fn with_runtime(mut self, runtime: &Runtime) -> Self {
        self.runtime = Some(runtime.clone());
        self
    }

    fn to_table_options(&self, column_names: &[&str]) -> CsvWriterOptions {
        let mut options = CsvWriterOptions::new(&self.filepath)
            .column_names(column_names)
            .delimiter(self.delimiter)
            .include_header(self.include_header);
        if let Some(runtime) = self.runtime.as_ref() {
            options = options.with_runtime(runtime);
        }
        options
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

impl DataFrame {
    /// Read a CSV file into a `DataFrame`.
    ///
    /// If the caller does not provide explicit names through
    /// [`DataFrameCsvReaderOptions::column_names`], names come from the header
    /// row when `has_header == true`, or fall back to generated names such as
    /// `"column_0"` when reading headerless CSV.
    pub fn read_csv(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_csv_with_options(DataFrameCsvReaderOptions::new(path))
    }

    /// Read a CSV file into a `DataFrame` using caller-provided options.
    ///
    /// When `has_header == true`, this reader API parses the header row on the
    /// host to recover schema names. This metadata parsing does not
    /// materialize data columns through host memory.
    pub fn read_csv_with_options(options: DataFrameCsvReaderOptions) -> Result<Self> {
        let table = csv::read_csv(&options.to_table_options())?;
        let num_columns = {
            let view = table.view()?;
            view.num_columns()? as usize
        };
        let names = options.resolve_column_names(num_columns)?;
        Self::from_table(table, names)
    }

    /// Write this `DataFrame` to a CSV file, preserving the DataFrame schema
    /// as the output header row.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_csv_with_options(DataFrameCsvWriterOptions::new(path))
    }

    /// Write this `DataFrame` to a CSV file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output column
    /// names so the high-level round-trip path remains truthful.
    pub fn write_csv_with_options(&self, options: DataFrameCsvWriterOptions) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let opts = options.to_table_options(&names);
        let write_domain = if let Some(runtime) = opts.runtime() {
            Arc::clone(runtime.execution_domain())
        } else {
            Arc::clone(&self.domain)
        };
        if !ExecutionDomain::same_provenance(&self.domain, &write_domain) {
            self.domain.fence_stream("dataframe::write_csv_with_options")?;
        }
        let tv = self.build_table_view()?;
        let exec_options = write_domain.explicit_exec_options_raw();
        csv::write_csv_with_exec(tv.table(), &opts, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_csv_with_options")
    }

    /// Read a Parquet file into a `DataFrame`, using column names from the
    /// file's schema metadata.
    ///
    /// This is a convenience wrapper around
    /// [`parquet::read_parquet_with_metadata`].
    pub fn read_parquet(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_parquet_with_options(ParquetReaderOptions::new(path))
    }

    /// Read a Parquet file into a `DataFrame` using caller-provided options.
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    pub fn read_parquet_with_options(options: ParquetReaderOptions) -> Result<Self> {
        let result = parquet::read_parquet_with_metadata(options)?;
        Self::from_table(result.table, result.column_names)
    }

    /// Write this `DataFrame` to a Parquet file, embedding column names in
    /// the file's schema metadata.
    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let opts = ParquetWriterOptions::new(path).column_names(&names)?;
        self.write_parquet_with_options(opts)
    }

    /// Write this `DataFrame` to a Parquet file using caller-provided options.
    pub fn write_parquet_with_options(&self, options: ParquetWriterOptions) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let options = options.ensure_column_names(&names)?;
        let write_domain = if let Some(runtime) = options.runtime() {
            Arc::clone(runtime.execution_domain())
        } else {
            Arc::clone(&self.domain)
        };
        if !ExecutionDomain::same_provenance(&self.domain, &write_domain) {
            self.domain.fence_stream("dataframe::write_parquet_with_options")?;
        }
        let tv = self.build_table_view()?;
        let exec_options = write_domain.explicit_exec_options_raw();
        parquet::write_parquet_with_exec(tv.table(), &options, exec_options.as_ref())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_parquet_with_options")
    }
}
