use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use typed_builder::TypedBuilder;

use super::{info::InfoDetail, owned::DataFrame};
use crate::{
    core::{
        error::{Error, ErrorCode, Result},
        scalar::Scalar,
        schema,
        types::{DataType, TimestampResolution, TypeId, cudf_null_mask_allocation_size_bytes},
    },
    frame::series::Series,
    internal::hash::{FastHashSet, fast_hash_set_with_capacity},
    io::{
        TableReadOutput,
        csv::{self, CsvTableReadOptions, CsvTableWriteOptions},
        json::{self, JsonTableReadOptions, JsonTableWriteOptions},
        parquet::{self, ParquetTableReadOptions, ParquetTableWriteOptions},
        source::{
            LocalParquetSource, LocalParquetSourceKind, PartitionValue, SourceChunk, SourceColumnRef, SourceReadMode,
            plan_local_parquet_source_chunks,
        },
    },
    ops::ast::AstExpression,
    runtime::{
        BoundaryKind, BudgetedExecutionOutcome, ExecutionBudgetToken, QueryStageKind, Runtime,
        execution_domain::ExecutionDomain,
    },
};

const PARQUET_SOURCE_DECODED_OUTPUT_PREFLIGHT_CONTRACT_GAP: &str =
    "libcudf_parquet_decoded_output_preflight_unavailable";
const PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES: usize = 64 * 1024;

/// Lower-level byte preflight for a bounded local Parquet source chunk read.
///
/// This is returned only for source shapes where cudf-rs can prove the retained
/// output bytes before execution from the checked source contract, selected
/// row-count metadata, and fixed-width output types. Variable-width source
/// outputs and chunks without row-count metadata currently return `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParquetSourceReadPreflight {
    retained_output_bytes: usize,
    required_budget_bytes: usize,
}

impl ParquetSourceReadPreflight {
    fn new(retained_output_bytes: usize) -> Self {
        Self {
            retained_output_bytes,
            required_budget_bytes: retained_output_bytes.max(PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES),
        }
    }

    /// Exact retained-output byte upper bound for the requested source frame.
    pub fn retained_output_bytes(&self) -> usize {
        self.retained_output_bytes
    }

    /// Minimum execution budget cudf-rs recommends for this source read.
    pub fn required_budget_bytes(&self) -> usize {
        self.required_budget_bytes
    }
}

fn comma_separated_strings(values: &[String]) -> String {
    values.join(",")
}

fn comma_separated_usize(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn source_column_names<'a>(columns: impl IntoIterator<Item = &'a crate::io::source::SourceOutputColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

fn required_column_names<'a>(columns: impl IntoIterator<Item = &'a crate::io::source::SourceRequiredColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

fn partition_column_names<'a>(columns: impl IntoIterator<Item = &'a crate::io::source::PartitionColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

fn local_parquet_source_kind_str(kind: LocalParquetSourceKind) -> &'static str {
    match kind {
        LocalParquetSourceKind::Data => "data",
        LocalParquetSourceKind::MetadataRowCount => "metadata_row_count",
    }
}

fn source_read_mode_str(mode: SourceReadMode) -> &'static str {
    match mode {
        SourceReadMode::FinalOutput => "final_output",
        SourceReadMode::ExecutionFrame => "execution_frame",
    }
}

fn source_preflight_row_count(source: &LocalParquetSource, chunk: &SourceChunk) -> Result<Option<usize>> {
    if source.kind() == LocalParquetSourceKind::MetadataRowCount {
        return Ok(Some(1));
    }
    chunk
        .known_row_count()
        .map(|rows| {
            usize::try_from(rows).map_err(|_| {
                Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    "dataframe::read_parquet_source_chunk_preflight",
                    "source chunk row count exceeds usize",
                )
                .with_detail("reason", "row_count_exceeds_usize")
                .with_detail("row_count", rows.to_string())
                .with_detail("chunk_index", chunk.chunk_index().to_string())
            })
        })
        .transpose()
}

fn source_preflight_column_bytes(column: &crate::io::source::SourceColumn, rows: usize) -> Result<Option<usize>> {
    let Some(element_size) = column.data_type().id().size_in_bytes()? else {
        return Ok(None);
    };
    let element_bytes = rows.checked_mul(element_size as usize).ok_or_else(|| {
        Error::new_with(
            ErrorCode::UnsupportedOperation,
            "dataframe::read_parquet_source_chunk_preflight",
            "source preflight fixed-width column byte count overflowed usize",
        )
        .with_detail("reason", "column_byte_count_overflow")
        .with_detail("column", column.name().to_string())
        .with_detail("row_count", rows.to_string())
        .with_detail("element_size_bytes", element_size.to_string())
    })?;
    let mask_bytes = if column.nullable() { cudf_null_mask_allocation_size_bytes(rows) } else { 0 };
    element_bytes.checked_add(mask_bytes).map(Some).ok_or_else(|| {
        Error::new_with(
            ErrorCode::UnsupportedOperation,
            "dataframe::read_parquet_source_chunk_preflight",
            "source preflight nullable fixed-width column byte count overflowed usize",
        )
        .with_detail("reason", "nullable_column_byte_count_overflow")
        .with_detail("column", column.name().to_string())
        .with_detail("row_count", rows.to_string())
        .with_detail("element_bytes", element_bytes.to_string())
        .with_detail("mask_bytes", mask_bytes.to_string())
    })
}

fn source_preflight_named_columns(
    source: &LocalParquetSource,
    mode: SourceReadMode,
) -> impl Iterator<Item = &str> + '_ {
    let mut names = Vec::new();
    let seen_capacity = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen = fast_hash_set_with_capacity(seen_capacity);
    for output in source.output_columns() {
        if seen.insert(output.name().to_string()) {
            names.push(output.name());
        }
    }
    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if seen.insert(required.name().to_string()) {
                names.push(required.name());
            }
        }
    }
    names.into_iter()
}

fn parquet_source_budget_context(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    mode: SourceReadMode,
) -> BTreeMap<String, String> {
    let mut facts = BTreeMap::new();
    facts.insert("source_kind".to_string(), local_parquet_source_kind_str(source.kind()).to_string());
    facts.insert("source_read_mode".to_string(), source_read_mode_str(mode).to_string());
    facts.insert("path".to_string(), chunk.path().display().to_string());
    facts.insert("stable_id".to_string(), chunk.stable_id().to_string());
    facts.insert("chunk_index".to_string(), chunk.chunk_index().to_string());
    facts.insert("file_index".to_string(), chunk.file_index().to_string());
    facts.insert("reads_all_row_groups".to_string(), chunk.reads_all_row_groups().to_string());
    facts.insert(
        "selected_row_groups".to_string(),
        if chunk.reads_all_row_groups() { "all".to_string() } else { comma_separated_usize(chunk.row_group_indexes()) },
    );
    if !chunk.reads_all_row_groups() {
        facts.insert("selected_row_group_count".to_string(), chunk.row_group_indexes().len().to_string());
    }
    facts.insert("read_columns".to_string(), comma_separated_strings(chunk.read_columns()));
    facts.insert("read_column_count".to_string(), chunk.read_columns().len().to_string());
    facts.insert("projected_columns".to_string(), source_column_names(source.output_columns()));
    facts.insert("projected_column_count".to_string(), source.output_columns().len().to_string());
    facts.insert("required_columns".to_string(), required_column_names(source.required_columns()));
    facts.insert("required_column_count".to_string(), source.required_columns().len().to_string());
    facts.insert("partition_columns".to_string(), partition_column_names(source.partition_columns()));
    facts.insert("partition_column_count".to_string(), source.partition_columns().len().to_string());
    if let Some(known_file_bytes) = chunk.known_file_bytes() {
        facts.insert("known_file_bytes".to_string(), known_file_bytes.to_string());
    }
    if let Some(known_row_group_bytes) = chunk.known_row_group_bytes() {
        facts.insert("known_row_group_bytes".to_string(), known_row_group_bytes.to_string());
    }
    if let Some(known_projected_compressed_bytes) = chunk.known_projected_compressed_bytes() {
        facts.insert("known_projected_compressed_bytes".to_string(), known_projected_compressed_bytes.to_string());
    }
    if let Some(known_projected_uncompressed_bytes) = chunk.known_projected_uncompressed_bytes() {
        facts.insert("known_projected_uncompressed_bytes".to_string(), known_projected_uncompressed_bytes.to_string());
    }
    if let Some(known_pure_filter_compressed_bytes) = chunk.known_pure_filter_compressed_bytes() {
        facts.insert("known_pure_filter_compressed_bytes".to_string(), known_pure_filter_compressed_bytes.to_string());
    }
    if let Some(known_pure_filter_uncompressed_bytes) = chunk.known_pure_filter_uncompressed_bytes() {
        facts.insert(
            "known_pure_filter_uncompressed_bytes".to_string(),
            known_pure_filter_uncompressed_bytes.to_string(),
        );
    }
    if let Some(estimated_chunk_bytes) = chunk.estimated_chunk_bytes() {
        facts.insert("estimated_chunk_bytes".to_string(), estimated_chunk_bytes.to_string());
    }
    if let Some(estimated_source_read_bytes) = chunk.estimated_source_read_bytes() {
        facts.insert("estimated_source_read_bytes".to_string(), estimated_source_read_bytes.to_string());
    }
    if let Some(known_row_count) = chunk.known_row_count() {
        facts.insert("known_row_count".to_string(), known_row_count.to_string());
    }
    facts
}

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
/// schema metadata into the resulting `DataFrame`. A supplied projection must
/// contain at least one column name; omit `columns` to read all columns.
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
    #[builder(default, setter(strip_option, into))]
    predicate: Option<Arc<AstExpression>>,
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
            self.predicate.clone(),
            self.runtime.clone(),
        )
    }
}

/// Options for [`DataFrame::read_parquet_many_with_options`].
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
    filepaths: Vec<PathBuf>,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    columns: Option<Vec<String>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
}

impl MultiParquetReadOptions {
    fn from_parts(filepaths: Vec<PathBuf>, columns: Option<Vec<String>>, runtime: Runtime) -> Self {
        Self { filepaths, columns, runtime }
    }

    fn to_single_file_options(&self, filepath: PathBuf) -> ParquetTableReadOptions {
        ParquetTableReadOptions::from_parts(filepath, self.columns.clone(), 0, None, None, None, self.runtime.clone())
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

fn resolve_csv_column_names(options: &CsvReadOptions, result: &TableReadOutput) -> Result<Vec<String>> {
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
        return Err(schema::column_name_count_mismatch_error(
            operation,
            format!("column name count ({name_count}) does not match table column count ({num_columns})"),
            name_count,
            num_columns,
        ));
    }

    Ok(())
}

fn require_reader_schema_names(
    result: TableReadOutput,
    operation: &'static str,
    reader_label: &'static str,
) -> Result<(crate::core::Table, Vec<String>)> {
    let num_columns = {
        let view = result.table.view()?;
        view.num_columns()?
    };
    let names = result.column_names.unwrap_or_default();

    if names.len() != num_columns {
        let (message, reason) = if names.is_empty() && num_columns > 0 {
            (format!("{reader_label} did not return column names for a non-empty table"), "missing_reader_schema_names")
        } else {
            (
                format!(
                    "{reader_label} returned {} column names for a table with {} columns",
                    names.len(),
                    num_columns
                ),
                "mismatch_with_table_column_count",
            )
        };
        return Err(schema::foreign_column_name_count_contract_error(
            operation,
            message,
            reason,
            names.len(),
            num_columns,
            "ensure the reader returns one schema name per table column for non-empty results",
        ));
    }

    Ok((result.table, names))
}

fn empty_parquet_projection_error(operation: &'static str) -> Error {
    Error::new_with(ErrorCode::EmptyInput, operation, "parquet projection requires at least one column name")
        .with_detail("resource", "columns")
        .with_detail("reason", "empty_projection")
        .with_detail("column_name_count", "0")
        .with_detail("remedy", "omit the projection to read all columns, or provide at least one column name")
}

fn validate_parquet_projection(columns: Option<&[String]>, operation: &'static str) -> Result<()> {
    if matches!(columns, Some(names) if names.is_empty()) {
        return Err(empty_parquet_projection_error(operation));
    }

    Ok(())
}

fn reanchor_multi_parquet_file_error(
    error: Error,
    operation: &'static str,
    file_index: usize,
    filepath: &Path,
) -> Error {
    error
        .reanchor(operation)
        .with_detail("file_index", file_index.to_string())
        .with_detail_if_absent("path", filepath.display().to_string())
}

fn reanchor_multi_parquet_schema_error(
    error: Error,
    operation: &'static str,
    reference_filepath: &Path,
    file_index: usize,
    filepath: &Path,
) -> Error {
    error
        .reanchor(operation)
        .with_detail("reference_file_index", "0")
        .with_detail("reference_path", reference_filepath.display().to_string())
        .with_detail("file_index", file_index.to_string())
        .with_detail("path", filepath.display().to_string())
}

fn prepare_write_domain(
    frame_domain: &Arc<ExecutionDomain>,
    write_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<()> {
    ExecutionDomain::require_same_device_for_mixed_inputs(frame_domain, write_domain, operation)?;

    if !ExecutionDomain::same_provenance(frame_domain, write_domain) {
        ExecutionDomain::insert_stream_dependency(frame_domain, write_domain, operation)?;
    }

    Ok(())
}

fn missing_source_row_count_error(operation: &'static str, chunk: &SourceChunk) -> Error {
    Error::new_with(
        ErrorCode::ForeignInvalidArgument,
        operation,
        "source chunk requires row-count metadata because it has no decoded read columns",
    )
    .with_detail("reason", "missing_chunk_row_count")
    .with_detail("chunk_index", chunk.chunk_index().to_string())
    .with_detail("file_index", chunk.file_index().to_string())
    .with_detail("stable_id", chunk.stable_id().to_string())
    .with_detail("path", chunk.path().display().to_string())
}

fn source_ref_series(
    frame: &DataFrame,
    source: &LocalParquetSource,
    source_ref: SourceColumnRef,
    materialized_name: &str,
) -> Result<Series> {
    match source_ref {
        SourceColumnRef::ReadColumn(index) => {
            let name = source.read_columns().get(index).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    "dataframe::read_parquet_source",
                    "read column reference index is out of bounds during source projection",
                )
                .with_detail("reason", "read_column_ref_out_of_bounds")
                .with_detail("read_column_count", source.read_columns().len().to_string())
            })?;
            frame.series(name)
        },
        SourceColumnRef::Partition(index) => {
            let column = source.partition_columns().get(index).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    "dataframe::read_parquet_source",
                    "partition column reference index is out of bounds during source projection",
                )
                .with_detail("reason", "partition_ref_out_of_bounds")
                .with_detail("partition_column_count", source.partition_columns().len().to_string())
            })?;
            frame.series(column.name())
        },
        SourceColumnRef::ZeroColumnCount => frame.series(crate::io::source::ZERO_COLUMN_COUNT_COLUMN),
        SourceColumnRef::MissingNull => frame.series(materialized_name),
    }
}

fn project_source_frame(frame: &DataFrame, source: &LocalParquetSource, mode: SourceReadMode) -> Result<DataFrame> {
    let mut pairs: Vec<(String, Series)> = Vec::new();
    let seen_capacity = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen: FastHashSet<String> = fast_hash_set_with_capacity(seen_capacity);

    for output in source.output_columns() {
        let series = source_ref_series(frame, source, output.source(), output.name())?;
        seen.insert(output.name().to_string());
        pairs.push((output.name().to_string(), series));
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if seen.contains(required.name()) {
                continue;
            }
            let series = source_ref_series(frame, source, required.source(), required.name())?;
            seen.insert(required.name().to_string());
            pairs.push((required.name().to_string(), series));
        }
    }

    if pairs.is_empty() {
        return frame.select(Vec::<&str>::new()).map_err(|error| error.reanchor("dataframe::read_parquet_source"));
    }

    DataFrame::from_series(pairs).map_err(|error| error.reanchor("dataframe::read_parquet_source"))
}

fn empty_parquet_source_frame(
    runtime: &Runtime,
    source: &LocalParquetSource,
    mode: SourceReadMode,
) -> Result<DataFrame> {
    let mut pairs: Vec<(String, Series)> = Vec::new();
    let seen_capacity = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen: FastHashSet<String> = fast_hash_set_with_capacity(seen_capacity);

    for output in source.output_columns() {
        let schema_column = source.schema().column(output.name()).ok_or_else(|| {
            Error::new_with(
                ErrorCode::MissingColumn,
                "dataframe::read_parquet_source",
                "source schema does not contain output column for typed empty source",
            )
            .with_column(output.name().to_string())
            .with_detail("reason", "schema_column_missing")
        })?;
        pairs.push((output.name().to_string(), runtime.empty_series(schema_column.data_type())?));
        seen.insert(output.name().to_string());
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if seen.contains(required.name()) {
                continue;
            }
            let schema_column = source.schema().column(required.name()).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::MissingColumn,
                    "dataframe::read_parquet_source",
                    "source schema does not contain required column for typed empty source",
                )
                .with_column(required.name().to_string())
                .with_detail("reason", "schema_column_missing")
            })?;
            pairs.push((required.name().to_string(), runtime.empty_series(schema_column.data_type())?));
            seen.insert(required.name().to_string());
        }
    }

    if pairs.is_empty() {
        return Ok(runtime.empty_dataframe());
    }

    DataFrame::from_series(pairs).map_err(|error| error.reanchor("dataframe::read_parquet_source"))
}

fn null_source_scalar(runtime: &Runtime, data_type: DataType) -> Result<Scalar> {
    match data_type.id() {
        TypeId::Bool8 => runtime.null_scalar::<bool>(),
        TypeId::Int8 => runtime.null_scalar::<i8>(),
        TypeId::Int16 => runtime.null_scalar::<i16>(),
        TypeId::Int32 => runtime.null_scalar::<i32>(),
        TypeId::Int64 => runtime.null_scalar::<i64>(),
        TypeId::UInt8 => runtime.null_scalar::<u8>(),
        TypeId::UInt16 => runtime.null_scalar::<u16>(),
        TypeId::UInt32 => runtime.null_scalar::<u32>(),
        TypeId::UInt64 => runtime.null_scalar::<u64>(),
        TypeId::Float32 => runtime.null_scalar::<f32>(),
        TypeId::Float64 => runtime.null_scalar::<f64>(),
        TypeId::Decimal128 => runtime.null_decimal128_raw_scalar(data_type.scale()),
        TypeId::TimestampDays
        | TypeId::TimestampSeconds
        | TypeId::TimestampMilliseconds
        | TypeId::TimestampMicroseconds
        | TypeId::TimestampNanoseconds => {
            let resolution = data_type.id().timestamp_resolution().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    "dataframe::read_parquet_source",
                    "source null timestamp dtype is not supported",
                )
                .with_detail("reason", "unsupported_source_null_dtype")
            })?;
            runtime.null_timestamp_raw_scalar(resolution)
        },
        TypeId::DurationDays
        | TypeId::DurationSeconds
        | TypeId::DurationMilliseconds
        | TypeId::DurationMicroseconds
        | TypeId::DurationNanoseconds => {
            let resolution = data_type.id().duration_resolution().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    "dataframe::read_parquet_source",
                    "source null duration dtype is not supported",
                )
                .with_detail("reason", "unsupported_source_null_dtype")
            })?;
            runtime.null_duration_raw_scalar(resolution)
        },
        TypeId::String => runtime.null_string_scalar(),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "dataframe::read_parquet_source",
            "source null dtype is not supported",
        )
        .with_detail("reason", "unsupported_source_null_dtype")),
    }
}

fn partition_scalar_series(runtime: &Runtime, value: &PartitionValue, len: usize) -> Result<Series> {
    let scalar = match value {
        PartitionValue::Null(data_type) => null_source_scalar(runtime, *data_type)?,
        PartitionValue::Boolean(value) => runtime.scalar(*value)?,
        PartitionValue::Int8(value) => runtime.scalar(*value)?,
        PartitionValue::Int16(value) => runtime.scalar(*value)?,
        PartitionValue::Int32(value) => runtime.scalar(*value)?,
        PartitionValue::Int64(value) => runtime.scalar(*value)?,
        PartitionValue::UInt8(value) => runtime.scalar(*value)?,
        PartitionValue::UInt16(value) => runtime.scalar(*value)?,
        PartitionValue::UInt32(value) => runtime.scalar(*value)?,
        PartitionValue::UInt64(value) => runtime.scalar(*value)?,
        PartitionValue::Float32(value) => runtime.scalar(*value)?,
        PartitionValue::Float64(value) => runtime.scalar(*value)?,
        PartitionValue::Decimal128 { raw, scale } => runtime.decimal128_raw_scalar(*raw, *scale)?,
        PartitionValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days)?,
        PartitionValue::Timestamp { raw, resolution } => runtime.timestamp_raw_scalar(*raw, *resolution)?,
        PartitionValue::Duration { raw, resolution } => runtime.duration_raw_scalar(*raw, *resolution)?,
        PartitionValue::String(value) => runtime.string_scalar(value)?,
    };
    runtime.series_from_scalar(&scalar, len)
}

fn materialize_missing_null_columns(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    row_count_override: Option<usize>,
    mode: SourceReadMode,
) -> Result<DataFrame> {
    let row_count = row_count_override.unwrap_or_else(|| frame.num_rows());
    let mut frame = frame;
    let planned_count = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen = fast_hash_set_with_capacity(planned_count);

    for output in source.output_columns() {
        if output.source() != SourceColumnRef::MissingNull || !seen.insert(output.name().to_string()) {
            continue;
        }
        frame = materialize_one_missing_null_column(runtime, source, frame, output.name(), row_count)?;
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if required.source() != SourceColumnRef::MissingNull || !seen.insert(required.name().to_string()) {
                continue;
            }
            frame = materialize_one_missing_null_column(runtime, source, frame, required.name(), row_count)?;
        }
    }

    Ok(frame)
}

fn materialize_one_missing_null_column(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    name: &str,
    row_count: usize,
) -> Result<DataFrame> {
    let schema_column = source.schema().column(name).ok_or_else(|| {
        Error::new_with(
            ErrorCode::MissingColumn,
            "dataframe::read_parquet_source",
            "source schema does not contain missing nullable column",
        )
        .with_column(name.to_string())
        .with_detail("reason", "schema_column_missing")
    })?;
    if !schema_column.nullable() {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            "dataframe::read_parquet_source",
            "missing source columns can only materialize nullable target columns",
        )
        .with_column(name.to_string())
        .with_detail("reason", "missing_null_for_non_nullable_column"));
    }
    let scalar = null_source_scalar(runtime, schema_column.data_type()).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "missing_null_materialization_failed")
            .with_column(name.to_string())
    })?;
    let series = runtime.series_from_scalar(&scalar, row_count).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "missing_null_materialization_failed")
            .with_column(name.to_string())
    })?;
    frame.with_column(name, series).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "missing_null_column_insert_failed")
            .with_column(name.to_string())
    })
}

fn materialize_partition_columns(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    frame: DataFrame,
    row_count_override: Option<usize>,
) -> Result<DataFrame> {
    let mut frame = frame;
    let row_count = row_count_override.unwrap_or_else(|| frame.num_rows());
    for (partition_index, (column, value)) in
        source.partition_columns().iter().zip(chunk.partition_values()).enumerate()
    {
        let series = partition_scalar_series(runtime, value, row_count).map_err(|error| {
            error
                .reanchor("dataframe::read_parquet_source")
                .with_detail("reason", "partition_materialization_failed")
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name().to_string())
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("stable_id", chunk.stable_id().to_string())
                .with_detail("path", chunk.path().display().to_string())
        })?;
        frame = frame.with_column(column.name(), series).map_err(|error| {
            error
                .reanchor("dataframe::read_parquet_source")
                .with_detail("reason", "partition_column_insert_failed")
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name().to_string())
        })?;
    }
    Ok(frame)
}

fn read_parquet_source_chunk_frame(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    mode: SourceReadMode,
    operation: &'static str,
) -> Result<DataFrame> {
    source.validate_chunk_partition_values(chunk, operation)?;

    if source.kind() == LocalParquetSourceKind::MetadataRowCount {
        let count = chunk.known_row_count_i64()?.ok_or_else(|| missing_source_row_count_error(operation, chunk))?;
        let count_series = runtime.series(&[count])?;
        let frame = DataFrame::from_series([(crate::io::source::ZERO_COLUMN_COUNT_COLUMN, count_series)])
            .map_err(|error| error.reanchor(operation))?;
        return project_source_frame(&frame, source, mode).map_err(|error| error.reanchor(operation));
    }

    let (mut frame, row_count_override) = if chunk.read_columns().is_empty() {
        let row_count = chunk.known_row_count().ok_or_else(|| missing_source_row_count_error(operation, chunk))?;
        let row_count = usize::try_from(row_count).map_err(|_| {
            Error::new_with(ErrorCode::UnsupportedOperation, operation, "source chunk row count exceeds usize")
                .with_detail("reason", "row_count_exceeds_usize")
                .with_detail("row_count", row_count.to_string())
                .with_detail("chunk_index", chunk.chunk_index().to_string())
        })?;
        (runtime.empty_dataframe(), Some(row_count))
    } else {
        let row_groups = if chunk.reads_all_row_groups() { None } else { Some(chunk.row_group_indexes_i32()?) };
        let table_options = ParquetTableReadOptions::from_parts(
            chunk.path().to_path_buf(),
            Some(chunk.read_columns().to_vec()),
            0,
            None,
            row_groups,
            source.reader_predicate().cloned(),
            runtime.clone(),
        );
        let result = parquet::read_parquet_table(&table_options).map_err(|error| {
            error
                .reanchor(operation)
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("stable_id", chunk.stable_id().to_string())
                .with_detail_if_absent("path", chunk.path().display().to_string())
        })?;
        let (table, names) = require_reader_schema_names(result, operation, "parquet reader").map_err(|error| {
            error
                .reanchor(operation)
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("stable_id", chunk.stable_id().to_string())
                .with_detail("path", chunk.path().display().to_string())
        })?;
        (DataFrame::from_table(table, names).map_err(|error| error.reanchor(operation))?, None)
    };

    frame = materialize_partition_columns(runtime, source, chunk, frame, row_count_override)?;
    frame = materialize_missing_null_columns(runtime, source, frame, row_count_override, mode)?;
    project_source_frame(&frame, source, mode).map_err(|error| error.reanchor(operation))
}

fn dataframe_device_allocated_size_bytes(frame: &DataFrame, operation: &'static str) -> Result<usize> {
    let info = frame.info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| error.reanchor(operation))?;
    let memory = info.memory_accounting.ok_or_else(|| {
        crate::core::error::invariant_violation(operation, "device allocation accounting was not populated")
    })?;
    memory
        .device_allocated_bytes
        .ok_or_else(|| crate::core::error::invariant_violation(operation, "device allocated bytes were not populated"))
}

impl DataFrame {
    /// Minimum hard execution budget worth attempting for a local Parquet
    /// source chunk read.
    ///
    /// This is not an exact decoded-output preflight contract. It is a lower
    /// bound for the scoped limiting allocator used by
    /// [`read_parquet_source_chunk_with_budget`](Self::read_parquet_source_chunk_with_budget),
    /// covering small lower-level allocations that happen before retained
    /// output bytes can be observed.
    pub fn minimum_parquet_source_read_budget_bytes() -> usize {
        PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES
    }

    /// Return a source-read byte preflight when cudf-rs can prove the retained
    /// output bytes before execution.
    ///
    /// The current coverage is intentionally narrow: every output column in
    /// the requested source mode must be fixed width and the planned chunk must
    /// carry exact row-count metadata. Variable-width columns or missing
    /// row-count metadata return `Ok(None)` so callers can keep their existing
    /// fallback contract.
    pub fn parquet_source_chunk_exact_preflight(
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
    ) -> Result<Option<ParquetSourceReadPreflight>> {
        let operation = "dataframe::read_parquet_source_chunk_preflight";
        source.validate_chunk_partition_values(chunk, operation)?;
        let Some(rows) = source_preflight_row_count(source, chunk)? else {
            return Ok(None);
        };

        let mut retained_output_bytes = 0usize;
        for name in source_preflight_named_columns(source, mode) {
            let column = source.schema().column(name).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::MissingColumn,
                    operation,
                    "source preflight output column is missing from source schema",
                )
                .with_column(name.to_string())
                .with_detail("reason", "schema_column_missing")
            })?;
            let Some(column_bytes) = source_preflight_column_bytes(column, rows)? else {
                return Ok(None);
            };
            retained_output_bytes = retained_output_bytes.checked_add(column_bytes).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    operation,
                    "source preflight output byte count overflowed usize",
                )
                .with_detail("reason", "source_preflight_byte_count_overflow")
                .with_detail("column", name.to_string())
            })?;
        }

        Ok(Some(ParquetSourceReadPreflight::new(retained_output_bytes)))
    }

    /// Read a CSV file into a `DataFrame`.
    ///
    /// If the caller does not provide explicit names through
    /// `CsvReadOptions::column_names`, names come from the header
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
        let operation = "dataframe::read_csv_with_options";
        let table_options = options.to_table_options();
        let result = csv::read_csv(&table_options).map_err(|error| error.reanchor(operation))?;
        let names = resolve_csv_column_names(&options, &result)?;
        let frame = Self::from_table(result.table, names).map_err(|error| error.reanchor(operation))?;
        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
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
        let exec_options = write_domain.exec_options_raw();
        csv::write_csv_with_exec(&tv, &opts, &exec_options)?;
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
        let operation = "dataframe::read_parquet_with_options";
        validate_parquet_projection(options.columns.as_deref(), operation)?;
        let table_options = options.to_table_options();
        let result = parquet::read_parquet_table(&table_options).map_err(|error| error.reanchor(operation))?;
        let (table, names) = require_reader_schema_names(result, operation, "parquet reader")?;
        let frame = Self::from_table(table, names).map_err(|error| error.reanchor(operation))?;
        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
    }

    /// Read multiple Parquet files into one `DataFrame`.
    ///
    /// Files are read in the order supplied by `paths`, using the provided
    /// runtime for each single-file read. The decoded per-file tables are then
    /// concatenated on device. Glob expansion is intentionally not performed
    /// by this API: each path is treated as a literal filesystem path. If
    /// globbing is desired, expand patterns in the shell or caller before
    /// calling this method. Directories are not enumerated.
    ///
    /// # Cost
    /// Performs host file I/O for each file, allocates GPU-backed columns for
    /// each decoded table, then performs one device-side table concatenation.
    /// This is not predicate pushdown, GDS, or zero-copy multi-file interop.
    pub fn read_parquet_many<P>(runtime: &Runtime, paths: impl IntoIterator<Item = P>) -> Result<Self>
    where P: AsRef<Path> {
        let filepaths = paths.into_iter().map(|path| path.as_ref().to_path_buf()).collect();
        Self::read_parquet_many_with_options(MultiParquetReadOptions::from_parts(filepaths, None, runtime.clone()))
    }

    /// Read multiple Parquet files into one `DataFrame` using caller-provided
    /// [`MultiParquetReadOptions`].
    ///
    /// The same projection is applied to every file. All files must return
    /// the same schema names and compatible column dtypes after projection.
    /// Row order follows the order of `filepaths`.
    ///
    /// # Cost
    /// Performs host file I/O for each file, allocates GPU-backed columns for
    /// each decoded table, then performs one device-side table concatenation.
    /// This is not predicate pushdown, GDS, or zero-copy multi-file interop.
    pub fn read_parquet_many_with_options(options: MultiParquetReadOptions) -> Result<Self> {
        let operation = "dataframe::read_parquet_many_with_options";
        if options.filepaths.is_empty() {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                operation,
                "read_parquet_many requires at least one file",
            )
            .with_detail("resource", "filepaths")
            .with_detail("reason", "empty_input")
            .with_detail("remedy", "provide one or more concrete Parquet file paths"));
        }
        validate_parquet_projection(options.columns.as_deref(), operation)?;

        let mut frames: Vec<DataFrame> = Vec::with_capacity(options.filepaths.len());
        for (file_index, filepath) in options.filepaths.iter().enumerate() {
            let table_options = options.to_single_file_options(filepath.clone());
            let result = parquet::read_parquet_table(&table_options)
                .map_err(|error| reanchor_multi_parquet_file_error(error, operation, file_index, filepath))?;
            let (table, names) = require_reader_schema_names(result, operation, "parquet reader")
                .map_err(|error| reanchor_multi_parquet_file_error(error, operation, file_index, filepath))?;
            let frame = Self::from_table(table, names)
                .map_err(|error| reanchor_multi_parquet_file_error(error, operation, file_index, filepath))?;
            if let Some(reference) = frames.first() {
                reference.validate_concat_schema(&frame, "dataframe::concat_many").map_err(|error| {
                    reanchor_multi_parquet_schema_error(error, operation, &options.filepaths[0], file_index, filepath)
                })?;
            }
            frames.push(frame);
        }

        let Some((first, rest)) = frames.split_first() else {
            unreachable!("empty file list was rejected before reading");
        };
        let rest_refs: Vec<&DataFrame> = rest.iter().collect();
        let frame = first.concat_many(&rest_refs).map_err(|error| error.reanchor(operation))?;
        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
    }

    /// Read a checked local Parquet source and return its final output.
    ///
    /// This is equivalent to
    /// [`read_parquet_source_with_mode`](Self::read_parquet_source_with_mode)
    /// with [`SourceReadMode::FinalOutput`].
    ///
    /// # Cost
    /// Performs host filesystem reads through the existing single-file
    /// Parquet reader, materializes partition columns explicitly on device,
    /// and concatenates per-file/chunk frames on device when more than one
    /// chunk is planned. Partition materialization and concatenation are not
    /// zero-copy.
    pub fn read_parquet_source(runtime: &Runtime, source: &LocalParquetSource) -> Result<Self> {
        Self::read_parquet_source_with_mode(runtime, source, SourceReadMode::FinalOutput)
    }

    /// Read a checked local Parquet source in the requested output mode.
    ///
    /// [`SourceReadMode::FinalOutput`] applies final source projection and
    /// aliases. [`SourceReadMode::ExecutionFrame`] keeps pure-filter columns
    /// required for source-local query execution.
    pub fn read_parquet_source_with_mode(
        runtime: &Runtime,
        source: &LocalParquetSource,
        mode: SourceReadMode,
    ) -> Result<Self> {
        let operation = "dataframe::read_parquet_source";
        let plan =
            plan_local_parquet_source_chunks(source, Default::default()).map_err(|error| error.reanchor(operation))?;
        let frame = if plan.chunks().is_empty() {
            empty_parquet_source_frame(runtime, source, mode)?
        } else {
            let mut frames = Vec::with_capacity(plan.chunks().len());
            for chunk in plan.chunks() {
                frames.push(
                    read_parquet_source_chunk_frame(runtime, source, chunk, mode, operation)
                        .map_err(|error| error.reanchor(operation))?,
                );
            }

            let Some((first, rest)) = frames.split_first() else {
                unreachable!("empty chunks handled before split_first");
            };
            let rest_refs: Vec<&DataFrame> = rest.iter().collect();
            first.concat_many(&rest_refs).map_err(|error| error.reanchor(operation))?
        };

        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
    }

    /// Execute one planned local Parquet source chunk.
    ///
    /// This is the bounded-read hook for downstream query engines. Query
    /// engines should usually request [`SourceReadMode::ExecutionFrame`].
    pub fn read_parquet_source_chunk(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
    ) -> Result<Self> {
        let operation = "dataframe::read_parquet_source_chunk";
        let frame = read_parquet_source_chunk_frame(runtime, source, chunk, mode, operation)?;
        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
    }

    /// Execute one planned local Parquet source chunk under a hard budget token.
    ///
    /// The Parquet decode, partition-column materialization, and final source
    /// projection run in a scoped execution domain whose memory resource is
    /// the token's stream-bound reservation. The report records observed
    /// retained/output device allocation bytes and output rows after the read.
    ///
    /// For source chunks covered by
    /// [`DataFrame::parquet_source_chunk_exact_preflight`], the report records
    /// the exact source-read preflight facts. Other chunks keep the unresolved
    /// decoded-output preflight marker. Source compressed bytes are not
    /// reported as decoded output bytes.
    pub fn read_parquet_source_chunk_with_budget(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let operation = "dataframe::read_parquet_source_chunk_with_budget";
        let budget_runtime = runtime
            .with_memory_resource_override(token.memory_resource(), operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let frame = read_parquet_source_chunk_frame(&budget_runtime, source, chunk, mode, operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output_rows = frame.num_rows();
        let output_bytes = dataframe_device_allocated_size_bytes(&frame, operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        frame
            .record_boundary_observation_for_query_stage(
                BoundaryKind::HostToDevice,
                QueryStageKind::SourceRead,
                false,
                operation,
            )
            .map_err(|error| token.annotate_error(error, operation))?;
        let preflight = Self::parquet_source_chunk_exact_preflight(source, chunk, mode)?;
        let mut report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output_rows)
            .with_context_facts(parquet_source_budget_context(source, chunk, mode));
        if let Some(preflight) = preflight {
            report = report.with_context_facts([
                ("source_read_preflight_contract", "exact_retained_output_upper_bound".to_string()),
                ("source_read_preflight_retained_output_bytes", preflight.retained_output_bytes().to_string()),
                ("source_read_preflight_required_budget_bytes", preflight.required_budget_bytes().to_string()),
            ]);
        } else {
            report = report.with_unresolved_contract(PARQUET_SOURCE_DECODED_OUTPUT_PREFLIGHT_CONTRACT_GAP);
        }

        Ok(BudgetedExecutionOutcome::new(frame, report))
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
        let exec_options = write_domain.exec_options_raw();
        parquet::write_parquet_with_exec(&tv, &table_options, &exec_options)?;
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
        let operation = "dataframe::read_json_with_options";
        let table_options = options.to_table_options();
        let result = json::read_json_table(&table_options).map_err(|error| error.reanchor(operation))?;
        let (table, names) = require_reader_schema_names(result, operation, "json reader")?;
        let frame = Self::from_table(table, names).map_err(|error| error.reanchor(operation))?;
        frame.record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(frame)
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
        let exec_options = write_domain.exec_options_raw();
        json::write_json_with_exec(&tv, &opts, &exec_options)?;
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
