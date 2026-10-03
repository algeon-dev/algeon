//! Parquet reader and writer entry points for [`DataFrame`].

mod options;
mod source_budget;
mod source_columns;
mod source_read;

use std::{path::Path, sync::Arc};

use options::validate_parquet_projection;
pub use options::{MultiParquetReadOptions, ParquetReadOptions, ParquetWriteOptions};
use source_budget::{
    PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES, parquet_source_batch_budget_context, single_chunk_report_facts,
    source_preflight_column_bytes, source_preflight_named_columns, source_preflight_row_count,
};
pub use source_budget::{ParquetSourceChunkReadOutput, ParquetSourceReadPreflight};
use source_columns::empty_parquet_source_frame;
pub use source_read::{SourceRead, SourceReadInput};
use source_read::{
    dataframe_from_parquet_table_options, read_parquet_source_chunk_frame, read_parquet_source_chunks_frame,
};

use super::owned::DataFrame;
use crate::{
    Exec,
    core::{
        error::{Error, ErrorCode, Result},
        schema,
        table::Table,
    },
    io::{
        TableReadOutput, parquet,
        source::{LocalParquetSource, SourceChunk, SourceReadMode},
    },
    runtime::{
        Runtime,
        execution_domain::{ConsumerCompletionFence, ExecutionDomain},
    },
};

fn require_reader_schema_names(
    result: TableReadOutput,
    operation: &'static str,
    reader_label: &'static str,
) -> Result<(Table, Vec<String>)> {
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

/// Order the frame's retained stream into the write stream and return the
/// consumer-completion fence. Callers must hold the fence across the write
/// FFI call so the frame's stream waits for the in-flight write on drop.
fn prepare_write_domain(
    frame_domain: &Arc<ExecutionDomain>,
    write_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<ConsumerCompletionFence> {
    ExecutionDomain::require_same_device_for_mixed_inputs(frame_domain, write_domain, operation)?;
    ExecutionDomain::insert_stream_dependency(frame_domain, write_domain, operation)
}

impl DataFrame {
    /// Minimum hard execution budget worth attempting for a local Parquet
    /// source chunk read.
    ///
    /// This is not an exact decoded-output preflight contract. It is a lower
    /// bound for the member-enforced execution budget of a single-chunk
    /// [`read_parquet_source`](Self::read_parquet_source), covering small
    /// lower-level allocations that happen before retained output bytes can
    /// be observed.
    pub fn minimum_parquet_source_read_budget_bytes() -> usize {
        PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES
    }

    /// Return a source-read byte preflight when cudf can prove the retained
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

    /// Read one Parquet file or host buffer selected by `options`, running
    /// where `exec` selects on the options' runtime.
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    ///
    /// # Cost
    /// Performs host file I/O (or decodes the borrowed host buffer), allocates
    /// new GPU-backed columns for the decoded result, and copies decoded
    /// column data into device memory. Column names are recovered from Parquet
    /// schema metadata rather than synthesized heuristically.
    pub fn read_parquet(options: ParquetReadOptions<'_>, exec: Exec<'_>) -> Result<Self> {
        let operation = "dataframe::read_parquet";
        validate_parquet_projection(options.columns.as_deref(), operation)?;
        let table_options = options.to_table_options(exec).map_err(|error| error.reanchor_if_needed(operation))?;
        dataframe_from_parquet_table_options(&table_options, operation)
    }

    /// Read multiple Parquet files into one `DataFrame`, running where `exec`
    /// selects on the options' runtime.
    ///
    /// Files are read in the order of `filepaths` with the same projection,
    /// then concatenated on device. All files must return the same schema
    /// names and compatible column dtypes after projection. Glob expansion is
    /// intentionally not performed: each path is treated as a literal
    /// filesystem path, and directories are not enumerated.
    ///
    /// # Cost
    /// Performs host file I/O for each file, allocates GPU-backed columns for
    /// each decoded table, then performs one device-side table concatenation.
    /// This is not predicate pushdown, GDS, or zero-copy multi-file interop.
    pub fn read_parquet_many(options: MultiParquetReadOptions, exec: Exec<'_>) -> Result<Self> {
        let operation = "dataframe::read_parquet_many";
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
        let runtime = options.runtime().on(exec).map_err(|error| error.reanchor(operation))?;

        let mut frames: Vec<DataFrame> = Vec::with_capacity(options.filepaths.len());
        for (file_index, filepath) in options.filepaths.iter().enumerate() {
            let table_options = options.to_single_file_options(filepath.clone(), &runtime);
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
        first.on(Exec::RETAINED).concat_many(&rest_refs).map_err(|error| error.reanchor(operation))
    }

    /// Read planned chunks of a checked local Parquet source, running where
    /// `exec` selects on `runtime`'s device.
    ///
    /// One chunk reads through the single-file reader from its path or from
    /// the request's host buffer. Several chunks read in one path-based
    /// multi-file reader call, which requires explicit non-empty row-group
    /// selections or known whole-file row-group coverage for every chunk and
    /// rejects partition columns. No chunks returns the source's empty output.
    ///
    /// The output carries the decoded frame, the lower-layer read evidence when
    /// the reader reports it, and the source-read facts that
    /// [`ExecutionBudgetToken::run`](crate::ExecutionBudgetToken::run) records. For a single chunk covered by
    /// [`DataFrame::parquet_source_chunk_exact_preflight`], those facts include
    /// the exact source-read preflight; other chunks record the unresolved
    /// preflight marker. Source compressed bytes are not reported as decoded
    /// output bytes.
    ///
    /// # Cost
    /// Performs host filesystem reads through the Parquet reader, materializes
    /// partition columns explicitly on device, and applies the source
    /// projection. Partition materialization is not zero-copy.
    pub fn read_parquet_source(
        request: SourceRead<'_>,
        runtime: &Runtime,
        exec: Exec<'_>,
    ) -> Result<ParquetSourceChunkReadOutput> {
        let operation = "dataframe::read_parquet_source";
        let runtime = runtime.on(exec).map_err(|error| error.reanchor(operation))?;
        let SourceRead { source, chunks, mode, input, pass_read_limit } = request;
        match chunks {
            [] => Ok(ParquetSourceChunkReadOutput::new(empty_parquet_source_frame(&runtime, source, mode)?, None)),
            [chunk] => {
                let output =
                    read_parquet_source_chunk_frame(&runtime, source, chunk, mode, input, pass_read_limit, operation)?;
                Ok(output.with_report_facts(single_chunk_report_facts(source, chunk, mode)))
            },
            chunks => {
                if matches!(input, SourceReadInput::HostBuffer(_)) {
                    return Err(Error::new_with(
                        ErrorCode::UnsupportedOperation,
                        operation,
                        "a host-buffer source read covers exactly one chunk",
                    )
                    .with_detail("reason", "host_buffer_multi_chunk")
                    .with_detail("chunk_count", chunks.len().to_string())
                    .with_detail("remedy", "read each chunk with its own host buffer"));
                }
                let output =
                    read_parquet_source_chunks_frame(&runtime, source, chunks, mode, pass_read_limit, operation)?;
                Ok(output.with_report_facts(parquet_source_batch_budget_context(source, chunks, mode)))
            },
        }
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
        let _consumer_fence =
            prepare_write_domain(&self.domain, &write_domain, "dataframe::write_parquet_with_options")?;
        let tv = self.build_table_view()?;
        let exec_options = write_domain.enter_execution()?;
        parquet::write_parquet_with_exec(&tv, &table_options, exec_options.options())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_parquet_with_options")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::{source_columns::materialize_all_null_read_columns, *};
    use crate::{
        core::types::{DataType, TypeId},
        frame::series::Series,
        io::source::{LocalParquetFile, SourceColumn, SourceColumnRef, SourceOutputColumn, SourceSchema},
    };

    /// A bounded pass-read limit must decode the exact same rows, in the
    /// same order, as the unbounded one-shot reader.
    #[test]
    fn parquet_pass_read_limit_preserves_row_order_across_multiple_passes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pass_read_limit.parquet");
        let runtime = Runtime::new().unwrap();

        let row_count = 20_000usize;
        let ids: Vec<i64> = (0..row_count as i64).collect();
        let values: Vec<i64> = ids.iter().map(|id| id * 7 + 3).collect();
        let id_series = Series::from_column(runtime.column_from_slice(&ids).unwrap()).unwrap();
        let value_series = Series::from_column(runtime.column_from_slice(&values).unwrap()).unwrap();
        let frame = DataFrame::from_series([("id", id_series), ("value", value_series)]).unwrap();

        let write_options =
            ParquetWriteOptions::builder().filepath(&path).row_group_size_rows(2_000).with_runtime(&runtime).build();
        frame.write_parquet_with_options(write_options).unwrap();

        let unbounded = DataFrame::read_parquet(
            ParquetReadOptions::builder().filepath(&path).with_runtime(&runtime).build(),
            Exec::RETAINED,
        )
        .unwrap();
        assert_eq!(unbounded.num_rows(), row_count);

        // Well under the ~320 KB of decoded i64 data, so the read spans several
        // passes rather than one.
        let bounded_options =
            ParquetReadOptions::builder().filepath(&path).with_runtime(&runtime).pass_read_limit(64 * 1024).build();
        let bounded = DataFrame::read_parquet(bounded_options, Exec::RETAINED).unwrap();
        assert_eq!(bounded.num_rows(), row_count);

        assert_eq!(
            unbounded.series("id").unwrap().to_vec_strict::<i64>().unwrap(),
            bounded.series("id").unwrap().to_vec_strict::<i64>().unwrap(),
        );
        assert_eq!(
            unbounded.series("value").unwrap().to_vec_strict::<i64>().unwrap(),
            bounded.series("value").unwrap().to_vec_strict::<i64>().unwrap(),
        );
        assert_eq!(bounded.series("id").unwrap().to_vec_strict::<i64>().unwrap(), ids);
        assert_eq!(bounded.series("value").unwrap().to_vec_strict::<i64>().unwrap(), values);
    }

    #[test]
    fn all_null_read_column_length_mismatch_returns_structured_error() {
        let runtime = Runtime::new().unwrap();
        let ids = vec![1i64, 2, 3];
        let id_series = Series::from_column(runtime.column_from_slice(&ids).unwrap()).unwrap();
        let frame = DataFrame::from_series([("id", id_series)]).unwrap();

        let schema = SourceSchema::new([
            SourceColumn::new("id", DataType::new(TypeId::Int64), false).unwrap(),
            SourceColumn::new("extra", DataType::new(TypeId::Int64), true).unwrap(),
        ])
        .unwrap();
        let source = LocalParquetSource::data(
            [LocalParquetFile::new("part-0.parquet", "part-0", []).unwrap()],
            schema,
            ["id"],
            [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
            [],
            [],
        )
        .unwrap();

        let error = materialize_all_null_read_columns(&runtime, &source, frame, &["extra".to_string()], 5).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ColumnLengthMismatch);
    }
}
