//! Parquet and JSON reader and writer entry points for [`DataFrame`].

mod options;
mod source_budget;
mod source_columns;
mod source_read;

use std::{path::Path, sync::Arc};

use options::validate_parquet_projection;
pub use options::{
    JsonReadOptions, JsonWriteOptions, MultiParquetReadOptions, ParquetReadOptions, ParquetWriteOptions,
};
use source_budget::{
    PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES, parquet_source_batch_budget_context, source_preflight_column_bytes,
    source_preflight_named_columns, source_preflight_row_count,
};
pub use source_budget::{ParquetSourceChunkReadOutput, ParquetSourceReadPreflight};
use source_columns::empty_parquet_source_frame;
use source_read::{
    SourceChunkReadInput, dataframe_from_parquet_table_options, read_parquet_source_chunk_frame,
    read_parquet_source_chunk_frame_from_host_buffer, read_parquet_source_chunk_with_budget_input,
    read_parquet_source_chunks_frame,
};

use super::owned::DataFrame;
use crate::{
    core::{
        error::{Error, ErrorCode, Result},
        schema,
        table::Table,
    },
    io::{
        TableReadOutput, json, parquet,
        source::{LocalParquetSource, SourceChunk, SourceReadMode, plan_local_parquet_source_chunks},
    },
    runtime::{
        BoundaryKind, BudgetedExecutionOutcome, ExecutionBudgetToken, QueryStageKind, Runtime,
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
    /// bound for the scoped limiting allocator used by
    /// [`read_parquet_source_chunk_with_budget`](Self::read_parquet_source_chunk_with_budget),
    /// covering small lower-level allocations that happen before retained
    /// output bytes can be observed.
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
    pub fn read_parquet_with_options(options: ParquetReadOptions<'_>) -> Result<Self> {
        let operation = "dataframe::read_parquet_with_options";
        validate_parquet_projection(options.columns.as_deref(), operation)?;
        let table_options = options.to_table_options()?;
        dataframe_from_parquet_table_options(&table_options, operation)
    }

    /// Read a Parquet file under a hard execution-budget token.
    ///
    /// This is for executor-owned cache and source paths that already hold a
    /// reservation and need the generic Parquet reader to allocate from the
    /// token's scoped memory resource.
    pub fn read_parquet_with_budget(
        mut options: ParquetReadOptions<'_>,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let operation = "dataframe::read_parquet_with_budget";
        validate_parquet_projection(options.columns.as_deref(), operation)?;
        options.runtime = options
            .runtime
            .with_memory_resource_override(token.memory_resource(), operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let table_options = options.to_table_options().map_err(|error| token.annotate_error(error, operation))?;
        let frame = dataframe_from_parquet_table_options(&table_options, operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output_rows = frame.num_rows();
        let output_bytes = frame
            .device_allocated_size_bytes()
            .map_err(|error| error.reanchor(operation))
            .map_err(|error| token.annotate_error(error, operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output_rows);
        Ok(BudgetedExecutionOutcome::new(frame, report))
    }

    /// Read a Parquet payload borrowed from host memory into a `DataFrame`.
    ///
    /// The host buffer is borrowed only for the duration of this call. The
    /// returned `DataFrame` owns device-resident decoded columns and does not
    /// retain a pointer into `buffer`, so callers may drop or reuse the buffer
    /// after this method returns.
    ///
    /// File-format options such as projection, row-group selection, reader
    /// predicate, and runtime are taken from `options`; any path or filebytes
    /// source already present in `options` is ignored in favor of `buffer`.
    ///
    /// # Cost
    /// Decodes Parquet from caller-owned host memory, allocates GPU-backed
    /// columns for the decoded result, and copies decoded column data into
    /// device memory. This path does not use KvikIO or path-based object-store
    /// reads.
    pub fn read_parquet_from_host_buffer(buffer: &[u8], options: ParquetReadOptions<'_>) -> Result<Self> {
        let operation = "dataframe::read_parquet_from_host_buffer";
        validate_parquet_projection(options.columns.as_deref(), operation)?;
        let table_options = options.to_host_buffer_table_options(buffer);
        dataframe_from_parquet_table_options(&table_options, operation)
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
                        .map_err(|error| error.reanchor(operation))?
                        .into_frame(),
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
        Self::read_parquet_source_chunk_with_evidence(runtime, source, chunk, mode)
            .map(ParquetSourceChunkReadOutput::into_frame)
    }

    /// Execute one planned local Parquet source chunk and return lower-layer
    /// read evidence alongside the decoded frame.
    pub fn read_parquet_source_chunk_with_evidence(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
    ) -> Result<ParquetSourceChunkReadOutput> {
        let operation = "dataframe::read_parquet_source_chunk";
        let output = read_parquet_source_chunk_frame(runtime, source, chunk, mode, operation)?;
        output.frame().record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(output)
    }

    /// Execute one planned local Parquet source chunk from a caller-owned host
    /// buffer instead of a path-based reader.
    ///
    /// `host_buffer` must contain the full Parquet file represented by
    /// `chunk`. The buffer is borrowed only for the duration of this call and
    /// is not retained by the returned `DataFrame`.
    pub fn read_parquet_source_chunk_from_host_buffer(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        host_buffer: &[u8],
    ) -> Result<Self> {
        Self::read_parquet_source_chunk_from_host_buffer_with_evidence(runtime, source, chunk, mode, host_buffer)
            .map(ParquetSourceChunkReadOutput::into_frame)
    }

    /// Execute one planned local Parquet source chunk from host memory and
    /// return lower-layer read evidence alongside the decoded frame.
    pub fn read_parquet_source_chunk_from_host_buffer_with_evidence(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        host_buffer: &[u8],
    ) -> Result<ParquetSourceChunkReadOutput> {
        let operation = "dataframe::read_parquet_source_chunk_from_host_buffer";
        let output =
            read_parquet_source_chunk_frame_from_host_buffer(runtime, source, chunk, mode, host_buffer, operation)?;
        output.frame().record_boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::SourceRead,
            false,
            operation,
        )?;
        Ok(output)
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
        let outcome = Self::read_parquet_source_chunk_with_budget_with_evidence(runtime, source, chunk, mode, token)?;
        let (output, report) = outcome.into_parts();
        Ok(BudgetedExecutionOutcome::new(output.into_frame(), report))
    }

    /// Execute one planned local Parquet source chunk under a hard budget
    /// token and return lower-layer read evidence alongside the decoded frame.
    pub fn read_parquet_source_chunk_with_budget_with_evidence(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<ParquetSourceChunkReadOutput>> {
        let operation = "dataframe::read_parquet_source_chunk_with_budget";
        read_parquet_source_chunk_with_budget_input(
            runtime,
            source,
            chunk,
            mode,
            token,
            operation,
            SourceChunkReadInput::Path,
        )
    }

    /// Execute multiple planned local Parquet source chunks in one path-based
    /// libcudf reader call under a hard budget token.
    ///
    /// This is intentionally narrower than the single-chunk reader: it reads
    /// path sources only, requires either explicit non-empty row-group
    /// selections or known whole-file row-group coverage for every child chunk,
    /// and rejects partition columns until the API can represent per-child
    /// partition materialization without losing source identity.
    pub fn read_parquet_source_chunks_with_budget(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        mode: SourceReadMode,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let outcome = Self::read_parquet_source_chunks_with_budget_with_evidence(runtime, source, chunks, mode, token)?;
        let (output, report) = outcome.into_parts();
        Ok(BudgetedExecutionOutcome::new(output.into_frame(), report))
    }

    /// Execute multiple planned local Parquet source chunks under a hard
    /// budget token and return lower-layer read evidence alongside the decoded
    /// frame.
    pub fn read_parquet_source_chunks_with_budget_with_evidence(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunks: &[SourceChunk],
        mode: SourceReadMode,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<ParquetSourceChunkReadOutput>> {
        let operation = "dataframe::read_parquet_source_chunks_with_budget";
        let budget_runtime = runtime
            .with_memory_resource_override(token.memory_resource(), operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output = read_parquet_source_chunks_frame(&budget_runtime, source, chunks, mode, operation)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output_rows = output.frame().num_rows();
        let output_bytes = output
            .frame()
            .device_allocated_size_bytes()
            .map_err(|error| error.reanchor(operation))
            .map_err(|error| token.annotate_error(error, operation))?;
        output
            .frame()
            .record_boundary_observation_for_query_stage(
                BoundaryKind::HostToDevice,
                QueryStageKind::SourceRead,
                false,
                operation,
            )
            .map_err(|error| token.annotate_error(error, operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output_rows)
            .with_context_facts(parquet_source_batch_budget_context(source, chunks, mode));

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    /// Execute one planned local Parquet source chunk from caller-owned host
    /// memory under a hard budget token.
    ///
    /// This is the host-buffer counterpart to
    /// [`read_parquet_source_chunk_with_budget`](Self::read_parquet_source_chunk_with_budget).
    /// The host buffer is charged by the caller's host-budget layer, not by
    /// this device-memory budget token.
    pub fn read_parquet_source_chunk_with_budget_from_host_buffer(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        host_buffer: &[u8],
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let outcome = Self::read_parquet_source_chunk_with_budget_from_host_buffer_with_evidence(
            runtime,
            source,
            chunk,
            mode,
            host_buffer,
            token,
        )?;
        let (output, report) = outcome.into_parts();
        Ok(BudgetedExecutionOutcome::new(output.into_frame(), report))
    }

    /// Execute one planned local Parquet source chunk from host memory under a
    /// hard budget token and return lower-layer read evidence alongside the
    /// decoded frame.
    pub fn read_parquet_source_chunk_with_budget_from_host_buffer_with_evidence(
        runtime: &Runtime,
        source: &LocalParquetSource,
        chunk: &SourceChunk,
        mode: SourceReadMode,
        host_buffer: &[u8],
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<ParquetSourceChunkReadOutput>> {
        let operation = "dataframe::read_parquet_source_chunk_with_budget_from_host_buffer";
        read_parquet_source_chunk_with_budget_input(
            runtime,
            source,
            chunk,
            mode,
            token,
            operation,
            SourceChunkReadInput::HostBuffer(host_buffer),
        )
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
        let _consumer_fence = prepare_write_domain(&self.domain, &write_domain, "dataframe::write_json_with_options")?;
        let tv = self.build_table_view()?;
        let exec_options = write_domain.enter_execution()?;
        json::write_json_with_exec(&tv, &opts, exec_options.options())?;
        write_domain.synchronize_for_host_visibility("dataframe::write_json_with_options")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::frame::series::Series;

    /// A bounded pass-read limit must decode the exact same rows, in the
    /// same order, as the unbounded one-shot reader.
    ///
    /// `Runtime::with_memory_resource_override_and_parquet_pass_read_limit`
    /// is `pub(crate)`, so this contract is exercised as a unit test here
    /// rather than from `tests/`, which only sees the crate's public API.
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

        let unbounded = DataFrame::read_parquet(&runtime, &path).unwrap();
        assert_eq!(unbounded.num_rows(), row_count);

        // Well under the ~320 KB of decoded i64 data, so the read spans several
        // passes rather than one.
        let pass_read_limit = 64 * 1024;
        let bounded_runtime = runtime
            .with_memory_resource_override_and_parquet_pass_read_limit(
                runtime.memory_resource(),
                pass_read_limit,
                "test::parquet_pass_read_limit",
            )
            .unwrap();
        let bounded_options = ParquetReadOptions::builder().filepath(&path).with_runtime(&bounded_runtime).build();
        let bounded = DataFrame::read_parquet_with_options(bounded_options).unwrap();
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
}
