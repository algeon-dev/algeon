//! Physical Parquet reads for planned source chunks.

use super::{
    super::owned::DataFrame,
    require_reader_schema_names,
    source_budget::{
        ParquetSourceChunkReadOutput, local_parquet_source_kind_str, source_chunk_continues_previous_reader_input,
    },
    source_columns::{
        materialize_all_null_read_columns, materialize_missing_null_columns, materialize_partition_columns,
        missing_source_row_count_error, project_source_frame, source_chunk_diagnostic_path,
    },
};
use crate::{
    Exec,
    core::error::{Error, ErrorCode, Result},
    hash::FastHashSet,
    io::{
        parquet::{
            self, ParquetMultiTableInput, ParquetMultiTableReadOptions, ParquetReadEvidence, ParquetTableReadOptions,
        },
        source::{LocalParquetSource, LocalParquetSourceKind, SourceChunk, SourceReadMode, ZERO_COLUMN_COUNT_COLUMN},
    },
    runtime::Runtime,
};

/// Where a source read takes its Parquet bytes from.
#[derive(Debug, Clone, Copy)]
pub enum SourceReadInput<'a> {
    /// Read each chunk's file through its path or remote source.
    Path,
    /// Read the one requested chunk from a caller-owned buffer holding the
    /// full Parquet file. The buffer is borrowed only for the call.
    HostBuffer(&'a [u8]),
}

/// One read of planned chunks of a checked local Parquet source, for
/// [`DataFrame::read_parquet_source`].
#[derive(Debug, Clone, Copy)]
pub struct SourceRead<'a> {
    /// The checked source the chunks belong to.
    pub source: &'a LocalParquetSource,
    /// Planned chunks to read; see [`DataFrame::read_parquet_source`] for how
    /// the chunk count selects the reader.
    pub chunks: &'a [SourceChunk],
    /// Final source output or the query-execution frame.
    pub mode: SourceReadMode,
    /// Path-based or host-buffer input.
    pub input: SourceReadInput<'a>,
    /// Bound on Parquet read and decompression scratch in bytes. The rows
    /// returned are unchanged; a row group larger than the bound still loads.
    pub pass_read_limit: Option<u64>,
}

impl<'a> SourceRead<'a> {
    /// Read `chunks` of `source` from their paths in `mode`, with unbounded
    /// read scratch.
    pub fn new(source: &'a LocalParquetSource, chunks: &'a [SourceChunk], mode: SourceReadMode) -> Self {
        Self { source, chunks, mode, input: SourceReadInput::Path, pass_read_limit: None }
    }

    /// Read the one requested chunk from `buffer` instead of its path.
    pub fn from_host_buffer(mut self, buffer: &'a [u8]) -> Self {
        self.input = SourceReadInput::HostBuffer(buffer);
        self
    }

    /// Bound read and decompression scratch to a share of `budget_bytes`, the
    /// allocation limit the read runs under. The decoded output must fit in
    /// the same budget, so scratch takes half of it; the divisor is a starting
    /// point, not a measured optimum.
    pub fn with_pass_read_limit_for_budget(mut self, budget_bytes: usize) -> Self {
        const SCRATCH_SHARE_DIVISOR: u64 = 2;
        self.pass_read_limit = Some((budget_bytes as u64) / SCRATCH_SHARE_DIVISOR);
        self
    }
}

#[derive(Debug, Clone)]
struct ParquetPhysicalReadPlan {
    read_columns: Vec<String>,
    all_null_read_columns: Vec<String>,
    row_count: Option<usize>,
    requires_per_chunk_batch: bool,
}

impl ParquetPhysicalReadPlan {
    fn new(read_columns: Vec<String>) -> Self {
        Self { read_columns, all_null_read_columns: Vec::new(), row_count: None, requires_per_chunk_batch: false }
    }

    fn without_all_null_columns(
        read_columns: Vec<String>,
        all_null_read_columns: Vec<String>,
        selected_row_count: usize,
    ) -> Self {
        let all_null_set: FastHashSet<&str> = all_null_read_columns.iter().map(String::as_str).collect();
        let read_columns = read_columns.into_iter().filter(|column| !all_null_set.contains(column.as_str())).collect();
        Self {
            read_columns,
            all_null_read_columns,
            row_count: Some(selected_row_count),
            requires_per_chunk_batch: false,
        }
    }

    fn per_chunk_batch(read_columns: Vec<String>) -> Self {
        Self { read_columns, all_null_read_columns: Vec::new(), row_count: None, requires_per_chunk_batch: true }
    }
}

fn parquet_physical_read_plan(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    input: &SourceReadInput<'_>,
) -> Result<ParquetPhysicalReadPlan> {
    let read_columns = chunk.read_columns().to_vec();
    if read_columns.is_empty()
        || source.reader_predicate().is_some()
        || chunk.remote_source().is_some()
        || !matches!(input, SourceReadInput::Path)
    {
        return Ok(ParquetPhysicalReadPlan::new(read_columns));
    }

    match parquet_all_null_read_columns(source, chunk, &read_columns)? {
        Some((all_null_read_columns, selected_row_count)) => Ok(ParquetPhysicalReadPlan::without_all_null_columns(
            read_columns,
            all_null_read_columns,
            selected_row_count,
        )),
        None => Ok(ParquetPhysicalReadPlan::new(read_columns)),
    }
}

/// Returns the nullable read columns that the source file's row-group null
/// counts prove all-null across the chunk's row groups, with the selected row
/// count, or `None` when no read column is proven all-null.
fn parquet_all_null_read_columns(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    read_columns: &[String],
) -> Result<Option<(Vec<String>, usize)>> {
    let Some(file) = source.files().get(chunk.file_index()) else {
        return Ok(None);
    };
    let Some(row_group_rows) = file.row_group_rows() else {
        return Ok(None);
    };
    let selected_row_groups = if chunk.reads_all_row_groups() {
        (0..row_group_rows.len()).collect()
    } else {
        chunk.row_group_indexes().to_vec()
    };
    if selected_row_groups.is_empty() || selected_row_groups.iter().any(|row_group| *row_group >= row_group_rows.len())
    {
        return Ok(None);
    }
    let all_null_read_columns = read_columns
        .iter()
        .filter(|read_column| source.schema().column(read_column).is_some_and(|column| column.nullable()))
        .filter(|read_column| {
            file.row_group_null_counts().get(read_column.as_str()).is_some_and(|null_counts| {
                selected_row_groups
                    .iter()
                    .all(|row_group| null_counts.get(*row_group).copied().flatten() == Some(row_group_rows[*row_group]))
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    if all_null_read_columns.is_empty() {
        return Ok(None);
    }
    let selected_row_count = selected_row_group_row_count(row_group_rows, &selected_row_groups)?;
    Ok(Some((all_null_read_columns, selected_row_count)))
}

fn parquet_multi_physical_read_plan(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
) -> Result<ParquetPhysicalReadPlan> {
    let read_columns = source.read_columns().to_vec();
    if read_columns.is_empty()
        || source.reader_predicate().is_some()
        || chunks.iter().any(|chunk| chunk.remote_source().is_some())
    {
        return Ok(ParquetPhysicalReadPlan::new(read_columns));
    }

    let mut chunk_plans = Vec::with_capacity(chunks.len());
    let mut selected_row_count = 0usize;
    for chunk in chunks {
        let plan = parquet_physical_read_plan(source, chunk, &SourceReadInput::Path)?;
        if let Some(row_count) = plan.row_count {
            selected_row_count = selected_row_count.checked_add(row_count).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    "dataframe::read_parquet_source",
                    "Parquet footer selected row count overflows usize",
                )
                .with_detail("reason", "parquet_footer_selected_row_count_overflow")
            })?;
        }
        chunk_plans.push(plan);
    }

    let Some(first_plan) = chunk_plans.first() else {
        return Ok(ParquetPhysicalReadPlan::new(read_columns));
    };
    if chunk_plans.iter().any(|plan| plan.read_columns != first_plan.read_columns) {
        return Ok(ParquetPhysicalReadPlan::per_chunk_batch(read_columns));
    }
    let all_null_read_columns = first_plan.all_null_read_columns.clone();
    if all_null_read_columns.is_empty() {
        return Ok(ParquetPhysicalReadPlan::new(read_columns));
    }

    Ok(ParquetPhysicalReadPlan::without_all_null_columns(read_columns, all_null_read_columns, selected_row_count))
}

fn selected_row_group_row_count(row_group_rows: &[u64], row_groups: &[usize]) -> Result<usize> {
    let row_count = row_groups
        .iter()
        .try_fold(0u64, |total, row_group| total.checked_add(row_group_rows[*row_group]))
        .and_then(|row_count| usize::try_from(row_count).ok());
    row_count.ok_or_else(|| {
        Error::new_with(
            ErrorCode::UnsupportedOperation,
            "dataframe::read_parquet_source",
            "selected Parquet row-group row count exceeds usize",
        )
        .with_detail("reason", "selected_row_group_row_count_exceeds_usize")
        .with_detail("chunk_row_groups", row_groups.len().to_string())
    })
}

pub(super) fn read_parquet_source_chunk_frame(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    mode: SourceReadMode,
    input: SourceReadInput<'_>,
    pass_read_limit: Option<u64>,
    operation: &'static str,
) -> Result<ParquetSourceChunkReadOutput> {
    source.validate_chunk_partition_values(chunk, operation)?;

    if source.kind() == LocalParquetSourceKind::MetadataRowCount {
        let count = chunk.known_row_count_i64()?.ok_or_else(|| missing_source_row_count_error(operation, chunk))?;
        let count_series = runtime.series(&[count])?;
        let frame = DataFrame::from_series([(ZERO_COLUMN_COUNT_COLUMN, count_series)])
            .map_err(|error| error.reanchor(operation))?;
        let frame = project_source_frame(&frame, source, mode).map_err(|error| error.reanchor(operation))?;
        return Ok(ParquetSourceChunkReadOutput::new(frame, None));
    }

    let physical_read_plan = parquet_physical_read_plan(source, chunk, &input)?;
    let (mut frame, row_count_override, read_evidence) = if physical_read_plan.read_columns.is_empty() {
        let row_count = if let Some(row_count) = physical_read_plan.row_count {
            row_count
        } else {
            let row_count = chunk.known_row_count().ok_or_else(|| missing_source_row_count_error(operation, chunk))?;
            usize::try_from(row_count).map_err(|_| {
                Error::new_with(ErrorCode::UnsupportedOperation, operation, "source chunk row count exceeds usize")
                    .with_detail("reason", "row_count_exceeds_usize")
                    .with_detail("row_count", row_count.to_string())
                    .with_detail("chunk_index", chunk.chunk_index().to_string())
            })?
        };
        let read_evidence =
            physical_read_plan.row_count.is_some().then(ParquetReadEvidence::local_path_footer_materialized);
        (runtime.empty_dataframe(), Some(row_count), read_evidence)
    } else {
        let row_groups = if chunk.reads_all_row_groups() { None } else { Some(chunk.row_group_indexes_i32()?) };
        let table_options = match input {
            SourceReadInput::Path => {
                if let Some(remote_source) = chunk.remote_source() {
                    let mut remote_source = remote_source.clone();
                    if let Some(known_file_bytes) = chunk.known_file_bytes() {
                        remote_source = remote_source.with_known_size_bytes(known_file_bytes);
                    }
                    ParquetTableReadOptions::from_remote_parts(
                        remote_source,
                        Some(physical_read_plan.read_columns.clone()),
                        0,
                        None,
                        row_groups,
                        source.reader_predicate().cloned(),
                        runtime.clone(),
                    )
                } else {
                    ParquetTableReadOptions::from_parts(
                        std::path::PathBuf::from(chunk.reader_path()),
                        Some(physical_read_plan.read_columns.clone()),
                        0,
                        None,
                        row_groups,
                        source.reader_predicate().cloned(),
                        runtime.clone(),
                    )
                }
            },
            SourceReadInput::HostBuffer(buffer) => ParquetTableReadOptions::from_host_buffer_parts(
                buffer,
                Some(physical_read_plan.read_columns.clone()),
                0,
                None,
                row_groups,
                source.reader_predicate().cloned(),
                runtime.clone(),
            ),
        }
        .with_pass_read_limit(pass_read_limit);
        let result = parquet::read_parquet_table_with_evidence(&table_options).map_err(|error| {
            error
                .reanchor(operation)
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("stable_id", chunk.stable_id().to_string())
                .with_detail_if_absent("path", source_chunk_diagnostic_path(chunk))
        })?;
        let read_evidence = result.read_evidence().cloned();
        let (table, names) = require_reader_schema_names(result.into_table_read_output(), operation, "parquet reader")
            .map_err(|error| {
                error
                    .reanchor(operation)
                    .with_detail("chunk_index", chunk.chunk_index().to_string())
                    .with_detail("file_index", chunk.file_index().to_string())
                    .with_detail("stable_id", chunk.stable_id().to_string())
                    .with_detail("path", source_chunk_diagnostic_path(chunk))
            })?;
        (
            DataFrame::from_table(table, names).map_err(|error| error.reanchor(operation))?,
            physical_read_plan.row_count,
            read_evidence,
        )
    };

    if !physical_read_plan.all_null_read_columns.is_empty() {
        let row_count = row_count_override.unwrap_or_else(|| frame.num_rows());
        frame = materialize_all_null_read_columns(
            runtime,
            source,
            frame,
            &physical_read_plan.all_null_read_columns,
            row_count,
        )?;
    }
    frame = materialize_partition_columns(runtime, source, chunk, frame, row_count_override)?;
    frame = materialize_missing_null_columns(runtime, source, frame, row_count_override, mode)?;
    let frame = project_source_frame(&frame, source, mode).map_err(|error| error.reanchor(operation))?;
    Ok(ParquetSourceChunkReadOutput::new(frame, read_evidence))
}

pub(super) fn read_parquet_source_chunks_frame(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    mode: SourceReadMode,
    pass_read_limit: Option<u64>,
    operation: &'static str,
) -> Result<ParquetSourceChunkReadOutput> {
    if chunks.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "source chunk batch must not be empty")
            .with_detail("reason", "empty_source_chunk_batch"));
    }
    if source.kind() != LocalParquetSourceKind::Data {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "multi-chunk Parquet source reads require data sources",
        )
        .with_detail("reason", "unsupported_source_kind")
        .with_detail("source_kind", local_parquet_source_kind_str(source.kind())));
    }
    if source.read_columns().is_empty() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "multi-chunk Parquet source reads require at least one physical read column",
        )
        .with_detail("reason", "empty_read_projection"));
    }
    if !source.partition_columns().is_empty() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "multi-chunk Parquet source reads do not yet materialize per-file partition columns",
        )
        .with_detail("reason", "partition_columns_not_supported")
        .with_detail("partition_column_count", source.partition_columns().len().to_string()));
    }

    let omit_row_groups = chunks.iter().all(SourceChunk::selects_all_known_file_row_groups);
    let mut inputs = Vec::with_capacity(chunks.len());
    let mut row_groups_per_file = (!omit_row_groups).then(|| Vec::<Vec<i32>>::with_capacity(chunks.len()));
    let mut previous_file_index: Option<usize> = None;
    for chunk in chunks {
        source.validate_chunk_partition_values(chunk, operation)?;
        if !omit_row_groups && chunk.reads_all_row_groups() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                operation,
                "multi-chunk Parquet source reads require explicit row groups for each child chunk",
            )
            .with_detail("reason", "reads_all_row_groups_not_supported")
            .with_detail("chunk_index", chunk.chunk_index().to_string())
            .with_detail("file_index", chunk.file_index().to_string())
            .with_detail("path", source_chunk_diagnostic_path(chunk)));
        }
        let merges_into_previous =
            source_chunk_continues_previous_reader_input(chunk, previous_file_index, omit_row_groups);
        previous_file_index = Some(chunk.file_index());

        if !merges_into_previous {
            if let Some(remote_source) = chunk.remote_source() {
                let mut remote_source = remote_source.clone();
                if let Some(known_file_bytes) = chunk.known_file_bytes() {
                    remote_source = remote_source.with_known_size_bytes(known_file_bytes);
                }
                inputs.push(ParquetMultiTableInput::RemoteSource(remote_source));
            } else {
                inputs.push(ParquetMultiTableInput::LocalPath(std::path::PathBuf::from(chunk.reader_path())));
            }
        }
        if let Some(row_groups_per_file) = row_groups_per_file.as_mut() {
            let row_groups = chunk.row_group_indexes_i32()?;
            if row_groups.is_empty() {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    operation,
                    "multi-chunk Parquet source reads require non-empty row-group selections",
                )
                .with_detail("reason", "empty_row_group_selection")
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("path", source_chunk_diagnostic_path(chunk)));
            }
            match row_groups_per_file.last_mut() {
                Some(selection) if merges_into_previous => selection.extend(row_groups),
                _ => row_groups_per_file.push(row_groups),
            }
        }
    }

    let physical_read_plan = parquet_multi_physical_read_plan(source, chunks)?;
    if physical_read_plan.requires_per_chunk_batch {
        return read_parquet_source_chunks_frame_per_chunk(runtime, source, chunks, mode, pass_read_limit, operation);
    }
    let (mut frame, read_evidence) = if physical_read_plan.read_columns.is_empty() {
        (runtime.empty_dataframe(), Some(ParquetReadEvidence::local_path_multi_footer_materialized()))
    } else {
        let table_options = ParquetMultiTableReadOptions::from_inputs(
            inputs,
            Some(physical_read_plan.read_columns.clone()),
            row_groups_per_file,
            source.reader_predicate().cloned(),
            runtime.clone(),
        );
        let table_options = ParquetMultiTableReadOptions { pass_read_limit, ..table_options };
        let result = parquet::read_parquet_multi_table(&table_options).map_err(|error| error.reanchor(operation))?;
        let read_evidence = result.read_evidence().cloned();
        let (table, names) =
            require_reader_schema_names(result.into_table_read_output(), operation, "multi-source parquet reader")?;
        (DataFrame::from_table(table, names).map_err(|error| error.reanchor(operation))?, read_evidence)
    };
    if !physical_read_plan.all_null_read_columns.is_empty() {
        let row_count = physical_read_plan.row_count.unwrap_or_else(|| frame.num_rows());
        frame = materialize_all_null_read_columns(
            runtime,
            source,
            frame,
            &physical_read_plan.all_null_read_columns,
            row_count,
        )?;
    }
    frame = materialize_missing_null_columns(runtime, source, frame, None, mode)?;
    let frame = project_source_frame(&frame, source, mode).map_err(|error| error.reanchor(operation))?;
    Ok(ParquetSourceChunkReadOutput::new(frame, read_evidence))
}

fn read_parquet_source_chunks_frame_per_chunk(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    mode: SourceReadMode,
    pass_read_limit: Option<u64>,
    operation: &'static str,
) -> Result<ParquetSourceChunkReadOutput> {
    let mut frames = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let (frame, _read_evidence) = read_parquet_source_chunk_frame(
            runtime,
            source,
            chunk,
            mode,
            SourceReadInput::Path,
            pass_read_limit,
            operation,
        )?
        .into_parts();
        frames.push(frame);
    }

    let Some((first, rest)) = frames.split_first() else {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "source chunk batch must not be empty")
            .with_detail("reason", "empty_source_chunk_batch"));
    };
    let rest_refs: Vec<&DataFrame> = rest.iter().collect();
    let frame = first.on(Exec::RETAINED).concat_many(&rest_refs).map_err(|error| error.reanchor(operation))?;
    Ok(ParquetSourceChunkReadOutput::new(frame, Some(ParquetReadEvidence::local_path_multi_footer_materialized())))
}

pub(super) fn dataframe_from_parquet_table_options(
    table_options: &ParquetTableReadOptions<'_>,
    operation: &'static str,
) -> Result<DataFrame> {
    let result = parquet::read_parquet_table(table_options).map_err(|error| error.reanchor(operation))?;
    let (table, names) = require_reader_schema_names(result, operation, "parquet reader")?;
    DataFrame::from_table(table, names).map_err(|error| error.reanchor(operation))
}
