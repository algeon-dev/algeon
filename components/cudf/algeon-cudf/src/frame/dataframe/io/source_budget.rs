//! Byte preflight and budget facts for Parquet source reads.

use std::collections::BTreeMap;

use super::{super::owned::DataFrame, source_columns::source_chunk_diagnostic_path};
use crate::{
    core::{
        error::{Error, ErrorCode, Result},
        types::cudf_null_mask_allocation_size_bytes,
    },
    hash::fast_hash_set_with_capacity,
    io::{
        parquet::ParquetReadEvidence,
        source::{
            LocalParquetSource, LocalParquetSourceKind, PartitionColumn, SourceChunk, SourceColumn, SourceOutputColumn,
            SourceReadMode, SourceRequiredColumn,
        },
    },
};

pub(super) const PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES: usize = 64 * 1024;

/// Lower-level byte preflight for a bounded local Parquet source chunk read.
///
/// This is returned only for source shapes where cudf can prove the retained
/// output bytes before execution from the checked source contract, selected
/// row-count metadata, and fixed-width output types. Variable-width source
/// outputs and chunks without row-count metadata currently return `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParquetSourceReadPreflight {
    retained_output_bytes: usize,
    required_budget_bytes: usize,
}

impl ParquetSourceReadPreflight {
    pub(super) fn new(retained_output_bytes: usize) -> Self {
        Self {
            retained_output_bytes,
            required_budget_bytes: retained_output_bytes.max(PARQUET_SOURCE_MINIMUM_HARD_BUDGET_BYTES),
        }
    }

    /// Exact retained-output byte upper bound for the requested source frame.
    pub fn retained_output_bytes(&self) -> usize {
        self.retained_output_bytes
    }

    /// Minimum execution budget cudf recommends for this source read.
    pub fn required_budget_bytes(&self) -> usize {
        self.required_budget_bytes
    }
}

/// Output envelope for a planned Parquet source-chunk read.
#[derive(Debug)]
pub struct ParquetSourceChunkReadOutput {
    frame: DataFrame,
    read_evidence: Option<ParquetReadEvidence>,
}

impl ParquetSourceChunkReadOutput {
    pub(super) fn new(frame: DataFrame, read_evidence: Option<ParquetReadEvidence>) -> Self {
        Self { frame, read_evidence }
    }

    /// Decoded DataFrame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Per-read Parquet evidence reported by the lower-level reader.
    pub fn read_evidence(&self) -> Option<&ParquetReadEvidence> {
        self.read_evidence.as_ref()
    }

    /// Consume the output and return only the decoded DataFrame.
    pub fn into_frame(self) -> DataFrame {
        self.frame
    }

    /// Consume the output and return the decoded DataFrame plus read evidence.
    pub fn into_parts(self) -> (DataFrame, Option<ParquetReadEvidence>) {
        (self.frame, self.read_evidence)
    }
}

fn comma_separated_strings(values: &[String]) -> String {
    values.join(",")
}

fn comma_separated_usize(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn source_column_names<'a>(columns: impl IntoIterator<Item = &'a SourceOutputColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

fn required_column_names<'a>(columns: impl IntoIterator<Item = &'a SourceRequiredColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

fn partition_column_names<'a>(columns: impl IntoIterator<Item = &'a PartitionColumn>) -> String {
    columns.into_iter().map(|column| column.name()).collect::<Vec<_>>().join(",")
}

pub(super) fn local_parquet_source_kind_str(kind: LocalParquetSourceKind) -> &'static str {
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

pub(super) fn source_preflight_row_count(source: &LocalParquetSource, chunk: &SourceChunk) -> Result<Option<usize>> {
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

pub(super) fn source_preflight_column_bytes(column: &SourceColumn, rows: usize) -> Result<Option<usize>> {
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

pub(super) fn source_preflight_named_columns(
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

pub(super) fn parquet_source_budget_context(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    mode: SourceReadMode,
) -> BTreeMap<String, String> {
    let mut facts = BTreeMap::new();
    facts.insert("source_kind".to_string(), local_parquet_source_kind_str(source.kind()).to_string());
    facts.insert("source_read_mode".to_string(), source_read_mode_str(mode).to_string());
    facts.insert("path".to_string(), source_chunk_diagnostic_path(chunk));
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

/// Whether this chunk continues the previous chunk's reader input instead of
/// opening its own.
///
/// Chunk planning splits one file into row-group chunks, so a batch repeats the
/// same file once per chunk. Each reader input opens its own file handle and
/// parses the footer again, so an unmerged batch consumes file descriptors and
/// footer parses proportional to row groups rather than to files.
///
/// Merging is exact rather than approximate: chunks of a file stay adjacent and
/// ascending, and libcudf reads each input's selection in the given order, so
/// the concatenated selection is the order the separate inputs would have
/// produced. Whole-file chunks carry no selection to concatenate, and remote
/// sources carry per-input read state, so both keep one input each.
pub(super) fn source_chunk_continues_previous_reader_input(
    chunk: &SourceChunk,
    previous_file_index: Option<usize>,
    omit_row_groups: bool,
) -> bool {
    !omit_row_groups && chunk.remote_source().is_none() && previous_file_index == Some(chunk.file_index())
}

fn merged_reader_input_count(chunks: &[SourceChunk]) -> usize {
    let omit_row_groups = chunks.iter().all(SourceChunk::selects_all_known_file_row_groups);
    let mut previous_file_index = None;
    let mut inputs = 0usize;
    for chunk in chunks {
        if !source_chunk_continues_previous_reader_input(chunk, previous_file_index, omit_row_groups) {
            inputs += 1;
        }
        previous_file_index = Some(chunk.file_index());
    }
    inputs
}

pub(super) fn parquet_source_batch_budget_context(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    mode: SourceReadMode,
) -> BTreeMap<String, String> {
    let mut facts = BTreeMap::new();
    facts.insert("source_kind".to_string(), local_parquet_source_kind_str(source.kind()).to_string());
    facts.insert("source_read_mode".to_string(), source_read_mode_str(mode).to_string());
    facts.insert("chunk_count".to_string(), chunks.len().to_string());
    facts.insert("reader_input_count".to_string(), merged_reader_input_count(chunks).to_string());
    facts.insert("read_columns".to_string(), comma_separated_strings(source.read_columns()));
    facts.insert("read_column_count".to_string(), source.read_columns().len().to_string());
    facts.insert("projected_columns".to_string(), source_column_names(source.output_columns()));
    facts.insert("projected_column_count".to_string(), source.output_columns().len().to_string());
    facts.insert("required_columns".to_string(), required_column_names(source.required_columns()));
    facts.insert("required_column_count".to_string(), source.required_columns().len().to_string());
    facts.insert("partition_column_count".to_string(), source.partition_columns().len().to_string());
    if let Some(first) = chunks.first() {
        facts.insert("first_chunk_index".to_string(), first.chunk_index().to_string());
        facts.insert("first_file_index".to_string(), first.file_index().to_string());
    }
    if let Some(last) = chunks.last() {
        facts.insert("last_chunk_index".to_string(), last.chunk_index().to_string());
        facts.insert("last_file_index".to_string(), last.file_index().to_string());
    }
    let mut known_rows = 0u64;
    let mut all_rows_known = true;
    let mut known_source_read_bytes = 0u64;
    let mut all_source_read_bytes_known = true;
    for chunk in chunks {
        if let Some(rows) = chunk.known_row_count() {
            known_rows = known_rows.saturating_add(rows);
        } else {
            all_rows_known = false;
        }
        if let Some(bytes) = chunk.estimated_source_read_bytes() {
            known_source_read_bytes = known_source_read_bytes.saturating_add(bytes);
        } else {
            all_source_read_bytes_known = false;
        }
    }
    if all_rows_known {
        facts.insert("known_row_count".to_string(), known_rows.to_string());
    }
    if all_source_read_bytes_known {
        facts.insert("estimated_source_read_bytes".to_string(), known_source_read_bytes.to_string());
    }
    facts
}
