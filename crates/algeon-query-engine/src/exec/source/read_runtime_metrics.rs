//! Builders for the `SourceReadRuntimeMetrics` rows that source reads offer to the plan
//! execution's recorder.
//!
//! The row type, its field names and `to_stable_kv_row` are a public metrics contract owned by
//! `query_runtime::metrics`; this module only fills them. `read_kind` identifies the shape:
//! `chunk`, `chunk_batch`, `optimized_chunk_batch`, `file`, and the two enqueue-only kinds
//! `background_cache_fill_file` and `background_cache_fill_chunk`. Byte fields are plan-time
//! metadata and are 0 when the plan has no value for them. `output_rows` and `elapsed` are measured.
//! `query_id` is left `None`; the recorder side owns correlation.

use std::{collections::BTreeSet, time::Duration};

use cudf::io::parquet::{ParquetPruningEvidence, ParquetReadEvidence, RemoteReadMode};
use parquet::file::reader::SerializedFileReader;

use super::{
    ExecutionContext, FileReader, LocalParquetFile, LocalParquetSource, NodeId, SourceChunk,
    SourceReadParquetEvidenceMetrics, SourceReadRuntimeMetrics, record_source_read_row,
};
use crate::chunk::SourceChunkBatch;

/// Records that a background cache fill was considered for a file. The row describes the enqueue
/// decision, so `output_rows` and `elapsed` are zero, `source_batching_reason` holds
/// the enqueue outcome (`enqueued`, a skip reason, or an `enqueue_failed_*` string), and
/// `source_chunk_planning_reason` holds the fill mode.
pub(super) fn record_background_cache_file_fill_enqueue_runtime(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    outcome: String,
    context: &ExecutionContext,
) {
    record_source_read_row(
        context,
        SourceReadRuntimeMetrics {
            query_id: None,
            source_node_id: source_node_id.to_string(),
            read_kind: "background_cache_fill_file".to_string(),
            read_strategy: file.resolved_source.read_strategy().as_str().to_string(),
            parquet_read_evidence: SourceReadParquetEvidenceMetrics::default(),
            batch_index: file_index as u64,
            chunk_count: 0,
            first_chunk_index: file_index as u64,
            last_chunk_index: file_index as u64,
            first_file_index: file_index as u64,
            last_file_index: file_index as u64,
            file_count: 1,
            row_group_count: file.row_groups.as_ref().map_or(0, |row_groups| row_groups.len() as u64),
            read_columns_per_file: source.read_columns.len() as u64,
            read_columns_total: source.read_columns.len() as u64,
            known_file_bytes: file.file_size_bytes.unwrap_or(0),
            estimated_source_read_bytes: file.file_size_bytes.unwrap_or(0),
            known_row_group_bytes: selected_file_metadata_sum(
                file.row_group_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_projected_compressed_bytes: selected_file_metadata_sum(
                file.projected_row_group_compressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_projected_uncompressed_bytes: selected_file_metadata_sum(
                file.projected_row_group_uncompressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_pure_filter_compressed_bytes: selected_file_metadata_sum(
                file.pure_filter_row_group_compressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_pure_filter_uncompressed_bytes: selected_file_metadata_sum(
                file.pure_filter_row_group_uncompressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            output_rows: 0,
            reader_predicate: false,
            reader_dynamic_filter_count: 0,
            source_batching_reason: outcome,
            source_chunk_planning_mode: "background_cache_fill".to_string(),
            source_chunk_planning_reason: file.resolved_source.lakehouse_cache_fill_mode.as_str().to_string(),
            elapsed: Duration::ZERO,
        },
    );
}

/// Chunk counterpart of `record_background_cache_file_fill_enqueue_runtime`, with the same
/// repurposed outcome and mode fields.
pub(super) fn record_background_cache_chunk_fill_enqueue_runtime(
    chunk: &SourceChunk,
    outcome: String,
    context: &ExecutionContext,
) {
    record_source_read_row(
        context,
        SourceReadRuntimeMetrics {
            query_id: None,
            source_node_id: chunk.source_node_id.to_string(),
            read_kind: "background_cache_fill_chunk".to_string(),
            read_strategy: chunk.resolved_source.read_strategy().as_str().to_string(),
            parquet_read_evidence: SourceReadParquetEvidenceMetrics::default(),
            batch_index: chunk.chunk_index as u64,
            chunk_count: 0,
            first_chunk_index: chunk.chunk_index as u64,
            last_chunk_index: chunk.chunk_index as u64,
            first_file_index: chunk.file_index as u64,
            last_file_index: chunk.file_index as u64,
            file_count: 1,
            row_group_count: chunk.row_group_count() as u64,
            read_columns_per_file: chunk.read_columns.len() as u64,
            read_columns_total: chunk.read_columns.len() as u64,
            known_file_bytes: chunk.known_file_bytes.unwrap_or(0),
            estimated_source_read_bytes: chunk.known_file_bytes.unwrap_or(0),
            known_row_group_bytes: chunk.known_row_group_bytes.unwrap_or(0),
            known_projected_compressed_bytes: chunk.known_projected_compressed_bytes.unwrap_or(0),
            known_projected_uncompressed_bytes: chunk.known_projected_uncompressed_bytes.unwrap_or(0),
            known_pure_filter_compressed_bytes: chunk.known_pure_filter_compressed_bytes.unwrap_or(0),
            known_pure_filter_uncompressed_bytes: chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0),
            output_rows: 0,
            reader_predicate: false,
            reader_dynamic_filter_count: 0,
            source_batching_reason: outcome,
            source_chunk_planning_mode: "background_cache_fill".to_string(),
            source_chunk_planning_reason: chunk.resolved_source.lakehouse_cache_fill_mode.as_str().to_string(),
            elapsed: Duration::ZERO,
        },
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
/// Records one chunk read. `row_group_count` prefers the count observed for this read, then the
/// count derivable from the chunk (`observed_chunk_row_group_count`), then the planned count.
pub(super) fn record_source_chunk_read_runtime(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    observed_row_group_count: Option<usize>,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    read_evidence: Option<&ParquetReadEvidence>,
    context: &ExecutionContext,
) {
    record_source_read_row(
        context,
        SourceReadRuntimeMetrics {
            query_id: None,
            source_node_id: chunk.source_node_id.to_string(),
            read_kind: "chunk".to_string(),
            read_strategy: chunk.resolved_source.read_strategy().as_str().to_string(),
            parquet_read_evidence: source_read_parquet_evidence_metrics(read_evidence),
            batch_index: chunk.chunk_index as u64,
            chunk_count: 1,
            first_chunk_index: chunk.chunk_index as u64,
            last_chunk_index: chunk.chunk_index as u64,
            first_file_index: chunk.file_index as u64,
            last_file_index: chunk.file_index as u64,
            file_count: 1,
            row_group_count: observed_row_group_count
                .or_else(|| observed_chunk_row_group_count(chunk))
                .unwrap_or_else(|| chunk.row_group_count()) as u64,
            read_columns_per_file: chunk.read_columns.len() as u64,
            read_columns_total: chunk.read_columns.len() as u64,
            known_file_bytes: chunk.known_file_bytes.unwrap_or(0),
            estimated_source_read_bytes: best_chunk_source_read_bytes(chunk),
            known_row_group_bytes: chunk.known_row_group_bytes.unwrap_or(0),
            known_projected_compressed_bytes: chunk.known_projected_compressed_bytes.unwrap_or(0),
            known_projected_uncompressed_bytes: chunk.known_projected_uncompressed_bytes.unwrap_or(0),
            known_pure_filter_compressed_bytes: chunk.known_pure_filter_compressed_bytes.unwrap_or(0),
            known_pure_filter_uncompressed_bytes: chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0),
            output_rows,
            reader_predicate: source.reader_predicate.is_some(),
            reader_dynamic_filter_count: reader_dynamic_filter_count as u64,
            source_batching_reason: chunk.source_batching_reason.clone().unwrap_or_else(|| "not_evaluated".to_string()),
            source_chunk_planning_mode: chunk.source_chunk_planning_mode.as_str().to_string(),
            source_chunk_planning_reason: chunk.source_chunk_planning_reason.clone(),
            elapsed,
        },
    );
}

/// Row groups a chunk read covers, or `None` when that cannot be determined cheaply.
///
/// Uses the planned count when the plan records it. A chunk that reads all row groups of a
/// file with no recorded count opens the file footer on the calling thread, which only works for
/// a path that exists locally; an object-store chunk without a count returns `None`.
pub(in crate::exec) fn observed_chunk_row_group_count(chunk: &SourceChunk) -> Option<usize> {
    chunk.known_read_row_group_count().or_else(|| {
        (chunk.reads_all_row_groups && chunk.path.is_file())
            .then(|| {
                SerializedFileReader::try_from(chunk.path.as_path())
                    .ok()
                    .map(|reader| reader.metadata().num_row_groups())
            })
            .flatten()
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
/// Records one read of a batch of chunks as a single row. `file_count` counts distinct file
/// indexes, and the byte fields are sums over the chunks. `estimated_source_read_bytes` is the
/// batch's own estimate when it has one, otherwise the sum of the per-chunk best estimates.
pub(super) fn record_source_chunk_batch_read_runtime(
    source: &LocalParquetSource,
    batch: &SourceChunkBatch,
    read_kind: &'static str,
    output_rows: u64,
    elapsed: Duration,
    reader_dynamic_filter_count: usize,
    read_evidence: Option<&ParquetReadEvidence>,
    context: &ExecutionContext,
) {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    let last_chunk = batch.chunks.last().expect("source chunk batch is non-empty");
    let file_count = batch.chunks.iter().map(|chunk| chunk.file_index).collect::<BTreeSet<_>>().len();
    let estimated_source_read_bytes = batch
        .estimated_source_read_bytes
        .unwrap_or_else(|| batch.chunks.iter().map(best_chunk_source_read_bytes).sum());
    record_source_read_row(
        context,
        SourceReadRuntimeMetrics {
            query_id: None,
            source_node_id: first_chunk.source_node_id.to_string(),
            read_kind: read_kind.to_string(),
            read_strategy: first_chunk.resolved_source.read_strategy().as_str().to_string(),
            parquet_read_evidence: source_read_parquet_evidence_metrics(read_evidence),
            batch_index: batch.batch_index as u64,
            chunk_count: batch.chunk_count() as u64,
            first_chunk_index: first_chunk.chunk_index as u64,
            last_chunk_index: last_chunk.chunk_index as u64,
            first_file_index: first_chunk.file_index as u64,
            last_file_index: last_chunk.file_index as u64,
            file_count: file_count as u64,
            row_group_count: batch.row_group_count() as u64,
            read_columns_per_file: source.read_column_count() as u64,
            read_columns_total: batch.chunks.iter().map(|chunk| chunk.read_columns.len() as u64).sum(),
            known_file_bytes: batch.chunks.iter().map(|chunk| chunk.known_file_bytes.unwrap_or(0)).sum(),
            estimated_source_read_bytes,
            known_row_group_bytes: batch.chunks.iter().map(|chunk| chunk.known_row_group_bytes.unwrap_or(0)).sum(),
            known_projected_compressed_bytes: batch
                .chunks
                .iter()
                .map(|chunk| chunk.known_projected_compressed_bytes.unwrap_or(0))
                .sum(),
            known_projected_uncompressed_bytes: batch
                .chunks
                .iter()
                .map(|chunk| chunk.known_projected_uncompressed_bytes.unwrap_or(0))
                .sum(),
            known_pure_filter_compressed_bytes: batch
                .chunks
                .iter()
                .map(|chunk| chunk.known_pure_filter_compressed_bytes.unwrap_or(0))
                .sum(),
            known_pure_filter_uncompressed_bytes: batch
                .chunks
                .iter()
                .map(|chunk| chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0))
                .sum(),
            output_rows,
            reader_predicate: source.reader_predicate.is_some(),
            reader_dynamic_filter_count: reader_dynamic_filter_count as u64,
            source_batching_reason: batch.source_batching_reason.clone(),
            source_chunk_planning_mode: first_chunk.source_chunk_planning_mode.as_str().to_string(),
            source_chunk_planning_reason: first_chunk.source_chunk_planning_reason.clone(),
            elapsed,
        },
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
/// Records a whole-file read. It does not go through chunk planning, so the planning mode is
/// `whole_file`, the batching reason is `not_evaluated`, and no dynamic filters apply.
pub(super) fn record_source_file_read_runtime(
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    observed_row_group_count: u64,
    output_rows: u64,
    elapsed: Duration,
    read_evidence: Option<&ParquetReadEvidence>,
    context: &ExecutionContext,
) {
    record_source_read_row(
        context,
        SourceReadRuntimeMetrics {
            query_id: None,
            source_node_id: source_node_id.to_string(),
            read_kind: "file".to_string(),
            read_strategy: file.resolved_source.read_strategy().as_str().to_string(),
            parquet_read_evidence: source_read_parquet_evidence_metrics(read_evidence),
            batch_index: chunk_index as u64,
            chunk_count: 1,
            first_chunk_index: chunk_index as u64,
            last_chunk_index: chunk_index as u64,
            first_file_index: file_index as u64,
            last_file_index: file_index as u64,
            file_count: 1,
            row_group_count: observed_row_group_count,
            read_columns_per_file: source.read_columns.len() as u64,
            read_columns_total: source.read_columns.len() as u64,
            known_file_bytes: file.file_size_bytes.unwrap_or(0),
            estimated_source_read_bytes: best_file_source_read_bytes(file),
            known_row_group_bytes: selected_file_metadata_sum(
                file.row_group_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_projected_compressed_bytes: selected_file_metadata_sum(
                file.projected_row_group_compressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_projected_uncompressed_bytes: selected_file_metadata_sum(
                file.projected_row_group_uncompressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_pure_filter_compressed_bytes: selected_file_metadata_sum(
                file.pure_filter_row_group_compressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            known_pure_filter_uncompressed_bytes: selected_file_metadata_sum(
                file.pure_filter_row_group_uncompressed_bytes.as_deref(),
                file.row_groups.as_deref(),
            ),
            output_rows,
            reader_predicate: source.reader_predicate.is_some(),
            reader_dynamic_filter_count: 0,
            source_batching_reason: "not_evaluated".to_string(),
            source_chunk_planning_mode: "whole_file".to_string(),
            source_chunk_planning_reason: "whole_file_read".to_string(),
            elapsed,
        },
    );
}

/// Converts cuDF's read evidence into the metric fields. `None` evidence yields the default
/// (empty) evidence. `bloom_pruning_status` is `not_reported` when cuDF reported no pruning
/// evidence at all. The pruning counts are `None` for a stage cuDF did not report; `None` does not
/// mean zero row groups survived.
pub(super) fn source_read_parquet_evidence_metrics(
    evidence: Option<&ParquetReadEvidence>,
) -> SourceReadParquetEvidenceMetrics {
    let Some(evidence) = evidence else {
        return SourceReadParquetEvidenceMetrics::default();
    };
    let pruning = evidence.pruning;
    SourceReadParquetEvidenceMetrics {
        input_kind: evidence.input_kind.to_string(),
        remote_read_mode: evidence.requested_remote_read_mode.map(RemoteReadMode::as_str).unwrap_or("none").to_string(),
        backend: evidence.backend.to_string(),
        backend_build_status: evidence.backend_build_status.as_str().to_string(),
        remote_bytes: evidence.remote_bytes,
        remote_range_count: evidence.remote_range_count,
        device_read_support: evidence.device_read_support.to_string(),
        pruning_input_row_groups: pruning.map(|pruning| pruning.input_row_groups),
        row_groups_after_stats_filter: pruning.and_then(|pruning| pruning.row_groups_after_stats_filter),
        row_groups_after_bloom_filter: pruning.and_then(|pruning| pruning.row_groups_after_bloom_filter),
        bloom_pruning_status: bloom_pruning_status(pruning).unwrap_or("not_reported").to_string(),
    }
}

/// Classifies Bloom-filter pruning from the stage counts cuDF reported.
///
/// `bloom_pruned` when the Bloom stage left fewer row groups than the stage before it (the
/// statistics stage, or the input when statistics did not report). `bloom_not_pruned` when the
/// Bloom stage reported and removed nothing. `bloom_not_reported` when only the statistics stage
/// reported. `no_reader_predicate` when pruning evidence exists but neither stage reported; the
/// stage counts are documented as present only for a read with a predicate. `None` when no
/// pruning evidence exists.
pub(super) fn bloom_pruning_status(pruning: Option<ParquetPruningEvidence>) -> Option<&'static str> {
    let pruning = pruning?;
    match (pruning.row_groups_after_stats_filter, pruning.row_groups_after_bloom_filter) {
        (Some(after_stats), Some(after_bloom)) if after_bloom < after_stats => Some("bloom_pruned"),
        (None, Some(after_bloom)) if after_bloom < pruning.input_row_groups => Some("bloom_pruned"),
        (_, Some(_)) => Some("bloom_not_pruned"),
        (Some(_), None) => Some("bloom_not_reported"),
        (None, None) => Some("no_reader_predicate"),
    }
}

/// Byte estimate for a whole-file read: the largest of the selected row-group bytes, the selected
/// projected compressed bytes and the file size. When the file size is known the estimate is
/// therefore at least the file size, even if only some row groups are selected.
pub(super) fn best_file_source_read_bytes(file: &LocalParquetFile) -> u64 {
    selected_file_metadata_sum(file.row_group_bytes.as_deref(), file.row_groups.as_deref())
        .max(selected_file_metadata_sum(
            file.projected_row_group_compressed_bytes.as_deref(),
            file.row_groups.as_deref(),
        ))
        .max(file.file_size_bytes.unwrap_or(0))
}

/// Sums per-row-group `values` over the selected row groups, or over all of them when `row_groups`
/// is `None`. Both are file-local row-group ordinals. Missing metadata and out-of-range ordinals
/// contribute 0.
pub(super) fn selected_file_metadata_sum(values: Option<&[u64]>, row_groups: Option<&[usize]>) -> u64 {
    let Some(values) = values else {
        return 0;
    };
    match row_groups {
        Some(row_groups) => row_groups.iter().filter_map(|index| values.get(*index)).sum(),
        None => values.iter().sum(),
    }
}

/// Byte estimate for a chunk, taking the first fact the plan has, from the most specific
/// (`estimated_source_read_bytes`) to the least (`known_file_bytes`). 0 when none is known.
pub(super) fn best_chunk_source_read_bytes(chunk: &SourceChunk) -> u64 {
    chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes)
        .unwrap_or(0)
}
