use super::{SourceChunk, SourceChunkBatch};
use crate::source::ParquetReadStrategy;

impl SourceChunkBatch {
    /// Splits a retryable multi-source batch into two ordered child batches.
    ///
    /// Returns `None` for singleton batches. Child estimates are recomputed from
    /// the same best-available source-read bytes used by initial batch planning.
    pub fn split_for_retry(self) -> Option<(Self, Self)> {
        if self.chunks.len() <= 1 {
            return None;
        }
        let split_index = self.chunks.len() / 2;
        let mut left_chunks = self.chunks;
        let right_chunks = left_chunks.split_off(split_index);
        Some((source_chunk_batch_from_retry_chunks(left_chunks), source_chunk_batch_from_retry_chunks(right_chunks)))
    }
}

fn source_chunk_batch_from_retry_chunks(chunks: Vec<SourceChunk>) -> SourceChunkBatch {
    let batch_index = chunks[0].chunk_index;
    SourceChunkBatch {
        batch_index,
        estimated_source_read_bytes: source_read_bytes_for_batch_chunks(&chunks),
        source_batching_reason: "retry_split".to_string(),
        chunks,
    }
}

fn source_read_bytes_for_batch_chunks(chunks: &[SourceChunk]) -> Option<u64> {
    chunks.iter().try_fold(0u64, |total, chunk| {
        batchable_chunk_source_read_bytes(chunk).map(|bytes| total.saturating_add(bytes))
    })
}

/// Groups compatible adjacent chunks for multi-source Parquet reads.
///
/// Incompatible chunks remain singleton batches. `max_batch_source_read_bytes`
/// is a planning target; chunks without source-read byte estimates are kept as
/// singleton batches so batching cannot silently defeat bounded memory policy.
pub fn plan_source_chunk_batches(
    chunks: &[SourceChunk],
    max_batch_source_read_bytes: Option<u64>,
) -> Vec<SourceChunkBatch> {
    plan_source_chunk_batches_with_row_cap(chunks, max_batch_source_read_bytes, None)
}

/// Groups compatible adjacent chunks like [`plan_source_chunk_batches`] and
/// additionally caps each batch's known row count.
///
/// Source-read bytes describe encoded input; a row cap lets a consumer bound
/// the decoded working set it observed on earlier batches. Chunks without a
/// known row count stay singleton batches while a row cap is in effect.
pub fn plan_source_chunk_batches_with_row_cap(
    chunks: &[SourceChunk],
    max_batch_source_read_bytes: Option<u64>,
    max_batch_rows: Option<u64>,
) -> Vec<SourceChunkBatch> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes: Option<u64> = Some(0);
    let mut current_rows: Option<u64> = Some(0);
    let mut current_reason: Option<&'static str> = None;

    for chunk in chunks {
        let chunk_bytes = batchable_chunk_source_read_bytes(chunk);
        let extension_rejection = current.last().and_then(|previous| {
            source_chunk_batch_extension_rejection(
                previous,
                current_bytes,
                chunk,
                chunk_bytes,
                max_batch_source_read_bytes,
            )
            .or_else(|| source_chunk_batch_row_extension_rejection(current_rows, chunk, max_batch_rows))
        });
        let can_extend = current.last().is_some() && extension_rejection.is_none();
        if !current.is_empty() && !can_extend {
            if current.len() == 1 && current_reason.is_none() {
                current_reason = extension_rejection;
            }
            push_source_chunk_batch(&mut batches, &mut current, current_bytes, current_reason);
            current_bytes = Some(0);
            current_rows = Some(0);
            current_reason = None;
        }

        if let Some(reason) = chunk_multi_source_batch_rejection(chunk) {
            current_bytes = None;
            current_reason.get_or_insert(reason);
        } else {
            current_bytes = current_bytes.and_then(|total| chunk_bytes.and_then(|bytes| total.checked_add(bytes)));
        }
        current_rows = current_rows.and_then(|total| chunk.known_row_count.and_then(|rows| total.checked_add(rows)));
        current.push(chunk.clone());
    }

    if !current.is_empty() {
        push_source_chunk_batch(&mut batches, &mut current, current_bytes, current_reason);
    }
    batches
}

fn push_source_chunk_batch(
    batches: &mut Vec<SourceChunkBatch>,
    current: &mut Vec<SourceChunk>,
    estimated_source_read_bytes: Option<u64>,
    reason: Option<&'static str>,
) {
    if current.is_empty() {
        return;
    }
    let source_batching_reason = if current.len() > 1 { "batched" } else { reason.unwrap_or("trailing_singleton") };
    batches.push(SourceChunkBatch {
        batch_index: batches.len(),
        chunks: std::mem::take(current),
        estimated_source_read_bytes,
        source_batching_reason: source_batching_reason.to_string(),
    });
}

fn chunk_multi_source_batch_rejection(chunk: &SourceChunk) -> Option<&'static str> {
    if chunk.zero_column_count {
        return Some("ineligible_zero_column_count");
    }
    if !matches!(
        chunk.resolved_source.read_strategy(),
        ParquetReadStrategy::LocalKvikio | ParquetReadStrategy::LocalCacheKvikio | ParquetReadStrategy::RemoteDirect
    ) {
        return Some("ineligible_read_strategy");
    }
    if chunk.read_columns.is_empty() {
        return Some("ineligible_empty_read_columns");
    }
    if !chunk.partition_values.is_empty() {
        return Some("ineligible_partition_values");
    }
    if chunk.scan_limit_rows.is_some() {
        return Some("ineligible_scan_limit");
    }
    if chunk.reads_all_row_groups {
        return Some("ineligible_reads_all_row_groups");
    }
    if chunk.row_group_indexes.is_empty() {
        return Some("ineligible_empty_row_groups");
    }
    if batchable_chunk_source_read_bytes(chunk).is_none() {
        return Some("ineligible_missing_source_read_bytes");
    }
    None
}

fn source_chunk_batch_extension_rejection(
    left: &SourceChunk,
    current_bytes: Option<u64>,
    right: &SourceChunk,
    right_bytes: Option<u64>,
    max_batch_source_read_bytes: Option<u64>,
) -> Option<&'static str> {
    if let Some(reason) = chunk_multi_source_batch_rejection(left) {
        return Some(reason);
    }
    if let Some(reason) = chunk_multi_source_batch_rejection(right) {
        return Some(reason);
    }
    if left.source_node_id != right.source_node_id {
        return Some("incompatible_source_node");
    }
    if left.chunk_index + 1 != right.chunk_index {
        return Some("incompatible_chunk_index_gap");
    }
    if left.file_index == right.file_index {
        if !source_chunks_have_adjacent_row_groups(left, right) {
            return Some("incompatible_row_group_gap");
        }
    } else if left.file_index + 1 != right.file_index {
        return Some("incompatible_file_index_gap");
    }
    if left.read_columns != right.read_columns {
        return Some("incompatible_read_columns");
    }
    if left.resolved_source.read_strategy() != right.resolved_source.read_strategy() {
        return Some("incompatible_read_strategy");
    }
    let Some(current_bytes) = current_bytes else {
        return Some("incompatible_missing_current_bytes");
    };
    let Some(right_bytes) = right_bytes else {
        return Some("incompatible_missing_next_bytes");
    };
    if let Some(max_bytes) = max_batch_source_read_bytes
        && current_bytes.saturating_add(right_bytes) > max_bytes
    {
        return Some("byte_target_exceeded");
    }
    None
}

fn source_chunk_batch_row_extension_rejection(
    current_rows: Option<u64>,
    right: &SourceChunk,
    max_batch_rows: Option<u64>,
) -> Option<&'static str> {
    let max_rows = max_batch_rows?;
    let Some(current_rows) = current_rows else {
        return Some("incompatible_missing_current_rows");
    };
    let Some(right_rows) = right.known_row_count else {
        return Some("incompatible_missing_next_rows");
    };
    (current_rows.saturating_add(right_rows) > max_rows).then_some("row_target_exceeded")
}

fn source_chunks_have_adjacent_row_groups(left: &SourceChunk, right: &SourceChunk) -> bool {
    let (Some(left_last), Some(right_first)) = (left.row_group_indexes.last(), right.row_group_indexes.first()) else {
        return false;
    };
    left_last.checked_add(1).is_some_and(|next| next == *right_first)
}

/// Annotates chunks with the batching decision that would apply under the
/// provided source-level gate and byte target.
pub fn annotate_source_chunk_batching_reasons(
    chunks: &mut [SourceChunk],
    source_level_reason: Option<&'static str>,
    max_batch_source_read_bytes: Option<u64>,
) {
    if let Some(reason) = source_level_reason {
        for chunk in chunks {
            chunk.source_batching_reason = Some(reason.to_string());
        }
        return;
    }
    let batches = plan_source_chunk_batches(chunks, max_batch_source_read_bytes);
    for batch in batches {
        for chunk in batch.chunks {
            if let Some(target) = chunks.get_mut(chunk.chunk_index) {
                target.source_batching_reason = Some(batch.source_batching_reason.clone());
            }
        }
    }
}

fn batchable_chunk_source_read_bytes(chunk: &SourceChunk) -> Option<u64> {
    chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes)
}
