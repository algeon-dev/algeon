use super::*;

pub(super) fn stream_has_multi_source_batches(stream: &BoundedJoinLeafLocalParquetSourceStream) -> bool {
    stream.source.partition_columns.is_empty()
        && source_chunk_batching_reader_predicate_supported(&stream.source)
        && source_chunk_batches(stream).iter().any(|batch| batch.chunk_count() > 1)
}

pub(super) fn source_chunk_batches(stream: &BoundedJoinLeafLocalParquetSourceStream) -> Vec<SourceChunkBatch> {
    join_leaf_source_chunk_batches(
        &stream.source,
        &stream.chunks,
        &stream.reader_dynamic_filters,
        stream.planning_policy.max_source_chunk_bytes,
    )
}

pub(super) fn join_leaf_source_chunk_batches(
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    _reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    max_batch_source_read_bytes: Option<u64>,
) -> Vec<SourceChunkBatch> {
    if let Some(reason) = join_leaf_source_batching_disabled_reason(source) {
        return singleton_source_chunk_batches(chunks, reason);
    }
    let target_bytes = if source_chunk_batch_decode_scratch_risk(source, chunks) {
        capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes)
    } else {
        max_batch_source_read_bytes
    };
    plan_source_chunk_batches(chunks, target_bytes)
}

pub(super) fn source_chunk_batching_reader_predicate_supported(source: &LocalParquetSource) -> bool {
    source.reader_predicate.as_ref().is_none_or(|predicate| source.reader_predicate_is_supported(predicate))
}

pub(super) fn join_leaf_source_batching_disabled_reason(source: &LocalParquetSource) -> Option<&'static str> {
    if !source_chunk_batching_reader_predicate_supported(source) {
        return Some("source_gate:unsupported_reader_predicate");
    }
    if !source.partition_columns.is_empty() {
        return Some("source_gate:partition_columns");
    }
    None
}

pub(super) fn annotate_join_leaf_source_chunk_batching_reasons(
    source: &LocalParquetSource,
    chunks: &mut [SourceChunk],
    max_batch_source_read_bytes: Option<u64>,
) {
    let target_bytes = if source_chunk_batch_decode_scratch_risk(source, chunks) {
        capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes)
    } else {
        max_batch_source_read_bytes
    };
    annotate_source_chunk_batching_reasons(chunks, join_leaf_source_batching_disabled_reason(source), target_bytes);
}

pub(super) fn singleton_source_chunk_batches(chunks: &[SourceChunk], reason: &'static str) -> Vec<SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| SourceChunkBatch {
            batch_index,
            chunks: vec![chunk.clone()],
            estimated_source_read_bytes: chunk
                .estimated_source_read_bytes
                .or(chunk.known_row_group_bytes)
                .or(chunk.estimated_chunk_bytes)
                .or(chunk.known_file_bytes),
            source_batching_reason: reason.to_string(),
        })
        .collect()
}
pub(super) fn source_chunk_batch_target_bytes(policy: MemoryControlPolicy) -> Option<u64> {
    match policy {
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => Some(max_source_chunk_bytes),
        MemoryControlPolicy::Unbounded => None,
    }
}

pub(super) fn capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes: Option<u64>) -> Option<u64> {
    max_batch_source_read_bytes
        .map(|max_source_chunk_bytes| max_source_chunk_bytes.min(SOURCE_BATCH_READ_MAX_PLANNED_BYTES))
        .or(Some(SOURCE_BATCH_READ_MAX_PLANNED_BYTES))
}

pub(super) fn source_chunk_batch_decode_scratch_risk(source: &LocalParquetSource, chunks: &[SourceChunk]) -> bool {
    source.reader_predicate.is_some()
        && chunks.iter().any(|chunk| {
            chunk.known_pure_filter_compressed_bytes.unwrap_or(0) > 0
                || chunk.known_pure_filter_uncompressed_bytes.unwrap_or(0) > 0
        })
}

pub(super) fn source_chunk_batch_requested_bytes(policy: MemoryControlPolicy, batch: &SourceChunkBatch) -> u64 {
    batch.chunks.iter().fold(0u64, |total, chunk| total.saturating_add(chunk_requested_bytes(policy, chunk)))
}
