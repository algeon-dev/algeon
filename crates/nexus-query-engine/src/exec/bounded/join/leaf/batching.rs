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

pub(super) fn materialized_source_chunk_batches(
    stream: &BoundedJoinLeafLocalParquetSourceStream,
) -> Vec<SourceChunkBatch> {
    if let Some(reason) = join_leaf_source_batching_disabled_reason(&stream.source) {
        return singleton_source_chunk_batches(&stream.chunks, reason);
    }
    plan_source_chunk_batches(&stream.chunks, materialized_source_batch_target_bytes(stream))
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
    let target_bytes = capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes);
    plan_source_chunk_batches(chunks, target_bytes)
}

fn materialized_source_batch_target_bytes(stream: &BoundedJoinLeafLocalParquetSourceStream) -> Option<u64> {
    if stream.stages.is_empty()
        && stream.source.scan_limit.is_none()
        && stream.source.partition_columns.is_empty()
        && stream.chunks.iter().all(source_chunk_requires_remote_direct_read)
    {
        return Some(BoundedSourceBatchingPolicy::default().remote_direct_materialized_join_source_batch_max_bytes());
    }
    stream.planning_policy.max_source_chunk_bytes
}

fn source_chunk_requires_remote_direct_read(chunk: &SourceChunk) -> bool {
    matches!(chunk.resolved_source.read_strategy, crate::source::ParquetReadStrategy::RemoteDirect)
        && chunk.resolved_source.remote_read_mode == crate::source::RemoteParquetReadMode::RequireDirect
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
    if source.files.iter().any(LocalParquetFile::has_position_delete_files) {
        return Some("source_gate:position_deletes");
    }
    None
}

pub(super) fn annotate_join_leaf_source_chunk_batching_reasons(
    source: &LocalParquetSource,
    chunks: &mut [SourceChunk],
    max_batch_source_read_bytes: Option<u64>,
) {
    let target_bytes = capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes);
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
    BoundedSourceBatchingPolicy::default().generic_source_batch_target_bytes(policy, false)
}

pub(super) fn capped_source_chunk_batch_target_bytes(max_batch_source_read_bytes: Option<u64>) -> Option<u64> {
    BoundedSourceBatchingPolicy::default().capped_source_batch_target_bytes(max_batch_source_read_bytes)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        runtime::SourceChunkPlanningMode,
        source::{
            CredentialMode, LocalParquetSource, ParquetSourceLocation, RemoteParquetReadMode, ResolvedParquetSource,
        },
    };

    #[test]
    fn join_leaf_batches_cap_large_non_predicate_source_reads() {
        let source = test_source();
        let chunk_bytes = BoundedSourceBatchingPolicy::DEFAULT_SOURCE_BATCH_READ_MAX_PLANNED_BYTES / 2 + 1;
        let chunks = vec![test_chunk(0, chunk_bytes), test_chunk(1, chunk_bytes)];

        let batches = join_leaf_source_chunk_batches(&source, &chunks, &[], Some(32 * 1024 * 1024 * 1024));

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].chunk_count(), 1);
        assert_eq!(batches[1].chunk_count(), 1);
        assert!(batches.iter().all(|batch| batch.estimated_source_read_bytes <= Some(chunk_bytes)));
    }

    #[test]
    fn materialized_remote_direct_join_source_batches_above_stream_cap_to_reduce_concat_fan_in() {
        let source = test_source();
        let chunk_bytes = 4 * 1024 * 1024 * 1024;
        let chunks = vec![
            test_remote_direct_chunk(0, chunk_bytes),
            test_remote_direct_chunk(1, chunk_bytes),
            test_remote_direct_chunk(2, chunk_bytes),
        ];
        let regular_batches = join_leaf_source_chunk_batches(&source, &chunks, &[], Some(4 * 1024 * 1024 * 1024));
        assert_eq!(regular_batches.len(), 3);

        let stream = BoundedJoinLeafLocalParquetSourceStream {
            source,
            node_id: NodeId::from_index_for_internal_use(1),
            chunks,
            stages: Vec::new(),
            reader_dynamic_filters: Vec::new(),
            planning_policy: SourceChunkPlanningPolicy::new(Some(4 * 1024 * 1024 * 1024), 512),
            max_attempts: 1,
            min_retry_chunk_row_groups: 1,
        };

        let materialized_batches = materialized_source_chunk_batches(&stream);
        assert_eq!(materialized_batches.len(), 2);
        assert_eq!(materialized_batches[0].chunk_count(), 2);
        assert_eq!(materialized_batches[0].estimated_source_read_bytes, Some(2 * chunk_bytes));
        assert_eq!(materialized_batches[1].chunk_count(), 1);
        assert_eq!(materialized_batches[1].estimated_source_read_bytes, Some(chunk_bytes));
    }

    fn test_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        LocalParquetSource::from_files(Vec::new(), schema, vec!["c0".to_string()])
    }

    fn test_chunk(index: usize, bytes: u64) -> SourceChunk {
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index: index,
            location: ParquetSourceLocation::LocalPath(PathBuf::from(format!("part-{index}.parquet"))),
            path: PathBuf::from(format!("part-{index}.parquet")),
            resolved_source: ResolvedParquetSource::local_kvikio(format!("part-{index}.parquet")),
            stable_id: format!("file-{index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(bytes),
            known_row_group_bytes: Some(bytes),
            known_projected_compressed_bytes: Some(bytes),
            known_projected_uncompressed_bytes: Some(bytes),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(bytes),
            estimated_source_read_bytes: Some(bytes),
            exact_source_read_preflight_bytes: Some(bytes),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![bytes]),
            source_file_projected_compressed_bytes: Some(vec![bytes]),
            source_file_projected_uncompressed_bytes: Some(vec![bytes]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![1]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["c0".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(1),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }

    fn test_remote_direct_chunk(index: usize, bytes: u64) -> SourceChunk {
        SourceChunk {
            resolved_source: ResolvedParquetSource::remote_direct(
                format!("s3://bucket/table/part-{index}.parquet"),
                CredentialMode::SharedProcessEnv,
            )
            .with_remote_read_mode(RemoteParquetReadMode::RequireDirect),
            location: ParquetSourceLocation::ObjectStoreUri {
                scheme: "s3".to_string(),
                uri: format!("s3://bucket/table/part-{index}.parquet"),
            },
            path: PathBuf::from(format!("s3://bucket/table/part-{index}.parquet")),
            ..test_chunk(index, bytes)
        }
    }
}
