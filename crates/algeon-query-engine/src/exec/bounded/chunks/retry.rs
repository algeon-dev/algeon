use query_planning::chunk::assembly::proportional_bytes;

use super::{BoundedExecutionAttempt, ExecutionMetrics, QueryError, RetryFacts, SourceChunk};
use crate::exec::retry::{is_retryable_resource_error, record_retryable_resource_error};

pub(in crate::exec::bounded) fn is_retryable_chunk_error(error: &QueryError) -> bool {
    is_retryable_resource_error(error)
}

pub(in crate::exec) fn record_chunk_retry_error(error: &QueryError, metrics: &ExecutionMetrics) {
    record_retryable_resource_error(error, metrics);
}

/// Halves a chunk by row group for a retry after a resource failure.
///
/// `None` means the chunk cannot shrink: it reads a whole file (no row-group
/// list to split) or already holds at most `min_retry_chunk_row_groups`
/// row groups. The caller then fails with the original error and the retry
/// facts. Both halves keep the parent's file order so the outputs can be
/// concatenated in place of the parent.
pub(in crate::exec::bounded) fn split_retryable_chunk(
    chunk: &SourceChunk,
    min_retry_chunk_row_groups: usize,
) -> Option<(SourceChunk, SourceChunk)> {
    if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= min_retry_chunk_row_groups {
        return None;
    }

    let split_at = chunk.row_group_indexes.len() / 2;
    if split_at == 0 || split_at == chunk.row_group_indexes.len() {
        return None;
    }

    let left_row_groups = chunk.row_group_indexes[..split_at].to_vec();
    let right_row_groups = chunk.row_group_indexes[split_at..].to_vec();
    let mut left = retry_child_chunk(chunk, left_row_groups);
    let mut right = retry_child_chunk(chunk, right_row_groups);
    apply_parent_scan_limit_to_retry_children(chunk, &mut left, &mut right);
    Some((left, right))
}

/// Redistributes a parent's scan limit in file order: the left half may take
/// up to the whole limit, the right half gets what the left half's known row
/// count leaves. A child without a known row count takes the full remainder
/// as its own limit, because the reader enforces the limit per chunk.
fn apply_parent_scan_limit_to_retry_children(parent: &SourceChunk, left: &mut SourceChunk, right: &mut SourceChunk) {
    let Some(parent_limit) = parent.scan_limit_rows else {
        return;
    };
    let mut remaining = parent_limit as u64;
    apply_retry_child_scan_limit(left, &mut remaining);
    apply_retry_child_scan_limit(right, &mut remaining);
}

fn apply_retry_child_scan_limit(child: &mut SourceChunk, remaining: &mut u64) {
    let Some(row_count) = child.known_row_count else {
        child.scan_limit_rows = Some(*remaining as usize);
        return;
    };
    if *remaining == 0 {
        child.scan_limit_rows = Some(0);
        child.known_row_count = Some(0);
        return;
    }
    if row_count <= *remaining {
        *remaining = remaining.saturating_sub(row_count);
        return;
    }
    child.scan_limit_rows = Some(*remaining as usize);
    child.known_row_count = Some(*remaining);
    *remaining = 0;
}

/// Builds a child chunk over `row_group_indexes` of `parent`.
///
/// Size and row figures come from the footer's row-group vectors when the
/// parent carries them and are otherwise scaled from the parent's totals by
/// row-group share, which is why the child is marked `size_estimate_based`.
/// The exact preflight byte count is dropped: it measured the parent's read
/// and does not apply to a subset.
pub(in crate::exec::bounded) fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
    let parent_row_group_count = parent.row_group_indexes.len();
    let known_row_group_bytes =
        checked_row_group_sum(parent.source_file_row_group_bytes.as_deref(), &row_group_indexes).or_else(|| {
            proportional_bytes(parent.known_row_group_bytes, row_group_indexes.len(), parent_row_group_count)
        });
    let known_projected_compressed_bytes = checked_row_group_sum(
        parent.source_file_projected_compressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_projected_compressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_projected_uncompressed_bytes = checked_row_group_sum(
        parent.source_file_projected_uncompressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_projected_uncompressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_pure_filter_compressed_bytes = checked_row_group_sum(
        parent.source_file_pure_filter_compressed_bytes.as_deref(),
        &row_group_indexes,
    )
    .or_else(|| {
        proportional_bytes(parent.known_pure_filter_compressed_bytes, row_group_indexes.len(), parent_row_group_count)
    });
    let known_pure_filter_uncompressed_bytes =
        checked_row_group_sum(parent.source_file_pure_filter_uncompressed_bytes.as_deref(), &row_group_indexes)
            .or_else(|| {
                proportional_bytes(
                    parent.known_pure_filter_uncompressed_bytes,
                    row_group_indexes.len(),
                    parent_row_group_count,
                )
            });
    let known_file_bytes = parent.known_file_bytes;
    let estimated_chunk_bytes = proportional_bytes(
        parent.estimated_chunk_bytes.or(parent.known_row_group_bytes).or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );
    let estimated_source_read_bytes = proportional_bytes(
        parent
            .estimated_source_read_bytes
            .or(parent.known_row_group_bytes)
            .or(parent.estimated_chunk_bytes)
            .or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );
    let known_row_count = checked_row_group_sum(parent.source_file_row_group_rows.as_deref(), &row_group_indexes)
        .or_else(|| proportional_bytes(parent.known_row_count, row_group_indexes.len(), parent_row_group_count));

    SourceChunk {
        source_node_id: parent.source_node_id,
        chunk_index: parent.chunk_index,
        file_index: parent.file_index,
        location: parent.location.clone(),
        path: parent.path.clone(),
        resolved_source: parent.resolved_source.clone(),
        stable_id: parent.stable_id.clone(),
        row_group_indexes,
        reads_all_row_groups: false,
        known_file_bytes,
        known_row_group_bytes,
        known_projected_compressed_bytes,
        known_projected_uncompressed_bytes,
        known_pure_filter_compressed_bytes,
        known_pure_filter_uncompressed_bytes,
        estimated_chunk_bytes,
        estimated_source_read_bytes,
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: parent.source_file_row_groups.clone(),
        source_file_row_group_bytes: parent.source_file_row_group_bytes.clone(),
        source_file_projected_compressed_bytes: parent.source_file_projected_compressed_bytes.clone(),
        source_file_projected_uncompressed_bytes: parent.source_file_projected_uncompressed_bytes.clone(),
        source_file_pure_filter_compressed_bytes: parent.source_file_pure_filter_compressed_bytes.clone(),
        source_file_pure_filter_uncompressed_bytes: parent.source_file_pure_filter_uncompressed_bytes.clone(),
        source_file_row_group_rows: parent.source_file_row_group_rows.clone(),
        reader_decimal_types: parent.reader_decimal_types.clone(),
        reader_decimal_types_footer_verified: parent.reader_decimal_types_footer_verified,
        read_columns: parent.read_columns.clone(),
        partition_values: parent.partition_values.clone(),
        zero_column_count: parent.zero_column_count,
        known_row_count,
        scan_limit_rows: None,
        size_estimate_based: true,
        source_batching_reason: parent.source_batching_reason.clone(),
        source_chunk_planning_mode: parent.source_chunk_planning_mode,
        source_chunk_planning_reason: parent.source_chunk_planning_reason.clone(),
    }
}

fn checked_row_group_sum(values: Option<&[u64]>, row_group_indexes: &[usize]) -> Option<u64> {
    row_group_indexes.iter().try_fold(0u64, |total, row_group| total.checked_add(*values?.get(*row_group)?))
}

/// Facts for a chunk that failed with a retryable error and will not be
/// retried: either it was already at minimum size or the attempt budget ran
/// out. The committed counts tell a reader how much of the source survived.
pub(in crate::exec::bounded) fn chunk_retry_facts(
    chunk: &SourceChunk,
    execution: &BoundedExecutionAttempt<'_>,
) -> RetryFacts {
    let retry_action = if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= 1 {
        "fail_min_chunk"
    } else {
        "retry_attempts_exhausted"
    };

    let progress = execution.metrics().progress();
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("source_chunk")
        .with_committed_chunks(progress.source_chunks_committed as usize)
        .with_committed_partials(progress.partial_aggregate_count as usize)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use query_planning::{
        execution_contracts::SourceChunkPlanningMode,
        plan::NodeId,
        source::{ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, ResolvedParquetSource},
    };

    use super::*;

    fn retry_parent_chunk() -> SourceChunk {
        SourceChunk {
            source_node_id: NodeId::from_index_for_test(1),
            chunk_index: 0,
            file_index: 0,
            location: ParquetSourceLocation::LocalPath(PathBuf::from("part-0.parquet")),
            path: PathBuf::from("part-0.parquet"),
            resolved_source: ResolvedParquetSource::local_kvikio("part-0.parquet"),
            stable_id: "file-0".to_string(),
            row_group_indexes: vec![0, 1, 2, 3],
            reads_all_row_groups: false,
            known_file_bytes: Some(400),
            known_row_group_bytes: Some(400),
            known_projected_compressed_bytes: Some(200),
            known_projected_uncompressed_bytes: Some(400),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(400),
            estimated_source_read_bytes: Some(400),
            exact_source_read_preflight_bytes: Some(400),
            source_file_row_groups: Some(vec![0, 1, 2, 3]),
            source_file_row_group_bytes: Some(vec![100, 100, 100, 100]),
            source_file_projected_compressed_bytes: Some(vec![50, 50, 50, 50]),
            source_file_projected_uncompressed_bytes: Some(vec![100, 100, 100, 100]),
            source_file_pure_filter_compressed_bytes: Some(vec![0, 0, 0, 0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0, 0, 0, 0]),
            source_file_row_group_rows: Some(vec![10, 10, 10, 10]),
            reader_decimal_types: vec![ParquetReaderDecimalType::new(
                "price",
                ParquetReaderDecimalTypeId::Decimal64,
                -2,
            )],
            reader_decimal_types_footer_verified: true,
            read_columns: vec!["price".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(40),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: Some("batched".to_string()),
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }

    #[test]
    fn split_retryable_chunk_preserves_verified_decimal_reader_types() {
        let parent = retry_parent_chunk();

        let (left, right) = split_retryable_chunk(&parent, 1).expect("chunk should split");

        assert_eq!(left.row_group_indexes, vec![0, 1]);
        assert_eq!(right.row_group_indexes, vec![2, 3]);
        assert_eq!(left.reader_decimal_types, parent.reader_decimal_types);
        assert_eq!(right.reader_decimal_types, parent.reader_decimal_types);
        assert!(left.reader_decimal_types_footer_verified);
        assert!(right.reader_decimal_types_footer_verified);
    }

    #[test]
    fn split_retryable_chunk_preserves_full_file_size_hint_without_row_group_bytes() {
        let mut parent = retry_parent_chunk();
        parent.known_row_group_bytes = None;
        parent.source_file_row_group_bytes = None;

        let (left, right) = split_retryable_chunk(&parent, 1).expect("chunk should split");

        assert_eq!(left.known_file_bytes, Some(400));
        assert_eq!(right.known_file_bytes, Some(400));
        assert_eq!(left.estimated_chunk_bytes, Some(200));
        assert_eq!(right.estimated_chunk_bytes, Some(200));
        assert_eq!(left.estimated_source_read_bytes, Some(200));
        assert_eq!(right.estimated_source_read_bytes, Some(200));
    }
}
