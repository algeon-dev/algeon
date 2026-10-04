use query_runtime::observability::{RuntimeExecutionObservationScope, SourceReadFact};

use super::{ExecutionMetrics, LocalParquetSource, QueryError, SourceChunk, SourceChunkPlan, comma_separated_usize};

/// Records what a source's chunk plan already knows about its Parquet files.
pub(in crate::exec) fn record_bounded_parquet_source_metrics(
    source: &LocalParquetSource,
    chunk_plan: &SourceChunkPlan,
    metrics: &ExecutionMetrics,
) {
    let mut planned_row_group_bytes_by_file = vec![0u64; source.files.len()];
    let mut extra_planned_row_group_bytes = 0u64;
    for chunk in &chunk_plan.chunks {
        let Some(bytes) = chunk.known_row_group_bytes else {
            continue;
        };
        if let Some(total) = planned_row_group_bytes_by_file.get_mut(chunk.file_index) {
            *total = total.saturating_add(bytes);
        } else {
            extra_planned_row_group_bytes = extra_planned_row_group_bytes.saturating_add(bytes);
        }
    }
    let known_bytes = source
        .files
        .iter()
        .enumerate()
        .map(|(index, file)| file.file_size_bytes.unwrap_or(0).max(planned_row_group_bytes_by_file[index]))
        .sum::<u64>()
        .saturating_add(extra_planned_row_group_bytes);

    metrics.record_source_read(SourceReadFact::ParquetSource {
        files: source.file_count() as u64,
        known_bytes,
        selected_row_groups: chunk_plan
            .chunks
            .iter()
            .map(|chunk| chunk.known_read_row_group_count().unwrap_or(0) as u64)
            .sum::<u64>(),
        read_columns: (source.file_count() * source.read_column_count()) as u64,
    });
    metrics.record_source_read(SourceReadFact::MetadataLoadTime(chunk_plan.metadata_load_time));
}

/// The plan execution a source-read fact belongs to.
///
/// Every production execution admits a plan-execution scope before any source
/// is read (`execute_query_plan_with_observation`), so a missing scope means a
/// harness skipped admission. That is a bug to surface, not a reason to lose
/// facts production would have counted.
pub(in crate::exec) fn admitted_source_read_scope(
    scope: Option<&RuntimeExecutionObservationScope>,
) -> Option<&RuntimeExecutionObservationScope> {
    debug_assert!(scope.is_some(), "source-read facts require an admitted plan-execution scope");
    scope
}

pub(in crate::exec) fn format_row_groups(chunk: &SourceChunk) -> String {
    if chunk.reads_all_row_groups {
        return "all".to_string();
    }

    comma_separated_usize(&chunk.row_group_indexes)
}

pub(in crate::exec) fn attach_row_group_facts(error: QueryError, chunk: &SourceChunk) -> QueryError {
    let error = error.with_fact("row_group_indexes", format_row_groups(chunk));
    if chunk.reads_all_row_groups { error } else { error.with_fact("row_group_count", chunk.row_group_indexes.len()) }
}

pub(in crate::exec) fn attach_row_group_facts_if_absent(error: QueryError, chunk: &SourceChunk) -> QueryError {
    let error = error.with_fact_if_absent("row_group_indexes", format_row_groups(chunk));
    if chunk.reads_all_row_groups {
        error
    } else {
        error.with_fact_if_absent("row_group_count", chunk.row_group_indexes.len())
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc, time::Duration};

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        execution_contracts::SourceChunkPlanningMode,
        plan::NodeId,
        source::{LocalParquetFile, LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
    };
    use query_runtime::observability::SourceReadTotals;

    use super::*;
    use crate::chunk::{SourceChunk, SourceChunkPlan};

    /// Runs `record` against a freshly admitted plan execution and returns its
    /// exact source-read totals, mirroring the checkpoint every real caller
    /// projects from.
    fn recorded_source_read_totals(label: &str, record: impl FnOnce(&ExecutionMetrics)) -> SourceReadTotals {
        let metrics = ExecutionMetrics::detached(label);
        record(&metrics);
        metrics.checkpoint().expect("a detached execution owns its metrics").summary.source_read
    }

    #[test]
    fn bounded_source_metrics_do_not_report_estimated_bytes_as_known() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/estimated-only.parquet")],
            Arc::clone(&schema),
            vec!["id".to_string()],
        );
        let chunk = estimated_only_chunk(0);
        let plan = SourceChunkPlan {
            chunks: vec![chunk],
            source_node_id: NodeId::from_index_for_test(0),
            row_group_metadata_available: true,
            size_estimate_based: true,
            metadata_unavailable_files: 0,
            metadata_load_time: Duration::ZERO,
            metadata_load_count: 0,
        };

        let totals = recorded_source_read_totals("estimated-only-single-file", |metrics| {
            record_bounded_parquet_source_metrics(&source, &plan, metrics);
        });

        assert_eq!(totals.parquet_bytes_known, 0);
        assert_eq!(totals.parquet_files, 1);
        assert_eq!(totals.row_groups_selected, 1);

        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/estimated-only.parquet").with_file_size_bytes(100)],
            Arc::clone(&schema),
            vec!["id".to_string()],
        );
        let first_file_chunk = estimated_only_chunk(0);
        let second_file_chunk = SourceChunk { chunk_index: 1, ..estimated_only_chunk(1) };
        let plan = SourceChunkPlan {
            chunks: vec![
                SourceChunk { known_file_bytes: Some(100), ..first_file_chunk },
                SourceChunk { known_file_bytes: Some(100), ..second_file_chunk },
            ],
            source_node_id: NodeId::from_index_for_test(0),
            row_group_metadata_available: true,
            size_estimate_based: true,
            metadata_unavailable_files: 0,
            metadata_load_time: Duration::ZERO,
            metadata_load_count: 0,
        };

        let totals = recorded_source_read_totals("estimated-only-known-file-size", |metrics| {
            record_bounded_parquet_source_metrics(&source, &plan, metrics);
        });

        assert_eq!(totals.parquet_bytes_known, 100);

        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/file-size-known.parquet").with_file_size_bytes(100),
                LocalParquetFile::new("/tmp/row-group-known.parquet"),
            ],
            schema,
            vec!["id".to_string()],
        );
        let plan = SourceChunkPlan {
            chunks: vec![
                SourceChunk {
                    known_file_bytes: Some(100),
                    ..estimated_only_chunk_for(0, 0, "/tmp/file-size-known.parquet")
                },
                SourceChunk {
                    known_row_group_bytes: Some(60),
                    ..estimated_only_chunk_for(1, 1, "/tmp/row-group-known.parquet")
                },
            ],
            source_node_id: NodeId::from_index_for_test(0),
            row_group_metadata_available: true,
            size_estimate_based: true,
            metadata_unavailable_files: 0,
            metadata_load_time: Duration::ZERO,
            metadata_load_count: 0,
        };

        let totals = recorded_source_read_totals("mixed-file-size-and-row-group-known", |metrics| {
            record_bounded_parquet_source_metrics(&source, &plan, metrics);
        });

        assert_eq!(totals.parquet_bytes_known, 160);
    }

    fn estimated_only_chunk(chunk_index: usize) -> SourceChunk {
        estimated_only_chunk_for(0, chunk_index, "/tmp/estimated-only.parquet")
    }

    fn estimated_only_chunk_for(file_index: usize, chunk_index: usize, path: &'static str) -> SourceChunk {
        SourceChunk {
            source_node_id: NodeId::from_index_for_test(0),
            chunk_index,
            file_index,
            location: ParquetSourceLocation::local_path(path),
            path: PathBuf::from(path),
            resolved_source: ResolvedParquetSource::local_kvikio(path),
            stable_id: "estimated-only".to_string(),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: None,
            known_row_group_bytes: None,
            known_projected_compressed_bytes: None,
            known_projected_uncompressed_bytes: None,
            known_pure_filter_compressed_bytes: None,
            known_pure_filter_uncompressed_bytes: None,
            estimated_chunk_bytes: Some(32),
            estimated_source_read_bytes: Some(64),
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: Some(vec![0]),
            source_file_row_group_bytes: None,
            source_file_projected_compressed_bytes: None,
            source_file_projected_uncompressed_bytes: None,
            source_file_pure_filter_compressed_bytes: None,
            source_file_pure_filter_uncompressed_bytes: None,
            source_file_row_group_rows: Some(vec![10]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["id".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: true,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        }
    }
}
