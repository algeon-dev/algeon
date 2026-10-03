use super::{ExecutionMetrics, NodeId, QueryError, RetryFacts, SourceChunk, SourceChunkBatch};
use crate::{
    error::ResourceBoundary,
    exec::{
        retry::{ResourceFailureRecovery, resource_failure_recovery},
        source::{attach_row_group_facts_if_absent, parquet_chunk_path_fact},
    },
};

#[derive(Debug, Clone, Copy)]
pub(in crate::exec) struct RuntimeResourceFacts<'a> {
    pub(in crate::exec) memory_boundary: &'static str,
    pub(in crate::exec) stage: &'static str,
    pub(in crate::exec) node_id: NodeId,
    pub(in crate::exec) attempt: usize,
    pub(in crate::exec) max_attempts: usize,
    pub(in crate::exec) resume_boundary: &'static str,
    pub(in crate::exec) metrics: &'a ExecutionMetrics,
}

impl RuntimeResourceFacts<'_> {
    pub(in crate::exec) fn attach_retry_context(self, error: QueryError) -> QueryError {
        let progress = self.metrics.progress();
        RetryFacts::new()
            .with_resume_boundary(self.resume_boundary)
            .with_committed_chunks(progress.source_chunks_committed as usize)
            .with_committed_partials(progress.partial_aggregate_count as usize)
            .attach_to_error(error)
    }

    /// Attaches the boundary that owned the failed work to every resource
    /// failure, terminal ones included. Whether the boundary may retry stays
    /// with [`resource_failure_recovery`]; an inner boundary wins.
    pub(in crate::exec) fn attach_to(self, error: QueryError) -> QueryError {
        if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
            return error;
        }

        let error = error
            .with_resource_boundary(
                ResourceBoundary::new(self.memory_boundary, self.node_id)
                    .with_stage(self.stage)
                    .with_resume(self.resume_boundary),
            )
            .with_fact_if_absent("attempt", self.attempt)
            .with_fact_if_absent("max_attempts", self.max_attempts);
        self.attach_retry_context(error)
    }
}

pub(in crate::exec) fn attach_chunk_resource_facts(
    error: QueryError,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    metrics: &ExecutionMetrics,
) -> QueryError {
    if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
        return error;
    }

    let error = RuntimeResourceFacts {
        memory_boundary,
        stage: memory_boundary,
        node_id: chunk.source_node_id,
        attempt,
        max_attempts,
        resume_boundary: "source_chunk",
        metrics,
    }
    .attach_to(error)
    .with_fact_if_absent("chunk_index", chunk.chunk_index)
    .with_fact_if_absent("file_index", chunk.file_index)
    .with_fact_if_absent("path", parquet_chunk_path_fact(chunk));
    attach_row_group_facts_if_absent(error, chunk)
}

pub(in crate::exec) fn attach_source_batch_resource_facts(
    error: QueryError,
    batch: &SourceChunkBatch,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    metrics: &ExecutionMetrics,
) -> QueryError {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return attach_chunk_resource_facts(error, first_chunk, memory_boundary, attempt, max_attempts, metrics);
    }
    if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
        return error;
    }

    RuntimeResourceFacts {
        memory_boundary,
        stage: memory_boundary,
        node_id: first_chunk.source_node_id,
        attempt,
        max_attempts,
        resume_boundary: "source_chunk_batch",
        metrics,
    }
    .attach_to(error)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use cudf::{CudfErrorDetail, CudfErrorFailureSource, testing::error as cudf_error};

    use super::*;
    use crate::{
        error::QueryErrorCode,
        native_error::cudf_query_error,
        runtime::SourceChunkPlanningMode,
        source::{ParquetSourceLocation, ResolvedParquetSource},
    };

    fn test_chunk() -> SourceChunk {
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: 0,
            file_index: 0,
            location: ParquetSourceLocation::LocalPath(PathBuf::from("part-0.parquet")),
            path: PathBuf::from("part-0.parquet"),
            resolved_source: ResolvedParquetSource::local_kvikio("part-0.parquet"),
            stable_id: "file-0".to_string(),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(1024),
            known_row_group_bytes: Some(1024),
            known_projected_compressed_bytes: Some(1024),
            known_projected_uncompressed_bytes: Some(1024),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(1024),
            estimated_source_read_bytes: Some(1024),
            exact_source_read_preflight_bytes: Some(1024),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![1024]),
            source_file_projected_compressed_bytes: Some(vec![1024]),
            source_file_projected_uncompressed_bytes: Some(vec![1024]),
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

    /// An allocation failure converted the way production converts cuDF errors.
    fn converted_allocation_failure(failure_source: CudfErrorFailureSource) -> QueryError {
        let source = cudf_error::with_ffi_detail_for_test(
            cudf_error::new_with_for_test(cudf::ErrorCode::OutOfMemory, "cudf.test.chunk_read", "synthetic"),
            CudfErrorDetail {
                status: 0,
                failure_source,
                domain_id: 0,
                member_id: 0,
                generation: 0,
                requested_bytes: 4096,
                alignment: 256,
                cuda_error: -1,
                message: "synthetic allocation failure".to_string(),
            },
        );
        cudf_query_error("query_engine.test.chunk_facts", "converted allocation failure", source)
    }

    #[test]
    fn chunk_resource_facts_attach_a_typed_boundary_to_every_resource_failure() {
        let metrics = ExecutionMetrics::detached("chunk_resource_facts_attach");
        let chunk = test_chunk();
        for (failure_source, code, recovery) in [
            (
                CudfErrorFailureSource::ReservationLimit,
                QueryErrorCode::QueryMemoryLimitExceeded,
                ResourceFailureRecovery::ShrinkWorkingSet,
            ),
            (CudfErrorFailureSource::Physical, QueryErrorCode::DeviceOutOfMemory, ResourceFailureRecovery::Terminal),
            (
                CudfErrorFailureSource::Unknown,
                QueryErrorCode::DeviceAllocationFailed,
                ResourceFailureRecovery::Terminal,
            ),
        ] {
            let error = attach_chunk_resource_facts(
                converted_allocation_failure(failure_source),
                &chunk,
                "source_chunk_read",
                1,
                3,
                &metrics,
            );
            assert_eq!(error.code(), code, "the original allocation cause survives");
            assert_eq!(resource_failure_recovery(&error), recovery, "a boundary never widens retry");
            let boundary = error.context().boundary.expect("every resource failure carries its boundary");
            assert_eq!(boundary.memory_boundary, "source_chunk_read");
            assert_eq!(boundary.node_id, Some(chunk.source_node_id));
            assert_eq!(boundary.resume, Some("source_chunk"));
            assert_eq!(error.fact_value("chunk_index"), Some("0"));
            assert_eq!(error.fact_value("max_attempts"), Some("3"));

            let outer = RuntimeResourceFacts {
                memory_boundary: "outer_stage",
                stage: "outer_stage",
                node_id: NodeId::from_index_for_internal_use(9),
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "outer_stage",
                metrics: &metrics,
            }
            .attach_to(error);
            assert_eq!(outer.context().boundary, Some(boundary), "an outer scope keeps the inner location");
        }

        let not_resource =
            QueryError::new(QueryErrorCode::UnsupportedExpression, "query_engine.test.chunk_facts", "not a resource");
        let not_resource = attach_chunk_resource_facts(not_resource, &chunk, "source_chunk_read", 1, 3, &metrics);
        assert!(not_resource.context().boundary.is_none());
        assert_eq!(not_resource.fact_value("chunk_index"), None);
    }
}
