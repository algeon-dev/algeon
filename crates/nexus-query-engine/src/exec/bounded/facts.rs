use super::*;
use crate::exec::source::{SourceChunkRowGroupFacts, parquet_chunk_path_fact};

#[derive(Debug, Clone, Copy)]
pub(in crate::exec) struct RuntimeResourceFacts<'a> {
    pub(in crate::exec) memory_boundary: &'static str,
    pub(in crate::exec) stage: &'static str,
    pub(in crate::exec) node_id: NodeId,
    pub(in crate::exec) attempt: usize,
    pub(in crate::exec) max_attempts: usize,
    pub(in crate::exec) resume_boundary: &'static str,
    pub(in crate::exec) metrics: &'a QueryMetrics,
}

impl RuntimeResourceFacts<'_> {
    pub(in crate::exec) fn attach_retry_context(self, error: QueryError) -> QueryError {
        RetryFacts::new()
            .with_resume_boundary(self.resume_boundary)
            .with_committed_chunks(self.metrics.source_chunks_committed as usize)
            .with_committed_partials(self.metrics.partial_aggregate_count as usize)
            .attach_to_error(error)
    }

    pub(in crate::exec) fn attach_to(self, error: QueryError) -> QueryError {
        if error.code() != QueryErrorCode::CudfResourceExhausted {
            return error;
        }

        let error = error
            .with_fact_if_absent("memory_boundary", self.memory_boundary)
            .with_fact_if_absent("stage", self.stage)
            .with_fact_if_absent("node_id", self.node_id)
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
    metrics: &QueryMetrics,
) -> QueryError {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
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
    error.with_row_group_facts_if_absent(chunk)
}

pub(in crate::exec) fn attach_source_batch_resource_facts(
    error: QueryError,
    batch: &SourceChunkBatch,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    metrics: &QueryMetrics,
) -> QueryError {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return attach_chunk_resource_facts(error, first_chunk, memory_boundary, attempt, max_attempts, metrics);
    }
    if error.code() != QueryErrorCode::CudfResourceExhausted {
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
