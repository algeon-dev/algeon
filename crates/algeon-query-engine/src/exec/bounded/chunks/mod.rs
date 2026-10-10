//! Source chunk work shared by compiled source tasks and the bounded
//! aggregate path: reading one chunk or chunk batch, the source-adjacent
//! filter and projection stages that run on the decoded frame, and the retry
//! protocol that halves a chunk by row group after a resource failure.
//!
//! Everything here runs on whichever execution attempt it is given, so the
//! same functions serve the coordinator inline and a lane job on its own
//! stream. Outputs stay private to the caller until it commits them; a
//! `SourceReadFact::ChunkCommitted` is recorded only by the function that
//! owns that commit.

use super::{
    BoundedExecutionAttempt, BoundedPlan, BoundedSourceBatchingPolicy, CommittedPartial, ExecutionBudgetToken,
    ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetSource, MemoryControlPolicy, NodeId,
    PartialAggregateInputEvidence, ProjectionNode, QueryError, ResidentGpuFrame, Result, RetryFacts, SourceChunk,
    SourceChunkBatch, attach_chunk_resource_facts, attach_source_batch_resource_facts, cudf_query_error,
    execute_filter_with_budget, execute_partial_aggregate_with_budget, execute_projection_with_budget,
    execute_projection_with_budget_and_prepared, execution_budget_token_for_boundary,
    is_source_adjacent_grouped_aggregate, observe_device_memory_bytes, partial_aggregate_source_chunk_fingerprint,
    plan_source_chunk_batches, projection_regexp_replace_prepare,
};
use crate::exec::source::{ScheduledLocalParquetReadOutput, read_local_parquet_chunk, read_local_parquet_chunk_batch};

mod execution;
mod retry;
mod source_adjacent;
mod source_budget;
mod source_read;

pub(in crate::exec::bounded) use execution::{
    RawSourceInput, execute_bounded_raw_source_chunk, execute_bounded_raw_source_chunk_batch,
};
pub(in crate::exec) use retry::record_chunk_retry_error;
pub(super) use retry::{chunk_retry_facts, is_retryable_chunk_error, split_retryable_chunk};
pub(in crate::exec::bounded) use source_adjacent::{
    bounded_source_adjacent_chunk_batches_for_work_width, bounded_source_adjacent_chunk_batching_enabled,
    execute_source_adjacent_batch_uncommitted,
};
pub(super) use source_budget::join_leaf_prefetch_admitted;
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunk_batch_required_only_no_commit,
    read_local_parquet_chunks_required_only,
};
