//! Chunk planning for a deferred Parquet leaf stream.
//!
//! Each constructor validates the source, derives the chunk planning policy from the
//! memory-control policy, planning mode, a `SourceTaskMemoryModel` for the consumer and the
//! row-group metadata that sizes the chunks, and plans chunks. The aggregate variants take the
//! number of chunks their consumer may run at once on execution lanes, and every such chunk keeps
//! its own model slots live. Planning may load Parquet footers on the calling thread. The stream
//! starts with no stages or dynamic filters; later builders add them. The aggregate variants also
//! cap chunk sizes from the aggregate input hints. Every planned chunk is counted in the
//! source metrics, and the retry limits of the context are copied into the stream.

use query_planning::{chunk::row_group::RowGroupMetadata, pipeline::AggregateInputHints};

use super::{
    BoundedJoinLeafLocalParquetSourceStream, ExecutionContext, ExecutionMetrics, LocalParquetSource, NodeId, Result,
    SourceChunkPlanningPolicy, SourceTaskMemoryModel, annotate_join_leaf_source_chunk_batching_reasons,
    apply_source_adjacent_aggregate_chunk_caps, record_bounded_parquet_source_metrics, validate_local_parquet_source,
};
use crate::{
    chunk::plan_local_parquet_source_chunks_with_derived_policy, exec::retry::RetryLimits,
    rapids_backend::rapids_analysis_context, source_chunk::SourceMetadataCache,
};

pub(in crate::exec::bounded) fn plan_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::join_leaf_stream(has_source_adjacent_stages),
        None,
        metrics,
    )
}

pub(in crate::exec::bounded) fn plan_bounded_aggregate_input_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    chunk_cap_hints: AggregateInputHints,
    concurrent_chunks: usize,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::aggregate_pipeline(has_source_adjacent_stages).with_concurrent_chunks(concurrent_chunks),
        Some(chunk_cap_hints),
        metrics,
    )
}

pub(in crate::exec::bounded) fn plan_bounded_shared_join_aggregate_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    chunk_cap_hints: AggregateInputHints,
    concurrent_chunks: usize,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::shared_join_aggregate_stream(has_source_adjacent_stages)
            .with_concurrent_chunks(concurrent_chunks),
        Some(chunk_cap_hints),
        metrics,
    )
}

pub(in crate::exec::bounded) fn plan_bounded_unary_chain_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::unary_chain_frame(),
        None,
        metrics,
    )
}

fn plan_bounded_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    memory_model: SourceTaskMemoryModel,
    chunk_cap_hints: Option<AggregateInputHints>,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    validate_local_parquet_source(source, rapids_analysis_context(context.planning_policy()))?;
    let (mut chunk_plan, planning_policy) = plan_local_parquet_source_chunks_with_derived_policy(
        node_id,
        source,
        |file_metadata| {
            let policy =
                local_parquet_source_stream_chunk_planning_policy(context, source, file_metadata, memory_model);
            match chunk_cap_hints {
                Some(chunk_cap_hints) => apply_source_adjacent_aggregate_chunk_caps(policy, chunk_cap_hints),
                None => policy,
            }
        },
        &mut SourceMetadataCache::default(),
        Some(context),
    )?;
    annotate_join_leaf_source_chunk_batching_reasons(
        source,
        &mut chunk_plan.chunks,
        planning_policy.max_source_chunk_bytes,
    );
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    let RetryLimits { max_attempts, min_split_row_groups: min_retry_chunk_row_groups } = RetryLimits::of(context);
    Ok(BoundedJoinLeafLocalParquetSourceStream {
        source: source.clone(),
        node_id,
        chunks: chunk_plan.chunks,
        stages: Vec::new(),
        reader_dynamic_filters: Vec::new(),
        planning_policy,
        max_attempts,
        min_retry_chunk_row_groups,
    })
}

fn local_parquet_source_stream_chunk_planning_policy(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file_metadata: &[Option<RowGroupMetadata>],
    memory_model: SourceTaskMemoryModel,
) -> SourceChunkPlanningPolicy {
    SourceChunkPlanningPolicy::for_source_task_with_mode_and_options(
        source,
        file_metadata,
        context.memory_control_policy(),
        memory_model,
        context.source_chunk_planning_mode(),
        context.source_chunk_planning_options(),
    )
}
