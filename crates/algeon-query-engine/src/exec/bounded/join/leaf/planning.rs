use super::{
    BoundedJoinLeafLocalParquetSourceStream, ExecutionContext, ExecutionMetrics, LocalParquetSource,
    MemoryControlPolicy, NodeId, Result, SourceAdjacentAggregateChunkCapHints, SourceChunkPlanningPolicy,
    SourceTaskMemoryModel, annotate_join_leaf_source_chunk_batching_reasons,
    apply_source_adjacent_aggregate_chunk_caps, plan_local_parquet_source_chunks_with_execution_context,
    record_bounded_parquet_source_metrics, validate_local_parquet_source,
};
use crate::rapids_backend::rapids_analysis_context;

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
    chunk_cap_hints: SourceAdjacentAggregateChunkCapHints,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::aggregate_pipeline(has_source_adjacent_stages),
        Some(chunk_cap_hints),
        metrics,
    )
}

pub(in crate::exec::bounded) fn plan_bounded_shared_join_aggregate_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    chunk_cap_hints: SourceAdjacentAggregateChunkCapHints,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    plan_bounded_local_parquet_source_stream(
        context,
        source,
        node_id,
        SourceTaskMemoryModel::shared_join_aggregate_stream(has_source_adjacent_stages),
        Some(chunk_cap_hints),
        metrics,
    )
}

fn plan_bounded_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    memory_model: SourceTaskMemoryModel,
    chunk_cap_hints: Option<SourceAdjacentAggregateChunkCapHints>,
    metrics: &ExecutionMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    validate_local_parquet_source(source, rapids_analysis_context())?;
    let mut planning_policy = local_parquet_source_stream_chunk_planning_policy(context, source, memory_model);
    if let Some(chunk_cap_hints) = chunk_cap_hints {
        planning_policy = apply_source_adjacent_aggregate_chunk_caps(planning_policy, chunk_cap_hints);
    }
    let mut chunk_plan =
        plan_local_parquet_source_chunks_with_execution_context(node_id, source, planning_policy, context)?;
    annotate_join_leaf_source_chunk_batching_reasons(
        source,
        &mut chunk_plan.chunks,
        planning_policy.max_source_chunk_bytes,
    );
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);
    let (max_attempts, min_retry_chunk_row_groups) = join_source_retry_policy(context);
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

pub(super) fn local_parquet_source_stream_chunk_planning_policy(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    memory_model: SourceTaskMemoryModel,
) -> SourceChunkPlanningPolicy {
    SourceChunkPlanningPolicy::for_source_task_with_mode_and_options(
        source,
        context.memory_control_policy(),
        memory_model,
        context.source_chunk_planning_mode(),
        context.source_chunk_planning_options(),
    )
}

pub(super) fn join_source_retry_policy(context: &ExecutionContext) -> (usize, usize) {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } => {
            (max_retry_attempts.max(1), min_retry_chunk_row_groups.max(1))
        },
        MemoryControlPolicy::Unbounded => (1, 1),
    }
}
