use super::*;

pub(in crate::exec::bounded) fn plan_bounded_join_leaf_local_parquet_source_stream(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    has_source_adjacent_stages: bool,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinLeafLocalParquetSourceStream> {
    validate_local_parquet_source(source)?;
    let planning_policy = join_source_chunk_planning_policy(context, source, has_source_adjacent_stages);
    let mut chunk_plan = plan_local_parquet_source_chunks(node_id, source, planning_policy)?;
    annotate_join_leaf_source_chunk_batching_reasons(
        source,
        &mut chunk_plan.chunks,
        planning_policy.max_source_chunk_bytes,
    );
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.join.source",
            "bounded join leaf local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
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

pub(super) fn join_source_chunk_planning_policy(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    has_source_adjacent_stages: bool,
) -> SourceChunkPlanningPolicy {
    SourceChunkPlanningPolicy::for_source_task_with_mode(
        source,
        context.memory_control_policy(),
        SourceTaskMemoryModel::join_leaf_stream(has_source_adjacent_stages),
        context.source_chunk_planning_mode(),
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
