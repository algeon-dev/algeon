use std::time::Duration;

use super::{
    AggregateExecutionStage, BoundedExecutionAttempt, BoundedPlan, CommittedPartial,
    CompiledIncrementalPartitionedMerge, ExecutionContext, ExecutionMetrics, Instant, PartitionedFinalMergeTopKStats,
    Result, aggregate_strategy_capability_decision, apply_partitioned_final_merge_topk,
    concatenate_partitioned_final_merge_outputs, execute_compiled_bounded_aggregate_from_empty_input,
    execute_final_aggregate_projection, observe_device_memory_bytes, partitioned_final_merge_topk_candidate,
    record_grouped_aggregate_strategy_decision,
};

pub(in crate::exec::bounded::aggregate) fn execute_incremental_source_adjacent_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    merge: CompiledIncrementalPartitionedMerge<'_>,
    aligned_key_columns: Option<&[String]>,
) -> Result<CommittedPartial> {
    let states = merge.finish_partition_states(execution)?;
    if states.partitions.is_empty() {
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, device_bytes: frame.device_bytes });
    }
    let topk = partitioned_final_merge_topk_candidate(bounded)?;
    let mut topk_stats = PartitionedFinalMergeTopKStats::default();
    let timer = Instant::now();
    let non_empty_partition_count = states.partitions.len();
    let mut merged_partitions = Vec::with_capacity(non_empty_partition_count);
    for (partition_index, state) in states.partitions {
        let frame = match apply_partitioned_final_merge_topk(
            execution,
            bounded,
            partition_index,
            state.frame,
            topk.as_ref(),
            &mut topk_stats,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                return Err(error);
            },
        };
        let bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        merged_partitions.push(CommittedPartial::with_device_bytes(frame, bytes));
    }
    execution.metrics().record_final_merge_attempt(states.max_partition_fan_in.max(1));
    execution.metrics().record_final_merge_level();
    let merged = concatenate_partitioned_final_merge_outputs(execution, bounded.aggregate_id, merged_partitions)?;
    let output_rows = merged.frame.num_rows() as u64;
    let mut extra_facts = vec![
        ("partitioned_merge_incremental", "true".to_string()),
        ("partitioned_merge_incremental_partials", states.partial_count.to_string()),
        ("partitioned_merge_incremental_flushes", states.flush_count.to_string()),
        ("partitioned_merge_incremental_max_pending_partials", states.max_pending_partials.to_string()),
    ];
    if let Some(key_columns) = aligned_key_columns {
        extra_facts.extend([
            ("partitioned_merge_aligned_input", "group_key_source_adjacent".to_string()),
            ("partitioned_merge_rehash_skipped", "true".to_string()),
            ("partitioned_merge_key_columns", key_columns.join(",")),
        ]);
    }
    record_partitioned_final_merge_strategy(
        execution.context(),
        bounded,
        states.input_rows,
        output_rows,
        states.partition_count,
        non_empty_partition_count,
        states.max_partition_fan_in,
        &topk_stats,
        states.merge_time.saturating_add(timer.elapsed()),
        extra_facts,
        execution.metrics(),
    );
    execute_final_aggregate_projection(execution, bounded, merged)
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_partitioned_final_merge_strategy(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input_rows: u64,
    output_rows: u64,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
    topk_stats: &PartitionedFinalMergeTopKStats,
    elapsed: Duration,
    extra_facts: impl IntoIterator<Item = (&'static str, String)>,
    metrics: &ExecutionMetrics,
) {
    let mut decision = aggregate_strategy_capability_decision(
        &bounded.aggregate.0.spec,
        bounded.aggregate_input_schema.as_ref(),
        AggregateExecutionStage::FinalMerge,
    );
    decision.facts.insert("partitioned_merge_partition_count", partition_count.to_string());
    decision.facts.insert("partitioned_merge_non_empty_partition_count", non_empty_partition_count.to_string());
    decision.facts.insert("partitioned_merge_max_partition_fan_in", max_partition_fan_in.to_string());
    decision.facts.insert("partitioned_merge_topk_pruned", topk_stats.enabled.to_string());
    decision.facts.insert(
        "partitioned_merge_topk_keep_rows",
        if topk_stats.enabled { topk_stats.keep_rows.to_string() } else { "none".to_string() },
    );
    decision.facts.insert("partitioned_merge_topk_partition_count", topk_stats.partition_count.to_string());
    decision.facts.insert("partitioned_merge_topk_input_rows", topk_stats.input_rows.to_string());
    decision.facts.insert("partitioned_merge_topk_output_rows", topk_stats.output_rows.to_string());
    decision.facts.extend(extra_facts);
    record_grouped_aggregate_strategy_decision(
        metrics,
        Some(bounded.aggregate_id.to_string()),
        AggregateExecutionStage::FinalMerge,
        context.aggregate_strategy_execution_options(),
        &bounded.aggregate.0.spec,
        bounded.aggregate_input_schema,
        input_rows,
        output_rows,
        decision,
        Duration::ZERO,
        Duration::ZERO,
        elapsed,
        Duration::ZERO,
        elapsed,
        None,
    );
}
