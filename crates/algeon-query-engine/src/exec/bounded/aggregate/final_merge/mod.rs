//! Final merge of aggregate partials: regrouping the partial states of all chunks into the
//! aggregate's result. The partial-to-final contract is described in `aggregate.rs`.
//!
//! - `compiled.rs` merges a retained set of partials. A lone partial only needs its projection.
//!   When the aggregate capability decision selects the partitioned merge and the partials
//!   justify more than one partition, every partial is hash-partitioned on the group keys and
//!   each partition is merged on its own. Otherwise groups of `merge_fan_in` consecutive partials
//!   merge level by level.
//! - `incremental.rs` merges while partials arrive, into per-partition states; retained bytes
//!   follow the merged state.
//! - `partitioned.rs` finishes an incremental merge for source-adjacent aggregates: per-partition
//!   top-k pruning, concatenation, then the final projection.
//!
//! This file holds the pieces they share: the partition count for a partial set, hash-partitioning
//! a partial by group key, the top-k candidate analysis, and the concatenation of merged
//! partitions.

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use query_planning::{
    capability::{AggregateExecutionStage, AggregateExecutionStrategy, aggregate_strategy_capability_decision},
    expr::{SortExpr, SortSpec},
};
use query_runtime::observability::{
    RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
};

use super::{final_merge_policy::BoundedFinalMergePolicy, *};

mod compiled;
mod incremental;
mod partitioned;

pub(in crate::exec::bounded) use compiled::execute_compiled_bounded_final_aggregate_merge;
pub(in crate::exec::bounded::aggregate) use compiled::{
    execute_compiled_final_merge_group_attempt, normalize_compiled_aggregate_partial,
};
pub(in crate::exec::bounded::aggregate) use incremental::{
    CompiledIncrementalPartitionedMerge, should_execute_incremental_partitioned_final_aggregate_merge,
};
pub(in crate::exec::bounded::aggregate) use partitioned::execute_incremental_source_adjacent_final_aggregate_merge;

/// A sort with a limit that the partitioned merge can apply to each merged partition before the
/// partitions are concatenated. Each group lives in one partition; a row outside its partition's
/// top `keep_rows` cannot belong to the global top rows.
pub(super) struct PartitionedFinalMergeTopK<'a> {
    sort: SortNode,
    limit: &'a LimitNode,
    keep_rows: usize,
}

/// Totals reported in the `partitioned_merge_topk_*` strategy facts.
#[derive(Default)]
pub(super) struct PartitionedFinalMergeTopKStats {
    enabled: bool,
    keep_rows: usize,
    input_rows: u64,
    output_rows: u64,
    partition_count: usize,
}

/// Partition count for merging a set of partials: from their total device bytes against the
/// aggregate partition byte target, and never below the minimum that the final-merge policy derives
/// from the number of partials.
pub(super) fn partitioned_final_merge_partition_count(
    context: &ExecutionContext,
    partials: &[CommittedPartial],
) -> usize {
    let input_bytes = partials.iter().map(|partial| partial.device_bytes()).sum::<u64>().max(1);
    let partial_partition_floor =
        BoundedFinalMergePolicy::default().partitioned_final_merge_partial_partition_floor(partials.len());
    partitioned_aggregate_partition_count(context, input_bytes).max(partial_partition_floor)
}

/// Hash-partitions one partial on the group-key columns and slices it into one partial per
/// partition. Consumes `partial`; the partitioned copy and the slices exist beside each other while
/// the slices are made. The hash partition runs under a budget token for
/// `final_aggregate_hash_partition` and records a hash-partition stage.
pub(super) fn partition_final_merge_partial(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partial_index: usize,
    partial: CommittedPartial,
    key_columns: &[String],
    partition_count: usize,
) -> Result<Vec<CommittedPartial>> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "final_aggregate_hash_partition",
        "query_engine.final_merge.partitioned_hash.budget_token",
        "failed to create partitioned final aggregate hash budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };

    let frame = partial.frame.into_frame();
    let input_rows = frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::HashPartition,
        RuntimeOperatorStageIdentity::new().with_node_id(node_id.to_string()).with_input_rows(input_rows),
    );
    let partitioned = match frame
        .on(token.exec())
        .hash_partition_by(key_columns.iter().map(String::as_str), partition_count)
        .map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.final_merge.hash_partition",
                    "failed to hash-partition final aggregate partial output",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("key_count", key_columns.len())
                .with_fact("partition_count", partition_count),
                &token,
            )
        }) {
        Ok(partitioned) => {
            stage_guard.finish(Some(input_rows), None);
            Ok(partitioned)
        },
        Err(error) => {
            stage_guard.fail(&error);
            Err(error)
        },
    };
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            return Err(error);
        },
    };
    // Hash partition owns its output; release the consumed concatenation
    // before partition slices allocate their independently owned storage.
    drop(frame);
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    record_device_memory_observation(&partitioned_gpu, execution.metrics())?;

    let mut partitions = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        match slice_partitioned_final_merge_partition(
            execution,
            node_id,
            partial_index,
            partition,
            &partitioned_frame,
            &offsets,
        ) {
            Ok(frame) => partitions.push(frame),
            Err(error) => {
                return Err(error);
            },
        }
    }
    Ok(partitions)
}

pub(super) fn slice_partitioned_final_merge_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partial_index: usize,
    partition: usize,
    frame: &cudf::DataFrame,
    offsets: &[usize],
) -> Result<CommittedPartial> {
    let start = *offsets.get(partition).ok_or_else(|| {
        bounded_invariant_violation(
            "partitioned final merge is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        bounded_invariant_violation(
            "partitioned final merge is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let sliced = frame.on(overrides).slice(start, len).map_err(|error| {
        cudf_query_error(
            "query_engine.final_merge.hash_partition_slice",
            "failed to slice partitioned final aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partial_index", partial_index)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(gpu, bytes))
}

/// Returns the top-k that the partitioned merge may apply per partition, or `None`.
///
/// It needs a sort and a limit without an offset, no global final state (the sort columns would not
/// be final yet), and a sort that maps through the post-aggregate projections onto aggregate output
/// columns.
pub(super) fn partitioned_final_merge_topk_candidate<'a>(
    bounded: &'a BoundedPlan<'a>,
) -> Result<Option<PartitionedFinalMergeTopK<'a>>> {
    let Some(sort) = bounded.sort else {
        return Ok(None);
    };
    let Some((_, limit)) = bounded.limit else {
        return Ok(None);
    };
    if limit.spec.skip != 0 || aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
        return Ok(None);
    }
    let Some(sort) = partitioned_final_merge_topk_sort(bounded, sort)? else {
        return Ok(None);
    };
    let keep_rows = topk_candidate_rows(limit.spec)?;
    if keep_rows == 0 {
        return Ok(None);
    }
    Ok(Some(PartitionedFinalMergeTopK { sort, limit, keep_rows }))
}

pub(super) fn partitioned_final_merge_topk_sort(
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
) -> Result<Option<SortNode>> {
    let mut expressions = Vec::with_capacity(sort.spec.expressions.len());
    for sort_expr in &sort.spec.expressions {
        let mut column = lower_column_expr(&sort_expr.expr, "query_engine.final_merge.topk_sort")?;
        for stage in bounded.post_aggregate_stages.iter().rev() {
            let Some(mapped) = project_topk_sort_column_through_stage(stage, column.as_str())? else {
                return Ok(None);
            };
            column = mapped;
        }
        if bounded.aggregate.1.field_with_name(column.as_str()).is_err() {
            return Ok(None);
        }
        expressions.push(SortExpr::new(Expr::column(column), sort_expr.descending, sort_expr.nulls_first));
    }
    Ok(Some(SortNode { input: sort.input, spec: SortSpec::new(expressions) }))
}

pub(super) fn project_topk_sort_column_through_stage(
    stage: &BoundedUnaryStage<'_>,
    column: &str,
) -> Result<Option<String>> {
    match stage {
        BoundedUnaryStage::Projection { node } => {
            let Some(output) = node.outputs.iter().find(|output| output.field.name() == column) else {
                return Ok(None);
            };
            lower_column_expr(&output.expr, "query_engine.final_merge.topk_projection").map(Some)
        },
    }
}

/// Keeps at most the candidate rows of one merged partition, using the final sort with a fetch.
/// Without a candidate the frame is returned unchanged. The totals accumulate in `topk_stats`.
pub(super) fn apply_partitioned_final_merge_topk(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    frame: GpuDataFrame,
    topk: Option<&PartitionedFinalMergeTopK<'_>>,
    topk_stats: &mut PartitionedFinalMergeTopKStats,
) -> Result<GpuDataFrame> {
    let Some(topk) = topk else {
        return Ok(frame);
    };
    topk_stats.enabled = true;
    topk_stats.keep_rows = topk.keep_rows;
    topk_stats.partition_count += 1;
    topk_stats.input_rows = topk_stats.input_rows.saturating_add(frame.num_rows() as u64);

    let input_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    let input = ResidentGpuFrame::with_device_bytes(frame, input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::PartitionLocalTopK,
        RuntimeOperatorStageIdentity::new()
            .with_node_id(bounded.aggregate_id.to_string())
            .with_partition_index(partition_index)
            .with_input_rows(topk_stats.input_rows),
    );
    let output = match execute_final_sort_node_with_retry(
        execution,
        bounded.aggregate_id,
        &topk.sort,
        Some(topk.limit),
        input,
        None,
    )
    .map_err(|error| error.with_fact_if_absent("partitioned_final_merge_topk_partition", partition_index))
    {
        Ok(output) => {
            stage_guard.finish(Some(output.frame.num_rows() as u64), None);
            output
        },
        Err(error) => {
            stage_guard.fail(&error);
            return Err(error);
        },
    };
    topk_stats.output_rows = topk_stats.output_rows.saturating_add(output.frame.num_rows() as u64);
    let output_frame = output.frame;
    Ok(output_frame)
}

/// Concatenates merged partitions in input order under a budget token for
/// `partitioned_final_aggregate_output_concat`. One input is returned as it is. Consumes `inputs`.
pub(in crate::exec::bounded::aggregate) fn concatenate_partitioned_final_merge_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut inputs: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "partitioned_final_aggregate_output_concat",
        "query_engine.final_merge.partitioned_output_concat.budget_token",
        "failed to create partitioned final aggregate output concat budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };
    let first = inputs.remove(0);
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.on(token.exec()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.final_merge.partitioned_output_concat",
                    "failed to concatenate partitioned final aggregate outputs",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };
    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    Ok(ResidentGpuFrame::with_device_bytes(concatenated, output_bytes))
}

/// The smaller fan-in to retry with after a retryable resource error, or `None` once the policy
/// minimum is reached.
pub(in crate::exec::bounded) fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    BoundedFinalMergePolicy::default().reduce_merge_fan_in(merge_fan_in)
}
