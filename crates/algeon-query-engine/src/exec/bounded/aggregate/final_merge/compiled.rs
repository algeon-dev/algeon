//! Compiled final merge over a retained set of partials, and the final-aggregate kernels shared
//! with the other merge paths.
//!
//! `execute_compiled_bounded_final_aggregate_merge` is the entry point. The merge concatenates
//! partial frames and runs the final aggregate over the concatenation: a groupby on the group keys
//! (or a global reduction) that sums COUNT partials, merges AVG sum and count, and sums the valid
//! counts of global SUM, MIN and MAX.
//!
//! Partition-wise, each partial is hash-partitioned on the group keys; a group's rows from all
//! partials then meet in one partition, and the partitions merge independently. Level-wise, groups
//! of consecutive partials are merged into one partial each, and levels repeat until one partial
//! remains. A level borrows its inputs, which lets a retryable resource error restart the level
//! with a smaller fan-in. The partitioned merge consumes each partial as it partitions it.

use std::{collections::VecDeque, time::Duration};

use query_planning::chunk::SourceChunkPlanningPolicy;

use super::{
    AggregateExecutionStage, AggregateExecutionStrategy, AggregateNode, Arc, ArrowSchemaRef, BoundedExecutionAttempt,
    BoundedFinalMergePolicy, CommittedPartial, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, GpuDataFrame,
    Instant, NodeId, QueryError, QueryErrorCode, ResidentGpuFrame, Result, aggregate_spec_has_avg,
    aggregate_spec_has_global_final_state, aggregate_strategy_capability_decision, attach_budget_token_failure_facts,
    cast_count_output_columns_with_budget, concatenate_partitioned_final_merge_outputs, cudf_query_error,
    execute_aggregate_with_budget, execute_compiled_bounded_aggregate_from_empty_input,
    execute_global_aggregate_with_budget, execute_groupby_aggregate_with_budget_observed,
    execution_budget_token_for_boundary, final_global_aggregate_spec, final_global_aggregate_state_output_schema,
    finalize_avg_output_columns_with_budget, lower_final_aggregate_spec, observe_device_memory_bytes,
    partition_final_merge_partial, partitioned_final_merge_partition_count,
    record_default_grouped_aggregate_strategy_decision, record_grouped_aggregate_strategy_decision,
    reduce_merge_fan_in, validate_aggregate_output_names,
};
use crate::exec::retry::{
    RetryLimits, backoff_before_retry, is_retryable_resource_error, record_retryable_resource_error,
};

/// Merges all partials of one aggregate into its final output frame.
///
/// An empty input runs the single-stage aggregate so the grand-total row is still emitted; a
/// single partial is only projected; a partitioned merge is used when the capability decision
/// and the partial sizes call for it; otherwise consecutive partials are merged in groups of
/// `merge_fan_in`, level by level, and a level's merged partials replace it only after the whole
/// level succeeds.
///
/// A level that fails with a retryable resource error restarts from the level's original partials
/// with a smaller fan-in, until the attempt limit or the minimum fan-in. The partials of a level
/// are replaced only after the whole level succeeded; a failed attempt loses nothing. Any other
/// error returns at once, and the partials still held drop with the call.
pub(in crate::exec::bounded) fn execute_compiled_bounded_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    mut partials: Vec<CommittedPartial>,
) -> Result<ResidentGpuFrame> {
    // Empty grouping-set / ROLLUP partials still need the final grand-total row.
    // Treat all-empty partials like a true empty final-merge input so synthesis
    // runs on the single-stage path with the original aggregate input schema.
    let all_empty_grouping_sets = aggregate.spec.grouping_sets.is_some()
        && (partials.is_empty() || partials.iter().all(|partial| partial.frame.num_rows() == 0));
    if all_empty_grouping_sets {
        // The empty-input aggregate allocates below, so the partials cannot wait for end of scope.
        drop(partials);
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }
    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
        );
    }
    if partials.len() == 1 {
        let partial = partials.pop().expect("single partial checked");
        return execute_compiled_final_aggregate_projection(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partial,
        );
    }

    let RetryLimits { max_attempts, .. } = RetryLimits::of(execution.context());

    if should_execute_compiled_partitioned_final_aggregate_merge(
        execution.context(),
        aggregate,
        input_schema,
        &partials,
    ) {
        let partial = execute_compiled_partitioned_final_aggregate_merge(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partials,
        )?;
        return execute_compiled_final_aggregate_projection(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partial,
        );
    }

    while partials.len() > 1 {
        let mut merge_fan_in = initial_compiled_final_merge_fan_in(execution.context(), &partials);
        let mut attempt = 1;

        loop {
            match execute_compiled_final_merge_level_attempt(
                execution,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &partials,
                merge_fan_in,
            ) {
                Ok(level_output) => {
                    execution.metrics().record_final_merge_level();
                    partials = commit_compiled_final_merge_level(partials, level_output);
                    break;
                },
                Err(error) if is_retryable_compiled_final_merge_error(&error) => {
                    record_compiled_final_merge_retry_error(&error, execution.metrics());
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        backoff_before_retry(
                            execution.context(),
                            "query_engine.exec.bounded.compiled_final_merge_fan_in_retry",
                        )?;
                        continue;
                    }

                    return Err(error);
                },
                Err(error) => {
                    return Err(error);
                },
            }
        }
    }

    let partial = partials.pop().expect("compiled final merge leaves exactly one partial");
    execute_compiled_final_aggregate_projection(execution, node_id, aggregate, input_schema, output_schema, partial)
}

fn should_execute_compiled_partitioned_final_aggregate_merge(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
) -> bool {
    partials.len() > 1
        && aggregate_strategy_capability_decision(
            &aggregate.spec,
            input_schema.as_ref(),
            AggregateExecutionStage::FinalMerge,
        )
        .selected_strategy
            == AggregateExecutionStrategy::PartitionedMerge
        && partitioned_final_merge_partition_count(context, partials) > 1
}

/// Hash-partitions each partial on the final group keys, one partial at a time, merges every
/// non-empty partition's slices and concatenates the merged partitions in partition order.
///
/// Because a group's rows share a partition, partitions merge independently. Each partial is
/// consumed by `partition_final_merge_partial` as it is partitioned. The call records one
/// final-merge attempt whose fan-in is the largest partition's input count, and one strategy
/// decision with the partition counts.
fn execute_compiled_partitioned_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    let partition_count = partitioned_final_merge_partition_count(execution.context(), &partials);
    let final_aggregate = lower_final_aggregate_spec(&aggregate.spec)?;
    let key_columns = final_aggregate.key_columns;
    let input_rows = partials.iter().map(|partial| partial.frame.num_rows() as u64).sum::<u64>();
    let mut pending = partials;
    let mut partition_inputs = (0..partition_count).map(|_| Vec::new()).collect::<Vec<Vec<CommittedPartial>>>();

    let mut partial_index = 0usize;
    while !pending.is_empty() {
        let partial = pending.remove(0);
        let partitions = match partition_final_merge_partial(
            execution,
            node_id,
            partial_index,
            partial,
            &key_columns,
            partition_count,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                return Err(error);
            },
        };
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if partition.frame.num_rows() > 0 {
                partition_inputs[partition_index].push(partition);
            }
        }
        partial_index += 1;
    }

    let timer = Instant::now();
    let mut merged_partitions = Vec::with_capacity(partition_count);
    let mut non_empty_partition_count = 0usize;
    let mut max_partition_fan_in = 0usize;
    let mut pending_partition_inputs = partition_inputs.into_iter().enumerate().collect::<VecDeque<_>>();
    while let Some((partition_index, inputs)) = pending_partition_inputs.pop_front() {
        if inputs.is_empty() {
            continue;
        }
        non_empty_partition_count += 1;
        max_partition_fan_in = max_partition_fan_in.max(inputs.len());
        match execute_compiled_partitioned_final_merge_group(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_index,
            inputs,
        ) {
            Ok(merged) => merged_partitions.push(merged),
            Err(error) => {
                return Err(error);
            },
        }
    }
    if merged_partitions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.compiled_final_merge.partitioned",
            "compiled partitioned final aggregate merge produced no non-empty partitions",
        )
        .with_fact("node_id", node_id)
        .with_fact("partition_count", partition_count));
    }
    execution.metrics().record_final_merge_attempt(max_partition_fan_in.max(1));
    execution.metrics().record_final_merge_level();
    let merged = concatenate_partitioned_final_merge_outputs(execution, node_id, merged_partitions)?;
    let elapsed = timer.elapsed();
    let output_rows = merged.frame.num_rows() as u64;
    record_compiled_partitioned_final_merge_strategy(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        input_rows,
        output_rows,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
        elapsed,
        [],
        execution.metrics(),
    );
    Ok(merged)
}

/// Merges one partition's slices under a budget token for
/// `compiled_partitioned_final_aggregate_merge`, consuming them. The elapsed time is added to the
/// final-merge time whether the merge succeeds or fails.
fn execute_compiled_partitioned_final_merge_group(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partition_index: usize,
    inputs: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "compiled_partitioned_final_aggregate_merge",
        "query_engine.final_merge.compiled_partitioned_merge.budget_token",
        "failed to create compiled partitioned final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };
    let timer = Instant::now();
    let result = merge_compiled_partitioned_final_aggregate_partials_consuming(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partition_index,
        inputs,
        &token,
    );
    let merge_elapsed = timer.elapsed();
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.metrics().add_final_merge_time(merge_elapsed);
            return Err(error);
        },
    };
    execution.metrics().add_final_merge_time(merge_elapsed);

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}

/// Merges borrowed partition-aligned partials into one merged partial state.
///
/// Inputs stay owned by the caller so a rejected merge leaves them intact for
/// a smaller retry; the concatenation is the only transient copy.
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn merge_compiled_partitioned_final_aggregate_partial_refs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partition_index: usize,
    partials: &[&CommittedPartial],
    token: &ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let concatenated = if partial_count == 1 {
        partials[0].frame.frame().clone()
    } else {
        let first_frame = partials[0].frame.frame().clone();
        let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
        let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.on(token.exec()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.compiled_final_merge.partitioned_concat_partials",
                    "failed to concatenate compiled partitioned final aggregate partials",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("partition", partition_index)
                .with_fact("partial_aggregate_count", partial_count),
                token,
            )
        })?
    };
    let concatenated = GpuDataFrame::new(concatenated);
    observe_device_memory_bytes(&concatenated, execution.metrics())?;
    execute_compiled_final_aggregate_state(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        output_schema,
        concatenated,
        token,
        execution.metrics(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
/// Consuming counterpart of the borrowed merge: the first partial's frame moves into the
/// concatenation. The remaining slices are dropped before the final aggregate allocates.
fn merge_compiled_partitioned_final_aggregate_partials_consuming(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partition_index: usize,
    partials: Vec<CommittedPartial>,
    token: &ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    let mut partials = partials;
    let partial_count = partials.len();
    let first = partials.remove(0);
    let first_frame = first.frame.into_frame();
    let concatenated = if partials.is_empty() {
        first_frame
    } else {
        let other_frames = partials.iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
        let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
        let concatenated = first_frame.on(token.exec()).concat_many(&other_frame_refs);
        match concatenated {
            Ok(frame) => frame,
            Err(error) => {
                return Err(attach_budget_token_failure_facts(
                    cudf_query_error(
                        "query_engine.compiled_final_merge.partitioned_concat_partials",
                        "failed to concatenate compiled partitioned final aggregate partials",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("partition", partition_index)
                    .with_fact("partial_aggregate_count", partial_count),
                    token,
                ));
            },
        }
    };
    let concatenated = GpuDataFrame::new(concatenated);
    match observe_device_memory_bytes(&concatenated, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };

    // The final aggregate allocates below, so the concatenated inputs cannot wait for end of scope.
    drop(partials);
    execute_compiled_final_aggregate(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        output_schema,
        concatenated,
        token,
        execution.metrics(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
/// Writes the final-merge strategy decision for a partitioned merge, with the partition count, the
/// non-empty partition count and the largest partition fan-in. The top-k facts are zeroed, because
/// this merge does no per-partition pruning; `extra_facts` adds caller-specific entries.
pub(super) fn record_compiled_partitioned_final_merge_strategy(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    input_rows: u64,
    output_rows: u64,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
    elapsed: Duration,
    extra_facts: impl IntoIterator<Item = (&'static str, String)>,
    metrics: &ExecutionMetrics,
) {
    let mut decision = aggregate_strategy_capability_decision(
        &aggregate.spec,
        input_schema.as_ref(),
        AggregateExecutionStage::FinalMerge,
    );
    decision.facts.insert("partitioned_merge_partition_count", partition_count.to_string());
    decision.facts.insert("partitioned_merge_non_empty_partition_count", non_empty_partition_count.to_string());
    decision.facts.insert("partitioned_merge_max_partition_fan_in", max_partition_fan_in.to_string());
    decision.facts.insert("partitioned_merge_topk_pruned", "false".to_string());
    decision.facts.insert("partitioned_merge_topk_keep_rows", "none".to_string());
    decision.facts.insert("partitioned_merge_topk_partition_count", "0".to_string());
    decision.facts.insert("partitioned_merge_topk_input_rows", "0".to_string());
    decision.facts.insert("partitioned_merge_topk_output_rows", "0".to_string());
    decision.facts.extend(extra_facts);
    record_grouped_aggregate_strategy_decision(
        metrics,
        Some(node_id.to_string()),
        AggregateExecutionStage::FinalMerge,
        context.aggregate_strategy_execution_options(),
        &aggregate.spec,
        input_schema,
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

/// Fan-in of the first merge level: every partial when their sizes sum to no more than the source
/// chunk byte target, otherwise the longest prefix that fits it, and never fewer than two.
fn initial_compiled_final_merge_fan_in(context: &ExecutionContext, partials: &[CommittedPartial]) -> usize {
    let threshold_bytes = SourceChunkPlanningPolicy::from(context.memory_control_policy())
        .max_source_chunk_bytes
        .expect("bounded source planning has a finite byte target");
    let total_bytes = partials.iter().map(compiled_partial_merge_basis_bytes).sum::<u64>();
    if total_bytes <= threshold_bytes {
        return partials.len();
    }

    let policy = BoundedFinalMergePolicy::default();
    let mut fan_in = 0usize;
    let mut selected_bytes = 0u64;
    for partial in partials {
        let partial_bytes = compiled_partial_merge_basis_bytes(partial);
        if policy.should_stop_compiled_final_merge_fan_in_selection(
            fan_in,
            selected_bytes,
            partial_bytes,
            threshold_bytes,
        ) {
            break;
        }
        selected_bytes = selected_bytes.saturating_add(partial_bytes);
        fan_in += 1;
    }

    policy.compiled_final_merge_fan_in(fan_in, partials.len())
}

/// Size used to group partials: the observed device bytes, with a minimum of one byte per row.
fn compiled_partial_merge_basis_bytes(partial: &CommittedPartial) -> u64 {
    partial.device_bytes().max(partial.frame.num_rows() as u64)
}

/// Runs one attempt at a whole merge level with the given fan-in.
///
/// `partials` is borrowed. A group of one partial is carried by its index, with no copy; the
/// returned items are enough to build the next level (`commit_compiled_final_merge_level`). A
/// failed group ends the attempt and drops the groups merged so far.
fn execute_compiled_final_merge_level_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
) -> Result<Vec<CompiledMergeLevelItem>> {
    execution.metrics().record_final_merge_attempt(merge_fan_in);
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(CompiledMergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_compiled_final_merge_group_attempt(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            group,
        ) {
            Ok(partial) => output.push(CompiledMergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                return Err(error);
            },
        }
    }

    Ok(output)
}

/// Merges two or more partials into one: concatenates them and runs the final aggregate over the
/// concatenation, under a budget token for `final_aggregate_merge`.
///
/// A failure leaves the borrowed inputs intact for a retry. The folded accumulator in `stream.rs`
/// also calls it, always with two states. The elapsed time is added to the final-merge time whether
/// the merge succeeds or fails.
pub(in crate::exec::bounded::aggregate) fn execute_compiled_final_merge_group_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "final_aggregate_merge",
        "query_engine.bounded.aggregate.compiled_stream_final_merge.budget_token",
        "failed to create bounded native compiled stream final-aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };

    let timer = Instant::now();
    let result = merge_compiled_final_aggregate_partials(
        execution,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        &token,
    );
    let merge_elapsed = timer.elapsed();
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.metrics().add_final_merge_time(merge_elapsed);
            return Err(error);
        },
    };
    execution.metrics().add_final_merge_time(merge_elapsed);

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}

fn merge_compiled_final_aggregate_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let concatenated = first_frame.on(token.exec()).concat_many(&other_frame_refs);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.aggregate.compiled_stream_concat_partials",
            "failed to concatenate compiled stream aggregate partials",
            error,
        )
    })?;
    observe_device_memory_bytes(&concatenated, execution.metrics())?;

    execute_compiled_final_aggregate(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        output_schema,
        concatenated,
        token,
        execution.metrics(),
    )
}

/// One entry of a merge level's output: a newly merged partial, or the index of a partial from the
/// previous level that was alone in its group and stays as it is.
enum CompiledMergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

/// Builds the next level from a successful attempt. Carried partials move out of the old level, and
/// the partials that were merged drop when this call returns.
fn commit_compiled_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<CompiledMergeLevelItem>,
) -> Vec<CommittedPartial> {
    let mut partials = partials.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_partials = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            CompiledMergeLevelItem::Merged(partial) => next_partials.push(*partial),
            CompiledMergeLevelItem::Carry { partial_index } => {
                let partial = partials[partial_index].take().expect("carry index references a source partial");
                next_partials.push(partial);
            },
        }
    }

    next_partials
}

fn is_retryable_compiled_final_merge_error(error: &QueryError) -> bool {
    is_retryable_resource_error(error)
}

fn record_compiled_final_merge_retry_error(error: &QueryError, metrics: &ExecutionMetrics) {
    record_retryable_resource_error(error, metrics);
}

/// Normalizes a raw partial into merge state without finalizing AVG. Raw
/// partials and merged states must not share a concatenation when COUNT/AVG
/// have different intermediate column names or types.
pub(in crate::exec::bounded::aggregate) fn normalize_compiled_aggregate_partial(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    let token = execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "final_aggregate_merge",
        "query_engine.bounded.aggregate.partial_normalize",
        "failed to create aggregate normalization token",
    )?;
    let frame = execute_compiled_final_aggregate(
        execution.context(),
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        execution.metrics(),
    )?;
    let bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, bytes))
}

/// Produces the final output from one partial or one merged state.
///
/// A lone partial still goes through the final aggregate, which turns raw partial state into the
/// shape the projection expects, and then through `finalize_compiled_final_aggregate_output`. Empty
/// grouping sets take the single-stage aggregate instead. The two steps run under separate budget
/// tokens, `final_aggregate_merge` and `final_aggregate_projection`.
pub(super) fn execute_compiled_final_aggregate_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: CommittedPartial,
) -> Result<ResidentGpuFrame> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "final_aggregate_merge",
        "query_engine.bounded.aggregate.compiled_stream_final_projection.budget_token",
        "failed to create bounded native compiled stream final-projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };
    // Empty grouping-set finalization is a full single-stage shrink (including
    // AVG finalize). Skip the usual partial-state final-projection path, which would
    // look for missing AVG state aliases on an already-final frame.
    let empty_grouping_sets = aggregate.spec.grouping_sets.is_some() && input.frame.num_rows() == 0;
    let frame = if empty_grouping_sets {
        execute_aggregate_with_budget(
            execution.context(),
            aggregate,
            input_schema,
            input.frame,
            output_schema,
            &token,
            execution.metrics(),
        )
    } else {
        execute_compiled_final_aggregate(
            execution.context(),
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            &token,
            execution.metrics(),
        )
    };
    let frame = frame?;
    let frame = if empty_grouping_sets {
        frame
    } else {
        let projection_token = execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            "final_aggregate_projection",
            "query_engine.bounded.aggregate.compiled_stream_final_projection.budget_token",
            "failed to create bounded native compiled stream final-projection budget token",
        )?;
        finalize_compiled_final_aggregate_output(
            execution.context(),
            aggregate,
            output_schema,
            frame,
            &projection_token,
            execution.metrics(),
        )?
    };
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}

/// Divides AVG state into the output column when the spec has an AVG, then selects the output
/// schema's columns, which drops the remaining state columns. `post_stages.rs` performs the same
/// step for the source-adjacent merge.
fn finalize_compiled_final_aggregate_output(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&aggregate.spec) {
        finalize_avg_output_columns_with_budget(
            context,
            input.into_frame(),
            &aggregate.spec,
            output_schema,
            token,
            metrics,
        )
    } else {
        Ok(input.into_frame())
    };
    result
        .and_then(|frame| {
            frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
                cudf_query_error(
                    "query_engine.compiled_final_projection.select",
                    "failed to select bounded native compiled final aggregate output columns",
                    error,
                )
            })
        })
        .map(GpuDataFrame::new)
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate state merge preserves its operator occurrence and schema contract"
)]
/// Final aggregate over a concatenation of partials. For grouped aggregates it records the
/// final-merge strategy decision on every call.
fn execute_compiled_final_aggregate(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let bounded = CompiledAggregatePlan { node_id, aggregate, input_schema, output_schema };
    execute_final_aggregate_for_plan(context, &bounded, input, token, metrics, true)
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate state merge preserves its operator occurrence and schema contract"
)]
/// Same final aggregate without the strategy record. The incremental merge calls it for each of
/// its many small merges and writes one decision when it finishes.
fn execute_compiled_final_aggregate_state(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let bounded = CompiledAggregatePlan { node_id, aggregate, input_schema, output_schema };
    execute_final_aggregate_for_plan(context, &bounded, input, token, metrics, false)
}

struct CompiledAggregatePlan<'a> {
    node_id: NodeId,
    aggregate: &'a AggregateNode,
    input_schema: &'a ArrowSchemaRef,
    output_schema: &'a ArrowSchemaRef,
}

/// Runs the aggregate that turns concatenated partial state into merged state.
///
/// A global aggregate uses the final global spec and, when it has global final state, keeps the
/// state schema. A grouped aggregate runs a groupby on the final spec's keys, restores COUNT output
/// types, and selects the output names. Empty grouping sets fall back to the single-stage aggregate
/// so the grand-total identity row exists. `record_strategy` writes the default grouped-aggregate
/// decision for the final-merge stage.
fn execute_final_aggregate_for_plan(
    context: &ExecutionContext,
    bounded: &CompiledAggregatePlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
    record_strategy: bool,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.spec);
        let output_schema = if aggregate_spec_has_global_final_state(&bounded.aggregate.spec) {
            final_global_aggregate_state_output_schema(
                &bounded.aggregate.spec,
                bounded.input_schema,
                bounded.output_schema,
            )?
        } else {
            Arc::clone(bounded.output_schema)
        };
        return execute_global_aggregate_with_budget(context, &final_spec, input, &output_schema, token, metrics);
    }

    // Empty ROLLUP/CUBE/GROUPING SETS still need the grand-total identity row.
    // Regular final-merge groupby over an empty partial drops it. Route through
    // the full single-stage aggregate path, which synthesizes identity rows.
    if bounded.aggregate.spec.grouping_sets.is_some() && input.num_rows() == 0 {
        return execute_aggregate_with_budget(
            context,
            bounded.aggregate,
            bounded.input_schema,
            input,
            bounded.output_schema,
            token,
            metrics,
        );
    }

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.spec) {
        validate_aggregate_output_names(
            &aggregate.output_names,
            bounded.output_schema,
            "query_engine.final_merge.validate",
            "final aggregate output names must match output schema fields",
        )?;
    }
    let expected_output_names = if aggregate_spec_has_avg(&bounded.aggregate.spec) {
        aggregate.output_names.clone()
    } else {
        bounded.output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>()
    };

    let frame = input.into_frame();
    let input_rows = frame.num_rows() as u64;
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.spec, bounded.output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        cudf_query_error(
            "query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    if record_strategy {
        record_default_grouped_aggregate_strategy_decision(
            metrics,
            Some(bounded.node_id.to_string()),
            AggregateExecutionStage::FinalMerge,
            context.aggregate_strategy_execution_options(),
            &bounded.aggregate.spec,
            bounded.input_schema,
            input_rows,
            output_rows,
            elapsed,
        );
    }
    Ok(GpuDataFrame::new(output))
}
