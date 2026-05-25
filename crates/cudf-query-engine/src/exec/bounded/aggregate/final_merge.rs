use std::time::Duration;

use super::*;
use crate::{
    SortExpr, SortSpec,
    capability::{AggregateExecutionStage, AggregateExecutionStrategy, aggregate_strategy_capability_decision},
};

struct PartitionedFinalMergeTopK<'a> {
    sort: SortNode,
    limit: &'a LimitNode,
    keep_rows: usize,
}

#[derive(Default)]
struct PartitionedFinalMergeTopKStats {
    enabled: bool,
    keep_rows: usize,
    input_rows: u64,
    output_rows: u64,
    partition_count: usize,
}

pub(in crate::exec::bounded) fn execute_final_aggregate_merge(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let mut partials = partials;
    let mut merge_level = 1;

    if partials.is_empty() {
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            context,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            reservations,
            metrics,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, reservation: frame.reservation });
    }

    if should_execute_partitioned_final_aggregate_merge(context, bounded, &partials) {
        let partial = execute_partitioned_final_aggregate_merge(context, bounded, partials, reservations, metrics)?;
        return execute_final_aggregate_projection(context, bounded, partial, reservations, metrics);
    }

    while partials.len() > 1 {
        let mut merge_fan_in = partials.len();
        let mut attempt = 1;

        loop {
            match execute_final_merge_level_attempt(
                context,
                bounded,
                &partials,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    metrics.record_final_merge_level();
                    partials = commit_final_merge_level(partials, level_output, reservations, metrics);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_merge_error(&error) => {
                    record_final_merge_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    return Err(final_merge_retry_facts(merge_level, merge_fan_in, metrics).attach_to_error(error));
                },
                Err(error) => return Err(error),
            }
        }
    }

    let partial = partials.pop().expect("final merge loop leaves exactly one partial");
    execute_final_aggregate_projection(context, bounded, partial, reservations, metrics)
}
fn execute_final_merge_level_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<MergeLevelItem>> {
    metrics.record_final_merge_attempt(merge_fan_in);
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(MergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_final_merge_group_attempt(
            context,
            bounded,
            group,
            group_index,
            attempt,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(partial) => output.push(MergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                release_merge_level_items(output, reservations, metrics);
                return Err(error);
            },
        }
    }

    Ok(output)
}

fn execute_final_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    group_index: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let merge_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts)
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt));
    let mut merge_reservation = reservations.try_reserve(merge_request, metrics)?;
    let merge_token_budget = reservations.available_bytes_including(&merge_reservation);
    let merge_token = match execution_budget_token_for_boundary(
        context,
        merge_token_budget,
        "final_aggregate_merge",
        "cudf_query_engine.final_merge.budget_token",
        "failed to create bounded native final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(merge_reservation, metrics);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics,
            }
            .attach_to(error));
        },
    };

    let timer = Instant::now();
    let result = merge_final_aggregate_partials(
        context,
        bounded,
        partials,
        group_index,
        &merge_token,
        &mut merge_reservation,
        reservations,
        attempt,
        max_attempts,
        metrics,
    );
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics,
            }
            .attach_to(error));
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_merge_output",
            stage: "final_merge",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_merge",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts)
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt));
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}
fn merge_final_aggregate_partials(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    group_index: usize,
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    attempt: usize,
    max_attempts: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = token.execution_overrides();
    let concatenated = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "cudf_query_engine.final_merge.concat_partials",
                "failed to concatenate bounded native aggregate partials",
                error,
            )
            .with_fact("partial_aggregate_count", partial_count),
            token,
        )
    })?;
    let concatenated = GpuDataFrame::new(concatenated);
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("final_aggregate_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_merge")
        .with_attempt(attempt, max_attempts)
        .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt));
        reservations.try_grow(merge_reservation, grow_request, metrics)?;
    }

    execute_final_aggregate(context, bounded, concatenated, token, true, metrics)
}

fn should_execute_partitioned_final_aggregate_merge(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
) -> bool {
    partials.len() > 1
        && aggregate_strategy_capability_decision(
            &bounded.aggregate.0.spec,
            bounded.aggregate_input_schema.as_ref(),
            AggregateExecutionStage::FinalMerge,
        )
        .selected_strategy
            == AggregateExecutionStrategy::PartitionedMerge
        && partitioned_final_merge_partition_count(context, partials) > 1
}

fn partitioned_final_merge_partition_count(context: &ExecutionContext, partials: &[CommittedPartial]) -> usize {
    let input_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>().max(1);
    partitioned_aggregate_partition_count(context, input_bytes).max(partials.len().clamp(1, 32))
}

fn execute_partitioned_final_aggregate_merge(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let partition_count = partitioned_final_merge_partition_count(context, &partials);
    let final_aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    let key_columns = final_aggregate.key_columns;
    let input_rows = partials.iter().map(|partial| partial.frame.num_rows() as u64).sum::<u64>();
    let topk = partitioned_final_merge_topk_candidate(bounded)?;
    let mut topk_stats = PartitionedFinalMergeTopKStats::default();
    let mut pending = partials;
    let mut partition_inputs = (0..partition_count).map(|_| Vec::new()).collect::<Vec<Vec<CommittedPartial>>>();

    let mut partial_index = 0usize;
    while !pending.is_empty() {
        let partial = pending.remove(0);
        let partitions = match partition_final_merge_partial(
            context,
            bounded.aggregate_id,
            partial_index,
            partial,
            &key_columns,
            partition_count,
            reservations,
            metrics,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                release_nested_partials_reverse(reservations, partition_inputs, metrics);
                release_partials_reverse(reservations, pending, metrics);
                return Err(error);
            },
        };
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if partition.frame.num_rows() > 0 {
                partition_inputs[partition_index].push(partition);
            } else {
                reservations.release(partition.reservation, metrics);
            }
        }
        partial_index += 1;
    }

    let timer = Instant::now();
    let mut merged_partitions = Vec::with_capacity(partition_count);
    let mut non_empty_partition_count = 0usize;
    let mut max_partition_fan_in = 0usize;
    for (partition_index, inputs) in partition_inputs.into_iter().enumerate() {
        if inputs.is_empty() {
            continue;
        }
        non_empty_partition_count += 1;
        max_partition_fan_in = max_partition_fan_in.max(inputs.len());
        let merged = execute_partitioned_final_merge_group(
            context,
            bounded,
            partition_index,
            inputs,
            topk.as_ref(),
            &mut topk_stats,
            reservations,
            metrics,
        )?;
        merged_partitions.push(merged);
    }
    if merged_partitions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.final_merge.partitioned",
            "partitioned final aggregate merge produced no non-empty partitions",
        )
        .with_fact("node_id", bounded.aggregate_id)
        .with_fact("partition_count", partition_count));
    }
    metrics.record_final_merge_attempt(max_partition_fan_in.max(1));
    metrics.record_final_merge_level();
    let mut merged = concatenate_partitioned_final_merge_outputs(
        context,
        bounded.aggregate_id,
        merged_partitions,
        reservations,
        metrics,
    )?;
    let elapsed = timer.elapsed();
    let output_rows = merged.frame.num_rows() as u64;
    record_partitioned_final_merge_strategy(
        context,
        bounded,
        input_rows,
        output_rows,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
        &topk_stats,
        elapsed,
        metrics,
    );
    let merged_reservation_bytes = merged.reservation.bytes();
    merged.reservation = reservations.transfer(
        merged.reservation,
        MemoryReservationRequest::new(ReservationKind::FinalMerge, merged_reservation_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge")
            .with_compiled_task_id(partitioned_final_merge_task_id(bounded.aggregate_id, "combined_output")),
        metrics,
    );
    Ok(merged)
}

fn partition_final_merge_partial(
    context: &ExecutionContext,
    node_id: NodeId,
    partial_index: usize,
    partial: CommittedPartial,
    key_columns: &[String],
    partition_count: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let task_id = partitioned_final_merge_task_id(node_id, format!("hash_partition.partial{partial_index}"));
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, partial.reservation.bytes(), reservations.budget())
            .with_memory_boundary("final_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(partial.frame);
            reservations.release(partial.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&partition_reservation),
        "final_aggregate_hash_partition",
        "cudf_query_engine.final_merge.partitioned_hash.budget_token",
        "failed to create partitioned final aggregate hash budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            drop(partial.frame);
            reservations.release(partial.reservation, metrics);
            return Err(error);
        },
    };

    let frame = partial.frame.into_frame();
    let partitioned = frame
        .with_execution(&token.execution_overrides())
        .hash_partition_by(key_columns.iter().map(String::as_str), partition_count)
        .map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.final_merge.hash_partition",
                    "failed to hash-partition final aggregate partial output",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("key_count", key_columns.len())
                .with_fact("partition_count", partition_count),
                &token,
            )
        });
    drop(frame);
    reservations.release(partial.reservation, metrics);
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("final_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("partitioned_final_merge")
        .with_compiled_task_id(task_id);
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partitions = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        match slice_partitioned_final_merge_partition(
            context,
            node_id,
            partial_index,
            partition,
            &partitioned_frame,
            &offsets,
            reservations,
            metrics,
        ) {
            Ok(frame) => partitions.push(frame),
            Err(error) => {
                release_partials_reverse(reservations, partitions, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);
    Ok(partitions)
}

fn slice_partitioned_final_merge_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    partial_index: usize,
    partition: usize,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned final merge is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned final merge is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let overrides = execution_overrides(context);
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.hash_partition_slice",
            "failed to slice partitioned final aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partial_index", partial_index)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_compiled_task_id(partitioned_final_merge_task_id(
                node_id,
                format!("partition.partial{partial_index}.partition{partition}"),
            )),
        metrics,
    )?;
    Ok(CommittedPartial { frame: gpu, reservation })
}

fn execute_partitioned_final_merge_group(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    inputs: Vec<CommittedPartial>,
    topk: Option<&PartitionedFinalMergeTopK<'_>>,
    topk_stats: &mut PartitionedFinalMergeTopKStats,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let requested_bytes = inputs.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let task_id = partitioned_final_merge_task_id(bounded.aggregate_id, format!("merge.partition{partition_index}"));
    let mut merge_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&merge_reservation),
        "partitioned_final_aggregate_merge",
        "cudf_query_engine.final_merge.partitioned_merge.budget_token",
        "failed to create partitioned final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(merge_reservation, metrics);
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let result = merge_partitioned_final_aggregate_partials(
        context,
        bounded,
        partition_index,
        &inputs,
        &token,
        &mut merge_reservation,
        reservations,
        metrics,
    );
    drop(token);
    release_partials_reverse(reservations, inputs, metrics);
    reservations.release(merge_reservation, metrics);
    let frame = result?;
    let frame = apply_partitioned_final_merge_topk(
        context,
        bounded,
        partition_index,
        frame,
        topk,
        topk_stats,
        reservations,
        metrics,
    )?;
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id),
        metrics,
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn partitioned_final_merge_topk_candidate<'a>(
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

fn partitioned_final_merge_topk_sort(bounded: &BoundedPlan<'_>, sort: &SortNode) -> Result<Option<SortNode>> {
    if bounded.post_aggregate_stages.iter().any(|stage| matches!(stage, BoundedUnaryStage::Filter { .. })) {
        return Ok(None);
    }
    let mut expressions = Vec::with_capacity(sort.spec.expressions.len());
    for sort_expr in &sort.spec.expressions {
        let mut column = lower_column_expr(&sort_expr.expr, "cudf_query_engine.final_merge.topk_sort")?;
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

fn project_topk_sort_column_through_stage(stage: &BoundedUnaryStage<'_>, column: &str) -> Result<Option<String>> {
    match stage {
        BoundedUnaryStage::Filter { .. } => Ok(None),
        BoundedUnaryStage::Projection { node, .. } => {
            let Some(output) = node.outputs.iter().find(|output| output.field.name() == column) else {
                return Ok(None);
            };
            lower_column_expr(&output.expr, "cudf_query_engine.final_merge.topk_projection").map(Some)
        },
    }
}

fn apply_partitioned_final_merge_topk(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    frame: GpuDataFrame,
    topk: Option<&PartitionedFinalMergeTopK<'_>>,
    topk_stats: &mut PartitionedFinalMergeTopKStats,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let Some(topk) = topk else {
        return Ok(frame);
    };
    topk_stats.enabled = true;
    topk_stats.keep_rows = topk.keep_rows;
    topk_stats.partition_count += 1;
    topk_stats.input_rows = topk_stats.input_rows.saturating_add(frame.num_rows() as u64);

    let input_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let input_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalSort, input_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_topk_input")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge_topk")
            .with_compiled_task_id(partitioned_final_merge_task_id(
                bounded.aggregate_id,
                format!("topk_input.partition{partition_index}"),
            )),
        metrics,
    )?;
    let input = CommittedPartial { frame, reservation: input_reservation };
    let output = execute_final_sort_node_with_retry(
        context,
        bounded.aggregate_id,
        &topk.sort,
        Some(topk.limit),
        input,
        reservations,
        metrics,
    )
    .map_err(|error| error.with_fact_if_absent("partitioned_final_merge_topk_partition", partition_index))?;
    topk_stats.output_rows = topk_stats.output_rows.saturating_add(output.frame.num_rows() as u64);
    let output_frame = output.frame;
    reservations.release(output.reservation, metrics);
    Ok(output_frame)
}

fn merge_partitioned_final_aggregate_partials(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let concatenated = if partial_count == 1 {
        partials[0].frame.frame().clone()
    } else {
        let first_frame = partials[0].frame.frame().clone();
        let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
        let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.final_merge.partitioned_concat_partials",
                    "failed to concatenate partitioned final aggregate partials",
                    error,
                )
                .with_fact("node_id", bounded.aggregate_id)
                .with_fact("partition", partition_index)
                .with_fact("partial_aggregate_count", partial_count),
                token,
            )
        })?
    };
    let concatenated = GpuDataFrame::new(concatenated);
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("partitioned_final_aggregate_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("partitioned_final_merge")
        .with_attempt(1, 1)
        .with_compiled_task_id(partitioned_final_merge_task_id(
            bounded.aggregate_id,
            format!("merge.partition{partition_index}"),
        ));
        reservations.try_grow(merge_reservation, grow_request, metrics)?;
    } else {
        reservations.shrink_to_observed(merge_reservation, concatenated_bytes, metrics);
    }

    execute_final_aggregate(context, bounded, concatenated, token, false, metrics)
}

fn concatenate_partitioned_final_merge_outputs(
    context: &ExecutionContext,
    node_id: NodeId,
    mut inputs: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    let requested_bytes = inputs.iter().map(|input| input.reservation.bytes()).sum::<u64>();
    let task_id = partitioned_final_merge_task_id(node_id, "output_concat");
    let mut concat_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_output_concat")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&concat_reservation),
        "partitioned_final_aggregate_output_concat",
        "cudf_query_engine.final_merge.partitioned_output_concat.budget_token",
        "failed to create partitioned final aggregate output concat budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let first = inputs.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.final_merge.partitioned_output_concat",
                    "failed to concatenate partitioned final aggregate outputs",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };
    drop(first_frame);
    reservations.release(first_reservation, metrics);
    release_partials_reverse(reservations, inputs, metrics);
    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("partitioned_final_aggregate_output_concat")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_compiled_task_id(task_id);
    if output_bytes > concat_reservation.bytes() {
        let delta = output_bytes.saturating_sub(concat_reservation.bytes());
        reservations.try_grow(&mut concat_reservation, output_request.clone().with_requested_bytes(delta), metrics)?;
    } else {
        reservations.shrink_to_observed_with_request(
            &mut concat_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }
    let output_reservation = reservations.transfer(concat_reservation, output_request, metrics);
    Ok(CommittedPartial { frame: concatenated, reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
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
    metrics: &mut QueryMetrics,
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
    record_grouped_aggregate_strategy_decision(
        metrics,
        Some(bounded.aggregate_id.to_string()),
        AggregateExecutionStage::FinalMerge,
        context.memory_control_policy().as_str(),
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

fn partitioned_final_merge_task_id(node_id: NodeId, stage: impl std::fmt::Display) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.partitioned.{stage}")
}

fn legacy_bounded_final_merge_task_id(node_id: NodeId, group_index: usize, attempt: usize) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.group{group_index}.attempt{attempt}")
}

enum MergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

fn commit_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<MergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Vec<CommittedPartial> {
    let mut partials = partials.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_partials = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            MergeLevelItem::Merged(partial) => next_partials.push(*partial),
            MergeLevelItem::Carry { partial_index } => {
                let partial = partials[partial_index].take().expect("carry index references a source partial");
                next_partials.push(partial);
            },
        }
    }

    for partial in partials.into_iter().flatten() {
        reservations.release(partial.reservation, metrics);
    }

    next_partials
}

fn release_merge_level_items(
    items: Vec<MergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    for item in items {
        if let MergeLevelItem::Merged(partial) = item {
            let partial = *partial;
            reservations.release(partial.reservation, metrics);
        }
    }
}

fn is_retryable_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(in crate::exec::bounded) fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    if merge_fan_in <= 2 {
        return None;
    }

    Some(merge_fan_in.div_ceil(2).max(2))
}

fn final_merge_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if merge_fan_in <= 2 { "fail_min_merge_fan_in" } else { "retry_attempts_exhausted" };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_merge")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
        .with_merge_level(merge_level)
        .with_merge_fan_in(merge_fan_in)
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_final_aggregate_merge(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    mut partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if partials.is_empty() {
        return execute_compiled_bounded_aggregate_from_empty_input(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            reservations,
            metrics,
        );
    }
    if partials.len() == 1 {
        let partial = partials.pop().expect("single partial checked");
        return execute_compiled_final_aggregate_projection(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partial,
            reservations,
            metrics,
        );
    }

    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_compiled_bounded_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);

    while partials.len() > 1 {
        let mut merge_fan_in = initial_compiled_final_merge_fan_in(context, &partials);
        let mut attempt = 1;

        loop {
            match execute_compiled_final_merge_level_attempt(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                &partials,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    metrics.record_final_merge_level();
                    partials = commit_compiled_final_merge_level(partials, level_output, reservations, metrics);
                    break;
                },
                Err(error) if is_retryable_compiled_final_merge_error(&error) => {
                    record_compiled_final_merge_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
            }
        }
    }

    let partial = partials.pop().expect("compiled final merge leaves exactly one partial");
    execute_compiled_final_aggregate_projection(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partial,
        reservations,
        metrics,
    )
}

fn initial_compiled_final_merge_fan_in(context: &ExecutionContext, partials: &[CommittedPartial]) -> usize {
    let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } = context.memory_control_policy() else {
        unreachable!("initial_compiled_final_merge_fan_in is called only for bounded policies");
    };
    let threshold_bytes = max_source_chunk_bytes.max(1);
    let total_bytes = partials.iter().map(compiled_partial_merge_basis_bytes).sum::<u64>();
    if total_bytes <= threshold_bytes {
        return partials.len();
    }

    let mut fan_in = 0usize;
    let mut selected_bytes = 0u64;
    for partial in partials {
        let partial_bytes = compiled_partial_merge_basis_bytes(partial);
        if fan_in >= 2 && selected_bytes.saturating_add(partial_bytes) > threshold_bytes {
            break;
        }
        selected_bytes = selected_bytes.saturating_add(partial_bytes);
        fan_in += 1;
    }

    fan_in.clamp(2, partials.len())
}

fn compiled_partial_merge_basis_bytes(partial: &CommittedPartial) -> u64 {
    partial.reservation.bytes().max(partial.frame.num_rows() as u64)
}
fn execute_compiled_final_merge_level_attempt(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CompiledMergeLevelItem>> {
    metrics.record_final_merge_attempt(merge_fan_in);
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(CompiledMergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_compiled_final_merge_group_attempt(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            group,
            attempt,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(partial) => output.push(CompiledMergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                release_compiled_merge_level_items(output, reservations, metrics);
                return Err(error);
            },
        }
    }

    Ok(output)
}
fn execute_compiled_final_merge_group_attempt(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let mut merge_reservation = reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_merge")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
        metrics,
    )?;
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&merge_reservation),
        "compiled_aggregate_final_merge",
        "cudf_query_engine.bounded.aggregate.compiled_stream_final_merge.budget_token",
        "failed to create bounded native compiled stream final-aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(merge_reservation, metrics);
            return Err(error);
        },
    };

    let timer = Instant::now();
    let result = merge_compiled_final_aggregate_partials(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        &token,
        &mut merge_reservation,
        reservations,
        attempt,
        max_attempts,
        metrics,
    );
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(error);
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_merge_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
        metrics,
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}
fn merge_compiled_final_aggregate_partials(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    attempt: usize,
    max_attempts: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let concatenated = first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
    drop(other_frame_refs);
    drop(other_frames);
    drop(first_frame);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.compiled_stream_concat_partials",
            "failed to concatenate compiled stream aggregate partials",
            error,
        )
    })?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_final_merge")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_final_merge")
        .with_attempt(attempt, max_attempts);
        reservations.try_grow(merge_reservation, grow_request, metrics)?;
    } else {
        reservations.shrink_to_observed(merge_reservation, concatenated_bytes, metrics);
    }

    execute_compiled_final_aggregate(context, aggregate, input_schema, output_schema, concatenated, token, metrics)
}

enum CompiledMergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

fn commit_compiled_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<CompiledMergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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

    for partial in partials.into_iter().flatten() {
        reservations.release(partial.reservation, metrics);
    }

    next_partials
}

fn release_compiled_merge_level_items(
    items: Vec<CompiledMergeLevelItem>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    for item in items {
        if let CompiledMergeLevelItem::Merged(partial) = item {
            let partial = *partial;
            reservations.release(partial.reservation, metrics);
        }
    }
}

fn is_retryable_compiled_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_compiled_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn execute_compiled_final_aggregate_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let work_reservation = reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_final_projection")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_projection"),
        metrics,
    )?;
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "compiled_aggregate_final_projection",
        "cudf_query_engine.bounded.aggregate.compiled_stream_final_projection.budget_token",
        "failed to create bounded native compiled stream final-projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let frame =
        execute_compiled_final_aggregate(context, aggregate, input_schema, output_schema, input.frame, &token, metrics)
            .and_then(|frame| {
                finalize_compiled_final_aggregate_output(context, aggregate, output_schema, frame, &token, metrics)
            });
    reservations.release(work_reservation, metrics);
    reservations.release(input.reservation, metrics);
    let frame = frame?;
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate"),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn finalize_compiled_final_aggregate_output(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
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
                QueryError::from_cudf(
                    "cudf_query_engine.compiled_final_projection.select",
                    "failed to select bounded native compiled final aggregate output columns",
                    error,
                )
            })
        })
        .map(GpuDataFrame::new)
}

fn execute_compiled_final_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let bounded = CompiledAggregatePlan { aggregate, input_schema, output_schema };
    execute_final_aggregate_for_plan(context, &bounded, input, token, metrics)
}

struct CompiledAggregatePlan<'a> {
    aggregate: &'a AggregateNode,
    input_schema: &'a arrow::datatypes::SchemaRef,
    output_schema: &'a arrow::datatypes::SchemaRef,
}

fn execute_final_aggregate_for_plan(
    context: &ExecutionContext,
    bounded: &CompiledAggregatePlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
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

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.output_schema,
            "cudf_query_engine.final_merge.validate",
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
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.spec, bounded.output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    record_default_grouped_aggregate_strategy_decision(
        metrics,
        None,
        AggregateExecutionStage::FinalMerge,
        context.memory_control_policy().as_str(),
        &bounded.aggregate.spec,
        bounded.input_schema,
        input_rows,
        output_rows,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}
