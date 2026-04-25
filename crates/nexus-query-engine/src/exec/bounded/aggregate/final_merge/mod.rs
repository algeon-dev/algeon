use std::{collections::VecDeque, time::Duration};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{final_merge_policy::BoundedFinalMergePolicy, *};
use crate::{
    capability::{AggregateExecutionStage, AggregateExecutionStrategy, aggregate_strategy_capability_decision},
    expr::{SortExpr, SortSpec},
};

mod compiled;
mod partitioned;

pub(in crate::exec::bounded) use compiled::execute_compiled_bounded_final_aggregate_merge;
pub(in crate::exec::bounded) use partitioned::execute_final_aggregate_merge;

pub(super) struct PartitionedFinalMergeTopK<'a> {
    sort: SortNode,
    limit: &'a LimitNode,
    keep_rows: usize,
}

#[derive(Default)]
pub(super) struct PartitionedFinalMergeTopKStats {
    enabled: bool,
    keep_rows: usize,
    input_rows: u64,
    output_rows: u64,
    partition_count: usize,
}

pub(super) fn partitioned_final_merge_partition_count(
    context: &ExecutionContext,
    partials: &[CommittedPartial],
) -> usize {
    let input_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>().max(1);
    let partial_partition_floor =
        BoundedFinalMergePolicy::default().partitioned_final_merge_partial_partition_floor(partials.len());
    partitioned_aggregate_partition_count(context, input_bytes).max(partial_partition_floor)
}

pub(super) fn partition_final_merge_partial(
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
            .with_task_registration(task_id.clone(), 1, "aggregate"),
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
        "nexus_query_engine.final_merge.partitioned_hash.budget_token",
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
                    "nexus_query_engine.final_merge.hash_partition",
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
        .with_task_registration(task_id, 1, "aggregate");
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

pub(super) fn slice_partitioned_final_merge_partition(
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.final_merge.hash_partition_slice",
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
            .with_task_registration(
                partitioned_final_merge_task_id(
                    node_id,
                    format!("partition.partial{partial_index}.partition{partition}"),
                ),
                1,
                "aggregate",
            ),
        metrics,
    )?;
    Ok(CommittedPartial { frame: gpu, reservation })
}

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
    if bounded.post_aggregate_stages.iter().any(|stage| matches!(stage, BoundedUnaryStage::Filter { .. })) {
        return Ok(None);
    }
    let mut expressions = Vec::with_capacity(sort.spec.expressions.len());
    for sort_expr in &sort.spec.expressions {
        let mut column = lower_column_expr(&sort_expr.expr, "nexus_query_engine.final_merge.topk_sort")?;
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
        BoundedUnaryStage::Filter { .. } => Ok(None),
        BoundedUnaryStage::Projection { node, .. } => {
            let Some(output) = node.outputs.iter().find(|output| output.field.name() == column) else {
                return Ok(None);
            };
            lower_column_expr(&output.expr, "nexus_query_engine.final_merge.topk_projection").map(Some)
        },
    }
}

pub(super) fn apply_partitioned_final_merge_topk(
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
            .with_task_registration(
                partitioned_final_merge_task_id(bounded.aggregate_id, format!("topk_input.partition{partition_index}")),
                1,
                "aggregate",
            ),
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

pub(super) fn concatenate_partitioned_final_merge_outputs(
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
            .with_task_registration(task_id.clone(), 1, "aggregate"),
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
        "nexus_query_engine.final_merge.partitioned_output_concat.budget_token",
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
                    "nexus_query_engine.final_merge.partitioned_output_concat",
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
            .with_task_registration(task_id, 1, "aggregate");
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut concat_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(concat_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(concat_reservation, output_request, metrics);
    Ok(CommittedPartial { frame: concatenated, reservation: output_reservation })
}

pub(super) fn partitioned_final_merge_task_id(node_id: NodeId, stage: impl std::fmt::Display) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.partitioned.{stage}")
}

pub(super) fn legacy_bounded_final_merge_task_id(node_id: NodeId, group_index: usize, attempt: usize) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.group{group_index}.attempt{attempt}")
}

pub(in crate::exec::bounded) fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    BoundedFinalMergePolicy::default().reduce_merge_fan_in(merge_fan_in)
}

pub(super) fn final_merge_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if BoundedFinalMergePolicy::default().merge_fan_in_is_at_retry_floor(merge_fan_in) {
        "fail_min_merge_fan_in"
    } else {
        "retry_attempts_exhausted"
    };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_merge")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
        .with_merge_level(merge_level)
        .with_merge_fan_in(merge_fan_in)
}
