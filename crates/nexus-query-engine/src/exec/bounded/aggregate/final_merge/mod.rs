use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{final_merge_policy::BoundedFinalMergePolicy, *};
use crate::{
    capability::{AggregateExecutionStage, AggregateExecutionStrategy, aggregate_strategy_capability_decision},
    expr::{SortExpr, SortSpec},
};

mod compiled;
mod partitioned;

pub(in crate::exec::bounded) use compiled::execute_compiled_bounded_final_aggregate_merge;
pub(in crate::exec::bounded) use partitioned::{
    AlignedGroupKeyPartitionPartial, execute_aligned_group_key_partitioned_final_aggregate_merge,
    execute_final_aggregate_merge,
};

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
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    partial_index: usize,
    partial: CommittedPartial,
    key_columns: &[String],
    partition_count: usize,
) -> Result<Vec<CommittedPartial>> {
    let task_id = partitioned_final_merge_task_id(node_id, format!("hash_partition.partial{partial_index}"));
    let mut partition_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            partial.reservation.bytes(),
            execution.reservations().budget(),
        )
        .with_memory_boundary("final_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("partitioned_final_merge")
        .with_task_registration(task_id.clone(), 1, "aggregate"),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(partial.reservation);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&partition_reservation),
        "final_aggregate_hash_partition",
        "nexus_query_engine.final_merge.partitioned_hash.budget_token",
        "failed to create partitioned final aggregate hash budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(partition_reservation);
            execution.release(partial.reservation);
            return Err(error);
        },
    };

    let frame = partial.frame.into_frame();
    let input_rows = frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) =
        crate::observability::stage_context_from_execution(execution.context());
    let stage_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::HashPartition,
        crate::observability::RuntimeOperatorStageIdentity::new()
            .with_node_id(node_id.to_string())
            .with_input_rows(input_rows),
    );
    let partitioned = match frame
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
    execution.release(partial.reservation);
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            execution.release(partition_reservation);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(partition_reservation);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("final_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("partitioned_final_merge")
        .with_task_registration(task_id, 1, "aggregate");
        if let Err(error) = execution.try_grow(&mut partition_reservation, grow_request) {
            execution.release(partition_reservation);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut partition_reservation, partitioned_bytes);
    }

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
                release_partials_reverse(execution, partitions);
                execution.release(partition_reservation);
                return Err(error);
            },
        }
    }
    execution.release(partition_reservation);
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
    let overrides_lease = execution_overrides(execution.context())?;
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
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics_mut())?;
    let reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, bytes, execution.reservations().budget())
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

    let input_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let input_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalSort, input_bytes, execution.reservations().budget())
            .with_memory_boundary("partitioned_final_aggregate_topk_input")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge_topk")
            .with_task_registration(
                partitioned_final_merge_task_id(bounded.aggregate_id, format!("topk_input.partition{partition_index}")),
                1,
                "aggregate",
            ),
    )?;
    let input = CommittedPartial { frame, reservation: input_reservation };
    let (observation, plan_execution_id, parent_span_id) =
        crate::observability::stage_context_from_execution(execution.context());
    let stage_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::PartitionLocalTopK,
        crate::observability::RuntimeOperatorStageIdentity::new()
            .with_node_id(bounded.aggregate_id.to_string())
            .with_partition_index(partition_index)
            .with_input_rows(topk_stats.input_rows),
    );
    let output =
        match execute_final_sort_node_with_retry(execution, bounded.aggregate_id, &topk.sort, Some(topk.limit), input)
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
    execution.release(output.reservation);
    Ok(output_frame)
}

pub(super) fn concatenate_partitioned_final_merge_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut inputs: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    let requested_bytes = inputs.iter().map(|input| input.reservation.bytes()).sum::<u64>();
    let task_id = partitioned_final_merge_task_id(node_id, "output_concat");
    let mut concat_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("partitioned_final_aggregate_output_concat")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_task_registration(task_id.clone(), 1, "aggregate"),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_partials_reverse(execution, inputs);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&concat_reservation),
        "partitioned_final_aggregate_output_concat",
        "nexus_query_engine.final_merge.partitioned_output_concat.budget_token",
        "failed to create partitioned final aggregate output concat budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(concat_reservation);
            release_partials_reverse(execution, inputs);
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
    execution.release(first_reservation);
    release_partials_reverse(execution, inputs);
    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            execution.release(concat_reservation);
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(concat_reservation);
            return Err(error);
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, execution.reservations().budget())
            .with_memory_boundary("partitioned_final_aggregate_output_concat")
            .with_node_id(node_id)
            .with_stage("partitioned_final_merge")
            .with_task_registration(task_id, 1, "aggregate");
    if let Err(error) =
        execution.reconcile_work_reservation_to_output(&mut concat_reservation, output_bytes, &output_request)
    {
        execution.release(concat_reservation);
        return Err(error);
    }
    let output_reservation = execution.transfer(concat_reservation, output_request);
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
