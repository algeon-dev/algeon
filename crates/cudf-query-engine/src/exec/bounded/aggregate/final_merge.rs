use super::*;

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

    execute_final_aggregate(context, bounded, concatenated, token, metrics)
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
