use super::*;

/// Pre-aligned group-key partition partials from the source-adjacent streaming path.
/// Compatible membership means every item shares the same partition count and key identity.
pub(in crate::exec::bounded) struct AlignedGroupKeyPartitionPartial {
    pub(in crate::exec::bounded) partition_index: usize,
    pub(in crate::exec::bounded) partition_count: usize,
    pub(in crate::exec::bounded) key_columns: Vec<String>,
    pub(in crate::exec::bounded) partial: CommittedPartial,
}

/// Merge already group-key-aligned partials without re-hashing.
pub(in crate::exec::bounded) fn execute_aligned_group_key_partitioned_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partition_count: usize,
    key_columns: &[String],
    partials: Vec<AlignedGroupKeyPartitionPartial>,
) -> Result<CommittedPartial> {
    if partials.is_empty() {
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, reservation: frame.reservation });
    }

    for partial in &partials {
        if partial.partition_count != partition_count {
            let error = QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.final_merge.aligned_partition_metadata",
                "aligned group-key final merge received mismatched partition_count metadata",
            )
            .with_fact("expected_partition_count", partition_count)
            .with_fact("actual_partition_count", partial.partition_count)
            .with_fact("partition_index", partial.partition_index);
            release_aligned_group_key_partials_reverse(execution, partials);
            return Err(error);
        }
        if partial.partition_index >= partition_count {
            let error = QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.final_merge.aligned_partition_metadata",
                "aligned group-key final merge received out-of-range partition_index",
            )
            .with_fact("partition_count", partition_count)
            .with_fact("partition_index", partial.partition_index);
            release_aligned_group_key_partials_reverse(execution, partials);
            return Err(error);
        }
        if partial.key_columns.as_slice() != key_columns {
            let error = QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.final_merge.aligned_partition_metadata",
                "aligned group-key final merge received mismatched partition key identity",
            )
            .with_fact("expected_keys", key_columns.join(","))
            .with_fact("actual_keys", partial.key_columns.join(","))
            .with_fact("partition_index", partial.partition_index);
            release_aligned_group_key_partials_reverse(execution, partials);
            return Err(error);
        }
    }

    let topk = match partitioned_final_merge_topk_candidate(bounded) {
        Ok(topk) => topk,
        Err(error) => {
            release_aligned_group_key_partials_reverse(execution, partials);
            return Err(error);
        },
    };
    let mut topk_stats = PartitionedFinalMergeTopKStats::default();
    let input_rows = partials.iter().map(|partial| partial.partial.frame.num_rows() as u64).sum::<u64>();
    let mut partition_inputs = (0..partition_count).map(|_| Vec::new()).collect::<Vec<Vec<CommittedPartial>>>();
    let mut max_partition_fan_in = 0usize;
    for item in partials {
        let partition_index = item.partition_index;
        if item.partial.frame.num_rows() == 0 {
            execution.release(item.partial.reservation);
            continue;
        }
        partition_inputs[partition_index].push(item.partial);
        max_partition_fan_in = max_partition_fan_in.max(partition_inputs[partition_index].len());
    }

    let timer = Instant::now();
    let mut merged_partitions = Vec::with_capacity(partition_count);
    let mut non_empty_partition_count = 0usize;
    for (partition_index, inputs) in partition_inputs.into_iter().enumerate() {
        if inputs.is_empty() {
            continue;
        }
        non_empty_partition_count += 1;
        max_partition_fan_in = max_partition_fan_in.max(inputs.len());
        let merged = match execute_partitioned_final_merge_group(
            execution,
            bounded,
            partition_index,
            inputs,
            topk.as_ref(),
            &mut topk_stats,
        ) {
            Ok(merged) => merged,
            Err(error) => {
                release_partials_reverse(execution, merged_partitions);
                return Err(error);
            },
        };
        merged_partitions.push(merged);
    }
    if merged_partitions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.final_merge.aligned_partitioned",
            "aligned group-key final aggregate merge produced no non-empty partitions",
        )
        .with_fact("node_id", bounded.aggregate_id)
        .with_fact("partition_count", partition_count));
    }
    execution.metrics_mut().record_final_merge_attempt(max_partition_fan_in.max(1));
    execution.metrics_mut().record_final_merge_level();
    let mut merged = concatenate_partitioned_final_merge_outputs(execution, bounded.aggregate_id, merged_partitions)?;
    let elapsed = timer.elapsed();
    let output_rows = merged.frame.num_rows() as u64;
    record_partitioned_final_merge_strategy(
        execution.context(),
        bounded,
        input_rows,
        output_rows,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
        &topk_stats,
        elapsed,
        execution.metrics_mut(),
    );
    // Mark the merge as consuming already-aligned group-key partition identity.
    if let Some(decision) = execution.metrics_mut().aggregate_strategy_decisions.last_mut() {
        decision.facts.insert("partitioned_merge_aligned_input".to_string(), "group_key_source_adjacent".to_string());
        decision.facts.insert("partitioned_merge_rehash_skipped".to_string(), "true".to_string());
        decision.facts.insert("partitioned_merge_key_columns".to_string(), key_columns.join(","));
    }
    let merged_reservation_bytes = merged.reservation.bytes();
    merged.reservation = execution.transfer(
        merged.reservation,
        MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            merged_reservation_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("partitioned_final_aggregate_merge_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("partitioned_final_merge")
        .with_task_registration(
            partitioned_final_merge_task_id(bounded.aggregate_id, "aligned_combined_output"),
            1,
            "aggregate",
        ),
    );
    execute_final_aggregate_projection(execution, bounded, merged)
}

fn release_aligned_group_key_partials_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut partials: Vec<AlignedGroupKeyPartitionPartial>,
) {
    while let Some(partial) = partials.pop() {
        execution.release(partial.partial.reservation);
    }
}

pub(in crate::exec::bounded) fn execute_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = execution.context().memory_control_policy() else {
        unreachable!("execute_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let mut partials = partials;
    let mut merge_level = 1;

    // Empty ROLLUP/CUBE/GROUPING SETS still need the grand-total identity row.
    // A single empty partial used to short-circuit through final projection and
    // drop that row because SUM/COUNT partial state needs no AVG finalize pass.
    let all_empty_grouping_sets = bounded.aggregate.0.spec.grouping_sets.is_some()
        && (partials.is_empty() || partials.iter().all(|partial| partial.frame.num_rows() == 0));
    if all_empty_grouping_sets {
        release_partials_reverse(execution, partials);
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, reservation: frame.reservation });
    }

    if partials.is_empty() {
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, reservation: frame.reservation });
    }

    if should_execute_partitioned_final_aggregate_merge(execution.context(), bounded, &partials) {
        let partial = execute_partitioned_final_aggregate_merge(execution, bounded, partials)?;
        return execute_final_aggregate_projection(execution, bounded, partial);
    }

    while partials.len() > 1 {
        let mut merge_fan_in = partials.len();
        let mut attempt = 1;

        loop {
            match execute_final_merge_level_attempt(execution, bounded, &partials, merge_fan_in, attempt, max_attempts)
            {
                Ok(level_output) => {
                    execution.metrics_mut().record_final_merge_level();
                    partials = commit_final_merge_level(execution, partials, level_output);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_merge_error(&error) => {
                    record_final_merge_retry_error(&error, execution.metrics_mut());
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        crate::exec::retry::backoff_before_retry(
                            execution.context(),
                            "nexus_query_engine.exec.bounded.final_merge_fan_in_retry",
                        )?;
                        continue;
                    }

                    return Err(final_merge_retry_facts(merge_level, merge_fan_in, execution.metrics_mut())
                        .attach_to_error(error));
                },
                Err(error) => return Err(error),
            }
        }
    }

    let partial = partials.pop().expect("final merge loop leaves exactly one partial");
    execute_final_aggregate_projection(execution, bounded, partial)
}
fn execute_final_merge_level_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
) -> Result<Vec<MergeLevelItem>> {
    execution.metrics_mut().record_final_merge_attempt(merge_fan_in);
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(MergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_final_merge_group_attempt(execution, bounded, group, group_index, attempt, max_attempts) {
            Ok(partial) => output.push(MergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                release_merge_level_items(execution, output);
                return Err(error);
            },
        }
    }

    Ok(output)
}

fn execute_final_merge_group_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    group_index: usize,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let merge_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts)
            .with_task_registration(
                legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt),
                u32::try_from(attempt).unwrap_or(u32::MAX),
                "aggregate",
            );
    let mut merge_reservation = execution.try_reserve(merge_request)?;
    let merge_token_budget = execution.reservations().available_bytes_including(&merge_reservation);
    let merge_token = match execution_budget_token_for_boundary(
        execution.context(),
        merge_token_budget,
        "final_aggregate_merge",
        "nexus_query_engine.final_merge.budget_token",
        "failed to create bounded native final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(merge_reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics: execution.metrics_mut(),
            }
            .attach_to(error));
        },
    };

    let timer = Instant::now();
    let input_rows = partials.iter().map(|partial| partial.frame.num_rows() as u64).sum::<u64>();
    let (observation, plan_execution_id, parent_span_id) =
        crate::observability::stage_context_from_execution(execution.context());
    let stage_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::FinalMerge,
        crate::observability::RuntimeOperatorStageIdentity::new()
            .with_node_id(bounded.aggregate_id.to_string())
            .with_attempt(u32::try_from(attempt).unwrap_or(u32::MAX))
            .with_input_rows(input_rows),
    );
    let result = merge_final_aggregate_partials(
        execution,
        bounded,
        partials,
        group_index,
        &merge_token,
        &mut merge_reservation,
        attempt,
        max_attempts,
    );
    let merge_elapsed = timer.elapsed();
    execution.release(merge_reservation);
    let frame = match result {
        Ok(frame) => {
            stage_guard.finish(Some(frame.num_rows() as u64), None);
            frame
        },
        Err(error) => {
            execution.metrics_mut().final_merge_time += merge_elapsed;
            let error = RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics: execution.metrics_mut(),
            }
            .attach_to(error);
            stage_guard.fail(&error);
            return Err(error);
        },
    };
    execution.metrics_mut().final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut()).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_merge_output",
            stage: "final_merge",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_merge",
            metrics: execution.metrics_mut(),
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, execution.reservations().budget())
            .with_memory_boundary("final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts)
            .with_task_registration(
                legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt),
                u32::try_from(attempt).unwrap_or(u32::MAX),
                "aggregate",
            );
    let output_reservation = execution.try_reserve(output_request)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
fn merge_final_aggregate_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    group_index: usize,
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    attempt: usize,
    max_attempts: usize,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = token.execution_overrides();
    let concatenated = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "nexus_query_engine.final_merge.concat_partials",
                "failed to concatenate bounded native aggregate partials",
                error,
            )
            .with_fact("partial_aggregate_count", partial_count),
            token,
        )
    })?;
    let concatenated = GpuDataFrame::new(concatenated);
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, execution.metrics_mut())?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("final_aggregate_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_merge")
        .with_attempt(attempt, max_attempts)
        .with_task_registration(
            legacy_bounded_final_merge_task_id(bounded.aggregate_id, group_index, attempt),
            u32::try_from(attempt).unwrap_or(u32::MAX),
            "aggregate",
        );
        execution.try_grow(merge_reservation, grow_request)?;
    }

    execute_final_aggregate(execution.context(), bounded, concatenated, token, true, execution.metrics_mut())
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

fn execute_partitioned_final_aggregate_merge(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    let partition_count = partitioned_final_merge_partition_count(execution.context(), &partials);
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
            execution,
            bounded.aggregate_id,
            partial_index,
            partial,
            &key_columns,
            partition_count,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                release_nested_partials_reverse(execution, partition_inputs);
                release_partials_reverse(execution, pending);
                return Err(error);
            },
        };
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if partition.frame.num_rows() > 0 {
                partition_inputs[partition_index].push(partition);
            } else {
                execution.release(partition.reservation);
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
            execution,
            bounded,
            partition_index,
            inputs,
            topk.as_ref(),
            &mut topk_stats,
        )?;
        merged_partitions.push(merged);
    }
    if merged_partitions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.final_merge.partitioned",
            "partitioned final aggregate merge produced no non-empty partitions",
        )
        .with_fact("node_id", bounded.aggregate_id)
        .with_fact("partition_count", partition_count));
    }
    execution.metrics_mut().record_final_merge_attempt(max_partition_fan_in.max(1));
    execution.metrics_mut().record_final_merge_level();
    let mut merged = concatenate_partitioned_final_merge_outputs(execution, bounded.aggregate_id, merged_partitions)?;
    let elapsed = timer.elapsed();
    let output_rows = merged.frame.num_rows() as u64;
    record_partitioned_final_merge_strategy(
        execution.context(),
        bounded,
        input_rows,
        output_rows,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
        &topk_stats,
        elapsed,
        execution.metrics_mut(),
    );
    let merged_reservation_bytes = merged.reservation.bytes();
    merged.reservation = execution.transfer(
        merged.reservation,
        MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            merged_reservation_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("partitioned_final_aggregate_merge_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("partitioned_final_merge")
        .with_task_registration(
            partitioned_final_merge_task_id(bounded.aggregate_id, "combined_output"),
            1,
            "aggregate",
        ),
    );
    Ok(merged)
}

fn execute_partitioned_final_merge_group(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    inputs: Vec<CommittedPartial>,
    topk: Option<&PartitionedFinalMergeTopK<'_>>,
    topk_stats: &mut PartitionedFinalMergeTopKStats,
) -> Result<CommittedPartial> {
    let requested_bytes = inputs.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let task_id = partitioned_final_merge_task_id(bounded.aggregate_id, format!("merge.partition{partition_index}"));
    let mut merge_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("partitioned_final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge")
            .with_attempt(1, 1)
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
        execution.reservations().available_bytes_including(&merge_reservation),
        "partitioned_final_aggregate_merge",
        "nexus_query_engine.final_merge.partitioned_merge.budget_token",
        "failed to create partitioned final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(merge_reservation);
            release_partials_reverse(execution, inputs);
            return Err(error);
        },
    };
    let result = merge_partitioned_final_aggregate_partials(
        execution,
        bounded,
        partition_index,
        &inputs,
        &token,
        &mut merge_reservation,
    );
    // Detach the native stream reservation before releasing the engine reservations below.
    drop(token);
    release_partials_reverse(execution, inputs);
    execution.release(merge_reservation);
    let frame = result?;
    let frame = apply_partitioned_final_merge_topk(execution, bounded, partition_index, frame, topk, topk_stats)?;
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let output_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, execution.reservations().budget())
            .with_memory_boundary("partitioned_final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("partitioned_final_merge")
            .with_attempt(1, 1)
            .with_task_registration(task_id, 1, "aggregate"),
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn merge_partitioned_final_aggregate_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
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
                    "nexus_query_engine.final_merge.partitioned_concat_partials",
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
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, execution.metrics_mut())?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("partitioned_final_aggregate_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("partitioned_final_merge")
        .with_attempt(1, 1)
        .with_task_registration(
            partitioned_final_merge_task_id(bounded.aggregate_id, format!("merge.partition{partition_index}")),
            1,
            "aggregate",
        );
        execution.try_grow(merge_reservation, grow_request)?;
    } else {
        execution.shrink_to_observed(merge_reservation, concatenated_bytes);
    }

    execute_final_aggregate(execution.context(), bounded, concatenated, token, false, execution.metrics_mut())
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

enum MergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

fn commit_final_merge_level(
    execution: &mut BoundedExecutionAttempt<'_>,
    partials: Vec<CommittedPartial>,
    level_output: Vec<MergeLevelItem>,
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
        execution.release(partial.reservation);
    }

    next_partials
}

fn release_merge_level_items(execution: &mut BoundedExecutionAttempt<'_>, items: Vec<MergeLevelItem>) {
    for item in items {
        if let MergeLevelItem::Merged(partial) = item {
            let partial = *partial;
            execution.release(partial.reservation);
        }
    }
}

fn is_retryable_final_merge_error(error: &QueryError) -> bool {
    crate::exec::retry::is_retryable_resource_error(error)
}

fn record_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    crate::exec::retry::record_retryable_resource_error(error, metrics);
}
