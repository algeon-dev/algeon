use super::*;

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
        release_partials_reverse(execution, partials);
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

    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = execution.context().memory_control_policy() else {
        unreachable!("execute_compiled_bounded_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);

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
                attempt,
                max_attempts,
            ) {
                Ok(level_output) => {
                    execution.metrics_mut().record_final_merge_level();
                    partials = commit_compiled_final_merge_level(execution, partials, level_output);
                    break;
                },
                Err(error) if is_retryable_compiled_final_merge_error(&error) => {
                    record_compiled_final_merge_retry_error(&error, execution.metrics_mut());
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        crate::exec::retry::backoff_before_retry(
                            execution.context(),
                            "nexus_query_engine.exec.bounded.compiled_final_merge_fan_in_retry",
                        )?;
                        continue;
                    }

                    release_partials_reverse(execution, partials);
                    return Err(error);
                },
                Err(error) => {
                    release_partials_reverse(execution, partials);
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
                release_partials_reverse(execution, merged_partitions);
                release_nested_partials_reverse(
                    execution,
                    pending_partition_inputs.into_iter().map(|(_, inputs)| inputs).collect(),
                );
                return Err(error);
            },
        }
    }
    if merged_partitions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.compiled_final_merge.partitioned",
            "compiled partitioned final aggregate merge produced no non-empty partitions",
        )
        .with_fact("node_id", node_id)
        .with_fact("partition_count", partition_count));
    }
    execution.metrics_mut().record_final_merge_attempt(max_partition_fan_in.max(1));
    execution.metrics_mut().record_final_merge_level();
    let mut merged = concatenate_partitioned_final_merge_outputs(execution, node_id, merged_partitions)?;
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
        .with_memory_boundary("compiled_partitioned_final_aggregate_merge_output")
        .with_node_id(node_id)
        .with_stage("compiled_partitioned_final_merge")
        .with_task_registration(
            partitioned_final_merge_task_id(node_id, "compiled_combined_output"),
            1,
            "aggregate",
        ),
    );
    Ok(merged)
}

fn execute_compiled_partitioned_final_merge_group(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partition_index: usize,
    inputs: Vec<CommittedPartial>,
) -> Result<CommittedPartial> {
    let requested_bytes = inputs.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let task_id = partitioned_final_merge_task_id(node_id, format!("compiled_merge.partition{partition_index}"));
    let mut merge_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_partitioned_final_aggregate_merge")
            .with_node_id(node_id)
            .with_stage("compiled_partitioned_final_merge")
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
        "compiled_partitioned_final_aggregate_merge",
        "nexus_query_engine.final_merge.compiled_partitioned_merge.budget_token",
        "failed to create compiled partitioned final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(merge_reservation);
            release_partials_reverse(execution, inputs);
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
        &mut merge_reservation,
    );
    let merge_elapsed = timer.elapsed();
    // Detach the native stream reservation before releasing the engine reservation below.
    drop(token);
    execution.release(merge_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.metrics_mut().final_merge_time += merge_elapsed;
            return Err(error);
        },
    };
    execution.metrics_mut().final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let output_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_partitioned_final_aggregate_merge_output")
            .with_node_id(node_id)
            .with_stage("compiled_partitioned_final_merge")
            .with_attempt(1, 1)
            .with_task_registration(task_id, 1, "aggregate"),
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn merge_compiled_partitioned_final_aggregate_partials_consuming(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partition_index: usize,
    partials: Vec<CommittedPartial>,
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
) -> Result<GpuDataFrame> {
    let mut partials = partials;
    let partial_count = partials.len();
    let first = partials.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = if partials.is_empty() {
        first_frame
    } else {
        let other_frames = partials.iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
        let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
        let concatenated = first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
        match concatenated {
            Ok(frame) => frame,
            Err(error) => {
                execution.release(first_reservation);
                release_partials_reverse(execution, partials);
                return Err(attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "nexus_query_engine.compiled_final_merge.partitioned_concat_partials",
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
    execution.release(first_reservation);
    let concatenated = GpuDataFrame::new(concatenated);
    let concatenated_bytes = match observe_device_memory_bytes(&concatenated, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_partials_reverse(execution, partials);
            return Err(error);
        },
    };
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_partitioned_final_aggregate_merge")
        .with_node_id(node_id)
        .with_stage("compiled_partitioned_final_merge")
        .with_attempt(1, 1)
        .with_task_registration(
            partitioned_final_merge_task_id(node_id, format!("compiled_merge.partition{partition_index}")),
            1,
            "aggregate",
        );
        if let Err(error) = execution.try_grow(merge_reservation, grow_request) {
            release_partials_reverse(execution, partials);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(merge_reservation, concatenated_bytes);
    }

    release_partials_reverse(execution, partials);
    execute_compiled_final_aggregate(
        execution.context(),
        aggregate,
        input_schema,
        output_schema,
        concatenated,
        token,
        execution.metrics_mut(),
    )
}

fn record_compiled_partitioned_final_merge_strategy(
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
    metrics: &mut QueryMetrics,
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
    record_grouped_aggregate_strategy_decision(
        metrics,
        Some(node_id.to_string()),
        AggregateExecutionStage::FinalMerge,
        context.memory_control_policy().as_str(),
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

fn initial_compiled_final_merge_fan_in(context: &ExecutionContext, partials: &[CommittedPartial]) -> usize {
    let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } = context.memory_control_policy() else {
        unreachable!("initial_compiled_final_merge_fan_in is called only for bounded policies");
    };
    let threshold_bytes = max_source_chunk_bytes.max(1);
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

fn compiled_partial_merge_basis_bytes(partial: &CommittedPartial) -> u64 {
    partial.reservation.bytes().max(partial.frame.num_rows() as u64)
}

fn execute_compiled_final_merge_level_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
) -> Result<Vec<CompiledMergeLevelItem>> {
    execution.metrics_mut().record_final_merge_attempt(merge_fan_in);
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
            attempt,
            max_attempts,
        ) {
            Ok(partial) => output.push(CompiledMergeLevelItem::Merged(Box::new(partial))),
            Err(error) => {
                release_compiled_merge_level_items(execution, output);
                return Err(error);
            },
        }
    }

    Ok(output)
}

fn execute_compiled_final_merge_group_attempt(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let mut merge_reservation = execution.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_final_merge")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
    )?;
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&merge_reservation),
        "compiled_aggregate_final_merge",
        "nexus_query_engine.bounded.aggregate.compiled_stream_final_merge.budget_token",
        "failed to create bounded native compiled stream final-aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(merge_reservation);
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
        &mut merge_reservation,
        attempt,
        max_attempts,
    );
    let merge_elapsed = timer.elapsed();
    execution.release(merge_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.metrics_mut().final_merge_time += merge_elapsed;
            return Err(error);
        },
    };
    execution.metrics_mut().final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let output_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_final_merge_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_final_merge")
            .with_attempt(attempt, max_attempts),
    )?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn merge_compiled_final_aggregate_partials(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    attempt: usize,
    max_attempts: usize,
) -> Result<GpuDataFrame> {
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let concatenated = first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs);
    let concatenated = concatenated.map(GpuDataFrame::new).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.aggregate.compiled_stream_concat_partials",
            "failed to concatenate compiled stream aggregate partials",
            error,
        )
    })?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, execution.metrics_mut())?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_aggregate_final_merge")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_final_merge")
        .with_attempt(attempt, max_attempts);
        execution.try_grow(merge_reservation, grow_request)?;
    } else {
        execution.shrink_to_observed(merge_reservation, concatenated_bytes);
    }

    execute_compiled_final_aggregate(
        execution.context(),
        aggregate,
        input_schema,
        output_schema,
        concatenated,
        token,
        execution.metrics_mut(),
    )
}

enum CompiledMergeLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

fn commit_compiled_final_merge_level(
    execution: &mut BoundedExecutionAttempt<'_>,
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

    for partial in partials.into_iter().flatten() {
        execution.release(partial.reservation);
    }

    next_partials
}

fn release_compiled_merge_level_items(execution: &mut BoundedExecutionAttempt<'_>, items: Vec<CompiledMergeLevelItem>) {
    for item in items {
        if let CompiledMergeLevelItem::Merged(partial) = item {
            let partial = *partial;
            execution.release(partial.reservation);
        }
    }
}

fn is_retryable_compiled_final_merge_error(error: &QueryError) -> bool {
    crate::exec::retry::is_retryable_resource_error(error)
}

fn record_compiled_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    crate::exec::retry::record_retryable_resource_error(error, metrics);
}

fn execute_compiled_final_aggregate_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: CommittedPartial,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let work_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            input_reserved_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_aggregate_final_projection")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_final_projection"),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&work_reservation),
        "compiled_aggregate_final_projection",
        "nexus_query_engine.bounded.aggregate.compiled_stream_final_projection.budget_token",
        "failed to create bounded native compiled stream final-projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error);
        },
    };
    // Empty grouping-set finalization is a full single-stage shrink (including
    // AVG finalize). Skip the usual partial-state final-projection path so we
    // do not look for missing AVG state aliases on an already-final frame.
    let frame = if aggregate.spec.grouping_sets.is_some() && input.frame.num_rows() == 0 {
        execute_aggregate_with_budget(
            execution.context(),
            aggregate,
            input_schema,
            input.frame,
            output_schema,
            &token,
            execution.metrics_mut(),
        )
    } else {
        execute_compiled_final_aggregate(
            execution.context(),
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            &token,
            execution.metrics_mut(),
        )
        .and_then(|frame| {
            finalize_compiled_final_aggregate_output(
                execution.context(),
                aggregate,
                output_schema,
                frame,
                &token,
                execution.metrics_mut(),
            )
        })
    };
    execution.release(work_reservation);
    execution.release(input.reservation);
    let frame = frame?;
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let output_reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_output")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate"),
    )?;
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn finalize_compiled_final_aggregate_output(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    output_schema: &ArrowSchemaRef,
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
                    "nexus_query_engine.compiled_final_projection.select",
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
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let bounded = CompiledAggregatePlan { aggregate, input_schema, output_schema };
    execute_final_aggregate_for_plan(context, &bounded, input, token, metrics)
}

struct CompiledAggregatePlan<'a> {
    aggregate: &'a AggregateNode,
    input_schema: &'a ArrowSchemaRef,
    output_schema: &'a ArrowSchemaRef,
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

    // Empty ROLLUP/CUBE/GROUPING SETS still need the grand-total identity row.
    // Regular final-merge groupby over an empty partial drops it. Route through
    // the full single-stage aggregatemapath, which synthesizes identity rows.
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
            &aggregate,
            bounded.output_schema,
            "nexus_query_engine.final_merge.validate",
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.spec, bounded.output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.final_merge.select",
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
        context.aggregate_strategy_execution_options(),
        &bounded.aggregate.spec,
        bounded.input_schema,
        input_rows,
        output_rows,
        elapsed,
    );
    Ok(GpuDataFrame::new(output))
}
