use super::{stream::execute_compiled_bounded_partial_aggregate_from_frame, *};

pub(super) fn execute_compiled_bounded_partitioned_aggregate_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
) -> Result<ResidentGpuFrame> {
    let mut lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    if aggregate.spec.grouping_sets.is_some() {
        lowered_keys.key_columns.truncate(aggregate.spec.group_exprs.len());
    }
    let partition_count = partitioned_aggregate_partition_count(execution.context(), input_reserved_bytes);
    let mut partition_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            input_reserved_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_hash_partition"),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            return Err(error);
        },
    };
    let frame = input.into_frame();
    let partitioned = {
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            execution.reservations().available_bytes_including(&partition_reservation),
            "compiled_aggregate_hash_partition",
            "nexus_query_engine.bounded.aggregate.compiled_partitioned_hash.budget_token",
            "failed to create bounded native compiled partitioned aggregate budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                execution.release(partition_reservation);
                return Err(error);
            },
        };

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
        match frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(lowered_keys.key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "nexus_query_engine.bounded.aggregate.hash_partition",
                        "failed to hash-partition compiled bounded aggregate input",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("key_count", lowered_keys.key_columns.len())
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
        }
    };
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
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            execution.reservations().budget(),
        )
        .with_memory_boundary("compiled_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_hash_partition");
        if let Err(error) = execution.try_grow(&mut partition_reservation, grow_request) {
            execution.release(partition_reservation);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut partition_reservation, partitioned_bytes);
    }

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame =
            match slice_compiled_aggregate_partition(execution, node_id, &partitioned_frame, &offsets, partition) {
                Ok(frame) => frame,
                Err(error) => {
                    release_partials_reverse(execution, partials);
                    execution.release(partition_reservation);
                    return Err(error);
                },
            };
        match execute_compiled_bounded_partial_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_frame,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(execution, partials);
                execution.release(partition_reservation);
                return Err(error);
            },
        }
    }
    execution.release(partition_reservation);

    execute_compiled_bounded_final_aggregate_merge(execution, node_id, aggregate, input_schema, output_schema, partials)
}

fn slice_compiled_aggregate_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    partition: usize,
) -> Result<ResidentGpuFrame> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.aggregate.hash_partition_slice",
            "failed to slice compiled bounded hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics_mut())?;
    let reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, execution.reservations().budget())
            .with_memory_boundary("compiled_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_partition"),
    )?;
    Ok(ResidentGpuFrame { frame: gpu, reservation })
}

pub(super) fn should_partition_compiled_grouped_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_reserved_bytes: u64,
) -> bool {
    !aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
        && !aggregate_spec_has_median(&aggregate.spec)
        && !aggregate.spec.group_exprs.is_empty()
        && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
}

pub(super) fn partitioned_aggregate_partition_count(context: &ExecutionContext, input_bytes: u64) -> usize {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return 1;
    };
    let target = BoundedPartitionTargetPolicy::default().aggregate_partition_target_bytes(device_budget_bytes);
    let partitions = input_bytes.max(1).saturating_add(target.saturating_sub(1)) / target.max(1);
    partitions.clamp(1, 1024) as usize
}
