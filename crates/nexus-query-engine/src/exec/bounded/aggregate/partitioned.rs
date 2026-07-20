use super::{stream::execute_compiled_bounded_partial_aggregate_from_frame, *};

pub(super) fn execute_compiled_bounded_partitioned_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    if aggregate.spec.grouping_sets.is_some() {
        lowered_keys.key_columns.truncate(aggregate.spec.group_exprs.len());
    }
    let partition_count = partitioned_aggregate_partition_count(context, input_reserved_bytes);
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_hash_partition"),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input);
            return Err(error);
        },
    };
    let frame = input.into_frame();
    let partitioned = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&partition_reservation),
            "compiled_aggregate_hash_partition",
            "nexus_query_engine.bounded.aggregate.compiled_partitioned_hash.budget_token",
            "failed to create bounded native compiled partitioned aggregate budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(partition_reservation, metrics);
                drop(frame);
                return Err(error);
            },
        };

        frame
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
            })
    };
    drop(frame);
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
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_hash_partition");
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame = match slice_compiled_aggregate_partition(
            context,
            node_id,
            &partitioned_frame,
            &offsets,
            partition,
            reservations,
            metrics,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_frame,
            reservations,
            metrics,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);

    execute_compiled_bounded_final_aggregate_merge(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        reservations,
        metrics,
    )
}

fn slice_compiled_aggregate_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    partition: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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
    let overrides_lease = execution_overrides(context)?;
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
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_partition"),
        metrics,
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
    usize::try_from(partitions.clamp(1, 1024)).unwrap_or(1024)
}
