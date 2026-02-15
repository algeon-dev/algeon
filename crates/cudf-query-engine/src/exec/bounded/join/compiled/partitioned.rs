use super::*;

pub(super) fn execute_compiled_partitioned_hash_equi_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_partitioned_hash_equi_join(node)?;
    let original_left_rows = left.rows;
    let original_left_bytes = left.bytes;
    let original_right_rows = right.rows;
    let original_right_bytes = right.bytes;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let partition_count = partitioned_join_partition_count(context, left.bytes, right.bytes);
    let partition_target_bytes = match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { device_budget_bytes, .. } => {
            partitioned_join_target_partition_bytes(device_budget_bytes)
        },
        _ => 0,
    };
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("join_type", node.spec.join_type.to_string()),
            ("build_side", node.spec.build_side.to_string()),
            ("key_count", node.spec.keys.len().to_string()),
            ("observed_build_rows", side_metric(node.spec.build_side, left.rows, right.rows).to_string()),
            ("build_side_bytes", side_metric(node.spec.build_side, left.bytes, right.bytes).to_string()),
            ("probe_side_bytes", side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes).to_string()),
            ("probe_reuse_count", "1".to_string()),
            (
                "reusable_hash_join_rejected_reason",
                partitioned_join_reusable_rejected_reason(
                    side_metric(node.spec.build_side, left.rows, right.rows),
                    side_metric(node.spec.build_side, left.bytes, right.bytes),
                    side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes),
                )
                .to_string(),
            ),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
        ]),
    );
    let overrides = execution_overrides(context);
    let left_partitioned = hash_partition_join_side(
        context,
        node_id,
        JoinSide::Left,
        left,
        &left_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let right_partitioned = match hash_partition_join_side(
        context,
        node_id,
        JoinSide::Right,
        right,
        &right_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    ) {
        Ok(partitioned) => partitioned,
        Err(error) => {
            release_partitioned_join_side(left_partitioned, reservations, metrics);
            return Err(error);
        },
    };
    let left_partitioned_bytes = left_partitioned.reservation.bytes();
    let right_partitioned_bytes = right_partitioned.reservation.bytes();
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("probe_partition_batch_count", partition_count.to_string()),
            (
                "build_partitioned_bytes",
                side_metric(node.spec.build_side, left_partitioned_bytes, right_partitioned_bytes).to_string(),
            ),
            (
                "probe_partitioned_bytes",
                side_metric(probe_side(node.spec.build_side), left_partitioned_bytes, right_partitioned_bytes)
                    .to_string(),
            ),
        ]),
    );

    let mut outputs = Vec::with_capacity(partition_count);
    let mut total_index_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let mut total_output_rows = 0u64;
    let mut total_output_reservation_bytes = 0u64;
    for partition in 0..partition_count {
        let left_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Left,
            node.spec.build_side,
            &left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        let right_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Right,
            node.spec.build_side,
            &right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        match materialize_partitioned_hash_equi_join_partition(
            context,
            node_id,
            node,
            left_part,
            right_part,
            reservations,
            metrics,
        ) {
            Ok(output) => {
                total_index_bytes = total_index_bytes.saturating_add(output.index_bytes);
                total_output_rows = total_output_rows.saturating_add(output.output_rows);
                total_output_bytes = total_output_bytes.saturating_add(output.output_bytes);
                total_output_reservation_bytes =
                    total_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                outputs.push(output.frame);
            },
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        }
    }
    release_partitioned_join_side(left_partitioned, reservations, metrics);
    release_partitioned_join_side(right_partitioned, reservations, metrics);
    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    let output_bytes = output.reservation.bytes();
    total_output_bytes = total_output_bytes.max(output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_partitioned_hash_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, original_left_rows, original_right_rows),
            build_bytes: side_metric(node.spec.build_side, original_left_bytes, original_right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), original_left_rows, original_right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), original_left_bytes, original_right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: total_index_bytes,
            output_reservation_bytes: total_output_reservation_bytes,
            output_rows: total_output_rows,
            output_bytes: total_output_bytes,
        },
    ));
    Ok(output)
}

pub(super) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_partitioned_mixed_keyed_residual_join(node)?;
    let original_left_rows = left.rows;
    let original_left_bytes = left.bytes;
    let original_right_rows = right.rows;
    let original_right_bytes = right.bytes;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let partition_count = partitioned_join_partition_count(context, left.bytes, right.bytes);
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
            ("join_type", node.spec.join_type.to_string()),
            ("build_side", node.spec.build_side.to_string()),
            ("key_count", node.spec.keys.len().to_string()),
            ("observed_build_rows", side_metric(node.spec.build_side, left.rows, right.rows).to_string()),
            ("build_side_bytes", side_metric(node.spec.build_side, left.bytes, right.bytes).to_string()),
            ("probe_side_bytes", side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes).to_string()),
            ("probe_reuse_count", "1".to_string()),
            (
                "reusable_hash_join_rejected_reason",
                partitioned_join_reusable_rejected_reason(
                    side_metric(node.spec.build_side, left.rows, right.rows),
                    side_metric(node.spec.build_side, left.bytes, right.bytes),
                    side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes),
                )
                .to_string(),
            ),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
        ]),
    );
    let overrides = execution_overrides(context);
    let left_partitioned = hash_partition_join_side(
        context,
        node_id,
        JoinSide::Left,
        left,
        &left_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let right_partitioned = match hash_partition_join_side(
        context,
        node_id,
        JoinSide::Right,
        right,
        &right_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    ) {
        Ok(partitioned) => partitioned,
        Err(error) => {
            release_partitioned_join_side(left_partitioned, reservations, metrics);
            return Err(error);
        },
    };
    let left_partitioned_bytes = left_partitioned.reservation.bytes();
    let right_partitioned_bytes = right_partitioned.reservation.bytes();
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("probe_partition_batch_count", partition_count.to_string()),
            (
                "build_partitioned_bytes",
                side_metric(node.spec.build_side, left_partitioned_bytes, right_partitioned_bytes).to_string(),
            ),
            (
                "probe_partitioned_bytes",
                side_metric(probe_side(node.spec.build_side), left_partitioned_bytes, right_partitioned_bytes)
                    .to_string(),
            ),
        ]),
    );

    let mut outputs = Vec::with_capacity(partition_count);
    let mut total_index_bytes = 0u64;
    let mut total_mixed_size_data_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let mut total_output_rows = 0u64;
    let mut total_output_reservation_bytes = 0u64;
    for partition in 0..partition_count {
        let left_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Left,
            node.spec.build_side,
            &left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        let right_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Right,
            node.spec.build_side,
            &right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        match materialize_partitioned_mixed_keyed_residual_join_partition(
            context,
            node_id,
            node,
            left_part,
            right_part,
            reservations,
            metrics,
        ) {
            Ok(output) => {
                total_index_bytes = total_index_bytes.saturating_add(output.index_bytes);
                total_mixed_size_data_bytes = total_mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
                total_output_rows = total_output_rows.saturating_add(output.output_rows);
                total_output_bytes = total_output_bytes.saturating_add(output.output_bytes);
                total_output_reservation_bytes =
                    total_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                outputs.push(output.frame);
            },
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        }
    }
    release_partitioned_join_side(left_partitioned, reservations, metrics);
    release_partitioned_join_side(right_partitioned, reservations, metrics);
    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    let output_bytes = output.reservation.bytes();
    total_output_bytes = total_output_bytes.max(output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_partitioned_mixed_keyed_residual_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, original_left_rows, original_right_rows),
            build_bytes: side_metric(node.spec.build_side, original_left_bytes, original_right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), original_left_rows, original_right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), original_left_bytes, original_right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: total_mixed_size_data_bytes,
            gather_map_bytes: total_index_bytes,
            output_reservation_bytes: total_output_reservation_bytes,
            output_rows: total_output_rows,
            output_bytes: total_output_bytes,
        },
    ));
    Ok(output)
}

pub(super) fn validate_partitioned_hash_equi_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)?;
    if matches!(node.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && !node.spec.keys.is_empty()
        && node.spec.residual_predicate.is_none()
        && (node.spec.join_type != JoinType::Left || node.spec.build_side == JoinSide::Right)
    {
        return Ok(());
    }
    Err(unsupported_bounded_shape(
        "compiled partitioned hash equi join execution supports keyed inner, left, and full equi joins without residual predicates",
        "compiled_join_strategy_unsupported",
    )
    .with_fact("join_type", node.spec.join_type)
    .with_fact("build_side", node.spec.build_side)
    .with_fact("key_count", node.spec.keys.len())
    .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
    .with_fact("output_projection_len", node.spec.output_projection.len()))
}

pub(super) fn validate_partitioned_mixed_keyed_residual_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)?;
    if !node.spec.keys.is_empty()
        && node.spec.residual_predicate.is_some()
        && !node.spec.output_projection.is_empty()
        && matches!(
            node.spec.join_type,
            JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::LeftSemi | JoinType::LeftAnti
        )
        && (node.spec.join_type != JoinType::Left || node.spec.build_side == JoinSide::Right)
        && (!matches!(node.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
            || (node.spec.build_side == JoinSide::Right
                && node.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)))
    {
        return Ok(());
    }
    Err(unsupported_bounded_shape(
        "compiled partitioned mixed keyed residual join execution supports keyed residual inner/left/full and left-only semi/anti joins",
        "compiled_join_strategy_unsupported",
    )
    .with_fact("join_type", node.spec.join_type)
    .with_fact("build_side", node.spec.build_side)
    .with_fact("key_count", node.spec.keys.len())
    .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
    .with_fact("output_projection_len", node.spec.output_projection.len()))
}

pub(super) fn partitioned_join_partition_count(context: &ExecutionContext, left_bytes: u64, right_bytes: u64) -> usize {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return 2;
    };
    let target_partition_bytes = partitioned_join_target_partition_bytes(device_budget_bytes);
    let basis = left_bytes.max(right_bytes).max(1);
    let partitions = basis.saturating_add(target_partition_bytes.saturating_sub(1)) / target_partition_bytes.max(1);
    usize::try_from(partitions.clamp(2, 1024)).unwrap_or(1024)
}

pub(super) fn partitioned_join_target_bytes_for_context(context: &ExecutionContext) -> u64 {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { device_budget_bytes, .. } => {
            partitioned_join_target_partition_bytes(device_budget_bytes)
        },
        _ => 0,
    }
}

pub(super) fn partitioned_join_target_partition_bytes(device_budget_bytes: u64) -> u64 {
    const MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    const MAX_TARGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    (device_budget_bytes / 8).clamp(MIN_TARGET_BYTES, MAX_TARGET_BYTES)
}

pub(super) fn partitioned_join_reusable_rejected_reason(
    build_rows: u64,
    build_bytes: u64,
    probe_bytes: u64,
) -> &'static str {
    if build_rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD || build_bytes >= LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD {
        "build_side_too_large"
    } else if probe_bytes >= LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD {
        "probe_side_too_large"
    } else {
        "partitioned_strategy_selected"
    }
}

pub(super) fn hash_partition_join_side(
    context: &ExecutionContext,
    node_id: NodeId,
    side: JoinSide,
    input: BoundedJoinSideFrame,
    keys: &[String],
    partition_count: usize,
    overrides: &ExecutionOverrides<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinSide> {
    let BoundedJoinSideFrame { frame, _reservation: input_reservation, rows: _, bytes } = input;
    let boundary =
        if side == JoinSide::Left { "join_hash_partitioned_left_side" } else { "join_hash_partitioned_right_side" };
    let mut reservation = reservations.try_reserve(
        bounded_join_request(ReservationKind::OperatorOutput, boundary, bytes, reservations.budget(), node_id),
        metrics,
    )?;
    let partitioned = frame
        .with_execution(overrides)
        .hash_partition_by(keys.iter().map(String::as_str), partition_count)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.hash_partition",
                "failed to hash-partition compiled bounded join input",
                error,
            )
            .with_fact("side", side)
            .with_fact("key_count", keys.len())
            .with_fact("partition_count", partition_count)
        })?;
    let (partitioned, offsets) = partitioned.into_parts();
    let partitioned_frame = GpuDataFrame::new(partitioned.clone());
    let partitioned_bytes = observe_device_memory_bytes(&partitioned_frame, metrics)?;
    if partitioned_bytes > reservation.bytes() {
        let delta = partitioned_bytes.saturating_sub(reservation.bytes());
        reservations.try_grow(
            &mut reservation,
            bounded_join_request(ReservationKind::OperatorOutput, boundary, delta, reservations.budget(), node_id),
            metrics,
        )?;
    } else {
        reservations.shrink_to_observed(&mut reservation, partitioned_bytes, metrics);
    }
    drop(frame);
    drop(input_reservation);
    let _ = context;
    Ok(PartitionedJoinSide { frame: partitioned, reservation, offsets })
}

pub(super) fn release_partitioned_join_side(
    side: PartitionedJoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let PartitionedJoinSide { frame, reservation, .. } = side;
    drop(frame);
    reservations.release(reservation, metrics);
}

pub(super) fn slice_partition_join_side(
    context: &ExecutionContext,
    node_id: NodeId,
    side: JoinSide,
    build_side: JoinSide,
    input: &PartitionedJoinSide,
    partition: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    let start = *input.offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned join side is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *input.offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape("partitioned join side is missing a partition end offset", "partition_offset_missing")
    })?;
    let len = end.saturating_sub(start);
    let overrides = execution_overrides(context);
    let frame = input.frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.hash_partition_slice",
            "failed to slice compiled bounded hash-partitioned join input",
            error,
        )
        .with_fact("side", side)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(frame.clone());
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation_kind = if side == build_side { ReservationKind::BuildSide } else { ReservationKind::ProbeSide };
    let boundary = if side == JoinSide::Left { "join_partition_left_side" } else { "join_partition_right_side" };
    let reservation = reservations.try_reserve(
        bounded_join_request(reservation_kind, boundary, bytes, reservations.budget(), node_id),
        metrics,
    )?;
    Ok(BoundedJoinSideFrame { frame, _reservation: reservation, rows: len as u64, bytes })
}

pub(super) fn materialize_partitioned_hash_equi_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinPartitionOutput> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let options = dataframe_join_options(&left_keys, &right_keys, node.spec.null_equality);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let pair_indices = match node.spec.join_type {
        JoinType::Inner => left.frame.with_execution(&overrides).inner_join_indices(&right.frame, options),
        JoinType::Left => left.frame.with_execution(&overrides).left_join_indices(&right.frame, options),
        JoinType::Full => left.frame.with_execution(&overrides).full_join_indices(&right.frame, options),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let index_bytes = pair_indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.partitioned_index_bytes",
            "failed to observe compiled partitioned join index memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_partition_indices",
            index_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(pair_indices);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let output_rows = pair_indices.len() as u64;
    let indices = CompiledJoinIndices::Pair { left: pair_indices.left().clone(), right: pair_indices.right().clone() };
    drop(pair_indices);
    let output_reservation_bytes = match indices.output_reservation_size_bytes(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let mut output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_partition_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let output = match indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection) {
        Ok(output) => output,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        reservations.try_grow(
            &mut output_reservation,
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_partition_output",
                delta,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )?;
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes, metrics);
    }
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(left);
    drop(right);
    Ok(PartitionedJoinPartitionOutput {
        frame: ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation },
        index_bytes,
        mixed_size_data_bytes: 0,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

pub(super) fn materialize_partitioned_mixed_keyed_residual_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinPartitionOutput> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    let size_data_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_partition_mixed_size_data",
            mixed_size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_mixed_join_size_data_bytes(mixed_size_data_bytes);
    let gather_map_reservation_bytes = size_data.gather_map_reservation_size_bytes(node)?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_partition_mixed_indices",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(size_data_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let indices = match size_data.materialize_indices_in(&overrides, node) {
        Ok(indices) => indices,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            reservations.release(size_data_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    reservations.release(size_data_reservation, metrics);
    let index_bytes = match indices.device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let output_reservation_bytes = match indices.output_reservation_size_bytes(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let mut output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_partition_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let output = match indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection) {
        Ok(output) => output,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        reservations.try_grow(
            &mut output_reservation,
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_partition_output",
                delta,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )?;
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes, metrics);
    }
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(left);
    drop(right);
    Ok(PartitionedJoinPartitionOutput {
        frame: ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation },
        index_bytes,
        mixed_size_data_bytes,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

pub(super) fn partitioned_streaming_probe_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::Inner)
        || (node.spec.join_type == JoinType::Left && node.spec.build_side == JoinSide::Right)
}

pub(super) fn join_keys_for_side(node: &JoinNode, side: JoinSide) -> Vec<String> {
    node.spec
        .keys
        .iter()
        .map(|key| match side {
            JoinSide::Left => key.left.name().to_string(),
            JoinSide::Right => key.right.name().to_string(),
        })
        .collect()
}

pub(super) fn execute_compiled_partitioned_hash_equi_join_probe_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_partitioned_hash_equi_join(node)?;
    if !partitioned_streaming_probe_supported(node) {
        return Err(unsupported_bounded_shape(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    let probe_estimated_bytes = stream.estimated_source_read_bytes().unwrap_or(0);
    let partition_count = partitioned_join_partition_count(context, build.input.bytes, probe_estimated_bytes);
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let overrides = execution_overrides(context);
    let build_partitioned = hash_partition_join_side(
        context,
        node_id,
        build_side,
        build.input,
        &build_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let build_partitioned_bytes = build_partitioned.reservation.bytes();
    let mut outputs = Vec::new();
    let mut summary =
        StreamedPartitionedJoinSummary::new(partition_count, partition_target_bytes, build_partitioned_bytes);
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );

    let probe_columns =
        compiled_join_input_columns(context, node, CompiledJoinExecutionStrategy::PartitionedHashEqui, probe_side)?;
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(
        context,
        stream,
        reservations,
        metrics,
        |input, reservations, metrics| {
            let probe = prepare_bounded_join_side_input_from_resident_frame(
                input,
                &probe_columns,
                probe_side,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let probe_rows = probe.rows;
            let probe_bytes = probe.bytes;
            let probe_partitioned = hash_partition_join_side(
                context,
                node_id,
                probe_side,
                probe,
                &probe_keys,
                partition_count,
                &overrides,
                reservations,
                metrics,
            )?;
            let probe_partitioned_bytes = probe_partitioned.reservation.bytes();
            summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
            for partition in 0..partition_count {
                let build_part = match slice_partition_join_side(
                    context,
                    node_id,
                    build_side,
                    build_side,
                    &build_partitioned,
                    partition,
                    reservations,
                    metrics,
                ) {
                    Ok(partition) => partition,
                    Err(error) => {
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                };
                let probe_part = match slice_partition_join_side(
                    context,
                    node_id,
                    probe_side,
                    build_side,
                    &probe_partitioned,
                    partition,
                    reservations,
                    metrics,
                ) {
                    Ok(partition) => partition,
                    Err(error) => {
                        drop(build_part);
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                };
                let (left_part, right_part) = match build_side {
                    JoinSide::Left => (build_part, probe_part),
                    JoinSide::Right => (probe_part, build_part),
                };
                match materialize_partitioned_hash_equi_join_partition(
                    context,
                    node_id,
                    node,
                    left_part,
                    right_part,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => {
                        summary.record_output(&output);
                        outputs.push(output.frame);
                    },
                    Err(error) => {
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                }
            }
            release_partitioned_join_side(probe_partitioned, reservations, metrics);
            Ok(())
        },
    );
    match stream_result {
        Ok(()) => {
            metrics.record_join_strategy_decision(
                node_id,
                &BTreeMap::from([
                    ("join_strategy", "partitioned_hash_equi".to_string()),
                    ("partition_count", partition_count.to_string()),
                    ("partition_target_bytes", partition_target_bytes.to_string()),
                    ("probe_partition_batch_count", summary.probe_partition_batch_count.to_string()),
                    ("probe_partitioned_bytes", summary.probe_partitioned_bytes.to_string()),
                ]),
            );
            release_partitioned_join_side(build_partitioned, reservations, metrics);
            Ok(BoundedCompiledJoinProbeState::PartitionedStreamed {
                build_side,
                build_rows,
                build_bytes,
                outputs,
                summary,
            })
        },
        Err(error) => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_partitioned_join_side(build_partitioned, reservations, metrics);
            Err(error)
        },
    }
}
