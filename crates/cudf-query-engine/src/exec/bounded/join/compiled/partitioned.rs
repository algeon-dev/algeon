use super::*;
use crate::exec::bounded::compiled::ResidentFrameStreamSource;

pub(super) fn execute_compiled_partitioned_hash_equi_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        false,
        reservations,
        metrics,
    )? {
        BoundedCompiledJoinMaterializeOutput::Frame(frame) => Ok(*frame),
        BoundedCompiledJoinMaterializeOutput::Stream(_) | BoundedCompiledJoinMaterializeOutput::LazyStream(_) => {
            unreachable!("non-preserving partitioned join returns frame")
        },
    }
}

pub(super) fn execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        true,
        reservations,
        metrics,
    )
}

fn execute_compiled_partitioned_hash_equi_join_materialize_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    preserve_stream: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
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
    let mut stream_outputs = Some(outputs);
    let output = if preserve_stream {
        None
    } else {
        let outputs = stream_outputs.take().expect("partitioned join outputs should be available");
        let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
        let output_bytes = output.reservation.bytes();
        total_output_bytes = total_output_bytes.max(output_bytes);
        Some(output)
    };
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
    if let Some(output) = output {
        Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
    } else {
        Ok(BoundedCompiledJoinMaterializeOutput::Stream(
            stream_outputs.expect("partitioned stream outputs should be available"),
        ))
    }
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
    match execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        false,
        reservations,
        metrics,
    )? {
        BoundedCompiledJoinMaterializeOutput::Frame(frame) => Ok(*frame),
        BoundedCompiledJoinMaterializeOutput::Stream(_) | BoundedCompiledJoinMaterializeOutput::LazyStream(_) => {
            unreachable!("non-preserving partitioned mixed join returns frame")
        },
    }
}

pub(super) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        true,
        reservations,
        metrics,
    )
}

fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    preserve_stream: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_mixed_keyed_residual_join(node)?;
    let original_left_rows = left.rows;
    let original_left_bytes = left.bytes;
    let original_right_rows = right.rows;
    let original_right_bytes = right.bytes;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let partition_count =
        partitioned_mixed_join_partition_count(context, left.rows, left.bytes, right.rows, right.bytes);
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

    if preserve_stream {
        let forecast = match forecast_partitioned_mixed_join_stream(
            context,
            node_id,
            node,
            &left_partitioned,
            &right_partitioned,
            partition_count,
            reservations,
            metrics,
        ) {
            Ok(forecast) => forecast,
            Err(error) => {
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        let left_partitioned_bytes = left_partitioned.reservation.bytes();
        let right_partitioned_bytes = right_partitioned.reservation.bytes();
        let stream = PartitionedMixedJoinStreamSource {
            context: context.clone(),
            node_id,
            node: node.clone(),
            left_partitioned: Some(left_partitioned),
            right_partitioned: Some(right_partitioned),
            partition_count,
            next_partition: 0,
            original_left_rows,
            original_left_bytes,
            original_right_rows,
            original_right_bytes,
            left_partitioned_bytes,
            right_partitioned_bytes,
            total_index_bytes: 0,
            total_mixed_size_data_bytes: 0,
            total_output_bytes: 0,
            total_output_rows: 0,
            total_output_reservation_bytes: 0,
            metrics_recorded: false,
        };
        let payload = ResidentFrameStreamPayload::from_producer(
            forecast.max_per_frame_reservation_bytes,
            forecast.aggregate_reservation_bytes,
            forecast.aggregate_num_rows,
            node.spec.output_projection.len(),
            Vec::new(),
            Box::new(stream),
        );
        return Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(payload));
    }

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
    Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
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

fn partitioned_mixed_join_partition_count(
    context: &ExecutionContext,
    left_rows: u64,
    left_bytes: u64,
    right_rows: u64,
    right_bytes: u64,
) -> usize {
    let byte_partitions = partitioned_join_partition_count(context, left_bytes, right_bytes);
    let row_basis = left_rows.max(right_rows).max(1);
    let row_partitions = row_basis.saturating_add(PARTITIONED_MIXED_JOIN_TARGET_ROWS.saturating_sub(1))
        / PARTITIONED_MIXED_JOIN_TARGET_ROWS;
    let row_partitions = usize::try_from(row_partitions.clamp(2, PARTITIONED_MIXED_JOIN_MAX_PARTITIONS as u64))
        .unwrap_or(PARTITIONED_MIXED_JOIN_MAX_PARTITIONS);
    byte_partitions.max(row_partitions)
}

const PARTITIONED_MIXED_JOIN_TARGET_ROWS: u64 = 50_000;
const PARTITIONED_MIXED_JOIN_MAX_PARTITIONS: usize = 1024;
const PARTITIONED_HASH_FRAME_STREAM_MIN_PARTITIONS: usize = 2;

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
    const MAX_TARGET_BYTES: u64 = 512 * 1024 * 1024;
    (device_budget_bytes / 8).clamp(MIN_TARGET_BYTES, MAX_TARGET_BYTES)
}

pub(super) fn partitioned_join_reusable_rejected_reason(
    build_rows: u64,
    build_bytes: u64,
    probe_bytes: u64,
) -> &'static str {
    if build_rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD || build_bytes >= LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD {
        "build_side_too_large"
    } else if probe_bytes >= LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD
        && build_rows > SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD
        && build_bytes > SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD
    {
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
    reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics)?;
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
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
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
    super::super::common::trace_preflight_entry(
        "partitioned_keyed_partition",
        "partitioned_keyed",
        node_id,
        reservations,
        0,
        output_rows,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let output_reservation_bytes = match indices.output_upper_bound_size_bytes(
        context,
        node_id,
        &left.frame,
        &right.frame,
        output_rows,
        &node.spec.output_projection,
        reservations,
        metrics,
    ) {
        Ok(bytes) => {
            super::super::common::trace_preflight_success("partitioned_keyed_partition", node_id, reservations, bytes);
            bytes
        },
        Err(error) => {
            super::super::common::trace_preflight_failed("partitioned_keyed_partition", node_id, reservations, &error);
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
    reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics)?;
    let output = match indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
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
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        reservations.release(output_reservation, metrics);
        drop(left);
        drop(right);
        return Err(error);
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
        JoinType::Cross | JoinType::LeftMark => unreachable!("validated above"),
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
    let forecast_output_reservation_bytes = match partitioned_mixed_join_output_upper_bound_bytes(
        context,
        node_id,
        node,
        &left,
        &right,
        output_rows,
        reservations,
        metrics,
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
    super::super::common::trace_preflight_entry(
        "partitioned_mixed_partition",
        "partitioned_mixed",
        node_id,
        reservations,
        index_bytes,
        output_rows,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let output_reservation_bytes = forecast_output_reservation_bytes;
    super::super::common::trace_preflight_success(
        "partitioned_mixed_partition",
        node_id,
        reservations,
        output_reservation_bytes,
    );
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
    reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics)?;
    let output = match indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
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
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        reservations.release(output_reservation, metrics);
        drop(left);
        drop(right);
        return Err(error);
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

fn partitioned_mixed_join_output_upper_bound_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    expected_output_rows: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<u64> {
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_bytes = join_side_output_upper_bound_bytes(
        context,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    )?;
    let right_bytes = join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    Ok(left_bytes.saturating_add(right_bytes))
}

struct PartitionedMixedJoinStreamForecast {
    max_per_frame_reservation_bytes: u64,
    aggregate_reservation_bytes: u64,
    aggregate_num_rows: u64,
}

fn forecast_partitioned_mixed_join_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    partition_count: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedMixedJoinStreamForecast> {
    let mut max_per_frame_reservation_bytes = 0u64;
    let mut aggregate_reservation_bytes = 0u64;
    let mut aggregate_num_rows = 0u64;
    for partition in 0..partition_count {
        let left_part = slice_partition_join_side(
            context,
            node_id,
            JoinSide::Left,
            node.spec.build_side,
            left_partitioned,
            partition,
            reservations,
            metrics,
        )?;
        let right_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Right,
            node.spec.build_side,
            right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                return Err(error);
            },
        };
        let forecast = match forecast_partitioned_mixed_join_partition(
            context,
            node_id,
            node,
            &left_part,
            &right_part,
            reservations,
            metrics,
        ) {
            Ok(forecast) => forecast,
            Err(error) => {
                drop(left_part);
                drop(right_part);
                return Err(error);
            },
        };
        aggregate_num_rows = aggregate_num_rows.saturating_add(forecast.output_rows);
        aggregate_reservation_bytes = aggregate_reservation_bytes.saturating_add(forecast.output_reservation_bytes);
        max_per_frame_reservation_bytes = max_per_frame_reservation_bytes.max(forecast.output_reservation_bytes);
        drop(left_part);
        drop(right_part);
    }
    Ok(PartitionedMixedJoinStreamForecast {
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
    })
}

struct PartitionedMixedJoinPartitionForecast {
    output_rows: u64,
    output_reservation_bytes: u64,
}

fn forecast_partitioned_mixed_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedMixedJoinPartitionForecast> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
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
        JoinType::Cross | JoinType::LeftMark => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    let size_data_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_partition_mixed_size_data_forecast",
            mixed_size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    drop(size_data);
    reservations.release(size_data_reservation, metrics);
    let output_reservation_bytes = partitioned_mixed_join_output_upper_bound_bytes(
        context,
        node_id,
        node,
        left,
        right,
        output_rows,
        reservations,
        metrics,
    )?;
    Ok(PartitionedMixedJoinPartitionForecast { output_rows, output_reservation_bytes })
}

struct PartitionedMixedJoinStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    left_partitioned: Option<PartitionedJoinSide>,
    right_partitioned: Option<PartitionedJoinSide>,
    partition_count: usize,
    next_partition: usize,
    original_left_rows: u64,
    original_left_bytes: u64,
    original_right_rows: u64,
    original_right_bytes: u64,
    left_partitioned_bytes: u64,
    right_partitioned_bytes: u64,
    total_index_bytes: u64,
    total_mixed_size_data_bytes: u64,
    total_output_bytes: u64,
    total_output_rows: u64,
    total_output_reservation_bytes: u64,
    metrics_recorded: bool,
}

impl PartitionedMixedJoinStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        metrics.record_join_strategy_decision(
            self.node_id,
            &BTreeMap::from([
                ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
                ("partition_count", self.partition_count.to_string()),
                ("partition_target_bytes", partitioned_join_target_bytes_for_context(&self.context).to_string()),
                ("build_partition_batch_count", self.partition_count.to_string()),
                ("probe_partition_batch_count", self.partition_count.to_string()),
                (
                    "build_partitioned_bytes",
                    side_metric(self.node.spec.build_side, self.left_partitioned_bytes, self.right_partitioned_bytes)
                        .to_string(),
                ),
                (
                    "probe_partitioned_bytes",
                    side_metric(
                        probe_side(self.node.spec.build_side),
                        self.left_partitioned_bytes,
                        self.right_partitioned_bytes,
                    )
                    .to_string(),
                ),
            ]),
        );
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            self.node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: self.node.spec.join_type.to_string(),
                execution_strategy: "bounded_partitioned_mixed_keyed_residual_join".to_string(),
                key_count: self.node.spec.keys.len(),
                residual_predicate: true,
                null_equality: self.node.spec.null_equality.to_string(),
                build_side: self.node.spec.build_side.to_string(),
                probe_side: probe_side(self.node.spec.build_side).to_string(),
                build_rows: side_metric(self.node.spec.build_side, self.original_left_rows, self.original_right_rows),
                build_bytes: side_metric(
                    self.node.spec.build_side,
                    self.original_left_bytes,
                    self.original_right_bytes,
                ),
                probe_rows: side_metric(
                    probe_side(self.node.spec.build_side),
                    self.original_left_rows,
                    self.original_right_rows,
                ),
                probe_bytes: side_metric(
                    probe_side(self.node.spec.build_side),
                    self.original_left_bytes,
                    self.original_right_bytes,
                ),
                hash_table_bytes: 0,
                filtered_probe_state_bytes: 0,
                mixed_size_data_bytes: self.total_mixed_size_data_bytes,
                gather_map_bytes: self.total_index_bytes,
                output_reservation_bytes: self.total_output_reservation_bytes,
                output_rows: self.total_output_rows,
                output_bytes: self.total_output_bytes,
            },
        ));
    }
}

impl ResidentFrameStreamSource for PartitionedMixedJoinStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        if self.next_partition >= self.partition_count {
            self.record_success_metrics(metrics);
            self.release(reservations, metrics);
            return None;
        }
        let partition = self.next_partition;
        self.next_partition += 1;
        let left_partitioned =
            self.left_partitioned.as_ref().expect("left partitioned side released before stream end");
        let right_partitioned =
            self.right_partitioned.as_ref().expect("right partitioned side released before stream end");
        let left_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            JoinSide::Left,
            self.node.spec.build_side,
            left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => return Some(Err(error)),
        };
        let right_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            JoinSide::Right,
            self.node.spec.build_side,
            right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                return Some(Err(error));
            },
        };
        match materialize_partitioned_mixed_keyed_residual_join_partition(
            &self.context,
            self.node_id,
            &self.node,
            left_part,
            right_part,
            reservations,
            metrics,
        ) {
            Ok(output) => {
                self.total_index_bytes = self.total_index_bytes.saturating_add(output.index_bytes);
                self.total_mixed_size_data_bytes =
                    self.total_mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
                self.total_output_rows = self.total_output_rows.saturating_add(output.output_rows);
                self.total_output_bytes = self.total_output_bytes.saturating_add(output.output_bytes);
                self.total_output_reservation_bytes =
                    self.total_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                Some(Ok(output.frame))
            },
            Err(error) => Some(Err(error)),
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        if let Some(left) = self.left_partitioned.take() {
            release_partitioned_join_side(left, reservations, metrics);
        }
        if let Some(right) = self.right_partitioned.take() {
            release_partitioned_join_side(right, reservations, metrics);
        }
    }
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
    let partition_count = partitioned_join_partition_count(context, build.input.bytes, probe_estimated_bytes)
        .max(PARTITIONED_HASH_FRAME_STREAM_MIN_PARTITIONS);
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

pub(super) fn execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
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
    let probe_estimated_bytes = inputs.max_per_frame_reservation_bytes();
    let partition_count = partitioned_join_partition_count(context, build.input.bytes, probe_estimated_bytes)
        .max(PARTITIONED_HASH_FRAME_STREAM_MIN_PARTITIONS);
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

    let probe_columns = match compiled_join_input_columns(
        context,
        node,
        CompiledJoinExecutionStrategy::PartitionedHashEqui,
        probe_side,
    ) {
        Ok(columns) => columns,
        Err(error) => {
            inputs.release_all(reservations, metrics);
            release_partitioned_join_side(build_partitioned, reservations, metrics);
            return Err(error);
        },
    };
    Ok(BoundedCompiledJoinProbeState::PartitionedLazyStream(PartitionedHashEquiJoinFrameStreamProbeState {
        build_side,
        build_rows,
        build_bytes,
        build_partitioned,
        build_partitioned_bytes,
        inputs,
        partition_count,
        partition_target_bytes,
        probe_columns,
        probe_keys,
    }))
}

pub(super) fn partitioned_hash_equi_join_frame_stream_probe_to_payload(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_per_frame_reservation_bytes = probe.partition_target_bytes;
    let aggregate_reservation_bytes = probe.inputs.aggregate_reservation_bytes();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = PartitionedHashEquiJoinProbeStreamSource {
        context: context.clone(),
        node_id,
        node: node.clone(),
        build_side: probe.build_side,
        build_rows: probe.build_rows,
        build_bytes: probe.build_bytes,
        build_partitioned: Some(probe.build_partitioned),
        inputs: Some(probe.inputs),
        partition_count: probe.partition_count,
        probe_columns: probe.probe_columns,
        probe_keys: probe.probe_keys,
        active_probe_partitioned: None,
        next_partition: 0,
        summary: StreamedPartitionedJoinSummary::new(
            probe.partition_count,
            probe.partition_target_bytes,
            probe.build_partitioned_bytes,
        ),
        metrics_recorded: false,
    };
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
        Box::new(source),
    )
}

struct PartitionedHashEquiJoinProbeStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    build_partitioned: Option<PartitionedJoinSide>,
    inputs: Option<ResidentFrameStreamPayload>,
    partition_count: usize,
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
    active_probe_partitioned: Option<PartitionedJoinSide>,
    next_partition: usize,
    summary: StreamedPartitionedJoinSummary,
    metrics_recorded: bool,
}

impl PartitionedHashEquiJoinProbeStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        record_streamed_partitioned_hash_join_metrics(
            self.node_id,
            &self.node,
            self.build_side,
            probe_side(self.build_side),
            self.build_rows,
            self.build_bytes,
            &self.summary,
            metrics,
        );
    }

    fn load_next_probe_chunk(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<bool>> {
        let inputs = self.inputs.as_mut()?;
        let mut input = match inputs.next_yielded_frame(reservations, metrics) {
            Some(Ok(input)) => input,
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        };
        let input = input.take_for_release();
        let probe_side = probe_side(self.build_side);
        let probe = match prepare_bounded_join_side_input_from_resident_frame(
            input,
            &self.probe_columns,
            probe_side,
            self.node_id,
            self.build_side,
            reservations,
            metrics,
        ) {
            Ok(probe) => probe,
            Err(error) => return Some(Err(error)),
        };
        let probe_rows = probe.rows;
        let probe_bytes = probe.bytes;
        let overrides = execution_overrides(&self.context);
        let probe_partitioned = match hash_partition_join_side(
            &self.context,
            self.node_id,
            probe_side,
            probe,
            &self.probe_keys,
            self.partition_count,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(partitioned) => partitioned,
            Err(error) => return Some(Err(error)),
        };
        let probe_partitioned_bytes = probe_partitioned.reservation.bytes();
        self.summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
        self.active_probe_partitioned = Some(probe_partitioned);
        self.next_partition = 0;
        Some(Ok(true))
    }

    fn materialize_next_active_partition(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Result<ResidentGpuFrame> {
        let partition = self.next_partition;
        self.next_partition += 1;
        let is_last_active_partition = self.next_partition >= self.partition_count;
        let build_partitioned =
            self.build_partitioned.as_ref().expect("partitioned hash join build side released before stream end");
        let probe_partitioned = self
            .active_probe_partitioned
            .as_ref()
            .expect("partitioned hash join probe chunk missing before stream end");
        let probe_side = probe_side(self.build_side);
        let build_part = slice_partition_join_side(
            &self.context,
            self.node_id,
            self.build_side,
            self.build_side,
            build_partitioned,
            partition,
            reservations,
            metrics,
        )?;
        let probe_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            probe_side,
            self.build_side,
            probe_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(build_part);
                return Err(error);
            },
        };
        let (left_part, right_part) = match self.build_side {
            JoinSide::Left => (build_part, probe_part),
            JoinSide::Right => (probe_part, build_part),
        };
        if is_last_active_partition {
            self.release_active_probe_partitioned(reservations, metrics);
        }
        let output = materialize_partitioned_hash_equi_join_partition(
            &self.context,
            self.node_id,
            &self.node,
            left_part,
            right_part,
            reservations,
            metrics,
        )
        .map_err(|error| error.with_fact("partition", partition))?;
        self.summary.record_output(&output);
        Ok(output.frame)
    }

    fn release_active_probe_partitioned(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        if let Some(probe_partitioned) = self.active_probe_partitioned.take() {
            release_partitioned_join_side(probe_partitioned, reservations, metrics);
        }
    }
}

impl ResidentFrameStreamSource for PartitionedHashEquiJoinProbeStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if self.active_probe_partitioned.is_some() {
                if self.next_partition < self.partition_count {
                    return Some(self.materialize_next_active_partition(reservations, metrics));
                }
                self.release_active_probe_partitioned(reservations, metrics);
            }

            match self.load_next_probe_chunk(reservations, metrics) {
                Some(Ok(true)) => {},
                Some(Ok(false)) => unreachable!("partitioned hash stream probe chunk loading is boolean-true only"),
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    self.record_success_metrics(metrics);
                    self.release(reservations, metrics);
                    return None;
                },
            }
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        self.release_active_probe_partitioned(reservations, metrics);
        if let Some(input) = self.inputs.take() {
            input.release_all(reservations, metrics);
        }
        if let Some(build_partitioned) = self.build_partitioned.take() {
            release_partitioned_join_side(build_partitioned, reservations, metrics);
        }
    }
}
