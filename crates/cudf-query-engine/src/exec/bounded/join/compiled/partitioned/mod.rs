use super::*;
use crate::exec::bounded::compiled::ResidentFrameStreamSource;


mod hash_equi;
mod mixed_keyed;

pub(super) use hash_equi::{
    execute_compiled_partitioned_hash_equi_join_materialize,
    execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_stream,
    partitioned_hash_equi_join_frame_stream_probe_to_payload,
};
pub(super) use mixed_keyed::{
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream,
};

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

