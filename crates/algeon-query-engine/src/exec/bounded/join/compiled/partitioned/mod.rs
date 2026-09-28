use super::{
    BTreeMap, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedCompiledPartitionedJoinBuildState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, BoundedPartitionRowPolicy,
    BoundedPartitionTargetPolicy, CompiledEqualityJoinPreflightIndices, CompiledJoinExecutionStrategy,
    CompiledMixedSizeData, DataFrame, ExecutionContext, ExecutionMetrics, ExecutionOverrides, GpuDataFrame, Instant,
    JoinMaterializationTxn, JoinNode, JoinSide, JoinType, KeyedHashJoinStrategyOptions, MemoryControlPolicy, NodeId,
    PartitionedHashEquiJoinFrameStreamProbeState, PartitionedJoinPartitionOutput, PartitionedJoinSide, QueryError,
    QueryErrorCode, ResidentFrameStreamPayload, ResidentGpuFrame, Result, RetryFacts, StreamedPartitionedJoinSummary,
    VecDeque, bounded_invariant_violation, compiled_join_input_columns, concatenate_streamed_hash_join_outputs,
    cudf_query_error, dataframe_mixed_join_options, execution_overrides, join_index_error,
    join_side_output_upper_bound_bytes, lower_join_residual_predicate,
    materialize_bounded_join_leaf_local_parquet_source_stream,
    materialize_compiled_keyed_equality_join_indices_with_preflight, observe_device_memory_bytes,
    prepare_bounded_join_side_input_from_resident_frame, probe_side, projected_output_columns_for_retained_sizing,
    record_streamed_partitioned_hash_join_metrics, release_bounded_join_side_frame, residual_input_columns,
    side_metric, stream_bounded_join_leaf_local_parquet_source, stream_bounded_join_leaf_local_parquet_source_serial,
    validate_executable_join_subset,
};
use crate::{
    error::ResourceBoundary,
    exec::{
        bounded::compiled::ResidentFrameStreamSource,
        retry::{ResourceFailureRecovery, resource_failure_recovery},
    },
};

mod build;
mod driver;
mod hash_equi;
mod mixed_keyed;

pub(in crate::exec::bounded) use build::{
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    PartitionedJoinBuildCommitment, consume_bounded_join_build_input_stream, partitioned_join_build_commitment,
};
use driver::*;
pub(super) use hash_equi::{
    execute_compiled_partitioned_hash_equi_join_materialize,
    execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_split_build_frame_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_stream, parallel_partitioned_lazy_drain,
    partitioned_hash_equi_join_frame_stream_probe_to_payload, partitioned_parallel_probe_supported,
};
pub(super) use mixed_keyed::{
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream,
};

#[derive(Clone, Copy)]
pub(super) struct PartitionedJoinRowRange {
    pub(super) offset: usize,
    pub(super) len: usize,
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
    Err(bounded_invariant_violation(
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
    Err(bounded_invariant_violation(
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
    let MemoryControlPolicy { device_budget_bytes, .. } = context.memory_control_policy();
    let target_partition_bytes = partitioned_join_target_partition_bytes(device_budget_bytes);
    let basis = left_bytes.max(right_bytes).max(1);
    let partitions = basis.saturating_add(target_partition_bytes.saturating_sub(1)) / target_partition_bytes.max(1);
    partitions.clamp(2, 1024) as usize
}

fn partitioned_mixed_join_partition_count(
    context: &ExecutionContext,
    left_rows: u64,
    left_bytes: u64,
    right_rows: u64,
    right_bytes: u64,
) -> usize {
    let byte_partitions = partitioned_join_partition_count(context, left_bytes, right_bytes);
    let target_rows = partitioned_mixed_join_target_rows(context);
    BoundedPartitionRowPolicy::default().mixed_join_partition_count(byte_partitions, target_rows, left_rows, right_rows)
}

fn partitioned_mixed_join_target_rows(context: &ExecutionContext) -> u64 {
    BoundedPartitionRowPolicy::default().mixed_join_target_rows(context.memory_control_policy())
}

pub(super) fn partitioned_join_target_bytes_for_context(context: &ExecutionContext) -> u64 {
    let MemoryControlPolicy { device_budget_bytes, .. } = context.memory_control_policy();
    partitioned_join_target_partition_bytes(device_budget_bytes)
}

pub(super) fn partitioned_join_target_partition_bytes(device_budget_bytes: u64) -> u64 {
    BoundedPartitionTargetPolicy::default().partitioned_join_target_bytes(device_budget_bytes)
}

pub(super) fn partitioned_join_reusable_rejected_reason(
    build_rows: u64,
    build_bytes: u64,
    probe_bytes: u64,
    options: KeyedHashJoinStrategyOptions,
) -> &'static str {
    if build_rows >= options.large_keyed_hash_build_row_threshold()
        || build_bytes >= options.large_keyed_hash_build_bytes_threshold()
    {
        "build_side_too_large"
    } else if probe_bytes >= options.large_keyed_hash_probe_bytes_threshold()
        && build_rows > options.small_reusable_hash_build_row_threshold()
        && build_bytes > options.small_reusable_hash_build_bytes_threshold()
    {
        "probe_side_too_large"
    } else {
        "partitioned_strategy_selected"
    }
}

pub(super) fn hash_partition_join_side(
    execution: &mut BoundedExecutionAttempt<'_>,
    side: JoinSide,
    input: BoundedJoinSideFrame,
    keys: &[String],
    partition_count: usize,
    overrides: &ExecutionOverrides<'_>,
) -> Result<PartitionedJoinSide> {
    let BoundedJoinSideFrame { frame, .. } = input;
    let partitioned = match frame
        .with_execution(overrides)
        .hash_partition_by(keys.iter().map(String::as_str), partition_count)
        .map_err(|error| {
            with_partition_side(
                cudf_query_error(
                    "query_engine.bounded.join.hash_partition",
                    "failed to hash-partition compiled bounded join input",
                    error,
                ),
                side,
                "join_hash_partition",
            )
            .with_fact("key_count", keys.len())
            .with_fact("partition_count", partition_count)
        }) {
        Ok(partitioned) => partitioned,
        Err(error) => {
            return Err(error);
        },
    };
    let (partitioned, offsets) = partitioned.into_parts();
    let partitioned_frame = GpuDataFrame::new(partitioned.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    Ok(PartitionedJoinSide::Contiguous { frame: partitioned, device_bytes: partitioned_bytes, offsets })
}

/// Records which join input's working set failed. A resource failure carries
/// it as its typed boundary, which the repartition decision reads.
pub(super) fn with_partition_side(error: QueryError, side: JoinSide, memory_boundary: &'static str) -> QueryError {
    if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
        return error.with_fact("side", side);
    }
    error.with_resource_boundary(ResourceBoundary {
        memory_boundary,
        stage: "partitioned_join",
        node_id: None,
        side: Some(side),
        resume: None,
    })
}

pub(super) fn release_partitioned_join_side(side: PartitionedJoinSide) {
    // Release all partitions before the caller starts another allocation on the query domain.
    drop(side);
}

pub(super) fn slice_partition_join_side_range(
    execution: &mut BoundedExecutionAttempt<'_>,
    side: JoinSide,
    input: &PartitionedJoinSide,
    partition: usize,
    range: PartitionedJoinRowRange,
) -> Result<BoundedJoinSideFrame> {
    let (partition_frame, partition_start, partition_len) = partition_bounds(input, partition)?;
    if range.offset > partition_len || range.len > partition_len.saturating_sub(range.offset) {
        return Err(bounded_invariant_violation(
            "partitioned join retry range exceeds partition bounds",
            "partition_retry_range_invalid",
        )
        .with_fact("partition", partition)
        .with_fact("partition_rows", partition_len)
        .with_fact("range_offset", range.offset)
        .with_fact("range_rows", range.len));
    }
    if matches!(input, PartitionedJoinSide::Split { .. }) && range.offset == 0 && range.len == partition_len {
        // A split partition already is the whole join input for this partition;
        // sharing its columns avoids one partition-sized copy per probe chunk.
        let frame = partition_frame.clone();
        let bytes = observe_device_memory_bytes(&GpuDataFrame::new(frame.clone()), execution.metrics())?;
        return Ok(BoundedJoinSideFrame { frame, rows: partition_len as u64, bytes });
    }
    let start = partition_start.saturating_add(range.offset);
    let len = range.len;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let frame = partition_frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        with_partition_side(
            cudf_query_error(
                "query_engine.bounded.join.hash_partition_slice",
                "failed to slice compiled bounded hash-partitioned join input",
                error,
            ),
            side,
            "join_partition_slice",
        )
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(frame.clone());
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics())?;
    Ok(BoundedJoinSideFrame { frame, rows: len as u64, bytes })
}

pub(super) fn partition_row_range(input: &PartitionedJoinSide, partition: usize) -> Result<PartitionedJoinRowRange> {
    let (_, _, len) = partition_bounds(input, partition)?;
    Ok(PartitionedJoinRowRange { offset: 0, len })
}

fn partition_bounds(input: &PartitionedJoinSide, partition: usize) -> Result<(&DataFrame, usize, usize)> {
    match input {
        PartitionedJoinSide::Contiguous { frame, offsets, .. } => {
            let start = *offsets.get(partition).ok_or_else(|| {
                bounded_invariant_violation(
                    "partitioned join side is missing a partition start offset",
                    "partition_offset_missing",
                )
            })?;
            let end = *offsets.get(partition + 1).ok_or_else(|| {
                bounded_invariant_violation(
                    "partitioned join side is missing a partition end offset",
                    "partition_offset_missing",
                )
            })?;
            Ok((frame, start, end.saturating_sub(start)))
        },
        PartitionedJoinSide::Split { partitions, .. } => {
            let frame = partitions.get(partition).ok_or_else(|| {
                bounded_invariant_violation(
                    "partitioned join side is missing a partition frame",
                    "partition_offset_missing",
                )
            })?;
            Ok((frame, 0, frame.num_rows()))
        },
    }
}

pub(super) fn partitioned_streaming_probe_supported(node: &JoinNode) -> bool {
    super::stateless_streaming_probe_supported(node)
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
