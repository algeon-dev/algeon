use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState,
    CommittedPartial, ExecutionOverrides, GpuDataFrame, Instant, JoinNode, JoinSide, NodeId, QueryError,
    ResidentFrameStreamPayload, ResidentGpuFrame, Result, execution_overrides, join_side_output_upper_bound_bytes,
    materialize_join_output_projection, observe_device_memory_bytes,
    probe::reusable_hash_join_execution_strategy,
    projected_output_columns_for_retained_sizing, release_bounded_root_hash_join_build_state,
    reusable_hash_join_stream_target_output_bytes, side_metric,
    state::validate_root_hash_join,
    stream::{
        RootHashJoinMaterializeStreamSource, is_retryable_join_stream_slice_error, record_join_stream_slice_retry_error,
    },
};

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedRootHashJoinProbeState,
) -> Result<CommittedPartial> {
    validate_root_hash_join(node)?;
    let timer = state.build.timer;
    let BoundedRootHashJoinProbeState { build, probe, indices, gather_map_bytes, .. } = state;
    let output = materialize_bounded_root_hash_join_probe_chunk(
        execution,
        node_id,
        node,
        &build,
        probe,
        indices,
        gather_map_bytes,
        timer,
    )?;
    release_bounded_root_hash_join_build_state(build);
    Ok(output)
}

pub(in crate::exec::bounded) fn execute_bounded_root_hash_join_materialize_preserving_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedRootHashJoinProbeState,
) -> Result<ResidentFrameStreamPayload> {
    validate_root_hash_join(node)?;
    let stream_target_bytes = reusable_hash_join_stream_target_output_bytes(execution);
    let max_per_frame_reservation_bytes = state.materialize_admission_bytes(stream_target_bytes);
    let aggregate_reservation_bytes = state.output_reservation_bytes;
    let aggregate_num_rows = state.indices.build().len() as u64;
    let num_columns = node.spec.output_projection.len();
    Ok(ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Box::new(RootHashJoinMaterializeStreamSource::new(execution.context(), node_id, node, state)),
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded::join::hash) fn materialize_bounded_root_hash_join_probe_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_bytes: u64,
    timer: Instant,
) -> Result<CommittedPartial> {
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let build_side = build.build_side;
    let probe_side = build.probe_side;
    let hash_table_bytes = build.hash_table_bytes;
    let (left, right) = match build_side {
        JoinSide::Left => (&build.build, &probe),
        JoinSide::Right => (&probe, &build.build),
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);
    let (left_indices, right_indices) = if build.left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let expected_output_rows = indices.build().len() as u64;
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let output = match materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
        node.spec.datafusion_order_preserved_side,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(error);
        },
    };
    execution.metrics_mut().operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    execution.metrics_mut().record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: reusable_hash_join_execution_strategy(node, build).to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));

    Ok(ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes))
}

pub(in crate::exec::bounded::join::hash) struct RootHashJoinStreamChunkOutput {
    pub(in crate::exec::bounded::join::hash) frame: ResidentGpuFrame,
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_rows: u64,
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded::join::hash) fn materialize_bounded_root_hash_join_index_slice(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    left_indices: &cudf::DataFrameJoinIndices,
    right_indices: &cudf::DataFrameJoinIndices,
    offset: usize,
    len: usize,
    overrides: &ExecutionOverrides<'_>,
) -> Result<Vec<RootHashJoinStreamChunkOutput>> {
    match materialize_bounded_root_hash_join_index_slice_once(
        execution,
        node_id,
        node,
        left,
        right,
        left_indices,
        right_indices,
        offset,
        len,
        overrides,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if len > 1 && is_retryable_join_stream_slice_error(&error) => {
            record_join_stream_slice_retry_error(&error, execution.metrics_mut());
            crate::exec::retry::backoff_before_retry(
                execution.context(),
                "nexus_query_engine.exec.bounded.hash_join_slice_retry",
            )?;
            let left_len = len / 2;
            let right_len = len - left_len;
            let mut outputs = materialize_bounded_root_hash_join_index_slice(
                execution,
                node_id,
                node,
                left,
                right,
                left_indices,
                right_indices,
                offset,
                left_len,
                overrides,
            )?;
            match materialize_bounded_root_hash_join_index_slice(
                execution,
                node_id,
                node,
                left,
                right,
                left_indices,
                right_indices,
                offset + left_len,
                right_len,
                overrides,
            ) {
                Ok(mut right_outputs) => {
                    outputs.append(&mut right_outputs);
                    Ok(outputs)
                },
                Err(error) => {
                    release_root_hash_join_stream_outputs(outputs.into_iter().map(|output| output.frame).collect());
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded::join::hash) fn materialize_bounded_root_hash_join_index_slice_once(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    left_indices: &cudf::DataFrameJoinIndices,
    right_indices: &cudf::DataFrameJoinIndices,
    offset: usize,
    len: usize,
    overrides: &ExecutionOverrides<'_>,
) -> Result<RootHashJoinStreamChunkOutput> {
    let left_slice = match left_indices.indices().slice(offset, len) {
        Ok(slice) => cudf::DataFrameJoinIndices::from_series(slice),
        Err(error) => {
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.reusable_hash_stream_left_indices",
                "failed to slice reusable hash join left gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };
    let right_slice = match right_indices.indices().slice(offset, len) {
        Ok(slice) => cudf::DataFrameJoinIndices::from_series(slice),
        Err(error) => {
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.reusable_hash_stream_right_indices",
                "failed to slice reusable hash join right gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type));
        },
    };
    let expected_output_rows = len as u64;
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let output = match materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_slice,
        &right_slice,
        overrides,
        &node.spec.output_projection,
        node.spec.join_type,
        node.spec.datafusion_order_preserved_side,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(error);
        },
    };
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    Ok(RootHashJoinStreamChunkOutput {
        frame: ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes),
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

pub(in crate::exec::bounded::join::hash) fn release_root_hash_join_stream_outputs(mut outputs: Vec<ResidentGpuFrame>) {
    outputs.clear();
}
