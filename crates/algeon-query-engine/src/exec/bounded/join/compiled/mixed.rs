use std::time::Instant;

mod left_state;
pub(super) use left_state::*;

use super::{
    BoundedCompiledJoinProbeState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, CompiledJoinMaterializationRequest,
    CompiledMixedSizeData, JoinNode, JoinSide, JoinType, NodeId, ResidentGpuFrame, Result, bounded_invariant_violation,
    compiled_join_input_columns, dataframe_mixed_join_options, execution_overrides, join_index_error,
    lower_join_residual_predicate, materialize_compiled_indices_output, orient_prepared_inputs,
    prepare_bounded_join_side_input_from_resident_frame, probe_side, release_bounded_join_side_frame,
    residual_input_columns, stream_bounded_join_leaf_local_parquet_source_serial,
};

pub(super) fn execute_compiled_mixed_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let predicate = lower_join_residual_predicate(execution.context(), node, &residual_columns)?;
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
        JoinType::LeftAnti { .. } => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross | JoinType::LeftMark { .. } => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let expected_output_rows = size_data.output_size() as u64;
    let size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    execution.metrics().record_mixed_join_size_data_bytes(size_data_bytes);
    let indices = size_data.materialize_indices_in(&overrides, node)?;
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    materialize_compiled_indices_output(
        execution,
        node,
        CompiledJoinMaterializationRequest {
            node_id,
            execution_strategy: "bounded_mixed_keyed_residual_join",
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            build_side: node.spec.build_side,
            probe_side: probe_side(node.spec.build_side),
            left,
            right,
            indices,
            mixed_size_data_bytes: size_data_bytes,
            filtered_probe_state_bytes: 0,
            expected_output_rows,
            gather_map_bytes,
            timer,
            one_shot_index_bytes: None,
        },
    )
}

pub(super) fn execute_compiled_mixed_join_probe_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<BoundedCompiledJoinProbeState> {
    if node.spec.join_type != JoinType::Inner {
        release_bounded_join_side_frame(build.input);
        return Err(bounded_invariant_violation(
            "streamed compiled mixed keyed residual join probe supports inner joins only",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", &node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }

    let probe_side = probe_side(build.side);
    let probe_columns = compiled_join_input_columns(node, build.strategy, probe_side)?;
    let mut outputs = Vec::new();
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(execution, stream, |input, execution| {
        let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, &probe_columns, probe_side)?;
        let output =
            materialize_compiled_mixed_join_probe_chunk(execution, node_id, node, build.side, &build.input, probe)?;
        outputs.push(output);
        Ok(())
    });
    match stream_result {
        Ok(()) => Ok(BoundedCompiledJoinProbeState::MixedStreamed { build, outputs }),
        Err(error) => {
            release_bounded_join_side_frame(build.input);
            Err(error)
        },
    }
}

fn materialize_compiled_mixed_join_probe_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build: &BoundedJoinSideFrame,
    probe: BoundedJoinSideFrame,
) -> Result<ResidentGpuFrame> {
    let build = alias_compiled_mixed_join_build_chunk(build);
    let probe_side = probe_side(build_side);
    let (left, right) = orient_prepared_inputs(build_side, build, probe_side, probe)?;
    execute_compiled_mixed_join_materialize(execution, node_id, node, left, right)
}

fn alias_compiled_mixed_join_build_chunk(build: &BoundedJoinSideFrame) -> BoundedJoinSideFrame {
    BoundedJoinSideFrame { frame: build.frame.clone(), rows: build.rows, bytes: build.bytes }
}
