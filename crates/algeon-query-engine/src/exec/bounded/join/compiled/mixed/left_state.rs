use std::time::Instant;

use query_runtime::metrics::{JoinOperatorMetrics, OperatorMetrics};

use crate::exec::bounded::join::compiled::{
    BoundedCompiledJoinProbeState, BoundedCompiledMixedLeftJoinProbeState, BoundedCompiledPreparedJoinBuildState,
    BoundedExecutionAttempt, GpuDataFrame, JoinNode, JoinSide, JoinType, NodeId, ResidentFrameStreamPayload,
    ResidentGpuFrame, Result, compiled_join_input_columns, dataframe_mixed_join_options, join_index_error,
    join_side_output_upper_bound_bytes, lower_join_residual_predicate, materialize_left_only_join_output_projection,
    observe_device_memory_bytes, prepare_bounded_join_side_input_from_resident_frame,
    projected_output_columns_for_retained_sizing, residual_input_columns,
};

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_mixed_left_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    mut inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    let result: Result<_> = (|| {
        execution.context().check_cancellation("query_engine.bounded.join.mixed_left_build")?;
        let residual_columns = residual_input_columns(node)?;
        let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
        let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
        let probe_columns = compiled_join_input_columns(node, build.strategy, JoinSide::Right)?;
        let exec_runtime = execution.context().runtime()?;
        let overrides = exec_runtime.exec();
        let predicate = lower_join_residual_predicate(execution.context(), node, &residual_columns)?;
        let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
        let timer = Instant::now();
        let mut state = cudf::DataFrameMixedLeftJoinState::new(&build.input.frame, predicate, options, overrides)
            .map_err(|error| join_index_error(error, node))?;
        let mut operator_time = timer.elapsed();
        let mut state_bytes =
            state.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64;
        let mut probe_rows = 0_u64;
        let mut probe_bytes = 0_u64;
        while let Some(input) = inputs.next_frame(execution) {
            let input = input?;
            execution.context().check_cancellation("query_engine.bounded.join.mixed_left_probe")?;
            let probe =
                prepare_bounded_join_side_input_from_resident_frame(execution, input, &probe_columns, JoinSide::Right)?;
            let timer = Instant::now();
            state.consume(&probe.frame).map_err(|error| join_index_error(error, node))?;
            operator_time += timer.elapsed();
            state_bytes = state_bytes
                .max(state.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64);
            probe_rows = probe_rows.saturating_add(probe.rows);
            probe_bytes = probe_bytes.saturating_add(probe.bytes);
        }
        execution.context().check_cancellation("query_engine.bounded.join.mixed_left_finalize")?;
        let timer = Instant::now();
        // Unmatched left rows become anti output only after every right batch has completed.
        let finalized = state
            .seal(matches!(node.spec.join_type, JoinType::LeftAnti { .. }))
            .map_err(|error| join_index_error(error, node))?;
        operator_time += timer.elapsed();
        state_bytes = state_bytes
            .max(finalized.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64);
        execution.metrics().record_mixed_join_size_data_bytes(state_bytes);
        Ok(BoundedCompiledJoinProbeState::MixedLeftMatched(BoundedCompiledMixedLeftJoinProbeState {
            left: build.input,
            finalized,
            probe_rows,
            probe_bytes,
            state_bytes,
            operator_time,
        }))
    })();
    inputs.release_all(execution);
    result.map_err(|error| {
        error.with_fact_if_absent("build_side", JoinSide::Left).with_fact_if_absent("probe_side", JoinSide::Right)
    })
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_mixed_left_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledMixedLeftJoinProbeState,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledMixedLeftJoinProbeState { left, finalized, probe_rows, probe_bytes, state_bytes, operator_time } =
        probe;
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let output_rows = finalized.output_size() as u64;
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let output_reservation_bytes = join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &left.frame,
        output_rows,
        &left_projection,
        JoinSide::Left,
    )?;
    let timer = Instant::now();
    let indices = finalized.materialize_indices().map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64;
    // Reclaim the match mask and hash state before allocating the gathered output.
    drop(finalized);
    let output = materialize_left_only_join_output_projection(
        &left.frame,
        &indices,
        overrides,
        &node.spec.output_projection,
        node.spec.join_type.clone(),
    )?;
    execution.metrics().add_operator_time(operator_time + timer.elapsed());
    let output = GpuDataFrame::new(output);
    let output_bytes = observe_device_memory_bytes(&output, execution.metrics())?;
    execution.metrics().record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_mixed_left_match_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: JoinSide::Left.to_string(),
            probe_side: JoinSide::Right.to_string(),
            build_rows: left.rows,
            build_bytes: left.bytes,
            probe_rows,
            probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    Ok(ResidentGpuFrame::with_device_bytes(output, output_bytes))
}

pub(in crate::exec::bounded::join::compiled) fn release_compiled_mixed_left_join_probe_state(
    probe: BoundedCompiledMixedLeftJoinProbeState,
) {
    // Release the match state before the executor admits the next allocation boundary.
    drop(probe);
}
