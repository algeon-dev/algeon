//! Keyed joins whose residual predicate is evaluated while matching (`MixedKeyedResidual`).
//!
//! The planner picks this strategy for keyed inner, left, full, semi and anti joins whose residual
//! can run inside the matching kernel. For an inner join, a residual containing CASE, IS DISTINCT
//! FROM, IS NOT DISTINCT FROM or a decimal division filters after matching and goes to the one-shot
//! path (`residual.rs`); other join types reject such a residual during planning. Large inputs use
//! the partitioned variant instead (`partitioned/mixed_keyed.rs`).
//!
//! The cuDF primitive is the mixed join: it hashes the equality keys and evaluates the lowered AST
//! predicate on each candidate pair. The call has two steps. `mixed_*_join_size_data` counts the
//! matches and returns a size-data object, and its `materialize_indices` produces the gather maps.
//! The size data is consumed by that second step and its device size is reported as
//! `mixed_size_data_bytes`.
//!
//! Only the build frame persists across probe chunks. The streamed inner variant runs the whole
//! size-then-indices sequence once per chunk against it. Semi and anti joins that build their left
//! side keep per-left-row match state across chunks.

use std::time::Instant;

use query_runtime::metrics::{JoinOperatorMetrics, OperatorMetrics};

use super::{
    BoundedCompiledJoinProbeState, BoundedCompiledMixedLeftJoinProbeState, BoundedCompiledPreparedJoinBuildState,
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame,
    CompiledJoinMaterializationRequest, CompiledMixedJoinCall, GpuDataFrame, JoinNode, JoinSide, JoinType, NodeId,
    ResidentFrameStreamPayload, ResidentGpuFrame, Result, bounded_invariant_violation, compiled_join_input_columns,
    dataframe_mixed_join_options, join_index_error, join_side_output_upper_bound_bytes, lower_join_residual_predicate,
    materialize_compiled_indices_output, materialize_left_only_join_output_projection, observe_device_memory_bytes,
    orient_prepared_inputs, prepare_bounded_join_side_input_from_resident_frame, probe_side,
    projected_output_columns_for_retained_sizing, release_bounded_join_side_frame, residual_input_columns,
    stream_bounded_join_leaf_local_parquet_source_serial,
};

/// Joins two resident sides with the mixed kernel and gathers the output.
///
/// When the plan builds the left side of an inner or full join, `CompiledMixedJoinCall` runs cuDF
/// with the inputs exchanged and restores plan orientation in the gather maps. The shared index
/// materializer owns both frames from the point the gather maps exist.
pub(super) fn execute_compiled_mixed_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<ResidentGpuFrame> {
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let call = CompiledMixedJoinCall::lower(execution.context(), node)?;
    let timer = Instant::now();
    let size_data = call.size_data(node, &left.frame, &right.frame, overrides)?;
    let expected_output_rows = size_data.output_size() as u64;
    let size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    execution.metrics().record_mixed_join_size_data_bytes(size_data_bytes);
    let indices = size_data.materialize_indices_in(overrides, node)?;
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

/// Probes a deferred Parquet leaf against the resident build frame, one chunk at a time (inner
/// joins only).
///
/// Each chunk is projected and joined on its own with `build.input`, and its output is appended to
/// the `MixedStreamed` state in chunk order. Concatenating chunk outputs is valid for an inner join
/// only; a join that preserves unmatched rows needs the complement across all chunks and is
/// rejected here as an invariant violation. The leaf stream may call the consumer again with a
/// smaller piece of a chunk after a retryable resource error. An output is therefore appended only
/// after the whole chunk succeeded. On error the build frame is released before the error returns.
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

/// Shares the build frame's columns so one chunk's join can own and release a
/// `BoundedJoinSideFrame` without freeing the build that later chunks still read.
fn alias_compiled_mixed_join_build_chunk(build: &BoundedJoinSideFrame) -> BoundedJoinSideFrame {
    BoundedJoinSideFrame { frame: build.frame.clone(), rows: build.rows, bytes: build.bytes }
}

/// Builds the match state over the left input and folds every right chunk into it.
///
/// This is the whole probe stage of the form: the returned `MixedLeftMatched` state holds the left
/// frame and the sealed match state, and only the final gather remains. Cancellation is checked
/// before the state is built, before each chunk and before sealing. `inputs` is released with
/// `release_all` whether the fold succeeds or fails, and a failure drops the left frame together
/// with `build`. Errors gain `build_side` and `probe_side` facts when they carry none.
pub(super) fn execute_compiled_mixed_left_join_probe_from_frame_stream(
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

/// Gathers the left rows the sealed match state selected and records the join metrics under
/// `bounded_mixed_left_match_join`.
///
/// The match state is released before the gather allocates the output.
pub(super) fn execute_compiled_mixed_left_join_materialize(
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

pub(super) fn release_compiled_mixed_left_join_probe_state(probe: BoundedCompiledMixedLeftJoinProbeState) {
    // Release the match state before the executor admits the next allocation boundary.
    drop(probe);
}
