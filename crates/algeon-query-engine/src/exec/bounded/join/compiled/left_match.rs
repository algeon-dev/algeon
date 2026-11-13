//! Semi and anti joins that build their left side and fold every right chunk into per-left-row match
//! state (`KeyedLeftMatch` and `KeyedResidual { left_match: true }`).
//!
//! The planner gives a keyed semi or anti join a left build when the left input is the smaller one.
//! Neither form retains the right input: each chunk marks the left rows it matches and is released,
//! so device memory across the probe is the left frame plus a state bounded by the left row count.
//! A join without a residual uses `cudf::DataFrameHashLeftMatchState` (a hash table and one byte per
//! left row); a residual join uses `cudf::DataFrameMixedLeftJoinState`, which evaluates the residual
//! while matching. Sealing after the last chunk selects the matched rows for a semi join and the
//! unmatched rows for an anti join, so the output is one gather of the left frame. Null-aware anti
//! joins never reach this module; they build their right side (`filtered.rs`).

use std::time::Instant;

use query_runtime::metrics::{JoinOperatorMetrics, OperatorMetrics};

use super::{
    BoundedCompiledJoinProbeState, BoundedCompiledLeftMatchJoinProbeState, BoundedCompiledPreparedJoinBuildState,
    BoundedExecutionAttempt, DataFrame, Exec, ExecutionContext, GpuDataFrame, JoinNode, JoinSide, JoinStrategy,
    JoinType, NodeId, ResidentFrameStreamPayload, ResidentGpuFrame, Result, compiled_join_input_columns,
    dataframe_mixed_join_options, join_index_error, join_side_output_upper_bound_bytes, lower_join_residual_predicate,
    lower_null_equality, materialize_left_only_join_output_projection, observe_device_memory_bytes,
    prepare_bounded_join_side_input_from_resident_frame, projected_output_columns_for_retained_sizing,
    residual_input_columns,
};

enum LeftMatchState {
    Hash(cudf::DataFrameHashLeftMatchState),
    Mixed(cudf::DataFrameMixedLeftJoinState),
}

/// Sealed match state of either form; the probe state carries it to the materialize task.
pub(super) enum LeftMatchFinalized {
    Hash(cudf::DataFrameHashLeftMatchFinalized),
    Mixed(cudf::DataFrameMixedLeftJoinFinalized),
}

impl LeftMatchState {
    fn new(
        context: &ExecutionContext,
        node: &JoinNode,
        strategy: JoinStrategy,
        left: &DataFrame,
        exec: Exec<'_>,
    ) -> Result<Self> {
        let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
        let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
        if strategy == JoinStrategy::HashLeftMatch {
            let options = cudf::DataFrameJoinOptions::new()
                .left_on(&left_keys)
                .right_on(&right_keys)
                .null_equality(lower_null_equality(node.spec.null_equality));
            return cudf::DataFrameHashLeftMatchState::new(left, options, exec)
                .map(Self::Hash)
                .map_err(|error| join_index_error(error, node));
        }
        let residual_columns = residual_input_columns(node)?;
        let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
        let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
        cudf::DataFrameMixedLeftJoinState::new(left, predicate, options, exec)
            .map(Self::Mixed)
            .map_err(|error| join_index_error(error, node))
    }

    fn consume(&mut self, right: &DataFrame) -> cudf::Result<()> {
        match self {
            Self::Hash(state) => state.consume(right),
            Self::Mixed(state) => state.consume(right),
        }
    }

    fn device_allocated_size_bytes(&self) -> cudf::Result<usize> {
        match self {
            Self::Hash(state) => state.device_allocated_size_bytes(),
            Self::Mixed(state) => state.device_allocated_size_bytes(),
        }
    }

    fn seal(self, anti: bool) -> cudf::Result<LeftMatchFinalized> {
        match self {
            Self::Hash(state) => state.seal(anti).map(LeftMatchFinalized::Hash),
            Self::Mixed(state) => state.seal(anti).map(LeftMatchFinalized::Mixed),
        }
    }
}

impl LeftMatchFinalized {
    fn output_size(&self) -> usize {
        match self {
            Self::Hash(finalized) => finalized.output_size(),
            Self::Mixed(finalized) => finalized.output_size(),
        }
    }

    fn device_allocated_size_bytes(&self) -> cudf::Result<usize> {
        match self {
            Self::Hash(finalized) => finalized.device_allocated_size_bytes(),
            Self::Mixed(finalized) => finalized.device_allocated_size_bytes(),
        }
    }

    fn materialize_indices(&self) -> cudf::Result<cudf::DataFrameJoinIndices> {
        match self {
            Self::Hash(finalized) => finalized.materialize_indices(),
            Self::Mixed(finalized) => finalized.materialize_indices(),
        }
    }
}

/// Builds the match state over the left input and folds every right chunk into it.
///
/// This is the whole probe stage of the form: the returned `LeftMatched` state holds the left frame
/// and the sealed match state, and only the final gather remains. Cancellation is checked before the
/// state is built, before each chunk and before sealing. `inputs` is released with `release_all`
/// whether the fold succeeds or fails, and a failure drops the left frame together with `build`.
/// Errors gain `build_side` and `probe_side` facts when they carry none.
pub(super) fn execute_compiled_left_match_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    mut inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    let result: Result<_> = (|| {
        execution.context().check_cancellation("query_engine.bounded.join.left_match_build")?;
        let probe_columns = compiled_join_input_columns(node, build.strategy, JoinSide::Right)?;
        let exec_runtime = execution.context().runtime()?;
        let overrides = exec_runtime.exec();
        let timer = Instant::now();
        let mut state = LeftMatchState::new(execution.context(), node, build.strategy, &build.input.frame, overrides)?;
        let mut operator_time = timer.elapsed();
        let mut state_bytes =
            state.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64;
        let mut probe_rows = 0_u64;
        let mut probe_bytes = 0_u64;
        while let Some(input) = inputs.next_frame(execution) {
            let input = input?;
            execution.context().check_cancellation("query_engine.bounded.join.left_match_probe")?;
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
        execution.context().check_cancellation("query_engine.bounded.join.left_match_finalize")?;
        let timer = Instant::now();
        // Unmatched left rows become anti output only after every right batch has completed.
        let finalized = state
            .seal(matches!(node.spec.join_type, JoinType::LeftAnti { .. }))
            .map_err(|error| join_index_error(error, node))?;
        operator_time += timer.elapsed();
        state_bytes = state_bytes
            .max(finalized.device_allocated_size_bytes().map_err(|error| join_index_error(error, node))? as u64);
        match &finalized {
            LeftMatchFinalized::Hash(_) => {
                execution.metrics().record_hash_join_build();
                execution.metrics().record_hash_join_hash_table_bytes(state_bytes);
            },
            LeftMatchFinalized::Mixed(_) => execution.metrics().record_mixed_join_size_data_bytes(state_bytes),
        }
        Ok(BoundedCompiledJoinProbeState::LeftMatched(BoundedCompiledLeftMatchJoinProbeState {
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
/// `bounded_hash_left_match_join` or `bounded_mixed_left_match_join`.
///
/// The match state is released before the gather allocates the output.
pub(super) fn execute_compiled_left_match_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledLeftMatchJoinProbeState,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledLeftMatchJoinProbeState { left, finalized, probe_rows, probe_bytes, state_bytes, operator_time } =
        probe;
    let (execution_strategy, residual_predicate, hash_table_bytes, mixed_size_data_bytes) = match &finalized {
        LeftMatchFinalized::Hash(_) => ("bounded_hash_left_match_join", false, state_bytes, 0),
        LeftMatchFinalized::Mixed(_) => ("bounded_mixed_left_match_join", true, 0, state_bytes),
    };
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
    // Reclaim the match flags and hash state before allocating the gathered output.
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
            execution_strategy: execution_strategy.to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate,
            null_equality: node.spec.null_equality.to_string(),
            build_side: JoinSide::Left.to_string(),
            probe_side: JoinSide::Right.to_string(),
            build_rows: left.rows,
            build_bytes: left.bytes,
            probe_rows,
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    Ok(ResidentGpuFrame::with_device_bytes(output, output_bytes))
}

pub(super) fn release_compiled_left_match_join_probe_state(probe: BoundedCompiledLeftMatchJoinProbeState) {
    // Release the match state before the executor admits the next allocation boundary.
    drop(probe);
}
