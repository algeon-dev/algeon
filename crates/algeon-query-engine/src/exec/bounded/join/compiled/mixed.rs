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
//! side use `left_state.rs`, which keeps match state across chunks.

use std::time::Instant;

mod left_state;
pub(super) use left_state::*;

use super::{
    BoundedCompiledJoinProbeState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, CompiledJoinMaterializationRequest,
    CompiledMixedJoinCall, JoinNode, JoinSide, JoinType, NodeId, ResidentGpuFrame, Result, bounded_invariant_violation,
    compiled_join_input_columns, materialize_compiled_indices_output, orient_prepared_inputs,
    prepare_bounded_join_side_input_from_resident_frame, probe_side, release_bounded_join_side_frame,
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
