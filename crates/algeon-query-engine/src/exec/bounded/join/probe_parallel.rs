//! Reusable join build with parallel probe chunks.
//!
//! One hash or filtered build serves many probe chunks. Each chunk probes the
//! shared build on its own execution lane with private workspace: the build
//! handle is read-only (`&self` probes on `Arc<HashJoinHandle>` Send+Sync,
//! per-call `ExecutionOverrides` streams, no shared mutable scratch), while
//! probe frames, indices, and gather maps stay lane-private. The join owner
//! merges chunk outputs after every lane completes.
//!
//! Parallel probe is enabled only where concatenation preserves semantics:
//! inner joins, probe-preserved left joins, and filtered semi/anti/mark
//! (membership is per probe row). Build-preserved outer joins (`Left` with a
//! left build, `Full`) need a global matched-bitmap complement that per-chunk
//! `full_probe_in` would duplicate, and `MixedKeyedResidual` accumulates
//! `&mut` native state, so those strategies keep the serial probe path and
//! are documented as such.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use super::{
    BoundedExecutionAttempt, ResidentGpuFrame, Result,
    compiled::stateless_streaming_probe_supported,
    hash::{BoundedRootHashJoinBuildState, execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame},
};
use crate::{
    exec::bounded::compiled::{ResidentFrameStreamPayload, lanes::AggregateLaneDispatch},
    plan::{JoinNode, JoinSide, JoinType, NodeId},
    runtime::resident::GpuHashJoinBuildHandle,
};

/// Whether a hash join probe may run chunks in parallel with concatenation.
///
/// `stateless_streaming_probe_supported` already names the chunk-independent
/// cases: inner joins and left joins with a right (probe-preserved) build.
/// All other hash strategies need the join owner's global complement and stay
/// serial.
pub(in crate::exec::bounded) fn hash_parallel_probe_supported(node: &JoinNode) -> bool {
    stateless_streaming_probe_supported(node)
}

/// Strategies that keep the serial probe path, with the reason recorded for
/// the D-stage contract.
///
/// Build-preserved outer joins need a global matched-bitmap complement that
/// per-chunk `full_probe_in` would duplicate; residual predicates need owner
/// merge; mixed/keyless/cross keep exclusive access until their owners prove
/// safe overlap.
pub(in crate::exec::bounded) fn serial_probe_reason(node: &JoinNode) -> &'static str {
    if node.spec.build_side == JoinSide::Left && matches!(node.spec.join_type, JoinType::Left | JoinType::Full) {
        "build_preserved_outer_needs_global_matched_complement"
    } else if matches!(node.spec.join_type, JoinType::Full) {
        "full_outer_needs_global_matched_complement"
    } else if node.spec.residual_predicate.is_some() {
        "residual_predicate_needs_owner_merge"
    } else {
        "strategy_requires_exclusive_probe_access"
    }
}

struct ProbeWorkState {
    pending: VecDeque<ResidentGpuFrame>,
    outputs: Vec<ResidentGpuFrame>,
    chunk_count: usize,
}

/// Probe an eager frame stream's chunks in parallel sharing one hash build.
///
/// `inputs` must hold eager frames; lazy producers fall back to the serial
/// stream path. Outputs keep input chunk order so the materialize owner
/// concatenates deterministically. The shared build stays owned by the caller
/// while lanes run; each lane works through an `Arc` clone whose device
/// memory the forward fence keeps visible on the lane's stream.
/// `build_count` never grows with probe batch count: the table is built once
/// and only probed per chunk.
///
/// `outer` lends the compiled run's dispatcher so probe chunks reuse the
/// run's own lanes; without it the helper borrows currently available lanes
/// instead (serial scheduler path).
pub(in crate::exec::bounded) fn parallel_hash_probe_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    inputs: ResidentFrameStreamPayload,
    outer: Option<&AggregateLaneDispatch>,
) -> Result<Vec<ResidentGpuFrame>> {
    let frames = inputs.into_remaining_frames_owned(execution)?;
    if frames.len() <= 1 {
        return serial_hash_probe_frames(execution, node_id, node, build, frames.into());
    }
    let mut state = ProbeWorkState { pending: frames.into(), outputs: Vec::new(), chunk_count: 0 };
    state.chunk_count = state.pending.len();
    let chunk_count = state.chunk_count;
    let shared = Arc::new(SharedHashBuild::from_build(build));
    let node_owned = node.clone();
    let mut run_on = |dispatch: &AggregateLaneDispatch, execution: &mut BoundedExecutionAttempt<'_>| {
        let width = dispatch.width();
        execution.metrics().record_join_strategy_decision(
            node_id,
            &BTreeMap::from([
                ("parallel_probe", "hash".to_string()),
                ("parallel_probe_chunks", chunk_count.to_string()),
                ("parallel_probe_width", width.to_string()),
            ]),
        );
        dispatch.execute_work(
            execution,
            &mut state,
            |state, _| Ok(state.pending.pop_front()),
            {
                let shared = Arc::clone(&shared);
                let node_owned = node_owned.clone();
                move |lane_execution, chunk| {
                    let build_ref = shared.as_build_ref();
                    execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
                        lane_execution,
                        node_id,
                        &node_owned,
                        &build_ref,
                        chunk,
                    )
                }
            },
            |state, _, chunk_outputs| {
                state.outputs.extend(chunk_outputs);
                Ok(())
            },
        )
    };
    match outer {
        Some(dispatch) => run_on(dispatch, execution)?,
        None => {
            AggregateLaneDispatch::with_available_lanes(execution, |dispatch, execution| run_on(dispatch, execution))?
        },
    }
    // `execute_work` commits in ordinal order, so `outputs` already follows
    // input chunk order. A single-chunk fallback above keeps width-1 on the
    // serial path with identical output order.
    Ok(state.outputs)
}

/// Serial fallback for hash probe frames, used when lanes are unavailable.
fn serial_hash_probe_frames(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    mut frames: VecDeque<ResidentGpuFrame>,
) -> Result<Vec<ResidentGpuFrame>> {
    let mut outputs = Vec::new();
    while let Some(chunk) = frames.pop_front() {
        outputs.extend(execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
            execution, node_id, node, build, chunk,
        )?);
    }
    Ok(outputs)
}

/// Minimal Send+Sync view of a hash build shared across probe lanes.
///
/// The native `HashJoinHandle` is already `Send+Sync` behind an `Arc`, probes
/// take `&self`, and each lane supplies its own `ExecutionOverrides` stream,
/// so sharing this immutable view needs no new `unsafe` impl. The timer is
/// intentionally excluded: it measures the coordinator's build, not lanes.
struct SharedHashBuild {
    build_side: JoinSide,
    probe_side: JoinSide,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_columns: Vec<String>,
    right_columns: Vec<String>,
    build_frame: cudf::DataFrame,
    build_rows: u64,
    build_bytes: u64,
    hash_join: GpuHashJoinBuildHandle,
    hash_table_bytes: u64,
    left_indices_from_probe: bool,
    build_keys_unique: bool,
}

impl SharedHashBuild {
    fn from_build(build: &BoundedRootHashJoinBuildState) -> Self {
        Self {
            build_side: build.build_side,
            probe_side: build.probe_side,
            left_keys: build.left_keys.clone(),
            right_keys: build.right_keys.clone(),
            left_columns: build.left_columns.clone(),
            right_columns: build.right_columns.clone(),
            build_frame: build.build.frame.clone(),
            build_rows: build.build.rows,
            build_bytes: build.build.bytes,
            hash_join: build.hash_join.clone(),
            hash_table_bytes: build.hash_table_bytes,
            left_indices_from_probe: build.left_indices_from_probe,
            build_keys_unique: build.build_keys_unique,
        }
    }

    fn as_build_ref(&self) -> BoundedRootHashJoinBuildState {
        BoundedRootHashJoinBuildState {
            build_side: self.build_side,
            probe_side: self.probe_side,
            left_keys: self.left_keys.clone(),
            right_keys: self.right_keys.clone(),
            left_columns: self.left_columns.clone(),
            right_columns: self.right_columns.clone(),
            build: super::common::BoundedJoinSideFrame {
                frame: self.build_frame.clone(),
                rows: self.build_rows,
                bytes: self.build_bytes,
            },
            hash_join: self.hash_join.clone(),
            hash_table_bytes: self.hash_table_bytes,
            left_indices_from_probe: self.left_indices_from_probe,
            build_keys_unique: self.build_keys_unique,
            timer: std::time::Instant::now(),
        }
    }
}
