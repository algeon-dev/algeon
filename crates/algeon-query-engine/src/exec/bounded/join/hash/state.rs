use super::{
    BoundedJoinSideFrame, GpuHashJoinBuildHandle, Instant, JoinNode, JoinSide, Result, release_bounded_join_side_frame,
    stream::{proportional_join_stream_bytes, reusable_hash_join_stream_chunk_rows},
    sum_bounds, validate_executable_join_subset,
};

// Reusable build artifact plus its observed native allocation.
// Fields stay visible across the join owner so the parallel-probe owner can
// share an immutable build view without new unsafe impls; per-lane scratch
// remains private to each probe call.
pub(in crate::exec::bounded) struct BoundedRootHashJoinBuildState {
    pub(in crate::exec::bounded) build_side: JoinSide,
    pub(in crate::exec::bounded) probe_side: JoinSide,
    pub(in crate::exec::bounded) left_keys: Vec<String>,
    pub(in crate::exec::bounded) right_keys: Vec<String>,
    pub(in crate::exec::bounded) left_columns: Vec<String>,
    pub(in crate::exec::bounded) right_columns: Vec<String>,
    pub(in crate::exec::bounded) build: BoundedJoinSideFrame,
    pub(in crate::exec::bounded) hash_join: GpuHashJoinBuildHandle,
    pub(in crate::exec::bounded) hash_table_bytes: u64,
    pub(in crate::exec::bounded) left_indices_from_probe: bool,
    pub(in crate::exec::bounded) build_keys_unique: bool,
    pub(in crate::exec::bounded) timer: Instant,
}

// Probe indices and observed allocation facts captured before materializing output.
pub(in crate::exec::bounded) struct BoundedRootHashJoinProbeState {
    pub(in crate::exec::bounded::join::hash) build: BoundedRootHashJoinBuildState,
    pub(in crate::exec::bounded::join::hash) probe: BoundedJoinSideFrame,
    pub(in crate::exec::bounded::join::hash) indices: cudf::DataFrameHashJoinPairIndices,
    pub(in crate::exec::bounded::join::hash) gather_map_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: Option<u64>,
    pub(in crate::exec::bounded::join::hash) minimum_output_reservation_bytes: Option<u64>,
}

impl BoundedRootHashJoinProbeState {
    /// Device bytes one materialized slice can need, or `None` when the output
    /// has no byte bound.
    pub(in crate::exec::bounded) fn materialize_admission_bytes(&self, stream_target_bytes: u64) -> Option<u64> {
        let output_rows = self.indices.build().len() as u64;
        if output_rows == 0 {
            return Some(0);
        }
        let chunk_rows = reusable_hash_join_stream_chunk_rows(
            output_rows,
            self.output_reservation_bytes,
            stream_target_bytes.max(1),
        );
        let chunk_output_bytes = sum_bounds(
            proportional_join_stream_bytes(self.output_reservation_bytes, chunk_rows as u64, output_rows),
            self.minimum_output_reservation_bytes,
        );
        let slice_gather_map_bytes =
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(chunk_rows).ok().map(|bytes| bytes as u64);
        sum_bounds(chunk_output_bytes, slice_gather_map_bytes)
    }
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_build_state(build: BoundedRootHashJoinBuildState) {
    let BoundedRootHashJoinBuildState { build: build_frame, .. } = build;
    release_bounded_join_side_frame(build_frame);
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_probe_state(probe: BoundedRootHashJoinProbeState) {
    let BoundedRootHashJoinProbeState { build, probe, .. } = probe;
    release_bounded_root_hash_join_build_state(build);
    release_bounded_join_side_frame(probe);
}

pub(in crate::exec::bounded::join::hash) fn validate_root_hash_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)
}
