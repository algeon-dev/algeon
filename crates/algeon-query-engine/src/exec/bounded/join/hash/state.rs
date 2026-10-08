use super::{
    BoundedJoinSideFrame, GpuHashJoinBuildHandle, Instant, JoinNode, JoinSide, Result, release_bounded_join_side_frame,
    stream::{proportional_join_stream_bytes, reusable_hash_join_stream_chunk_rows},
    sum_bounds, validate_executable_join_subset,
};

/// Resident build side of a root hash join: the prepared build frame and the cuDF hash
/// table built over its key columns.
///
/// Created by `execute_bounded_root_hash_join_build_from_resident_frame` on the
/// coordinator. The holder owns the device allocations of `build` and `hash_join`; they
/// free when the last clone drops, and `release_bounded_root_hash_join_build_state` is the
/// explicit release. The serial probe step consumes the state and the chunk-stream probe
/// step borrows it; one build serves any number of probe chunks. Fields are visible
/// across the join owner so the parallel-probe owner can share an immutable view with
/// execution lanes without new `unsafe` impls; per-lane scratch stays private to each
/// probe call.
pub(in crate::exec::bounded) struct BoundedRootHashJoinBuildState {
    pub(in crate::exec::bounded) build_side: JoinSide,
    pub(in crate::exec::bounded) probe_side: JoinSide,
    pub(in crate::exec::bounded) left_keys: Vec<String>,
    pub(in crate::exec::bounded) right_keys: Vec<String>,
    /// Columns each plan side keeps from its input: keys, then output-projected columns.
    /// Hidden DataFusion order columns are added when a side is prepared.
    pub(in crate::exec::bounded) left_columns: Vec<String>,
    pub(in crate::exec::bounded) right_columns: Vec<String>,
    pub(in crate::exec::bounded) build: BoundedJoinSideFrame,
    pub(in crate::exec::bounded) hash_join: GpuHashJoinBuildHandle,
    /// Device bytes of the hash table, read from the handle when it was built.
    pub(in crate::exec::bounded) hash_table_bytes: u64,
    /// True when the build input is the plan's right side. The left gather map is then the
    /// probe-row map of the cuDF index pair and the right gather map is the build-row map.
    pub(in crate::exec::bounded) left_indices_from_probe: bool,
    /// Set only for a single key column whose distinct count equals the build row count.
    /// It permits the single-pass inner probe: each probe row matches at most one build
    /// row, and the output-size pass is skipped.
    pub(in crate::exec::bounded) build_keys_unique: bool,
    /// Started when the build step begins. Materialization reports its elapsed time as
    /// operator time; the figure covers every interval between the stages.
    pub(in crate::exec::bounded) timer: Instant,
}

/// A finished probe awaiting materialization: the build state, the prepared probe frame
/// and the gather maps that pair probe rows with build rows.
///
/// Holds the build state by value; dropping a probe state also drops the build.
/// Materialization consumes it; `release_bounded_root_hash_join_probe_state` is the
/// explicit release for a probe that will not be materialized.
pub(in crate::exec::bounded) struct BoundedRootHashJoinProbeState {
    pub(in crate::exec::bounded::join::hash) build: BoundedRootHashJoinBuildState,
    pub(in crate::exec::bounded::join::hash) probe: BoundedJoinSideFrame,
    pub(in crate::exec::bounded::join::hash) indices: cudf::DataFrameHashJoinPairIndices,
    /// Device bytes of `indices`, read right after the probe.
    pub(in crate::exec::bounded::join::hash) gather_map_bytes: u64,
    /// Upper bound on the bytes of the full materialized output, or `None` when a column
    /// has no closed-form bound. This sizes slices and is reported in metrics; it is not
    /// a reservation against the grant, whose cap the query allocator enforces.
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: Option<u64>,
    /// The same bound for a one-row output (zero when the probe matched no rows).
    pub(in crate::exec::bounded::join::hash) minimum_output_reservation_bytes: Option<u64>,
}

impl BoundedRootHashJoinProbeState {
    /// Device bytes one materialized slice can need, or `None` when the output
    /// has no byte bound.
    ///
    /// Sums three terms: the slice's share of the output bound, the one-row bound (the
    /// bound is not linear in row count because of offset and padding terms), and the
    /// gather-map bytes a slice of that length reserves. The materialized stream reports
    /// this as its largest frame hint. An empty probe result needs no bytes.
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

/// Drops the build frame and the hash table handle. Other clones of the handle, such as
/// those held by running lane jobs, keep the table alive until they drop.
pub(in crate::exec::bounded) fn release_bounded_root_hash_join_build_state(build: BoundedRootHashJoinBuildState) {
    let BoundedRootHashJoinBuildState { build: build_frame, .. } = build;
    release_bounded_join_side_frame(build_frame);
}

/// Drops the build state, then the probe frame. The gather maps drop when the function
/// returns.
pub(in crate::exec::bounded) fn release_bounded_root_hash_join_probe_state(probe: BoundedRootHashJoinProbeState) {
    let BoundedRootHashJoinProbeState { build, probe, .. } = probe;
    release_bounded_root_hash_join_build_state(build);
    release_bounded_join_side_frame(probe);
}

pub(in crate::exec::bounded::join::hash) fn validate_root_hash_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)
}
