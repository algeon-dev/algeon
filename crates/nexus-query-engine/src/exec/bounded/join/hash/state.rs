use super::*;

// Reusable build artifact plus the reservation that keeps its hash table resident.
pub(in crate::exec::bounded) struct BoundedRootHashJoinBuildState {
    pub(in crate::exec::bounded::join::hash) build_side: JoinSide,
    pub(in crate::exec::bounded::join::hash) probe_side: JoinSide,
    pub(in crate::exec::bounded::join::hash) left_keys: Vec<String>,
    pub(in crate::exec::bounded::join::hash) right_keys: Vec<String>,
    pub(in crate::exec::bounded::join::hash) left_columns: Vec<String>,
    pub(in crate::exec::bounded::join::hash) right_columns: Vec<String>,
    pub(in crate::exec::bounded::join::hash) build: BoundedJoinSideFrame,
    pub(in crate::exec::bounded::join::hash) hash_join: GpuHashJoinBuildHandle,
    pub(in crate::exec::bounded::join::hash) hash_table_reservation: ActiveReservation,
    pub(in crate::exec::bounded::join::hash) hash_table_bytes: u64,
    pub(in crate::exec::bounded::join::hash) left_indices_from_probe: bool,
    pub(in crate::exec::bounded::join::hash) build_keys_unique: bool,
    pub(in crate::exec::bounded::join::hash) timer: Instant,
}

// Probe indices and reservations captured before materializing root hash-join output.
pub(in crate::exec::bounded) struct BoundedRootHashJoinProbeState {
    pub(in crate::exec::bounded::join::hash) build: BoundedRootHashJoinBuildState,
    pub(in crate::exec::bounded::join::hash) probe: BoundedJoinSideFrame,
    pub(in crate::exec::bounded::join::hash) indices: cudf::DataFrameHashJoinPairIndices,
    pub(in crate::exec::bounded::join::hash) gather_map_reservation: ActiveReservation,
    pub(in crate::exec::bounded::join::hash) gather_map_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: u64,
    pub(in crate::exec::bounded::join::hash) minimum_output_reservation_bytes: u64,
}

impl BoundedRootHashJoinProbeState {
    pub(in crate::exec::bounded) fn materialize_admission_bytes(&self, stream_target_bytes: u64) -> u64 {
        let output_rows = self.indices.build().len() as u64;
        if output_rows == 0 {
            return 0;
        }
        let chunk_rows = reusable_hash_join_stream_chunk_rows(
            output_rows,
            self.output_reservation_bytes,
            stream_target_bytes.max(1),
        );
        let chunk_output_bytes =
            proportional_join_stream_bytes(self.output_reservation_bytes, chunk_rows as u64, output_rows)
                .saturating_add(self.minimum_output_reservation_bytes);
        let slice_gather_map_bytes = cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(chunk_rows)
            .map(|bytes| bytes as u64)
            .unwrap_or(64);
        chunk_output_bytes.saturating_add(slice_gather_map_bytes)
    }
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_build_state(
    build: BoundedRootHashJoinBuildState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedRootHashJoinBuildState { build: build_frame, hash_join: _, hash_table_reservation, .. } = build;
    reservations.release(hash_table_reservation, metrics);
    release_bounded_join_side_frame(build_frame, reservations, metrics);
}

pub(in crate::exec::bounded) fn release_bounded_root_hash_join_probe_state(
    probe: BoundedRootHashJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedRootHashJoinProbeState { build, probe, indices: _, gather_map_reservation, .. } = probe;
    reservations.release(gather_map_reservation, metrics);
    release_bounded_root_hash_join_build_state(build, reservations, metrics);
    release_bounded_join_side_frame(probe, reservations, metrics);
}

pub(in crate::exec::bounded::join::hash) fn validate_root_hash_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)?;
    if !keyed_hash_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled hash join execution requires keyed inner, left, or full equi join capability",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }
    Ok(())
}
