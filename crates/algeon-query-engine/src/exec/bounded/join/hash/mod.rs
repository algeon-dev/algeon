//! Reusable root hash join: one cuDF hash table, many probe chunks.
//!
//! `build` turns the build input into a hash table that stays resident while probe chunks
//! run against it. `probe` produces gather maps (pairs of row indices into the probe and
//! build frames). `materialize` and `stream` gather the projected output columns
//! from those maps in slices whose size bound fits the attempt's slice target.
//!
//! Ownership follows the state types in `state`. The build state is created by the build
//! step and consumed by the probe step, or borrowed by the chunk-stream probe step so one
//! build serves many chunks. A probe state carries its build state and is consumed by
//! materialize. Materialize frees the build when it finishes. Lane jobs hold clones of the
//! table handle and the build frame, and the device buffers free when the last clone drops.
//!
//! A slice whose gather fails with a query-cap rejection is halved and retried; see
//! `materialize::materialize_bounded_root_hash_join_index_slice` for what a retry leaves
//! behind. Gather maps and the probe frame are never recomputed by a retry.

use super::{
    BoundedExecutionAttempt, BoundedPartitionTargetPolicy, CommittedPartial, DataFrame, Exec, ExecutionContext,
    ExecutionMetrics, GpuDataFrame, GpuHashJoinBuildHandle, Instant, JoinNode, JoinSide, JoinType, NodeId, QueryError,
    ResidentGpuFrame, Result,
    common::{
        BoundedJoinSideFrame, join_side_output_upper_bound_bytes, join_side_output_upper_bound_bytes_for_row_counts,
        prepare_bounded_join_side_input_from_resident_frame, projected_output_columns_for_retained_sizing,
        release_bounded_join_side_frame, sum_bounds,
    },
    cudf_query_error, join_index_error, lower_null_equality, materialize_join_output_projection,
    observe_device_memory_bytes, probe_side, projected_input_columns, side_metric, validate_executable_join_subset,
};
use crate::exec::bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource};

mod build;
mod materialize;
mod probe;
mod state;
mod stream;

pub(in crate::exec::bounded::join) use build::build_bounded_hash_join;
pub(in crate::exec::bounded) use build::execute_bounded_root_hash_join_build_from_resident_frame;
pub(in crate::exec::bounded) use materialize::{
    execute_bounded_root_hash_join_materialize, execute_bounded_root_hash_join_materialize_preserving_stream,
};
pub(in crate::exec::bounded) use probe::{
    execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame,
    execute_bounded_root_hash_join_probe_from_resident_frame,
};
pub(in crate::exec::bounded) use state::{
    BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState, release_bounded_root_hash_join_build_state,
    release_bounded_root_hash_join_probe_state,
};
pub(in crate::exec::bounded) use stream::reusable_hash_join_stream_target_output_bytes;
