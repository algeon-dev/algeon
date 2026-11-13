use std::ptr::{self, NonNull};

use super::*;
use crate::core::{error, native_teardown::drop_device_owned_handle};

/// Cumulative keyed semi/anti membership for a retained logical-left frame,
/// without a residual predicate.
///
/// Construction builds a hash table over the left keys and a one-byte match
/// flag per left row. Each right batch marks the left rows it matches without
/// emitting rows, so the device bytes retained across batches are bounded by
/// the left input. Sealing declares right EOF; it consumes this state and makes
/// the exact semi/anti output count available. Calls after construction run on
/// the construction device, stream and memory resource.
pub struct DataFrameHashLeftMatchState {
    handle: NonNull<cudf_sys::cudf_hash_left_match_state_handle_state>,
    domain: Arc<ExecutionDomain>,
    left: DataFrame,
    right_keys: Vec<String>,
}

impl std::fmt::Debug for DataFrameHashLeftMatchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFrameHashLeftMatchState")
            .field("left_rows", &self.left.num_rows())
            .field("right_keys", &self.right_keys)
            .finish_non_exhaustive()
    }
}

// SAFETY: The state retains the left frame whose key storage the native hash
// table borrows. Mutation requires exclusive Rust access; device-aware
// destruction and the retained domain keep the stream/resource live after a
// move to another thread.
unsafe impl Send for DataFrameHashLeftMatchState {}

impl Drop for DataFrameHashLeftMatchState {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self.domain.device_id(),
            "HashLeftMatchState::drop",
            |owned| unsafe { cudf_sys::cudf_hash_left_match_state_destroy(owned) },
        );
    }
}

/// Sealed keyed semi/anti membership and its exact output count.
#[derive(Debug)]
pub struct DataFrameHashLeftMatchFinalized {
    state: DataFrameHashLeftMatchState,
    output_size: usize,
}

impl DataFrameHashLeftMatchState {
    /// Build the hash table over the left keys of `options` without copying
    /// frame payloads, running where `exec` selects; later calls on the state
    /// stay there. Null keys compare as `options` selects.
    pub fn new(left: &DataFrame, options: DataFrameJoinOptions, exec: Exec<'_>) -> Result<Self> {
        let operation = "dataframe::hash_left_match_state::new";
        let (left_keys, right_keys, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
        validate_hash_join_key_names(&left_keys, operation)?;
        validate_hash_join_key_names(&right_keys, operation)?;
        if left_keys.len() != right_keys.len() {
            return Err(merge_key_count_mismatch_error(operation, left_keys.len(), right_keys.len()));
        }
        let keys = resolve_column_names(left.columns_map(), &string_refs(&left_keys), operation)?;
        let keys = build_borrowed_table_view_from_columns(left.columns_map(), &keys)?;
        let execution = ResolvedExecution::unary(keys.execution_domain(), exec, operation)?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_hash_left_match_state_create_ex(
                    keys.raw_handle(),
                    options.null_eq().to_join_raw(),
                    &mut handle,
                    execution.exec_options(),
                )
            },
            operation,
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, operation, "hash left match state")?,
            domain: Arc::clone(execution.output_domain()),
            left: left.clone(),
            right_keys,
        })
    }

    /// Mark the left rows that one right batch matches. An empty batch is not EOF.
    pub fn consume(&mut self, right: &DataFrame) -> Result<()> {
        let operation = "dataframe::hash_left_match_state::consume";
        let keys = resolve_probe_key_table(right, &self.right_keys, self.right_keys.len(), operation)?;
        // ResolvedExecution orders the right producer and fences its release
        // after the probe, including right frames on a different stream.
        let execution =
            ResolvedExecution::many(&self.domain, vec![keys.execution_domain()], Exec::RETAINED, operation)?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_hash_left_match_state_consume_ex(
                    self.handle.as_ptr(),
                    keys.raw_handle(),
                    execution.exec_options(),
                )
            },
            operation,
        )
    }

    /// Retained native hash table and match flag bytes, excluding input frames.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut bytes = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_hash_left_match_state_device_allocated_size(self.handle.as_ptr(), &mut bytes) },
            "dataframe::hash_left_match_state::device_allocated_size_bytes",
        )?;
        Ok(bytes)
    }

    /// Declare right EOF and choose anti (`true`) or semi (`false`) output.
    pub fn seal(self, anti: bool) -> Result<DataFrameHashLeftMatchFinalized> {
        let operation = "dataframe::hash_left_match_state::seal";
        let execution = ResolvedExecution::unary(&self.domain, Exec::RETAINED, operation)?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_hash_left_match_state_seal_ex(
                    self.handle.as_ptr(),
                    if anti {
                        cudf_sys::CUDF_JOIN_KIND_LEFT_ANTI_JOIN
                    } else {
                        cudf_sys::CUDF_JOIN_KIND_LEFT_SEMI_JOIN
                    },
                    execution.exec_options(),
                )
            },
            operation,
        )?;
        let mut output_size = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_hash_left_match_state_output_size(self.handle.as_ptr(), &mut output_size) },
            operation,
        )?;
        Ok(DataFrameHashLeftMatchFinalized { state: self, output_size })
    }
}

impl DataFrameHashLeftMatchFinalized {
    /// Exact selected logical-left rows after every right batch has been consumed.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Retained native hash table and match flag bytes, excluding input frames.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        self.state.device_allocated_size_bytes()
    }

    /// Materialize the selected logical-left indices in ascending order without
    /// re-probing right input.
    pub fn materialize_indices(&self) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::hash_left_match_finalized::materialize_indices";
        let execution = ResolvedExecution::unary(&self.state.domain, Exec::RETAINED, operation)?;
        let mut out = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_hash_left_match_state_materialize_indices_ex(
                    self.state.handle.as_ptr(),
                    &mut out,
                    execution.exec_options(),
                )
            },
            operation,
        )?;
        let indices = unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain())) }?;
        single_side_indices_from_raw(join::SingleSideJoinIndices { indices }, operation)
    }
}
