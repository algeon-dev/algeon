use std::ptr::{self, NonNull};

use super::*;
use crate::core::error;

/// Cumulative keyed mixed semi/anti membership for a retained logical-left frame.
///
/// Construction retains the left frame and owns the residual predicate. Each
/// right batch adds matches without emitting rows. Sealing declares right EOF;
/// it consumes this state and makes the exact semi/anti output count available.
/// Every call must use the construction device, stream and memory resource.
pub struct DataFrameMixedLeftJoinState {
    handle: NonNull<cudf_sys::cudf_mixed_left_join_state_handle_state>,
    domain: Arc<ExecutionDomain>,
    left: DataFrame,
    predicate: AstExpression,
    options: DataFrameMixedJoinOptions,
}

impl std::fmt::Debug for DataFrameMixedLeftJoinState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFrameMixedLeftJoinState")
            .field("left_rows", &self.left.num_rows())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

// SAFETY: The state retains all borrowed native left storage and AST literals.
// Mutation requires exclusive Rust access; device-aware destruction and the
// retained domain keep the stream/resource live after a move to another thread.
unsafe impl Send for DataFrameMixedLeftJoinState {}

impl Drop for DataFrameMixedLeftJoinState {
    fn drop(&mut self) {
        crate::core::native_teardown::drop_device_owned_handle(
            self.handle.as_ptr(),
            self.domain.device_id(),
            "MixedLeftJoinState::drop",
            |owned| unsafe { cudf_sys::cudf_mixed_left_join_state_destroy(owned) },
        );
    }
}

/// Sealed mixed semi/anti membership and its exact output count.
#[derive(Debug)]
pub struct DataFrameMixedLeftJoinFinalized {
    state: DataFrameMixedLeftJoinState,
    output_size: usize,
}

impl DataFrameMixedLeftJoinState {
    /// Build the logical-left equality index without copying frame payloads.
    pub fn new_in(
        left: &DataFrame,
        predicate: AstExpression,
        options: DataFrameMixedJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        let operation = "dataframe::mixed_left_join_state::new_in";
        let (left_keys, _, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
        if options.left_predicate_column_names().is_empty() || options.right_predicate_column_names().is_empty() {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                operation,
                "mixed join requires predicate columns",
            ));
        }
        let keys = resolve_column_names(left.columns_map(), &string_refs(&left_keys), operation)?;
        let conditional =
            resolve_column_names(left.columns_map(), &string_refs(options.left_predicate_column_names()), operation)?;
        let keys = build_borrowed_table_view_from_columns(left.columns_map(), &keys)?;
        let conditional = build_borrowed_table_view_from_columns(left.columns_map(), &conditional)?;
        let mut domains = vec![conditional.table().execution_domain()];
        predicate.literal_domains(&mut domains);
        let execution = ResolvedExecution::many(keys.table().execution_domain(), domains, ctx, operation)?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_mixed_left_join_state_create_ex(
                    keys.table().raw_handle(),
                    conditional.table().raw_handle(),
                    predicate.raw_handle(),
                    options.null_eq().to_join_raw(),
                    &mut handle,
                    execution.exec_options(),
                )
            },
            operation,
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, operation, "mixed left join state")?,
            domain: Arc::clone(execution.output_domain()),
            left: left.clone(),
            predicate,
            options,
        })
    }

    /// Accumulate matches from one right batch. An empty batch is not EOF.
    pub fn consume_in(&mut self, right: &DataFrame, ctx: &ExecutionOverrides<'_>) -> Result<()> {
        let operation = "dataframe::mixed_left_join_state::consume_in";
        let tables = mixed_join_tables(&self.left, right, &self.options, operation)?;
        let mut domains = vec![
            tables.left_equality.table().execution_domain(),
            tables.left_conditional.table().execution_domain(),
            tables.right_equality.table().execution_domain(),
            tables.right_conditional.table().execution_domain(),
        ];
        self.predicate.literal_domains(&mut domains);
        // ResolvedExecution orders input producers and fences their release after
        // the consuming kernel, including right frames on a different stream.
        let execution = ResolvedExecution::many(&self.domain, domains, ctx, operation)?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_mixed_left_join_state_consume_ex(
                    self.handle.as_ptr(),
                    tables.right_equality.table().raw_handle(),
                    tables.right_conditional.table().raw_handle(),
                    execution.exec_options(),
                )
            },
            operation,
        )
    }

    /// Retained native index and membership bytes, excluding input frames.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        let mut bytes = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_mixed_left_join_state_device_allocated_size(self.handle.as_ptr(), &mut bytes) },
            "dataframe::mixed_left_join_state::device_allocated_size_bytes",
        )?;
        Ok(bytes)
    }

    /// Declare right EOF and choose anti (`true`) or semi (`false`) output.
    pub fn seal_in(self, anti: bool, ctx: &ExecutionOverrides<'_>) -> Result<DataFrameMixedLeftJoinFinalized> {
        let operation = "dataframe::mixed_left_join_state::seal_in";
        let execution = ResolvedExecution::unary(&self.domain, ctx, operation)?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_mixed_left_join_state_seal_ex(
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
            unsafe { cudf_sys::cudf_mixed_left_join_state_output_size(self.handle.as_ptr(), &mut output_size) },
            operation,
        )?;
        Ok(DataFrameMixedLeftJoinFinalized { state: self, output_size })
    }
}

impl DataFrameMixedLeftJoinFinalized {
    /// Exact selected logical-left rows after every right batch has been consumed.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Retained native index and membership bytes, excluding input frames.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        self.state.device_allocated_size_bytes()
    }

    /// Exact bytes for the final INT32 gather map.
    pub fn gather_map_reservation_size_bytes(&self) -> Result<usize> {
        self.output_size.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "dataframe::mixed_left_join_finalized::gather_map_reservation_size_bytes",
                "gather-map bytes overflow usize",
            )
        })
    }

    /// Materialize the selected logical-left indices without re-probing right input.
    pub fn materialize_indices_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::mixed_left_join_finalized::materialize_indices_in";
        let execution = ResolvedExecution::unary(&self.state.domain, ctx, operation)?;
        let mut out = ptr::null_mut();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_mixed_left_join_state_materialize_indices_ex(
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
