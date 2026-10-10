//! GPU-accelerated join operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::merge`](crate::FrameOps::merge) with
//! [`MergeOptions`].
//!
//! # Example
//!
//! ```no_run
//! use cudf::{MergeOptions, Runtime};
//!
//! let rt = Runtime::new().unwrap();
//! let left = cudf::dataframe! {
//!     "id" => rt.series(&[1i32, 2, 3]).unwrap(),
//!     "left_val" => rt.series(&[10i32, 20, 30]).unwrap(),
//! }
//! .unwrap();
//! let right = cudf::dataframe! {
//!     "id" => rt.series(&[2i32, 3, 4]).unwrap(),
//!     "right_val" => rt.series(&[200i32, 300, 400]).unwrap(),
//! }
//! .unwrap();
//!
//! let result = left
//!     .on(rt.exec())
//!     .merge(&right, MergeOptions::new().on(["id"]).how_inner())
//!     .unwrap();
//! assert_eq!(
//!     result.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
//!     vec![2, 3]
//! );
//! ```
//!
//! # Join-Index `_in` Execution Context
//!
//! Join-index APIs ending in `_in` share the same execution-context contract.
//! The borrowed [`crate::Exec`] selects the stream and/or memory
//! resource for GPU index allocation when compatible with the primary left
//! input's device (`left_keys` for equality joins, `probe_keys` for
//! `HashJoin` probes, `left` for conditional joins, and `left_equality` for
//! mixed joins). Output provenance is resolved from that primary input plus the
//! override. Same-device table inputs and predicate literal domains are
//! ordered against the resolved output stream; cross-device inputs are rejected
//! before calling libcudf. These calls enqueue GPU `INT32` index allocation and
//! do not gather payload columns or materialize host-visible data before
//! returning.
//!
//! `filter_join_indices_in` follows the same split, but uses the left index
//! column as the primary input because it transforms already-computed join
//! gather maps and does not build equality matches itself.

use std::{
    fmt,
    ptr::{self, NonNull},
    sync::Arc,
};

mod dataframe;
pub use dataframe::{DataFrameMixedLeftJoinFinalized, DataFrameMixedLeftJoinState};
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use dataframe::{
    trigger_merge_reassembly_key_count_error_for_test, trigger_projected_cross_join_input_row_limit_for_test,
    trigger_projected_cross_join_output_row_overflow_for_test,
};
mod options;

use domain::{
    compatible_conditional_join_domain, compatible_filter_join_indices_domain, compatible_join_indices_domain,
    compatible_mixed_join_domain,
};
pub(crate) use options::{FilteredJoinOptions, HashJoinOptions};
pub use options::{JoinType, MergeOptions, NullEquality, NullableJoin};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result,
        },
        native_teardown::drop_device_owned_handle,
        table::{Table, TableView},
        types::TypeId,
    },
    ops::ast::AstExpression,
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

/// Device-side gather maps produced by an equality join.
///
/// Both columns are `INT32` device columns. They identify rows in the left and
/// right key tables respectively; unmatched rows contain libcudf's no-match
/// sentinel and are intended for gather paths that use nullifying
/// out-of-bounds handling. Payload columns are not copied until callers gather
/// with these indices.
#[derive(Debug)]
pub(crate) struct JoinIndices {
    /// Row indices for the left/probe table.
    pub left: Column,
    /// Row indices for the right/build table.
    pub right: Column,
}

/// Device-side gather map produced by a single-side join.
///
/// The column is an `INT32` device column indexing rows in the left key table.
/// Payload columns are not copied by these index-producing operations. For
/// left semi joins, duplicate matches on the right side do not duplicate left
/// rows in the output index map.
#[derive(Debug)]
pub(crate) struct SingleSideJoinIndices {
    /// Row indices for the left table.
    pub indices: Column,
}

/// Split output of a low-level join operation.
#[derive(Debug)]
pub(crate) struct JoinSides {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
}

#[derive(Clone, Copy)]
pub(crate) struct SplitJoinTables<'a, 'view> {
    pub(crate) left_keys: &'a TableView<'view>,
    pub(crate) right_keys: &'a TableView<'view>,
    pub(crate) left_payload: &'a TableView<'view>,
    pub(crate) right_payload: &'a TableView<'view>,
}

#[derive(Clone, Copy)]
struct FilterJoinInputs<'a, 'view> {
    left: &'a TableView<'view>,
    right: &'a TableView<'view>,
    left_indices: &'a ColumnView<'view>,
    right_indices: &'a ColumnView<'view>,
    predicate: &'a AstExpression,
}

#[derive(Clone, Copy)]
struct MixedJoinTables<'a, 'view> {
    left_equality: &'a TableView<'view>,
    right_equality: &'a TableView<'view>,
    left_conditional: &'a TableView<'view>,
    right_conditional: &'a TableView<'view>,
}

#[derive(Clone, Copy)]
pub(crate) struct JoinCallContext<'a> {
    output_domain: &'a Arc<ExecutionDomain>,
    exec_options: &'a cudf_sys::cudf_exec_options_t,
    operation: &'static str,
}

impl<'a> JoinCallContext<'a> {
    pub(crate) fn new(
        output_domain: &'a Arc<ExecutionDomain>,
        exec_options: &'a cudf_sys::cudf_exec_options_t,
        operation: &'static str,
    ) -> Self {
        Self { output_domain, exec_options, operation }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizedConditionalPairJoinType {
    Inner,
    Left,
}

impl SizedConditionalPairJoinType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inner => "inner",
            Self::Left => "left",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizedMixedJoinType {
    Inner,
    Left,
    Full,
}

impl SizedMixedJoinType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inner => "inner",
            Self::Left => "left",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SizedMixedSingleSideJoinType {
    LeftSemi,
    LeftAnti,
}

impl SizedMixedSingleSideJoinType {
    fn as_str(self) -> &'static str {
        match self {
            Self::LeftSemi => "left_semi",
            Self::LeftAnti => "left_anti",
        }
    }
}

mod conditional;
mod domain;
mod filter;
mod hash;
mod mixed;

pub(crate) use conditional::*;
pub(crate) use filter::*;
pub(crate) use hash::*;
pub(crate) use mixed::*;

fn check_join_indices_status(
    status: cudf_sys::cudf_status_t,
    null_equality: NullEquality,
    operation: &'static str,
) -> Result<()> {
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_authority", "resolved_domain".to_string()),
        ],
    )
}

struct HashJoinHandle {
    handle: NonNull<cudf_sys::cudf_hash_join_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

struct FilteredJoinHandle {
    handle: NonNull<cudf_sys::cudf_filtered_join_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

struct FilteredJoinProbeHandle {
    handle: NonNull<cudf_sys::cudf_filtered_join_probe_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

struct ConditionalFullJoinSizeDataHandle {
    handle: NonNull<cudf_sys::cudf_conditional_full_join_size_data_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

struct MixedJoinSizeDataHandle {
    handle: NonNull<cudf_sys::cudf_mixed_join_size_data_handle_state>,
    _domain: Arc<ExecutionDomain>,
}

impl fmt::Debug for HashJoinHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for FilteredJoinHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for FilteredJoinProbeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinProbeHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for ConditionalFullJoinSizeDataHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalFullJoinSizeDataHandle").field("handle", &self.handle).finish()
    }
}

impl fmt::Debug for MixedJoinSizeDataHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixedJoinSizeDataHandle").field("handle", &self.handle).finish()
    }
}

// SAFETY: libcudf `hash_join` is documented as reusable for repeated and
// possibly parallel probes. The safe wrapper retains the build table so the
// borrowed build-side device memory outlives the handle.
unsafe impl Send for HashJoinHandle {}
unsafe impl Sync for HashJoinHandle {}

// SAFETY: libcudf `filtered_join` is documented as reusable for repeated
// probes. The safe wrapper retains the build table so the borrowed build-side
// device memory outlives the handle.
unsafe impl Send for FilteredJoinHandle {}
unsafe impl Sync for FilteredJoinHandle {}

// SAFETY: The retained probe-state handle owns its device membership buffer and
// no longer borrows the probe table. The wrapper retains the filtered join so
// build-side device memory remains live until the state is dropped.
unsafe impl Send for FilteredJoinProbeHandle {}
unsafe impl Sync for FilteredJoinProbeHandle {}

// SAFETY: The retained conditional full join size-data handle owns its
// device-side size state. The safe DataFrame wrapper retains input DataFrames,
// and the Rust state lifetime keeps AST literal owners alive.
unsafe impl Send for ConditionalFullJoinSizeDataHandle {}
unsafe impl Sync for ConditionalFullJoinSizeDataHandle {}

// SAFETY: The retained mixed-join size-data handle owns its device-side
// matches-per-row buffer. The safe DataFrame wrapper retains the input
// DataFrames, and the Rust state lifetime keeps AST literal owners alive.
unsafe impl Send for MixedJoinSizeDataHandle {}
unsafe impl Sync for MixedJoinSizeDataHandle {}

impl Drop for HashJoinHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "HashJoinHandle::drop",
            |owned| unsafe { cudf_sys::cudf_hash_join_destroy(owned) },
        );
    }
}

impl Drop for FilteredJoinHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "FilteredJoinHandle::drop",
            |owned| unsafe { cudf_sys::cudf_filtered_join_destroy(owned) },
        );
    }
}

impl Drop for FilteredJoinProbeHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "FilteredJoinProbeHandle::drop",
            |owned| unsafe { cudf_sys::cudf_filtered_join_probe_destroy(owned) },
        );
    }
}

impl Drop for ConditionalFullJoinSizeDataHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "ConditionalFullJoinSizeDataHandle::drop",
            |owned| unsafe { cudf_sys::cudf_conditional_full_join_size_data_destroy(owned) },
        );
    }
}

impl Drop for MixedJoinSizeDataHandle {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self._domain.device_id(),
            "MixedJoinSizeDataHandle::drop",
            |owned| unsafe { cudf_sys::cudf_mixed_join_size_data_destroy(owned) },
        );
    }
}

/// Reusable duplicate-key hash-join build artifact.
///
/// `HashJoin` owns the build key table and a libcudf hash table that borrows
/// that build table's device memory. Probe methods return `INT32` device
/// gather maps; they do not gather payload columns.
///
/// The handle is `Send + Sync`: cloned handles may be probed from multiple
/// host threads while the retained build table stays alive. Each probe still
/// resolves execution provenance from its probe table and returns device
/// columns in that resolved execution domain.
#[derive(Clone)]
pub(crate) struct HashJoin {
    handle: Arc<HashJoinHandle>,
    build: Arc<Table>,
    options: HashJoinOptions,
}

/// Reusable filtered-join build artifact for equality left semi/anti probes.
///
/// `FilteredJoin` owns the right/build key table and a libcudf filtered-join
/// hash table that borrows that build table's device memory. Probe methods
/// return `INT32` device gather maps indexing rows in the left/probe table;
/// they do not gather payload columns.
#[derive(Clone)]
pub(crate) struct FilteredJoin {
    handle: Arc<FilteredJoinHandle>,
    build: Arc<Table>,
    options: FilteredJoinOptions,
}

/// Retained filtered-join probe state for equality left semi/anti probes.
///
/// `FilteredJoinProbeState` owns the device-side membership map produced by a
/// single filtered-join hash-table probe. It exposes the exact selected row
/// count before the gather map is allocated, then materializes the gather map
/// from retained state without probing the hash table again.
pub(crate) struct FilteredJoinProbeState {
    handle: FilteredJoinProbeHandle,
    join: FilteredJoin,
    output_domain: Arc<ExecutionDomain>,
    output_size: usize,
    semi_join: bool,
}

/// Retained conditional full join size data.
///
/// This state owns libcudf's device-side size data for one conditional full
/// join. It exposes the exact output size before gather-map allocation and
/// materializes the two gather maps from the retained size data. The lifetime
/// keeps AST literal owners alive; higher-level wrappers retain input tables
/// whose column memory is borrowed by the C handle.
pub(crate) struct ConditionalFullJoinSizeData<'a> {
    handle: ConditionalFullJoinSizeDataHandle,
    output_domain: Arc<ExecutionDomain>,
    output_size: usize,
    _predicate: &'a AstExpression,
}

/// Retained mixed equality-plus-predicate pair-output join size data.
///
/// This state owns libcudf's device-side size-data buffer for one mixed inner,
/// left, or full join. It exposes the exact output size before gather-map
/// allocation and materializes the two gather maps from the retained size data. The
/// lifetime keeps AST literal owners alive; higher-level wrappers retain input
/// tables whose column memory is borrowed by the C handle.
pub(crate) struct MixedJoinSizeData<'a> {
    handle: MixedJoinSizeDataHandle,
    output_domain: Arc<ExecutionDomain>,
    output_size: usize,
    join_type: SizedMixedJoinType,
    _predicate: &'a AstExpression,
}

/// Retained mixed equality-plus-predicate single-output join size data.
///
/// This state owns libcudf's device-side size-data buffer for one mixed left
/// semi or left anti join. It exposes the exact output size before gather-map
/// allocation and materializes one gather map from the retained size data. The
/// lifetime keeps AST literal owners alive; higher-level wrappers retain input
/// tables whose column memory is borrowed by the C handle.
pub(crate) struct MixedSingleSideJoinSizeData<'a> {
    handle: MixedJoinSizeDataHandle,
    output_domain: Arc<ExecutionDomain>,
    output_size: usize,
    join_type: SizedMixedSingleSideJoinType,
    _predicate: &'a AstExpression,
}

impl fmt::Debug for HashJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoin")
            .field("options", &self.options)
            .field("placement", &self.build.execution_domain().placement())
            .finish()
    }
}

impl fmt::Debug for FilteredJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoin")
            .field("options", &self.options)
            .field("placement", &self.build.execution_domain().placement())
            .finish()
    }
}

impl fmt::Debug for FilteredJoinProbeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredJoinProbeState")
            .field("join_type", &if self.semi_join { "left_semi" } else { "left_anti" })
            .field("output_size", &self.output_size)
            .field("placement", &self.output_domain.placement())
            .finish()
    }
}

impl fmt::Debug for ConditionalFullJoinSizeData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalFullJoinSizeData")
            .field("join_type", &"full")
            .field("output_size", &self.output_size)
            .field("placement", &self.output_domain.placement())
            .finish()
    }
}

impl fmt::Debug for MixedJoinSizeData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixedJoinSizeData")
            .field("join_type", &self.join_type.as_str())
            .field("output_size", &self.output_size)
            .field("placement", &self.output_domain.placement())
            .finish()
    }
}

impl fmt::Debug for MixedSingleSideJoinSizeData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MixedSingleSideJoinSizeData")
            .field("join_type", &self.join_type.as_str())
            .field("output_size", &self.output_size)
            .field("placement", &self.output_domain.placement())
            .finish()
    }
}
