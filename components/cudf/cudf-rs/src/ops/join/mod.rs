//! GPU-accelerated join operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::merge`](crate::DataFrame::merge) with
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
//! let result =
//!     left.merge(&right, MergeOptions::new().on(["id"]).how_inner()).unwrap();
//! assert_eq!(
//!     result.series("id").unwrap().to_vec_strict::<i32>().unwrap(),
//!     vec![2, 3]
//! );
//! ```

use std::{
    fmt,
    os::raw::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

mod dataframe;
mod options;
pub(crate) mod testing_hooks;

pub use options::{HashJoinOptions, JoinType, MergeOptions, NullEquality, NullableJoin};

use crate::{
    core::{
        column::Column,
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Device-side gather maps produced by an equality join.
///
/// Both columns are `INT32` device columns. They identify rows in the left and
/// right key tables respectively; unmatched rows contain libcudf's no-match
/// sentinel and are intended for gather paths that use nullifying
/// out-of-bounds handling. Payload columns are not copied until callers gather
/// with these indices.
pub struct JoinIndices {
    /// Row indices for the left/probe table.
    pub left: Column,
    /// Row indices for the right/build table.
    pub right: Column,
}

/// Split output of a low-level join operation.
pub struct JoinSides {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
}

pub(crate) fn join_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinSides> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (join_type, exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_inner_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_left_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_full_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_inner_join_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_left_join_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_full_join_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    Ok(JoinSides {
        left: unsafe { Table::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Table::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

fn check_join_indices_status(
    status: cudf_sys::cudf_status_t,
    null_equality: NullEquality,
    exec_options_present: bool,
    operation: &'static str,
) -> Result<()> {
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_context_provided", exec_options_present.to_string()),
        ],
    )
}

pub(crate) fn join_indices_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinIndices> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    let null_eq_ffi = null_equality.to_join_raw();
    let status = match (join_type, exec_options) {
        (JoinType::Inner, None) => unsafe {
            cudf_sys::cudf_inner_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Left, None) => unsafe {
            cudf_sys::cudf_left_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Full, None) => unsafe {
            cudf_sys::cudf_full_join_indices(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
            )
        },
        (JoinType::Inner, Some(options)) => unsafe {
            cudf_sys::cudf_inner_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Left, Some(options)) => unsafe {
            cudf_sys::cudf_left_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
        (JoinType::Full, Some(options)) => unsafe {
            cudf_sys::cudf_full_join_indices_ex(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                null_eq_ffi,
                &mut out_left,
                &mut out_right,
                options,
            )
        },
    };

    check_join_indices_status(status, null_equality, exec_options.is_some(), operation)?;

    Ok(JoinIndices {
        left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
        right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
    })
}

fn compatible_join_domain(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        left_keys.execution_domain(),
        [right_keys.execution_domain(), left_payload.execution_domain(), right_payload.execution_domain()],
        &ctx,
        operation,
    )
}

fn compatible_join_indices_domain(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::binary(left_keys.execution_domain(), right_keys.execution_domain(), &ctx, operation)
}

/// Inner join: returns only rows with matching keys in both tables.
///
/// `left_keys` and `right_keys` define the join condition.
/// `left_payload` and `right_payload` are additional columns carried through.
/// The result tables contain keys + payload columns concatenated.
pub fn inner_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let execution = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::inner_join")?;
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Inner,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::inner_join",
    )
}

/// Inner join indices for matching rows in both key tables.
///
/// Returns `INT32` device columns that can be used as gather maps for the
/// left and right tables. This allocates only the index columns; payload
/// columns are not copied or gathered by this operation.
pub fn inner_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let execution = compatible_join_indices_domain(left_keys, right_keys, "join::inner_join_indices")?;
    join_indices_impl(
        left_keys,
        right_keys,
        null_equality,
        JoinType::Inner,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::inner_join_indices",
    )
}

/// Left join: all left rows are preserved; right side is null-padded
/// for non-matching rows.
pub fn left_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let execution = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::left_join")?;
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Left,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::left_join",
    )
}

/// Left join indices preserving all rows from the left key table.
///
/// Misses in `right` contain libcudf's no-match sentinel and should be used
/// with nullifying gather behavior when materializing the right side.
pub fn left_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let execution = compatible_join_indices_domain(left_keys, right_keys, "join::left_join_indices")?;
    join_indices_impl(
        left_keys,
        right_keys,
        null_equality,
        JoinType::Left,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::left_join_indices",
    )
}

// ---------------------------------------------------------------------------
// Low-level join functions
// ---------------------------------------------------------------------------

/// Full (outer) join: all rows from both sides are preserved, with
/// null-padding where there is no match.
pub fn full_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinSides> {
    let execution = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::full_join")?;
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Full,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::full_join",
    )
}

/// Full join indices preserving rows from both key tables.
///
/// Misses on either side contain libcudf's no-match sentinel and should be
/// used with nullifying gather behavior when materializing either side.
pub fn full_join_indices(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinIndices> {
    let execution = compatible_join_indices_domain(left_keys, right_keys, "join::full_join_indices")?;
    join_indices_impl(
        left_keys,
        right_keys,
        null_equality,
        JoinType::Full,
        execution.output_domain(),
        Some(execution.exec_options()),
        "join::full_join_indices",
    )
}

struct HashJoinHandle {
    handle: NonNull<c_void>,
    _domain: Arc<ExecutionDomain>,
}

impl fmt::Debug for HashJoinHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinHandle").field("handle", &self.handle).finish()
    }
}

// SAFETY: libcudf `hash_join` is documented as reusable for repeated and
// possibly parallel probes. The safe wrapper retains the build table so the
// borrowed build-side device memory outlives the handle.
unsafe impl Send for HashJoinHandle {}
unsafe impl Sync for HashJoinHandle {}

impl Drop for HashJoinHandle {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_hash_join_destroy(self.handle.as_ptr()) };
    }
}

/// Reusable duplicate-key hash-join build artifact.
///
/// `HashJoin` owns the build key table and a libcudf hash table that borrows
/// that build table's device memory. Probe methods return `INT32` device
/// gather maps; they do not gather payload columns.
#[derive(Clone)]
pub struct HashJoin {
    handle: Arc<HashJoinHandle>,
    build: Arc<Table>,
    options: HashJoinOptions,
}

impl fmt::Debug for HashJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoin")
            .field("options", &self.options)
            .field("placement", &self.build.execution_domain().placement())
            .finish()
    }
}

impl HashJoin {
    /// Build a reusable hash-join artifact from an owned build key table.
    ///
    /// The build table is retained by the returned handle because libcudf's
    /// hash-join state borrows build-side device memory. Construction may
    /// allocate GPU memory for the hash table but does not copy payload
    /// columns.
    pub fn new(build: Table, options: HashJoinOptions) -> Result<Self> {
        let operation = "hash_join::new";
        let handle = {
            let build_view = build.view().map_err(|e| e.reanchor_if_needed(operation))?;
            let exec_options = build.execution_domain().exec_options_raw();
            let mut handle = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_hash_join_create_ex(
                        build_view.raw_handle(),
                        options.nullable_join_setting().to_raw(),
                        options.null_eq().to_join_raw(),
                        options.load_factor_value(),
                        &mut handle,
                        &exec_options,
                    )
                },
                operation,
                [
                    ("nullable_join", format!("{:?}", options.nullable_join_setting()).to_lowercase()),
                    ("null_equality", format!("{:?}", options.null_eq()).to_lowercase()),
                    ("load_factor", options.load_factor_value().to_string()),
                    ("execution_context_provided", true.to_string()),
                ],
            )?;
            HashJoinHandle {
                handle: error::require_non_null(handle, operation, "hash join")?,
                _domain: Arc::clone(build.execution_domain()),
            }
        };

        Ok(Self { handle: Arc::new(handle), build: Arc::new(build), options })
    }

    /// Returns the retained build key table.
    pub fn build_keys(&self) -> &Table {
        &self.build
    }

    /// Returns the options used when this hash table was built.
    pub fn options(&self) -> HashJoinOptions {
        self.options
    }

    /// Probe with inner-join semantics.
    ///
    /// The returned `left` column indexes the probe table, and `right` indexes
    /// the retained build table.
    pub fn inner_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        self.probe_indices(probe_keys, JoinType::Inner, "hash_join::inner_join_indices")
    }

    /// Probe with left-join semantics.
    ///
    /// Misses in the returned `right` column contain libcudf's no-match
    /// sentinel for nullifying gather.
    pub fn left_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        self.probe_indices(probe_keys, JoinType::Left, "hash_join::left_join_indices")
    }

    /// Probe with full-join semantics.
    ///
    /// Misses on either side contain libcudf's no-match sentinel for
    /// nullifying gather.
    pub fn full_join_indices(&self, probe_keys: &TableView<'_>) -> Result<JoinIndices> {
        self.probe_indices(probe_keys, JoinType::Full, "hash_join::full_join_indices")
    }

    fn probe_indices(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let ctx = ExecutionOverrides::new();
        let execution =
            ResolvedExecution::many(probe_keys.execution_domain(), [self.build.execution_domain()], &ctx, operation)?;
        self.probe_indices_impl(
            probe_keys,
            join_type,
            execution.output_domain(),
            Some(execution.exec_options()),
            operation,
        )
    }

    fn probe_indices_impl(
        &self,
        probe_keys: &TableView<'_>,
        join_type: JoinType,
        output_domain: &Arc<ExecutionDomain>,
        exec_options: Option<&cudf_sys::cudf_exec_options_t>,
        operation: &'static str,
    ) -> Result<JoinIndices> {
        let mut out_left = ptr::null_mut();
        let mut out_right = ptr::null_mut();
        let status = match (join_type, exec_options) {
            (JoinType::Inner, None) => unsafe {
                cudf_sys::cudf_hash_join_inner_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Left, None) => unsafe {
                cudf_sys::cudf_hash_join_left_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Full, None) => unsafe {
                cudf_sys::cudf_hash_join_full_join(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                )
            },
            (JoinType::Inner, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_inner_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Left, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_left_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
            (JoinType::Full, Some(options)) => unsafe {
                cudf_sys::cudf_hash_join_full_join_ex(
                    self.handle.handle.as_ptr(),
                    probe_keys.raw_handle(),
                    &mut out_left,
                    &mut out_right,
                    options,
                )
            },
        };

        error::check_status_in_with_details(
            status,
            operation,
            [
                ("nullable_join", format!("{:?}", self.options.nullable_join_setting()).to_lowercase()),
                ("null_equality", format!("{:?}", self.options.null_eq()).to_lowercase()),
                ("execution_context_provided", exec_options.is_some().to_string()),
            ],
        )?;

        Ok(JoinIndices {
            left: unsafe { Column::from_raw_with_domain(out_left, Arc::clone(output_domain)) }?,
            right: unsafe { Column::from_raw_with_domain(out_right, Arc::clone(output_domain)) }?,
        })
    }
}
