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

use std::{ptr, sync::Arc};

mod dataframe;
mod options;
pub(crate) mod testing_hooks;

pub use options::{JoinType, MergeOptions, NullEquality};

use crate::{
    core::{
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

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
