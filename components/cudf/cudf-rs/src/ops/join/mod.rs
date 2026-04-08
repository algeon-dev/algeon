//! GPU-accelerated join operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::merge`](crate::DataFrame::merge) with
//! [`MergeOptions`](crate::MergeOptions).
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     core::{Column, Table},
//!     ops::join,
//! };
//!
//! let left_key = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let left_val = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let right_key = Column::from_slice(&[2i32, 3, 4]).unwrap();
//! let right_val = Column::from_slice(&[200i32, 300, 400]).unwrap();
//!
//! let lk = Table::from_columns(vec![left_key]).unwrap();
//! let lv = Table::from_columns(vec![left_val]).unwrap();
//! let rk = Table::from_columns(vec![right_key]).unwrap();
//! let rv = Table::from_columns(vec![right_val]).unwrap();
//!
//! let result = join::inner_join(
//!     &lk.view().unwrap(),
//!     &rk.view().unwrap(),
//!     &lv.view().unwrap(),
//!     &rv.view().unwrap(),
//!     join::NullEquality::Equal,
//! )
//! .unwrap();
//! // result matches keys 2 and 3
//! ```

use std::ptr;

mod dataframe;
mod options;

pub use options::{JoinType, MergeOptions, NullEquality};

use crate::{
    core::{
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

/// Result of a join operation.
pub struct JoinResult {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn join_impl(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
    join_type: JoinType,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<JoinResult> {
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

    error::check_status_in_with(
        status,
        operation,
        [
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}

fn compatible_join_domain(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    operation: &'static str,
) -> Result<std::sync::Arc<ExecutionDomain>> {
    let output_domain = std::sync::Arc::clone(left_keys.execution_domain());

    for input in [right_keys, left_payload, right_payload] {
        ExecutionDomain::require_same_device_for_mixed_inputs(
            left_keys.execution_domain(),
            input.execution_domain(),
            operation,
        )?;
        if !ExecutionDomain::same_provenance(input.execution_domain(), &output_domain) {
            input.execution_domain().fence_stream(operation)?;
        }
    }

    Ok(output_domain)
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
) -> Result<JoinResult> {
    let output_domain = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::inner_join")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Inner,
        exec_options.as_ref(),
        "join::inner_join",
    )
    .map(|result| JoinResult {
        left: result.left.with_execution_domain(std::sync::Arc::clone(&output_domain)),
        right: result.right.with_execution_domain(output_domain),
    })
}

/// Left join: all left rows are preserved; right side is null-padded
/// for non-matching rows.
pub fn left_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinResult> {
    let output_domain = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::left_join")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Left,
        exec_options.as_ref(),
        "join::left_join",
    )
    .map(|result| JoinResult {
        left: result.left.with_execution_domain(std::sync::Arc::clone(&output_domain)),
        right: result.right.with_execution_domain(output_domain),
    })
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
) -> Result<JoinResult> {
    let output_domain = compatible_join_domain(left_keys, right_keys, left_payload, right_payload, "join::full_join")?;
    let exec_options = output_domain.explicit_exec_options_raw();
    join_impl(
        left_keys,
        right_keys,
        left_payload,
        right_payload,
        null_equality,
        JoinType::Full,
        exec_options.as_ref(),
        "join::full_join",
    )
    .map(|result| JoinResult {
        left: result.left.with_execution_domain(std::sync::Arc::clone(&output_domain)),
        right: result.right.with_execution_domain(output_domain),
    })
}

