//! GPU-accelerated join operations.

use std::ptr;

use crate::{
    error::{self, Result},
    table::{Table, TableView},
};

/// How to compare null keys in join operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullEquality {
    /// Null keys are considered equal (null == null is true).
    Equal = 0,
    /// Null keys are considered unequal (null == null is false).
    Unequal = 1,
}

/// Result of a join operation.
pub struct JoinResult {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
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
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_inner_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::inner_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
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
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_left_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::left_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}

/// Full (outer) join: all rows from both sides are preserved, with
/// null-padding where there is no match.
pub fn full_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinResult> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_full_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::full_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}
