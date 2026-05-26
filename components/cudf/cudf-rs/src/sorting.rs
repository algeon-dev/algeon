//! GPU-accelerated sorting operations.

use crate::column::Column;
use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::ptr;

/// Column sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Order {
    Ascending = 0,
    Descending = 1,
}

/// Where nulls appear in the sorted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullOrder {
    After = 0,
    Before = 1,
}

/// Helper: convert Option slices to raw pointers and key count.
fn sort_params(
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> (*const i32, *const i32) {
    let co = match column_order {
        Some(s) => s.as_ptr() as *const i32,
        None => ptr::null(),
    };
    let np = match null_precedence {
        Some(s) => s.as_ptr() as *const i32,
        None => ptr::null(),
    };
    (co, np)
}

/// Sort the rows of a table by all its columns.
///
/// If `column_order` or `null_precedence` are `None`, cuDF defaults are
/// used (ascending, nulls after).
pub fn sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_keys = input.num_columns()?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_sort(input.raw_handle(), co, np, num_keys, &mut out)
    })?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Stable sort (preserves relative order of equal elements).
pub fn stable_sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_keys = input.num_columns()?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_stable_sort(input.raw_handle(), co, np, num_keys, &mut out)
    })?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Sort the `values` table by a separate `keys` table.
pub fn sort_by_key(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_key_columns = keys.num_columns()?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_sort_by_key(
            values.raw_handle(),
            keys.raw_handle(),
            co,
            np,
            num_key_columns,
            &mut out,
        )
    })?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Return the row indices that would sort the table.
///
/// The result is a single INT32 column.
pub fn sorted_order(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Column> {
    let num_keys = input.num_columns()?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_sorted_order(input.raw_handle(), co, np, num_keys, &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Check whether the table is already sorted according to the given order.
pub fn is_sorted(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<bool> {
    let num_keys = input.num_columns()?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out: i32 = 0;
    error::check_status(unsafe {
        cudf_sys::cudf_is_sorted(input.raw_handle(), co, np, num_keys, &mut out)
    })?;
    Ok(out != 0)
}
