//! GPU-accelerated sorting operations.

use std::ptr;

use crate::{
    column::Column,
    error::{self, Error, ErrorKind, Result},
    table::{Table, TableView},
};

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

fn validate_sort_params(
    operation: &'static str,
    num_keys: i32,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<()> {
    let expected = usize::try_from(num_keys).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "number of sort keys must be >= 0")
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
    })?;

    if let Some(orders) = column_order {
        if orders.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column_order length ({}) must equal number of keys ({})", orders.len(), expected),
            )
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
            .with_context("column_order_len", orders.len().to_string()));
        }
    }

    if let Some(nulls) = null_precedence {
        if nulls.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("null_precedence length ({}) must equal number of keys ({})", nulls.len(), expected),
            )
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
            .with_context("null_precedence_len", nulls.len().to_string()));
        }
    }

    Ok(())
}

/// Helper: convert Option slices to raw pointers.
fn sort_params(column_order: Option<&[Order]>, null_precedence: Option<&[NullOrder]>) -> (*const i32, *const i32) {
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
    validate_sort_params("sorting::sort", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_sort(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::sort",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Stable sort (preserves relative order of equal elements).
pub fn stable_sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::stable_sort", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_stable_sort(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::stable_sort",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
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
    validate_sort_params("sorting::sort_by_key", num_key_columns, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_sort_by_key(values.raw_handle(), keys.raw_handle(), co, np, num_key_columns, &mut out)
        },
        "sorting::sort_by_key",
        [
            ("num_key_columns", num_key_columns.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
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
    validate_sort_params("sorting::sorted_order", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_sorted_order(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::sorted_order",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Check whether the table is already sorted according to the given order.
pub fn is_sorted(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<bool> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::is_sorted", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out: i32 = 0;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_is_sorted(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::is_sorted",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(out != 0)
}
