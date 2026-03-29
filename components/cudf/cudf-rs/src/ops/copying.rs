//! GPU-accelerated row gathering (take / index-select) operations.

use std::ptr;

use crate::core::{
    column::ColumnView,
    error::{self, Result},
    table::{Table, TableView},
};

fn gather_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: i32,
    bounds_policy_label: &'static str,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_gather_ex(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_gather(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out) },
    };
    error::check_status_in_with(
        status,
        operation,
        [
            ("bounds_policy", bounds_policy_label.to_owned()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Gather (select) rows from a table by index.
///
/// Produces a new table whose row `i` is `source_table[gather_map[i]]`.
/// The `gather_map` must be a non-nullable column of `INT32` indices.
///
/// A negative value `i` in the gather map is interpreted as `i + n`,
/// where `n` is the number of rows in the source table.
///
/// Out-of-bounds indices produce null rows in the output.
///
/// # Example
/// ```no_run
/// use cudf::{Column, Table, ops::copying};
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// let result =
///     copying::gather(&table.view().unwrap(), &map.view().unwrap()).unwrap();
/// // result row 0 = source row 2 (30), result row 1 = source row 0 (10)
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    gather_impl(input, gather_map, 0, "nullify", None, "copying::gather")
}

pub(crate) fn gather_with_exec_options(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    gather_impl(input, gather_map, 0, "nullify", exec_options, operation)
}

/// Gather (select) rows from a table by index, without bounds checking.
///
/// This is identical to [`gather`] but skips bounds validation for
/// better performance. Out-of-bounds indices cause **undefined behavior**
/// (invalid GPU memory access).
///
/// # Safety
///
/// Caller must ensure that every value in `gather_map` is a valid index
/// into `input` (i.e., in the range `[-n, n)` where `n` is the number
/// of rows in `input`). Violating this causes undefined behavior.
///
/// # Example
/// ```no_run
/// use cudf::{Column, Table, ops::copying};
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// // SAFETY: indices 2 and 0 are valid for a 3-row table.
/// let result = unsafe {
///     copying::gather_unchecked(&table.view().unwrap(), &map.view().unwrap())
/// }
/// .unwrap();
/// ```
///
/// Prefer [`gather`] unless profiling shows the bounds check is a bottleneck.
pub unsafe fn gather_unchecked(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    gather_impl(input, gather_map, 1, "dont_check", None, "copying::gather_unchecked")
}

pub(crate) unsafe fn gather_unchecked_with_exec_options(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    gather_impl(input, gather_map, 1, "dont_check", exec_options, operation)
}
