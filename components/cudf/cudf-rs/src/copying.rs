//! GPU-accelerated row gathering (take / index-select) operations.

use std::ptr;

use crate::{
    column::ColumnView,
    error::{self, Result},
    table::{Table, TableView},
};

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
/// use cudf::{column::Column, copying, table::Table};
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// let result =
///     copying::gather(&table.view().unwrap(), &map.view().unwrap()).unwrap();
/// // result row 0 = source row 2 (30), result row 1 = source row 0 (10)
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let mut out = ptr::null_mut();
    let policy = 0i32; // Nullify
    error::check_status_in_with(
        // SAFETY: input and gather_map handles are valid (NonNull maintained by their types).
        unsafe { cudf_sys::cudf_gather(input.raw_handle(), gather_map.raw_handle(), policy, &mut out) },
        "copying::gather",
        [("bounds_policy", "nullify".to_owned())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw(out) })
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
/// use cudf::{column::Column, copying, table::Table};
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
    let mut out = ptr::null_mut();
    let policy = 1i32; // DontCheck
    error::check_status_in_with(
        // SAFETY: input and gather_map handles are valid. Caller guarantees
        // all indices are in bounds (unsafe fn contract).
        unsafe { cudf_sys::cudf_gather(input.raw_handle(), gather_map.raw_handle(), policy, &mut out) },
        "copying::gather_unchecked",
        [("bounds_policy", "dont_check".to_owned())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw(out) })
}
