//! GPU-accelerated row gathering (take / index-select) operations.

use std::ptr;

use crate::{
    column::ColumnView,
    error::{self, Result},
    table::{Table, TableView},
};

/// Policy for handling out-of-bounds indices in a gather map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OutOfBoundsPolicy {
    /// Out-of-bounds indices produce null rows in the output.
    Nullify = 0,
    /// No bounds checking — out-of-bounds indices cause undefined behavior.
    /// Use when you are certain all indices are valid, for better performance.
    DontCheck = 1,
}

/// Gather (select) rows from a table by index.
///
/// Produces a new table whose row `i` is `source_table[gather_map[i]]`.
/// The `gather_map` must be a non-nullable column of `INT32` indices.
///
/// A negative value `i` in the gather map is interpreted as `i + n`,
/// where `n` is the number of rows in the source table.
///
/// # Example
/// ```no_run
/// use cudf::{column::Column, copying, table::Table};
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// let result = copying::gather(
///     &table.view().unwrap(),
///     &map.view().unwrap(),
///     copying::OutOfBoundsPolicy::DontCheck,
/// )
/// .unwrap();
/// // result row 0 = source row 2 (30), result row 1 = source row 0 (10)
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>, bounds_policy: OutOfBoundsPolicy) -> Result<Table> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_gather(input.raw_handle(), gather_map.raw_handle(), bounds_policy as i32, &mut out) },
        "copying::gather",
        [("bounds_policy", format!("{:?}", bounds_policy).to_lowercase())],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}
