//! GPU-accelerated filtering operations.

use crate::column::ColumnView;
use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::ptr;

/// Filter rows of a table using a boolean mask column.
///
/// Rows where the mask is `true` are kept; rows where the mask is `false`
/// or null are dropped. The mask must be a `BOOL8` column with the same
/// number of rows as the input table.
pub fn apply_boolean_mask(input: &TableView<'_>, mask: &ColumnView<'_>) -> Result<Table> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        unsafe {
            cudf_sys::cudf_apply_boolean_mask(
                input.raw_handle(),
                mask.raw_handle(),
                &mut out,
            )
        },
        "filter::apply_boolean_mask",
    )?;
    Ok(unsafe { Table::from_raw(out) })
}
