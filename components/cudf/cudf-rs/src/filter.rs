//! GPU-accelerated filtering operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{column::Column, filter, table::Table};
//!
//! let col = Column::from_slice(&[10i32, 20, 30, 40]).unwrap();
//! let table = Table::from_columns(vec![col]).unwrap();
//! let mask = Column::from_slice(&[true, false, true, false]).unwrap();
//!
//! let result = filter::apply_boolean_mask(
//!     &table.view().unwrap(),
//!     &mask.view().unwrap(),
//! )
//! .unwrap();
//! // result contains rows 0 and 2: [10, 30]
//! ```

use std::ptr;

use crate::{
    column::ColumnView,
    error::{self, Result},
    table::{Table, TableView},
};

fn apply_boolean_mask_impl(
    input: &TableView<'_>,
    mask: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_apply_boolean_mask_ex(input.raw_handle(), mask.raw_handle(), &mut out, options) },
        None => unsafe { cudf_sys::cudf_apply_boolean_mask(input.raw_handle(), mask.raw_handle(), &mut out) },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Filter rows of a table using a boolean mask column.
///
/// Rows where the mask is `true` are kept; rows where the mask is `false`
/// or null are dropped. The mask must be a `BOOL8` column with the same
/// number of rows as the input table.
pub fn apply_boolean_mask(input: &TableView<'_>, mask: &ColumnView<'_>) -> Result<Table> {
    apply_boolean_mask_impl(input, mask, None, "filter::apply_boolean_mask")
}

pub(crate) fn apply_boolean_mask_with_exec_options(
    input: &TableView<'_>,
    mask: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    apply_boolean_mask_impl(input, mask, exec_options, operation)
}
