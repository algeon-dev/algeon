use std::{ptr, sync::Arc};

use super::{owned::Column, view::ColumnView};
use crate::core::error::{self, Result};

/// Deep-copy a `ColumnView` into an owned `Column`.
///
/// Uses `cudf_column_copy_from_view`, which supports fixed-width, string,
/// and nested types.
pub(crate) fn copy_view_to_owned(cv: &ColumnView<'_>) -> Result<Column> {
    let len = cv.len()?;
    if len == 0 {
        return Column::empty_in_domain(
            cv.data_type()?.id(),
            Arc::clone(cv.execution_domain()),
            "column::copy_view_to_owned",
        );
    }
    let mut handle = ptr::null_mut();
    let exec_options = cv.execution_domain().exec_options_raw();
    error::check_status_in(
        unsafe { cudf_sys::cudf_column_copy_from_view_ex(cv.raw_handle(), &exec_options, &mut handle) },
        "column::copy_view_to_owned",
    )?;
    unsafe { Column::from_raw_with_domain(handle, Arc::clone(cv.execution_domain())) }
}
