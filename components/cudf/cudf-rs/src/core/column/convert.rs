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
        return Column::empty(cv.data_type()?.id())
            .map(|col| col.with_execution_domain(Arc::clone(cv.execution_domain())));
    }
    let mut handle = ptr::null_mut();
    let exec_options = cv.execution_domain().explicit_exec_options_raw();
    error::check_status_in(
        match exec_options.as_ref() {
            Some(options) => unsafe { cudf_sys::cudf_column_copy_from_view_ex(cv.raw_handle(), options, &mut handle) },
            None => unsafe { cudf_sys::cudf_column_copy_from_view(cv.raw_handle(), &mut handle) },
        },
        "column::copy_view_to_owned",
    )?;
    Ok(unsafe { Column::from_raw_with_domain(handle, Arc::clone(cv.execution_domain())) })
}
