use std::sync::Arc;

use super::{Column, ColumnView};
use crate::core::error::{self, ErrorKind, Result};

pub(crate) fn slice_view_for_test<'a>(column: &'a Column, offset: usize, length: usize) -> Result<ColumnView<'a>> {
    let operation = "testing::column::slice_view";
    let view = Column::view(column)?;
    let start = i32::try_from(offset).map_err(|_| error::row_range_limit_error(operation, "start", offset))?;
    let end_usize =
        offset.checked_add(length).ok_or_else(|| error::row_range_overflow_error(operation, offset, length))?;
    let len = i32::try_from(length)
        .map_err(|_| error::range_limit_error(ErrorKind::InvalidInput, operation, "row_count", "i32::MAX", length))?;
    let _ = i32::try_from(end_usize).map_err(|_| {
        error::row_range_limit_error(operation, "end", end_usize).with_context("start_row", offset.to_string())
    })?;

    let mut handle = std::ptr::null_mut();
    crate::core::error::check_status_in(
        unsafe { cudf_sys::cudf_column_view_slice(view.raw_handle(), start, len, &mut handle) },
        operation,
    )?;
    unsafe { ColumnView::from_raw_with_domain(handle, Arc::clone(view.execution_domain())) }
}
