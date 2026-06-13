use std::sync::Arc;

use super::{Column, ColumnView};
use crate::core::{
    error::{self, Result},
    row_range,
};

pub(crate) fn slice_view_for_test<'a>(column: &'a Column, offset: usize, length: usize) -> Result<ColumnView<'a>> {
    let operation = "testing::column::slice_view";
    let view = Column::view(column)?;
    let start = i32::try_from(offset).map_err(|_| error::row_range_limit_error(operation, "start", offset))?;
    let len = match i32::try_from(length) {
        Ok(len) => len,
        Err(_) if offset == 0 => return Err(error::range_limit_error(operation, "row_count", "i32::MAX", length)),
        Err(_) => {
            let end =
                offset.checked_add(length).ok_or_else(|| error::row_range_overflow_error(operation, offset, length))?;
            return Err(
                error::row_range_limit_error(operation, "end", end).with_detail("start_row", offset.to_string())
            );
        },
    };
    let (_start, _end) = row_range::checked_bounds_i32(offset, length, operation)?;

    let mut handle = std::ptr::null_mut();
    crate::core::error::check_status_in(
        unsafe { cudf_sys::cudf_column_view_slice(view.raw_handle(), start, len, &mut handle) },
        operation,
    )?;
    unsafe { ColumnView::from_raw_with_domain(handle, Arc::clone(view.execution_domain())) }
}
