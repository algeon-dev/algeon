use std::{ptr, sync::Arc};

use super::{owned::Column, view::ColumnView};
use crate::{
    core::error::{self, Result},
    runtime::execution_domain::ExecutionDomain,
};

/// Deep-copy a `ColumnView` into an owned `Column`.
///
/// Uses `cudf_column_copy_from_view`, which supports fixed-width, string,
/// and nested types.
pub(crate) fn copy_view_to_owned(cv: &ColumnView<'_>) -> Result<Column> {
    let len = cv.len()?;
    if len == 0 {
        return Column::empty_with_dtype_in_domain(
            cv.data_type()?,
            Arc::clone(cv.execution_domain()),
            "column::copy_view_to_owned",
        );
    }
    let mut handle = ptr::null_mut();
    let exec_options = cv.execution_domain().enter_execution()?;
    error::check_status_in(
        unsafe { cudf_sys::cudf_column_copy_from_view_ex(cv.raw_handle(), exec_options.options(), &mut handle) },
        "column::copy_view_to_owned",
    )?;
    unsafe { Column::from_raw_with_domain(handle, Arc::clone(cv.execution_domain())) }
}

/// Deep-copy a `ColumnView` into an owned `Column` on the output domain.
///
/// Same-dtype casts use this instead of copying on the input domain and
/// relabeling: the copy executes on the output stream and memory resource,
/// so later output-stream reads are ordered after it, teardown frees on the
/// right stream, and the charge lands on the budgeting resource.
pub(crate) fn copy_view_in(
    cv: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<Column> {
    let len = cv.len()?;
    if len == 0 {
        return Column::empty_with_dtype_in_domain(cv.data_type()?, Arc::clone(output_domain), "column::copy_view_in");
    }
    let mut handle = ptr::null_mut();
    error::check_status_in(
        unsafe { cudf_sys::cudf_column_copy_from_view_ex(cv.raw_handle(), exec_options, &mut handle) },
        "column::copy_view_in",
    )?;
    unsafe { Column::from_raw_with_domain(handle, Arc::clone(output_domain)) }
}
