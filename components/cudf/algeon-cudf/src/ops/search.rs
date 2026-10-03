//! GPU-accelerated search / set-membership operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
    },
    runtime::execution_domain::ExecutionDomain,
};

pub(crate) fn contains_impl(
    haystack: &ColumnView<'_>,
    needles: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_contains_ex(haystack.raw_handle(), needles.raw_handle(), &mut out, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
