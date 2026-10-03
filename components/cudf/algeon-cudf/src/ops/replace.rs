//! GPU-accelerated null replacement operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        scalar::Scalar,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// Fill policy for forward and backward null replacement.
///
/// - `Preceding` (forward fill): replace each null with the nearest preceding non-null value.
/// - `Following` (backward fill): replace each null with the nearest following non-null value.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillPolicy {
    /// Forward fill: nulls are replaced by the nearest preceding non-null value.
    Preceding = 0,
    /// Backward fill: nulls are replaced by the nearest following non-null value.
    Following = 1,
}

pub(crate) fn replace_nulls_with_exec_options(
    input: &ColumnView<'_>,
    replacement: &Scalar,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_replace_nulls_scalar_ex(input.raw_handle(), replacement.raw_handle(), &mut out, exec_options)
    };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn replace_nulls_column_with_exec_options(
    input: &ColumnView<'_>,
    replacement: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_replace_nulls_column_ex(input.raw_handle(), replacement.raw_handle(), &mut out, exec_options)
    };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn replace_nulls_policy_with_exec_options(
    input: &ColumnView<'_>,
    policy: FillPolicy,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_replace_nulls_policy_ex(input.raw_handle(), policy as i32, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [("execution_authority", "retained_domain".to_string()), ("policy", format!("{policy:?}"))],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
