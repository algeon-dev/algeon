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

/// Fill policy for [`replace_nulls_policy`].
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

/// Replace null values in a column with a scalar.
///
/// Returns a new column where every null element in `input` is replaced
/// by `replacement`. Non-null elements are copied unchanged.
/// The scalar type must match the column type.
///
/// # Example
/// ```no_run
/// use cudf::{
///     core::{Column, Scalar},
///     ops::replace,
/// };
///
/// let col = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
///     .unwrap();
/// let fill = Scalar::from_value(99i32).unwrap();
/// let result = replace::replace_nulls(&col.view().unwrap(), &fill).unwrap();
/// // result: [1, 99, 3]
/// ```
pub fn replace_nulls(input: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    ExecutionDomain::require_same_device_and_fence_secondary(
        &output_domain,
        replacement.execution_domain(),
        "replace::replace_nulls",
    )?;
    let exec_options = output_domain.explicit_exec_options_raw();
    replace_nulls_with_exec_options(input, replacement, exec_options.as_ref(), "replace::replace_nulls")
        .map(|column| column.with_execution_domain(output_domain))
}

pub(crate) fn replace_nulls_with_exec_options(
    input: &ColumnView<'_>,
    replacement: &Scalar,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        None => unsafe { cudf_sys::cudf_replace_nulls_scalar(input.raw_handle(), replacement.raw_handle(), &mut out) },
        Some(options) => unsafe {
            cudf_sys::cudf_replace_nulls_scalar_ex(input.raw_handle(), replacement.raw_handle(), &mut out, options)
        },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Replace null values in a column with corresponding values from another column.
///
/// Returns a new column where every null element in `input` is replaced
/// by the corresponding element from `replacement`. Non-null elements in
/// `input` are copied unchanged. Both columns must have the same type and
/// length.
///
/// # Example
/// ```no_run
/// use cudf::{core::Column, ops::replace};
///
/// let left = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
///     .unwrap();
/// let right = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let result = replace::replace_nulls_column(
///     &left.view().unwrap(),
///     &right.view().unwrap(),
/// )
/// .unwrap();
/// // result: [1, 20, 3]
/// ```
pub fn replace_nulls_column(input: &ColumnView<'_>, replacement: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(input.execution_domain());
    ExecutionDomain::require_same_device_and_fence_secondary(
        &output_domain,
        replacement.execution_domain(),
        "replace::replace_nulls_column",
    )?;
    let exec_options = output_domain.explicit_exec_options_raw();
    replace_nulls_column_with_exec_options(input, replacement, exec_options.as_ref(), "replace::replace_nulls_column")
        .map(|column| column.with_execution_domain(output_domain))
}

pub(crate) fn replace_nulls_column_with_exec_options(
    input: &ColumnView<'_>,
    replacement: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        None => unsafe { cudf_sys::cudf_replace_nulls_column(input.raw_handle(), replacement.raw_handle(), &mut out) },
        Some(options) => unsafe {
            cudf_sys::cudf_replace_nulls_column_ex(input.raw_handle(), replacement.raw_handle(), &mut out, options)
        },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Replace null values using a directional fill policy (forward or backward fill).
///
/// - [`FillPolicy::Preceding`] (forward fill): each null is replaced with the
///   nearest non-null value that precedes it. Leading nulls remain null.
/// - [`FillPolicy::Following`] (backward fill): each null is replaced with the
///   nearest non-null value that follows it. Trailing nulls remain null.
///
/// # Example
/// ```no_run
/// use cudf::{
///     core::Column,
///     ops::replace::{self, FillPolicy},
/// };
///
/// let col = Column::from_slice_nullable(
///     &[1i32, 0, 0, 4],
///     &[true, false, false, true],
/// )
/// .unwrap();
/// let result = replace::replace_nulls_policy(
///     &col.view().unwrap(),
///     FillPolicy::Preceding,
/// )
/// .unwrap();
/// // result: [1, 1, 1, 4]
/// ```
pub fn replace_nulls_policy(input: &ColumnView<'_>, policy: FillPolicy) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    replace_nulls_policy_with_exec_options(input, policy, exec_options.as_ref(), "replace::replace_nulls_policy")
        .map(|column| column.with_execution_domain(output_domain))
}

pub(crate) fn replace_nulls_policy_with_exec_options(
    input: &ColumnView<'_>,
    policy: FillPolicy,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        None => unsafe { cudf_sys::cudf_replace_nulls_policy(input.raw_handle(), policy as i32, &mut out) },
        Some(options) => unsafe {
            cudf_sys::cudf_replace_nulls_policy_ex(input.raw_handle(), policy as i32, &mut out, options)
        },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string()), ("policy", format!("{policy:?}"))],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}
