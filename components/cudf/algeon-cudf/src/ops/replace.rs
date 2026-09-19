//! GPU-accelerated null replacement operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        scalar::Scalar,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
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
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let series =
///     rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap();
/// let result = series.fillna(99i32).unwrap();
/// assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 99, 3]);
/// ```
pub fn replace_nulls(input: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        input.execution_domain(),
        replacement.execution_domain(),
        &ctx,
        "replace::replace_nulls",
    )?;
    replace_nulls_with_exec_options(
        input,
        replacement,
        execution.output_domain(),
        execution.exec_options(),
        "replace::replace_nulls",
    )
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

/// Replace null values in a column with corresponding values from another column.
///
/// Returns a new column where every null element in `input` is replaced
/// by the corresponding element from `replacement`. Non-null elements in
/// `input` are copied unchanged. Both columns must have the same type and
/// length.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let left = rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap();
/// let right = rt.series(&[10i32, 20, 30]).unwrap();
/// let result = left.fillna_with_series(&right).unwrap();
/// assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
/// ```
pub fn replace_nulls_column(input: &ColumnView<'_>, replacement: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        input.execution_domain(),
        replacement.execution_domain(),
        &ctx,
        "replace::replace_nulls_column",
    )?;
    replace_nulls_column_with_exec_options(
        input,
        replacement,
        execution.output_domain(),
        execution.exec_options(),
        "replace::replace_nulls_column",
    )
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

/// Replace null values using a directional fill policy (forward or backward fill).
///
/// - [`FillPolicy::Preceding`] (forward fill): each null is replaced with the
///   nearest non-null value that precedes it. Leading nulls remain null.
/// - [`FillPolicy::Following`] (backward fill): each null is replaced with the
///   nearest non-null value that follows it. Trailing nulls remain null.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let series = rt
///     .nullable_series(&[1i32, 0, 0, 4], &[true, false, false, true])
///     .unwrap();
/// let result = series.ffill().unwrap();
/// assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 1, 1, 4]);
/// ```
pub fn replace_nulls_policy(input: &ColumnView<'_>, policy: FillPolicy) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "replace::replace_nulls_policy")?;
    replace_nulls_policy_with_exec_options(
        input,
        policy,
        execution.output_domain(),
        execution.exec_options(),
        "replace::replace_nulls_policy",
    )
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
