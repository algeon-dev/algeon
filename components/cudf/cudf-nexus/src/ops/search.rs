//! GPU-accelerated search / set-membership operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
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

/// Test which elements of `needles` exist anywhere in `haystack`
/// (element-wise set membership / "isin").
///
/// Returns a `BOOL8` column with the same size as `needles`. Element `i`
/// is `true` if `needles[i]` is found anywhere in `haystack`, `false`
/// otherwise. Null rows in `needles` produce null rows in the output.
///
/// Both columns must have the same data type.
///
/// This is the building block for [`DataFrame::filter_isin`](crate::frame::dataframe::DataFrame::filter_isin),
/// which wraps `contains` + `filter` into a one-liner convenience method.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let haystack = rt.series(&[10i32, 20, 30]).unwrap();
/// let needles = rt.series(&[20i32, 40]).unwrap();
/// let result = needles.isin(&haystack).unwrap();
/// assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false]);
/// ```
pub fn contains(haystack: &ColumnView<'_>, needles: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(haystack.execution_domain(), needles.execution_domain(), &ctx, "search::contains")?;
    contains_impl(haystack, needles, execution.output_domain(), execution.exec_options(), "search::contains")
}
