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
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_contains_ex(haystack.raw_handle(), needles.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_contains(haystack.raw_handle(), needles.raw_handle(), &mut out) },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    Ok(unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) })
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
/// let haystack = rt.series("haystack", &[10i32, 20, 30]).unwrap();
/// let needles = rt.series("needles", &[20i32, 40]).unwrap();
/// let result = needles.isin(&haystack).unwrap();
/// assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false]);
/// ```
pub fn contains(haystack: &ColumnView<'_>, needles: &ColumnView<'_>) -> Result<Column> {
    let output_domain = Arc::clone(haystack.execution_domain());
    ExecutionDomain::require_same_device_for_mixed_inputs(
        haystack.execution_domain(),
        needles.execution_domain(),
        "search::contains",
    )?;
    if !ExecutionDomain::same_provenance(needles.execution_domain(), &output_domain) {
        needles.execution_domain().fence_stream("search::contains")?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    contains_impl(haystack, needles, &output_domain, exec_options.as_ref(), "search::contains")
}
