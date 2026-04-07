//! GPU-accelerated search / set-membership operations.

use std::ptr;

use crate::core::{
    column::{Column, ColumnView},
    error::{self, Result},
};

fn contains_impl(
    haystack: &ColumnView<'_>,
    needles: &ColumnView<'_>,
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
    Ok(unsafe { Column::from_raw(out) })
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
/// use cudf::{core::Column, ops::search};
///
/// let haystack = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let needles = Column::from_slice(&[20i32, 40]).unwrap();
/// let result =
///     search::contains(&haystack.view().unwrap(), &needles.view().unwrap())
///         .unwrap();
/// // result: [true, false]
/// ```
pub fn contains(haystack: &ColumnView<'_>, needles: &ColumnView<'_>) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(haystack.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    contains_with_exec_options(haystack, needles, exec_options.as_ref(), "search::contains")
        .map(|column| column.with_execution_domain(output_domain))
}

pub(crate) fn contains_with_exec_options(
    haystack: &ColumnView<'_>,
    needles: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    contains_impl(haystack, needles, exec_options, operation)
}
