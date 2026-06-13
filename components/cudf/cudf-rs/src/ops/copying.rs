//! GPU-accelerated row gathering (take / index-select) and conditional copy operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

pub(crate) fn gather_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_gather_ex(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_gather(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out) },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("bounds_policy", bounds_policy_label.to_owned()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Gather (select) rows from a table by index.
///
/// Produces a new table whose row `i` is `source_table[gather_map[i]]`.
/// The `gather_map` must be a non-nullable column of `INT32` indices.
///
/// A negative value `i` in the gather map is interpreted as `i + n`,
/// where `n` is the number of rows in the source table.
///
/// Out-of-bounds indices produce null rows in the output.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.series(&[10i32, 20, 30]).unwrap(),
/// }
/// .unwrap();
/// let map = rt.series(&[2i32, 0]).unwrap();
/// let result = df.gather(&map).unwrap();
/// assert_eq!(
///     result.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![30, 10]
/// );
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(input.execution_domain(), gather_map.execution_domain(), &ctx, "copying::gather")?;
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        execution.output_domain(),
        Some(execution.exec_options()),
        "copying::gather",
    )
}

/// Gather (select) rows from a table by index, without bounds checking.
///
/// This is identical to [`gather`] but skips bounds validation for
/// better performance. Out-of-bounds indices cause **undefined behavior**
/// (invalid GPU memory access).
///
/// # Safety
///
/// Caller must ensure that every value in `gather_map` is a valid index
/// into `input` (i.e., in the range `[-n, n)` where `n` is the number
/// of rows in `input`). Violating this causes undefined behavior.
///
/// # Example
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.series(&[10i32, 20, 30]).unwrap(),
/// }
/// .unwrap();
/// let map = rt.series(&[2i32, 0]).unwrap();
/// let result = df.gather(&map).unwrap();
/// assert_eq!(
///     result.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![30, 10]
/// );
/// ```
///
/// Prefer [`gather`] unless profiling shows the bounds check is a bottleneck.
pub unsafe fn gather_unchecked(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        input.execution_domain(),
        gather_map.execution_domain(),
        &ctx,
        "copying::gather_unchecked",
    )?;
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
        "dont_check",
        execution.output_domain(),
        Some(execution.exec_options()),
        "copying::gather_unchecked",
    )
}

// ---------------------------------------------------------------------------
// copy_if_else (column-column)
// ---------------------------------------------------------------------------

pub(crate) fn copy_if_else_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_copy_if_else_vv_ex(
                lhs.raw_handle(),
                rhs.raw_handle(),
                boolean_mask.raw_handle(),
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_copy_if_else_vv(lhs.raw_handle(), rhs.raw_handle(), boolean_mask.raw_handle(), &mut out)
        },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Conditionally select elements from two columns based on a boolean mask.
///
/// Produces a new column where:
/// - `output[i] = lhs[i]` when `boolean_mask[i]` is valid and `true`
/// - `output[i] = rhs[i]` when `boolean_mask[i]` is null or `false`
///
/// `lhs` and `rhs` must have the same type and length as `boolean_mask`.
/// `boolean_mask` must be of type `BOOL8`.
///
/// # Example
/// ```no_run
/// use cudf::ops::expr::{col, when};
///
/// let rt = cudf::Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "lhs" => rt.series(&[10i32, 20, 30]).unwrap(),
///     "rhs" => rt.series(&[1i32, 2, 3]).unwrap(),
///     "keep_lhs" => rt.series(&[true, false, true]).unwrap(),
/// }
/// .unwrap();
/// let out = df
///     .with_columns([when(col("keep_lhs"))
///         .then(col("lhs"))
///         .otherwise(col("rhs"))
///         .alias("out")])
///     .unwrap();
/// assert_eq!(
///     out.series("out").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![10, 2, 30]
/// );
/// ```
pub fn copy_if_else(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>, boolean_mask: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::many(
        lhs.execution_domain(),
        [rhs.execution_domain(), boolean_mask.execution_domain()],
        &ctx,
        "copying::copy_if_else",
    )?;
    copy_if_else_impl(
        lhs,
        rhs,
        boolean_mask,
        execution.output_domain(),
        Some(execution.exec_options()),
        "copying::copy_if_else",
    )
}
