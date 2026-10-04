//! GPU-accelerated row gathering (take / index-select), scatter, and conditional copy operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

pub(crate) fn gather_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_gather_ex(input.raw_handle(), gather_map.raw_handle(), bounds_policy, &mut out, exec_options)
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("bounds_policy", bounds_policy_label.to_owned()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn gather_device_allocated_size_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
    reservation_contract: &'static str,
) -> Result<usize> {
    let mut out = 0usize;
    let status = unsafe {
        cudf_sys::cudf_gather_device_allocated_size_ex(
            input.raw_handle(),
            gather_map.raw_handle(),
            bounds_policy,
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("bounds_policy", bounds_policy_label.to_owned()),
            ("execution_authority", "retained_domain".to_string()),
            ("reservation_contract", reservation_contract.to_owned()),
        ],
    )?;
    Ok(out)
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
/// let result = df.on(rt.exec()).gather(&map).unwrap();
/// assert_eq!(
///     result.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
///     vec![30, 10]
/// );
/// ```
pub(crate) fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let ctx = Exec::RETAINED;
    let execution =
        ResolvedExecution::binary(input.execution_domain(), gather_map.execution_domain(), ctx, "copying::gather")?;
    gather_impl(
        input,
        gather_map,
        cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        execution.output_domain(),
        execution.exec_options(),
        "copying::gather",
    )
}

pub(crate) fn copy_if_else_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_copy_if_else_vv_ex(
            lhs.raw_handle(),
            rhs.raw_handle(),
            boolean_mask.raw_handle(),
            &mut out,
            exec_options,
        )
    };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Conditionally select elements from two columns using an explicit execution override.
pub(crate) fn copy_if_else_in(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
    ctx: Exec<'_>,
) -> Result<Column> {
    let execution = ResolvedExecution::many(
        lhs.execution_domain(),
        [rhs.execution_domain(), boolean_mask.execution_domain()],
        ctx,
        "copying::copy_if_else_in",
    )?;
    copy_if_else_impl(
        lhs,
        rhs,
        boolean_mask,
        execution.output_domain(),
        execution.exec_options(),
        "copying::copy_if_else_in",
    )
}
