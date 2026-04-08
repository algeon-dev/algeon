//! GPU-accelerated row gathering (take / index-select) and conditional copy operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

pub(crate) fn gather_impl(
    input: &TableView<'_>,
    gather_map: &ColumnView<'_>,
    bounds_policy: cudf_sys::cudf_out_of_bounds_policy_t,
    bounds_policy_label: &'static str,
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
    error::check_status_in_with(
        status,
        operation,
        [
            ("bounds_policy", bounds_policy_label.to_owned()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
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
/// use cudf::{
///     core::{Column, Table},
///     ops::copying,
/// };
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// let result =
///     copying::gather(&table.view().unwrap(), &map.view().unwrap()).unwrap();
/// // result row 0 = source row 2 (30), result row 1 = source row 0 (10)
/// ```
pub fn gather(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    ExecutionDomain::require_same_device_for_mixed_inputs(
        input.execution_domain(),
        gather_map.execution_domain(),
        "copying::gather",
    )?;
    if !ExecutionDomain::same_provenance(gather_map.execution_domain(), &output_domain) {
        gather_map.execution_domain().fence_stream("copying::gather")?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
        "nullify",
        exec_options.as_ref(),
        "copying::gather",
    )
    .map(|table| table.with_execution_domain(output_domain))
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
/// use cudf::{
///     core::{Column, Table},
///     ops::copying,
/// };
///
/// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let map = Column::from_slice(&[2i32, 0]).unwrap();
/// // SAFETY: indices 2 and 0 are valid for a 3-row table.
/// let result = unsafe {
///     copying::gather_unchecked(&table.view().unwrap(), &map.view().unwrap())
/// }
/// .unwrap();
/// ```
///
/// Prefer [`gather`] unless profiling shows the bounds check is a bottleneck.
pub unsafe fn gather_unchecked(input: &TableView<'_>, gather_map: &ColumnView<'_>) -> Result<Table> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    ExecutionDomain::require_same_device_for_mixed_inputs(
        input.execution_domain(),
        gather_map.execution_domain(),
        "copying::gather_unchecked",
    )?;
    if !ExecutionDomain::same_provenance(gather_map.execution_domain(), &output_domain) {
        gather_map.execution_domain().fence_stream("copying::gather_unchecked")?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    gather_impl(
        input,
        gather_map,
        cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
        "dont_check",
        exec_options.as_ref(),
        "copying::gather_unchecked",
    )
    .map(|table| table.with_execution_domain(output_domain))
}

// ---------------------------------------------------------------------------
// copy_if_else (column-column)
// ---------------------------------------------------------------------------

fn copy_if_else_impl(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
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
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
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
/// use cudf::{
///     core::Column,
///     ops::copying,
/// };
///
/// let lhs = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let rhs = Column::from_slice(&[1i32, 2, 3]).unwrap();
/// let mask = Column::from_slice(&[true, false, true]).unwrap();
/// let result = copying::copy_if_else(
///     &lhs.view().unwrap(),
///     &rhs.view().unwrap(),
///     &mask.view().unwrap(),
/// )
/// .unwrap();
/// // result: [10, 2, 30]
/// ```
pub fn copy_if_else(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    boolean_mask: &ColumnView<'_>,
) -> Result<Column> {
    let output_domain = Arc::clone(lhs.execution_domain());
    ExecutionDomain::require_same_device_and_fence_secondary(
        &output_domain,
        rhs.execution_domain(),
        "copying::copy_if_else",
    )?;
    ExecutionDomain::require_same_device_and_fence_secondary(
        &output_domain,
        boolean_mask.execution_domain(),
        "copying::copy_if_else",
    )?;
    let exec_options = output_domain.explicit_exec_options_raw();
    copy_if_else_impl(lhs, rhs, boolean_mask, exec_options.as_ref(), "copying::copy_if_else")
        .map(|col| col.with_execution_domain(output_domain))
}
