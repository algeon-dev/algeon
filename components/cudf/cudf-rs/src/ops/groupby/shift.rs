//! Per-group row shift.
//!
//! This is the building block for SQL `LAG(value, offset, default)` and
//! `LEAD(value, offset, default)` window functions. Within each group, libcudf
//! shifts the value column by the requested offset, filling indeterminate
//! output positions with the per-column fill scalar.
//!
//! libcudf guarantees that rows within each group appear in the input row
//! order; the order of groups themselves is libcudf-defined and may differ
//! between calls. Callers that need the input row order back should add a
//! row-position column to their inputs and sort the result by that column.

use std::{os::raw::c_void, ptr, sync::Arc};

use smallvec::SmallVec;

use super::{agg::NullPolicy, errors::foreign_groupby_contract_error};
use crate::{
    core::{
        column::ColumnView,
        error::{self, Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        scalar::Scalar,
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

const GROUPBY_SHIFT_VALUE_HANDLE_INLINE_CAPACITY: usize = 8;

/// Output payload of a low-level grouped shift.
pub struct GroupByShiftOutput {
    /// Unique key groups, one row per output values row, with rows clustered
    /// by group and inter-group order libcudf-defined.
    pub keys: Table,
    /// Shifted value columns, in the same column order as the input request.
    /// Rows align with `keys` row-for-row, and within each group the row order
    /// matches the input row order.
    pub values: Table,
}

/// A single column shift request.
pub struct GroupByShiftRequest<'a> {
    /// Values column to shift.
    pub values: &'a ColumnView<'a>,
    /// Row shift offset (positive shifts rows toward larger indices within
    /// the group; negative shifts toward smaller indices).
    pub offset: i32,
    /// Fill value for indeterminate output rows. Must share the column's
    /// data type. A null-valued scalar forces the output column to be
    /// nullable.
    pub fill_value: &'a Scalar,
}

/// Perform a per-group row shift.
///
/// Each request shifts one value column by its `offset`, filling output rows
/// that fall outside the group with the request's `fill_value`. libcudf
/// rejects columns whose data type is neither fixed-width nor string and
/// rejects fill scalars whose data type does not match the corresponding
/// column.
///
/// libcudf returns one output column per request, in the same order. The
/// output keys and values tables have identical row counts; row order within
/// each group matches the input row order, but inter-group order is
/// libcudf-defined.
pub fn groupby_shift(
    keys: &TableView<'_>,
    requests: &[GroupByShiftRequest<'_>],
    null_policy: NullPolicy,
    keys_are_sorted: bool,
) -> Result<GroupByShiftOutput> {
    let execution = compatible_shift_domain(keys, requests)?;
    groupby_shift_impl(
        keys,
        requests,
        null_policy,
        keys_are_sorted,
        execution.output_domain(),
        Some(execution.exec_options()),
        "groupby::groupby_shift",
    )
}

fn compatible_shift_domain(keys: &TableView<'_>, requests: &[GroupByShiftRequest<'_>]) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        keys.execution_domain(),
        requests
            .iter()
            .flat_map(|req| [req.values.execution_domain(), req.fill_value.execution_domain()])
            .collect::<Vec<_>>(),
        &ctx,
        "groupby::groupby_shift",
    )
}

pub(crate) fn groupby_shift_impl(
    keys: &TableView<'_>,
    requests: &[GroupByShiftRequest<'_>],
    null_policy: NullPolicy,
    keys_are_sorted: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<GroupByShiftOutput> {
    validate_request_dtypes(requests, operation)?;

    let num_value_columns_usize = requests.len();
    let num_value_columns = i32::try_from(num_value_columns_usize).map_err(|_| {
        error::limit_validation_error(
            operation,
            "number of grouped shift columns exceeds i32::MAX",
            "shift_requests",
            "request_count",
            "i32::MAX",
            num_value_columns_usize,
            "reduce the number of shift requests to fit within libcudf's i32 range",
        )
    })?;

    let value_handles: SmallVec<[*mut c_void; GROUPBY_SHIFT_VALUE_HANDLE_INLINE_CAPACITY]> =
        requests.iter().map(|r| r.values.raw_handle()).collect();
    let offsets: SmallVec<[i32; GROUPBY_SHIFT_VALUE_HANDLE_INLINE_CAPACITY]> =
        requests.iter().map(|r| r.offset).collect();
    let fill_handles: SmallVec<[*mut c_void; GROUPBY_SHIFT_VALUE_HANDLE_INLINE_CAPACITY]> =
        requests.iter().map(|r| r.fill_value.raw_handle()).collect();

    let mut out_keys = ptr::null_mut();
    let mut out_values = ptr::null_mut();

    let null_policy_ffi = null_policy.to_raw();
    let keys_are_sorted_ffi = i32::from(keys_are_sorted);
    let status = if let Some(options) = exec_options {
        unsafe {
            cudf_sys::cudf_groupby_shift_ex(
                keys.raw_handle(),
                value_handles.as_ptr(),
                offsets.as_ptr(),
                fill_handles.as_ptr(),
                num_value_columns,
                null_policy_ffi,
                keys_are_sorted_ffi,
                &mut out_keys,
                &mut out_values,
                options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_groupby_shift(
                keys.raw_handle(),
                value_handles.as_ptr(),
                offsets.as_ptr(),
                fill_handles.as_ptr(),
                num_value_columns,
                null_policy_ffi,
                keys_are_sorted_ffi,
                &mut out_keys,
                &mut out_values,
            )
        }
    };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("num_value_columns", num_value_columns.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
            ("keys_are_sorted", keys_are_sorted.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    let keys_table = unsafe { Table::from_raw_with_domain(out_keys, Arc::clone(output_domain)) }?;
    let values_out = unsafe { Table::from_raw_with_domain(out_values, Arc::clone(output_domain)) }?;

    let result_value_columns = values_out.num_columns().map_err(|e| e.reanchor_if_needed(operation))?;
    if result_value_columns != num_value_columns_usize {
        return Err(foreign_groupby_contract_error(
            operation,
            "FFI returned a values table with a different column count than requested",
            "result_values",
            "column_count_mismatch",
            "ensure the foreign grouped shift backend returns one output column per request",
        )
        .with_fact(FACT_ACTUAL_COUNT, result_value_columns)
        .with_detail("requested_value_columns", num_value_columns.to_string()));
    }
    let keys_row_count = keys_table.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    let values_row_count = values_out.num_rows().map_err(|e| e.reanchor_if_needed(operation))?;
    if values_row_count != keys_row_count {
        return Err(foreign_groupby_contract_error(
            operation,
            "FFI returned keys and values tables with different row counts",
            "result_row_count",
            "row_count_mismatch",
            "ensure the foreign grouped shift backend returns keys and values with identical row counts",
        )
        .with_detail("keys_row_count", keys_row_count.to_string())
        .with_detail("values_row_count", values_row_count.to_string()));
    }

    Ok(GroupByShiftOutput { keys: keys_table, values: values_out })
}

fn validate_request_dtypes(requests: &[GroupByShiftRequest<'_>], operation: &'static str) -> Result<()> {
    for (request_index, req) in requests.iter().enumerate() {
        let value_dtype = req.values.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
        let fill_dtype = req.fill_value.data_type().map_err(|e| e.reanchor_if_needed(operation))?;
        if value_dtype != fill_dtype {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                format!(
                    "shift fill_value dtype ({:?}) at request {request_index} must match values dtype ({:?})",
                    fill_dtype.id(),
                    value_dtype.id()
                ),
            )
            .with_fact(FACT_LEFT_TYPE_ID, value_dtype.id())
            .with_fact(FACT_RIGHT_TYPE_ID, fill_dtype.id())
            .with_detail("resource", "shift_fill_value")
            .with_detail("reason", "dtype_mismatch")
            .with_detail("request_index", request_index.to_string())
            .with_detail("remedy", "construct fill_value with the same DataType as values"));
        }
    }
    Ok(())
}
