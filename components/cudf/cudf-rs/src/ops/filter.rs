//! GPU-accelerated filtering operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::Runtime;
//!
//! let rt = Runtime::new().unwrap();
//! let df = cudf::dataframe! {
//!     "value" => rt.series(&[10i32, 20, 30, 40]).unwrap(),
//! }
//! .unwrap();
//! let mask = rt.series(&[true, false, true, false]).unwrap();
//!
//! let result = df.filter(&mask).unwrap();
//! assert_eq!(
//!     result.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
//!     vec![10, 30]
//! );
//! ```

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::ColumnView,
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result,
        },
        scalar::Scalar,
        table::{Table, TableView},
        types::TypeId,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

pub(crate) fn validate_boolean_mask(
    mask: &ColumnView<'_>,
    expected_rows: usize,
    expected_rows_label: &'static str,
    operation: &'static str,
) -> Result<()> {
    let mask_dtype = mask.data_type()?;
    if mask_dtype.id() != TypeId::Bool8 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            format!("filter mask must be Bool8, got {:?}", mask_dtype.id()),
        )
        .with_fact(FACT_LEFT_TYPE_ID, mask_dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Bool8));
    }

    let mask_len = mask.len()?;
    if mask_len != expected_rows {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            format!("filter mask length ({mask_len}) does not match {expected_rows_label} ({expected_rows})"),
        )
        .with_fact(FACT_ACTUAL_ROWS, mask_len)
        .with_fact(FACT_EXPECTED_ROWS, expected_rows));
    }

    Ok(())
}

pub(crate) fn apply_boolean_mask_impl(
    input: &TableView<'_>,
    mask: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_apply_boolean_mask_ex(input.raw_handle(), mask.raw_handle(), &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_apply_boolean_mask(input.raw_handle(), mask.raw_handle(), &mut out) },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Filter rows of a table using a boolean mask column.
///
/// Rows where the mask is `true` are kept; rows where the mask is `false`
/// or null are dropped. The mask must be a `BOOL8` column with the same
/// number of rows as the input table.
pub fn apply_boolean_mask(input: &TableView<'_>, mask: &ColumnView<'_>) -> Result<Table> {
    let input_rows = input.num_rows()?;
    validate_boolean_mask(mask, input_rows, "table row count", "filter::apply_boolean_mask")?;

    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        input.execution_domain(),
        mask.execution_domain(),
        &ctx,
        "filter::apply_boolean_mask",
    )?;
    apply_boolean_mask_impl(
        input,
        mask,
        execution.output_domain(),
        Some(execution.exec_options()),
        "filter::apply_boolean_mask",
    )
}

pub(crate) fn apply_row_predicate_impl(
    input: &TableView<'_>,
    nodes: &[cudf_sys::cudf_row_predicate_node_t],
    root_index: i32,
    retained_scalars: &[Scalar],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let _retained_scalars = retained_scalars;
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_filter_rows_by_predicate_ex(
                input.raw_handle(),
                nodes.as_ptr(),
                error::usize_to_i32(nodes.len(), operation, "node_count")?,
                root_index,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_filter_rows_by_predicate(
                input.raw_handle(),
                nodes.as_ptr(),
                error::usize_to_i32(nodes.len(), operation, "node_count")?,
                root_index,
                &mut out,
            )
        },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("node_count", nodes.len().to_string()),
            ("root_index", root_index.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
