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

use std::ptr;

use crate::{
    core::{
        column::ColumnView,
        error::{self, Result},
        scalar::Scalar,
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

pub(crate) fn apply_boolean_mask_impl(
    input: &TableView<'_>,
    mask: &ColumnView<'_>,
    output_domain: &std::sync::Arc<ExecutionDomain>,
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
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Table::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) }
}

/// Filter rows of a table using a boolean mask column.
///
/// Rows where the mask is `true` are kept; rows where the mask is `false`
/// or null are dropped. The mask must be a `BOOL8` column with the same
/// number of rows as the input table.
pub fn apply_boolean_mask(input: &TableView<'_>, mask: &ColumnView<'_>) -> Result<Table> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    ExecutionDomain::require_same_device_for_mixed_inputs(
        input.execution_domain(),
        mask.execution_domain(),
        "filter::apply_boolean_mask",
    )?;
    if !ExecutionDomain::same_provenance(mask.execution_domain(), &output_domain) {
        ExecutionDomain::insert_stream_dependency(
            mask.execution_domain(),
            &output_domain,
            "filter::apply_boolean_mask",
        )?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    apply_boolean_mask_impl(input, mask, &output_domain, exec_options.as_ref(), "filter::apply_boolean_mask")
}

pub(crate) fn apply_row_predicate_impl(
    input: &TableView<'_>,
    nodes: &[cudf_sys::cudf_row_predicate_node_t],
    root_index: i32,
    retained_scalars: &[Scalar],
    output_domain: &std::sync::Arc<ExecutionDomain>,
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
    error::check_status_in_with(
        status,
        operation,
        [
            ("node_count", nodes.len().to_string()),
            ("root_index", root_index.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, std::sync::Arc::clone(output_domain)) }
}
