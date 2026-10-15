//! GPU-accelerated dictionary encoding (factorize / categorize).
//!
//! Dictionary encoding maps a column's values to dense integer indices backed
//! by a sorted, unique key set, similar to `pandas.Categorical` and to
//! `pandas.factorize(sort=True)`.
//!
//! The returned indices are row-aligned with the encoded input. They are not,
//! by themselves, a reusable key-to-code dictionary for some separately
//! deduplicated key set. To construct a stable lookup mapping for later
//! lookup or gather calls, pair the codebook keys with an explicit positional
//! code column `0..keys.len()`.
//!
//! # Example
//!
//! ```no_run
//! use cudf::Runtime;
//!
//! let rt = Runtime::new().unwrap();
//! let series = rt.series(&[429i32, 111, 213, 111, 213, 429, 213]).unwrap();
//! let factorized = series.on(rt.exec()).factorize().unwrap();
//!
//! // A reusable lookup mapping pairs keys with explicit positional codes; the
//! // row-aligned codes are not a key-to-code mapping.
//! assert_eq!(factorized.codebook().keys().len(), 3);
//! assert_eq!(factorized.row_codes().len(), 7);
//! ```

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        types::TypeId,
    },
    runtime::execution_domain::ExecutionDomain,
};

pub(crate) fn encode_impl(
    input: &ColumnView<'_>,
    indices_type: TypeId,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_dictionary_encode_ex(input.raw_handle(), indices_type.to_raw(), &mut out, exec_options)
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("indices_type_id", format!("{:?}", indices_type).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Get the keys column view from a dictionary column.
///
/// The returned view borrows from the dictionary column. It contains the
/// sorted, unique values.
pub(crate) fn keys<'a>(dict_column: &ColumnView<'a>) -> Result<ColumnView<'a>> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        // SAFETY: dict_column handle is valid (NonNull maintained by ColumnView).
        unsafe { cudf_sys::cudf_dictionary_keys(dict_column.raw_handle(), &mut out) },
        "dictionary::keys",
    )?;
    // SAFETY: out was returned by a successful FFI call. The view borrows from
    // dict_column, enforced by the lifetime parameter 'a.
    unsafe { ColumnView::from_raw_with_domain(out, Arc::clone(dict_column.execution_domain())) }
}

/// Get the indices column view from a dictionary column.
///
/// The returned view borrows from the dictionary column. Each element is an
/// integer index into the keys column.
pub(crate) fn indices<'a>(dict_column: &ColumnView<'a>) -> Result<ColumnView<'a>> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        // SAFETY: dict_column handle is valid (NonNull maintained by ColumnView).
        unsafe { cudf_sys::cudf_dictionary_indices(dict_column.raw_handle(), &mut out) },
        "dictionary::indices",
    )?;
    // SAFETY: out was returned by a successful FFI call. The view borrows from
    // dict_column, enforced by the lifetime parameter 'a.
    unsafe { ColumnView::from_raw_with_domain(out, Arc::clone(dict_column.execution_domain())) }
}
