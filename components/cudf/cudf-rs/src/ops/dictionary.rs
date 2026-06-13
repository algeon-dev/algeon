//! GPU-accelerated dictionary encoding (factorize / categorize).
//!
//! Dictionary encoding maps a column's values to dense integer indices backed
//! by a sorted, unique key set — equivalent to `pandas.factorize()` or
//! `pandas.Categorical`.
//!
//! The returned indices are row-aligned with the encoded input. They are not,
//! by themselves, a reusable key-to-code dictionary for some separately
//! deduplicated key set. If you need a stable lookup mapping for later
//! lookup/gather calls, pair [`keys`] with an explicit positional code column
//! `0..keys.len()`.
//!
//! # Example
//!
//! ```no_run
//! use cudf::Runtime;
//!
//! let rt = Runtime::new().unwrap();
//! let series = rt.series(&[429i32, 111, 213, 111, 213, 429, 213]).unwrap();
//! let factorized = series.factorize().unwrap();
//!
//! // For a reusable lookup mapping, pair keys with explicit positional codes
//! // instead of reusing the row-aligned codes.
//! assert_eq!(factorized.codebook().keys().len(), 3);
//! assert_eq!(factorized.row_codes().len(), 7);
//! ```

mod errors;
mod result_shape;
pub(crate) mod testing_hooks;

use std::{ptr, sync::Arc};

use self::{
    errors::{
        dictionary_column_count_limit_error, require_dictionary_column, unsupported_dictionary_validation_error,
        validate_set_keys_inputs,
    },
    result_shape::validate_match_dictionaries_result_shape,
};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        types::TypeId,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

pub(crate) fn encode_impl(
    input: &ColumnView<'_>,
    indices_type: TypeId,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_encode_ex(input.raw_handle(), indices_type.to_raw(), &mut out, options)
            },
            None => unsafe { cudf_sys::cudf_dictionary_encode(input.raw_handle(), indices_type.to_raw(), &mut out) },
        },
        operation,
        [
            ("indices_type_id", format!("{:?}", indices_type).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn decode_impl(
    dict_column: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_decode_ex(dict_column.raw_handle(), &mut out, options)
            },
            None => unsafe { cudf_sys::cudf_dictionary_decode(dict_column.raw_handle(), &mut out) },
        },
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn set_keys_impl(
    dict_column: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_set_keys_ex(dict_column.raw_handle(), keys.raw_handle(), &mut out, options)
            },
            None => unsafe {
                cudf_sys::cudf_dictionary_set_keys(dict_column.raw_handle(), keys.raw_handle(), &mut out)
            },
        },
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

fn match_dictionaries_impl(
    dict_columns: &[&ColumnView<'_>],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Vec<Column>> {
    let n = dict_columns.len();
    let n_i32 = i32::try_from(n).map_err(|_| dictionary_column_count_limit_error(operation, n))?;
    let handles: Vec<*mut std::ffi::c_void> = dict_columns.iter().map(|c| c.raw_handle()).collect();
    let mut out_columns: *mut *mut std::ffi::c_void = ptr::null_mut();
    let mut out_count: i32 = 0;

    error::check_status_in_with_details(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_match_dictionaries_ex(
                    handles.as_ptr(),
                    n_i32,
                    &mut out_columns,
                    &mut out_count,
                    options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_dictionary_match_dictionaries(handles.as_ptr(), n_i32, &mut out_columns, &mut out_count)
            },
        },
        operation,
        [("num_columns", n.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;

    validate_match_dictionaries_result_shape(out_columns, out_count, n_i32, operation)?;

    let returned = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
    let columns = returned
        .iter()
        .copied()
        .map(|handle| unsafe { Column::from_raw_with_domain(handle, Arc::clone(output_domain)) })
        .collect::<Result<Vec<_>>>()?;
    unsafe { cudf_sys::cudf_free_column_array(out_columns) };
    Ok(columns)
}

/// Encode a column into a dictionary column.
///
/// The result is a `DICTIONARY32` column whose children contain:
/// - **indices**: integer column (of type `indices_type`) mapping each input
///   row to its position in the keys.
/// - **keys**: sorted, unique values from the input.
///
/// # Example
///
/// Input:  `[429, 111, 213, 111, 213, 429, 213]`
/// Keys:   `[111, 213, 429]`
/// Indices: `[2, 0, 1, 0, 1, 2, 1]`
pub fn encode(input: &ColumnView<'_>, indices_type: TypeId) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "dictionary::encode")?;
    encode_impl(input, indices_type, execution.output_domain(), Some(execution.exec_options()), "dictionary::encode")
}

/// Decode a dictionary column back to a regular column.
///
/// The input must be a `DICTIONARY32` column. The result has the same type as
/// the dictionary keys.
pub fn decode(dict_column: &ColumnView<'_>) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(dict_column.execution_domain(), &ctx, "dictionary::decode")?;
    decode_impl(dict_column, execution.output_domain(), Some(execution.exec_options()), "dictionary::decode")
}

/// Get the keys column view from a dictionary column.
///
/// The returned view borrows from the dictionary column. It contains the
/// sorted, unique values.
pub fn keys<'a>(dict_column: &ColumnView<'a>) -> Result<ColumnView<'a>> {
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
pub fn indices<'a>(dict_column: &ColumnView<'a>) -> Result<ColumnView<'a>> {
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

/// Replace the keys of a dictionary column.
///
/// Any new elements found in `keys` are added to the output dictionary.
/// Any existing keys not in `keys` are removed (rows that referenced
/// removed keys become null). The indices are updated to reflect the new
/// key positions.
///
/// This is useful for aligning two dictionary columns to the same key set
/// so their indices are directly comparable.
pub fn set_keys(dict_column: &ColumnView<'_>, keys: &ColumnView<'_>) -> Result<Column> {
    validate_set_keys_inputs(dict_column, keys, "dictionary::set_keys")?;

    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        dict_column.execution_domain(),
        keys.execution_domain(),
        &ctx,
        "dictionary::set_keys",
    )?;
    set_keys_impl(dict_column, keys, execution.output_domain(), Some(execution.exec_options()), "dictionary::set_keys")
}

/// Align multiple dictionary columns to a shared, deduplicated key set.
///
/// The keys from all input dictionary columns are concatenated,
/// deduplicated, and used to re-key each column. After this call every
/// output column shares the same key set, so their integer indices are
/// directly comparable.
///
/// This is the correct way to prepare two or more dictionary-encoded
/// columns for use as graph vertex IDs — it guarantees the same string
/// always maps to the same integer across all columns.
///
/// No high-level `Series` wrapper exists yet for shared dictionary-key
/// alignment across multiple independently encoded columns. Most users should
/// stay on the higher-level factorization surface unless they explicitly need
/// shared low-level dictionary keys across multiple columns.
pub fn match_dictionaries(dict_columns: &[&ColumnView<'_>]) -> Result<Vec<Column>> {
    let n = dict_columns.len();
    if n == 0 {
        return Err(unsupported_dictionary_validation_error(
            "dictionary::match_dictionaries",
            "match_dictionaries requires at least one column",
            "empty_input",
            "provide at least one input dictionary column",
        ));
    }

    for (column_index, dict_column) in dict_columns.iter().enumerate() {
        require_dictionary_column(dict_column, "dictionary::match_dictionaries", "dict_column", Some(column_index))?;
    }

    let ctx = ExecutionOverrides::new();
    let (primary, secondaries) = dict_columns.split_first().expect("empty input was rejected above");
    let execution = ResolvedExecution::many(
        primary.execution_domain(),
        secondaries.iter().map(|column| column.execution_domain()),
        &ctx,
        "dictionary::match_dictionaries",
    )?;
    match_dictionaries_impl(
        dict_columns,
        execution.output_domain(),
        Some(execution.exec_options()),
        "dictionary::match_dictionaries",
    )
}
