//! GPU-accelerated dictionary encoding (factorize / categorize).
//!
//! Dictionary encoding maps a column's values to dense integer indices backed
//! by a sorted, unique key set — equivalent to `pandas.factorize()` or
//! `pandas.Categorical`.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{Column, TypeId, ops::dictionary};
//!
//! let col =
//!     Column::from_slice(&[429i32, 111, 213, 111, 213, 429, 213]).unwrap();
//! let view = col.view().unwrap();
//!
//! let dict = dictionary::encode(&view, TypeId::Int32).unwrap();
//! let dict_view = dict.view().unwrap();
//!
//! // Access keys and indices without copying to host.
//! let keys = dictionary::keys(&dict_view).unwrap();
//! let indices = dictionary::indices(&dict_view).unwrap();
//! ```

use std::ptr;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        types::TypeId,
    },
    runtime::execution_domain::ExecutionDomain,
};

fn encode_impl(
    input: &ColumnView<'_>,
    indices_type: TypeId,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
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
    Ok(unsafe { Column::from_raw(out) })
}

fn decode_impl(
    dict_column: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_decode_ex(dict_column.raw_handle(), &mut out, options)
            },
            None => unsafe { cudf_sys::cudf_dictionary_decode(dict_column.raw_handle(), &mut out) },
        },
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

fn set_keys_impl(
    dict_column: &ColumnView<'_>,
    keys: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
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
    Ok(unsafe { Column::from_raw(out) })
}

fn primary_column_domain(columns: &[&ColumnView<'_>]) -> Result<std::sync::Arc<ExecutionDomain>> {
    let Some(first) = columns.first() else {
        return Ok(ExecutionDomain::ambient_default());
    };
    Ok(std::sync::Arc::clone(first.execution_domain()))
}

fn match_dictionaries_impl(
    dict_columns: &[&ColumnView<'_>],
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Vec<Column>> {
    let n = dict_columns.len();
    let handles: Vec<*mut std::ffi::c_void> = dict_columns.iter().map(|c| c.raw_handle()).collect();
    let mut out_columns: *mut *mut std::ffi::c_void = ptr::null_mut();
    let mut out_count: i32 = 0;

    error::check_status_in_with(
        match exec_options {
            Some(options) => unsafe {
                cudf_sys::cudf_dictionary_match_dictionaries_ex(
                    handles.as_ptr(),
                    n as i32,
                    &mut out_columns,
                    &mut out_count,
                    options,
                )
            },
            None => unsafe {
                cudf_sys::cudf_dictionary_match_dictionaries(
                    handles.as_ptr(),
                    n as i32,
                    &mut out_columns,
                    &mut out_count,
                )
            },
        },
        operation,
        [("num_columns", n.to_string()), ("execution_context_provided", exec_options.is_some().to_string())],
    )?;

    if out_count != n as i32 {
        if !out_columns.is_null() {
            let returned = unsafe { std::slice::from_raw_parts(out_columns, out_count.max(0) as usize) };
            for &handle in returned {
                if !handle.is_null() {
                    unsafe { cudf_sys::cudf_column_destroy(handle) };
                }
            }
            unsafe { cudf_sys::cudf_free_column_array(out_columns) };
        }
        return Err(error::Error::new(
            error::ErrorKind::InvalidOperation,
            "ffi returned unexpected dictionary match result count",
        )
        .with_operation(operation)
        .with_context("expected_count", n.to_string())
        .with_context("actual_count", out_count.to_string()));
    }
    if out_count > 0 && out_columns.is_null() {
        return Err(error::Error::new(
            error::ErrorKind::InvalidOperation,
            "ffi returned null dictionary match result array with positive count",
        )
        .with_operation(operation)
        .with_context("num_columns", n.to_string()));
    }

    let returned = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
    let columns = returned.iter().copied().map(|handle| unsafe { Column::from_raw(handle) }).collect();
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
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    encode_impl(input, indices_type, exec_options.as_ref(), "dictionary::encode")
        .map(|column| column.with_execution_domain(output_domain))
}

/// Decode a dictionary column back to a regular column.
///
/// The input must be a `DICTIONARY32` column. The result has the same type as
/// the dictionary keys.
pub fn decode(dict_column: &ColumnView<'_>) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(dict_column.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    decode_impl(dict_column, exec_options.as_ref(), "dictionary::decode")
        .map(|column| column.with_execution_domain(output_domain))
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
    Ok(unsafe { ColumnView::from_raw_with_domain(out, std::sync::Arc::clone(dict_column.execution_domain())) })
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
    Ok(unsafe { ColumnView::from_raw_with_domain(out, std::sync::Arc::clone(dict_column.execution_domain())) })
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
    let output_domain = std::sync::Arc::clone(dict_column.execution_domain());
    if !ExecutionDomain::same_provenance(keys.execution_domain(), &output_domain) {
        keys.execution_domain().fence_stream("dictionary::set_keys")?;
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    set_keys_impl(dict_column, keys, exec_options.as_ref(), "dictionary::set_keys")
        .map(|column| column.with_execution_domain(output_domain))
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
/// # Example
///
/// ```no_run
/// use cudf::{Column, TypeId, ops::dictionary};
///
/// let src = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let dst = Column::from_slice(&[20i32, 30, 40]).unwrap();
///
/// let src_dict =
///     dictionary::encode(&src.view().unwrap(), TypeId::Int32).unwrap();
/// let dst_dict =
///     dictionary::encode(&dst.view().unwrap(), TypeId::Int32).unwrap();
///
/// let matched = dictionary::match_dictionaries(&[
///     &src_dict.view().unwrap(),
///     &dst_dict.view().unwrap(),
/// ])
/// .unwrap();
/// // matched[0] and matched[1] now share keys [10, 20, 30, 40].
/// ```
pub fn match_dictionaries(dict_columns: &[&ColumnView<'_>]) -> Result<Vec<Column>> {
    let n = dict_columns.len();
    if n == 0 {
        return Err(error::Error::new(
            error::ErrorKind::InvalidOperation,
            "match_dictionaries requires at least one column",
        )
        .with_operation("dictionary::match_dictionaries"));
    }

    let output_domain = primary_column_domain(dict_columns)?;
    for dict_column in &dict_columns[1..] {
        if !ExecutionDomain::same_provenance(dict_column.execution_domain(), &output_domain) {
            dict_column.execution_domain().fence_stream("dictionary::match_dictionaries")?;
        }
    }
    let exec_options = output_domain.explicit_exec_options_raw();
    match_dictionaries_impl(dict_columns, exec_options.as_ref(), "dictionary::match_dictionaries").map(|columns| {
        columns.into_iter().map(|column| column.with_execution_domain(std::sync::Arc::clone(&output_domain))).collect()
    })
}
