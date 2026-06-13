//! GPU-accelerated stream compaction: drop_nulls, distinct (deduplicate).

use std::{ptr, sync::Arc};

pub use crate::core::types::NullEquality;
use crate::{
    core::{
        error::{self, Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

/// Policy for which duplicate rows to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DuplicateKeepOption {
    /// Keep an unspecified occurrence of each duplicate group.
    KeepAny = 0,
    /// Keep the first occurrence.
    KeepFirst = 1,
    /// Keep the last occurrence.
    KeepLast = 2,
    /// Remove all duplicates (keep none).
    KeepNone = 3,
}

impl DuplicateKeepOption {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_duplicate_keep_option_t {
        match self {
            DuplicateKeepOption::KeepAny => cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            DuplicateKeepOption::KeepFirst => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST
            },
            DuplicateKeepOption::KeepLast => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST
            },
            DuplicateKeepOption::KeepNone => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE
            },
        }
    }
}

/// Whether NaN elements should be considered equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NanEquality {
    /// All NaN values compare equal.
    AllEqual = 0,
    /// NaN values compare unequal (IEEE 754 behavior).
    Unequal = 1,
}

impl NanEquality {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_stream_compaction_nan_equality_t {
        match self {
            NanEquality::AllEqual => {
                cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL
            },
            NanEquality::Unequal => {
                cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL
            },
        }
    }
}

fn checked_key_indices_i32(input: &TableView<'_>, keys: &[usize], operation: &'static str) -> Result<Vec<i32>> {
    let num_columns = input.num_columns()?;

    keys.iter()
        .map(|&key| {
            if key >= num_columns {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("key column index {} out of bounds for {} columns", key, num_columns),
                )
                .with_operation(operation)
                .with_context("key_index", key.to_string())
                .with_context("num_columns", num_columns.to_string()))
            } else {
                i32::try_from(key).map_err(|_| {
                    Error::new(ErrorKind::InvalidOperation, "key column index exceeds i32::MAX")
                        .with_operation(operation)
                        .with_context("key_index", key.to_string())
                })
            }
        })
        .collect()
}

pub(crate) fn drop_nulls_impl(
    input: &TableView<'_>,
    keys: &[usize],
    keep_threshold: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let keys_i32 = checked_key_indices_i32(input, keys, operation)?;
    let num_keys = i32::try_from(keys_i32.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "num_keys exceeds i32::MAX")
            .with_operation(operation)
            .with_context("num_keys", keys_i32.len().to_string())
    })?;

    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_drop_nulls_ex(
                input.raw_handle(),
                keys_i32.as_ptr(),
                num_keys,
                keep_threshold,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_drop_nulls(input.raw_handle(), keys_i32.as_ptr(), num_keys, keep_threshold, &mut out)
        },
    };
    error::check_status_in_with(
        status,
        operation,
        [
            ("num_keys", num_keys.to_string()),
            ("keep_threshold", keep_threshold.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn distinct_impl(
    input: &TableView<'_>,
    keys: &[usize],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    nan_equality: NanEquality,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let keys_i32 = checked_key_indices_i32(input, keys, operation)?;
    let num_keys = i32::try_from(keys_i32.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "num_keys exceeds i32::MAX")
            .with_operation(operation)
            .with_context("num_keys", keys_i32.len().to_string())
    })?;

    let mut out = ptr::null_mut();
    let keep_ffi = keep.to_raw();
    let null_eq_ffi = null_equality.to_stream_compaction_raw();
    let nan_eq_ffi = nan_equality.to_raw();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_distinct_ex(
                input.raw_handle(),
                keys_i32.as_ptr(),
                num_keys,
                keep_ffi,
                null_eq_ffi,
                nan_eq_ffi,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_distinct(
                input.raw_handle(),
                keys_i32.as_ptr(),
                num_keys,
                keep_ffi,
                null_eq_ffi,
                nan_eq_ffi,
                &mut out,
            )
        },
    };
    error::check_status_in_with(
        status,
        operation,
        [
            ("num_keys", num_keys.to_string()),
            ("keep", format!("{:?}", keep).to_lowercase()),
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("nan_equality", format!("{:?}", nan_equality).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Drop rows from a table where key columns contain nulls.
///
/// Rows are kept only if at least `keep_threshold` of the specified `keys`
/// columns are non-null. For the common "drop any null" behavior, set
/// `keep_threshold` equal to the number of key columns.
///
/// # Example
/// ```no_run
/// use cudf::{Runtime, frame::dataframe::DropNaOptions};
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap(),
/// }
/// .unwrap();
/// let result = df.dropna(DropNaOptions::new()).unwrap();
/// assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3]);
/// ```
pub fn drop_nulls(input: &TableView<'_>, keys: &[usize], keep_threshold: i32) -> Result<Table> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    drop_nulls_impl(input, keys, keep_threshold, &output_domain, exec_options.as_ref(), "stream_compaction::drop_nulls")
}

/// Create a new table with distinct rows based on key columns.
///
/// This is the GPU equivalent of `drop_duplicates(subset=[...])`.
/// The order of rows in the output is **not** specified.
///
/// # Example
/// ```no_run
/// use cudf::{Runtime, frame::dataframe::DropDuplicatesOptions};
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.series(&[1i32, 2, 1, 3]).unwrap(),
/// }
/// .unwrap();
/// let result = df.drop_duplicates(DropDuplicatesOptions::new()).unwrap();
/// assert_eq!(result.num_rows(), 3);
/// ```
pub fn distinct(
    input: &TableView<'_>,
    keys: &[usize],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    nan_equality: NanEquality,
) -> Result<Table> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    distinct_impl(
        input,
        keys,
        keep,
        null_equality,
        nan_equality,
        &output_domain,
        exec_options.as_ref(),
        "stream_compaction::distinct",
    )
}
