//! GPU-accelerated stream compaction: drop_nulls, distinct (deduplicate).

use std::ptr;

use crate::core::{
    error::{self, Error, ErrorKind, Result},
    table::{Table, TableView},
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

pub use crate::core::types::NullEquality;

/// Whether NaN elements should be considered equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NanEquality {
    /// All NaN values compare equal.
    AllEqual = 0,
    /// NaN values compare unequal (IEEE 754 behavior).
    Unequal = 1,
}

/// Drop rows from a table where key columns contain nulls.
///
/// Rows are kept only if at least `keep_threshold` of the specified `keys`
/// columns are non-null. For the common "drop any null" behavior, set
/// `keep_threshold` equal to the number of key columns.
///
/// # Example
/// ```no_run
/// use cudf::{Column, Table, ops::stream_compaction};
///
/// let c0 = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
///     .unwrap();
/// let table = Table::from_columns(vec![c0]).unwrap();
/// let result =
///     stream_compaction::drop_nulls(&table.view().unwrap(), &[0], 1).unwrap();
/// // Rows with null in column 0 are removed.
/// ```
pub fn drop_nulls(input: &TableView<'_>, keys: &[i32], keep_threshold: i32) -> Result<Table> {
    let num_keys = i32::try_from(keys.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "num_keys exceeds i32::MAX")
            .with_operation("stream_compaction::drop_nulls")
            .with_context("num_keys", keys.len().to_string())
    })?;

    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: input handle is valid (NonNull). keys slice is valid for num_keys elements.
        unsafe { cudf_sys::cudf_drop_nulls(input.raw_handle(), keys.as_ptr(), num_keys, keep_threshold, &mut out) },
        "stream_compaction::drop_nulls",
        [("num_keys", num_keys.to_string()), ("keep_threshold", keep_threshold.to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw(out) })
}

/// Create a new table with distinct rows based on key columns.
///
///
/// This is the GPU equivalent of `drop_duplicates(subset=[...])`.
/// The order of rows in the output is **not** specified.
///
/// # Example
/// ```no_run
/// use cudf::{
///     Column, Table,
///     ops::stream_compaction::{
///         self, DuplicateKeepOption, NanEquality, NullEquality,
///     },
/// };
///
/// let col = Column::from_slice(&[1i32, 2, 1, 3]).unwrap();
/// let table = Table::from_columns(vec![col]).unwrap();
/// let result = stream_compaction::distinct(
///     &table.view().unwrap(),
///     &[0],
///     DuplicateKeepOption::KeepAny,
///     NullEquality::Equal,
///     NanEquality::AllEqual,
/// )
/// .unwrap();
/// // result has 3 rows: {1, 2, 3} in unspecified order.
/// ```
pub fn distinct(
    input: &TableView<'_>,
    keys: &[i32],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    nan_equality: NanEquality,
) -> Result<Table> {
    let num_keys = i32::try_from(keys.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "num_keys exceeds i32::MAX")
            .with_operation("stream_compaction::distinct")
            .with_context("num_keys", keys.len().to_string())
    })?;

    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: input handle is valid (NonNull). keys slice is valid for num_keys elements.
        unsafe {
            cudf_sys::cudf_distinct(
                input.raw_handle(),
                keys.as_ptr(),
                num_keys,
                keep as i32,
                null_equality as i32,
                nan_equality as i32,
                &mut out,
            )
        },
        "stream_compaction::distinct",
        [
            ("num_keys", num_keys.to_string()),
            ("keep", format!("{:?}", keep).to_lowercase()),
            ("null_equality", format!("{:?}", null_equality).to_lowercase()),
            ("nan_equality", format!("{:?}", nan_equality).to_lowercase()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Table::from_raw(out) })
}
