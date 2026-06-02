//! GPU-accelerated join operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::merge`](crate::DataFrame::merge) with
//! [`MergeOptions`](crate::MergeOptions).
//!
//! # Example
//!
//! ```no_run
//! use cudf::{column::Column, join, table::Table};
//!
//! let left_key = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let left_val = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let right_key = Column::from_slice(&[2i32, 3, 4]).unwrap();
//! let right_val = Column::from_slice(&[200i32, 300, 400]).unwrap();
//!
//! let lk = Table::from_columns(vec![left_key]).unwrap();
//! let lv = Table::from_columns(vec![left_val]).unwrap();
//! let rk = Table::from_columns(vec![right_key]).unwrap();
//! let rv = Table::from_columns(vec![right_val]).unwrap();
//!
//! let result = join::inner_join(
//!     &lk.view().unwrap(),
//!     &rk.view().unwrap(),
//!     &lv.view().unwrap(),
//!     &rv.view().unwrap(),
//!     join::NullEquality::Equal,
//! )
//! .unwrap();
//! // result matches keys 2 and 3
//! ```

use std::ptr;

pub use crate::types::NullEquality;
use crate::{
    error::{self, Result},
    table::{Table, TableView},
};

/// Result of a join operation.
pub struct JoinResult {
    /// The left-side result table (keys + payload for matching rows).
    pub left: Table,
    /// The right-side result table (keys + payload for matching rows).
    pub right: Table,
}

/// Inner join: returns only rows with matching keys in both tables.
///
/// `left_keys` and `right_keys` define the join condition.
/// `left_payload` and `right_payload` are additional columns carried through.
/// The result tables contain keys + payload columns concatenated.
pub fn inner_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinResult> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All raw handles are valid (NonNull maintained by their types).
        unsafe {
            cudf_sys::cudf_inner_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::inner_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    // SAFETY: out_left/out_right were returned by a successful FFI call, transferring ownership.
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}

/// Left join: all left rows are preserved; right side is null-padded
/// for non-matching rows.
pub fn left_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinResult> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All raw handles are valid (NonNull maintained by their types).
        unsafe {
            cudf_sys::cudf_left_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::left_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    // SAFETY: out_left/out_right were returned by a successful FFI call, transferring ownership.
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}

// ---------------------------------------------------------------------------
// High-level types
// ---------------------------------------------------------------------------

/// The type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Only rows with matching keys in both sides.
    Inner,
    /// All left rows preserved; right side null-padded for non-matches.
    Left,
    /// All rows from both sides; null-padded where no match.
    Full,
}

/// Options for `DataFrame::merge`, the cuDF-style unified join entry point.
///
/// Use the builder methods to configure the merge. At minimum, either `on()`
/// or both `left_on()` and `right_on()` must be set before calling `merge`.
///
/// # Example
///
/// ```no_run
/// use cudf::join::{JoinType, MergeOptions};
///
/// let opts = MergeOptions::new()
///     .on(&["id"])
///     .how(JoinType::Left)
///     .suffixes("_x", "_y");
/// ```
#[derive(Debug, Clone)]
pub struct MergeOptions {
    join_type: JoinType,
    on: Option<Vec<String>>,
    left_on: Option<Vec<String>>,
    right_on: Option<Vec<String>>,
    null_eq: NullEquality,
    suffix_left: String,
    suffix_right: String,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeOptions {
    /// Create a new `MergeOptions` with default settings.
    ///
    /// Defaults: inner join, suffixes `("_left", "_right")`, null equality `Equal`.
    pub fn new() -> Self {
        Self {
            join_type: JoinType::Inner,
            on: None,
            left_on: None,
            right_on: None,
            null_eq: NullEquality::Equal,
            suffix_left: "_left".to_string(),
            suffix_right: "_right".to_string(),
        }
    }

    /// Set shared key column names (both sides must have these names).
    pub fn on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set left-side key column names.
    pub fn left_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set right-side key column names.
    pub fn right_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set the join type.
    pub fn how(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Shorthand for `how(JoinType::Inner)`.
    pub fn how_inner(self) -> Self {
        self.how(JoinType::Inner)
    }

    /// Shorthand for `how(JoinType::Left)`.
    pub fn how_left(self) -> Self {
        self.how(JoinType::Left)
    }

    /// Shorthand for `how(JoinType::Full)`.
    pub fn how_full(self) -> Self {
        self.how(JoinType::Full)
    }

    /// Set how null keys are compared.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Set suffixes for conflicting payload column names.
    pub fn suffixes(mut self, left: &str, right: &str) -> Self {
        self.suffix_left = left.to_string();
        self.suffix_right = right.to_string();
        self
    }

    // -- Accessors (used by DataFrame::merge and tests) -----------------------

    /// Returns the configured join type.
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Returns the shared `on` key names, if set.
    pub fn on_keys(&self) -> Option<&[String]> {
        self.on.as_deref()
    }

    /// Returns the left-side key names, if set.
    pub fn left_on_keys(&self) -> Option<&[String]> {
        self.left_on.as_deref()
    }

    /// Returns the right-side key names, if set.
    pub fn right_on_keys(&self) -> Option<&[String]> {
        self.right_on.as_deref()
    }

    /// Returns the configured left suffix.
    pub fn suffix_left(&self) -> &str {
        &self.suffix_left
    }

    /// Returns the configured right suffix.
    pub fn suffix_right(&self) -> &str {
        &self.suffix_right
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }
}

// ---------------------------------------------------------------------------
// Low-level join functions
// ---------------------------------------------------------------------------

/// Full (outer) join: all rows from both sides are preserved, with
/// null-padding where there is no match.
pub fn full_join(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    left_payload: &TableView<'_>,
    right_payload: &TableView<'_>,
    null_equality: NullEquality,
) -> Result<JoinResult> {
    let mut out_left = ptr::null_mut();
    let mut out_right = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: All raw handles are valid (NonNull maintained by their types).
        unsafe {
            cudf_sys::cudf_full_join(
                left_keys.raw_handle(),
                right_keys.raw_handle(),
                left_payload.raw_handle(),
                right_payload.raw_handle(),
                null_equality as i32,
                &mut out_left,
                &mut out_right,
            )
        },
        "join::full_join",
        [("null_equality", format!("{:?}", null_equality).to_lowercase())],
    )?;
    // SAFETY: out_left/out_right were returned by a successful FFI call, transferring ownership.
    Ok(JoinResult { left: unsafe { Table::from_raw(out_left) }, right: unsafe { Table::from_raw(out_right) } })
}
