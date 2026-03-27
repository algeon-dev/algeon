//! GPU-accelerated sorting operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{column::Column, sorting, table::Table};
//!
//! let col = Column::from_slice(&[30i32, 10, 20]).unwrap();
//! let table = Table::from_columns(vec![col]).unwrap();
//!
//! let sorted = sorting::sort(
//!     &table.view().unwrap(),
//!     Some(&[sorting::Order::Ascending]),
//!     None,
//! )
//! .unwrap();
//! // sorted: [10, 20, 30]
//! ```

use std::ptr;

use crate::{
    column::Column,
    error::{self, Error, ErrorKind, Result},
    table::{Table, TableView},
};

/// Column sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Order {
    Ascending = 0,
    Descending = 1,
}

/// Where nulls appear in the sorted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullOrder {
    After = 0,
    Before = 1,
}

fn validate_sort_params(
    operation: &'static str,
    num_keys: i32,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<()> {
    let expected = usize::try_from(num_keys).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "number of sort keys must be >= 0")
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
    })?;

    if let Some(orders) = column_order {
        if orders.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column_order length ({}) must equal number of keys ({})", orders.len(), expected),
            )
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
            .with_context("column_order_len", orders.len().to_string()));
        }
    }

    if let Some(nulls) = null_precedence {
        if nulls.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("null_precedence length ({}) must equal number of keys ({})", nulls.len(), expected),
            )
            .with_operation(operation)
            .with_context("num_keys", num_keys.to_string())
            .with_context("null_precedence_len", nulls.len().to_string()));
        }
    }

    Ok(())
}

/// Helper: convert Option slices to raw pointers.
fn sort_params(column_order: Option<&[Order]>, null_precedence: Option<&[NullOrder]>) -> (*const i32, *const i32) {
    let co = match column_order {
        Some(s) => s.as_ptr() as *const i32,
        None => ptr::null(),
    };
    let np = match null_precedence {
        Some(s) => s.as_ptr() as *const i32,
        None => ptr::null(),
    };
    (co, np)
}

/// Sort the rows of a table by all its columns.
///
/// If `column_order` or `null_precedence` are `None`, cuDF defaults are
/// used (ascending, nulls after).
pub fn sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::sort", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_sort(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::sort",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Stable sort (preserves relative order of equal elements).
pub fn stable_sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::stable_sort", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_stable_sort(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::stable_sort",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Sort the `values` table by a separate `keys` table.
pub fn sort_by_key(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_key_columns = keys.num_columns()?;
    validate_sort_params("sorting::sort_by_key", num_key_columns, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_sort_by_key(values.raw_handle(), keys.raw_handle(), co, np, num_key_columns, &mut out)
        },
        "sorting::sort_by_key",
        [
            ("num_key_columns", num_key_columns.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Stable sort the `values` table by a separate `keys` table.
///
/// Preserves the relative order of rows with equal keys.
pub fn stable_sort_by_key(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let num_key_columns = keys.num_columns()?;
    validate_sort_params("sorting::stable_sort_by_key", num_key_columns, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe {
            cudf_sys::cudf_stable_sort_by_key(values.raw_handle(), keys.raw_handle(), co, np, num_key_columns, &mut out)
        },
        "sorting::stable_sort_by_key",
        [
            ("num_key_columns", num_key_columns.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Table::from_raw(out) })
}

/// Return the row indices that would sort the table.
///
/// The result is a single INT32 column.
pub fn sorted_order(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Column> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::sorted_order", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_sorted_order(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::sorted_order",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Check whether the table is already sorted according to the given order.
pub fn is_sorted(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<bool> {
    let num_keys = input.num_columns()?;
    validate_sort_params("sorting::is_sorted", num_keys, column_order, null_precedence)?;
    let (co, np) = sort_params(column_order, null_precedence);
    let mut out: i32 = 0;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_is_sorted(input.raw_handle(), co, np, num_keys, &mut out) },
        "sorting::is_sorted",
        [
            ("num_keys", num_keys.to_string()),
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
        ],
    )?;
    Ok(out != 0)
}

// ---------------------------------------------------------------------------
// SortOptions — cuDF-style façade for sort configuration
// ---------------------------------------------------------------------------

/// High-level sort configuration for [`DataFrame::sort_values`](crate::DataFrame::sort_values).
///
/// Provides a cuDF-Python-like interface over the low-level `Order`/`NullOrder`
/// slices used by the core sorting functions.
///
/// # Defaults
///
/// - `descending`: empty (all columns ascending)
/// - `nulls_first`: empty (all columns nulls-last)
/// - `stable`: `false` (unstable sort)
///
/// # Length rules
///
/// `descending` and `nulls_first` vectors must be either:
/// - **empty** — meaning "use defaults for all sort keys", or
/// - **exactly equal** in length to the number of sort-key columns.
///
/// Any other length is rejected at sort time with an `InvalidOperation` error.
///
/// # Example
///
/// ```no_run
/// use cudf::sorting::SortOptions;
///
/// let opts = SortOptions::new()
///     .with_descending(vec![true, false])
///     .with_nulls_first(vec![false, true])
///     .with_stable(true);
/// ```
#[derive(Debug, Clone)]
pub struct SortOptions {
    /// Per-column descending flags. Empty means all ascending.
    pub descending: Vec<bool>,
    /// Per-column nulls-first flags. Empty means all nulls-last.
    pub nulls_first: Vec<bool>,
    /// If `true`, use a stable sort that preserves the relative order of
    /// equal elements. Default is `false` (unstable).
    pub stable: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SortOptions {
    /// Create a new `SortOptions` with all defaults (ascending, nulls-last,
    /// unstable).
    pub fn new() -> Self {
        Self { descending: Vec::new(), nulls_first: Vec::new(), stable: false }
    }

    /// Set per-column descending flags.
    pub fn with_descending(mut self, descending: Vec<bool>) -> Self {
        self.descending = descending;
        self
    }

    /// Set per-column nulls-first flags.
    pub fn with_nulls_first(mut self, nulls_first: Vec<bool>) -> Self {
        self.nulls_first = nulls_first;
        self
    }

    /// Set whether to use a stable sort.
    pub fn with_stable(mut self, stable: bool) -> Self {
        self.stable = stable;
        self
    }

    /// Validate and expand option vectors against the number of sort keys,
    /// returning `(Vec<Order>, Option<Vec<NullOrder>>)`.
    ///
    /// - Empty vectors are expanded to defaults for all keys.
    /// - Non-empty vectors must match `num_keys` exactly.
    pub(crate) fn resolve(&self, num_keys: usize) -> Result<(Vec<Order>, Option<Vec<NullOrder>>)> {
        let orders = self.resolve_orders(num_keys)?;
        let null_orders = self.resolve_null_orders(num_keys)?;
        Ok((orders, null_orders))
    }

    fn resolve_orders(&self, num_keys: usize) -> Result<Vec<Order>> {
        if self.descending.is_empty() {
            Ok(vec![Order::Ascending; num_keys])
        } else if self.descending.len() == num_keys {
            Ok(self.descending.iter().map(|&d| if d { Order::Descending } else { Order::Ascending }).collect())
        } else {
            Err(Error::new(
                ErrorKind::InvalidOperation,
                format!(
                    "descending length ({}) must be 0 or equal to number of sort keys ({})",
                    self.descending.len(),
                    num_keys
                ),
            )
            .with_operation("SortOptions::resolve")
            .with_context("descending_len", self.descending.len().to_string())
            .with_context("num_keys", num_keys.to_string()))
        }
    }

    fn resolve_null_orders(&self, num_keys: usize) -> Result<Option<Vec<NullOrder>>> {
        if self.nulls_first.is_empty() {
            // None → cuDF default (nulls after).
            Ok(None)
        } else if self.nulls_first.len() == num_keys {
            Ok(Some(self.nulls_first.iter().map(|&nf| if nf { NullOrder::Before } else { NullOrder::After }).collect()))
        } else {
            Err(Error::new(
                ErrorKind::InvalidOperation,
                format!(
                    "nulls_first length ({}) must be 0 or equal to number of sort keys ({})",
                    self.nulls_first.len(),
                    num_keys
                ),
            )
            .with_operation("SortOptions::resolve")
            .with_context("nulls_first_len", self.nulls_first.len().to_string())
            .with_context("num_keys", num_keys.to_string()))
        }
    }
}
