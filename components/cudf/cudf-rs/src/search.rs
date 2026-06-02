//! GPU-accelerated search / set-membership operations.

use std::ptr;

use crate::{
    column::{Column, ColumnView},
    error::{self, Result},
};

/// Check if elements of `needles` exist in `haystack` (set membership / "isin").
///
/// Returns a `BOOL8` column with the same size as `needles`. Element `i`
/// is `true` if `needles[i]` is found anywhere in `haystack`, `false`
/// otherwise. Null rows in `needles` produce null rows in the output.
///
/// Both columns must have the same data type.
///
/// This is the building block for [`DataFrame::filter_isin`](crate::dataframe::DataFrame::filter_isin),
/// which wraps `contains` + `filter` into a one-liner convenience method.
///
/// # Example
/// ```no_run
/// use cudf::{column::Column, search};
///
/// let haystack = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let needles = Column::from_slice(&[20i32, 40]).unwrap();
/// let result =
///     search::contains(&haystack.view().unwrap(), &needles.view().unwrap())
///         .unwrap();
/// // result: [true, false]
/// ```
pub fn contains(haystack: &ColumnView<'_>, needles: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        // SAFETY: haystack and needles handles are valid (NonNull maintained by ColumnView).
        unsafe { cudf_sys::cudf_contains(haystack.raw_handle(), needles.raw_handle(), &mut out) },
        "search::contains",
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}
