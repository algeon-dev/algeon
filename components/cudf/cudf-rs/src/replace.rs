//! GPU-accelerated null replacement operations.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::scalar::Scalar;
use std::ptr;

/// Replace null values in a column with a scalar.
///
/// Returns a new column where every null element in `input` is replaced
/// by `replacement`. Non-null elements are copied unchanged.
/// The scalar type must match the column type.
///
/// # Example
/// ```no_run
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::replace;
///
/// let col = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap();
/// let fill = Scalar::from_value(99i32).unwrap();
/// let result = replace::replace_nulls(&col.view().unwrap(), &fill).unwrap();
/// // result: [1, 99, 3]
/// ```
pub fn replace_nulls(input: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        unsafe {
            cudf_sys::cudf_replace_nulls_scalar(
                input.raw_handle(),
                replacement.raw_handle(),
                &mut out,
            )
        },
        "replace::replace_nulls",
    )?;
    Ok(unsafe { Column::from_raw(out) })
}
