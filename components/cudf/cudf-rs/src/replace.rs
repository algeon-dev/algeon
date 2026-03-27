//! GPU-accelerated null replacement operations.

use std::ptr;

use crate::{
    column::{Column, ColumnView},
    error::{self, Result},
    scalar::Scalar,
};

/// Replace null values in a column with a scalar.
///
/// Returns a new column where every null element in `input` is replaced
/// by `replacement`. Non-null elements are copied unchanged.
/// The scalar type must match the column type.
///
/// # Example
/// ```no_run
/// use cudf::{column::Column, replace, scalar::Scalar};
///
/// let col = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
///     .unwrap();
/// let fill = Scalar::from_value(99i32).unwrap();
/// let result = replace::replace_nulls(&col.view().unwrap(), &fill).unwrap();
/// // result: [1, 99, 3]
/// ```
pub fn replace_nulls(input: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        unsafe { cudf_sys::cudf_replace_nulls_scalar(input.raw_handle(), replacement.raw_handle(), &mut out) },
        "replace::replace_nulls",
    )?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Replace null values in a column with corresponding values from another column.
///
/// Returns a new column where every null element in `input` is replaced
/// by the corresponding element from `replacement`. Non-null elements in
/// `input` are copied unchanged. Both columns must have the same type and
/// length.
///
/// # Example
/// ```no_run
/// use cudf::{column::Column, replace};
///
/// let left = Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true])
///     .unwrap();
/// let right = Column::from_slice(&[10i32, 20, 30]).unwrap();
/// let result = replace::replace_nulls_column(
///     &left.view().unwrap(),
///     &right.view().unwrap(),
/// )
/// .unwrap();
/// // result: [1, 20, 3]
/// ```
pub fn replace_nulls_column(input: &ColumnView<'_>, replacement: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in(
        unsafe { cudf_sys::cudf_replace_nulls_column(input.raw_handle(), replacement.raw_handle(), &mut out) },
        "replace::replace_nulls_column",
    )?;
    Ok(unsafe { Column::from_raw(out) })
}
