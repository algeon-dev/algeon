//! GPU-accelerated unary operations.

use std::ptr;

use crate::core::{
    column::{Column, ColumnView},
    error::{self, Result},
    types::DataType,
};

/// Unary operators matching `cudf::unary_operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum UnaryOperator {
    /// Trigonometric sine.
    Sin = 0,
    /// Trigonometric cosine.
    Cos,
    /// Trigonometric tangent.
    Tan,
    /// Inverse sine (arcsin).
    ArcSin,
    /// Inverse cosine (arccos).
    ArcCos,
    /// Inverse tangent (arctan).
    ArcTan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    ArcSinh,
    /// Inverse hyperbolic cosine.
    ArcCosh,
    /// Inverse hyperbolic tangent.
    ArcTanh,
    /// Natural exponential (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Square root.
    Sqrt,
    /// Cube root.
    Cbrt,
    /// Ceiling — smallest integer >= x.
    Ceil,
    /// Floor — largest integer <= x.
    Floor,
    /// Absolute value.
    Abs,
    /// Round to nearest integer (half-to-even / banker's rounding).
    Rint,
    /// Population count — number of `1`-bits in the binary representation.
    BitCount,
    /// Bitwise inversion (flips every bit). For integer types.
    BitInvert,
    /// Logical NOT. For `Bool8` columns: `true` becomes `false` and vice versa.
    Not,
    /// Arithmetic negation (`-x`). For numeric types.
    Negate,
}

/// Apply a unary operation element-wise.
pub fn unary_operation(input: &ColumnView<'_>, op: UnaryOperator) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status_in_with(
        // SAFETY: input handle is valid (NonNull maintained by ColumnView).
        unsafe { cudf_sys::cudf_unary_operation(input.raw_handle(), op as i32, &mut out) },
        "unary::unary_operation",
        [("operator", format!("{:?}", op).to_lowercase())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column with each element logically negated.
///
/// Input must be Bool8. Equivalent to `unary_operation(input, UnaryOperator::Not)`.
pub fn logical_not(input: &ColumnView<'_>) -> Result<Column> {
    unary_operation(input, UnaryOperator::Not)
}

/// Return a boolean column: `true` where elements are null.
pub fn is_null(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull maintained by ColumnView).
    error::check_status_in(unsafe { cudf_sys::cudf_is_null(input.raw_handle(), &mut out) }, "unary::is_null")?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column: `true` where elements are valid (non-null).
pub fn is_valid(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull maintained by ColumnView).
    error::check_status_in(unsafe { cudf_sys::cudf_is_valid(input.raw_handle(), &mut out) }, "unary::is_valid")?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Cast a column to the specified data type.
pub fn cast(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle_for("unary::cast")?;
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). dtype_handle was returned by
    // a successful to_raw_handle_for call.
    let status = unsafe { cudf_sys::cudf_cast(input.raw_handle(), dtype_handle, &mut out) };
    // SAFETY: dtype_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status_in_with(
        status,
        "unary::cast",
        [("output_type_id", format!("{:?}", output_type.id()).to_lowercase())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Check whether a cast between two data types is supported.
pub fn is_supported_cast(from: &DataType, to: &DataType) -> Result<bool> {
    let from_handle = from.to_raw_handle_for("unary::is_supported_cast")?;
    let to_handle = to.to_raw_handle_for("unary::is_supported_cast")?;
    let mut out: i32 = 0;
    // SAFETY: from_handle and to_handle were returned by successful
    // to_raw_handle_for calls.
    let status = unsafe { cudf_sys::cudf_is_supported_cast(from_handle, to_handle, &mut out) };
    // SAFETY: Both handles are valid and must be destroyed.
    unsafe {
        cudf_sys::cudf_data_type_destroy(from_handle);
        cudf_sys::cudf_data_type_destroy(to_handle);
    }
    error::check_status_in_with(
        status,
        "unary::is_supported_cast",
        [
            ("from_type_id", format!("{:?}", from.id()).to_lowercase()),
            ("to_type_id", format!("{:?}", to.id()).to_lowercase()),
        ],
    )?;
    Ok(out != 0)
}

/// Return a boolean column: `true` where elements are NaN.
pub fn is_nan(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull maintained by ColumnView).
    error::check_status_in(unsafe { cudf_sys::cudf_is_nan(input.raw_handle(), &mut out) }, "unary::is_nan")?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column: `true` where elements are not NaN.
pub fn is_not_nan(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull maintained by ColumnView).
    error::check_status_in(unsafe { cudf_sys::cudf_is_not_nan(input.raw_handle(), &mut out) }, "unary::is_not_nan")?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}
