//! GPU-accelerated unary operations.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::types::DataType;
use std::ptr;

/// Unary operators matching `cudf::unary_operator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum UnaryOperator {
    Sin = 0,
    Cos,
    Tan,
    ArcSin,
    ArcCos,
    ArcTan,
    Sinh,
    Cosh,
    Tanh,
    ArcSinh,
    ArcCosh,
    ArcTanh,
    Exp,
    Log,
    Sqrt,
    Cbrt,
    Ceil,
    Floor,
    Abs,
    Rint,
    BitCount,
    BitInvert,
    Not,
    Negate,
}

/// Apply a unary operation element-wise.
pub fn unary_operation(input: &ColumnView<'_>, op: UnaryOperator) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_unary_operation(input.raw_handle(), op as i32, &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column: `true` where elements are null.
pub fn is_null(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_is_null(input.raw_handle(), &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column: `true` where elements are valid (non-null).
pub fn is_valid(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_is_valid(input.raw_handle(), &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Cast a column to the specified data type.
pub fn cast(input: &ColumnView<'_>, output_type: &DataType) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle()?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_cast(input.raw_handle(), dtype_handle, &mut out)
    };
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status(status)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Check whether a cast between two data types is supported.
pub fn is_supported_cast(from: &DataType, to: &DataType) -> Result<bool> {
    let from_handle = from.to_raw_handle()?;
    let to_handle = to.to_raw_handle()?;
    let mut out: i32 = 0;
    let status = unsafe {
        cudf_sys::cudf_is_supported_cast(from_handle, to_handle, &mut out)
    };
    unsafe {
        cudf_sys::cudf_data_type_destroy(from_handle);
        cudf_sys::cudf_data_type_destroy(to_handle);
    }
    error::check_status(status)?;
    Ok(out != 0)
}

/// Return a boolean column: `true` where elements are NaN.
pub fn is_nan(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_is_nan(input.raw_handle(), &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Return a boolean column: `true` where elements are not NaN.
pub fn is_not_nan(input: &ColumnView<'_>) -> Result<Column> {
    let mut out = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_is_not_nan(input.raw_handle(), &mut out)
    })?;
    Ok(unsafe { Column::from_raw(out) })
}
