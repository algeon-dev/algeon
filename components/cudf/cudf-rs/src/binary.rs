//! GPU-accelerated binary operations.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::scalar::Scalar;
use crate::types::DataType;
use std::ptr;

/// Binary operators matching `cudf::binary_operator` (same integer values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BinaryOperator {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    TrueDiv = 4,
    FloorDiv = 5,
    Mod = 6,
    Pmod = 7,
    Pymod = 8,
    Pow = 9,
    IntPow = 10,
    LogBase = 11,
    Atan2 = 12,
    ShiftLeft = 13,
    ShiftRight = 14,
    ShiftRightUnsigned = 15,
    BitwiseAnd = 16,
    BitwiseOr = 17,
    BitwiseXor = 18,
    LogicalAnd = 19,
    LogicalOr = 20,
    Equal = 21,
    NotEqual = 22,
    Less = 23,
    Greater = 24,
    LessEqual = 25,
    GreaterEqual = 26,
    NullEquals = 27,
    NullNotEquals = 28,
    NullMax = 29,
    NullMin = 30,
}

/// Binary operation: column op column.
pub fn binary_operation_vv(
    lhs: &ColumnView<'_>,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle()?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_binary_operation_vv(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op as i32,
            dtype_handle,
            &mut out,
        )
    };
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status(status)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Binary operation: column op scalar.
pub fn binary_operation_vs(
    lhs: &ColumnView<'_>,
    rhs: &Scalar,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle()?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_binary_operation_vs(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op as i32,
            dtype_handle,
            &mut out,
        )
    };
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status(status)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Binary operation: scalar op column.
pub fn binary_operation_sv(
    lhs: &Scalar,
    rhs: &ColumnView<'_>,
    op: BinaryOperator,
    output_type: &DataType,
) -> Result<Column> {
    let dtype_handle = output_type.to_raw_handle()?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_binary_operation_sv(
            lhs.raw_handle(),
            rhs.raw_handle(),
            op as i32,
            dtype_handle,
            &mut out,
        )
    };
    unsafe { cudf_sys::cudf_data_type_destroy(dtype_handle) };
    error::check_status(status)?;
    Ok(unsafe { Column::from_raw(out) })
}
