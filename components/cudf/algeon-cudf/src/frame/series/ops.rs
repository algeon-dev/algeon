//! `std::ops` trait implementations for `Series` arithmetic.
//!
//! All operators return `Result<Series>` — GPU computation, type promotion,
//! and FFI calls can fail at runtime.
//!
//! Supported forms:
//! - `Series op Series` (all value/ref combos): `Add`, `Sub`, `Mul`, `Div`, `Rem`
//! - `Series op scalar` / `scalar op Series` for all arithmetic numeric
//!   `CudfType`s (`u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`,
//!   `f32`, `f64`)
//! - `-Series` (`Neg`) and `!Series` (`Not`)
//!
//! Named methods (`add`, `sub`, `logical_and`, `logical_or`, `mul_scalar`,
//! etc.) remain on `Series` in `owned.rs`. In particular, boolean mask
//! composition stays on `logical_and` / `logical_or` instead of overloading
//! Rust's `&` / `|`, because those operators conventionally imply bitwise
//! semantics for integer-valued data.

use std::sync::Arc;

use super::owned::Series;
use crate::{
    core::{error::Result, scalar::Scalar},
    ops::binary::{self, BinaryOperator},
};

/// Generates `std::ops::{Add, Sub, Mul, Div}` for all owned/ref combos of
/// `Series op Series`, returning `Result<Series>`.
macro_rules! impl_series_series_op {
    ($trait:ident, $method:ident, $named:ident) => {
        impl std::ops::$trait for Series {
            type Output = Result<Series>;

            fn $method(self, rhs: Series) -> Self::Output {
                Series::$named(&self, &rhs)
            }
        }

        impl std::ops::$trait<&Series> for Series {
            type Output = Result<Series>;

            fn $method(self, rhs: &Series) -> Self::Output {
                Series::$named(&self, rhs)
            }
        }

        impl std::ops::$trait<Series> for &Series {
            type Output = Result<Series>;

            fn $method(self, rhs: Series) -> Self::Output {
                Series::$named(self, &rhs)
            }
        }

        impl std::ops::$trait<&Series> for &Series {
            type Output = Result<Series>;

            fn $method(self, rhs: &Series) -> Self::Output {
                Series::$named(self, rhs)
            }
        }
    };
}

impl_series_series_op!(Add, add, add);
impl_series_series_op!(Sub, sub, sub);
impl_series_series_op!(Mul, mul, mul);
impl_series_series_op!(Div, div, div);
impl_series_series_op!(Rem, rem, rem);

/// Generates `std::ops::{Add, Sub, Mul, Div}` for `Series op $scalar_ty`
/// and `&Series op $scalar_ty`, returning `Result<Series>`.
macro_rules! impl_series_scalar_op {
    ($trait:ident, $method:ident, $named:ident, $($scalar_ty:ty),+) => {
        $(
            impl std::ops::$trait<$scalar_ty> for Series {
                type Output = Result<Series>;

                fn $method(self, rhs: $scalar_ty) -> Self::Output {
                    self.$named(rhs)
                }
            }

            impl std::ops::$trait<$scalar_ty> for &Series {
                type Output = Result<Series>;

                fn $method(self, rhs: $scalar_ty) -> Self::Output {
                    self.$named(rhs)
                }
            }
        )+
    };
}

impl_series_scalar_op!(Add, add, add_scalar, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_series_scalar_op!(Sub, sub, sub_scalar, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_series_scalar_op!(Mul, mul, mul_scalar, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_series_scalar_op!(Div, div, div_scalar, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_series_scalar_op!(Rem, rem, rem_scalar, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// Generates `std::ops::{Add, Sub, Mul, Div}` for `$scalar_ty op Series`
/// and `$scalar_ty op &Series`, returning `Result<Series>`.
///
/// All forms use `scalar_left_op` which dispatches through the scalar-vs-column
/// binary primitive (`binary_operation_sv`) to preserve true operand order.
macro_rules! impl_scalar_series_op {
    ($trait:ident, $method:ident, $op:expr, $($scalar_ty:ty),+) => {
        $(
            impl std::ops::$trait<Series> for $scalar_ty {
                type Output = Result<Series>;

                fn $method(self, rhs: Series) -> Self::Output {
                    scalar_left_op(&rhs, self, $op)
                }
            }

            impl std::ops::$trait<&Series> for $scalar_ty {
                type Output = Result<Series>;

                fn $method(self, rhs: &Series) -> Self::Output {
                    scalar_left_op(rhs, self, $op)
                }
            }
        )+
    };
}

impl_scalar_series_op!(Add, add, BinaryOperator::Add, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_scalar_series_op!(Sub, sub, BinaryOperator::Sub, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_scalar_series_op!(Mul, mul, BinaryOperator::Mul, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_scalar_series_op!(Div, div, BinaryOperator::Div, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_scalar_series_op!(Rem, rem, BinaryOperator::Mod, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// Execute `scalar op series` using the scalar-vs-column binary primitive.
///
/// 1. Constructs a `Scalar` from the host value.
/// 2. Infers output dtype in left-to-right operand order (scalar, column).
/// 3. Dispatches to `binary_operation_sv`.
/// 4. Overrides the result column's execution domain to match the `Series`.
/// 5. Wraps the result as a `Series`.
fn scalar_left_op<T: crate::core::types::CudfType>(series: &Series, lhs: T, op: BinaryOperator) -> Result<Series> {
    let scalar = Scalar::from_value_in_domain(lhs, series.column.execution_domain())?;
    let col_view = series.column.view()?;

    // Infer output type: scalar type on the left, column type on the right.
    let out_type =
        binary::infer_binary_output_dtype_in(scalar.data_type()?, col_view.data_type()?, op, "series::scalar_left_op")?;

    // Dispatch through the low-level sv primitive (preserves operand order).
    // The temporary scalar is created in the Series' execution domain so
    // multi-GPU pipelines do not accidentally allocate it on the calling
    // thread's current device.
    let result_col = binary::binary_operation_sv(&scalar, &col_view, op, &out_type)
        .map(|c| c.with_execution_domain(Arc::clone(series.column.execution_domain())))?;

    Series::from_column(result_col)
}

impl std::ops::Neg for Series {
    type Output = Result<Series>;

    fn neg(self) -> Self::Output {
        Series::neg(&self)
    }
}

impl std::ops::Neg for &Series {
    type Output = Result<Series>;

    fn neg(self) -> Self::Output {
        Series::neg(self)
    }
}

impl std::ops::Not for Series {
    type Output = Result<Series>;

    fn not(self) -> Self::Output {
        Series::logical_not(&self)
    }
}

impl std::ops::Not for &Series {
    type Output = Result<Series>;

    fn not(self) -> Self::Output {
        Series::logical_not(self)
    }
}
