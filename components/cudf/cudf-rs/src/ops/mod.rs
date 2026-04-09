//! Low-level operation modules and expression DSL entry points.
//!
//! `cudf::ops` is intentionally namespaced by operation domain. Most APIs live
//! under submodules such as [`join`], [`groupby`], [`sorting`], or [`filter`].
//! The root keeps only the high-frequency expression DSL entry points
//! ([`Expr`], [`col`], [`lit`], and [`when`]) plus the standard arithmetic trait
//! implementations for [`Column`](crate::core::column::Column).

pub mod binary;
pub mod concatenate;
pub mod copying;
pub mod dictionary;
pub mod expr;
pub mod filter;
pub mod groupby;
pub mod hashing;
pub mod join;
pub mod reduction;
pub mod replace;
pub mod search;
pub mod sorting;
pub mod stream_compaction;
pub mod strings;
pub mod unary;

use std::ops::{Add, Div, Mul, Sub};

pub use expr::{Expr, col, lit, when};

use crate::core::{column::Column, error::Result, scalar::Scalar};

// ---------------------------------------------------------------------------
// Column op Scalar (for each supported arithmetic numeric Rust type)
// ---------------------------------------------------------------------------

macro_rules! impl_column_scalar_ops {
    ($($rust_type:ty),+) => {
        $(
            impl Add<$rust_type> for &Column {
                type Output = Result<Column>;
                fn add(self, rhs: $rust_type) -> Self::Output {
                    self.add_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Sub<$rust_type> for &Column {
                type Output = Result<Column>;
                fn sub(self, rhs: $rust_type) -> Self::Output {
                    self.sub_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Mul<$rust_type> for &Column {
                type Output = Result<Column>;
                fn mul(self, rhs: $rust_type) -> Self::Output {
                    self.mul_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Div<$rust_type> for &Column {
                type Output = Result<Column>;
                fn div(self, rhs: $rust_type) -> Self::Output {
                    self.div_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
        )+
    };
}

impl_column_scalar_ops!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// ---------------------------------------------------------------------------
// Column op Column
// ---------------------------------------------------------------------------

impl Add<&Column> for &Column {
    type Output = Result<Column>;
    fn add(self, rhs: &Column) -> Self::Output {
        Column::add(self, rhs)
    }
}

impl Sub<&Column> for &Column {
    type Output = Result<Column>;
    fn sub(self, rhs: &Column) -> Self::Output {
        Column::sub(self, rhs)
    }
}

impl Mul<&Column> for &Column {
    type Output = Result<Column>;
    fn mul(self, rhs: &Column) -> Self::Output {
        Column::mul(self, rhs)
    }
}

impl Div<&Column> for &Column {
    type Output = Result<Column>;
    fn div(self, rhs: &Column) -> Self::Output {
        Column::div(self, rhs)
    }
}
