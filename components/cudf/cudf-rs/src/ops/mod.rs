//! `std::ops` trait implementations for ergonomic Column arithmetic.
//!
//! Enables syntax like `&col + 1.0f32` and `&col_a * &col_b`.
//! All operations return `Result<Column>` since they involve GPU computation.

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

pub use binary::*;
pub use concatenate::*;
pub use copying::*;
pub use dictionary::*;
pub use expr::*;
pub use filter::*;
pub use groupby::*;
pub use hashing::*;
pub use join::*;
pub use reduction::*;
pub use replace::*;
pub use search::*;
pub use sorting::*;
pub use stream_compaction::*;
pub use strings::*;
pub use unary::*;

use crate::core::{column::Column, error::Result, scalar::Scalar};

// ---------------------------------------------------------------------------
// Column op Scalar (for each supported Rust numeric type)
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

impl_column_scalar_ops!(f32, f64, i32, i64);

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
