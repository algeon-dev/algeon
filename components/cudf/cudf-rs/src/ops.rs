//! `std::ops` trait implementations for ergonomic Column arithmetic.
//!
//! Enables syntax like `&col + 1.0f32` and `&col_a * &col_b`.
//! All operations return `Result<Column>` since they involve GPU computation.

use std::ops::{Add, Div, Mul, Sub};

use crate::{column::Column, error::Result, scalar::Scalar};

// ---------------------------------------------------------------------------
// Column op Scalar (for each supported Rust numeric type)
// ---------------------------------------------------------------------------

macro_rules! impl_column_scalar_ops {
    ($($rust_type:ty),+) => {
        $(
            impl Add<$rust_type> for &Column {
                type Output = Result<Column>;
                fn add(self, rhs: $rust_type) -> Self::Output {
                    self.add_scalar(&Scalar::from_value(rhs)?)
                }
            }
            impl Sub<$rust_type> for &Column {
                type Output = Result<Column>;
                fn sub(self, rhs: $rust_type) -> Self::Output {
                    self.sub_scalar(&Scalar::from_value(rhs)?)
                }
            }
            impl Mul<$rust_type> for &Column {
                type Output = Result<Column>;
                fn mul(self, rhs: $rust_type) -> Self::Output {
                    self.mul_scalar(&Scalar::from_value(rhs)?)
                }
            }
            impl Div<$rust_type> for &Column {
                type Output = Result<Column>;
                fn div(self, rhs: $rust_type) -> Self::Output {
                    self.div_scalar(&Scalar::from_value(rhs)?)
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
